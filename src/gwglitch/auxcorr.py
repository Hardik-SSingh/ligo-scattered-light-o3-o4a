"""Auxiliary-channel correlation analysis.

Two distinct questions are asked of the auxiliary data, and they need
different statistics:

*Rate level.*  Does the hourly rate of scattered-light triggers track ground
motion?  The natural estimator is a rank correlation between the hourly rate
and the hourly band-limited RMS of a seismometer.  Hourly rates are strongly
autocorrelated -- the microseism varies on a timescale of days, not hours --
so the naive standard error on a Spearman coefficient is far too small.  We
use a moving-block bootstrap with a block length set in the configuration.

*Trigger level.*  For an individual glitch, does the arch frequency follow
``f = 2 n |v| / lambda`` computed from a motion witness?  That is handled by
:mod:`gwglitch.scattering`; this module supplies the surrounding bookkeeping.

Availability
------------
The rate-level analysis needs ground-motion trends, which exist in the GWOSC
O3 auxiliary release (``*_BLRMS_*.rms``, 1 Hz second-trend frames) and do
**not** exist in the O4a release.  Calling the O3 functions with an O4 epoch
raises rather than silently returning nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .dataio import DataFetcher

log = logging.getLogger(__name__)

__all__ = [
    "CorrelationResult",
    "hourly_ground_motion",
    "spearman_block_bootstrap",
    "rate_ratio_by_quantile",
    "lagged_correlation",
    "partial_spearman",
    "correlation_table",
]


@dataclass
class CorrelationResult:
    """Rank correlation with a block-bootstrap confidence interval."""

    channel: str
    detector: str
    epoch: str
    rho: float
    ci_low: float
    ci_high: float
    p_value: float
    n_hours: float
    block_hours: float

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def as_row(self) -> dict:
        return {
            "detector": self.detector,
            "epoch": self.epoch,
            "channel": self.channel,
            "spearman_rho": self.rho,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_hours": self.n_hours,
            "block_hours": self.block_hours,
            "significant": self.significant,
        }


# ---------------------------------------------------------------------------
# Ground-motion trends
# ---------------------------------------------------------------------------


def hourly_ground_motion(
    fetcher: DataFetcher,
    cfg: Config,
    detector: str,
    epoch: str,
    channels: list[str] | None = None,
    chunk_hours: int = 4,
    gps_start: float | None = None,
    gps_end: float | None = None,
    segments=None,
) -> pd.DataFrame:
    """Median band-limited RMS per GPS hour, for each configured trend channel.

    The second-trend channels are sampled at 1 Hz, so one observing epoch is
    ~1.4e7 samples per channel.  They are fetched in ``chunk_hours`` blocks so
    that a dropped connection costs one chunk, not the whole run, and so that
    the on-disk cache is reusable at chunk granularity.

    ``chunk_hours`` defaults to 4 because that is what the GWOSC NDS server
    will serve: a 4 h request succeeds and an 8 h request fails.  All channels
    for a chunk are requested together, since the per-request overhead
    dominates -- nine channels cost 11.3 s batched against 40.8 s separately.

    The *median* within an hour is used rather than the mean because the trend
    channels contain occasional large excursions (earthquakes, calibration
    activity) whose effect on an hourly mean is out of proportion to how much
    of the hour they occupy.
    """
    ep = cfg.epoch(epoch)
    if ep.run != "O3":
        raise ValueError(
            f"epoch {epoch!r} is in run {ep.run}. Ground-motion trend channels "
            "are published only for O3; the O4a auxiliary release contains no "
            "seismometer, accelerometer or suspension channel. See "
            "docs/DATA_AVAILABILITY.md."
        )

    channels = channels or list(cfg.auxiliary.ground_motion_trends[detector])
    chunk_hours = int(getattr(cfg.auxiliary, 'chunk_hours', chunk_hours))
    begin = int(gps_start if gps_start is not None else ep.gps_start)
    stop = int(gps_end if gps_end is not None else ep.gps_end)
    step = chunk_hours * 3600
    records: dict[str, list[pd.Series]] = {ch: [] for ch in channels}

    starts = list(range(begin, stop, step))
    if segments is not None:
        # Only ask for chunks that contain analysis-ready time.  A chunk with
        # no livetime contributes nothing to a rate correlation, and the
        # detector is down for roughly a quarter of calendar time.
        keep = [s for s in starts
                if _livetime_between(segments, s, min(s + step, stop)) > 0]
        if len(keep) < len(starts):
            log.info(
                "%s %s: skipping %d of %d chunks with no analysis-ready time",
                detector, epoch, len(starts) - len(keep), len(starts),
            )
        starts = keep

    n_empty = 0
    for start in starts:
        end = min(start + step, stop)
        series = fetcher.auxiliary_many(channels, start, end)
        if not series:
            n_empty += 1
        for ch, data in series.items():
            if len(data.values) == 0:
                continue
            hours = np.floor(data.times / 3600.0).astype(np.int64)
            s = pd.Series(data.values, index=pd.Index(hours, name="hour_bin"))
            # Trend channels use 0 or NaN as a "no data" sentinel.
            s = s.replace(0.0, np.nan).dropna()
            if len(s):
                records[ch].append(s.groupby(level=0).median())

    frames = {}
    for ch, pieces in records.items():
        if pieces:
            frames[ch] = pd.concat(pieces).groupby(level=0).median()
        else:
            log.warning("no data recovered for %s in %s %s", ch, detector, epoch)

    if not frames:
        raise RuntimeError(
            f"no ground-motion trend data recovered for {detector} {epoch} "
            f"over [{begin}, {stop}) -- {n_empty} of {len(starts)} "
            "chunks came back empty. The O3 auxiliary release has documented "
            "gaps that do not line up with the strain segments (the first day "
            "of O3b is one), so try a later stretch before assuming a fault."
        )
    if n_empty:
        log.warning(
            "%s %s: %d of %d chunks had no auxiliary data at all",
            detector, epoch, n_empty, len(starts),
        )
    out = pd.DataFrame(frames)
    out.index.name = "hour_bin"
    return out.sort_index()



def _livetime_between(segments, start: float, end: float) -> float:
    """Analysis-ready seconds inside ``[start, end)``."""
    return float(sum(
        max(0.0, min(end, s_end) - max(start, s_start))
        for s_start, s_end in segments.segments
    ))


# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------


def spearman_block_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    block_length: int,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Spearman rho with a moving-block-bootstrap confidence interval.

    Parameters
    ----------
    block_length
        Number of consecutive samples per block.  It must exceed the
        autocorrelation length of the series, otherwise the interval is still
        too narrow.  For hourly rates driven by the microseism, six hours is a
        conservative choice.

    Returns
    -------
    rho, ci_low, ci_high, p_value
        ``p_value`` is the asymptotic Spearman p-value, reported for
        reference only; the bootstrap interval is the statistic to quote.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = len(x)
    if n < 3 * block_length:
        return (np.nan, np.nan, np.nan, np.nan)

    rho, p = stats.spearmanr(x, y)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    draws = np.empty(n_resamples)

    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:n]
        r, _ = stats.spearmanr(x[idx], y[idx])
        draws[i] = r

    draws = draws[np.isfinite(draws)]
    if draws.size < 100:
        return (float(rho), np.nan, np.nan, float(p))

    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return (float(rho), lo, hi, float(p))


def rate_ratio_by_quantile(
    rate: np.ndarray,
    driver: np.ndarray,
    n_quantiles: int = 4,
) -> pd.DataFrame:
    """Mean trigger rate in each quantile of a driving channel.

    A rank correlation says *whether* rate follows ground motion; this says
    *by how much*, in units a commissioner cares about.  The last row of the
    returned table gives the ratio of the top to the bottom quantile.
    """
    rate = np.asarray(rate, dtype=float)
    driver = np.asarray(driver, dtype=float)
    good = np.isfinite(rate) & np.isfinite(driver)
    rate, driver = rate[good], driver[good]
    if len(rate) < n_quantiles * 5:
        return pd.DataFrame()

    edges = np.percentile(driver, np.linspace(0, 100, n_quantiles + 1))
    edges[-1] += 1e-9
    which = np.clip(np.digitize(driver, edges) - 1, 0, n_quantiles - 1)

    rows = []
    for q in range(n_quantiles):
        sel = which == q
        rows.append(
            {
                "quantile": q + 1,
                "driver_low": float(edges[q]),
                "driver_high": float(edges[q + 1]),
                "n_hours": int(sel.sum()),
                "mean_rate": float(rate[sel].mean()),
                "median_rate": float(np.median(rate[sel])),
                "sem_rate": float(rate[sel].std(ddof=1) / np.sqrt(max(sel.sum(), 1))),
            }
        )
    out = pd.DataFrame(rows)
    bottom = out.loc[0, "mean_rate"]
    out.attrs["top_to_bottom_ratio"] = (
        float(out.loc[n_quantiles - 1, "mean_rate"] / bottom) if bottom > 0 else np.nan
    )
    return out


def lagged_correlation(
    rate: np.ndarray,
    driver: np.ndarray,
    max_lag_hours: int = 12,
    min_points: int | None = None,
) -> pd.DataFrame:
    """Spearman rho as a function of lag, to check causal ordering.

    Sign convention: a **positive** lag means the trigger rate responds that
    many hours *after* the ground motion.  Scattering is a mechanical response
    and should be effectively instantaneous, so the scan must peak at zero.  A
    peak at a large lag of either sign indicates that both series are being
    driven by a slow common trend -- diurnal or weather -- rather than by the
    mechanism under test, and a peak at negative lag would mean the rate
    anticipates the ground motion, which is not a thing.
    """
    rate = np.asarray(rate, dtype=float)
    driver = np.asarray(driver, dtype=float)

    # A fixed floor of 50 points silently returns an all-NaN scan for any
    # sample shorter than a couple of days, which is exactly what a trial run
    # is.  Scale it to the series instead, keeping a small absolute minimum.
    if min_points is None:
        min_points = max(10, min(50, len(rate) // 4))

    rows = []
    for lag in range(-max_lag_hours, max_lag_hours + 1):
        if lag > 0:
            # Rate delayed with respect to the driver.
            a, b = rate[lag:], driver[: len(driver) - lag]
        elif lag < 0:
            a, b = rate[: len(rate) + lag], driver[-lag:]
        else:
            a, b = rate, driver
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < min_points:
            rows.append({"lag_hours": lag, "rho": np.nan, "n": int(good.sum())})
            continue
        r, _ = stats.spearmanr(a[good], b[good])
        rows.append({"lag_hours": lag, "rho": float(r), "n": int(good.sum())})
    return pd.DataFrame(rows)


def partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Spearman correlation of ``x`` and ``y`` controlling for ``z``.

    Applied here with ``z`` = the microseismic BLRMS, to test whether an
    apparent correlation with, say, the 1-6 Hz anthropogenic band survives
    once the microseism is accounted for.  Computed as the correlation of the
    residuals of the rank-transformed variables after linear regression on the
    rank-transformed control.
    """
    arrays = [np.asarray(a, dtype=float) for a in (x, y, z)]
    good = np.all([np.isfinite(a) for a in arrays], axis=0)
    if good.sum() < 20:
        return (np.nan, np.nan)
    rx, ry, rz = (stats.rankdata(a[good]) for a in arrays)

    def residual(v, control):
        design = np.vstack([control, np.ones_like(control)]).T
        coef, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ coef

    r, p = stats.spearmanr(residual(rx, rz), residual(ry, rz))
    return (float(r), float(p))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def correlation_table(
    rates: pd.DataFrame,
    ground_motion: pd.DataFrame,
    detector: str,
    epoch: str,
    cfg: Config,
) -> pd.DataFrame:
    """Correlate the hourly trigger rate against every ground-motion channel.

    ``rates`` must have a ``rate`` column and a ``hour_bin`` index; hours with
    zero livetime must already have been dropped (see
    :func:`gwglitch.catalog.hourly_rate`).
    """
    stat = cfg.statistics
    joined = rates[["rate"]].join(ground_motion, how="inner")
    if joined.empty:
        raise ValueError(
            f"no overlapping hours between trigger rates and ground motion for "
            f"{detector} {epoch}"
        )

    results = []
    for channel in ground_motion.columns:
        rho, lo, hi, p = spearman_block_bootstrap(
            joined["rate"].to_numpy(),
            joined[channel].to_numpy(),
            block_length=int(stat.block_length_hours),
            n_resamples=int(stat.bootstrap_resamples),
            alpha=float(stat.alpha),
            seed=int(cfg.sampling.seed),
        )
        results.append(
            CorrelationResult(
                channel=channel,
                detector=detector,
                epoch=epoch,
                rho=rho,
                ci_low=lo,
                ci_high=hi,
                p_value=p,
                n_hours=float(len(joined)),
                block_hours=float(stat.block_length_hours),
            ).as_row()
        )

    out = pd.DataFrame(results)
    return out.sort_values("spearman_rho", ascending=False).reset_index(drop=True)
