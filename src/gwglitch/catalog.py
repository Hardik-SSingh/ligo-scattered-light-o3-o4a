"""Gravity Spy trigger catalogue: download, parse, filter, sample.

The Gravity Spy machine-learning classification release (Zenodo record
5649212, Glanzer et al. 2023) is the labelled input for the O3 half of this
study.  It contains, for every Omicron trigger with SNR > 7.5 and peak
frequency in [10, 2048] Hz, the trigger metadata plus a 22-element vector of
class posteriors and the arg-max label.

There is no equivalent public release for O4a.  ``load_epoch`` will refuse to
be called with ``O4a`` rather than silently return an empty frame; the O4a
sample is built instead by :mod:`gwglitch.identify` from strain alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Config
from .constants import GRAVITYSPY_CLASSES
from .utils import human_duration, retry

log = logging.getLogger(__name__)

__all__ = [
    "download_catalog",
    "load_epoch",
    "load_run",
    "select_class",
    "cluster_triggers",
    "add_time_columns",
    "stratified_sample",
    "block_stratified_sample",
    "sample_within_blocks",
    "hourly_rate",
]

#: Columns we keep.  The 22 class-posterior columns are dropped after the
#: confidence cut to keep the working frame small, except for the target
#: class, which is retained for diagnostics.
_KEEP = [
    "event_time",
    "ifo",
    "peak_time",
    "duration",
    "peak_frequency",
    "central_freq",
    "bandwidth",
    "amplitude",
    "snr",
    "q_value",
    "gravityspy_id",
    "ml_label",
    "ml_confidence",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@retry(n_attempts=4, delay=5.0, exceptions=(requests.RequestException, OSError))
def _download_one(url: str, dest: Path, chunk_bytes: int = 1 << 20) -> Path:
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        written = 0
        with tmp.open("wb") as fh:
            for block in resp.iter_content(chunk_size=chunk_bytes):
                fh.write(block)
                written += len(block)
        if total and written != total:
            tmp.unlink(missing_ok=True)
            raise OSError(f"short read for {url}: {written} of {total} bytes")
        tmp.replace(dest)
    log.info("wrote %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_catalog(cfg: Config, overwrite: bool = False) -> dict[str, Path]:
    """Fetch the Gravity Spy CSVs listed in the configuration.

    Returns a mapping ``{"H1_O3a": Path(...), ...}``.  Files already present
    are not re-downloaded unless ``overwrite`` is set.
    """
    out_dir = cfg.path_for("catalog_dir")
    base = cfg.gravityspy.base_url
    paths: dict[str, Path] = {}

    for key, filename in cfg.gravityspy.files.items():
        dest = out_dir / filename
        if dest.exists() and not overwrite:
            log.info("%s already present (%.1f MB), skipping",
                     filename, dest.stat().st_size / 1e6)
        else:
            _download_one(f"{base}/{filename}?download=1", dest)
        paths[key] = dest
    return paths


# ---------------------------------------------------------------------------
# Load / filter
# ---------------------------------------------------------------------------


def _catalog_path(cfg: Config, detector: str, epoch: str) -> Path:
    key = f"{detector}_{epoch}"
    try:
        filename = cfg.gravityspy.files[key]
    except KeyError as exc:
        raise KeyError(
            f"no Gravity Spy file configured for {key}. "
            "The public Gravity Spy release covers O1, O2, O3a and O3b only "
            "(Zenodo 5649212); see docs/DATA_AVAILABILITY.md."
        ) from exc
    return cfg.path_for("catalog_dir") / filename


def load_epoch(
    cfg: Config,
    detector: str,
    epoch: str,
    apply_cuts: bool = True,
    keep_posteriors: bool = False,
) -> pd.DataFrame:
    """Load one detector/epoch Gravity Spy CSV.

    Parameters
    ----------
    apply_cuts
        Apply the SNR, peak-frequency and ML-confidence cuts from the config.
        Cuts are applied to *all* classes so that the target and control
        samples are drawn under identical conditions.
    keep_posteriors
        Retain all 22 class-posterior columns (large; only needed for the
        confidence-threshold sensitivity study).
    """
    if epoch not in cfg.epoch_names:
        raise KeyError(f"unknown epoch {epoch!r}")
    if cfg.epoch(epoch).run != "O3":
        raise ValueError(
            f"epoch {epoch!r} belongs to run {cfg.epoch(epoch).run}; the "
            "Gravity Spy release only covers O1-O3b. Build the O4a sample "
            "with gwglitch.identify instead."
        )

    path = _catalog_path(cfg, detector, epoch)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run scripts/01_fetch_gravityspy_catalog.py first"
        )

    usecols = None if keep_posteriors else _KEEP + [cfg.gravityspy.target_label]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    n_raw = len(df)

    # Sanity: the file should contain only this detector.
    if "ifo" in df.columns:
        bad = set(df["ifo"].unique()) - {detector}
        if bad:
            log.warning("%s contains unexpected ifo values %s", path.name, bad)

    df["epoch"] = epoch
    df["detector"] = detector
    df["run"] = cfg.epoch(epoch).run

    if apply_cuts:
        df = apply_quality_cuts(cfg, df, epoch)

    log.info(
        "%s %s: %d triggers loaded, %d after cuts (%.1f%%)",
        detector, epoch, n_raw, len(df), 100.0 * len(df) / max(n_raw, 1),
    )
    return df.reset_index(drop=True)


def apply_quality_cuts(cfg: Config, df: pd.DataFrame, epoch: str) -> pd.DataFrame:
    """SNR / peak-frequency / confidence / in-epoch cuts."""
    gs = cfg.gravityspy
    ep = cfg.epoch(epoch)
    fmin, fmax = gs.peak_frequency_range

    mask = (
        (df["snr"] >= gs.min_snr)
        & df["peak_frequency"].between(fmin, fmax)
        & (df["ml_confidence"] >= gs.min_ml_confidence)
        & df["event_time"].between(ep.gps_start, ep.gps_end)
        & np.isfinite(df["event_time"])
    )
    return df.loc[mask].copy()


def load_run(
    cfg: Config,
    run: str = "O3",
    detectors: tuple[str, ...] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Concatenate every configured epoch of ``run`` for the given detectors."""
    detectors = detectors or ("H1", "L1")
    frames = [
        load_epoch(cfg, det, epoch, **kwargs)
        for epoch in cfg.epochs_for_run(run)
        for det in detectors
    ]
    if not frames:
        raise ValueError(f"no epochs configured for run {run!r}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("event_time").reset_index(drop=True)


def select_class(df: pd.DataFrame, label: str | list[str]) -> pd.DataFrame:
    """Rows whose ``ml_label`` matches ``label`` (string or list)."""
    labels = [label] if isinstance(label, str) else list(label)
    unknown = set(labels) - set(GRAVITYSPY_CLASSES)
    if unknown:
        raise ValueError(f"not Gravity Spy classes: {sorted(unknown)}")
    return df.loc[df["ml_label"].isin(labels)].copy()



def cluster_triggers(
    df: pd.DataFrame,
    window: float,
    time_column: str = "event_time",
    rank_column: str = "snr",
    group_columns: tuple[str, ...] = ("detector",),
) -> pd.DataFrame:
    """Collapse triggers that describe the same physical event.

    The Gravity Spy release is a list of *Omicron triggers*, not of events.
    Two things follow from that, and both bias the analysis if ignored:

    * 12.3 per cent of rows share an exact ``(event_time, detector)``: the same
      Omicron trigger appears under several Gravity Spy subject identifiers,
      because the release makes a subject per Q-scan duration view.
    * About half of all triggers fall within eight seconds of another one.
      Since the analysed Q-scan window is +/- 8 s, those triggers land in each
      other's windows and yield *identical* feature vectors.

    Left uncorrected, every morphology distribution is weighted by how many
    Omicron triggers a given arch train happened to produce, so loud, long
    trains are counted dozens of times and quiet ones once.  That is a
    selection effect on exactly the axis the study measures.

    This function groups triggers whose times fall within ``window`` of the
    running cluster start and keeps the loudest as the representative, adding
    ``cluster_id`` and ``cluster_size``.  Choosing ``window`` equal to the
    width of the analysed Q-scan window makes the statement precise: no two
    retained events share a Q-scan.

    The unclustered trigger list is kept separately, because trigger *rate* is
    still a meaningful quantity for the auxiliary correlation -- it just is not
    an event rate, and the write-up says which one it is quoting.
    """
    if df.empty:
        return df.copy()

    pieces = []
    for _, group in df.groupby(list(group_columns), sort=True):
        group = group.sort_values(time_column, kind="mergesort").reset_index(drop=True)
        times = group[time_column].to_numpy(dtype=float)

        # A new cluster starts whenever the gap to the previous trigger exceeds
        # the window.  Chaining is deliberate: a continuous scattering episode
        # is one event even if it lasts longer than the window.
        gaps = np.diff(times, prepend=times[0])
        cluster = np.cumsum(gaps > window)
        group = group.assign(_cluster=cluster)

        sizes = group.groupby("_cluster")[time_column].transform("size")
        group = group.assign(cluster_size=sizes)

        if rank_column in group.columns:
            idx = group.groupby("_cluster")[rank_column].idxmax()
        else:
            idx = group.groupby("_cluster").head(1).index
        pieces.append(group.loc[idx])

    out = pd.concat(pieces, ignore_index=True)
    out = out.drop(columns=["_cluster"])
    out["cluster_id"] = np.arange(len(out))

    log.info(
        "clustering at %.1f s: %d triggers -> %d events (%.2f triggers per event)",
        window, len(df), len(out), len(df) / max(len(out), 1),
    )
    return out.sort_values(time_column).reset_index(drop=True)


def add_time_columns(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Add ``days_into_epoch`` and an integer ``hour_bin`` for rate work."""
    out = df.copy()
    starts = out["epoch"].map(
        {name: cfg.epoch(name).gps_start for name in cfg.epoch_names}
    )
    out["days_into_epoch"] = (out["event_time"] - starts) / 86400.0
    out["hour_bin"] = np.floor(out["event_time"] / 3600.0).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pd.DataFrame,
    n_total: int,
    seed: int,
    strata: tuple[str, ...] = ("detector", "epoch"),
    time_column: str = "event_time",
    n_time_bins: int = 20,
) -> pd.DataFrame:
    """Draw ``n_total`` rows per stratum, spread evenly over the run.

    A naive random draw over a whole observing run is dominated by whichever
    weeks happened to be noisiest, which biases the morphology distribution
    towards the conditions of those weeks.  Splitting each stratum into
    ``n_time_bins`` equal-duration time bins and drawing proportionally from
    each keeps the sample representative of the run as a whole while still
    reflecting real duty-cycle variation.
    """
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []

    for _, group in df.groupby(list(strata), sort=True):
        if len(group) <= n_total:
            pieces.append(group)
            continue

        t = group[time_column].to_numpy(dtype=float)
        edges = np.linspace(t.min(), t.max() + 1e-6, n_time_bins + 1)
        which = np.clip(np.digitize(t, edges) - 1, 0, n_time_bins - 1)

        # Largest-remainder allocation so the quotas sum exactly to n_total.
        counts = np.bincount(which, minlength=n_time_bins)
        exact = n_total * counts / counts.sum()
        quota = np.floor(exact).astype(int)
        quota = np.minimum(quota, counts)
        deficit = n_total - quota.sum()
        if deficit > 0:
            order = np.argsort(-(exact - quota))
            for idx in order:
                if deficit == 0:
                    break
                room = counts[idx] - quota[idx]
                take = min(room, deficit)
                quota[idx] += take
                deficit -= take

        chosen: list[np.ndarray] = []
        positions = np.arange(len(group))
        for b in range(n_time_bins):
            if quota[b] == 0:
                continue
            pool = positions[which == b]
            chosen.append(rng.choice(pool, size=quota[b], replace=False))
        picked = np.sort(np.concatenate(chosen)) if chosen else np.array([], int)
        pieces.append(group.iloc[picked])

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(time_column).reset_index(drop=True)



#: GWOSC serves open strain in HDF5 files of this length.  A request for any
#: sub-interval downloads the whole containing file, so this is the true unit
#: of cost when building a sample.
GWOSC_BLOCK_SECONDS = 4096


def block_stratified_sample(
    df: pd.DataFrame,
    n_blocks: int,
    max_per_block: int,
    seed: int,
    block_seconds: int = GWOSC_BLOCK_SECONDS,
    strata: tuple[str, ...] = ("detector", "epoch"),
    time_column: str = "event_time",
) -> tuple[pd.DataFrame, set[tuple[str, int]]]:
    """Sample GWOSC *file blocks*, then events within them.

    Why not sample events directly.  GWOSC stores open strain in files of
    ``block_seconds``, and asking for a 64 s window downloads the whole file --
    60 to 90 seconds on a domestic connection, essentially independent of how
    much was requested.  A sample of individual events spread across a run
    therefore costs close to one full download per event: for 4800 events that
    measured out at 3484 downloads, or roughly three days of wall clock.

    Sampling blocks first and taking up to ``max_per_block`` events from each
    collapses that by a factor of three to five, because scattered-light events
    cluster strongly in time.  Blocks are chosen stratified across each
    stratum's span, so the sample still represents the whole run rather than
    whichever fortnight was noisiest.

    The cost is honest and must be stated: events drawn from one block share
    ground-motion conditions, so they are correlated and the effective sample
    size is smaller than the row count.  Drawing many blocks and capping
    ``max_per_block`` modestly keeps that in check; the block count, not the
    event count, is what sets the statistical weight.

    Events are drawn at random within a block rather than by loudness, which
    would bias the morphology sample towards the tail.

    Returns
    -------
    sample, blocks
        The chosen rows, and the set of ``(detector, block index)`` pairs used,
        so that a second sample (the control classes) can be drawn from exactly
        the same blocks at no extra download cost.
    """
    if df.empty:
        return df.copy(), set()

    rng = np.random.default_rng(seed)
    work = df.assign(_block=(df[time_column] // block_seconds).astype(np.int64))

    pieces: list[pd.DataFrame] = []
    chosen_blocks: set[tuple[str, int]] = set()

    for _, group in work.groupby(list(strata), sort=True):
        blocks = np.sort(group["_block"].unique())

        if len(blocks) <= n_blocks:
            picked = blocks
        else:
            # One block from each of n_blocks contiguous bins of the block
            # list: stratification in time, without assuming blocks are evenly
            # spaced (they are not -- the detector is not always locked).
            edges = np.linspace(0, len(blocks), n_blocks + 1).astype(int)
            picked = np.array([
                blocks[rng.integers(lo, hi)]
                for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo
            ])

        detector = str(group["detector"].iloc[0])
        chosen_blocks |= {(detector, int(b)) for b in picked}

        selected = group.loc[group["_block"].isin(picked)]
        for _, block_rows in selected.groupby("_block", sort=True):
            if len(block_rows) <= max_per_block:
                pieces.append(block_rows)
            else:
                take = rng.choice(len(block_rows), size=max_per_block, replace=False)
                pieces.append(block_rows.iloc[np.sort(take)])

    out = pd.concat(pieces, ignore_index=True).drop(columns=["_block"])
    out = out.sort_values(time_column).reset_index(drop=True)

    log.info(
        "block sampling: %d blocks, %d events (%.2f per block) from %d available",
        len(chosen_blocks), len(out),
        len(out) / max(len(chosen_blocks), 1), len(df),
    )
    return out, chosen_blocks


def sample_within_blocks(
    df: pd.DataFrame,
    blocks: set[tuple[str, int]],
    max_per_block: int,
    seed: int,
    block_seconds: int = GWOSC_BLOCK_SECONDS,
    time_column: str = "event_time",
    balance_column: str | None = "ml_label",
) -> pd.DataFrame:
    """Draw events from blocks that have already been chosen.

    Used for the control sample.  Reusing the signal sample's blocks makes the
    controls free in download terms, and it also removes a confound: signal and
    background are then drawn from the same stretches of data, so the
    classifier cannot separate them by learning what a noisy night looks like.

    It introduces a different confound, though, which ``balance_column`` exists
    to remove.  Those blocks were chosen because they contain scattering, so
    the glitch classes that co-occur with scattering dominate them: a first
    real run returned 41 Low_Frequency_Burst out of 46 controls.  A classifier
    trained on that separates scattering from Low_Frequency_Burst and little
    else, which is not the job it is given in the O4a search.  Sampling
    round-robin across the classes present in each block keeps the shared-block
    advantages and spreads the quota over whatever classes are there.

    Pass ``balance_column=None`` to sample uniformly within a block instead.
    """
    if df.empty or not blocks:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    work = df.assign(_block=(df[time_column] // block_seconds).astype(np.int64))
    keys = pd.Series(list(zip(work["detector"], work["_block"])), index=work.index)
    work = work.loc[keys.isin(blocks)]

    if work.empty:
        log.warning("no control events fall inside the %d chosen blocks", len(blocks))
        return work.drop(columns=["_block"])

    pieces = []
    for _, block_rows in work.groupby(["detector", "_block"], sort=True):
        if len(block_rows) <= max_per_block:
            pieces.append(block_rows)
            continue

        if balance_column and balance_column in block_rows.columns:
            pieces.append(
                _round_robin_sample(block_rows, balance_column, max_per_block, rng)
            )
        else:
            take = rng.choice(len(block_rows), size=max_per_block, replace=False)
            pieces.append(block_rows.iloc[np.sort(take)])

    out = pd.concat(pieces, ignore_index=True).drop(columns=["_block"])
    log.info(
        "control sample: %d events from the same %d blocks; class mix %s",
        len(out), len(blocks),
        dict(out[balance_column].value_counts()) if balance_column
        and balance_column in out.columns else "n/a",
    )
    return out.sort_values(time_column).reset_index(drop=True)


def _round_robin_sample(
    rows: pd.DataFrame, column: str, quota: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Take ``quota`` rows, spread as evenly as possible over ``column``.

    One row is taken from each group in turn until the quota is met, so a class
    with three members in the block contributes up to three rows before a class
    with three hundred contributes its fourth.
    """
    shuffled = {
        value: list(rng.permutation(group.index))
        for value, group in rows.groupby(column, sort=True)
    }
    chosen: list = []
    while len(chosen) < quota and any(shuffled.values()):
        for value in sorted(shuffled):
            if not shuffled[value]:
                continue
            chosen.append(shuffled[value].pop())
            if len(chosen) == quota:
                break
    return rows.loc[sorted(chosen)]


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def hourly_rate(
    df: pd.DataFrame,
    gps_start: float,
    gps_end: float,
    livetime_hours: pd.Series | None = None,
) -> pd.DataFrame:
    """Bin triggers into UTC-aligned hours.

    Returns a frame indexed by ``hour_bin`` (GPS hour number) with columns
    ``count``, ``gps_mid`` and, if ``livetime_hours`` is supplied, ``rate``
    in triggers per hour of analysis-ready livetime.

    Hours with zero *livetime* and hours with zero *triggers* are different
    things: the former must be dropped, the latter must be kept as a genuine
    zero.  Conflating them is the classic way to manufacture a spurious
    rate-versus-ground-motion correlation.
    """
    lo = int(np.floor(gps_start / 3600.0))
    hi = int(np.floor(gps_end / 3600.0))
    index = pd.RangeIndex(lo, hi + 1, name="hour_bin")

    counts = (
        df["hour_bin"].value_counts().reindex(index, fill_value=0).astype(int)
        if "hour_bin" in df.columns
        else pd.Series(0, index=index, dtype=int)
    )

    out = pd.DataFrame({"count": counts})
    out["gps_mid"] = (out.index.to_numpy() + 0.5) * 3600.0

    if livetime_hours is not None:
        lt = livetime_hours.reindex(index).fillna(0.0)
        out["livetime_hours"] = lt
        out = out.loc[lt > 0.0]
        out["rate"] = out["count"] / out["livetime_hours"]
    return out


def summarise(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per detector/epoch counts of the target class, for the results table."""
    target = cfg.gravityspy.target_label
    rows = []
    for (det, epoch), group in df.groupby(["detector", "epoch"]):
        ep = cfg.epoch(epoch)
        span_days = (ep.gps_end - ep.gps_start) / 86400.0
        n_target = int((group["ml_label"] == target).sum())
        rows.append(
            {
                "detector": det,
                "epoch": epoch,
                "n_triggers": len(group),
                f"n_{target}": n_target,
                "fraction": n_target / max(len(group), 1),
                "epoch_span_days": span_days,
                "per_day_calendar": n_target / span_days,
                "median_snr": float(
                    group.loc[group["ml_label"] == target, "snr"].median()
                )
                if n_target
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["detector", "epoch"]).reset_index(drop=True)


def describe_download_plan(cfg: Config) -> str:
    """Human-readable summary printed by the fetch script before it starts."""
    files = cfg.gravityspy.files
    lines = [
        f"Gravity Spy ML classifications, Zenodo record {cfg.gravityspy.zenodo_record}",
        f"  target class      : {cfg.gravityspy.target_label}",
        f"  ML confidence cut : >= {cfg.gravityspy.min_ml_confidence}",
        f"  SNR cut           : >= {cfg.gravityspy.min_snr}",
        f"  files             : {', '.join(files.values())}",
        "  approximate total : 560 MB",
        f"  estimated time    : {human_duration(560e6 / 2e6)} at 2 MB/s",
    ]
    return "\n".join(lines)
