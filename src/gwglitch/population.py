"""Population-level comparison of the O3 and O4a scattered-light samples.

The descriptive results of the study are here: how many scattered-light
transients occur per hour of observing time, how that rate changed between O3
and O4a, and whether the *shape* of the population changed -- specifically
whether the slow-scattering component driven by the microseism was suppressed
relative to the fast-scattering component.

Statistical choices
-------------------
* Rates are Poisson counts over a known livetime, so intervals are Poisson
  (Garwood) intervals, not Gaussian.
* Feature distributions are compared with the two-sample Kolmogorov-Smirnov
  test for the p-value and with Cliff's delta for the effect size.  With tens
  of thousands of triggers per sample, essentially every KS test rejects; the
  effect size is what carries the physics, and it is quoted first.
* Many features are tested at once, so p-values are corrected with the
  Benjamini-Hochberg procedure and the corrected values are the ones reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .scattering import classify_slow_fast, driving_frequency

log = logging.getLogger(__name__)

__all__ = [
    "RateEstimate",
    "poisson_rate",
    "rate_table",
    "compare_feature",
    "compare_all_features",
    "slow_fast_breakdown",
    "benjamini_hochberg",
    "cliffs_delta",
]


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


@dataclass
class RateEstimate:
    """A Poisson rate with a two-sided confidence interval."""

    count: int
    livetime_hours: float
    rate: float
    lo: float
    hi: float
    confidence: float

    def as_row(self) -> dict:
        return {
            "count": self.count,
            "livetime_hours": self.livetime_hours,
            "rate_per_hour": self.rate,
            "rate_lo": self.lo,
            "rate_hi": self.hi,
        }


def poisson_rate(count: int, livetime_hours: float, alpha: float = 0.05) -> RateEstimate:
    """Rate per hour with an exact (Garwood) Poisson interval.

    The interval uses the chi-squared quantiles rather than ``sqrt(N)``,
    which matters for the sparsely-populated bins of the per-week rate curves.
    """
    if livetime_hours <= 0:
        return RateEstimate(count, 0.0, np.nan, np.nan, np.nan, 1 - alpha)

    if count == 0:
        lo = 0.0
    else:
        lo = 0.5 * stats.chi2.ppf(alpha / 2, 2 * count)
    hi = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1))

    return RateEstimate(
        count=int(count),
        livetime_hours=float(livetime_hours),
        rate=count / livetime_hours,
        lo=lo / livetime_hours,
        hi=hi / livetime_hours,
        confidence=1 - alpha,
    )


def rate_table(
    counts: dict[tuple[str, str], int],
    livetimes: dict[tuple[str, str], float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Assemble a (detector, epoch) rate table from counts and livetimes."""
    rows = []
    for key, count in counts.items():
        detector, epoch = key
        lt = livetimes.get(key, 0.0)
        est = poisson_rate(count, lt, alpha)
        rows.append({"detector": detector, "epoch": epoch, **est.as_row()})
    return (
        pd.DataFrame(rows)
        .sort_values(["detector", "epoch"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 20000, seed: int = 0) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    A non-parametric effect size that needs no distributional assumption and
    is invariant under monotone transformation of the feature -- important
    here because several features (energy, slope) span orders of magnitude.

    For large samples the exact O(n m) computation is replaced by a rank-based
    identity, and both arrays are subsampled to ``max_n`` to bound the cost.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    if a.size > max_n:
        a = rng.choice(a, max_n, replace=False)
    if b.size > max_n:
        b = rng.choice(b, max_n, replace=False)

    n, m = a.size, b.size
    # Mann-Whitney U relates to Cliff's delta by delta = 2U/(nm) - 1.
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (n * m) - 1.0)


def _bootstrap_median_ci(
    x: np.ndarray, n_resamples: int, alpha: float, seed: int
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    meds = np.median(x[idx], axis=1)
    return (
        float(np.median(x)),
        float(np.percentile(meds, 100 * alpha / 2)),
        float(np.percentile(meds, 100 * (1 - alpha / 2))),
    )


def compare_feature(
    a: np.ndarray,
    b: np.ndarray,
    name: str,
    label_a: str = "A",
    label_b: str = "B",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Compare one feature between two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]

    if fa.size < 20 or fb.size < 20:
        return {
            "feature": name,
            "n_a": int(fa.size),
            "n_b": int(fb.size),
            "median_a": np.nan,
            "median_b": np.nan,
            "delta": np.nan,
            "ks_stat": np.nan,
            "p_value": np.nan,
        }

    med_a, lo_a, hi_a = _bootstrap_median_ci(fa, n_resamples, alpha, seed)
    med_b, lo_b, hi_b = _bootstrap_median_ci(fb, n_resamples, alpha, seed + 1)
    ks = stats.ks_2samp(fa, fb)

    return {
        "feature": name,
        "label_a": label_a,
        "label_b": label_b,
        "n_a": int(fa.size),
        "n_b": int(fb.size),
        "median_a": med_a,
        "median_a_lo": lo_a,
        "median_a_hi": hi_a,
        "median_b": med_b,
        "median_b_lo": lo_b,
        "median_b_hi": hi_b,
        "median_ratio": med_b / med_a if med_a not in (0.0, np.nan) else np.nan,
        "delta": cliffs_delta(fa, fb, seed=seed),
        "ks_stat": float(ks.statistic),
        "p_value": float(ks.pvalue),
    }


def compare_all_features(
    df: pd.DataFrame,
    group_column: str,
    label_a: str,
    label_b: str,
    features: list[str],
    cfg: Config,
) -> pd.DataFrame:
    """Compare every feature between two groups, with BH correction."""
    stat = cfg.statistics
    sub_a = df.loc[df[group_column] == label_a]
    sub_b = df.loc[df[group_column] == label_b]
    if sub_a.empty or sub_b.empty:
        raise ValueError(
            f"one of the groups is empty: {label_a}={len(sub_a)}, {label_b}={len(sub_b)}"
        )

    rows = [
        compare_feature(
            sub_a[f].to_numpy(),
            sub_b[f].to_numpy(),
            name=f,
            label_a=label_a,
            label_b=label_b,
            n_resamples=min(int(stat.bootstrap_resamples), 2000),
            alpha=float(stat.alpha),
            seed=int(cfg.sampling.seed) + i,
        )
        for i, f in enumerate(features)
        if f in df.columns
    ]
    out = pd.DataFrame(rows)
    out["p_adjusted"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant"] = out["p_adjusted"] < float(stat.alpha)
    return out.sort_values("delta", key=np.abs, ascending=False).reset_index(drop=True)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate control)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out

    pv = p[good]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    out[good] = adjusted
    return out


# ---------------------------------------------------------------------------
# Slow / fast decomposition
# ---------------------------------------------------------------------------


def slow_fast_breakdown(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Split each (detector, epoch) sample into slow and fast scattering.

    The split is on the measured arch repetition period; the implied driving
    frequency is ``1 / (2 T_arch)`` because ``f ~ |v|`` gives two arches per
    cycle of the motion.  Reporting the implied driving frequency alongside
    the fractions makes the physical claim checkable: the slow population
    should land in the 0.1-0.3 Hz microseism band and the fast population in
    the 1-6 Hz anthropogenic band.
    """
    split = float(cfg.scattering.slow_fast_split_period)
    micro = tuple(cfg.scattering.microseism_band)
    anthro = tuple(cfg.scattering.anthropogenic_band)

    work = df.copy()
    work["population"] = [
        classify_slow_fast(p, split) for p in work["arch_period"].to_numpy()
    ]
    work["driving_frequency"] = [
        driving_frequency(p) for p in work["arch_period"].to_numpy()
    ]

    rows = []
    for (det, epoch), group in work.groupby(["detector", "epoch"], sort=True):
        n_total = len(group)
        counts = group["population"].value_counts()
        n_slow = int(counts.get("slow", 0))
        n_fast = int(counts.get("fast", 0))
        n_unknown = int(counts.get("unknown", 0))
        classified = n_slow + n_fast

        fd_slow = group.loc[group["population"] == "slow", "driving_frequency"]
        fd_fast = group.loc[group["population"] == "fast", "driving_frequency"]

        rows.append(
            {
                "detector": det,
                "epoch": epoch,
                "n_total": n_total,
                "n_slow": n_slow,
                "n_fast": n_fast,
                "n_unknown": n_unknown,
                "frac_slow": n_slow / classified if classified else np.nan,
                "frac_fast": n_fast / classified if classified else np.nan,
                "frac_unclassified": n_unknown / n_total if n_total else np.nan,
                "median_fdrive_slow": float(fd_slow.median()) if len(fd_slow) else np.nan,
                "median_fdrive_fast": float(fd_fast.median()) if len(fd_fast) else np.nan,
                "slow_in_microseism_band": (
                    float(fd_slow.between(*micro).mean()) if len(fd_slow) else np.nan
                ),
                "fast_in_anthropogenic_band": (
                    float(fd_fast.between(*anthro).mean()) if len(fd_fast) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def proportion_difference(
    k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05
) -> dict:
    """Difference of two binomial proportions with a Newcombe interval.

    Used for the headline O3-versus-O4a statement, "the slow fraction fell
    from X to Y".  The Newcombe (score-based) interval behaves correctly when
    a proportion approaches 0 or 1, where the Wald interval does not.
    """
    if n1 == 0 or n2 == 0:
        return {"p1": np.nan, "p2": np.nan, "difference": np.nan,
                "ci_low": np.nan, "ci_high": np.nan}

    z = stats.norm.ppf(1 - alpha / 2)

    def wilson(k, n):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return centre - half, centre + half

    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    diff = p2 - p1

    lower = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

    return {
        "p1": float(p1),
        "p2": float(p2),
        "difference": float(diff),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }
