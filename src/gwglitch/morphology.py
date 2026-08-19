"""Morphology feature extraction from a Q-scan.

Scattered light is not distinguished from other glitch classes by loudness or
by peak frequency -- it is distinguished by *shape*.  The characteristic
signature is a train of smooth, symmetric, harmonically-spaced arches whose
apex frequency is set by the peak velocity of a moving scattering surface and
whose repetition period is half the period of that motion.

This module reduces a :class:`~gwglitch.qscan.QMap` to the fixed-length,
physically interpretable feature vector named in
:data:`gwglitch.constants.FEATURE_NAMES`.  Those features feed three things:

1. the O3 morphology distributions that are the descriptive result;
2. the strain-only classifier that transfers the O3 definition to O4a;
3. the comparison between the auxiliary scattering predictor and the observed
   ridge (:mod:`gwglitch.scattering`).

Every feature is computed from the *normalised energy* map, so it is
insensitive to the absolute calibration and directly comparable between
detectors and between observing runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage, signal, stats

from .constants import FEATURE_NAMES
from .qscan import QMap

log = logging.getLogger(__name__)

__all__ = ["RidgeTrack", "extract_ridge", "extract_features", "feature_frame"]


# ---------------------------------------------------------------------------
# Ridge extraction
# ---------------------------------------------------------------------------


@dataclass
class RidgeTrack:
    """Time-resolved description of the supra-threshold structure in a Q-map.

    Attributes
    ----------
    times
        Time bin centres, relative to the trigger.
    ridge
        Frequency of the loudest tile in each time bin; NaN where that tile is
        below threshold.
    upper
        Highest frequency exceeding threshold in each time bin; NaN elsewhere.
        For a scattering arch train this is the arch envelope, which is the
        quantity the aux predictor predicts.
    lower
        Lowest frequency exceeding threshold in each time bin.
    active
        Boolean mask of time bins containing any supra-threshold tile.
    energy_t
        Energy summed over frequency in each time bin.
    energy_f
        Energy summed over time in each frequency bin.
    """

    times: np.ndarray
    ridge: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    active: np.ndarray
    energy_t: np.ndarray
    energy_f: np.ndarray
    frequencies: np.ndarray
    threshold: float

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    @property
    def duty_cycle(self) -> float:
        return float(self.active.mean()) if self.active.size else 0.0


def extract_ridge(
    qmap: QMap,
    threshold: float,
    min_island_bins: int = 40,
    smooth_bins: int = 5,
) -> RidgeTrack:
    """Threshold a Q-map and trace its frequency envelope in time.

    Parameters
    ----------
    threshold
        Normalised-energy threshold, ``Z = SNR^2 / 2``.  It has to sit above
        the largest energy a *noise-only* map produces, not merely above the
        mean: a map of this size holds order 1e4 independent tiles, so its
        noise maximum is around ``ln(1e4) ~ 9`` before the multi-Q search is
        taken into account and around 16 after.  The configured default of 20
        clears that.  Setting it lower does not recover faint harmonics, it
        just fills ``f_max`` with noise.
    min_island_bins
        Connected supra-threshold regions smaller than this many output pixels
        are removed.  Because the output grid is interpolated from far coarser
        Q tiles, a single hot tile already spans tens of pixels, so this is a
        speck filter rather than a significance cut.
    smooth_bins
        Width of the median filter applied along the time axis to the ridge
        and envelope tracks before differentiation.
    """
    energy = qmap.energy
    freqs = qmap.frequencies
    times = qmap.relative_times

    mask = energy > threshold
    if mask.any() and min_island_bins > 1:
        labels, n_labels = ndimage.label(mask)
        if n_labels:
            sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n_labels + 1))
            keep = np.zeros(n_labels + 1, dtype=bool)
            keep[1:] = sizes >= min_island_bins
            mask = keep[labels]

    active = mask.any(axis=1)

    ridge = np.full(len(times), np.nan)
    upper = np.full(len(times), np.nan)
    lower = np.full(len(times), np.nan)

    if active.any():
        masked_energy = np.where(mask, energy, -np.inf)
        idx_peak = np.argmax(masked_energy, axis=1)
        ridge[active] = freqs[idx_peak[active]]

        # Highest / lowest supra-threshold frequency per time bin.
        any_f = mask.any(axis=1)
        first = np.argmax(mask, axis=1)
        last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
        lower[any_f] = freqs[first[any_f]]
        upper[any_f] = freqs[last[any_f]]

    if smooth_bins > 1:
        ridge = _nan_median_filter(ridge, smooth_bins)
        upper = _nan_median_filter(upper, smooth_bins)
        lower = _nan_median_filter(lower, smooth_bins)

    energy_t = np.where(mask, energy, 0.0).sum(axis=1)
    energy_f = np.where(mask, energy, 0.0).sum(axis=0)

    return RidgeTrack(
        times=times,
        ridge=ridge,
        upper=upper,
        lower=lower,
        active=active,
        energy_t=energy_t,
        energy_f=energy_f,
        frequencies=freqs,
        threshold=float(threshold),
    )


def _nan_median_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Median filter that leaves NaNs in place instead of spreading them."""
    out = x.copy()
    finite = np.isfinite(x)
    if finite.sum() < size:
        return out
    filled = np.interp(np.arange(len(x)), np.flatnonzero(finite), x[finite])
    smoothed = signal.medfilt(filled, kernel_size=size | 1)
    out[finite] = smoothed[finite]
    return out


# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------


def _arch_statistics(track: RidgeTrack, dt: float) -> dict[str, float]:
    """Count arches and measure their repetition period.

    Arches are the local maxima of the frequency envelope.  Because the
    scattering frequency goes as ``|v|``, one full cycle of the driving motion
    produces *two* arches, so the measured repetition period is half the
    driving period -- the factor of two is applied where the period is
    interpreted physically, not here.

    The period is taken from the *autocorrelation* of the envelope rather than
    from the spacing of detected peaks.  Peak spacing is fragile in exactly
    the situation that matters: when the topmost harmonic dips below threshold
    the envelope drops to the next one down, creating a spurious local maximum
    that splits one arch into two and biases the period low.  The
    autocorrelation is insensitive to that because the underlying periodicity
    survives in every harmonic simultaneously.  The peak count is still
    reported, as ``n_arches``, since it is a useful independent feature.
    """
    empty = {"n_arches": 0.0, "arch_period": np.nan, "arch_period_scatter": np.nan}

    env = track.upper.copy()
    finite = np.isfinite(env)
    if finite.sum() < 16:
        return empty

    # Work on the contiguous span between the first and last supra-threshold
    # bin, interpolating across short internal dropouts.
    first, last = np.flatnonzero(finite)[[0, -1]]
    idx = np.arange(first, last + 1)
    inner = finite[first : last + 1]
    if inner.sum() < 16:
        return empty
    y = np.interp(idx, idx[inner], env[first : last + 1][inner])
    span = len(y) * dt

    # -- peak count -----------------------------------------------------
    spread = np.percentile(y, 90) - np.percentile(y, 10)
    prominence = max(0.2 * spread, 1.0)              # Hz
    min_distance = max(int(round(0.05 / dt)), 3)     # >= 50 ms between apexes
    peaks, _ = signal.find_peaks(y, prominence=prominence, distance=min_distance)
    n_arches = float(len(peaks))

    # -- period from the autocorrelation --------------------------------
    z = y - y.mean()
    denom = float(np.dot(z, z))
    if denom <= 0 or span < 4 * dt:
        return {**empty, "n_arches": n_arches}

    acf = signal.correlate(z, z, mode="full")[len(z) - 1 :] / denom

    min_lag = max(int(round(0.05 / dt)), 3)
    max_lag = int(len(acf) // 2)
    if max_lag <= min_lag + 2:
        return {**empty, "n_arches": n_arches}

    window = acf[min_lag:max_lag]
    acf_peaks, props = signal.find_peaks(window, prominence=0.05)
    if acf_peaks.size == 0:
        period = np.nan
    else:
        # The first prominent maximum is the fundamental repetition; later
        # ones are its multiples.
        best = int(acf_peaks[0])
        period = float((best + min_lag) * dt)

    # -- scatter --------------------------------------------------------
    if len(peaks) >= 3:
        spacings = np.diff(peaks) * dt
        med = float(np.median(spacings))
        scatter = (
            float(stats.median_abs_deviation(spacings, scale="normal") / med)
            if med > 0
            else np.nan
        )
        if not np.isfinite(period):
            period = med
    else:
        scatter = np.nan

    return {
        "n_arches": n_arches,
        "arch_period": period,
        "arch_period_scatter": scatter,
    }


def _harmonic_statistics(qmap: QMap, track: RidgeTrack) -> dict[str, float]:
    """Count harmonically spaced ridges at the arch apex.

    At the instant of peak surface velocity every harmonic ``f_n = n f_1``
    reaches its own maximum simultaneously, so a frequency slice through the
    apex shows a comb.  The number of teeth and their spacing measure how
    strongly non-linear the scattering is -- higher harmonic content means a
    larger phase excursion ``4 pi x / lambda`` and therefore larger surface
    displacement.
    """
    if not track.active.any():
        return {"n_harmonics": 0.0, "harmonic_spacing": np.nan}

    apex = int(np.nanargmax(np.where(np.isfinite(track.upper), track.upper, -np.inf)))
    lo = max(apex - 2, 0)
    hi = min(apex + 3, qmap.energy.shape[0])
    slice_energy = qmap.energy[lo:hi].mean(axis=0)

    freqs = qmap.frequencies
    if slice_energy.max() <= track.threshold:
        return {"n_harmonics": 0.0, "harmonic_spacing": np.nan}

    df = float(np.median(np.diff(freqs)))
    peaks, _ = signal.find_peaks(
        slice_energy,
        height=track.threshold,
        distance=max(int(round(1.5 / df)), 2),   # harmonics are >= 1.5 Hz apart
        prominence=0.3 * track.threshold,
    )
    if len(peaks) < 2:
        return {"n_harmonics": float(len(peaks)), "harmonic_spacing": np.nan}

    fpeaks = freqs[peaks]
    spacing = float(np.median(np.diff(fpeaks)))
    return {"n_harmonics": float(len(peaks)), "harmonic_spacing": spacing}


def _shape_statistics(track: RidgeTrack, dt: float) -> dict[str, float]:
    """Slope, curvature and time-symmetry of the frequency envelope."""
    env = track.upper
    finite = np.isfinite(env)
    if finite.sum() < 8:
        return {
            "ridge_slope_rms": np.nan,
            "ridge_curvature_rms": np.nan,
            "upward_downward_ratio": np.nan,
        }

    idx = np.arange(len(env))
    y = np.interp(idx, idx[finite], env[finite])
    first_good, last_good = np.flatnonzero(finite)[[0, -1]]
    y = y[first_good : last_good + 1]

    slope = np.gradient(y, dt)
    curvature = np.gradient(slope, dt)

    up = float((slope > 0).sum())
    down = float((slope < 0).sum())
    ratio = up / down if down > 0 else np.nan

    return {
        "ridge_slope_rms": float(np.sqrt(np.mean(slope**2))),
        "ridge_curvature_rms": float(np.sqrt(np.mean(curvature**2))),
        "upward_downward_ratio": ratio,
    }


def _energy_statistics(qmap: QMap, track: RidgeTrack) -> dict[str, float]:
    """Where the supra-threshold energy sits in frequency and in time."""
    freqs = qmap.frequencies
    ef = track.energy_f
    total = float(ef.sum())

    if total <= 0:
        return {
            "energy_total": 0.0,
            "frac_energy_below_40hz": np.nan,
            "frac_energy_above_100hz": np.nan,
            "spectral_centroid": np.nan,
            "band_concentration": np.nan,
            "time_symmetry": np.nan,
        }

    centroid = float(np.sum(freqs * ef) / total)

    # Concentration: fraction of energy inside the narrowest band holding
    # half of it, expressed as (that band's width) / (full analysis band).
    order = np.argsort(-ef)
    cumulative = np.cumsum(ef[order]) / total
    n_half = int(np.searchsorted(cumulative, 0.5) + 1)
    band_width = n_half * float(np.median(np.diff(freqs)))
    concentration = 1.0 - band_width / (freqs[-1] - freqs[0])

    et = track.energy_t
    if et.sum() > 0:
        t = track.times
        mean_t = float(np.sum(t * et) / et.sum())
        var_t = float(np.sum((t - mean_t) ** 2 * et) / et.sum())
        third = float(np.sum((t - mean_t) ** 3 * et) / et.sum())
        symmetry = third / var_t**1.5 if var_t > 0 else np.nan
    else:
        symmetry = np.nan

    return {
        "energy_total": total,
        "frac_energy_below_40hz": float(ef[freqs < 40.0].sum() / total),
        "frac_energy_above_100hz": float(ef[freqs > 100.0].sum() / total),
        "spectral_centroid": centroid,
        "band_concentration": float(concentration),
        "time_symmetry": float(symmetry) if np.isfinite(symmetry) else np.nan,
    }


def _extent_statistics(qmap: QMap, track: RidgeTrack, dt: float) -> dict[str, float]:
    """Peak / maximum / minimum frequency, bandwidth and duration."""
    if not track.active.any():
        return {
            "snr_qscan": float(np.sqrt(2.0 * qmap.energy.max())),
            "f_peak": np.nan,
            "f_max": np.nan,
            "f_min": np.nan,
            "bandwidth": np.nan,
            "duration": 0.0,
        }

    it, iff = np.unravel_index(np.argmax(qmap.energy), qmap.energy.shape)
    f_peak = float(qmap.frequencies[iff])

    upper = track.upper[np.isfinite(track.upper)]
    lower = track.lower[np.isfinite(track.lower)]
    # Robust extremes: the 99th/1st percentile rather than the literal max/min,
    # so one surviving noise tile cannot set the bandwidth.
    f_max = float(np.percentile(upper, 99)) if upper.size else np.nan
    f_min = float(np.percentile(lower, 1)) if lower.size else np.nan

    duration = float(track.active.sum() * dt)

    return {
        "snr_qscan": float(np.sqrt(2.0 * qmap.energy.max())),
        "f_peak": f_peak,
        "f_max": f_max,
        "f_min": f_min,
        "bandwidth": f_max - f_min if np.isfinite(f_max) and np.isfinite(f_min) else np.nan,
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_features(
    qmap: QMap,
    threshold: float,
    min_island_tiles: int = 40,
    return_track: bool = False,
) -> dict[str, float] | tuple[dict[str, float], RidgeTrack]:
    """Reduce a Q-map to the standard feature dictionary.

    The returned dictionary always contains exactly the keys in
    :data:`gwglitch.constants.FEATURE_NAMES`, in that order, with ``NaN`` for
    features that are undefined for this map (for example ``arch_period``
    when fewer than two arches are found).  Downstream code can therefore
    assume a rectangular table.
    """
    track = extract_ridge(qmap, threshold, min_island_bins=min_island_tiles)
    dt = qmap.dt

    features: dict[str, float] = {}
    features.update(_extent_statistics(qmap, track, dt))
    features.update(_arch_statistics(track, dt))
    features.update(_harmonic_statistics(qmap, track))
    features.update(_shape_statistics(track, dt))
    features.update(_energy_statistics(qmap, track))

    ordered = {name: float(features.get(name, np.nan)) for name in FEATURE_NAMES}
    missing = set(FEATURE_NAMES) - set(features)
    if missing:  # pragma: no cover - guards against a refactor dropping a key
        log.debug("features not computed, filled with NaN: %s", sorted(missing))

    if return_track:
        return ordered, track
    return ordered


def feature_frame(records: list[dict]) -> "pd.DataFrame":  # noqa: F821
    """Assemble per-trigger feature dictionaries into a tidy DataFrame."""
    import pandas as pd

    if not records:
        return pd.DataFrame(columns=list(FEATURE_NAMES))
    df = pd.DataFrame.from_records(records)
    ordered = [c for c in df.columns if c not in FEATURE_NAMES] + list(FEATURE_NAMES)
    return df.reindex(columns=ordered)


def track_to_dict(track: RidgeTrack) -> dict:
    """Serialisable view of a ridge track, for figure reproduction."""
    d = asdict(track)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
