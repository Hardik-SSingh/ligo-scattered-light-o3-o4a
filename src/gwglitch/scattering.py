"""The scattering predictor: from optic motion to arch frequency.

Physics
-------
A small fraction of the circulating light leaves the beam, strikes a nearby
surface, and recombines with the main beam.  If that surface is displaced by
``x(t)`` along the beam, the recombining field has acquired a phase

    phi(t) = 4 pi x(t) / lambda

(the factor 4 rather than 2 because the light traverses the extra path twice).
When the phase excursion exceeds a radian the modulation is strongly
non-linear and the noise appears at every harmonic of the instantaneous
frequency

    f_n(t) = (n / 2 pi) |d phi / d t| = n (2 / lambda) |v(t)| .

With ``lambda = 1064 nm`` this is ``f_n = n x 1.88 MHz per (m/s)``, i.e.
1.88 Hz for every micron per second of surface velocity.  Because the
frequency depends on ``|v|`` rather than ``v``, each cycle of the driving
motion produces **two** arches; the observed arch repetition period is half
the period of the motion.

Practical complication
----------------------
The GWOSC auxiliary channels are published in raw ADC/digital counts with no
calibration to metres.  Rather than invent one, we fit a single scale factor
``alpha`` (metres per count) per detector and epoch by maximising the
agreement between the predicted harmonic comb and the measured arch envelope
over a training set of triggers.  ``alpha`` is then frozen and applied to the
validation set.  This is the one free parameter in the auxiliary analysis and
it is reported with its uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .constants import SCATTER_FREQ_PER_VELOCITY
from .dataio import TimeSeriesData
from .morphology import RidgeTrack

log = logging.getLogger(__name__)

__all__ = [
    "ScatteringPrediction",
    "surface_velocity",
    "predict_arch_frequencies",
    "track_agreement",
    "calibrate_witness",
    "classify_slow_fast",
    "band_limited_rms",
]


# ---------------------------------------------------------------------------
# Velocity from a motion witness
# ---------------------------------------------------------------------------


def surface_velocity(
    witness: TimeSeriesData,
    band: tuple[float, float] = (0.03, 10.0),
    calibration: float = 1.0,
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Band-limited velocity implied by a displacement-like witness channel.

    Parameters
    ----------
    witness
        Time series proportional to the displacement of the scattering
        surface (e.g. a suspension damping-loop error signal).
    band
        Band-pass corners in Hz.  The lower corner removes the DC drift that
        differentiation would otherwise amplify into a spurious ramp; the
        upper corner is set below the witness Nyquist frequency.
    calibration
        Metres per count.  ``1.0`` returns velocity in counts/s, which is what
        :func:`calibrate_witness` consumes.

    Returns
    -------
    times, velocity
        GPS times and the (calibrated) velocity.
    """
    fs = witness.sample_rate
    x = np.asarray(witness.values, dtype=float)

    if detrend:
        x = signal.detrend(x, type="linear")

    lo, hi = band
    hi = min(hi, 0.45 * fs)
    if hi <= lo:
        raise ValueError(
            f"witness sampled at {fs} Hz cannot support the band {band}; "
            "its Nyquist frequency is too low"
        )

    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = signal.sosfiltfilt(sos, x)

    v = np.gradient(xf, 1.0 / fs) * float(calibration)
    return witness.times, v


@dataclass
class ScatteringPrediction:
    """Predicted arch frequencies for harmonics 1..N.

    ``frequencies`` has shape ``(n_harmonics, n_times)`` with row ``i``
    corresponding to harmonic ``harmonics[i]``.
    """

    times: np.ndarray
    frequencies: np.ndarray
    harmonics: np.ndarray
    calibration: float
    channel: str = ""

    @property
    def fundamental(self) -> np.ndarray:
        return self.frequencies[0]

    def at(self, times: np.ndarray) -> "ScatteringPrediction":
        """Interpolate onto a new time grid (e.g. the Q-map time bins)."""
        interp = np.vstack(
            [np.interp(times, self.times, row, left=np.nan, right=np.nan)
             for row in self.frequencies]
        )
        return ScatteringPrediction(
            times=np.asarray(times, dtype=float),
            frequencies=interp,
            harmonics=self.harmonics,
            calibration=self.calibration,
            channel=self.channel,
        )


def predict_arch_frequencies(
    witness: TimeSeriesData,
    max_harmonic: int = 8,
    calibration: float = 1.0,
    band: tuple[float, float] = (0.03, 10.0),
    wavelength_factor: float = SCATTER_FREQ_PER_VELOCITY,
) -> ScatteringPrediction:
    """Apply ``f_n = n (2 / lambda) |v|`` to a witness channel."""
    times, v = surface_velocity(witness, band=band, calibration=calibration)
    f1 = wavelength_factor * np.abs(v)
    harmonics = np.arange(1, int(max_harmonic) + 1)
    frequencies = np.outer(harmonics, f1)
    return ScatteringPrediction(
        times=times,
        frequencies=frequencies,
        harmonics=harmonics,
        calibration=float(calibration),
        channel=witness.channel,
    )


# ---------------------------------------------------------------------------
# Agreement between prediction and observation
# ---------------------------------------------------------------------------


def track_agreement(
    track: RidgeTrack,
    prediction: ScatteringPrediction,
    t0: float,
    tolerance_frac: float = 0.15,
    tolerance_floor: float = 1.5,
    min_points: int = 20,
) -> dict[str, float]:
    """Quantify how well a predicted harmonic comb explains an arch envelope.

    A ridge point at frequency ``f`` is counted as *explained* if some
    harmonic ``f_n`` of the prediction satisfies
    ``|f - f_n| <= max(tolerance_floor, tolerance_frac * f)``.

    Returns
    -------
    dict with keys
        ``matched_fraction``
            Fraction of supra-threshold time bins that are explained.  This is
            the headline agreement statistic.
        ``best_harmonic``
            The harmonic number that alone explains the largest fraction.
        ``pearson_r``
            Correlation between the observed envelope and the best-matching
            harmonic, over the bins where both are finite.
        ``median_residual_hz``
            Median absolute distance to the nearest harmonic.
        ``n_points``
            Number of bins used.
    """
    empty = {
        "matched_fraction": np.nan,
        "best_harmonic": np.nan,
        "pearson_r": np.nan,
        "median_residual_hz": np.nan,
        "n_points": 0.0,
    }

    obs_times = track.times + t0
    pred = prediction.at(obs_times)

    finite_obs = np.isfinite(track.upper)
    finite_pred = np.isfinite(pred.frequencies).all(axis=0)
    use = finite_obs & finite_pred
    if use.sum() < min_points:
        return empty

    f_obs = track.upper[use]
    f_pred = pred.frequencies[:, use]                     # (n_harm, n_use)

    tol = np.maximum(tolerance_floor, tolerance_frac * f_obs)
    residual = np.abs(f_pred - f_obs[None, :])            # (n_harm, n_use)
    nearest = residual.min(axis=0)

    matched = float(np.mean(nearest <= tol))
    per_harmonic = (residual <= tol[None, :]).mean(axis=1)
    best_idx = int(np.argmax(per_harmonic))
    best_harmonic = float(prediction.harmonics[best_idx])

    y = f_pred[best_idx]
    if np.std(y) > 0 and np.std(f_obs) > 0:
        r = float(np.corrcoef(f_obs, y)[0, 1])
    else:
        r = np.nan

    return {
        "matched_fraction": matched,
        "best_harmonic": best_harmonic,
        "pearson_r": r,
        "median_residual_hz": float(np.median(nearest)),
        "n_points": float(use.sum()),
    }


def calibrate_witness(
    tracks: list[RidgeTrack],
    witnesses: list[TimeSeriesData],
    t0s: list[float],
    max_harmonic: int = 8,
    band: tuple[float, float] = (0.03, 10.0),
    log10_range: tuple[float, float] = (-14.0, -6.0),
    n_grid: int = 161,
    refine: bool = True,
) -> dict[str, float]:
    """Fit the single scale factor turning witness counts into metres.

    The objective is the mean ``matched_fraction`` over the training triggers.
    It is scanned on a logarithmic grid (the plausible range spans eight
    decades) and then refined by a golden-section search on the best decade.

    Returns
    -------
    dict with ``calibration``, ``objective``, ``n_triggers`` and
    ``calibration_lo`` / ``calibration_hi``, the range over which the
    objective stays within 1 per cent of its maximum.  A narrow range means
    the fit is well constrained; a wide one is a warning that the witness does
    not actually track the scattering surface.
    """
    if not (len(tracks) == len(witnesses) == len(t0s)):
        raise ValueError("tracks, witnesses and t0s must have the same length")
    if not tracks:
        raise ValueError("no training triggers supplied")

    def objective(alpha: float) -> float:
        scores = []
        for track, wit, t0 in zip(tracks, witnesses, t0s):
            try:
                pred = predict_arch_frequencies(
                    wit, max_harmonic=max_harmonic, calibration=alpha, band=band
                )
                res = track_agreement(track, pred, t0)
            except Exception as exc:  # noqa: BLE001
                log.debug("calibration point failed: %s", exc)
                continue
            if np.isfinite(res["matched_fraction"]):
                scores.append(res["matched_fraction"])
        return float(np.mean(scores)) if scores else 0.0

    grid = np.logspace(log10_range[0], log10_range[1], n_grid)
    values = np.array([objective(a) for a in grid])
    best = int(np.argmax(values))
    alpha = float(grid[best])
    peak = float(values[best])

    if refine and 0 < best < len(grid) - 1:
        lo, hi = grid[best - 1], grid[best + 1]
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = np.log10(lo), np.log10(hi)
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = objective(10**c), objective(10**d)
        for _ in range(20):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = objective(10**c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = objective(10**d)
            if b - a < 1e-3:
                break
        alpha = float(10 ** ((a + b) / 2.0))
        peak = max(peak, objective(alpha))

    within = values >= 0.99 * peak
    lo_edge = float(grid[within].min()) if within.any() else np.nan
    hi_edge = float(grid[within].max()) if within.any() else np.nan

    log.info(
        "witness calibration: alpha = %.3e m/count, matched fraction %.3f "
        "(1%% range %.2e - %.2e, %d training triggers)",
        alpha, peak, lo_edge, hi_edge, len(tracks),
    )
    return {
        "calibration": alpha,
        "objective": peak,
        "calibration_lo": lo_edge,
        "calibration_hi": hi_edge,
        "n_triggers": float(len(tracks)),
    }


# ---------------------------------------------------------------------------
# Population labels and band-limited RMS
# ---------------------------------------------------------------------------


def classify_slow_fast(arch_period: float, split: float = 1.5) -> str:
    """Label a trigger ``slow``, ``fast`` or ``unknown`` from its arch period.

    Soni et al. (2021) identify two distinct scattering populations at LIGO:
    *slow* scattering driven by the 0.1-0.3 Hz microseism, whose arches recur
    every few seconds, and *fast* scattering driven by 1-6 Hz anthropogenic
    and control-loop motion, whose arches recur several times per second.
    The split is on the arch repetition period, which is half the period of
    the driving motion.
    """
    if not np.isfinite(arch_period):
        return "unknown"
    return "slow" if arch_period > split else "fast"


def driving_frequency(arch_period: float) -> float:
    """Frequency of the motion implied by an arch repetition period.

    Two arches per cycle, so ``f_drive = 1 / (2 T_arch)``.
    """
    if not np.isfinite(arch_period) or arch_period <= 0:
        return np.nan
    return 1.0 / (2.0 * arch_period)


def band_limited_rms(
    data: TimeSeriesData,
    band: tuple[float, float],
    stride: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Band-limited RMS of a time series, in ``stride``-second bins.

    Used to reproduce, from full-rate channels, the same quantity that the
    ``*_BLRMS_*`` trend channels provide -- which lets a result derived from
    the trends be cross-checked against one derived from the raw data.
    """
    fs = data.sample_rate
    lo, hi = band
    hi = min(hi, 0.45 * fs)
    if hi <= lo:
        raise ValueError(f"band {band} is not representable at {fs} Hz")

    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(data.values, dtype=float))

    n_per_bin = max(int(round(stride * fs)), 1)
    n_bins = len(filtered) // n_per_bin
    if n_bins == 0:
        raise ValueError("series is shorter than one stride")

    trimmed = filtered[: n_bins * n_per_bin].reshape(n_bins, n_per_bin)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    times = data.t0 + (np.arange(n_bins) + 0.5) * stride
    return times, rms
