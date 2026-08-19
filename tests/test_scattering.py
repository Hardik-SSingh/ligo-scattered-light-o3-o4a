"""The scattering predictor and its calibration.

The important test here is the round trip: generate a witness in known counts
per metre, hand it to the calibration fit without telling it the answer, and
check the answer comes back.  If that works, the same machinery applied to a
real GWOSC auxiliary channel -- which is published in uncalibrated counts --
is doing something meaningful rather than fitting noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwglitch.constants import LASER_WAVELENGTH_M, SCATTER_FREQ_PER_VELOCITY
from gwglitch.scattering import (
    band_limited_rms,
    calibrate_witness,
    classify_slow_fast,
    driving_frequency,
    predict_arch_frequencies,
    surface_velocity,
    track_agreement,
)
from gwglitch.simulate import (
    displacement_for_apex,
    expected_arch_period,
    expected_max_arch_frequency,
)


# ---------------------------------------------------------------------------
# The physics constants themselves
# ---------------------------------------------------------------------------


def test_scatter_prefactor():
    """1 micron per second of surface velocity is 1.88 Hz of arch frequency."""
    assert SCATTER_FREQ_PER_VELOCITY == pytest.approx(2.0 / LASER_WAVELENGTH_M)
    assert SCATTER_FREQ_PER_VELOCITY * 1e-6 == pytest.approx(1.88, rel=0.01)


def test_arch_period_is_half_the_motion_period():
    assert expected_arch_period(0.15) == pytest.approx(1.0 / 0.30)
    assert driving_frequency(expected_arch_period(0.15)) == pytest.approx(0.15)


def test_displacement_for_apex_inverts_the_apex_formula():
    x0 = displacement_for_apex(apex_frequency=20.0, motion_frequency=0.2)
    assert expected_max_arch_frequency(x0, 0.2, harmonic=1) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Velocity and prediction
# ---------------------------------------------------------------------------


def test_surface_velocity_recovers_peak_velocity(slow_witness, slow_injection):
    _, v = surface_velocity(
        slow_witness, band=(0.03, 10.0), calibration=1.0e-9
    )
    # Ignore the filter transients at the ends.
    n = len(v)
    interior = v[n // 10 : -n // 10]
    assert np.max(np.abs(interior)) == pytest.approx(
        slow_injection.peak_velocity, rel=0.10
    )


def test_prediction_apex_matches_theory(slow_witness, slow_injection):
    pred = predict_arch_frequencies(
        slow_witness, max_harmonic=4, calibration=1.0e-9, band=(0.03, 10.0)
    )
    n = pred.frequencies.shape[1]
    interior = pred.fundamental[n // 10 : -n // 10]
    assert np.max(interior) == pytest.approx(
        slow_injection.fundamental_max_frequency, rel=0.10
    )
    # Harmonics are exact integer multiples by construction.
    assert np.allclose(pred.frequencies[3], 4.0 * pred.frequencies[0])


def test_agreement_is_high_for_the_true_calibration(
    slow_analysis, slow_witness, slow_injection
):
    _, _, track = slow_analysis
    pred = predict_arch_frequencies(
        slow_witness, max_harmonic=8, calibration=1.0e-9, band=(0.03, 10.0)
    )
    result = track_agreement(track, pred, slow_injection.event_centre)
    assert result["matched_fraction"] > 0.7
    assert result["pearson_r"] > 0.7


def test_agreement_collapses_for_a_wrong_calibration(
    slow_analysis, slow_witness, slow_injection
):
    """A predictor that is wrong by three decades must not still "agree"."""
    _, _, track = slow_analysis
    good = track_agreement(
        track,
        predict_arch_frequencies(slow_witness, 8, 1.0e-9, (0.03, 10.0)),
        slow_injection.event_centre,
    )
    bad = track_agreement(
        track,
        predict_arch_frequencies(slow_witness, 8, 1.0e-12, (0.03, 10.0)),
        slow_injection.event_centre,
    )
    assert bad["matched_fraction"] < 0.5 * good["matched_fraction"]


# ---------------------------------------------------------------------------
# Calibration round trip
# ---------------------------------------------------------------------------


def test_calibration_round_trip(slow_analysis, slow_witness, slow_injection):
    _, _, track = slow_analysis
    fit = calibrate_witness(
        [track],
        [slow_witness],
        [slow_injection.event_centre],
        max_harmonic=8,
        log10_range=(-12.0, -6.0),
        n_grid=61,
    )
    assert fit["objective"] > 0.7
    # The witness was generated at 1e9 counts per metre, so the calibration
    # that should come back is 1e-9 metres per count.
    assert fit["calibration"] == pytest.approx(1.0e-9, rel=0.30)


def test_calibration_requires_training_data():
    with pytest.raises(ValueError):
        calibrate_witness([], [], [])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_classify_slow_fast_handles_nan():
    assert classify_slow_fast(np.nan) == "unknown"
    assert classify_slow_fast(5.0, split=1.5) == "slow"
    assert classify_slow_fast(0.2, split=1.5) == "fast"


def test_band_limited_rms_matches_a_known_sinusoid():
    from gwglitch.dataio import TimeSeriesData

    fs = 256.0
    t = np.arange(int(60 * fs)) / fs
    amplitude = 3.0
    data = TimeSeriesData(
        values=amplitude * np.sin(2 * np.pi * 0.2 * t), t0=0.0, dt=1.0 / fs
    )
    times, rms = band_limited_rms(data, band=(0.1, 0.3), stride=5.0)
    # RMS of a sinusoid is A / sqrt(2); drop the filter transients.
    assert np.median(rms[1:-1]) == pytest.approx(amplitude / np.sqrt(2), rel=0.10)
    assert len(times) == len(rms)


def test_band_limited_rms_rejects_an_impossible_band():
    from gwglitch.dataio import TimeSeriesData

    data = TimeSeriesData(values=np.zeros(1024), t0=0.0, dt=1.0)  # 1 Hz
    with pytest.raises(ValueError):
        band_limited_rms(data, band=(2.0, 8.0))
