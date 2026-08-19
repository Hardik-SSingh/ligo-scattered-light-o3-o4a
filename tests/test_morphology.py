"""Morphology extractor against injections with known parameters.

Tolerances here are not arbitrary.  Where a measurement is expected to be
biased -- ``f_max`` is, because a Q tile of bandwidth ``f/Q`` extends above the
true apex of the arch -- the test asserts the bias is within the expected size
rather than pretending it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwglitch.constants import FEATURE_NAMES
from gwglitch.morphology import extract_features


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_feature_dict_is_rectangular(slow_analysis):
    _, features, _ = slow_analysis
    assert list(features) == list(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


def test_pure_noise_has_no_ridge(noise_analysis, cfg):
    """The threshold must sit above the noise maximum of a whole Q-map.

    This is the test that caught the original threshold of Z = 5.5, at which
    a noise-only map produced hundreds of supra-threshold bins and ``f_max``
    saturated at the top of the analysis band.
    """
    qmap, features, track = noise_analysis
    assert qmap.peak_energy < float(cfg.qtransform.energy_threshold)
    assert track.n_active == 0
    assert features["duration"] == 0.0
    assert np.isnan(features["f_max"])


# ---------------------------------------------------------------------------
# Slow scattering
# ---------------------------------------------------------------------------


def test_slow_scattering_is_detected(slow_analysis):
    qmap, features, track = slow_analysis
    assert qmap.peak_energy > 100.0
    assert features["snr_qscan"] > 20.0
    assert track.n_active > 100


def test_slow_scattering_arch_period(slow_analysis, slow_injection):
    _, features, _ = slow_analysis
    measured = features["arch_period"]
    assert np.isfinite(measured)
    assert measured == pytest.approx(slow_injection.arch_period, rel=0.15)


def test_slow_scattering_harmonic_spacing(slow_analysis, slow_injection):
    """The comb spacing at the apex measures the fundamental directly."""
    _, features, _ = slow_analysis
    assert features["n_harmonics"] >= 2
    assert features["harmonic_spacing"] == pytest.approx(
        slow_injection.harmonic_spacing, rel=0.20
    )


def test_slow_scattering_fmax_is_high_but_bounded(slow_analysis, slow_injection):
    """``f_max`` overshoots the true apex by about half a tile bandwidth."""
    _, features, _ = slow_analysis
    truth = slow_injection.max_arch_frequency
    assert truth * 0.8 < features["f_max"] < truth * 1.35


def test_slow_scattering_energy_is_low_frequency(slow_analysis):
    _, features, _ = slow_analysis
    assert features["frac_energy_below_40hz"] > 0.5
    assert features["frac_energy_above_100hz"] < 0.05


def test_arches_are_time_symmetric(slow_analysis):
    """An arch train rises and falls equally; a whistle or blip does not."""
    _, features, _ = slow_analysis
    assert features["upward_downward_ratio"] == pytest.approx(1.0, abs=0.35)


# ---------------------------------------------------------------------------
# Fast scattering
# ---------------------------------------------------------------------------


def test_fast_scattering_arch_period(fast_analysis, fast_injection):
    """Fast arches are only resolved if the Q cap keeps tiles short.

    Without the cap in ``config/analysis.yaml`` the multi-Q search selects
    Q ~ 45, whose tiles last 0.16 s at 45 Hz -- comparable to the 0.25 s arch
    period -- and the measured period comes out an order of magnitude too
    large.
    """
    _, features, _ = fast_analysis
    assert features["arch_period"] == pytest.approx(
        fast_injection.arch_period, rel=0.20
    )


def test_fast_scattering_counts_many_arches(fast_analysis, fast_injection):
    _, features, _ = fast_analysis
    expected = fast_injection.event_duration / fast_injection.arch_period
    assert features["n_arches"] == pytest.approx(expected, rel=0.35)


def test_slow_and_fast_are_separated_by_arch_period(
    slow_analysis, fast_analysis, cfg
):
    """The slow/fast split must actually split the two populations."""
    from gwglitch.scattering import classify_slow_fast

    split = float(cfg.scattering.slow_fast_split_period)
    _, slow_features, _ = slow_analysis
    _, fast_features, _ = fast_analysis

    assert classify_slow_fast(slow_features["arch_period"], split) == "slow"
    assert classify_slow_fast(fast_features["arch_period"], split) == "fast"


def test_driving_frequency_lands_in_the_right_band(
    slow_analysis, fast_analysis, cfg
):
    from gwglitch.scattering import driving_frequency

    micro = tuple(cfg.scattering.microseism_band)
    anthro = tuple(cfg.scattering.anthropogenic_band)

    _, slow_features, _ = slow_analysis
    _, fast_features, _ = fast_analysis

    f_slow = driving_frequency(slow_features["arch_period"])
    f_fast = driving_frequency(fast_features["arch_period"])

    assert micro[0] <= f_slow <= micro[1]
    assert anthro[0] <= f_fast <= anthro[1]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_threshold_above_peak_gives_empty_track(slow_analysis):
    qmap, _, _ = slow_analysis
    features = extract_features(qmap, qmap.peak_energy * 2.0)
    assert features["duration"] == 0.0
    assert np.isnan(features["arch_period"])


def test_features_never_raise_on_degenerate_maps(slow_analysis):
    """Cropping to a sliver must return NaNs, not an exception."""
    qmap, _, _ = slow_analysis
    sliver = qmap.crop_time(0.1)
    features = extract_features(sliver, 20.0)
    assert list(features) == list(FEATURE_NAMES)
