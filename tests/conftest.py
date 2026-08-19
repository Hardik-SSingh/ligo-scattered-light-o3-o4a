"""Shared fixtures.

The Q-transform of a 64 s, 1 kHz stretch takes a few hundred milliseconds, and
several tests need the same maps, so they are computed once per session.
"""

from __future__ import annotations

import pytest

from gwglitch import load_config
from gwglitch.morphology import extract_features
from gwglitch.qscan import compute_qmap
from gwglitch.simulate import (
    ScatteringInjection,
    fast_scattering_example,
    make_scattering_strain,
    make_witness,
    slow_scattering_example,
)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def slow_injection():
    return slow_scattering_example(seed=3)


@pytest.fixture(scope="session")
def fast_injection():
    return fast_scattering_example(seed=3)


def _analyse(injection, cfg):
    data = make_scattering_strain(injection)
    qmap = compute_qmap(data, injection.event_centre, cfg)
    features, track = extract_features(
        qmap,
        float(cfg.qtransform.energy_threshold),
        int(cfg.qtransform.min_island_tiles),
        return_track=True,
    )
    return qmap, features, track


@pytest.fixture(scope="session")
def slow_analysis(slow_injection, cfg):
    return _analyse(slow_injection, cfg)


@pytest.fixture(scope="session")
def fast_analysis(fast_injection, cfg):
    return _analyse(fast_injection, cfg)


@pytest.fixture(scope="session")
def noise_analysis(cfg):
    injection = ScatteringInjection(
        displacement_amplitude=0.0,
        motion_frequency=0.15,
        coupling=0.0,
        duration=64.0,
        sample_rate=1024.0,
        seed=11,
    )
    return _analyse(injection, cfg)


@pytest.fixture(scope="session")
def slow_witness(slow_injection):
    return make_witness(slow_injection, sample_rate=256.0, counts_per_metre=1.0e9)
