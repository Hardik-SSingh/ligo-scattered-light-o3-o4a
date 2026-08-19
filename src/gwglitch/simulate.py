"""Synthetic scattering signals with known ground truth.

Every quantity the morphology module measures -- arch period, maximum arch
frequency, harmonic count -- is a statement about a physical model.  Without a
signal whose parameters are known exactly, there is no way to tell a genuine
measurement from a plausible-looking artefact of the ridge tracker.

This module generates such signals.  The model is the standard one:

    x(t) = X0 sin(2 pi f_m t + phi)                  surface displacement
    h(t) = sum_n K_n sin(n * 4 pi x(t) / lambda)     scattered-light strain

The sum over ``n`` is over *bounce orders*: light that makes ``n`` round trips
between the main optic and the scattering surface accumulates ``n`` times the
phase, and each successive bounce is weaker.  This is where the harmonic
arches come from, and it is worth being explicit about it, because there is a
tempting wrong answer: the Jacobi-Anger expansion of a single ``sin(beta sin
theta)`` term also produces harmonics, but they are harmonics of the *motion*
frequency ``f_m``, and they terminate at ``k ~ beta`` -- that is, at exactly
the apex frequency ``beta f_m`` of the single arch.  A one-bounce model
therefore produces one arch, however large ``beta`` is, and simulating "many
harmonics" by cranking up the displacement only produces one very tall arch.

The predictions that follow, and that ``tests/test_morphology.py`` checks:

    f_n(t)  = n (2 / lambda) |dx/dt|
    f_1,apex = (2 / lambda) 2 pi f_m X0 = 4 pi f_m X0 / lambda
    f_apex   = N_bounce * f_1,apex             top of the arch train
    T_arch   = 1 / (2 f_m)                     two arches per motion cycle
    harmonic spacing at the apex = f_1,apex
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import LASER_WAVELENGTH_M, SCATTER_FREQ_PER_VELOCITY
from .dataio import TimeSeriesData

__all__ = [
    "ScatteringInjection",
    "make_scattering_strain",
    "expected_arch_period",
    "expected_max_arch_frequency",
    "expected_harmonic_count",
]


@dataclass
class ScatteringInjection:
    """Parameters of a synthetic scattering event and its predictions."""

    displacement_amplitude: float   # X0, metres
    motion_frequency: float         # f_m, Hz
    coupling: float                 # K, strain units
    duration: float                 # s, total length of the generated stretch
    sample_rate: float              # Hz
    event_duration: float = 16.0    # s, how long the scattering itself lasts
    n_bounces: int = 4              # number of arch harmonics
    bounce_decay: float = 0.6       # amplitude ratio between successive bounces
    t0: float = 0.0
    phase: float = 0.0
    noise_amplitude: float = 1.0
    seed: int = 0

    @property
    def event_centre(self) -> float:
        return self.t0 + self.duration / 2.0

    # -- analytic predictions -------------------------------------------
    @property
    def peak_velocity(self) -> float:
        return 2.0 * np.pi * self.motion_frequency * self.displacement_amplitude

    @property
    def fundamental_max_frequency(self) -> float:
        return SCATTER_FREQ_PER_VELOCITY * self.peak_velocity

    @property
    def arch_period(self) -> float:
        return expected_arch_period(self.motion_frequency)

    @property
    def phase_modulation_index(self) -> float:
        """beta = 4 pi X0 / lambda, the one-bounce modulation index in radians.

        Note that ``beta * f_m`` equals :attr:`fundamental_max_frequency`; the
        two are the same statement seen from the frequency and the time side.
        """
        return 4.0 * np.pi * self.displacement_amplitude / LASER_WAVELENGTH_M

    @property
    def harmonic_count(self) -> int:
        """Number of arch harmonics, i.e. the number of bounce orders."""
        return int(self.n_bounces)

    @property
    def max_arch_frequency(self) -> float:
        """Apex frequency of the highest arch."""
        return self.n_bounces * self.fundamental_max_frequency

    @property
    def harmonic_spacing(self) -> float:
        """Frequency spacing of the harmonic comb at the apex."""
        return self.fundamental_max_frequency


def expected_arch_period(motion_frequency: float) -> float:
    """Arch repetition period for a sinusoidal motion at ``motion_frequency``.

    The scattering frequency follows ``|v|``, so it peaks twice per cycle.
    """
    return 1.0 / (2.0 * motion_frequency)


def expected_max_arch_frequency(
    displacement_amplitude: float, motion_frequency: float, harmonic: int = 1
) -> float:
    """Apex frequency of harmonic ``n`` for a sinusoidal surface motion."""
    v_peak = 2.0 * np.pi * motion_frequency * displacement_amplitude
    return harmonic * SCATTER_FREQ_PER_VELOCITY * v_peak


def displacement_for_apex(
    apex_frequency: float, motion_frequency: float, harmonic: int = 1
) -> float:
    """Displacement amplitude that puts harmonic ``n`` at ``apex_frequency``.

    The inverse of :func:`expected_max_arch_frequency`.  Useful when designing
    an injection: pick where you want the arches, get the motion you need.
    """
    return (
        apex_frequency
        * LASER_WAVELENGTH_M
        / (4.0 * np.pi * motion_frequency * harmonic)
    )


def make_scattering_strain(
    injection: ScatteringInjection,
    add_noise: bool = True,
) -> TimeSeriesData:
    """Generate a strain-like series containing a synthetic scattering event.

    The scattering term is confined to a Tukey-windowed stretch of
    ``injection.event_duration`` seconds centred in a much longer stretch of
    noise.  This is not cosmetic.  The Q-transform whitens using an amplitude
    spectral density estimated from the data it is given; a scattering train
    that fills the whole window is indistinguishable from the noise floor and
    is whitened away entirely, which is exactly what happened the first time
    this simulator was written.  Keeping the event to a modest fraction of the
    window -- as a real transient is -- leaves the median-averaged ASD
    estimate dominated by noise, and the arches survive.

    Parameters
    ----------
    add_noise
        Add unit-variance white Gaussian noise.  White noise is used rather
        than a coloured aLIGO-like spectrum because after whitening the two
        are equivalent for the Q-transform, and white noise makes the expected
        normalised-energy distribution exactly ``Exp(1)`` -- which is what the
        threshold in the configuration is calibrated against.
    """
    from scipy.signal import windows

    n = int(round(injection.duration * injection.sample_rate))
    dt = 1.0 / injection.sample_rate
    t = np.arange(n) * dt

    x = injection.displacement_amplitude * np.sin(
        2.0 * np.pi * injection.motion_frequency * t + injection.phase
    )
    phi = 4.0 * np.pi * x / LASER_WAVELENGTH_M

    h = np.zeros(n)
    for order in range(1, int(injection.n_bounces) + 1):
        h += (
            injection.coupling
            * injection.bounce_decay ** (order - 1)
            * np.sin(order * phi)
        )

    n_event = min(int(round(injection.event_duration * injection.sample_rate)), n)
    start = (n - n_event) // 2
    gate = np.zeros(n)
    gate[start : start + n_event] = windows.tukey(n_event, alpha=0.15)
    h = h * gate

    if add_noise:
        rng = np.random.default_rng(injection.seed)
        h = h + injection.noise_amplitude * rng.standard_normal(n)

    return TimeSeriesData(
        values=h,
        t0=injection.t0,
        dt=dt,
        channel="SIM:SCATTERING_STRAIN",
    )


def make_witness(
    injection: ScatteringInjection,
    sample_rate: float = 256.0,
    counts_per_metre: float = 1.0e9,
    noise_counts: float = 0.0,
) -> TimeSeriesData:
    """The displacement witness a suspension damping loop would record.

    ``counts_per_metre`` is the inverse of the calibration constant that
    :func:`gwglitch.scattering.calibrate_witness` is designed to recover, so a
    round trip through the two is a useful end-to-end check: fitting the
    witness produced here should return ``1 / counts_per_metre``.
    """
    n = int(round(injection.duration * sample_rate))
    dt = 1.0 / sample_rate
    t = np.arange(n) * dt

    x = injection.displacement_amplitude * np.sin(
        2.0 * np.pi * injection.motion_frequency * t + injection.phase
    )
    counts = x * counts_per_metre

    if noise_counts > 0:
        rng = np.random.default_rng(injection.seed + 1)
        counts = counts + noise_counts * rng.standard_normal(n)

    return TimeSeriesData(
        values=counts,
        t0=injection.t0,
        dt=dt,
        channel="SIM:WITNESS_DISPLACEMENT",
        unit="count",
    )


def slow_scattering_example(seed: int = 0) -> ScatteringInjection:
    """A canonical *slow* scattering event.

    Microseism-driven: 0.15 Hz motion of 8 um amplitude puts the fundamental
    apex at 14.2 Hz, so a four-bounce train reaches 57 Hz, with arches
    recurring every 3.33 s.  Several microns of *relative* motion between the
    main chain and the reaction chain is what Soni et al. (2021) report during
    high microseism, so this is a realistic rather than a convenient number.

    The amplitude matters: the analysis band starts at 8 Hz, so an injection
    whose fundamental apex lands below that contributes nothing however loud
    it is.  Use :func:`displacement_for_apex` when changing these values.
    """
    return ScatteringInjection(
        displacement_amplitude=8.0e-6,
        motion_frequency=0.15,
        coupling=6.0,
        duration=64.0,
        event_duration=16.0,
        n_bounces=4,
        sample_rate=1024.0,
        noise_amplitude=1.0,
        seed=seed,
    )


def fast_scattering_example(seed: int = 0) -> ScatteringInjection:
    """A canonical *fast* scattering event.

    2 Hz anthropogenic motion of 640 nm amplitude: fundamental apex at 15.1 Hz,
    three bounces reaching 45 Hz, arches recurring every 0.25 s.
    """
    return ScatteringInjection(
        displacement_amplitude=6.4e-7,
        motion_frequency=2.0,
        coupling=6.0,
        duration=64.0,
        event_duration=8.0,
        n_bounces=3,
        sample_rate=1024.0,
        noise_amplitude=1.0,
        seed=seed + 101,
    )
