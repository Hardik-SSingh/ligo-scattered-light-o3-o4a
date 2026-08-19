"""Physical and instrumental constants used throughout the analysis.

Only quantities that are genuinely fixed live here.  Anything that is an
analysis *choice* belongs in ``config/analysis.yaml`` so that it can be
varied without touching code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------

#: Advanced LIGO pre-stabilised-laser wavelength (Nd:YAG), metres.
LASER_WAVELENGTH_M: float = 1.064e-6

#: Round-trip phase modulation imparted by a scattering surface displaced by
#: ``x`` is ``4 * pi * x / lambda``.  Differentiating and dividing by 2*pi
#: gives the instantaneous frequency of the fundamental scattering arch,
#: ``f_1 = (2 / lambda) * |v|``.  This is that prefactor, in Hz per (m/s).
SCATTER_FREQ_PER_VELOCITY: float = 2.0 / LASER_WAVELENGTH_M  # ~1.8797e6 Hz/(m/s)

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

#: Offset between POSIX time and GPS time at the GPS epoch (1980-01-06).
GPS_UNIX_OFFSET: int = 315964800

#: Leap seconds accumulated between 1980-01-06 and the O3/O4 era.  Constant
#: across both runs (no leap second has been inserted since 2016-12-31).
LEAP_SECONDS_O3_O4: int = 18

SECONDS_PER_HOUR: int = 3600
SECONDS_PER_DAY: int = 86400

# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

DETECTORS: tuple[str, ...] = ("H1", "L1")

DETECTOR_NAMES: dict[str, str] = {
    "H1": "LIGO Hanford",
    "L1": "LIGO Livingston",
}

# ---------------------------------------------------------------------------
# Gravity Spy
# ---------------------------------------------------------------------------

#: The 22 classes used by the Gravity Spy model in the Zenodo 5649212 release.
GRAVITYSPY_CLASSES: tuple[str, ...] = (
    "1080Lines",
    "1400Ripples",
    "Air_Compressor",
    "Blip",
    "Chirp",
    "Extremely_Loud",
    "Helix",
    "Koi_Fish",
    "Light_Modulation",
    "Low_Frequency_Burst",
    "Low_Frequency_Lines",
    "No_Glitch",
    "None_of_the_Above",
    "Paired_Doves",
    "Power_Line",
    "Repeating_Blips",
    "Scattered_Light",
    "Scratchy",
    "Tomte",
    "Violin_Mode",
    "Wandering_Line",
    "Whistle",
)

#: Columns of the Gravity Spy ML-classification CSVs that are not class
#: confidence scores.
GRAVITYSPY_META_COLUMNS: tuple[str, ...] = (
    "event_time",
    "ifo",
    "peak_time",
    "peak_time_ns",
    "start_time",
    "start_time_ns",
    "duration",
    "peak_frequency",
    "central_freq",
    "bandwidth",
    "channel",
    "amplitude",
    "snr",
    "q_value",
    "gravityspy_id",
    "ml_label",
    "ml_confidence",
    "url1",
    "url2",
    "url3",
    "url4",
)

# ---------------------------------------------------------------------------
# Morphology feature vector
# ---------------------------------------------------------------------------

#: Ordered names of the features produced by ``morphology.extract_features``.
#: Kept here (rather than inferred) so that a feature table written by one
#: version of the code can be validated against another.
FEATURE_NAMES: tuple[str, ...] = (
    "snr_qscan",
    "energy_total",
    "f_peak",
    "f_max",
    "f_min",
    "bandwidth",
    "duration",
    "n_arches",
    "arch_period",
    "arch_period_scatter",
    "n_harmonics",
    "harmonic_spacing",
    "ridge_slope_rms",
    "ridge_curvature_rms",
    "time_symmetry",
    "frac_energy_below_40hz",
    "frac_energy_above_100hz",
    "spectral_centroid",
    "band_concentration",
    "upward_downward_ratio",
)
