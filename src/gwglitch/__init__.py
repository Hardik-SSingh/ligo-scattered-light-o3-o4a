"""gwglitch -- scattered-light transient characterisation for LIGO O3 and O4a.

The package is organised as a linear pipeline:

    catalog   -> obtain and filter the Gravity Spy trigger list (O3 only)
    dataio    -> fetch strain and auxiliary time series, with a local cache
    qscan     -> compute normalised-energy Q-transform maps
    morphology-> reduce a Q-map to a vector of scattering-specific features
    scattering-> the f = 2 n |v| / lambda predictor and its calibration
    auxcorr   -> ground-motion / glitch-rate correlation statistics
    identify  -> strain-only classifier trained on O3, applied to O4a
    population-> O3 vs O4a comparison and hypothesis tests
    plotting  -> every figure that appears in the write-up

Nothing in this package requires LIGO credentials: all inputs are public
(GWOSC strain, GWOSC auxiliary channels via NDS, Gravity Spy on Zenodo).
"""

from .config import Config, load_config  # noqa: F401

__version__ = "0.1.0"
__author__ = "Hardik Sher Singh"

__all__ = ["Config", "load_config", "__version__"]
