"""Q-transform ("Q-scan") computation and storage.

The Q-transform tiles the time-frequency plane with constant-Q bisquare
windows and reports, for each tile, the normalised energy

    Z = |X|^2 / <|X|^2>

where the mean is taken over tiles at the same frequency.  For stationary
Gaussian noise, Z is exponentially distributed with unit mean, so
``sqrt(2 Z)`` is the familiar single-tile SNR.  Scattered light appears as a
sequence of narrow, smoothly curved ridges ("arches") in this plane.

We call ``gwpy.timeseries.TimeSeries.q_transform`` rather than
``gwdetchar.scattering``/``gwdetchar.omega`` because gwdetchar's dependency
chain (lalsuite, dqsegdb2, gwdatafind) has no Windows build; the underlying
transform is identical.  See docs/DECISIONS.md (D-04).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .dataio import DataFetcher, TimeSeriesData
from .utils import hash_key

log = logging.getLogger(__name__)

__all__ = ["QMap", "compute_qmap", "qmap_for_trigger"]


@dataclass
class QMap:
    """A normalised-energy Q-transform map.

    Attributes
    ----------
    energy
        Array of shape ``(n_time, n_freq)`` of normalised energy Z.
    times
        GPS times of the time bins (length ``n_time``).
    frequencies
        Centre frequencies of the frequency bins (length ``n_freq``).
    q
        The Q value selected by the transform's own search.
    t0
        Reference (trigger) time; ``times - t0`` is the plotted abscissa.
    """

    energy: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    q: float
    t0: float
    detector: str = ""
    meta: dict | None = None

    # -- geometry --------------------------------------------------------
    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.times)))

    @property
    def df(self) -> float:
        return float(np.median(np.diff(self.frequencies)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.energy.shape

    @property
    def relative_times(self) -> np.ndarray:
        return self.times - self.t0

    @property
    def peak_energy(self) -> float:
        return float(self.energy.max())

    @property
    def peak_snr(self) -> float:
        """Single-tile SNR of the loudest tile, ``sqrt(2 Z_max)``."""
        return float(np.sqrt(2.0 * self.energy.max()))

    def crop_time(self, half_width: float) -> "QMap":
        mask = np.abs(self.times - self.t0) <= half_width
        if mask.sum() < 8:
            raise ValueError("crop_time left fewer than 8 time bins")
        return QMap(
            energy=self.energy[mask],
            times=self.times[mask],
            frequencies=self.frequencies,
            q=self.q,
            t0=self.t0,
            detector=self.detector,
            meta=self.meta,
        )

    def band(self, fmin: float, fmax: float) -> "QMap":
        mask = (self.frequencies >= fmin) & (self.frequencies <= fmax)
        return QMap(
            energy=self.energy[:, mask],
            times=self.times,
            frequencies=self.frequencies[mask],
            q=self.q,
            t0=self.t0,
            detector=self.detector,
            meta=self.meta,
        )

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        """Write via a temporary file and an atomic rename.

        As in :meth:`gwglitch.dataio.TimeSeriesData.save`, the temporary file
        is passed as an open handle: ``numpy.savez_compressed`` would otherwise
        append ``.npz`` to the temporary name and the rename would fail.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                energy=self.energy.astype(np.float32),
                times=self.times.astype(np.float64),
                frequencies=self.frequencies.astype(np.float64),
                meta=np.array(
                    json.dumps(
                        {
                            "q": self.q,
                            "t0": self.t0,
                            "detector": self.detector,
                            "meta": self.meta or {},
                        }
                    )
                ),
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "QMap":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return cls(
                energy=f["energy"].astype(np.float64),
                times=f["times"],
                frequencies=f["frequencies"],
                q=meta["q"],
                t0=meta["t0"],
                detector=meta.get("detector", ""),
                meta=meta.get("meta") or {},
            )


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_qmap(
    data: TimeSeriesData,
    t0: float,
    cfg: Config,
    outseg_half_width: float | None = None,
) -> QMap:
    """Q-transform ``data`` and return the map around ``t0``.

    ``data`` must extend well beyond the requested output segment: whitening
    and the constant-Q filters both need padding, and the resulting transients
    are of exactly the sort this analysis would otherwise mistake for arches.
    The configuration enforces ``window_half_width > qscan_half_width``.
    """
    qt = cfg.qtransform
    half = float(outseg_half_width or cfg.strain.qscan_half_width)

    available = data.duration / 2.0
    if half >= available:
        raise ValueError(
            f"requested output half-width {half} s but only {available:.1f} s "
            "of data either side of the trigger were supplied"
        )

    from gwpy.segments import Segment

    ts = data.to_gwpy()

    # The high-pass is applied *inside* the whitening filter rather than as a
    # separate pass, so there is only one filter transient to pad away.
    spec = ts.q_transform(
        qrange=tuple(qt.q_range),
        frange=tuple(qt.frequency_range),
        gps=t0,
        outseg=Segment(t0 - half, t0 + half),
        tres=float(qt.tres),
        fres=float(qt.fres),
        whiten=bool(qt.whiten),
        highpass=float(cfg.strain.highpass) or None,
        logf=False,
    )

    energy = np.asarray(spec.value, dtype=np.float64)          # (n_time, n_freq)
    times = np.asarray(spec.times.value, dtype=np.float64)
    freqs = np.asarray(spec.frequencies.value, dtype=np.float64)

    q_used = float(getattr(spec, "q", np.nan))
    if not np.isfinite(q_used):
        q_used = float(np.mean(qt.q_range))

    return QMap(
        energy=energy,
        times=times,
        frequencies=freqs,
        q=q_used,
        t0=float(t0),
        detector=data.channel.split(":")[0] if ":" in data.channel else "",
        meta={
            "highpass": float(cfg.strain.highpass),
            "whiten": bool(qt.whiten),
            "qrange": list(qt.q_range),
            "frange": list(qt.frequency_range),
            "sample_rate": data.sample_rate,
        },
    )


def qmap_for_trigger(
    fetcher: DataFetcher,
    cfg: Config,
    detector: str,
    t0: float,
    cache: bool = True,
) -> QMap:
    """Fetch strain around ``t0`` and return its Q-map, with caching.

    Caching Q-maps as well as strain roughly halves the cost of a re-run: the
    transform of a 64 s, 4 kHz window is a few hundred milliseconds, which is
    the dominant CPU cost once the strain cache is warm.
    """
    key = hash_key(
        "qmap",
        detector,
        round(float(t0), 4),
        list(cfg.qtransform.q_range),
        list(cfg.qtransform.frequency_range),
        float(cfg.qtransform.tres),
        float(cfg.qtransform.fres),
        float(cfg.strain.qscan_half_width),
        float(cfg.strain.highpass),
        int(cfg.strain.sample_rate),
    )
    path = fetcher.cache_root / "qmap" / detector / f"{key}.npz"
    if cache and path.exists():
        return QMap.load(path)

    half = float(cfg.strain.window_half_width)
    data = fetcher.strain(detector, t0 - half, t0 + half)
    qmap = compute_qmap(data, t0, cfg)
    if cache:
        qmap.save(path)
    return qmap
