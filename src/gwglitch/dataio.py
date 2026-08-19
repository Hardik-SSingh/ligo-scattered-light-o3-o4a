"""Data access: GWOSC strain, GWOSC auxiliary channels over NDS, segments.

Everything here is wrapped in an on-disk cache.  The analysis makes tens of
thousands of small requests to remote servers; without a cache, a re-run after
a crash costs another full night of downloading, and iterating on the feature
extractor would be impossible.

Cache layout
------------
``data/cache/strain/<det>/<hash>.npz``   one array + metadata per window
``data/cache/aux/<channel>/<hash>.npz``  ditto for auxiliary channels
``data/cache/segments/<name>.json``      analysis-ready segment lists

The hash is computed from (channel, start, end, sample_rate), so a change to
any of them is a cache miss rather than a silently wrong hit.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import Config
from .utils import hash_key, retry

log = logging.getLogger(__name__)

__all__ = [
    "TimeSeriesData",
    "DataFetcher",
    "SegmentList",
    "BlockUnavailable",
    "segments_livetime_per_hour",
]


class BlockUnavailable(RuntimeError):
    """A GWOSC file block is not served at all.

    Distinguished from a transient network failure because the response is
    different: a transient failure is worth retrying, a genuine data gap is
    not.  Every event inside a missing block would otherwise burn the full
    retry budget twice over -- once on the block, once on the fallback -- and a
    single bad block holds a lot of events.
    """


# ---------------------------------------------------------------------------
# A minimal, dependency-light time-series container
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesData:
    """Uniformly sampled data plus the metadata needed to interpret it.

    We do not pass ``gwpy.TimeSeries`` around because it is expensive to
    pickle and to store; conversion happens only at the boundaries.
    """

    values: np.ndarray
    t0: float
    dt: float
    channel: str = ""
    unit: str = ""

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def duration(self) -> float:
        return len(self.values) * self.dt

    @property
    def times(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(len(self.values))

    def crop(self, start: float, end: float) -> "TimeSeriesData":
        i0 = max(0, int(round((start - self.t0) / self.dt)))
        i1 = min(len(self.values), int(round((end - self.t0) / self.dt)))
        if i1 <= i0:
            raise ValueError(
                f"crop({start}, {end}) is empty for a series covering "
                f"[{self.t0}, {self.t0 + self.duration}]"
            )
        return TimeSeriesData(
            values=self.values[i0:i1],
            t0=self.t0 + i0 * self.dt,
            dt=self.dt,
            channel=self.channel,
            unit=self.unit,
        )

    def to_gwpy(self):
        """Convert to ``gwpy.timeseries.TimeSeries`` (imported lazily)."""
        from gwpy.timeseries import TimeSeries

        return TimeSeries(
            self.values, t0=self.t0, dt=self.dt, channel=self.channel or None
        )

    @classmethod
    def from_gwpy(cls, ts) -> "TimeSeriesData":
        return cls(
            values=np.asarray(ts.value, dtype=np.float64),
            t0=float(ts.t0.value),
            dt=float(ts.dt.value),
            channel=str(ts.channel) if ts.channel is not None else "",
            unit=str(ts.unit) if ts.unit is not None else "",
        )

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        """Write to ``path`` via a temporary file and an atomic rename.

        The temporary file is opened as a *handle* rather than passed by name.
        ``numpy.savez_compressed`` appends ``.npz`` to any filename that does
        not already end in it, so passing ``<hash>.npz.tmp`` writes
        ``<hash>.npz.tmp.npz`` and the subsequent rename fails on a file that
        was never created.  Handing it an open file object suppresses that.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                values=self.values.astype(np.float32),
                meta=np.array(
                    json.dumps(
                        {
                            "t0": self.t0,
                            "dt": self.dt,
                            "channel": self.channel,
                            "unit": self.unit,
                        }
                    )
                ),
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TimeSeriesData":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return cls(
                values=f["values"].astype(np.float64),
                t0=meta["t0"],
                dt=meta["dt"],
                channel=meta.get("channel", ""),
                unit=meta.get("unit", ""),
            )


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


class SegmentList:
    """Analysis-ready segments as an (N, 2) array of GPS intervals."""

    def __init__(self, segments: np.ndarray) -> None:
        seg = np.atleast_2d(np.asarray(segments, dtype=float))
        if seg.size == 0:
            seg = np.zeros((0, 2))
        if seg.shape[1] != 2:
            raise ValueError("segments must have shape (N, 2)")
        order = np.argsort(seg[:, 0])
        self.segments = seg[order]

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterable[tuple[float, float]]:
        return iter(map(tuple, self.segments))

    @property
    def livetime(self) -> float:
        return float(np.sum(self.segments[:, 1] - self.segments[:, 0]))

    def contains(self, gps: np.ndarray | float, pad: float = 0.0) -> np.ndarray:
        """Boolean mask: is ``gps`` inside a segment, with ``pad`` margin?"""
        g = np.atleast_1d(np.asarray(gps, dtype=float))
        starts = self.segments[:, 0] + pad
        ends = self.segments[:, 1] - pad
        idx = np.searchsorted(starts, g, side="right") - 1
        ok = idx >= 0
        out = np.zeros(g.shape, dtype=bool)
        out[ok] = g[ok] <= ends[idx[ok]]
        return out

    def livetime_per_hour(self, gps_start: float, gps_end: float) -> np.ndarray:
        """Livetime (seconds) in each GPS hour of ``[gps_start, gps_end)``."""
        lo = int(np.floor(gps_start / 3600.0))
        hi = int(np.floor(gps_end / 3600.0))
        out = np.zeros(hi - lo + 1)
        for s, e in self.segments:
            s = max(s, gps_start)
            e = min(e, gps_end)
            if e <= s:
                continue
            first = int(np.floor(s / 3600.0))
            last = int(np.floor((e - 1e-9) / 3600.0))
            for h in range(first, last + 1):
                hs = max(s, h * 3600.0)
                he = min(e, (h + 1) * 3600.0)
                out[h - lo] += he - hs
        return out

    def restrict(self, gps_start: float, gps_end: float) -> "SegmentList":
        seg = self.segments.copy()
        seg[:, 0] = np.maximum(seg[:, 0], gps_start)
        seg[:, 1] = np.minimum(seg[:, 1], gps_end)
        return SegmentList(seg[seg[:, 1] > seg[:, 0]])

    def sample_windows(
        self, length: float, n: int, seed: int, pad: float = 4.0
    ) -> np.ndarray:
        """Draw ``n`` non-overlapping start times for windows of ``length`` s.

        Windows are drawn uniformly in *livetime*, not uniformly in segment
        index, so long lock stretches are represented in proportion to how
        much data they actually contribute.
        """
        rng = np.random.default_rng(seed)
        usable = np.maximum(
            0.0, (self.segments[:, 1] - pad) - (self.segments[:, 0] + pad) - length
        )
        total = usable.sum()
        if total <= 0:
            raise ValueError(
                f"no segment is long enough for a {length} s window with {pad} s padding"
            )

        picked: list[float] = []
        cdf = np.cumsum(usable) / total
        attempts = 0
        max_attempts = 200 * n
        while len(picked) < n and attempts < max_attempts:
            attempts += 1
            k = int(np.searchsorted(cdf, rng.random()))
            k = min(k, len(usable) - 1)
            if usable[k] <= 0:
                continue
            start = self.segments[k, 0] + pad + rng.random() * usable[k]
            if any(abs(start - p) < length for p in picked):
                continue
            picked.append(start)

        if len(picked) < n:
            log.warning(
                "requested %d windows but only %d non-overlapping placements "
                "were found in %.1f h of livetime",
                n, len(picked), self.livetime / 3600.0,
            )
        return np.sort(np.array(picked))

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.segments.tolist()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SegmentList":
        return cls(np.array(json.loads(path.read_text(encoding="utf-8"))))


def segments_livetime_per_hour(segments: SegmentList, gps_start: float, gps_end: float):
    """Convenience wrapper returning a pandas Series indexed by GPS hour."""
    import pandas as pd

    lo = int(np.floor(gps_start / 3600.0))
    values = segments.livetime_per_hour(gps_start, gps_end) / 3600.0
    return pd.Series(values, index=pd.RangeIndex(lo, lo + len(values), name="hour_bin"))


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class DataFetcher:
    """Cached access to strain, auxiliary channels and segment lists."""

    #: GWOSC serves open strain as HDF5 files covering this many seconds.  A
    #: request for any sub-interval downloads the whole containing file, so
    #: this is the true granularity of a fetch and the unit we cache in memory.
    BLOCK_SECONDS = 4096

    def __init__(
        self, cfg: Config, offline: bool = False, block_cache_size: int = 3
    ) -> None:
        self.cfg = cfg
        self.offline = offline
        self.cache_root = cfg.path_for("cache_dir")
        self._misses = 0
        self._hits = 0
        self._downloads = 0
        # Small LRU of whole GWOSC blocks, keyed by (detector, block index,
        # sample rate).  Three entries is enough: two so a window straddling a
        # block boundary does not evict the block it is half-way through, and
        # one spare.
        self._block_cache: "OrderedDict[tuple, TimeSeriesData]" = OrderedDict()
        self._block_cache_size = int(block_cache_size)
        # Blocks GWOSC does not serve at all.  Remembering them turns a data
        # gap from "retry budget per event" into "one failed attempt, ever".
        self._failed_blocks: set[tuple] = set()

    # -- cache bookkeeping ----------------------------------------------
    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "downloads": self._downloads,
            "dead_blocks": len(self._failed_blocks),
        }

    def _cache_path(self, kind: str, name: str, key: str) -> Path:
        safe = name.replace(":", "_").replace("/", "_")
        return self.cache_root / kind / safe / f"{key}.npz"

    # -- strain ----------------------------------------------------------
    def strain(
        self,
        detector: str,
        start: float,
        end: float,
        sample_rate: int | None = None,
    ) -> TimeSeriesData:
        """Open GWOSC strain for ``[start, end)``.

        Works identically for O3 and O4a: the O4a strain release (public since
        2025-08-26) is served through the same ``fetch_open_data`` endpoint.

        On a cache miss this does *not* fetch exactly ``[start, end)``.  GWOSC
        stores open strain in HDF5 files of :attr:`BLOCK_SECONDS` seconds, and
        a request for any sub-interval downloads the whole containing file --
        measured at 60 to 90 seconds each on a domestic connection, essentially
        independent of how much of it was asked for.  Fetching a 64 s window
        per trigger therefore costs one full download per trigger.

        Instead the containing block (or blocks, for a window that straddles a
        boundary) is fetched once, held in a small in-memory LRU, and sliced.
        Processing triggers in time order then collapses every trigger in a
        block down to a single download.  Scattered-light triggers cluster
        strongly in time, so this is worth roughly a factor of five to nine.
        """
        sample_rate = int(sample_rate or self.cfg.strain.sample_rate)
        key = hash_key("strain", detector, round(start, 3), round(end, 3), sample_rate)
        path = self._cache_path("strain", detector, key)

        if path.exists():
            self._hits += 1
            return TimeSeriesData.load(path)
        if self.offline:
            raise FileNotFoundError(
                f"offline mode: strain {detector} [{start}, {end}) not cached"
            )

        data = self._from_blocks(detector, start, end, sample_rate)
        data.channel = f"{detector}:{self.cfg.strain.channel_suffix}"
        data.save(path)
        self._misses += 1
        return data

    # -- block-level fetching -------------------------------------------
    def _block_index(self, gps: float) -> int:
        return int(np.floor(gps / self.BLOCK_SECONDS))

    def _get_block(
        self, detector: str, index: int, sample_rate: int
    ) -> TimeSeriesData:
        """Return one whole GWOSC block, from memory if it is already there."""
        cache_key = (detector, index, sample_rate)
        if cache_key in self._block_cache:
            self._block_cache.move_to_end(cache_key)
            return self._block_cache[cache_key]
        if cache_key in self._failed_blocks:
            raise BlockUnavailable(
                f"{detector} block {index} is known to be unavailable"
            )

        start = index * self.BLOCK_SECONDS
        end = start + self.BLOCK_SECONDS
        log.debug("downloading %s block %d [%d, %d)", detector, index, start, end)
        try:
            ts = self._fetch_open_data(detector, start, end, sample_rate)
        except Exception as exc:  # noqa: BLE001
            self._failed_blocks.add(cache_key)
            raise BlockUnavailable(
                f"{detector} block {index} [{start}, {end}) is not served: {exc}"
            ) from exc
        block = TimeSeriesData.from_gwpy(ts)
        self._downloads += 1

        self._block_cache[cache_key] = block
        while len(self._block_cache) > self._block_cache_size:
            self._block_cache.popitem(last=False)
        return block

    def _from_blocks(
        self, detector: str, start: float, end: float, sample_rate: int
    ) -> TimeSeriesData:
        """Assemble ``[start, end)`` out of whole blocks.

        A request longer than one block, or one that falls in a stretch with a
        data gap, falls back to fetching the interval directly -- for the long
        search windows of stage 05 the block machinery would buy nothing, and
        near a gap the block may not exist at all.
        """
        first = self._block_index(start)
        last = self._block_index(end - 1e-9)

        if last - first > 1:
            ts = self._fetch_open_data(detector, start, end, sample_rate)
            self._downloads += 1
            return TimeSeriesData.from_gwpy(ts)

        already_dead = any(
            (detector, i, sample_rate) in self._failed_blocks
            for i in range(first, last + 1)
        )
        try:
            pieces = [
                self._get_block(detector, i, sample_rate)
                for i in range(first, last + 1)
            ]
        except BlockUnavailable as exc:
            if already_dead:
                # A previous event already established that neither the block
                # nor a direct window fetch works here.  Fail immediately.
                raise
            log.debug(
                "block unavailable for %s [%.0f, %.0f) (%s); trying the window "
                "directly once before giving up on it",
                detector, start, end, exc,
            )
            ts = self._fetch_open_data(detector, start, end, sample_rate)
            self._downloads += 1
            return TimeSeriesData.from_gwpy(ts)

        if len(pieces) == 1:
            return pieces[0].crop(start, end)

        joined = TimeSeriesData(
            values=np.concatenate([p.values for p in pieces]),
            t0=pieces[0].t0,
            dt=pieces[0].dt,
            channel=pieces[0].channel,
        )
        return joined.crop(start, end)

    @retry(n_attempts=4, delay=3.0)
    def _fetch_open_data(self, detector, start, end, sample_rate):
        from gwpy.timeseries import TimeSeries

        return TimeSeries.fetch_open_data(
            detector, start, end, sample_rate=sample_rate, cache=False, verbose=False
        )

    # -- auxiliary -------------------------------------------------------
    def auxiliary(
        self,
        channel: str,
        start: float,
        end: float,
        resample: float | None = None,
    ) -> TimeSeriesData:
        """Auxiliary channel from the GWOSC NDS server.

        ``channel`` must appear in the relevant GWOSC auxiliary release; see
        ``config/channels_*.csv``.  Channels whose name ends in ``.rms`` come
        from the second-trend frames and are sampled at 1 Hz.
        """
        key = hash_key("aux", channel, round(start, 3), round(end, 3), resample)
        path = self._cache_path("aux", channel, key)

        if path.exists():
            self._hits += 1
            return TimeSeriesData.load(path)
        if self.offline:
            raise FileNotFoundError(
                f"offline mode: aux {channel} [{start}, {end}) not cached"
            )

        ts = self._fetch_nds(channel, start, end)
        if resample is not None and abs(1.0 / float(ts.dt.value) - resample) > 1e-9:
            ts = ts.resample(resample)
        data = TimeSeriesData.from_gwpy(ts)
        data.channel = channel
        data.save(path)
        self._misses += 1
        return data

    def _fallback_one_at_a_time(self, pending, start, end, batch_error):
        """Recover from a failed batch request without wasting the retry budget.

        A batch can fail because one channel in it is missing, or because the
        whole chunk has no auxiliary data.  Retrying every channel four times
        answers the first case and wastes nine minutes on the second, which is
        the common one -- the O3 release has documented gaps and the detector
        is not always observing.

        So a single channel is probed first.  If that fails too, the chunk is
        taken to be empty and abandoned; if it succeeds, the batch failure was
        caused by a particular channel and the rest are worth fetching.
        """
        canary = pending[0]
        try:
            first = self._fetch_nds(canary, start, end)
        except Exception:  # noqa: BLE001
            log.info(
                "no auxiliary data at all for [%.0f, %.0f) (%s); "
                "skipping its %d channels",
                start, end, batch_error, len(pending),
            )
            return {}

        log.warning(
            "batch fetch failed for [%.0f, %.0f) (%s) but %s is present; "
            "fetching the remaining %d channels individually",
            start, end, batch_error, canary, len(pending) - 1,
        )
        fetched = {canary: first}
        for ch in pending[1:]:
            try:
                fetched[ch] = self._fetch_nds(ch, start, end)
            except Exception as inner:  # noqa: BLE001
                log.warning("auxiliary channel %s unavailable for "
                            "[%.0f, %.0f): %s", ch, start, end, inner)
        return fetched

    @retry(n_attempts=2, delay=4.0)
    def _fetch_nds(self, channel, start, end):
        from gwpy.timeseries import TimeSeries

        return TimeSeries.fetch(
            channel,
            start,
            end,
            host=self.cfg.auxiliary.nds_host,
            port=int(self.cfg.auxiliary.nds_port),
            verbose=False,
        )

    #: The GWOSC NDS server refuses auxiliary requests longer than this.
    #: Measured: 4 h succeeds, 8 h fails.  The first version of this code used
    #: 24 h chunks and every single request failed.
    MAX_NDS_SPAN = 4 * 3600

    def auxiliary_many(
        self, channels: Sequence[str], start: float, end: float
    ) -> dict[str, TimeSeriesData]:
        """Fetch several channels for the same interval, skipping failures.

        Channels not already cached are requested together in a single NDS
        call.  The per-request overhead dominates: nine trend channels over a
        4 h span take 11.3 s in one request against 40.8 s one at a time.

        A missing channel is logged and omitted rather than raising.  The O3
        auxiliary release has documented gaps, and one absent seismometer must
        not abort a multi-day correlation run.
        """
        if end - start > self.MAX_NDS_SPAN:
            raise ValueError(
                f"requested {(end - start) / 3600:.1f} h of auxiliary data but "
                f"the GWOSC NDS server refuses spans beyond "
                f"{self.MAX_NDS_SPAN / 3600:.0f} h; fetch in chunks"
            )

        out: dict[str, TimeSeriesData] = {}
        pending: list[str] = []

        for ch in channels:
            key = hash_key("aux", ch, round(start, 3), round(end, 3), None)
            path = self._cache_path("aux", ch, key)
            if path.exists():
                self._hits += 1
                out[ch] = TimeSeriesData.load(path)
            else:
                pending.append(ch)

        if not pending:
            return out
        if self.offline:
            raise FileNotFoundError(
                f"offline mode: {len(pending)} auxiliary channels not cached "
                f"for [{start}, {end})"
            )

        try:
            fetched = self._fetch_nds_many(pending, start, end)
        except Exception as exc:  # noqa: BLE001
            fetched = self._fallback_one_at_a_time(pending, start, end, exc)

        for ch, ts in fetched.items():
            data = TimeSeriesData.from_gwpy(ts)
            data.channel = ch
            key = hash_key("aux", ch, round(start, 3), round(end, 3), None)
            data.save(self._cache_path("aux", ch, key))
            self._misses += 1
            out[ch] = data

        missing = set(pending) - set(fetched)
        if missing:
            log.warning("%d auxiliary channels unavailable for [%.0f, %.0f): %s",
                        len(missing), start, end, sorted(missing))
        return out

    # Two attempts, not four: an auxiliary gap is a permanent property of the
    # release, and the fall-back path below will try each channel separately
    # anyway.  Four attempts on both paths turned a single missing chunk into
    # several minutes of retries.
    @retry(n_attempts=2, delay=4.0)
    def _fetch_nds_many(self, channels, start, end):
        from gwpy.timeseries import TimeSeriesDict

        return TimeSeriesDict.fetch(
            list(channels),
            start,
            end,
            host=self.cfg.auxiliary.nds_host,
            port=int(self.cfg.auxiliary.nds_port),
            verbose=False,
        )

    # -- segments --------------------------------------------------------
    def segments(self, detector: str, epoch: str, flag: str = "DATA") -> SegmentList:
        """Analysis-ready segments for one detector and epoch.

        Uses the GWOSC segment API via ``gwpy.segments.DataQualityFlag``.  The
        ``DATA`` flag marks times when the detector was locked and the strain
        channel is usable, which is the correct denominator for a glitch rate.
        """
        ep = self.cfg.epoch(epoch)
        name = f"{detector}_{flag}_{epoch}"
        path = self.cache_root / "segments" / f"{name}.json"

        if path.exists():
            self._hits += 1
            return SegmentList.load(path)
        if self.offline:
            raise FileNotFoundError(f"offline mode: segments {name} not cached")

        seg = self._fetch_segments(detector, flag, ep.gps_start, ep.gps_end)
        seg.save(path)
        self._misses += 1
        return seg

    @retry(n_attempts=4, delay=5.0)
    def _fetch_segments(self, detector, flag, start, end) -> SegmentList:
        from gwpy.segments import DataQualityFlag

        dq = DataQualityFlag.fetch_open_data(f"{detector}_{flag}", start, end)
        active = np.array([[float(s[0]), float(s[1])] for s in dq.active])
        log.info(
            "%s_%s [%d, %d): %d segments, %.1f h livetime",
            detector, flag, start, end, len(active),
            (active[:, 1] - active[:, 0]).sum() / 3600.0 if len(active) else 0.0,
        )
        return SegmentList(active)
