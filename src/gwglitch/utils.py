"""Small shared utilities: GPS/UTC conversion, logging, caching, hashing.

Deliberately free of GW-specific dependencies so it can be imported without
``gwpy`` present (``scripts/00_check_environment.py`` relies on that).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .constants import GPS_UNIX_OFFSET, LEAP_SECONDS_O3_O4, SECONDS_PER_DAY

__all__ = [
    "setup_logging",
    "gps_to_utc",
    "utc_to_gps",
    "gps_to_datetime",
    "gps_day_index",
    "hash_key",
    "atomic_write_bytes",
    "retry",
    "chunk",
    "human_duration",
    "JsonSidecar",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logging(level: str | int = "INFO", logfile: Path | None = None) -> logging.Logger:
    """Configure root logging once and return the package logger."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(fh)
    root.setLevel(level)
    # matplotlib and urllib3 are extremely chatty at DEBUG.
    for noisy in ("matplotlib", "urllib3", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("gwglitch")


# ---------------------------------------------------------------------------
# GPS time
# ---------------------------------------------------------------------------
#
# GPS time runs without leap seconds; UTC does not.  No leap second has been
# inserted since 2016-12-31, so throughout O3 and O4 the offset is a constant
# 18 s.  Hard-coding it keeps this module free of an astropy/lal dependency
# and is exact for every epoch this project touches.


def gps_to_datetime(gps: float) -> datetime:
    """Convert GPS seconds to a timezone-aware UTC ``datetime``."""
    unix = float(gps) + GPS_UNIX_OFFSET - LEAP_SECONDS_O3_O4
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=unix)


def gps_to_utc(gps: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format GPS seconds as a UTC string."""
    return gps_to_datetime(gps).strftime(fmt)


def utc_to_gps(when: str | datetime) -> float:
    """Convert a UTC timestamp (ISO-ish string or ``datetime``) to GPS seconds."""
    if isinstance(when, str):
        dt = datetime.fromisoformat(when)
    else:
        dt = when
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix = dt.timestamp()
    return unix - GPS_UNIX_OFFSET + LEAP_SECONDS_O3_O4


def gps_day_index(gps: np.ndarray | float, reference_gps: float) -> np.ndarray | float:
    """Days elapsed since ``reference_gps`` (fractional)."""
    return (np.asarray(gps, dtype=float) - float(reference_gps)) / SECONDS_PER_DAY


def human_duration(seconds: float) -> str:
    """Format a duration for log messages."""
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 3 * SECONDS_PER_DAY:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / SECONDS_PER_DAY:.2f} d"


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def hash_key(*parts: Any, length: int = 16) -> str:
    """Stable short hash of the arguments, for cache filenames.

    ``json`` with ``sort_keys`` gives a canonical representation, so the same
    logical request always maps to the same file regardless of dict ordering.
    """
    payload = json.dumps(parts, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary file and an atomic rename.

    Prevents a half-written cache entry from surviving an interrupted run --
    which on a laptop with a flaky connection is the common failure mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{id(data):x}")
    tmp.write_bytes(data)
    tmp.replace(path)


class JsonSidecar:
    """A tiny JSON file recording provenance next to a derived data product."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write(self, **fields: Any) -> None:
        payload = dict(fields)
        payload.setdefault("written_utc", datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


def retry(
    n_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """Decorator: retry a flaky network call with exponential backoff.

    Remote data servers (GWOSC, NDS, Zenodo) fail transiently often enough that
    a bare call inside a 10 000-iteration loop is not viable.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logging.getLogger("gwglitch.retry")
            wait = delay
            last: BaseException | None = None
            for attempt in range(1, n_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    last = exc
                    if attempt == n_attempts:
                        break
                    log.warning(
                        "%s failed (attempt %d/%d): %s -- retrying in %.1f s",
                        func.__name__, attempt, n_attempts, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
            raise RuntimeError(
                f"{func.__name__} failed after {n_attempts} attempts"
            ) from last

        return wrapper

    return decorator


def chunk(seq: Sequence, size: int) -> Iterable[Sequence]:
    """Yield consecutive slices of ``seq`` of length at most ``size``."""
    if size < 1:
        raise ValueError("size must be >= 1")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]
