"""Reading and writing the intermediate tables that link the pipeline stages.

Parquet is used when an engine is installed (it is faster and keeps dtypes),
and gzipped CSV otherwise.  ``environment.yml`` does not require ``pyarrow``,
so both paths are live and the reader has to work out which one it is looking
at -- hence a single pair of functions rather than scattered ``to_parquet``
calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["write_table", "read_table", "table_path"]

_PARQUET = ".parquet"
_CSV = ".csv.gz"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def table_path(stem: str | Path) -> Path:
    """Resolve ``stem`` to whichever of the two formats exists on disk."""
    stem = Path(stem)
    for suffix in (_PARQUET, _CSV):
        candidate = stem.with_name(stem.name + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no table found at {stem}{_PARQUET} or {stem}{_CSV}. "
        "Has the preceding pipeline stage been run?"
    )


def write_table(df: pd.DataFrame, stem: str | Path) -> Path:
    """Write ``df`` to ``<stem>.parquet`` or ``<stem>.csv.gz``."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    if _parquet_available():
        path = stem.with_name(stem.name + _PARQUET)
        df.to_parquet(path, index=False)
    else:
        path = stem.with_name(stem.name + _CSV)
        df.to_csv(path, index=False, compression="gzip")

    log.info("wrote %s (%d rows, %d columns)", path.name, len(df), df.shape[1])
    return path


def read_table(stem: str | Path) -> pd.DataFrame:
    """Read a table written by :func:`write_table`."""
    path = table_path(stem)
    if path.name.endswith(_PARQUET):
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="gzip", low_memory=False)
