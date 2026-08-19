"""Configuration loading.

A single YAML file (``config/analysis.yaml``) drives the whole pipeline.  This
module turns it into an attribute-accessible object, resolves every relative
path against the repository root, and validates the handful of entries that
would otherwise fail late and confusingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

__all__ = ["Config", "load_config", "repo_root"]


def repo_root() -> Path:
    """Return the repository root.

    Resolution order:

    1. ``$GWGLITCH_ROOT`` if it is set (useful on cluster submissions);
    2. the directory two levels above this file, which is ``src/gwglitch/..``
       -> ``src/..`` -> the repository root when installed in editable mode;
    3. the current working directory, as a last resort.
    """
    env = os.environ.get("GWGLITCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    candidate = here.parent.parent.parent
    if (candidate / "config" / "analysis.yaml").exists():
        return candidate
    return Path.cwd().resolve()


class _Node(Mapping):
    """Read-only mapping that also supports attribute access.

    ``cfg.qtransform.q_range`` and ``cfg["qtransform"]["q_range"]`` are
    equivalent.  Nested dicts are wrapped lazily; lists of dicts are wrapped
    element-wise.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    # -- Mapping protocol ------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access ------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self._data[name])
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"configuration has no entry {name!r}; available: "
                f"{sorted(self._data)}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        raise TypeError("configuration is read-only")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Node({sorted(self._data)})"

    def to_dict(self) -> dict:
        """Return a deep plain-``dict`` copy."""
        return _unwrap(self._data)


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _Node(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, _Node):
        return _unwrap(value._data)
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


@dataclass
class Config:
    """The parsed analysis configuration."""

    raw: dict
    root: Path
    path: Path = field(default_factory=Path)

    # -- passthrough -----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, i.e. for YAML keys.
        try:
            return _wrap(self.__dict__["raw"][name])
        except KeyError as exc:
            raise AttributeError(
                f"configuration has no top-level section {name!r}; available: "
                f"{sorted(self.__dict__['raw'])}"
            ) from exc

    def __getitem__(self, key: str) -> Any:
        return _wrap(self.raw[key])

    # -- paths -----------------------------------------------------------
    def path_for(self, key: str) -> Path:
        """Resolve ``paths.<key>`` against the repository root, creating it."""
        rel = self.raw["paths"][key]
        out = (self.root / rel).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    def resolve(self, relative: str | os.PathLike) -> Path:
        """Resolve an arbitrary repo-relative path (no directory creation)."""
        return (self.root / Path(relative)).resolve()

    # -- convenience -----------------------------------------------------
    def epoch(self, name: str) -> _Node:
        """Return the epoch block for ``O3a`` / ``O3b`` / ``O4a``."""
        try:
            return _wrap(self.raw["epochs"][name])
        except KeyError as exc:
            raise KeyError(
                f"unknown epoch {name!r}; configured epochs are "
                f"{sorted(self.raw['epochs'])}"
            ) from exc

    def epochs_for_run(self, run: str) -> list[str]:
        """Names of the epochs belonging to observing run ``run`` (O3 / O4)."""
        return [
            name
            for name, block in self.raw["epochs"].items()
            if block["run"] == run
        ]

    @property
    def epoch_names(self) -> list[str]:
        return list(self.raw["epochs"])


_REQUIRED_SECTIONS = (
    "epochs",
    "gravityspy",
    "strain",
    "qtransform",
    "scattering",
    "auxiliary",
    "sampling",
    "classifier",
    "statistics",
    "paths",
)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load and validate the analysis configuration.

    Parameters
    ----------
    path
        Path to the YAML file.  Defaults to ``<repo root>/config/analysis.yaml``.
    """
    root = repo_root()
    cfg_path = Path(path) if path is not None else root / "config" / "analysis.yaml"
    cfg_path = cfg_path.expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"analysis configuration not found at {cfg_path}. "
            "Set $GWGLITCH_ROOT or pass an explicit path."
        )

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping")

    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"{cfg_path} is missing required sections: {missing}")

    _validate(raw, cfg_path)
    return Config(raw=raw, root=root, path=cfg_path)


def _validate(raw: dict, cfg_path: Path) -> None:
    """Fail fast on the mistakes that are expensive to discover late."""
    for name, block in raw["epochs"].items():
        if block["gps_end"] <= block["gps_start"]:
            raise ValueError(f"epoch {name}: gps_end must exceed gps_start")

    qt = raw["qtransform"]
    if qt["q_range"][0] >= qt["q_range"][1]:
        raise ValueError("qtransform.q_range must be increasing")
    if qt["frequency_range"][0] >= qt["frequency_range"][1]:
        raise ValueError("qtransform.frequency_range must be increasing")

    strain = raw["strain"]
    if strain["qscan_half_width"] >= strain["window_half_width"]:
        raise ValueError(
            "strain.qscan_half_width must be smaller than window_half_width so "
            "that filter transients fall outside the analysed region"
        )

    if not 0.0 < raw["gravityspy"]["min_ml_confidence"] <= 1.0:
        raise ValueError("gravityspy.min_ml_confidence must be in (0, 1]")

    root = repo_root()
    for key, rel in raw["auxiliary"]["channel_list_files"].items():
        if not (root / rel).exists():
            raise FileNotFoundError(
                f"auxiliary.channel_list_files.{key} points at {rel}, "
                f"which does not exist relative to {root}"
            )

    target = raw["gravityspy"]["target_label"]
    controls = raw["sampling"]["control_classes"]
    if target in controls:
        raise ValueError(
            f"{cfg_path}: the target class {target!r} also appears in "
            "sampling.control_classes"
        )
