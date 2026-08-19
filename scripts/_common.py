"""Shared argument parsing and bootstrapping for the pipeline scripts.

Each stage is a standalone command-line program.  They share a small amount of
setup -- locating the repository, loading the configuration, configuring
logging, writing a provenance sidecar -- which lives here.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running the scripts directly out of a checkout without installing.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gwglitch import load_config  # noqa: E402
from gwglitch.utils import setup_logging  # noqa: E402

__all__ = ["base_parser", "bootstrap", "provenance", "banner"]


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="path to analysis.yaml (default: <repo>/config/analysis.yaml)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="fail rather than hit the network; use only cached data",
    )
    return parser


def bootstrap(args: argparse.Namespace, stage: str):
    """Load the configuration and start logging to console and file."""
    cfg = load_config(args.config)
    logdir = cfg.path_for("results_dir") / "logs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = setup_logging(args.log_level, logdir / f"{stage}_{stamp}.log")
    log.info("stage %s starting", stage)
    log.info("repository root: %s", cfg.root)
    log.info("configuration  : %s", cfg.path)
    return cfg, log


def provenance(cfg, stage: str, **extra) -> dict:
    """Metadata written alongside every derived product."""
    return {
        "stage": stage,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": str(cfg.path),
        "seed": int(cfg.sampling.seed),
        **extra,
    }


def banner(text: str) -> str:
    line = "=" * max(len(text), 60)
    return f"\n{line}\n{text}\n{line}"
