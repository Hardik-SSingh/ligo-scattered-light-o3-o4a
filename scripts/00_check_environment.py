#!/usr/bin/env python
"""Stage 00 -- verify the environment before anything expensive is attempted.

Checks, in order of how annoying they are to discover late:

1. every required package imports, at a usable version;
2. the linear-algebra backend actually works (see below);
3. the configuration parses and its channel lists are present;
4. GWOSC open strain is reachable for O3 *and* O4a;
5. the GWOSC NDS server serves auxiliary channels for O3 and O4a;
6. the segment API answers.

Check 2 is not paranoia.  A conda environment solved with MKL 2026.1.0 on
Windows terminates the interpreter with STATUS_STACK_BUFFER_OVERRUN on any
LAPACK call -- ``numpy.linalg.lstsq``, ``scipy.signal.detrend``,
``sklearn``'s regressions -- with no Python traceback at all.  The fix is to
pin the OpenBLAS builds, as ``environment.yml`` does.  This check catches a
recurrence in one second instead of during a twelve-hour analysis run.

Exit status is 0 if every mandatory check passes, 1 otherwise.
"""

from __future__ import annotations

import importlib
import sys
import traceback

from _common import banner, base_parser

REQUIRED = {
    "numpy": "1.24",
    "scipy": "1.11",
    "pandas": "2.0",
    "matplotlib": "3.8",
    "sklearn": "1.3",
    "h5py": "3.9",
    "yaml": "6.0",
    "gwpy": "3.0",
    "gwosc": "0.7",
}

OPTIONAL = {"nds2": "auxiliary channel access over NDS", "joblib": "model persistence"}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  [ ok ] {what}{(' -- ' + detail) if detail else ''}")

    def warn(self, what: str, detail: str) -> None:
        print(f"  [warn] {what} -- {detail}")
        self.warnings.append(what)

    def fail(self, what: str, detail: str) -> None:
        print(f"  [FAIL] {what} -- {detail}")
        self.failures.append(what)


def _version_at_least(found: str, wanted: str) -> bool:
    def parts(v):
        out = []
        for piece in v.split(".")[:3]:
            digits = "".join(c for c in piece if c.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    return parts(found) >= parts(wanted)


def check_packages(rep: Report) -> None:
    print(banner("1. packages"))
    print(f"  python {sys.version.split()[0]} at {sys.executable}")
    for name, minimum in REQUIRED.items():
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            rep.fail(name, f"import failed: {exc}")
            continue
        version = getattr(mod, "__version__", "unknown")
        if version != "unknown" and not _version_at_least(version, minimum):
            rep.fail(name, f"version {version} is older than the required {minimum}")
        else:
            rep.ok(name, version)

    for name, why in OPTIONAL.items():
        try:
            importlib.import_module(name)
            rep.ok(name, why)
        except Exception:  # noqa: BLE001
            rep.warn(name, f"absent -- {why} will not work")


def check_linalg(rep: Report) -> None:
    print(banner("2. linear algebra backend"))
    try:
        import numpy as np
        from scipy import signal

        rng = np.random.default_rng(0)
        a = rng.standard_normal((128, 128))
        b = rng.standard_normal(128)

        np.linalg.solve(a, b)
        np.linalg.svd(a, compute_uv=False)
        np.linalg.lstsq(a, b, rcond=None)
        signal.detrend(rng.standard_normal(4096), type="linear")
        sos = signal.butter(4, [0.03, 10.0], btype="bandpass", fs=256.0, output="sos")
        signal.sosfiltfilt(sos, rng.standard_normal(8192))
    except Exception as exc:  # noqa: BLE001
        rep.fail("LAPACK / SciPy filtering", f"{type(exc).__name__}: {exc}")
        return
    rep.ok("LAPACK solve/svd/lstsq and SciPy detrend/filtfilt")

    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")
        name = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
        rep.ok("BLAS", str(name))
    except Exception:  # noqa: BLE001
        pass


def check_config(rep: Report, args) -> object | None:
    print(banner("3. configuration"))
    try:
        from gwglitch import load_config

        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        rep.fail("configuration", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None

    rep.ok("analysis.yaml", str(cfg.path))
    rep.ok("epochs", ", ".join(cfg.epoch_names))
    for key, rel in cfg.auxiliary.channel_list_files.items():
        path = cfg.resolve(rel)
        n_lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines()) - 1
        rep.ok(f"channel list {key}", f"{n_lines} channels")
    return cfg


def check_strain(rep: Report, cfg) -> None:
    print(banner("4. GWOSC open strain"))
    from gwpy.timeseries import TimeSeries

    probes = [
        ("O3b", "L1", 1256655700),
        ("O4a", "L1", 1368980000),
        ("O4a", "H1", 1368980000),
    ]
    for epoch, detector, gps in probes:
        try:
            ts = TimeSeries.fetch_open_data(
                detector, gps, gps + 32, sample_rate=4096, cache=False
            )
            rep.ok(f"{epoch} {detector} strain",
                   f"{len(ts)} samples at {float(ts.sample_rate.value):.0f} Hz")
        except Exception as exc:  # noqa: BLE001
            rep.fail(f"{epoch} {detector} strain", f"{type(exc).__name__}: {exc}")


def check_nds(rep: Report, cfg) -> None:
    print(banner("5. GWOSC auxiliary channels over NDS"))
    from gwpy.timeseries import TimeSeries

    host = cfg.auxiliary.nds_host
    probes = [
        ("O3 full-rate witness", cfg.auxiliary.witness["L1"]["channel"], 1256655700, 60),
        ("O3 ground motion", "L1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M", 1256655700, 60),
        ("O3 second trend", "L1:ISI-GND_STS_ITMY_Z_BLRMS_100M_300M.rms", 1256655700, 600),
        ("O4a auxiliary", "L1:CAL-PCALX_RX_PD_OUT_DQ", 1368980000, 10),
    ]
    for label, channel, gps, length in probes:
        try:
            ts = TimeSeries.fetch(channel, gps, gps + length, host=host)
            rep.ok(f"{label} ({channel})",
                   f"{len(ts)} samples at {float(ts.sample_rate.value):g} Hz")
        except Exception as exc:  # noqa: BLE001
            rep.warn(f"{label} ({channel})", f"{type(exc).__name__}: {exc}")


def check_segments(rep: Report, cfg) -> None:
    print(banner("6. segment database"))
    from gwpy.segments import DataQualityFlag

    for detector, epoch in (("L1", "O3b"), ("L1", "O4a")):
        ep = cfg.epoch(epoch)
        try:
            dq = DataQualityFlag.fetch_open_data(
                f"{detector}_DATA", ep.gps_start, ep.gps_start + 86400
            )
            livetime = sum(float(s[1] - s[0]) for s in dq.active) / 3600.0
            rep.ok(f"{detector} {epoch} segments",
                   f"{len(dq.active)} in the first day, {livetime:.2f} h livetime")
        except Exception as exc:  # noqa: BLE001
            rep.fail(f"{detector} {epoch} segments", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--skip-network", action="store_true",
                        help="run only the local checks")
    args = parser.parse_args()

    rep = Report()
    check_packages(rep)
    check_linalg(rep)
    cfg = check_config(rep, args)

    if cfg is not None and not (args.skip_network or args.offline):
        check_strain(rep, cfg)
        check_nds(rep, cfg)
        check_segments(rep, cfg)
    else:
        print("\n(network checks skipped)")

    print(banner("summary"))
    if rep.failures:
        print(f"  {len(rep.failures)} FAILED: {', '.join(rep.failures)}")
    if rep.warnings:
        print(f"  {len(rep.warnings)} warnings: {', '.join(rep.warnings)}")
    if not rep.failures:
        print("  environment is ready.")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
