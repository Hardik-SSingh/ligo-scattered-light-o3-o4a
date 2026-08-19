#!/usr/bin/env python
"""Stage 02 -- Q-scan the O3 samples and extract morphology features.

Runs over the target and control samples written by stage 01.  For each
trigger it fetches a padded strain window, computes the Q-transform, extracts
the feature vector, and appends a row to the output table.

The stage is *resumable*.  Every completed trigger is flushed to a partial
file every ``--checkpoint`` rows, and a re-run skips triggers that are already
present.  With a few thousand network fetches on a home connection this is not
a luxury; the run will be interrupted.

Output
------
    data/derived/o3_features_<label>.parquet   one row per trigger
    results/tables/qscan_failures.csv          triggers that could not be scanned
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from _common import banner, base_parser, bootstrap, provenance

from gwglitch.dataio import DataFetcher
from gwglitch.morphology import extract_features
from gwglitch.qscan import qmap_for_trigger
from gwglitch.tables import read_table, write_table
from gwglitch.utils import JsonSidecar, human_duration

SAMPLES = {
    "scattered": "o3_scattered_light_sample",
    "control": "o3_control_sample",
}


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--sample", choices=[*SAMPLES, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most this many triggers (for a trial run)")
    parser.add_argument("--checkpoint", type=int, default=100,
                        help="flush partial results every N triggers")
    parser.add_argument("--restart", action="store_true",
                        help="ignore any existing partial output and start over")
    args = parser.parse_args()

    cfg, log = bootstrap(args, "02_run_qscans")
    fetcher = DataFetcher(cfg, offline=args.offline)
    derived = cfg.path_for("derived_dir")

    which = list(SAMPLES) if args.sample == "all" else [args.sample]
    failures: list[dict] = []

    for label in which:
        stem = derived / SAMPLES[label]
        try:
            triggers = read_table(stem)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 1

        # Order by detector and then by time.  The strain cache in
        # gwglitch.dataio holds whole GWOSC file blocks in memory, so
        # consecutive triggers from the same block cost one download between
        # them; interleaving detectors would thrash that cache and put the
        # download count back up by a factor of several.
        triggers = triggers.sort_values(
            ["detector", "event_time"], kind="mergesort"
        ).reset_index(drop=True)

        if args.limit:
            triggers = triggers.head(args.limit)

        out_stem = derived / f"o3_features_{label}"
        done: set[float] = set()
        rows: list[dict] = []

        if not args.restart:
            try:
                existing = read_table(out_stem)
                rows = existing.to_dict("records")
                done = set(np.round(existing["event_time"].to_numpy(), 4))
                log.info("resuming %s: %d triggers already done", label, len(done))
            except FileNotFoundError:
                pass

        if done:
            keys = pd.Series(
                np.round(triggers["event_time"].to_numpy(), 4), index=triggers.index
            )
            todo = triggers.loc[~keys.isin(done)]
        else:
            todo = triggers

        print(banner(f"{label}: {len(todo)} triggers to scan "
                     f"({len(triggers)} in sample)"))

        threshold = float(cfg.qtransform.energy_threshold)
        islands = int(cfg.qtransform.min_island_tiles)
        started = time.time()

        for i, trigger in enumerate(todo.itertuples(), start=1):
            t0 = float(trigger.event_time)
            detector = str(trigger.detector)
            try:
                qmap = qmap_for_trigger(fetcher, cfg, detector, t0)
                features = extract_features(qmap, threshold, islands)
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "event_time": t0, "detector": detector, "sample": label,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                log.debug("failed %s %.4f: %s", detector, t0, exc)
                continue

            rows.append({
                "event_time": t0,
                "detector": detector,
                "epoch": str(trigger.epoch),
                "run": str(getattr(trigger, "run", "O3")),
                "ml_label": str(getattr(trigger, "ml_label", "")),
                "ml_confidence": float(getattr(trigger, "ml_confidence", np.nan)),
                "snr_omicron": float(getattr(trigger, "snr", np.nan)),
                "peak_frequency_omicron": float(
                    getattr(trigger, "peak_frequency", np.nan)
                ),
                "q_selected": float(qmap.q),
                **features,
            })

            if i % args.checkpoint == 0:
                write_table(pd.DataFrame(rows), out_stem)
                rate = i / (time.time() - started)
                remaining = (len(todo) - i) / rate if rate > 0 else float("nan")
                stats = fetcher.stats
                log.info(
                    "%s %d/%d (%.2f trigger/s, ~%s remaining) | "
                    "%d downloads, %.2f events per download",
                    label, i, len(todo), rate, human_duration(remaining),
                    stats["downloads"],
                    i / max(stats["downloads"], 1),
                )

        frame = pd.DataFrame(rows)
        write_table(frame, out_stem)
        log.info("%s: %d rows written, %d failures so far",
                 label, len(frame), len(failures))

        JsonSidecar(derived / f"o3_features_{label}.provenance.json").write(
            **provenance(
                cfg, "02_run_qscans",
                sample=label,
                n_requested=int(len(triggers)),
                n_extracted=int(len(frame)),
                energy_threshold=threshold,
                min_island_tiles=islands,
                q_range=list(cfg.qtransform.q_range),
                frequency_range=list(cfg.qtransform.frequency_range),
                cache_stats=fetcher.stats,
            )
        )

    if failures:
        path = cfg.path_for("table_dir") / "qscan_failures.csv"
        pd.DataFrame(failures).to_csv(path, index=False)
        log.warning("%d triggers could not be scanned; details in %s",
                    len(failures), path.name)

    log.info("stage 02 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
