#!/usr/bin/env python
"""Stage 05 -- blind scattered-light search in O4a strain.

There is no public Gravity Spy catalogue for O4a and no motion witness in the
O4a auxiliary release, so the O4a sample has to be built from strain alone:

    sample analysis-ready segments  ->  excess-power trigger finder
        ->  Q-scan  ->  morphology features  ->  O3-trained classifier

Livetime, not calendar time, is the unit of the result.  The stage draws
``sampling.o4a_n_segments`` windows of ``sampling.o4a_segment_length`` seconds,
uniformly in livetime, from each detector's analysis-ready segments.  With the
configured 1200 x 512 s that is about 7.1 days per detector -- roughly 4 per
cent of O4a -- which is what a laptop and a home connection can actually do.
Rates are therefore quoted per hour of *analysed* livetime and are directly
comparable with the O3 rates, which are computed the same way.

For a like-for-like comparison the same search is also run on a matched
livetime sample of O3, so that any rate difference cannot be blamed on the
change of trigger generator between the Gravity Spy catalogue and this one.
Use ``--epoch O3b`` for that.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from _common import banner, base_parser, bootstrap, provenance

from gwglitch.dataio import DataFetcher
from gwglitch.identify import ClassifierBundle, score_frame, search_segment
from gwglitch.tables import read_table, write_table
from gwglitch.utils import JsonSidecar, human_duration


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--epoch", default="O4a", choices=["O3a", "O3b", "O4a"])
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--n-segments", type=int, default=None,
                        help="override sampling.o4a_n_segments")
    parser.add_argument("--z-threshold", type=float, default=5.0,
                        help="excess-power trigger threshold, robust z-score")
    parser.add_argument("--checkpoint", type=int, default=20,
                        help="flush partial results every N segments")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    cfg, log = bootstrap(args, "05_search_o4a")
    fetcher = DataFetcher(cfg, offline=args.offline)
    derived = cfg.path_for("derived_dir")

    try:
        bundle = ClassifierBundle.load(derived / "identifier.joblib")
    except FileNotFoundError:
        log.error("identifier.joblib missing -- run scripts/04_train_identifier.py")
        return 1
    log.info("loaded identifier:\n%s", bundle.summary())

    n_segments = int(args.n_segments or cfg.sampling.o4a_n_segments)
    seg_length = float(cfg.sampling.o4a_segment_length)
    epoch = args.epoch

    all_rows: list[pd.DataFrame] = []
    livetime_records = []

    for detector in args.detectors:
        print(banner(f"searching {detector} {epoch}"))
        segments = fetcher.segments(detector, epoch)
        log.info("%s %s: %d segments, %.1f h total livetime",
                 detector, epoch, len(segments), segments.livetime / 3600.0)

        starts = segments.sample_windows(
            length=seg_length,
            n=n_segments,
            seed=int(cfg.sampling.seed) + hash(detector) % 1000,
            pad=float(cfg.strain.window_half_width) + 4.0,
        )
        analysed_hours = len(starts) * seg_length / 3600.0
        log.info("drew %d windows -> %.2f h of analysed livetime",
                 len(starts), analysed_hours)

        out_stem = derived / f"{epoch.lower()}_candidates_{detector}"
        rows: list[dict] = []
        done_segments: set[float] = set()

        if not args.restart:
            try:
                existing = read_table(out_stem)
                rows = existing.to_dict("records")
                done_segments = set(np.round(existing["segment_start"], 3))
                log.info("resuming: %d segments already searched", len(done_segments))
            except FileNotFoundError:
                pass

        todo = [s for s in starts if round(float(s), 3) not in done_segments]
        started = time.time()

        for i, start in enumerate(todo, start=1):
            try:
                found = search_segment(
                    fetcher, cfg, detector, float(start), seg_length,
                    z_threshold=args.z_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("segment %.0f failed: %s", start, exc)
                continue
            if not found.empty:
                found["epoch"] = epoch
                found["run"] = cfg.epoch(epoch).run
                rows.extend(found.to_dict("records"))

            if i % args.checkpoint == 0:
                if rows:
                    write_table(pd.DataFrame(rows), out_stem)
                rate = i / (time.time() - started)
                log.info(
                    "%s %d/%d segments (%.3f seg/s, ~%s remaining, %d candidates)",
                    detector, i, len(todo), rate,
                    human_duration((len(todo) - i) / rate if rate > 0 else np.nan),
                    len(rows),
                )

        frame = pd.DataFrame(rows)
        if frame.empty:
            log.warning("%s %s: no candidates found at all", detector, epoch)
            continue

        scored = score_frame(bundle, frame)
        write_table(scored, out_stem)

        n_scatter = int(scored["is_scattering"].sum())
        log.info(
            "%s %s: %d candidates, %d classified as scattered light "
            "(%.2f per hour of analysed livetime)",
            detector, epoch, len(scored), n_scatter, n_scatter / analysed_hours,
        )

        livetime_records.append({
            "detector": detector,
            "epoch": epoch,
            "n_windows": int(len(starts)),
            "segment_length_s": seg_length,
            "analysed_livetime_hours": analysed_hours,
            "total_epoch_livetime_hours": segments.livetime / 3600.0,
            "coverage_fraction": (len(starts) * seg_length) / segments.livetime,
            "n_candidates": int(len(scored)),
            "n_scattering": n_scatter,
            "rate_per_hour": n_scatter / analysed_hours,
        })
        all_rows.append(scored)

    if not all_rows:
        log.error("nothing was found; aborting")
        return 1

    combined = pd.concat(all_rows, ignore_index=True)
    write_table(combined, derived / f"{epoch.lower()}_candidates")

    livetime = pd.DataFrame(livetime_records)
    livetime.to_csv(
        cfg.path_for("table_dir") / f"{epoch.lower()}_search_livetime.csv", index=False
    )
    print(banner("search summary"))
    print(livetime.to_string(index=False))

    JsonSidecar(derived / f"{epoch.lower()}_search.provenance.json").write(
        **provenance(
            cfg, "05_search_o4a",
            epoch=epoch,
            detectors=args.detectors,
            n_segments=n_segments,
            segment_length=seg_length,
            z_threshold=args.z_threshold,
            classifier_threshold=bundle.threshold,
            classifier_cv_auc=bundle.cv_auc,
            classifier_cv_efficiency=bundle.cv_efficiency,
            cache_stats=fetcher.stats,
        )
    )
    log.info("stage 05 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
