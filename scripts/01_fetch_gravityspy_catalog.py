#!/usr/bin/env python
"""Stage 01 -- download the Gravity Spy catalogue and summarise the O3 sample.

Downloads the four O3 CSVs from Zenodo record 5649212 into ``data/catalog``
(about 560 MB in total), applies the SNR / peak-frequency / ML-confidence cuts
from the configuration, and writes:

    data/derived/o3_scattered_light.parquet   the target-class triggers
    data/derived/o3_control.parquet           the control-class triggers
    results/tables/catalog_summary.csv        counts per detector and epoch

There is no O4a stage here.  The Gravity Spy release covers O1 through O3b
only; the O4a sample is produced by ``05_search_o4a.py`` from strain.
"""

from __future__ import annotations

import sys

from _common import banner, base_parser, bootstrap, provenance

from gwglitch.catalog import (
    add_time_columns,
    block_stratified_sample,
    cluster_triggers,
    describe_download_plan,
    download_catalog,
    load_run,
    sample_within_blocks,
    select_class,
    summarise,
)
from gwglitch.tables import write_table
from gwglitch.utils import JsonSidecar


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="re-download files that are already present")
    parser.add_argument("--skip-download", action="store_true",
                        help="assume the CSVs are already in data/catalog")
    args = parser.parse_args()

    cfg, log = bootstrap(args, "01_fetch_gravityspy_catalog")

    print(banner("download plan"))
    print(describe_download_plan(cfg))

    if not args.skip_download:
        if args.offline:
            log.error("--offline and no --skip-download: nothing to do")
            return 1
        download_catalog(cfg, overwrite=args.overwrite)

    print(banner("loading and cutting"))
    triggers = load_run(cfg, run="O3")
    triggers = add_time_columns(cfg, triggers)

    target = cfg.gravityspy.target_label
    scattered = select_class(triggers, target)
    control = select_class(triggers, list(cfg.sampling.control_classes))

    log.info("after cuts: %d %s, %d control, %d total",
             len(scattered), target, len(control), len(triggers))

    # The release lists Omicron *triggers*, not events.  Collapse triggers that
    # share a Q-scan window, otherwise the morphology distributions are weighted
    # by how many triggers each arch train happened to produce.  The unclustered
    # list is kept as well, because the auxiliary rate correlation legitimately
    # uses trigger rate -- the write-up says which is which.
    window = float(cfg.sampling.cluster_window)
    print(banner("clustering triggers into events"))
    scattered_events = cluster_triggers(scattered, window)
    control_events = cluster_triggers(control, window)

    if scattered.empty:
        log.error(
            "no %s triggers survived the cuts. Check gravityspy.min_ml_confidence "
            "(currently %.2f) and that the CSVs downloaded completely.",
            target, cfg.gravityspy.min_ml_confidence,
        )
        return 1

    # Morphology sub-samples, stratified over detector, epoch and time.
    seed = int(cfg.sampling.seed)
    print(banner("block-stratified sampling"))
    scattered_sample, blocks = block_stratified_sample(
        scattered_events,
        n_blocks=int(cfg.sampling.blocks_per_detector_epoch),
        max_per_block=int(cfg.sampling.max_events_per_block),
        seed=seed,
    )
    control_sample = sample_within_blocks(
        control_events,
        blocks=blocks,
        max_per_block=int(cfg.sampling.max_control_per_block),
        seed=seed + 1,
    )

    # Process in time order so that the block-level strain cache in
    # gwglitch.dataio can serve many triggers from one GWOSC file download.
    scattered_sample = scattered_sample.sort_values("event_time").reset_index(drop=True)
    control_sample = control_sample.sort_values("event_time").reset_index(drop=True)

    derived = cfg.path_for("derived_dir")
    tables = cfg.path_for("table_dir")

    write_table(scattered, derived / "o3_scattered_light_all")
    write_table(scattered_events, derived / "o3_scattered_light_events")
    write_table(scattered_sample, derived / "o3_scattered_light_sample")
    write_table(control_sample, derived / "o3_control_sample")

    summary = summarise(triggers, cfg)
    summary.to_csv(tables / "catalog_summary.csv", index=False)
    print(banner("catalogue summary"))
    print(summary.to_string(index=False))

    JsonSidecar(derived / "o3_catalog.provenance.json").write(
        **provenance(
            cfg, "01_fetch_gravityspy_catalog",
            zenodo_record=int(cfg.gravityspy.zenodo_record),
            n_triggers_after_cuts=int(len(triggers)),
            n_target_triggers=int(len(scattered)),
            n_target_events=int(len(scattered_events)),
            cluster_window=window,
            n_blocks=len(blocks),
            n_target_sampled=int(len(scattered_sample)),
            n_control_sampled=int(len(control_sample)),
            cuts={
                "min_snr": float(cfg.gravityspy.min_snr),
                "min_ml_confidence": float(cfg.gravityspy.min_ml_confidence),
                "peak_frequency_range": list(cfg.gravityspy.peak_frequency_range),
            },
        )
    )
    log.info("stage 01 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
