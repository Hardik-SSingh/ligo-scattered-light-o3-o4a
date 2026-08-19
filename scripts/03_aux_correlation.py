#!/usr/bin/env python
"""Stage 03 -- auxiliary-channel correlation (O3 only).

Two analyses, at two different levels.

**Rate level.**  Bin the full O3 scattered-light catalogue into hours, divide
by the analysis-ready livetime in each hour, and rank-correlate against the
band-limited RMS of each configured ground-motion trend channel.  Confidence
intervals come from a moving-block bootstrap because hourly rates are strongly
autocorrelated.  A lag scan checks that the correlation peaks at zero lag, as
a causal mechanism requires, rather than at some long lag that would betray a
shared diurnal trend.

**Trigger level.**  For Livingston, where the public O3 release contains a
full-rate suspension witness (``L1:SUS-OM3_M1_DAMP_L_IN1_DQ``, 256 Hz), fit
the single counts-to-metres calibration on a training subset, then measure on
a disjoint validation subset how much of each arch envelope the predictor
``f_n = 2 n |v| / lambda`` explains.

This stage refuses to run on O4a.  The O4a auxiliary release contains 25
channels, all of them calibration lines, input-mode-cleaner wavefront sensors,
mains monitors or parametric-instability monitors; there is no seismometer,
accelerometer or suspension witness of any kind.  See docs/DATA_AVAILABILITY.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _common import banner, base_parser, bootstrap, provenance

from gwglitch.auxcorr import (
    correlation_table,
    hourly_ground_motion,
    lagged_correlation,
    partial_spearman,
    rate_ratio_by_quantile,
)
from gwglitch.catalog import hourly_rate
from gwglitch.dataio import DataFetcher, segments_livetime_per_hour
from gwglitch.morphology import extract_ridge
from gwglitch.qscan import qmap_for_trigger
from gwglitch.scattering import calibrate_witness, predict_arch_frequencies, track_agreement
from gwglitch.tables import read_table, write_table
from gwglitch.utils import JsonSidecar


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--epochs", nargs="+", default=["O3a", "O3b"])
    parser.add_argument("--skip-rate", action="store_true")
    parser.add_argument("--skip-witness", action="store_true")
    parser.add_argument("--n-train", type=int, default=60,
                        help="triggers used to fit the witness calibration")
    parser.add_argument("--n-validate", type=int, default=300,
                        help="disjoint triggers used to measure the agreement")
    parser.add_argument("--days", type=float, default=None,
                        help="restrict each epoch to N days (trial runs; the "
                             "full epoch is several hours)")
    parser.add_argument("--start-day", type=float, default=0.0,
                        help="begin N days after the epoch start. The first "
                             "chunk of O3b has livetime but no auxiliary data, "
                             "so a trial run starting at day 0 finds nothing")
    args = parser.parse_args()

    cfg, log = bootstrap(args, "03_aux_correlation")
    fetcher = DataFetcher(cfg, offline=args.offline)
    derived = cfg.path_for("derived_dir")
    tables = cfg.path_for("table_dir")

    for epoch in args.epochs:
        if cfg.epoch(epoch).run != "O3":
            log.error(
                "epoch %s is in run %s; ground-motion channels are published for "
                "O3 only. See docs/DATA_AVAILABILITY.md.",
                epoch, cfg.epoch(epoch).run,
            )
            return 1

    catalogue = read_table(derived / "o3_scattered_light_all")

    # ------------------------------------------------------------------
    # Rate-level correlation
    # ------------------------------------------------------------------
    if not args.skip_rate:
        all_corr, all_quantiles, all_lags = [], [], []

        for detector in args.detectors:
            for epoch in args.epochs:
                print(banner(f"rate correlation: {detector} {epoch}"))
                ep = cfg.epoch(epoch)

                gps_start = ep.gps_start + args.start_day * 86400.0
                gps_end = (
                    gps_start + args.days * 86400.0
                    if args.days else ep.gps_end
                )
                if args.days or args.start_day:
                    log.warning(
                        "trial run: %s %s restricted to days %.1f-%.1f of the epoch",
                        detector, epoch, args.start_day,
                        args.start_day + (args.days or 0.0),
                    )

                segments = fetcher.segments(detector, epoch).restrict(
                    gps_start, gps_end
                )
                livetime = segments_livetime_per_hour(
                    segments, gps_start, gps_end
                )
                log.info("%s %s livetime: %.1f h", detector, epoch,
                         segments.livetime / 3600.0)

                subset = catalogue.loc[
                    (catalogue["detector"] == detector)
                    & (catalogue["epoch"] == epoch)
                ]
                subset = subset.loc[
                    subset["event_time"].between(gps_start, gps_end)
                ]
                rates = hourly_rate(
                    subset, gps_start, gps_end, livetime_hours=livetime
                )
                log.info("%d live hours, median rate %.2f /h",
                         len(rates), rates["rate"].median())

                ground = hourly_ground_motion(
                    fetcher, cfg, detector, epoch,
                    gps_start=gps_start, gps_end=gps_end, segments=segments,
                )
                corr = correlation_table(rates, ground, detector, epoch, cfg)
                print(corr.to_string(index=False))
                all_corr.append(corr)

                # Strongest channel: quantile table and lag scan.
                best = corr.iloc[0]["channel"]
                joined = rates[["rate"]].join(ground[[best]], how="inner").dropna()

                q = rate_ratio_by_quantile(
                    joined["rate"].to_numpy(), joined[best].to_numpy()
                )
                if not q.empty:
                    q.insert(0, "channel", best)
                    q.insert(0, "epoch", epoch)
                    q.insert(0, "detector", detector)
                    all_quantiles.append(q)
                    log.info("top/bottom quartile rate ratio for %s: %.2f",
                             best, q.attrs.get("top_to_bottom_ratio", np.nan))

                lag = lagged_correlation(
                    joined["rate"].to_numpy(), joined[best].to_numpy()
                )
                lag.insert(0, "channel", best)
                lag.insert(0, "epoch", epoch)
                lag.insert(0, "detector", detector)
                all_lags.append(lag)
                if lag["rho"].notna().any():
                    peak_lag = int(lag.loc[lag["rho"].idxmax(), "lag_hours"])
                    log.info("lag scan peaks at %+d h (zero lag expected)",
                             peak_lag)
                else:
                    log.warning(
                        "lag scan is empty for %s %s: %d hours is too few to "
                        "correlate at every lag", detector, epoch, len(joined),
                    )

                # Does the anthropogenic band survive control for microseism?
                micro = [c for c in ground.columns if "100M_300M" in c]
                anthro = [c for c in ground.columns if "_1_3" in c or "_3_10" in c]
                if micro and anthro:
                    joined2 = rates[["rate"]].join(
                        ground[[micro[0], anthro[0]]], how="inner"
                    ).dropna()
                    r, p = partial_spearman(
                        joined2["rate"].to_numpy(),
                        joined2[anthro[0]].to_numpy(),
                        joined2[micro[0]].to_numpy(),
                    )
                    log.info(
                        "partial correlation of rate with %s, controlling for %s: "
                        "rho = %.3f (p = %.3g)", anthro[0], micro[0], r, p,
                    )

                write_table(
                    rates.reset_index(), derived / f"rates_{detector}_{epoch}"
                )
                write_table(
                    ground.reset_index(), derived / f"groundmotion_{detector}_{epoch}"
                )

        pd.concat(all_corr, ignore_index=True).to_csv(
            tables / "aux_rate_correlations.csv", index=False
        )
        if all_quantiles:
            pd.concat(all_quantiles, ignore_index=True).to_csv(
                tables / "aux_rate_quantiles.csv", index=False
            )
        pd.concat(all_lags, ignore_index=True).to_csv(
            tables / "aux_rate_lagscan.csv", index=False
        )

    # ------------------------------------------------------------------
    # Trigger-level scattering predictor
    # ------------------------------------------------------------------
    if not args.skip_witness:
        results = []
        for detector in args.detectors:
            witness_cfg = cfg.auxiliary.witness[detector]
            if not bool(witness_cfg.get("full_rate_witness_available", True)):
                log.warning(
                    "%s has no full-rate motion witness in the public O3 release "
                    "(%s is only %g Hz); skipping the per-trigger predictor.",
                    detector, witness_cfg["channel"], witness_cfg["sample_rate"],
                )
                continue

            for epoch in args.epochs:
                print(banner(f"scattering predictor: {detector} {epoch}"))
                out = _witness_analysis(
                    cfg, fetcher, log, catalogue, detector, epoch,
                    witness_cfg["channel"], args.n_train, args.n_validate,
                )
                if out is not None:
                    results.append(out)

        if results:
            frame = pd.concat([r["per_trigger"] for r in results], ignore_index=True)
            write_table(frame, derived / "witness_agreement")
            summary = pd.DataFrame([
                {k: v for k, v in r.items() if k != "per_trigger"} for r in results
            ])
            summary.to_csv(tables / "witness_calibration.csv", index=False)
            print(banner("witness summary"))
            print(summary.to_string(index=False))

    JsonSidecar(derived / "aux_correlation.provenance.json").write(
        **provenance(cfg, "03_aux_correlation",
                     detectors=args.detectors, epochs=args.epochs,
                     cache_stats=fetcher.stats)
    )
    log.info("stage 03 complete")
    return 0


def _witness_analysis(
    cfg, fetcher, log, catalogue, detector, epoch, channel,
    n_train: int, n_validate: int,
):
    """Fit the calibration on one subset and measure agreement on another."""
    subset = catalogue.loc[
        (catalogue["detector"] == detector) & (catalogue["epoch"] == epoch)
    ]
    if len(subset) < n_train + 20:
        log.warning("%s %s: only %d triggers, skipping", detector, epoch, len(subset))
        return None

    rng = np.random.default_rng(int(cfg.sampling.seed))
    order = rng.permutation(len(subset))
    train_idx = order[:n_train]
    validate_idx = order[n_train : n_train + n_validate]

    half = float(cfg.strain.window_half_width)
    threshold = float(cfg.qtransform.energy_threshold)
    islands = int(cfg.qtransform.min_island_tiles)
    max_harmonic = int(cfg.scattering.max_harmonic)

    def collect(indices):
        tracks, witnesses, times = [], [], []
        for pos in indices:
            t0 = float(subset.iloc[pos]["event_time"])
            try:
                qmap = qmap_for_trigger(fetcher, cfg, detector, t0)
                track = extract_ridge(qmap, threshold, min_island_bins=islands)
                if track.n_active < 50:
                    continue
                witness = fetcher.auxiliary(channel, t0 - half, t0 + half)
            except Exception as exc:  # noqa: BLE001
                log.debug("witness collection failed at %.4f: %s", t0, exc)
                continue
            tracks.append(track)
            witnesses.append(witness)
            times.append(t0)
        return tracks, witnesses, times

    log.info("collecting %d training triggers", len(train_idx))
    tr_tracks, tr_wit, tr_t0 = collect(train_idx)
    if len(tr_tracks) < 10:
        log.warning("%s %s: only %d usable training triggers, skipping",
                    detector, epoch, len(tr_tracks))
        return None

    fit = calibrate_witness(
        tr_tracks, tr_wit, tr_t0, max_harmonic=max_harmonic
    )

    log.info("measuring agreement on %d validation triggers", len(validate_idx))
    va_tracks, va_wit, va_t0 = collect(validate_idx)

    rows = []
    for track, witness, t0 in zip(va_tracks, va_wit, va_t0):
        prediction = predict_arch_frequencies(
            witness, max_harmonic=max_harmonic, calibration=fit["calibration"]
        )
        result = track_agreement(track, prediction, t0)
        rows.append({"detector": detector, "epoch": epoch,
                     "event_time": t0, "channel": channel, **result})

    per_trigger = pd.DataFrame(rows)
    matched = per_trigger["matched_fraction"].dropna()

    log.info(
        "%s %s: median matched fraction %.3f over %d validation triggers "
        "(%.1f%% above 0.5)",
        detector, epoch, matched.median(), len(matched),
        100.0 * float((matched > 0.5).mean()) if len(matched) else np.nan,
    )

    return {
        "detector": detector,
        "epoch": epoch,
        "channel": channel,
        "calibration_m_per_count": fit["calibration"],
        "calibration_lo": fit["calibration_lo"],
        "calibration_hi": fit["calibration_hi"],
        "training_objective": fit["objective"],
        "n_train": len(tr_tracks),
        "n_validate": len(per_trigger),
        "median_matched_fraction": float(matched.median()) if len(matched) else np.nan,
        "frac_above_half": float((matched > 0.5).mean()) if len(matched) else np.nan,
        "median_pearson_r": float(per_trigger["pearson_r"].median()),
        "per_trigger": per_trigger,
    }


if __name__ == "__main__":
    sys.exit(main())
