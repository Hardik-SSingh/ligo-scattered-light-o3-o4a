#!/usr/bin/env python
"""Stage 07 -- render every figure the write-up needs.

Figures that depend only on earlier stages' tables are regenerated from those
tables.  The example Q-scans are the exception: they re-fetch strain, because
a Q-map is far too large to keep for every trigger and the point of the figure
is to show a real one.

Output goes to ``results/figures`` as both PNG and PDF.  ``paper/main.tex``
includes the PDFs, so re-running this stage updates the document.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _common import banner, base_parser, bootstrap

from gwglitch.dataio import DataFetcher
from gwglitch.morphology import extract_ridge
from gwglitch.plotting import (
    plot_arch_period_histogram,
    plot_correlation_summary,
    plot_feature_distributions,
    plot_qscan,
    plot_rate_vs_ground_motion,
    plot_slow_fast,
    savefig,
    set_style,
)
from gwglitch.qscan import qmap_for_trigger
from gwglitch.scattering import predict_arch_frequencies
from gwglitch.tables import read_table


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--n-examples", type=int, default=4,
                        help="how many example Q-scans to render")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=["examples", "validation", "distributions",
                                 "correlation", "predictor"])
    args = parser.parse_args()

    cfg, log = bootstrap(args, "07_make_figures")
    set_style()
    figures = cfg.path_for("figure_dir")
    derived = cfg.path_for("derived_dir")
    tables = cfg.path_for("table_dir")
    wanted = set(args.only) if args.only else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    # ------------------------------------------------------------------
    if want("validation"):
        print(banner("validation figure: simulated injections"))
        _validation_figure(cfg, figures, log)

    # ------------------------------------------------------------------
    if want("examples"):
        print(banner("example Q-scans"))
        _example_qscans(cfg, args, figures, derived, log)

    # ------------------------------------------------------------------
    if want("distributions"):
        print(banner("feature distributions"))
        try:
            combined = read_table(derived / "population_combined")
        except FileNotFoundError:
            log.warning("population_combined missing; run stage 06 first")
        else:
            headline = ["arch_period", "f_max", "harmonic_spacing", "n_harmonics",
                        "duration", "bandwidth", "frac_energy_below_40hz",
                        "snr_qscan", "band_concentration"]
            savefig(plot_feature_distributions(combined, headline, "sample"),
                    figures / "feature_distributions_o3_vs_o4a")
            savefig(
                plot_arch_period_histogram(
                    combined, "sample",
                    float(cfg.scattering.slow_fast_split_period)),
                figures / "arch_period_o3_vs_o4a",
            )

        path = tables / "slow_fast_breakdown.csv"
        if path.exists():
            savefig(plot_slow_fast(pd.read_csv(path)),
                    figures / "slow_fast_fractions")

    # ------------------------------------------------------------------
    if want("correlation"):
        print(banner("auxiliary correlation figures"))
        path = tables / "aux_rate_correlations.csv"
        if not path.exists():
            log.warning("%s missing; run stage 03 first", path.name)
        else:
            corr = pd.read_csv(path)
            for (detector, epoch), group in corr.groupby(["detector", "epoch"]):
                savefig(plot_correlation_summary(group),
                        figures / f"aux_correlation_{detector}_{epoch}")

                try:
                    rates = read_table(derived / f"rates_{detector}_{epoch}")
                    ground = read_table(derived / f"groundmotion_{detector}_{epoch}")
                except FileNotFoundError:
                    continue
                rates = rates.set_index("hour_bin")
                ground = ground.set_index("hour_bin")
                best = group.sort_values("spearman_rho", ascending=False)
                best = str(best.iloc[0]["channel"])
                if best in ground.columns:
                    savefig(
                        plot_rate_vs_ground_motion(rates, ground[best], best),
                        figures / f"rate_vs_groundmotion_{detector}_{epoch}",
                    )

    # ------------------------------------------------------------------
    if want("predictor"):
        print(banner("scattering-predictor overlay"))
        _predictor_figure(cfg, figures, derived, tables, log, args)

    log.info("stage 07 complete; figures in %s", figures)
    return 0


def _validation_figure(cfg, figures, log) -> None:
    """Q-scans of the two canonical injections, with truth annotated.

    This is the figure that justifies every measured number downstream: it
    shows the extractor recovering parameters that were put in by hand.
    """
    import matplotlib.pyplot as plt

    from gwglitch.morphology import extract_features
    from gwglitch.qscan import compute_qmap
    from gwglitch.simulate import (
        fast_scattering_example,
        make_scattering_strain,
        slow_scattering_example,
    )

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.0))
    for ax, injection, name in (
        (axes[0], slow_scattering_example(seed=3), "slow"),
        (axes[1], fast_scattering_example(seed=3), "fast"),
    ):
        data = make_scattering_strain(injection)
        qmap = compute_qmap(data, injection.event_centre, cfg)
        features, track = extract_features(
            qmap,
            float(cfg.qtransform.energy_threshold),
            int(cfg.qtransform.min_island_tiles),
            return_track=True,
        )
        plot_qscan(
            qmap, track=track, ax=ax, fmax=120.0,
            title=(
                f"simulated {name} scattering: "
                rf"$T_{{\rm arch}}$ = {injection.arch_period:.2f} s injected, "
                f"{features['arch_period']:.2f} s recovered; "
                rf"$f_{{\rm apex}}$ = {injection.max_arch_frequency:.0f} Hz injected, "
                f"{features['f_max']:.0f} Hz recovered"
            ),
        )
        log.info("%s injection: recovered arch period %.3f s (truth %.3f s)",
                 name, features["arch_period"], injection.arch_period)
    fig.tight_layout()
    savefig(fig, figures / "validation_injections")


def _example_qscans(cfg, args, figures, derived, log) -> None:
    """A handful of real Q-scans, chosen to span the slow/fast range."""
    try:
        features = read_table(derived / "o3_features_scattered")
    except FileNotFoundError:
        log.warning("o3_features_scattered missing; run stage 02 first")
        return

    fetcher = DataFetcher(cfg, offline=args.offline)
    split = float(cfg.scattering.slow_fast_split_period)
    threshold = float(cfg.qtransform.energy_threshold)
    islands = int(cfg.qtransform.min_island_tiles)

    loud = features.loc[np.isfinite(features["arch_period"])]
    loud = loud.nlargest(400, "snr_qscan")
    slow = loud.loc[loud["arch_period"] > split].nlargest(args.n_examples // 2, "snr_qscan")
    fast = loud.loc[loud["arch_period"] <= split].nlargest(
        args.n_examples - len(slow), "snr_qscan"
    )

    for kind, subset in (("slow", slow), ("fast", fast)):
        for i, row in enumerate(subset.itertuples(), start=1):
            try:
                qmap = qmap_for_trigger(fetcher, cfg, row.detector, row.event_time)
                track = extract_ridge(qmap, threshold, min_island_bins=islands)
            except Exception as exc:  # noqa: BLE001
                log.warning("example %s %d failed: %s", kind, i, exc)
                continue
            title = (
                f"{row.detector} {row.epoch}  GPS {row.event_time:.2f}\n"
                rf"$T_{{\rm arch}}$ = {row.arch_period:.2f} s, "
                rf"$f_{{\rm max}}$ = {row.f_max:.0f} Hz, SNR = {row.snr_qscan:.0f}"
            )
            savefig(
                plot_qscan(qmap, track=track, title=title, fmax=120.0),
                figures / f"example_{kind}_{i}_{row.detector}",
            )


def _predictor_figure(cfg, figures, derived, tables, log, args) -> None:
    """Overlay the auxiliary-derived harmonic comb on a real arch train."""
    path = tables / "witness_calibration.csv"
    if not path.exists():
        log.warning("witness_calibration.csv missing; run stage 03 first")
        return
    calibration = pd.read_csv(path)

    try:
        agreement = read_table(derived / "witness_agreement")
    except FileNotFoundError:
        log.warning("witness_agreement missing; run stage 03 first")
        return

    fetcher = DataFetcher(cfg, offline=args.offline)
    best = agreement.nlargest(3, "matched_fraction")

    for i, row in enumerate(best.itertuples(), start=1):
        entry = calibration.loc[
            (calibration["detector"] == row.detector)
            & (calibration["epoch"] == row.epoch)
        ]
        if entry.empty:
            continue
        alpha = float(entry.iloc[0]["calibration_m_per_count"])
        half = float(cfg.strain.window_half_width)

        try:
            qmap = qmap_for_trigger(fetcher, cfg, row.detector, row.event_time)
            track = extract_ridge(
                qmap, float(cfg.qtransform.energy_threshold),
                min_island_bins=int(cfg.qtransform.min_island_tiles),
            )
            witness = fetcher.auxiliary(
                row.channel, row.event_time - half, row.event_time + half
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("predictor example %d failed: %s", i, exc)
            continue

        prediction = predict_arch_frequencies(
            witness, max_harmonic=int(cfg.scattering.max_harmonic), calibration=alpha
        )
        title = (
            f"{row.detector} {row.epoch}  GPS {row.event_time:.2f}\n"
            f"{row.channel}, {row.matched_fraction:.0%} of the envelope explained"
        )
        savefig(
            plot_qscan(qmap, track=track, prediction=prediction,
                       title=title, fmax=120.0),
            figures / f"predictor_overlay_{i}_{row.detector}",
        )


if __name__ == "__main__":
    sys.exit(main())
