#!/usr/bin/env python
"""Stage 06 -- compare the O3 and O4a scattered-light populations.

Produces the three results the study is built around:

1. the rate of scattered-light transients per hour of analysed livetime in
   each run, with Poisson intervals;
2. the slow/fast decomposition -- the fraction of triggers whose arch
   repetition period implies microseismic rather than anthropogenic driving --
   and how it changed between runs;
3. a feature-by-feature comparison with Cliff's delta as the effect size and
   Benjamini-Hochberg-corrected KS p-values.

The comparison is deliberately run in two modes.  ``--mode like-for-like``
compares the O4a search output against the *same search* run on O3, so the
trigger generator and the classifier are identical on both sides and only the
detector has changed.  ``--mode catalogue`` compares against the Gravity Spy
O3 sample instead, which has better labels but a different trigger generator.
Where the two modes agree, the result is robust; where they disagree, the
difference is a selection effect and is reported as such.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _common import banner, base_parser, bootstrap, provenance

from gwglitch.constants import FEATURE_NAMES
from gwglitch.latex import code, fmt, interval, write_macros, write_table_body
from gwglitch.plotting import (
    plot_arch_period_histogram,
    plot_feature_distributions,
    plot_slow_fast,
    savefig,
    set_style,
)
from gwglitch.population import (
    compare_all_features,
    poisson_rate,
    proportion_difference,
    slow_fast_breakdown,
)
from gwglitch.scattering import classify_slow_fast
from gwglitch.tables import read_table, write_table
from gwglitch.utils import JsonSidecar


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["like-for-like", "catalogue"],
                        default="like-for-like")
    parser.add_argument("--o3-epoch", default="O3b",
                        help="which O3 epoch to compare against")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg, log = bootstrap(args, "06_population_comparison")
    derived = cfg.path_for("derived_dir")
    tables = cfg.path_for("table_dir")

    o4a = _load_search(derived, "o4a", log)
    if o4a is None:
        return 1

    if args.mode == "like-for-like":
        o3 = _load_search(derived, args.o3_epoch.lower(), log)
        if o3 is None:
            log.error(
                "like-for-like mode needs the same search run on O3:\n"
                "    python scripts/05_search_o4a.py --epoch %s", args.o3_epoch,
            )
            return 1
        o3_label = f"{args.o3_epoch} (same search)"
    else:
        try:
            o3 = read_table(derived / "o3_features_scattered")
            o3 = o3.loc[o3["epoch"] == args.o3_epoch]
        except FileNotFoundError as exc:
            log.error("%s -- run scripts/02_run_qscans.py first", exc)
            return 1
        o3_label = f"{args.o3_epoch} (Gravity Spy)"

    log.info("comparing %d O4a against %d O3 triggers (%s)",
             len(o4a), len(o3), o3_label)

    combined = pd.concat(
        [o3.assign(sample="O3"), o4a.assign(sample="O4a")], ignore_index=True
    )

    # ------------------------------------------------------------------
    # 1. Rates
    # ------------------------------------------------------------------
    print(banner("1. rate per hour of analysed livetime"))
    rate_rows = []
    for tag, name in (("o3", args.o3_epoch.lower()), ("o4a", "o4a")):
        path = tables / f"{name}_search_livetime.csv"
        if not path.exists():
            log.warning("%s not found; rates for %s will be omitted", path.name, tag)
            continue
        livetime = pd.read_csv(path)
        for row in livetime.itertuples():
            est = poisson_rate(int(row.n_scattering), float(row.analysed_livetime_hours))
            rate_rows.append({
                "sample": tag.upper(), "detector": row.detector, "epoch": row.epoch,
                **est.as_row(),
                "coverage_fraction": float(row.coverage_fraction),
            })
    rates = pd.DataFrame(rate_rows)
    if not rates.empty:
        rates.to_csv(tables / "population_rates.csv", index=False)
        print(rates.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Slow / fast decomposition
    # ------------------------------------------------------------------
    print(banner("2. slow / fast decomposition"))
    breakdown = slow_fast_breakdown(combined, cfg)
    breakdown.to_csv(tables / "slow_fast_breakdown.csv", index=False)
    print(breakdown.to_string(index=False))

    split = float(cfg.scattering.slow_fast_split_period)
    labels = {
        name: np.array([classify_slow_fast(p, split)
                        for p in group["arch_period"].to_numpy()])
        for name, group in combined.groupby("sample")
    }
    shift_rows = []
    if "O3" in labels and "O4a" in labels:
        for det in sorted(set(combined["detector"])):
            sel3 = (combined["sample"] == "O3") & (combined["detector"] == det)
            sel4 = (combined["sample"] == "O4a") & (combined["detector"] == det)
            lab3 = np.array([classify_slow_fast(p, split)
                             for p in combined.loc[sel3, "arch_period"]])
            lab4 = np.array([classify_slow_fast(p, split)
                             for p in combined.loc[sel4, "arch_period"]])
            n3 = int((lab3 != "unknown").sum())
            n4 = int((lab4 != "unknown").sum())
            if n3 < 20 or n4 < 20:
                continue
            result = proportion_difference(
                int((lab3 == "slow").sum()), n3, int((lab4 == "slow").sum()), n4
            )
            shift_rows.append({"detector": det, "slow_fraction_O3": result["p1"],
                               "slow_fraction_O4a": result["p2"], **result})
            log.info(
                "%s: slow fraction %.3f (O3) -> %.3f (O4a), change %.3f "
                "[%.3f, %.3f]",
                det, result["p1"], result["p2"], result["difference"],
                result["ci_low"], result["ci_high"],
            )
    if shift_rows:
        pd.DataFrame(shift_rows).to_csv(
            tables / "slow_fraction_shift.csv", index=False
        )

    # ------------------------------------------------------------------
    # 3. Feature-by-feature comparison
    # ------------------------------------------------------------------
    print(banner("3. morphology feature comparison"))
    comparison = compare_all_features(
        combined, "sample", "O3", "O4a", list(FEATURE_NAMES), cfg
    )
    comparison.to_csv(tables / "feature_comparison.csv", index=False)
    show = comparison[
        ["feature", "median_a", "median_b", "delta", "p_adjusted", "significant"]
    ]
    print(show.to_string(index=False))

    strongest = comparison.iloc[0]
    log.info(
        "largest effect: %s, median %.4g (O3) vs %.4g (O4a), Cliff's delta %.3f",
        strongest["feature"], strongest["median_a"], strongest["median_b"],
        strongest["delta"],
    )

    write_table(combined, derived / "population_combined")

    # ------------------------------------------------------------------
    print(banner("4. LaTeX fragments"))
    _emit_latex(cfg, log, tables, rates, breakdown, comparison, shift_rows)

    # ------------------------------------------------------------------
    if not args.no_figures:
        set_style()
        figures = cfg.path_for("figure_dir")
        headline = [
            "arch_period", "f_max", "harmonic_spacing", "n_harmonics",
            "duration", "bandwidth", "frac_energy_below_40hz", "snr_qscan",
            "ridge_slope_rms",
        ]
        savefig(
            plot_feature_distributions(combined, headline, "sample"),
            figures / "feature_distributions_o3_vs_o4a",
        )
        savefig(
            plot_arch_period_histogram(combined, "sample", split),
            figures / "arch_period_o3_vs_o4a",
        )
        if not breakdown.empty:
            savefig(plot_slow_fast(breakdown), figures / "slow_fast_fractions")

    JsonSidecar(derived / "population_comparison.provenance.json").write(
        **provenance(
            cfg, "06_population_comparison",
            mode=args.mode, o3_epoch=args.o3_epoch,
            n_o3=int(len(o3)), n_o4a=int(len(o4a)),
            slow_fast_split_period=split,
        )
    )
    log.info("stage 06 complete")
    return 0


def _emit_latex(cfg, log, tables, rates, breakdown, comparison, shift_rows):
    """Write paper/numbers.tex and the generated table bodies.

    Only tables whose source CSV exists are emitted, so a partial pipeline run
    still produces a compilable document: the section files fall back to their
    placeholder rows for anything missing.
    """
    paper = cfg.root / "paper"
    generated = paper / "generated"

    # -- table bodies ---------------------------------------------------
    def emit(name, frame, columns, formatters=None, max_rows=None):
        try:
            write_table_body(
                generated / f"{name}.tex", frame, columns, formatters, max_rows
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not emit %s: %s", name, exc)

    catalog_path = tables / "catalog_summary.csv"
    if catalog_path.exists():
        summary = pd.read_csv(catalog_path)
        target_col = f"n_{cfg.gravityspy.target_label}"
        emit(
            "tab_sample", summary,
            ["detector", "epoch", "n_triggers", target_col, "fraction",
             "median_snr"],
            {
                "detector": str, "epoch": str,
                "n_triggers": lambda v: fmt(v, thousands=True),
                target_col: lambda v: fmt(v, thousands=True),
                "fraction": lambda v: fmt(v, percent=True, precision=2),
                "median_snr": lambda v: fmt(v, precision=3),
            },
        )

    if not breakdown.empty:
        emit(
            "tab_slowfast", breakdown,
            ["detector", "epoch", "n_total", "frac_slow", "frac_fast",
             "slow_in_microseism_band", "fast_in_anthropogenic_band"],
            {
                "detector": str, "epoch": str,
                "n_total": lambda v: fmt(v, thousands=True),
                "frac_slow": lambda v: fmt(v, percent=True, precision=2),
                "frac_fast": lambda v: fmt(v, percent=True, precision=2),
                "slow_in_microseism_band": lambda v: fmt(v, percent=True, precision=2),
                "fast_in_anthropogenic_band": lambda v: fmt(v, percent=True, precision=2),
            },
        )

    corr_path = tables / "aux_rate_correlations.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        corr = corr.assign(
            short=[c.replace("L1:ISI-GND_STS_", "").replace("H1:ISI-GND_STS_", "")
                     .replace(".rms", "") for c in corr["channel"]],
            band=[_band_of(c) for c in corr["channel"]],
            ci=[interval(lo, hi) for lo, hi in zip(corr["ci_low"], corr["ci_high"])],
        )
        emit(
            "tab_ratecorr", corr.head(6),
            ["short", "band", "spearman_rho", "ci"],
            {"short": code, "band": str,
             "spearman_rho": lambda v: fmt(v, precision=2), "ci": str},
        )

    witness_path = tables / "witness_calibration.csv"
    if witness_path.exists():
        wit = pd.read_csv(witness_path)
        wit = wit.assign(
            crange=[interval(lo, hi) for lo, hi in
                    zip(wit["calibration_lo"], wit["calibration_hi"])]
        )
        emit(
            "tab_witness", wit,
            ["epoch", "calibration_m_per_count", "crange", "n_validate",
             "median_matched_fraction", "median_pearson_r"],
            {
                "epoch": str,
                "calibration_m_per_count": lambda v: fmt(v, scientific=True, precision=2),
                "crange": str,
                "n_validate": lambda v: fmt(v, thousands=True),
                "median_matched_fraction": lambda v: fmt(v, percent=True, precision=2),
                "median_pearson_r": lambda v: fmt(v, precision=2),
            },
        )

    search_frames = []
    for tag in ("o3b", "o3a", "o4a"):
        path = tables / f"{tag}_search_livetime.csv"
        if path.exists():
            search_frames.append(pd.read_csv(path))
    if search_frames:
        search = pd.concat(search_frames, ignore_index=True)
        search = search.sort_values(["detector", "epoch"])
        emit(
            "tab_search", search,
            ["detector", "epoch", "n_windows", "analysed_livetime_hours",
             "coverage_fraction", "n_scattering", "rate_per_hour"],
            {
                "detector": str, "epoch": str,
                "n_windows": lambda v: fmt(v, thousands=True),
                "analysed_livetime_hours": lambda v: fmt(v, precision=4),
                "coverage_fraction": lambda v: fmt(v, percent=True, precision=2),
                "n_scattering": lambda v: fmt(v, thousands=True),
                "rate_per_hour": lambda v: fmt(v, precision=3),
            },
        )

    emit(
        "tab_featurecomp", comparison,
        ["feature", "median_a", "median_b", "delta", "p_adjusted"],
        {
            "feature": lambda v: code(str(v)),
            "median_a": lambda v: fmt(v, precision=3),
            "median_b": lambda v: fmt(v, precision=3),
            "delta": lambda v: fmt(v, precision=2),
            "p_adjusted": lambda v: fmt(v, precision=2, scientific=True),
        },
        max_rows=8,
    )

    # -- inline numbers -------------------------------------------------
    macros: dict[str, str] = {}
    strongest = comparison.iloc[0]
    macros["strongest_feature"] = code(str(strongest["feature"]))
    macros["strongest_delta"] = fmt(strongest["delta"], precision=2)

    for row in shift_rows:
        det = str(row["detector"]).lower()
        macros[f"slow_shift_{det}"] = (
            f"{fmt(row['difference'], precision=2)} "
            f"{interval(row['ci_low'], row['ci_high'])}"
        )
        macros[f"slow_fraction_o3_{det}"] = fmt(row["p1"], percent=True, precision=2)
        macros[f"slow_fraction_o4a_{det}"] = fmt(row["p2"], percent=True, precision=2)

    if not rates.empty:
        for det in sorted(set(rates["detector"])):
            sub = rates.loc[rates["detector"] == det]
            o3 = sub.loc[sub["sample"] == "O3", "rate_per_hour"]
            o4 = sub.loc[sub["sample"] == "O4A", "rate_per_hour"]
            if len(o3) and len(o4) and float(o3.iloc[0]) > 0:
                macros[f"rate_ratio_{det.lower()}"] = fmt(
                    float(o4.iloc[0]) / float(o3.iloc[0]), precision=2
                )
        o4a_cov = rates.loc[rates["sample"] == "O4A", "coverage_fraction"]
        if len(o4a_cov):
            macros["coverage_o4a"] = fmt(
                float(o4a_cov.mean()), percent=True, precision=2
            )

    perf_path = tables / "identifier_performance.csv"
    if perf_path.exists():
        perf = pd.read_csv(perf_path).set_index("metric")["value"]
        macros["classifier_auc"] = fmt(perf.get("cv_auc"), precision=4)
        macros["classifier_efficiency"] = fmt(
            perf.get("cv_efficiency"), percent=True, precision=3
        )

    if corr_path.exists():
        corr = pd.read_csv(corr_path).sort_values("spearman_rho", ascending=False)
        if len(corr):
            macros["strongest_channel"] = code(
                str(corr.iloc[0]["channel"]).replace(".rms", "")
            )
            macros["strongest_rho"] = fmt(corr.iloc[0]["spearman_rho"], precision=2)

    quant_path = tables / "aux_rate_quantiles.csv"
    if quant_path.exists():
        quant = pd.read_csv(quant_path)
        grouped = quant.groupby(["detector", "epoch"])
        ratios = []
        for _, group in grouped:
            group = group.sort_values("quantile")
            bottom = float(group.iloc[0]["mean_rate"])
            if bottom > 0:
                ratios.append(float(group.iloc[-1]["mean_rate"]) / bottom)
        if ratios:
            macros["quartile_ratio"] = fmt(float(np.median(ratios)), precision=2)

    lag_path = tables / "aux_rate_lagscan.csv"
    if lag_path.exists():
        lag = pd.read_csv(lag_path)
        peak = lag.loc[lag["rho"].idxmax(), "lag_hours"]
        macros["lag_peak"] = f"{int(peak):+d}"

    write_macros(paper / "numbers.tex", macros)
    log.info(
        "paper/numbers.tex written with %d macros; recompile paper/main.tex "
        "to replace the placeholders", len(macros),
    )


def _band_of(channel: str) -> str:
    """Human-readable ground-motion band from a BLRMS channel name."""
    if "100M_300M" in channel:
        return "microseism"
    if "30M_100M" in channel:
        return "earthquake"
    if "_1_3" in channel or "_3_10" in channel:
        return "anthropogenic"
    if "10_30" in channel or "30_100" in channel:
        return "high frequency"
    return "---"


def _load_search(derived, epoch_tag: str, log):
    try:
        frame = read_table(derived / f"{epoch_tag}_candidates")
    except FileNotFoundError:
        log.error(
            "no search output for %s -- run "
            "scripts/05_search_o4a.py --epoch %s", epoch_tag, epoch_tag.upper(),
        )
        return None
    if "is_scattering" in frame.columns:
        frame = frame.loc[frame["is_scattering"]]
    return frame.reset_index(drop=True)


if __name__ == "__main__":
    sys.exit(main())
