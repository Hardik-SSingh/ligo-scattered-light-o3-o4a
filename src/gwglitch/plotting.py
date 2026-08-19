"""Figures.

Every figure in the write-up is produced by a function in this module, so the
LaTeX document and the analysis cannot drift apart.  Functions take already-
computed objects (a ``QMap``, a features table) and return a Matplotlib
``Figure``; none of them fetch data or compute statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .morphology import RidgeTrack
from .qscan import QMap

log = logging.getLogger(__name__)

__all__ = [
    "set_style",
    "savefig",
    "plot_qscan",
    "plot_feature_distributions",
    "plot_rate_timeline",
    "plot_rate_vs_ground_motion",
    "plot_correlation_summary",
    "plot_roc",
    "plot_slow_fast",
    "plot_arch_period_histogram",
    "plot_feature_importance",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = {
    "H1": "#D55E00",
    "L1": "#0072B2",
    "O3a": "#009E73",
    "O3b": "#56B4E9",
    "O4a": "#CC79A7",
    "slow": "#0072B2",
    "fast": "#D55E00",
    "signal": "#0072B2",
    "background": "#999999",
    "accent": "#E69F00",
}


def set_style(context: str = "paper") -> None:
    """Apply the house style.  Called once by each figure script."""
    base = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "image.cmap": "viridis",
    }
    if context == "paper":
        base.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "lines.linewidth": 1.2})
    else:
        base.update({"font.size": 12, "lines.linewidth": 1.8})
    mpl.rcParams.update(base)


def savefig(fig: plt.Figure, path: Path, also_pdf: bool = True) -> Path:
    """Save a figure as PNG (and PDF for LaTeX inclusion)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s", path.with_suffix(".png"))
    return path.with_suffix(".png")


# ---------------------------------------------------------------------------
# Q-scans
# ---------------------------------------------------------------------------


def plot_qscan(
    qmap: QMap,
    track: RidgeTrack | None = None,
    prediction=None,
    title: str | None = None,
    vmax: float | None = None,
    fmax: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render a Q-scan, optionally overlaid with the ridge and the predictor.

    The colour axis is normalised energy ``Z``; the colour bar is labelled in
    those units rather than in SNR so that the threshold used for ridge
    extraction can be read directly off the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
    else:
        fig = ax.figure

    energy = qmap.energy
    freqs = qmap.frequencies
    times = qmap.relative_times

    if fmax is not None:
        keep = freqs <= fmax
        energy = energy[:, keep]
        freqs = freqs[keep]

    vmax = vmax if vmax is not None else float(np.percentile(energy, 99.9))
    mesh = ax.pcolormesh(
        times, freqs, energy.T, vmin=0.0, vmax=vmax, shading="auto", rasterized=True
    )

    if track is not None:
        finite = np.isfinite(track.upper)
        if finite.any():
            ax.plot(
                track.times[finite], track.upper[finite],
                color="white", lw=1.0, alpha=0.85,
                label="arch envelope",
            )

    if prediction is not None:
        pred = prediction.at(qmap.times)
        for i, n in enumerate(pred.harmonics):
            row = pred.frequencies[i]
            if not np.isfinite(row).any() or np.nanmax(row) > freqs[-1] * 1.5:
                continue
            ax.plot(
                qmap.relative_times, row,
                color=PALETTE["accent"], lw=0.8, ls="--", alpha=0.8,
                label=r"$f = 2n|v|/\lambda$" if i == 0 else None,
            )

    ax.set_xlabel(f"Time since {qmap.t0:.3f} (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(freqs[0], freqs[-1])
    ax.grid(False)
    if title:
        ax.set_title(title)
    if track is not None or prediction is not None:
        ax.legend(loc="upper right", framealpha=0.3, facecolor="black",
                  labelcolor="white")

    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Normalised energy")
    return fig


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    group_column: str,
    log_features: tuple[str, ...] = ("energy_total", "ridge_slope_rms",
                                     "ridge_curvature_rms"),
    n_cols: int = 3,
    bins: int = 45,
) -> plt.Figure:
    """Grid of normalised histograms, one panel per feature, split by group."""
    features = [f for f in features if f in df.columns]
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.1 * n_cols, 2.3 * n_rows), squeeze=False
    )
    groups = list(pd.unique(df[group_column].dropna()))

    for k, feature in enumerate(features):
        ax = axes[k // n_cols][k % n_cols]
        values = df[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 10:
            ax.set_visible(False)
            continue

        use_log = feature in log_features and (finite > 0).all()
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1e-9
        edges = (
            np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi), bins)
            if use_log
            else np.linspace(lo, hi, bins)
        )

        for group in groups:
            sub = df.loc[df[group_column] == group, feature].to_numpy(dtype=float)
            sub = sub[np.isfinite(sub)]
            if sub.size < 10:
                continue
            ax.hist(
                sub, bins=edges, density=True, histtype="step", lw=1.3,
                color=PALETTE.get(str(group)), label=str(group),
            )
        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel(feature.replace("_", " "))
        ax.set_yticks([])
        if k == 0:
            ax.legend()

    for k in range(len(features), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_arch_period_histogram(
    df: pd.DataFrame,
    group_column: str = "epoch",
    split_period: float = 1.5,
) -> plt.Figure:
    """Arch repetition period, with the slow/fast split and driving bands.

    The upper axis shows the implied driving frequency ``1 / (2 T)``, so the
    microseism and anthropogenic bands can be marked directly.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    edges = np.logspace(-1.4, 1.2, 60)

    for group in pd.unique(df[group_column].dropna()):
        vals = df.loc[df[group_column] == group, "arch_period"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 10:
            continue
        ax.hist(vals, bins=edges, density=True, histtype="step", lw=1.4,
                color=PALETTE.get(str(group)), label=str(group))

    ax.axvline(split_period, color="k", ls=":", lw=1.0)
    ax.text(split_period * 1.05, ax.get_ylim()[1] * 0.9, "slow $\\rightarrow$",
            fontsize=8)
    ax.text(split_period * 0.95, ax.get_ylim()[1] * 0.9, "$\\leftarrow$ fast",
            fontsize=8, ha="right")

    ax.set_xscale("log")
    ax.set_xlabel(r"Arch repetition period $T_{\rm arch}$ (s)")
    ax.set_ylabel("Probability density")
    ax.legend()

    top = ax.secondary_xaxis(
        "top", functions=(lambda t: 1.0 / (2.0 * t), lambda f: 1.0 / (2.0 * f))
    )
    top.set_xlabel(r"Implied driving frequency $f_{\rm drive}=1/2T_{\rm arch}$ (Hz)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rates and correlations
# ---------------------------------------------------------------------------


def plot_rate_timeline(
    rates: dict[str, pd.DataFrame],
    reference_gps: dict[str, float] | None = None,
) -> plt.Figure:
    """Daily scattered-light rate through each epoch, one panel per epoch."""
    fig, axes = plt.subplots(
        len(rates), 1, figsize=(6.6, 2.0 * len(rates)), squeeze=False, sharey=True
    )
    for ax, (name, table) in zip(axes[:, 0], rates.items()):
        if table.empty:
            ax.set_visible(False)
            continue
        ref = (reference_gps or {}).get(name, table["gps_mid"].min())
        days = (table["gps_mid"] - ref) / 86400.0
        daily = (
            pd.DataFrame({"day": np.floor(days), "rate": table["rate"].to_numpy()})
            .groupby("day")["rate"]
            .mean()
        )
        ax.plot(daily.index, daily.to_numpy(), lw=1.0,
                color=PALETTE.get(name.split()[-1], "#333333"))
        ax.set_ylabel("triggers h$^{-1}$")
        ax.set_title(name, fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("Days into epoch")
    fig.tight_layout()
    return fig


def plot_rate_vs_ground_motion(
    rates: pd.DataFrame,
    ground_motion: pd.Series,
    channel: str,
    n_quantiles: int = 12,
) -> plt.Figure:
    """Hourly rate against ground motion: scatter plus binned medians."""
    joined = pd.DataFrame({"rate": rates["rate"]}).join(
        ground_motion.rename("driver"), how="inner"
    ).dropna()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(joined["driver"], joined["rate"], s=4, alpha=0.15,
               color=PALETTE["background"], edgecolors="none")

    edges = np.percentile(joined["driver"], np.linspace(0, 100, n_quantiles + 1))
    centres, medians, los, his = [], [], [], []
    for i in range(n_quantiles):
        sel = joined["driver"].between(edges[i], edges[i + 1])
        if sel.sum() < 5:
            continue
        r = joined.loc[sel, "rate"]
        centres.append(np.median(joined.loc[sel, "driver"]))
        medians.append(r.median())
        los.append(r.quantile(0.25))
        his.append(r.quantile(0.75))

    ax.plot(centres, medians, "o-", color=PALETTE["signal"], ms=4,
            label="binned median")
    ax.fill_between(centres, los, his, color=PALETTE["signal"], alpha=0.2,
                    label="interquartile range")

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(f"{channel}\nband-limited RMS (counts)")
    ax.set_ylabel("Scattered-light rate (h$^{-1}$)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_summary(table: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Forest plot of Spearman rho with block-bootstrap intervals."""
    sub = table.reindex(table["spearman_rho"].abs().sort_values(
        ascending=False).index).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(sub) + 1.4))
    y = np.arange(len(sub))
    ax.errorbar(
        sub["spearman_rho"], y,
        xerr=[sub["spearman_rho"] - sub["ci_low"], sub["ci_high"] - sub["spearman_rho"]],
        fmt="o", ms=4, capsize=2.5, lw=1.0,
        color=PALETTE["signal"],
    )
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [c.replace(".rms", "").replace("ISI-GND_STS_", "") for c in sub["channel"]],
        fontsize=7,
    )
    ax.set_xlabel(r"Spearman $\rho$ (hourly rate vs band-limited RMS)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classifier diagnostics
# ---------------------------------------------------------------------------


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    operating_point: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(fpr, tpr, color=PALETTE["signal"], label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls=":", color="k", lw=0.8)
    if operating_point is not None:
        ax.plot(*operating_point, "o", ms=6, color=PALETTE["accent"],
                label=f"operating point ({operating_point[0]:.3f}, "
                      f"{operating_point[1]:.3f})")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate (control classes)")
    ax.set_ylabel("Efficiency (scattered light)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: dict[str, float], top_n: int = 15) -> plt.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1])[-top_n:]
    names = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(5.4, 0.28 * len(items) + 1.2))
    ax.barh(np.arange(len(items)), values, color=PALETTE["signal"])
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Gradient-boosting feature importance")
    fig.tight_layout()
    return fig


def plot_slow_fast(breakdown: pd.DataFrame) -> plt.Figure:
    """Stacked slow/fast fractions per detector and epoch."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels = [f"{r.detector}\n{r.epoch}" for r in breakdown.itertuples()]
    x = np.arange(len(breakdown))

    ax.bar(x, breakdown["frac_slow"], color=PALETTE["slow"], label="slow")
    ax.bar(x, breakdown["frac_fast"], bottom=breakdown["frac_slow"],
           color=PALETTE["fast"], label="fast")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of classified triggers")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    return fig
