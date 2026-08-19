#!/usr/bin/env python
"""Stage 04 -- train the strain-only scattered-light identifier on O3.

Combines the target and control feature tables from stage 02, trains a
gradient-boosted tree with stratified K-fold cross-validation, fixes the
operating point at the configured false-positive rate, and saves the fitted
pipeline for stage 05 to apply to O4a.

What the cross-validated numbers do and do not mean
---------------------------------------------------
The efficiency and false-positive rate are measured against the *Gravity Spy
control classes*, so they answer "how well does this separate scattered light
from other known glitch classes in O3".  They do not measure performance
against whatever unlabelled O4a contains, and they cannot: no O4a labels
exist.  The folds are also not grouped in time, so triggers minutes apart can
land in different folds and the AUC is mildly optimistic.  Both caveats are
carried into the write-up rather than buried.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from _common import banner, base_parser, bootstrap, provenance

from gwglitch.constants import FEATURE_NAMES
from gwglitch.identify import score_frame, train_identifier
from gwglitch.plotting import plot_feature_importance, plot_roc, savefig, set_style
from gwglitch.tables import read_table, write_table
from gwglitch.utils import JsonSidecar


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg, log = bootstrap(args, "04_train_identifier")
    derived = cfg.path_for("derived_dir")
    tables = cfg.path_for("table_dir")

    try:
        target = read_table(derived / "o3_features_scattered")
        control = read_table(derived / "o3_features_control")
    except FileNotFoundError as exc:
        log.error("%s -- run scripts/02_run_qscans.py first", exc)
        return 1

    features = pd.concat([target, control], ignore_index=True)
    log.info("training set: %d target, %d control", len(target), len(control))

    print(banner("class balance"))
    print(features["ml_label"].value_counts().to_string())

    bundle = train_identifier(cfg, features)
    print(banner("cross-validated performance"))
    print(bundle.summary())

    print(banner("feature importance"))
    for name, value in list(bundle.feature_importance.items())[:12]:
        print(f"  {name:<26s} {value:.4f}")

    model_path = derived / "identifier.joblib"
    bundle.save(model_path)

    scored = score_frame(bundle, features)
    write_table(scored, derived / "o3_features_scored")

    # A full-sample ROC, for the figure only; the quoted numbers are the
    # cross-validated ones above.
    y = (scored["ml_label"] == cfg.gravityspy.target_label).to_numpy(dtype=int)
    fpr, tpr, _ = roc_curve(y, scored["score"].to_numpy())
    k = int(np.searchsorted(fpr, bundle.target_fpr, side="right") - 1)

    pd.DataFrame({
        "metric": ["cv_auc", "cv_auc_std", "cv_efficiency", "cv_efficiency_std",
                   "threshold", "target_fpr", "n_signal", "n_background"],
        "value": [bundle.cv_auc, bundle.cv_auc_std, bundle.cv_efficiency,
                  bundle.cv_efficiency_std, bundle.threshold, bundle.target_fpr,
                  bundle.n_signal, bundle.n_background],
    }).to_csv(tables / "identifier_performance.csv", index=False)

    pd.DataFrame(
        sorted(bundle.feature_importance.items(), key=lambda kv: -kv[1]),
        columns=["feature", "importance"],
    ).to_csv(tables / "identifier_feature_importance.csv", index=False)

    if not args.no_figures:
        set_style()
        figures = cfg.path_for("figure_dir")
        savefig(
            plot_roc(fpr, tpr, bundle.cv_auc,
                     operating_point=(float(fpr[max(k, 0)]), float(tpr[max(k, 0)]))),
            figures / "identifier_roc",
        )
        savefig(
            plot_feature_importance(bundle.feature_importance),
            figures / "identifier_feature_importance",
        )

    JsonSidecar(derived / "identifier.provenance.json").write(
        **provenance(
            cfg, "04_train_identifier",
            model=str(cfg.classifier.model),
            n_features=len(FEATURE_NAMES),
            cv_auc=bundle.cv_auc,
            cv_efficiency=bundle.cv_efficiency,
            threshold=bundle.threshold,
            control_classes=list(cfg.sampling.control_classes),
        )
    )
    log.info("stage 04 complete; model at %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
