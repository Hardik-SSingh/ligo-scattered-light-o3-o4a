"""Transferring the O3 definition of "scattered light" to O4a.

Why this module exists
----------------------
The O3 half of this study rests on the Gravity Spy machine-learning
classification release (Zenodo 5649212), which supplies labelled triggers.
**No equivalent public catalogue exists for O4a**, and the O4a auxiliary
release contains no motion witness from which scattering times could be
predicted.  The only public O4a input is the strain itself.

So the O4a sample is built in three steps:

1. *Trigger generation.*  A cheap excess-power finder on whitened,
   band-passed strain stands in for Omicron, which cannot be run here.  It is
   deliberately loose: its job is to lose no scattering, not to be pure.
2. *Feature extraction.*  Identical Q-scan and morphology code as for O3, so
   the feature vectors are directly comparable.
3. *Classification.*  A gradient-boosted tree trained on labelled O3 triggers
   (Scattered_Light against the configured control classes) scores each
   candidate.  The operating point is fixed on held-out O3 folds at a chosen
   false-positive rate, then frozen.

Every quantity reported for O4a therefore inherits an O3-derived definition,
and the paper must -- and does -- state the assumption that the morphological
boundary of the class did not move between runs.  The efficiency and purity
measured on the O3 validation folds are the quantitative statement of how
well that transfer is expected to hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .constants import FEATURE_NAMES
from .dataio import DataFetcher, TimeSeriesData
from .morphology import extract_features
from .qscan import compute_qmap

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierBundle",
    "find_triggers",
    "search_segment",
    "train_identifier",
    "score_frame",
]


# ---------------------------------------------------------------------------
# Trigger generation (an Omicron stand-in)
# ---------------------------------------------------------------------------


def find_triggers(
    data: TimeSeriesData,
    band: tuple[float, float] = (10.0, 120.0),
    z_threshold: float = 5.0,
    rms_stride: float = 0.25,
    cluster_window: float = 4.0,
    edge_pad: float = 8.0,
    fftlength: float = 8.0,
) -> pd.DataFrame:
    """Locate excess-power transients in a stretch of strain.

    The statistic is the ``rms_stride``-second RMS of whitened, band-passed
    strain, converted to a robust z-score using the median and the
    median-absolute-deviation of the whole stretch.  Robust statistics matter:
    during a scattering-heavy hour a large fraction of the data is glitching,
    and a mean/standard-deviation normalisation would be dragged upward by the
    very signal it is meant to detect.

    Parameters
    ----------
    band
        Analysis band.  Slow scattering rarely exceeds 60 Hz and fast
        scattering rarely exceeds 120 Hz; going higher only admits blips.
    z_threshold
        Detection threshold on the robust z-score.
    cluster_window
        Minimum separation between reported triggers.  A scattering arch
        train is one physical event lasting several seconds; without
        clustering it would be reported as a dozen triggers.
    edge_pad
        Candidates within this distance of either end are discarded, because
        the whitening filter transient lives there.

    Returns
    -------
    DataFrame with columns ``time``, ``z``, ``rms``.
    """
    fs = data.sample_rate
    if data.duration < 4 * fftlength:
        raise ValueError(
            f"need at least {4 * fftlength} s of data to whiten with "
            f"fftlength={fftlength}; got {data.duration:.1f} s"
        )

    ts = data.to_gwpy()
    white = ts.whiten(fftlength=fftlength, overlap=fftlength / 2)
    lo, hi = band
    white = white.bandpass(lo, min(hi, 0.45 * fs))
    x = np.asarray(white.value, dtype=float)

    n_per_bin = max(int(round(rms_stride * fs)), 1)
    n_bins = len(x) // n_per_bin
    if n_bins < 16:
        return pd.DataFrame(columns=["time", "z", "rms"])

    blocks = x[: n_bins * n_per_bin].reshape(n_bins, n_per_bin)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    times = data.t0 + (np.arange(n_bins) + 0.5) * rms_stride

    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med))) * 1.4826
    if mad <= 0:
        return pd.DataFrame(columns=["time", "z", "rms"])
    z = (rms - med) / mad

    distance = max(int(round(cluster_window / rms_stride)), 1)
    peaks, _ = sps.find_peaks(z, height=z_threshold, distance=distance)
    if peaks.size == 0:
        return pd.DataFrame(columns=["time", "z", "rms"])

    keep = (times[peaks] - data.t0 > edge_pad) & (
        data.t0 + data.duration - times[peaks] > edge_pad
    )
    peaks = peaks[keep]

    return pd.DataFrame(
        {"time": times[peaks], "z": z[peaks], "rms": rms[peaks]}
    ).reset_index(drop=True)


def search_segment(
    fetcher: DataFetcher,
    cfg: Config,
    detector: str,
    seg_start: float,
    seg_length: float,
    z_threshold: float = 5.0,
    max_triggers: int = 200,
) -> pd.DataFrame:
    """Generate triggers in one segment and extract their morphology features.

    Returns one row per candidate with the trigger metadata and the full
    feature vector.  Candidates whose Q-scan fails (a data gap inside the
    padding window, most often) are dropped with a debug message rather than
    aborting the segment.
    """
    pad = float(cfg.strain.window_half_width)
    data = fetcher.strain(detector, seg_start - pad, seg_start + seg_length + pad)

    # The clustering window is the same one used to turn the Gravity Spy
    # trigger list into events (sampling.cluster_window), so the O3 and O4a
    # samples count the same kind of thing.  Comparing an event rate against a
    # trigger rate would show a factor of several that has nothing to do with
    # the detectors.
    triggers = find_triggers(
        data,
        band=(10.0, 120.0),
        z_threshold=z_threshold,
        cluster_window=float(cfg.sampling.cluster_window),
        edge_pad=pad,
    )
    if triggers.empty:
        return pd.DataFrame()

    if len(triggers) > max_triggers:
        # Keep the loudest; an hour with >200 triggers in 512 s is saturated
        # and the sample would otherwise be dominated by a single bad night.
        triggers = triggers.nlargest(max_triggers, "z")
        log.debug(
            "%s %.0f: truncated to the %d loudest candidates",
            detector, seg_start, max_triggers,
        )

    # Both arguments must come from the configuration, exactly as stage 02
    # does for O3.  If the search used the function defaults while the O3
    # extraction used the configured values, the two feature sets would be
    # computed under different thresholds and the like-for-like comparison
    # would be measuring the pipeline rather than the detectors.
    threshold = float(cfg.qtransform.energy_threshold)
    islands = int(cfg.qtransform.min_island_tiles)
    rows = []
    for _, trig in triggers.iterrows():
        t0 = float(trig["time"])
        try:
            window = data.crop(t0 - pad, t0 + pad)
            qmap = compute_qmap(window, t0, cfg)
            feats = extract_features(qmap, threshold, islands)
        except Exception as exc:  # noqa: BLE001
            log.debug("feature extraction failed at %s %.3f: %s", detector, t0, exc)
            continue
        rows.append(
            {
                "event_time": t0,
                "detector": detector,
                "trigger_z": float(trig["z"]),
                "segment_start": float(seg_start),
                **feats,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@dataclass
class ClassifierBundle:
    """A trained identifier plus everything needed to interpret its scores."""

    pipeline: Pipeline
    feature_names: list[str]
    threshold: float
    target_fpr: float
    cv_auc: float
    cv_auc_std: float
    cv_efficiency: float
    cv_efficiency_std: float
    n_signal: int
    n_background: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    #: True when the background sample is too small to place the operating
    #: point at ``target_fpr``.  The efficiency should not be quoted if so;
    #: the AUC, being threshold-free, is unaffected.
    efficiency_underresolved: bool = False

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_names).to_numpy(dtype=float)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df) >= self.threshold

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("saved classifier to %s", path)

    @staticmethod
    def load(path: Path) -> "ClassifierBundle":
        import joblib

        return joblib.load(path)

    def summary(self) -> str:
        caveat = (
            "   [under-resolved: too few controls for this operating point]"
            if self.efficiency_underresolved else ""
        )
        return (
            f"identifier: {self.n_signal} signal / {self.n_background} background\n"
            f"  cross-validated ROC AUC : {self.cv_auc:.4f} +/- {self.cv_auc_std:.4f}\n"
            f"  efficiency at FPR={self.target_fpr:.3f} : "
            f"{self.cv_efficiency:.3f} +/- {self.cv_efficiency_std:.3f}{caveat}\n"
            f"  score threshold         : {self.threshold:.4f}"
        )


def _build_pipeline(cfg: Config) -> Pipeline:
    clf = cfg.classifier
    return Pipeline(
        [
            # NaN features are informative (e.g. arch_period is undefined when
            # fewer than two arches are found), so they are imputed to the
            # median and flagged rather than dropped.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=int(clf.n_estimators),
                    max_depth=int(clf.max_depth),
                    learning_rate=float(clf.learning_rate),
                    random_state=int(cfg.sampling.seed),
                ),
            ),
        ]
    )


def train_identifier(
    cfg: Config,
    features: pd.DataFrame,
    label_column: str = "ml_label",
    feature_names: list[str] | None = None,
) -> ClassifierBundle:
    """Train and cross-validate the strain-only scattered-light identifier.

    ``features`` must contain the O3 target-class rows and the control-class
    rows, each with the columns in :data:`gwglitch.constants.FEATURE_NAMES`.

    Cross-validation is stratified K-fold.  Note that it is *not* grouped in
    time: triggers minutes apart are not independent, so the quoted AUC is
    mildly optimistic.  This is stated in the write-up; the mitigation is that
    the sample is drawn stratified across the whole run, so within-fold
    neighbours are rare.
    """
    target = cfg.gravityspy.target_label
    names = list(feature_names or FEATURE_NAMES)

    missing = [c for c in names if c not in features.columns]
    if missing:
        raise KeyError(f"feature table is missing columns: {missing}")

    y = (features[label_column] == target).to_numpy(dtype=int)
    X = features[names].to_numpy(dtype=float)

    if y.sum() < 50 or (1 - y).sum() < 50:
        raise ValueError(
            f"need at least 50 examples of each class; got {y.sum()} signal "
            f"and {(1 - y).sum()} background"
        )

    n_splits = int(cfg.classifier.n_splits)
    target_fpr = float(cfg.classifier.target_fpr)

    # An operating point can only be placed as finely as the background sample
    # resolves.  With n background events the smallest measurable false
    # positive rate is 1/n, and each cross-validation fold sees only n/k of
    # them -- so asking for FPR = 0.01 from 66 controls means the threshold is
    # set by whether zero or one event happens to land above it, and the
    # efficiency then swings wildly between folds (0.24 to 0.76 was measured on
    # exactly that sample).  The AUC is unaffected, being threshold-free.
    n_background = int((1 - y).sum())
    per_fold = n_background / n_splits
    expected_false_positives = target_fpr * per_fold
    if expected_false_positives < 5.0:
        log.warning(
            "target FPR %.3g is finer than the background sample resolves: "
            "%d control events over %d folds is %.0f per fold, so the "
            "operating point is set by only %.1f expected false positives and "
            "the quoted efficiency will be unstable. The AUC is unaffected. "
            "Enlarge the control sample or coarsen classifier.target_fpr "
            "before quoting the efficiency.",
            target_fpr, n_background, n_splits, per_fold,
            expected_false_positives,
        )
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(cfg.sampling.seed)
    )

    aucs, effs, thresholds = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        pipe = _build_pipeline(cfg)
        pipe.fit(X[train_idx], y[train_idx])
        scores = pipe.predict_proba(X[test_idx])[:, 1]

        auc = roc_auc_score(y[test_idx], scores)
        fpr, tpr, thr = roc_curve(y[test_idx], scores)
        k = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
        k = max(k, 0)
        aucs.append(auc)
        effs.append(float(tpr[k]))
        thresholds.append(float(thr[k]))
        log.info(
            "fold %d/%d: AUC = %.4f, efficiency at FPR=%.3f is %.3f (threshold %.3f)",
            fold, n_splits, auc, target_fpr, tpr[k], thr[k],
        )

    final = _build_pipeline(cfg)
    final.fit(X, y)

    model = final.named_steps["model"]
    # SimpleImputer(add_indicator=True) appends one column per feature that had
    # a missing value in training, so importances can be longer than `names`.
    importance = {
        name: float(v)
        for name, v in zip(names, model.feature_importances_[: len(names)])
    }

    bundle = ClassifierBundle(
        pipeline=final,
        feature_names=names,
        threshold=float(np.median(thresholds)),
        target_fpr=target_fpr,
        cv_auc=float(np.mean(aucs)),
        cv_auc_std=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        cv_efficiency=float(np.mean(effs)),
        cv_efficiency_std=float(np.std(effs, ddof=1)) if len(effs) > 1 else 0.0,
        n_signal=int(y.sum()),
        n_background=n_background,
        efficiency_underresolved=bool(expected_false_positives < 5.0),
        feature_importance=dict(
            sorted(importance.items(), key=lambda kv: -kv[1])
        ),
    )
    log.info("\n%s", bundle.summary())
    return bundle


def score_frame(bundle: ClassifierBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Attach ``score`` and ``is_scattering`` columns to a feature table."""
    out = features.copy()
    out["score"] = bundle.predict_proba(out)
    out["is_scattering"] = out["score"] >= bundle.threshold
    return out
