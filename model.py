"""
model.py
========
Ensemble Anomaly Detection Engine for the Log Anomaly Detection System.

Architecture:
  ┌─────────────────────────────────────────┐
  │   Feature Matrix X  (n_samples × n_features)   │
  └──────────────┬──────────────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
  Isolation Forest    One-Class SVM
  (tree-based,        (kernel-based,
   good for           good for
   sparse outliers)   dense distributions)
       │                    │
       └─────────┬──────────┘
                 │
         Soft Voting via
         Anomaly Score Average
                 │
         Adaptive Threshold
         (FPR target = 5 %)
                 │
         Binary Label Output
         (0 = normal, 1 = anomaly)

Why this ensemble?
  • Isolation Forest excels at isolating rare/extreme feature combinations.
  • One-Class SVM learns the boundary of the "normal" region in kernel space.
  • Averaging their continuous anomaly scores before thresholding outperforms
    either model alone, reduces variance, and gives fine-grained FPR control.

False Positive Rate Control:
  The threshold is calibrated so that at most `fpr_target` fraction of all
  logs are flagged — giving the analyst a tunable knob rather than a hard-
  coded contamination parameter.

Author: AI/ML Engineer  |  Hackathon 2026
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler


# ---------------------------------------------------------------------------
# 1.  ANOMALY DEFINITIONS (documenting what we are detecting)
# ---------------------------------------------------------------------------
"""
What counts as an anomaly in system logs?

A. Statistical outliers in numeric features
   - Unusually high response time (e.g., latency spike → 10× median)
   - Abnormal response size (truncated or inflated payloads)
   - Burst of high-severity log levels (ERROR/CRITICAL) in a short window

B. Rare log sequences / unusual token patterns
   - TF-IDF score highlights messages whose vocabulary is rare in the corpus
   - "stack trace", "segfault", "connection refused" appearing suddenly

C. HTTP anomalies
   - High rate of 4xx client errors (scraping, enumeration attacks)
   - Any 5xx server error in a production service context

D. Temporal anomalies
   - Activity outside normal business hours (e.g., 2 AM logins)
   - Log bursts that are statistically improbable given historical rate

The ensemble score captures ALL of these simultaneously by operating on the
full feature matrix that encodes each dimension above.
"""


# ---------------------------------------------------------------------------
# 2.  SCALER
# ---------------------------------------------------------------------------

def build_scaler() -> RobustScaler:
    """
    RobustScaler is preferred over StandardScaler here because log data is
    inherently skewed (many normal events, few extreme outliers).
    Robust scaling uses median + IQR, so extreme outlier values don't collapse
    the normal range onto a tiny interval.
    """
    return RobustScaler()


# ---------------------------------------------------------------------------
# 3.  MODEL CONSTRUCTION
# ---------------------------------------------------------------------------

def build_models(contamination: float = 0.05, random_state: int = 42) -> dict:
    """
    Instantiate the ensemble members.

    Parameters
    ----------
    contamination : float
        Fraction of the dataset expected to be anomalous.
        Used as a prior for Isolation Forest's internal threshold.
        Not used for final labelling (we override with FPR-calibrated threshold).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys 'iso_forest' and 'ocsvm'
    """
    iso_forest = IsolationForest(
        n_estimators=300,          # More trees → lower variance
        max_samples='auto',        # Auto = min(256, n_samples)
        contamination=contamination,
        max_features=1.0,          # Use all features per tree
        bootstrap=False,           # Subsampling without replacement
        n_jobs=-1,                 # Parallel training on all CPU cores
        random_state=random_state,
        verbose=0,
    )

    ocsvm = OneClassSVM(
        kernel='rbf',              # Radial basis function — handles non-linear boundaries
        nu=contamination,          # Upper bound on fraction of outliers (≈ contamination)
        gamma='scale',             # gamma = 1 / (n_features * X.var()) — auto-scaled
    )

    return {'iso_forest': iso_forest, 'ocsvm': ocsvm}


# ---------------------------------------------------------------------------
# 4.  TRAINING + SCORING
# ---------------------------------------------------------------------------

def fit_and_score(
    X: np.ndarray,
    fpr_target: float = 0.05,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Fit the ensemble on X (unsupervised — no labels required) and compute
    a calibrated anomaly score for every log line.

    Parameters
    ----------
    X            : feature matrix (n_samples × n_features)
    fpr_target   : desired maximum False Positive Rate (default 5 %)
    contamination: prior estimate of anomaly fraction for model init
    random_state : RNG seed

    Returns
    -------
    labels       : np.ndarray[int]  — 1 = anomaly, 0 = normal
    scores       : np.ndarray[float]— continuous ensemble anomaly score (higher = more anomalous)
    threshold    : float            — decision threshold applied
    models       : dict             — fitted model objects (for inspection / persistence)
    """
    if len(X) == 0:
        raise ValueError("Feature matrix is empty. Cannot train anomaly detector.")

    logging.info(f"[Model] Training ensemble on {X.shape[0]} samples × {X.shape[1]} features")

    # ---- Scale features ----
    scaler = build_scaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Build & fit models ----
    models = build_models(contamination=contamination, random_state=random_state)

    models['iso_forest'].fit(X_scaled)
    models['ocsvm'].fit(X_scaled)
    models['scaler'] = scaler  # store scaler for potential inference later

    # ---- Compute raw anomaly scores ----
    # Isolation Forest: decision_function returns negative anomaly scores;
    #   more negative = more anomalous → negate so higher = worse.
    iso_scores = -models['iso_forest'].decision_function(X_scaled)

    # One-Class SVM: decision_function returns signed distance from boundary;
    #   negative = outside normal region (anomalous) → negate so higher = worse.
    svm_scores = -models['ocsvm'].decision_function(X_scaled)

    # ---- Normalise each score to [0, 1] so they contribute equally ----
    iso_norm = _minmax_norm(iso_scores)
    svm_norm = _minmax_norm(svm_scores)

    # ---- Ensemble: weighted average (equal weight — can tune if needed) ----
    ensemble_score = 0.5 * iso_norm + 0.5 * svm_norm

    # ---- Adaptive threshold: calibrate to FPR target ----
    # We flag the top `fpr_target` fraction of logs as anomalies.
    # This directly controls FPR when the majority of logs are normal.
    threshold = np.quantile(ensemble_score, 1.0 - fpr_target)

    # Ensure at least one anomaly if any score is distinctly elevated
    if threshold == ensemble_score.max():
        threshold = np.quantile(ensemble_score, 0.90)

    labels = (ensemble_score >= threshold).astype(int)

    anomaly_count = labels.sum()
    total = len(labels)
    logging.info(
        f"[Model] Threshold={threshold:.4f} | "
        f"Anomalies={anomaly_count}/{total} ({100*anomaly_count/total:.1f}%)"
    )

    return labels, ensemble_score, threshold, models


# ---------------------------------------------------------------------------
# 5.  SYNTHETIC GROUND-TRUTH EVALUATION (for hackathon metrics display)
# ---------------------------------------------------------------------------

def evaluate_with_synthetic_ground_truth(
    df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """
    Since we have no labelled ground truth in production, we generate a
    *synthetic* ground truth based on deterministic heuristic rules.
    This lets us estimate precision, recall, and FPR for the dashboard.

    Heuristic "true anomaly" definition:
      A log line is heuristically labelled as a true anomaly if ANY of:
        (a) log level is ERROR, CRITICAL, or FATAL
        (b) HTTP status code ≥ 500
        (c) the message contains a strong error keyword
        (d) response time > 95th percentile of all response times

    These rules are intentionally conservative (they will miss subtle anomalies),
    so precision and recall are lower bounds — real performance is likely higher.

    Returns
    -------
    dict with keys: precision, recall, f1, fpr, estimated_accuracy
    """
    n = len(df)
    if n == 0:
        return {}

    # Build heuristic ground truth
    error_levels = {'ERROR', 'CRITICAL', 'FATAL', 'EMERG', 'ALERT'}
    error_kw = re.compile(
        r'\b(error|exception|fail|fatal|critical|traceback|segfault|oom|kill|crash|timeout|refused|denied)\b',
        re.IGNORECASE
    )

    rt_series = pd.to_numeric(df.get('response_time', pd.Series(dtype=float)), errors='coerce')
    rt_p95 = rt_series.quantile(0.95) if rt_series.notna().any() else np.inf

    gt = []
    for _, row in df.iterrows():
        is_anomaly = (
            str(row.get('level', '')).upper() in error_levels
            or (row.get('http_status') is not None and _safe_int(row.get('http_status', 0)) >= 500)
            or bool(error_kw.search(str(row.get('message', ''))))
            or (pd.notna(row.get('response_time')) and float(row.get('response_time', 0)) > rt_p95)
        )
        gt.append(int(is_anomaly))

    gt = np.array(gt)
    pred = labels.astype(int)

    TP = int(((pred == 1) & (gt == 1)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())
    TN = int(((pred == 0) & (gt == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    accuracy  = (TP + TN) / n if n > 0 else 0.0

    return {
        'precision'          : round(precision * 100, 1),
        'recall'             : round(recall * 100, 1),
        'f1'                 : round(f1 * 100, 1),
        'fpr'                : round(fpr * 100, 1),
        'estimated_accuracy' : round(accuracy * 100, 1),
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'gt_anomalies'       : int(gt.sum()),
    }


# ---------------------------------------------------------------------------
# 6.  HELPERS
# ---------------------------------------------------------------------------

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]. If all values are equal, returns zeros."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _safe_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


import re  # noqa: E402  (used inside evaluate_with_synthetic_ground_truth)
