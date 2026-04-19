from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import chi2_contingency, ks_2samp
except ImportError:  # pragma: no cover
    chi2_contingency = None
    ks_2samp = None
    jensenshannon = None


EXCLUDE_COLUMNS = {
    "data_split",
    "churn_status",
    "customerID",  # Identifier column; drift here is not actionable.
}
CATEGORICAL_OVERRIDE_COLUMNS = {"SeniorCitizen"}
NUMERIC_DRIFT_THRESHOLD = 0.10
CATEGORICAL_DRIFT_THRESHOLD = 0.10

# Critical thresholds: crossing these suggests the model needs retraining.
PSI_CRITICAL = 0.25
KS_CRITICAL = 0.20
JS_CRITICAL = 0.20
DRIFT_FRACTION_CRITICAL = 0.30


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [
        col
        for col in df.select_dtypes(include=["number"]).columns.tolist()
        if col not in EXCLUDE_COLUMNS and col not in CATEGORICAL_OVERRIDE_COLUMNS
    ]
    categorical_columns = [
        col
        for col in df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if col not in EXCLUDE_COLUMNS
    ]

    for column in sorted(CATEGORICAL_OVERRIDE_COLUMNS):
        if column in df.columns and column not in EXCLUDE_COLUMNS and column not in categorical_columns:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def ks_distance(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.sort(reference)
    current = np.sort(current)
    combined = np.sort(np.unique(np.concatenate([reference, current])))
    cdf_ref = np.searchsorted(reference, combined, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, combined, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float | None]:
    if ks_2samp is not None:
        result = ks_2samp(reference, current, alternative="two-sided", mode="auto")
        return float(result.statistic), float(result.pvalue)
    return ks_distance(reference, current), None


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    breakpoints = np.linspace(0.0, 1.0, buckets + 1)
    reference_bins = np.quantile(reference, breakpoints)
    reference_counts, _ = np.histogram(reference, bins=reference_bins)
    current_counts, _ = np.histogram(current, bins=reference_bins)
    reference_rates = reference_counts.astype(float) / reference_counts.sum()
    current_rates = current_counts.astype(float) / current_counts.sum()
    reference_rates = np.where(reference_rates == 0, 1e-8, reference_rates)
    current_rates = np.where(current_rates == 0, 1e-8, current_rates)
    return float(np.sum((reference_rates - current_rates) * np.log(reference_rates / current_rates)))


def entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def js_divergence(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.astype(str).fillna("<NA>")
    current = current.astype(str).fillna("<NA>")
    categories = sorted(set(reference.unique()).union(set(current.unique())))
    p = np.array([(reference == cat).sum() for cat in categories], dtype=float)
    q = np.array([(current == cat).sum() for cat in categories], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p /= p.sum()
    q /= q.sum()

    # Prefer scipy implementation when available, but guard against non-finite outputs.
    if jensenshannon is not None:
        js_distance = float(jensenshannon(p, q, base=2.0))
        if np.isfinite(js_distance):
            return js_distance

    # Stable fallback: JS distance = sqrt(0.5 * (KL(p||m) + KL(q||m))).
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()

    m = 0.5 * (p_safe + q_safe)
    kl_pm = np.sum(p_safe * np.log2(p_safe / m))
    kl_qm = np.sum(q_safe * np.log2(q_safe / m))
    js_div = 0.5 * (kl_pm + kl_qm)
    js_div = max(js_div, 0.0)
    return float(np.sqrt(js_div))


def chi2_test(reference: pd.Series, current: pd.Series) -> tuple[float | None, float | None]:
    if chi2_contingency is None:
        return None, None
    reference = reference.astype(str).fillna("<NA>")
    current = current.astype(str).fillna("<NA>")
    categories = sorted(set(reference.unique()).union(set(current.unique())))
    contingency = np.array(
        [[(reference == category).sum(), (current == category).sum()] for category in categories],
        dtype=float,
    )
    chi2_stat, p_value, _, _ = chi2_contingency(contingency, correction=False)
    return float(chi2_stat), float(p_value)


def assess_retraining_need(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Determine if model retraining is recommended based solely on input drift."""
    total = len(metrics)
    if total == 0:
        return {"recommended": False, "confidence": "low", "reasons": []}

    drifted_count = sum(1 for m in metrics if m["alert"])
    drift_fraction = drifted_count / total

    critical_psi = [m["feature"] for m in metrics if m.get("psi") is not None and m["psi"] > PSI_CRITICAL]
    critical_ks = [m["feature"] for m in metrics if m["feature_type"] == "numeric" and m["value"] > KS_CRITICAL]
    critical_js = [m["feature"] for m in metrics if m["feature_type"] == "categorical" and m["value"] > JS_CRITICAL]
    avg_severity = sum(m["severity"] for m in metrics if m["alert"]) / drifted_count if drifted_count > 0 else 0.0

    reasons: list[str] = []
    if critical_psi:
        reasons.append(f"PSI > {PSI_CRITICAL} on: {', '.join(critical_psi)}")
    if critical_ks:
        reasons.append(f"KS > {KS_CRITICAL} on: {', '.join(critical_ks)}")
    if critical_js:
        reasons.append(f"JS > {JS_CRITICAL} on: {', '.join(critical_js)}")
    if drift_fraction > DRIFT_FRACTION_CRITICAL:
        reasons.append(f"{drift_fraction:.0%} of features are drifted (threshold: {DRIFT_FRACTION_CRITICAL:.0%})")

    recommended = len(reasons) > 0

    if len(reasons) >= 3 or (critical_psi and drift_fraction > DRIFT_FRACTION_CRITICAL):
        confidence = "high"
    elif len(reasons) == 2 or avg_severity > 0.6:
        confidence = "medium"
    else:
        confidence = "low" if not recommended else "medium"

    return {
        "recommended": recommended,
        "confidence": confidence,
        "reasons": reasons,
        "drift_fraction": drift_fraction,
        "avg_severity": avg_severity,
    }


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    numeric_columns, categorical_columns = get_feature_columns(reference)
    metrics: list[dict[str, Any]] = []

    for feature in numeric_columns:
        if feature not in current.columns:
            continue
        ref_values = reference[feature].dropna().to_numpy(dtype=float)
        cur_values = current[feature].dropna().to_numpy(dtype=float)
        if len(ref_values) < 2 or len(cur_values) < 2:
            continue
        ks_stat, ks_p = ks_test(ref_values, cur_values)
        psi_value = psi(ref_values, cur_values)
        alert = ks_stat > NUMERIC_DRIFT_THRESHOLD or psi_value > 0.1
        reasons: list[str] = []
        if ks_stat > NUMERIC_DRIFT_THRESHOLD:
            reasons.append(f"KS>{NUMERIC_DRIFT_THRESHOLD}")
        if psi_value > 0.1:
            reasons.append("PSI>0.1")
        severity = (
            0.6 * min(ks_stat, 1.0)
            + 0.4 * min(psi_value / 0.2, 1.0)
        )
        metrics.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "ks": ks_stat,
                "js": None,
                "psi": psi_value,
                "value": ks_stat,
                "p_value": ks_p,
                "severity": min(max(severity, 0.0), 1.0),
                "alert": alert,
                "reason": ", ".join(reasons) if reasons else "ok",
            }
        )

    for feature in categorical_columns:
        if feature not in current.columns:
            continue
        js_score = js_divergence(reference[feature], current[feature])
        _chi2_stat, chi2_p = chi2_test(reference[feature], current[feature])
        alert = js_score > CATEGORICAL_DRIFT_THRESHOLD
        reasons: list[str] = []
        if js_score > CATEGORICAL_DRIFT_THRESHOLD:
            reasons.append(f"JS>{CATEGORICAL_DRIFT_THRESHOLD}")
        severity = min(js_score, 1.0)
        metrics.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "ks": None,
                "js": js_score,
                "psi": None,
                "value": js_score,
                "p_value": chi2_p,
                "severity": min(max(severity, 0.0), 1.0),
                "alert": alert,
                "reason": ", ".join(reasons) if reasons else "ok",
            }
        )

    metrics.sort(key=lambda item: (item["alert"], item["value"]), reverse=True)
    return metrics
