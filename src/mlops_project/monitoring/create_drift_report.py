from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, ks_2samp
    from scipy.spatial.distance import jensenshannon
except ImportError:  # pragma: no cover
    chi2_contingency = None
    ks_2samp = None
    jensenshannon = None


REFERENCE_DATA_PATH = Path("data/processed/cleaned_data.csv")
CURRENT_DATA_PATH = Path("data/processed/inference_drift_data.csv")
OUTPUT_REPORT_PATH = Path("reports/data_drift_report.html")
EXCLUDE_COLUMNS = {"data_split", "churn_status"}
NUMERIC_DRIFT_THRESHOLD = 0.10
CATEGORICAL_DRIFT_THRESHOLD = 0.10


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [
        col
        for col in df.select_dtypes(include=["number"]).columns.tolist()
        if col not in EXCLUDE_COLUMNS
    ]
    categorical_columns = [
        col
        for col in df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if col not in EXCLUDE_COLUMNS
    ]
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
    if jensenshannon is not None:
        return float(jensenshannon(p, q, base=2.0))
    m = 0.5 * (p + q)
    return float(np.sqrt(0.5 * (entropy(p) + entropy(q) - 2 * entropy(m))))


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


def build_numeric_plot(reference: pd.Series, current: pd.Series, name: str, score: float) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reference.dropna(), bins=20, alpha=0.5, label="reference", color="#1f77b4", density=True)
    ax.hist(current.dropna(), bins=20, alpha=0.5, label="current", color="#d62728", density=True)
    ax.set_title(f"{name} (KS distance = {score:.3f})")
    ax.legend()
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    return plot_to_base64(fig)


def build_categorical_plot(reference: pd.Series, current: pd.Series, name: str, score: float) -> str:
    categories = sorted(set(reference.dropna().astype(str).unique()).union(current.dropna().astype(str).unique()))
    reference_counts = [(reference.astype(str) == cat).sum() for cat in categories]
    current_counts = [(current.astype(str) == cat).sum() for cat in categories]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    ax.bar(x - width / 2, reference_counts, width, label="reference", color="#1f77b4")
    ax.bar(x + width / 2, current_counts, width, label="current", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_title(f"{name} (Jensen-Shannon = {score:.3f})")
    ax.set_ylabel("Count")
    ax.legend()
    return plot_to_base64(fig)


def plot_to_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_severity_plot(metrics: list[dict[str, Any]]) -> str:
    top_metrics = sorted(metrics, key=lambda item: item["severity"], reverse=True)[:10]
    labels = [metric["feature"] for metric in top_metrics]
    scores = [metric["severity"] for metric in top_metrics]
    colors = ["#c0392b" if metric["alert"] else "#7f8c8d" for metric in top_metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, scores, color=colors)
    ax.set_title("Top 10 Feature Drift Severity")
    ax.set_ylabel("Severity score")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return plot_to_base64(fig)


def create_html_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    plots_html = []
    for metric in metrics[:10]:
        if metric["feature_type"] == "numeric":
            img = build_numeric_plot(
                reference[metric["feature"]], current[metric["feature"]], metric["feature"], metric["value"]
            )
        else:
            img = build_categorical_plot(
                reference[metric["feature"]], current[metric["feature"]], metric["feature"], metric["value"]
            )
        plots_html.append(
            f"<div class='chart-card'><h3>{metric['feature']} ({metric['feature_type']})</h3><p><strong>Reason:</strong> {metric['reason']}</p><img src='data:image/png;base64,{img}' /></div>"
        )

    rows_html = "".join(
        """
        <tr>
            <td>{feature}</td>
            <td>{feature_type}</td>
            <td>{severity:.3f}</td>
            <td>{value:.4f}</td>
            <td>{p_text}</td>
            <td>{psi_text}</td>
            <td>{alert}</td>
        </tr>
        """.format(
            feature=metric["feature"],
            feature_type=metric["feature_type"],
            severity=metric["severity"],
            value=metric["value"],
            p_text=f"{metric['p_value']:.3g}" if metric.get("p_value") is not None else "n/a",
            psi_text=f"{metric['psi']:.3f}" if metric.get("psi") is not None else "-",
            alert="<strong style='color:red'>drift</strong>" if metric["alert"] else "ok",
        )
        for metric in metrics
    )

    drift_count = sum(1 for metric in metrics if metric["alert"])
    numeric_drift = sum(1 for metric in metrics if metric["alert"] and metric["feature_type"] == "numeric")
    categorical_drift = sum(1 for metric in metrics if metric["alert"] and metric["feature_type"] == "categorical")
    fallback_note = "" if all(metric.get("p_value") is not None for metric in metrics) else "<li>Some statistical tests used fallback approximations because SciPy is not installed.</li>"

    severity_plot = build_severity_plot(metrics)

    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Data Drift Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f2f2f2; }}
            .section {{ margin-bottom: 40px; }}
            .summary-card {{ background: #f8f9fa; border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 24px; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
            .chart-card {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; background: white; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
            .chart-card h3 {{ margin: 0 0 8px; font-size: 1rem; color: #2c3e50; }}
            .chart-card p {{ margin: 0 0 12px; font-size: 0.95rem; color: #333; }}
            img {{ width: 100%; max-width: 900px; border: 1px solid #ddd; padding: 8px; background: white; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.9rem; }}
            .badge-ok {{ background: #27ae60; color: white; }}
            .badge-drift {{ background: #c0392b; color: white; }}
        </style>
    </head>
    <body>
        <h1>Data Drift Report</h1>
        <div class="summary-card">
            <p><strong>Reference rows:</strong> {len(reference)}</p>
            <p><strong>Current rows:</strong> {len(current)}</p>
            <p><strong>Features evaluated:</strong> {len(metrics)}</p>
            <p><strong>Drift alerts:</strong> <span class="badge badge-drift">{drift_count}</span></p>
            <p><strong>Numeric drifts:</strong> <span class="badge badge-drift">{numeric_drift}</span></p>
            <p><strong>Categorical drifts:</strong> <span class="badge badge-drift">{categorical_drift}</span></p>
            <ul>{fallback_note}</ul>
        </div>
        <div class="section">
            <h2>Severity Score Chart</h2>
            <img src='data:image/png;base64,{severity_plot}' />
        </div>
        <div class="section">
            <h2>Feature Drift Summary</h2>
            <table>
                <thead>
                    <tr><th>Feature</th><th>Type</th><th>Severity</th><th>Drift score</th><th>P-value</th><th>PSI</th><th>Status</th></tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        <div class="section">
            <h2>Top Drift Visualizations</h2>
            <div class="chart-grid">
                {''.join(plots_html)}
            </div>
        </div>
    </body>
    </html>
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")
    print(f"Saved data drift report to: {output_path}")


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
        alert = (ks_p is not None and ks_p < 0.05) or ks_stat > NUMERIC_DRIFT_THRESHOLD or psi_value > 0.1
        reasons: list[str] = []
        if ks_p is not None and ks_p < 0.05:
            reasons.append("KS p<0.05")
        if ks_stat > NUMERIC_DRIFT_THRESHOLD:
            reasons.append(f"KS>{NUMERIC_DRIFT_THRESHOLD}")
        if psi_value > 0.1:
            reasons.append("PSI>0.1")
        severity = (
            0.5 * min(ks_stat, 1.0)
            + 0.35 * min(psi_value / 0.2, 1.0)
            + 0.15 * (1.0 if ks_p is not None and ks_p < 0.05 else 0.0)
        )
        metrics.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "value": ks_stat,
                "p_value": ks_p,
                "psi": psi_value,
                "severity": min(max(severity, 0.0), 1.0),
                "alert": alert,
                "reason": ", ".join(reasons) if reasons else "ok",
            }
        )

    for feature in categorical_columns:
        if feature not in current.columns:
            continue
        js_score = js_divergence(reference[feature], current[feature])
        chi2_stat, chi2_p = chi2_test(reference[feature], current[feature])
        alert = (chi2_p is not None and chi2_p < 0.05) or js_score > CATEGORICAL_DRIFT_THRESHOLD
        reasons: list[str] = []
        if chi2_p is not None and chi2_p < 0.05:
            reasons.append("chi2 p<0.05")
        if js_score > CATEGORICAL_DRIFT_THRESHOLD:
            reasons.append(f"JS>{CATEGORICAL_DRIFT_THRESHOLD}")
        severity = 0.6 * min(js_score, 1.0) + 0.4 * (1.0 if chi2_p is not None and chi2_p < 0.05 else 0.0)
        metrics.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "value": js_score,
                "p_value": chi2_p,
                "psi": None,
                "severity": min(max(severity, 0.0), 1.0),
                "alert": alert,
                "reason": ", ".join(reasons) if reasons else "ok",
            }
        )

    metrics.sort(key=lambda item: (item["alert"], item["value"]), reverse=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a data drift HTML report.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_DATA_PATH,
        help="Reference (train) dataset CSV path.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=CURRENT_DATA_PATH,
        help="Current (inference) dataset CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_REPORT_PATH,
        help="Output HTML report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_df = load_dataframe(args.reference)
    current_df = load_dataframe(args.current)
    metrics = evaluate_drift(reference_df, current_df)
    create_html_report(reference_df, current_df, metrics, args.output)


if __name__ == "__main__":
    main()
