from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mlops_project.api.schema import CustomerInput, PredictionOutput
from src.mlops_project.api.service import artifacts_status
from src.mlops_project.utils.logger import customer_id_exists


METRICS_PATH = ROOT_DIR / "artifacts" / "metrics" / "metrics.json"
API_BASE_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")

YES_NO_OPTIONS = ["Yes", "No"]
MULTIPLE_LINES_OPTIONS = ["Yes", "No", "No phone service"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
INTERNET_ADDON_OPTIONS = ["Yes", "No", "No internet service"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

CSV_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_metrics() -> dict[str, float | int | str] | None:
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_sample_batch_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            },
            {
                "customerID": "5575-GNVDE",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 34,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 56.95,
                "TotalCharges": 1889.5,
            },
        ],
        columns=CSV_COLUMNS,
    )


def render_sidebar() -> tuple[dict, dict | None]:
    status = artifacts_status()
    metrics = load_metrics()

    st.sidebar.header("Model Status")
    st.sidebar.write(f"Model: `{status.get('model_name') or 'N/A'}`")
    st.sidebar.write(f"Threshold: `{status.get('threshold')}`")
    st.sidebar.write(
        "Artifacts loaded: "
        f"`model={status.get('model_loaded')}` | "
        f"`preprocessor={status.get('preprocessor_loaded')}`"
    )

    if metrics:
        st.sidebar.header("Offline Metrics")
        st.sidebar.metric("PR AUC", f"{metrics.get('pr_auc', 0):.3f}")
        st.sidebar.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.sidebar.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")

    return status, metrics


def _sync_total_charges_with_monthly() -> None:
    if not st.session_state.get("single_total_charges_touched", False):
        st.session_state["single_total_charges"] = st.session_state["single_monthly_charges"]


def _mark_total_charges_overridden() -> None:
    st.session_state["single_total_charges_touched"] = True


def build_customer_payload() -> dict[str, object]:
    if "single_monthly_charges" not in st.session_state:
        st.session_state["single_monthly_charges"] = 70.35
    if "single_total_charges" not in st.session_state:
        st.session_state["single_total_charges"] = st.session_state["single_monthly_charges"]
    if "single_total_charges_touched" not in st.session_state:
        st.session_state["single_total_charges_touched"] = False

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.text_input("Customer ID", value="7590-VHVEG")
        if not str(customer_id).strip():
            st.warning("Customer ID cannot be empty.")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], index=0)
        partner = st.selectbox("Partner", YES_NO_OPTIONS)
        dependents = st.selectbox("Dependents", YES_NO_OPTIONS, index=1)
        tenure = st.number_input("Tenure (months)", min_value=0, value=12, step=1)
        phone_service = st.selectbox("Phone Service", YES_NO_OPTIONS)
        multiple_lines = st.selectbox("Multiple Lines", MULTIPLE_LINES_OPTIONS)
        internet_service = st.selectbox("Internet Service", INTERNET_SERVICE_OPTIONS)
        contract = st.selectbox("Contract", CONTRACT_OPTIONS)

    with col2:
        online_security = st.selectbox("Online Security", INTERNET_ADDON_OPTIONS)
        online_backup = st.selectbox("Online Backup", INTERNET_ADDON_OPTIONS)
        device_protection = st.selectbox("Device Protection", INTERNET_ADDON_OPTIONS)
        tech_support = st.selectbox("Tech Support", INTERNET_ADDON_OPTIONS)
        streaming_tv = st.selectbox("Streaming TV", INTERNET_ADDON_OPTIONS)
        streaming_movies = st.selectbox("Streaming Movies", INTERNET_ADDON_OPTIONS)
        paperless = st.selectbox("Paperless Billing", YES_NO_OPTIONS)
        payment_method = st.selectbox("Payment Method", PAYMENT_METHOD_OPTIONS)
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            step=0.01,
            key="single_monthly_charges",
            on_change=_sync_total_charges_with_monthly,
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            step=0.01,
            key="single_total_charges",
            on_change=_mark_total_charges_overridden,
        )
        if float(total_charges) < float(monthly_charges):
            st.warning("Total Charges must be greater than or equal to Monthly Charges.")

    return {
        "customerID": str(customer_id).strip(),
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }


def api_predict(customer: CustomerInput) -> PredictionOutput:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{API_BASE_URL}/predict",
            json=customer.model_dump(),
        )
    response.raise_for_status()
    return PredictionOutput.model_validate(response.json())


def api_batch_predict(rows: list[CustomerInput]) -> list[PredictionOutput]:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{API_BASE_URL}/batch-predict",
            json=[row.model_dump() for row in rows],
        )
    response.raise_for_status()
    return [PredictionOutput.model_validate(item) for item in response.json()]


def _format_row_numbers(indexes: pd.Index) -> str:
    # Show user-facing row numbers (1-based within uploaded data table).
    return ", ".join(str(int(i) + 1) for i in indexes.tolist())


def validate_batch_rules(batch_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    customer_ids = batch_df["customerID"].fillna("").astype(str).str.strip()
    empty_id_mask = customer_ids.eq("")
    if empty_id_mask.any():
        issues.append(
            "Customer ID cannot be empty. Rows: "
            + _format_row_numbers(batch_df.index[empty_id_mask])
        )

    non_empty_ids = customer_ids[~empty_id_mask]
    duplicate_mask = non_empty_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_rows = non_empty_ids.index[duplicate_mask]
        issues.append(
            "Batch contains duplicate customerID values. Rows: "
            + _format_row_numbers(duplicate_rows)
        )

    existing_rows = [
        idx for idx, cid in non_empty_ids.items() if customer_id_exists(str(cid))
    ]
    if existing_rows:
        issues.append(
            "Some customerID values already exist in inference log. Rows: "
            + _format_row_numbers(pd.Index(existing_rows))
        )

    monthly = pd.to_numeric(batch_df["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(batch_df["TotalCharges"], errors="coerce")
    total_lt_monthly_mask = (total < monthly).fillna(False)
    if total_lt_monthly_mask.any():
        issues.append(
            "Total Charges must be greater than or equal to Monthly Charges. Rows: "
            + _format_row_numbers(batch_df.index[total_lt_monthly_mask])
        )

    return issues


def render_single_prediction() -> None:
    st.subheader("Single Prediction")
    st.caption("Enter customer information to predict churn risk.")

    payload = build_customer_payload()
    customer_id = str(payload.get("customerID") or "").strip()
    is_customer_id_empty = len(customer_id) == 0
    is_customer_id_taken = (not is_customer_id_empty) and customer_id_exists(customer_id)
    total_lt_monthly = float(payload["TotalCharges"]) < float(payload["MonthlyCharges"])

    if is_customer_id_taken:
        st.error(f"Customer ID '{customer_id}' already exists.")

    submitted = st.button(
        "Predict",
        use_container_width=True,
        disabled=is_customer_id_empty or is_customer_id_taken or total_lt_monthly,
    )

    if not submitted:
        return

    try:
        customer = CustomerInput(**payload)
        result = api_predict(customer)

        churn_prob = result.churn_probability
        st.success("Prediction completed.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{churn_prob:.2%}")
        col2.metric("Prediction", "Churn" if result.prediction == 1 else "Stay")
        col3.metric("Threshold", f"{result.threshold:.2f}")

        st.progress(min(max(float(churn_prob), 0.0), 1.0))
        st.json(
            {
                "input": customer.model_dump(),
                "output": result.model_dump(),
            }
        )
    except ValidationError as exc:
        st.error("Input validation failed.")
        st.code(str(exc))
    except httpx.HTTPStatusError as exc:
        st.error(f"API returned an error: {exc.response.status_code}")
        st.code(exc.response.text)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach API at {API_BASE_URL}: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def render_batch_prediction() -> None:
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV in the Telco schema or use the sample file for batch prediction.")

    sample_df = build_sample_batch_df()
    st.download_button(
        "Download Sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="telco_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.dataframe(sample_df, use_container_width=True)
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview")
        st.dataframe(batch_df.head(10), use_container_width=True)

        missing_columns = [col for col in CSV_COLUMNS if col not in batch_df.columns]
        batch_issues: list[str] = []
        if missing_columns:
            st.error(f"Missing required columns in the CSV: {missing_columns}")
        else:
            batch_issues = validate_batch_rules(batch_df)

        if batch_issues:
            for issue in batch_issues:
                st.error(issue)

        run_batch = st.button(
            "Run Batch Prediction",
            use_container_width=True,
            disabled=bool(missing_columns or batch_issues),
        )
        if not run_batch:
            return

        rows = [CustomerInput(**record) for record in batch_df.to_dict(orient="records")]
        results = api_batch_predict(rows)

        result_df = batch_df.copy()
        result_df["churn_probability"] = [item.churn_probability for item in results]
        result_df["prediction"] = [
            "Churn" if item.prediction == 1 else "Stay" for item in results
        ]
        result_df["threshold"] = [item.threshold for item in results]
        result_df["model_name"] = [item.model_name for item in results]

        churn_rate = (result_df["prediction"] == "Churn").mean()
        avg_probability = result_df["churn_probability"].mean()

        col1, col2 = st.columns(2)
        col1.metric("Rows", len(result_df))
        col2.metric("Predicted Churn Rate", f"{churn_rate:.2%}")
        st.metric("Average Churn Probability", f"{avg_probability:.2%}")

        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download Predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
    except ValidationError as exc:
        st.error("CSV validation failed.")
        st.code(str(exc))
    except httpx.HTTPStatusError as exc:
        st.error(f"API returned an error: {exc.response.status_code}")
        st.code(exc.response.text)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach API at {API_BASE_URL}: {exc}")
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")


def render_overview(status: dict, metrics: dict | None) -> None:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Loaded", "Yes" if status.get("model_loaded") else "No")
    col2.metric("Preprocessor", "Yes" if status.get("preprocessor_loaded") else "No")
    col3.metric("Model Name", status.get("model_name") or "N/A")
    col4.metric(
        "Threshold",
        "N/A" if status.get("threshold") is None else f"{status.get('threshold'):.2f}",
    )

    if metrics:
        st.write("Latest offline evaluation metrics")
        col1, col2, col3 = st.columns(3)

        metrics_df = pd.DataFrame(
            [
                {"metric": "pr_auc", "value": metrics.get("pr_auc")},
                {"metric": "roc_auc", "value": metrics.get("roc_auc")},
                {"metric": "f1_score", "value": metrics.get("f1_score")},
                {"metric": "accuracy", "value": metrics.get("accuracy")},
                {"metric": "precision", "value": metrics.get("precision")},
                {"metric": "recall", "value": metrics.get("recall")},
                {"metric": "log_loss", "value": metrics.get("log_loss")},
            ]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("artifacts/metrics/metrics.json")


def main() -> None:
    st.set_page_config(
        page_title="Churn Prediction Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("Churn Prediction Dashboard")
    st.caption("Streamlit UI for churn prediction, artifact monitoring, and batch scoring.")

    status, metrics = render_sidebar()

    overview_tab, single_tab, batch_tab = st.tabs(
        ["Overview", "Single Predict", "Batch Predict"]
    )

    with overview_tab:
        render_overview(status, metrics)

    with single_tab:
        render_single_prediction()

    with batch_tab:
        render_batch_prediction()


if __name__ == "__main__":
    main()
