import subprocess
import pandas as pd
from datetime import datetime
import os

INFERENCE_LOG_PATH = "monitoring/inference/inference_log.csv"
DRIFT_REPORT_TRIGGER_EVERY = 100  # run drift report every N new predictions


def log_inference(input_data: dict, prediction):
    log_entry = {
        **input_data,
        "prediction": prediction,
        "timestamp": datetime.now()
    }

    df = pd.DataFrame([log_entry])

    if os.path.exists(INFERENCE_LOG_PATH):
        df.to_csv(INFERENCE_LOG_PATH, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname(INFERENCE_LOG_PATH), exist_ok=True)
        df.to_csv(INFERENCE_LOG_PATH, index=False)

    _maybe_trigger_drift_report()


def _maybe_trigger_drift_report():
    if not os.path.exists(INFERENCE_LOG_PATH):
        return
    try:
        row_count = sum(1 for _ in open(INFERENCE_LOG_PATH)) - 1  # subtract header
    except OSError:
        return
    if row_count > 0 and row_count % DRIFT_REPORT_TRIGGER_EVERY == 0:
        subprocess.Popen(
            [
                "python", "src/mlops_project/monitoring/create_drift_report.py",
                "--current", INFERENCE_LOG_PATH,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
