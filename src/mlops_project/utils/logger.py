import pandas as pd
from datetime import datetime
import os

def log_inference(input_data: dict, prediction):
    log_entry = {
        **input_data,
        "prediction": prediction,
        "timestamp": datetime.now()
    }

    file_path = "src/mlops_project/data/inference_log.csv"

    df = pd.DataFrame([log_entry])

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)
