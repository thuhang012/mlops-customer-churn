import argparse
import pandas as pd
import joblib
import os
from sklearn.dummy import DummyClassifier

def main(data_path, model_path):
    # Enforce input validation
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Mock data file not found: {data_path}")

    # Load mock data
    df = pd.read_csv(data_path)
    
    # Assume the last column is the target, remaining are features
    # This establishes the input schema vector length
    y = df["churn_status"]
    X = df.drop(columns=["churn_status"])

    # Initialize Dummy Model with a static return value
    # Strategy 'constant' bypasses ML logic and guarantees predictable output for CI/CD
    model = DummyClassifier(strategy="constant", constant=1)
    
    # Fit the mock data to register the I/O schema internally
    model.fit(X, y)

    # Ensure output directory exists before dumping
    output_dir = os.path.dirname(model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Export the mock object
    joblib.dump(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mock Model for Pipeline Validation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dummy CSV data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save dummy_model.pkl")
    
    args = parser.parse_args()
    main(args.data_path, args.model_path)
    