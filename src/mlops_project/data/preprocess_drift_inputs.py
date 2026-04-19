from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.mlops_project.data.load_data import load_raw_data
from src.mlops_project.data.validate_data import (
    clean_drift_current_dataframe,
    clean_drift_reference_dataframe,
)


DEFAULT_REFERENCE_INPUT = Path("data/raw/telcom_churn.csv")
DEFAULT_CURRENT_INPUT = Path("data/raw/inference_log_raw.csv")
DEFAULT_REFERENCE_OUTPUT = Path("data/processed/drift_reference_clean.csv")
DEFAULT_CURRENT_OUTPUT = Path("data/processed/inference_log_clean.csv")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def run(
    reference_input: Path,
    current_input: Path,
    reference_output: Path,
    current_output: Path,
) -> None:
    # Reuse canonical Telco raw loader for reference data.
    reference_raw_df = load_raw_data(reference_input)
    current_raw_df = _load_csv(current_input)

    reference_clean_df, reference_report = clean_drift_reference_dataframe(reference_raw_df)
    current_clean_df, current_report = clean_drift_current_dataframe(current_raw_df)

    reference_output.parent.mkdir(parents=True, exist_ok=True)
    current_output.parent.mkdir(parents=True, exist_ok=True)

    reference_clean_df.to_csv(reference_output, index=False)
    current_clean_df.to_csv(current_output, index=False)

    print(f"Saved cleaned drift reference CSV to: {reference_output}")
    print(f"Saved cleaned drift current CSV to: {current_output}")
    print(f"Reference shape: {reference_raw_df.shape} -> {reference_clean_df.shape}")
    print(f"Current shape: {current_raw_df.shape} -> {current_clean_df.shape}")
    print(f"Reference duplicates removed: {reference_report.get('duplicates_removed', 0)}")
    print(f"Current duplicates removed: {current_report.get('duplicates_removed', 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare drift reference/current CSVs using only raw data cleaning."
    )
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=DEFAULT_REFERENCE_INPUT,
        help="Raw reference CSV path.",
    )
    parser.add_argument(
        "--current-input",
        type=Path,
        default=DEFAULT_CURRENT_INPUT,
        help="Raw current/inference CSV path.",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=DEFAULT_REFERENCE_OUTPUT,
        help="Output path for cleaned reference CSV.",
    )
    parser.add_argument(
        "--current-output",
        type=Path,
        default=DEFAULT_CURRENT_OUTPUT,
        help="Output path for cleaned current CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        reference_input=args.reference_input,
        current_input=args.current_input,
        reference_output=args.reference_output,
        current_output=args.current_output,
    )


if __name__ == "__main__":
    main()
