"""
Data validation script.
Validates that a real dataset exists and has minimal structure before training.
"""

from pathlib import Path
import os
import sys

import pandas as pd


def _write_output(passed: bool) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"passed={str(passed).lower()}\n")


def validate_data(data_path: str) -> bool:
    """Return True only when the real input dataset passes basic checks."""
    try:
        path = Path(data_path)
        if not path.exists():
            print(f"ERROR: Data file not found at {path}")
            _write_output(False)
            return False

        df = pd.read_csv(path)
        if df.empty:
            print("ERROR: Data file is empty")
            _write_output(False)
            return False

        if len(df.columns) < 2:
            print("ERROR: Data must have at least 2 columns")
            _write_output(False)
            return False

        print("Data validation passed")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        _write_output(True)
        return True
    except Exception as exc:
        print(f"ERROR: Data validation failed: {exc}")
        _write_output(False)
        return False


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/netflix_large.csv"
    ok = validate_data(target_path)
    sys.exit(0 if ok else 1)
