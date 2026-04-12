"""
Data Validation Script
Validates data before training
"""
import pandas as pd
import sys
from pathlib import Path


def validate_data(data_path: str) -> bool:
    """
    Validate data file exists and has expected structure
    
    Args:
        data_path: Path to data file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"❌ ERROR: Data file not found at {data_path}")
            return False
        
        # Load and validate
        df = pd.read_csv(data_path)
        
        # Check minimum requirements
        if len(df) == 0:
            print("❌ ERROR: Data file is empty")
            return False
        
        if len(df.columns) < 2:
            print("❌ ERROR: Data must have at least 2 columns")
            return False
        
        # Print validation summary
        print(f"✅ Data validation passed:")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Columns: {', '.join(df.columns.tolist())}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Data validation failed: {e}")
        return False


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/netflix_large.csv"
    
    is_valid = validate_data(data_path)
    sys.exit(0 if is_valid else 1)
