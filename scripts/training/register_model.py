"""
Model Registration Script
Registers trained model as new baseline
"""
import json
import sys
from pathlib import Path


def register_model(
    metrics_path: str,
    baseline_output_path: str = "artifacts/baseline/metrics.json"
) -> bool:
    """
    Register model by saving current metrics as baseline
    
    Args:
        metrics_path: Path to current metrics.json
        baseline_output_path: Path to save as baseline
    
    Returns:
        True if successful
    """
    try:
        metrics_path = Path(metrics_path)
        baseline_output_path = Path(baseline_output_path)
        
        # Load current metrics
        if not metrics_path.exists():
            print(f"❌ ERROR: Metrics file not found: {metrics_path}")
            return False
        
        with open(metrics_path) as f:
            current = json.load(f)
        
        # Ensure output directory exists
        baseline_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as baseline
        with open(baseline_output_path, 'w') as f:
            json.dump(current, f, indent=2)
        
        print("✅ Model Registered Successfully")
        print(f"   Baseline saved: {baseline_output_path}")
        print(f"   Metrics:")
        for key, value in current.items():
            if isinstance(value, float):
                print(f"     - {key}: {value:.4f}")
            else:
                print(f"     - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Model registration failed: {e}")
        return False


if __name__ == "__main__":
    metrics_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/metrics/metrics.json"
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/baseline/metrics.json"
    
    success = register_model(metrics_path, baseline_path)
    sys.exit(0 if success else 1)
