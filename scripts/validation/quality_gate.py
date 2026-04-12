"""
Quality Gate Script
Compares current metrics against baseline
"""
import json
import sys
from pathlib import Path
>>>>>>> 331fce0f5feb6c369994404231f4c01687224982

<<<<<<< HEAD

def check_quality_gate(
    current_metrics_path: str,
    baseline_metrics_path: str = "artifacts/baseline/metrics.json",
    threshold: float = 0.95
) -> bool:
    """
    Check if current metrics pass quality gate
    
    Args:
        current_metrics_path: Path to current metrics.json
        baseline_metrics_path: Path to baseline metrics.json
        threshold: Allowed degradation (0.95 = 5% max degradation)
    
    Returns:
        True if passes, False otherwise
    """
    try:
        current_metrics_path = Path(current_metrics_path)
        baseline_metrics_path = Path(baseline_metrics_path)
        # Load current metrics
        if not current_metrics_path.exists():
            print(f"❌ ERROR: Current metrics not found: {current_metrics_path}")
            return False
        
        with open(current_metrics_path) as f:
            current = json.load(f)
        
        # Load or create baseline
        if baseline_metrics_path.exists():
            with open(baseline_metrics_path) as f:
                baseline = json.load(f)
        else:
            # First run - use conservative defaults
            baseline = {
                'f1_score': 0.60,
                'roc_auc': 0.65
            }
            print(f"⚠️  No baseline found - using defaults: {baseline}")
        
        # Quality gate checks
        print("\n📋 Quality Gate Checks:")
        print("=" * 50)
        
        passed = True
        
        # Check F1 Score
        current_f1 = current.get('f1_score', 0)
        baseline_f1 = baseline.get('f1_score', 0.60)
        threshold_f1 = baseline_f1 * threshold
        f1_pass = current_f1 >= threshold_f1
        
        print(f"F1 Score:    {current_f1:.4f} >= {threshold_f1:.4f}? {'✅' if f1_pass else '❌'}")
        if not f1_pass:
            print(f"  Degradation: {((baseline_f1 - current_f1) / baseline_f1 * 100):.1f}% (max: {(1-threshold)*100:.1f}%)")
        passed = passed and f1_pass
        
        # Check ROC AUC
        current_auc = current.get('roc_auc', 0)
        baseline_auc = baseline.get('roc_auc', 0.65)
        threshold_auc = baseline_auc * threshold
        auc_pass = current_auc >= threshold_auc
        
        print(f"ROC AUC:     {current_auc:.4f} >= {threshold_auc:.4f}? {'✅' if auc_pass else '❌'}")
        if not auc_pass:
            print(f"  Degradation: {((baseline_auc - current_auc) / baseline_auc * 100):.1f}% (max: {(1-threshold)*100:.1f}%)")
        passed = passed and auc_pass
        
        print("=" * 50)
        
        if passed:
            print("\n✅ Quality Gate PASSED")
            print(f"   Current metrics are acceptable for deployment")
        else:
            print("\n❌ Quality Gate FAILED")
            print(f"   Model degradation exceeds threshold")
        
        _write_output(passed)
        return passed
        
    except Exception as e:
        print(f"❌ ERROR: Quality gate check failed: {e}")
        _write_output(False)
        return False


def _write_output(passed: bool) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"passed={str(passed).lower()}\n")


if __name__ == "__main__":
    current_metrics = sys.argv[1] if len(sys.argv) > 1 else "artifacts/metrics/metrics.json"
    baseline_metrics = sys.argv[2] if len(sys.argv) > 2 else "artifacts/baseline/metrics.json"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
    
    passed = check_quality_gate(current_metrics, baseline_metrics, threshold)
    sys.exit(0 if passed else 1)

=======

def check_quality_gate(
    current_metrics_path: str,
    baseline_metrics_path: str = "artifacts/baseline/metrics.json",
    threshold: float = 0.95
) -> bool:
    """
    Check if current metrics pass quality gate
    
    Args:
        current_metrics_path: Path to current metrics.json
        baseline_metrics_path: Path to baseline metrics.json
        threshold: Allowed degradation (0.95 = 5% max degradation)
    
    Returns:
        True if passes, False otherwise
    """
    try:
        current_metrics_path = Path(current_metrics_path)
        baseline_metrics_path = Path(baseline_metrics_path)
        
        # Load current metrics
        if not current_metrics_path.exists():
            print(f"❌ ERROR: Current metrics not found: {current_metrics_path}")
            return False
        
        with open(current_metrics_path) as f:
            current = json.load(f)
        
        # Load or create baseline
        if baseline_metrics_path.exists():
            with open(baseline_metrics_path) as f:
                baseline = json.load(f)
        else:
            # First run - use conservative defaults
            baseline = {
                'f1_score': 0.60,
                'roc_auc': 0.65
            }
            print(f"⚠️  No baseline found - using defaults: {baseline}")
        
        # Quality gate checks
        print("\n📋 Quality Gate Checks:")
        print("=" * 50)
        
        passed = True
        
        # Check F1 Score
        current_f1 = current.get('f1_score', 0)
        baseline_f1 = baseline.get('f1_score', 0.60)
        threshold_f1 = baseline_f1 * threshold
        f1_pass = current_f1 >= threshold_f1
        
        print(f"F1 Score:    {current_f1:.4f} >= {threshold_f1:.4f}? {'✅' if f1_pass else '❌'}")
        if not f1_pass:
            print(f"  Degradation: {((baseline_f1 - current_f1) / baseline_f1 * 100):.1f}% (max: {(1-threshold)*100:.1f}%)")
        passed = passed and f1_pass
        
        # Check ROC AUC
        current_auc = current.get('roc_auc', 0)
        baseline_auc = baseline.get('roc_auc', 0.65)
        threshold_auc = baseline_auc * threshold
        auc_pass = current_auc >= threshold_auc
        
        print(f"ROC AUC:     {current_auc:.4f} >= {threshold_auc:.4f}? {'✅' if auc_pass else '❌'}")
        if not auc_pass:
            print(f"  Degradation: {((baseline_auc - current_auc) / baseline_auc * 100):.1f}% (max: {(1-threshold)*100:.1f}%)")
        passed = passed and auc_pass
        
        print("=" * 50)
        
        if passed:
            print("\n✅ Quality Gate PASSED")
            print(f"   Current metrics are acceptable for deployment")
        else:
            print("\n❌ Quality Gate FAILED")
            print(f"   Model degradation exceeds threshold")
        
        return passed
        
    except Exception as e:
        print(f"❌ ERROR: Quality gate check failed: {e}")
        return False


if __name__ == "__main__":
    current_metrics = sys.argv[1] if len(sys.argv) > 1 else "artifacts/metrics/metrics.json"
    baseline_metrics = sys.argv[2] if len(sys.argv) > 2 else "artifacts/baseline/metrics.json"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
    
    passed = check_quality_gate(current_metrics, baseline_metrics, threshold)
    sys.exit(0 if passed else 1)

>>>>>>> 331fce0f5feb6c369994404231f4c01687224982