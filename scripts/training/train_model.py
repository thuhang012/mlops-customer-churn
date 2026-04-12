"""
Model Training Script
Trains model with mock data (ready for real training)
"""
import joblib
import json
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


def train_model(output_dir: str = "artifacts") -> dict:
    """
    Train model and save metrics
    
    Args:
        output_dir: Directory to save model and metrics
    
    Returns:
        Dictionary with model_id, f1_score, roc_auc
    """
    try:
        # Setup directories
        output_dir = Path(output_dir)
        models_dir = output_dir / "models"
        metrics_dir = output_dir / "metrics"
        
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 Training model...")
        
        # Generate mock data for demonstration
        # TODO: Replace with real data loading
        X, y = make_classification(
            n_samples=100, 
            n_features=20, 
            n_informative=15, 
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = models_dir / "model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Save metrics
        metrics = {
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'n_samples': len(X_test),
            'n_features': X.shape[1]
        }
        
        metrics_path = metrics_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics saved: {metrics_path}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - ROC AUC: {roc_auc:.4f}")
        
        return {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'model_path': str(model_path)
        }
        
    except Exception as e:
        print(f"❌ ERROR: Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    results = train_model(output_dir)
    
    # Output for GitHub Actions
    import subprocess
    subprocess.run([
        "bash", "-c",
        f"echo 'f1_score={results[\"f1_score\"]}' >> $GITHUB_OUTPUT"
    ])
    subprocess.run([
        "bash", "-c",
        f"echo 'roc_auc={results[\"roc_auc\"]}' >> $GITHUB_OUTPUT"
    ])
