#!/bin/bash
set -e

echo "============================================================"
echo "          MLOps Cluster & Observability Bootstrapper        "
echo "============================================================"

# Ensure .env exists 
if [ ! -f .env ]; then
    echo "❌ FATAL: .env file is missing."
    echo "Hệ thống đạt chuẩn Reproducibility Zero yêu cầu file .env!"
    echo "Please create a .env file with DAGSHUB_USERNAME (or DAGSHUB_USER) and DAGSHUB_TOKEN."
    exit 1
fi

export $(grep -v '^#' .env | xargs)

# Extract robust DAGSHUB_USER
DAGS_USER=${DAGSHUB_USERNAME:-$DAGSHUB_USER}

if [ -z "$DAGS_USER" ] || [ -z "$DAGSHUB_TOKEN" ]; then
    echo "❌ FATAL: DAGS_USER or DAGSHUB_TOKEN missing in .env"
    exit 1
fi

echo "🚀 Bootstrapping cluster..."
kind create cluster --name mlops-cluster --config deployment/kind-config.yaml || echo "Cluster already exists"

echo "⏳ Loading churn-api:v1 image into kind..."
# Giả sử image đã được build bằng 'docker build -t churn-api:v1 .' ở terminal
kind load docker-image churn-api:v1 --name mlops-cluster || echo "Please build the image first using 'docker build -t churn-api:v1 .'"

echo "🔐 Creating Kubernetes Secrets..."
kubectl create secret generic dagshub-secret \
  --from-literal=username="$DAGS_USER" \
  --from-literal=token="$DAGSHUB_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "🌐 Bootstrapping Prometheus/Grafana Stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace default \
  -f k8s/monitoring-values.yaml

echo "🛠️ Applying application manifests..."
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/ui-deployment.yaml
kubectl apply -f k8s/ui-service.yaml
kubectl apply -f k8s/servicemonitor.yaml
kubectl apply -f k8s/mlflow.yaml

echo "============================================================"
echo "✅ Setup Complete!"
echo "📍 Access Endpoints:"
echo "   - Streamlit UI: http://localhost:30000"
echo "   - FastAPI: http://localhost:30100"
echo "   - Grafana: http://localhost:30200"
echo "   - Prometheus: http://localhost:30300"
echo "   - MLflow Tracking: http://localhost:30500"
echo ""
echo "🔐 Grafana Admin Password Retrieval Command:"
echo "👉 Copy and paste the following line to reveal your Grafana login password:"
echo "   kubectl get secret --namespace default prometheus-grafana -o jsonpath=\"{.data.admin-password}\" | base64 --decode ; echo"
echo "============================================================"
