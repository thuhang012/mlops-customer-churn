#!/bin/bash
set -e

echo "============================================================"
echo "      MLOps Project: Bootstrap      "
echo "============================================================"

# 1. .env Validation
if [ ! -f .env ]; then
    echo "[FATAL] .env file is missing."
    echo ""
    echo "Please create '.env' and fill in the information:"
    echo "------------------------------------------------------------"
    echo "DAGSHUB_USERNAME=your_username"
    echo "DAGSHUB_TOKEN=your_token"
    echo "------------------------------------------------------------"
    exit 1
fi

export $(grep -v '^#' .env | xargs)
DAGS_USER=${DAGSHUB_USERNAME:-$DAGSHUB_USER}

# 2. Destructive Cleanup
echo "[CLEANUP] Cleaning up previous cluster if exists..."
kind delete cluster --name mlops-cluster || true

# 3. Build Application Image
echo "[BUILD] Building Churn App image (FastAPI + Streamlit)..."
docker build -t churn-api:v1 .

# 4. Cluster Creation
echo "[CLUSTER] Creating Kind Cluster (3 Nodes)..."
kind create cluster --name mlops-cluster --config deployment/kind-config.yaml

# 5. Image Sideloading
echo "[LOAD] Loading image into Kind nodes..."
kind load docker-image churn-api:v1 --name mlops-cluster

# 6. Kubernetes Secrets
echo "[SECRET] Configuring DagsHub Secrets..."
kubectl create secret generic dagshub-secret \
  --from-literal=username="$DAGS_USER" \
  --from-literal=token="$DAGSHUB_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

# 7. Monitoring Stack (Helm)
echo "[MONITOR] Installing Observability Stack (Prometheus & Grafana)..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace default \
  -f deployment/monitoring-values.yaml

# 8. Application Deployment
echo "[DEPLOY] Deploying Application Manifests..."
# Apply ConfigMap first for Grafana
kubectl apply -f k8s/grafana-dashboard-configmap.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/ui-deployment.yaml
kubectl apply -f k8s/ui-service.yaml
kubectl apply -f k8s/servicemonitor.yaml

# 9. Wait for Health
echo "[WAIT] Waiting for Pods to be Ready (Training/DVC may take a few minutes)..."
kubectl wait --for=condition=ready pod -l app=churn-api --timeout=600s
kubectl wait --for=condition=ready pod -l app=churn-ui --timeout=600s

# 10. Demo Data Injection
echo "[DATA] Injecting Demo Data for Grafana preview..."
# We use the NodePort 30100 for the API
for i in {1..5}
do
   # Retry logic for data injection
   success=false
   retries=0
   until [ "$success" = true ] || [ $retries -eq 5 ]
   do
     if curl -s -X POST http://localhost:30100/predict \
       -H "Content-Type: application/json" \
       -d '{"customerID": "demo-'"$i"'", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 845.5}' > /dev/null; then
       success=true
       echo "Sent demo prediction $i..."
     else
       retries=$((retries+1))
       echo "Waiting for API to respond (Attempt $retries/5)..."
       sleep 5
     fi
   done
   sleep 1
done

echo "============================================================"
echo "[SUCCESS] DEPLOYMENT COMPLETE!"
echo "Access Endpoints:"
echo "   - Streamlit UI: http://localhost:30000"
echo "   - FastAPI API:  http://localhost:30100"
echo "   - Grafana:      http://localhost:30200"
echo "   - MLflow:       http://localhost:30500"
echo "============================================================"
echo "Check README_M4.md for hand-off details."
echo "============================================================"
