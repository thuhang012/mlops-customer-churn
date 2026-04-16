$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "          MLOps Cluster & Observability Bootstrapper        "
Write-Host "============================================================"

if (-Not (Test-Path ".env")) {
    Write-Host "FATAL: .env file is missing." -ForegroundColor Red
    Write-Host "Hệ thống đạt chuẩn Reproducibility Zero yêu cầu file .env!" -ForegroundColor Yellow
    Write-Host "Please create a .env file with DAGSHUB_USERNAME and DAGSHUB_TOKEN."
    exit 1
}

# Parse .env file
Get-Content .env | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
    $name, $value = $_.Split('=', 2)
    Set-Variable -Name $name.Trim() -Value $value.Trim() -Scope Script
}

$DAGS_USER = $DAGSHUB_USERNAME
if ([string]::IsNullOrWhitespace($DAGS_USER)) {
    $DAGS_USER = $DAGSHUB_USER
}

if ([string]::IsNullOrWhitespace($DAGS_USER) -or [string]::IsNullOrWhitespace($DAGSHUB_TOKEN)) {
    Write-Host "FATAL: DAGSHUB_USER or DAGSHUB_TOKEN missing in .env" -ForegroundColor Red
    exit 1
}

Write-Host "Bootstrapping cluster..." -ForegroundColor Cyan
& kind create cluster --name mlops-cluster --config deployment/kind-config.yaml
if ($LASTEXITCODE -ne 0) { Write-Host "Warning: Cluster might already exist, continuing..." -ForegroundColor Yellow }

Write-Host "Loading churn-api:v1 image into kind..." -ForegroundColor Cyan
& kind load docker-image churn-api:v1 --name mlops-cluster
if ($LASTEXITCODE -ne 0) { Write-Host "Warning: Please build the image first using 'docker build -t churn-api:v1 .'" -ForegroundColor Yellow }

Write-Host "Creating Kubernetes Secrets..." -ForegroundColor Cyan
& kubectl create secret generic dagshub-secret --from-literal=username="$DAGS_USER" --from-literal=token="$DAGSHUB_TOKEN" --dry-run=client -o yaml | Out-File "temp-secret.yaml" -Encoding utf8
& kubectl apply -f temp-secret.yaml
Remove-Item temp-secret.yaml

Write-Host "Bootstrapping Prometheus/Grafana Stack..." -ForegroundColor Cyan
& helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
& helm repo update
& helm upgrade --install prometheus prometheus-community/kube-prometheus-stack --namespace default -f k8s/monitoring-values.yaml

Write-Host "Applying application manifests..." -ForegroundColor Cyan
& kubectl apply -f k8s/api-deployment.yaml
& kubectl apply -f k8s/api-service.yaml
& kubectl apply -f k8s/ui-deployment.yaml
& kubectl apply -f k8s/ui-service.yaml
& kubectl apply -f k8s/servicemonitor.yaml
& kubectl apply -f k8s/mlflow.yaml

Write-Host "============================================================"
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "📍 Access Endpoints:"
Write-Host "   - Streamlit UI: http://localhost:30000"
Write-Host "   - FastAPI: http://localhost:30100"
Write-Host "   - Grafana: http://localhost:30200"
Write-Host "   - Prometheus: http://localhost:30300"
Write-Host "   - MLflow Tracking: http://localhost:30500"
Write-Host ""
Write-Host "🔐 Grafana Admin Password Retrieval Command:"
Write-Host "👉 Run the following command in PowerShell to get your Grafana password:"
Write-Host "   [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String((kubectl get secret --namespace default prometheus-grafana -o jsonpath=`"{.data.admin-password}`")))"
Write-Host "============================================================"
