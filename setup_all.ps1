$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "      MLOps Project:   Bootstrap          " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. .env Validation
if (-not (Test-Path ".env")) {
    Write-Host "[FATAL] .env file is missing." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please create '.env' and fill in the information:"
    Write-Host "------------------------------------------------------------"
    Write-Host "DAGSHUB_USERNAME=your_username"
    Write-Host "DAGSHUB_TOKEN=your_token"
    Write-Host "------------------------------------------------------------"
    exit 1
}

# Load variables from .env
Get-Content .env | ForEach-Object {
    if ($_ -match "^(?<name>[^#=]+)=(?<value>.*)$") {
        $name = $Matches["name"].Trim()
        $value = $Matches["value"].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value)
    }
}

$DAGS_USER = $env:DAGSHUB_USERNAME
if (-not $DAGS_USER) { $DAGS_USER = $env:DAGSHUB_USER }

# 2. Destructive Cleanup
Write-Host "[CLEANUP] Cleaning up previous cluster if exists..." -ForegroundColor Yellow
try {
    # kind writes status to stderr, which triggers NativeCommandError if Stop is used.
    # We temporarily allow errors here.
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    kind delete cluster --name mlops-cluster 2>$null
    $ErrorActionPreference = $oldPreference
} catch {
    # Ignore errors during cleanup
}

# 3. Build Application Image
Write-Host "[BUILD] Building Churn App image (FastAPI and Streamlit)..." -ForegroundColor Yellow
docker build -t churn-api:v1 .

# 4. Cluster Creation
Write-Host "[CLUSTER] Creating Kind Cluster (3 Nodes)..." -ForegroundColor Yellow
kind create cluster --name mlops-cluster --config deployment/kind-config.yaml

# 5. Image Sideloading
Write-Host "[LOAD] Loading image into Kind nodes..." -ForegroundColor Yellow
kind load docker-image churn-api:v1 --name mlops-cluster

# 6. Kubernetes Secrets
Write-Host "[SECRET] Configuring DagsHub Secrets..." -ForegroundColor Yellow
$DagsSecret = "kubectl create secret generic dagshub-secret --from-literal=username=`"$DAGS_USER`" --from-literal=token=`"$env:DAGSHUB_TOKEN`" --dry-run=client -o yaml"
Invoke-Expression $DagsSecret | kubectl apply -f -

# 7. Monitoring Stack (Helm)
Write-Host "[MONITOR] Installing Observability Stack (Prometheus and Grafana)..." -ForegroundColor Yellow
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack --namespace default -f deployment/monitoring-values.yaml

# 8. Application Deployment
Write-Host "[DEPLOY] Deploying Application Manifests..." -ForegroundColor Yellow
kubectl apply -f k8s/grafana-dashboard-configmap.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/ui-deployment.yaml
kubectl apply -f k8s/ui-service.yaml
kubectl apply -f k8s/servicemonitor.yaml

# 9. Wait for Health
Write-Host "[WAIT] Waiting for Pods to be Ready (Training/DVC may take a few minutes)..." -ForegroundColor Yellow
kubectl wait --for=condition=ready pod -l app=churn-api --timeout=600s
kubectl wait --for=condition=ready pod -l app=churn-ui --timeout=600s

# 10. Demo Data Injection
Write-Host "[DATA] Injecting Demo Data for Grafana preview..." -ForegroundColor Yellow
for ($i=1; $i -le 5; $i++) {
    $custId = "demo-$i"
    $payloadObj = @{
        customerID = $custId
        gender = "Male"
        SeniorCitizen = 0
        Partner = "Yes"
        Dependents = "No"
        tenure = 12
        PhoneService = "Yes"
        MultipleLines = "No"
        InternetService = "Fiber optic"
        OnlineSecurity = "No"
        OnlineBackup = "No"
        DeviceProtection = "No"
        TechSupport = "No"
        StreamingTV = "No"
        StreamingMovies = "No"
        Contract = "Month-to-month"
        PaperlessBilling = "Yes"
        PaymentMethod = "Electronic check"
        MonthlyCharges = 70.35
        TotalCharges = 845.5
    }
    $jsonPayload = $payloadObj | ConvertTo-Json -Compress
    
    # Retry logic for data injection
    $success = $false
    $retries = 0
    while (-not $success -and $retries -lt 5) {
        try {
            Invoke-RestMethod -Uri "http://localhost:30100/predict" -Method Post -Body $jsonPayload -ContentType "application/json" | Out-Null
            $success = $true
            Write-Host "Sent demo prediction $i..."
        } catch {
            $retries++
            Write-Host "Waiting for API to respond (Attempt $retries/5)..." -ForegroundColor Gray
            Start-Sleep -Seconds 5
        }
    }
    Start-Sleep -Seconds 1
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "[SUCCESS] DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "Access Endpoints:" -ForegroundColor Green
Write-Host "   - Streamlit UI: http://localhost:30000"
Write-Host "   - FastAPI API:  http://localhost:30100"
Write-Host "   - Grafana:      http://localhost:30200"
Write-Host "   - MLflow:       http://localhost:30500"
Write-Host ""
Write-Host "To get Grafana Admin Password (PowerShell):"
Write-Host '   $pw = kubectl get secret prometheus-grafana -o jsonpath="{.data.admin-password}"; [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($pw))'
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Check README_M4.md for hand-off details and Demo Guide."
Write-Host "============================================================" -ForegroundColor Green
