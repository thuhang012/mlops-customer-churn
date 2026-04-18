# Churn Prediction MLOps Project: Hand-off Documentation (M4)

This document provides technical details and operational instructions for the Churn Prediction infrastructure deployed on Kubernetes (Kind).

## Infrastructure Overview

The project is hosted on a **3-node Kind cluster** (`mlops-cluster`) consisting of 1 control-plane node and 2 worker nodes. The services are exposed via **NodePorts** on `localhost`.

### Access Endpoints

| Service | Protocol | Host Port | Internal Port |
|---------|----------|-----------|---------------|
| **Streamlit UI** | HTTP | **30000** | 8501 |
| **FastAPI (Churn API)** | HTTP | **30100** | 8000 |
| **Grafana Dashboard** | HTTP | **30200** | 3000 |
| **Prometheus UI** | HTTP | **30300** | 9090 |
| **MLflow Tracking** | HTTP | **30500** | 5000 |

---

## Credentials & Secrets

### Grafana Admin Password
The Grafana instance is configured with persistent storage. 

**For Windows (PowerShell):**
```powershell
$pw = kubectl get secret --namespace default prometheus-grafana -o jsonpath="{.data.admin-password}"
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($pw))
```

**For Linux/macOS:**
```bash
kubectl get secret --namespace default prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
```
*Username: `admin`*

### DagsHub Secrets
Stored as a Kubernetes Secret named `dagshub-secret`. Ensure the `.env` file is set up correctly before bootstrap.

---

## Master Setup Scripts

We provide two master scripts for zero-config reproducibility:
1.  **Windows**: `.\setup_all.ps1`
2.  **Linux/macOS**: `./setup_all.sh`

**What these scripts do:**
- Delete any existing cluster named `mlops-cluster`.
- Build the `churn-api:v1` Docker image locally.
- Create the 3-node Kind cluster and load the image.
- Install the Monitoring Stack (Kube-Prometheus-Stack) via Helm.
- Deploy API, UI, and MLflow with **Resource Quotas** (Limits/Requests).
- Inject 5 sample prediction requests to populate the Grafana dashboard immediately.

---

## Professor Demo: Self-Healing & Rolling Update

Use these scenarios to demonstrate the resilience of your infrastructure:

### 1. Self-Healing (Tự hồi phục)
**Scenario**: Demonstrates that Kubernetes automatically restarts failed pods.
- **Step 1**: Open a terminal and watch the pods: `kubectl get pods -w`
- **Step 2**: In another terminal, delete one of the API pods:
  ```powershell
  kubectl delete pod -l app=churn-api --grace-period=0
  ```
- **Step 3**: Show that a new pod is immediately created to maintain the desired state (2 replicas).

### 2. Rolling Update (Cập nhật không gián đoạn)
**Scenario**: Demonstrates updating the application without downtime.
- **Step 1**: Check the current rollout history: `kubectl rollout history deployment/churn-api`
- **Step 2**: Trigger an update by changing an environment variable (or changing image version):
  ```powershell
  kubectl set env deployment/churn-api UPDATE_DATE=$(date +%s)
  ```
- **Step 3**: Watch the "Rolling" process: `kubectl get pods -l app=churn-api`
- **Step 4**: Show the "MaxSurge" behavior: a new pod starts before the old one is killed, ensuring 100% availability.

### 3. Resource Pressure (OOM Check)
- Show the resource limits in `k8s/api-deployment.yaml`.
- Explain how `Limits` protect the node from a single pod consuming all Memory (OOMKill) while `Requests` guarantee baseline performance.
