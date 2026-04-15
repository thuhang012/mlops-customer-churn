# M4 - Docker Infrastructure & Deployment SOP

Tài liệu này quy chuẩn hóa quy trình vận hành (SOP) cho hạ tầng Docker của dự án Netflix Churn Prediction. Hạ tầng được thiết kế để tự động hóa hoàn toàn luồng MLOps từ tải dữ liệu, huấn luyện đến phục vụ API.

## 1. Kiến trúc Hạ tầng "Zero-Config"

Hệ thống được thiết kế theo nguyên tắc Vận hành Tự động (Idempotent Boot) và chống xung đột kỹ thuật:

- **DVC Automation:** Container `api` tự động xác thực với DagsHub thông qua Basic Auth, bật cờ `--force` để tự tin pull dữ liệu về ổ đĩa chia sẻ mà không vướng kiểm duyệt của SCM/Git.
- **Idempotent Training (Huấn luyện thông minh):** Hệ thống chỉ tốn tài nguyên chạy bước làm sạch dữ liệu (Preprocessing) và huấn luyện (RF Model) **khi và chỉ khi** bị thiếu file `Netflix_Prediction_final.pkl`. Nếu file đã tồn tại trên thư mục Host, quá trình này được bỏ qua để ưu tiên nhảy ngay sang bật Server API.
- **MLflow Artifact Proxying:** Thay vì để Container API tự quyền ghi file vào các ổ gắn nối (gây ra lỗi `Permission Denied` vì lệch quyền Root/User), **MLflow Tracking Server** được cấu hình chế độ `--serve-artifacts`. API chỉ đẩy Model sang qua sóng HTTP, còn việc ghi ổ cứng sẽ do Server lo trọn tuyến, giúp đảm bảo chuẩn bảo mật cao nhất (UID 1000 user).

## 2. Quy trình Vận hành Chuẩn (Operating Procedure)

### Bước 1: Khai báo Xác thực
Để DVC có quyền tải dataset từ kho DagsHub, bạn cần điền thông tin vào file `.env`:
```bash
DAGSHUB_USER=ten_dang_nhap_cua_ban
DAGSHUB_TOKEN=token_lay_tu_dagshub
```

### Bước 2: Triển khai & Khởi động
Chỉ cần dùng một lệnh duy nhất để Orchestrator (`docker-compose`) lo liệu mọi thứ:
```powershell
docker-compose up --build
```
> **Tip:** Nếu muốn chạy ngầm, hãy dùng `docker-compose up -d --build`.

### Bước 3: Theo dõi Log & Giám sát
Khi container boot lên, hãy nhìn vào Terminal. Một hệ thống khởi chạy thành công sẽ có các mốc thời gian:
1. `✅ DVC Sync successful. Clean remote dataset located.` -> Kéo data mượt mà.
2. Quá trình Huấn luyện song song bắt đầu (nếu file Model bị xóa).
3. `✅ Active Model found... Bypassing training sequence.` -> Tìm thấy Model cũ, bỏ qua train.
4. `👉 Swagger UI is available at: http://localhost:8000/docs` -> FastAPI đã chạy!

## 3. Quản trị Chuyên sâu

### A. Làm thế nào để ép hệ thống Huấn luyện (Retrain) lại từ đầu?
Vì cơ chế Idempotent, hệ thống sẽ chối từ việc Train nếu đã có Model. Để ép máy học lại từ đầu:
1. Vào thư mục `artifacts/models/` trên VSCode của bạn.
2. Xóa đi file `Netflix_Prediction_final.pkl`.
3. Chờ hoặc khởi động lại bằng tổ hợp `Ctrl + C` và chạy lại lệnh `docker-compose up`.

### B. Đường dẫn (Bảng điều khiển)
Khi hệ thống vào guồng:
- **FastAPI / Swagger UI (Dự đoán khách hàng):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Local MLflow Server (Xem các chỉ số mô hình RMSE, PR-AUC):** [http://localhost:5555](http://localhost:5555)

### C. Reset Tận Gốc (Xóa Database rác)
Nếu MLflow gặp lỗi dính cấu hình cũ, hãy xóa sạch các bộ nhớ trung gian:
```powershell
docker-compose down -v
```
*(Lệnh này chỉ xóa Database của Docker, KHÔNG XÓA file mô hình và mã nguồn của bạn ở ngoài Host).*

## 4. K8s Production Deployment (Bảo Mật & Storage)

### Khởi tạo Dagshub Secret (Bảo mật tuyệt đối)
Tuyệt đối KHÔNG hardcode username và token vào file `.yaml`. Để API có thể kết nối tải data từ Dagshub khi chạy trên Kubernetes, bạn cần chạy tay lệnh tạo K8s Secret từ Terminal/Powershell:
```powershell
kubectl create secret generic dagshub-secret --from-literal=username="<ten-dang-nhap>" --from-literal=token="<token-cua-ban>"
```
> *Lưu ý: K8s Secret sẽ được lưu tự động trong internal namespace Kube System, đảm bảo Hacker xem trộm mã nguồn trên Github cũng không lấy cắp được cấu hình của bạn.*

### Lưu ý về Volume
Môi trường K8s đã được thiết lập `PersistentVolumeClaim` để đảm bảo dữ liệu Lịch sử Training (MLflow db/artifacts) không bị mất khi Pod bị khởi động lại (tránh dùng `emptyDir`).
