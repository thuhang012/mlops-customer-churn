# Tài liệu Kỹ thuật về Triển khai Hạ tầng Docker (Docker Infrastructure & Orchestration)

Tài liệu này trình bày chi tiết về kiến trúc hạ tầng và quy trình triển khai ứng dụng dự đoán Churn khách hàng thông qua công nghệ Containerization. Mục tiêu trọng tâm là thiết lập một môi trường thực thi nhất quán, bền bỉ (Robustness) và sẵn sàng cho việc mở rộng.

## 1. Tổng quan Kiến trúc Hạ tầng (Architectural Overview)

Hệ thống được thiết kế theo mô hình **Multi-Container Microservices**, bao gồm hai thành phần chính được điều phối bởi Docker Compose:

*   **API Service (Consumer):** Phát triển trên nền tảng FastAPI, thực hiện nhiệm vụ suy luận (Inference) và cung cấp RESTful API.
*   **MLflow Tracking Server (Provider):** Quản lý vòng đời mô hình, lưu trữ metrics và metadata của các thực nghiệm.

Hai dịch vụ này được cô lập trong một mạng nội bộ (`mlops_bridge`) để đảm bảo tính an ninh và tối ưu hóa hiệu suất giao tiếp nội bộ.

## 2. Các Quyết định Kỹ thuật Trọng yếu (Key Infrastructure Decisions)

### 2.1. Cơ chế Tự động Khởi tạo (Self-healing Bootstrap)
Để đảm bảo API Service không bị crash khi model artifacts chưa được huấn luyện hoặc bị thất lạc, dự án triển khai một script `entrypoint.sh` thực hiện chuỗi logic bootstrap:
1.  **Health-check:** Kiểm tra tính sẵn sàng của MLflow Server thông qua TCP socket.
2.  **Artifact Validation:** Kiểm tra sự hiện diện của file `.pkl`.
3.  **Automated Training:** Tự động kích hoạt pipeline huấn luyện nếu thiếu artifacts, sử dụng tập dữ liệu mock để đảm bảo API luôn khởi chạy được (Robustness).
4.  **Placeholder Fail-safe:** Trong trường hợp huấn luyện thực tế thất bại, hệ thống tự tạo Model Placeholder đúng cấu trúc để duy trì tính hoạt động của API server.

### 2.2. Quản lý Dữ liệu Bền vững (Data Persistence)
Hệ thống sử dụng cơ chế **Volume Mounting** để ánh xạ dữ liệu giữa Host và Container:
*   Mô hình đã huấn luyện và preprocessor được lưu trữ trực tiếp tại máy Host (`./artifacts`), đảm bảo không bị mất dữ liệu khi tái khởi động container.
*   Database của MLflow được quản lý bởi một Docker Volume định danh (`mlflow_data`) để bảo toàn lịch sử thực nghiệm.

### 2.3. Tối ưu hóa Docker Image
*   **Base Image:** Sử dụng `python:3.11-slim` để giảm thiểu dung lượng image và hạn chế bề mặt tấn công (Security).
*   **Non-root User:** Ứng dụng được thực thi dưới quyền user (`UID 1000`) thay vì root để tuân thủ các nguyên tắc bảo mật tiêu chuẩn.
*   **Package Management:** Sử dụng `uv` (Rust-based) để tăng tốc độ cài đặt phụ thuộc và đảm bảo tính tái lập (Reproducibility).

## 3. Hướng dẫn Vận hành (Operating Instructions)

Hệ thống được vận hành theo triết lý "Zero-Config". Toàn bộ hạ tầng có thể được thiết lập thông qua một lệnh điều phối duy nhất:

```bash
# Xây dựng và khởi chạy toàn bộ hệ thống
docker-compose up -d --build
```

### Các tham số quản lý chính:
*   **Dừng hệ thống:** `docker-compose down`
*   **Xóa bỏ hoàn toàn (bao gồm volumes):** `docker-compose down -v`
*   **Kiểm tra trạng thái Bootstrap:** `docker logs -f churn_api_container`

## 4. Các Điểm cuối Truy cập (Service Endpoints)

Sau khi hạ tầng sẵn sàng, các dịch vụ được ánh xạ ra cổng vật lý của máy Host như sau:

| Thành phần | Địa chỉ truy cập (Host) | Chức năng |
|---|---|---|
| **FastAPI Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Tài liệu hóa và kiểm thử API |
| **MLflow Dashboard** | [http://localhost:5555](http://localhost:5555) | Quản lý metrics và artifacts |
| **Health Endpoint** | [http://localhost:8000/health](http://localhost:8000/health) | Giám sát trạng thái hoạt động |

> **Lưu ý về Networking:** Mặc dù log hệ thống hiển thị lắng nghe tại `0.0.0.0`, người dùng cuối truy cập từ môi trường Host cần sử dụng định danh `localhost` hoặc `127.0.0.1` để thông qua cơ chế Port Forwarding của Docker.
