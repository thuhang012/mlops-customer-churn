
# 📦 Data Version Control (DVC) & DagsHub Setup

Dự án này sử dụng mô hình **DVC (Data Version Control)** kết hợp với **DagsHub Storage** để quản lý dữ liệu lớn. Điều này giúp mã nguồn Git luôn nhẹ nhàng và quá trình cộng tác giữa các máy tính (Local C, Local D, Cloud) trở nên đồng bộ.

---

## 🏗️ Cơ chế hoạt động (How it works)

Mô hình quản lý này tách biệt Mã nguồn và Dữ liệu thật:

* **Git (GitHub/DagsHub Git):** Quản lý mã nguồn và các file **biên lai (.dvc)**. File `.dvc` rất nhẹ, chỉ chứa mã băm (MD5 hash) để định danh file dữ liệu.
* **DVC (DagsHub Storage):** Quản lý "ruột" của các file dữ liệu nặng (ví dụ: `telcom_churn.csv`). Dữ liệu này được lưu trữ trong kho riêng của DagsHub.
* **Sự đồng bộ:** Khi bạn `git pull`, bạn nhận được "biên lai". Sau đó, bạn dùng `dvc pull` để cầm biên lai đó lên kho DagsHub đổi lấy dữ liệu thật.

---

## 🚀 Hướng dẫn lấy dữ liệu cho máy mới (Collaborator Guide)

Nếu bạn vừa clone dự án hoặc làm việc trên một máy tính mới, hãy thực hiện các bước sau:

### 1. Cài đặt môi trường
```powershell
# Tạo môi trường ảo (khuyên dùng)
python -m venv venv
.\venv\Scripts\activate

# Cài đặt DVC và thư viện hỗ trợ DagsHub
pip install dvc[s3] dagshub
```

### 2. Cấu hình xác thực thủ công (Dùng Token - Ổn định nhất)
Lấy Token tại: **DagsHub > Settings > Tokens**

```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user bich-le
dvc remote modify origin --local password 42faca3ef5d1242b9f72c7f5be0f09e97acd4c82
```

### 3. Tải dữ liệu về máy
```powershell
dvc pull -r origin
```
*Sau khi chạy thành công, file dữ liệu thật sẽ xuất hiện tại `data/raw/telcom_churn.csv`.*

---

## 🛠️ Quy trình làm việc hàng ngày (Workflow)

Khi bạn có dữ liệu mới hoặc thay đổi nội dung dữ liệu hiện tại, hãy thực hiện theo thứ tự:

1.  **Đăng ký với DVC:**
    ```powershell
    dvc add data/raw/your_data.csv
    ```

2.  **Đẩy dữ liệu thật lên kho DagsHub:**
    ```powershell
    dvc push -r origin
    ```

3.  **Lưu biên lai (.dvc) vào Git và đẩy lên GitHub:**
    ```powershell
    git add data/raw/your_data.csv.dvc .gitignore
    git commit -m "Update dataset: [tên thay đổi]"
    git push origin main
    ```

---

## ⚠️ Lưu ý quan trọng
* **KHÔNG bao giờ** dùng `git add` trực tiếp lên các file dữ liệu lớn (.csv, .zip, .pkl).
* Luôn đảm bảo file dữ liệu thật đã được liệt kê trong `.gitignore` sau khi `dvc add`.
* Nếu lệnh `dvc pull` báo `Everything is up to date` mà không thấy file xuất hiện, hãy chạy `dvc status` để kiểm tra trạng thái cache.
