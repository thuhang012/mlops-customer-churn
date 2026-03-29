# Sử dụng base image mỏng nhẹ Python 3.11 (Khoảng 150MB)
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong Container
WORKDIR /app

# Khai báo môi trường để Python không nhả bộ đệm (giúp đọc log realtime), tối ưu kích thước
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cài đặt các thư viện lõi hệ thống nếu cần (Bỏ qua nếu không dùng OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# Lệnh ăn tiền nhất: --no-cache-dir ép pip xóa file nén cài đặt ngay lập tức -> Size < 500MB
RUN pip install --no-cache-dir -r requirements.txt

# Copy bộ não code nháp API vào
COPY src/ /app/src/

# Báo hiệu cổng kết nối
EXPOSE 8000

# Lệnh khởi chạy uvicorn thẳng tiến tới file serve.py
CMD ["uvicorn", "src.mlops_project.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
