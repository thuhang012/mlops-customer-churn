# Sử dụng base image mỏng nhẹ Python 3.11 (Khoảng 150MB) -> Tối ưu < 200MB
FROM python:3.11-slim

# Bảo mật: Cấm chạy quyền Root. Bắt buộc tạo một user thường tên là 'user' với ID 1000
RUN useradd -m -u 1000 user \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Chuyển quyền điều khiển sang cho user
USER user

# Thiết lập đường dẫn môi trường cho user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dọn nhà sang khu vực của user và tạo thư mục log monitoring
WORKDIR $HOME/app
RUN mkdir -p monitoring/inference && chown -R user:user monitoring

# Khởi tạo Virtual Environment cho User và đặt vào PATH
ENV VIRTUAL_ENV=/home/user/app/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Tách riêng cài đặt UV (Rust-based) vào trong venv
RUN pip install --no-cache-dir uv

# Mượn quyền cấp cho user, copy file text (Tận dụng Layer Caching)
COPY --chown=user:user requirements.txt .

# Cài đặt thư viện siêu tốc bằng uv (Đã có venv nên không cần --system)
RUN uv pip install --no-cache -r requirements.txt

# Copy entrypoint bootstrap script (M4) và cấp quyền thực thi
COPY --chown=user:user scripts/entrypoint.sh ./scripts/entrypoint.sh
RUN chmod +x ./scripts/entrypoint.sh

# Copy cấu hình model
COPY --chown=user:user artifacts/ ./artifacts/

# Copy source code
COPY --chown=user:user src/ ./src/

# Mở cổng kết nối 8000
EXPOSE 8000

# Sử dụng entrypoint script: tự kiểm tra model, train nếu thiếu, rồi chạy API
ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
