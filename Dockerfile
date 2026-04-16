# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install system dependencies required for building and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally for extreme fast dependency resolving
RUN pip install --no-cache-dir uv

# Create virtual environment and install dependencies
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    UV_HTTP_TIMEOUT=300 \
    PYTHONPATH=/home/user/app/src

WORKDIR /build
COPY requirements.txt .

# Install reqs + dvc, dagshub, mlflow
# Removing extra dvc[s3] if not needed tightly keeping < 500MB 
# though dvc natively supports dagshub
RUN uv pip install --no-cache -r requirements.txt \
    && uv pip install --no-cache dvc dagshub "mlflow==2.10.0"

# Stage 2: Runtime
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 user \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER user
ENV HOME=/home/user \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/home/user/app/src

WORKDIR $HOME/app

# Copy venv from builder
COPY --from=builder --chown=user:user /opt/venv /opt/venv

# Copy scripts first and ensure entrypoint is executable
COPY --chown=user:user scripts/ ./scripts/
RUN sed -i 's/\r$//' ./scripts/entrypoint.sh && chmod +x ./scripts/entrypoint.sh

# Copy Application Code and Configs
# This ensures that no mock files or raw files are copied, dependent on .dockerignore
COPY --chown=user:user src/ ./src/
COPY --chown=user:user configs/ ./configs/
COPY --chown=user:user data/ ./data/
COPY --chown=user:user artifacts/ ./artifacts/
COPY --chown=user:user dvc.yaml dvc.lock ./
COPY --chown=user:user .dvc/ ./.dvc/

EXPOSE 8000

# Executing the bootstrap
ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
