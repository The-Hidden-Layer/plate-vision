# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Docker Desktop's build network stalls on large wheels at uv's defaults.
    # opencv-python-headless + numpy are ~50MB, so this matters more here.
    UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=2 \
    # Outside /app: the source bind mount would otherwise hide the venv.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# opencv-python-headless still needs libGL-free glib bits for video decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
# INSTALL_ML: false = stub backends only; cpu = trained models on CPU-only
# PyTorch (default in compose); true = CUDA PyTorch (NVIDIA overlay).
ARG INSTALL_ML=false
ARG INSTALL_TENSORRT=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_TENSORRT" = "true" ]; then uv sync --frozen --extra ml --extra export --extra tensorrt; \
    elif [ "$INSTALL_ML" = "true" ]; then uv sync --frozen --extra ml --extra export; \
    elif [ "$INSTALL_ML" = "cpu" ]; then uv sync --frozen --extra cpu; \
    else uv sync --frozen; fi

COPY . .

# Non-root; /app/media owned here so the named volume inherits the ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/media \
    && chown -R app:app /app /opt/venv
USER app

EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
