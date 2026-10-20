# syntax=docker/dockerfile:1

# ---- build stage: resolve and install dependencies with uv -------------------
FROM python:3.11-slim-bookworm AS builder

# uv ships as a static binary in its own image; copying it avoids a pip bootstrap.
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /srv

# Dependency metadata first, so edits to source code do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups

# `--no-default-groups` installs neither `dev` nor `research`, so TensorFlow,
# AutoKeras, Jupyter and matplotlib never enter the image.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups

# ---- runtime stage: copy only the virtualenv and what the app serves ---------
FROM python:3.11-slim-bookworm AS runtime

# libgomp is LightGBM's OpenMP runtime; the wheel links against it but does not
# bundle it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser

WORKDIR /srv
ENV PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder --chown=appuser:appuser /srv/.venv /srv/.venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser configs ./configs
# models/ carries the per-crop models, their cards, and weekly_snapshot.csv -- the
# processed price table the API computes its answers from. The snapshot is CSV rather
# than parquet so the runtime needs no pyarrow, which alone would add ~170 MB.
COPY --chown=appuser:appuser models ./models

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# uvicorn directly: FastAPI is ASGI, and this image serves both the API and the UI
# from one port, so there is no second process to supervise.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--access-log"]
