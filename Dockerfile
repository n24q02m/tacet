# Production image for the TACET service.
#
#   docker build -t ghcr.io/n24q02m/tacet:latest .
#   docker run --rm -p 8088:8088 \
#       -e TACET_TEACHER=gemini \
#       -e TACET_GEMINI_API_KEY=$GEMINI_API_KEY \
#       ghcr.io/n24q02m/tacet:latest
#
# Multi-stage build: a `builder` stage installs uv + dependencies and a
# `runtime` stage copies only the installed venv + source. ~150 MB final
# image on the python:3.13-slim base.

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY src/ ./src/
# Install the "service" extra so FastAPI + pydantic-settings come with us.
RUN pip install --prefix=/install '.[service]'

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8088 \
    TACET_PORT=8088 \
    TACET_HOST=0.0.0.0

# Run as a non-root user for safety.
RUN groupadd -g 1000 tacet \
 && useradd -u 1000 -g 1000 -m -s /sbin/nologin tacet

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /app/src /app/src

USER tacet
EXPOSE 8088

HEALTHCHECK --interval=20s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8088/healthz').status==200 else 1)"

# Default command runs the demo "blank" service. In production the user
# supplies their own bootstrap script and overrides CMD.
CMD ["python", "-m", "tacet.serve.server"]
