# percepta — container image (spec §12, milestone M5).
#
# Build from the repository root so the source tree is in the build context:
#
#     docker build -f docker/Dockerfile -t percepta:0.1.0 .
#
# Model weights are NEVER baked into the image (spec §12). Mount them read-only and
# point `tools.<name>.checkpoint` at the mount, for example:
#
#     docker run --rm -p 127.0.0.1:8790:8790 \
#         -v /srv/models:/models:ro \
#         -v /srv/percepta/config.yaml:/app/configs/example.yaml:ro \
#         percepta:0.1.0
#
# The service binds loopback only; reach it from another machine through an SSH tunnel
# (spec §12): `ssh -N -L 8790:127.0.0.1:8790 user@host`.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Non-root runtime user: the service only needs to read configs and listen on 8790.
RUN groupadd --system percepta \
    && useradd --system --gid percepta --create-home --home-dir /home/percepta \
        --shell /usr/sbin/nologin percepta

# Dependency layer: only what the build backend needs, then install. The `service` extra
# is the runtime surface (FastAPI + uvicorn); the `sam3` extra is deliberately NOT
# installed — torch and the gated weights stay outside the image (spec §12/§13).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[service]"

# Example configs for `percepta serve`; production configs are usually mounted instead.
COPY configs ./configs

# No COPY of models/ or weights/: spec §12 forbids vendoring model weights.
USER percepta

EXPOSE 8790

# Liveness probe: expects the default loopback port; change both EXPOSE and this URL if
# `server.port` differs. urllib is stdlib, so the health check needs no extra package.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8790/healthz', timeout=3).status == 200 else 1)"]

CMD ["percepta", "serve", "--config", "/app/configs/example.yaml"]
