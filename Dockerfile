FROM python:3.11-slim

ARG INSTALL_PHOTOMETRIC=false
ARG INSTALL_MEDIAPIPE=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt requirements-photometric.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && if [ "$INSTALL_PHOTOMETRIC" = "true" ]; then python -m pip install -r requirements-photometric.txt; fi \
    && if [ "$INSTALL_MEDIAPIPE" = "true" ]; then python -m pip install -e ".[mediapipe]"; else python -m pip install -e .; fi

CMD ["python", "-m", "eyewear.cli", "--help"]
