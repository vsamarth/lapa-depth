FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    pkg-config \
    unzip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install uv

WORKDIR /workspace/lapa

COPY . /workspace/lapa

RUN mkdir -p /workspace/data /workspace/outputs

CMD ["/bin/bash"]
