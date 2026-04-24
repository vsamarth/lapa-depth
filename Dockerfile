FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    pkg-config vim unzip \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/LAPA

RUN python3 -m pip install --upgrade pip setuptools wheel

# 1. PyTorch CUDA 12.1
RUN python3 -m pip install \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. JAX CUDA 12 + cuDNN 8.9
RUN python3 -m pip install \
    jax==0.4.23 \
    jaxlib==0.4.23+cuda12.cudnn89 \
    jax-cuda12-pjrt==0.4.23 \
    jax-cuda12-plugin==0.4.23 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. TensorFlow 2.15, do not use tensorflow[and-cuda]
RUN python3 -m pip install \
    tensorflow==2.15.0 \
    keras==2.15.0 \
    tensorboard==2.15.2 \
    tensorflow-estimator==2.15.0 \
    tensorflow-io-gcs-filesystem==0.37.1 \
    ml-dtypes==0.2.0
    

# 4. Other Python packages
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# 5. Force correct JAX version after requirements installation
RUN python3 -m pip uninstall -y jax jaxlib jax-cuda12-pjrt jax-cuda12-plugin && \
    python3 -m pip install \
    jax==0.4.23 \
    jaxlib==0.4.23+cuda12.cudnn89 \
    jax-cuda12-pjrt==0.4.23 \
    jax-cuda12-plugin==0.4.23 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 6. Install extra packages added later
COPY requirements_update.txt /tmp/requirements_update.txt
RUN python3 -m pip install -r /tmp/requirements_update.txt

RUN ln -sf /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]