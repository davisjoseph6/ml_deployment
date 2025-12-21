FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_DEVICE=cpu \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Git is required because requirements.txt contains a git+https dependency (MobileSAM).
# build-essential helps if any deps compile native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.prod.txt /app/requirements.prod.txt

# IMPORTANT: avoid installing CUDA torch wheels inside a CPU container.
# Install CPU-only PyTorch explicitly, then install the rest.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /app/requirements.prod.txt

RUN python -c "import torch; print('torch', torch.__version__); assert '+cpu' in torch.__version__, torch.__version__"

COPY . /app

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300"]

