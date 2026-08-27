# NeuroScribe backend — FastAPI + PyTorch inference server.
#
# Build:
#   docker build -t neuroscribe-backend ./backend
# Run:
#   docker run -p 8000:8000 neuroscribe-backend
#
# On first startup, if no trained checkpoint is baked into the image (see the
# `RUN python train.py` step below), the app will fall back to training a
# small model in-process for a few seconds. Baking in a checkpoint at build
# time (as done here) makes container startup fast and deterministic.

FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch separately first to keep layer caching effective and
# avoid pulling CUDA wheels inside the image.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    || pip install --no-cache-dir "torch>=2.2"
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY train.py ./train.py

# Train the synthetic-data model at build time so the image ships with real,
# already-trained weights (see app/weights/unet_synthetic.pt after this step).
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
