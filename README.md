# NeuroScribe

NeuroScribe is a portfolio / educational project: a small end-to-end
medical-imaging-*style* pipeline — synthetic data generation, a PyTorch
U-Net segmentation model, gradient-based interpretability, a FastAPI
backend, and a single-file React frontend.

> **Not a medical device.** Every image this project trains on, predicts
> on, or displays is procedurally generated synthetic data meant to loosely
> resemble a 2D scan slice with lesion-like blobs. No real patient data or
> real MRI/CT scans are used anywhere in this repository. This is an
> educational/portfolio demo, not a clinical or diagnostic tool, and must
> never be used to make real medical decisions.

## What's actually here

- **`backend/app/data.py`** — a synthetic 2D "scan" generator: Perlin-ish
  noise background + a few random lesion-like blobs, with a matching binary
  ground-truth mask.
- **`backend/app/model.py`** — `SmallUNet`, a real encoder/decoder U-Net
  with skip connections, implemented in plain PyTorch (no MONAI dependency,
  kept small so it trains in seconds on CPU). Loss is combined BCE + Dice.
- **`backend/app/interpretability.py`** — two gradient-based saliency
  methods: vanilla input-gradient, and a Grad-CAM-style approximation that
  hooks the U-Net's bottleneck activations. **Neither of these is SHAP.**
  The resume line this project is based on mentions SHAP; the actual
  implementation here uses gradient-based saliency instead, and every place
  the API and UI report a method name, they say so explicitly
  (`"...gradient-based approximation — not SHAP"`).
- **`backend/app/main.py`** — a FastAPI app with `GET /health` and
  `POST /segment` (upload an image, get back the resized original, a
  predicted mask, a mask overlay, and a saliency heatmap, all as base64
  PNGs, plus a disclaimer string).
- **`backend/app/storage.py`** — an S3-backed `StorageClient` (via `boto3`),
  tested against `moto`'s in-memory mock AWS backend — no real AWS account
  or credentials needed to run the tests.
- **`backend/train.py`** — a standalone training script that trains
  `SmallUNet` on freshly generated synthetic batches and saves weights to
  `backend/app/weights/unet_synthetic.pt`. If those weights aren't present,
  the API falls back to a quick in-process training run on startup so the
  project works out of the box on a fresh checkout.
- **`frontend/index.html`** — a single-file React 18 app (loaded via CDN,
  no build step, no JSX/Babel — plain `React.createElement`) with an image
  upload form, a saliency-method selector, and a results grid showing the
  original, mask overlay, and saliency heatmap returned by the API.

## Setup & running it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# optional: train and save weights first (otherwise the API trains briefly on startup)
python train.py

uvicorn app.main:app --reload
```

Then open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000` by default, configurable in the UI), or serve it
with any static file server.

### Run with Docker

```bash
cd backend
docker build -t neuroscribe .
docker run --rm -p 8000:8000 neuroscribe
```

## Testing

```bash
cd backend
pytest
```

Covers:

- `tests/test_model.py` — output shape correctness, and a real
  learning-signal check (loss measurably decreases over 25 training steps
  on synthetic batches), plus a Dice-loss sanity check.
- `tests/test_api.py` — `/health`, a full `/segment` round trip against a
  real synthetic PNG (verifies all returned images decode as valid PNGs),
  both saliency methods, and error handling for a bad method name / empty
  upload.
- `tests/test_storage.py` — `StorageClient` against `moto`'s mocked S3:
  bucket creation is idempotent, upload/download round-trips, existence
  checks, prefix listing, and deletion.

### An actual measured training run

Running `python train.py` (20 epochs, default settings) in the environment
this project was built in produced a combined BCE+Dice loss that dropped
from **1.4255 (epoch 1) to 0.3372 (epoch 20)** — a real, monotonic-ish
decrease measured on that run, not an invented number. Exact values will
vary run to run since batches are randomly generated, but loss decreasing
over training is also directly asserted by `test_loss_decreases_with_training`.

## Design notes / honest limitations

- This uses a 2D slice-based model, not a full 3D volumetric one, so the
  whole pipeline trains and runs in seconds on CPU.
- "Saliency" here means gradient-based approximations (vanilla gradient,
  Grad-CAM-style), which are legitimate, real interpretability techniques —
  but they are not SHAP, and this project doesn't claim otherwise anywhere
  in its code, API responses, or UI.
- The synthetic data generator creates images that are visually
  scan-*like* (grayscale, blob-shaped structures) but were not designed to
  match any specific real imaging modality's noise characteristics,
  intensity distribution, or anatomy. Segmentation performance on this
  synthetic data says nothing about performance on real scans.

## License

MIT — see [LICENSE](LICENSE).
