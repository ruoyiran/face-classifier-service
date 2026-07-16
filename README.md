# face-classifier-service

Private, local FastAPI service for phone-album face atomic capabilities:
face detection, 112x112 ArcFace alignment, face quality scoring, and L2-normalized
face embeddings.

V1 intentionally does not provide clustering, naming, face-library management,
deduplication, liveness, age or gender prediction, cloud sync, or a frontend UI.

## API

All face endpoints are mounted under `{API_BASE_PATH}/v1/faces/...`.
`API_BASE_PATH` defaults to no prefix. Empty string and `/` both mean no prefix.

- `GET /healthz`
- `GET /readyz`
- `GET /v1/encryption/public-key`
- `POST /v1/faces/detect`
- `POST /v1/faces/detect/encrypted`
- `POST /v1/faces/align`
- `POST /v1/faces/align/encrypted`
- `POST /v1/faces/quality`
- `POST /v1/faces/quality/encrypted`
- `POST /v1/faces/embed`
- `POST /v1/faces/embed/encrypted`
- `POST /v1/faces/analyze`
- `POST /v1/faces/analyze/encrypted`

Plain endpoints accept multipart `files` and return `results[]`, where each item
contains `image_id`, original `width`/`height`, `faces[]`, and `warnings[]`.
Embeddings are L2-normalized by the service and returned as `float32_base64` by
default with `normalized: true`.

Encrypted endpoints use `AES-256-GCM+SealedBox-v1`, matching the sibling
`clip-classifier-service` implementation. They accept encrypted JSON uploads and
return an encrypted envelope; decrypting that envelope yields the same
`results[]` payload as the corresponding plain endpoint. Enable Bearer auth with
`FACE_AUTH_ENABLED=true` and comma-separated `FACE_API_TOKENS`. Enable encrypted
uploads with `FACE_ENCRYPTION_ENABLED=true`; set `FACE_REQUIRE_ENCRYPTED_UPLOADS=true`
to reject plaintext multipart face endpoints.

See `docs/API_TESTING.md` for curl, Python, and Go test commands/scripts for every API.

## Model Artifacts

Production model files are expected to be pre-seeded locally under `/models`.
The service validates the required artifact manifest and `.sha256` sidecars at
startup and never downloads model weights during live traffic.

Expected manifest:

- `insightface/scrfd_34g.onnx`
- `insightface/scrfd_34g.onnx.sha256`
- `insightface/2d106det.onnx`
- `insightface/2d106det.onnx.sha256`
- `fdbtrs/CR-FIQA-L/181952backbone.pth`
- `fdbtrs/CR-FIQA-L/181952backbone.pth.sha256`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/model.safetensors`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/model.safetensors.sha256`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/wrapper.py`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/wrapper.py.sha256`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.yaml`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.yaml.sha256`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.pt`
- `minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.pt.sha256`

Before production rollout, keep license, weight source, and training-data
restriction records beside the artifact bundle. The concrete native model runner
is loaded with:

```env
FACE_ANALYZER_FACTORY=face_classifier_service.native_analyzer:create_analyzer
```

The factory is called after local artifact validation and must return a ready
`FaceAnalyzer` implementation. Without a configured factory, `/readyz` and face
analysis requests return `503`.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Install native model runtime dependencies from `requirements-models.txt` so pip
uses the required PyTorch CUDA index:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

Run locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn face_classifier_service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Bind to `0.0.0.0` only behind an internal gateway or reverse proxy that provides
TLS, authentication, rate limits, access logs with image/embedding redaction, and
network allowlisting.
