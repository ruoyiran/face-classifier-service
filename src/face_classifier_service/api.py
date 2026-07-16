from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from face_classifier_service import __version__
from face_classifier_service.analysis import (
    AlignmentResult,
    AnalysisOptions,
    FaceAnalyzer,
    FaceObservation,
    ImageAnalysisResult,
    QualityResult,
    create_default_analyzer,
)
from face_classifier_service.auth import create_auth_dependency, validate_bearer_token
from face_classifier_service.embedding import encode_float32_base64, l2_normalize
from face_classifier_service.encryption import (
    ENCRYPTION_ALGORITHM,
    decrypt_encrypted_image_payload_with_key,
    encrypt_response_for_client,
    read_key_b64,
)
from face_classifier_service.preprocessing import ImageTooLargeError, decode_image
from face_classifier_service.settings import Settings

EmbeddingFormat = Literal["float32_base64", "json"]


class EncryptedImagePayload(BaseModel):
    alg: str
    encrypted_key: str
    nonce: str
    ciphertext: str
    filename: str | None = None


class EncryptedBatchPayload(BaseModel):
    items: list[EncryptedImagePayload]


async def read_upload_limited(file: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")
        chunks.append(chunk)
    return b"".join(chunks)


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app,
        *,
        route_prefix: str,
        max_plain_body_bytes: int,
        max_encrypted_body_bytes: int,
    ) -> None:
        self.app = app
        self.route_prefix = route_prefix
        self.max_plain_body_bytes = max_plain_body_bytes
        self.max_encrypted_body_bytes = max_encrypted_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.route_prefix):
            await self.app(scope, receive, send)
            return

        total = 0
        messages = []
        max_body_bytes = self.max_body_bytes_for_path(scope.get("path", ""))
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > max_body_bytes:
                    response = Response(content="request body too large", status_code=413)
                    await response(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    def max_body_bytes_for_path(self, path: str) -> int:
        if path.endswith("/encrypted"):
            return self.max_encrypted_body_bytes
        return self.max_plain_body_bytes


def docs_path(base_path: str, suffix: str) -> str:
    return f"{base_path}{suffix}" if base_path else suffix


def create_app(
    settings: Settings | None = None,
    analyzer: FaceAnalyzer | None = None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    state: dict[str, FaceAnalyzer | None] = {"analyzer": analyzer}
    gpu_semaphore = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["analyzer"] is None and load_on_startup:
            state["analyzer"] = create_default_analyzer(settings)
        app.state.analyzer = state["analyzer"]
        yield

    disable_docs = settings.auth_enabled and settings.auth_disable_docs
    app = FastAPI(
        title=settings.service_name,
        version=__version__,
        lifespan=lifespan,
        docs_url=None if disable_docs else docs_path(settings.api_base_path, "/docs"),
        redoc_url=None if disable_docs else docs_path(settings.api_base_path, "/redoc"),
        openapi_url=None if disable_docs else docs_path(settings.api_base_path, "/openapi.json"),
    )
    router_dependencies = (
        [Depends(create_auth_dependency(settings))] if settings.auth_enabled else []
    )
    router = APIRouter(
        prefix=f"{settings.api_base_path}/v1/faces",
        dependencies=router_dependencies,
    )
    utility_router = APIRouter(prefix=settings.api_base_path, dependencies=router_dependencies)
    face_route_prefix = f"{settings.api_base_path}/v1/faces"
    max_plain_face_body_bytes = (
        settings.max_image_bytes + settings.max_multipart_overhead_bytes
    ) * settings.max_batch_size
    max_encrypted_face_body_bytes = settings.max_encrypted_image_bytes * settings.max_batch_size
    app.add_middleware(
        BodySizeLimitMiddleware,
        route_prefix=face_route_prefix,
        max_plain_body_bytes=max_plain_face_body_bytes,
        max_encrypted_body_bytes=max_encrypted_face_body_bytes,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.middleware("http")
    async def content_length_limit_middleware(request: Request, call_next):
        if request.url.path.startswith(face_route_prefix):
            if settings.auth_enabled:
                try:
                    validate_bearer_token(request.headers.get("authorization"), settings)
                except HTTPException as exc:
                    return Response(
                        content=f'{{"detail":"{exc.detail}"}}',
                        status_code=exc.status_code,
                        media_type="application/json",
                        headers=exc.headers,
                    )
            content_length = request.headers.get("content-length")
            try:
                parsed_content_length = int(content_length) if content_length else None
            except ValueError:
                return Response(content="invalid content-length", status_code=400)
            max_body_bytes = (
                max_encrypted_face_body_bytes
                if request.url.path.endswith("/encrypted")
                else max_plain_face_body_bytes
            )
            if parsed_content_length and parsed_content_length > max_body_bytes:
                return Response(content="request body too large", status_code=413)
        return await call_next(request)

    @utility_router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @utility_router.get("/readyz")
    def readyz() -> dict[str, str]:
        current = state.get("analyzer")
        if current is None or not getattr(current, "ready", False):
            raise HTTPException(status_code=503, detail="model is not ready")
        return {"status": "ready"}

    @utility_router.get("/v1/encryption/public-key")
    def encryption_public_key() -> dict[str, str]:
        if not settings.encryption_enabled:
            raise HTTPException(status_code=404, detail="encrypted uploads are not enabled")
        try:
            public_key = read_key_b64(settings.encryption_public_key_path)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"alg": ENCRYPTION_ALGORITHM, "public_key": public_key}

    @router.post("/detect")
    async def detect(files: Annotated[list[UploadFile], File()]):
        return await run_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            files=files,
            options=AnalysisOptions(),
            embedding_format="float32_base64",
        )

    @router.post("/detect/encrypted")
    async def detect_encrypted(payload: EncryptedBatchPayload):
        return await run_encrypted_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            payload=payload,
            options=AnalysisOptions(),
            embedding_format="float32_base64",
        )

    @router.post("/align")
    async def align(
        files: Annotated[list[UploadFile], File()],
        include_landmarks106: bool = False,
        include_aligned_crop: bool = False,
    ):
        return await run_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            files=files,
            options=AnalysisOptions(
                need_alignment=True,
                include_landmarks106=include_landmarks106,
                include_aligned_crop=include_aligned_crop,
            ),
            embedding_format="float32_base64",
        )

    @router.post("/align/encrypted")
    async def align_encrypted(
        payload: EncryptedBatchPayload,
        include_landmarks106: bool = False,
        include_aligned_crop: bool = False,
    ):
        return await run_encrypted_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            payload=payload,
            options=AnalysisOptions(
                need_alignment=True,
                include_landmarks106=include_landmarks106,
                include_aligned_crop=include_aligned_crop,
            ),
            embedding_format="float32_base64",
        )

    @router.post("/quality")
    async def quality(
        files: Annotated[list[UploadFile], File()],
        include_landmarks106: bool = False,
    ):
        return await run_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            files=files,
            options=AnalysisOptions(
                need_alignment=True,
                need_quality=True,
                include_landmarks106=include_landmarks106,
            ),
            embedding_format="float32_base64",
        )

    @router.post("/quality/encrypted")
    async def quality_encrypted(
        payload: EncryptedBatchPayload,
        include_landmarks106: bool = False,
    ):
        return await run_encrypted_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            payload=payload,
            options=AnalysisOptions(
                need_alignment=True,
                need_quality=True,
                include_landmarks106=include_landmarks106,
            ),
            embedding_format="float32_base64",
        )

    @router.post("/embed")
    async def embed(
        files: Annotated[list[UploadFile], File()],
        include_landmarks106: bool = False,
        embedding_format: Annotated[EmbeddingFormat, Query()] = "float32_base64",
    ):
        if embedding_format == "json" and not settings.allow_debug_embedding_json:
            raise HTTPException(
                status_code=403,
                detail="json embedding output requires allow_debug_embedding_json=true",
            )
        return await run_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            files=files,
            options=AnalysisOptions(
                need_alignment=True,
                need_embedding=True,
                include_landmarks106=include_landmarks106,
            ),
            embedding_format=embedding_format,
        )

    @router.post("/embed/encrypted")
    async def embed_encrypted(
        payload: EncryptedBatchPayload,
        include_landmarks106: bool = False,
        embedding_format: Annotated[EmbeddingFormat, Query()] = "float32_base64",
    ):
        validate_embedding_format(settings, embedding_format)
        return await run_encrypted_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            payload=payload,
            options=AnalysisOptions(
                need_alignment=True,
                need_embedding=True,
                include_landmarks106=include_landmarks106,
            ),
            embedding_format=embedding_format,
        )

    @router.post("/analyze")
    async def analyze(
        files: Annotated[list[UploadFile], File()],
        include_landmarks106: bool = False,
        include_aligned_crop: bool = False,
        embedding_format: Annotated[EmbeddingFormat, Query()] = "float32_base64",
    ):
        if embedding_format == "json" and not settings.allow_debug_embedding_json:
            raise HTTPException(
                status_code=403,
                detail="json embedding output requires allow_debug_embedding_json=true",
            )
        return await run_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            files=files,
            options=AnalysisOptions(
                need_alignment=True,
                need_quality=True,
                need_embedding=True,
                include_landmarks106=include_landmarks106,
                include_aligned_crop=include_aligned_crop,
            ),
            embedding_format=embedding_format,
        )

    @router.post("/analyze/encrypted")
    async def analyze_encrypted(
        payload: EncryptedBatchPayload,
        include_landmarks106: bool = False,
        include_aligned_crop: bool = False,
        embedding_format: Annotated[EmbeddingFormat, Query()] = "float32_base64",
    ):
        validate_embedding_format(settings, embedding_format)
        return await run_encrypted_faces_request(
            settings=settings,
            analyzer=state.get("analyzer"),
            semaphore=gpu_semaphore,
            payload=payload,
            options=AnalysisOptions(
                need_alignment=True,
                need_quality=True,
                need_embedding=True,
                include_landmarks106=include_landmarks106,
                include_aligned_crop=include_aligned_crop,
            ),
            embedding_format=embedding_format,
        )

    app.include_router(utility_router)
    app.include_router(router)
    return app


async def run_faces_request(
    *,
    settings: Settings,
    analyzer: FaceAnalyzer | None,
    semaphore: asyncio.Semaphore,
    files: list[UploadFile],
    options: AnalysisOptions,
    embedding_format: EmbeddingFormat,
) -> dict[str, list[dict]]:
    if settings.require_encrypted_uploads:
        raise HTTPException(status_code=403, detail="encrypted uploads are required")
    if len(files) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail="batch exceeds max_batch_size")
    if analyzer is None or not getattr(analyzer, "ready", False):
        raise HTTPException(status_code=503, detail="model is not ready")

    image_items = []
    for file in files:
        if file.size is not None and file.size > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")
        image_bytes = await read_upload_limited(file, settings.max_image_bytes)
        image_items.append((file.filename or str(uuid4()), image_bytes))

    return await run_image_bytes_request(
        settings=settings,
        analyzer=analyzer,
        semaphore=semaphore,
        image_items=image_items,
        options=options,
        embedding_format=embedding_format,
    )


async def run_encrypted_faces_request(
    *,
    settings: Settings,
    analyzer: FaceAnalyzer | None,
    semaphore: asyncio.Semaphore,
    payload: EncryptedBatchPayload,
    options: AnalysisOptions,
    embedding_format: EmbeddingFormat,
) -> dict[str, list[dict]]:
    if len(payload.items) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail="batch exceeds max_batch_size")
    if analyzer is None or not getattr(analyzer, "ready", False):
        raise HTTPException(status_code=503, detail="model is not ready")

    private_key_b64 = get_encryption_private_key(settings)
    image_items: list[tuple[str, bytes]] = []
    response_key: bytes | None = None
    for item in payload.items:
        try:
            decrypted = decrypt_encrypted_image_payload_with_key(
                encrypted_payload_to_dict(item),
                private_key_b64,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if response_key is None:
            response_key = decrypted.aes_key
        image_bytes = decrypted.image_bytes
        if len(image_bytes) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")
        image_items.append((item.filename or str(uuid4()), image_bytes))

    response_payload = await run_image_bytes_request(
        settings=settings,
        analyzer=analyzer,
        semaphore=semaphore,
        image_items=image_items,
        options=options,
        embedding_format=embedding_format,
    )
    if response_key is None:
        raise HTTPException(
            status_code=400,
            detail="encrypted batch must contain at least one item",
        )
    return encrypt_response_for_client(
        aes_key=response_key,
        response_payload=response_payload,
        key_index=0,
    )


async def run_image_bytes_request(
    *,
    settings: Settings,
    analyzer: FaceAnalyzer,
    semaphore: asyncio.Semaphore,
    image_items: list[tuple[str, bytes]],
    options: AnalysisOptions,
    embedding_format: EmbeddingFormat,
) -> dict[str, list[dict]]:
    decoded_images = []
    for image_id, image_bytes in image_items:
        try:
            decoded = decode_image(
                image_bytes,
                image_id=image_id,
                max_pixels=settings.max_image_pixels,
            )
        except ImageTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        decoded_images.append(decoded)

    results: list[ImageAnalysisResult] = []
    try:
        async with semaphore:
            for decoded in decoded_images:
                result = await asyncio.to_thread(analyzer.analyze, decoded, options)
                results.append(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return {
            "results": [serialize_image_result(result, embedding_format) for result in results]
        }
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"invalid model output: {exc}") from exc


def validate_embedding_format(settings: Settings, embedding_format: EmbeddingFormat) -> None:
    if embedding_format == "json" and not settings.allow_debug_embedding_json:
        raise HTTPException(
            status_code=403,
            detail="json embedding output requires allow_debug_embedding_json=true",
        )


def get_encryption_private_key(settings: Settings) -> str:
    if not settings.encryption_enabled:
        raise HTTPException(status_code=404, detail="encrypted uploads are not enabled")
    try:
        return read_key_b64(settings.encryption_private_key_path)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def encrypted_payload_to_dict(payload: EncryptedImagePayload) -> dict[str, str]:
    return {
        "alg": payload.alg,
        "encrypted_key": payload.encrypted_key,
        "nonce": payload.nonce,
        "ciphertext": payload.ciphertext,
    }


def serialize_image_result(
    result: ImageAnalysisResult,
    embedding_format: EmbeddingFormat,
) -> dict:
    return {
        "image_id": result.image_id,
        "width": result.width,
        "height": result.height,
        "faces": [serialize_face(face, embedding_format) for face in result.faces],
        "warnings": result.warnings,
    }


def serialize_face(face: FaceObservation, embedding_format: EmbeddingFormat) -> dict:
    data = {
        "bbox_xyxy": face.bbox_xyxy,
        "det_score": face.det_score,
        "warnings": face.warnings,
    }
    if face.landmarks5 is not None:
        data["landmarks5"] = face.landmarks5
    if face.landmarks106 is not None:
        data["landmarks106"] = face.landmarks106
    if face.alignment is not None:
        data["alignment"] = serialize_alignment(face.alignment)
    if face.quality is not None:
        data["quality"] = serialize_quality(face.quality)
    if face.embedding is not None:
        normalized = l2_normalize(face.embedding.vector)
        data["embedding"] = {
            "model_id": face.embedding.model_id,
            "dim": int(normalized.shape[0]),
            "preprocess_id": face.embedding.preprocess_id,
            "normalized": True,
            "format": embedding_format,
            "vector": encode_float32_base64(normalized)
            if embedding_format == "float32_base64"
            else normalized.tolist(),
        }
    return data


def serialize_alignment(alignment: AlignmentResult) -> dict:
    data = {
        "image_size": alignment.image_size,
        "matrix": alignment.matrix,
        "landmarks5_crop": alignment.landmarks5_crop,
    }
    if alignment.aligned_crop_rgb_base64 is not None:
        data["aligned_crop_rgb_base64"] = alignment.aligned_crop_rgb_base64
    return data


def serialize_quality(quality: QualityResult) -> dict:
    return {
        "model_id": quality.model_id,
        "score": quality.score,
        "diagnostics": quality.diagnostics,
    }
