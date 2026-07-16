import asyncio
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from face_classifier_service.analysis import (
    AlignmentResult,
    AnalysisOptions,
    EmbeddingResult,
    FaceObservation,
    ImageAnalysisResult,
    QualityResult,
)
from face_classifier_service.api import create_app
from face_classifier_service.embedding import decode_float32_base64
from face_classifier_service.encryption import (
    ENCRYPTION_ALGORITHM,
    decrypt_response_from_server,
    encrypt_image_for_server,
    encrypt_image_for_server_with_key,
    generate_key_pair,
)
from face_classifier_service.settings import Settings


def make_png(size: tuple[int, int] = (8, 6)) -> bytes:
    image = Image.new("RGB", size, color=(255, 0, 0))
    data = BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


@dataclass
class FakeAnalyzer:
    ready: bool = True
    no_faces: bool = False
    embedding_vector: tuple[float, ...] = (3.0, 4.0)

    def analyze(self, image, options: AnalysisOptions) -> ImageAnalysisResult:
        assert image.image.mode == "RGB"
        assert image.width == 8
        assert image.height == 6
        if self.no_faces:
            return ImageAnalysisResult(
                image_id=image.image_id,
                width=image.width,
                height=image.height,
                faces=[],
            )

        face = FaceObservation(
            bbox_xyxy=[1.0, 2.0, 5.0, 6.0],
            det_score=0.91,
            landmarks5=[
                [2.0, 3.0],
                [4.0, 3.0],
                [3.0, 4.0],
                [2.4, 5.0],
                [3.8, 5.0],
            ]
            if options.need_alignment or options.need_embedding
            else None,
            landmarks106=[[0.0, 0.0]] * 106 if options.include_landmarks106 else None,
            alignment=AlignmentResult(
                image_size=[112, 112],
                matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                landmarks5_crop=[
                    [38.0, 52.0],
                    [74.0, 52.0],
                    [56.0, 72.0],
                    [42.0, 92.0],
                    [70.0, 92.0],
                ],
                aligned_crop_rgb_base64="ZmFrZS1jcm9w" if options.include_aligned_crop else None,
            )
            if options.need_alignment or options.need_embedding
            else None,
            quality=QualityResult(
                model_id="fdbtrs/CR-FIQA-L iresnet100",
                score=0.77,
                diagnostics={"blur": 0.12},
            )
            if options.need_quality
            else None,
            embedding=EmbeddingResult(
                model_id="minchul/cvlface_adaface_vit_base_kprpe_webface12m",
                preprocess_id="arcface_112_rgb_mean0.5_std0.5",
                vector=np.array(self.embedding_vector, dtype=np.float32),
            )
            if options.need_embedding
            else None,
            warnings=["profile_face"],
        )
        return ImageAnalysisResult(
            image_id=image.image_id,
            width=image.width,
            height=image.height,
            faces=[face],
            warnings=["low_light"],
        )


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer primary-token"}


@pytest.fixture
def encryption_key_paths(tmp_path):
    key_pair = generate_key_pair()
    public_key_path = tmp_path / "server_public.key"
    private_key_path = tmp_path / "server_private.key"
    public_key_path.write_text(key_pair.public_key_b64, encoding="ascii")
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    return key_pair, public_key_path, private_key_path


def test_all_face_endpoints_are_available_under_default_base_path() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    for endpoint in ["detect", "align", "quality", "embed", "analyze"]:
        response = client.post(
            f"/v1/faces/{endpoint}",
            files=[("files", ("phone.png", make_png(), "image/png"))],
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["image_id"] == "phone.png"


def test_api_base_path_prefixes_faces_routes_and_docs() -> None:
    app = create_app(Settings(api_base_path="/face-api"), analyzer=FakeAnalyzer())
    client = TestClient(app)

    assert client.post(
        "/v1/faces/analyze",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    ).status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404

    response = client.post(
        "/face-api/v1/faces/analyze",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    assert client.get("/face-api/docs").status_code == 200
    assert client.get("/face-api/openapi.json").status_code == 200


def test_auth_rejects_missing_token_on_public_routes(encryption_key_paths) -> None:
    key_pair, public_key_path, private_key_path = encryption_key_paths
    encrypted = encrypt_image_for_server(key_pair.public_key_b64, make_png())
    app = create_app(
        Settings(
            auth_enabled=True,
            api_tokens=("primary-token",),
            encryption_enabled=True,
            encryption_public_key_path=public_key_path,
            encryption_private_key_path=private_key_path,
        ),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)

    requests = [
        ("get", "/healthz", {}),
        ("get", "/readyz", {}),
        ("get", "/v1/encryption/public-key", {}),
        (
            "post",
            "/v1/faces/detect",
            {"files": [("files", ("phone.png", make_png(), "image/png"))]},
        ),
        ("post", "/v1/faces/analyze/encrypted", {"json": {"items": [encrypted]}}),
    ]

    for method, path, kwargs in requests:
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["detail"] == "missing bearer token"


def test_auth_rejects_invalid_token() -> None:
    app = create_app(
        Settings(auth_enabled=True, api_tokens=("primary-token",)),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)

    response = client.get("/healthz", headers={"Authorization": "Bearer wrong-token"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "invalid bearer token"


def test_auth_accepts_any_configured_token() -> None:
    app = create_app(
        Settings(auth_enabled=True, api_tokens=("primary-token", "secondary-token")),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)

    first = client.get("/healthz", headers={"Authorization": "Bearer primary-token"})
    second = client.get("/readyz", headers={"Authorization": "Bearer secondary-token"})

    assert first.status_code == 200
    assert second.status_code == 200


def test_auth_disables_docs_when_configured(auth_headers: dict[str, str]) -> None:
    app = create_app(
        Settings(
            auth_enabled=True,
            api_tokens=("primary-token",),
            auth_disable_docs=True,
        ),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)

    assert client.get("/docs", headers=auth_headers).status_code == 404
    assert client.get("/openapi.json", headers=auth_headers).status_code == 404


def test_public_key_endpoint_returns_encryption_metadata(encryption_key_paths) -> None:
    key_pair, public_key_path, private_key_path = encryption_key_paths
    app = create_app(
        Settings(
            encryption_enabled=True,
            encryption_public_key_path=public_key_path,
            encryption_private_key_path=private_key_path,
        ),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)

    response = client.get("/v1/encryption/public-key")

    assert response.status_code == 200
    assert response.json() == {
        "alg": ENCRYPTION_ALGORITHM,
        "public_key": key_pair.public_key_b64,
    }


def test_public_key_endpoint_returns_404_when_encryption_disabled() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.get("/v1/encryption/public-key")

    assert response.status_code == 404
    assert response.json()["detail"] == "encrypted uploads are not enabled"


@pytest.mark.parametrize("endpoint", ["detect", "align", "quality", "embed", "analyze"])
def test_encrypted_face_endpoints_accept_batch_payload(endpoint: str, encryption_key_paths) -> None:
    key_pair, _, private_key_path = encryption_key_paths
    app = create_app(
        Settings(encryption_enabled=True, encryption_private_key_path=private_key_path),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)
    encrypted = encrypt_image_for_server_with_key(key_pair.public_key_b64, make_png())

    response = client.post(
        f"/v1/faces/{endpoint}/encrypted",
        json={"items": [{"filename": "phone.png", **encrypted.payload}]},
    )

    assert response.status_code == 200
    encrypted_response = response.json()
    assert encrypted_response["alg"] == ENCRYPTION_ALGORITHM
    assert encrypted_response["key_index"] == 0
    assert "results" not in encrypted_response
    decrypted = decrypt_response_from_server(encrypted_response, encrypted.aes_key)
    assert decrypted["results"][0]["image_id"] == "phone.png"


def test_encrypted_embed_can_return_debug_json_embedding(encryption_key_paths) -> None:
    key_pair, _, private_key_path = encryption_key_paths
    app = create_app(
        Settings(
            encryption_enabled=True,
            encryption_private_key_path=private_key_path,
            allow_debug_embedding_json=True,
        ),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)
    encrypted = encrypt_image_for_server_with_key(key_pair.public_key_b64, make_png())

    response = client.post(
        "/v1/faces/embed/encrypted",
        params={"embedding_format": "json"},
        json={"items": [{"filename": "phone.png", **encrypted.payload}]},
    )

    assert response.status_code == 200
    decrypted = decrypt_response_from_server(response.json(), encrypted.aes_key)
    assert decrypted["results"][0]["faces"][0]["embedding"]["vector"] == [
        pytest.approx(0.6),
        pytest.approx(0.8),
    ]


def test_encrypted_endpoint_rejects_tampered_payload(encryption_key_paths) -> None:
    key_pair, _, private_key_path = encryption_key_paths
    app = create_app(
        Settings(encryption_enabled=True, encryption_private_key_path=private_key_path),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)
    encrypted = encrypt_image_for_server(key_pair.public_key_b64, make_png())
    encrypted["ciphertext"] = encrypted["ciphertext"][:-2] + "AA"

    response = client.post(
        "/v1/faces/analyze/encrypted",
        json={"items": [{"filename": "phone.png", **encrypted}]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid encrypted payload"


def test_require_encrypted_uploads_rejects_plaintext_faces(encryption_key_paths) -> None:
    _, public_key_path, private_key_path = encryption_key_paths
    app = create_app(
        Settings(
            encryption_enabled=True,
            require_encrypted_uploads=True,
            encryption_public_key_path=public_key_path,
            encryption_private_key_path=private_key_path,
        ),
        analyzer=FakeAnalyzer(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/faces/analyze",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "encrypted uploads are required"


def test_detect_returns_boxes_scores_and_original_image_size_only() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/detect",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["width"] == 8
    assert result["height"] == 6
    assert result["faces"][0]["bbox_xyxy"] == [1.0, 2.0, 5.0, 6.0]
    assert result["faces"][0]["det_score"] == 0.91
    assert "embedding" not in result["faces"][0]
    assert "quality" not in result["faces"][0]


def test_align_can_include_landmarks_matrix_and_aligned_crop() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/align",
        params={"include_aligned_crop": "true", "include_landmarks106": "true"},
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    face = response.json()["results"][0]["faces"][0]
    assert len(face["landmarks5"]) == 5
    assert len(face["landmarks106"]) == 106
    assert face["alignment"]["image_size"] == [112, 112]
    assert face["alignment"]["aligned_crop_rgb_base64"] == "ZmFrZS1jcm9w"


def test_quality_returns_quality_score_and_diagnostics() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/quality",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    quality = response.json()["results"][0]["faces"][0]["quality"]
    assert quality == {
        "model_id": "fdbtrs/CR-FIQA-L iresnet100",
        "score": 0.77,
        "diagnostics": {"blur": 0.12},
    }


def test_embed_returns_l2_normalized_float32_base64_embedding_by_default() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/embed",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    embedding = response.json()["results"][0]["faces"][0]["embedding"]
    assert embedding["model_id"] == "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
    assert embedding["dim"] == 2
    assert embedding["preprocess_id"] == "arcface_112_rgb_mean0.5_std0.5"
    assert embedding["normalized"] is True
    assert embedding["format"] == "float32_base64"
    np.testing.assert_allclose(decode_float32_base64(embedding["vector"]), [0.6, 0.8])


def test_embed_can_return_debug_json_embedding_vector() -> None:
    app = create_app(Settings(allow_debug_embedding_json=True), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/embed",
        params={"embedding_format": "json"},
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["faces"][0]["embedding"]["vector"] == [
        pytest.approx(0.6),
        pytest.approx(0.8),
    ]


def test_json_embedding_vector_is_rejected_when_debug_mode_is_disabled() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/embed",
        params={"embedding_format": "json"},
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 403


def test_analyze_combines_detection_alignment_quality_and_embedding() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/analyze",
        params={"include_landmarks106": "true"},
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    face = response.json()["results"][0]["faces"][0]
    assert face["bbox_xyxy"] == [1.0, 2.0, 5.0, 6.0]
    assert "alignment" in face
    assert "quality" in face
    assert "embedding" in face
    assert len(face["landmarks106"]) == 106
    assert response.json()["results"][0]["warnings"] == ["low_light"]


def test_no_faces_returns_200_with_empty_faces() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer(no_faces=True))
    client = TestClient(app)

    response = client.post(
        "/v1/faces/analyze",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["faces"] == []


def test_bad_image_returns_400() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/analyze",
        files=[("files", ("bad.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid image"


def test_oversized_image_returns_413() -> None:
    app = create_app(Settings(max_image_bytes=4), analyzer=FakeAnalyzer())
    client = TestClient(app)

    response = client.post(
        "/v1/faces/analyze",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 413


def test_model_not_ready_returns_503() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer(ready=False))
    client = TestClient(app)

    response = client.post(
        "/v1/faces/analyze",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "model is not ready"


def test_invalid_model_embedding_returns_503_not_500() -> None:
    app = create_app(Settings(), analyzer=FakeAnalyzer(embedding_vector=(0.0, 0.0)))
    client = TestClient(app)

    response = client.post(
        "/v1/faces/embed",
        files=[("files", ("phone.png", make_png(), "image/png"))],
    )

    assert response.status_code == 503
    assert "invalid model output" in response.json()["detail"]


def test_body_limit_rejects_stream_without_content_length() -> None:
    app = create_app(
        Settings(max_image_bytes=1, max_multipart_overhead_bytes=1, max_batch_size=1),
        analyzer=FakeAnalyzer(),
    )

    async def call_app() -> int:
        messages = []
        received = False
        body = (
            b"--BOUNDARY\r\n"
            b'Content-Disposition: form-data; name="files"; filename="phone.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
            b"too-large\r\n"
            b"--BOUNDARY--\r\n"
        )

        async def receive():
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/faces/analyze",
                "raw_path": b"/v1/faces/analyze",
                "query_string": b"",
                "headers": [(b"content-type", b"multipart/form-data; boundary=BOUNDARY")],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        return next(
            message["status"]
            for message in messages
            if message["type"] == "http.response.start"
        )

    assert asyncio.run(call_app()) == 413
