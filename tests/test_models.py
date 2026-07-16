from hashlib import sha256
from pathlib import Path

import pytest

from face_classifier_service.models import (
    MODEL_MANIFEST,
    ModelArtifactError,
    validate_model_artifacts,
)


def test_validate_model_artifacts_accepts_complete_local_manifest(tmp_path: Path) -> None:
    for artifact in MODEL_MANIFEST:
        path = tmp_path / artifact.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"model"
        path.write_bytes(payload)
        path.with_name(f"{path.name}.sha256").write_text(
            sha256(payload).hexdigest(),
            encoding="ascii",
        )

    validate_model_artifacts(tmp_path)


def test_model_manifest_uses_official_cr_fiqa_l_pth_weight() -> None:
    quality_artifact = next(artifact for artifact in MODEL_MANIFEST if artifact.role == "quality")

    assert quality_artifact.model_id == "fdbtrs/CR-FIQA-L iresnet100"
    assert quality_artifact.relative_path == Path("fdbtrs/CR-FIQA-L/181952backbone.pth")


def test_model_manifest_covers_cvlface_runtime_artifacts() -> None:
    embedder_paths = {
        artifact.relative_path
        for artifact in MODEL_MANIFEST
        if artifact.role == "embedder"
    }

    root = Path("minchul/cvlface_adaface_vit_base_kprpe_webface12m")
    assert root / "model.safetensors" in embedder_paths
    assert root / "wrapper.py" in embedder_paths
    assert root / "pretrained_model" / "model.yaml" in embedder_paths
    assert root / "pretrained_model" / "model.pt" in embedder_paths


def test_validate_model_artifacts_reports_missing_local_models(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError) as exc_info:
        validate_model_artifacts(tmp_path)

    message = str(exc_info.value)
    assert "/models" not in message
    assert "scrfd_34g" in message
    assert "2d106det" in message
    assert "CR-FIQA-L" in message
    assert "adaface_vit_base_kprpe" in message


def test_validate_model_artifacts_requires_sha256_sidecar(tmp_path: Path) -> None:
    for artifact in MODEL_MANIFEST:
        path = tmp_path / artifact.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")

    with pytest.raises(ModelArtifactError) as exc_info:
        validate_model_artifacts(tmp_path)

    assert ".sha256" in str(exc_info.value)


def test_validate_model_artifacts_rejects_checksum_mismatch(tmp_path: Path) -> None:
    for artifact in MODEL_MANIFEST:
        path = tmp_path / artifact.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        path.with_name(f"{path.name}.sha256").write_text("0" * 64, encoding="ascii")

    with pytest.raises(ModelArtifactError) as exc_info:
        validate_model_artifacts(tmp_path)

    assert "sha256 mismatch" in str(exc_info.value)
