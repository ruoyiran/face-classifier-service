from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    role: str
    model_id: str
    relative_path: Path


MODEL_MANIFEST = (
    ModelArtifact(
        role="detector",
        model_id="insightface/scrfd_34g",
        relative_path=Path("insightface/scrfd_34g.onnx"),
    ),
    ModelArtifact(
        role="landmark",
        model_id="insightface/2d106det",
        relative_path=Path("insightface/2d106det.onnx"),
    ),
    ModelArtifact(
        role="quality",
        model_id="fdbtrs/CR-FIQA-L iresnet100",
        relative_path=Path("fdbtrs/CR-FIQA-L/181952backbone.pth"),
    ),
    ModelArtifact(
        role="embedder",
        model_id="minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        relative_path=Path("minchul/cvlface_adaface_vit_base_kprpe_webface12m/model.safetensors"),
    ),
    ModelArtifact(
        role="embedder",
        model_id="minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        relative_path=Path("minchul/cvlface_adaface_vit_base_kprpe_webface12m/wrapper.py"),
    ),
    ModelArtifact(
        role="embedder",
        model_id="minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        relative_path=Path(
            "minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.yaml"
        ),
    ),
    ModelArtifact(
        role="embedder",
        model_id="minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        relative_path=Path(
            "minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.pt"
        ),
    ),
)


class ModelArtifactError(RuntimeError):
    pass


def validate_model_artifacts(models_dir: Path) -> None:
    problems: list[str] = []
    for artifact in MODEL_MANIFEST:
        path = models_dir / artifact.relative_path
        label = f"{artifact.role} {artifact.model_id}: {artifact.relative_path.as_posix()}"
        if not path.is_file():
            problems.append(label)
            continue
        if path.stat().st_size <= 0:
            problems.append(f"empty artifact {label}")
            continue
        sidecar = path.with_name(f"{path.name}.sha256")
        if not sidecar.is_file():
            problems.append(f"missing sha256 sidecar {sidecar.relative_to(models_dir).as_posix()}")
            continue
        expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
        actual = _sha256_file(path)
        if expected != actual:
            problems.append(f"sha256 mismatch {label}")
    if problems:
        details = "\n".join(f"- {item}" for item in problems)
        raise ModelArtifactError(f"invalid local model artifacts:\n{details}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
