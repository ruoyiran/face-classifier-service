from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from face_classifier_service.models import ModelArtifactError, validate_model_artifacts
from face_classifier_service.preprocessing import DecodedImage
from face_classifier_service.settings import Settings


@dataclass(frozen=True)
class AnalysisOptions:
    need_alignment: bool = False
    need_quality: bool = False
    need_embedding: bool = False
    include_landmarks106: bool = False
    include_aligned_crop: bool = False


@dataclass(frozen=True)
class AlignmentResult:
    image_size: list[int]
    matrix: list[list[float]]
    landmarks5_crop: list[list[float]]
    aligned_crop_rgb_base64: str | None = None


@dataclass(frozen=True)
class QualityResult:
    model_id: str
    score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    model_id: str
    preprocess_id: str
    vector: np.ndarray | list[float]


@dataclass(frozen=True)
class FaceObservation:
    bbox_xyxy: list[float]
    det_score: float
    landmarks5: list[list[float]] | None = None
    landmarks106: list[list[float]] | None = None
    alignment: AlignmentResult | None = None
    quality: QualityResult | None = None
    embedding: EmbeddingResult | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageAnalysisResult:
    image_id: str
    width: int
    height: int
    faces: list[FaceObservation]
    warnings: list[str] = field(default_factory=list)


class FaceAnalyzer(Protocol):
    ready: bool

    def analyze(self, image: DecodedImage, options: AnalysisOptions) -> ImageAnalysisResult:
        ...


class UnavailableAnalyzer:
    ready = False

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def analyze(self, image: DecodedImage, options: AnalysisOptions) -> ImageAnalysisResult:
        raise RuntimeError(self.reason)


def create_default_analyzer(settings: Settings) -> FaceAnalyzer:
    try:
        validate_model_artifacts(settings.models_dir)
    except ModelArtifactError as exc:
        return UnavailableAnalyzer(str(exc))

    if settings.analyzer_factory:
        try:
            module_name, function_name = settings.analyzer_factory.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            factory = getattr(module, function_name)
            analyzer = factory(settings)
        except (ImportError, AttributeError, TypeError, ValueError, OSError, RuntimeError) as exc:
            return UnavailableAnalyzer(f"failed to load analyzer factory: {exc}")
        if not getattr(analyzer, "ready", False):
            return UnavailableAnalyzer("analyzer factory returned a model that is not ready")
        return analyzer

    return UnavailableAnalyzer(
        "native face model runner is not configured; set FACE_ANALYZER_FACTORY"
    )
