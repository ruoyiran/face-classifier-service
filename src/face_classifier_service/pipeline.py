from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from face_classifier_service.analysis import (
    AlignmentResult,
    AnalysisOptions,
    EmbeddingResult,
    FaceObservation,
    ImageAnalysisResult,
    QualityResult,
)
from face_classifier_service.preprocessing import (
    ALIGNED_CROP_SIZE,
    DecodedImage,
    aligned_crop_rgb_base64,
    arcface_affine_matrix,
    build_kprpe_input,
    derive_landmarks5_from_106,
    normalize_rgb_for_arcface,
    transform_points,
    warp_arcface_crop,
)


@dataclass(frozen=True)
class DetectedFace:
    bbox_xyxy: list[float]
    det_score: float


class Detector(Protocol):
    def detect(self, image: DecodedImage) -> list[DetectedFace]:
        ...


class LandmarkModel(Protocol):
    def predict(self, image: DecodedImage, face: DetectedFace) -> np.ndarray:
        ...


class QualityModel(Protocol):
    def score(self, normalized_aligned_crop_rgb: np.ndarray) -> QualityResult:
        ...


class Embedder(Protocol):
    def embed(
        self,
        normalized_aligned_crop_rgb: np.ndarray,
        landmarks5_crop: np.ndarray,
    ) -> EmbeddingResult:
        ...


class FacePipelineAnalyzer:
    ready = True

    def __init__(
        self,
        *,
        detector: Detector,
        landmark_model: LandmarkModel,
        quality_model: QualityModel,
        embedder: Embedder,
    ) -> None:
        self.detector = detector
        self.landmark_model = landmark_model
        self.quality_model = quality_model
        self.embedder = embedder

    def analyze(self, image: DecodedImage, options: AnalysisOptions) -> ImageAnalysisResult:
        observations = [
            self._analyze_face(image, detection, options)
            for detection in self.detector.detect(image)
        ]
        return ImageAnalysisResult(
            image_id=image.image_id,
            width=image.width,
            height=image.height,
            faces=observations,
        )

    def _analyze_face(
        self,
        image: DecodedImage,
        detection: DetectedFace,
        options: AnalysisOptions,
    ) -> FaceObservation:
        needs_landmarks = (
            options.need_alignment
            or options.need_quality
            or options.need_embedding
            or options.include_landmarks106
        )
        landmarks5 = None
        landmarks106 = None
        alignment = None
        quality = None
        embedding = None
        aligned_crop = None
        landmarks5_crop = None

        if needs_landmarks:
            landmarks106_array = self.landmark_model.predict(image, detection)
            landmarks5_array = derive_landmarks5_from_106(landmarks106_array)
            matrix = arcface_affine_matrix(landmarks5_array)
            landmarks5_crop_array = transform_points(landmarks5_array, matrix)
            landmarks5 = landmarks5_array.tolist()
            landmarks5_crop = landmarks5_crop_array
            if options.include_landmarks106:
                landmarks106 = np.asarray(landmarks106_array, dtype=np.float32).tolist()

            if options.need_alignment or options.need_quality or options.need_embedding:
                aligned_crop = warp_arcface_crop(image.image, matrix)
                alignment = AlignmentResult(
                    image_size=[ALIGNED_CROP_SIZE, ALIGNED_CROP_SIZE],
                    matrix=matrix.tolist(),
                    landmarks5_crop=landmarks5_crop_array.tolist(),
                    aligned_crop_rgb_base64=aligned_crop_rgb_base64(aligned_crop)
                    if options.include_aligned_crop
                    else None,
                )

        if options.need_quality:
            if aligned_crop is None:
                raise ValueError("aligned crop is required for quality scoring")
            quality = self.quality_model.score(normalize_rgb_for_arcface(aligned_crop))

        if options.need_embedding:
            if aligned_crop is None or landmarks5_crop is None:
                raise ValueError("aligned crop and crop landmarks are required for embedding")
            kprpe_input = build_kprpe_input(aligned_crop, landmarks5_crop)
            embedding = self.embedder.embed(
                kprpe_input.normalized_aligned_crop_rgb,
                kprpe_input.landmarks5_crop,
            )

        return FaceObservation(
            bbox_xyxy=detection.bbox_xyxy,
            det_score=detection.det_score,
            landmarks5=landmarks5,
            landmarks106=landmarks106,
            alignment=alignment,
            quality=quality,
            embedding=embedding,
        )
