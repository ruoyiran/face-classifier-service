from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from face_classifier_service.analysis import AnalysisOptions, EmbeddingResult, QualityResult
from face_classifier_service.pipeline import DetectedFace, FacePipelineAnalyzer
from face_classifier_service.preprocessing import (
    ARCFACE_TEMPLATE_112,
    LANDMARK106_TO_5,
    PREPROCESS_ID,
    DecodedImage,
)


def landmarks106_from_landmarks5(landmarks5: np.ndarray) -> np.ndarray:
    landmarks106 = np.zeros((106, 2), dtype=np.float32)
    landmarks106[LANDMARK106_TO_5.left_eye] = landmarks5[0]
    landmarks106[LANDMARK106_TO_5.right_eye] = landmarks5[1]
    landmarks106[LANDMARK106_TO_5.nose] = landmarks5[2]
    landmarks106[LANDMARK106_TO_5.left_mouth] = landmarks5[3]
    landmarks106[LANDMARK106_TO_5.right_mouth] = landmarks5[4]
    return landmarks106


@dataclass
class FakeDetector:
    faces: list[DetectedFace]

    def detect(self, image: DecodedImage) -> list[DetectedFace]:
        assert image.image.mode == "RGB"
        return self.faces


@dataclass
class FakeLandmarkModel:
    landmarks106: np.ndarray
    calls: int = 0

    def predict(self, image: DecodedImage, face: DetectedFace) -> np.ndarray:
        self.calls += 1
        assert face.det_score == 0.95
        return self.landmarks106


@dataclass
class FakeQualityModel:
    seen_crop_shape: tuple[int, ...] | None = None
    seen_dtype: np.dtype | None = None

    def score(self, normalized_aligned_crop_rgb: np.ndarray) -> QualityResult:
        self.seen_crop_shape = normalized_aligned_crop_rgb.shape
        self.seen_dtype = normalized_aligned_crop_rgb.dtype
        assert normalized_aligned_crop_rgb.min() >= -1.0
        assert normalized_aligned_crop_rgb.max() <= 1.0
        return QualityResult(
            model_id="fdbtrs/CR-FIQA-L iresnet100",
            score=0.81,
            diagnostics={"test": 1.0},
        )


@dataclass
class FakeEmbedder:
    seen_landmarks5_crop: np.ndarray | None = None
    seen_crop_shape: tuple[int, ...] | None = None
    seen_dtype: np.dtype | None = None

    def embed(
        self,
        normalized_aligned_crop_rgb: np.ndarray,
        landmarks5_crop: np.ndarray,
    ) -> EmbeddingResult:
        self.seen_crop_shape = normalized_aligned_crop_rgb.shape
        self.seen_dtype = normalized_aligned_crop_rgb.dtype
        assert normalized_aligned_crop_rgb.min() >= -1.0
        assert normalized_aligned_crop_rgb.max() <= 1.0
        self.seen_landmarks5_crop = landmarks5_crop
        return EmbeddingResult(
            model_id="minchul/cvlface_adaface_vit_base_kprpe_webface12m",
            preprocess_id=PREPROCESS_ID,
            vector=np.array([3.0, 4.0], dtype=np.float32),
        )


@dataclass
class Components:
    detector: FakeDetector
    landmark: FakeLandmarkModel
    quality: FakeQualityModel = field(default_factory=FakeQualityModel)
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)


def make_analyzer(components: Components) -> FacePipelineAnalyzer:
    return FacePipelineAnalyzer(
        detector=components.detector,
        landmark_model=components.landmark,
        quality_model=components.quality,
        embedder=components.embedder,
    )


def test_pipeline_returns_empty_faces_without_calling_landmark_model() -> None:
    source_landmarks5 = np.array(ARCFACE_TEMPLATE_112, dtype=np.float32)
    components = Components(
        detector=FakeDetector(faces=[]),
        landmark=FakeLandmarkModel(landmarks106_from_landmarks5(source_landmarks5)),
    )
    analyzer = make_analyzer(components)
    image = DecodedImage("empty", Image.new("RGB", (160, 160)), 160, 160)

    result = analyzer.analyze(image, AnalysisOptions(need_embedding=True))

    assert result.faces == []
    assert components.landmark.calls == 0


def test_pipeline_derives_landmarks_aligns_crop_and_passes_kprpe_inputs() -> None:
    source_landmarks5 = np.array(ARCFACE_TEMPLATE_112, dtype=np.float32) + np.array([12.0, -7.0])
    components = Components(
        detector=FakeDetector(faces=[DetectedFace([10.0, 20.0, 120.0, 140.0], 0.95)]),
        landmark=FakeLandmarkModel(landmarks106_from_landmarks5(source_landmarks5)),
    )
    analyzer = make_analyzer(components)
    image = DecodedImage("person", Image.new("RGB", (180, 180), color=(120, 80, 40)), 180, 180)

    result = analyzer.analyze(
        image,
        AnalysisOptions(
            need_alignment=True,
            need_quality=True,
            need_embedding=True,
            include_landmarks106=True,
            include_aligned_crop=True,
        ),
    )

    face = result.faces[0]
    assert face.bbox_xyxy == [10.0, 20.0, 120.0, 140.0]
    assert face.det_score == 0.95
    assert len(face.landmarks106) == 106
    assert face.alignment.image_size == [112, 112]
    assert face.alignment.aligned_crop_rgb_base64
    assert face.quality.score == 0.81
    assert face.embedding.preprocess_id == PREPROCESS_ID
    assert components.quality.seen_crop_shape == (112, 112, 3)
    assert components.quality.seen_dtype == np.float32
    assert components.embedder.seen_crop_shape == (112, 112, 3)
    assert components.embedder.seen_dtype == np.float32
    np.testing.assert_allclose(
        components.embedder.seen_landmarks5_crop,
        np.array(ARCFACE_TEMPLATE_112, dtype=np.float32),
        atol=1e-3,
    )
