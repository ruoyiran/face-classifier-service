import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import face_classifier_service.native_analyzer as native_analyzer
from face_classifier_service.analysis import QualityResult
from face_classifier_service.native_analyzer import (
    NativeCvlFaceEmbedder,
    NativeLandmark106Model,
    NativeScrfdDetector,
    _prepare_onnxruntime_cuda_dlls,
    create_analyzer,
    onnx_providers_for_device,
)
from face_classifier_service.pipeline import FacePipelineAnalyzer
from face_classifier_service.preprocessing import DecodedImage
from face_classifier_service.settings import Settings


class FakeIo:
    def __init__(self, name: str, shape: list[int | str], kind: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = kind


class FakeScrfdSession:
    def get_inputs(self):
        return [FakeIo("input.1", [1, 3, 640, 640])]

    def get_outputs(self):
        return [
            FakeIo("score_8", [12800, 1]),
            FakeIo("score_16", [3200, 1]),
            FakeIo("score_32", [800, 1]),
            FakeIo("bbox_8", [12800, 4]),
            FakeIo("bbox_16", [3200, 4]),
            FakeIo("bbox_32", [800, 4]),
            FakeIo("kps_8", [12800, 10]),
            FakeIo("kps_16", [3200, 10]),
            FakeIo("kps_32", [800, 10]),
        ]

    def run(self, output_names, feeds):
        scores = [
            np.zeros((12800, 1), dtype=np.float32),
            np.zeros((3200, 1), dtype=np.float32),
            np.zeros((800, 1), dtype=np.float32),
        ]
        scores[0][0, 0] = 0.95
        boxes = [
            np.zeros((12800, 4), dtype=np.float32),
            np.zeros((3200, 4), dtype=np.float32),
            np.zeros((800, 4), dtype=np.float32),
        ]
        boxes[0][0] = [1.0, 1.0, 2.0, 2.0]
        kps = [
            np.zeros((12800, 10), dtype=np.float32),
            np.zeros((3200, 10), dtype=np.float32),
            np.zeros((800, 10), dtype=np.float32),
        ]
        return [*scores, *boxes, *kps]


class FakeLandmarkSession:
    def get_inputs(self):
        return [FakeIo("data", ["None", 3, 192, 192])]

    def get_outputs(self):
        return [FakeIo("fc1", [1, 212])]

    def run(self, output_names, feeds):
        return [np.zeros((1, 212), dtype=np.float32)]


class FakeTorchTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def unsqueeze(self, dim):
        return FakeTorchTensor(np.expand_dims(self.array, dim))

    def to(self, device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorchModule:
    @staticmethod
    def from_numpy(array):
        return FakeTorchTensor(array)

    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


class FakeZeroEmbedderModel:
    def __call__(self, tensor, keypoints):
        return FakeTorchTensor(np.zeros((1, 512), dtype=np.float32))


def decoded_image(size: tuple[int, int] = (64, 64)) -> DecodedImage:
    image = Image.new("RGB", size, color=(120, 100, 80))
    return DecodedImage(image_id="test.png", image=image, width=size[0], height=size[1])


def test_onnx_providers_for_device_prefers_cuda_then_cpu() -> None:
    assert onnx_providers_for_device("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert onnx_providers_for_device("cpu") == ["CPUExecutionProvider"]


def test_prepare_onnxruntime_cuda_dlls_adds_torch_lib(
    monkeypatch,
    tmp_path: Path,
) -> None:
    torch_package = tmp_path / "torch"
    torch_lib = torch_package / "lib"
    torch_lib.mkdir(parents=True)
    (torch_package / "__init__.py").write_text("", encoding="utf-8")
    fake_torch = type("FakeTorch", (), {"__file__": str(torch_package / "__init__.py")})()
    added: list[Path] = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        native_analyzer.os,
        "add_dll_directory",
        lambda path: added.append(Path(path)),
    )
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    native_analyzer._CUDA_DLL_DIRECTORY_HANDLES.clear()
    native_analyzer._CUDA_DLL_DIRECTORY_PATHS.clear()

    _prepare_onnxruntime_cuda_dlls(["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert added == [torch_lib.resolve()]


def test_create_analyzer_wires_pipeline_with_deployment_model_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: dict[str, Path] = {}

    class FakeDetector:
        def __init__(self, model_path, providers):
            created["detector"] = Path(model_path)

    class FakeLandmark:
        def __init__(self, model_path, providers):
            created["landmark"] = Path(model_path)

    class FakeQuality:
        def __init__(self, model_path, device):
            created["quality"] = Path(model_path)

        def score(self, normalized_aligned_crop_rgb):
            return QualityResult(model_id="quality", score=0.0)

    class FakeEmbedder:
        def __init__(self, model_dir, device):
            created["embedder"] = Path(model_dir)

    monkeypatch.setattr(
        "face_classifier_service.native_analyzer.NativeScrfdDetector",
        FakeDetector,
    )
    monkeypatch.setattr(
        "face_classifier_service.native_analyzer.NativeLandmark106Model",
        FakeLandmark,
    )
    monkeypatch.setattr(
        "face_classifier_service.native_analyzer.NativeCrFiqaQualityModel",
        FakeQuality,
    )
    monkeypatch.setattr(
        "face_classifier_service.native_analyzer.NativeCvlFaceEmbedder",
        FakeEmbedder,
    )

    analyzer = create_analyzer(Settings(models_dir=tmp_path, device="cuda"))

    assert isinstance(analyzer, FacePipelineAnalyzer)
    assert created == {
        "detector": tmp_path / "insightface" / "scrfd_34g.onnx",
        "landmark": tmp_path / "insightface" / "2d106det.onnx",
        "quality": tmp_path / "fdbtrs" / "CR-FIQA-L" / "181952backbone.pth",
        "embedder": tmp_path / "minchul" / "cvlface_adaface_vit_base_kprpe_webface12m",
    }


def test_native_scrfd_detector_decodes_onnx_outputs_to_original_coordinates() -> None:
    detector = NativeScrfdDetector(
        model_path=Path("scrfd.onnx"),
        providers=["CPUExecutionProvider"],
        session=FakeScrfdSession(),
        det_threshold=0.5,
    )

    faces = detector.detect(decoded_image())

    assert len(faces) == 1
    assert faces[0].det_score == 0.95
    assert faces[0].bbox_xyxy == [-0.8, -0.8, 1.6, 1.6]


def test_native_landmark_model_maps_192_crop_points_back_to_original_image() -> None:
    model = NativeLandmark106Model(
        model_path=Path("2d106det.onnx"),
        providers=["CPUExecutionProvider"],
        session=FakeLandmarkSession(),
    )
    face = type("Face", (), {"bbox_xyxy": [16.0, 16.0, 48.0, 48.0]})()

    landmarks = model.predict(decoded_image(), face)

    assert landmarks.shape == (106, 2)
    np.testing.assert_allclose(landmarks[0], [32.0, 32.0], atol=1e-5)


def test_native_cvlface_embedder_rejects_zero_norm_embedding() -> None:
    embedder = object.__new__(NativeCvlFaceEmbedder)
    embedder.torch = FakeTorchModule()
    embedder.device = "cpu"
    embedder.model = FakeZeroEmbedderModel()

    with pytest.raises(RuntimeError, match="zero-norm embedding"):
        embedder.embed(
            np.zeros((112, 112, 3), dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
        )
