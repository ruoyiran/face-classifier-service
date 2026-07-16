from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from face_classifier_service.analysis import EmbeddingResult, QualityResult
from face_classifier_service.pipeline import DetectedFace, FacePipelineAnalyzer
from face_classifier_service.preprocessing import (
    ALIGNED_CROP_SIZE,
    PREPROCESS_ID,
    DecodedImage,
    non_max_suppression,
)
from face_classifier_service.settings import Settings

DETECTOR_MODEL_ID = "insightface/scrfd_34g"
QUALITY_MODEL_ID = "fdbtrs/CR-FIQA-L iresnet100"
EMBEDDER_MODEL_ID = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"

SCRFD_INPUT_SIZE = 640
SCRFD_STRIDES = (8, 16, 32)
LANDMARK_INPUT_SIZE = 192

_CUDA_DLL_DIRECTORY_HANDLES: list[Any] = []
_CUDA_DLL_DIRECTORY_PATHS: set[str] = set()


def onnx_providers_for_device(device: str) -> list[str]:
    normalized = device.strip().lower()
    if normalized in {"cuda", "gpu", "cuda:0"} or normalized.startswith("cuda:"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_analyzer(settings: Settings) -> FacePipelineAnalyzer:
    models_dir = Path(settings.models_dir)
    providers = onnx_providers_for_device(settings.device)
    return FacePipelineAnalyzer(
        detector=NativeScrfdDetector(
            model_path=models_dir / "insightface" / "scrfd_34g.onnx",
            providers=providers,
        ),
        landmark_model=NativeLandmark106Model(
            model_path=models_dir / "insightface" / "2d106det.onnx",
            providers=providers,
        ),
        quality_model=NativeCrFiqaQualityModel(
            model_path=models_dir / "fdbtrs" / "CR-FIQA-L" / "181952backbone.pth",
            device=settings.device,
        ),
        embedder=NativeCvlFaceEmbedder(
            model_dir=models_dir / "minchul" / "cvlface_adaface_vit_base_kprpe_webface12m",
            device=settings.device,
        ),
    )


class NativeScrfdDetector:
    def __init__(
        self,
        model_path: Path,
        providers: list[str],
        *,
        session: Any | None = None,
        det_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_faces: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.max_faces = max_faces
        self.session = session or _create_onnx_session(self.model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(self, image: DecodedImage) -> list[DetectedFace]:
        blob, det_scale = _scrfd_blob(image.image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        output_by_name = dict(zip(self.output_names, outputs, strict=True))
        boxes_by_stride: list[np.ndarray] = []
        scores_by_stride: list[np.ndarray] = []

        for stride in SCRFD_STRIDES:
            scores = output_by_name[f"score_{stride}"].reshape(-1).astype(np.float32)
            bbox_distances = output_by_name[f"bbox_{stride}"].reshape(-1, 4).astype(np.float32)
            positive = np.where(scores >= self.det_threshold)[0]
            if positive.size == 0:
                continue

            feature_h = SCRFD_INPUT_SIZE // stride
            feature_w = SCRFD_INPUT_SIZE // stride
            anchor_count = scores.shape[0] // (feature_h * feature_w)
            centers = _scrfd_anchor_centers(feature_h, feature_w, stride, anchor_count)
            boxes = _distance_to_bbox(
                centers[positive],
                bbox_distances[positive] * float(stride),
            )
            boxes_by_stride.append(boxes / det_scale)
            scores_by_stride.append(scores[positive])

        if not boxes_by_stride:
            return []

        boxes = np.vstack(boxes_by_stride).astype(np.float32)
        scores = np.concatenate(scores_by_stride).astype(np.float32)
        keep = non_max_suppression(boxes, scores, iou_threshold=self.nms_threshold)
        if self.max_faces > 0:
            keep = keep[: self.max_faces]
        return [
            DetectedFace(
                bbox_xyxy=[round(float(value), 6) for value in boxes[index].tolist()],
                det_score=round(float(scores[index]), 6),
            )
            for index in keep
        ]


class NativeLandmark106Model:
    def __init__(
        self,
        model_path: Path,
        providers: list[str],
        *,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.session = session or _create_onnx_session(self.model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, image: DecodedImage, face: DetectedFace) -> np.ndarray:
        matrix = _landmark_crop_matrix(face.bbox_xyxy)
        crop = _warp_rgb_with_matrix(image.image, matrix, LANDMARK_INPUT_SIZE)
        blob = _nchw_normalized(crop, mean=127.5, std=128.0)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        landmarks = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 2)
        if landmarks.shape[0] != 106:
            raise RuntimeError(f"2d106det returned {landmarks.shape[0]} landmarks, expected 106")

        landmarks = (landmarks + 1.0) * (LANDMARK_INPUT_SIZE / 2.0)
        inverse = _invert_affine(matrix)
        return _transform_points(landmarks, inverse).astype(np.float32)


class NativeCrFiqaQualityModel:
    def __init__(self, model_path: Path, device: str) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"CR-FIQA model does not exist: {self.model_path}")
        import torch

        self.torch = torch
        self.device = _torch_device(torch, device)
        self.model = _build_cr_fiqa_iresnet100(torch)
        state_dict = _torch_load_state_dict(torch, self.model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def score(self, normalized_aligned_crop_rgb: np.ndarray) -> QualityResult:
        crop = np.asarray(normalized_aligned_crop_rgb, dtype=np.float32)
        if crop.shape != (ALIGNED_CROP_SIZE, ALIGNED_CROP_SIZE, 3):
            raise ValueError("CR-FIQA input must be 112x112 RGB normalized crop")
        tensor = self.torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            _features, quality = self.model(tensor)
        return QualityResult(
            model_id=QUALITY_MODEL_ID,
            score=float(quality.reshape(-1)[0].detach().cpu().item()),
            diagnostics={
                "score_type": "raw_cr_fiqa_qs",
                "preprocess_id": PREPROCESS_ID,
            },
        )


class NativeCvlFaceEmbedder:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"CVLFace model directory does not exist: {self.model_dir}")
        model_pt = self.model_dir / "pretrained_model" / "model.pt"
        if not model_pt.is_file():
            raise FileNotFoundError(f"CVLFace model weights do not exist: {model_pt}")
        import torch

        self.torch = torch
        self.device = _torch_device(torch, device)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def embed(
        self,
        normalized_aligned_crop_rgb: np.ndarray,
        landmarks5_crop: np.ndarray,
    ) -> EmbeddingResult:
        crop = np.asarray(normalized_aligned_crop_rgb, dtype=np.float32)
        points = np.asarray(landmarks5_crop, dtype=np.float32)
        if crop.shape != (ALIGNED_CROP_SIZE, ALIGNED_CROP_SIZE, 3):
            raise ValueError("CVLFace input must be 112x112 RGB normalized crop")
        if points.shape != (5, 2):
            raise ValueError("CVLFace KPRPE landmarks must have shape (5, 2)")

        tensor = self.torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(self.device)
        keypoints = self.torch.from_numpy(points / float(ALIGNED_CROP_SIZE))
        keypoints = keypoints.unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor, keypoints)
        if isinstance(output, (tuple, list)):
            output = output[0]
        vector = output.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise RuntimeError("CVLFace returned zero-norm embedding")
        vector = vector / norm
        return EmbeddingResult(
            model_id=EMBEDDER_MODEL_ID,
            preprocess_id=PREPROCESS_ID,
            vector=vector,
        )

    def _load_model(self):
        _install_kprpe_python_fallback()
        module_name = "_face_classifier_service_cvlface_wrapper"
        wrapper_path = self.model_dir / "wrapper.py"
        if not wrapper_path.is_file():
            raise FileNotFoundError(f"CVLFace wrapper does not exist: {wrapper_path}")
        spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load CVLFace wrapper: {wrapper_path}")

        old_cwd = os.getcwd()
        sys.path.insert(0, str(self.model_dir))
        try:
            os.chdir(self.model_dir)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            config = module.ModelConfig()
            return module.CVLFaceRecognitionModel(config)
        finally:
            os.chdir(old_cwd)
            try:
                sys.path.remove(str(self.model_dir))
            except ValueError:  # pragma: no cover - defensive cleanup
                pass


def _create_onnx_session(model_path: Path, providers: list[str]):
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    _prepare_onnxruntime_cuda_dlls(providers)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def _prepare_onnxruntime_cuda_dlls(providers: list[str]) -> None:
    if "CUDAExecutionProvider" not in providers:
        return
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates: list[Path] = []
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            candidates.append(torch_lib)

    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        cuda_bin = Path(cuda_path) / "bin"
        if cuda_bin.is_dir():
            candidates.append(cuda_bin)

    for candidate in candidates:
        _add_windows_dll_directory(candidate)


def _add_windows_dll_directory(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _CUDA_DLL_DIRECTORY_PATHS:
        return
    handle = os.add_dll_directory(resolved)
    _CUDA_DLL_DIRECTORY_HANDLES.append(handle)
    _CUDA_DLL_DIRECTORY_PATHS.add(resolved)


def _scrfd_blob(image: Image.Image) -> tuple[np.ndarray, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if height > width:
        resized_h = SCRFD_INPUT_SIZE
        resized_w = int(round(SCRFD_INPUT_SIZE * width / height))
    else:
        resized_w = SCRFD_INPUT_SIZE
        resized_h = int(round(SCRFD_INPUT_SIZE * height / width))
    det_scale = resized_h / float(height)

    resized = np.asarray(
        image.convert("RGB").resize((resized_w, resized_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.zeros((SCRFD_INPUT_SIZE, SCRFD_INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:resized_h, :resized_w] = resized
    return _nchw_normalized(canvas, mean=127.5, std=128.0), det_scale


def _nchw_normalized(rgb: np.ndarray, *, mean: float, std: float) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32)
    normalized = (array - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def _scrfd_anchor_centers(
    feature_h: int,
    feature_w: int,
    stride: int,
    anchor_count: int,
) -> np.ndarray:
    grid_y, grid_x = np.mgrid[:feature_h, :feature_w]
    centers = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
    centers = (centers * float(stride)).reshape(-1, 2)
    if anchor_count > 1:
        centers = np.repeat(centers, anchor_count, axis=0)
    return centers


def _distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ],
        axis=1,
    ).astype(np.float32)


def _landmark_crop_matrix(bbox_xyxy: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    width = x2 - x1
    height = y2 - y1
    box_size = max(width, height)
    if box_size <= 0.0:
        raise ValueError("face bbox must have positive area")
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    scale = LANDMARK_INPUT_SIZE / (box_size * 1.5)
    return np.array(
        [
            [scale, 0.0, LANDMARK_INPUT_SIZE * 0.5 - center_x * scale],
            [0.0, scale, LANDMARK_INPUT_SIZE * 0.5 - center_y * scale],
        ],
        dtype=np.float32,
    )


def _warp_rgb_with_matrix(image: Image.Image, matrix: np.ndarray, size: int) -> np.ndarray:
    inverse = _invert_affine(matrix)
    transform = tuple(float(value) for value in inverse.reshape(-1))
    crop = image.convert("RGB").transform(
        (size, size),
        Image.Transform.AFFINE,
        transform,
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(crop, dtype=np.uint8)


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    affine = np.asarray(matrix, dtype=np.float32)
    if affine.shape != (2, 3):
        raise ValueError("affine matrix must have shape (2, 3)")
    homogeneous = np.eye(3, dtype=np.float32)
    homogeneous[:2, :] = affine
    return np.linalg.inv(homogeneous)[:2, :].astype(np.float32)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float32)
    ones = np.ones((point_array.shape[0], 1), dtype=np.float32)
    return np.concatenate([point_array, ones], axis=1) @ matrix.T


def _torch_device(torch, device: str):
    requested = device.strip().lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _torch_load_state_dict(torch, path: Path) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch fallback
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid torch checkpoint: {path}")
    return state


def _install_kprpe_python_fallback() -> None:
    module_name = "models.vit_kprpe.RPE.rpe_ops.rpe_index"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.RPEIndexFunction = None
    sys.modules[module_name] = module


def _build_cr_fiqa_iresnet100(torch):
    nn = torch.nn

    def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class IBasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Any | None = None,
        ) -> None:
            super().__init__()
            self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
            self.conv1 = conv3x3(inplanes, planes)
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
            self.prelu = nn.PReLU(planes)
            self.conv2 = conv3x3(planes, planes, stride)
            self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.bn1(x)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.prelu(out)
            out = self.conv2(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            return out + identity

    class IResNet(nn.Module):
        def __init__(self, layers: list[int], num_features: int = 512, qs: int = 1) -> None:
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
            self.prelu = nn.PReLU(64)
            self.layer1 = self._make_layer(64, layers[0], stride=2)
            self.layer2 = self._make_layer(128, layers[1], stride=2)
            self.layer3 = self._make_layer(256, layers[2], stride=2)
            self.layer4 = self._make_layer(512, layers[3], stride=2)
            self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
            self.dropout = nn.Dropout(p=0.0, inplace=True)
            self.fc = nn.Linear(512 * 7 * 7, num_features)
            self.features = nn.BatchNorm1d(num_features, eps=1e-5)
            self.qs = nn.Linear(num_features, qs)

        def _make_layer(self, planes: int, blocks: int, stride: int = 1):
            downsample = None
            if stride != 1 or self.inplanes != planes * IBasicBlock.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * IBasicBlock.expansion, stride),
                    nn.BatchNorm2d(planes * IBasicBlock.expansion, eps=1e-5),
                )
            layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * IBasicBlock.expansion
            for _ in range(1, blocks):
                layers.append(IBasicBlock(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = self.fc(x.float())
            features = self.features(x)
            quality = self.qs(features)
            return features, quality

    return IResNet([3, 13, 30, 3], num_features=512, qs=1)


__all__ = [
    "NativeCrFiqaQualityModel",
    "NativeCvlFaceEmbedder",
    "NativeLandmark106Model",
    "NativeScrfdDetector",
    "create_analyzer",
    "onnx_providers_for_device",
]
