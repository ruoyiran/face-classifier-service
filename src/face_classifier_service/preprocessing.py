from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional runtime format support
    register_heif_opener = None

if register_heif_opener is not None:  # pragma: no cover - depends on optional package
    register_heif_opener(thumbnails=False)


ALIGNED_CROP_SIZE = 112
PREPROCESS_ID = "arcface_112_similarity_rgb_mean0.5_std0.5"
ARCFACE_TEMPLATE_112 = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]


@dataclass(frozen=True)
class Landmark106To5:
    left_eye: list[int]
    right_eye: list[int]
    nose: int
    left_mouth: int
    right_mouth: int


LANDMARK106_TO_5 = Landmark106To5(
    left_eye=[35, 36],
    right_eye=[89, 90],
    nose=86,
    left_mouth=52,
    right_mouth=61,
)


@dataclass(frozen=True)
class DecodedImage:
    image_id: str
    image: Image.Image
    width: int
    height: int


@dataclass(frozen=True)
class KprpeInput:
    normalized_aligned_crop_rgb: np.ndarray
    landmarks5_crop: np.ndarray


class ImageTooLargeError(ValueError):
    pass


def decode_image(image_bytes: bytes, image_id: str, max_pixels: int | None = None) -> DecodedImage:
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            if max_pixels is not None and opened.width * opened.height > max_pixels:
                raise ImageTooLargeError("image exceeds max_image_pixels")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            if max_pixels is not None and image.width * image.height > max_pixels:
                raise ImageTooLargeError("image exceeds max_image_pixels")
    except ImageTooLargeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("invalid image") from exc
    return DecodedImage(
        image_id=image_id,
        image=image,
        width=image.width,
        height=image.height,
    )


def scale_xyxy(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> list[float]:
    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w <= 0 or from_h <= 0:
        raise ValueError("from_size dimensions must be positive")
    sx = to_w / from_w
    sy = to_h / from_h
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)]


def non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
) -> list[int]:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return []
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if score_array.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have the same length as boxes")

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = score_array.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= iou_threshold]

    return keep


def arcface_affine_matrix(
    landmarks5: list[list[float]] | np.ndarray,
    template: list[list[float]] | np.ndarray = ARCFACE_TEMPLATE_112,
) -> np.ndarray:
    source = np.asarray(landmarks5, dtype=np.float32)
    target = np.asarray(template, dtype=np.float32)
    if source.shape != (5, 2):
        raise ValueError("landmarks5 must have shape (5, 2)")
    if target.shape != (5, 2):
        raise ValueError("template must have shape (5, 2)")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= np.finfo(np.float32).eps:
        raise ValueError("landmarks5 points must not be degenerate")

    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    no_reflection = np.ones(2, dtype=np.float32)
    if np.linalg.det(u @ vt) < 0.0:
        no_reflection[-1] = -1.0

    rotation = u @ np.diag(no_reflection) @ vt
    scale = float(np.sum(singular_values * no_reflection) / source_variance)
    linear = (scale * rotation).astype(np.float32)
    translation = target_mean - linear @ source_mean

    matrix = np.empty((2, 3), dtype=np.float32)
    matrix[:, :2] = linear
    matrix[:, 2] = translation.astype(np.float32)
    return matrix


def transform_points(points: list[list[float]] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float32)
    affine = np.asarray(matrix, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if affine.shape != (2, 3):
        raise ValueError("matrix must have shape (2, 3)")
    homogeneous = np.concatenate(
        [point_array, np.ones((point_array.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (homogeneous @ affine.T).astype(np.float32)


def warp_arcface_crop(image: Image.Image, matrix: np.ndarray) -> np.ndarray:
    affine = np.asarray(matrix, dtype=np.float32)
    if affine.shape != (2, 3):
        raise ValueError("matrix must have shape (2, 3)")
    forward = np.eye(3, dtype=np.float32)
    forward[:2, :] = affine
    inverse = np.linalg.inv(forward)
    transform = tuple(float(value) for value in inverse[:2, :].reshape(-1))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    crop = image.convert("RGB").transform(
        (ALIGNED_CROP_SIZE, ALIGNED_CROP_SIZE),
        Image.Transform.AFFINE,
        transform,
        resample=resampling,
    )
    return np.asarray(crop, dtype=np.uint8)


def aligned_crop_rgb_base64(aligned_crop_rgb: np.ndarray) -> str:
    crop = np.asarray(aligned_crop_rgb)
    if crop.shape != (ALIGNED_CROP_SIZE, ALIGNED_CROP_SIZE, 3) or crop.dtype != np.uint8:
        raise ValueError("aligned_crop_rgb must be 112x112 RGB uint8")
    return base64.b64encode(crop.tobytes()).decode("ascii")


def build_kprpe_input(
    aligned_crop_rgb: np.ndarray,
    landmarks5_crop: list[list[float]] | np.ndarray,
) -> KprpeInput:
    crop = np.asarray(aligned_crop_rgb)
    points = np.asarray(landmarks5_crop, dtype=np.float32)
    if crop.shape != (ALIGNED_CROP_SIZE, ALIGNED_CROP_SIZE, 3):
        raise ValueError("aligned_crop_rgb must have shape (112, 112, 3)")
    if crop.dtype != np.uint8:
        raise ValueError("aligned_crop_rgb must be uint8")
    if points.shape != (5, 2):
        raise ValueError("landmarks5_crop must have shape (5, 2)")
    return KprpeInput(
        normalized_aligned_crop_rgb=normalize_rgb_for_arcface(crop),
        landmarks5_crop=points,
    )


def derive_landmarks5_from_106(landmarks106: list[list[float]] | np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks106, dtype=np.float32)
    if points.shape != (106, 2):
        raise ValueError("landmarks106 must have shape (106, 2)")
    return np.array(
        [
            points[LANDMARK106_TO_5.left_eye].mean(axis=0),
            points[LANDMARK106_TO_5.right_eye].mean(axis=0),
            points[LANDMARK106_TO_5.nose],
            points[LANDMARK106_TO_5.left_mouth],
            points[LANDMARK106_TO_5.right_mouth],
        ],
        dtype=np.float32,
    )


def normalize_rgb_for_arcface(crop_rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(crop_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("crop_rgb must have shape (H, W, 3)")
    if array.dtype != np.uint8:
        raise ValueError("crop_rgb must be uint8")
    return (array.astype(np.float32) / 255.0 - 0.5) / 0.5
