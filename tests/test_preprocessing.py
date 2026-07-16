from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from face_classifier_service.embedding import (
    decode_float32_base64,
    encode_float32_base64,
    l2_normalize,
)
from face_classifier_service.preprocessing import (
    ARCFACE_TEMPLATE_112,
    LANDMARK106_TO_5,
    ImageTooLargeError,
    arcface_affine_matrix,
    build_kprpe_input,
    decode_image,
    derive_landmarks5_from_106,
    non_max_suppression,
    normalize_rgb_for_arcface,
    scale_xyxy,
    transform_points,
)


def test_decode_image_applies_exif_orientation_and_converts_to_rgb() -> None:
    image = Image.new("L", (2, 3), color=128)
    exif = Image.Exif()
    exif[274] = 6
    data = BytesIO()
    image.save(data, format="JPEG", exif=exif)

    decoded = decode_image(data.getvalue(), image_id="rotated")

    assert decoded.image_id == "rotated"
    assert decoded.image.mode == "RGB"
    assert decoded.width == 3
    assert decoded.height == 2


def test_decode_image_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="invalid image"):
        decode_image(b"not an image", image_id="bad")


def test_decode_image_rejects_oversized_pixels_before_service_analysis() -> None:
    data = BytesIO()
    Image.new("RGB", (10, 10)).save(data, format="PNG")

    with pytest.raises(ImageTooLargeError, match="image exceeds max_image_pixels"):
        decode_image(data.getvalue(), image_id="large", max_pixels=99)


def test_scale_xyxy_scales_coordinates_between_image_sizes() -> None:
    assert scale_xyxy([10, 20, 30, 40], from_size=(100, 200), to_size=(50, 100)) == [
        5.0,
        10.0,
        15.0,
        20.0,
    ]


def test_non_max_suppression_keeps_highest_scoring_overlapping_box() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert non_max_suppression(boxes, scores, iou_threshold=0.5) == [0, 2]


def test_arcface_affine_matrix_maps_landmarks_to_fixed_template() -> None:
    source = np.array(ARCFACE_TEMPLATE_112, dtype=np.float32) + np.array([5.0, -3.0])

    matrix = arcface_affine_matrix(source)
    homogeneous = np.concatenate([source, np.ones((5, 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ matrix.T

    np.testing.assert_allclose(transformed, ARCFACE_TEMPLATE_112, atol=1e-4)


def test_arcface_affine_matrix_is_similarity_transform_without_shear() -> None:
    source = np.array(ARCFACE_TEMPLATE_112, dtype=np.float32)
    source = source + np.array(
        [
            [2.5, -1.0],
            [-1.0, 1.5],
            [0.5, 0.0],
            [-2.0, 1.0],
            [1.0, -1.5],
        ],
        dtype=np.float32,
    )

    matrix = arcface_affine_matrix(source)
    linear = matrix[:, :2]

    gram = linear.T @ linear
    scale_sq = float(np.trace(gram) / 2.0)
    np.testing.assert_allclose(gram, np.eye(2, dtype=np.float32) * scale_sq, atol=1e-5)
    assert np.linalg.det(linear) > 0.0


def test_transform_points_applies_affine_matrix() -> None:
    points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]], dtype=np.float32)

    transformed = transform_points(points, matrix)

    np.testing.assert_allclose(transformed, [[21.0, 58.0], [61.0, 118.0]])


def test_derive_landmarks5_from_106_uses_eye_means_and_anchor_points() -> None:
    landmarks106 = np.zeros((106, 2), dtype=np.float32)
    landmarks106[LANDMARK106_TO_5.left_eye] = np.array([[10.0, 20.0], [14.0, 24.0]])
    landmarks106[LANDMARK106_TO_5.right_eye] = np.array([[30.0, 20.0], [34.0, 24.0]])
    landmarks106[LANDMARK106_TO_5.nose] = np.array([22.0, 35.0])
    landmarks106[LANDMARK106_TO_5.left_mouth] = np.array([14.0, 50.0])
    landmarks106[LANDMARK106_TO_5.right_mouth] = np.array([32.0, 50.0])

    landmarks5 = derive_landmarks5_from_106(landmarks106)

    np.testing.assert_allclose(
        landmarks5,
        np.array(
            [
                [12.0, 22.0],
                [32.0, 22.0],
                [22.0, 35.0],
                [14.0, 50.0],
                [32.0, 50.0],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_rgb_for_arcface_returns_float32_minus_one_to_one() -> None:
    crop = np.array([[[0, 127, 255]]], dtype=np.uint8)

    normalized = normalize_rgb_for_arcface(crop)

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 1, 3)
    np.testing.assert_allclose(normalized[0, 0, 0], -1.0)
    np.testing.assert_allclose(normalized[0, 0, 2], 1.0)


def test_build_kprpe_input_contains_112_rgb_crop_and_crop_landmarks() -> None:
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    crop[..., 1] = 127
    crop[..., 2] = 255
    landmarks5_crop = np.array(ARCFACE_TEMPLATE_112, dtype=np.float32)

    kprpe_input = build_kprpe_input(crop, landmarks5_crop)

    assert kprpe_input.normalized_aligned_crop_rgb.shape == (112, 112, 3)
    assert kprpe_input.normalized_aligned_crop_rgb.dtype == np.float32
    np.testing.assert_allclose(
        kprpe_input.normalized_aligned_crop_rgb[0, 0],
        [-1.0, -0.00392157, 1.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(kprpe_input.landmarks5_crop, landmarks5_crop)


def test_l2_normalize_returns_unit_vector() -> None:
    normalized = l2_normalize([3.0, 4.0])

    np.testing.assert_allclose(normalized, np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(normalized), 1.0)


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize([0.0, 0.0])


def test_float32_base64_round_trips_little_endian_vectors() -> None:
    vector = np.array([0.6, 0.8], dtype=np.float32)

    encoded = encode_float32_base64(vector)

    np.testing.assert_allclose(decode_float32_base64(encoded), vector)
