from __future__ import annotations

import base64
from collections.abc import Sequence

import numpy as np


def l2_normalize(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("cannot normalize zero-norm embedding")
    return (array / norm).astype(np.float32, copy=False)


def encode_float32_base64(vector: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(vector, dtype="<f4")
    return base64.b64encode(array.tobytes()).decode("ascii")


def decode_float32_base64(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"), validate=True)
    if len(raw) % 4:
        raise ValueError("float32_base64 payload length must be divisible by 4")
    return np.frombuffer(raw, dtype="<f4").copy()
