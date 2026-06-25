from __future__ import annotations

import numpy as np


def _as_uint8_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.copy()
    padded = np.pad(image.astype(np.float32), radius, mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return np.clip(out / (size * size), 0, 255).astype(np.uint8)


def _dilate_dark_lines(image: np.ndarray, iterations: int) -> np.ndarray:
    out = image.copy()
    for _ in range(max(iterations, 0)):
        padded = np.pad(out, 1, mode="edge")
        windows = [
            padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        out = np.minimum.reduce(windows).astype(np.uint8)
    return out


def _erode_dark_lines(image: np.ndarray, iterations: int) -> np.ndarray:
    out = image.copy()
    for _ in range(max(iterations, 0)):
        padded = np.pad(out, 1, mode="edge")
        windows = [
            padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        out = np.maximum.reduce(windows).astype(np.uint8)
    return out


def apply_sketch_perturbation(
    image: np.ndarray,
    mode: str,
    severity: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Apply deterministic sketch-domain perturbations for robustness tests."""
    arr = _as_uint8_gray(image)
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    if mode == "thickness":
        iterations = max(1, int(round(1 + 3 * severity)))
        return _dilate_dark_lines(arr, iterations)

    if mode == "dropout":
        line_mask = arr < 128
        drop_prob = 0.05 + 0.45 * severity
        keep_mask = rng.random(arr.shape) > drop_prob
        out = arr.copy()
        out[line_mask & ~keep_mask] = 255
        return out.astype(np.uint8)

    if mode == "noise":
        out = arr.astype(np.int16)
        stroke_prob = 0.003 + 0.025 * severity
        black_noise = rng.random(arr.shape) < stroke_prob
        white_noise = rng.random(arr.shape) < (stroke_prob * 0.5)
        out[black_noise] = 0
        out[white_noise] = 255
        return np.clip(out, 0, 255).astype(np.uint8)

    if mode == "blur":
        radius = max(1, int(round(1 + 3 * severity)))
        return _box_filter(arr, radius)

    if mode == "simplify":
        eroded = _erode_dark_lines(arr, max(1, int(round(1 + 2 * severity))))
        threshold = 96 + int(64 * severity)
        return np.where(eroded < threshold, 0, 255).astype(np.uint8)

    raise ValueError(f"unknown perturbation mode: {mode}")
