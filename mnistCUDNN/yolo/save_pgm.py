from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


def crop_digit(frame: np.ndarray, bbox: BBox, padding: float = 0.18) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * padding
    pad_y = box_h * padding

    left = int(max(0, np.floor(x1 - pad_x)))
    top = int(max(0, np.floor(y1 - pad_y)))
    right = int(min(width, np.ceil(x2 + pad_x)))
    bottom = int(min(height, np.ceil(y2 + pad_y)))

    if right <= left or bottom <= top:
        return frame.copy()

    return frame[top:bottom, left:right].copy()


def digit_to_mnist_pgm(
    crop: np.ndarray,
    target_size: int = 28,
    digit_box_size: int = 20,
) -> np.ndarray:
    """Converts a crop to an MNIST-like white digit on a black background."""

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    foreground = _foreground_mask(gray)
    foreground = _largest_component(foreground)

    ys, xs = np.where(foreground > 0)
    if len(xs) == 0 or len(ys) == 0:
        fallback = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return _normalize_to_white_digit(fallback)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    digit = foreground[y1:y2, x1:x2]

    h, w = digit.shape[:2]
    scale = min(digit_box_size / max(1, w), digit_box_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    return _center_by_mass(canvas)


def save_pgm(path: str | Path, image: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(image, 0, 255).astype(np.uint8)
    height, width = img.shape[:2]
    with out.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )
    border_mean = float(np.mean(border))
    image_mean = float(np.mean(gray))

    if border_mean >= image_mean:
        mode = cv2.THRESH_BINARY_INV
    else:
        mode = cv2.THRESH_BINARY

    _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _normalize_to_white_digit(gray: np.ndarray) -> np.ndarray:
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border)) > 127.0:
        gray = 255 - gray
    return gray.astype(np.uint8)


def _center_by_mass(image: np.ndarray) -> np.ndarray:
    moments = cv2.moments(image)
    if moments["m00"] == 0:
        return image

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    target = (image.shape[1] - 1) / 2.0
    shift_x = target - cx
    shift_y = target - cy
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
