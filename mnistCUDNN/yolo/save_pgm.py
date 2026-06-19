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

    return _stretch_digit(_center_by_mass(canvas))


def save_pgm(path: str | Path, image: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(image, 0, 255).astype(np.uint8)
    height, width = img.shape[:2]
    with out.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    candidates = []
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, otsu = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
        candidates.append(otsu)

    for kernel_size in _blackhat_kernel_sizes(gray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        dark_strokes = cv2.subtract(background, gray)
        dark_strokes = cv2.normalize(dark_strokes, None, 0, 255, cv2.NORM_MINMAX)
        _, mask = cv2.threshold(dark_strokes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        candidates.append(mask)

    block = max(3, min(gray.shape[:2]) // 3)
    if block % 2 == 0:
        block += 1
    if block >= 3:
        candidates.append(
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                3,
            )
        )
        candidates.append(
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block,
                3,
            )
        )

    scored = []
    for candidate in candidates:
        cleaned = _clean_mask(candidate)
        component = _digit_components(cleaned)
        scored.append((_score_digit_mask(component), component))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else np.zeros_like(gray, dtype=np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _blackhat_kernel_sizes(gray: np.ndarray) -> list[int]:
    shortest = min(gray.shape[:2])
    sizes = {
        max(5, shortest // 12),
        max(7, shortest // 8),
        max(9, shortest // 5),
    }
    normalized = []
    for size in sorted(sizes):
        if size % 2 == 0:
            size += 1
        normalized.append(max(3, size))
    return normalized


def _digit_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    height, width = mask.shape[:2]
    image_area = max(1, height * width)
    kept = np.zeros_like(mask, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(np.max(areas)) if len(areas) else 0

    for label in range(1, count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        area_frac = area / image_area
        width_frac = w / max(1, width)
        height_frac = h / max(1, height)
        aspect = w / max(1, h)

        if area < max(6, int(max_area * 0.12)) and area_frac < 0.015:
            continue
        if w < 2 or h < 2:
            continue

        touches_many_edges = int(x <= 0) + int(y <= 0) + int(x + w >= width) + int(y + h >= height)
        if touches_many_edges >= 3 and area_frac > 0.25:
            continue
        if touches_many_edges >= 2 and (width_frac > 0.70 or height_frac > 0.70):
            continue
        if (x <= 1 or x + w >= width - 1) and height_frac > 0.55 and width_frac < 0.35:
            continue
        if (y <= 1 or y + h >= height - 1) and width_frac > 0.55 and height_frac < 0.30:
            continue
        if aspect > 3.0 and width_frac > 0.45:
            continue

        kept[labels == label] = 255

    if np.count_nonzero(kept) == 0:
        largest = int(np.argmax(areas)) + 1
        kept[labels == largest] = 255

    return kept


def _score_digit_mask(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return -1.0

    height, width = mask.shape[:2]
    image_area = max(1, height * width)
    area_frac = len(xs) / image_area
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    bbox_w = (x2 - x1) / max(1, width)
    bbox_h = (y2 - y1) / max(1, height)
    aspect = bbox_w / max(0.01, bbox_h)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    fill_ratio = len(xs) / bbox_area
    cx = (x1 + x2) * 0.5 / max(1, width)
    cy = (y1 + y2) * 0.5 / max(1, height)
    center_penalty = abs(cx - 0.5) + abs(cy - 0.5)
    border_pixels = (
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    border_penalty = border_pixels / max(1, 2 * (height + width))

    score = 0.0
    score += min(area_frac, 0.35) * 4.0
    score += min(bbox_w, 0.9) + min(bbox_h, 0.9)
    score -= center_penalty
    score -= border_penalty * 1.5

    if area_frac < 0.015 or area_frac > 0.75:
        score -= 2.0
    if area_frac > 0.45:
        score -= 1.5
    if fill_ratio > 0.65:
        score -= 2.0
    elif 0.08 <= fill_ratio <= 0.45:
        score += 0.8
    if bbox_w < 0.12 or bbox_h < 0.20:
        score -= 1.0
    if aspect > 2.5 or aspect < 0.15:
        score -= 1.0
    if (y1 <= 1 or y2 >= height - 1) and bbox_w > 0.55:
        score -= 1.5
    if (x1 <= 1 or x2 >= width - 1) and bbox_h > 0.55:
        score -= 1.5
    if (int(x1 <= 1) + int(y1 <= 1) + int(x2 >= width - 1) + int(y2 >= height - 1)) >= 2:
        score -= 1.0

    return score


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


def _stretch_digit(image: np.ndarray) -> np.ndarray:
    max_value = int(image.max())
    if max_value <= 0:
        return image
    return np.clip(image.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)
