from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


DIGIT_NAMES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class InkConfig:
    target_size: int
    digit_box_size: int
    max_side: int
    blur_radius: float
    max_gray: int
    min_contrast: int
    contrast_percentile: float
    min_area: int
    min_area_ratio: float
    max_fill_ratio: float
    dilate: int
    center: bool


Component = Tuple[int, int, int, int, int, List[Tuple[int, int]]]


def main() -> int:
    args = parse_args()
    config = InkConfig(
        target_size=args.target_size,
        digit_box_size=args.digit_box_size,
        max_side=args.max_side,
        blur_radius=args.blur_radius,
        max_gray=args.max_gray,
        min_contrast=args.min_contrast,
        contrast_percentile=args.contrast_percentile,
        min_area=args.min_area,
        min_area_ratio=args.min_area_ratio,
        max_fill_ratio=args.max_fill_ratio,
        dilate=args.dilate,
        center=not args.no_center,
    )

    source_root = Path(args.source)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = collect_digit_samples(source_root)
    if not samples:
        raise ValueError(f"no labeled digit images found under {source_root}")

    previews: List[Tuple[Path, Image.Image, Image.Image]] = []
    for index, (path, label) in enumerate(samples):
        original = load_image(path, args.orientation)
        digit = black_ink_to_mnist(original, config)
        out_dir = output_root / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}_{index:04d}_{path.stem}.pgm"
        save_pgm(out_path, digit)
        previews.append((out_path, original, Image.fromarray(digit)))

    write_contact_sheet(previews, output_root / "contact_sheet.png")
    print(f"converted samples: {len(samples)}")
    print(f"output: {output_root}")
    print(f"contact sheet: {output_root / 'contact_sheet.png'}")
    return 0


def collect_digit_samples(root: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    seen: set[Path] = set()
    if not root.exists():
        raise FileNotFoundError(root)

    root_label = parse_digit_label(root.name)
    for path in sorted(root.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = parse_digit_label_from_filename(path.name)
        if label is None:
            label = root_label
        if label is not None:
            samples.append((path, label))
            seen.add(path)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        label = parse_digit_label(child.name)
        if label is None:
            continue
        for path in sorted(child.glob("**/*")):
            if path in seen:
                continue
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, label))
                seen.add(path)
    return samples


def parse_digit_label(name: str) -> int | None:
    lowered = name.strip().lower()
    if lowered.isdigit() and len(lowered) == 1:
        return int(lowered)
    return DIGIT_NAMES.get(lowered)


def parse_digit_label_from_filename(name: str) -> int | None:
    lowered = name.strip().lower()
    match = re.search(r"(?:^|[_-])digit[_-]?([0-9])(?:[_\.-]|$)", lowered)
    if match:
        return int(match.group(1))
    match = re.match(r"([0-9])(?:[_-]|\.)", lowered)
    if match:
        return int(match.group(1))
    for word, digit in DIGIT_NAMES.items():
        if re.search(rf"(?:^|[_-]){word}(?:[_\.-]|$)", lowered):
            return digit
    return None


def load_image(path: Path, orientation: str) -> Image.Image:
    with Image.open(path) as image:
        loaded = image.copy()
    if orientation == "exif":
        loaded = ImageOps.exif_transpose(loaded)
    elif orientation == "rotate90cw":
        loaded = loaded.rotate(-90, expand=True)
    elif orientation == "rotate90ccw":
        loaded = loaded.rotate(90, expand=True)
    elif orientation == "rotate180":
        loaded = loaded.rotate(180, expand=True)
    return loaded.convert("L")


def black_ink_to_mnist(image: Image.Image, config: InkConfig) -> np.ndarray:
    gray = resize_for_preprocess(image, config.max_side)
    array = np.asarray(gray, dtype=np.float32)
    radius = config.blur_radius if config.blur_radius > 0 else max(10.0, min(gray.size) / 14.0)
    background = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    local_dark = np.clip(background - array, 0, 255)

    contrast_threshold = max(config.min_contrast, float(np.percentile(local_dark, config.contrast_percentile)))
    mask = (local_dark >= contrast_threshold) & (array <= config.max_gray)
    mask = clean_mask(mask)
    mask = keep_ink_components(mask, config)

    if config.dilate > 0:
        for _ in range(config.dilate):
            mask = dilate_mask(mask)

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((config.target_size, config.target_size), dtype=np.uint8)

    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    digit = fit_digit(cropped, config.target_size, config.digit_box_size)
    if config.center:
        digit = center_by_mass(digit)
    return digit


def resize_for_preprocess(image: Image.Image, max_side: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.BILINEAR)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    mask = mask & (neighbor_count(mask) >= 2)
    mask = erode_mask(dilate_mask(mask))
    mask = mask & (neighbor_count(mask) >= 3)
    return mask


def keep_ink_components(mask: np.ndarray, config: InkConfig) -> np.ndarray:
    components = connected_components(mask)
    if not components:
        return mask

    height, width = mask.shape
    image_area = max(1, height * width)
    max_area = max(component[0] for component in components)
    kept = np.zeros_like(mask, dtype=bool)

    for area, x1, y1, x2, y2, points in sorted(components, key=lambda item: item[0], reverse=True):
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = max(1, box_w * box_h)
        fill_ratio = area / box_area
        aspect = box_w / max(1, box_h)
        area_frac = area / image_area
        width_frac = box_w / max(1, width)
        height_frac = box_h / max(1, height)
        touches_edges = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= width - 1) + int(y2 >= height - 1)

        if area < config.min_area and area < max_area * config.min_area_ratio:
            continue
        if fill_ratio > config.max_fill_ratio:
            continue
        if area_frac > 0.18:
            continue
        if touches_edges >= 1 and (width_frac > 0.75 or height_frac > 0.75):
            continue
        if touches_edges >= 2 and (width_frac > 0.50 or height_frac > 0.50):
            continue
        if aspect < 0.08 and height_frac > 0.25:
            continue
        if aspect > 12.0 and width_frac > 0.25:
            continue
        if box_w < 2 or box_h < 2:
            continue

        for y, x in points:
            kept[y, x] = True

    if np.count_nonzero(kept) == 0:
        largest = max(components, key=lambda item: item[0])
        for y, x in largest[5]:
            kept[y, x] = True
    return kept


def connected_components(mask: np.ndarray) -> List[Component]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[Component] = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if seen[start_y, start_x] or not mask[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        points: List[Tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))

        point_array = np.asarray(points)
        y1, x1 = point_array.min(axis=0)
        y2, x2 = point_array.max(axis=0) + 1
        components.append((len(points), int(x1), int(y1), int(x2), int(y2), points))
    return components


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            count += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return count


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def erode_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=True)
    out = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def fit_digit(cropped: np.ndarray, target_size: int, digit_box_size: int) -> np.ndarray:
    height, width = cropped.shape
    scale = min(digit_box_size / max(1, width), digit_box_size / max(1, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    digit = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(digit, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def center_by_mass(image: np.ndarray) -> np.ndarray:
    ys, xs = np.where(image > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image

    weights = image[ys, xs].astype(np.float64)
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))
    target = (image.shape[0] - 1) / 2.0
    shift_x = int(round(target - cx))
    shift_y = int(round(target - cy))
    return shift_image(image, shift_x, shift_y)


def shift_image(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    out = np.zeros_like(image)
    height, width = image.shape
    src_x1 = max(0, -shift_x)
    src_x2 = min(width, width - shift_x)
    src_y1 = max(0, -shift_y)
    src_y2 = min(height, height - shift_y)
    dst_x1 = max(0, shift_x)
    dst_x2 = min(width, width + shift_x)
    dst_y1 = max(0, shift_y)
    dst_y2 = min(height, height + shift_y)
    if src_x2 > src_x1 and src_y2 > src_y1:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    return out


def save_pgm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(image, 0, 255).astype(np.uint8)
    height, width = img.shape
    with path.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def write_contact_sheet(items: Sequence[Tuple[Path, Image.Image, Image.Image]], output_path: Path) -> None:
    cell_w, cell_h = 180, 142
    cols = min(4, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (235, 235, 235))
    for index, (path, original, digit) in enumerate(items):
        cell = Image.new("RGB", (cell_w, cell_h), "white")
        original_preview = ImageOps.autocontrast(original).convert("RGB")
        original_preview.thumbnail((108, 80))
        digit_preview = digit.resize((56, 56), Image.Resampling.NEAREST).convert("RGB")
        cell.paste(original_preview, (6, 4))
        cell.paste(digit_preview, (118, 10))
        draw = ImageDraw.Draw(cell)
        draw.text((6, 96), path.parent.name + "/" + path.name[:22], fill=(0, 0, 0))
        sheet.paste(cell, ((index % cols) * cell_w, (index // cols) * cell_h))
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert extra digit photos to MNIST-like PGM training data.")
    parser.add_argument("--source", default="extra", help="Labeled source image root.")
    parser.add_argument("--output", default="mnistCUDNN/datasets/extra_pgm_ink", help="Output labeled PGM root.")
    parser.add_argument(
        "--orientation",
        default="exif",
        choices=["exif", "raw", "rotate90cw", "rotate90ccw", "rotate180"],
        help="Source image orientation handling.",
    )
    parser.add_argument("--target-size", type=int, default=28)
    parser.add_argument("--digit-box-size", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--blur-radius", type=float, default=0.0, help="0 chooses a size-dependent radius.")
    parser.add_argument("--max-gray", type=int, default=150, help="Only pixels darker than this can become ink.")
    parser.add_argument("--min-contrast", type=int, default=18, help="Minimum local dark contrast to keep.")
    parser.add_argument("--contrast-percentile", type=float, default=96.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--min-area-ratio", type=float, default=0.06)
    parser.add_argument("--max-fill-ratio", type=float, default=0.52)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--no-center", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
