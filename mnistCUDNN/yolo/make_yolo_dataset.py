from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from config_utils import load_config_section

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    cv2 = None
    np = None
    CV_IMPORT_ERROR = exc
else:
    CV_IMPORT_ERROR = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm", ".webp"}
BUILTIN_DIGITS = {
    1: "one_28x28.pgm",
    3: "three_28x28.pgm",
    5: "five_28x28.pgm",
}
DEFAULTS = {
    "source_root": "..",
    "output_dir": "datasets/yolo",
    "classes": "1,3,5,6,8",
    "train_per_class": 120,
    "val_per_class": 30,
    "imgsz": 640,
    "min_digit_frac": 0.18,
    "max_digit_frac": 0.55,
    "max_rotate": 14.0,
    "seed": 0,
    "jpeg_quality": 95,
    "include_builtins": True,
    "allow_missing_classes": False,
}


@dataclass(frozen=True)
class DigitSource:
    label: int
    path: Path


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    config_path = resolve_path(root_dir, args.config)
    apply_config(args, load_config_section(config_path, "dataset_generation"), DEFAULTS)

    if cv2 is None or np is None:
        print(f"ERROR: OpenCV/Numpy dependency is missing: {CV_IMPORT_ERROR}", file=sys.stderr)
        print("Install dependencies with: pip install -r requirements-yolo.txt", file=sys.stderr)
        return 2

    source_root = resolve_path(root_dir, args.source_root)
    output_dir = resolve_path(root_dir, args.output_dir)
    classes = parse_classes(args.classes)

    sources = collect_sources(
        root_dir=root_dir,
        source_root=source_root,
        classes=classes,
        include_builtins=args.include_builtins,
    )
    missing = [label for label in classes if not sources.get(label)]
    if missing and not args.allow_missing_classes:
        print(
            "ERROR: missing source images for classes: "
            + ", ".join(str(label) for label in missing),
            file=sys.stderr,
        )
        print(
            "Put images under digit-named folders such as ../6 and ../8, "
            "or pass --allow-missing-classes.",
            file=sys.stderr,
        )
        return 2

    active_classes = [label for label in classes if sources.get(label)]
    if not active_classes:
        print("ERROR: no usable digit source images found.", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    prepare_output_dirs(output_dir)

    total = 0
    for split, count in (("train", args.train_per_class), ("val", args.val_per_class)):
        for label in active_classes:
            for index in range(count):
                source = rng.choice(sources[label])
                image, yolo_box = synthesize_sample(source.path, args, rng, np_rng)
                stem = f"syn_{split}_{label}_{index:05d}"
                image_path = output_dir / "images" / split / f"{stem}.jpg"
                label_path = output_dir / "labels" / split / f"{stem}.txt"
                write_image(image_path, image, args.jpeg_quality)
                write_label(label_path, label, yolo_box)
                total += 1

    print("YOLO dataset generated")
    print(f"source_root : {source_root}")
    print(f"output_dir  : {output_dir}")
    print(f"classes     : {', '.join(str(label) for label in active_classes)}")
    print(f"images      : {total}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO training data from digit source images.")
    parser.add_argument("--config", default="yolo/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--source-root", default=DEFAULTS["source_root"], help="Directory containing digit folders like 6/ and 8/.")
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"], help="YOLO dataset output directory.")
    parser.add_argument("--classes", default=DEFAULTS["classes"], help="Comma-separated class ids or 'all'.")
    parser.add_argument("--train-per-class", type=int, default=DEFAULTS["train_per_class"])
    parser.add_argument("--val-per-class", type=int, default=DEFAULTS["val_per_class"])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--min-digit-frac", type=float, default=DEFAULTS["min_digit_frac"])
    parser.add_argument("--max-digit-frac", type=float, default=DEFAULTS["max_digit_frac"])
    parser.add_argument("--max-rotate", type=float, default=DEFAULTS["max_rotate"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULTS["jpeg_quality"])
    parser.add_argument("--include-builtins", default=DEFAULTS["include_builtins"], action=argparse.BooleanOptionalAction)
    parser.add_argument("--allow-missing-classes", default=DEFAULTS["allow_missing_classes"], action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def apply_config(args: argparse.Namespace, config: dict, defaults: dict) -> None:
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)


def collect_sources(
    root_dir: Path,
    source_root: Path,
    classes: list[int],
    include_builtins: bool,
) -> dict[int, list[DigitSource]]:
    class_set = set(classes)
    sources: dict[int, list[DigitSource]] = defaultdict(list)

    if source_root.exists():
        if source_root.name.isdigit() and int(source_root.name) in class_set:
            add_images_from_dir(sources, int(source_root.name), source_root)

        for child in sorted(source_root.iterdir()):
            if child.is_dir() and child.name.isdigit():
                label = int(child.name)
                if label in class_set:
                    add_images_from_dir(sources, label, child)
            elif child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                label = infer_label_from_name(child)
                if label is not None and label in class_set:
                    sources[label].append(DigitSource(label=label, path=child))

    if include_builtins:
        for label, name in BUILTIN_DIGITS.items():
            if label in class_set:
                path = root_dir / "data" / name
                if path.exists():
                    sources[label].append(DigitSource(label=label, path=path))

    return dict(sources)


def add_images_from_dir(sources: dict[int, list[DigitSource]], label: int, directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            sources[label].append(DigitSource(label=label, path=path))


def infer_label_from_name(path: Path) -> int | None:
    match = re.search(r"(^|[^0-9])([0-9])([^0-9]|$)", path.stem)
    if not match:
        return None
    return int(match.group(2))


def synthesize_sample(
    source_path: Path,
    args: argparse.Namespace,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    canvas = make_background(args.imgsz, rng, np_rng)
    mask = read_digit_mask(source_path)
    mask = augment_mask(mask, args, rng)

    h, w = mask.shape[:2]
    if h >= args.imgsz or w >= args.imgsz:
        scale = min((args.imgsz - 2) / max(1, w), (args.imgsz - 2) / max(1, h))
        mask = resize_mask(mask, max(1, int(w * scale)), max(1, int(h * scale)))
        h, w = mask.shape[:2]

    x = rng.randint(0, max(0, args.imgsz - w - 1))
    y = rng.randint(0, max(0, args.imgsz - h - 1))
    ink = choose_ink_color(canvas, rng)
    canvas = paste_digit(canvas, mask, x, y, ink)

    ys, xs = np.where(mask > 8)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"empty digit mask after augmentation: {source_path}")

    x1 = x + int(xs.min())
    x2 = x + int(xs.max()) + 1
    y1 = y + int(ys.min())
    y2 = y + int(ys.max()) + 1
    return canvas, to_yolo_box(x1, y1, x2, y2, args.imgsz, args.imgsz)


def read_digit_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read image: {path}")

    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if alpha is not None and int(alpha.max()) > int(alpha.min()):
        mask = np.where(alpha > 8, 255, 0).astype(np.uint8)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        mode = cv2.THRESH_BINARY_INV if float(np.mean(border)) >= float(np.mean(gray)) else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = largest_component(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"no foreground digit found: {path}")

    pad = 4
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(mask.shape[1], int(xs.max()) + 1 + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(mask.shape[0], int(ys.max()) + 1 + pad)
    return mask[y1:y2, x1:x2]


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def augment_mask(mask: np.ndarray, args: argparse.Namespace, rng: random.Random) -> np.ndarray:
    target_h = rng.uniform(args.min_digit_frac, args.max_digit_frac) * args.imgsz
    scale = target_h / max(1, mask.shape[0])
    new_w = max(4, int(round(mask.shape[1] * scale)))
    new_h = max(4, int(round(mask.shape[0] * scale)))
    mask = resize_mask(mask, new_w, new_h)

    angle = rng.uniform(-args.max_rotate, args.max_rotate)
    return rotate_mask(mask, angle)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(rotated, 0, 255).astype(np.uint8)


def make_background(size: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.85:
        base = rng.randint(205, 248)
    else:
        base = rng.randint(35, 90)
    background = np.full((size, size, 3), base, dtype=np.int16)
    noise = np_rng.normal(0, rng.uniform(3.0, 12.0), size=(size, size, 1))
    background = background + noise.astype(np.int16)

    gradient = np.linspace(rng.randint(-18, 18), rng.randint(-18, 18), size, dtype=np.int16)
    background = background + gradient.reshape(1, size, 1)
    return np.clip(background, 0, 255).astype(np.uint8)


def choose_ink_color(canvas: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    if float(np.mean(canvas)) > 128.0:
        value = rng.randint(0, 55)
    else:
        value = rng.randint(200, 255)
    jitter = [rng.randint(-8, 8) for _ in range(3)]
    return tuple(int(np.clip(value + item, 0, 255)) for item in jitter)


def paste_digit(
    canvas: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    h, w = mask.shape[:2]
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    roi = canvas[y : y + h, x : x + w].astype(np.float32)
    ink = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    canvas[y : y + h, x : x + w] = np.clip((roi * (1.0 - alpha)) + (ink * alpha), 0, 255).astype(np.uint8)
    return canvas


def to_yolo_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0
    return (
        x_center / width,
        y_center / height,
        box_w / width,
        box_h / height,
    )


def prepare_output_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def write_label(path: Path, label: int, box: tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_center, y_center, width, height = box
    path.write_text(
        f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )


def parse_classes(value: str | int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(item) for item in value]
    if str(value).strip().lower() == "all":
        return list(range(10))
    labels = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not labels:
        raise ValueError("--classes must not be empty")
    return labels


def resolve_path(root_dir: Path, path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (root_dir / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
