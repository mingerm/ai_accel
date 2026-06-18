from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2

from detect_video import YOLO, YOLO_IMPORT_ERROR, as_bool, detect_one, load_config
from save_pgm import crop_digit, digit_to_mnist_pgm, save_pgm


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".tif", ".tiff", ".webp"}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    image_source = choose_image_source(args, config)
    weights_path = Path(args.weights or str(config.get("weights", "yolo/weights/best.pt")))
    output_dir = Path(args.output_dir or str(config.get("output_dir", "pgm_output")))
    expected_count = args.expected_count
    recursive = args.recursive if args.recursive is not None else as_bool(config.get("recursive_image_search", False))

    if args.image is not None and args.image_dir is not None:
        print("ERROR: use only one of --image or --image-dir", file=sys.stderr)
        return 2
    if image_source is None:
        print("ERROR: provide --image or --image-dir", file=sys.stderr)
        return 2
    if YOLO is None:
        print(f"ERROR: ultralytics is not installed: {YOLO_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if not image_source.exists():
        print(f"ERROR: image source not found: {image_source}", file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(f"ERROR: YOLO weights not found: {weights_path}", file=sys.stderr)
        return 2

    image_paths = list(collect_image_paths(image_source, recursive=recursive))
    if not image_paths:
        print(f"ERROR: no supported image files found: {image_source}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    confidence = float(config.get("confidence", 0.45))
    iou = float(config.get("iou", 0.45))
    imgsz = int(config.get("imgsz", 640))
    device = str(config.get("device", "") or "")
    target_size = int(config.get("target_size", 28))
    digit_box_size = int(config.get("digit_box_size", 20))
    bbox_padding = float(config.get("bbox_padding", 0.18))
    selection = str(config.get("selection", "center_conf"))

    processed = 0
    saved = 0
    for image_index, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"WARNING: could not read image: {image_path}", file=sys.stderr)
            continue

        processed += 1
        detection = detect_one(
            model=model,
            frame=frame,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            device=device,
            selection=selection,
        )
        if detection is None:
            print(f"SKIP {image_path} reason=no_detection")
            continue

        crop = crop_digit(frame, detection.bbox, padding=bbox_padding)
        pgm_image = digit_to_mnist_pgm(
            crop,
            target_size=target_size,
            digit_box_size=digit_box_size,
        )
        name = (
            f"image_{image_index:06d}_{safe_stem(image_path)}_digit_{detection.label}_"
            f"conf_{detection.confidence:.2f}_{saved:04d}.pgm"
        )
        save_pgm(output_dir / name, pgm_image)
        saved += 1
        print(
            f"SAVED {output_dir / name} "
            f"source={image_path} label={detection.label} "
            f"conf={detection.confidence:.3f}"
        )

        if expected_count is not None and saved >= expected_count:
            break

    print(
        f"YOLO image summary: images={len(image_paths)} processed={processed} "
        f"saved={saved} output_dir={output_dir}"
    )
    return 0


def choose_image_source(args: argparse.Namespace, config: Dict[str, Any]) -> Optional[Path]:
    if args.image is not None:
        return Path(args.image)
    if args.image_dir is not None:
        return Path(args.image_dir)
    if config.get("image_path"):
        return Path(str(config["image_path"]))
    if config.get("image_dir"):
        return Path(str(config["image_dir"]))
    return None


def collect_image_paths(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_SUFFIXES:
            yield source
        return

    pattern = "**/*" if recursive else "*"
    paths: List[Path] = []
    for path in source.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    yield from sorted(paths)


def safe_stem(path: Path) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in path.stem)
    return (value or "image")[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect handwritten digits from images and save MNIST PGM crops.")
    parser.add_argument("--config", default="yolo/config.yaml")
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--image-dir", default=None, help="Directory containing image files.")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
