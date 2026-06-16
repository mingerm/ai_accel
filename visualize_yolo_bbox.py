from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO bbox labels on images.")
    parser.add_argument("--images", default="yolo_auto/images", help="Image directory.")
    parser.add_argument("--labels", default="yolo_auto/labels", help="YOLO label directory.")
    parser.add_argument("--output", default="yolo_auto/bbox_preview", help="Output directory.")
    parser.add_argument("--limit", type=int, default=69, help="Maximum number of images to render.")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional image stems to render, e.g. 1_one_1_0001 1_one_1_0002.",
    )
    return parser.parse_args()


def read_label(label_path: Path) -> tuple[str, float, float, float, float] | None:
    if not label_path.exists():
        return None

    line = label_path.read_text().strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label: {label_path}")

    class_id = parts[0]
    cx, cy, width, height = map(float, parts[1:])
    return class_id, cx, cy, width, height


def yolo_to_pixels(
    cx: float,
    cy: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = int(round((cx - width / 2) * image_width))
    y1 = int(round((cy - height / 2) * image_height))
    x2 = int(round((cx + width / 2) * image_width))
    y2 = int(round((cy + height / 2) * image_height))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    return x1, y1, x2, y2


def draw_bbox(image_path: Path, label_path: Path, output_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    label = read_label(label_path)
    if label is None:
        raise ValueError(f"Missing or empty label: {label_path}")

    class_id, cx, cy, width, height = label
    x1, y1, x2, y2 = yolo_to_pixels(cx, cy, width, height, image_width, image_height)

    draw = ImageDraw.Draw(image)
    line_width = max(3, min(image_width, image_height) // 160)
    for offset in range(line_width):
        draw.rectangle((x1 + offset, y1 + offset, x2 - offset, y2 - offset), outline=(255, 0, 0))

    text = f"class={class_id}  cx={cx:.3f} cy={cy:.3f} w={width:.3f} h={height:.3f}"
    text_box = draw.textbbox((0, 0), text)
    draw.rectangle((0, 0, text_box[2] + 16, text_box[3] + 16), fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(255, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return image


def make_contact_sheet(images: list[tuple[str, Image.Image]], output_path: Path) -> None:
    if not images:
        return

    cell_width = 420
    cell_height = 320
    cols = 2
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_width * cols, cell_height * rows), (240, 240, 240))

    for index, (name, image) in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((cell_width - 20, cell_height - 55))

        x = (index % cols) * cell_width
        y = (index // cols) * cell_height
        cell = Image.new("RGB", (cell_width, cell_height), "white")
        cell.paste(thumb, ((cell_width - thumb.width) // 2, 10))
        ImageDraw.Draw(cell).text((10, cell_height - 35), name, fill=(0, 0, 0))
        sheet.paste(cell, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def find_images(image_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        selected = []
        for name in names:
            stem = Path(name).stem
            matches = [image_dir / f"{stem}{ext}" for ext in IMAGE_EXTENSIONS]
            selected.extend(path for path in matches if path.exists())
        return selected

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
    return sorted(images)


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_dir = Path(args.output)

    rendered = []
    for image_path in find_images(image_dir, args.names)[: args.limit]:
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = output_dir / f"{image_path.stem}_bbox.jpg"
        try:
            image = draw_bbox(image_path, label_path, output_path)
        except ValueError as exc:
            print(f"SKIP: {exc}")
            continue

        rendered.append((image_path.stem, image))
        print(f"WROTE: {output_path}")

    make_contact_sheet(rendered, output_dir / "contact_sheet.jpg")
    if rendered:
        print(f"WROTE: {output_dir / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
