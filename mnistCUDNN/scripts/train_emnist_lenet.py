from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms


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
EXPECTED_BIN_BYTES = {
    "conv1.bin": 20 * 1 * 5 * 5 * 4,
    "conv1.bias.bin": 20 * 4,
    "conv2.bin": 50 * 20 * 5 * 5 * 4,
    "conv2.bias.bin": 50 * 4,
    "ip1.bin": 500 * 800 * 4,
    "ip1.bias.bin": 500 * 4,
    "ip2.bin": 10 * 500 * 4,
    "ip2.bias.bin": 10 * 4,
}


class LeNetForCUDNN(nn.Module):
    """Matches mnistCUDNN.cpp layer order and tensor shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.lrn(x[:, :, None, None]).squeeze(-1).squeeze(-1)
        return self.fc2(x)


class ExtraDigitDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        auto_crop: bool = True,
        preprocess: str = "auto",
        orientation: str = "exif",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        recursive: bool = True,
        cache: bool = True,
    ) -> None:
        self.samples = collect_digit_samples(root, recursive=recursive)
        self.auto_crop = auto_crop
        self.preprocess = preprocess
        self.orientation = orientation
        self.transform = transform
        self.cached_images: List[Tuple[Image.Image, int]] | None = None
        if not self.samples:
            raise ValueError(f"no labeled digit images found under {root}")
        if cache:
            self.cached_images = [self.load_image(path, label) for path, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if self.cached_images is not None:
            image, label = self.cached_images[index]
            image = image.copy()
        else:
            image, label = self.load_image(*self.samples[index])

        if self.transform is not None:
            return self.transform(image), label

        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), label

    def load_image(self, path: Path, label: int) -> Tuple[Image.Image, int]:
        with Image.open(path) as source:
            image = source.copy()
        if self.orientation == "exif":
            image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        if self.auto_crop:
            image = mnist_like_image(image, mode=self.preprocess)
        else:
            image = image.resize((28, 28), Image.Resampling.BILINEAR)
        return image, label


@dataclass
class Metrics:
    loss: float
    accuracy: float
    per_digit: Dict[int, Tuple[int, int]]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)

    if args.preview_extra_dir:
        roots = args.extra_data_root or args.eval_data_root
        if not roots:
            raise ValueError("--preview-extra-dir requires --extra-data-root or --eval-data-root")
        export_extra_previews(roots, args, resolve_project_path(args.preview_extra_dir, project_root))
        if args.preview_only:
            return 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    train_dataset, test_dataset = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    eval_loaders: List[Tuple[str, DataLoader[Tuple[torch.Tensor, int]]]] = [("emnist_test", test_loader)]
    for eval_root in args.eval_data_root:
        dataset = ExtraDigitDataset(
            Path(eval_root),
            auto_crop=not args.no_extra_auto_crop,
            preprocess=args.extra_preprocess,
            orientation=args.extra_orientation,
            cache=not args.no_extra_cache,
        )
        eval_loaders.append(
            (
                f"extra:{eval_root}",
                DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
            )
        )

    model = LeNetForCUDNN().to(device)
    if args.init_bin_dir:
        init_bin_dir = resolve_project_path(args.init_bin_dir, project_root)
        load_bins(model, init_bin_dir)
        print(f"loaded initial weights: {init_bin_dir}")
        if args.eval_initial:
            print("initial evaluation:")
            for name, loader in eval_loaders:
                print_metrics(name, evaluate(model, loader, device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        test_metrics = evaluate(model, test_loader, device)
        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.accuracy:.4f} "
            f"elapsed={elapsed:.1f}s"
        )

        if test_metrics.accuracy > best_accuracy:
            best_accuracy = test_metrics.accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("final evaluation:")
    final_metrics = {}
    for name, loader in eval_loaders:
        metrics = evaluate(model, loader, device)
        final_metrics[name] = metrics_to_dict(metrics)
        print_metrics(name, metrics)

    if final_metrics["emnist_test"]["accuracy"] < args.min_test_accuracy:
        raise RuntimeError(
            "EMNIST test accuracy "
            f"{final_metrics['emnist_test']['accuracy']:.4f} is below --min-test-accuracy "
            f"{args.min_test_accuracy:.4f}; refusing to export weights"
        )

    export_dir = Path(args.export_dir)
    if not export_dir.is_absolute():
        export_dir = project_root / export_dir
    if export_dir.exists() and args.backup_existing:
        backup_bins(export_dir)
    export_bins(model, export_dir, overwrite=args.overwrite)

    checkpoint_path = export_dir / "lenet_emnist_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "final_metrics": final_metrics,
        },
        checkpoint_path,
    )
    with (export_dir / "lenet_emnist_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "final_metrics": final_metrics}, f, indent=2)

    print(f"exported weights: {export_dir}")
    print(f"checkpoint: {checkpoint_path}")
    return 0


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset[Tuple[torch.Tensor, int]], Dataset[Tuple[torch.Tensor, int]]]:
    train_transform = build_emnist_transform(args, train=True)
    test_transform = build_emnist_transform(args, train=False)
    train_dataset = datasets.EMNIST(
        root=args.data_root,
        split=args.emnist_split,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    test_dataset = datasets.EMNIST(
        root=args.data_root,
        split=args.emnist_split,
        train=False,
        download=args.download,
        transform=test_transform,
    )

    datasets_to_concat: List[Dataset[Tuple[torch.Tensor, int]]] = [train_dataset]
    extra_transform = build_extra_transform(args, train=True)
    for extra_root in args.extra_data_root:
        extra_dataset = ExtraDigitDataset(
            Path(extra_root),
            auto_crop=not args.no_extra_auto_crop,
            preprocess=args.extra_preprocess,
            orientation=args.extra_orientation,
            transform=extra_transform,
            cache=not args.no_extra_cache,
        )
        print(f"extra dataset: {extra_root} samples={len(extra_dataset)} repeat={args.extra_repeat}")
        for _ in range(args.extra_repeat):
            datasets_to_concat.append(extra_dataset)

    if len(datasets_to_concat) == 1:
        return train_dataset, test_dataset
    return ConcatDataset(datasets_to_concat), test_dataset


def build_emnist_transform(args: argparse.Namespace, train: bool) -> transforms.Compose:
    ops: List[object] = []
    if not args.no_emnist_orientation_fix:
        ops.append(transforms.Lambda(fix_emnist_orientation))
    if train and args.augment:
        ops.append(
            transforms.RandomAffine(
                degrees=args.augment_degrees,
                translate=(args.augment_translate, args.augment_translate),
                scale=(1.0 - args.augment_scale, 1.0 + args.augment_scale),
                shear=args.augment_shear,
                fill=0,
            )
        )
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_extra_transform(args: argparse.Namespace, train: bool) -> transforms.Compose | None:
    if not train or not args.extra_augment:
        return None
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=args.extra_augment_degrees,
                translate=(args.extra_augment_translate, args.extra_augment_translate),
                scale=(1.0 - args.extra_augment_scale, 1.0 + args.extra_augment_scale),
                shear=args.extra_augment_shear,
                fill=0,
            ),
            transforms.ToTensor(),
        ]
    )


def fix_emnist_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.mirror(image.rotate(-90, fillcolor=0))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.numel()
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.numel())
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, int]],
    device: torch.device,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    per_digit: Dict[int, List[int]] = {i: [0, 0] for i in range(10)}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        correct = predictions == labels

        total_loss += float(loss.item()) * labels.numel()
        total_correct += int(correct.sum().item())
        total_count += int(labels.numel())
        for label, ok in zip(labels.detach().cpu().tolist(), correct.detach().cpu().tolist()):
            per_digit[int(label)][1] += 1
            if ok:
                per_digit[int(label)][0] += 1

    return Metrics(
        loss=total_loss / max(1, total_count),
        accuracy=total_correct / max(1, total_count),
        per_digit={digit: (values[0], values[1]) for digit, values in per_digit.items()},
    )


def print_metrics(name: str, metrics: Metrics) -> None:
    print(f"{name}: loss={metrics.loss:.4f} accuracy={metrics.accuracy:.4f}")
    for digit in range(10):
        correct, total = metrics.per_digit.get(digit, (0, 0))
        accuracy = correct / total if total else 0.0
        print(f"  digit={digit} accuracy={accuracy:.4f} ({correct}/{total})")


def metrics_to_dict(metrics: Metrics) -> Dict[str, object]:
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "per_digit": {
            str(digit): {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}
            for digit, (correct, total) in metrics.per_digit.items()
        },
    }


def collect_digit_samples(root: Path, recursive: bool) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    seen: set[Path] = set()
    if not root.exists():
        raise FileNotFoundError(root)

    root_label = parse_digit_label(root.name)
    root_pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(root_pattern)):
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
        pattern = "**/*" if recursive else "*"
        for path in sorted(child.glob(pattern)):
            if path in seen:
                continue
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, label))
                seen.add(path)
    return samples


def export_extra_previews(roots: Sequence[str], args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_images: List[Tuple[Path, Image.Image]] = []
    for root in roots:
        dataset = ExtraDigitDataset(
            Path(root),
            auto_crop=not args.no_extra_auto_crop,
            preprocess=args.extra_preprocess,
            orientation=args.extra_orientation,
            cache=True,
        )
        root_name = Path(root).name or "extra"
        for index, (image, label) in enumerate(dataset.cached_images or []):
            source = dataset.samples[index][0]
            stem = f"{label}_{index:04d}_{source.stem}".replace(" ", "_")
            out_path = output_dir / root_name / f"{stem}.pgm"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            preview_images.append((out_path, image.copy()))
        print(f"preview dataset: {root} samples={len(dataset)} output={output_dir / root_name}")

    if preview_images:
        write_contact_sheet(preview_images, output_dir / "contact_sheet.png")


def write_contact_sheet(items: Sequence[Tuple[Path, Image.Image]], output_path: Path) -> None:
    cell_w, cell_h = 116, 96
    cols = min(6, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (235, 235, 235))
    for index, (path, image) in enumerate(items):
        cell = Image.new("RGB", (cell_w, cell_h), "white")
        preview = image.resize((56, 56), Image.Resampling.NEAREST).convert("RGB")
        cell.paste(preview, ((cell_w - preview.width) // 2, 4))
        draw = ImageDraw.Draw(cell)
        draw.text((4, 64), path.name[:18], fill=(0, 0, 0))
        sheet.paste(cell, ((index % cols) * cell_w, (index // cols) * cell_h))
    sheet.save(output_path)
    print(f"preview contact sheet: {output_path}")


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


def mnist_like_image(
    image: Image.Image,
    target_size: int = 28,
    digit_box_size: int = 20,
    mode: str = "auto",
) -> Image.Image:
    gray = image.convert("L")
    if mode == "pipeline":
        return pipeline_mnist_like_image(gray, target_size=target_size, digit_box_size=digit_box_size)

    gray = resize_for_preprocess(gray)
    array = np.asarray(gray, dtype=np.uint8)
    candidates: List[np.ndarray] = []

    if mode in ("auto", "simple"):
        candidates.append(foreground_mask(array))
    if mode in ("auto", "ink"):
        candidates.extend(ink_foreground_candidates(gray, array))
    if not candidates:
        raise ValueError(f"unknown extra preprocessing mode: {mode}")

    scored = [(score_digit_mask(candidate), candidate) for candidate in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    mask = scored[0][1]
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return ImageOps.autocontrast(gray.resize((target_size, target_size), Image.Resampling.BILINEAR))

    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    h, w = cropped.shape
    scale = min(digit_box_size / max(1, w), digit_box_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(digit, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def pipeline_mnist_like_image(image: Image.Image, target_size: int = 28, digit_box_size: int = 20) -> Image.Image:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from yolo.save_pgm import digit_to_mnist_pgm
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise RuntimeError(
                "--extra-preprocess pipeline requires opencv-python. "
                "Install it in Colab with: pip install opencv-python"
            ) from exc
        raise

    array = np.asarray(image, dtype=np.uint8)
    output = digit_to_mnist_pgm(array, target_size=target_size, digit_box_size=digit_box_size)
    return Image.fromarray(output)


def resize_for_preprocess(image: Image.Image, max_side: int = 640) -> Image.Image:
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.BILINEAR)


def foreground_mask(array: np.ndarray) -> np.ndarray:
    border = np.concatenate([array[0, :], array[-1, :], array[:, 0], array[:, -1]])
    threshold = otsu_threshold(array)
    if float(border.mean()) >= float(array.mean()):
        return clean_digit_mask(array < threshold)
    return clean_digit_mask(array > threshold)


def ink_foreground_candidates(gray: Image.Image, array: np.ndarray) -> List[np.ndarray]:
    radius = max(6, min(gray.size) // 18)
    background = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    dark_strokes = np.clip(background - array.astype(np.float32), 0, 255).astype(np.uint8)
    dark_threshold = max(8, otsu_threshold(dark_strokes), int(np.percentile(dark_strokes, 88)))
    gray_threshold = int(np.percentile(array, 62))
    candidates = [
        dark_strokes > dark_threshold,
        (dark_strokes > max(6, int(np.percentile(dark_strokes, 92)))) & (array < gray_threshold),
        (array < min(otsu_threshold(array), int(np.percentile(array, 12)) + 12)) | (dark_strokes > dark_threshold),
    ]
    return [clean_digit_mask(candidate) for candidate in candidates]


def clean_digit_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    mask = mask & (neighbor_count(mask) >= 2)
    mask = erode_mask(dilate_mask(mask))
    mask = mask & (neighbor_count(mask) >= 3)
    return keep_digit_components(mask)


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


def keep_digit_components(mask: np.ndarray) -> np.ndarray:
    components = connected_components(mask)
    if not components:
        return mask

    height, width = mask.shape
    image_area = max(1, height * width)
    max_area = max(component[0] for component in components)
    kept = np.zeros_like(mask, dtype=bool)
    for area, x1, y1, x2, y2, points in components:
        box_w = x2 - x1
        box_h = y2 - y1
        area_frac = area / image_area
        width_frac = box_w / max(1, width)
        height_frac = box_h / max(1, height)
        touches_edges = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= width - 1) + int(y2 >= height - 1)

        if area < max(6, int(max_area * 0.08)) and area_frac < 0.001:
            continue
        if box_w < 2 or box_h < 2:
            continue
        if touches_edges >= 2 and (width_frac > 0.55 or height_frac > 0.55):
            continue
        if area_frac > 0.45:
            continue

        for y, x in points:
            kept[y, x] = True

    if np.count_nonzero(kept) == 0:
        largest = max(components, key=lambda item: item[0])
        for y, x in largest[5]:
            kept[y, x] = True
    return kept


def connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int, List[Tuple[int, int]]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[Tuple[int, int, int, int, int, List[Tuple[int, int]]]] = []
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


def score_digit_mask(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return -1_000.0

    height, width = mask.shape
    image_area = max(1, height * width)
    area_frac = len(xs) / image_area
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    bbox_w = (x2 - x1) / max(1, width)
    bbox_h = (y2 - y1) / max(1, height)
    aspect = bbox_w / max(0.01, bbox_h)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    fill_ratio = len(xs) / bbox_area
    border_pixels = (
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    border_penalty = border_pixels / max(1, 2 * (height + width))

    score = 0.0
    score += min(area_frac, 0.25) * 6.0
    score += min(bbox_w, 0.90) + min(bbox_h, 0.90)
    score -= border_penalty * 2.0

    if area_frac < 0.0004 or area_frac > 0.45:
        score -= 3.0
    if fill_ratio > 0.65:
        score -= 2.0
    elif 0.03 <= fill_ratio <= 0.45:
        score += 0.5
    if bbox_w < 0.04 or bbox_h < 0.08:
        score -= 1.0
    if aspect > 3.0 or aspect < 0.08:
        score -= 1.0
    return score


def otsu_threshold(array: np.ndarray) -> int:
    hist = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    total = array.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 127
    best_variance = -1.0
    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def resolve_project_path(path: str, project_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved


def load_bins(model: LeNetForCUDNN, bin_dir: Path) -> None:
    tensors = {
        "conv1.bin": model.conv1.weight,
        "conv1.bias.bin": model.conv1.bias,
        "conv2.bin": model.conv2.weight,
        "conv2.bias.bin": model.conv2.bias,
        "ip1.bin": model.fc1.weight,
        "ip1.bias.bin": model.fc1.bias,
        "ip2.bin": model.fc2.weight,
        "ip2.bias.bin": model.fc2.bias,
    }
    for name, tensor in tensors.items():
        path = bin_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        expected_values = tensor.numel()
        array = np.fromfile(path, dtype=np.float32)
        if array.size != expected_values:
            raise RuntimeError(f"{path} has {array.size} float32 values; expected {expected_values}")
        loaded = torch.from_numpy(array.reshape(tuple(tensor.shape))).to(device=tensor.device, dtype=tensor.dtype)
        tensor.data.copy_(loaded)


def export_bins(model: LeNetForCUDNN, export_dir: Path, overwrite: bool) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "conv1.bin": model.conv1.weight,
        "conv1.bias.bin": model.conv1.bias,
        "conv2.bin": model.conv2.weight,
        "conv2.bias.bin": model.conv2.bias,
        "ip1.bin": model.fc1.weight,
        "ip1.bias.bin": model.fc1.bias,
        "ip2.bin": model.fc2.weight,
        "ip2.bias.bin": model.fc2.bias,
    }
    for name, tensor in tensors.items():
        path = export_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")
        array = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
        array.tofile(path)

    for name, expected_size in EXPECTED_BIN_BYTES.items():
        actual_size = (export_dir / name).stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"{name} has {actual_size} bytes; expected {expected_size}")


def backup_bins(export_dir: Path) -> None:
    existing = [export_dir / name for name in EXPECTED_BIN_BYTES if (export_dir / name).exists()]
    if not existing:
        return
    backup_dir = export_dir / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    print(f"backed up existing bins: {backup_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeNet-compatible EMNIST digit classifier and export mnistCUDNN .bin weights."
    )
    parser.add_argument("--data-root", default="datasets", help="Torchvision dataset cache root.")
    parser.add_argument("--emnist-split", default="digits", choices=["digits", "mnist"])
    parser.add_argument("--download", action="store_true", help="Download EMNIST if needed.")
    parser.add_argument("--extra-data-root", action="append", default=[], help="Labeled digit folders to mix in.")
    parser.add_argument("--eval-data-root", action="append", default=[], help="Labeled digit folders for evaluation.")
    parser.add_argument("--extra-repeat", type=int, default=5, help="Oversampling factor for extra data.")
    parser.add_argument("--extra-preprocess", default="auto", choices=["auto", "simple", "ink", "pipeline"])
    parser.add_argument("--extra-orientation", default="exif", choices=["exif", "raw"])
    parser.add_argument("--extra-augment", action="store_true", help="Apply random affine augmentation to extra data.")
    parser.add_argument("--extra-augment-degrees", type=float, default=8.0)
    parser.add_argument("--extra-augment-translate", type=float, default=0.06)
    parser.add_argument("--extra-augment-scale", type=float, default=0.08)
    parser.add_argument("--extra-augment-shear", type=float, default=4.0)
    parser.add_argument("--preview-extra-dir", help="Write preprocessed extra-data PGM previews before training.")
    parser.add_argument("--preview-only", action="store_true", help="Only write extra-data previews; do not train.")
    parser.add_argument("--no-extra-auto-crop", action="store_true")
    parser.add_argument("--no-extra-cache", action="store_true", help="Do not cache extra images after preprocessing.")
    parser.add_argument("--no-emnist-orientation-fix", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-degrees", type=float, default=10.0)
    parser.add_argument("--augment-translate", type=float, default=0.08)
    parser.add_argument("--augment-scale", type=float, default=0.08)
    parser.add_argument("--augment-shear", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-step", type=int, default=6)
    parser.add_argument("--lr-gamma", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="")
    parser.add_argument("--init-bin-dir", help="Initialize the model from an existing mnistCUDNN data/*.bin directory.")
    parser.add_argument("--eval-initial", action="store_true", help="Evaluate initialized weights before training.")
    parser.add_argument("--min-test-accuracy", type=float, default=0.0, help="Refuse export if EMNIST test accuracy is lower.")
    parser.add_argument("--export-dir", default="trained_weights/emnist_digits")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
