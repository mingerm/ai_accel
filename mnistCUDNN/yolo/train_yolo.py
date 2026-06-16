from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from config_utils import load_config_section

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency guard
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


DEFAULTS = {
    "data": "yolo/dataset.yaml",
    "model": "yolov8n.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": "16",
    "device": "",
    "workers": 4,
    "project": "runs/digit_yolo",
    "name": "train",
    "patience": 30,
    "optimizer": "auto",
    "lr0": None,
    "seed": 0,
    "cache": False,
    "exist_ok": False,
    "resume": False,
    "close_mosaic": 10,
    "copy_best_to": "yolo/weights/best.pt",
}


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    config_path = resolve_path(root_dir, args.config)
    apply_config(args, load_config_section(config_path, "training"), DEFAULTS)

    data_path = resolve_path(root_dir, args.data)
    if not data_path.exists():
        print(f"ERROR: dataset yaml not found: {data_path}", file=sys.stderr)
        return 2

    if YOLO is None:
        print(f"ERROR: ultralytics is not installed: {YOLO_IMPORT_ERROR}", file=sys.stderr)
        return 2

    project_dir = resolve_path(root_dir, args.project)
    resolved_data_path = prepare_dataset_yaml(root_dir, data_path, project_dir)

    model = YOLO(args.model)
    train_kwargs = build_train_kwargs(args, root_dir, resolved_data_path, project_dir)
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", ""))
    if not save_dir:
        save_dir = Path(train_kwargs["project"]) / str(train_kwargs["name"])
    if not save_dir.is_absolute():
        save_dir = root_dir / save_dir

    best_weight = save_dir / "weights" / "best.pt"
    if args.copy_best_to and best_weight.exists():
        destination = resolve_path(root_dir, args.copy_best_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, destination)
        print(f"Copied best weights: {best_weight} -> {destination}")
    else:
        print(f"Training finished. Best weights should be under: {save_dir / 'weights'}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for handwritten digit detection.")
    parser.add_argument("--config", default="yolo/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--data", default=DEFAULTS["data"], help="Dataset YAML path.")
    parser.add_argument("--model", default=DEFAULTS["model"], help="Base model, e.g. yolov8n.pt or yolo26n.pt.")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--batch", default=DEFAULTS["batch"], help="Batch size. Use -1 for auto batch.")
    parser.add_argument("--device", default=DEFAULTS["device"], help="CUDA device like 0, cpu, mps, or blank for auto.")
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--project", default=DEFAULTS["project"])
    parser.add_argument("--name", default=DEFAULTS["name"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--optimizer", default=DEFAULTS["optimizer"])
    parser.add_argument("--lr0", type=float, default=DEFAULTS["lr0"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--cache", default=DEFAULTS["cache"], action=argparse.BooleanOptionalAction)
    parser.add_argument("--exist-ok", default=DEFAULTS["exist_ok"], action=argparse.BooleanOptionalAction)
    parser.add_argument("--resume", default=DEFAULTS["resume"], action=argparse.BooleanOptionalAction)
    parser.add_argument("--close-mosaic", type=int, default=DEFAULTS["close_mosaic"])
    parser.add_argument("--copy-best-to", default=DEFAULTS["copy_best_to"])
    return parser.parse_args()


def apply_config(args: argparse.Namespace, config: dict, defaults: dict) -> None:
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)


def build_train_kwargs(
    args: argparse.Namespace,
    root_dir: Path,
    data_path: Path,
    project_dir: Path,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": parse_batch(args.batch),
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "cache": args.cache,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
        "close_mosaic": args.close_mosaic,
        "plots": True,
    }
    if args.device:
        kwargs["device"] = parse_device(args.device)
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0
    return kwargs


def prepare_dataset_yaml(root_dir: Path, data_path: Path, project_dir: Path) -> Path:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to resolve the dataset YAML path.") from exc

    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_root = data.get("path")
    if dataset_root:
        dataset_root_path = Path(str(dataset_root))
        if not dataset_root_path.is_absolute():
            data["path"] = str((root_dir / dataset_root_path).resolve())

    project_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = project_dir / "_resolved_dataset.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return resolved_path


def resolve_path(root_dir: Path, path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return root_dir / value


def parse_batch(value: str | int | float) -> int | float:
    if isinstance(value, (int, float)):
        return value
    try:
        as_int = int(str(value))
    except ValueError:
        return float(str(value))
    return as_int


def parse_device(value: str | int | list[int]) -> str | int | list[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [int(item) for item in value]
    if "," not in str(value):
        try:
            return int(str(value))
        except ValueError:
            return value
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
