from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None

from dedupe import Detection, SegmentTracker
from save_pgm import crop_digit, digit_to_mnist_pgm, save_pgm


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    video_path = Path(args.video or str(config.get("video", "videos/input.mp4")))
    weights_path = Path(args.weights or str(config.get("weights", "yolo/weights/best.pt")))
    output_dir = Path(args.output_dir or str(config.get("output_dir", "pgm_output")))
    expected_count = args.expected_count

    if YOLO is None:
        print(f"ERROR: ultralytics is not installed: {YOLO_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(f"ERROR: YOLO weights not found: {weights_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"ERROR: could not open video: {video_path}", file=sys.stderr)
        return 2

    tracker = SegmentTracker(
        min_stable_frames=int(config.get("min_stable_frames", 2)),
        missing_frames_to_close_segment=int(config.get("missing_frames_to_close_segment", 5)),
        min_frames_between_saves=int(config.get("min_frames_between_saves", 8)),
        allow_same_label_motion_split=as_bool(config.get("allow_same_label_motion_split", False)),
        same_label_center_distance=float(config.get("same_label_center_distance", 0.45)),
    )

    frame_index = 0
    saved = 0
    confidence = float(config.get("confidence", 0.45))
    iou = float(config.get("iou", 0.45))
    imgsz = int(config.get("imgsz", 640))
    device = str(config.get("device", "") or "")
    target_size = int(config.get("target_size", 28))
    digit_box_size = int(config.get("digit_box_size", 20))
    bbox_padding = float(config.get("bbox_padding", 0.18))
    selection = str(config.get("selection", "center_conf"))

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detection = detect_one(
            model=model,
            frame=frame,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            device=device,
            selection=selection,
        )
        height, width = frame.shape[:2]
        decision = tracker.update(detection, width, height)

        if decision.should_save and detection is not None:
            crop = crop_digit(frame, detection.bbox, padding=bbox_padding)
            pgm_image = digit_to_mnist_pgm(
                crop,
                target_size=target_size,
                digit_box_size=digit_box_size,
            )
            name = (
                f"frame_{frame_index:06d}_digit_{detection.label}_"
                f"conf_{detection.confidence:.2f}_{decision.segment_index:04d}.pgm"
            )
            save_pgm(output_dir / name, pgm_image)
            saved += 1
            print(
                f"SAVED {output_dir / name} "
                f"label={detection.label} conf={detection.confidence:.3f} "
                f"reason={decision.reason}"
            )

            if expected_count is not None and saved >= expected_count:
                break

        frame_index += 1

    capture.release()
    print(f"YOLO summary: frames={frame_index} saved={saved} output_dir={output_dir}")
    return 0


def detect_one(
    model: Any,
    frame: Any,
    confidence: float,
    iou: float,
    imgsz: int,
    device: str,
    selection: str,
) -> Optional[Detection]:
    kwargs: Dict[str, Any] = {
        "conf": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    result = model.predict(frame, **kwargs)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    detections: List[Detection] = []
    names = getattr(result, "names", {}) or {}
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_to_digit(cls_id, names)
        if label is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))

    if not detections:
        return None

    return select_detection(detections, frame.shape[1], frame.shape[0], selection)


def select_detection(
    detections: Iterable[Detection],
    frame_width: int,
    frame_height: int,
    selection: str,
) -> Detection:
    items = list(detections)
    if selection == "confidence":
        return max(items, key=lambda d: d.confidence)
    if selection == "center":
        return min(items, key=lambda d: center_distance(d.bbox, frame_width, frame_height))
    return max(
        items,
        key=lambda d: d.confidence - 0.25 * center_distance(d.bbox, frame_width, frame_height),
    )


def center_distance(bbox: tuple[float, float, float, float], frame_width: int, frame_height: int) -> float:
    cx = (bbox[0] + bbox[2]) * 0.5 / max(1, frame_width)
    cy = (bbox[1] + bbox[3]) * 0.5 / max(1, frame_height)
    dx = cx - 0.5
    dy = cy - 0.5
    return (dx * dx + dy * dy) ** 0.5


def class_to_digit(cls_id: int, names: Dict[int, str]) -> Optional[int]:
    name = str(names.get(cls_id, cls_id)).strip()
    if name.isdigit() and 0 <= int(name) <= 9:
        return int(name)
    if 0 <= cls_id <= 9:
        return cls_id
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect handwritten digits and save MNIST PGM crops.")
    parser.add_argument("--config", default="yolo/config.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    try:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return dict(data)
    except ImportError:
        return parse_simple_yaml(config_path)


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = parse_scalar(value.strip())
    return data


def parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
