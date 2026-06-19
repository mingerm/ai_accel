from __future__ import annotations

import argparse
import sys
import time
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

    source = choose_source(args, config)
    weights_path = Path(args.weights or str(config.get("weights", "yolo/weights/best.pt")))
    output_dir = Path(args.output_dir or str(config.get("output_dir", "pgm_output")))
    expected_count = args.expected_count
    display = args.display if args.display is not None else as_bool(config.get("display", True))
    window_title = str(config.get("window_title", "YOLO Live"))
    camera_width = int(config.get("camera_width", 0) or 0)
    camera_height = int(config.get("camera_height", 0) or 0)
    camera_fps = int(config.get("camera_fps", 0) or 0)
    max_frames = int(config.get("max_frames", 0) or 0)

    if YOLO is None:
        print(f"ERROR: ultralytics is not installed: {YOLO_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if isinstance(source, Path) and not source.exists():
        print(f"ERROR: video/source file not found: {source}", file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(f"ERROR: YOLO weights not found: {weights_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    capture_source = int(source) if isinstance(source, int) else str(source)
    capture = cv2.VideoCapture(capture_source)
    if camera_width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    if camera_height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    if camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, camera_fps)
    if not capture.isOpened():
        print(f"ERROR: could not open source: {source}", file=sys.stderr)
        return 2

    tracker = SegmentTracker(
        min_stable_frames=int(config.get("min_stable_frames", 2)),
        missing_frames_to_close_segment=int(config.get("missing_frames_to_close_segment", 5)),
        min_frames_between_saves=int(config.get("min_frames_between_saves", 8)),
        allow_same_label_motion_split=as_bool(config.get("allow_same_label_motion_split", False)),
        same_label_center_distance=float(config.get("same_label_center_distance", 0.45)),
        label_change_center_distance=float(config.get("label_change_center_distance", 0.18)),
    )

    frame_index = 0
    saved = 0
    prev_time = 0.0
    confidence = float(config.get("confidence", 0.45))
    iou = float(config.get("iou", 0.45))
    imgsz = int(config.get("imgsz", 640))
    device = str(config.get("device", "") or "")
    target_size = int(config.get("target_size", 28))
    digit_box_size = int(config.get("digit_box_size", 20))
    bbox_padding = float(config.get("bbox_padding", 0.18))
    selection = str(config.get("selection", "center_conf"))
    save_debug_crops = as_bool(config.get("save_debug_crops", False))
    save_gate = SaveGate(
        require_fully_visible=as_bool(config.get("require_fully_visible", False)),
        min_edge_margin=float(config.get("min_edge_margin", 0.0) or 0.0),
        require_centered=as_bool(config.get("require_centered", False)),
        center_x_min=float(config.get("center_x_min", 0.0) or 0.0),
        center_x_max=float(config.get("center_x_max", 1.0) or 1.0),
        center_y_min=float(config.get("center_y_min", 0.0) or 0.0),
        center_y_max=float(config.get("center_y_max", 1.0) or 1.0),
    )

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
        gated_detection, gate_reason = save_gate.apply(detection, width, height)
        decision = tracker.update(gated_detection, width, height)

        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now

        if decision.should_save and gated_detection is not None:
            crop = crop_digit(frame, gated_detection.bbox, padding=bbox_padding)
            pgm_image = digit_to_mnist_pgm(
                crop,
                target_size=target_size,
                digit_box_size=digit_box_size,
            )
            name = (
                f"frame_{frame_index:06d}_digit_{gated_detection.label}_"
                f"conf_{gated_detection.confidence:.2f}_{decision.segment_index:04d}.pgm"
            )
            save_pgm(output_dir / name, pgm_image)
            if save_debug_crops:
                debug_dir = output_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(name).stem
                cv2.imwrite(str(debug_dir / f"{stem}_crop.png"), crop)
                preview = cv2.resize(pgm_image, (280, 280), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(debug_dir / f"{stem}_pgm_preview.png"), preview)
            saved += 1
            print(
                f"SAVED {output_dir / name} "
                f"label={gated_detection.label} conf={gated_detection.confidence:.3f} "
                f"reason={decision.reason}"
            )

            if expected_count is not None and saved >= expected_count:
                break

        if display:
            annotated = draw_live_frame(frame, detection, fps, saved, gate_reason)
            try:
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as exc:
                print(f"WARNING: display disabled: {exc}", file=sys.stderr)
                display = False

        frame_index += 1
        if max_frames > 0 and frame_index >= max_frames:
            break

    capture.release()
    if display:
        cv2.destroyAllWindows()
    print(f"YOLO summary: frames={frame_index} saved={saved} output_dir={output_dir}")
    return 0


class SaveGate:
    def __init__(
        self,
        require_fully_visible: bool,
        min_edge_margin: float,
        require_centered: bool,
        center_x_min: float,
        center_x_max: float,
        center_y_min: float,
        center_y_max: float,
    ) -> None:
        self.require_fully_visible = require_fully_visible
        self.min_edge_margin = max(0.0, min(0.45, min_edge_margin))
        self.require_centered = require_centered
        self.center_x_min = max(0.0, min(1.0, center_x_min))
        self.center_x_max = max(0.0, min(1.0, center_x_max))
        self.center_y_min = max(0.0, min(1.0, center_y_min))
        self.center_y_max = max(0.0, min(1.0, center_y_max))

    def apply(
        self,
        detection: Optional[Detection],
        frame_width: int,
        frame_height: int,
    ) -> tuple[Optional[Detection], str]:
        if detection is None:
            return None, "no_detection"

        x1, y1, x2, y2 = detection.bbox
        width = max(1, frame_width)
        height = max(1, frame_height)

        if self.require_fully_visible:
            margin_x = width * self.min_edge_margin
            margin_y = height * self.min_edge_margin
            if x1 < margin_x or y1 < margin_y or x2 > width - margin_x or y2 > height - margin_y:
                return None, "waiting_full_digit"

        if self.require_centered:
            cx = (x1 + x2) * 0.5 / width
            cy = (y1 + y2) * 0.5 / height
            if not (self.center_x_min <= cx <= self.center_x_max and self.center_y_min <= cy <= self.center_y_max):
                return None, "waiting_center"

        return detection, "ready"


def choose_source(args: argparse.Namespace, config: Dict[str, Any]) -> int | Path:
    raw_source: Any
    if args.source is not None:
        raw_source = args.source
    elif args.video is not None:
        raw_source = args.video
    elif "source" in config:
        raw_source = config.get("source")
    else:
        raw_source = config.get("video", 0)

    if isinstance(raw_source, int):
        return raw_source

    text = str(raw_source).strip()
    if text.startswith("camera:"):
        return int(text.split(":", 1)[1])
    if text.isdigit():
        return int(text)
    return Path(text)


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


def draw_live_frame(
    frame: Any,
    detection: Optional[Detection],
    fps: float,
    saved: int,
    gate_reason: str,
) -> Any:
    annotated = frame.copy()
    if detection is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated,
        f"Saved: {saved}   q: finish",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    if gate_reason not in {"ready", "no_detection"}:
        cv2.putText(
            annotated,
            gate_reason,
            (20, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 255),
            2,
        )
    return annotated


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
    parser.add_argument("--source", default=None, help="Camera index like 0, camera:0, or a video path.")
    parser.add_argument("--video", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=None)
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
