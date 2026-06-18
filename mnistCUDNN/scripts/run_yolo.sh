#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${YOLO_CONFIG:-yolo/config.yaml}"
ARGS=(--config "$CONFIG")
SCRIPT="yolo/detect_video.py"

if [[ -n "${IMAGE_PATH:-}" ]]; then
  SCRIPT="yolo/detect_images.py"
  ARGS+=(--image "$IMAGE_PATH")
elif [[ -n "${IMAGE_DIR:-}" ]]; then
  SCRIPT="yolo/detect_images.py"
  ARGS+=(--image-dir "$IMAGE_DIR")
elif [[ -n "${YOLO_SOURCE:-}" ]]; then
  ARGS+=(--source "$YOLO_SOURCE")
elif [[ -n "${CAMERA_INDEX:-}" ]]; then
  ARGS+=(--source "$CAMERA_INDEX")
elif [[ -n "${VIDEO_PATH:-}" ]]; then
  ARGS+=(--video "$VIDEO_PATH")
fi

if [[ "$SCRIPT" == "yolo/detect_images.py" && -n "${IMAGE_RECURSIVE:-}" ]]; then
  if [[ "$IMAGE_RECURSIVE" == "0" || "$IMAGE_RECURSIVE" == "false" || "$IMAGE_RECURSIVE" == "False" ]]; then
    ARGS+=(--no-recursive)
  else
    ARGS+=(--recursive)
  fi
fi

if [[ -n "${YOLO_WEIGHTS:-}" ]]; then
  ARGS+=(--weights "$YOLO_WEIGHTS")
fi

if [[ -n "${PGM_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "$PGM_OUTPUT_DIR")
fi

if [[ -n "${EXPECTED_COUNT:-}" ]]; then
  ARGS+=(--expected-count "$EXPECTED_COUNT")
fi

python3 "$SCRIPT" "${ARGS[@]}" "$@"
