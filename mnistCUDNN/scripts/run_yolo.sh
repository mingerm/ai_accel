#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${YOLO_CONFIG:-yolo/config.yaml}"
ARGS=(--config "$CONFIG")

if [[ -n "${YOLO_SOURCE:-}" ]]; then
  ARGS+=(--source "$YOLO_SOURCE")
elif [[ -n "${CAMERA_INDEX:-}" ]]; then
  ARGS+=(--source "$CAMERA_INDEX")
elif [[ -n "${VIDEO_PATH:-}" ]]; then
  ARGS+=(--video "$VIDEO_PATH")
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

python3 yolo/detect_video.py "${ARGS[@]}" "$@"
