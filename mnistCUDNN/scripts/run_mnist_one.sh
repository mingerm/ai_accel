#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_mnist_one.sh <image.pgm>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGM="$1"
cd "$ROOT_DIR"

if [[ -n "${MNIST_INPUT_PATH:-}" ]]; then
  mkdir -p "$(dirname "$MNIST_INPUT_PATH")"
  cp "$PGM" "$MNIST_INPUT_PATH"
  PGM_FOR_COMMAND="$MNIST_INPUT_PATH"
else
  PGM_FOR_COMMAND="$PGM"
fi

if [[ -n "${MNIST_COMMAND:-}" ]]; then
  COMMAND="${MNIST_COMMAND//\{pgm\}/$PGM_FOR_COMMAND}"
  eval "$COMMAND"
  exit $?
fi

if [[ -f "ocr.py" ]]; then
  python3 ocr.py "$PGM_FOR_COMMAND"
  exit $?
fi

if [[ -x "${MNIST_BIN:-./mnistCUDNN}" ]]; then
  "${MNIST_BIN:-./mnistCUDNN}" "image=$PGM_FOR_COMMAND"
  exit $?
fi

if [[ -f "Makefile" ]]; then
  make
fi

if [[ -x "${MNIST_BIN:-./mnistCUDNN}" ]]; then
  "${MNIST_BIN:-./mnistCUDNN}" "image=$PGM_FOR_COMMAND"
  exit $?
fi

echo "ERROR: no MNIST runner found. Set MNIST_COMMAND='your_command {pgm}'." >&2
exit 2
