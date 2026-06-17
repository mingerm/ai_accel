#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPORT_DIR="${REPORT_DIR:-reports/mnist_profile_$(date +%Y%m%d_%H%M%S)}"
RUN_YOLO_FOR_PROFILE="${RUN_YOLO_FOR_PROFILE:-0}"

mkdir -p "$REPORT_DIR"

if [[ ! -x "${MNIST_BIN:-./mnistCUDNN}" ]]; then
  echo "ERROR: mnistCUDNN binary not found. Build it first with make." >&2
  exit 2
fi

if [[ "$RUN_YOLO_FOR_PROFILE" != "1" ]]; then
  if ! find pgm_output -maxdepth 1 -type f -name '*.pgm' | grep -q .; then
    echo "ERROR: pgm_output has no PGM files. Run YOLO once or set RUN_YOLO_FOR_PROFILE=1." >&2
    exit 2
  fi
fi

run_case() {
  local phase="$1"
  local batch="$2"
  local profile_csv="$REPORT_DIR/${phase}_mnist_profile.csv"
  local pipeline_report="$REPORT_DIR/${phase}_pipeline_report.txt"
  local stdout_log="$REPORT_DIR/${phase}_stdout.txt"

  MNIST_BATCH="$batch" \
  MNIST_PROFILE_CSV="$profile_csv" \
  MNIST_PROFILE_PHASE="$phase" \
  PIPELINE_REPORT="$pipeline_report" \
  RUN_YOLO="$RUN_YOLO_FOR_PROFILE" \
    ./run_pgm_all.sh | tee "$stdout_log"
}

run_case baseline 0
run_case optimized 1

python3 scripts/compare_mnist_profiles.py \
  "$REPORT_DIR/baseline_pipeline_report.txt" \
  "$REPORT_DIR/optimized_pipeline_report.txt" \
  "$REPORT_DIR/mnist_optimization_report.md"

echo "Reports written under: $REPORT_DIR"
