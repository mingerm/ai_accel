#!/usr/bin/env bash
set -euo pipefail

# Optional ground truth for demo scoring.
# For evaluation, fill this array at the top of the script:
# answers=(6 8 8 6 ...)
#
# If the array is left empty and answers.txt exists, whitespace-separated digits
# in that file are used as a local fallback. If both are empty, the pipeline
# still runs and prints predictions with unknown answers.
answers=()

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${DEMO_PREFLIGHT:-1}" == "1" ]]; then
  bash scripts/demo_preflight.sh
fi

if [[ ${#answers[@]} -eq 0 && -f answers.txt ]]; then
  # shellcheck disable=SC2207
  answers=($(tr -s '[:space:]' ' ' < answers.txt))
fi

source scripts/run_pipeline.sh
