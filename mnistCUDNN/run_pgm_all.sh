#!/usr/bin/env bash
set -euo pipefail

# Optional ground truth for demo scoring.
# If answers.txt exists, whitespace-separated digits in that file are used.
# Otherwise the pipeline still runs and prints predictions with unknown answers.
# Example answers.txt content: 5 6 5 6
answers=()

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${DEMO_PREFLIGHT:-1}" == "1" ]]; then
  bash scripts/demo_preflight.sh
fi

if [[ -f answers.txt ]]; then
  # shellcheck disable=SC2207
  answers=($(tr -s '[:space:]' ' ' < answers.txt))
fi

source scripts/run_pipeline.sh
