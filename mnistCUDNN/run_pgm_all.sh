#!/usr/bin/env bash
set -euo pipefail

# The evaluator can replace this array before the demo.
# Example: answers=(6 8 8 6)
answers=()

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source scripts/run_pipeline.sh
