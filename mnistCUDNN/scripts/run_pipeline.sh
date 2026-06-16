#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! declare -p answers >/dev/null 2>&1; then
  answers=()
fi

if [[ "${RUN_YOLO:-1}" == "1" ]]; then
  if [[ -z "${EXPECTED_COUNT:-}" && ${#answers[@]} -gt 0 ]]; then
    export EXPECTED_COUNT="${#answers[@]}"
  fi
  bash scripts/clean_outputs.sh
  bash scripts/run_yolo.sh
fi

OUT_DIR="${PGM_OUTPUT_DIR:-pgm_output}"
if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: PGM output directory not found: $OUT_DIR" >&2
  exit 2
fi

pgm_files=()
while IFS= read -r pgm_file; do
  pgm_files+=("$pgm_file")
done < <(find "$OUT_DIR" -maxdepth 1 -type f -name '*.pgm' | sort)

if [[ ${#answers[@]} -gt 0 ]]; then
  has_answers=1
  total_images=${#answers[@]}
else
  has_answers=0
  total_images=${#pgm_files[@]}
fi

correct=0
total_ms="0.000"

for ((i = 0; i < total_images; i++)); do
  if [[ $i -ge ${#pgm_files[@]} ]]; then
    echo "================================"
    echo "INPUT: MISSING"
    echo "정답: ${answers[$i]:-?} , 추론: ?"
    echo "Inference time: 0.000 ms"
    echo "결과: X"
    continue
  fi

  pgm="${pgm_files[$i]}"
  mnist_output="$(bash scripts/run_mnist_one.sh "$pgm" 2>&1 || true)"
  parsed="$(printf '%s\n' "$mnist_output" | python3 scripts/parse_mnist_output.py)"
  read -r prediction infer_ms <<<"$parsed"

  if [[ "$has_answers" == "1" ]]; then
    answer="${answers[$i]}"
    result="X"
    if [[ "$prediction" == "$answer" ]]; then
      result="O"
      correct=$((correct + 1))
    fi
  else
    answer="?"
    result="-"
  fi

  total_ms="$(awk "BEGIN {printf \"%.3f\", $total_ms + $infer_ms}")"

  echo "================================"
  echo "INPUT: $pgm"
  echo "정답: $answer , 추론: $prediction"
  echo "Inference time: $infer_ms ms"
  echo "결과: $result"
done

echo "============= SUMMARY ============="
echo "Total Images              : $total_images"
echo "Correct Predictions       : $correct"
echo "Total Inference Time      : $total_ms ms"
echo "==================================="
