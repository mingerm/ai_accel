#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
pipeline_start_ms="$(python3 -c 'import time; print(f"{time.perf_counter() * 1000.0:.3f}")')"

if [[ -n "${MNIST_PROFILE_CSV:-}" ]]; then
  mkdir -p "$(dirname "$MNIST_PROFILE_CSV")"
fi

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

batch_predictions=()
batch_times=()
use_batch=0
if [[ "${MNIST_BATCH:-1}" == "1" && -z "${MNIST_COMMAND:-}" && -z "${MNIST_INPUT_PATH:-}" && -x "${MNIST_BIN:-./mnistCUDNN}" && ${#pgm_files[@]} -gt 0 ]]; then
  list_file="$(mktemp)"
  for ((i = 0; i < total_images && i < ${#pgm_files[@]}; i++)); do
    printf '%s\n' "${pgm_files[$i]}" >> "$list_file"
  done

  batch_args=(images="$list_file")
  if [[ -n "${MNIST_PROFILE_CSV:-}" ]]; then
    batch_args+=(profile="$MNIST_PROFILE_CSV")
    batch_args+=(phase="${MNIST_PROFILE_PHASE:-optimized}")
  fi

  mnist_batch_output="$("${MNIST_BIN:-./mnistCUDNN}" "${batch_args[@]}" 2>&1 || true)"
  while IFS=$'\t' read -r _input prediction latency; do
    batch_predictions+=("$prediction")
    batch_times+=("$latency")
  done < <(printf '%s\n' "$mnist_batch_output" | python3 scripts/parse_mnist_batch_output.py)
  rm -f "$list_file"

  if [[ ${#batch_predictions[@]} -gt 0 ]]; then
    use_batch=1
  else
    echo "WARNING: batch MNIST runner produced no parseable results; falling back to per-image runner" >&2
  fi
fi

correct=0
total_ms="0.000"
declare -A total_by_digit=()
declare -A correct_by_digit=()

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
  if [[ "$use_batch" == "1" && $i -lt ${#batch_predictions[@]} ]]; then
    prediction="${batch_predictions[$i]}"
    infer_ms="${batch_times[$i]}"
  else
    mnist_output="$(bash scripts/run_mnist_one.sh "$pgm" 2>&1 || true)"
    parsed="$(printf '%s\n' "$mnist_output" | python3 scripts/parse_mnist_output.py)"
    read -r prediction infer_ms <<<"$parsed"
  fi

  if [[ "$has_answers" == "1" ]]; then
    answer="${answers[$i]}"
    total_by_digit["$answer"]=$(( ${total_by_digit["$answer"]:-0} + 1 ))
    result="X"
    if [[ "$prediction" == "$answer" ]]; then
      result="O"
      correct=$((correct + 1))
      correct_by_digit["$answer"]=$(( ${correct_by_digit["$answer"]:-0} + 1 ))
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

avg_ms="$(awk "BEGIN {if ($total_images > 0) printf \"%.3f\", $total_ms / $total_images; else printf \"0.000\"}")"
pipeline_end_ms="$(python3 -c 'import time; print(f"{time.perf_counter() * 1000.0:.3f}")')"
pipeline_wall_ms="$(awk "BEGIN {printf \"%.3f\", $pipeline_end_ms - $pipeline_start_ms}")"
accuracy="$(awk "BEGIN {if ($total_images > 0) printf \"%.6f\", $correct / $total_images; else printf \"0.000000\"}")"

echo "============= SUMMARY ============="
echo "Total Images              : $total_images"
echo "Correct Predictions       : $correct"
echo "Total Inference Time      : $total_ms ms"
echo "Average Latency           : $avg_ms ms"
echo "Pipeline Wall Time        : $pipeline_wall_ms ms"
echo "==================================="

if [[ -n "${PIPELINE_REPORT:-}" ]]; then
  mkdir -p "$(dirname "$PIPELINE_REPORT")"
  {
    echo "execution_environment: $(uname -a)"
    echo "mode: $([[ "$use_batch" == "1" ]] && echo batch || echo per_image)"
    echo "modified_files: mnistCUDNN.cpp, ocr.py, scripts/run_pipeline.sh, scripts/parse_mnist_batch_output.py"
    echo "modified_functions: network_t::classify_example, network_t::convoluteForward, network_t::resize, main"
    echo "optimization_purpose: reuse model handles, weights, CUDA buffers, and convolution workspace across repeated PGM inference"
    echo "total_images: $total_images"
    echo "correct_predictions: $correct"
    echo "accuracy: $accuracy"
    echo "total_inference_time_ms: $total_ms"
    echo "average_latency_ms: $avg_ms"
    echo "pipeline_wall_time_ms: $pipeline_wall_ms"
    echo "mnist_profile_csv: ${MNIST_PROFILE_CSV:-}"
    if [[ "$has_answers" == "1" ]]; then
      for digit in "${!total_by_digit[@]}"; do
        digit_total="${total_by_digit[$digit]}"
        digit_correct="${correct_by_digit[$digit]:-0}"
        digit_accuracy="$(awk "BEGIN {if ($digit_total > 0) printf \"%.6f\", $digit_correct / $digit_total; else printf \"0.000000\"}")"
        echo "digit_${digit}_accuracy: $digit_correct/$digit_total ($digit_accuracy)"
      done | sort
    fi
  } > "$PIPELINE_REPORT"
fi
