#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

info() {
  echo "[demo] $*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "'$1' command not found"
}

resolve_demo_paths() {
  python3 - "$ROOT_DIR" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / os.environ.get("YOLO_CONFIG", "yolo/config.yaml")

data = {}
try:
    import yaml
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
except Exception:
    pass

source = (
    os.environ.get("YOLO_SOURCE")
    or os.environ.get("CAMERA_INDEX")
    or os.environ.get("VIDEO_PATH")
    or data.get("source")
    or data.get("video")
    or "0"
)
weights = os.environ.get("YOLO_WEIGHTS") or data.get("weights") or "yolo/weights/best.pt"
out_dir = os.environ.get("PGM_OUTPUT_DIR") or data.get("output_dir") or "pgm_output"

def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value

source_text = str(source).strip()
if source_text.startswith("camera:") or source_text.isdigit():
    print(source_text)
    print("camera")
else:
    print(resolve(source_text))
    print("file")
print(resolve(str(weights)))
print(resolve(str(out_dir)))
PY
}

check_python_deps() {
  python3 - <<'PY'
missing = []
for module in ("cv2", "numpy", "yaml", "ultralytics"):
    try:
        __import__(module)
    except Exception:
        missing.append(module)

if missing:
    print("missing Python modules: " + ", ".join(missing))
    raise SystemExit(1)
PY
}

build_mnist_if_needed() {
  local mnist_bin="${MNIST_BIN:-./mnistCUDNN}"
  if [[ "$mnist_bin" != /* ]]; then
    mnist_bin="$ROOT_DIR/${mnist_bin#./}"
  fi

  if [[ -x "$mnist_bin" ]]; then
    info "mnistCUDNN binary: $mnist_bin"
    return
  fi

  info "mnistCUDNN binary not found; building with Makefile"
  if [[ ! -x "${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" ]] && ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc not found. Set CUDA_PATH or install CUDA Toolkit before the demo."
  fi

  make ${MAKE_ARGS:-}

  [[ -x "$mnist_bin" ]] || fail "mnistCUDNN build did not create executable: $mnist_bin"
}

require_command python3
require_command awk
require_command find

if ! check_python_deps; then
  fail "Python dependencies are missing. Install once with: python3 -m pip install -r requirements-yolo.txt"
fi

mapfile -t resolved_paths < <(resolve_demo_paths)
source_value="${resolved_paths[0]}"
source_kind="${resolved_paths[1]}"
weights_path="${resolved_paths[2]}"
output_dir="${resolved_paths[3]}"

if [[ "$source_kind" == "file" ]]; then
  [[ -f "$source_value" ]] || fail "demo video/source file not found: $source_value"
fi
[[ -f "$weights_path" ]] || fail "YOLO weights not found: $weights_path"
mkdir -p "$output_dir"

build_mnist_if_needed

info "source: $source_value ($source_kind)"
info "weights: $weights_path"
info "output_dir: $output_dir"
info "preflight complete"
