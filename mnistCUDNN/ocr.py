from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    if should_run_yolo_pipeline(args):
        return run_yolo_pipeline(args, root_dir)

    if args.image is None:
        print("usage: python3 ocr.py <image.pgm> OR python3 ocr.py --video <video.mp4>", file=sys.stderr)
        return 2

    return run_mnist_image(args.image, root_dir, args.mnist_bin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run handwritten digit OCR from a PGM image, video, image folder, or camera."
    )
    parser.add_argument("image", nargs="?", help="PGM image to classify with mnistCUDNN.")
    parser.add_argument("--video", help="Video file to process through the full YOLO-to-MNIST pipeline.")
    parser.add_argument("--source", help="YOLO source: camera index, camera:0, or video path.")
    parser.add_argument("--camera-index", help="Camera index for live capture, for example 0.")
    parser.add_argument("--image-path", help="Still image to process through YOLO.")
    parser.add_argument("--image-dir", help="Image directory to process through YOLO.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --image-dir.")
    parser.add_argument("--config", help="YOLO config path.")
    parser.add_argument("--yolo-weights", help="YOLO detector weights path.")
    parser.add_argument("--output-dir", help="PGM output directory.")
    parser.add_argument("--expected-count", type=int, help="Stop after this many saved PGM files.")
    parser.add_argument("--mnist-bin", help="mnistCUDNN binary path.")
    parser.add_argument("--no-preflight", action="store_true", help="Skip demo preflight checks.")
    return parser.parse_args()


def should_run_yolo_pipeline(args: argparse.Namespace) -> bool:
    return any(
        (
            args.video,
            args.source,
            args.camera_index,
            args.image_path,
            args.image_dir,
        )
    )


def run_mnist_image(image: str, root_dir: Path, mnist_bin_arg: str | None = None) -> int:
    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = (root_dir / image_path).resolve()

    if not image_path.exists():
        print(f"ERROR: PGM image not found: {image_path}", file=sys.stderr)
        return 2

    mnist_bin = Path(mnist_bin_arg or os.environ.get("MNIST_BIN", "./mnistCUDNN"))
    if not mnist_bin.is_absolute():
        mnist_bin = root_dir / mnist_bin

    if not mnist_bin.exists() and (root_dir / "Makefile").exists():
        build = subprocess.run(["make"], cwd=root_dir)
        if build.returncode != 0:
            return build.returncode

    if not mnist_bin.exists():
        print(
            "ERROR: mnistCUDNN binary not found. Build it with make or set MNIST_BIN.",
            file=sys.stderr,
        )
        return 2

    command = [str(mnist_bin), f"image={image_path}"]
    profile_csv = os.environ.get("MNIST_PROFILE_CSV")
    if profile_csv:
        command.append(f"profile={profile_csv}")
    profile_phase = os.environ.get("MNIST_PROFILE_PHASE")
    if profile_phase:
        command.append(f"phase={profile_phase}")

    result = subprocess.run(command, cwd=root_dir)
    return result.returncode


def run_yolo_pipeline(args: argparse.Namespace, root_dir: Path) -> int:
    env = os.environ.copy()
    set_env(env, "VIDEO_PATH", args.video)
    set_env(env, "YOLO_SOURCE", args.source)
    set_env(env, "CAMERA_INDEX", args.camera_index)
    set_env(env, "IMAGE_PATH", args.image_path)
    set_env(env, "IMAGE_DIR", args.image_dir)
    set_env(env, "YOLO_CONFIG", args.config)
    set_env(env, "YOLO_WEIGHTS", args.yolo_weights)
    set_env(env, "PGM_OUTPUT_DIR", args.output_dir)
    set_env(env, "MNIST_BIN", args.mnist_bin)
    if args.expected_count is not None:
        env["EXPECTED_COUNT"] = str(args.expected_count)
    if args.recursive:
        env["IMAGE_RECURSIVE"] = "1"
    if args.no_preflight:
        env["DEMO_PREFLIGHT"] = "0"

    script = root_dir / "run_pgm_all.sh"
    if not script.exists():
        print(f"ERROR: pipeline script not found: {script}", file=sys.stderr)
        return 2

    result = subprocess.run(["bash", str(script)], cwd=root_dir, env=env)
    return result.returncode


def set_env(env: dict[str, str], key: str, value: str | None) -> None:
    if value:
        env[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
