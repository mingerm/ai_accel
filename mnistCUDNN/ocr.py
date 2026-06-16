from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python3 ocr.py <image.pgm>", file=sys.stderr)
        return 2

    root_dir = Path(__file__).resolve().parent
    image_path = Path(sys.argv[1])
    if not image_path.is_absolute():
        image_path = (root_dir / image_path).resolve()

    if not image_path.exists():
        print(f"ERROR: PGM image not found: {image_path}", file=sys.stderr)
        return 2

    mnist_bin = Path(os.environ.get("MNIST_BIN", "./mnistCUDNN"))
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

    result = subprocess.run([str(mnist_bin), f"image={image_path}"], cwd=root_dir)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
