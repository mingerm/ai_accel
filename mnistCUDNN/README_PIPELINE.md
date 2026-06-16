# YOLO to mnistCUDNN Pipeline

For a code-level explanation of each file, see `CODE_EXPLANATION.md`.

## Setup

1. Put the handwritten digit video at `videos/input.mp4`.
2. Put the trained YOLO model at `yolo/weights/best.pt`.
3. Install Python dependencies if needed:

```bash
pip install -r requirements-yolo.txt
```

4. Edit the answer array at the top of `run_pgm_all.sh` before evaluation:

```bash
answers=(6 8 8 6)
```

5. Make scripts executable once:

```bash
chmod +x run_pgm_all.sh scripts/*.sh
```

## Run

```bash
./run_pgm_all.sh
```

The script runs YOLO first, writes generated images into `pgm_output/`, then
evaluates only the first `answers` count of PGM files.

## Custom Paths

```bash
VIDEO_PATH=/path/to/demo.mp4 ./run_pgm_all.sh
YOLO_WEIGHTS=/path/to/best.pt ./run_pgm_all.sh
PGM_OUTPUT_DIR=pgm_output ./run_pgm_all.sh
```

## MNIST Runner Detection

`scripts/run_mnist_one.sh` tries the following order:

1. `MNIST_COMMAND`, if provided
2. `python3 ocr.py <pgm>`
3. `./mnistCUDNN image=<pgm>`
4. `make`, then `./mnistCUDNN image=<pgm>`

For a fixed custom command, use `{pgm}` as the image placeholder:

```bash
MNIST_COMMAND='./mnistCUDNN image={pgm}' ./run_pgm_all.sh
```

If the original code reads a fixed input path instead of a command-line PGM,
copy each generated PGM into that path before running:

```bash
MNIST_INPUT_PATH=data/input.pgm MNIST_COMMAND='./mnistCUDNN image={pgm}' ./run_pgm_all.sh
```

## Duplicate Filtering

Tune these values in `yolo/config.yaml`:

- `min_stable_frames`: frames required before saving a new digit.
- `missing_frames_to_close_segment`: frames without detection before a segment ends.
- `min_frames_between_saves`: guard interval between saved PGM files.
- `allow_same_label_motion_split`: optional handling for repeated same digits at
  different board positions.

## Output Format

The final output is printed in the requested format:

```text
================================
INPUT: pgm_output/frame_000095_digit_6_conf_0.79_0000.pgm
정답: 6 , 추론: 6
Inference time: 439.939 ms
결과: O
...
============= SUMMARY =============
Total Images              : 12
Correct Predictions       : 11
Total Inference Time      : 2927.677 ms
===================================
```
