# YOLO to mnistCUDNN Pipeline

For a code-level explanation of each file, see `CODE_EXPLANATION.md`.

## Setup

1. Connect the final demo camera. By default `yolo/config.yaml` uses camera
   source `0`.
2. Put the trained YOLO detector at `yolo/weights/best.pt`.
   `yolo11n.pt` is the base checkpoint used by the training config; the live
   demo still expects the trained detector path unless `YOLO_WEIGHTS` is set.
3. Install Python dependencies if needed:

```bash
pip install -r requirements-yolo.txt
```

4. Before evaluation, fill the answer array at the top of `run_pgm_all.sh`:

```bash
answers=(6 8 8 6)
```

If that array is left empty, `run_pgm_all.sh` can read whitespace-separated
digits from `answers.txt` as a local fallback. If both are empty, the pipeline
still runs and prints predictions with unknown answers.

5. Make scripts executable once:

```bash
chmod +x run_pgm_all.sh scripts/*.sh
```

## Train YOLO

First generate a YOLO-format dataset from digit source images. The script looks
for digit folders such as `../1/`, `../3/`, `../5/`, `../6/`, and `../8/` when
run from `mnistCUDNN/`, and can also use the built-in sample PGM files.

```bash
python3 yolo/make_yolo_dataset.py
```

Dataset generation options are read from `yolo/train_config.yaml`.

Put YOLO-format training data under `datasets/yolo/`:

```text
datasets/yolo/images/train/
datasets/yolo/images/val/
datasets/yolo/labels/train/
datasets/yolo/labels/val/
```

Each label file must use normalized YOLO boxes:

```text
class_id x_center y_center width height
```

Then run:

```bash
python3 yolo/train_yolo.py
```

The script copies the best checkpoint to `yolo/weights/best.pt` by default, so
`./run_pgm_all.sh` can use it directly.

## Run

```bash
./run_pgm_all.sh
```

The script runs the demo preflight, opens the camera or configured source,
writes generated PGM files into `pgm_output/`, then evaluates only the
`answers.txt` count of PGM files when answers are provided.

For a camera-only final check:

```bash
CAMERA_INDEX=0 ./run_pgm_all.sh
```

For a recorded video regression check:

```bash
VIDEO_PATH=videos/input.mp4 ./run_pgm_all.sh
```

The same video path can be launched through `ocr.py`, which delegates to the
same shell pipeline:

```bash
python3 ocr.py --video videos/input.mp4
```

For the report, use the sample video and keep the same answer sequence used in
the shell script:

```bash
VIDEO_PATH=videos/sample.mp4 PIPELINE_REPORT=reports/sample_video_report.txt ./run_pgm_all.sh
```

## Custom Paths

```bash
CAMERA_INDEX=1 ./run_pgm_all.sh
YOLO_SOURCE=camera:1 ./run_pgm_all.sh
VIDEO_PATH=/path/to/demo.mp4 ./run_pgm_all.sh
IMAGE_DIR=/path/to/image_folder ./run_pgm_all.sh
IMAGE_PATH=/path/to/image.jpg ./run_pgm_all.sh
YOLO_WEIGHTS=/path/to/best.pt ./run_pgm_all.sh
PGM_OUTPUT_DIR=pgm_output ./run_pgm_all.sh
EXPECTED_COUNT=4 ./run_pgm_all.sh
```

For nested image folders, enable recursive image search:

```bash
IMAGE_DIR=/path/to/image_folder IMAGE_RECURSIVE=1 ./run_pgm_all.sh
```

You can also run only the YOLO image-to-PGM step:

```bash
python3 yolo/detect_images.py --image-dir /path/to/image_folder --output-dir pgm_output
```

## Train EMNIST LeNet Weights

The C++ runner reads raw float32 weight files from `data/*.bin`. Train a
compatible LeNet model in PyTorch, then export those files:

```bash
python3 -m pip install -r requirements-training.txt
python3 scripts/train_emnist_lenet.py \
  --download \
  --augment \
  --epochs 10 \
  --export-dir trained_weights/emnist_digits \
  --overwrite
```

To mix in labeled project images, place them in digit folders such as
`custom_digits/1/`, `custom_digits/3/`, `custom_digits/5/`, `custom_digits/6/`,
and `custom_digits/8/`, or use YOLO PGM filenames containing `digit_8`.

```bash
python3 scripts/train_emnist_lenet.py \
  --download \
  --augment \
  --extra-data-root custom_digits \
  --eval-data-root custom_digits \
  --epochs 10 \
  --export-dir trained_weights/emnist_custom \
  --overwrite
```

After checking metrics, export directly into the runtime `data/` directory:

```bash
python3 scripts/train_emnist_lenet.py \
  --download \
  --augment \
  --extra-data-root custom_digits \
  --eval-data-root custom_digits \
  --epochs 10 \
  --export-dir data \
  --backup-existing \
  --overwrite
```

## MNIST Runner Detection

`scripts/run_mnist_one.sh` tries the following order:

1. `MNIST_COMMAND`, if provided
2. `python3 ocr.py <pgm>`
3. `./mnistCUDNN image=<pgm>`
4. `make`, then `./mnistCUDNN image=<pgm>`

`ocr.py` also accepts video/camera inputs for compatibility with the project
specification:

```bash
python3 ocr.py --video videos/input.mp4
python3 ocr.py --source camera:0
```

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

- `require_fully_visible`, `min_edge_margin`: wait until a digit is away from the frame edge before saving.
- `require_centered`, `center_x_min`, `center_x_max`: wait until a digit reaches the center region before saving.
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

## Report Artifacts

The pipeline can write machine-readable notes for the report:

```bash
PIPELINE_REPORT=reports/final_pipeline_report.txt ./run_pgm_all.sh
```

Report accuracy and time should be recorded from the sample video run, not from
old generated PGM files.

For MNIST baseline-vs-optimized timing comparison, first keep a set of PGM files
under `pgm_output/`, then run:

```bash
bash scripts/profile_mnist_pipeline.sh
```

The profiling script writes CSV, stdout logs, pipeline summaries, and
`mnist_optimization_report.md` under `reports/`.
