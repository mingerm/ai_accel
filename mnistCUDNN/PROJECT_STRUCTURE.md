# Project Structure

For an easy code walkthrough, see `CODE_EXPLANATION.md`.

This project keeps the original mnistCUDNN files unchanged and adds a wrapper
pipeline around them.

```text
mnistCUDNN/
├── Makefile
├── ocr.py
├── run_pgm_all.sh
├── yolo/
│   ├── config.yaml
│   ├── dataset.yaml
│   ├── detect_video.py
│   ├── dedupe.py
│   ├── make_yolo_dataset.py
│   ├── save_pgm.py
│   ├── train_yolo.py
│   ├── train_config.yaml
│   └── weights/
│       └── best.pt
├── datasets/
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── scripts/
│   ├── clean_outputs.sh
│   ├── parse_mnist_output.py
│   ├── run_mnist_one.sh
│   ├── run_pipeline.sh
│   └── run_yolo.sh
├── videos/
│   └── input.mp4
└── pgm_output/
```

## Runtime Flow

```text
./run_pgm_all.sh
  -> scripts/clean_outputs.sh
  -> scripts/run_yolo.sh
  -> yolo/detect_video.py
  -> pgm_output/*.pgm
  -> scripts/run_mnist_one.sh for each PGM
  -> summary
```

## Important Files

- `run_pgm_all.sh`: entry point for the demonstration.
- `yolo/config.yaml`: video, YOLO weight, PGM output, and duplicate filtering settings.
- `yolo/detect_video.py`: detects digits from video and saves segment-level PGM files.
- `yolo/dedupe.py`: saves only one PGM per visible digit segment.
- `yolo/save_pgm.py`: converts detected crops into 28x28 MNIST-style PGM files.
- `scripts/run_mnist_one.sh`: runs the existing MNIST code for one PGM image.

## Duplicate Save Rule

The tracker saves a digit when a stable new segment is detected. Repeated
detections across adjacent frames are ignored while the same digit remains
active. After enough missing frames, the segment closes and a later detection of
the same number can be saved again.

## Submission Rule

Submit code, scripts, configuration, and trained weights. Do not submit generated
`pgm_output/*.pgm` files.
