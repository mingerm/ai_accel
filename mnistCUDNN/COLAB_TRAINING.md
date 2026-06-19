# Colab Training for 6/8 Fine-Tuning

This workflow fine-tunes the LeNet-compatible `mnistCUDNN` weights with the
small `extra/6` and `extra/8` image set while keeping the current working
`data/*.bin` weights as the starting point. The raw photos are first converted
to MNIST-like PGM images so training uses exactly the previewed data.

## 1. Make the Colab upload zip locally

Run this from `/Users/kim/Desktop/project`:

```bash
zip -r mnist_colab_aug68.zip \
  mnistCUDNN/scripts/train_emnist_lenet.py \
  mnistCUDNN/scripts/make_extra_pgm_dataset.py \
  mnistCUDNN/requirements-training.txt \
  mnistCUDNN/yolo/save_pgm.py \
  mnistCUDNN/data/*.bin \
  mnistCUDNN/datasets/extra_pgm_ink \
  mnistCUDNN/COLAB_TRAINING.md \
  extra \
  -x "*.DS_Store"
```

Upload `mnist_colab_aug68.zip` to Colab.

## 2. Set up Colab

```python
from google.colab import files
uploaded = files.upload()
```

```bash
!rm -rf /content/project
!mkdir -p /content/project
!unzip -q mnist_colab_aug68.zip -d /content/project
%cd /content/project/mnistCUDNN
!pip -q install -r requirements-training.txt
```

Use `Runtime > Change runtime type > T4 GPU` before training.

## 3. Preview or regenerate the converted PGM data

The zip already includes `datasets/extra_pgm_ink`, generated from `extra/` by
keeping only locally dark black-pen strokes. Check the contact sheet before
training:

```python
from IPython.display import Image, display
display(Image("datasets/extra_pgm_ink/contact_sheet.png"))
```

To regenerate it in Colab from the raw uploaded photos:

```bash
!python scripts/make_extra_pgm_dataset.py \
  --source ../extra \
  --output datasets/extra_pgm_ink \
  --orientation exif
```

```python
from IPython.display import Image, display
display(Image("datasets/extra_pgm_ink/contact_sheet.png"))
```

If the strokes are too faint, regenerate with a slightly lower contrast
threshold:

```bash
!python scripts/make_extra_pgm_dataset.py \
  --source ../extra \
  --output datasets/extra_pgm_ink \
  --orientation exif \
  --contrast-percentile 94 \
  --max-gray 170 \
  --min-contrast 14
```

If some preview cells are clearly not a digit, move those source files out of
`../extra` and regenerate. With such a small dataset, one bad sample can matter.
If the digits look rotated, try:

```bash
!python scripts/make_extra_pgm_dataset.py \
  --source ../extra \
  --output datasets/extra_pgm_ink \
  --orientation raw
```

## 4. Recommended fine-tune

This starts from the current `data/*.bin`, keeps EMNIST in the training loop as
rehearsal data, oversamples the converted PGM set, and exports new `.bin` files
only if EMNIST test accuracy remains above the guard threshold. The
`--no-extra-auto-crop` flag is important because `datasets/extra_pgm_ink`
already contains final 28x28 training images.

```bash
!python scripts/train_emnist_lenet.py \
  --data-root /content/emnist \
  --download \
  --augment \
  --extra-data-root datasets/extra_pgm_ink \
  --eval-data-root datasets/extra_pgm_ink \
  --no-extra-auto-crop \
  --extra-augment \
  --extra-repeat 600 \
  --epochs 8 \
  --batch-size 256 \
  --lr 2e-4 \
  --lr-step 5 \
  --lr-gamma 0.3 \
  --init-bin-dir data \
  --eval-initial \
  --min-test-accuracy 0.98 \
  --export-dir trained_weights/aug68_colab \
  --overwrite
```

If the final `extra:datasets/extra_pgm_ink` score improves but the EMNIST guard
fails by a small amount, rerun with `--min-test-accuracy 0.0`, inspect
`trained_weights/aug68_colab/lenet_emnist_metrics.json`, and only use the bins
if the per-digit metrics are acceptable.

## 5. Optional from-scratch training

Use this only if fine-tuning from the current bins is not good enough.

```bash
!python scripts/train_emnist_lenet.py \
  --data-root /content/emnist \
  --download \
  --augment \
  --extra-data-root datasets/extra_pgm_ink \
  --eval-data-root datasets/extra_pgm_ink \
  --no-extra-auto-crop \
  --extra-augment \
  --extra-repeat 600 \
  --epochs 12 \
  --batch-size 256 \
  --lr 1e-3 \
  --lr-step 7 \
  --lr-gamma 0.3 \
  --min-test-accuracy 0.98 \
  --export-dir trained_weights/aug68_from_scratch \
  --overwrite
```

## 6. Download the trained bins

```bash
!cd trained_weights/aug68_colab && zip -q -r /content/aug68_colab_bins.zip .
files.download("/content/aug68_colab_bins.zip")
```

Back on the local machine, back up the current runtime bins before copying the
new ones into `mnistCUDNN/data/`.
