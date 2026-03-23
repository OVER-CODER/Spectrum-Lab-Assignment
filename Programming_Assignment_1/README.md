# Programming_Assignment_1 — Assignment 1: ISTA-Net for Image Deblurring

This assignment focuses on implementing and training an unrolled ISTA-Net for **single-image deblurring** on CBSD500-style data.

## 1) Assignment Objective

The goal is to bridge classical optimization and deep learning by unrolling the ISTA algorithm into a finite-depth network.
Each iteration becomes a learnable layer that performs:

1. A data-consistency gradient step using the blur model transpose.
2. A soft-thresholding update to promote sparse structure.

The degraded observation model is:

\[
y = Hx + w
\]

where:

- \(H\): uniform **5×5 box blur** operator
- \(w\): additive Gaussian noise with \(\sigma = 0.05\)

## 2) Data Protocol (as implemented)

- Input images are center-cropped/resized to **256×256 RGB**.
- Training images are loaded from `archive/images/train`.
- Test images are loaded from `archive/images/test`.
- Evaluation runs on a held-out test subset of **up to 25 images** (the test script caps evaluation at 25).

## 3) ISTA-Net Architecture (Deblurring)

The deblurring model is defined in:

- `Train_ISTA_Net_Deblur.py`
- `TEST_ISTA_Net_Deblur.py`

Key properties:

- Unrolled iterative architecture (`ISTANet`) with configurable depth via `--layer_num`.
- Typical depths for this assignment: **10** or **15** layers.
- Each stage learns:
  - Step size parameter \(\eta_k\)
  - Threshold parameter \(\lambda_k\)
- Forward degradation and transpose operators are implemented with convolution / transpose-convolution using the 5×5 box kernel.

## 4) Environment Setup

From this folder:

```bash
cd Programming_Assignment_1
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy scipy opencv-python scikit-image matplotlib
```

> If you use Apple Silicon/MPS or CUDA, install the matching PyTorch build from the official PyTorch instructions.

## 5) Training

### Train with 10 layers

```bash
python Train_ISTA_Net_Deblur.py --end_epoch 50 --layer_num 10 --learning_rate 1e-4 --gpu_list 0
```

### Train with 15 layers

```bash
python Train_ISTA_Net_Deblur.py --end_epoch 50 --layer_num 15 --learning_rate 1e-4 --gpu_list 0
```

Notes:

- The script synthesizes blurred-noisy observations internally using the assignment model \(y = Hx + w\) with noise std `0.05`.
- Checkpoints are saved every 5 epochs.

## 6) Testing and Evaluation

### Evaluate 10-layer model

```bash
python TEST_ISTA_Net_Deblur.py --epoch_num 50 --layer_num 10 --learning_rate 1e-4 --gpu_list 0
```

### Evaluate 15-layer model

```bash
python TEST_ISTA_Net_Deblur.py --epoch_num 50 --layer_num 15 --learning_rate 1e-4 --gpu_list 0
```

Evaluation metrics:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

The script also reports baseline blurred metrics and improvement deltas.

## 7) Outputs

- Checkpoints: `model/ISTA_Net_Deblur_layer_<L>_lr_<LR>/net_params_<epoch>.pkl`
- Restored images: `result/deblurred_*.png`
- Summary file: `result/evaluation_summary.txt`

## 8) Expected Folder Layout

- `archive/images/train`
- `archive/images/test`
- `archive/ground_truth/test`
- `model/`
- `log/`
- `result/`

## 9) Troubleshooting

- If running on CPU-only systems, keep `--gpu_list 0`; script falls back to CPU when CUDA is unavailable.
- If testing fails to load a checkpoint, confirm `--epoch_num`, `--layer_num`, and `--learning_rate` exactly match the training run.
- If no images are found, verify `archive/images/train` and `archive/images/test` contain `.png`, `.jpg`, or `.bmp` files.

---

### Note on extra scripts in this folder

This directory also includes CS / MRI-CS ISTA-Net scripts from other tracks, but **Assignment 1 in this README is specifically the deblurring track** using:

- `Train_ISTA_Net_Deblur.py`
- `TEST_ISTA_Net_Deblur.py`
