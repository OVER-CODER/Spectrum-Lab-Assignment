# Spectral Lab Assignment

This workspace contains two assignments:

- **Programming_Assignment_1**: Assignment 1 — ISTA-Net for image deblurring (CBSD500-style protocol)
- **Programming_Assignment_2**: Plug-and-Play (PnP) super-resolution with pretrained denoiser priors

For full assignment-specific instructions, use:

- [Programming_Assignment_1/README.md](Programming_Assignment_1/README.md)
- [Programming_Assignment_2/README.md](Programming_Assignment_2/README.md)

## Workspace Structure

- `Programming_Assignment_1/` → [PA1 README](Programming_Assignment_1/README.md)
- `Programming_Assignment_2/` → [PA2 README](Programming_Assignment_2/README.md)

## Prerequisites

- Python 3.9+ recommended
- `pip` and `venv`
- Optional GPU backend (`cuda` or `mps`) for faster runs

## Quick Start

### 1) Clone / open workspace

```bash
git clone <your-repo-url>
cd "Spectral Lab Assignment"
```

### 2) Run PA1 (Deblurring track)

```bash
cd Programming_Assignment_1
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy scipy opencv-python scikit-image matplotlib
python Train_ISTA_Net_Deblur.py --end_epoch 50 --layer_num 10 --learning_rate 1e-4 --gpu_list 0
python TEST_ISTA_Net_Deblur.py --epoch_num 50 --layer_num 10 --learning_rate 1e-4 --gpu_list 0
```

See full details in [Programming_Assignment_1/README.md](Programming_Assignment_1/README.md).

What this run does (PA1):

- Trains unrolled ISTA-Net for deblurring with a 5×5 box blur and Gaussian noise (`σ=0.05`)
- Evaluates on up to 25 held-out test images with PSNR and SSIM

### 3) Run PA2 (PnP super-resolution)

```bash
cd ../Programming_Assignment_2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib scikit-image pillow deepinv
python pnp_superres.py
```

See full details in [Programming_Assignment_2/README.md](Programming_Assignment_2/README.md).

What this run does (PA2):

- Runs iterative PnP super-resolution with pretrained denoiser priors
- Saves convergence and reconstruction figures to `Programming_Assignment_2/outputs/`

## Notes

- PA1 includes legacy CS/MRI-CS scripts in the same folder, but Assignment 1 is the deblurring track.
- PA1 CS/MRI-CS scripts require additional `data/` and `sampling_matrix/` assets.
- PA2 auto-selects device (`mps` → `cuda` → `cpu`) unless overridden.
- Keep outputs under each assignment folder (`model/`, `result/`, `outputs/`).

## Typical Output Paths

- PA1 checkpoints: `Programming_Assignment_1/model/ISTA_Net_Deblur_layer_<L>_lr_<LR>/`
- PA1 results: `Programming_Assignment_1/result/`
- PA2 results: `Programming_Assignment_2/outputs/`
