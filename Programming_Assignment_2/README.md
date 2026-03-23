# Programming_Assignment_2 — Plug-and-Play (PnP) Super-Resolution

This assignment implements iterative PnP super-resolution on CBSD-style test images using a pretrained denoiser (`DnCNN`) from `deepinv`.

Main script:

- `pnp_superres.py`

## 1) Environment Setup

From this folder:

```bash
cd Programming_Assignment_2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib scikit-image pillow deepinv
```

`pnp_superres.py` auto-selects device priority in this order:

1. `mps` (Apple Silicon)
2. `cuda`
3. `cpu`

You can override with `--device`.

## 2) Dataset Layout

Default input directory:

- `archive/images/test`

The script expects image files (configured dataset loader currently reads `*.jpg`).

## 3) Run

Basic command:

```bash
python pnp_superres.py
```

Recommended explicit command:

```bash
python pnp_superres.py \
  --data-dir archive/images/test \
  --crop-size 256 \
  --scale 2 \
  --noise-std 0.01 \
  --denoiser dncnn \
  --num-iters 10 \
  --schedule linear \
  --step-size 1.0 \
  --step-size-end 0.2 \
  --denoiser-std 0.02 \
  --denoiser-std-end 0.01 \
  --max-images 8
```

## 4) Key Arguments

- `--data-dir`: input image directory (default: `archive/images/test`)
- `--crop-size`: center crop size (**must be 256** in current assignment code)
- `--scale`: downsampling factor (default: `2`)
- `--noise-std`: LR Gaussian noise std (**must be 0.01** in current code)
- `--denoiser`: `dncnn` or `fne_dncnn`
- `--num-iters`: number of PnP iterations
- `--schedule`: `constant`, `linear`, or `exp` for both step-size and denoiser sigma schedules
- `--device`: e.g., `mps`, `cuda`, `cpu`
- `--max-images`: cap number of processed images

## 5) Outputs

Saved under `outputs/`:

- `pnp_convergence.png` (PSNR, SSIM, residual vs iteration)
- `pnp_reconstruction.png` (LR input, PnP reconstruction, GT for one sample)

Console prints:

- Adjoint consistency check
- Per-image processing info
- Final average PSNR / SSIM
- Effective schedule range for `eta` and denoiser `sigma`

## 6) Implementation Summary

The script performs:

1. Forward degradation: blur + decimation + additive Gaussian noise
2. Iterative update: gradient/data-consistency step with \(H^T(Hx-y)\)
3. Denoising prior step via pretrained DnCNN
4. Metric tracking (PSNR/SSIM) across iterations

## 7) Troubleshooting

- If `deepinv` download/init fails, check internet connectivity and retry.
- If you see backend/device errors, set `--device cpu` to verify functionality first.
- If no images are processed, verify `archive/images/test` contains `.jpg` files.
- Matplotlib cache issues are mitigated by setting `MPLCONFIGDIR` inside the script.
