#!/usr/bin/env python
"""
Test script for ISTA-Net Deblur (fixed model).
Evaluates a chosen checkpoint on the test set and saves plots/results.
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset


parser = ArgumentParser(description="Evaluate fixed ISTA-Net deblur checkpoint")
parser.add_argument("--test_dir", type=str, default="./archive/ground_truth/test", help="test .mat directory")
parser.add_argument("--layer_num", type=int, default=10, help="number of ISTA layers")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate used for naming")
parser.add_argument("--epoch_num", type=int, default=150, help="checkpoint epoch to evaluate")
parser.add_argument("--model_dir", type=str, default="./model", help="root model directory")
parser.add_argument("--result_dir", type=str, default="./result/fixed_epoch_150", help="output directory")
parser.add_argument("--force_cpu", action="store_true", help="force CPU even if MPS is available")
args = parser.parse_args()


def resolve_device():
    if args.force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


device = resolve_device()
is_real_degradation = False
os.makedirs(args.result_dir, exist_ok=True)

model_dir = os.path.join(
    args.model_dir,
    f"ISTA_Net_Deblur_layer_{args.layer_num}_lr_{args.learning_rate:.4f}_FIXED",
)
model_path = os.path.join(model_dir, f"net_params_{args.epoch_num}.pkl")

print(f"Loading fixed model from: {model_dir}")
print(f"Checkpoint epoch: {args.epoch_num}")
print(f"Device: {device}")


class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.image_names = sorted(
            name for name in os.listdir(test_dir) if name.lower().endswith(".mat")
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        mat = scipy.io.loadmat(os.path.join(self.test_dir, image_name))
        image = mat["data"].astype(np.float32)

        if not is_real_degradation:
            degraded_image = mat["degraded_image"].astype(np.float32)
        else:
            degraded_image = image

        image = torch.from_numpy(image).unsqueeze(0)
        degraded_image = torch.from_numpy(degraded_image).unsqueeze(0)

        return image_name, degraded_image, image


blur_kernel = torch.ones((5, 5), dtype=torch.float32, device=device) / 25.0


def apply_blur(x):
    _, channels, _, _ = x.shape
    kernel = blur_kernel.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return nn.functional.conv2d(x, kernel, padding=2, groups=channels)


def apply_transpose_blur(x):
    _, channels, _, _ = x.shape
    kernel = blur_kernel.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return nn.functional.conv_transpose2d(x, kernel, padding=2, groups=channels)


def soft_threshold(z, lam):
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)


class ISTABlock(nn.Module):
    """Single ISTA layer for image deblurring with fixed initialization."""

    def __init__(self):
        super().__init__()
        self.eta_raw = nn.Parameter(torch.tensor(0.0))
        self.lam_raw = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, y):
        eta = nn.functional.softplus(self.eta_raw) + 1e-6
        lam = nn.functional.softplus(self.lam_raw) + 1e-6

        residual = apply_blur(x) - y
        z = x - eta * apply_transpose_blur(residual)
        return soft_threshold(z, lam)


class ISTANet(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([ISTABlock() for _ in range(num_layers)])

    def forward(self, y):
        x = y.clone()
        for layer in self.layers:
            x = layer(x, y)
        return x


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

model = ISTANet(num_layers=args.layer_num).to(device)
state_dict = torch.load(model_path, map_location=device)
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"Loaded model from {model_path}")

test_dataset = TestDataset(args.test_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("\n" + "=" * 80)
print("EVALUATION RESULTS - FIXED MODEL")
print("=" * 80)

records = []
psnr_values = []
ssim_values = []

with torch.no_grad():
    for image_name, degraded_image, original_image in test_loader:
        degraded_image = degraded_image.to(device)
        original_image = original_image.to(device)
        restored_image = torch.clamp(model(degraded_image), 0.0, 1.0)

        original_np = original_image.squeeze().cpu().numpy()
        degraded_np = degraded_image.squeeze().cpu().numpy()
        restored_np = restored_image.squeeze().cpu().numpy()
        data_range = float(original_np.max() - original_np.min())
        if data_range == 0:
            data_range = 1.0

        psnr_degraded = psnr(original_np, degraded_np, data_range=data_range)
        ssim_degraded = ssim(original_np, degraded_np, data_range=data_range)
        psnr_restored = psnr(original_np, restored_np, data_range=data_range)
        ssim_restored = ssim(original_np, restored_np, data_range=data_range)

        record = {
            "image_name": image_name[0],
            "psnr_degraded": psnr_degraded,
            "ssim_degraded": ssim_degraded,
            "psnr_restored": psnr_restored,
            "ssim_restored": ssim_restored,
        }
        records.append(record)
        psnr_values.append(psnr_restored)
        ssim_values.append(ssim_restored)

        print(
            f"{image_name[0]:20s} | "
            f"PSNR: {psnr_restored:7.4f} dB | "
            f"SSIM: {ssim_restored:7.4f}"
        )

psnr_values = np.array(psnr_values)
ssim_values = np.array(ssim_values)
psnr_mean = float(np.mean(psnr_values))
psnr_std = float(np.std(psnr_values))
ssim_mean = float(np.mean(ssim_values))
ssim_std = float(np.std(ssim_values))
degraded_psnr_mean = float(np.mean([r["psnr_degraded"] for r in records]))
degraded_ssim_mean = float(np.mean([r["ssim_degraded"] for r in records]))

print("=" * 80)
print(f"\nFIXED MODEL METRICS (Epoch {args.epoch_num}):")
print(f"  Restored PSNR: {psnr_mean:.4f} ± {psnr_std:.4f} dB")
print(f"  Restored SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
print(f"  Degraded PSNR: {degraded_psnr_mean:.4f} dB")
print(f"  Degraded SSIM: {degraded_ssim_mean:.4f}")
print(f"  Delta PSNR: {psnr_mean - degraded_psnr_mean:+.4f} dB")
print(f"  Delta SSIM: {ssim_mean - degraded_ssim_mean:+.4f}")
print(f"  Test images evaluated: {len(records)}")
print("=" * 80)

results_file = os.path.join(args.result_dir, f"evaluation_summary_epoch_{args.epoch_num}.txt")
csv_file = os.path.join(args.result_dir, f"per_image_metrics_epoch_{args.epoch_num}.csv")

with open(results_file, "w") as handle:
    handle.write("ISTA-Net FIXED Model Evaluation Results\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Model: {model_dir}\n")
    handle.write(f"Checkpoint: {model_path}\n")
    handle.write(f"Epoch: {args.epoch_num}\n")
    handle.write(f"Device: {device}\n")
    handle.write("=" * 80 + "\n\n")
    handle.write("Summary Statistics:\n")
    handle.write(f"Restored PSNR: {psnr_mean:.4f} ± {psnr_std:.4f} dB\n")
    handle.write(f"Restored SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n")
    handle.write(f"Degraded PSNR: {degraded_psnr_mean:.4f} dB\n")
    handle.write(f"Degraded SSIM: {degraded_ssim_mean:.4f}\n")
    handle.write(f"Delta PSNR: {psnr_mean - degraded_psnr_mean:+.4f} dB\n")
    handle.write(f"Delta SSIM: {ssim_mean - degraded_ssim_mean:+.4f}\n")
    handle.write(f"Test images evaluated: {len(records)}\n\n")
    handle.write("Per-Image Results:\n")
    handle.write("-" * 80 + "\n")
    for record in records:
        handle.write(
            f"{record['image_name']:20s} | "
            f"Blur PSNR: {record['psnr_degraded']:7.4f} dB | "
            f"Blur SSIM: {record['ssim_degraded']:7.4f} | "
            f"Restored PSNR: {record['psnr_restored']:7.4f} dB | "
            f"Restored SSIM: {record['ssim_restored']:7.4f}\n"
        )

with open(csv_file, "w") as handle:
    handle.write("image_name,psnr_degraded,ssim_degraded,psnr_restored,ssim_restored\n")
    for record in records:
        handle.write(
            f"{record['image_name']},{record['psnr_degraded']:.6f},{record['ssim_degraded']:.6f},"
            f"{record['psnr_restored']:.6f},{record['ssim_restored']:.6f}\n"
        )

print(f"\nResults saved to: {results_file}")
print(f"Per-image CSV saved to: {csv_file}")

# Comparison plot for the first 5 images.
fig = plt.figure(figsize=(16, 10))
grid = GridSpec(3, min(5, len(test_dataset)), figure=fig, hspace=0.3, wspace=0.3)

for idx in range(min(5, len(test_dataset))):
    image_name, degraded_image, original_image = test_dataset[idx]
    degraded_image = degraded_image.unsqueeze(0).to(device)
    original_image = original_image.to(device)

    with torch.no_grad():
        restored_image = torch.clamp(model(degraded_image), 0.0, 1.0)

    original_np = original_image.squeeze().cpu().numpy()
    degraded_np = degraded_image.squeeze().cpu().numpy()
    restored_np = restored_image.squeeze().cpu().numpy()
    data_range = float(original_np.max() - original_np.min())
    if data_range == 0:
        data_range = 1.0

    psnr_val = psnr(original_np, restored_np, data_range=data_range)
    ssim_val = ssim(original_np, restored_np, data_range=data_range)

    ax = fig.add_subplot(grid[0, idx])
    ax.imshow(original_np, cmap="gray")
    ax.set_title("Ground Truth", fontsize=10, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(grid[1, idx])
    ax.imshow(degraded_np, cmap="gray")
    ax.set_title("Degraded", fontsize=10, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(grid[2, idx])
    ax.imshow(restored_np, cmap="gray")
    ax.set_title(
        f"Restored\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}",
        fontsize=10,
        fontweight="bold",
        color="green",
    )
    ax.axis("off")

fig.suptitle(
    f"ISTA-Net FIXED Model Results (Epoch {args.epoch_num}, Layer {args.layer_num}, LR {args.learning_rate})",
    fontsize=14,
    fontweight="bold",
    color="darkgreen",
)
comparison_path = os.path.join(args.result_dir, f"test_results_comparison_epoch_{args.epoch_num}.png")
fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Comparison plot saved to: {comparison_path}")

# Distribution plots.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(psnr_values, bins=10, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].axvline(psnr_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {psnr_mean:.4f}")
axes[0].set_xlabel("PSNR (dB)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
axes[0].set_title("PSNR Distribution", fontsize=13, fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(ssim_values, bins=10, color="forestgreen", edgecolor="black", alpha=0.7)
axes[1].axvline(ssim_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {ssim_mean:.4f}")
axes[1].set_xlabel("SSIM", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Frequency", fontsize=12, fontweight="bold")
axes[1].set_title("SSIM Distribution", fontsize=13, fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle(
    f"FIXED Model Distribution (Epoch {args.epoch_num})",
    fontsize=14,
    fontweight="bold",
    color="darkgreen",
)
fig.tight_layout()
distribution_path = os.path.join(args.result_dir, f"metrics_distribution_epoch_{args.epoch_num}.png")
fig.savefig(distribution_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Metrics distribution saved to: {distribution_path}")

print("\nFIXED MODEL EVALUATION COMPLETE")
