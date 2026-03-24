import os
from pathlib import Path
import argparse
import csv

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="PnP super-resolution on CBSD500.")
    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    parser.add_argument("--data-dir", type=str, default="archive/images/test")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--denoiser-std", type=float, default=0.02)
    parser.add_argument("--denoiser", type=str, default="dncnn", choices=["dncnn", "fne_dncnn"])
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--step-size-end", type=float, default=0.2)
    parser.add_argument("--denoiser-std-end", type=float, default=0.01)
    parser.add_argument("--schedule", type=str, default="linear", choices=["constant", "linear", "exp"])
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=-1, help="Max images to evaluate (-1 for all)")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class CBSD500Dataset(Dataset):
    def __init__(self, root, crop_size):
        self.root = Path(root)
        self.files = sorted(
            path for path in self.root.glob("*.jpg") if path.name.lower() != "thumbs.db"
        )
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),  # RGB in [0, 1]
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        return self.transform(image), self.files[idx].name


def box_kernel(device):
    kernel = torch.full((3, 1, 3, 3), 1.0 / 9.0, device=device)
    return kernel


def apply_blur(x, kernel):
    c = x.shape[1]
    weight = kernel.expand(c, 1, 3, 3)
    return F.conv2d(x, weight, padding=1, groups=c)


def downsample(x, scale=2):
    return x[:, :, ::scale, ::scale]


def apply_H(x, kernel, scale=2):
    return downsample(apply_blur(x, kernel), scale=scale)


def upsample_adjoint(y, scale=2, output_size=(256, 256)):
    b, c, _, _ = y.shape
    out = torch.zeros((b, c, output_size[0], output_size[1]), device=y.device, dtype=y.dtype)
    out[:, :, ::scale, ::scale] = y
    return out


def apply_transpose_blur(x, kernel):
    c = x.shape[1]
    weight = kernel.expand(c, 1, 3, 3)
    return F.conv2d(x, weight, padding=1, groups=c)


def apply_HT(y, kernel, scale=2, output_size=(256, 256)):
    return apply_transpose_blur(upsample_adjoint(y, scale=scale, output_size=output_size), kernel)


def get_denoiser(device, denoiser_name):
    import deepinv as dinv

    pretrained_name = {
        "dncnn": "download",
        "fne_dncnn": "download_lipschitz",
    }[denoiser_name]
    model = dinv.models.DnCNN(
        in_channels=3,
        out_channels=3,
        pretrained=pretrained_name,
        device=device,
    ).to(device)
    model.eval()

    def denoiser(x, sigma):
        noise_level = torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            return model(x, sigma=noise_level)

    print(f"Using deepinv pretrained {denoiser_name}.")
    return denoiser


def add_noise(y, noise_std):
    return y + noise_std * torch.randn_like(y)


def build_schedule(num_iters, start_value, end_value, mode):
    if num_iters <= 0:
        return []
    if mode == "constant":
        return [start_value for _ in range(num_iters)]
    if num_iters == 1:
        return [start_value]
    if mode == "linear":
        return np.linspace(start_value, end_value, num_iters).tolist()
    if mode == "exp":
        if start_value <= 0 or end_value <= 0:
            raise ValueError("Exponential schedule requires positive start/end values.")
        return np.geomspace(start_value, end_value, num_iters).tolist()
    raise ValueError(f"Unknown schedule mode: {mode}")


def tensor_to_image(x):
    return x.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def compute_metrics(x, gt):
    x_np = tensor_to_image(x)
    gt_np = tensor_to_image(gt)
    psnr = peak_signal_noise_ratio(gt_np, x_np, data_range=1.0)
    ssim = structural_similarity(gt_np, x_np, data_range=1.0, channel_axis=2)
    return psnr, ssim


def compute_batch_metrics(x_batch, gt_batch):
    psnr_vals = []
    ssim_vals = []
    for idx in range(x_batch.shape[0]):
        psnr, ssim = compute_metrics(x_batch[idx], gt_batch[idx])
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def run_pnp(y, gt, kernel, denoiser, eta_schedule, sigma_schedule, args):
    x = F.interpolate(y, scale_factor=args.scale, mode="bicubic", align_corners=False).clamp(0, 1)
    history = {"psnr": [], "ssim": [], "residual": [], "eta": [], "sigma": []}

    for k in range(args.num_iters):
        x_prev = x.clone()
        eta_k = float(eta_schedule[k])
        sigma_k = float(sigma_schedule[k])

        grad = apply_HT(
            apply_H(x, kernel, scale=args.scale) - y,
            kernel,
            scale=args.scale,
            output_size=(args.crop_size, args.crop_size),
        )
        z = x - eta_k * grad
        x = torch.clamp(denoiser(z, sigma_k), 0.0, 1.0)

        residual = torch.norm(x - x_prev).item() / float(x.shape[0])
        psnr, ssim = compute_batch_metrics(x, gt)
        history["residual"].append(residual)
        history["psnr"].append(psnr)
        history["ssim"].append(ssim)
        history["eta"].append(eta_k)
        history["sigma"].append(sigma_k)

    return x, history


def plot_curves(history, output_dir, denoiser):
    iterations = np.arange(1, len(history["psnr"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(iterations, history["psnr"], marker="o", color="blue")
    axes[0].set_title("Average PSNR vs iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(iterations, history["ssim"], marker="o", color="green")
    axes[1].set_title("Average SSIM vs iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].plot(iterations, history["residual"], marker="o", color="red")
    axes[2].set_title("Average Residual vs iteration")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(r"$\|x_k - x_{k-1}\|$")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_dir / f"pnp_convergence_{denoiser}.png", dpi=150)
    plt.close(fig)


def plot_distribution(psnr_base, ssim_base, psnr_recon, ssim_recon, output_dir, denoiser):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(psnr_base, bins=15, alpha=0.5, label='Bicubic Baseline', color='red')
    axes[0].hist(psnr_recon, bins=15, alpha=0.5, label='PnP Recon', color='blue')
    axes[0].set_title(f"PSNR Distribution ({denoiser})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(ssim_base, bins=15, alpha=0.5, label='Bicubic Baseline', color='orange')
    axes[1].hist(ssim_recon, bins=15, alpha=0.5, label='PnP Recon', color='green')
    axes[1].set_title(f"SSIM Distribution ({denoiser})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("SSIM")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / f"pnp_distribution_{denoiser}.png", dpi=150)
    plt.close(fig)


def plot_comprehensive_comparison(saved_images, output_dir, denoiser):
    n = len(saved_images)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    if n == 1:
        axes = [axes]
    
    for i, (lr, recon, gt, name, p_base, s_base, p_rec, s_rec) in enumerate(saved_images):
        axes[i][0].imshow(tensor_to_image(lr))
        axes[i][0].set_title(f"Bicubic (Baseline)\nPSNR: {p_base:.2f}dB | SSIM: {s_base:.4f}")
        axes[i][0].axis("off")
        
        axes[i][1].imshow(tensor_to_image(recon))
        axes[i][1].set_title(f"PnP Recon ({denoiser})\nPSNR: {p_rec:.2f}dB | SSIM: {s_rec:.4f}")
        axes[i][1].axis("off")
        
        axes[i][2].imshow(tensor_to_image(gt))
        axes[i][2].set_title(f"Ground Truth\n({name})")
        axes[i][2].axis("off")

    fig.suptitle(f'Comprehensive Visual Comparison ({denoiser})', fontsize=16, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / f"pnp_comprehensive_{denoiser}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if args.crop_size != 256:
        raise ValueError("Assignment requires center crop size of 256.")
    if abs(args.noise_std - 0.01) > 1e-12:
        raise ValueError("Assignment requires Gaussian noise standard deviation sigma = 0.01.")

    dataset = CBSD500Dataset(args.data_dir, args.crop_size)
    if args.max_images > 0:
        dataset.files = dataset.files[: args.max_images]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    kernel = box_kernel(device)
    eta_schedule = build_schedule(args.num_iters, args.step_size, args.step_size_end, args.schedule)
    sigma_schedule = build_schedule(args.num_iters, args.denoiser_std, args.denoiser_std_end, args.schedule)
    denoiser = get_denoiser(device, args.denoiser)

    avg_history = {"psnr": np.zeros(args.num_iters), "ssim": np.zeros(args.num_iters), "residual": np.zeros(args.num_iters)}
    
    baseline_psnr_list, baseline_ssim_list = [], []
    final_psnr_list, final_ssim_list = [], []
    
    saved_images = []
    
    csv_file = output_dir / f"per_image_metrics_{args.denoiser}.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "bicubic_psnr", "bicubic_ssim", "recon_psnr", "recon_ssim"])
        
        for gt, name in loader:
            gt = gt.to(device)
            y = apply_H(gt, kernel, scale=args.scale)
            y = add_noise(y, args.noise_std)
            
            # Baseline: Bicubic upsampling
            y_bicubic = F.interpolate(y, scale_factor=args.scale, mode="bicubic", align_corners=False).clamp(0, 1)
            b_psnr, b_ssim = compute_batch_metrics(y_bicubic, gt)
            baseline_psnr_list.append(b_psnr)
            baseline_ssim_list.append(b_ssim)

            recon, history = run_pnp(y, gt, kernel, denoiser, eta_schedule, sigma_schedule, args)

            avg_history["psnr"] += np.array(history["psnr"])
            avg_history["ssim"] += np.array(history["ssim"])
            avg_history["residual"] += np.array(history["residual"])
            
            f_psnr = history["psnr"][-1]
            f_ssim = history["ssim"][-1]
            final_psnr_list.append(f_psnr)
            final_ssim_list.append(f_ssim)
            
            writer.writerow([name[0], b_psnr, b_ssim, f_psnr, f_ssim])
            print(f"Processed {name[0]:20s} | Base PSNR/SSIM: {b_psnr:.2f}/{b_ssim:.4f} | Recon PSNR/SSIM: {f_psnr:.2f}/{f_ssim:.4f}")
            
            if len(saved_images) < 6:
                saved_images.append((y_bicubic[0].cpu(), recon[0].cpu(), gt[0].cpu(), name[0], b_psnr, b_ssim, f_psnr, f_ssim))

    num_images = len(final_psnr_list)
    avg_history = {k: (v / max(num_images, 1)).tolist() for k, v in avg_history.items()}

    plot_curves(avg_history, output_dir, args.denoiser)
    plot_distribution(baseline_psnr_list, baseline_ssim_list, final_psnr_list, final_ssim_list, output_dir, args.denoiser)
    plot_comprehensive_comparison(saved_images, output_dir, args.denoiser)
    
    # Text summary
    summary_file = output_dir / f"evaluation_summary_{args.denoiser}.txt"
    with open(summary_file, "w") as f:
        f.write(f"PnP Super-Resolution Evaluation Summary\n")
        f.write(f"=======================================\n")
        f.write(f"Denoiser: {args.denoiser}\n")
        f.write(f"Number of test images: {num_images}\n")
        f.write(f"\nBaseline (Bicubic Interpolation):\n")
        f.write(f"  Average PSNR: {np.mean(baseline_psnr_list):.4f} dB\n")
        f.write(f"  Average SSIM: {np.mean(baseline_ssim_list):.4f}\n")
        f.write(f"\nPnP Reconstruction:\n")
        f.write(f"  Average PSNR: {np.mean(final_psnr_list):.4f} dB\n")
        f.write(f"  Average SSIM: {np.mean(final_ssim_list):.4f}\n")
        f.write(f"\nImprovement:\n")
        f.write(f"  Delta PSNR: {np.mean(final_psnr_list) - np.mean(baseline_psnr_list):+.4f} dB\n")
        f.write(f"  Delta SSIM: {np.mean(final_ssim_list) - np.mean(baseline_ssim_list):+.4f}\n")
        f.write(f"\nSaved Visualization Assets:\n")
        f.write(f"  - pnp_convergence_{args.denoiser}.png\n")
        f.write(f"  - pnp_distribution_{args.denoiser}.png\n")
        f.write(f"  - pnp_comprehensive_{args.denoiser}.png\n")
    
    print("\n" + "="*50)
    print(f"Final Evaluation for {args.denoiser}")
    print("="*50)
    print(f"Baseline Average PSNR: {np.mean(baseline_psnr_list):.4f} dB,  SSIM: {np.mean(baseline_ssim_list):.4f}")
    print(f"Recon Average PSNR:    {np.mean(final_psnr_list):.4f} dB,  SSIM: {np.mean(final_ssim_list):.4f}")
    print(f"Improvement PSNR:      {np.mean(final_psnr_list) - np.mean(baseline_psnr_list):+.4f} dB")
    print(f"Improvement SSIM:      {np.mean(final_ssim_list) - np.mean(baseline_ssim_list):+.4f}")
    print("==================================================")
    print(f"Saved evaluation summaries to {output_dir}")

if __name__ == "__main__":
    main()
