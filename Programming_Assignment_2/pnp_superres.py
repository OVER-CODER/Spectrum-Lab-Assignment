import os
from pathlib import Path
import argparse

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
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=0)
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
    # Hx = S(Bx): blur, then decimate
    return downsample(apply_blur(x, kernel), scale=scale)


def upsample_adjoint(y, scale=2, output_size=(256, 256)):
    # S^T y: insert zeros at sampled locations, exact transpose of slicing
    b, c, _, _ = y.shape
    out = torch.zeros((b, c, output_size[0], output_size[1]), device=y.device, dtype=y.dtype)
    out[:, :, ::scale, ::scale] = y
    return out


def apply_transpose_blur(x, kernel):
    c = x.shape[1]
    weight = kernel.expand(c, 1, 3, 3)
    return F.conv2d(x, weight, padding=1, groups=c)


def apply_HT(y, kernel, scale=2, output_size=(256, 256)):
    # H^T y = B^T(S^T y)
    return apply_transpose_blur(upsample_adjoint(y, scale=scale, output_size=output_size), kernel)


def verify_adjoint(device, kernel, crop_size, scale):
    x = torch.randn(1, 3, crop_size, crop_size, device=device)
    z = torch.randn(1, 3, crop_size // scale, crop_size // scale, device=device)
    lhs = torch.sum(apply_H(x, kernel, scale=scale) * z)
    rhs = torch.sum(x * apply_HT(z, kernel, scale=scale, output_size=(crop_size, crop_size)))
    error = abs(lhs.item() - rhs.item())
    print(f"Adjoint check |<Hx,z> - <x,H^Tz>| = {error:.6e}")


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
    x = F.interpolate(y, scale_factor=args.scale, mode="nearest")
    history = {"psnr": [], "ssim": [], "residual": [], "eta": [], "sigma": []}

    for k in range(args.num_iters):
        x_prev = x.clone()
        eta_k = float(eta_schedule[k])
        sigma_k = float(sigma_schedule[k])

        # z_{k+1} = x_k - eta * H^T(Hx_k - y)
        grad = apply_HT(
            apply_H(x, kernel, scale=args.scale) - y,
            kernel,
            scale=args.scale,
            output_size=(args.crop_size, args.crop_size),
        )
        z = x - eta_k * grad

        # x_{k+1} = D_theta(z_{k+1}, sigma_k)
        x = torch.clamp(denoiser(z, sigma_k), 0.0, 1.0)

        residual = torch.norm(x - x_prev).item() / float(x.shape[0])
        psnr, ssim = compute_batch_metrics(x, gt)
        history["residual"].append(residual)
        history["psnr"].append(psnr)
        history["ssim"].append(ssim)
        history["eta"].append(eta_k)
        history["sigma"].append(sigma_k)

    return x, history


def plot_curves(history, output_dir):
    iterations = np.arange(1, len(history["psnr"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iterations, history["psnr"], marker="o")
    axes[0].set_title("PSNR vs iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("PSNR")

    axes[1].plot(iterations, history["ssim"], marker="o")
    axes[1].set_title("SSIM vs iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("SSIM")

    axes[2].plot(iterations, history["residual"], marker="o")
    axes[2].set_title("Residual vs iteration")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(r"$\|x_k - x_{k-1}\|$")

    fig.tight_layout()
    fig.savefig(output_dir / "pnp_convergence.png", dpi=150)
    plt.close(fig)


def plot_images(lr, recon, gt, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(tensor_to_image(lr))
    axes[0].set_title("Input (LR noisy)")
    axes[1].imshow(tensor_to_image(recon))
    axes[1].set_title("PnP reconstruction")
    axes[2].imshow(tensor_to_image(gt))
    axes[2].set_title("Ground truth")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "pnp_reconstruction.png", dpi=150)
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    kernel = box_kernel(device)
    verify_adjoint(device, kernel, args.crop_size, args.scale)
    eta_schedule = build_schedule(args.num_iters, args.step_size, args.step_size_end, args.schedule)
    sigma_schedule = build_schedule(args.num_iters, args.denoiser_std, args.denoiser_std_end, args.schedule)
    denoiser = get_denoiser(device, args.denoiser)

    avg_history = {"psnr": np.zeros(args.num_iters), "ssim": np.zeros(args.num_iters), "residual": np.zeros(args.num_iters)}
    final_psnr = []
    final_ssim = []
    saved_triplet = None

    for gt, name in loader:
        gt = gt.to(device)
        print(f"Processing {name[0]} ...")

        y = apply_H(gt, kernel, scale=args.scale)
        y = add_noise(y, args.noise_std)

        recon, history = run_pnp(y, gt, kernel, denoiser, eta_schedule, sigma_schedule, args)

        avg_history["psnr"] += np.array(history["psnr"])
        avg_history["ssim"] += np.array(history["ssim"])
        avg_history["residual"] += np.array(history["residual"])
        final_psnr.append(history["psnr"][-1])
        final_ssim.append(history["ssim"][-1])

        if saved_triplet is None:
            lr_vis = F.interpolate(y, scale_factor=args.scale, mode="nearest")
            saved_triplet = (lr_vis[0].cpu(), recon[0].cpu(), gt[0].cpu(), name[0])

    num_images = len(final_psnr)
    avg_history = {k: (v / max(num_images, 1)).tolist() for k, v in avg_history.items()}

    plot_curves(avg_history, output_dir)
    if saved_triplet is not None:
        lr_vis, recon_vis, gt_vis, sample_name = saved_triplet
        plot_images(lr_vis, recon_vis, gt_vis, output_dir)
        print(f"Saved sample visualization for {sample_name} to {output_dir / 'pnp_reconstruction.png'}")

    print(f"Final average PSNR: {np.mean(final_psnr):.4f} dB")
    print(f"Final average SSIM: {np.mean(final_ssim):.4f}")
    if len(eta_schedule) > 0 and len(sigma_schedule) > 0:
        print(
            "Schedule used: "
            f"eta [{eta_schedule[0]:.4f} -> {eta_schedule[-1]:.4f}], "
            f"sigma [{sigma_schedule[0]:.4f} -> {sigma_schedule[-1]:.4f}] ({args.schedule})"
        )
    print(f"Saved convergence plot to {output_dir / 'pnp_convergence.png'}")


if __name__ == "__main__":
    main()
