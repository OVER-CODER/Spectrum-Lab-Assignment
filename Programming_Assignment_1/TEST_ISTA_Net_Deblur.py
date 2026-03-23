import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from argparse import ArgumentParser
import glob
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim

parser = ArgumentParser(description='ISTA-Net Test for Image Deblurring')

parser.add_argument('--epoch_num', type=int, default=50, help='epoch of model to test')
parser.add_argument('--layer_num', type=int, default=10, help='number of ISTA layers')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (for model naming)')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='model directory')
parser.add_argument('--data_dir', type=str, default='archive', help='data directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()

epoch_num = args.epoch_num
layer_num = args.layer_num
learning_rate = args.learning_rate
gpu_list = args.gpu_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except:
    pass

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# BLUR FUNCTIONS (same as training)
# ============================================================================

blur_kernel_np = np.ones((5, 5), dtype=np.float32) / 25.0
blur_kernel = torch.from_numpy(blur_kernel_np).to(device)

def apply_blur(x):
    """Apply blur operator H to image x."""
    _, C, _, _ = x.shape
    kernel = blur_kernel.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 5]
    kernel = kernel.repeat(C, 1, 1, 1)  # [C, 1, 5, 5]
    y = F.conv2d(x, kernel, padding=2, groups=C)
    return y


def apply_transpose_blur(x):
    _, C, _, _ = x.shape
    kernel = blur_kernel.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    return F.conv_transpose2d(x, kernel, padding=2, groups=C)

def add_noise(x, sigma=0.05):
    """Add Gaussian noise to image."""
    noise = torch.randn_like(x) * sigma
    return x + noise


def soft_threshold(z, lam):
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)

# ============================================================================
# MODEL DEFINITION (same as training)
# ============================================================================

class ISTABlock(nn.Module):
    """Single ISTA layer for image deblurring."""
    def __init__(self):
        super(ISTABlock, self).__init__()
        
        self.eta_raw = nn.Parameter(torch.tensor(-3.0))
        self.lam_raw = nn.Parameter(torch.tensor(-8.0))
    
    def forward(self, x, y):
        eta = F.softplus(self.eta_raw) + 1e-6
        lam = F.softplus(self.lam_raw) + 1e-6

        Hx = apply_blur(x)
        residual = Hx - y
        HT_residual = apply_transpose_blur(residual)
        
        z = x - eta * HT_residual
        
        x_soft = soft_threshold(z, lam)

        return x_soft

class ISTANet(nn.Module):
    """ISTA-Net for image deblurring."""
    def __init__(self, num_layers=10):
        super(ISTANet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([ISTABlock() for _ in range(num_layers)])
    
    def forward(self, y):
        x = y.clone()
        for layer in self.layers:
            x = layer(x, y)
        return x

# Load model
model = ISTANet(num_layers=layer_num)
model = nn.DataParallel(model)
model = model.to(device)

model_dir = "./%s/ISTA_Net_Deblur_layer_%d_lr_%.4f" % (args.model_dir, layer_num, learning_rate)
model_path = os.path.join(model_dir, f'net_params_{epoch_num}.pkl')

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Loaded model from {model_path}")
model.eval()

# ============================================================================
# LOAD TEST IMAGES
# ============================================================================

test_dir_clean = os.path.join(args.data_dir, 'images', 'test')
test_dir_gt = os.path.join(args.data_dir, 'ground_truth', 'test')

if not os.path.exists(test_dir_clean):
    print(f"Error: Test image directory not found: {test_dir_clean}")
    exit(1)

# Get list of test images
test_images = sorted(glob.glob(os.path.join(test_dir_clean, '*.png'))) + \
              sorted(glob.glob(os.path.join(test_dir_clean, '*.jpg'))) + \
              sorted(glob.glob(os.path.join(test_dir_clean, '*.bmp')))

print(f"Found {len(test_images)} test images")

# Limit to first 25 images if more exist
if len(test_images) > 25:
    test_images = test_images[:25]

print(f"Testing on {len(test_images)} images")

# ============================================================================
# EVALUATION
# ============================================================================

os.makedirs(args.result_dir, exist_ok=True)

psnr_values = []
ssim_values = []
blur_psnr_values = []
blur_ssim_values = []

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

with torch.no_grad():
    for idx, img_path in enumerate(test_images):
        # Load image
        img_clean = cv2.imread(img_path)
        if img_clean is None:
            print(f"Skipping {img_path}: could not load")
            continue
        
        # Convert BGR to RGB
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256 (center crop)
        h, w = img_clean.shape[:2]
        if h < 256 or w < 256:
            scale = 256 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_clean = cv2.resize(img_clean, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        h, w = img_clean.shape[:2]
        top = (h - 256) // 2
        left = (w - 256) // 2
        img_clean = img_clean[top:top+256, left:left+256, :]
        
        # Normalize to [0, 1]
        img_clean_np = img_clean.astype(np.float32) / 255.0
        img_clean_tensor = torch.from_numpy(img_clean_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Create blurred version
        with torch.no_grad():
            img_blurred = apply_blur(img_clean_tensor)
            img_blurred = add_noise(img_blurred, sigma=0.05)
            img_blurred = torch.clamp(img_blurred, 0, 1)
        
        # Deblur
        img_deblurred = model(img_blurred)
        img_deblurred = torch.clamp(img_deblurred, 0, 1)
        
        # Convert to numpy for metrics
        img_clean_np = img_clean_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_deblurred_np = img_deblurred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        img_blurred_np = img_blurred.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        psnr_blur = compare_psnr(img_clean_np, img_blurred_np, data_range=1.0)
        ssim_blur = ssim(img_clean_np, img_blurred_np, data_range=1.0, channel_axis=2)

        psnr = compare_psnr(img_clean_np, img_deblurred_np, data_range=1.0)
        ssim_val = ssim(img_clean_np, img_deblurred_np, data_range=1.0, channel_axis=2)
        
        blur_psnr_values.append(psnr_blur)
        blur_ssim_values.append(ssim_blur)
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
        
        print(f"Image {idx+1:2d}: PSNR = {psnr:.4f} dB, SSIM = {ssim_val:.4f}")
        
        # Save result
        img_deblurred_uint8 = (img_deblurred_np * 255).astype(np.uint8)
        img_deblurred_bgr = cv2.cvtColor(img_deblurred_uint8, cv2.COLOR_RGB2BGR)
        result_path = os.path.join(args.result_dir, f'deblurred_{idx+1:03d}.png')
        cv2.imwrite(result_path, img_deblurred_bgr)

print("="*60)
print(f"\nAverage PSNR: {np.mean(psnr_values):.4f} dB")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
print(f"Average Blurred PSNR: {np.mean(blur_psnr_values):.4f} dB")
print(f"Average Blurred SSIM: {np.mean(blur_ssim_values):.4f}")
print(f"Delta PSNR (Recon-Blur): {(np.mean(psnr_values) - np.mean(blur_psnr_values)):.4f} dB")
print(f"Delta SSIM (Recon-Blur): {(np.mean(ssim_values) - np.mean(blur_ssim_values)):.4f}")
print("="*60)

# Save results summary
summary_path = os.path.join(args.result_dir, 'evaluation_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f"ISTA-Net Deblurring Evaluation\n")
    f.write(f"Epoch: {epoch_num}\n")
    f.write(f"Layer Num: {layer_num}\n")
    f.write(f"Number of test images: {len(psnr_values)}\n")
    f.write(f"Average PSNR: {np.mean(psnr_values):.4f} dB\n")
    f.write(f"Average SSIM: {np.mean(ssim_values):.4f}\n")
    f.write(f"Average Blurred PSNR: {np.mean(blur_psnr_values):.4f} dB\n")
    f.write(f"Average Blurred SSIM: {np.mean(blur_ssim_values):.4f}\n")
    f.write(f"Delta PSNR (Recon-Blur): {(np.mean(psnr_values) - np.mean(blur_psnr_values)):.4f} dB\n")
    f.write(f"Delta SSIM (Recon-Blur): {(np.mean(ssim_values) - np.mean(blur_ssim_values)):.4f}\n")
    f.write(f"\nPer-image results:\n")
    for i, (psnr, ssim_val) in enumerate(zip(psnr_values, ssim_values)):
        f.write(f"Image {i+1}: PSNR = {psnr:.4f} dB, SSIM = {ssim_val:.4f}\n")

print(f"\nResults saved to: {args.result_dir}")
