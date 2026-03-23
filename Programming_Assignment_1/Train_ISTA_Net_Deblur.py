import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import glob
import cv2

parser = ArgumentParser(description='ISTA-Net for Image Deblurring')

parser.add_argument('--end_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--layer_num', type=int, default=10, help='number of ISTA layers')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='model directory')
parser.add_argument('--data_dir', type=str, default='archive', help='training data directory')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()

end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
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

batch_size = 64

# ============================================================================
# BLUR FUNCTIONS
# ============================================================================

def create_blur_kernel(kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return torch.from_numpy(kernel).to(device)


blur_kernel = create_blur_kernel(5)

def apply_blur(x):
    """
    Apply blur operator H to image x.
    x: [B, C, H, W] of CLIPPED values [0, 1] or similar
    """
    _, C, _, _ = x.shape
    
    # Prepare kernel for conv2d: [out_channels, in_channels, kH, kW]
    # Apply same kernel to all channels (depthwise convolution)
    kernel = blur_kernel.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 5]
    kernel = kernel.repeat(C, 1, 1, 1)  # [C, 1, 5, 5]
    
    # Apply same convolution (padding to maintain size)
    y = F.conv2d(x, kernel, padding=2, groups=C)
    
    return y


def apply_transpose_blur(x):
    """Apply H^T via conv_transpose2d."""
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
# DATASET
# ============================================================================

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform_size=256):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png'))) + \
                          sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
                          sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
        self.transform_size = transform_size
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Return random tensor if load fails (shouldn't happen in practice)
            return torch.randn(3, 256, 256)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize or center crop to 256x256
        h, w = img.shape[:2]
        if h < self.transform_size or w < self.transform_size:
            # Resize
            scale = self.transform_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        h, w = img.shape[:2]
        # Center crop
        top = (h - self.transform_size) // 2
        left = (w - self.transform_size) // 2
        img = img[top:top+self.transform_size, left:left+self.transform_size, :]
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        assert img.ndim == 3 and img.shape[2] == 3, f"Expected RGB image, got shape {img.shape}"
        assert img.shape[0] == 256 and img.shape[1] == 256, f"Expected 256x256 crop, got {img.shape[:2]}"
        assert img.min() >= 0.0 and img.max() <= 1.0, "Expected normalized image range [0,1]"
        
        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img_tensor

# Load dataset
train_dir = os.path.join(args.data_dir, 'images', 'train')
if not os.path.exists(train_dir):
    print(f"Error: Training directory not found: {train_dir}")
    exit(1)

train_dataset = DeblurDataset(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=0)

print(f"Loaded {len(train_dataset)} training images")

# ============================================================================
# MODEL: ISTA-Net for Deblurring
# ============================================================================

class ISTABlock(nn.Module):
    """Single ISTA layer for image deblurring."""
    def __init__(self):
        super(ISTABlock, self).__init__()
        
        self.eta_raw = nn.Parameter(torch.tensor(-3.0))
        self.lam_raw = nn.Parameter(torch.tensor(-8.0))
    
    def forward(self, x, y):
        """
        ISTA update step:
        z = x - eta * H^T(Hx - y)
        x_new = soft_threshold(z, lambda)
        """
        eta = F.softplus(self.eta_raw) + 1e-6
        lam = F.softplus(self.lam_raw) + 1e-6

        # Gradient step: z = x - eta * H^T(Hx - y)
        Hx = apply_blur(x)
        residual = Hx - y
        HT_residual = apply_transpose_blur(residual)
        
        z = x - eta * HT_residual
        
        # Soft thresholding
        x_soft = soft_threshold(z, lam)
        
        return x_soft

class ISTANet(nn.Module):
    """ISTA-Net for image deblurring."""
    def __init__(self, num_layers=10):
        super(ISTANet, self).__init__()
        self.num_layers = num_layers
        
        # Stack of ISTA blocks
        self.layers = nn.ModuleList([ISTABlock() for _ in range(num_layers)])
    
    def forward(self, y):
        """
        Args:
            y: blurred image [B, C, H, W]
        
        Returns:
            x: deblurred image [B, C, H, W]
        """
        # Initialize with blurred image
        x = y.clone()
        
        # Iterate through ISTA layers
        for layer in self.layers:
            x = layer(x, y)
        
        return x

model = ISTANet(num_layers=layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print(f"Initialized ISTA-Net with {layer_num} layers")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {num_params}")

# ============================================================================
# OPTIMIZER AND SETUP
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model_dir = "./%s/ISTA_Net_Deblur_layer_%d_lr_%.4f" % (args.model_dir, layer_num, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("Starting training...")

for epoch in range(1, end_epoch + 1):
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, x_clean in enumerate(train_loader):
        x_clean = x_clean.to(device)
        
        # Create blurred (degraded) image: y = Hx + w
        with torch.no_grad():
            y = apply_blur(x_clean)
            y = add_noise(y, sigma=0.05)
            y = torch.clamp(y, 0, 1)
        
        # Forward pass
        x_deblurred = model(y)
        
        # Compute loss
        loss = criterion(x_deblurred, x_clean)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    print(f"[Epoch {epoch}] Average Loss: {avg_loss:.6f}")
    
    # Save model every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'net_params_{epoch}.pkl'))
        print(f"Saved model at epoch {epoch}")

print("Training complete!")
print(f"Model saved to: {model_dir}")
