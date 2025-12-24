#!/usr/bin/env python3
"""
Model G: Virchow2 + Img2ST Decoder + MSE Loss (Clean 2x2 Factorial)

Part of the CLEAN 2x2 factorial design for comparing:
- Decoder architecture: Hist2ST (Transformer) vs Img2ST (UNet)
- Loss function: MSE vs Poisson

EXPERIMENTAL DESIGN (2x2 factorial - FIXED):
                  | MSE Loss (Softplus)  | Poisson Loss (log_rate) |
    Hist2ST       | Model D' (simple)    | Model E                 |
    Img2ST        | Model G (this)       | Model F                 |

CRITICAL: This script uses IDENTICAL settings to Model D' for fair comparison:
- Same loss: Simple masked MSE at 2µm only
- Same activation: Softplus (not ReLU!) for non-negative outputs
- Same supervision: 2µm only (no multi-scale pyramid)
- Same hyperparameters: lr=5e-5, batch=8, grad_accum=4, epochs=40

KEY ARCHITECTURE:
- Encoder: Virchow2 (1.1B params, frozen)
- Decoder: Img2ST-style MiniUNet with upsampling
- Output: Softplus activation (matches D' for fair comparison)
- Loss: Simple MSE at 2µm (masked to tissue regions)

USAGE:
    python train_2um_mse_img2st.py --test_patient P5 --save_dir results/model_g_img2st_mse
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Performance optimization for modern GPUs
torch.set_float32_matmul_precision('high')


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Joint Augmentation
# ============================================================================

class JointGeometricTransform:
    """Apply identical geometric transforms to image, labels, and mask."""

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.5, p_rot90: float = 0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(self, image: torch.Tensor, label_2um: torch.Tensor,
                 label_8um: torch.Tensor, mask_2um: torch.Tensor):
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[2])
            label_2um = torch.flip(label_2um, dims=[2])
            label_8um = torch.flip(label_8um, dims=[2])
            mask_2um = torch.flip(mask_2um, dims=[2])

        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[1])
            label_2um = torch.flip(label_2um, dims=[1])
            label_8um = torch.flip(label_8um, dims=[1])
            mask_2um = torch.flip(mask_2um, dims=[1])

        if torch.rand(1).item() < self.p_rot90:
            k = 1
            image = torch.rot90(image, k, dims=[1, 2])
            label_2um = torch.rot90(label_2um, k, dims=[1, 2])
            label_8um = torch.rot90(label_8um, k, dims=[1, 2])
            mask_2um = torch.rot90(mask_2um, k, dims=[1, 2])

        return image, label_2um, label_8um, mask_2um


# Add model path
model_base = Path(os.environ.get('VISIUM_MODEL_BASE', '/home/user/visium-hd-2um-benchmark'))
sys.path.insert(0, str(model_base))

from model.encoder_wrapper import get_spatial_encoder


# ============================================================================
# Infrastructure
# ============================================================================

def log_epoch(log_file: Path, epoch_data: dict):
    """Append one JSON line per epoch."""
    def convert_to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_to_native(x) for x in obj.tolist()]
        elif isinstance(obj, list):
            return [convert_to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        else:
            return obj

    clean_data = convert_to_native(epoch_data)
    with open(log_file, 'a') as f:
        f.write(json.dumps(clean_data) + '\n')


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def discover_patients(data_dir: str) -> list:
    """Discover patient directories with required patch files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    patients = []
    for child in sorted(data_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / 'patches_raw_counts.npy').exists():
            patients.append(child.name)
    return patients


# ============================================================================
# Dataset
# ============================================================================

class RawCountsSTDataset(Dataset):
    """Dataset with RAW COUNTS for training at 2um."""

    def __init__(self, data_dir, patient_id, num_genes=50, transform=None,
                 input_size=224, joint_transform=None):
        self.data_dir = Path(data_dir) / patient_id
        self.patient_id = patient_id
        self.num_genes = num_genes
        self.transform = transform
        self.joint_transform = joint_transform
        self.input_size = input_size

        patches_file = self.data_dir / 'patches_raw_counts.npy'
        if not patches_file.exists():
            raise FileNotFoundError(f"Raw counts file not found: {patches_file}")
        self.patches = np.load(patches_file, allow_pickle=True).tolist()

        with open(self.data_dir / 'gene_names.json') as f:
            self.gene_names = json.load(f)
            if isinstance(self.gene_names, dict):
                self.gene_names = self.gene_names.get('gene_names', list(self.gene_names.keys()))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        item = self.patches[idx]

        img_path_key = 'img_path' if 'img_path' in item else 'image_path'
        img_name = Path(item[img_path_key]).name
        img_path = self.data_dir / 'images' / img_name

        if not img_path.exists():
            alt_paths = [
                Path('/mnt/x/img2st_rotation_demo/processed_crc_raw_counts') / self.data_dir.name / 'images' / img_name,
            ]
            for alt in alt_paths:
                if alt.exists():
                    img_path = alt
                    break

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Load RAW COUNTS at 2um (128x128)
        label_2um_flat = torch.tensor(item['label_2um'], dtype=torch.float32)
        label_2um = label_2um_flat.reshape(128, 128, self.num_genes).permute(2, 0, 1)
        mask_2um = torch.tensor(item['mask_2um'], dtype=torch.float32).unsqueeze(0)

        # Generate 8um labels via sum-pooling
        label_4um = F.avg_pool2d(label_2um.unsqueeze(0), 2, 2).squeeze(0) * 4
        label_8um = F.avg_pool2d(label_4um.unsqueeze(0), 2, 2).squeeze(0) * 4

        if self.joint_transform is not None:
            img, label_2um, label_8um, mask_2um = self.joint_transform(
                img, label_2um, label_8um, mask_2um
            )

        raw_patch_id = item.get('patch_id', f"{item.get('patch_row', idx)}_{item.get('patch_col', 0)}")
        full_patch_id = f"{self.patient_id}_{raw_patch_id}"

        return {
            'image': img,
            'label_2um': label_2um,
            'label_8um': label_8um,
            'mask_2um': mask_2um,
            'patch_id': full_patch_id,
        }


# ============================================================================
# Img2ST-Style MiniUNet Decoder
# ============================================================================

class ConvBlock(nn.Module):
    """Two-layer conv block with BN and ReLU (from Img2ST)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Img2STDecoder(nn.Module):
    """
    Img2ST-style MiniUNet decoder adapted for Virchow2 encoder.

    For MSE training: outputs raw values (no log transform).
    Uses Softplus at output to ensure non-negative predictions (matches D').

    Note: Virchow2 outputs 14x14 features, so we need explicit 128x128 resize.
    """
    def __init__(self, in_ch=1024, mid_ch=512, out_ch=50):
        super().__init__()

        # Initial projection from Virchow2 features (14x14)
        self.input_proj = nn.Conv2d(in_ch, mid_ch, 1)

        # Encoder path
        self.enc1 = ConvBlock(mid_ch, mid_ch)  # 14x14
        self.pool = nn.MaxPool2d(2)            # 7x7
        self.enc2 = ConvBlock(mid_ch, mid_ch)  # 7x7

        # Decoder path with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(mid_ch + mid_ch, mid_ch)  # 14x14

        # Progressive upsampling (14->28->56->112)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(mid_ch, mid_ch // 2),  # 28x28, 256ch
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(mid_ch // 2, mid_ch // 4),  # 56x56, 128ch
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(mid_ch // 4, mid_ch // 8),  # 112x112, 64ch
        )

        # Final resize to exact 128x128
        self.final_resize = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)

        # Output head: raw values for MSE (with Softplus for non-negative)
        self.out_conv = nn.Conv2d(mid_ch // 8, out_ch, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Args:
            x: (B, 1024, 14, 14) Virchow2 encoder features
        Returns:
            (B, 50, 128, 128) predictions (non-negative)
        """
        # Project input
        x = self.input_proj(x)  # (B, 512, 14, 14)

        # Encoder with skip
        e1 = self.enc1(x)       # (B, 512, 14, 14)
        x = self.pool(e1)       # (B, 512, 7, 7)
        e2 = self.enc2(x)       # (B, 512, 7, 7)

        # Decoder with skip connection
        d1 = self.up1(e2)       # (B, 512, 14, 14)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 512, 14, 14)

        # Progressive upsampling
        d2 = self.up2(d1)       # (B, 256, 28, 28)
        d3 = self.up3(d2)       # (B, 128, 56, 56)
        d4 = self.up4(d3)       # (B, 64, 112, 112)

        # Final resize to exact 128x128
        d4 = self.final_resize(d4)  # (B, 64, 128, 128)

        # Output: raw values with Softplus for non-negative (matches D' for fair comparison)
        out = self.softplus(self.out_conv(d4))  # (B, 50, 128, 128)

        return out


# ============================================================================
# Full Model: Virchow2 + Img2ST Decoder
# ============================================================================

class HybridModel(nn.Module):
    """
    Hybrid Model G: Virchow2 encoder + Img2ST decoder + MSE.

    - Encoder: Virchow2 (frozen, 1.1B params)
    - Decoder: Img2ST-style MiniUNet (trainable)
    - Output: Non-negative values at 128x128 (2µm resolution)
    """

    def __init__(self, encoder_name='virchow2', num_genes=50, input_size=224):
        super().__init__()
        self.encoder_name = encoder_name
        self.input_size = input_size

        # Frozen encoder
        self.encoder = get_spatial_encoder(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Img2ST decoder
        self.decoder = Img2STDecoder(in_ch=1024, mid_ch=512, out_ch=num_genes)

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)  # (B, 1024, 16, 16)

        pred = self.decoder(features)  # (B, 50, 128, 128)

        return pred


# ============================================================================
# Training with 2um MSE Loss (MASKED)
# ============================================================================

def train_epoch_mse_2um(model, loader, optimizer, device, grad_accum=1, scaler=None):
    """Train for one epoch with MSE loss at 2um. Uses AMP for ~2x speedup."""
    model.train()
    total_loss = 0
    metrics = {'mean_pred': 0, 'mean_target': 0, 'valid_fraction': 0}
    n_batches = 0
    optimizer.zero_grad()

    total_batches = len(loader)
    last_window_size = total_batches % grad_accum
    if last_window_size == 0:
        last_window_size = grad_accum

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training (Img2ST + MSE)', leave=False)):
        images = batch['image'].to(device, non_blocking=True)
        label_2um = batch['label_2um'].to(device, non_blocking=True)  # RAW COUNTS at 2um
        mask_2um = batch['mask_2um'].to(device, non_blocking=True)  # [B, 1, 128, 128]

        # Forward pass with AMP autocast
        with torch.cuda.amp.autocast():
            # Forward: model outputs predictions at 2um (128x128)
            pred_2um = model(images)

            # Expand mask to match gene dimension
            mask_expanded = mask_2um.expand_as(pred_2um)  # [B, G, 128, 128]

            # MSE at 2um with masking
            mse = (pred_2um - label_2um) ** 2  # [B, G, 128, 128]

            # Apply mask and average only over valid regions
            valid_mask = mask_expanded > 0.5
            n_valid = valid_mask.sum()

            if n_valid > 0:
                loss = (mse * valid_mask.float()).sum() / n_valid
            else:
                # Don't learn from empty/garbage patches - zero gradient loss
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Gradient accumulation
        is_last_batch = (batch_idx + 1) == total_batches
        is_step_batch = (batch_idx + 1) % grad_accum == 0
        is_last_window = batch_idx >= (total_batches - last_window_size)
        actual_accum = last_window_size if is_last_window else grad_accum

        scaled_loss = loss / actual_accum

        # Use scaler for AMP backward pass
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if is_step_batch or is_last_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        with torch.no_grad():
            if n_valid > 0:
                metrics['mean_pred'] += (pred_2um * valid_mask.float()).sum().item() / n_valid.item()
                metrics['mean_target'] += (label_2um * valid_mask.float()).sum().item() / n_valid.item()
            metrics['valid_fraction'] += valid_mask.float().mean().item()

        n_batches += 1

    return total_loss / n_batches, {k: v / n_batches for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate at multiple scales."""
    model.eval()

    all_pred_8um, all_label_8um = [], []
    all_pred_2um, all_label_2um, all_mask_2um = [], [], []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        images = batch['image'].to(device)
        label_2um = batch['label_2um'].to(device)
        label_8um = batch['label_8um'].to(device)
        mask_2um = batch['mask_2um'].to(device)

        pred_2um = model(images)

        # Pool to 8um for secondary metrics
        pred_4um = F.avg_pool2d(pred_2um, 2, 2) * 4
        pred_8um = F.avg_pool2d(pred_4um, 2, 2) * 4

        all_pred_8um.append(pred_8um.cpu().numpy())
        all_label_8um.append(label_8um.cpu().numpy())
        all_pred_2um.append(pred_2um.cpu().numpy())
        all_label_2um.append(label_2um.cpu().numpy())
        all_mask_2um.append(mask_2um.cpu().numpy())

    pred_8um = np.concatenate(all_pred_8um)
    label_8um = np.concatenate(all_label_8um)
    pred_2um = np.concatenate(all_pred_2um)
    label_2um = np.concatenate(all_label_2um)
    mask_2um = np.concatenate(all_mask_2um)

    # 2um PCC (PRIMARY - masked)
    mask_2um_broadcast = np.broadcast_to(mask_2um, pred_2um.shape)
    mask_2um_flat = mask_2um_broadcast.flatten()
    valid_2um = mask_2um_flat > 0.5
    if valid_2um.sum() > 100:
        p = pred_2um.flatten()[valid_2um]
        l = label_2um.flatten()[valid_2um]
        pcc_2um, _ = pearsonr(p, l)
    else:
        pcc_2um = 0.0

    # Per-gene PCC at 2um (masked)
    n_genes = pred_2um.shape[1]
    gene_pccs = []
    for g in range(n_genes):
        p_gene = pred_2um[:, g, :, :].flatten()
        l_gene = label_2um[:, g, :, :].flatten()
        m_gene = mask_2um[:, 0, :, :].flatten()
        valid = m_gene > 0.5
        if valid.sum() > 100:
            try:
                r, _ = pearsonr(p_gene[valid], l_gene[valid])
                if not np.isnan(r):
                    gene_pccs.append(r)
            except:
                pass

    # 8um PCC (SECONDARY - masked)
    # Pool mask to 8um (max pooling to preserve any tissue region)
    mask_8um = np.zeros((mask_2um.shape[0], 1, 32, 32))
    for b in range(mask_2um.shape[0]):
        # Max pool: if any 2um pixel in 4x4 block has tissue, the 8um pixel has tissue
        m = mask_2um[b, 0]  # [128, 128]
        for i in range(32):
            for j in range(32):
                mask_8um[b, 0, i, j] = m[i*4:(i+1)*4, j*4:(j+1)*4].max()

    mask_8um_broadcast = np.broadcast_to(mask_8um, pred_8um.shape)
    mask_8um_flat = mask_8um_broadcast.flatten()
    valid_8um = mask_8um_flat > 0.5
    if valid_8um.sum() > 100:
        p8 = pred_8um.flatten()[valid_8um]
        l8 = label_8um.flatten()[valid_8um]
        pcc_8um, _ = pearsonr(p8, l8)
    else:
        pcc_8um = 0.0

    # SSIM at 2um (masked)
    ssim_2um_list = []
    n_samples = pred_2um.shape[0]
    for b in range(n_samples):
        sample_mask = mask_2um[b, 0]
        if sample_mask.mean() > 0.05:
            for g in range(n_genes):
                p_img = pred_2um[b, g]
                l_img = label_2um[b, g]
                masked_p = p_img * sample_mask
                masked_l = l_img * sample_mask
                combined = np.concatenate([masked_p.flatten(), masked_l.flatten()])
                vmin, vmax = combined.min(), combined.max()
                if vmax - vmin > 1e-6:
                    p_norm = (masked_p - vmin) / (vmax - vmin)
                    l_norm = (masked_l - vmin) / (vmax - vmin)
                    try:
                        s = ssim_metric(p_norm, l_norm, data_range=1.0)
                        if not np.isnan(s):
                            ssim_2um_list.append(s)
                    except Exception:
                        pass
    ssim_2um = np.mean(ssim_2um_list) if ssim_2um_list else 0.0

    # SSIM at 8um (masked to match PCC)
    ssim_8um_list = []
    for b in range(n_samples):
        sample_mask_8um = mask_8um[b, 0]  # [32, 32]
        if sample_mask_8um.mean() > 0.05:  # Skip mostly-empty patches
            for g in range(n_genes):
                p_img = pred_8um[b, g]
                l_img = label_8um[b, g]
                # Apply mask to focus on tissue regions
                masked_p = p_img * sample_mask_8um
                masked_l = l_img * sample_mask_8um
                combined = np.concatenate([masked_p.flatten(), masked_l.flatten()])
                vmin, vmax = combined.min(), combined.max()
                if vmax - vmin > 1e-6:
                    p_norm = (masked_p - vmin) / (vmax - vmin)
                    l_norm = (masked_l - vmin) / (vmax - vmin)
                    try:
                        s = ssim_metric(p_norm, l_norm, data_range=1.0)
                        if not np.isnan(s):
                            ssim_8um_list.append(s)
                    except Exception:
                        pass
    ssim_8um = np.mean(ssim_8um_list) if ssim_8um_list else 0.0

    return {
        'pcc_2um': pcc_2um,
        'pcc_8um': pcc_8um,
        'ssim_2um': ssim_2um,
        'ssim_8um': ssim_8um,
        'pcc_2um_per_gene_mean': np.mean(gene_pccs) if gene_pccs else 0.0,
        'pcc_2um_per_gene_std': np.std(gene_pccs) if gene_pccs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Comparison Model G: Virchow2 + Img2ST + MSE')

    # Data
    parser.add_argument('--test_patient', type=str, required=True)
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/x/img2st_rotation_demo/processed_crc_raw_counts')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/x/virchow2-decoder-benchmark/results/hybrid_virchow2_img2st_mse')

    # Model
    parser.add_argument('--encoder', type=str, default='virchow2')
    parser.add_argument('--num_genes', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=224)

    # Training (SAME as all other models for fair comparison)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discovered_patients = discover_patients(args.data_dir)
    all_patients = discovered_patients if discovered_patients else ['P1', 'P2', 'P5']

    if args.test_patient not in all_patients:
        raise ValueError(f"test_patient {args.test_patient} not found")
    train_patients = [p for p in all_patients if p != args.test_patient]

    print(f"\n{'='*60}")
    print(f"COMPARISON MODEL G: Virchow2 + Img2ST + MSE")
    print(f"{'='*60}")
    print(f"HYPOTHESIS: Img2ST + MSE should ALSO fail (~0.15-0.20 PCC)")
    print(f"Train: {train_patients}, Test: {args.test_patient}")
    print(f"Encoder: Virchow2 (frozen)")
    print(f"Decoder: Img2ST-style MiniUNet")
    print(f"Loss: MSE at 2um with tissue masking")
    print(f"{'='*60}\n")

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f'test{args.test_patient}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['git_commit'] = get_git_commit()
    config['model_type'] = 'hybrid_virchow2_img2st_mse'
    config['decoder_type'] = 'img2st_miniunet'
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Transforms (SAME as all other models)
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_joint_transform = JointGeometricTransform(p_hflip=0.5, p_vflip=0.5, p_rot90=0.5)

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_datasets = [RawCountsSTDataset(args.data_dir, p, args.num_genes, train_transform,
                                          joint_transform=train_joint_transform)
                      for p in train_patients]
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = RawCountsSTDataset(args.data_dir, args.test_patient,
                                       args.num_genes, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = HybridModel(args.encoder, args.num_genes, input_size=args.input_size).to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Trainable decoder params: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    # Training loop - SSIM as primary metric for structural fidelity
    best_ssim = -1
    patience_counter = 0
    log_file = save_dir / 'training_log.jsonl'

    # AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch_mse_2um(
            model, train_loader, optimizer, device, grad_accum=args.grad_accum, scaler=scaler
        )
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
        }
        log_epoch(log_file, epoch_data)

        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, "
              f"SSIM[2um={test_metrics['ssim_2um']:.3f}], PCC[2um={test_metrics['pcc_2um']:.3f}, 8um={test_metrics['pcc_8um']:.3f}], "
              f"pred={train_metrics['mean_pred']:.2f}/tgt={train_metrics['mean_target']:.2f}")

        # Save best based on 2um SSIM (structural fidelity over correlation)
        if test_metrics['ssim_2um'] > best_ssim:
            best_ssim = test_metrics['ssim_2um']
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')

            with open(save_dir / 'best_metrics.json', 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'ssim_2um': float(test_metrics['ssim_2um']),
                    'ssim_8um': float(test_metrics['ssim_8um']),
                    'pcc_2um': float(test_metrics['pcc_2um']),
                    'pcc_8um': float(test_metrics['pcc_8um']),
                    'pcc_2um_per_gene_mean': float(test_metrics['pcc_2um_per_gene_mean']),
                }, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest 2um SSIM: {best_ssim:.4f}")
    print(f"Results saved to {save_dir}")

    # Save predictions
    print("\nSaving predictions...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    model.eval()

    all_pred_2um = []
    all_label_2um = []
    all_mask_2um = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            pred = model(images)
            all_pred_2um.append(pred.cpu().numpy())
            all_label_2um.append(batch['label_2um'].numpy())
            all_mask_2um.append(batch['mask_2um'].numpy())

    np.save(save_dir / 'pred_2um.npy', np.concatenate(all_pred_2um))
    np.save(save_dir / 'label_2um.npy', np.concatenate(all_label_2um))
    np.save(save_dir / 'mask_2um.npy', np.concatenate(all_mask_2um))

    print("Done!")


if __name__ == '__main__':
    main()
