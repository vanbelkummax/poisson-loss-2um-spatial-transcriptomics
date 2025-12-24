#!/usr/bin/env python3
"""
Key Ablation (Model E): 2um Poisson Training

This script trains with Poisson NLL loss at 2um resolution ONLY.
This is the CRITICAL experiment that completes the 2x2 factorial design.

EXPERIMENTAL DESIGN:
                  | MSE Loss        | Poisson Loss     |
    8um Resolution| Model A (0.511) | Model C (0.524)  |
    2um Resolution| Model D (fails) | Model E (this)   |

HYPOTHESIS:
If Model E succeeds (high PCC at 2um) while Model D fails:
- Proves Poisson loss is CRITICAL for sparse 2um count data
- Proves MSE's "regression to the mean" is the cause of failure, not resolution

KEY DESIGN CHOICES:
- Loss: Poisson NLL at 2um ONLY (masked to valid tissue regions)
- Model output: log(rate) at 128x128 (native 2um)
- Masking: Only supervise tissue spots (using mask_2um)

EXPECTED RESULT:
- This model should WORK because Poisson properly handles sparse counts
- Model D fails because MSE penalizes deviations from zero weakly

USAGE:
    python train_2um_poisson.py --test_patient P5 --save_dir results/baseline_2um_poisson
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
from model.enhanced_decoder import Hist2STDecoder


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
    """Dataset with RAW COUNTS for Poisson loss training at 2um."""

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
# Model
# ============================================================================

class MiniUNet(nn.Module):
    """Simple UNet-style decoder."""
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, hidden_dim, 4, 2, 1)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1)
        self.up3 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1)
        self.up4 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1)
        self.conv4 = nn.Conv2d(hidden_dim // 8, out_channels, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.up1(x))
        x = self.act(self.conv1(x))
        x = self.act(self.up2(x))
        x = self.act(self.conv2(x))
        x = self.act(self.up3(x))
        x = self.act(self.conv3(x))
        x = self.act(self.up4(x))
        x = self.conv4(x)
        return x


class DecoderModel(nn.Module):
    """
    Encoder + Decoder for Poisson loss at 2um.

    KEY: use_softplus=False to output log(rate) for Poisson NLL.
    Initialize bias to -3.0 so initial predictions are low (exp(-3) ~ 0.05).
    """

    def __init__(self, encoder_name, decoder_type, num_genes=50, input_size=224):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_type = decoder_type
        self.input_size = input_size

        self.encoder = get_spatial_encoder(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        enc_dim = 1024

        if decoder_type == 'miniunet':
            self.decoder = MiniUNet(enc_dim, num_genes)
            nn.init.constant_(self.decoder.conv4.bias, -3.0)
        elif decoder_type == 'hist2st':
            self.decoder = Hist2STDecoder(
                in_ch=enc_dim,
                hidden_ch=512,
                out_ch=512,
                num_heads=8,
                k_neighbors=8,
                dropout=0.1
            )
            self.hist2st_head = nn.Conv2d(512, num_genes, 1)
            nn.init.constant_(self.hist2st_head.bias, -3.0)
        else:
            raise ValueError(f"Unknown decoder: {decoder_type}")

        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)

        pred = self.decoder(features)

        if hasattr(self, 'hist2st_head'):
            pred = self.hist2st_head(pred)

        if pred.shape[-1] != 128:
            pred = self.upsample(pred)

        # Output is log(rate) - no activation
        return pred


# ============================================================================
# Training with 2um Poisson Loss (MASKED)
# ============================================================================

def train_epoch_poisson_2um(model, loader, optimizer, device, grad_accum=1, scaler=None):
    """
    Train for one epoch with Poisson NLL loss at 2um ONLY.

    KEY DIFFERENCE from Model C (8um Poisson):
    - Supervise at 2um (128x128) instead of 8um (32x32)
    - Apply mask_2um to only supervise valid tissue regions
    - This is CRITICAL for sparse data - we don't want to penalize empty regions
    - Uses AMP (Automatic Mixed Precision) for ~2x speedup

    The model outputs log(rate) at 2um (128x128).
    We apply mask and compute Poisson NLL only on valid spots.
    """
    model.train()
    total_loss = 0
    metrics = {'mean_rate': 0, 'mean_target': 0, 'valid_fraction': 0}
    n_batches = 0
    optimizer.zero_grad()

    total_batches = len(loader)
    last_window_size = total_batches % grad_accum
    if last_window_size == 0:
        last_window_size = grad_accum

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training (2um Poisson)', leave=False)):
        images = batch['image'].to(device, non_blocking=True)
        label_2um = batch['label_2um'].to(device, non_blocking=True)  # RAW COUNTS at 2um
        mask_2um = batch['mask_2um'].to(device, non_blocking=True)  # [B, 1, 128, 128]

        # Forward pass with AMP autocast
        with torch.cuda.amp.autocast():
            # Forward: model outputs log(rate) at 2um (128x128)
            pred_log_rate_2um = model(images)

            # Expand mask to match gene dimension
            mask_expanded = mask_2um.expand_as(pred_log_rate_2um)  # [B, G, 128, 128]

            # Poisson NLL at 2um with masking
            # Poisson NLL: loss = exp(log_input) - target * log_input
            # Clamp log_rate to avoid exp() overflow (max ~85 for float32)
            pred_log_rate_2um = torch.clamp(pred_log_rate_2um, max=20.0)
            rate_2um = torch.exp(pred_log_rate_2um)

            # Masked Poisson NLL
            nll = rate_2um - label_2um * pred_log_rate_2um  # [B, G, 128, 128]

            # Apply mask and average only over valid regions
            valid_mask = mask_expanded > 0.5
            n_valid = valid_mask.sum()

            if n_valid > 0:
                loss = (nll * valid_mask.float()).sum() / n_valid
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
                metrics['mean_rate'] += (rate_2um * valid_mask.float()).sum().item() / n_valid.item()
                metrics['mean_target'] += (label_2um * valid_mask.float()).sum().item() / n_valid.item()
            metrics['valid_fraction'] += valid_mask.float().mean().item()

        n_batches += 1

    return total_loss / n_batches, {k: v / n_batches for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate at multiple scales.

    PRIMARY METRIC: 2um PCC (what we're training on)
    SECONDARY: 8um PCC (for comparison with Model C)
    """
    model.eval()

    all_pred_8um, all_label_8um = [], []
    all_pred_2um, all_label_2um, all_mask_2um = [], [], []
    per_gene_pccs = []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        images = batch['image'].to(device)
        label_2um = batch['label_2um'].to(device)
        label_8um = batch['label_8um'].to(device)
        mask_2um = batch['mask_2um'].to(device)

        # Clamp log-rates to prevent exp() overflow (matches training)
        pred_log_rate_2um = torch.clamp(model(images), max=20.0)
        pred_rate_2um = torch.exp(pred_log_rate_2um)

        # Pool to 8um for secondary metrics
        pred_rate_4um = F.avg_pool2d(pred_rate_2um, 2, 2) * 4
        pred_rate_8um = F.avg_pool2d(pred_rate_4um, 2, 2) * 4

        all_pred_8um.append(pred_rate_8um.cpu().numpy())
        all_label_8um.append(label_8um.cpu().numpy())
        all_pred_2um.append(pred_rate_2um.cpu().numpy())
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
    parser = argparse.ArgumentParser(description='Key Ablation (Model E): 2um Poisson')

    # Data
    parser.add_argument('--test_patient', type=str, required=True)
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/x/img2st_rotation_demo/processed_crc_raw_counts')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/x/virchow2-decoder-benchmark/results/baseline_2um_poisson')

    # Model
    parser.add_argument('--encoder', type=str, default='virchow2')
    parser.add_argument('--decoder', type=str, default='hist2st')
    parser.add_argument('--num_genes', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=224)

    # Training
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
    print(f"KEY ABLATION (Model E): 2um Poisson Training")
    print(f"{'='*60}")
    print(f"HYPOTHESIS: Poisson loss enables 2um prediction (Model D fails with MSE)")
    print(f"Train: {train_patients}, Test: {args.test_patient}")
    print(f"Loss: Poisson NLL at 2um with tissue masking")
    print(f"{'='*60}\n")

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f'test{args.test_patient}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['git_commit'] = get_git_commit()
    config['model_type'] = 'baseline_2um_poisson'
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Transforms
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
    model = DecoderModel(args.encoder, args.decoder, args.num_genes,
                         input_size=args.input_size).to(device)

    trainable_params = list(model.decoder.parameters())
    if hasattr(model, 'hist2st_head'):
        trainable_params += list(model.hist2st_head.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    # Training loop - SSIM as primary metric for structural fidelity
    best_ssim = -1
    patience_counter = 0
    log_file = save_dir / 'training_log.jsonl'

    # AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch_poisson_2um(
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
              f"rate={train_metrics['mean_rate']:.2f}/tgt={train_metrics['mean_target']:.2f}")

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
            pred_log_rate = model(images)
            pred_rate = torch.exp(pred_log_rate)
            all_pred_2um.append(pred_rate.cpu().numpy())
            all_label_2um.append(batch['label_2um'].numpy())
            all_mask_2um.append(batch['mask_2um'].numpy())

    np.save(save_dir / 'pred_2um.npy', np.concatenate(all_pred_2um))
    np.save(save_dir / 'label_2um.npy', np.concatenate(all_label_2um))
    np.save(save_dir / 'mask_2um.npy', np.concatenate(all_mask_2um))

    print("Done!")


if __name__ == '__main__':
    main()
