# Poisson Loss for 2um Spatial Transcriptomics

**Systematic proof that Poisson loss is critical for high-resolution (2um) spatial transcriptomics prediction.**

---

## The Sparsity Trap

At 2um resolution, ~95% of spatial bins contain **zero UMI counts**. Standard MSE loss fails catastrophically:

| Patient | Non-zero Fraction |
|---------|-------------------|
| P1 | 3.4% |
| P2 | 5.2% |
| P5 | 6.1% |

**MSE Loss**: Predicting zero everywhere minimizes loss on sparse data.

**Poisson NLL Loss**: `L = rate - k * log(rate)` - Predicting rate->0 when k>0 gives INFINITE loss, forcing the model to predict high values where counts exist.

---

## 2x2 Factorial Experiment

Clean factorial design: **Decoder Architecture** x **Loss Function**

|            | MSE Loss | Poisson Loss |
|------------|----------|--------------|
| **Hist2ST** | Model D' | Model E'    |
| **Img2ST**  | Model G  | Model F     |

All models use identical hyperparameters:
- Learning rate: 5e-5
- Batch size: 8
- Epochs: 40
- Patience: 10
- Gradient accumulation: 4

---

## Results (Patient P5)

| Model | Decoder | Loss | SSIM_2um | PCC_2um | PCC_8um | Best Epoch | Pred/Tgt |
|-------|---------|------|----------|---------|---------|------------|----------|
| **D'** | Hist2ST | MSE | 0.227 | 0.309 | 0.457 | 40 | 2.0x |
| **E'** | Hist2ST | Poisson | **0.576** | 0.313 | 0.466 | 22 | 1.0x |
| **F** | Img2ST | Poisson | 0.388 | 0.298 | 0.441 | 35 | 1.0x |
| **G** | Img2ST | MSE | 0.154 | 0.260 | 0.382 | 36 | 4.4x |

---

## Key Findings

### 1. Poisson Rescue (D' vs E')
Switching from MSE to Poisson loss **2.5x improves SSIM** (0.227 -> 0.576) while maintaining global coherence (PCC_8um: 0.457 -> 0.466).

### 2. Decoder Architecture Matters (E' vs F)
Hist2ST achieves **48% better SSIM** than Img2ST with identical Poisson loss (0.576 vs 0.388). The CNN+Transformer+GNN architecture captures fine texture detail better than pure convolutions.

### 3. MSE + Img2ST = Worst Combination (G)
Model G shows the most severe overshoot (4.4x) and lowest metrics (SSIM=0.154), confirming MSE fails catastrophically on sparse 2um data regardless of decoder architecture.

---

## Architecture

```
H&E Image (224x224 pixels)
       |
Virchow2 Encoder (frozen, 632M params)
       | [1024-dim features]
Decoder (Hist2ST or Img2ST)
       |
Gene expression predictions (128x128 x 50 genes)
```

**Hist2ST Decoder**: CNN + Transformer + GNN (~45M params)
**Img2ST Decoder**: MiniUNet with skip connections (~19M params)

---

## Usage

### Train Model E' (Hist2ST + Poisson)
```bash
python scripts/train_2um_poisson.py \
    --test_patient P5 \
    --data_dir /path/to/processed_crc_raw_counts \
    --save_dir results/model_E_prime
```

### Train Model G (Img2ST + MSE)
```bash
python scripts/train_2um_mse_img2st.py \
    --test_patient P5 \
    --data_dir /path/to/processed_crc_raw_counts \
    --save_dir results/model_G
```

---

## Data

- **Dataset**: 10x Genomics Visium HD CRC
- **Resolution**: 2um bins (128x128 per patch)
- **Genes**: Top 50 by variance
- **Patients**: P1, P2 (train), P5 (test)
- **Preprocessing**: Raw UMI counts, no normalization

---

## Scripts

| Script | Model | Decoder | Loss |
|--------|-------|---------|------|
| `train_2um_mse_hist2st.py` | D' | Hist2ST | MSE |
| `train_2um_poisson.py` | E' | Hist2ST | Poisson |
| `train_2um_poisson_img2st.py` | F | Img2ST | Poisson |
| `train_2um_mse_img2st.py` | G | Img2ST | MSE |

---

## Citation

```bibtex
@software{vanbelkum2024poisson_2um,
  author = {Van Belkum, Max},
  title = {Poisson Loss for 2um Spatial Transcriptomics: The Sparsity Trap},
  year = {2024},
  url = {https://github.com/vanbelkummax/poisson-loss-2um-spatial-transcriptomics}
}
```

---

## License

MIT License
