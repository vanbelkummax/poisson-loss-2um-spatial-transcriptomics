#!/usr/bin/env python3
"""
Generate figures for 2×2 Factorial Analysis (Decoder × Loss)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150

# Results from experiments
RESULTS = {
    "D'": {"decoder": "Hist2ST", "loss": "MSE", "ssim_2um": 0.227, "pcc_2um": 0.309, "pcc_8um": 0.457, "pred_tgt": 2.0, "epoch": 40},
    "E'": {"decoder": "Hist2ST", "loss": "Poisson", "ssim_2um": 0.576, "pcc_2um": 0.313, "pcc_8um": 0.466, "pred_tgt": 1.0, "epoch": 22},
    "F":  {"decoder": "Img2ST", "loss": "Poisson", "ssim_2um": 0.388, "pcc_2um": 0.298, "pcc_8um": 0.441, "pred_tgt": 1.0, "epoch": 35},
    "G":  {"decoder": "Img2ST", "loss": "MSE", "ssim_2um": 0.154, "pcc_2um": 0.260, "pcc_8um": 0.382, "pred_tgt": 4.4, "epoch": 36},
}

def setup_output_dir():
    output_dir = Path(__file__).parent.parent / "figures" / "factorial_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def figure1_interaction_plot(output_dir):
    """
    Figure 1: Interaction Plot - The signature 2×2 factorial visualization
    Shows non-parallel lines indicating interaction between Loss and Decoder
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points
    x = [0, 1]  # MSE, Poisson
    hist2st = [RESULTS["D'"]["ssim_2um"], RESULTS["E'"]["ssim_2um"]]
    img2st = [RESULTS["G"]["ssim_2um"], RESULTS["F"]["ssim_2um"]]

    # Plot lines
    ax.plot(x, hist2st, 'o-', linewidth=3, markersize=12, color='#2E86AB', label='Hist2ST (CNN+Trans+GNN)')
    ax.plot(x, img2st, 's--', linewidth=3, markersize=12, color='#E94F37', label='Img2ST (MiniUNet)')

    # Annotations
    ax.annotate(f"E' = {RESULTS['E\'']['ssim_2um']:.3f}", (1, RESULTS["E'"]["ssim_2um"]),
                xytext=(15, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    ax.annotate(f"D' = {RESULTS['D\'']['ssim_2um']:.3f}", (0, RESULTS["D'"]["ssim_2um"]),
                xytext=(-50, -15), textcoords='offset points', fontsize=11)
    ax.annotate(f"F = {RESULTS['F']['ssim_2um']:.3f}", (1, RESULTS["F"]["ssim_2um"]),
                xytext=(15, -10), textcoords='offset points', fontsize=11)
    ax.annotate(f"G = {RESULTS['G']['ssim_2um']:.3f}", (0, RESULTS["G"]["ssim_2um"]),
                xytext=(-50, 5), textcoords='offset points', fontsize=11)

    # Formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MSE', 'Poisson'], fontsize=14)
    ax.set_xlabel('Loss Function', fontsize=14)
    ax.set_ylabel('SSIM at 2µm', fontsize=14)
    ax.set_title('Interaction Plot: Decoder × Loss\n(Non-parallel lines = interaction effect)', fontsize=14)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation about interaction
    ax.text(0.5, 0.05, 'Poisson rescue stronger for Hist2ST (+0.349) than Img2ST (+0.234)',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "figure1_interaction_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure1_interaction_plot.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Interaction plot saved")


def figure2_grouped_bar(output_dir):
    """
    Figure 2: Grouped bar chart comparing all 4 models across metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    models = ["D'", "E'", "F", "G"]
    colors = ['#A8DADC', '#457B9D', '#F4A261', '#E76F51']  # MSE-Hist, Pois-Hist, Pois-Img, MSE-Img

    metrics = [
        ('ssim_2um', 'SSIM at 2µm\n(Structural Similarity)', (0, 0.7)),
        ('pcc_2um', 'PCC at 2µm\n(Pixel Correlation)', (0, 0.4)),
        ('pcc_8um', 'PCC at 8µm\n(Global Coherence)', (0, 0.6)),
    ]

    for ax, (metric, label, ylim) in zip(axes, metrics):
        values = [RESULTS[m][metric] for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(label)
        ax.set_ylim(ylim)
        ax.set_xlabel('Model')

        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('#2D6A4F')
        bars[best_idx].set_linewidth(3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#A8DADC', edgecolor='black', label="D': Hist2ST + MSE"),
        mpatches.Patch(facecolor='#457B9D', edgecolor='black', label="E': Hist2ST + Poisson"),
        mpatches.Patch(facecolor='#F4A261', edgecolor='black', label="F: Img2ST + Poisson"),
        mpatches.Patch(facecolor='#E76F51', edgecolor='black', label="G: Img2ST + MSE"),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.suptitle('2×2 Factorial Results: All Metrics', y=1.08, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_grouped_bar.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure2_grouped_bar.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Grouped bar chart saved")


def figure3_heatmap(output_dir):
    """
    Figure 3: Heatmap of all metrics across models
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ["D' (Hist+MSE)", "E' (Hist+Pois)", "F (Img+Pois)", "G (Img+MSE)"]
    metrics = ['SSIM_2µm', 'PCC_2µm', 'PCC_8µm', '1/Overshoot']

    # Build data matrix (normalize for visualization, higher = better)
    data = np.array([
        [RESULTS["D'"]["ssim_2um"], RESULTS["D'"]["pcc_2um"], RESULTS["D'"]["pcc_8um"], 1/RESULTS["D'"]["pred_tgt"]],
        [RESULTS["E'"]["ssim_2um"], RESULTS["E'"]["pcc_2um"], RESULTS["E'"]["pcc_8um"], 1/RESULTS["E'"]["pred_tgt"]],
        [RESULTS["F"]["ssim_2um"], RESULTS["F"]["pcc_2um"], RESULTS["F"]["pcc_8um"], 1/RESULTS["F"]["pred_tgt"]],
        [RESULTS["G"]["ssim_2um"], RESULTS["G"]["pcc_2um"], RESULTS["G"]["pcc_8um"], 1/RESULTS["G"]["pred_tgt"]],
    ])

    # Normalize each column to [0, 1] for heatmap
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_yticklabels(models, fontsize=12)

    # Add text annotations with actual values
    for i in range(len(models)):
        for j in range(len(metrics)):
            val = data[i, j]
            text = ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                          fontsize=11, fontweight='bold',
                          color='white' if data_norm[i, j] < 0.5 else 'black')

    ax.set_title('Performance Heatmap: Higher = Better (Green)', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Performance', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "figure3_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure3_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Heatmap saved")


def figure5_overshoot(output_dir):
    """
    Figure 5: Overshoot (Pred/Tgt) vs SSIM scatter plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ["D'", "E'", "F", "G"]
    colors = {'MSE': '#E63946', 'Poisson': '#2A9D8F'}
    markers = {'Hist2ST': 'o', 'Img2ST': 's'}

    for model in models:
        r = RESULTS[model]
        ax.scatter(r['ssim_2um'], r['pred_tgt'],
                  c=colors[r['loss']],
                  marker=markers[r['decoder']],
                  s=200, edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(model, (r['ssim_2um'], r['pred_tgt']),
                   xytext=(10, 5), textcoords='offset points', fontsize=12, fontweight='bold')

    # Reference line at pred/tgt = 1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Perfect calibration')

    # Formatting
    ax.set_xlabel('SSIM at 2µm', fontsize=14)
    ax.set_ylabel('Prediction / Target Ratio', fontsize=14)
    ax.set_title('The Overshoot Problem: MSE Predicts Too High\n(Optimal = 1.0×)', fontsize=14)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946',
                   markersize=12, markeredgecolor='black', label='MSE Loss'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2A9D8F',
                   markersize=12, markeredgecolor='black', label='Poisson Loss'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=12, markeredgecolor='black', label='Hist2ST'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=12, markeredgecolor='black', label='Img2ST'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Annotation
    ax.annotate('MSE causes\nmassive overshoot', xy=(0.19, 3.2), fontsize=11,
               ha='center', style='italic', color='#E63946',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate('Poisson = calibrated', xy=(0.48, 1.0), fontsize=11,
               ha='center', style='italic', color='#2A9D8F',
               xytext=(0.48, 1.8), arrowprops=dict(arrowstyle='->', color='#2A9D8F'))

    plt.tight_layout()
    plt.savefig(output_dir / "figure5_overshoot.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure5_overshoot.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Overshoot analysis saved")


def figure8_effect_sizes(output_dir):
    """
    Figure 8: Effect size forest plot
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Calculate effects
    effects = {
        'Loss Effect\n(Poisson - MSE)': {
            'ssim': (RESULTS["E'"]["ssim_2um"] + RESULTS["F"]["ssim_2um"])/2 -
                    (RESULTS["D'"]["ssim_2um"] + RESULTS["G"]["ssim_2um"])/2,
            'baseline': (RESULTS["D'"]["ssim_2um"] + RESULTS["G"]["ssim_2um"])/2
        },
        'Decoder Effect\n(Hist2ST - Img2ST)': {
            'ssim': (RESULTS["D'"]["ssim_2um"] + RESULTS["E'"]["ssim_2um"])/2 -
                    (RESULTS["G"]["ssim_2um"] + RESULTS["F"]["ssim_2um"])/2,
            'baseline': (RESULTS["G"]["ssim_2um"] + RESULTS["F"]["ssim_2um"])/2
        },
        'Poisson + Hist2ST\nvs MSE + Img2ST': {
            'ssim': RESULTS["E'"]["ssim_2um"] - RESULTS["G"]["ssim_2um"],
            'baseline': RESULTS["G"]["ssim_2um"]
        }
    }

    y_pos = np.arange(len(effects))
    effect_names = list(effects.keys())
    effect_values = [effects[k]['ssim'] for k in effect_names]
    percent_changes = [effects[k]['ssim'] / effects[k]['baseline'] * 100 for k in effect_names]

    # Horizontal bar chart
    colors = ['#2A9D8F', '#457B9D', '#E76F51']
    bars = ax.barh(y_pos, percent_changes, color=colors, edgecolor='black', linewidth=2, height=0.6)

    # Add value labels
    for bar, pct, delta in zip(bars, percent_changes, effect_values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
               f'+{pct:.0f}% (Δ={delta:.3f})', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(effect_names, fontsize=12)
    ax.set_xlabel('Effect Size (% improvement in SSIM_2µm)', fontsize=14)
    ax.set_title('Effect Sizes: What Matters Most?', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlim(-20, 350)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "figure8_effect_sizes.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure8_effect_sizes.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Effect sizes saved")


def table1_summary(output_dir):
    """
    Generate Table 1 as a figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Table data
    columns = ['Model', 'Decoder', 'Loss', 'SSIM_2µm', 'PCC_2µm', 'PCC_8µm', 'Pred/Tgt', 'Epoch']
    rows = []
    for model in ["D'", "E'", "F", "G"]:
        r = RESULTS[model]
        rows.append([model, r['decoder'], r['loss'], f"{r['ssim_2um']:.3f}",
                    f"{r['pcc_2um']:.3f}", f"{r['pcc_8um']:.3f}",
                    f"{r['pred_tgt']:.1f}×", str(r['epoch'])])

    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8']*8)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Highlight best SSIM
    table[(2, 3)].set_facecolor('#90EE90')  # E' SSIM
    table[(2, 3)].set_text_props(fontweight='bold')

    # Highlight worst
    table[(4, 3)].set_facecolor('#FFB6C1')  # G SSIM

    ax.set_title('Table 1: Complete 2×2 Factorial Results', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_dir / "table1_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "table1_summary.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Table 1: Summary table saved")


def table2_effects(output_dir):
    """
    Generate Table 2: Effect decomposition
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Main effects
    mse_mean_ssim = (RESULTS["D'"]["ssim_2um"] + RESULTS["G"]["ssim_2um"]) / 2
    pois_mean_ssim = (RESULTS["E'"]["ssim_2um"] + RESULTS["F"]["ssim_2um"]) / 2
    hist_mean_ssim = (RESULTS["D'"]["ssim_2um"] + RESULTS["E'"]["ssim_2um"]) / 2
    img_mean_ssim = (RESULTS["G"]["ssim_2um"] + RESULTS["F"]["ssim_2um"]) / 2

    columns = ['Effect', 'Condition 1', 'Condition 2', 'Δ SSIM', '% Change']
    rows = [
        ['Loss Effect', f'MSE: {mse_mean_ssim:.3f}', f'Poisson: {pois_mean_ssim:.3f}',
         f'+{pois_mean_ssim - mse_mean_ssim:.3f}', f'+{(pois_mean_ssim/mse_mean_ssim - 1)*100:.0f}%'],
        ['Decoder Effect', f'Img2ST: {img_mean_ssim:.3f}', f'Hist2ST: {hist_mean_ssim:.3f}',
         f'+{hist_mean_ssim - img_mean_ssim:.3f}', f'+{(hist_mean_ssim/img_mean_ssim - 1)*100:.0f}%'],
        ['Interaction', "D'→E': +0.349", "G→F: +0.234", '+0.115', 'Poisson helps Hist2ST more'],
    ]

    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    ax.set_title('Table 2: Factorial Effect Decomposition', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_dir / "table2_effects.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "table2_effects.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Table 2: Effect decomposition saved")


def main():
    print("=" * 60)
    print("Generating 2×2 Factorial Analysis Figures")
    print("=" * 60)

    output_dir = setup_output_dir()
    print(f"Output directory: {output_dir}\n")

    # Generate all figures
    figure1_interaction_plot(output_dir)
    figure2_grouped_bar(output_dir)
    figure3_heatmap(output_dir)
    figure5_overshoot(output_dir)
    figure8_effect_sizes(output_dir)
    table1_summary(output_dir)
    table2_effects(output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
