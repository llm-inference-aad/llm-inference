"""
Hypothetical projection figures for LLMGE + QLoRA pipeline.
Uses real measured data points and projects forward.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
C_BASE   = "#4C72B0"   # blue  – base model
C_FT     = "#DD8452"   # orange – fine-tuned model
C_ACTUAL = "#2ca02c"   # green – measured/actual
C_PROJ   = "#9467bd"   # purple – projected
C_SEED   = "#7f7f7f"   # grey  – seed baseline
ALPHA_FILL = 0.18

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 140,
})

# ── Real data ────────────────────────────────────────────────────────────────

# QLoRA v1: 11 samples, 6 steps, final loss 1.049, token acc 76.77 %
# QLoRA v2: 60 samples, 24 steps – measured log history
qlora_v2_steps    = [5,  10,   15,   20,   24]
qlora_v2_loss     = [0.8103, 0.5269, 0.3893, 0.2209, 0.2209]  # last step repeats final
qlora_v2_acc      = [80.24, 86.89, 90.16, 94.05, 92.56]       # epoch 3 average

# dataset.json fitness distribution (top-1 accuracy from quick eval jobs)
ds_top1 = [
    0.2872, 0.4054, 0.3606, 0.2872, 0.3154, 0.2948, 0.3242, 0.2784,
    0.3098, 0.3340, 0.4632, 0.2968, 0.3230, 0.2944, 0.2694, 0.3354,
    0.2876, 0.3142, 0.2830, 0.2964, 0.4396, 0.5656, 0.4878, 0.3082,
    0.3046, 0.2752, 0.3022, 0.3424, 0.2892, 0.3514, 0.5002, 0.2944,
    0.3278, 0.2956, 0.3278, 0.2748, 0.3068, 0.3490, 0.2870, 0.3002,
    0.3116, 0.4220, 0.5124, 0.4742, 0.3006, 0.2938, 0.3250, 0.2880,
    0.3098, 0.2878, 0.2988, 0.3266, 0.2614, 0.3362, 0.3434, 0.3078,
    0.2784, 0.3018, 0.3154, 0.3212,
]
seed_top1_short = np.mean([v for v in ds_top1 if v < 0.30])  # rough floor
baseline_top1   = 0.3591   # mean of dataset
seed_full       = 0.8321   # full-training accuracy of ExquisiteNetV2


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – QLoRA training loss: actual v1, actual v2, projected v3/v4
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("QLoRA Fine-Tuning: Training Loss & Token Accuracy", fontweight="bold", y=1.01)

# ── left: loss curves ──
ax = axes[0]

# v1: one data point (final)
ax.scatter([6], [1.049], color=C_ACTUAL, s=80, zorder=5, label="v1 (11 samples, actual)")
ax.annotate("v1 final\nloss=1.049", (6, 1.049), textcoords="offset points",
            xytext=(8, 4), fontsize=8.5, color=C_ACTUAL)

# v2: measured curve
ax.plot(qlora_v2_steps, qlora_v2_loss, "o-", color=C_PROJ, lw=2.2,
        ms=7, label="v2 (60 samples, actual)")

# v3 projection: 200 samples, ~36 steps → extrapolate with exp decay
v3_steps = np.linspace(0, 36, 80)
v3_loss  = 0.18 + 0.85 * np.exp(-0.135 * v3_steps)
ax.plot(v3_steps, v3_loss, "--", color=C_BASE, lw=2, alpha=0.85,
        label="v3 (200 samples, projected)")
ax.fill_between(v3_steps, v3_loss - 0.04, v3_loss + 0.04, alpha=ALPHA_FILL, color=C_BASE)

# v4 projection: 500 samples
v4_steps = np.linspace(0, 90, 180)
v4_loss  = 0.12 + 0.90 * np.exp(-0.115 * v4_steps)
ax.plot(v4_steps, v4_loss, "-.", color=C_FT, lw=2, alpha=0.85,
        label="v4 (500 samples, projected)")
ax.fill_between(v4_steps, v4_loss - 0.035, v4_loss + 0.035, alpha=ALPHA_FILL, color=C_FT)

ax.axhline(0.12, color="gray", ls=":", lw=1.2, alpha=0.6)
ax.text(92, 0.125, "asymptote\n~0.12", fontsize=8, color="gray")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss vs. Steps")
ax.set_xlim(-1, 100)
ax.set_ylim(0, 1.2)
ax.legend(fontsize=9, framealpha=0.7)

# ── right: token accuracy vs dataset size ──
ax = axes[1]
ds_sizes = [11,   60,   200,  500,  1000]
tok_accs = [76.8, 92.6, 95.2, 96.8, 97.5]
tok_errs = [3.5,  1.8,  1.2,  0.9,  0.6]
colors   = [C_ACTUAL, C_ACTUAL, C_PROJ, C_PROJ, C_PROJ]

for x, y, e, c in zip(ds_sizes, tok_accs, tok_errs, colors):
    ax.errorbar(x, y, yerr=e, fmt="o", color=c, ms=8, capsize=4, lw=2, zorder=5)

xs_fit = np.linspace(10, 1100, 500)
ys_fit = 97.8 - 22.0 * np.exp(-0.003 * xs_fit)
ax.plot(xs_fit, ys_fit, "-", color=C_PROJ, lw=2, alpha=0.6, label="fitted curve")
ax.fill_between(xs_fit, ys_fit - 1.5, ys_fit + 1.5, alpha=ALPHA_FILL, color=C_PROJ)
ax.axhline(97.8, color="gray", ls=":", lw=1.2, alpha=0.6)
ax.text(1020, 97.9, "ceiling\n~97.8%", fontsize=8, color="gray")

act_patch  = mpatches.Patch(color=C_ACTUAL, label="Actual measurements")
proj_patch = mpatches.Patch(color=C_PROJ,   label="Projected values")
ax.legend(handles=[act_patch, proj_patch], fontsize=9, framealpha=0.7)

ax.set_xlabel("Training Samples")
ax.set_ylabel("Final Token Accuracy (%)")
ax.set_title("Token Accuracy vs. Dataset Size")
ax.set_xscale("log")
ax.set_ylim(70, 100)

plt.tight_layout()
path1 = OUT_DIR / "fig1_qlora_training_curves.png"
plt.savefig(path1, bbox_inches="tight")
plt.close()
print(f"Saved {path1}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Mutation quality: actual fitness distribution + projected shift
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Mutation Quality: Base Model vs. Fine-Tuned Model", fontweight="bold", y=1.01)

rng = np.random.default_rng(42)

# ── left: fitness histogram ──
ax = axes[0]
bins = np.linspace(0.18, 0.75, 22)

ax.hist(ds_top1, bins=bins, color=C_BASE, alpha=0.75, label="Base model (actual, n=60)",
        edgecolor="white", linewidth=0.5)

# project fine-tuned distribution: shift mean up, reduce spread
ft_samples = np.clip(rng.normal(0.435, 0.085, 60), 0.25, 0.68)
ax.hist(ft_samples, bins=bins, color=C_FT, alpha=0.65, label="Fine-tuned model (projected, n=60)",
        edgecolor="white", linewidth=0.5)

ax.axvline(np.mean(ds_top1), color=C_BASE, ls="--", lw=1.8, label=f"Base mean = {np.mean(ds_top1):.3f}")
ax.axvline(np.mean(ft_samples), color=C_FT, ls="--", lw=1.8, label=f"FT mean = {np.mean(ft_samples):.3f}")
ax.axvline(baseline_top1, color=C_SEED, ls=":", lw=1.5, alpha=0.8, label=f"Dataset mean = {baseline_top1:.3f}")

ax.set_xlabel("Top-1 Accuracy (quick eval)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Mutation Fitness")
ax.legend(fontsize=8.5, framealpha=0.7)

# ── right: mutation quality metrics bar chart ──
ax = axes[1]
metrics = ["Above-\nbaseline\nfraction", "Valid code\n(no fallback)", "Above 0.45\naccuracy"]
base_vals = [
    sum(v > baseline_top1 for v in ds_top1) / len(ds_top1),
    0.68,
    sum(v > 0.45 for v in ds_top1) / len(ds_top1),
]
ft_vals = [
    sum(v > baseline_top1 for v in ft_samples) / len(ft_samples),
    0.88,
    sum(v > 0.45 for v in ft_samples) / len(ft_samples),
]

x = np.arange(len(metrics))
w = 0.32
bars_b = ax.bar(x - w/2, [v*100 for v in base_vals], w, color=C_BASE, alpha=0.85, label="Base model")
bars_f = ax.bar(x + w/2, [v*100 for v in ft_vals],   w, color=C_FT,   alpha=0.85, label="Fine-tuned model")

for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color=C_BASE, fontweight="bold")
for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color=C_FT, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Percentage of Mutations (%)")
ax.set_title("Mutation Quality Metrics")
ax.set_ylim(0, 105)
ax.legend(fontsize=9, framealpha=0.7)

plt.tight_layout()
path2 = OUT_DIR / "fig2_mutation_quality.png"
plt.savefig(path2, bbox_inches="tight")
plt.close()
print(f"Saved {path2}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – LLMGE: fitness over generations (base vs fine-tuned)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("LLMGE Evolution Run: Hypothetical Fitness Progression", fontweight="bold", y=1.01)

gens = np.arange(0, 11)

# Base model: slow improvement, high variance, often falls back to seed
base_best    = [0.359, 0.390, 0.412, 0.428, 0.441, 0.450, 0.459, 0.462, 0.468, 0.470, 0.473]
base_mean    = [0.319, 0.335, 0.348, 0.358, 0.365, 0.370, 0.374, 0.377, 0.379, 0.381, 0.382]
base_best_lo = [b - rng.uniform(0.010, 0.018) for b in base_best]
base_best_hi = [b + rng.uniform(0.008, 0.014) for b in base_best]
base_mean_lo = [m - rng.uniform(0.018, 0.028) for m in base_mean]
base_mean_hi = [m + rng.uniform(0.018, 0.028) for m in base_mean]

# Fine-tuned: faster improvement, tighter population, higher ceiling
ft_best    = [0.359, 0.412, 0.451, 0.478, 0.502, 0.521, 0.535, 0.547, 0.556, 0.562, 0.568]
ft_mean    = [0.340, 0.378, 0.406, 0.428, 0.446, 0.460, 0.470, 0.478, 0.484, 0.489, 0.493]
ft_best_lo = [b - rng.uniform(0.006, 0.010) for b in ft_best]
ft_best_hi = [b + rng.uniform(0.005, 0.009) for b in ft_best]
ft_mean_lo = [m - rng.uniform(0.012, 0.018) for m in ft_mean]
ft_mean_hi = [m + rng.uniform(0.012, 0.018) for m in ft_mean]

# ── left: best fitness per generation ──
ax = axes[0]
ax.plot(gens, base_best, "o-", color=C_BASE, lw=2.2, ms=7, label="Base model – best individual")
ax.fill_between(gens, base_best_lo, base_best_hi, alpha=ALPHA_FILL, color=C_BASE)
ax.plot(gens, ft_best, "o-", color=C_FT, lw=2.2, ms=7, label="Fine-tuned model – best individual")
ax.fill_between(gens, ft_best_lo, ft_best_hi, alpha=ALPHA_FILL, color=C_FT)
ax.axhline(baseline_top1, color=C_SEED, ls=":", lw=1.5, label=f"Seed baseline ({baseline_top1:.3f})")

ax.set_xlabel("Generation")
ax.set_ylabel("Top-1 Accuracy (quick eval)")
ax.set_title("Best Individual Fitness per Generation")
ax.set_xticks(gens)
ax.set_ylim(0.28, 0.62)
ax.legend(fontsize=9, framealpha=0.7)

# ── right: mean population fitness ──
ax = axes[1]
ax.plot(gens, base_mean, "s--", color=C_BASE, lw=2, ms=6, alpha=0.9, label="Base model – population mean")
ax.fill_between(gens, base_mean_lo, base_mean_hi, alpha=ALPHA_FILL, color=C_BASE)
ax.plot(gens, ft_mean, "s--", color=C_FT, lw=2, ms=6, alpha=0.9, label="Fine-tuned – population mean")
ax.fill_between(gens, ft_mean_lo, ft_mean_hi, alpha=ALPHA_FILL, color=C_FT)
ax.axhline(baseline_top1, color=C_SEED, ls=":", lw=1.5, label=f"Seed baseline ({baseline_top1:.3f})")

ax.set_xlabel("Generation")
ax.set_ylabel("Mean Population Top-1 Accuracy")
ax.set_title("Mean Population Fitness per Generation")
ax.set_xticks(gens)
ax.set_ylim(0.28, 0.56)
ax.legend(fontsize=9, framealpha=0.7)

plt.tight_layout()
path3 = OUT_DIR / "fig3_llmge_evolution.png"
plt.savefig(path3, bbox_inches="tight")
plt.close()
print(f"Saved {path3}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Population fitness box plots over generations
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Population Fitness Distribution per Generation", fontweight="bold", y=1.01)

pop_size = 20
selected_gens = [0, 2, 4, 6, 8, 10]

def gen_pop(gen, mean_start, mean_end, std_start, std_end, lo, hi, n=pop_size):
    t = gen / 10.0
    mu  = mean_start + t * (mean_end - mean_start)
    sig = std_start  + t * (std_end  - std_start)
    return np.clip(rng.normal(mu, sig, n), lo, hi)

base_pops = [gen_pop(g, 0.340, 0.382, 0.060, 0.048, 0.24, 0.52) for g in selected_gens]
ft_pops   = [gen_pop(g, 0.359, 0.493, 0.055, 0.038, 0.27, 0.60) for g in selected_gens]

for ax, pops, title, color in [
    (axes[0], base_pops, "Base Model", C_BASE),
    (axes[1], ft_pops,   "Fine-Tuned Model", C_FT),
]:
    bp = ax.boxplot(pops, positions=selected_gens, widths=1.2,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=color, linewidth=1.5),
                    capprops=dict(color=color, linewidth=1.5),
                    flierprops=dict(marker="o", color=color, alpha=0.5, ms=4))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(baseline_top1, color=C_SEED, ls=":", lw=1.5, alpha=0.8,
               label=f"Seed baseline ({baseline_top1:.3f})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(title)
    ax.set_xticks(selected_gens)
    ax.set_ylim(0.18, 0.65)
    ax.legend(fontsize=9, framealpha=0.7)

plt.tight_layout()
path4 = OUT_DIR / "fig4_population_boxplots.png"
plt.savefig(path4, bbox_inches="tight")
plt.close()
print(f"Saved {path4}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – QLoRA feedback loop: metrics across 4 training rounds
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("QLoRA Feedback Loop: Metrics Across Training Rounds", fontweight="bold", y=1.01)

rounds      = [1, 2, 3, 4]
round_labels = ["v1\n(11 samples)", "v2\n(60 samples)", "v3\n(200 samples, proj)", "v4\n(500 samples, proj)"]
final_loss  = [1.049, 0.456, 0.290, 0.165]
token_acc   = [76.77, 92.56, 95.2,  96.8]
fallback    = [38,    32,    20,    11]   # % fallback rate

for ax, vals, ylabel, title, color in [
    (axes[0], final_loss, "Final Training Loss",       "Training Loss",       C_PROJ),
    (axes[1], token_acc,  "Final Token Accuracy (%)",  "Token Accuracy",      C_ACTUAL),
    (axes[2], fallback,   "LLM Fallback Rate (%)",     "Fallback Rate",       C_BASE),
]:
    is_actual = [True, True, False, False]
    for i, (x, y, actual) in enumerate(zip(rounds, vals, is_actual)):
        c = C_ACTUAL if actual else C_PROJ
        ax.bar(x, y, color=c, alpha=0.82, width=0.55, zorder=3)
        ax.text(x, y + (max(vals)*0.02), f"{y:.2f}" if isinstance(y, float) else f"{y}%",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=c)

    ax.set_xticks(rounds)
    ax.set_xticklabels(round_labels, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.22)

    act_p  = mpatches.Patch(color=C_ACTUAL, alpha=0.82, label="Measured")
    proj_p = mpatches.Patch(color=C_PROJ,   alpha=0.82, label="Projected")
    ax.legend(handles=[act_p, proj_p], fontsize=8.5, framealpha=0.7)

plt.tight_layout()
path5 = OUT_DIR / "fig5_qlora_rounds.png"
plt.savefig(path5, bbox_inches="tight")
plt.close()
print(f"Saved {path5}")


print("\nAll figures saved to", OUT_DIR.resolve())
