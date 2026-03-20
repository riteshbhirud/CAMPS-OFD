#!/usr/bin/env python3
"""
Cross-Validation: GF(2) vs Benchmark OFD Counts — Publication Figures
======================================================================
Scatter plot of GF(2)-predicted OFD success count vs actual OFD success
count from the benchmark data.

Uses: results/experiment_cross_validation.csv

Output:
  experiment_cross_validation.png / .pdf  (scatter + correlation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "experiment_cross_validation.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df[df["success"] == True].copy()

print(f"Loaded {len(df)} cross-validation results")
print(f"Families: {df['family'].nunique()}")
print()

# ── Short names ────────────────────────────────────────────────────────────────
short_names = {
    "Random Clifford+T (All-to-all)": "Random (A2A)",
    "Random Clifford+T (Brick-wall)": "Random (Brick)",
    "Bernstein-Vazirani": "Bernstein-Vaz.",
    "Graph State": "Graph State",
    "Cluster State (1D)": "Cluster State",
    "Bell State / EPR Pairs": "Bell State",
    "Simon's Algorithm": "Simon's",
    "Surface Code": "Surface Code",
    "QAOA MaxCut (p=1, 3-regular)": "QAOA",
    "Deutsch-Jozsa": "Deutsch-Jozsa",
    "GHZ State": "GHZ",
    "Grover Search": "Grover",
    "VQE Hardware-Efficient Ansatz": "VQE",
    "Quantum Fourier Transform": "QFT",
}
df["short_name"] = df["family"].map(short_names).fillna(df["family"])

# ── Color scheme ──────────────────────────────────────────────────────────────
class_colors = {
    "Random (A2A)": "#1f77b4", "Random (Brick)": "#4a90d9",
    "QFT": "#d62728", "VQE": "#e45756", "Grover": "#ff7f0e", "GHZ": "#c44e52",
    "Bernstein-Vaz.": "#2ca02c", "Cluster State": "#98df8a",
    "Graph State": "#8c564b", "Bell State": "#9467bd",
    "Simon's": "#7f7f7f", "Deutsch-Jozsa": "#bcbd22",
    "QAOA": "#17becf", "Surface Code": "#aec7e8",
}

# ── Category classification ──────────────────────────────────────────────────
category_map = {
    "Random (A2A)": "Random", "Random (Brick)": "Random",
    "Bernstein-Vaz.": "Algorithm", "Simon's": "Algorithm",
    "Deutsch-Jozsa": "Algorithm", "QAOA": "Algorithm",
    "Surface Code": "Algorithm", "QFT": "Algorithm",
    "Grover": "Algorithm", "VQE": "Algorithm",
    "GHZ": "State", "Bell State": "State",
    "Graph State": "State", "Cluster State": "State",
}

# ── Publication-quality settings ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter — GF(2)-predicted vs Actual OFD success count
# ═══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(10, 8))

for sname in sorted(df["short_name"].unique()):
    fam_data = df[df["short_name"] == sname]
    color = class_colors.get(sname, "#333333")
    cat = category_map.get(sname, "Algorithm")

    marker = "o" if cat == "Random" else ("s" if cat == "State" else "^")

    ax1.scatter(fam_data["actual_ofd_success"], fam_data["gf2_ofd_success"],
                color=color, s=40, marker=marker, edgecolor="white",
                linewidth=0.3, alpha=0.7, label=sname, zorder=3)

# y=x reference line
lim_max = max(df["actual_ofd_success"].max(), df["gf2_ofd_success"].max()) * 1.1
ax1.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.0, alpha=0.3,
         label="Perfect agreement", zorder=1)

# Compute overall R²
actual = df["actual_ofd_success"].values.astype(float)
predicted = df["gf2_ofd_success"].values.astype(float)
if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0:
    corr = np.corrcoef(actual, predicted)[0, 1]
    r_sq = corr ** 2
    n_match = (df["exact_match"] == True).sum()
    n_total = len(df)
    ax1.text(0.05, 0.95,
             f"$R^2$ = {r_sq:.4f}\nExact match: {n_match}/{n_total} ({100*n_match/n_total:.1f}%)\nMAE = {np.mean(np.abs(actual - predicted)):.2f}",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"))

ax1.set_xlabel("Actual OFD Success Count (Benchmark)")
ax1.set_ylabel("GF(2)-Predicted OFD Success Count")
ax1.set_title("Cross-Validation: GF(2) Predictions vs Benchmark Results")
ax1.set_xlim(-0.5, lim_max)
ax1.set_ylim(-0.5, lim_max)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.15, linewidth=0.5)
ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1)

fig1.tight_layout()
out1_png = repo / "experiment_cross_validation.png"
out1_pdf = repo / "experiment_cross_validation.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-family match rate and MAE
# ═══════════════════════════════════════════════════════════════════════════════

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 6))

fam_stats = df.groupby("short_name").agg(
    match_rate=("exact_match", "mean"),
    mean_abs_error=("abs_error", "mean"),
    count=("exact_match", "count"),
).reset_index()

fam_stats["category"] = fam_stats["short_name"].map(category_map)
cat_color_map = {"Random": "#4C72B0", "Algorithm": "#55A868", "State": "#DD8452"}
fam_stats["color"] = fam_stats["category"].map(cat_color_map).fillna("#333333")

# Left: match rate
fam_stats_sorted = fam_stats.sort_values("match_rate", ascending=False).reset_index(drop=True)
x2a = np.arange(len(fam_stats_sorted))
ax2a.bar(x2a, fam_stats_sorted["match_rate"] * 100,
         color=fam_stats_sorted["color"], edgecolor="white", linewidth=0.5,
         width=0.72, zorder=3)

for i, (_, row) in enumerate(fam_stats_sorted.iterrows()):
    ax2a.text(i, row["match_rate"] * 100 + 1, f"{row['match_rate']*100:.0f}%",
              ha="center", va="bottom", fontsize=9)

ax2a.set_xticks(x2a)
ax2a.set_xticklabels(fam_stats_sorted["short_name"], rotation=35, ha="right")
ax2a.set_ylabel("Exact Match Rate (%)")
ax2a.set_title("GF(2) Exact Match Rate by Family")
ax2a.set_ylim(0, 115)
ax2a.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax2a.set_axisbelow(True)

# Right: MAE
fam_stats_mae = fam_stats.sort_values("mean_abs_error", ascending=True).reset_index(drop=True)
x2b = np.arange(len(fam_stats_mae))
ax2b.bar(x2b, fam_stats_mae["mean_abs_error"],
         color=fam_stats_mae["color"], edgecolor="white", linewidth=0.5,
         width=0.72, zorder=3)

for i, (_, row) in enumerate(fam_stats_mae.iterrows()):
    ax2b.text(i, row["mean_abs_error"] + 0.05, f"{row['mean_abs_error']:.2f}",
              ha="center", va="bottom", fontsize=9)

ax2b.set_xticks(x2b)
ax2b.set_xticklabels(fam_stats_mae["short_name"], rotation=35, ha="right")
ax2b.set_ylabel("Mean Absolute Error")
ax2b.set_title("GF(2) Prediction Error by Family")
ax2b.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax2b.set_axisbelow(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", label="Random"),
    Patch(facecolor="#55A868", label="Algorithm"),
    Patch(facecolor="#DD8452", label="State"),
]
ax2b.legend(handles=legend_elements, loc="upper right", frameon=True,
            framealpha=0.9, edgecolor="black", fancybox=False)

fig2.tight_layout()
out2_png = repo / "experiment_cross_validation_accuracy.png"
out2_pdf = repo / "experiment_cross_validation_accuracy.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("CROSS-VALIDATION SUMMARY")
print("="*90)
n_total = len(df)
n_match = (df["exact_match"] == True).sum()
mae = df["abs_error"].mean()
print(f"Total: {n_total} circuits")
print(f"Exact match: {n_match} ({100*n_match/n_total:.1f}%)")
print(f"MAE: {mae:.2f}")
if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0:
    print(f"R²: {r_sq:.4f}")
print("="*90)

plt.show()
