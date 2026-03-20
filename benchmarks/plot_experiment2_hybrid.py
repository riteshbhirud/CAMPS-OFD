#!/usr/bin/env python3
"""
Experiment 2: GF(2)-Screened OFD — Publication Figures
=======================================================
Validates whether GF(2) pre-analysis correctly identifies OFD-amenable T-gates.
Compares default (OFD on every T-gate) vs GF(2)-guided (OFD only on predicted-
amenable T-gates, rest naively absorbed).

Uses: results/experiment2_hybrid_scheduling.csv

Output:
  experiment2_chi_comparison.png / .pdf      (Default vs Guided chi)
  experiment2_amenability_vs_chi.png / .pdf   (Amenability scatter)
  experiment2_screening_accuracy.png / .pdf   (GF(2) accuracy + skip rate)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "experiment2_hybrid_scheduling.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df[df["success"] == True].copy()

print(f"Loaded {len(df)} successful experiments")
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
df["short_name"] = df["family"].map(short_names)

# ── Category colors ────────────────────────────────────────────────────────────
category_map = {
    "Random (A2A)": "Random", "Random (Brick)": "Random",
    "Bernstein-Vaz.": "Algorithm", "Simon's": "Algorithm",
    "Deutsch-Jozsa": "Algorithm", "QAOA": "Algorithm",
    "Surface Code": "Algorithm", "QFT": "Algorithm",
    "Grover": "Algorithm", "VQE": "Algorithm",
    "GHZ": "State", "Bell State": "State",
    "Graph State": "State", "Cluster State": "State",
}
cat_colors = {
    "Random": "#4C72B0",
    "Algorithm": "#55A868",
    "State": "#DD8452",
}

# ── Publication-quality settings ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Bond dimension — Default vs GF(2)-Guided
# ═══════════════════════════════════════════════════════════════════════════════

stats = df.groupby("short_name").agg(
    mean_chi_default=("default_chi", "mean"),
    mean_chi_guided=("guided_chi", "mean"),
    mean_amenability=("amenability_rate", "mean"),
    mean_predicted_chi=("predicted_chi", "mean"),
    chi_match_rate=("chi_match", "mean"),
    count=("default_chi", "count"),
).reset_index()

stats["category"] = stats["short_name"].map(category_map)
stats = stats.sort_values("mean_amenability", ascending=False).reset_index(drop=True)

fig1, ax1 = plt.subplots(figsize=(14, 6))

x = np.arange(len(stats))
width = 0.35

bars_def = ax1.bar(x - width/2, stats["mean_chi_default"], width,
                    label="Default (OFD on all T-gates)", color="#4C72B0",
                    edgecolor="white", linewidth=0.5, zorder=3)
bars_guided = ax1.bar(x + width/2, stats["mean_chi_guided"], width,
                       label=r"GF(2)-Guided (OFD on amenable only)", color="#DD8452",
                       edgecolor="white", linewidth=0.5, zorder=3)

# Value labels
for i, (_, row) in enumerate(stats.iterrows()):
    if row["mean_chi_default"] > 0:
        ax1.text(i - width/2, row["mean_chi_default"] * 1.05, f"{row['mean_chi_default']:.0f}",
                 ha="center", va="bottom", fontsize=8)
    if row["mean_chi_guided"] > 0:
        ax1.text(i + width/2, row["mean_chi_guided"] * 1.05, f"{row['mean_chi_guided']:.0f}",
                 ha="center", va="bottom", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(stats["short_name"], rotation=35, ha="right")
ax1.set_ylabel(r"Mean Bond Dimension $\chi$")
ax1.set_title(r"Bond Dimension: Default vs GF(2)-Guided OFD Screening")
ax1.set_yscale("symlog", linthresh=1)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="black", fancybox=False)
ax1.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)

fig1.tight_layout()
out1_png = repo / "experiment2_chi_comparison.png"
out1_pdf = repo / "experiment2_chi_comparison.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: GF(2) Amenability vs χ ratio (scatter)
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(10, 7))

df["chi_ratio"] = df["guided_chi"] / df["default_chi"].clip(lower=1)

for sname in df["short_name"].unique():
    fam_data = df[df["short_name"] == sname]
    cat = category_map.get(sname, "Algorithm")
    color = cat_colors.get(cat, "#333333")
    ax2.scatter(fam_data["amenability_rate"] * 100, fam_data["chi_ratio"],
                color=color, s=50, alpha=0.7, edgecolor="white", linewidth=0.3,
                label=sname, zorder=3)

# Reference line: ratio = 1 (perfect match)
ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
ax2.text(95, 1.05, "Perfect match", ha="right", fontsize=9, color="gray", style="italic")

ax2.set_xlabel("GF(2) OFD Amenability (%)")
ax2.set_ylabel(r"$\chi_{\mathrm{Guided}} / \chi_{\mathrm{Default}}$")
ax2.set_title("Amenability Rate vs Bond Dimension Ratio")
ax2.set_yscale("symlog", linthresh=1)
ax2.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1, fontsize=9)

fig2.tight_layout()
out2_png = repo / "experiment2_amenability_vs_chi.png"
out2_pdf = repo / "experiment2_amenability_vs_chi.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: GF(2) Screening Accuracy and OFD Skip Rate
# ═══════════════════════════════════════════════════════════════════════════════

cat_color_map = {"Random": "#4C72B0", "Algorithm": "#55A868", "State": "#DD8452"}

fig3, ax3 = plt.subplots(figsize=(10, 6))

# OFD skip rate (fraction of T-gates where OFD was skipped)
skip_stats = df.groupby("short_name").agg(
    mean_skipped=("n_ofd_skipped", "mean"),
    mean_t=("n_t_gates", "mean"),
    mean_amenability=("amenability_rate", "mean"),
).reset_index()
skip_stats["skip_rate"] = skip_stats["mean_skipped"] / skip_stats["mean_t"].clip(lower=1)
skip_stats["category"] = skip_stats["short_name"].map(category_map)
skip_stats["color"] = skip_stats["category"].map(cat_color_map)
skip_stats = skip_stats.sort_values("skip_rate", ascending=False).reset_index(drop=True)

x3 = np.arange(len(skip_stats))
ax3.bar(x3, skip_stats["skip_rate"] * 100,
        color=skip_stats["color"], edgecolor="white",
        linewidth=0.5, width=0.72, zorder=3)

for i, (_, row) in enumerate(skip_stats.iterrows()):
    ax3.text(i, row["skip_rate"] * 100 + 1, f"{row['skip_rate']*100:.0f}%",
             ha="center", va="bottom", fontsize=9)

ax3.set_xticks(x3)
ax3.set_xticklabels(skip_stats["short_name"], rotation=35, ha="right")
ax3.set_ylabel("OFD Attempts Skipped (%)")
ax3.set_title("Computation Saved by GF(2) Pre-Screening")
ax3.set_ylim(0, 110)
ax3.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax3.set_axisbelow(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", label="Random"),
    Patch(facecolor="#55A868", label="Algorithm"),
    Patch(facecolor="#DD8452", label="State"),
]
ax3.legend(handles=legend_elements, loc="upper right", frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False)

fig3.tight_layout()
out3_png = repo / "experiment2_screening_accuracy.png"
out3_pdf = repo / "experiment2_screening_accuracy.pdf"
fig3.savefig(out3_png)
fig3.savefig(out3_pdf)
print(f"Saved: {out3_png}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 2 SUMMARY: GF(2)-SCREENED OFD VALIDATION")
print("="*100)
print(f"{'Family':25s} {'Amenab%':>8s} {'chi_Def':>8s} {'chi_Guid':>9s} {'Match%':>8s} {'Skip%':>8s}")
print("-"*100)
for _, row in stats.iterrows():
    skip_row = skip_stats[skip_stats["short_name"] == row["short_name"]]
    skip_pct = skip_row["skip_rate"].values[0] * 100 if len(skip_row) > 0 else 0
    print(f"{row['short_name']:25s} {row['mean_amenability']*100:7.1f}% "
          f"{row['mean_chi_default']:8.1f} {row['mean_chi_guided']:9.1f} "
          f"{row['chi_match_rate']*100:7.1f}% {skip_pct:7.1f}%")

n_total = len(df)
n_match = df["chi_match"].sum()
print("="*100)
print(f"Overall chi match: {n_match}/{n_total} ({100*n_match/n_total:.1f}%)")
print("="*100)

plt.show()
