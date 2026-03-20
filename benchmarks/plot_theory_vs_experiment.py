#!/usr/bin/env python3
"""
Theory vs Experiment: Circuit-Level Validation
===============================================
Scatter plot of GF(2) Rank/T-gates (theoretical prediction) vs
Empirical OFD Success Rate from cross-validation experiment.

Uses:
  - results/experiment_cross_validation.csv (936 circuits, all 14 families)
    Contains both GF(2) rank and actual OFD success/fail in one dataset.

Output:
  gf2_vs_ofd_validation.png
  gf2_vs_ofd_validation.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
results_dir = repo / "results"

cv_csv = results_dir / "experiment_cross_validation.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(cv_csv)
df["success"] = df["success"].astype(str).str.lower() == "true"
df = df[df["success"]].copy()

print(f"Cross-validation circuits: {len(df)}")
print(f"Families: {df['family'].nunique()}")

# Compute derived columns
df["ofd_total"] = df["actual_ofd_success"] + df["actual_ofd_fail"]
df["ofd_rate"] = df["actual_ofd_success"] / df["ofd_total"]
df["rank_to_t_ratio"] = df["gf2_rank"] / df["n_t_gates"]

merged = df.copy()
print(f"Total circuits:     {len(merged)}")
print()

# ── Family categorization ─────────────────────────────────────────────────────
# Three natural categories matching the paper's Section 4.1 structure:
#   Random   – control circuits matching Liu & Clark's theoretical regime
#   Algorithm – quantum algorithms (oracle-based, variational, error-correcting)
#   State    – entangled state preparation circuits
category_map = {
    # Random circuits
    "Random Clifford+T (Brick-wall)":  "Random",
    "Random Clifford+T (All-to-all)":  "Random",
    # Quantum algorithms
    "Bernstein-Vazirani":              "Algorithm",
    "Simon's Algorithm":               "Algorithm",
    "Deutsch-Jozsa":                   "Algorithm",
    "QAOA MaxCut (p=1, 3-regular)":    "Algorithm",
    "Surface Code":                    "Algorithm",
    "Quantum Fourier Transform":       "Algorithm",
    "Grover Search":                   "Algorithm",
    "VQE Hardware-Efficient Ansatz":   "Algorithm",
    # State-preparation circuits
    "GHZ State":                       "State",
    "Bell State / EPR Pairs":          "State",
    "Graph State":                     "State",
    "Cluster State (1D)":              "State",
}

merged["category"] = merged["family"].map(category_map)

# Drop any rows that didn't map (shouldn't happen)
unmapped = merged[merged["category"].isna()]
if len(unmapped) > 0:
    print(f"WARNING: {len(unmapped)} rows with unmapped family names:")
    print(unmapped["family"].unique())
merged = merged.dropna(subset=["category"])

# Drop circuits with 0 T-gates (rank/T is undefined; these are pure-Clifford)
n_before = len(merged)
merged = merged.dropna(subset=["rank_to_t_ratio"])
merged = merged[merged["n_t_gates"] > 0].copy()
n_dropped = n_before - len(merged)
if n_dropped > 0:
    print(f"Dropped {n_dropped} circuits with 0 T-gates (rank/T undefined)")

# Print per-category stats
print("Per-category breakdown:")
for cat in ["Random", "Algorithm", "State"]:
    sub = merged[merged["category"] == cat]
    print(f"  {cat:14s}: {len(sub):4d} circuits, "
          f"mean OFD rate = {sub['ofd_rate'].mean():.3f}, "
          f"mean rank/T = {sub['rank_to_t_ratio'].mean():.3f}")
print()

# ── Correlation ───────────────────────────────────────────────────────────────
x = merged["rank_to_t_ratio"].values
y = merged["ofd_rate"].values
r = np.corrcoef(x, y)[0, 1]
n_plot = len(merged)
print(f"Pearson r = {r:.4f}, n = {n_plot}")

# ── Publication-quality plot ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

fig, ax = plt.subplots(figsize=(7, 7))

# Category colors and ordering (draw back-to-front so Random is on top)
cat_style = {
    "Random":    {"color": "#4C72B0", "zorder": 4, "alpha": 0.75},
    "Algorithm": {"color": "#55A868", "zorder": 3, "alpha": 0.70},
    "State":     {"color": "#DD8452", "zorder": 2, "alpha": 0.70},
}

for cat in ["State", "Algorithm", "Random"]:
    sub = merged[merged["category"] == cat]
    style = cat_style[cat]
    ax.scatter(
        sub["rank_to_t_ratio"], sub["ofd_rate"],
        label=cat,
        s=40,
        color=style["color"],
        alpha=style["alpha"],
        edgecolors="white",
        linewidths=0.3,
        zorder=style["zorder"],
    )

# y = x reference line
ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, alpha=0.7,
        label=r"$y = x$", zorder=1)

# Annotation box with correlation
textstr = f"$r = {r:.2f}$\n$n = {n_plot}$"
props = dict(boxstyle="round,pad=0.4", facecolor="white",
             edgecolor="black", alpha=0.9, linewidth=0.8)
ax.text(0.04, 0.96, textstr, transform=ax.transAxes,
        fontsize=13, verticalalignment="top", bbox=props)

# Axis formatting
ax.set_xlabel(r"GF(2) Rank / T-gates (Theoretical Prediction)")
ax.set_ylabel(r"Empirical OFD Success Rate")
ax.set_title("Theory vs Experiment: Circuit-Level Validation")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_aspect("equal")

# Legend
legend = ax.legend(loc="lower right", frameon=True, framealpha=0.9,
                   edgecolor="black", fancybox=False,
                   handletextpad=0.5, borderpad=0.6)
legend.get_frame().set_linewidth(0.8)

ax.grid(True, alpha=0.15, linewidth=0.5)

# ── Save ──────────────────────────────────────────────────────────────────────
out_png = repo / "gf2_vs_ofd_validation.png"
out_pdf = repo / "gf2_vs_ofd_validation.pdf"

fig.savefig(out_png)
fig.savefig(out_pdf)
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")

plt.show()
