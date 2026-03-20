#!/usr/bin/env python3
"""
Random Circuit Rank Verification — Publication Figures
=======================================================
Part A: Absolute nullity convergence — ν ≈ O(1) independent of N,
        meaning ν/t → 0 as N grows (random circuits are OFD-friendly).
Part B: Rank saturation with increasing T-gate count.

Uses: results/experiment_random_rank.csv

Output:
  experiment_random_nullity.png / .pdf       (absolute nullity + ν/t box plots)
  experiment_random_saturation.png / .pdf    (rank vs t scatter)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "experiment_random_rank.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df[df["success"] == True].copy()

print(f"Loaded {len(df)} successful experiments")
print()

# ── Publication-quality settings ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Absolute Nullity Convergence + ν/t → 0
# ═══════════════════════════════════════════════════════════════════════════════

part_a = df[df["part"] == "nullity_convergence"].copy()

if len(part_a) > 0:
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))

    n_vals = sorted(part_a["n_qubits"].unique())

    # Left panel: Absolute nullity ν (should be O(1), independent of N)
    data_nullity = [part_a[part_a["n_qubits"] == n]["nullity"].dropna().values
                    for n in n_vals]

    bp1 = ax1a.boxplot(data_nullity, positions=range(len(n_vals)), widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                       medianprops=dict(color="black", linewidth=1.5),
                       boxprops=dict(facecolor="#4C72B0", alpha=0.6))

    ax1a.set_xticks(range(len(n_vals)))
    ax1a.set_xticklabels([str(n) for n in n_vals])
    ax1a.set_xlabel("Qubit Count $N$")
    ax1a.set_ylabel(r"Absolute Nullity $\nu$")
    ax1a.set_title(r"Absolute Nullity $\nu$ vs $N$ ($t = N$)")

    ax1a.yaxis.grid(True, alpha=0.15, linewidth=0.5)
    ax1a.set_axisbelow(True)

    for i, n in enumerate(n_vals):
        vals = part_a[part_a["n_qubits"] == n]["nullity"].dropna()
        if len(vals) > 0:
            ax1a.text(i, vals.mean() + 0.15, f"{vals.mean():.2f}",
                      ha="center", va="bottom", fontsize=9, color="red")

    # Right panel: Nullity ratio ν/t (should → 0 as N grows)
    data_ratio = [part_a[part_a["n_qubits"] == n]["nullity_ratio"].dropna().values
                  for n in n_vals]

    bp2 = ax1b.boxplot(data_ratio, positions=range(len(n_vals)), widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                       medianprops=dict(color="black", linewidth=1.5),
                       boxprops=dict(facecolor="#55A868", alpha=0.6))

    ax1b.set_xticks(range(len(n_vals)))
    ax1b.set_xticklabels([str(n) for n in n_vals])
    ax1b.set_xlabel("Qubit Count $N$")
    ax1b.set_ylabel(r"Nullity Ratio $\nu / t$")
    ax1b.set_title(r"Nullity Ratio $\nu/t \to 0$ as $N$ grows ($t = N$)")

    ax1b.yaxis.grid(True, alpha=0.15, linewidth=0.5)
    ax1b.set_axisbelow(True)

    for i, n in enumerate(n_vals):
        vals = part_a[part_a["n_qubits"] == n]["nullity_ratio"].dropna()
        if len(vals) > 0:
            ax1b.text(i, vals.mean() + 0.005, f"{vals.mean():.3f}",
                      ha="center", va="bottom", fontsize=9, color="red")

    fig1.tight_layout()
    out1_png = repo / "experiment_random_nullity.png"
    out1_pdf = repo / "experiment_random_nullity.pdf"
    fig1.savefig(out1_png)
    fig1.savefig(out1_pdf)
    print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Rank Saturation (Rank vs T-count)
# ═══════════════════════════════════════════════════════════════════════════════

part_b = df[df["part"] == "rank_saturation"].copy()

if len(part_b) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    n_colors = {
        8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c",
        48: "#d62728", 64: "#9467bd",
    }

    n_vals = sorted(part_b["n_qubits"].unique())

    for n in n_vals:
        nd = part_b[part_b["n_qubits"] == n]
        color = n_colors.get(n, "#333333")

        # Scatter individual points
        ax2.scatter(nd["t"], nd["rank"], color=color, s=20, alpha=0.4, zorder=2)

        # Mean line per density
        avg = nd.groupby("t_density").agg(
            mean_t=("t", "mean"),
            mean_rank=("rank", "mean"),
        ).reset_index().sort_values("mean_t")

        ax2.plot(avg["mean_t"], avg["mean_rank"], "o-", color=color,
                 linewidth=2, markersize=6, label=f"N={n}", zorder=3)

        # Horizontal line at rank = N (saturation level)
        ax2.axhline(y=n, color=color, linestyle=":", linewidth=1.0, alpha=0.4)
        ax2.text(avg["mean_t"].max() * 1.05, n, f"N={n}", fontsize=9,
                 color=color, va="center")

    ax2.set_xlabel("Number of T-Gates $t$")
    ax2.set_ylabel("GF(2) Rank")
    ax2.set_title("Random Clifford+T: Rank Saturation with T-Gate Count")
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9,
               edgecolor="black", fancybox=False)

    fig2.tight_layout()
    out2_png = repo / "experiment_random_saturation.png"
    out2_pdf = repo / "experiment_random_saturation.pdf"
    fig2.savefig(out2_png)
    fig2.savefig(out2_pdf)
    print(f"Saved: {out2_png}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RANDOM RANK VERIFICATION SUMMARY")
print("="*70)
if len(part_a) > 0:
    for n in sorted(part_a["n_qubits"].unique()):
        nulls = part_a[part_a["n_qubits"] == n]["nullity"].dropna()
        ratios = part_a[part_a["n_qubits"] == n]["nullity_ratio"].dropna()
        print(f"  N={n:3d}: mean(ν) = {nulls.mean():.3f} ± {nulls.std():.3f}, mean(ν/t) = {ratios.mean():.4f}")
if len(part_b) > 0:
    for n in sorted(part_b["n_qubits"].unique()):
        nd = part_b[part_b["n_qubits"] == n]
        max_rank = nd["rank"].max()
        print(f"  N={n:3d}: max rank = {max_rank} (saturation at {100*max_rank/n:.0f}% of N)")
print("="*70)

plt.show()
