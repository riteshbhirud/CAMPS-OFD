#!/usr/bin/env python3
"""
Experiment 3: T-Gate Consolidation — Publication Figures
=========================================================
Generates figures showing the effect of T-gate consolidation preprocessing
on GF(2) metrics, T-count reduction, and predicted bond dimension.

Uses: results/experiment3_t_consolidation.csv

Output:
  experiment3_t_reduction.png / .pdf        (T-count reduction by family)
  experiment3_nullity_reduction.png / .pdf   (Nullity improvement)
  experiment3_chi_improvement.png / .pdf     (Bond dimension improvement)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "experiment3_t_consolidation.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df[df["success"] == True].copy()

# Filter out rows with no T gates
df = df[df["orig_t_count"] > 0].copy()

print(f"Loaded {len(df)} successful experiments with T gates")
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

# ── Per-family statistics ──────────────────────────────────────────────────────
stats = df.groupby("short_name").agg(
    mean_orig_t=("orig_t_count", "mean"),
    mean_consol_t=("consol_t_count", "mean"),
    mean_t_reduction_pct=("t_reduction_pct", "mean"),
    std_t_reduction_pct=("t_reduction_pct", "std"),
    mean_nullity_reduction=("nullity_reduction", "mean"),
    mean_orig_nullity=("orig_nullity", "mean"),
    mean_consol_nullity=("consol_nullity", "mean"),
    mean_chi_improvement=("chi_improvement", "mean"),
    mean_orig_ofd=("orig_ofd_rate", "mean"),
    mean_consol_ofd=("consol_ofd_rate", "mean"),
    mean_n_tt=("n_tt_to_s", "mean"),
    mean_n_cancel=("n_t_tdag_cancel", "mean"),
    count=("orig_t_count", "count"),
).reset_index()

stats["category"] = stats["short_name"].map(category_map)
stats["color"] = stats["category"].map(cat_colors)

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
# FIGURE 1: T-Count Reduction (%) by Family — bar chart
# ═══════════════════════════════════════════════════════════════════════════════

stats_sorted = stats.sort_values("mean_t_reduction_pct", ascending=False).reset_index(drop=True)

fig1, ax1 = plt.subplots(figsize=(14, 6))

x = np.arange(len(stats_sorted))
bars = ax1.bar(
    x,
    stats_sorted["mean_t_reduction_pct"] * 100,
    yerr=stats_sorted["std_t_reduction_pct"].fillna(0) * 100,
    capsize=4,
    color=stats_sorted["color"],
    edgecolor="white",
    linewidth=0.5,
    width=0.72,
    error_kw={"linewidth": 1.2, "capthick": 1.2},
    zorder=3,
)

# Value labels
for i, (_, row) in enumerate(stats_sorted.iterrows()):
    pct = row["mean_t_reduction_pct"] * 100
    y_pos = pct + (row["std_t_reduction_pct"] or 0) * 100 + 2
    ax1.text(i, y_pos, f"{pct:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="medium")

ax1.set_xticks(x)
ax1.set_xticklabels(stats_sorted["short_name"], rotation=35, ha="right")
ax1.set_ylabel("T-Count Reduction (%)")
ax1.set_title("T-Gate Consolidation: T-Count Reduction by Circuit Family")
ax1.set_ylim(0, 105)
ax1.set_xlim(-0.6, len(stats_sorted) - 0.4)
ax1.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cat_colors["Random"], label="Random"),
    Patch(facecolor=cat_colors["Algorithm"], label="Algorithm"),
    Patch(facecolor=cat_colors["State"], label="State"),
]
ax1.legend(handles=legend_elements, loc="upper right", frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, fontsize=11)

fig1.tight_layout()
out1_png = repo / "experiment3_t_reduction.png"
out1_pdf = repo / "experiment3_t_reduction.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Before/After Nullity — stacked comparison
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(14, 6))

stats_null = stats.sort_values("mean_orig_nullity", ascending=False).reset_index(drop=True)

x2 = np.arange(len(stats_null))
width = 0.35

bars_orig = ax2.bar(x2 - width/2, stats_null["mean_orig_nullity"], width,
                     label="Before consolidation", color="#d62728", edgecolor="white",
                     linewidth=0.5, zorder=3)
bars_cons = ax2.bar(x2 + width/2, stats_null["mean_consol_nullity"], width,
                     label="After consolidation", color="#2ca02c", edgecolor="white",
                     linewidth=0.5, zorder=3)

ax2.set_xticks(x2)
ax2.set_xticklabels(stats_null["short_name"], rotation=35, ha="right")
ax2.set_ylabel(r"Mean Nullity $\nu$")
ax2.set_title("Nullity Before vs After T-Gate Consolidation")
ax2.set_yscale("symlog", linthresh=1)
ax2.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="black", fancybox=False)
ax2.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)

fig2.tight_layout()
out2_png = repo / "experiment3_nullity_reduction.png"
out2_pdf = repo / "experiment3_nullity_reduction.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Chi improvement factor (log scale)
# ═══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(14, 6))

stats_chi = stats.sort_values("mean_chi_improvement", ascending=True).reset_index(drop=True)

x3 = np.arange(len(stats_chi))
colors = [cat_colors.get(category_map.get(sn, "Algorithm"), "#333333")
          for sn in stats_chi["short_name"]]

# chi_improvement < 1 means improvement (consol_chi / orig_chi)
bars3 = ax3.barh(x3, 1.0 / stats_chi["mean_chi_improvement"].clip(lower=1e-10), 0.72,
                  color=colors, edgecolor="white", linewidth=0.5, zorder=3)

ax3.set_yticks(x3)
ax3.set_yticklabels(stats_chi["short_name"])
ax3.set_xlabel(r"$\chi$ Improvement Factor ($\chi_{\mathrm{orig}} / \chi_{\mathrm{consol}}$)")
ax3.set_title("Predicted Bond Dimension Improvement from T-Gate Consolidation")
ax3.set_xscale("symlog", linthresh=1)
ax3.axvline(x=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
ax3.xaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax3.set_axisbelow(True)

fig3.tight_layout()
out3_png = repo / "experiment3_chi_improvement.png"
out3_pdf = repo / "experiment3_chi_improvement.pdf"
fig3.savefig(out3_png)
fig3.savefig(out3_pdf)
print(f"Saved: {out3_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Consolidation mechanism breakdown (TT→S vs T·T†→I)
# ═══════════════════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(14, 6))

stats_mech = stats.sort_values("mean_n_tt", ascending=False).reset_index(drop=True)
# Filter to families with any consolidation
stats_mech = stats_mech[(stats_mech["mean_n_tt"] > 0) | (stats_mech["mean_n_cancel"] > 0)].reset_index(drop=True)

if len(stats_mech) > 0:
    x4 = np.arange(len(stats_mech))
    width4 = 0.35

    ax4.bar(x4 - width4/2, stats_mech["mean_n_tt"], width4,
            label=r"$T \cdot T \to S$ (phase merge)", color="#4C72B0",
            edgecolor="white", linewidth=0.5, zorder=3)
    ax4.bar(x4 + width4/2, stats_mech["mean_n_cancel"], width4,
            label=r"$T \cdot T^\dagger \to I$ (cancellation)", color="#DD8452",
            edgecolor="white", linewidth=0.5, zorder=3)

    ax4.set_xticks(x4)
    ax4.set_xticklabels(stats_mech["short_name"], rotation=35, ha="right")
    ax4.set_ylabel("Mean Consolidation Operations")
    ax4.set_title("T-Gate Consolidation Mechanism Breakdown")
    ax4.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="black", fancybox=False)
    ax4.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
    ax4.set_axisbelow(True)
    ax4.set_yscale("symlog", linthresh=1)

fig4.tight_layout()
out4_png = repo / "experiment3_mechanism_breakdown.png"
out4_pdf = repo / "experiment3_mechanism_breakdown.pdf"
fig4.savefig(out4_png)
fig4.savefig(out4_pdf)
print(f"Saved: {out4_png}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 3 SUMMARY: T-GATE CONSOLIDATION")
print("="*100)
print(f"{'Family':25s} {'T_orig':>8s} {'T_consol':>8s} {'Red%':>7s} {'ν_orig':>8s} {'ν_consol':>8s} {'χ impr':>8s}")
print("-"*100)
for _, row in stats_sorted.iterrows():
    chi_impr = 1.0 / row["mean_chi_improvement"] if row["mean_chi_improvement"] > 0 else float("inf")
    print(f"{row['short_name']:25s} {row['mean_orig_t']:8.0f} {row['mean_consol_t']:8.0f} "
          f"{row['mean_t_reduction_pct']*100:6.1f}% {row['mean_orig_nullity']:8.1f} "
          f"{row['mean_consol_nullity']:8.1f} {chi_impr:8.1f}x")
print("="*100)

plt.show()
