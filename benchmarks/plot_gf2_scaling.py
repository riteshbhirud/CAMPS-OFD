#!/usr/bin/env python3
"""
GF(2) Null Space Scaling Classification — Publication Figures
==============================================================
Generates three key figures from GF(2) scaling analysis data:

Figure 1: Nullity ratio ν(n)/t(n) vs n for all families
Figure 2: Absolute GF(2) rank r(n) vs n
Figure 3: Nullity ν(n) vs n on log-log scale

Uses: results/gf2_scaling_analysis.csv

Output:
  gf2_nullity_ratio_scaling.png / .pdf   (Figure 1)
  gf2_rank_scaling.png / .pdf            (Figure 2)
  gf2_nullity_scaling.png / .pdf         (Figure 3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "gf2_scaling_analysis.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df[df["success"] == True].copy()

print(f"Loaded {len(df)} successful experiments")
print(f"Families: {df['family'].nunique()}")
print(f"Qubit range: {df['n_qubits'].min()} to {df['n_qubits'].max()}")
print()

# ── Per (family, n_qubits) averages ───────────────────────────────────────────
stats = df.groupby(["family", "n_qubits"]).agg(
    mean_t=("n_t_gates", "mean"),
    mean_rank=("gf2_rank", "mean"),
    mean_nullity=("nullity", "mean"),
    mean_nullity_ratio=("nullity_ratio", "mean"),
    mean_rank_ratio=("rank_ratio", "mean"),
    std_nullity_ratio=("nullity_ratio", "std"),
    count=("nullity_ratio", "count"),
).reset_index()

# ── Classification ────────────────────────────────────────────────────────────
# Three categories matching paper's Section 4.1 structure
category_map = {
    "Random Clifford+T (Brick-wall)":  "Random",
    "Random Clifford+T (All-to-all)":  "Random",
    "Bernstein-Vazirani":              "Algorithm",
    "Simon's Algorithm":               "Algorithm",
    "Deutsch-Jozsa":                   "Algorithm",
    "QAOA MaxCut (p=1, 3-regular)":    "Algorithm",
    "Surface Code":                    "Algorithm",
    "Quantum Fourier Transform":       "Algorithm",
    "Grover Search":                   "Algorithm",
    "VQE Hardware-Efficient Ansatz":    "Algorithm",
    "GHZ State":                       "State",
    "Bell State / EPR Pairs":          "State",
    "Graph State":                     "State",
    "Cluster State (1D)":              "State",
}

# Short names for legend
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

stats["category"] = stats["family"].map(category_map)
stats["short_name"] = stats["family"].map(short_names)

# ── Scaling class assignment (based on large-n nullity ratio) ─────────────────
scaling_class = {}
for fam in stats["family"].unique():
    fam_data = stats[stats["family"] == fam]
    max_n = fam_data["n_qubits"].max()
    large_n = fam_data[fam_data["n_qubits"] == max_n]
    nr = large_n["mean_nullity_ratio"].values[0]
    if nr < 0.3:
        scaling_class[fam] = "OFD-friendly"
    elif nr > 0.7:
        scaling_class[fam] = "OFD-hostile"
    else:
        scaling_class[fam] = "Intermediate"

stats["scaling_class"] = stats["family"].map(scaling_class)

# Print classification
print("Scaling Classification:")
for fam, cls in sorted(scaling_class.items(), key=lambda x: x[1]):
    fam_data = stats[stats["family"] == fam]
    max_n = fam_data["n_qubits"].max()
    nr = fam_data[fam_data["n_qubits"] == max_n]["mean_nullity_ratio"].values[0]
    print(f"  {short_names.get(fam, fam):20s}  {cls:15s}  ν/t={nr:.3f} (n={max_n})")
print()

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

# ── Color scheme by scaling class ────────────────────────────────────────────
# OFD-friendly: blues, OFD-hostile: reds/oranges, Intermediate: greens/grays
class_colors = {
    # OFD-friendly (blues)
    "Random (A2A)": "#1f77b4",
    "Random (Brick)": "#4a90d9",
    # OFD-hostile (reds/oranges)
    "QFT": "#d62728",
    "VQE": "#e45756",
    "Grover": "#ff7f0e",
    "GHZ": "#c44e52",
    # Intermediate (greens/grays/purples)
    "Bernstein-Vaz.": "#2ca02c",
    "Cluster State": "#98df8a",
    "Graph State": "#8c564b",
    "Bell State": "#9467bd",
    "Simon's": "#7f7f7f",
    "Deutsch-Jozsa": "#bcbd22",
    "QAOA": "#17becf",
    "Surface Code": "#aec7e8",
}

# Line styles by scaling class
class_linestyles = {
    "OFD-friendly": "-",
    "OFD-hostile": "--",
    "Intermediate": "-.",
}

class_markers = {
    "OFD-friendly": "o",
    "OFD-hostile": "s",
    "Intermediate": "^",
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Nullity Ratio ν/t vs n (THE key figure)
# ═══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(12, 7))

families_sorted = sorted(stats["family"].unique(),
                         key=lambda f: stats[stats["family"] == f]["mean_nullity_ratio"].iloc[-1])

for fam in families_sorted:
    fam_data = stats[stats["family"] == fam].sort_values("n_qubits")
    sname = short_names.get(fam, fam)
    cls = scaling_class.get(fam, "Intermediate")
    color = class_colors.get(sname, "#333333")
    ls = class_linestyles.get(cls, "-")
    marker = class_markers.get(cls, "o")

    ax1.plot(fam_data["n_qubits"], fam_data["mean_nullity_ratio"],
             color=color, linestyle=ls, marker=marker, markersize=5,
             linewidth=1.8, label=sname, alpha=0.85)


# Horizontal reference lines
ax1.axhline(y=0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax1.axhline(y=0.7, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

ax1.set_xlabel("Number of Qubits $n$")
ax1.set_ylabel(r"Nullity Ratio $\nu(n) / t(n)$")
ax1.set_title("GF(2) Null Space Scaling Classification")
ax1.set_ylim(-0.03, 1.03)
ax1.set_xscale("log", base=2)
ax1.set_xlim(3, ax1.get_xlim()[1] * 1.1)
ax1.grid(True, alpha=0.15, linewidth=0.5)
ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1)

fig1.tight_layout()
out1_png = repo / "gf2_nullity_ratio_scaling.png"
out1_pdf = repo / "gf2_nullity_ratio_scaling.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
# Also save as paper/figure1 for LaTeX
fig1.savefig(repo / "paper" / "figure1.png")
fig1.savefig(repo / "paper" / "figure1.pdf")
print(f"Saved: {out1_png} + paper/figure1.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Absolute GF(2) Rank vs n
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(12, 7))

for fam in families_sorted:
    fam_data = stats[stats["family"] == fam].sort_values("n_qubits")
    sname = short_names.get(fam, fam)
    cls = scaling_class.get(fam, "Intermediate")
    color = class_colors.get(sname, "#333333")
    ls = class_linestyles.get(cls, "-")
    marker = class_markers.get(cls, "o")

    ax2.plot(fam_data["n_qubits"], fam_data["mean_rank"],
             color=color, linestyle=ls, marker=marker, markersize=5,
             linewidth=1.8, label=sname, alpha=0.85)

# Reference line: rank = n (maximum possible)
n_ref = np.array(sorted(stats["n_qubits"].unique()))
ax2.plot(n_ref, n_ref, "k--", linewidth=1, alpha=0.4, label=r"$r = n$")

ax2.set_xlabel("Number of Qubits $n$")
ax2.set_ylabel("GF(2) Rank $r(n)$")
ax2.set_title("GF(2) Rank Growth by Circuit Family")
ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=2)
ax2.grid(True, alpha=0.15, linewidth=0.5)
ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1)

fig2.tight_layout()
out2_png = repo / "gf2_rank_scaling.png"
out2_pdf = repo / "gf2_rank_scaling.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Nullity ν(n) vs n on log-log scale
# ═══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(12, 7))

for fam in families_sorted:
    fam_data = stats[stats["family"] == fam].sort_values("n_qubits")
    sname = short_names.get(fam, fam)
    cls = scaling_class.get(fam, "Intermediate")
    color = class_colors.get(sname, "#333333")
    ls = class_linestyles.get(cls, "-")
    marker = class_markers.get(cls, "o")

    # Skip points with nullity = 0 (can't plot on log scale)
    plot_data = fam_data[fam_data["mean_nullity"] > 0]
    if len(plot_data) < 2:
        continue

    ax3.plot(plot_data["n_qubits"], plot_data["mean_nullity"],
             color=color, linestyle=ls, marker=marker, markersize=5,
             linewidth=1.8, label=sname, alpha=0.85)

ax3.set_xlabel("Number of Qubits $n$")
ax3.set_ylabel(r"Nullity $\nu(n)$")
ax3.set_title("Null Space Dimension Growth by Circuit Family")
ax3.set_xscale("log", base=2)
ax3.set_yscale("log", base=2)
ax3.grid(True, alpha=0.15, linewidth=0.5)
ax3.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1)

fig3.tight_layout()
out3_png = repo / "gf2_nullity_scaling.png"
out3_pdf = repo / "gf2_nullity_scaling.pdf"
fig3.savefig(out3_png)
fig3.savefig(out3_pdf)
# Also save as paper/figure2 for LaTeX (paper's Figure 2 = nullity log-log)
fig3.savefig(repo / "paper" / "figure2.png")
fig3.savefig(repo / "paper" / "figure2.pdf")
print(f"Saved: {out3_png} + paper/figure2.png")

# ── Print summary table for the paper ────────────────────────────────────────
print("\n" + "="*90)
print("TABLE FOR PAPER: GF(2) Scaling Classification")
print("="*90)
print(f"{'Family':30s} {'Class':15s} {'max n':>6s} {'ν/t':>8s} {'rank':>8s} {'T':>8s}")
print("-"*90)

for cls_name in ["OFD-friendly", "Intermediate", "OFD-hostile"]:
    fams_in_cls = [f for f, c in scaling_class.items() if c == cls_name]
    fams_in_cls.sort(key=lambda f: stats[stats["family"] == f]["mean_nullity_ratio"].iloc[-1])
    for fam in fams_in_cls:
        fam_data = stats[stats["family"] == fam]
        max_n = fam_data["n_qubits"].max()
        row = fam_data[fam_data["n_qubits"] == max_n].iloc[0]
        sname = short_names.get(fam, fam)
        print(f"{sname:30s} {cls_name:15s} {max_n:6d} {row['mean_nullity_ratio']:8.4f} "
              f"{row['mean_rank']:8.1f} {row['mean_t']:8.0f}")
    if fams_in_cls:
        print()

print("="*90)

plt.show()
