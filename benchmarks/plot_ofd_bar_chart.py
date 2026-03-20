#!/usr/bin/env python3
"""
OFD Success Rate by Circuit Family — Bar Chart
================================================
Publication-quality bar chart showing mean OFD success rate ± std
for all 14 circuit families, sorted descending.

Uses: results/results.csv (960 circuits)
Output: ofd_success_by_family.png / .pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
bench_csv = repo / "results" / "results.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
bench = pd.read_csv(bench_csv)
bench["success"] = bench["success"].astype(str).str.upper() == "TRUE"
bench = bench[bench["success"]].copy()

n_total = len(bench)

# ── Per-family statistics ─────────────────────────────────────────────────────
stats = bench.groupby("family").agg(
    mean_ofd=("ofd_rate", "mean"),
    std_ofd=("ofd_rate", "std"),
    count=("ofd_rate", "count"),
).sort_values("mean_ofd", ascending=False).reset_index()

# ── Short display names ──────────────────────────────────────────────────────
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
stats["short_name"] = stats["family"].map(short_names)

# ── Category colors ───────────────────────────────────────────────────────────
category_map = {
    "Random (A2A)": "Random",
    "Random (Brick)": "Random",
    "Bernstein-Vaz.": "Algorithm",
    "Simon's": "Algorithm",
    "Deutsch-Jozsa": "Algorithm",
    "QAOA": "Algorithm",
    "Surface Code": "Algorithm",
    "QFT": "Algorithm",
    "Grover": "Algorithm",
    "VQE": "Algorithm",
    "GHZ": "State",
    "Bell State": "State",
    "Graph State": "State",
    "Cluster State": "State",
}
cat_colors = {
    "Random": "#4C72B0",
    "Algorithm": "#55A868",
    "State": "#DD8452",
}
stats["category"] = stats["short_name"].map(category_map)
stats["color"] = stats["category"].map(cat_colors)

# ── Publication-quality plot ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(stats))
bars = ax.bar(
    x,
    stats["mean_ofd"] * 100,
    yerr=stats["std_ofd"] * 100,
    capsize=4,
    color=stats["color"],
    edgecolor="white",
    linewidth=0.5,
    width=0.72,
    error_kw={"linewidth": 1.2, "capthick": 1.2},
    zorder=3,
)

# Value labels above bars
for i, (_, row) in enumerate(stats.iterrows()):
    pct = row["mean_ofd"] * 100
    y_pos = pct + row["std_ofd"] * 100 + 1.5
    ax.text(i, y_pos, f"{pct:.0f}%", ha="center", va="bottom",
            fontsize=10, fontweight="medium")

# 50% reference line
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(stats["short_name"], rotation=35, ha="right")
ax.set_ylabel("OFD Success Rate (%)")
ax.set_xlabel("Circuit Family")
ax.set_title(f"OFD Success Rate by Circuit Family (n={n_total} circuits)")
ax.set_ylim(0, 105)
ax.set_xlim(-0.6, len(stats) - 0.4)

# Grid
ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cat_colors["Random"], label="Random"),
    Patch(facecolor=cat_colors["Algorithm"], label="Algorithm"),
    Patch(facecolor=cat_colors["State"], label="State"),
]
ax.legend(handles=legend_elements, loc="upper right", frameon=True,
          framealpha=0.9, edgecolor="black", fancybox=False,
          fontsize=11, borderpad=0.6)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_png = repo / "ofd_success_by_family.png"
out_pdf = repo / "ofd_success_by_family.pdf"
fig.savefig(out_png)
fig.savefig(out_pdf)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

plt.show()
