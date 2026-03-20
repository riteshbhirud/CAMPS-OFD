#!/usr/bin/env python3
"""
Experiment 4: Per-T-Gate Positional OFD Analysis — Publication Figures
=======================================================================
Generates figures showing how OFD success varies with T-gate position
in the circuit, identifying early-circuit vs late-circuit OFD failure.

Uses: results/experiment4_positional_ofd.csv
      results/experiment4_positional_ofd_summary.csv

Output:
  experiment4_ofd_vs_position.png / .pdf        (OFD success by position)
  experiment4_first_failure.png / .pdf           (First failure position)
  experiment4_pauli_weight_position.png / .pdf   (Pauli weight vs position)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
detail_csv = repo / "results" / "experiment4_positional_ofd.csv"
summary_csv = repo / "results" / "experiment4_positional_ofd_summary.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
detail = pd.read_csv(detail_csv)
summary = pd.read_csv(summary_csv)

print(f"Loaded {len(detail)} per-T-gate records across {summary['family'].nunique()} families")
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
detail["short_name"] = detail["family"].map(short_names)
summary["short_name"] = summary["family"].map(short_names)

# ── Scaling class ──────────────────────────────────────────────────────────────
ofd_friendly = {"Random (A2A)", "Random (Brick)"}
ofd_hostile = {"QFT", "VQE", "Grover", "GHZ"}

def get_class(sname):
    if sname in ofd_friendly:
        return "OFD-friendly"
    elif sname in ofd_hostile:
        return "OFD-hostile"
    return "Intermediate"

# ── Color scheme ───────────────────────────────────────────────────────────────
class_colors = {
    "Random (A2A)": "#1f77b4", "Random (Brick)": "#4a90d9",
    "QFT": "#d62728", "VQE": "#e45756", "Grover": "#ff7f0e", "GHZ": "#c44e52",
    "Bernstein-Vaz.": "#2ca02c", "Cluster State": "#98df8a",
    "Graph State": "#8c564b", "Bell State": "#9467bd",
    "Simon's": "#7f7f7f", "Deutsch-Jozsa": "#bcbd22",
    "QAOA": "#17becf", "Surface Code": "#aec7e8",
}

class_linestyles = {"OFD-friendly": "-", "OFD-hostile": "--", "Intermediate": "-."}
class_markers = {"OFD-friendly": "o", "OFD-hostile": "s", "Intermediate": "^"}

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
# FIGURE 1: OFD Success Rate vs Fractional Position (binned)
# ═══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(12, 7))

# Bin the fractional positions
n_bins = 10
detail["pos_bin"] = pd.cut(detail["fractional_position"], bins=n_bins, labels=False)
detail["pos_bin_center"] = (detail["pos_bin"] + 0.5) / n_bins

# Select representative families (one per scaling class with enough data)
representative_families = []
for sname in detail["short_name"].unique():
    fam_data = detail[detail["short_name"] == sname]
    if len(fam_data) >= 10:  # Need enough data points for binning
        representative_families.append(sname)

# Sort by scaling class for consistent ordering
representative_families.sort(key=lambda x: (
    0 if get_class(x) == "OFD-friendly" else
    1 if get_class(x) == "Intermediate" else 2,
    x
))

for sname in representative_families:
    fam_data = detail[detail["short_name"] == sname]
    cls = get_class(sname)
    color = class_colors.get(sname, "#333333")
    ls = class_linestyles.get(cls, "-")
    marker = class_markers.get(cls, "o")

    # Compute OFD success rate per bin
    binned = fam_data.groupby("pos_bin_center").agg(
        ofd_rate=("ofd_success", "mean"),
        count=("ofd_success", "count"),
    ).reset_index()

    # Only plot bins with enough data
    binned = binned[binned["count"] >= 2]

    ax1.plot(binned["pos_bin_center"], binned["ofd_rate"] * 100,
             color=color, linestyle=ls, marker=marker, markersize=5,
             linewidth=1.8, label=sname, alpha=0.85)

ax1.set_xlabel("Fractional Position in Circuit")
ax1.set_ylabel("OFD Success Rate (%)")
ax1.set_title("OFD Success Rate vs T-Gate Position")
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-3, 103)
ax1.grid(True, alpha=0.15, linewidth=0.5)
ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1)

fig1.tight_layout()
out1_png = repo / "experiment4_ofd_vs_position.png"
out1_pdf = repo / "experiment4_ofd_vs_position.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: First-Failure Position Distribution
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(14, 6))

# Per-family mean first-failure position
fam_stats = summary.groupby("short_name").agg(
    mean_first_fail=("first_fail_position", "mean"),
    std_first_fail=("first_fail_position", "std"),
    mean_ofd_rate=("ofd_rate", "mean"),
    count=("ofd_rate", "count"),
).reset_index()

fam_stats["category"] = fam_stats["short_name"].apply(
    lambda x: "Random" if "Random" in x else
    ("State" if x in {"GHZ", "Bell State", "Graph State", "Cluster State"} else "Algorithm")
)
cat_colors_map = {"Random": "#4C72B0", "Algorithm": "#55A868", "State": "#DD8452"}
fam_stats["color"] = fam_stats["category"].map(cat_colors_map)

fam_stats = fam_stats.sort_values("mean_first_fail", ascending=False).reset_index(drop=True)

x2 = np.arange(len(fam_stats))
bars = ax2.bar(
    x2,
    fam_stats["mean_first_fail"],
    yerr=fam_stats["std_first_fail"].fillna(0),
    capsize=4,
    color=fam_stats["color"],
    edgecolor="white",
    linewidth=0.5,
    width=0.72,
    error_kw={"linewidth": 1.2, "capthick": 1.2},
    zorder=3,
)

# Value labels
for i, (_, row) in enumerate(fam_stats.iterrows()):
    val = row["mean_first_fail"]
    y_pos = val + (row["std_first_fail"] if pd.notna(row["std_first_fail"]) else 0) + 0.02
    ax2.text(i, y_pos, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax2.set_xticks(x2)
ax2.set_xticklabels(fam_stats["short_name"], rotation=35, ha="right")
ax2.set_ylabel("Mean First-Failure Fractional Position")
ax2.set_title("Where OFD First Fails in the Circuit")
ax2.set_ylim(0, 1.15)
ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
ax2.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", label="Random"),
    Patch(facecolor="#55A868", label="Algorithm"),
    Patch(facecolor="#DD8452", label="State"),
]
ax2.legend(handles=legend_elements, loc="upper right", frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False)

fig2.tight_layout()
out2_png = repo / "experiment4_first_failure.png"
out2_pdf = repo / "experiment4_first_failure.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: OFD success vs GF(2) independence and pure-Z rate
# ═══════════════════════════════════════════════════════════════════════════════

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: OFD rate vs GF(2) independence rate
fam_corr = summary.groupby("short_name").agg(
    mean_ofd=("ofd_rate", "mean"),
    mean_indep=("gf2_independence_rate", "mean"),
    mean_pz=("pure_z_rate", "mean"),
).reset_index()

for _, row in fam_corr.iterrows():
    sname = row["short_name"]
    color = class_colors.get(sname, "#333333")
    cls = get_class(sname)
    marker = class_markers.get(cls, "o")
    ax3a.scatter(row["mean_indep"] * 100, row["mean_ofd"] * 100,
                 color=color, s=80, marker=marker, edgecolor="white",
                 linewidth=0.5, zorder=3, label=sname)

ax3a.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3, label="y=x")
ax3a.set_xlabel("GF(2) Independence Rate (%)")
ax3a.set_ylabel("OFD Success Rate (%)")
ax3a.set_title("OFD Success vs GF(2) Independence")
ax3a.set_xlim(-3, 103)
ax3a.set_ylim(-3, 103)
ax3a.grid(True, alpha=0.15, linewidth=0.5)

# Right panel: Pure-Z rate (indicator of OFD-hostile structure)
for _, row in fam_corr.iterrows():
    sname = row["short_name"]
    color = class_colors.get(sname, "#333333")
    cls = get_class(sname)
    marker = class_markers.get(cls, "o")
    ax3b.scatter(row["mean_pz"] * 100, row["mean_ofd"] * 100,
                 color=color, s=80, marker=marker, edgecolor="white",
                 linewidth=0.5, zorder=3, label=sname)

ax3b.set_xlabel("Pure-Z Twisted Pauli Rate (%)")
ax3b.set_ylabel("OFD Success Rate (%)")
ax3b.set_title("OFD Success vs Pure-Z Rate")
ax3b.set_xlim(-3, 103)
ax3b.set_ylim(-3, 103)
ax3b.grid(True, alpha=0.15, linewidth=0.5)

# Single legend for both panels
handles, labels = ax3a.get_legend_handles_labels()
fig3.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=True, framealpha=0.9, edgecolor="black", fancybox=False,
            ncol=1, fontsize=9)

fig3.tight_layout()
out3_png = repo / "experiment4_ofd_correlations.png"
out3_pdf = repo / "experiment4_ofd_correlations.pdf"
fig3.savefig(out3_png)
fig3.savefig(out3_pdf)
print(f"Saved: {out3_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Mean Pauli Weight vs Position (reveals structural buildup)
# ═══════════════════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(12, 7))

for sname in representative_families:
    fam_data = detail[detail["short_name"] == sname]
    cls = get_class(sname)
    color = class_colors.get(sname, "#333333")
    ls = class_linestyles.get(cls, "-")
    marker = class_markers.get(cls, "o")

    binned = fam_data.groupby("pos_bin_center").agg(
        mean_weight=("pauli_weight", "mean"),
        count=("pauli_weight", "count"),
    ).reset_index()
    binned = binned[binned["count"] >= 2]

    ax4.plot(binned["pos_bin_center"], binned["mean_weight"],
             color=color, linestyle=ls, marker=marker, markersize=5,
             linewidth=1.8, label=sname, alpha=0.85)

ax4.set_xlabel("Fractional Position in Circuit")
ax4.set_ylabel("Mean Twisted Pauli Weight")
ax4.set_title("Twisted Pauli Weight Growth Through Circuit")
ax4.set_xlim(-0.02, 1.02)
ax4.grid(True, alpha=0.15, linewidth=0.5)
ax4.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
           framealpha=0.9, edgecolor="black", fancybox=False, ncol=1)

fig4.tight_layout()
out4_png = repo / "experiment4_pauli_weight_position.png"
out4_pdf = repo / "experiment4_pauli_weight_position.pdf"
fig4.savefig(out4_png)
fig4.savefig(out4_pdf)
print(f"Saved: {out4_png}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 4 SUMMARY: POSITIONAL OFD ANALYSIS")
print("="*100)
print(f"{'Family':25s} {'OFD%':>7s} {'PureZ%':>8s} {'Indep%':>8s} {'1stFail':>8s} {'MeanWt':>8s}")
print("-"*100)
for _, row in fam_corr.sort_values("mean_ofd", ascending=False).iterrows():
    ff = fam_stats[fam_stats["short_name"] == row["short_name"]]
    ff_val = ff["mean_first_fail"].values[0] if len(ff) > 0 else float("nan")
    print(f"{row['short_name']:25s} {row['mean_ofd']*100:6.1f}% {row['mean_pz']*100:7.1f}% "
          f"{row['mean_indep']*100:7.1f}% {ff_val:8.3f}")
print("="*100)

plt.show()
