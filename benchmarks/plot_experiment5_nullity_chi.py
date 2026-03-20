#!/usr/bin/env python3
"""
Experiment 5: Nullity vs Bond Dimension Validation — Publication Figures
=========================================================================
Validates the theoretical bound chi <= 2^nu and shows how tight it is
across circuit families. Compares OFD (with naive absorption) vs
NoDisentangling (naive absorption only) baseline.

Uses: results/experiment5_nullity_vs_chi.csv

Output:
  experiment5_bound_validation.png / .pdf   (chi vs 2^nu scatter)
  experiment5_tightness.png / .pdf          (Tightness by family)
  experiment5_strategy_comparison.png / .pdf (OFD vs NoDisentangling)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "experiment5_nullity_vs_chi.csv"

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

# ── Scaling class ──────────────────────────────────────────────────────────────
ofd_friendly = {"Random (A2A)", "Random (Brick)"}
ofd_hostile = {"QFT", "VQE", "Grover", "GHZ"}

def get_class(sname):
    if sname in ofd_friendly:
        return "OFD-friendly"
    elif sname in ofd_hostile:
        return "OFD-hostile"
    return "Intermediate"

df["scaling_class"] = df["short_name"].apply(get_class)

# ── Color scheme ───────────────────────────────────────────────────────────────
class_colors = {
    "Random (A2A)": "#1f77b4", "Random (Brick)": "#4a90d9",
    "QFT": "#d62728", "VQE": "#e45756", "Grover": "#ff7f0e", "GHZ": "#c44e52",
    "Bernstein-Vaz.": "#2ca02c", "Cluster State": "#98df8a",
    "Graph State": "#8c564b", "Bell State": "#9467bd",
    "Simon's": "#7f7f7f", "Deutsch-Jozsa": "#bcbd22",
    "QAOA": "#17becf", "Surface Code": "#aec7e8",
}

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
# FIGURE 1: Two-panel — (a) Zoomed scatter  (b) Tightness bar chart
# ═══════════════════════════════════════════════════════════════════════════════

from matplotlib.patches import Patch

# Compute log2(chi_OFD), treating chi ≤ 1 as log2 = 0
df["log2_chi_ofd_plot"] = np.where(
    df["ofd_chi"] <= 1, 0.0, np.log2(df["ofd_chi"].astype(float))
)

# Compute tightness for panel (b): log2(chi_ofd) / nullity
df_with_null = df[df["nullity"] > 0].copy()
df_with_null["tightness"] = np.log2(df_with_null["ofd_chi"].clip(lower=1)) / df_with_null["nullity"]

tight_stats = df_with_null.groupby("short_name").agg(
    mean_tightness=("tightness", "mean"),
    std_tightness=("tightness", "std"),
    count=("tightness", "count"),
).reset_index()

tight_stats["category"] = tight_stats["short_name"].apply(
    lambda x: "Random" if "Random" in x else
    ("State" if x in {"GHZ", "Bell State", "Graph State", "Cluster State"} else "Algorithm")
)
cat_color_map = {"Random": "#4C72B0", "Algorithm": "#55A868", "State": "#DD8452"}
tight_stats["color"] = tight_stats["category"].map(cat_color_map)
tight_stats = tight_stats.sort_values("mean_tightness", ascending=False).reset_index(drop=True)

# ── Create two-panel figure ───────────────────────────────────────────────────
fig1, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

# ── Panel (a): Zoomed scatter, ν = 0..15 ─────────────────────────────────────
zoom_max = 15
df_zoom = df[df["nullity"] <= zoom_max]

# Shaded regions
ax_a.fill_between([0, zoom_max], [0, 0], [0, zoom_max],
                  color="#2ca02c", alpha=0.06, zorder=0)
ax_a.fill_between([0, zoom_max], [0, zoom_max], [zoom_max, zoom_max],
                  color="#d62728", alpha=0.06, zorder=0)

# Region labels
ax_a.text(zoom_max * 0.72, zoom_max * 0.12,
          r"$\chi \leq 2^\nu$ satisfied", fontsize=9, color="#2ca02c",
          alpha=0.5, style="italic", ha="center")
ax_a.text(zoom_max * 0.25, zoom_max * 0.88,
          "Bound violated", fontsize=9, color="#d62728",
          alpha=0.5, style="italic", ha="center")

# Diagonal reference line: y = x
ax_a.plot([0, zoom_max], [0, zoom_max], color="#555555", linestyle="--",
          linewidth=1.2, alpha=0.6,
          label=r"$\log_2 \chi = \nu$ (bound)", zorder=2)

# Plot each family (only points with nu <= 15)
for sname in sorted(df_zoom["short_name"].unique()):
    fam_data = df_zoom[df_zoom["short_name"] == sname]
    cls = get_class(sname)
    color = class_colors.get(sname, "#333333")
    marker = class_markers.get(cls, "o")

    ax_a.scatter(fam_data["nullity"], fam_data["log2_chi_ofd_plot"],
                 color=color, s=60, marker=marker, edgecolor="white",
                 linewidth=0.5, alpha=0.8, label=sname, zorder=3)

ax_a.set_xlabel(r"Nullity $\nu = t - \mathrm{rank}$")
ax_a.set_ylabel(r"$\log_2(\chi_{\mathrm{OFD}})$")
ax_a.set_title(r"(a) Zoomed: $\log_2(\chi_{\mathrm{OFD}})$ vs $\nu$")
ax_a.set_xlim(-0.5, zoom_max)
log2chi_zoom_max = df_zoom["log2_chi_ofd_plot"].max()
ax_a.set_ylim(-0.3, max(log2chi_zoom_max * 1.3 + 0.5, zoom_max))
ax_a.grid(True, alpha=0.15, linewidth=0.5)
ax_a.set_axisbelow(True)
ax_a.legend(loc="upper left", frameon=True, framealpha=0.9,
            edgecolor="black", fancybox=False, fontsize=8)

# Note: zoomed view caption (below x-axis label, in figure coordinates)
ax_a.text(0.5, -0.22,
          r"Zoomed to $\nu \leq 15$. Structured circuits ($\nu$ up to ${\sim}900$) "
          r"satisfy bound trivially (see panel b).",
          fontsize=7.5, color="#555555", ha="center", va="top", style="italic",
          transform=ax_a.transAxes)

# ── Panel (b): Tightness bar chart ───────────────────────────────────────────
x2 = np.arange(len(tight_stats))
ax_b.bar(
    x2,
    tight_stats["mean_tightness"],
    yerr=tight_stats["std_tightness"].fillna(0),
    capsize=4,
    color=tight_stats["color"],
    edgecolor="white",
    linewidth=0.5,
    width=0.72,
    error_kw={"linewidth": 1.2, "capthick": 1.2},
    zorder=3,
)

ax_b.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

ax_b.set_xticks(x2)
ax_b.set_xticklabels(tight_stats["short_name"], rotation=35, ha="right")
ax_b.set_ylabel(r"Tightness $\log_2(\chi) / \nu$")
ax_b.set_title(r"(b) Tightness of $\chi \leq 2^\nu$ by Family")
ax_b.set_ylim(0, 1.6)
ax_b.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax_b.set_axisbelow(True)

from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor="#4C72B0", label="Random"),
    Patch(facecolor="#55A868", label="Algorithm"),
    Patch(facecolor="#DD8452", label="State"),
    Line2D([0], [0], color="red", linestyle="--", linewidth=0.8, alpha=0.5,
           label="Tight bound (ratio = 1)"),
]
ax_b.legend(handles=legend_elements, loc="upper right", frameon=True,
            framealpha=0.9, edgecolor="black", fancybox=False)

fig1.tight_layout()
out1_png = repo / "experiment5_bound_validation.png"
out1_pdf = repo / "experiment5_bound_validation.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Two strategies compared — OFD vs No-Disentangling
# ═══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(14, 6))

strat_stats = df.groupby("short_name").agg(
    mean_chi_ofd=("ofd_chi", "mean"),
    mean_chi_none=("nodis_chi", "mean"),
    mean_predicted=("predicted_chi", "mean"),
).reset_index()

strat_stats = strat_stats.sort_values("mean_chi_none", ascending=False).reset_index(drop=True)

x3 = np.arange(len(strat_stats))
width3 = 0.30

bars_none = ax3.bar(x3 - width3/2, strat_stats["mean_chi_none"], width3,
                     label="No disentangling (naive absorption)", color="#d62728",
                     edgecolor="white", linewidth=0.5, zorder=3)
bars_ofd = ax3.bar(x3 + width3/2, strat_stats["mean_chi_ofd"], width3,
                    label="OFD (+ naive absorption on failure)", color="#2ca02c",
                    edgecolor="white", linewidth=0.5, zorder=3)

# Predicted chi markers
ax3.scatter(x3, strat_stats["mean_predicted"], color="black", marker="_",
            s=200, linewidths=2, zorder=4, label=r"$2^\nu$ (predicted)")

ax3.set_xticks(x3)
ax3.set_xticklabels(strat_stats["short_name"], rotation=35, ha="right")
ax3.set_ylabel("Mean Bond Dimension $\\chi$")
ax3.set_title("Bond Dimension: OFD vs No Disentangling")
ax3.set_yscale("symlog", linthresh=1)
ax3.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="black",
           fancybox=False, fontsize=10)
ax3.yaxis.grid(True, alpha=0.15, linewidth=0.5, zorder=0)
ax3.set_axisbelow(True)

fig3.tight_layout()
out3_png = repo / "experiment5_strategy_comparison.png"
out3_pdf = repo / "experiment5_strategy_comparison.pdf"
fig3.savefig(out3_png)
fig3.savefig(out3_pdf)
print(f"Saved: {out3_png}")

# ── Bound violation check ──────────────────────────────────────────────────────
print("\n" + "="*90)
print("BOUND VALIDATION CHECK")
print("="*90)
violations = df[df["ofd_chi"] > df["predicted_chi"]]
n_violations = len(violations)
n_total = len(df)
print(f"Total experiments: {n_total}")
print(f"Bound violations (chi_OFD > 2^nu): {n_violations}")
if n_violations > 0:
    print("\nViolation details:")
    for _, row in violations.iterrows():
        print(f"  {row['short_name']:25s} n={row['n_qubits']} nu={row['nullity']} "
              f"chi_pred={row['predicted_chi']} chi_ofd={row['ofd_chi']}")
else:
    print("No violations found — bound chi <= 2^nu holds universally!")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 5 SUMMARY")
print("="*100)
print(f"{'Family':25s} {'nu':>6s} {'2^nu':>8s} {'chi_OFD':>8s} {'chi_None':>8s} {'Tight':>7s}")
print("-"*100)
for _, row in strat_stats.iterrows():
    sname = row["short_name"]
    tight_row = tight_stats[tight_stats["short_name"] == sname]
    tightness = tight_row["mean_tightness"].values[0] if len(tight_row) > 0 else float("nan")
    print(f"{sname:25s} {df[df['short_name']==sname]['nullity'].mean():6.1f} "
          f"{row['mean_predicted']:8.1f} {row['mean_chi_ofd']:8.1f} "
          f"{row['mean_chi_none']:8.1f} {tightness:7.3f}")
print("="*100)

plt.show()
