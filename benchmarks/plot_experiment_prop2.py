#!/usr/bin/env python3
"""
Proposition 2: Repeated-Block Rank Bound — Publication Figure
==============================================================
Validates rank(z) ≤ g + (r−1)·N for VQE (varying layers) and Grover
(varying iterations).

Layout:
  Top row (2 panels): Zoomed-in rank vs r (no bound lines) with y=N
                       saturation lines. Shows actual rank behavior clearly.
  Bottom row (1 panel, centered): Tightness ratio (rank/bound vs r).
                       Demonstrates the bound is satisfied and very loose.

Uses: results/experiment_prop2_rank_bound.csv

Output:
  experiment_prop2_rank_vs_r.png / .pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
csv_path = repo / "results" / "experiment_prop2_rank_bound.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df[df["success"] == True].copy()

print(f"Loaded {len(df)} successful experiments")
print(f"Families: {df['family'].nunique()}")
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

# ── Color palette by qubit count ──────────────────────────────────────────────
n_colors = {
    4: "#1f77b4", 6: "#ff7f0e", 8: "#2ca02c",
    10: "#d62728", 12: "#9467bd", 16: "#8c564b",
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE: 2×2 layout
#   Top row:    Zoomed-in rank vs r (Grover left, VQE right)
#   Bottom row: Tightness ratio (rank/bound) spanning both columns
# ═══════════════════════════════════════════════════════════════════════════════

families = sorted(df["family"].unique())

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

# ── Top row: Zoomed-in rank vs r (no bound lines) ────────────────────────────

for col, family in enumerate(families):
    ax = fig.add_subplot(gs[0, col])
    fam_data = df[df["family"] == family]

    n_vals = sorted(fam_data["n_qubits"].unique())

    for n in n_vals:
        nd = fam_data[fam_data["n_qubits"] == n]
        color = n_colors.get(n, "#333333")

        # Average over realizations per (n, r)
        avg = nd.groupby("r").agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
        ).reset_index()

        avg["std_rank"] = avg["std_rank"].fillna(0)

        ax.plot(avg["r"], avg["mean_rank"], "o-", color=color, linewidth=1.8,
                markersize=5, label=f"$N = {n}$", zorder=3)

        # Shaded error band (±1 std)
        if avg["std_rank"].max() > 0:
            ax.fill_between(avg["r"],
                            avg["mean_rank"] - avg["std_rank"],
                            avg["mean_rank"] + avg["std_rank"],
                            color=color, alpha=0.12, zorder=1)

    # Faint gray dashed reference lines at y = N (saturation level)
    r_max = fam_data["r"].max()
    for n in n_vals:
        ax.axhline(y=n, color="#888888", linestyle="--", linewidth=0.7,
                   alpha=0.5, zorder=1)
        ax.annotate(f"$N\\!={n}$", xy=(1.0, n), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#555555", va="center", ha="left",
                    annotation_clip=False)

    # Y-axis scaled to actual data range
    max_rank = fam_data["rank"].max()
    max_n = max(n_vals)
    y_upper = max(max_rank, max_n) * 1.3 + 1
    ax.set_ylim(-0.5, y_upper)

    # Panel label
    panel_label = "(a)" if col == 0 else "(b)"
    ax.text(0.03, 0.07, panel_label, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="left")

    ax.set_xlabel("Repetitions $r$")
    ax.set_ylabel("GF(2) Rank")
    ax.set_title(f"{family}: Actual Rank vs Repetitions")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="black",
              fancybox=False, fontsize=9)

# ── Bottom row: Tightness ratio (spanning both columns) ──────────────────────

ax_bottom = fig.add_subplot(gs[1, :])

for family in families:
    fam_data = df[df["family"] == family]

    n_vals = sorted(fam_data["n_qubits"].unique())
    for n in n_vals:
        nd = fam_data[fam_data["n_qubits"] == n]
        avg = nd.groupby("r").agg(
            mean_ratio=("rank_to_bound_ratio", "mean"),
        ).reset_index()

        marker = "o" if family == "VQE" else "s"
        ax_bottom.plot(avg["r"], avg["mean_ratio"], marker=marker,
                       color=n_colors.get(n, "#333"),
                       linewidth=1.4, markersize=5, alpha=0.8,
                       label=f"{family}, $N={n}$", zorder=3)

ax_bottom.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                  zorder=1, clip_on=True)

# Panel label
ax_bottom.text(0.015, 0.95, "(c)", transform=ax_bottom.transAxes, fontsize=14,
               fontweight="bold", va="top", ha="left")

ax_bottom.set_xlabel("Repetitions $r$")
ax_bottom.set_ylabel(r"Rank / Bound  ($\mathrm{rank}\; /\; [g + (r{-}1)\,N]$)")
ax_bottom.set_title("Tightness of Proposition 2 Bound")
y_max_ratio = df["rank_to_bound_ratio"].max()
ax_bottom.set_ylim(0, min(1.15, max(0.25, y_max_ratio * 1.3)))
ax_bottom.grid(True, alpha=0.15, linewidth=0.5)
ax_bottom.set_axisbelow(True)

handles, labels = ax_bottom.get_legend_handles_labels()
ax_bottom.legend(handles, labels, loc="upper right", frameon=True,
                 framealpha=0.9, edgecolor="black", fancybox=False,
                 ncol=2, fontsize=9)

# ── Caption annotation ────────────────────────────────────────────────────────
fig.text(0.5, -0.01,
         "Rank saturates at exactly $N$ from $r=1$ for both families, meaning every "
         "T-gate beyond the first $N$\ncontributes only to GF(2) nullity. "
         "The Proposition 2 bound $[g + (r{-}1)N]$ is satisfied universally but "
         "extremely loose (ratio $< 0.1$).",
         ha="center", va="top", fontsize=11, style="italic",
         wrap=True)

fig.savefig(repo / "experiment_prop2_rank_vs_r.png")
fig.savefig(repo / "experiment_prop2_rank_vs_r.pdf")
print(f"Saved: {repo / 'experiment_prop2_rank_vs_r.png'}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("PROPOSITION 2 SUMMARY")
print("=" * 90)
n_total = len(df)
n_violations = (df["bound_satisfied"] == False).sum()
print(f"Total experiments: {n_total}")
print(f"Bound violations: {n_violations}")
if n_violations == 0:
    print("Proposition 2 bound holds universally!")

for family in families:
    fam_data = df[df["family"] == family]
    max_ratio = fam_data["rank_to_bound_ratio"].max()
    mean_ratio = fam_data["rank_to_bound_ratio"].mean()
    print(f"  {family}: max(rank/bound) = {max_ratio:.4f}, mean = {mean_ratio:.4f}")

print("=" * 90)

plt.show()
