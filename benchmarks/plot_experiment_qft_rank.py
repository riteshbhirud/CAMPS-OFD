#!/usr/bin/env python3
"""
QFT Rank Scaling — Publication Figures
========================================
Part A: Log-log rank vs N with power law fit.
Part B: Per-R_k cumulative rank staircase.

Uses: results/experiment_qft_rank_scaling.csv
      results/experiment_qft_rank_incremental.csv

Output:
  experiment_qft_rank_scaling.png / .pdf     (log-log rank vs N)
  experiment_qft_rank_incremental.png / .pdf (per-R_k staircase)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
scaling_csv = repo / "results" / "experiment_qft_rank_scaling.csv"
incr_csv = repo / "results" / "experiment_qft_rank_incremental.csv"

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
# FIGURE 1: Log-log rank vs N with power law fit
# ═══════════════════════════════════════════════════════════════════════════════

df_s = pd.read_csv(scaling_csv)
df_s = df_s[df_s["success"] == True].copy()

print(f"Loaded {len(df_s)} scaling experiments")

fig1, ax1 = plt.subplots(figsize=(8, 6))

# Per-N averages
avg = df_s.groupby("n_qubits").agg(
    mean_rank=("rank", "mean"),
    std_rank=("rank", "std"),
    mean_t=("t", "mean"),
).reset_index()

ns = avg["n_qubits"].values.astype(float)
ranks = avg["mean_rank"].values

# Data points
ax1.errorbar(ns, ranks, yerr=avg["std_rank"].fillna(0).values,
             fmt="o", color="#1f77b4", markersize=8, capsize=4,
             linewidth=1.5, label="Data", zorder=3)

# Power law fit on log-log
valid = (ranks > 0) & (ns > 0)
if valid.sum() >= 2:
    log_n = np.log(ns[valid])
    log_r = np.log(ranks[valid])
    slope, intercept = np.polyfit(log_n, log_r, 1)
    a = np.exp(intercept)
    b = slope

    # R²
    ss_res = np.sum((log_r - (intercept + slope * log_n))**2)
    ss_tot = np.sum((log_r - log_r.mean())**2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    n_fit = np.linspace(ns.min() * 0.8, ns.max() * 1.2, 100)
    rank_fit = a * n_fit**b
    ax1.plot(n_fit, rank_fit, "--", color="#d62728", linewidth=1.5,
             label=f"Fit: rank = {a:.2f} $\\cdot$ $N^{{{b:.2f}}}$ ($R^2$={r_sq:.3f})",
             zorder=2)

# Reference: rank = N line
n_ref = np.linspace(ns.min() * 0.8, ns.max() * 1.2, 100)
ax1.plot(n_ref, n_ref, ":", color="gray", linewidth=1.0, alpha=0.5,
         label="rank $= N$", zorder=1)

# Actual T-count t(N) — the gap between t and rank is the nullity
ax1.plot(ns, avg["mean_t"].values, "s--", color="#ff7f0e", markersize=6,
         linewidth=1.2, alpha=0.8, label="$t(N)$ (T-gate count)", zorder=2)

# Shade the nullity gap between t(N) and rank
ax1.fill_between(ns, ranks, avg["mean_t"].values,
                 color="#ff7f0e", alpha=0.08, zorder=0)
# Label the gap at midpoint
mid_idx = len(ns) // 2
mid_n = ns[mid_idx]
mid_rank = ranks[mid_idx]
mid_t = avg["mean_t"].values[mid_idx]
ax1.annotate("nullity $\\nu = t - \\mathrm{rank}$",
             xy=(mid_n, np.sqrt(mid_rank * mid_t)),
             fontsize=9, color="#ff7f0e", ha="left", va="center",
             xytext=(8, 0), textcoords="offset points", style="italic")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Qubit Count $N$")
ax1.set_ylabel("GF(2) Rank / T-count")
ax1.set_title("QFT GF(2) Rank Scaling with Qubit Count")
ax1.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax1.legend(loc="upper left", frameon=True, framealpha=0.9,
           edgecolor="black", fancybox=False)

fig1.tight_layout()
out1_png = repo / "experiment_qft_rank_scaling.png"
out1_pdf = repo / "experiment_qft_rank_scaling.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-R_k cumulative rank staircase
# ═══════════════════════════════════════════════════════════════════════════════

df_i = pd.read_csv(incr_csv)
print(f"Loaded {len(df_i)} incremental records")

n_vals = sorted(df_i["n_qubits"].unique())
n_colors = {
    8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c",
    48: "#d62728", 64: "#9467bd",
}

fig2, axes2 = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 5),
                           sharey=True)
if len(n_vals) == 1:
    axes2 = [axes2]

for i, n in enumerate(n_vals):
    ax2 = axes2[i]
    nd = df_i[df_i["n_qubits"] == n]
    color = n_colors.get(n, "#333333")
    t_max = nd["cumulative_t"].max()

    # Normalize x to fractional position [0, 1]
    x_frac = nd["cumulative_t"] / t_max

    ax2.plot(x_frac, nd["cumulative_rank"],
             color=color, linewidth=1.5, zorder=3)

    # Mark rank jumps (rank_delta > 0) with dots
    jumps = nd[nd["rank_delta"] > 0]
    ax2.scatter(jumps["cumulative_t"] / t_max, jumps["cumulative_rank"],
                color=color, s=20, zorder=4)

    # Draw horizontal line at rank = N
    ax2.axhline(y=n, color=color, linestyle=":", linewidth=0.8, alpha=0.3)

    ax2.set_xlabel("Fractional T-Gate Position")
    ax2.set_title(f"$N = {n}$ ({t_max:,} T-gates)")
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_xlim(0, 1)

axes2[0].set_ylabel("Cumulative GF(2) Rank")
fig2.suptitle("QFT: Incremental Rank Growth per Controlled Rotation",
              fontsize=15, y=1.02)

fig2.tight_layout()
out2_png = repo / "experiment_qft_rank_incremental.png"
out2_pdf = repo / "experiment_qft_rank_incremental.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ── Per-R_k summary ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PER-R_k RANK CONTRIBUTIONS")
print("="*70)
for n in n_vals:
    nd = df_i[df_i["n_qubits"] == n]
    k_vals = sorted(nd["rotation_k"].unique())
    k_vals = [k for k in k_vals if k > 0]
    print(f"\nN={n}: total T={len(nd)}, final rank={nd['cumulative_rank'].iloc[-1]}")
    for k in k_vals:
        kd = nd[nd["rotation_k"] == k]
        total_delta = kd["rank_delta"].sum()
        print(f"  R_{k}: {len(kd)} T-gates, rank contribution = {total_delta}")
print("="*70)

plt.show()
