#!/usr/bin/env python3
"""
Figure 5: QFT GF(2) Rank Scaling (Extended to N = 256)
========================================================
Two-panel publication figure for the paper.

Panel (a): Log-log rank vs N with exact N-1 model, power-law fit,
           log-corrected fit, T-count curve, and nullity shading.
Panel (b): Residuals on log scale for each model.

Uses: results/experiment_qft_large_n_genC.csv  (Generator C, exact angles)

Output:
  paper/figure5.pdf
  paper/figure5.png
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
data_csv = repo / "results" / "experiment_qft_large_n_genC.csv"
paper_dir = repo / "paper"
paper_dir.mkdir(exist_ok=True)

# ── Publication-quality settings ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(data_csv)
print(f"Loaded {len(df)} data points from Gen C QFT")

ns = df["n_qubits"].values.astype(float)
ranks = df["rank"].values.astype(float)
ts = df["t"].values.astype(float)

# Verify rank = N - 1
all_match = all(int(ranks[i]) == int(ns[i]) - 1 for i in range(len(ns)))
print(f"rank = N - 1 for all N: {all_match}")
print(f"N range: {int(ns.min())} to {int(ns.max())}")
print()

# ── Fit models ─────────────────────────────────────────────────────────────────

# Model 1: Pure power law  rank = a · N^b
log_ns = np.log(ns)
log_ranks = np.log(ranks)
slope, intercept = np.polyfit(log_ns, log_ranks, 1)
a1, b1 = np.exp(intercept), slope

# Model 2: Log-corrected linear  rank = c · N · log(N)^α
y_adj = log_ranks - log_ns
loglog_ns = np.log(log_ns)
a2_slope, a2_intercept = np.polyfit(loglog_ns, y_adj, 1)
c2 = np.exp(a2_intercept)

# R² computation
def r_squared_log(actual, predicted):
    log_a = np.log(actual)
    log_p = np.log(np.clip(predicted, 1e-10, None))
    ss_res = np.sum((log_a - log_p)**2)
    ss_tot = np.sum((log_a - log_a.mean())**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

r2_powerlaw = r_squared_log(ranks, a1 * ns**b1)
r2_loglin = r_squared_log(ranks, c2 * ns * np.log(ns)**a2_slope)
r2_exact = 1.0  # rank = N-1 is exact

print(f"Model 1 (power law): rank = {a1:.2f} · N^{{{b1:.2f}}}  R² = {r2_powerlaw:.6f}")
print(f"Model 2 (log-linear): rank = {c2:.2f} · N · log(N)^{{{a2_slope:.2f}}}  R² = {r2_loglin:.6f}")
print(f"Model 3 (exact): rank = N - 1  R² = {r2_exact:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: Two-panel — (a) Log-log with fits  (b) Residuals
# ═══════════════════════════════════════════════════════════════════════════════

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={"width_ratios": [1.2, 1]})

# ── Panel (a): Log-log rank vs N with model fits ────────────────────────────

n_ref = np.geomspace(ns.min() * 0.7, ns.max() * 1.3, 300)

# Data points
ax_a.plot(ns, ranks, "o", color="#1f77b4", markersize=8,
          label="Data", zorder=5, markeredgecolor="white", markeredgewidth=0.5)

# T-count curve (orange squares)
ax_a.plot(ns, ts, "s--", color="#ff7f0e", markersize=5,
          linewidth=1.0, alpha=0.6, label="$t(N)$ (T-gate count)", zorder=2)

# Shade nullity gap
ax_a.fill_between(ns, ranks, ts, color="#ff7f0e", alpha=0.06, zorder=0)
mid_idx = len(ns) // 2
mid_n = ns[mid_idx]
mid_rank = ranks[mid_idx]
mid_t = ts[mid_idx]
ax_a.annotate(r"nullity $\nu = t - \mathrm{rank}$",
              xy=(mid_n, np.sqrt(mid_rank * mid_t)),
              fontsize=9, color="#ff7f0e", ha="left", va="center",
              xytext=(8, 0), textcoords="offset points", style="italic")

# Reference: rank = N (dotted gray)
ax_a.plot(n_ref, n_ref, ":", color="gray", linewidth=1.0, alpha=0.5,
          label="rank $= N$", zorder=1)

# Exact model: rank = N - 1 (green dashed)
ax_a.plot(n_ref, n_ref - 1, "--", color="#2ca02c", linewidth=2.0, alpha=0.7,
          label="rank $= N - 1$ (exact)", zorder=3)

# Power law fit (red dashed, for comparison)
rank_fit1 = a1 * n_ref**b1
ax_a.plot(n_ref, rank_fit1, "--", color="#d62728", linewidth=1.2, alpha=0.6,
          label=f"Power law: ${a1:.2f} \\cdot N^{{{b1:.2f}}}$",
          zorder=3)

# Log-corrected fit (purple dashed, for comparison)
rank_fit2 = c2 * n_ref * np.log(n_ref)**a2_slope
ax_a.plot(n_ref, rank_fit2, "--", color="#9467bd", linewidth=1.2, alpha=0.6,
          label=f"Log-linear: ${c2:.2f} \\cdot N \\cdot \\log^{{{a2_slope:.2f}}}(N)$",
          zorder=3)

ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlabel("Qubit Count $N$")
ax_a.set_ylabel("GF(2) Rank / T-count")
ax_a.set_title("(a) QFT GF(2) Rank Scaling (Extended)")
ax_a.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax_a.legend(loc="upper left", frameon=True, framealpha=0.9,
            edgecolor="black", fancybox=False, fontsize=8.5)

# ── Panel (b): Residuals ────────────────────────────────────────────────────

# Residuals = log(rank) - log(predicted)
res_powerlaw = log_ranks - np.log(a1 * ns**b1)
res_loglin = log_ranks - np.log(c2 * ns * np.log(ns)**a2_slope)
res_exact = log_ranks - np.log(ns - 1)

ax_b.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

ax_b.plot(ns, res_powerlaw, "o--", color="#d62728", markersize=6, linewidth=1.0,
          label=f"Power law ($R^2$={r2_powerlaw:.4f})", zorder=3)
ax_b.plot(ns, res_loglin, "s--", color="#9467bd", markersize=6, linewidth=1.0,
          label=f"Log-linear ($R^2$={r2_loglin:.4f})", zorder=3)
ax_b.plot(ns, res_exact, "^-", color="#2ca02c", markersize=7, linewidth=1.5,
          label="$N - 1$ (exact, zero residuals)", zorder=4)

ax_b.set_xscale("log")
ax_b.set_xlabel("Qubit Count $N$")
ax_b.set_ylabel(r"Residual: $\log(\mathrm{rank}) - \log(\hat{\mathrm{rank}})$")
ax_b.set_title("(b) Model Residuals (log scale)")
ax_b.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax_b.legend(loc="best", frameon=True, framealpha=0.9,
            edgecolor="black", fancybox=False, fontsize=9)

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = paper_dir / "figure5.pdf"
out_png = paper_dir / "figure5.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# ── Print fit parameters for paper caption verification ──────────────────────
print()
print("="*70)
print("FIT PARAMETERS (for paper caption verification)")
print("="*70)
print(f"Power law fit:    rank = {a1:.2f} · N^{{{b1:.2f}}}")
print(f"Log-linear fit:   rank = {c2:.2f} · N · log(N)^{{{a2_slope:.2f}}}")
print(f"Exact model:      rank = N - 1")
print()
print("Paper caption should say:")
print(f'  "The power-law fit ${a1:.2f} \\cdot N^{{{b1:.2f}}}$ and log-corrected')
print(f'   fit are shown for comparison; both deviate systematically."')
print("="*70)

# plt.show()  # uncomment for interactive viewing
