#!/usr/bin/env python3
"""
Large-N QFT Rank Scaling — Publication Figures
================================================
Part A: Log-log rank vs N with all three model fits.
Part B: Residuals for each model.
Part C: Effective exponent α_eff vs N.

Uses: results/experiment_qft_large_n.csv
      results/experiment_qft_large_n_models.csv
      results/experiment_qft_large_n_exponents.csv

Output:
  experiment_qft_large_n_fits.png / .pdf       (log-log + fits)
  experiment_qft_large_n_residuals.png / .pdf   (residuals)
  experiment_qft_large_n_exponents.png / .pdf   (effective exponents)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
repo = Path(__file__).resolve().parent.parent
results_csv = repo / "results" / "experiment_qft_large_n.csv"
models_csv = repo / "results" / "experiment_qft_large_n_models.csv"
exponents_csv = repo / "results" / "experiment_qft_large_n_exponents.csv"

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

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(results_csv)
print(f"Loaded {len(df)} data points")
print(f"N range: {df['n_qubits'].min()} to {df['n_qubits'].max()}")
print()

ns = df["n_qubits"].values.astype(float)
ranks = df["rank"].values.astype(float)
ts = df["t"].values.astype(float)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Two-panel — (a) Log-log with fits  (b) Residuals
# ═══════════════════════════════════════════════════════════════════════════════

fig1, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={"width_ratios": [1, 1]})

# ── Panel (a): Log-log rank vs N with model fits ────────────────────────────

# Data points
ax_a.plot(ns, ranks, "o", color="#1f77b4", markersize=8,
          label="Data", zorder=4, markeredgecolor="white", markeredgewidth=0.5)

# t(N) reference (T-gate count)
ax_a.plot(ns, ts, "s--", color="#ff7f0e", markersize=5,
          linewidth=1.0, alpha=0.6, label="$t(N)$ (T-gate count)", zorder=2)

# Reference: rank = N line
n_ref = np.linspace(ns.min() * 0.8, ns.max() * 1.2, 200)
ax_a.plot(n_ref, n_ref, ":", color="gray", linewidth=1.0, alpha=0.5,
          label="rank $= N$", zorder=1)

# Reference: rank = N - 2 line
ax_a.plot(n_ref, n_ref - 2, "-.", color="#2ca02c", linewidth=1.5, alpha=0.7,
          label="rank $= N - 2$ (exact)", zorder=2)

# Fit Model 1: Pure power law
log_ns = np.log(ns)
log_ranks = np.log(ranks)
slope, intercept = np.polyfit(log_ns, log_ranks, 1)
a1, b1 = np.exp(intercept), slope
rank_fit1 = a1 * n_ref**b1
ax_a.plot(n_ref, rank_fit1, "--", color="#d62728", linewidth=1.2, alpha=0.8,
          label=f"Power law: ${{a}} \\cdot N^{{{b1:.2f}}}$ ($a$={a1:.3f})",
          zorder=3)

# Fit Model 2: Log-corrected linear
# log(rank) = log(c) + log(N) + a·log(log(N))
y_adj = log_ranks - log_ns
loglog_ns = np.log(log_ns)
a2_slope, a2_intercept = np.polyfit(loglog_ns, y_adj, 1)
c2 = np.exp(a2_intercept)
rank_fit2 = c2 * n_ref * np.log(n_ref)**a2_slope
ax_a.plot(n_ref, rank_fit2, "--", color="#9467bd", linewidth=1.2, alpha=0.8,
          label=f"Log-linear: $c \\cdot N \\cdot \\log^{{{a2_slope:.2f}}}(N)$ ($c$={c2:.3f})",
          zorder=3)

# Compute R² for each model
def r_squared_log(actual, predicted):
    log_a = np.log(actual)
    log_p = np.log(predicted)
    ss_res = np.sum((log_a - log_p)**2)
    ss_tot = np.sum((log_a - log_a.mean())**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

r2_1 = r_squared_log(ranks, a1 * ns**b1)
r2_2 = r_squared_log(ranks, c2 * ns * np.log(ns)**a2_slope)

ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlabel("Qubit Count $N$")
ax_a.set_ylabel("GF(2) Rank / T-count")
ax_a.set_title("(a) QFT GF(2) Rank Scaling (Extended)")
ax_a.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax_a.legend(loc="upper left", frameon=True, framealpha=0.9,
            edgecolor="black", fancybox=False, fontsize=8.5)

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

# ── Panel (b): Residuals ────────────────────────────────────────────────────

# Residuals = log(rank) - log(predicted)
res_powerlaw = log_ranks - np.log(a1 * ns**b1)
res_loglin = log_ranks - np.log(c2 * ns * np.log(ns)**a2_slope)
res_exact = log_ranks - np.log(ns - 2)

ax_b.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

ax_b.plot(ns, res_powerlaw, "o--", color="#d62728", markersize=6, linewidth=1.0,
          label=f"Power law ($R^2$={r2_1:.4f})", zorder=3)
ax_b.plot(ns, res_loglin, "s--", color="#9467bd", markersize=6, linewidth=1.0,
          label=f"Log-linear ($R^2$={r2_2:.4f})", zorder=3)
ax_b.plot(ns, res_exact, "^-", color="#2ca02c", markersize=6, linewidth=1.2,
          label="$N - 2$ (exact)", zorder=4)

ax_b.set_xscale("log")
ax_b.set_xlabel("Qubit Count $N$")
ax_b.set_ylabel(r"Residual: $\log(\mathrm{rank}) - \log(\hat{\mathrm{rank}})$")
ax_b.set_title("(b) Model Residuals (log scale)")
ax_b.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax_b.legend(loc="best", frameon=True, framealpha=0.9,
            edgecolor="black", fancybox=False, fontsize=9)

fig1.tight_layout()
out1_png = repo / "experiment_qft_large_n_fits.png"
out1_pdf = repo / "experiment_qft_large_n_fits.pdf"
fig1.savefig(out1_png)
fig1.savefig(out1_pdf)
print(f"Saved: {out1_png}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Effective exponent α_eff vs N
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Compute effective exponents
alpha_effs = []
n_mids = []
for i in range(len(ns) - 1):
    if ranks[i] > 0 and ranks[i+1] > 0:
        alpha = np.log(ranks[i+1] / ranks[i]) / np.log(ns[i+1] / ns[i])
        alpha_effs.append(alpha)
        n_mids.append(np.sqrt(ns[i] * ns[i+1]))  # geometric mean

ax2.plot(n_mids, alpha_effs, "o-", color="#1f77b4", markersize=8, linewidth=1.5,
         markeredgecolor="white", markeredgewidth=0.5, zorder=3)

# Reference lines
ax2.axhline(y=1.0, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7,
            label=r"$\alpha = 1$ (linear: rank $\propto N$)", zorder=1)

# Power law fit exponent
ax2.axhline(y=b1, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.7,
            label=f"$\\alpha = {b1:.2f}$ (power law fit)", zorder=1)

ax2.set_xscale("log")
ax2.set_xlabel("Qubit Count $N$ (geometric mean of consecutive pair)")
ax2.set_ylabel(r"Effective Exponent $\alpha_{\mathrm{eff}}$")
ax2.set_title("Effective Power-Law Exponent vs System Size")
ax2.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax2.legend(loc="best", frameon=True, framealpha=0.9,
           edgecolor="black", fancybox=False)

fig2.tight_layout()
out2_png = repo / "experiment_qft_large_n_exponents.png"
out2_pdf = repo / "experiment_qft_large_n_exponents.pdf"
fig2.savefig(out2_png)
fig2.savefig(out2_pdf)
print(f"Saved: {out2_png}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("QFT RANK SCALING SUMMARY")
print("="*80)
print(f"{'N':>6s} {'t':>10s} {'rank':>8s} {'N-2':>6s} {'rank==N-2':>10s}")
print("-"*45)
for i in range(len(ns)):
    n = int(ns[i])
    t = int(ts[i])
    r = int(ranks[i])
    match = "YES" if r == n - 2 else "NO"
    print(f"{n:6d} {t:10d} {r:8d} {n-2:6d} {match:>10s}")
print("="*80)

# Check if rank = N - 2 for all
all_match = all(int(ranks[i]) == int(ns[i]) - 2 for i in range(len(ns)))
if all_match:
    print("\nRESULT: rank = N - 2 holds EXACTLY for all tested N.")
    print("The QFT rank scaling is LINEAR (not superlinear).")
    print("Previous power-law fit α ≈ 1.20 was an artifact of small N range.")
else:
    print("\nRESULT: rank = N - 2 does NOT hold for all N.")
    # Find where it breaks
    for i in range(len(ns)):
        n = int(ns[i])
        r = int(ranks[i])
        if r != n - 2:
            print(f"  Breaks at N={n}: rank={r}, expected {n-2}")

print("="*80)

plt.show()
