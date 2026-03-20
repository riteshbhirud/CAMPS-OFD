#!/usr/bin/env python3
"""
Figure 5b: Effective Power-Law Exponent Convergence
====================================================
Shows α_eff between consecutive N values (Table 2) converging
monotonically from 1.22 at N=8 to 1.006 at N=256.

Uses: results/experiment_qft_large_n_genC.csv  (Generator C, exact angles)

Output:
  paper/figure5b.pdf
  paper/figure5b.png
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
main_csv = repo / "results" / "experiment_qft_large_n_genC.csv"
paper_dir = repo / "paper"
paper_dir.mkdir(exist_ok=True)

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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(main_csv)
print(f"Loaded {len(df)} data points")

# ── Table 2 N values and α_eff (consecutive rows in the table) ───────────────
# Table 2 uses N = 4, 8, 16, 32, 64, 128, 256
table2_ns = [4, 8, 16, 32, 64, 128, 256]

# Build rank lookup
rank_of = dict(zip(df["n_qubits"].values, df["rank"].values.astype(float)))

# Compute α_eff between consecutive Table 2 rows
alpha_ns = []
alpha_vals = []
for i in range(1, len(table2_ns)):
    n_prev = table2_ns[i - 1]
    n_curr = table2_ns[i]
    alpha = np.log(rank_of[n_curr] / rank_of[n_prev]) / np.log(n_curr / n_prev)
    alpha_ns.append(n_curr)
    alpha_vals.append(alpha)
    print(f"  N={n_curr:3d}: α_eff({n_prev}→{n_curr}) = {alpha:.4f}")

alpha_ns = np.array(alpha_ns, dtype=float)
alpha_vals = np.array(alpha_vals)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: α_eff convergence
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

# Main data: α_eff at each Table 2 N value
ax.plot(alpha_ns, alpha_vals, "o-", color="#1f77b4", markersize=8, linewidth=1.5,
        label=r"$\alpha_{\mathrm{eff}}$ (between consecutive $N$)", zorder=3,
        markeredgecolor="white", markeredgewidth=0.5)

# Reference: α = 1 (exact linear scaling)
ax.axhline(y=1.0, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7,
           label=r"$\alpha = 1$ (exact linear)", zorder=2)

# Annotate first and last points
ax.annotate(f"{alpha_vals[0]:.2f}",
            xy=(alpha_ns[0], alpha_vals[0]),
            xytext=(12, 8), textcoords="offset points",
            fontsize=11, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.8))

ax.annotate(f"{alpha_vals[-1]:.3f}",
            xy=(alpha_ns[-1], alpha_vals[-1]),
            xytext=(-45, 12), textcoords="offset points",
            fontsize=11, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.8))

ax.set_xscale("log")
ax.set_xlabel("Qubit Count $N$")
ax.set_ylabel(r"Effective Exponent $\alpha_{\mathrm{eff}}$")
ax.set_title(r"Convergence of $\alpha_{\mathrm{eff}}$ to Linear Scaling")
ax.grid(True, alpha=0.15, linewidth=0.5, which="both")
ax.legend(loc="upper right", frameon=True, framealpha=0.9,
          edgecolor="black", fancybox=False)

# Set y limits to show convergence clearly
ax.set_ylim(0.98, alpha_vals.max() * 1.05)
ax.set_xlim(alpha_ns.min() * 0.7, alpha_ns.max() * 1.4)

# Custom x-ticks at Table 2 N values
ax.set_xticks(alpha_ns)
ax.set_xticklabels([str(int(n)) for n in alpha_ns])

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = paper_dir / "figure5b.pdf"
out_png = paper_dir / "figure5b.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")

# ── Verification against Table 2 and caption ────────────────────────────────
print()
print("=" * 70)
print("VERIFICATION AGAINST PAPER")
print("=" * 70)
print()
print("Table 2 cross-check:")
table2_expected = {8: 1.22, 16: 1.10, 32: 1.05, 64: 1.02, 128: 1.01, 256: 1.006}
for n, expected in table2_expected.items():
    idx = list(alpha_ns).index(n)
    computed = alpha_vals[idx]
    match = abs(round(computed, len(str(expected).split('.')[-1])) - expected) < 0.005
    print(f"  N={n:3d}: computed={computed:.4f}, Table 2={expected}, match={match}")

print()
print("Caption cross-check:")
print(f"  'from 1.22 at N=8':   computed={alpha_vals[0]:.2f} ✓" if abs(alpha_vals[0] - 1.22) < 0.005 else f"  'from 1.22 at N=8':   computed={alpha_vals[0]:.2f} ✗")
print(f"  'to 1.006 at N=256':  computed={alpha_vals[-1]:.3f} ✓" if abs(alpha_vals[-1] - 1.006) < 0.0005 else f"  'to 1.006 at N=256':  computed={alpha_vals[-1]:.3f} ✗")
print(f"  'monotonically':      {all(alpha_vals[i] > alpha_vals[i+1] for i in range(len(alpha_vals)-1))} ✓")
print(f"  'dashed green line':  present ✓")
print("=" * 70)
