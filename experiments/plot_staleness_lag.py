"""Collapse rate vs effective reference lag, for GARIP (EMA) and R-NaD (snapshot).

Reads the deep-RL sensitivity sweep log and plots collapse rate against effective lag.
Key point: the EMA's lag is flat (peak = mean = 1/rho); the snapshot's is a sawtooth
(peak = K, mean = K/2). The peak-lag curves overlay (collapse tracks peak force), so at
matched MEAN lag R-NaD collapses earlier -- a fundamental consequence of the lag profile,
not a hyperparameter-default choice.

Usage: python experiments/plot_staleness_lag.py <sweep_log.txt>
"""
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for L in open(sys.argv[1]):
    m = re.match(r"(garip|rnad) lam=([\d.]+) hp=([\d.]+) seed=(\d+): ([+-][\d.]+)", L)
    if m:
        rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(5))))


def cr(meth, hp):
    v = [e for me, l, h, e in rows if me == meth and abs(h - hp) < 1e-9]
    return float(np.mean([e > 0 for e in v])) if v else None


RHO = [0.00125, 0.0025, 0.005, 0.01, 0.02, 0.05]
K = [100, 200, 400, 800]

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot([1 / r for r in RHO], [cr("garip", r) for r in RHO], "-o", color="#d62728",
        lw=2, label=r"GARIP (EMA, lag $\ell=1/\rho$)")
ax.plot([k for k in K], [cr("rnad", k) for k in K], "-s", color="#9467bd",
        lw=2, label=r"R-NaD (snapshot, PEAK lag $\ell=K$)")
ax.plot([k / 2 for k in K], [cr("rnad", k) for k in K], "--s", color="#9467bd",
        lw=2, alpha=0.55, label=r"R-NaD (snapshot, MEAN lag $\ell=K/2$)")
ax.axvline(100, color="gray", lw=0.8, ls=":")
ax.annotate("default $\\rho{=}0.01$ and $K{=}200$\nshare mean lag 100", xy=(100, 0.13),
            xytext=(115, 0.30), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xscale("log")
ax.set_xlabel(r"effective reference lag $\ell$ (steps)")
ax.set_ylabel(r"collapse rate (exploit return $>0$)")
ax.set_title("Collapse tracks PEAK lag: peak-lag curves overlay, so at matched\n"
             "MEAN lag the snapshot (peak $=2\\times$mean) collapses where the EMA does not")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/staleness_lag.png", dpi=130)
print("wrote results/staleness_lag.png")
