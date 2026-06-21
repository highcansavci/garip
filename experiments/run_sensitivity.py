"""Hyperparameter-sensitivity: GARIP (η, β) vs R-NaD (α, K) on matrix games.

The honest contribution candidate for GARIP is *not* beating R-NaD on performance (they
tie) but **matching it with less hyperparameter sensitivity** — GARIP has no reset-period
K, whereas R-NaD's last-iterate quality swings a lot with (α, K). This script sweeps each
method's 2-D hyperparameter grid, scores every config by mean last-iterate exploitability
over a panel of matrix games, and compares the *distribution* of scores across the grid
(tight & low = robust).

Usage:
    python experiments/run_sensitivity.py [--steps 8000]
Writes results/sensitivity_heatmaps.png and results/sensitivity_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garip.games import rps, matching_pennies, random_zero_sum
from garip import methods
from garip.train import run

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GARIP_ETA = [0.05, 0.1, 0.2, 0.3, 0.5]
GARIP_BETA = [0.005, 0.01, 0.02, 0.05, 0.1]
RNAD_ALPHA = [0.25, 0.5, 1.0, 2.0, 4.0]
RNAD_K = [100, 200, 300, 500, 1000]


def panel():
    return [rps(), matching_pennies(),
            random_zero_sum(jax.random.PRNGKey(0), 10, 10),
            random_zero_sum(jax.random.PRNGKey(1), 10, 10),
            random_zero_sum(jax.random.PRNGKey(2), 8, 8),
            random_zero_sum(jax.random.PRNGKey(3), 12, 6),
            random_zero_sum(jax.random.PRNGKey(4), 15, 15)]


def score(algo, games, steps):
    """Mean last-iterate exploitability over the panel (lower = better)."""
    vals = []
    for g in games:
        e, _, _ = run(algo, g, steps, jax.random.PRNGKey(0))
        vals.append(float(e[-1]))
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    games = panel()

    garip_grid = np.zeros((len(GARIP_ETA), len(GARIP_BETA)))
    for i, eta in enumerate(GARIP_ETA):
        for j, beta in enumerate(GARIP_BETA):
            garip_grid[i, j] = score(methods.garip(eta=eta, beta=beta), games, args.steps)
        print(f"GARIP eta={eta}: {np.round(garip_grid[i], 4)}", flush=True)

    rnad_grid = np.zeros((len(RNAD_ALPHA), len(RNAD_K)))
    for i, alpha in enumerate(RNAD_ALPHA):
        for j, K in enumerate(RNAD_K):
            rnad_grid[i, j] = score(methods.rnad(alpha=alpha, reset_every=K), games, args.steps)
        print(f"R-NaD alpha={alpha}: {np.round(rnad_grid[i], 4)}", flush=True)

    def stats(grid):
        f = grid.ravel()
        return dict(median=np.median(f), iqr=np.percentile(f, 75) - np.percentile(f, 25),
                    worst=f.max(), frac_conv=float(np.mean(f < 0.05)))

    gs, rs = stats(garip_grid), stats(rnad_grid)

    # Heatmaps (log color).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vmax = max(garip_grid.max(), rnad_grid.max())
    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=1e-4, vmax=vmax)
    for ax, grid, xt, yt, xl, yl, title in [
        (axes[0], garip_grid, GARIP_BETA, GARIP_ETA, "β (anchor)", "η (step)", "GARIP (ours)"),
        (axes[1], rnad_grid, RNAD_K, RNAD_ALPHA, "K (reset period)", "α (reg)", "R-NaD")]:
        im = ax.imshow(np.maximum(grid, 1e-4), norm=norm, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt)
        ax.set_yticks(range(len(yt)))
        ax.set_yticklabels(yt)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        for (yy, xx), v in np.ndenumerate(grid):
            ax.text(xx, yy, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.05 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Last-iterate exploitability vs. hyperparameters (lower = better; tighter = more robust)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "sensitivity_heatmaps.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    with open(os.path.join(RESULTS_DIR, "sensitivity_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "median", "iqr", "worst", "frac_configs_below_0.05"])
        w.writerow(["GARIP", f"{gs['median']:.4f}", f"{gs['iqr']:.4f}", f"{gs['worst']:.4f}", f"{gs['frac_conv']:.2f}"])
        w.writerow(["R-NaD", f"{rs['median']:.4f}", f"{rs['iqr']:.4f}", f"{rs['worst']:.4f}", f"{rs['frac_conv']:.2f}"])

    print("\n=== Hyperparameter robustness (over the 5×5 grid; lower median/IQR/worst = more robust) ===")
    print(f"  {'method':8s} {'median':>9s} {'IQR':>9s} {'worst':>9s} {'frac<0.05':>10s}")
    for name, s in (("GARIP", gs), ("R-NaD", rs)):
        print(f"  {name:8s} {s['median']:>9.4f} {s['iqr']:>9.4f} {s['worst']:>9.4f} {s['frac_conv']:>10.2f}")


if __name__ == "__main__":
    main()
