"""Deep CFR on Leduc: full external-sampling Monte-Carlo + a deterministic variant.

Runs both Deep CFR modes and plots their exact policy exploitability against the prior
results (tabular CFR, CGSP-quantal, fictitious play). See the README for the honest
finding: at this compact prototype scale the neural advantage approximation plateaus
above tabular CFR (and above CGSP) -- the bottleneck is the advantage network, not the
policy distillation (which is exact).

Usage:
    python experiments/run_deep_cfr.py [--sampling-iters 200] [--det-iters 500]
Writes results/deep_cfr_curves.png and results/deep_cfr_exploitability.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garip import deep_cfr, leduc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Prior results on Leduc (from experiments/run_leduc.py), drawn as reference lines.
CGSP_QUANTAL = 0.18
FICTITIOUS_PLAY = 0.21


def cfr_curve(iterations, eval_every):
    """Tabular CFR exploitability of the running average strategy (gold standard)."""
    import numpy as np
    legal = [np.asarray(leduc.GAME.legal0), np.asarray(leduc.GAME.legal1)]
    n = [leduc.GAME.n0, leduc.GAME.n1]
    R = [np.zeros((n[p], leduc.NUM_ACTIONS)) for p in (0, 1)]
    S = [np.zeros((n[p], leduc.NUM_ACTIONS)) for p in (0, 1)]
    its, expl = [], []
    for it in range(1, iterations + 1):
        s = [leduc.regret_matching(R[p], legal[p]) for p in (0, 1)]
        for p in (0, 1):
            S[p] += s[p]
        r0, r1 = leduc.counterfactual_regrets(leduc.GAME, s[0], s[1])
        R[0] += r0
        R[1] += r1
        if it % eval_every == 0:
            a = [S[p] / S[p].sum(1, keepdims=True) for p in (0, 1)]
            its.append(it)
            expl.append(leduc.exploitability(leduc.GAME, a[0], a[1]))
    return np.array(its), np.array(expl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-iters", type=int, default=200)
    parser.add_argument("--det-iters", type=int, default=500)
    parser.add_argument("--cfr-iters", type=int, default=400)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    runs = {}

    t0 = time.time()
    it, curve, _ = deep_cfr.run_deep_cfr(
        iters=args.sampling_iters, sampling=True, traversals=50,
        adv_steps=200, policy_steps=2000, hidden=64, eval_every=25, seed=0, verbose=True)
    runs["dcfr_sampling"] = (it, curve)
    print(f"Deep CFR (sampling) final = {curve[-1]:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    it, curve, _ = deep_cfr.run_deep_cfr(
        iters=args.det_iters, sampling=False, warm_start=True, adv_steps=120,
        policy_steps=2000, hidden=64, eval_every=25, seed=0, verbose=True)
    runs["dcfr_det"] = (it, curve)
    print(f"Deep CFR (deterministic) final = {curve[-1]:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    runs["cfr"] = cfr_curve(args.cfr_iters, 25)
    print(f"tabular CFR final = {runs['cfr'][1][-1]:.4f} ({time.time()-t0:.0f}s)")

    # Plot.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        "dcfr_sampling": ("#d62728", "Deep CFR (external-sampling MC)"),
        "dcfr_det": ("#ff7f0e", "Deep CFR (deterministic, exact regrets)"),
        "cfr": ("#2ca02c", "Tabular CFR (reference)"),
    }
    for name, (it, curve) in runs.items():
        c, label = styles[name]
        ax.plot(it, curve, color=c, lw=2, label=label)
    ax.axhline(CGSP_QUANTAL, ls="--", color="#1f77b4", lw=1.5, label=f"CGSP-quantal ({CGSP_QUANTAL})")
    ax.axhline(FICTITIOUS_PLAY, ls=":", color="#9467bd", lw=1.5, label=f"Fictitious play ({FICTITIOUS_PLAY})")
    ax.set_yscale("log")
    ax.set_xlabel("iteration (algorithm-specific)")
    ax.set_ylabel("exact policy exploitability (log)")
    ax.set_title("Deep CFR on Leduc hold'em vs prior results")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "deep_cfr_curves.png")
    fig.savefig(fig_path, dpi=130)
    print(f"wrote {fig_path}")

    path = os.path.join(RESULTS_DIR, "deep_cfr_exploitability.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "final_exploitability"])
        for name, (it, curve) in runs.items():
            w.writerow([name, f"{curve[-1]:.5f}"])
        w.writerow(["cgsp_quantal", CGSP_QUANTAL])
        w.writerow(["fictitious_play", FICTITIOUS_PLAY])
    print(f"wrote {path}")

    print("\n=== Leduc final exploitability ===")
    print(f"  Deep CFR (external-sampling MC)      {runs['dcfr_sampling'][1][-1]:.4f}")
    print(f"  Deep CFR (deterministic)            {runs['dcfr_det'][1][-1]:.4f}")
    print(f"  CGSP-quantal (prior)                {CGSP_QUANTAL:.4f}")
    print(f"  Fictitious play (prior)             {FICTITIOUS_PLAY:.4f}")
    print(f"  Tabular CFR (reference)             {runs['cfr'][1][-1]:.4f}")


if __name__ == "__main__":
    main()
