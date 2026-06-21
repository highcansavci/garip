"""Replicate the stale-reference collapse on Leduc hold'em (exact exploitability).

A second environment for the staleness analysis ([docs/staleness_analysis.md]). Tabular
self-play on the Leduc tree: each infoset's strategy is updated by mirror ascent on its
exact counterfactual value (`leduc.counterfactual_values`), regularized toward a magnet —
GARIP's running-average vs R-NaD's periodic snapshot. The metric is *exact* exploitability
(no best-response-budget caveats). If R-NaD's high-α / large-K region blows up while GARIP
stays low, the stale-reference collapse is not Coin-Game-specific.

Usage:
    python experiments/run_leduc_collapse.py [--steps 1500] [--seeds 5]
Writes results/leduc_collapse.png and results/leduc_collapse.csv.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GARIP_ETA = [0.1, 0.2, 0.3, 0.5]
GARIP_BETA = [0.005, 0.01, 0.02, 0.05]
RNAD_ALPHA = [0.25, 0.5, 1.0, 2.0]
RNAD_K = [100, 200, 400, 800]
MMD_ETA = 0.5  # fixed step for the closed-form (MMD/R-NaD) updates


def _normalize(s):
    return s / np.maximum(s.sum(axis=1, keepdims=True), 1e-12)


def _worker(job):
    method, p1, p2, seed, steps = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import jax
    from garip import leduc

    G = leduc.GAME
    legal = [np.asarray(G.legal0), np.asarray(G.legal1)]
    rng = np.random.default_rng(seed)
    s = [_normalize(np.exp(rng.normal(size=legal[p].shape)) * legal[p]) for p in (0, 1)]
    avg = [s[0].copy(), s[1].copy()]      # running-average iterate (GARIP magnet + the reported output)
    ssum = [s[0].copy(), s[1].copy()]     # cumulative for the time-average strategy
    snap = [s[0].copy(), s[1].copy()]
    q_prev = list(leduc.counterfactual_values(G, s[0], s[1]))

    for t in range(steps):
        q = list(leduc.counterfactual_values(G, s[0], s[1]))
        if method == "garip":
            eta, beta = p1, p2
            for p in (0, 1):
                h = _normalize(s[p] * np.exp(eta * (2 * q[p] - q_prev[p])) * legal[p])
                s[p] = (1 - beta) * h + beta * avg[p]
            q_prev = q
            for p in (0, 1):
                avg[p] = avg[p] + (s[p] - avg[p]) / (t + 2.0)
        else:  # rnad / mmd: closed-form KL-proximal update toward a magnet
            alpha, K = p1, int(p2)
            c = 1.0 / (1.0 + MMD_ETA * alpha)
            for p in (0, 1):
                mag = (np.ones_like(s[p]) * legal[p] / legal[p].sum(1, keepdims=True)
                       if method == "mmd" else snap[p])
                s[p] = _normalize((s[p] ** c) * (mag ** (1.0 - c))
                                  * np.exp(MMD_ETA * c * q[p]) * legal[p])
            if method == "rnad" and (t + 1) % K == 0:
                snap = [s[0].copy(), s[1].copy()]
        for p in (0, 1):
            ssum[p] += s[p]

    # Report the exact exploitability of the time-average strategy (the standard output
    # of no-regret / CFR-style dynamics). Staleness collapse corrupts even the average.
    avg_strat = [_normalize(ssum[p]) for p in (0, 1)]
    e = leduc.exploitability(G, avg_strat[0], avg_strat[1])
    print(f"{method} p1={p1} p2={p2} seed={seed}: expl={e:.3f}", flush=True)
    return method, p1, p2, e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = []
    for a in GARIP_ETA:
        for b in GARIP_BETA:
            for s in range(args.seeds):
                jobs.append(("garip", a, b, s, args.steps))
    for a in RNAD_ALPHA:
        for K in RNAD_K:
            for s in range(args.seeds):
                jobs.append(("rnad", a, K, s, args.steps))

    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)

    def grid(method, p1s, p2s):
        g = np.zeros((len(p1s), len(p2s)))
        for i, a in enumerate(p1s):
            for j, b in enumerate(p2s):
                g[i, j] = np.mean([e for (m, x, y, e) in raw if m == method and x == a and y == b])
        return g

    gg = grid("garip", GARIP_ETA, GARIP_BETA)
    rg = grid("rnad", RNAD_ALPHA, RNAD_K)

    def stats(method):
        f = np.array([e for (m, x, y, e) in raw if m == method])
        return dict(median=np.median(f), iqr=np.percentile(f, 75) - np.percentile(f, 25),
                    worst=f.max(), frac_good=float(np.mean(f < 0.1)),
                    frac_collapse=float(np.mean(f > 1.0)))

    gs, rs = stats("garip"), stats("rnad")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    norm = mcolors.LogNorm(vmin=1e-3, vmax=max(gg.max(), rg.max()))
    for ax, g, xt, yt, xl, yl, title in [
        (axes[0], gg, GARIP_BETA, GARIP_ETA, "β (anchor)", "η (step)", "GARIP (ours)"),
        (axes[1], rg, RNAD_K, RNAD_ALPHA, "K (reset period)", "α (reg)", "R-NaD")]:
        im = ax.imshow(np.maximum(g, 1e-3), norm=norm, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt)
        ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        for (yy, xx), v in np.ndenumerate(g):
            ax.text(xx, yy, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.1 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Leduc average-strategy exploitability vs. hyperparameters (exact; lower = better)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "leduc_collapse.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    with open(os.path.join(RESULTS_DIR, "leduc_collapse.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "median", "iqr", "worst", "frac_good(<0.1)", "frac_collapse(>1)"])
        for name, st in (("GARIP", gs), ("R-NaD", rs)):
            w.writerow([name, f"{st['median']:.3f}", f"{st['iqr']:.3f}", f"{st['worst']:.3f}",
                        f"{st['frac_good']:.2f}", f"{st['frac_collapse']:.2f}"])
    print(f"wrote {os.path.join(RESULTS_DIR, 'leduc_collapse.csv')}")

    print("\n=== Leduc exact-exploitability robustness (lower = better) ===")
    print(f"  {'method':8s} {'median':>8s} {'IQR':>8s} {'worst':>8s} {'good<0.1':>9s} {'collapse>1':>11s}")
    for name, st in (("GARIP", gs), ("R-NaD", rs)):
        print(f"  {name:8s} {st['median']:>8.3f} {st['iqr']:>8.3f} {st['worst']:>8.3f} "
              f"{st['frac_good']:>9.2f} {st['frac_collapse']:>11.2f}")


if __name__ == "__main__":
    main()
