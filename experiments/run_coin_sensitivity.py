"""Deep-RL hyperparameter sensitivity: GARIP (λ, Polyak ρ) vs R-NaD (λ, reset K).

Checks whether GARIP's matrix-game hyperparameter robustness transfers to the deep Coin
Game. Each (method, hyperparameter, seed) job trains self-play for `updates` steps and
reports the final best-response exploit return (lower = more robust). We then compare the
*distribution* of that score across each method's hyperparameter grid.

Jobs run in parallel via `multiprocessing` (single-threaded XLA per worker).

Usage:
    python experiments/run_coin_sensitivity.py [--updates 1200] [--seeds 2] [--workers 12]
Writes results/coin_sensitivity.png and results/coin_sensitivity.csv.
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

GARIP_LAM = [0.25, 0.5, 1.0, 2.0]
# EMA lag ~= 1/rho. To make the comparison fair the rho grid must reach into
# GARIP's *own* stale regime, symmetric to R-NaD's large-K stress point: rho=0.00125
# gives lag ~800, matching K=800. If GARIP stays robust here while R-NaD collapses,
# the robustness gap is a real property, not a grid artifact.
GARIP_RHO = [0.00125, 0.0025, 0.005, 0.01, 0.02, 0.05]
RNAD_LAM = [0.25, 0.5, 1.0, 2.0]
RNAD_K = [100, 200, 400, 800]  # large K probes R-NaD's stale-reference collapse


def _worker(job):
    method, lam, hp, seed, updates, br_updates = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import dataclasses
    import jax
    from garip.envs import CoinGame
    from garip.rl.ppo_selfplay import PPOConfig, make_selfplay_trainer
    from garip.rl.exploitability import approx_exploitability

    env = CoinGame(episode_length=16, zero_sum=True)
    if method == "garip":
        cfg = dataclasses.replace(PPOConfig(), polyak=hp)
        _, init, train_chunk = make_selfplay_trainer(env, cfg, "average", "moving", lam)
    else:  # rnad
        cfg = PPOConfig()
        _, init, train_chunk = make_selfplay_trainer(env, cfg, "current", "periodic", lam, int(hp))
    carry = init(jax.random.PRNGKey(seed))
    carry, _ = train_chunk(carry, updates)
    e = approx_exploitability(env, cfg, carry[0], jax.random.PRNGKey(10_000 + seed),
                              br_updates=br_updates)
    print(f"{method} lam={lam} hp={hp} seed={seed}: {e:+.2f}", flush=True)
    return method, lam, hp, e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--br-updates", type=int, default=250)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = []
    for lam in GARIP_LAM:
        for rho in GARIP_RHO:
            for s in range(args.seeds):
                jobs.append(("garip", lam, rho, s, args.updates, args.br_updates))
    for lam in RNAD_LAM:
        for K in RNAD_K:
            for s in range(args.seeds):
                jobs.append(("rnad", lam, K, s, args.updates, args.br_updates))

    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)

    def grid(method, lams, hps):
        g = np.zeros((len(lams), len(hps)))
        for i, lam in enumerate(lams):
            for j, hp in enumerate(hps):
                vals = [e for (m, l, h, e) in raw if m == method and l == lam and h == hp]
                g[i, j] = np.mean(vals)
        return g

    gg = grid("garip", GARIP_LAM, GARIP_RHO)
    rg = grid("rnad", RNAD_LAM, RNAD_K)

    def wilson(k, n, z=1.96):
        # 95% Wilson score interval for a binomial proportion k/n.
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def stats(method):
        # Over ALL individual (config, seed) runs -- so a single collapsing seed shows up.
        f = np.array([e for (m, l, h, e) in raw if m == method])
        n = len(f)
        n_collapse = int(np.sum(f > 0.0))
        lo, hi = wilson(n_collapse, n)
        return dict(median=np.median(f), iqr=np.percentile(f, 75) - np.percentile(f, 25),
                    worst=f.max(), frac_robust=float(np.mean(f < -5.0)),
                    frac_collapse=n_collapse / n if n else 0.0,
                    n=n, n_collapse=n_collapse, collapse_ci=(lo, hi))

    gs, rs = stats("garip"), stats("rnad")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vmin = min(gg.min(), rg.min())
    vmax = max(gg.max(), rg.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for ax, g, xt, yt, xl, title in [
        (axes[0], gg, GARIP_RHO, GARIP_LAM, "Polyak ρ (avg rate)", "GARIP (ours)"),
        (axes[1], rg, RNAD_K, RNAD_LAM, "K (reset period)", "R-NaD")]:
        im = ax.imshow(g, norm=norm, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt)
        ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
        ax.set_xlabel(xl); ax.set_ylabel("λ (KL weight)"); ax.set_title(title)
        for (yy, xx), v in np.ndenumerate(g):
            ax.text(xx, yy, f"{v:+.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Coin Game exploit return vs. hyperparameters (lower/greener = more robust)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "coin_sensitivity.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    with open(os.path.join(RESULTS_DIR, "coin_sensitivity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "median", "iqr", "worst", "frac_robust(<-5)",
                    "frac_collapse(>0)", "n_runs", "n_collapse", "collapse_ci95_lo", "collapse_ci95_hi"])
        for name, s in (("GARIP", gs), ("R-NaD", rs)):
            w.writerow([name, f"{s['median']:.3f}", f"{s['iqr']:.3f}", f"{s['worst']:.3f}",
                        f"{s['frac_robust']:.2f}", f"{s['frac_collapse']:.3f}",
                        s['n'], s['n_collapse'],
                        f"{s['collapse_ci'][0]:.3f}", f"{s['collapse_ci'][1]:.3f}"])

    print("\n=== Deep-RL hyperparameter robustness over all runs (lower = more robust) ===")
    print(f"  {'method':8s} {'median':>8s} {'IQR':>8s} {'worst':>8s} {'frac<-5':>8s} "
          f"{'collapse>0':>11s} {'collapse 95% CI':>22s}")
    for name, s in (("GARIP", gs), ("R-NaD", rs)):
        lo, hi = s['collapse_ci']
        print(f"  {name:8s} {s['median']:>8.2f} {s['iqr']:>8.2f} {s['worst']:>8.2f} "
              f"{s['frac_robust']:>8.2f} {s['frac_collapse']:>10.3f} "
              f"({s['n_collapse']}/{s['n']}) [{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
