"""Deep-RL self-play robustness across multiple pgx board games (generic over `--games`).

Same opponent x magnet ablation and best-response-win-rate metric as run_connect4.py, run
on several structurally different turn-based zero-sum games to show the moving-reference
robustness finding is not game-specific. One worker pool over all (game, method, seed) jobs.

Writes results/{game}_robustness.csv and a combined results/board_robustness.png.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

METHODS = [
    ("naive", "current", "none", 0.0, 200),
    ("fictitious", "average", "none", 0.0, 200),
    ("mmd", "current", "fixed", 0.5, 200),
    ("rnad", "current", "periodic", 0.5, 200),
    ("garip", "average", "moving", 0.5, 200),
]
COLORS = {"naive": "#7f7f7f", "fictitious": "#1f77b4", "mmd": "#2ca02c",
          "rnad": "#9467bd", "garip": "#d62728"}
LABELS = {"naive": "Naive", "fictitious": "Fictitious", "mmd": "MMD (fixed)",
          "rnad": "R-NaD (snapshot)", "garip": "GARIP (moving)"}


def _worker(job):
    game, name, opp, mag, lam, reset_every, seed, sp_updates, br_updates, mem_frac = job
    # GPU mode: share one GPU across a few memory-capped processes (no preallocation).
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_frac:.3f}"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import jax
    from garip.rl.pgx_selfplay import make_pgx_trainer, best_response_winrate, PgxConfig
    cfg = PgxConfig(num_envs=128, rollout_len=32)
    net, init, train_chunk, od, na = make_pgx_trainer(game, cfg, opp, mag, lam, reset_every)
    carry = init(jax.random.PRNGKey(seed))
    t0 = time.time()
    carry, _ = train_chunk(carry, sp_updates)
    br = best_response_winrate(game, cfg, carry[0], jax.random.PRNGKey(10_000 + seed),
                               br_updates=br_updates)
    print(f"[{game}] {name} seed {seed}: BR={br:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return game, name, seed, float(br)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=str, default="othello,hex,animal_shogi")
    parser.add_argument("--sp-updates", type=int, default=1000)
    parser.add_argument("--br-updates", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)  # processes sharing the GPU
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    games = args.games.split(",")
    mem_frac = min(0.85 / args.workers, 0.3)  # split GPU memory across workers

    jobs = [(g, n, o, m, l, r, s, args.sp_updates, args.br_updates, mem_frac)
            for g in games for (n, o, m, l, r) in METHODS for s in range(args.seeds)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)
    print(f"all {len(jobs)} jobs done in {time.time()-t0:.0f}s")

    fig, axes = plt.subplots(1, len(games), figsize=(4.6 * len(games), 4.6), squeeze=False)
    for gi, g in enumerate(games):
        stats = {}
        for name, *_ in METHODS:
            vals = np.array([b for (gg, n, s, b) in raw if gg == g and n == name])
            stats[name] = (vals.mean(), vals.std(), len(vals))
        with open(os.path.join(RESULTS_DIR, f"{g}_robustness.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "br_winrate", "std", "n"])
            for k in sorted(stats, key=lambda k: stats[k][0]):
                w.writerow([k, f"{stats[k][0]:.4f}", f"{stats[k][1]:.4f}", stats[k][2]])
        order = sorted(stats, key=lambda k: stats[k][0])
        ax = axes[0][gi]
        ax.bar(range(len(order)), [stats[k][0] for k in order],
               yerr=[stats[k][1] for k in order],
               color=[COLORS[k] for k in order], capsize=3)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([LABELS[k] for k in order], rotation=30, ha="right", fontsize=8)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_title(g); ax.grid(True, axis="y", alpha=0.3)
        if gi == 0:
            ax.set_ylabel("best-response win-rate\n(lower = more robust)")
        print(f"\n=== {g}: best-response win-rate (lower = more robust) ===")
        for k in order:
            print(f"  {LABELS[k]:18s} {stats[k][0]:.3f} +/- {stats[k][1]:.3f}")
    fig.suptitle("Self-play robustness across board games (best-response win-rate, lower = more robust)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "board_robustness.png"), dpi=130)
    print(f"\nwrote {os.path.join(RESULTS_DIR, 'board_robustness.png')}")


if __name__ == "__main__":
    main()
