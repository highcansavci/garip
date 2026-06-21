"""Deep-RL self-play on Connect Four (pgx) -- a clean second deep environment.

Board games give the faithful metric the simultaneous-move envs lacked: strictly zero-sum,
symmetric, the action distribution *is* the strategy, and no inertness loophole (every game
ends with a result). Robustness = win-rate of a freshly trained best-responder against the
frozen self-play policy (lower = harder to exploit = more robust). Same opponent x magnet
ablation as the Coin Game runner.

Writes results/connect4_exploitability.csv + results/connect4_curves.png.
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
GAME = "connect_four"

METHODS = [
    ("naive", "current", "none", 0.0, 200),
    ("fictitious", "average", "none", 0.0, 200),
    ("mmd", "current", "fixed", 0.5, 200),
    ("rnad", "current", "periodic", 0.5, 200),
    ("garip", "average", "moving", 0.5, 200),
]
COLORS = {"naive": "#7f7f7f", "fictitious": "#1f77b4", "mmd": "#2ca02c",
          "rnad": "#9467bd", "garip": "#d62728"}
LABELS = {"naive": "Naive self-play", "fictitious": "Fictitious (avg opponent)",
          "mmd": "MMD (fixed magnet)", "rnad": "R-NaD (periodic snapshot)",
          "garip": "GARIP (ours, moving magnet)"}


def _worker(job):
    name, opp, mag, lam, reset_every, seed, sp_updates, br_updates = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import jax
    from garip.rl.pgx_selfplay import make_pgx_trainer, best_response_winrate, PgxConfig

    cfg = PgxConfig(num_envs=128, rollout_len=32)
    net, init, train_chunk, od, na = make_pgx_trainer(GAME, cfg, opp, mag, lam, reset_every)
    carry = init(jax.random.PRNGKey(seed))
    t0 = time.time()
    carry, _ = train_chunk(carry, sp_updates)
    br = best_response_winrate(GAME, cfg, carry[0], jax.random.PRNGKey(10_000 + seed),
                               br_updates=br_updates)
    print(f"{name} seed {seed}: BR win-rate={br:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return name, seed, float(br)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-updates", type=int, default=1000)
    parser.add_argument("--br-updates", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = [(n, o, m, l, r, s, args.sp_updates, args.br_updates)
            for (n, o, m, l, r) in METHODS for s in range(args.seeds)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)
    print(f"all {len(jobs)} jobs done in {time.time()-t0:.0f}s")

    stats = {}
    for name, *_ in METHODS:
        vals = np.array([b for (n, s, b) in raw if n == name])
        stats[name] = (vals.mean(), vals.std(), len(vals))

    order = sorted(stats, key=lambda k: stats[k][0])
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(order))
    ax.bar(xs, [stats[k][0] for k in order], yerr=[stats[k][1] for k in order],
           color=[COLORS[k] for k in order], capsize=4)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABELS[k] for k in order], rotation=20, ha="right")
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_ylabel("best-response win-rate vs frozen policy\n(lower = harder to exploit = more robust)")
    ax.set_title(f"Connect Four self-play robustness ({args.seeds} seeds, "
                 f"{args.sp_updates} self-play + {args.br_updates} BR updates)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "connect4_curves.png"), dpi=130)
    print(f"wrote {os.path.join(RESULTS_DIR, 'connect4_curves.png')}")

    with open(os.path.join(RESULTS_DIR, "connect4_exploitability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "br_winrate", "std", "n"])
        for k in order:
            w.writerow([k, f"{stats[k][0]:.4f}", f"{stats[k][1]:.4f}", stats[k][2]])
    print("\n=== Connect Four: best-response win-rate (lower = more robust) ===")
    for k in order:
        print(f"  {LABELS[k]:32s} {stats[k][0]:.3f} +/- {stats[k][1]:.3f}")


if __name__ == "__main__":
    main()
