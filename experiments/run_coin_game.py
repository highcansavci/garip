"""CGSP deep-RL self-play stress test on the vendored Coin Game (JaxMARL port).

Trains three self-play methods and tracks an approximate-exploitability metric (the
exploit return of a fixed-budget best responder trained against the frozen policy;
**lower = more robust / less exploitable**, since a strong policy a budget-limited
adversary cannot beat yields a negative exploit return):

    naive self-play   : opponent = current policy
    fictitious        : opponent = Polyak running-average policy
    CGSP (ours)       : opponent = running-average + cycle KL penalty

The (method, seed) jobs are independent, so they run in parallel via `multiprocessing`
(each worker pins JAX to a single-threaded CPU device to avoid oversubscription).

Usage:
    python experiments/run_coin_game.py [--updates 1200] [--seeds 10] [--workers 12]
Writes results/coin_game_curves.png and results/coin_game_exploitability.csv.

NOTE: top-level imports stay JAX-free; JAX is imported inside each worker *after* the
thread-limiting environment is set, so the multiprocessing 'spawn' children configure
threads correctly.
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

# (name, opponent_mode, magnet_mode, lam, reset_every): the opponent x magnet ablation.
METHODS = [
    ("naive", "current", "none", 0.0, 200),       # latest self, no magnet
    ("fictitious", "average", "none", 0.0, 200),   # average opponent, no magnet
    ("mmd", "current", "fixed", 0.5, 200),         # latest self + FIXED magnet (MMD)
    ("rnad", "current", "periodic", 0.5, 200),     # latest self + PERIODIC-snapshot magnet (R-NaD)
    ("garip", "average", "moving", 0.5, 200),      # average opponent + MOVING-avg magnet (ours)
]
COLORS = {"naive": "#7f7f7f", "fictitious": "#1f77b4", "mmd": "#2ca02c",
          "rnad": "#9467bd", "garip": "#d62728"}
LABELS = {"naive": "Naive self-play", "fictitious": "Fictitious self-play",
          "mmd": "MMD (fixed magnet)", "rnad": "R-NaD (periodic-snapshot magnet)",
          "garip": "GARIP (ours, moving-avg magnet)"}


def _worker(job):
    """Run one (method, seed) to completion and return its exploitability curve."""
    name, opp_mode, magnet_mode, lam, reset_every, seed, updates, eval_every, br_updates = job

    # Pin this process to a single-threaded CPU JAX device BEFORE importing jax.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import jax
    from garip.envs import CoinGame
    from garip.rl.ppo_selfplay import PPOConfig, make_selfplay_trainer
    from garip.rl.exploitability import approx_exploitability

    env = CoinGame(episode_length=16, zero_sum=True)
    cfg = PPOConfig()
    net, init, train_chunk = make_selfplay_trainer(env, cfg, opp_mode, magnet_mode, lam, reset_every)
    carry = init(jax.random.PRNGKey(seed))
    checkpoints = list(range(0, updates + 1, eval_every))
    expl, done = [], 0
    t0 = time.time()
    for cp in checkpoints:
        if cp > done:
            carry, _ = train_chunk(carry, cp - done)
            done = cp
        e = approx_exploitability(env, cfg, carry[0], jax.random.PRNGKey(10_000 + seed),
                                  br_updates=br_updates)
        expl.append(e)
    print(f"{name} seed {seed}: {np.round(expl, 2)} ({time.time()-t0:.0f}s)", flush=True)
    return name, seed, checkpoints, expl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--br-updates", type=int, default=250)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = [(name, opp, magnet, lam, reset_every, seed, args.updates, args.eval_every, args.br_updates)
            for (name, opp, magnet, lam, reset_every) in METHODS for seed in range(args.seeds)]

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)
    print(f"all {len(jobs)} jobs done in {time.time()-t0:.0f}s")

    # Aggregate per method.
    results = {}
    for name, _opp, _magnet, _lam, _re in METHODS:
        rows = [np.array(e) for (n, s, cp, e) in raw if n == name]
        xs = [cp for (n, s, cp, e) in raw if n == name][0]
        arr = np.stack(rows)
        results[name] = (np.array(xs), arr.mean(0), arr.std(0))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, (xs, mean, std) in results.items():
        ax.plot(xs, mean, "-o", color=COLORS[name], lw=2, label=LABELS[name])
        ax.fill_between(xs, mean - std, mean + std, color=COLORS[name], alpha=0.15)
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("self-play PPO updates")
    ax.set_ylabel("best-response exploit return\n(lower = more robust / less exploitable)")
    ax.set_title(f"GARIP deep-RL self-play on Coin Game (zero-sum, {args.seeds} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "coin_game_curves.png")
    fig.savefig(fig_path, dpi=130)
    print(f"wrote {fig_path}")

    path = os.path.join(RESULTS_DIR, "coin_game_exploitability.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "final_exploit_return", "std"])
        for name, (xs, mean, std) in results.items():
            w.writerow([name, f"{mean[-1]:.4f}", f"{std[-1]:.4f}"])
    print(f"wrote {path}")

    print("\n=== Coin Game: final best-response exploit return (lower = more robust) ===")
    for name, (xs, mean, std) in results.items():
        print(f"  {LABELS[name]:24s} {mean[-1]:+.3f} ± {std[-1]:.3f}")


if __name__ == "__main__":
    main()
