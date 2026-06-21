"""Deep-RL self-play on STORM-2p with a *fair* metric (embedded-game exploitability).

The proxy best-response metric does not transfer to STORM: its payoff fires only on a
zap when both players hold inventory, so a non-engaging policy is unexploitable by
construction, and matching pennies' uniform Nash is hit for free by MMD's uniform magnet
and by random play (see `_diag_storm.py`: MMD interaction rate ~5e-4). We therefore (a)
embed a *non-uniform-Nash* zero-sum game (SKEWED, Nash (1/3,2/3), value 1/3) so uniform
play is exploitable, and (b) score each method by the **exact exploitability of its
effective inventory-mix strategy** in that 2x2 game --- i.e. does spatial self-play drive
the inventory mix to the embedded equilibrium? Lower = closer to Nash = more robust.

Writes results/storm_curves.png + results/storm_nashconv.csv.
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
LABELS = {"naive": "Naive self-play", "fictitious": "Fictitious (avg opponent)",
          "mmd": "MMD (fixed magnet)", "rnad": "R-NaD (periodic snapshot)",
          "garip": "GARIP (ours, moving magnet)"}
EPISODE_LENGTH = 64
# Row payoff of the embedded zero-sum game (matches SKEWED[0]); Nash (1/3,2/3), value 1/3.
A = np.array([[3.0, -1.0], [-1.0, 1.0]])


def _game_value(A):
    maximin = max(min(A[0]), min(A[1]))
    minimax = min(max(A[:, 0]), max(A[:, 1]))
    if abs(maximin - minimax) < 1e-9:  # pure saddle
        return maximin
    return (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / (A[0, 0] + A[1, 1] - A[0, 1] - A[1, 0])


def _exploitability(sigma, A, v):
    # Opponent best-responds (column) to minimise the row's payoff under mix sigma.
    return float(v - min(sigma @ A))


def _worker(job):
    name, opp_mode, magnet_mode, lam, reset_every, seed, updates = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import jax
    import jax.numpy as jnp
    from garip.envs import Storm2p, SKEWED
    from garip.rl.ppo_selfplay import PPOConfig, make_selfplay_trainer

    env = Storm2p(episode_length=EPISODE_LENGTH, zero_sum=True, payoff=SKEWED)
    cfg = PPOConfig()
    net, init, train_chunk = make_selfplay_trainer(env, cfg, opp_mode, magnet_mode, lam, reset_every)
    carry = init(jax.random.PRNGKey(seed))

    def mean_mix(params, key, n_eps=128):
        """Time-averaged normalised inventory mix (the effective matrix-game strategy)."""
        def rollout(k):
            (o0, o1), st = env.reset(k)
            def body(carry, _):
                o0, o1, st, k = carry
                k, k0, k1, ks = jax.random.split(k, 4)
                l0, _ = net.apply(params, o0); l1, _ = net.apply(params, o1)
                a0 = jax.random.categorical(k0, l0); a1 = jax.random.categorical(k1, l1)
                (o0, o1), st, _, _, _ = env.step(ks, st, a0, a1)
                inv = jnp.stack([st.red_inventory, st.blue_inventory])  # (2,2)
                s = inv.sum(-1, keepdims=True)
                mix = jnp.where(s > 0, inv / jnp.maximum(s, 1e-8), 0.0)
                w = (s[:, 0] > 0).astype(jnp.float32)                   # only count when engaged
                return (o0, o1, st, k), (mix * w[:, None], w)
            _, (mixes, ws) = jax.lax.scan(body, (o0, o1, st, k), None, length=EPISODE_LENGTH)
            return mixes.sum((0, 1)), ws.sum()                          # (2,) summed, scalar weight
        keys = jax.random.split(key, n_eps)
        num, den = jax.vmap(rollout)(keys)
        return np.asarray(num.sum(0) / max(float(den.sum()), 1e-8))

    expl_curve, checkpoints, done = [], list(range(0, updates + 1, updates // 3)), 0
    t0 = time.time()
    vstar = _game_value(A)
    for cp in checkpoints:
        if cp > done:
            carry, _ = train_chunk(carry, cp - done); done = cp
        sigma = mean_mix(carry[0], jax.random.PRNGKey(10_000 + seed))
        expl_curve.append(_exploitability(sigma, A, vstar))
    print(f"{name} seed {seed}: mix={np.round(sigma,3)} expl={np.round(expl_curve,3)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return name, seed, checkpoints, expl_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=2400)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = [(n, o, m, l, r, s, args.updates) for (n, o, m, l, r) in METHODS for s in range(args.seeds)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)
    print(f"all {len(jobs)} jobs done in {time.time()-t0:.0f}s")

    results = {}
    for name, *_ in METHODS:
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
    ax.set_ylabel("embedded-game exploitability of the\ninventory mix (lower = closer to Nash)")
    ax.set_title(f"STORM-2p (skewed zero-sum, Nash (1/3,2/3)): inventory-mix exploitability "
                 f"({args.seeds} seeds)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "storm_curves.png"), dpi=130)
    print(f"wrote {os.path.join(RESULTS_DIR, 'storm_curves.png')}")

    with open(os.path.join(RESULTS_DIR, "storm_nashconv.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "final_exploitability", "std"])
        for name, (_xs, mean, std) in results.items():
            w.writerow([name, f"{mean[-1]:.4f}", f"{std[-1]:.4f}"])
    print("\n=== STORM-2p inventory-mix exploitability (lower = closer to embedded Nash) ===")
    print(f"  (reference: uniform play scores {_exploitability(np.array([.5,.5]), A, _game_value(A)):.3f})")
    for name, (_xs, mean, std) in results.items():
        print(f"  {LABELS[name]:32s} {mean[-1]:.4f} +/- {std[-1]:.4f}")


if __name__ == "__main__":
    main()
