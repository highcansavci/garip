"""Coin Game opponent x magnet ablation -- isolating the reference (reviewer #3).

GARIP (average opponent + moving magnet) and R-NaD (current opponent + snapshot
magnet) differ on BOTH axes, so their head-to-head gap conflates two variables.
This script fills the full 2x2 grid at the standard config (lam=0.5, K=200,
polyak rho=0.01) so the effect of the *magnet* can be read holding the *opponent*
fixed, and vice versa:

    opponent \ magnet |   moving (avg)      periodic (snapshot)
    ------------------+-------------------------------------------
    average           |   GARIP (diag)      avg + snapshot  (NEW)
    current           |   cur + moving (NEW) R-NaD (diag)

Writes results/coin_ablation.csv. 10 seeds, exact same env/PPO config as
run_coin_game.py so the diagonal cells reproduce the headline numbers.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# (name, opponent_mode, magnet_mode): the 2x2 (+ labels). lam, K, rho fixed below.
CELLS = [
    ("garip",        "average", "moving"),    # avg + moving  (diagonal = GARIP)
    ("rnad",         "current", "periodic"),  # current + snapshot (diagonal = R-NaD)
    ("avg_snapshot", "average", "periodic"),  # avg opponent + snapshot magnet  (off-diagonal)
    ("cur_moving",   "current", "moving"),    # current opponent + moving magnet (off-diagonal)
]
LABELS = {
    "garip": "average + moving  (GARIP)",
    "rnad": "current + snapshot  (R-NaD)",
    "avg_snapshot": "average + snapshot",
    "cur_moving": "current + moving",
}
LAM, RESET_EVERY = 0.5, 200


def _worker(job):
    name, opp_mode, magnet_mode, seed, updates, br_updates = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import jax
    from garip.envs import CoinGame
    from garip.rl.ppo_selfplay import PPOConfig, make_selfplay_trainer
    from garip.rl.exploitability import approx_exploitability

    env = CoinGame(episode_length=16, zero_sum=True)
    cfg = PPOConfig()  # polyak rho = 0.01 (default), matching the headline run
    _, init, train_chunk = make_selfplay_trainer(env, cfg, opp_mode, magnet_mode, LAM, RESET_EVERY)
    carry = init(jax.random.PRNGKey(seed))
    t0 = time.time()
    carry, _ = train_chunk(carry, updates)
    e = approx_exploitability(env, cfg, carry[0], jax.random.PRNGKey(10_000 + seed),
                              br_updates=br_updates)
    print(f"{name} seed {seed}: {e:+.2f} ({time.time()-t0:.0f}s)", flush=True)
    return name, seed, float(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--br-updates", type=int, default=250)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = [(name, opp, mag, s, args.updates, args.br_updates)
            for (name, opp, mag) in CELLS for s in range(args.seeds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)

    rows = []
    for name, _opp, _mag in CELLS:
        vals = np.array([e for (m, s, e) in raw if m == name])
        rows.append((name, vals.mean(), vals.std(), float(np.mean(vals > 0.0)), len(vals)))

    with open(os.path.join(RESULTS_DIR, "coin_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "label", "mean_exploit_return", "std", "collapse_frac(>0)", "n"])
        for name, m, sd, cf, n in rows:
            w.writerow([name, LABELS[name], f"{m:.3f}", f"{sd:.3f}", f"{cf:.2f}", n])

    print("\n=== Coin Game opponent x magnet ablation (lam=0.5, K=200, rho=0.01; "
          "lower = more robust) ===")
    for name, m, sd, cf, n in rows:
        print(f"  {LABELS[name]:28s} {m:+7.2f} +/- {sd:4.2f}   collapse={cf:.2f}  (n={n})")
    # Decomposition: how much of GARIP-vs-R-NaD is opponent vs magnet?
    d = {name: m for name, m, _, _, _ in rows}
    print("\n  magnet effect (avg opp):   moving - snapshot = "
          f"{d['garip'] - d['avg_snapshot']:+.2f}")
    print("  magnet effect (cur opp):   moving - snapshot = "
          f"{d['cur_moving'] - d['rnad']:+.2f}")
    print("  opponent effect (moving):  avg - current     = "
          f"{d['garip'] - d['cur_moving']:+.2f}")
    print("  opponent effect (snapshot):avg - current     = "
          f"{d['avg_snapshot'] - d['rnad']:+.2f}")


if __name__ == "__main__":
    main()
