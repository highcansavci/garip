"""Test Remark 2: can an *anticipatory* (extrapolated, negative-weight) reference beat
the plain running average on collapse? Prop. 1's optimality is only within causal
convex averages; a double-EMA magnet (1+g)*avg - g*avg2 leads the policy (lower/negative
effective lag) and is excluded from that class. We sweep (lambda, rho) for plain GARIP
('moving') vs the extrapolated magnet ('extrap', gains g) on the Coin Game and compare
collapse rates per rho -- does extrapolation widen the safe basin, or overshoot?
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import multiprocessing as mp
import os
import sys

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LAM = [0.5, 1.0, 2.0]
RHO = [0.00125, 0.0025, 0.005, 0.01, 0.02, 0.05]


def _worker(job):
    method, lam, rho, gain, seed, updates, br, mem = job
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem:.3f}"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import jax
    from garip.envs import CoinGame
    from garip.rl.ppo_selfplay import PPOConfig, make_selfplay_trainer
    from garip.rl.exploitability import approx_exploitability
    env = CoinGame(episode_length=16, zero_sum=True)
    cfg = dataclasses.replace(PPOConfig(), polyak=rho)
    if method == "moving":
        _, init, train = make_selfplay_trainer(env, cfg, "average", "moving", lam)
    else:
        _, init, train = make_selfplay_trainer(env, cfg, "average", "extrap", lam, extrap_gain=gain)
    carry = init(jax.random.PRNGKey(seed))
    carry, _ = train(carry, updates)
    e = float(approx_exploitability(env, cfg, carry[0], jax.random.PRNGKey(10_000 + seed), br_updates=br))
    tag = method if method == "moving" else f"extrap{gain:g}"
    print(f"{tag} lam={lam} rho={rho} seed={seed}: {e:+.2f}", flush=True)
    return tag, lam, rho, e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--br-updates", type=int, default=250)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--gains", type=str, default="1.0")  # extrap gains to test
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gains = [float(g) for g in args.gains.split(",")]
    mem = min(0.85 / args.workers, 0.3)

    jobs = [("moving", lam, rho, 0.0, s, args.updates, args.br_updates, mem)
            for lam in LAM for rho in RHO for s in range(args.seeds)]
    for g in gains:
        jobs += [("extrap", lam, rho, g, s, args.updates, args.br_updates, mem)
                 for lam in LAM for rho in RHO for s in range(args.seeds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)

    tags = ["moving"] + [f"extrap{g:g}" for g in gains]
    rows = []
    print("\n=== collapse rate (exploit>0) per rho, over lambda x seeds ===")
    for tag in tags:
        print(f"  {tag}:")
        for rho in RHO:
            v = [e for (t, l, r, e) in raw if t == tag and abs(r - rho) < 1e-9]
            cr = float(np.mean([e > 0 for e in v])) if v else float("nan")
            med = float(np.median(v)) if v else float("nan")
            rows.append((tag, rho, cr, med, len(v)))
            print(f"    rho={rho:<8} collapse={cr:.2f}  median={med:+.2f}  (n={len(v)})")
    with open(os.path.join(RESULTS_DIR, "extrap_test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "rho", "collapse_rate", "median_exploit", "n"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}", r[4]])
    print("wrote results/extrap_test.csv")


if __name__ == "__main__":
    main()
