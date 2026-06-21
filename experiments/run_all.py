"""Run every method on every game across many seeds; save exploitability curves.

Usage:
    python experiments/run_all.py [--steps 5000] [--seeds 16]

Writes one CSV per game to results/exploitability_<game>.csv with columns:
    step, <method>_mean, <method>_p25, <method>_p75   (for each method)
and prints a final-exploitability summary table.
"""
from __future__ import annotations

import argparse
import csv
import os

import jax
import jax.numpy as jnp

# Project root on path when run as a script.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garip.games import rps, matching_pennies, random_zero_sum
from garip import methods
from garip.train import run_seeds

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def make_games():
    return [
        rps(),
        matching_pennies(),
        random_zero_sum(jax.random.PRNGKey(0), m=10, n=10),
        random_zero_sum(jax.random.PRNGKey(1), m=12, n=6),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    algos = methods.all_methods()
    games = make_games()

    summary = {}  # game_name -> {method_name -> final mean exploitability}

    for game in games:
        curves = {}  # method -> (steps+1,) mean curve
        bands = {}   # method -> (p25, p75)
        finals = {}
        for algo in algos:
            keys = jax.random.split(jax.random.PRNGKey(1234), args.seeds)
            expl, _, _ = run_seeds(algo, game, args.steps, keys)  # (seeds, steps+1)
            expl = jnp.asarray(expl)
            curves[algo.name] = jnp.mean(expl, axis=0)
            bands[algo.name] = (
                jnp.percentile(expl, 25, axis=0),
                jnp.percentile(expl, 75, axis=0),
            )
            finals[algo.name] = float(jnp.mean(expl[:, -1]))
        summary[game.name] = finals

        # Write CSV.
        path = os.path.join(RESULTS_DIR, f"exploitability_{game.name}.csv")
        n = args.steps + 1
        header = ["step"]
        for name in curves:
            header += [f"{name}_mean", f"{name}_p25", f"{name}_p75"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n):
                row = [i]
                for name in curves:
                    row += [
                        float(curves[name][i]),
                        float(bands[name][0][i]),
                        float(bands[name][1][i]),
                    ]
                w.writerow(row)
        print(f"wrote {path}")

    # Print summary table of final (last-iterate) exploitability.
    print("\n=== Final last-iterate exploitability (mean over seeds) ===")
    method_names = [a.name for a in algos]
    print("game".ljust(16) + "".join(m.ljust(18) for m in method_names))
    for game_name, finals in summary.items():
        line = game_name.ljust(16)
        for m in method_names:
            line += f"{finals[m]:.4f}".ljust(18)
        print(line)


if __name__ == "__main__":
    main()
