"""Generate figures from the experiment runs.

Produces:
    results/exploitability_curves.png  -- exploitability vs iteration, all games/methods.
    results/rps_simplex.png            -- row-player strategy trajectories on the RPS simplex,
                                          showing SGA orbiting while GARIP spirals into Nash.

Usage:
    python experiments/plot.py [--steps 5000]
"""
from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garip.games import rps, matching_pennies, random_zero_sum
from garip import methods
from garip.train import run_seeds, strategy_trajectory

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Consistent colors per method.
COLORS = {
    "garip": "#d62728",
    "rnad": "#17becf",
    "mmd": "#2ca02c",
    "cgsp": "#e377c2",
    "cgsp_quantal": "#ff7f0e",
    "sga": "#7f7f7f",
    "fictitious_play": "#1f77b4",
    "optimistic_md": "#8c564b",
    "mirror_descent": "#9467bd",
}
LABELS = {
    "garip": "GARIP (ours)",
    "rnad": "R-NaD",
    "mmd": "MMD",
    "cgsp": "CGSP last-iterate",
    "cgsp_quantal": "CGSP-quantal, avg",
    "sga": "Self-play grad. ascent",
    "fictitious_play": "Fictitious play",
    "optimistic_md": "Optimistic MD",
    "mirror_descent": "Mirror descent",
}


def make_games():
    return [
        rps(),
        matching_pennies(),
        random_zero_sum(jax.random.PRNGKey(0), m=10, n=10),
        random_zero_sum(jax.random.PRNGKey(1), m=12, n=6),
    ]


def plot_curves(steps: int, seeds: int = 10):
    games = make_games()
    # Drop plain CGSP last-iterate: GARIP supersedes it and it clutters the panel.
    algos = [a for a in methods.all_methods() if a.name != "cgsp"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, game in zip(axes, games):
        for algo in algos:
            keys = jax.random.split(jax.random.PRNGKey(1234), seeds)
            expl, _, _ = run_seeds(algo, game, steps, keys)
            expl = jnp.asarray(expl)
            mean = jnp.mean(expl, axis=0)
            p25 = jnp.percentile(expl, 25, axis=0)
            p75 = jnp.percentile(expl, 75, axis=0)
            xs = jnp.arange(mean.shape[0])
            c = COLORS.get(algo.name, None)
            ax.plot(xs, mean, label=LABELS.get(algo.name, algo.name), color=c, lw=2)
            ax.fill_between(xs, p25, p75, color=c, alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(game.name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("exploitability (log)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Last-iterate exploitability: GARIP vs. MMD and baselines", fontsize=13)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exploitability_curves.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _project_simplex_2d(p):
    """Map a 3-simplex point to 2D for plotting (equilateral triangle)."""
    # vertices: Rock=(0,0), Paper=(1,0), Scissors=(0.5, sqrt(3)/2)
    v = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, jnp.sqrt(3) / 2]])
    return p @ v


def plot_rps_simplex(steps: int):
    game = rps()
    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Triangle outline + vertex labels.
    tri = jnp.array([[0, 0], [1, 0], [0.5, jnp.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1)
    for name, xy in zip(["Rock", "Paper", "Scissors"],
                        [(-0.04, -0.04), (1.02, -0.04), (0.5, jnp.sqrt(3) / 2 + 0.03)]):
        ax.annotate(name, xy, ha="center", fontsize=11)
    nash = _project_simplex_2d(jnp.ones(3) / 3)
    ax.scatter([nash[0]], [nash[1]], color="black", marker="*", s=200, zorder=5, label="Nash")

    key = jax.random.PRNGKey(0)
    for algo in (methods.sga(), methods.garip()):
        traj = strategy_trajectory(algo, game, steps, key)  # (steps+1, 3)
        pts = _project_simplex_2d(traj)
        c = COLORS.get(algo.name)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.2, alpha=0.8,
                label=LABELS.get(algo.name, algo.name))
        ax.scatter([pts[0, 0]], [pts[0, 1]], color=c, marker="o", s=40, zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Row-player strategy trajectory on RPS\n(circle = start, star = Nash)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "rps_simplex.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_curves(args.steps)
    plot_rps_simplex(args.steps)


if __name__ == "__main__":
    main()
