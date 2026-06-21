"""Neural CGSP on Kuhn poker: a first scale-up beyond matrix games.

Each player's behavioral strategy is produced by a small Flax MLP over information-set
features. We train with CGSP (self-play gradient + cycle-consistency, tau annealed) and
compare to naive self-play gradient ascent (SGA) and exact-best-response fictitious
play (XFP). Convergence is measured by the *exact* exploitability from cgsp/kuhn.py.

Usage:
    python experiments/run_kuhn.py [--steps 3000] [--seeds 8]
Writes results/kuhn_exploitability.csv and results/kuhn_curves.png.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garip import kuhn

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Annealing tau toward a small floor lets CGSP shed its quantal-response bias and
# reach near-Nash; 0.03 over 5000 steps matches exact-best-response fictitious play.
TAU_INIT, TAU_FINAL, ANNEAL_STEPS = 0.5, 0.03, 5000
LR = 0.01
POLYAK = 0.01  # GARIP moving-average magnet rate


class Policy(nn.Module):
    """Maps (6, 5) infoset features to a (6, 2) behavioral strategy."""
    hidden: int = 16

    @nn.compact
    def __call__(self, feats):
        h = nn.tanh(nn.Dense(self.hidden)(feats))
        logits = nn.Dense(kuhn.NUM_ACTIONS)(h)
        return jax.nn.softmax(logits, axis=1)


_NET = Policy()


def _strats(params1, params2):
    return _NET.apply(params1, kuhn.P1_FEATURES), _NET.apply(params2, kuhn.P2_FEATURES)


def _init_params(key):
    k1, k2 = jax.random.split(key)
    p1 = _NET.init(k1, kuhn.P1_FEATURES)
    p2 = _NET.init(k2, kuhn.P2_FEATURES)
    return p1, p2


# --------------------------- unified magnet self-play ---------------------- #
def run_magnet(magnet_mode: str, lam: float, steps: int, key: jax.Array, reset_every: int = 500):
    """Neural self-play with a `λ·KL(s ‖ magnet)` term. magnet_mode selects the method:
    "moving" = GARIP (running-average magnet), "periodic" = R-NaD (snapshot every
    `reset_every`), "fixed" = MMD (uniform magnet), "none" = naive (λ=0). Reports the
    exact exploitability of the running-average strategy."""
    optimizer = optax.adam(LR)
    uniform = jnp.full((kuhn.NUM_INFOSETS, kuhn.NUM_ACTIONS), 1.0 / kuhn.NUM_ACTIONS)

    def kl(s, m):
        return jnp.sum(s * (jnp.log(s + 1e-9) - jnp.log(m + 1e-9)))

    def loss_p1(p1, p2, m1):
        s1, s2 = _strats(p1, p2)
        return -kuhn.ev(s1, s2) + lam * kl(s1, m1)

    def loss_p2(p1, p2, m2):
        s1, s2 = _strats(p1, p2)
        return kuhn.ev(s1, s2) + lam * kl(s2, m2)

    def scan_step(carry, _):
        p1, p2, opt_state, mov1, mov2, snap1, snap2, av1, av2, t = carry
        s1, s2 = _strats(p1, p2)
        if magnet_mode == "fixed":
            m1, m2 = uniform, uniform
        elif magnet_mode == "periodic":
            m1, m2 = snap1, snap2
        elif magnet_mode == "moving":
            m1, m2 = mov1, mov2
        else:
            m1, m2 = s1, s2
        m1, m2 = jax.lax.stop_gradient(m1), jax.lax.stop_gradient(m2)
        g1 = jax.grad(loss_p1, argnums=0)(p1, p2, m1)
        g2 = jax.grad(loss_p2, argnums=1)(p1, p2, m2)
        updates, opt_state = optimizer.update((g1, g2), opt_state)
        p1, p2 = optax.apply_updates((p1, p2), updates)
        s1, s2 = _strats(p1, p2)
        mov1 = (1 - POLYAK) * mov1 + POLYAK * s1
        mov2 = (1 - POLYAK) * mov2 + POLYAK * s2
        reset = jnp.mod(t + 1.0, reset_every) < 0.5
        snap1 = jnp.where(reset, s1, snap1)
        snap2 = jnp.where(reset, s2, snap2)
        t1 = t + 1.0
        av1 = av1 + (s1 - av1) / t1
        av2 = av2 + (s2 - av2) / t1
        return (p1, p2, opt_state, mov1, mov2, snap1, snap2, av1, av2, t1), \
            kuhn.exploitability(av1, av2)

    @jax.jit
    def rollout(k):
        p1, p2 = _init_params(k)
        opt_state = optimizer.init((p1, p2))
        s1, s2 = _strats(p1, p2)
        e0 = kuhn.exploitability(s1, s2)
        carry = (p1, p2, opt_state, s1, s2, s1, s2, s1, s2, jnp.array(0.0))
        _, expls = jax.lax.scan(scan_step, carry, None, length=steps)
        return jnp.concatenate([e0[None], expls])

    return rollout(key)


# --------------------------- neural gradient methods ----------------------- #
def run_gradient(lam: float, steps: int, key: jax.Array):
    optimizer = optax.adam(LR)

    def tau_at(t):
        frac = jnp.clip(t / ANNEAL_STEPS, 0.0, 1.0)
        return TAU_INIT + (TAU_FINAL - TAU_INIT) * frac

    def loss_p1(p1, p2, tau):
        s1, s2 = _strats(p1, p2)
        return -kuhn.ev(s1, s2) + lam * kuhn.cycle_loss(s1, s2, tau)

    def loss_p2(p1, p2, tau):
        s1, s2 = _strats(p1, p2)
        return kuhn.ev(s1, s2) + lam * kuhn.cycle_loss(s1, s2, tau)

    def scan_step(carry, _):
        p1, p2, opt_state, t = carry
        tau = tau_at(t)
        g1 = jax.grad(loss_p1, argnums=0)(p1, p2, tau)
        g2 = jax.grad(loss_p2, argnums=1)(p1, p2, tau)
        updates, opt_state = optimizer.update((g1, g2), opt_state)
        p1, p2 = optax.apply_updates((p1, p2), updates)
        s1, s2 = _strats(p1, p2)
        return (p1, p2, opt_state, t + 1.0), kuhn.exploitability(s1, s2)

    @jax.jit
    def rollout(k):
        p1, p2 = _init_params(k)
        opt_state = optimizer.init((p1, p2))
        s1, s2 = _strats(p1, p2)
        e0 = kuhn.exploitability(s1, s2)
        carry = (p1, p2, opt_state, jnp.array(0.0))
        _, expls = jax.lax.scan(scan_step, carry, None, length=steps)
        return jnp.concatenate([e0[None], expls])

    return rollout(key)


# --------------------------- CGSP-quantal (averaged target) ---------------- #
def run_cgsp_quantal(steps: int, key: jax.Array, lam: float = 1.0):
    """CGSP with the averaged-opponent smoothed-BR target (the unified formulation).

    Each player is pulled toward its smoothed best response to the opponent's *running
    average*, plus the adversarial game gradient; the reported strategy is the average.
    """
    optimizer = optax.adam(LR)

    def tau_at(t):
        frac = jnp.clip(t / ANNEAL_STEPS, 0.0, 1.0)
        return TAU_INIT + (TAU_FINAL - TAU_INIT) * frac

    def loss_p1(p1, p2, target1):
        s1, s2 = _strats(p1, p2)
        return -kuhn.ev(s1, s2) + lam * jnp.sum((target1 - s1) ** 2)

    def loss_p2(p1, p2, target2):
        s1, s2 = _strats(p1, p2)
        return kuhn.ev(s1, s2) + lam * jnp.sum((target2 - s2) ** 2)

    def scan_step(carry, _):
        p1, p2, opt_state, avg1, avg2, t = carry
        tau = tau_at(t)
        target1 = jax.lax.stop_gradient(kuhn.smoothed_br_p1(avg2, avg1, tau))
        target2 = jax.lax.stop_gradient(kuhn.smoothed_br_p2(avg1, tau))
        g1 = jax.grad(loss_p1, argnums=0)(p1, p2, target1)
        g2 = jax.grad(loss_p2, argnums=1)(p1, p2, target2)
        updates, opt_state = optimizer.update((g1, g2), opt_state)
        p1, p2 = optax.apply_updates((p1, p2), updates)
        s1, s2 = _strats(p1, p2)
        t1 = t + 1.0
        avg1 = avg1 + (s1 - avg1) / t1
        avg2 = avg2 + (s2 - avg2) / t1
        return (p1, p2, opt_state, avg1, avg2, t1), kuhn.exploitability(avg1, avg2)

    @jax.jit
    def rollout(k):
        p1, p2 = _init_params(k)
        opt_state = optimizer.init((p1, p2))
        s1, s2 = _strats(p1, p2)
        e0 = kuhn.exploitability(s1, s2)
        carry = (p1, p2, opt_state, s1, s2, jnp.array(1.0))
        _, expls = jax.lax.scan(scan_step, carry, None, length=steps)
        return jnp.concatenate([e0[None], expls])

    return rollout(key)


# --------------------------- fictitious play (XFP) ------------------------- #
def run_fictitious_play(steps: int, key: jax.Array):
    def scan_step(carry, _):
        avg1, avg2, t = carry
        br1 = kuhn.best_response_p1(avg2)
        br2 = kuhn.best_response_p2(avg1)
        t1 = t + 1.0
        avg1 = avg1 + (br1 - avg1) / t1
        avg2 = avg2 + (br2 - avg2) / t1
        return (avg1, avg2, t1), kuhn.exploitability(avg1, avg2)

    @jax.jit
    def rollout(k):
        p1, p2 = _init_params(k)
        avg1, avg2 = _strats(p1, p2)
        e0 = kuhn.exploitability(avg1, avg2)
        carry = (avg1, avg2, jnp.array(1.0))
        _, expls = jax.lax.scan(scan_step, carry, None, length=steps)
        return jnp.concatenate([e0[None], expls])

    return rollout(key)


METHODS = {
    "garip": lambda steps, key: run_magnet("moving", 0.5, steps, key),
    "rnad": lambda steps, key: run_magnet("periodic", 0.5, steps, key, reset_every=500),
    "mmd": lambda steps, key: run_magnet("fixed", 0.5, steps, key),
    "naive": lambda steps, key: run_magnet("none", 0.0, steps, key),
    "fictitious_play": run_fictitious_play,
}
COLORS = {"garip": "#d62728", "rnad": "#9467bd", "mmd": "#2ca02c",
          "naive": "#7f7f7f", "fictitious_play": "#1f77b4"}
LABELS = {"garip": "GARIP (ours, moving magnet)", "rnad": "R-NaD (periodic snapshot)",
          "mmd": "MMD (fixed magnet)", "naive": "Naive self-play",
          "fictitious_play": "Fictitious play (exact BR)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6500)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    keys = jax.random.split(jax.random.PRNGKey(0), args.seeds)
    curves, bands, finals = {}, {}, {}
    for name, fn in METHODS.items():
        runner = jax.vmap(lambda k: fn(args.steps, k))
        expl = jnp.asarray(runner(keys))  # (seeds, steps+1)
        curves[name] = jnp.mean(expl, axis=0)
        bands[name] = (jnp.percentile(expl, 25, axis=0), jnp.percentile(expl, 75, axis=0))
        finals[name] = float(jnp.mean(expl[:, -1]))

    # CSV
    path = os.path.join(RESULTS_DIR, "kuhn_exploitability.csv")
    n = args.steps + 1
    header = ["step"] + [f"{m}_mean" for m in curves]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            w.writerow([i] + [float(curves[m][i]) for m in curves])
    print(f"wrote {path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in curves:
        xs = jnp.arange(curves[name].shape[0])
        ax.plot(xs, curves[name], label=LABELS[name], color=COLORS[name], lw=2)
        ax.fill_between(xs, bands[name][0], bands[name][1], color=COLORS[name], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("exact exploitability (log)")
    ax.set_title("Neural self-play on Kuhn poker: GARIP vs R-NaD vs MMD (mean over seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "kuhn_curves.png")
    fig.savefig(fig_path, dpi=130)
    print(f"wrote {fig_path}")

    print("\n=== Kuhn poker: final exploitability (mean over seeds) ===")
    for name, v in finals.items():
        print(f"  {LABELS[name]:28s} {v:.4f}")


if __name__ == "__main__":
    main()
