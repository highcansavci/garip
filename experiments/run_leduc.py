"""Neural CGSP on Leduc hold'em -- the larger imperfect-information scale-up.

Each player's behavioral strategy is a Flax MLP over information-set features, trained
through the differentiable Leduc tree ([cgsp/leduc.py]). We compare:

    * CGSP (ours)  -- self-play gradient + cycle-consistency, tau annealed.
    * SGA          -- naive self-play gradient ascent (lambda = 0).
    * Fictitious play (XFP) -- exact best-response averaging (tabular reference).
    * CFR          -- counterfactual regret minimization (tabular gold standard).

Convergence is the *exact* exploitability from the tree. The cycle term uses a
gradient-based smoothed best response (one gradient of EV per player), pulling each
player's strategy toward its current best response -- the imperfect-information analog
of the F(G(x)) = x fixed-point condition.

Usage:
    python experiments/run_leduc.py [--steps 3000] [--seeds 3] [--cfr-iters 600]
Writes results/leduc_exploitability.csv and results/leduc_curves.png.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garip import leduc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
G = leduc.GAME

# The cycle target is the smoothed best response to the opponent's *running average*
# strategy (the fictitious-play insight). Because that target moves slowly, annealing
# the temperature is now stable -- so we anneal tau from 0.5 down to 0.1, which walks the
# averaged-strategy exploitability down to fictitious-play level. (Targeting the *current*
# opponent with a sharp tau instead reintroduces the cycling instability.)
TAU_INIT, TAU_FINAL = 0.5, 0.1
ANNEAL_FRAC = 0.7
LR = 0.01


class Policy(nn.Module):
    hidden: int = 32

    @nn.compact
    def __call__(self, feats):
        h = nn.relu(nn.Dense(self.hidden)(feats))
        h = nn.relu(nn.Dense(self.hidden)(h))
        return nn.Dense(leduc.NUM_ACTIONS)(h)  # raw logits, masked below


_NET = Policy()


def _masked_softmax(logits, legal):
    logits = jnp.where(legal > 0, logits, -1e9)
    return jax.nn.softmax(logits, axis=1)


def _strats(p0, p1):
    s0 = _masked_softmax(_NET.apply(p0, G.features0), G.legal0)
    s1 = _masked_softmax(_NET.apply(p1, G.features1), G.legal1)
    return s0, s1


def _init_params(key):
    k0, k1 = jax.random.split(key)
    return _NET.init(k0, G.features0), _NET.init(k1, G.features1)


def _make_update(lam):
    # The cycle targets (br0, br1) are the exact smoothed counterfactual best responses,
    # computed on the host and passed in as stop-gradient constants. The CGSP penalty
    # pulls each player toward being its own (regularized) best response -- the
    # imperfect-information fixed-point form of cycle-consistency.
    optimizer = optax.adam(LR)

    def loss_p0(p0, p1, br0):
        s0, s1 = _strats(p0, p1)
        return -leduc.ev(G, s0, s1) + lam * jnp.sum((br0 - s0) ** 2)

    def loss_p1(p0, p1, br1):
        s0, s1 = _strats(p0, p1)
        return leduc.ev(G, s0, s1) + lam * jnp.sum((br1 - s1) ** 2)

    @jax.jit
    def update(p0, p1, opt_state, br0, br1):
        g0 = jax.grad(loss_p0, argnums=0)(p0, p1, br0)
        g1 = jax.grad(loss_p1, argnums=1)(p0, p1, br1)
        updates, opt_state = optimizer.update((g0, g1), opt_state)
        p0, p1 = optax.apply_updates((p0, p1), updates)
        return p0, p1, opt_state

    return optimizer, update


def run_gradient(lam, steps, eval_every, key, target="quantal"):
    """Returns (iters, last_iterate_exploitability, average_strategy_exploitability).

    The average strategy is accumulated after a 20% burn-in; it is the apples-to-apples
    metric versus fictitious play and CFR, which are themselves average-strategy methods.

    `target` selects the cycle target when `lam > 0`:
      * "quantal" (default) -- smoothed counterfactual best response to the *average*
        opponent; smooth and slowly moving, so the neural policy tracks it well (~0.18).
      * "regret" -- CFR-style regret matching on accumulated counterfactual regrets.
        Mathematically the no-regret target, but it is sharp and fast-moving, so the
        lagging neural policy cannot track it and it underperforms badly (~1.0). Kept as
        a documented negative result: online distillation breaks the regret dynamics,
        which is exactly what Deep CFR's replay-buffer + retrain-from-scratch machinery
        exists to avoid.
    """
    optimizer, update = _make_update(lam)
    p0, p1 = _init_params(key)
    opt_state = optimizer.init((p0, p1))
    anneal_steps = max(1, int(steps * ANNEAL_FRAC))
    burn_in = steps // 5
    tavg0 = tavg1 = None   # target-average opponent (accumulated from step 0)
    eavg0 = eavg1 = None   # eval-average strategy (after burn-in) -- the reported metric
    cum_r0 = np.zeros((G.n0, leduc.NUM_ACTIONS))  # cumulative regret (regret target only)
    cum_r1 = np.zeros((G.n1, leduc.NUM_ACTIONS))
    tc = ec = 0
    iters, last_curve, avg_curve = [], [], []
    for step in range(steps + 1):
        s0, s1 = _strats(p0, p1)
        s0n, s1n = np.asarray(s0), np.asarray(s1)
        tc += 1
        if tavg0 is None:
            tavg0, tavg1 = s0n.copy(), s1n.copy()
        else:
            tavg0 += (s0n - tavg0) / tc
            tavg1 += (s1n - tavg1) / tc
        if step >= burn_in:
            ec += 1
            if eavg0 is None:
                eavg0, eavg1 = s0n.copy(), s1n.copy()
            else:
                eavg0 += (s0n - eavg0) / ec
                eavg1 += (s1n - eavg1) / ec
        if step % eval_every == 0:
            iters.append(step)
            last_curve.append(leduc.exploitability(G, s0n, s1n))
            avg_curve.append(leduc.exploitability(G, eavg0, eavg1) if eavg0 is not None
                             else last_curve[-1])
        if step < steps:
            if lam <= 0:
                p0, p1, opt_state = update(p0, p1, opt_state, s0, s1)  # lam=0: pull vanishes
            elif target == "regret":
                r0, r1 = leduc.counterfactual_regrets(G, s0n, s1n)
                cum_r0 += r0
                cum_r1 += r1
                br0 = leduc.regret_matching(cum_r0, G.legal0)
                br1 = leduc.regret_matching(cum_r1, G.legal1)
                p0, p1, opt_state = update(p0, p1, opt_state, jnp.asarray(br0), jnp.asarray(br1))
            else:  # "quantal": smoothed best response to the *average* opponent
                tau = TAU_INIT + (TAU_FINAL - TAU_INIT) * min(1.0, step / anneal_steps)
                br0, br1 = leduc.quantal_best_response(G, tavg0, tavg1, tau)
                p0, p1, opt_state = update(p0, p1, opt_state, jnp.asarray(br0), jnp.asarray(br1))
    return np.array(iters), np.array(last_curve), np.array(avg_curve)


def run_fictitious_play(iterations, eval_every, key):
    p0, p1 = _init_params(key)
    s0, s1 = _strats(p0, p1)
    avg0, avg1 = np.asarray(s0), np.asarray(s1)
    iters, expls = [], []
    for it in range(iterations + 1):
        if it % eval_every == 0:
            iters.append(it)
            expls.append(leduc.exploitability(G, avg0, avg1))
        if it < iterations:
            br0 = np.asarray(leduc.best_response_strategy(G, 0, avg1))
            br1 = np.asarray(leduc.best_response_strategy(G, 1, avg0))
            w = 1.0 / (it + 2.0)
            avg0 = avg0 + (br0 - avg0) * w
            avg1 = avg1 + (br1 - avg1) * w
    return np.array(iters), np.array(expls)


def run_cfr(iterations, eval_every):
    # CFR with periodic exploitability of the running average strategy.
    import numpy as np
    from collections import defaultdict
    legal = [np.asarray(G.legal0), np.asarray(G.legal1)]
    n = [G.n0, G.n1]
    regret = [np.zeros((n[p], leduc.NUM_ACTIONS)) for p in (0, 1)]
    strat_sum = [np.zeros((n[p], leduc.NUM_ACTIONS)) for p in (0, 1)]

    def current_strategy(p):
        pos = np.maximum(regret[p], 0.0)
        total = pos.sum(axis=1, keepdims=True)
        unif = legal[p] / legal[p].sum(axis=1, keepdims=True)
        strat = np.where(total > 0, pos / np.where(total > 0, total, 1.0), unif)
        return strat * legal[p]

    def avg_strategy(p):
        total = strat_sum[p].sum(axis=1, keepdims=True)
        unif = legal[p] / legal[p].sum(axis=1, keepdims=True)
        a = np.where(total > 0, strat_sum[p] / np.where(total > 0, total, 1.0), unif)
        return a * legal[p]

    def cfr(node, reach):
        if isinstance(node, leduc.Terminal):
            return np.array([node.payoff, -node.payoff])
        if isinstance(node, leduc.Chance):
            out = np.zeros(2)
            for prob, ch in node.children:
                out += prob * cfr(ch, [reach[0], reach[1], reach[2] * prob])
            return out
        p = node.player
        strat = current_strategy(p)[node.local_id]
        node_util = np.zeros(2)
        child_util = {}
        for a, ch in node.children.items():
            r2 = list(reach)
            r2[p] = reach[p] * strat[a]
            cu = cfr(ch, r2)
            child_util[a] = cu
            node_util += strat[a] * cu
        cf_reach = reach[1 - p] * reach[2]
        for a in node.children:
            regret[p][node.local_id, a] += cf_reach * (child_util[a][p] - node_util[p])
        strat_sum[p][node.local_id] += reach[p] * strat
        return node_util

    iters, expls = [], []
    for it in range(iterations + 1):
        if it % eval_every == 0:
            expls.append(leduc.exploitability(G, avg_strategy(0), avg_strategy(1)))
            iters.append(it)
        if it < iterations:
            cfr(G.root, [1.0, 1.0, 1.0])
    return np.array(iters), np.array(expls)


COLORS = {"cgsp": "#d62728", "cgsp_avg": "#d62728", "sga": "#7f7f7f",
          "fictitious_play": "#1f77b4", "cfr": "#2ca02c"}
LABELS = {"cgsp": "CGSP last-iterate (ours)", "cgsp_avg": "CGSP averaged (ours)",
          "sga": "Self-play grad. ascent (neural)",
          "fictitious_play": "Fictitious play (exact BR)", "cfr": "CFR (reference)"}


def _seed_worker(job):
    """Run one (method, seed) neural self-play; returns curves. Independent -> parallel.
    Launch the script with OMP_NUM_THREADS / XLA_FLAGS set so spawned children inherit
    single-threaded CPU JAX (avoids oversubscription)."""
    name, lam, target, seed, steps, eval_every = job
    t0 = time.time()
    it, last, avg = run_gradient(lam, steps, eval_every, jax.random.PRNGKey(seed), target=target)
    print(f"{name} seed {seed}: last={last[-1]:.4f} avg={avg[-1]:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    return name, seed, np.asarray(it), np.asarray(last), np.asarray(avg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=750)
    parser.add_argument("--cfr-iters", type=int, default=600)
    parser.add_argument("--fp-iters", type=int, default=600)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--target", choices=["quantal", "regret"], default="quantal",
                        help="CGSP cycle target ('regret' is a documented negative result)")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    curves = {}   # label -> (iters, mean, p25, p75)
    finals = {}   # label -> final exploitability

    # Neural methods (CGSP last-iterate + averaged, SGA): (method, seed) jobs in parallel.
    jobs = [(name, lam, args.target if name == "cgsp" else "quantal", s, args.steps, args.eval_every)
            for (name, lam) in (("cgsp", 1.0), ("sga", 0.0)) for s in range(args.seeds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_seed_worker, jobs)

    for name in ("cgsp", "sga"):
        iters = next(it for (n, s, it, last, avg) in raw if n == name)
        last_arr = np.stack([last for (n, s, it, last, avg) in raw if n == name])
        avg_arr = np.stack([avg for (n, s, it, last, avg) in raw if n == name])
        curves[name] = (iters, last_arr.mean(0), np.percentile(last_arr, 25, 0),
                        np.percentile(last_arr, 75, 0))
        finals[LABELS[name]] = last_arr.mean(0)[-1]
        if name == "cgsp":
            curves["cgsp_avg"] = (iters, avg_arr.mean(0), avg_arr.mean(0), avg_arr.mean(0))
            finals["CGSP (ours, averaged)"] = avg_arr.mean(0)[-1]

    t0 = time.time()
    it, ex = run_fictitious_play(args.fp_iters, max(1, args.fp_iters // 20), jax.random.PRNGKey(0))
    curves["fictitious_play"] = (it, ex, ex, ex)
    finals[LABELS["fictitious_play"]] = ex[-1]
    print(f"fictitious_play: final expl={ex[-1]:.4f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    it, ex = run_cfr(args.cfr_iters, max(1, args.cfr_iters // 20))
    curves["cfr"] = (it, ex, ex, ex)
    finals[LABELS["cfr"]] = ex[-1]
    print(f"cfr: final expl={ex[-1]:.4f} ({time.time()-t0:.1f}s)")

    # Plot. Neural methods on gradient-step axis; tabular references on sweep axis.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"cgsp": ("-", 2), "cgsp_avg": ("--", 2), "sga": ("-", 2),
              "fictitious_play": ("-", 2), "cfr": ("-", 2)}
    for name, (it, mean, p25, p75) in curves.items():
        ls, lw = styles[name]
        ax.plot(it, mean, ls, label=LABELS[name], color=COLORS[name], lw=lw)
        if not np.allclose(p25, p75):
            ax.fill_between(it, p25, p75, color=COLORS[name], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("iteration (gradient step or tabular sweep)")
    ax.set_ylabel("exact exploitability (log)")
    ax.set_title("Neural GARIP-style self-play on Leduc hold'em (144 infosets/player)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "leduc_neural.png")
    fig.savefig(fig_path, dpi=130)
    print(f"wrote {fig_path}")

    path = os.path.join(RESULTS_DIR, "leduc_exploitability.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "final_exploitability"])
        for label, v in finals.items():
            w.writerow([label, f"{v:.5f}"])
    print(f"wrote {path}")

    print("\n=== Leduc: final exploitability ===")
    for label, v in finals.items():
        print(f"  {label:34s} {v:.4f}")


if __name__ == "__main__":
    main()
