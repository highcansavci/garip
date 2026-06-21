"""Leduc hold'em methods comparison: GARIP vs R-NaD vs MMD vs naive vs CFR (exact).

Tabular self-play on the Leduc tree (mirror ascent on exact counterfactual values,
`leduc.counterfactual_values`), reporting the *exact* exploitability of the time-average
strategy over iterations. The neural raw-gradient self-play does not converge for the
magnet baselines on Leduc (the reach/credit-assignment issue the counterfactual target
was built to fix), so this exact tabular comparison is the faithful one. GARIP =
optimism + running-average anchor; R-NaD = KL-proximal to a periodic snapshot; MMD = KL
to a uniform magnet; naive = plain mirror ascent; CFR = reference.

Usage:
    python experiments/run_leduc_methods.py [--steps 3000] [--seeds 8]
Writes results/leduc_curves.png and results/leduc_methods.csv.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

COLORS = {"garip": "#d62728", "rnad": "#9467bd", "mmd": "#2ca02c",
          "naive": "#7f7f7f", "cfr": "#1f77b4"}
LABELS = {"garip": "GARIP (ours, moving magnet)", "rnad": "R-NaD (periodic snapshot)",
          "mmd": "MMD (fixed magnet)", "naive": "Naive self-play", "cfr": "CFR (reference)"}
ETA, BETA = 0.3, 0.02           # GARIP
RNAD_ALPHA, RNAD_K = 0.5, 300   # R-NaD
MMD_ETA = 0.5                   # closed-form step for R-NaD / MMD


def _norm(s):
    return s / np.maximum(s.sum(axis=1, keepdims=True), 1e-12)


def _worker(job):
    method, seed, steps, eval_every = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from garip import leduc
    G = leduc.GAME
    legal = [np.asarray(G.legal0), np.asarray(G.legal1)]

    if method == "cfr":
        R = [np.zeros((G.n0, leduc.NUM_ACTIONS)), np.zeros((G.n1, leduc.NUM_ACTIONS))]
        S = [np.zeros_like(R[0]), np.zeros_like(R[1])]
        iters, curve = [], []
        for t in range(steps + 1):
            s = [leduc.regret_matching(R[p], legal[p]) for p in (0, 1)]
            for p in (0, 1):
                S[p] += s[p]
            if t % eval_every == 0:
                a = [_norm(S[p] + 1e-12) for p in (0, 1)]
                iters.append(t)
                curve.append(leduc.exploitability(G, a[0], a[1]))
            r0, r1 = leduc.counterfactual_regrets(G, s[0], s[1])
            R[0] += r0
            R[1] += r1
        return method, seed, np.array(iters), np.array(curve)

    rng = np.random.default_rng(seed)
    s = [_norm(np.exp(rng.normal(size=legal[p].shape)) * legal[p]) for p in (0, 1)]
    mov = [s[0].copy(), s[1].copy()]
    snap = [s[0].copy(), s[1].copy()]
    ssum = [s[0].copy(), s[1].copy()]
    qprev = list(leduc.counterfactual_values(G, s[0], s[1]))
    iters, curve = [], []
    for t in range(steps + 1):
        if t % eval_every == 0:
            avg = [_norm(ssum[p]) for p in (0, 1)]
            iters.append(t)
            curve.append(leduc.exploitability(G, avg[0], avg[1]))
        if t < steps:
            q = list(leduc.counterfactual_values(G, s[0], s[1]))
            if method == "garip":
                for p in (0, 1):
                    h = _norm(s[p] * np.exp(ETA * (2 * q[p] - qprev[p])) * legal[p])
                    s[p] = (1 - BETA) * h + BETA * mov[p]
                qprev = q
                for p in (0, 1):
                    mov[p] = mov[p] + (s[p] - mov[p]) / (t + 2.0)
            elif method == "naive":
                for p in (0, 1):
                    s[p] = _norm(s[p] * np.exp(ETA * q[p]) * legal[p])
            else:  # rnad / mmd
                c = 1.0 / (1.0 + MMD_ETA * RNAD_ALPHA)
                for p in (0, 1):
                    mag = (legal[p] / legal[p].sum(1, keepdims=True)
                           if method == "mmd" else snap[p])
                    s[p] = _norm((s[p] ** c) * (mag ** (1.0 - c)) * np.exp(MMD_ETA * c * q[p]) * legal[p])
                if method == "rnad" and (t + 1) % RNAD_K == 0:
                    snap = [s[0].copy(), s[1].copy()]
            for p in (0, 1):
                ssum[p] += s[p]
    print(f"{method} seed {seed}: final expl={curve[-1]:.4f}", flush=True)
    return method, seed, np.array(iters), np.array(curve)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = [(m, s, args.steps, args.eval_every)
            for m in ("garip", "rnad", "mmd", "naive") for s in range(args.seeds)]
    jobs.append(("cfr", 0, args.steps, args.eval_every))  # deterministic reference
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(_worker, jobs)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    finals = {}
    for m in ("cfr", "garip", "rnad", "mmd", "naive"):
        runs = [c for (mm, s, it, c) in raw if mm == m]
        it = next(it for (mm, s, it, c) in raw if mm == m)
        arr = np.stack(runs)
        mean = arr.mean(0)
        ax.plot(it, mean, color=COLORS[m], lw=2, label=LABELS[m])
        if arr.shape[0] > 1:
            ax.fill_between(it, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0),
                            color=COLORS[m], alpha=0.15)
        finals[m] = mean[-1]
    ax.set_yscale("log")
    ax.set_xlabel("iteration (tabular self-play sweep)")
    ax.set_ylabel("exact exploitability of the average strategy (log)")
    ax.set_title(f"Leduc hold'em: GARIP vs R-NaD vs MMD vs CFR ({args.seeds} seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "leduc_curves.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    with open(os.path.join(RESULTS_DIR, "leduc_methods.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "final_exploitability"])
        for m, v in finals.items():
            w.writerow([m, f"{v:.5f}"])
    print("\n=== Leduc methods: final exact exploitability (average strategy) ===")
    for m in ("garip", "rnad", "mmd", "naive", "cfr"):
        print(f"  {LABELS[m]:30s} {finals[m]:.4f}")


if __name__ == "__main__":
    main()
