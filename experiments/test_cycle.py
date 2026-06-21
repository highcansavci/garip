"""Is a *reference-free* cycle-consistency regularizer (CycleGAN load-bearing) better
than R-NaD's stale snapshot and GARIP's running average?

CYCLE anchors each player not to a history-based reference (snapshot / EMA) but to its
own RECIPROCAL best response -- the cycle map F(G(x)) computed from the CURRENT policies
(G = column quantal-BR, F = row quantal-BR). There is no reference to go stale, so the
hypothesis is: CYCLE has no staleness-collapse regime, the failure mode that R-NaD (large
K) and GARIP (slow rho) both have.

Pure-numpy (tiny matrices). Two tests:
  (1) Last-iterate exploitability on matrix games (does it even converge to Nash?).
  (2) Robustness: push each method's stress knob and count collapses; the claim to
      falsify is that CYCLE stays converged where R-NaD/GARIP collapse.
"""
from __future__ import annotations

import numpy as np

ETA = 0.3


def sm(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def col_qbr(A, x, tau):   # G: column (minimizer) quantal BR to row x
    return sm(-(x @ A) / tau)


def row_qbr(A, y, tau):   # F: row (maximizer) quantal BR to column y
    return sm((A @ y) / tau)


def games():
    rps = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
    mp = np.array([[1., -1.], [-1., 1.]])
    rng = np.random.default_rng(0)
    R = rng.standard_normal((10, 10))
    R2 = rng.standard_normal((12, 6))
    return [("rps", rps), ("matching_pennies", mp), ("random_10x10", R), ("random_12x6", R2)]


def exploit(A, x, y):
    return float(np.max(A @ y) - np.min(x @ A))


def run(A, method, T, beta=0.1, K=200, tau=0.1, seed=0):
    m, n = A.shape
    rng = np.random.default_rng(seed)
    x, y = sm(rng.standard_normal(m)), sm(rng.standard_normal(n))
    ax, ay = x.copy(), y.copy()
    sx, sy = x.copy(), y.copy()
    gxp, gyp = A @ y, x @ A
    for t in range(1, T + 1):
        gx, gy = A @ y, x @ A
        xh = x * np.exp(ETA * (2 * gx - gxp)); xh /= xh.sum()
        yh = y * np.exp(-ETA * (2 * gy - gyp)); yh /= yh.sum()
        if method == "garip":
            rx, ry = ax, ay
        elif method == "rnad":
            rx, ry = sx, sy
        elif method == "cycle":
            rx = row_qbr(A, col_qbr(A, x, tau), tau)
            ry = col_qbr(A, row_qbr(A, y, tau), tau)
        else:
            rx, ry = xh, yh
        x = (1 - beta) * xh + beta * rx
        y = (1 - beta) * yh + beta * ry
        ax += (x - ax) / (t + 1)
        ay += (y - ay) / (t + 1)
        if t % K == 0:
            sx, sy = x.copy(), y.copy()
        gxp, gyp = gx, gy
    return exploit(A, x, y)


def main():
    print("=== (1) last-iterate exploitability (5000 steps, default knobs, median of 5) ===\n")
    print(f"{'game':>16} {'GARIP':>9} {'R-NaD':>9} {'CYCLE':>9}")
    for name, A in games():
        eg = np.median([run(A, "garip", 5000, beta=0.02, seed=s) for s in range(5)])
        er = np.median([run(A, "rnad", 5000, beta=0.1, K=200, seed=s) for s in range(5)])
        ec = np.median([run(A, "cycle", 5000, beta=0.1, tau=0.1, seed=s) for s in range(5)])
        print(f"{name:>16} {eg:9.4f} {er:9.4f} {ec:9.4f}")

    print("\n=== (2) robustness: collapse rate over each method's STRESS knob ===")
    print("    (random 10x10, 5 seeds; collapse = final exploit > 0.1)\n")
    A = games()[2][1]
    print("  R-NaD over reset period K (staleness knob):")
    for K in [50, 100, 200, 400, 800, 1600]:
        c = np.mean([run(A, "rnad", 5000, beta=0.1, K=K, seed=s) > 0.1 for s in range(5)])
        print(f"    K={K:<5} collapse={c:.1f}")
    print("  GARIP over anchor strength beta:")
    for b in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        c = np.mean([run(A, "garip", 5000, beta=b, seed=s) > 0.1 for s in range(5)])
        print(f"    beta={b:<6} collapse={c:.1f}")
    print("  CYCLE over regularizer strength beta (NO staleness knob exists):")
    for b in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        c = np.mean([run(A, "cycle", 5000, beta=b, tau=0.1, seed=s) > 0.1 for s in range(5)])
        print(f"    beta={b:<6} collapse={c:.1f}")
    print("  CYCLE over temperature tau:")
    for tau in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        c = np.mean([run(A, "cycle", 5000, beta=0.1, tau=tau, seed=s) > 0.1 for s in range(5)])
        print(f"    tau={tau:<6} collapse={c:.1f}")


if __name__ == "__main__":
    main()
