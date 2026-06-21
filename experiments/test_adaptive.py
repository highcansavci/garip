"""Can a DRIFT-ADAPTIVE moving reference remove the collapse region that R-NaD (large K)
and GARIP (slow rho) both have -- i.e. genuinely beat R-NaD by having no knob that
collapses?

Mechanism (ADA, novel): anchor to an EMA average whose rate rho_t is raised whenever the
average drifts too far from the current policy, capping the reference's *lag* by
construction. Concretely rho_t = clip(rho0 * max(1, ||x-xbar||/D), rho0, rho_max): when
the policy moves fast the reference catches up, so peak lag stays bounded and the
stale-force collapse (Prop 1) cannot trigger. This is exactly the paper's peak-lag theory
made into a controller -- a reference-with-bounded-peak-lag.

Testbed: OMWU base + anchor, on random zero-sum games (where R-NaD collapses). For each
method sweep its stress knob and report the collapse-free width. Falsifiable claim: ADA
has NO collapse across its entire knob range, where R-NaD and GARIP both have one.
"""
from __future__ import annotations

import numpy as np

ETA = 0.3


def sm(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def run(A, method, T, beta=0.1, rho=0.01, K=200, D=0.3, rho_max=0.5,
        L0=60.0, k=0.02, gmax=3.0, seed=0):
    m, n = A.shape
    rng = np.random.default_rng(seed)
    x, y = sm(rng.standard_normal(m)), sm(rng.standard_normal(n))
    ax, ay = x.copy(), y.copy()      # EMA average (GARIP / ADA / ADAEXT)
    ax2, ay2 = x.copy(), y.copy()    # EMA-of-average (for anticipation)
    sx, sy = x.copy(), y.copy()      # snapshot (R-NaD)
    gxp, gyp = A @ y, x @ A
    # adaptive anticipation gain: anticipate in proportion to how much the lag 1/rho
    # exceeds a floor L0 -- zero for fast averaging (no overshoot), large for slow.
    gamma = min(gmax, max(0.0, k * (1.0 / rho - L0)))
    for t in range(1, T + 1):
        gx, gy = A @ y, x @ A
        xh = x * np.exp(ETA * (2 * gx - gxp)); xh /= xh.sum()
        yh = y * np.exp(-ETA * (2 * gy - gyp)); yh /= yh.sum()
        if method == "rnad":
            rx, ry = sx, sy
        elif method == "garip":
            rx, ry = ax, ay
        elif method == "ada":
            rx, ry = ax, ay
        elif method == "adaext":
            rx = (1 + gamma) * ax - gamma * ax2     # anticipatory (double-EMA), prob-space
            ry = (1 + gamma) * ay - gamma * ay2
        elif method == "adaextlog":
            # simplex-safe anticipation: extrapolate in LOG space, renormalize -> valid dist
            rx = sm((1 + gamma) * np.log(ax + 1e-12) - gamma * np.log(ax2 + 1e-12))
            ry = sm((1 + gamma) * np.log(ay + 1e-12) - gamma * np.log(ay2 + 1e-12))
        x = (1 - beta) * xh + beta * rx
        y = (1 - beta) * yh + beta * ry
        if method == "ada":
            rx_rate = min(rho_max, rho * max(1.0, np.linalg.norm(x - ax) / D))
            ry_rate = min(rho_max, rho * max(1.0, np.linalg.norm(y - ay) / D))
        else:
            rx_rate = ry_rate = rho
        ax += rx_rate * (x - ax)
        ay += ry_rate * (y - ay)
        ax2 += rho * (ax - ax2)
        ay2 += rho * (ay - ay2)
        if t % K == 0:
            sx, sy = x.copy(), y.copy()
        gxp, gyp = gx, gy
    return float(np.max(A @ y) - np.min(x @ A))


def collapse_rate(A_list, method, knob, val, seeds=6):
    kw = {knob: val}
    # HONEST: a NaN/blow-up is a collapse, not a "safe" run (nan>0.1 is False in numpy).
    def bad(e):
        return (not np.isfinite(e)) or (e > 0.1)
    return np.mean([bad(run(A, method, 5000, seed=s, **kw))
                    for A in A_list for s in range(seeds)])


def main():
    rng = np.random.default_rng(0)
    A_list = [rng.standard_normal((d, d)) for d in (8, 10, 12) for _ in range(2)]
    print("Collapse rate (exploit>0.1), OMWU base, 6 random games x 6 seeds = 36 runs/cell\n")

    print("R-NaD over reset period K (its staleness knob):")
    for K in [100, 200, 400, 800, 1600, 3200]:
        print(f"  K={K:<5} collapse={collapse_rate(A_list,'rnad','K',K):.2f}")

    print("\nGARIP over EMA rate rho (slow rho = stale average):")
    for rho in [0.2, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        print(f"  rho={rho:<6} collapse={collapse_rate(A_list,'garip','rho',rho):.2f}")

    print("\nADA (drift-adaptive) over base rho (should NOT collapse -- lag is capped):")
    for rho in [0.2, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        print(f"  rho0={rho:<6} collapse={collapse_rate(A_list,'ada','rho',rho):.2f}")

    print("\nADAEXT-prob (naive prob-space anticipation -- expect NaN/off-simplex):")
    for rho in [0.05, 0.01, 0.005, 0.001]:
        print(f"  rho={rho:<6} collapse(NaN=fail)={collapse_rate(A_list,'adaext','rho',rho):.2f}")

    print("\nADAEXTLOG (simplex-safe log-space adaptive anticipation):")
    for rho in [0.2, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        print(f"  rho={rho:<6} collapse={collapse_rate(A_list,'adaextlog','rho',rho):.2f}")

    def med(method, **kw):
        v = [run(A, method, 5000, seed=s, **kw) for A in A_list for s in range(6)]
        v = [e for e in v if np.isfinite(e)]
        return np.median(v) if v else float('nan')
    print("\nConvergence check (median finite final exploit):")
    print(f"  R-NaD(K=200)={med('rnad',K=200):.4f}  GARIP(rho=0.05)={med('garip',rho=0.05):.4f}"
          f"  ADAEXTLOG(rho=0.01)={med('adaextlog',rho=0.01):.4f}"
          f"  ADAEXTLOG(rho=0.001)={med('adaextlog',rho=0.001):.4f}")
    print("\nWidth summary: collapse-free knob settings (NaN counts as collapse):")
    rn = sum(collapse_rate(A_list,'rnad','K',K) < 0.05 for K in [100,200,400,800,1600,3200])
    ga = sum(collapse_rate(A_list,'garip','rho',r) < 0.05 for r in [0.2,0.05,0.02,0.01,0.005,0.002,0.001])
    ae = sum(collapse_rate(A_list,'adaextlog','rho',r) < 0.05 for r in [0.2,0.05,0.02,0.01,0.005,0.002,0.001])
    print(f"  R-NaD: {rn}/6 K safe;  GARIP: {ga}/7 rho safe;  ADAEXTLOG: {ae}/7 rho safe")


if __name__ == "__main__":
    main()
