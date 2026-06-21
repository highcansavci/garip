"""Numerical verification of the LOCAL last-iterate proposition for GARIP on a
bilinear 2x2 zero-sum game (matching pennies). Three things to confirm:

  1. The last iterate (x_t, y_t) converges to the interior Nash (uniform), and the
     deviation u_t = x_t - xbar_t (policy minus its own running average) contracts
     geometrically at rate ~ (1-beta)|M|, where M is the OMWU one-step Jacobian at
     Nash (|eig M| = 1 - Theta(eta) < 1).
  2. The running average xbar_t -> Nash (closing open step (a) LOCALLY): the only
     frozen anchor consistent with u->0 is (M-I)xbar = 0 => xbar = Nash, since M-I
     is invertible (1 is not an eigenvalue of M).
  3. The asymptotic oscillation amplitude DECREASES in beta (the conjecture's 2nd
     clause), and the empirical contraction rate matches (1-beta) * rho(M).

Also: build the FULL linearized coupled Jacobian (OMWU companion + anchor + a
constant-rho average proxy) at Nash and check spectral radius < 1 for a grid of
(beta, rho), i.e. local asymptotic stability of the moving-anchor composition.
"""
from __future__ import annotations

import numpy as np

# ---- matching pennies, payoff to the row player; interior Nash = uniform -------
A = np.array([[1.0, -1.0], [-1.0, 1.0]])
NASH = np.array([0.5, 0.5])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run_garip(eta, beta, T, seed=0, cesaro=True, rho=0.01, optimistic=True):
    """Tabular GARIP exactly as in garip/methods.py. cesaro=True uses the 1/(t+1)
    running average (as in the paper); else a constant-rho EMA anchor. optimistic=False
    drops the 2g-g_prev prediction -> vanilla MWU base, which CYCLES (Poincare
    recurrence) without the anchor: the regime where the anchor is load-bearing."""
    rng = np.random.default_rng(seed)
    x = softmax(rng.normal(size=2) * 0.3)
    y = softmax(rng.normal(size=2) * 0.3)
    ax, ay = x.copy(), y.copy()
    gx_prev, gy_prev = A @ y, x @ A
    xs, axs = [], []
    for t in range(1, T + 1):
        gx, gy = A @ y, x @ A
        px = (2 * gx - gx_prev) if optimistic else gx
        py = (2 * gy - gy_prev) if optimistic else gy
        xh = x * np.exp(eta * px); xh /= xh.sum()
        yh = y * np.exp(-eta * py); yh /= yh.sum()
        x = (1 - beta) * xh + beta * ax
        y = (1 - beta) * yh + beta * ay
        w = 1.0 / (t + 1) if cesaro else rho
        ax = ax + w * (x - ax)
        ay = ay + w * (y - ay)
        gx_prev, gy_prev = gx, gy
        xs.append(x.copy()); axs.append(ax.copy())
    return np.array(xs), np.array(axs)


def omwu_jacobian(eta):
    """One-step OMWU Jacobian at the uniform Nash of matching pennies, in the 1-D
    reduced coordinate (x = [p, 1-p]) including the lagged gradient. State (dp, dg):
    half-step log-ratio uses 2 g_t - g_{t-1}. Linearizing softmax at uniform gives
    d(logit) gain 1, and g_x = A y depends on the column deviation, which (by the
    symmetric saddle structure) couples to the row through the rotation. We return
    the 2x2 companion [[1, c],[?]] numerically via finite differences on the joint
    4-state (dp_x, dp_y, dg_x, dg_y) map -- the cleanest correct route."""
    raise NotImplementedError  # done below via finite-difference of the true map


def full_jacobian_fd(eta, beta, rho, h=1e-6, optimistic=True):
    """Finite-difference Jacobian of the full GARIP map (constant-rho EMA anchor) at
    Nash, over the state s = (p_x, p_y, gx, gy, abar_x, abar_y) reduced to scalars
    p (since 2x2). Returns the 6x6 (here 4 active) Jacobian's spectral radius."""
    # state: x logit-free reduced to p_x in [0,1]; we carry full 2-vectors but the
    # simplex is 1-D, so track (px, py, gx0, gy0, apx, apy) where g*0 is first comp.
    def to_state(px, py, gx, gy, apx, apy):
        return np.array([px, py, gx, gy, apx, apy])

    def step_state(s):
        px, py, gx_prev_0, gy_prev_0, apx, apy = s
        x = np.array([px, 1 - px]); y = np.array([py, 1 - py])
        ax = np.array([apx, 1 - apx]); ay = np.array([apy, 1 - apy])
        gx = A @ y; gy = x @ A
        gx_prev = np.array([gx_prev_0, -gx_prev_0])  # matching pennies: gx = [d,-d]
        gy_prev = np.array([gy_prev_0, -gy_prev_0])
        pgx = (2 * gx - gx_prev) if optimistic else gx
        pgy = (2 * gy - gy_prev) if optimistic else gy
        xh = x * np.exp(eta * pgx); xh /= xh.sum()
        yh = y * np.exp(-eta * pgy); yh /= yh.sum()
        xn = (1 - beta) * xh + beta * ax
        yn = (1 - beta) * yh + beta * ay
        apxn = apx + rho * (xn[0] - apx)
        apyn = apy + rho * (yn[0] - apy)
        return to_state(xn[0], yn[0], gx[0], gy[0], apxn, apyn)

    s0 = to_state(0.5, 0.5, 0.0, 0.0, 0.5, 0.5)
    # confirm fixed point
    assert np.allclose(step_state(s0), s0, atol=1e-9), step_state(s0) - s0
    n = len(s0)
    J = np.zeros((n, n))
    for j in range(n):
        sp = s0.copy(); sp[j] += h
        sm = s0.copy(); sm[j] -= h
        J[:, j] = (step_state(sp) - step_state(sm)) / (2 * h)
    eig = np.linalg.eigvals(J)
    return np.max(np.abs(eig)), eig


def main():
    eta = 0.1
    print(f"=== matching pennies, eta={eta} ===\n")

    # 1-3: trajectories for a few beta, Cesaro average (paper's exact dynamics)
    print("LAST-ITERATE CONVERGENCE (Cesaro running average, as in the paper):")
    print(f"{'beta':>6} {'final |x-Nash|':>16} {'final |xbar-Nash|':>18} "
          f"{'u-contraction':>14} {'(1-b)*rho(M)':>13}")
    # estimate rho(M) from the beta=0 (pure OMWU) decay of |x-Nash|
    xs0, _ = run_garip(eta, 0.0, 4000, cesaro=True)
    d0 = np.linalg.norm(xs0 - NASH, axis=1)
    # geometric rate over a clean window
    lo, hi = 500, 3500
    rate_M = np.exp(np.polyfit(np.arange(hi - lo), np.log(d0[lo:hi] + 1e-300), 1)[0])
    print(f"  (estimated rho(M) from pure OMWU, beta=0): {rate_M:.5f}\n")

    for beta in [0.0, 0.05, 0.2, 0.5]:
        xs, axs = run_garip(eta, beta, 6000, cesaro=True)
        dx = np.linalg.norm(xs - NASH, axis=1)
        dax = np.linalg.norm(axs - NASH, axis=1)
        u = np.linalg.norm(xs - axs, axis=1)
        lo, hi = 1000, 5000
        urate = np.exp(np.polyfit(np.arange(hi - lo), np.log(u[lo:hi] + 1e-300), 1)[0])
        print(f"{beta:6.2f} {dx[-1]:16.2e} {dax[-1]:18.2e} "
              f"{urate:14.5f} {(1 - beta) * rate_M:13.5f}")

    # oscillation amplitude vs beta (peak-to-peak of x over a late window)
    print("\nOSCILLATION AMPLITUDE vs beta (late-window peak-to-peak of x_0):")
    for beta in [0.0, 0.05, 0.2, 0.5]:
        xs, _ = run_garip(eta, beta, 6000, cesaro=True)
        amp = xs[3000:, 0].max() - xs[3000:, 0].min()
        print(f"  beta={beta:.2f}: amplitude={amp:.3e}")

    # --- THE LOAD-BEARING REGIME: non-optimistic (cycling) base -----------------
    # Vanilla MWU cycles (Poincare recurrence) in zero-sum games: the last iterate
    # does NOT converge. Here the anchor is what *induces* convergence. This is the
    # tabular analog of the deep-RL collapse story (PPO cycles; the magnet stabilizes).
    print("\n=== CYCLING BASE (vanilla MWU, no optimism): anchor is load-bearing ===")
    print(f"{'beta':>6} {'final |x-Nash|':>16} {'late osc amplitude':>20}")
    for beta in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
        xs, _ = run_garip(eta, beta, 20000, cesaro=True, optimistic=False)
        dx = np.linalg.norm(xs[-1] - NASH)
        amp = xs[10000:, 0].max() - xs[10000:, 0].min()
        print(f"{beta:6.2f} {dx:16.2e} {amp:20.3e}")
    print("(beta=0 = pure MWU should CYCLE (large, non-vanishing amplitude);\n"
          " beta>0 should converge with amplitude DECREASING in beta.)")

    # 4: full linearized Jacobian spectral radius over (beta, rho) -- local stability
    print("\nLOCAL STABILITY: spectral radius of full linearized map (EMA anchor):")
    print(f"{'beta|rho':>10}", *[f"{r:>9}" for r in [0.005, 0.01, 0.05, 0.1]])
    for beta in [0.0, 0.05, 0.2, 0.5, 0.9]:
        row = []
        for rho in [0.005, 0.01, 0.05, 0.1]:
            sr, _ = full_jacobian_fd(eta, beta, rho)
            row.append(f"{sr:9.5f}")
        print(f"{beta:10.2f}", *row)
    print("\n(all entries < 1  =>  Nash is locally asymptotically stable in the last "
          "iterate\n for every beta in [0,1), rho in (0,1] -- the local last-iterate "
          "proposition.)")

    # high-precision check of the borderline large-beta entries (optimistic base):
    # is rho(J) STRICTLY < 1, or does the slow mode touch/exceed 1?
    print("\nHIGH-PRECISION slow-mode check (optimistic base), 1 - rho(J):")
    for base, opt in [("optimistic", True), ("cycling", False)]:
        print(f"  base={base}:")
        for beta in [0.5, 0.9, 0.99]:
            for rho in [0.005, 0.01, 0.05]:
                sr, _ = full_jacobian_fd(eta, beta, rho, optimistic=opt)
                print(f"    beta={beta:.2f} rho={rho:.3f}: rho(J)={sr:.9f}  "
                      f"1-rho(J)={1 - sr:+.2e}")

    # ---- the analytic mechanism: base is a rotation; anchor scales it by (1-beta) --
    print("\nMECHANISM CHECK (cycling base, EMA rho=0.01):")
    print(f"{'beta':>6} {'|mu_base| (rotation)':>20} {'rho(J) full':>13} "
          f"{'(1-beta)|mu|':>13} {'1-rho':>8}")
    rho = 0.01
    # base modulus = spectral radius at beta=0, very small rho (anchor frozen) on the
    # active policy+gradient modes (exclude the trivial slow-average eigenvalue ~1).
    _, eig0 = full_jacobian_fd(eta, 0.0, 1e-7, optimistic=False)
    mu_base = sorted(np.abs(eig0))[-2]  # largest non-average mode
    for beta in [0.0, 0.05, 0.2, 0.5, 0.9]:
        sr, _ = full_jacobian_fd(eta, beta, rho, optimistic=False)
        print(f"{beta:6.2f} {mu_base:20.5f} {sr:13.5f} "
              f"{(1 - beta) * mu_base:13.5f} {1 - rho:8.5f}")
    print("(|mu_base| ~ 1 = the base rotation/recurrence; the anchor scales the fast "
          "mode\n to (1-beta)|mu_base|<1, while the slow average mode sits near 1-rho.)")


if __name__ == "__main__":
    main()
