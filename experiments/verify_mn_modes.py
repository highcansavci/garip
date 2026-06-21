"""Validate the two upgrades to the local last-iterate proposition:

  (1) m x n generalization. The anchor and running average are SCALAR (coordinate-wise
      identical) linear operators, so they commute with the base (O)MWU operator; in a
      basis diagonalizing the base at the interior Nash into modes M_k, the full
      linearized map block-diagonalizes into independent 2x2 blocks J(M_k). Check:
      spectral radius of the full FD Jacobian for a 3x3 (RPS) and a random 4x4 zero-sum
      game equals max_k rho(J(M_k)) over the base eigenvalues M_k.

  (2) Analytic rho>0 slow mode. The marginal eigenvalue 1 (at rho=0) perturbs to
          lambda_slow = 1 - rho * (1-beta)(1-M) / (1-(1-beta)M) + O(rho^2),
      a COMPLEX coefficient (M is a rotation). Check this first-order form against the
      exact 2x2-block root, and confirm |lambda_slow| < 1 <=> Re[(1-b)(1-M)/(1-(1-b)M)]>0
      (true for small eta). The simpler real guess 1 - rho(1-|M|)(1-beta) is shown to be
      quantitatively wrong.
"""
from __future__ import annotations

import numpy as np

RPS = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def block(M, beta, rho):
    """The per-mode 2x2 block J(M)."""
    return np.array([[(1 - beta) * M, beta],
                     [rho * (1 - beta) * M, 1 - rho * (1 - beta)]], dtype=complex)


def base_jacobian(A, eta, optimistic, h=1e-6):
    """FD Jacobian of the pure base (O)MWU map on (x, y, gx_prev, gy_prev) at the
    uniform interior Nash -- no anchor, no average. Returns its eigenvalues (the M_k)."""
    m, n = A.shape
    xs, ys = np.ones(m) / m, np.ones(n) / n
    gx0, gy0 = A @ ys, xs @ A
    s0 = np.concatenate([xs, ys, gx0, gy0])

    def step(s):
        x, y = s[:m], s[m:m + n]
        gxp, gyp = s[m + n:2 * m + n], s[2 * m + n:]
        gx, gy = A @ y, x @ A
        px = (2 * gx - gxp) if optimistic else gx
        py = (2 * gy - gyp) if optimistic else gy
        xh = x * np.exp(eta * px); xh /= xh.sum()
        yh = y * np.exp(-eta * py); yh /= yh.sum()
        return np.concatenate([xh, yh, gx, gy])

    assert np.allclose(step(s0), s0, atol=1e-9)
    d = len(s0)
    J = np.zeros((d, d))
    for j in range(d):
        sp = s0.copy(); sp[j] += h
        sm = s0.copy(); sm[j] -= h
        J[:, j] = (step(sp) - step(sm)) / (2 * h)
    return np.linalg.eigvals(J)


def full_jacobian(A, eta, beta, rho, optimistic, h=1e-6):
    """FD Jacobian of the full GARIP map on (x, y, gx_prev, gy_prev, ax, ay)."""
    m, n = A.shape
    xs, ys = np.ones(m) / m, np.ones(n) / n
    gx0, gy0 = A @ ys, xs @ A
    s0 = np.concatenate([xs, ys, gx0, gy0, xs, ys])

    def step(s):
        x, y = s[:m], s[m:m + n]
        gxp, gyp = s[m + n:2 * m + n], s[2 * m + n:2 * m + 2 * n]
        ax, ay = s[2 * m + 2 * n:3 * m + 2 * n], s[3 * m + 2 * n:]
        gx, gy = A @ y, x @ A
        px = (2 * gx - gxp) if optimistic else gx
        py = (2 * gy - gyp) if optimistic else gy
        xh = x * np.exp(eta * px); xh /= xh.sum()
        yh = y * np.exp(-eta * py); yh /= yh.sum()
        xn = (1 - beta) * xh + beta * ax
        yn = (1 - beta) * yh + beta * ay
        axn = ax + rho * (xn - ax)
        ayn = ay + rho * (yn - ay)
        return np.concatenate([xn, yn, gx, gy, axn, ayn])

    assert np.allclose(step(s0), s0, atol=1e-9)
    d = len(s0)
    J = np.zeros((d, d))
    for j in range(d):
        sp = s0.copy(); sp[j] += h
        sm = s0.copy(); sm[j] -= h
        J[:, j] = (step(sp) - step(sm)) / (2 * h)
    return np.linalg.eigvals(J)


def main():
    eta, beta, rho = 0.1, 0.3, 0.02
    rng = np.random.default_rng(0)
    R4 = rng.standard_normal((4, 4)); R4 = R4 - R4.T
    # double-center -> zero row/col sums, so uniform is the interior Nash (still antisym)
    R4 = R4 - R4.mean(1, keepdims=True) - R4.mean(0, keepdims=True) + R4.mean()

    print("=== (1) m x n PER-MODE REDUCTION: rho(full FD) vs max_k rho(J(M_k)) ===")
    print(f"(eta={eta}, beta={beta}, rho={rho})\n")

    def disagreement(A, opt):
        Mk = base_jacobian(A, eta, opt)
        Mk = Mk[np.abs(Mk) > 1e-3]  # drop the softmax-null (all-ones) modes
        per_mode_sr = max(np.max(np.abs(np.linalg.eigvals(block(M, beta, rho)))) for M in Mk)
        full_sr = np.max(np.abs(full_jacobian(A, eta, beta, rho, opt)))
        return abs(per_mode_sr - full_sr)

    def rand_zerosum(d, seed):
        r = np.random.default_rng(seed)
        Z = r.standard_normal((d, d)); Z = Z - Z.T
        return Z - Z.mean(1, keepdims=True) - Z.mean(0, keepdims=True) + Z.mean()

    # size sweep: square zero-sum games (double-centered antisym -> uniform interior Nash)
    REPS = 10
    print(f"{'size':>5} {'one-step base (MWU)':>24} {'optimistic base (OMWU)':>26}")
    print(f"{'':>5} {'max |rho(full)-max_k rho|':>24} {'max |rho(full)-max_k rho|':>26}")
    worst_cyc, worst_opt, biggest = 0.0, 0.0, 0
    for d in [3, 4, 5, 6, 8]:
        cyc = max(disagreement(rand_zerosum(d, s), False) for s in range(REPS))
        opt = max(disagreement(rand_zerosum(d, s), True) for s in range(REPS))
        worst_cyc, worst_opt, biggest = max(worst_cyc, cyc), max(worst_opt, opt), d
        print(f"{d:>5} {cyc:>24.2e} {opt:>26.2e}")
    print(f"\n  one-step base: agreement to {worst_cyc:.0e} across sizes up to {biggest}x{biggest},"
          f" {REPS} payoffs each\n  => the 2x2 block-diagonalization is EXACT (analytic); "
          f"numerics only confirm it to FD precision.")
    print(f"  optimistic base: agreement to {worst_opt:.0e} (lagged gradient -> 3x3 per-mode "
          f"block; 2x2 is an approximation).")

    print("\n=== (2) ANALYTIC SLOW MODE: lambda_slow = 1 - rho*(1-b)(1-M)/(1-(1-b)M) ===\n")
    # use a representative rotation mode M from RPS optimistic
    Mk = base_jacobian(RPS, eta, True)
    Mk = Mk[np.abs(Mk) > 1e-3]
    # pick the mode with largest |Im| (a genuine rotation)
    M = Mk[np.argmax(np.abs(Mk.imag))]
    print(f"representative base mode M = {M:.6f}  (|M|={abs(M):.6f})\n")
    print(f"{'rho':>7} {'exact |lam_slow|':>17} {'1st-order |lam|':>16} "
          f"{'real-guess |lam|':>17}")
    K = (1 - beta) * (1 - M) / (1 - (1 - beta) * M)  # complex coefficient
    for rho_ in [0.0005, 0.002, 0.01, 0.05]:
        ev = np.linalg.eigvals(block(M, beta, rho_))
        lam_slow = ev[np.argmax(np.abs(ev))]  # the near-1 root
        approx = 1 - rho_ * K
        real_guess = 1 - rho_ * (1 - abs(M)) * (1 - beta)
        print(f"{rho_:7.4f} {abs(lam_slow):17.9f} {abs(approx):16.9f} "
              f"{abs(real_guess):17.9f}")
    print(f"\nRe(K) = {K.real:+.6f}  (>0  =>  |lambda_slow|<1).  Small-eta reduction: "
          f"Re(K) ~ (1-b)Re(1-M)/beta = {(1-beta)*(1-M).real/beta:+.6f}")
    print("The complex 1st-order form tracks the exact root; the real guess does not.")


if __name__ == "__main__":
    main()
