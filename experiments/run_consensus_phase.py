"""Premature-consensus phase boundary: where does a constant anchor freeze the iterate
at a non-Nash consensus before the running average reaches Nash?

For each game size d (antisymmetric d x d, interior Nash) and anchor strength beta, run
tabular GARIP from a fixed start and classify the run as STALLED (premature consensus)
vs CONVERGING by whether last-iterate exploitability is still decreasing at the horizon.
Output: the critical beta*(d) -- the smallest beta that stalls -- as a rough phase
boundary. Expectation: beta*(d) decreases with d (bigger games stall at weaker anchors,
because their average takes longer to reach Nash so consensus wins sooner).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from garip.games import ZeroSumGame
from garip.methods import garip, GaripState
from garip.strategies import to_simplex
from garip.exploitability import exploitability

SIZES = [2, 3, 4, 6, 8, 10, 12]
BETAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
ETA = 0.3


def final_expls(game, beta, seed, halfT):
    """Return exploitability at halfT and 2*halfT (to detect 'still decreasing')."""
    alg = garip(eta=ETA, beta=beta)
    kx, ky = jax.random.split(jax.random.PRNGKey(seed))
    d0, d1 = game.num_row_actions, game.num_col_actions
    x0 = to_simplex(2.0 * jax.random.normal(kx, (d0,)))
    y0 = to_simplex(2.0 * jax.random.normal(ky, (d1,)))
    s = GaripState(x0, y0, x0, y0, game.payoff @ y0, x0 @ game.payoff, jnp.array(1.0))

    @jax.jit
    def advance(s):
        return jax.lax.scan(lambda c, _: (alg.step(c, game), 0), s, None, length=halfT)[0]

    s = advance(s)
    xa, ya = alg.strategies(s)
    e1 = float(exploitability(game.payoff, xa, ya))
    s = advance(s)
    xb, yb = alg.strategies(s)
    e2 = float(exploitability(game.payoff, xb, yb))
    cons = float(jnp.linalg.norm(s.x - s.avg_x) + jnp.linalg.norm(s.y - s.avg_y))
    return e1, e2, cons


def stalled(e1, e2, cons):
    """Premature consensus: exploitability stuck high AND iterate ~ its average."""
    return (e2 > 0.02) and (e2 / max(e1, 1e-12) > 0.7) and (cons < 0.05)


def main():
    halfT = 100_000
    rng = np.random.default_rng(0)
    print(f"Premature-consensus phase boundary (eta={ETA}, horizon={2*halfT}/run)\n")
    print("STALL map (S=premature-consensus stall, .=converging):")
    header = "  d\\beta " + "".join(f"{b:>7}" for b in BETAS)
    print(header)
    crit = {}
    for d in SIZES:
        Z = jnp.array(rng.standard_normal((d, d)))
        game = ZeroSumGame(Z - Z.T, f"a{d}")
        row = []
        for b in BETAS:
            e1, e2, cons = final_expls(game, b, 0, halfT)
            st = stalled(e1, e2, cons)
            row.append("S" if st else ".")
            if st and d not in crit:
                crit[d] = b
        print(f"  {d:>4}   " + "".join(f"{c:>7}" for c in row))
    print("\ncritical beta*(d) = smallest beta that stalls:")
    for d in SIZES:
        print(f"  d={d:>2}: beta* = {crit.get(d, '>0.4 (none stalled)')}")
    print("\n(beta* decreasing in d => premature consensus is a large-anchor / large-game"
          " regime; the default beta=0.02 stays below the boundary on small/medium games.)")


if __name__ == "__main__":
    main()
