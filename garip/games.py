"""Two-player zero-sum games, represented by a single payoff matrix.

Convention: the row player is the *maximizer* and the column player is the
*minimizer* of `V = x^T A y`, where `x` is the row mixed strategy (in the simplex
`Delta_m`) and `y` is the column mixed strategy (`Delta_n`). For symmetric games we
store an antisymmetric `A` (`A = -A^T`), whose unique Nash value is 0.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ZeroSumGame(NamedTuple):
    """A two-player zero-sum game.

    Attributes:
        payoff: `(m, n)` payoff matrix `A`. Row player maximizes `x^T A y`,
            column player minimizes it.
        name: Human-readable identifier used in logs/plots.
    """

    payoff: jax.Array
    name: str

    @property
    def num_row_actions(self) -> int:
        return self.payoff.shape[0]

    @property
    def num_col_actions(self) -> int:
        return self.payoff.shape[1]


def rps() -> ZeroSumGame:
    """Rock-Paper-Scissors. Unique Nash is uniform `(1/3, 1/3, 1/3)`, value 0.

    The canonical non-transitive game on which naive self-play cycles forever.
    """
    a = jnp.array(
        [[0.0, -1.0, 1.0],
         [1.0, 0.0, -1.0],
         [-1.0, 1.0, 0.0]]
    )
    return ZeroSumGame(payoff=a, name="rps")


def matching_pennies() -> ZeroSumGame:
    """Matching Pennies. Unique Nash is uniform `(1/2, 1/2)`, value 0."""
    a = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    return ZeroSumGame(payoff=a, name="matching_pennies")


def random_zero_sum(key: jax.Array, m: int = 10, n: int = 10) -> ZeroSumGame:
    """A random general (not necessarily symmetric) zero-sum game.

    The payoff entries are i.i.d. standard normal. Such games still have a
    well-defined Nash equilibrium and value (von Neumann), but exploitability
    (our convergence metric) needs neither — only best-response values.
    """
    a = jax.random.normal(key, (m, n))
    return ZeroSumGame(payoff=a, name=f"random_{m}x{n}")
