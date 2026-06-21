"""Simplex utilities and the two smoothed best-response "generators" G and F.

These are the heart of the CycleGAN analogy: `G` maps the row player's strategy to
the column player's (smoothed) best response, and `F` maps the column player's
strategy back to the row player's best response. Their composition `F o G` is a map
`Delta_m -> Delta_m` whose fixed point is the (quantal-response) equilibrium.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def to_simplex(logits: jax.Array) -> jax.Array:
    """Map unconstrained logits to a point on the probability simplex."""
    return jax.nn.softmax(logits, axis=-1)


def col_best_response(payoff: jax.Array, x: jax.Array, tau: float) -> jax.Array:
    """G: smoothed best response of the *column* (minimizing) player to row `x`.

    The column player minimizes `x^T A y`, so it favors columns with small payoff
    `(x^T A)_j`. With temperature `tau -> 0` this approaches the hard arg-min.
    """
    col_payoff = x @ payoff  # shape (n,): expected payoff of each column vs x
    return jax.nn.softmax(-col_payoff / tau, axis=-1)


def row_best_response(payoff: jax.Array, y: jax.Array, tau: float) -> jax.Array:
    """F: smoothed best response of the *row* (maximizing) player to column `y`.

    The row player maximizes `x^T A y`, so it favors rows with large payoff
    `(A y)_i`. With `tau -> 0` this approaches the hard arg-max.
    """
    row_payoff = payoff @ y  # shape (m,): expected payoff of each row vs y
    return jax.nn.softmax(row_payoff / tau, axis=-1)


def hard_col_best_response(payoff: jax.Array, x: jax.Array) -> jax.Array:
    """One-hot arg-min best response of the column player (used by fictitious play)."""
    col_payoff = x @ payoff
    return jax.nn.one_hot(jnp.argmin(col_payoff), payoff.shape[1])


def hard_row_best_response(payoff: jax.Array, y: jax.Array) -> jax.Array:
    """One-hot arg-max best response of the row player (used by fictitious play)."""
    row_payoff = payoff @ y
    return jax.nn.one_hot(jnp.argmax(row_payoff), payoff.shape[0])
