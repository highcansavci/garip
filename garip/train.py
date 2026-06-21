"""Training loop: roll out a learning dynamic and record its exploitability."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from garip.games import ZeroSumGame
from garip.methods import Algorithm
from garip.exploitability import exploitability


def run(algo: Algorithm, game: ZeroSumGame, steps: int, key: jax.Array):
    """Run `algo` on `game` for `steps` iterations from a single seed.

    Returns:
        expl: `(steps + 1,)` array of exploitability, including the initial point.
        final_x, final_y: the last-iterate mixed strategies.
    """

    def scan_step(state, _):
        state = algo.step(state, game)
        x, y = algo.strategies(state)
        return state, exploitability(game.payoff, x, y)

    @jax.jit
    def rollout(k):
        state0 = algo.init(game, k)
        x0, y0 = algo.strategies(state0)
        e0 = exploitability(game.payoff, x0, y0)
        final_state, expls = jax.lax.scan(scan_step, state0, None, length=steps)
        fx, fy = algo.strategies(final_state)
        return jnp.concatenate([e0[None], expls]), fx, fy

    return rollout(key)


def run_seeds(algo: Algorithm, game: ZeroSumGame, steps: int, keys: jax.Array):
    """Vectorize `run` over a batch of seeds. `keys` has shape `(num_seeds, 2)`.

    Returns:
        expl: `(num_seeds, steps + 1)` exploitability curves.
        final_x, final_y: per-seed last-iterate strategies.
    """

    def scan_step(state, _):
        state = algo.step(state, game)
        x, y = algo.strategies(state)
        return state, exploitability(game.payoff, x, y)

    @partial(jax.jit)
    @partial(jax.vmap)
    def rollout(k):
        state0 = algo.init(game, k)
        x0, y0 = algo.strategies(state0)
        e0 = exploitability(game.payoff, x0, y0)
        final_state, expls = jax.lax.scan(scan_step, state0, None, length=steps)
        fx, fy = algo.strategies(final_state)
        return jnp.concatenate([e0[None], expls]), fx, fy

    return rollout(keys)


def strategy_trajectory(algo: Algorithm, game: ZeroSumGame, steps: int, key: jax.Array):
    """Record the row player's strategy at every step (for simplex trajectory plots)."""

    def scan_step(state, _):
        state = algo.step(state, game)
        x, _y = algo.strategies(state)
        return state, x

    @jax.jit
    def rollout(k):
        state0 = algo.init(game, k)
        x0, _y0 = algo.strategies(state0)
        final_state, xs = jax.lax.scan(scan_step, state0, None, length=steps)
        return jnp.concatenate([x0[None], xs], axis=0)

    return rollout(key)
