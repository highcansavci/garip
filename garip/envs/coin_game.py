"""Coin Game — the canonical 2-player self-play testbed, vendored from JaxMARL.

Ported directly from `jaxmarl/environments/coin_game/coin_game.py` and made fully
self-contained (no `MultiAgentEnv`/`spaces`/`chex` dependency, no version coupling). The
diagnostic statistics of the original are dropped; the mechanics are faithful:

  * 3x3 toroidal grid, two players (red, blue), one coin of each color.
  * Each step both players move (5 actions: right/left/up/down/stay). Landing on a coin
    collects it: +1 to the collector; collecting the *opponent's* coin additionally costs
    the opponent 2 ("stealing"). Collected coins respawn at a random cell.
  * Observations are egocentric (each player sees itself in channel 0), so a single
    shared policy can play either side. Flattened to a 36-vector (3x3x4 channels:
    [self, opponent, self-coin, opponent-coin]).

`zero_sum=True` returns `r0 = red - blue`, `r1 = -r0` (a strict two-player zero-sum
competition over coins, matching CGSP's setting); `zero_sum=False` keeps the original
general-sum social-dilemma rewards.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

NUM_ACTIONS = 5
GRID = 3
OBS_DIM = GRID * GRID * 4  # 36

# right, left, up, down, stay
MOVES = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])

# Original social-dilemma payoffs: collect own coin +1, steal opponent coin +1 (and the
# victim loses 2).
COLLECT = 1.0
STEAL_PENALTY = -2.0


class CoinState(NamedTuple):
    red_pos: jax.Array       # (2,)
    blue_pos: jax.Array      # (2,)
    red_coin: jax.Array      # (2,)
    blue_coin: jax.Array     # (2,)
    t: jax.Array             # scalar step counter within the episode


def _grid_obs(red_pos, blue_pos, red_coin, blue_coin):
    """Build the (3,3,4) tensor from the red player's perspective, then both egocentric views."""
    g = jnp.zeros((GRID, GRID, 4))
    g = g.at[red_pos[0], red_pos[1], 0].set(1.0)
    g = g.at[blue_pos[0], blue_pos[1], 1].set(1.0)
    g = g.at[red_coin[0], red_coin[1], 2].set(1.0)
    g = g.at[blue_coin[0], blue_coin[1], 3].set(1.0)
    obs_red = g.reshape(-1)
    # Blue sees itself as "red": swap player channels (0<->1) and coin channels (2<->3).
    obs_blue = g[:, :, jnp.array([1, 0, 3, 2])].reshape(-1)
    return obs_red, obs_blue


class CoinGame:
    """Stateless functional env: `reset(key)` and `step(key, state, a0, a1)`.

    All methods are pure and jit/vmap-friendly. `step` auto-resets at the episode
    boundary (returning the next episode's first observation with `done=True`).
    """

    num_actions = NUM_ACTIONS
    obs_dim = OBS_DIM

    def __init__(self, episode_length: int = 16, zero_sum: bool = True):
        self.episode_length = episode_length
        self.zero_sum = zero_sum

    def reset(self, key):
        pos = jax.random.randint(key, (4, 2), 0, GRID)
        state = CoinState(pos[0], pos[1], pos[2], pos[3], jnp.array(0))
        obs0, obs1 = _grid_obs(state.red_pos, state.blue_pos, state.red_coin, state.blue_coin)
        return (obs0, obs1), state

    def step(self, key, state: CoinState, a0, a1):
        new_red = (state.red_pos + MOVES[a0]) % GRID
        new_blue = (state.blue_pos + MOVES[a1]) % GRID

        rr = jnp.all(new_red == state.red_coin)      # red collects own coin
        rb = jnp.all(new_red == state.blue_coin)     # red steals blue's coin
        br = jnp.all(new_blue == state.red_coin)      # blue steals red's coin
        bb = jnp.all(new_blue == state.blue_coin)     # blue collects own coin

        red_r = rr * COLLECT + rb * COLLECT + br * STEAL_PENALTY
        blue_r = bb * COLLECT + br * COLLECT + rb * STEAL_PENALTY

        # Respawn collected coins.
        key, ck = jax.random.split(key)
        new_coins = jax.random.randint(ck, (2, 2), 0, GRID)
        red_coin = jnp.where(rr | br, new_coins[0], state.red_coin)
        blue_coin = jnp.where(rb | bb, new_coins[1], state.blue_coin)

        t1 = state.t + 1
        done = t1 >= self.episode_length

        cont_state = CoinState(new_red, new_blue, red_coin, blue_coin, t1)
        key, rk = jax.random.split(key)
        _, reset_state = self.reset(rk)
        next_state = jax.tree.map(lambda c, r: jnp.where(done, r, c), cont_state, reset_state)

        obs0, obs1 = _grid_obs(next_state.red_pos, next_state.blue_pos,
                               next_state.red_coin, next_state.blue_coin)

        if self.zero_sum:
            r0 = red_r - blue_r
            r1 = -r0
        else:
            r0, r1 = red_r, blue_r

        info = {
            "red_collect": rr, "blue_collect": bb,
            "red_steal": rb, "blue_steal": br,
        }
        return (obs0, obs1), next_state, (r0, r1), done, info
