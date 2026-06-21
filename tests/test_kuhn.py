import jax.numpy as jnp

from garip import kuhn


def _analytic_nash(xi=1.0 / 3.0):
    """A Kuhn-poker Nash equilibrium parameterized by the player-1 bluff rate xi.

    Player 1: bet J with prob xi, never bet Q, bet K with prob 3*xi at the start;
    at "checked-then-bet" call Q with prob xi+1/3, never J, always K.
    Player 2: facing a bet, call Q with prob 1/3, always K; facing a check, bet J with
    prob 1/3, always K. (Standard equilibrium family, see Kuhn 1950.)
    """
    s1 = jnp.array([
        [1 - xi, xi],            # (J, "")  bet with prob xi
        [1.0, 0.0],              # (Q, "")  never bet
        [1 - 3 * xi, 3 * xi],    # (K, "")  bet with prob 3*xi
        [1.0, 0.0],              # (J, "pb") never call
        [1 - (xi + 1 / 3), xi + 1 / 3],  # (Q, "pb") call with prob xi+1/3
        [0.0, 1.0],              # (K, "pb") always call
    ])
    s2 = jnp.array([
        [2 / 3, 1 / 3],          # (J, "p") bet with prob 1/3
        [1.0, 0.0],              # (Q, "p") never bet
        [0.0, 1.0],              # (K, "p") always bet
        [1.0, 0.0],              # (J, "b") never call
        [2 / 3, 1 / 3],          # (Q, "b") call with prob 1/3
        [0.0, 1.0],              # (K, "b") always call
    ])
    return s1, s2


def test_strategies_are_rowstochastic():
    s1, s2 = _analytic_nash()
    assert jnp.allclose(s1.sum(axis=1), 1.0)
    assert jnp.allclose(s2.sum(axis=1), 1.0)


def test_game_value_is_minus_one_eighteenth():
    s1, s2 = _analytic_nash()
    assert jnp.allclose(kuhn.ev(s1, s2), -1.0 / 18.0, atol=1e-4)


def test_nash_has_zero_exploitability():
    s1, s2 = _analytic_nash()
    assert float(kuhn.exploitability(s1, s2)) < 1e-3


def test_uniform_strategy_is_exploitable():
    s = jnp.full((kuhn.NUM_INFOSETS, kuhn.NUM_ACTIONS), 0.5)
    assert float(kuhn.exploitability(s, s)) > 0.1


def test_exploitability_nonnegative_random():
    import jax
    for seed in range(5):
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        s1 = jax.nn.softmax(jax.random.normal(k1, (6, 2)), axis=1)
        s2 = jax.nn.softmax(jax.random.normal(k2, (6, 2)), axis=1)
        assert float(kuhn.exploitability(s1, s2)) >= -1e-6
