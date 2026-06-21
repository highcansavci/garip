import jax
import jax.numpy as jnp

from garip.games import rps, matching_pennies, random_zero_sum


def test_rps_is_antisymmetric():
    # RPS is a symmetric game with an antisymmetric payoff matrix (A = -A^T).
    a = rps().payoff
    assert jnp.allclose(a, -a.T)


def test_known_games_have_zero_value_at_uniform():
    # Both RPS and matching pennies have value 0, attained at the uniform Nash.
    for game, k in ((rps(), 3), (matching_pennies(), 2)):
        uniform = jnp.ones(k) / k
        assert jnp.allclose(uniform @ game.payoff @ uniform, 0.0, atol=1e-6), game.name


def test_shapes():
    assert rps().payoff.shape == (3, 3)
    assert matching_pennies().payoff.shape == (2, 2)
    g = random_zero_sum(jax.random.PRNGKey(0), m=7, n=4)
    assert g.payoff.shape == (7, 4)
    assert g.num_row_actions == 7 and g.num_col_actions == 4
