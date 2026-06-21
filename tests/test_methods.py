import jax
import jax.numpy as jnp
import pytest

from garip.games import rps, random_zero_sum
from garip import methods
from garip.train import run


# Long enough for CGSP's tau-annealing (anneal_steps=4000) to finish and settle.
STEPS = 5000


@pytest.mark.parametrize("algo", methods.all_methods(), ids=lambda a: a.name)
def test_step_runs_and_is_jittable(algo):
    game = rps()
    expl, fx, fy = run(algo, game, steps=50, key=jax.random.PRNGKey(0))
    assert expl.shape == (51,)
    assert jnp.all(jnp.isfinite(expl))
    # strategies stay on the simplex
    assert jnp.allclose(jnp.sum(fx), 1.0, atol=1e-4)
    assert jnp.allclose(jnp.sum(fy), 1.0, atol=1e-4)
    assert jnp.all(fx >= -1e-6) and jnp.all(fy >= -1e-6)


def _final_expl(algo, game, key=jax.random.PRNGKey(0)):
    expl, _, _ = run(algo, game, steps=STEPS, key=key)
    return float(expl[-1])


def test_cgsp_converges_on_rps():
    # The core claim: cycle-consistency drives last-iterate exploitability toward 0.
    assert _final_expl(methods.cgsp(), rps()) < 0.05


def test_baselines_that_should_converge_do():
    assert _final_expl(methods.fictitious_play(), rps()) < 0.1
    assert _final_expl(methods.mirror_descent(optimistic=True), rps()) < 0.05


def test_sga_does_not_converge_in_last_iterate():
    # Naive self-play gradient ascent keeps cycling: its last-iterate exploitability
    # stays well above CGSP's. This is the pathology CGSP is designed to fix.
    sga_expl = _final_expl(methods.sga(), rps())
    cgsp_expl = _final_expl(methods.cgsp(), rps())
    assert sga_expl > cgsp_expl
    assert sga_expl > 0.1


def test_cgsp_converges_on_random_game():
    game = random_zero_sum(jax.random.PRNGKey(7), m=8, n=8)
    assert _final_expl(methods.cgsp(), game) < 0.2


def test_garip_last_iterate_converges_without_annealing():
    # GARIP reaches near-zero last-iterate exploitability on RPS and a random game with a
    # constant anchor strength (no temperature annealing).
    assert _final_expl(methods.garip(), rps()) < 0.02
    assert _final_expl(methods.garip(), random_zero_sum(jax.random.PRNGKey(7), 8, 8)) < 0.05


def test_garip_beats_mmd_fixed_magnet_on_random_game():
    # GARIP's moving anchor reaches Nash where MMD's fixed magnet leaves a quantal bias
    # (at equal, un-annealed regularization).
    game = random_zero_sum(jax.random.PRNGKey(0), m=10, n=10)
    assert _final_expl(methods.garip(), game) < _final_expl(methods.mmd(), game)
