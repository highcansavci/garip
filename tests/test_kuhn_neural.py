"""Smoke tests for the neural CGSP training on Kuhn poker (kept short for speed)."""
import jax
import jax.numpy as jnp

from garip import kuhn
import experiments.run_kuhn as rk


def test_cycle_loss_is_finite_and_nonneg():
    from tests.test_kuhn import _analytic_nash
    s1, s2 = _analytic_nash()
    uni = jnp.full((6, 2), 0.5)
    for a, b in ((s1, s2), (uni, uni)):
        c = float(kuhn.cycle_loss(a, b, tau=0.1))
        assert jnp.isfinite(c) and c >= 0.0


def test_smoothed_br_p2_matches_exact_at_low_tau():
    # Player 2's smoothed BR is exact one-shot, so at tiny tau it should agree with
    # the hard best response on the action it selects.
    key = jax.random.PRNGKey(0)
    s1 = jax.nn.softmax(jax.random.normal(key, (6, 2)), axis=1)
    soft = kuhn.smoothed_br_p2(s1, tau=0.01)
    hard = kuhn.best_response_p2(s1)
    assert jnp.allclose(jnp.argmax(soft, axis=1), jnp.argmax(hard, axis=1))


def test_cgsp_reduces_exploitability():
    expl = rk.run_gradient(lam=1.0, steps=400, key=jax.random.PRNGKey(0))
    assert jnp.all(jnp.isfinite(expl))
    assert float(expl[-1]) < float(expl[0])


def test_cgsp_beats_naive_self_play():
    # Use a fast anneal schedule so CGSP converges within a short run; naive self-play
    # keeps oscillating high (it never drops below ~0.4 in the full experiment).
    saved = (rk.TAU_FINAL, rk.ANNEAL_STEPS, rk.LR)
    rk.TAU_FINAL, rk.ANNEAL_STEPS, rk.LR = 0.05, 900, 0.02
    try:
        key = jax.random.PRNGKey(1)
        cgsp_expl = rk.run_gradient(lam=1.0, steps=1500, key=key)
        sga_expl = rk.run_gradient(lam=0.0, steps=1500, key=key)
    finally:
        rk.TAU_FINAL, rk.ANNEAL_STEPS, rk.LR = saved
    assert float(cgsp_expl[-1]) < 0.2          # CGSP converges
    assert float(cgsp_expl[-1]) < float(sga_expl[-1])  # and beats naive self-play
