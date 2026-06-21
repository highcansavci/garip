"""Smoke test for neural CGSP training on Leduc (kept short)."""
import jax

import experiments.run_leduc as rl


def test_cgsp_reduces_leduc_exploitability():
    # A short averaged-target CGSP run should pull exploitability far below the
    # uniform-strategy value (~4.5) -- evidence the cycle term scales to Leduc.
    it, last, avg = rl.run_gradient(lam=1.0, steps=1500, eval_every=500,
                                    key=jax.random.PRNGKey(0))
    assert avg[0] > 2.0          # starts near uniform
    assert avg[-1] < 1.5         # averaged strategy improves substantially


def test_sga_does_not_converge_on_leduc():
    # Naive neural self-play stays highly exploitable (it diverges/cycles).
    it, last, avg = rl.run_gradient(lam=0.0, steps=1500, eval_every=500,
                                    key=jax.random.PRNGKey(0))
    assert last[-1] > 2.0
