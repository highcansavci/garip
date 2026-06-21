"""Validation tests for the Leduc hold'em engine.

The decisive check is that CFR drives exploitability toward 0 and the game value lands
near the known Leduc value (~ -0.0856 to player 0): this can only happen if the tree,
EV, and exact best response are mutually consistent and correct.
"""
import numpy as np
import jax.numpy as jnp

from garip import leduc

G = leduc.GAME


def _uniform(n, legal):
    legal = np.asarray(legal)
    return jnp.asarray(legal / legal.sum(axis=1, keepdims=True))


def test_tree_has_canonical_leduc_size():
    # Standard Leduc has 288 information sets (144 per player).
    assert G.n0 == 144 and G.n1 == 144


def test_features_are_unique_per_infoset():
    for feats, n in ((G.features0, G.n0), (G.features1, G.n1)):
        arr = np.asarray(feats)
        assert len({tuple(row) for row in arr}) == n


def test_exploitability_nonnegative_and_uniform_is_exploitable():
    s0 = _uniform(G.n0, G.legal0)
    s1 = _uniform(G.n1, G.legal1)
    expl = leduc.exploitability(G, s0, s1)
    assert expl > 1.0  # uniform play is very exploitable in Leduc


def test_cfr_converges_and_value_is_correct():
    s0, s1 = leduc.solve_cfr(G, iterations=400)
    expl = leduc.exploitability(G, s0, s1)
    value = float(leduc.ev(G, s0, s1))
    assert expl < 0.1                       # CFR drives exploitability toward 0
    assert abs(value - (-0.0856)) < 0.02    # known Leduc game value to player 0


def test_best_response_value_matches_ev_bounds():
    # BR value to a player is at least the value they get under any fixed strategy pair.
    s0 = _uniform(G.n0, G.legal0)
    s1 = _uniform(G.n1, G.legal1)
    v0, _ = leduc._compute_best_response(G, 0, s1)
    assert v0 >= float(leduc.ev(G, s0, s1)) - 1e-6  # P0 can do at least as well by BR


def test_quantal_best_response_is_rowstochastic_on_legal_actions():
    s0 = _uniform(G.n0, G.legal0)
    s1 = _uniform(G.n1, G.legal1)
    q0, q1 = leduc.quantal_best_response(G, s0, s1, tau=0.3)
    for q, legal in ((q0, np.asarray(G.legal0)), (q1, np.asarray(G.legal1))):
        assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(q * (1 - legal) < 1e-8)  # zero mass on illegal actions


def test_regret_matching_is_rowstochastic_on_legal_actions():
    R = np.array([[1.0, -2.0, 3.0]])
    legal = np.array([[1.0, 1.0, 0.0]])  # action 2 illegal
    s = leduc.regret_matching(R, legal)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert s[0, 2] == 0.0                 # no mass on illegal action
    assert s[0, 0] == 1.0                 # only positive-regret legal action


def test_counterfactual_values_shapes_and_legality():
    s0 = _uniform(G.n0, G.legal0)
    s1 = _uniform(G.n1, G.legal1)
    q0, q1 = leduc.counterfactual_values(G, s0, s1)
    assert q0.shape == (G.n0, leduc.NUM_ACTIONS)
    assert q1.shape == (G.n1, leduc.NUM_ACTIONS)
    assert np.all(np.isfinite(q0)) and np.all(np.isfinite(q1))
    # zero on illegal actions
    assert np.all(q0 * (1 - np.asarray(G.legal0)) == 0.0)
    assert np.all(q1 * (1 - np.asarray(G.legal1)) == 0.0)


def test_counterfactual_regrets_drive_tabular_cfr_to_equilibrium():
    # Running plain tabular CFR built from these two primitives must reduce
    # exploitability -- the check that the regret computation is correct.
    R0 = np.zeros((G.n0, leduc.NUM_ACTIONS))
    R1 = np.zeros((G.n1, leduc.NUM_ACTIONS))
    S0 = np.zeros_like(R0)
    S1 = np.zeros_like(R1)
    for _ in range(300):
        s0 = leduc.regret_matching(R0, G.legal0)
        s1 = leduc.regret_matching(R1, G.legal1)
        S0 += s0
        S1 += s1
        r0, r1 = leduc.counterfactual_regrets(G, s0, s1)
        R0 += r0
        R1 += r1
    a0 = S0 / S0.sum(1, keepdims=True)
    a1 = S1 / S1.sum(1, keepdims=True)
    assert leduc.exploitability(G, a0, a1) < 0.4  # from ~4.7 at uniform
