"""Kuhn poker: game tree, expected value, and *exact* exploitability.

Kuhn poker is the standard tiny benchmark for imperfect-information self-play. Three
cards (J<Q<K), two players, each antes 1 and is dealt one card (6 equally likely
deals). Player 1 checks/bets; the betting can lead to a fold or a showdown where the
higher card wins the pot.

Each player has 6 information sets (3 cards x 2 histories), each with 2 actions
(pass=check/fold = 0, bet=call = 1). A behavioral strategy is therefore a `(6, 2)`
row-stochastic array. Payoffs below are from player 1's perspective; player 1 is the
maximizer, player 2 the minimizer.

Everything here is pure JAX and differentiable in the strategies, so neural policies
can be trained straight through `ev`. Exploitability is computed *exactly* by brute
force over each player's 64 pure strategies — that exact number, not the cycle term,
is what we report.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

# Information-set indexing.
#   Player 1 acts at history ""  -> ids 0,1,2 for cards J,Q,K
#              and at history "pb" (checked, then faced a bet) -> ids 3,4,5
#   Player 2 acts at history "p" -> ids 0,1,2
#              and at history "b" -> ids 3,4,5
P1_HISTORIES = {"": 0, "pb": 3}
P2_HISTORIES = {"p": 0, "b": 3}
NUM_INFOSETS = 6
NUM_ACTIONS = 2
CARDS = (0, 1, 2)  # J, Q, K
CARD_NAMES = ("J", "Q", "K")


def _p1_id(card: int, hist: str) -> int:
    return card + P1_HISTORIES[hist]


def _p2_id(card: int, hist: str) -> int:
    return card + P2_HISTORIES[hist]


def _build_terminals():
    """Enumerate every terminal play-through as (p1 decisions, p2 decisions, payoff).

    A decision is an `(infoset_id, action)` pair. Payoff is to player 1.
    """
    records = []
    for c1, c2 in itertools.permutations(CARDS, 2):  # 6 distinct deals
        win = 1.0 if c1 > c2 else -1.0  # player 1 wins the showdown?
        p1_start = _p1_id(c1, "")
        p1_pb = _p1_id(c1, "pb")
        p2_p = _p2_id(c2, "p")
        p2_b = _p2_id(c2, "b")
        # pp: check, check -> showdown for 1
        records.append(([(p1_start, 0)], [(p2_p, 0)], win * 1.0))
        # pbp: check, bet, fold -> player 1 loses the ante
        records.append(([(p1_start, 0), (p1_pb, 0)], [(p2_p, 1)], -1.0))
        # pbb: check, bet, call -> showdown for 2
        records.append(([(p1_start, 0), (p1_pb, 1)], [(p2_p, 1)], win * 2.0))
        # bp: bet, fold -> player 1 wins the ante
        records.append(([(p1_start, 1)], [(p2_b, 0)], 1.0))
        # bb: bet, call -> showdown for 2
        records.append(([(p1_start, 1)], [(p2_b, 1)], win * 2.0))
    return records


_RECORDS = _build_terminals()
_NUM_DEALS = 6.0
_K1 = 2  # max player-1 decisions on a single play-through
_K2 = 1  # player 2 always decides exactly once per play-through

# Pack into dense index/mask tensors for a vectorized, vmap-friendly `ev`.
_n = len(_RECORDS)
_p1_idx = [[0] * _K1 for _ in range(_n)]
_p1_act = [[0] * _K1 for _ in range(_n)]
_p1_mask = [[0.0] * _K1 for _ in range(_n)]
_p2_idx = [[0] * _K2 for _ in range(_n)]
_p2_act = [[0] * _K2 for _ in range(_n)]
_payoffs = [0.0] * _n
for r, (p1_dec, p2_dec, pay) in enumerate(_RECORDS):
    _payoffs[r] = pay
    for k, (i, a) in enumerate(p1_dec):
        _p1_idx[r][k] = i
        _p1_act[r][k] = a
        _p1_mask[r][k] = 1.0
    for k, (i, a) in enumerate(p2_dec):
        _p2_idx[r][k] = i
        _p2_act[r][k] = a

P1_IDX = jnp.array(_p1_idx)
P1_ACT = jnp.array(_p1_act)
P1_MASK = jnp.array(_p1_mask)
P2_IDX = jnp.array(_p2_idx)
P2_ACT = jnp.array(_p2_act)
PAYOFFS = jnp.array(_payoffs)


def ev(strat1: jax.Array, strat2: jax.Array) -> jax.Array:
    """Expected payoff to player 1 under behavioral strategies `strat1`, `strat2`.

    Both are `(6, 2)` row-stochastic arrays. Multilinear in each, hence differentiable.
    """
    g1 = strat1[P1_IDX, P1_ACT]                # (R, K1) action probs along each line
    g1 = jnp.where(P1_MASK > 0, g1, 1.0)       # masked-out decisions contribute 1
    reach1 = jnp.prod(g1, axis=1)
    g2 = strat2[P2_IDX, P2_ACT]                # (R, K2)
    reach2 = jnp.prod(g2, axis=1)
    return jnp.sum(PAYOFFS * reach1 * reach2) / _NUM_DEALS


# ----- exact exploitability via brute force over pure strategies ----------- #
def _pure_strategies() -> jax.Array:
    """All 2**6 = 64 pure strategies as one-hot `(64, 6, 2)` arrays."""
    rows = []
    for actions in itertools.product((0, 1), repeat=NUM_INFOSETS):
        rows.append(jax.nn.one_hot(jnp.array(actions), NUM_ACTIONS))
    return jnp.stack(rows)


PURE = _pure_strategies()  # (64, 6, 2)


def best_response_value_p1(strat2: jax.Array) -> jax.Array:
    """Value player 1 can guarantee by best-responding to `strat2` (maximizer)."""
    vals = jax.vmap(lambda p: ev(p, strat2))(PURE)
    return jnp.max(vals)


def best_response_value_p2(strat1: jax.Array) -> jax.Array:
    """Value player 2 forces by best-responding to `strat1` (minimizer of P1's EV)."""
    vals = jax.vmap(lambda p: ev(strat1, p))(PURE)
    return jnp.min(vals)


def exploitability(strat1: jax.Array, strat2: jax.Array) -> jax.Array:
    """Exact NashConv: how much both players together gain by best-responding.

    `>= 0`, and `= 0` iff `(strat1, strat2)` is a Nash equilibrium. The Kuhn game
    value is -1/18 to player 1; at equilibrium both best-response values meet there.
    """
    return best_response_value_p1(strat2) - best_response_value_p2(strat1)


def best_response_p1(strat2: jax.Array) -> jax.Array:
    """Player 1's exact pure best response to `strat2`, as a `(6, 2)` one-hot."""
    vals = jax.vmap(lambda p: ev(p, strat2))(PURE)
    return PURE[jnp.argmax(vals)]


def best_response_p2(strat1: jax.Array) -> jax.Array:
    """Player 2's exact pure best response to `strat1`, as a `(6, 2)` one-hot."""
    vals = jax.vmap(lambda p: ev(strat1, p))(PURE)
    return PURE[jnp.argmin(vals)]


# ----- smoothed (one-shot-deviation) best-response maps for the cycle term -- #
def smoothed_br_p2(strat1: jax.Array, tau: float) -> jax.Array:
    """Player 2's entropy-regularized best response to `strat1` (the map `G`).

    Each player-2 information set is visited on mutually exclusive play-throughs, so
    a one-shot-deviation value gives the *exact* counterfactual best response (the
    softmax is invariant to the additive contributions of the other infosets).
    """
    base = jnp.full((NUM_INFOSETS, NUM_ACTIONS), 0.5)

    def q(i, a):
        s2 = base.at[i].set(jax.nn.one_hot(a, NUM_ACTIONS))
        return ev(strat1, s2)

    ii, aa = jnp.meshgrid(jnp.arange(NUM_INFOSETS), jnp.arange(NUM_ACTIONS), indexing="ij")
    qvals = jax.vmap(jax.vmap(q))(ii, aa)        # (6, 2)
    return jax.nn.softmax(-qvals / tau, axis=1)  # P2 minimizes P1's EV


def smoothed_br_p1(strat2: jax.Array, ref_strat1: jax.Array, tau: float) -> jax.Array:
    """Player 1's entropy-regularized best response to `strat2` (the map `F`).

    Player 1's two infosets on the "checked then bet" line are not mutually exclusive,
    so the continuation is taken from `ref_strat1` (the current strategy). This makes
    `F` an *approximate* best-response map -- fine, since it is only used inside the
    cycle-consistency regularizer; the reported exploitability remains exact.
    """
    def q(i, a):
        s1 = ref_strat1.at[i].set(jax.nn.one_hot(a, NUM_ACTIONS))
        return ev(s1, strat2)

    ii, aa = jnp.meshgrid(jnp.arange(NUM_INFOSETS), jnp.arange(NUM_ACTIONS), indexing="ij")
    qvals = jax.vmap(jax.vmap(q))(ii, aa)
    return jax.nn.softmax(qvals / tau, axis=1)   # P1 maximizes


def cycle_loss(strat1: jax.Array, strat2: jax.Array, tau: float) -> jax.Array:
    """CycleGAN-style round-trip consistency `||F(G(s1)) - s1||^2 + ||G(F(s2)) - s2||^2`."""
    s1_roundtrip = smoothed_br_p1(smoothed_br_p2(strat1, tau), strat1, tau)
    s2_roundtrip = smoothed_br_p2(smoothed_br_p1(strat2, strat1, tau), tau)
    return jnp.sum((s1_roundtrip - strat1) ** 2) + jnp.sum((s2_roundtrip - strat2) ** 2)


# ----- infoset features for neural policies -------------------------------- #
def _features(histories: dict) -> jax.Array:
    """`(6, 5)` features per infoset: [card one-hot (3), history one-hot (2)]."""
    feats = [None] * NUM_INFOSETS
    for h_index, (_, base) in enumerate(histories.items()):
        for card in CARDS:
            card_oh = [0.0, 0.0, 0.0]
            card_oh[card] = 1.0
            hist_oh = [0.0, 0.0]
            hist_oh[h_index] = 1.0
            feats[card + base] = card_oh + hist_oh
    return jnp.array(feats)


P1_FEATURES = _features(P1_HISTORIES)  # (6, 5)
P2_FEATURES = _features(P2_HISTORIES)  # (6, 5)
FEATURE_DIM = 5
