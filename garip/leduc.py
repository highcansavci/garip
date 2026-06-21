"""Leduc hold'em: full game tree, differentiable EV, exact best response, and CFR.

Leduc is the standard step up from Kuhn poker. Deck of 6 cards = 3 ranks (J<Q<K) x 2
suits. Each player antes 1 and is dealt one private card; a betting round follows, then
one public card is revealed, then a second betting round, then a showdown. A player
whose private card matches the public rank ("a pair") beats any non-pair; otherwise the
higher private card wins. Bet/raise size is 2 in round 1 and 4 in round 2, with at most
two raises per round. Suits are irrelevant to payoffs, so information sets are keyed by
*rank* (the standard Leduc abstraction); deals still enumerate suited cards for correct
probabilities.

The tree is too large for brute-force exploitability (player strategy spaces are huge),
so this module provides an exact recursive **best response** (counterfactual traversal)
and a vanilla **CFR** solver. EV is tensorized and differentiable for training neural
policies through the tree.

Players are 0 and 1; payoffs are from player 0's perspective (player 0 maximizes).
Actions: FOLD=0, CALL=1 (check when no bet is pending), RAISE=2 (bet when none pending).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

FOLD, CALL, RAISE = 0, 1, 2
NUM_ACTIONS = 3
NUM_RANKS = 3
RANK_NAMES = ("J", "Q", "K")
RAISE_SIZE = {0: 2.0, 1: 4.0}
MAX_RAISES = 2
NUM_CARDS = 6  # 2 suits x 3 ranks


def _rank(card: int) -> int:
    return card // 2


# --------------------------------------------------------------------------- #
# Tree node types.
# --------------------------------------------------------------------------- #
@dataclass
class Terminal:
    payoff: float  # to player 0


@dataclass
class Chance:
    children: list  # list of (prob, node)


@dataclass
class Decision:
    player: int
    local_id: int           # index within the player's own infoset list
    legal: tuple            # legal actions
    children: dict = field(default_factory=dict)  # action -> node


@dataclass
class Game:
    root: Chance
    n0: int                 # number of player-0 infosets
    n1: int
    legal0: jax.Array       # (n0, 3) legal-action mask
    legal1: jax.Array       # (n1, 3)
    features0: jax.Array    # (n0, F)
    features1: jax.Array    # (n1, F)
    # tensorized terminals for differentiable EV:
    payoffs: jax.Array      # (R,)
    chance: jax.Array       # (R,)
    p0_idx: jax.Array       # (R, K) local infoset ids for player-0 decisions
    p0_act: jax.Array
    p0_mask: jax.Array
    p1_idx: jax.Array
    p1_act: jax.Array
    p1_mask: jax.Array


# --------------------------------------------------------------------------- #
# Builder.
# --------------------------------------------------------------------------- #
def build_game() -> Game:
    # Infoset registries, keyed by (rank, public_rank_or_-1, history_tuple) per player.
    info_keys = ({}, {})       # per player: key -> local_id
    info_meta = ([], [])       # per player: list of (own_rank, pub_rank_or_-1, history_tuple, legal)

    def get_local_id(player, own_rank, pub_rank, history, legal):
        key = (own_rank, pub_rank, history)
        table = info_keys[player]
        if key not in table:
            table[key] = len(table)
            info_meta[player].append((own_rank, pub_rank, history, legal))
        return table[key]

    def legal_actions(facing_bet, raises):
        if facing_bet:
            acts = [FOLD, CALL]
            if raises < MAX_RAISES:
                acts.append(RAISE)
        else:
            acts = [CALL]  # check
            if raises < MAX_RAISES:
                acts.append(RAISE)
        return tuple(acts)

    def showdown_payoff(c1r, c2r, pubr, commit):
        p0_pair = (c1r == pubr)
        p1_pair = (c2r == pubr)
        if p0_pair and not p1_pair:
            winner = 0
        elif p1_pair and not p0_pair:
            winner = 1
        elif c1r > c2r:
            winner = 0
        elif c2r > c1r:
            winner = 1
        else:
            return 0.0  # identical ranks -> split pot
        return commit if winner == 0 else -commit

    def build_round(round_idx, to_move, facing_bet, raises, a, b,
                    acted_this_round, history, c1_card, c2_card, pubr):
        c1r, c2r = _rank(c1_card), _rank(c2_card)
        own_rank = c1r if to_move == 0 else c2r
        pub_for_infoset = pubr if round_idx == 1 else -1
        legal = legal_actions(facing_bet, raises)
        local_id = get_local_id(to_move, own_rank, pub_for_infoset, tuple(history), legal)
        node = Decision(player=to_move, local_id=local_id, legal=legal)

        for act in legal:
            hist2 = history + [act]
            if act == FOLD:
                # The folder (to_move) loses; the winner takes the folder's commitment.
                # Payoff to player 0: +b if player 1 folded, -a if player 0 folded.
                payoff = b if to_move == 1 else -a
                node.children[act] = Terminal(payoff=payoff)
            elif act == CALL:
                if facing_bet:                     # call matches the outstanding bet
                    m = max(a, b)
                    na, nb = (m, b) if to_move == 0 else (a, m)
                    node.children[act] = _end_round(round_idx, na, nb, hist2,
                                                    c1_card, c2_card, pubr,
                                                    build_round, showdown_payoff)
                else:                              # check
                    if len(acted_this_round) == 0:  # first to act checks -> opponent acts
                        node.children[act] = build_round(
                            round_idx, 1 - to_move, False, raises, a, b,
                            acted_this_round + [act], hist2, c1_card, c2_card, pubr)
                    else:                          # check-check ends the round
                        node.children[act] = _end_round(round_idx, a, b, hist2,
                                                        c1_card, c2_card, pubr,
                                                        build_round, showdown_payoff)
            else:  # RAISE
                m = max(a, b)
                new_commit = m + RAISE_SIZE[round_idx]
                na, nb = (new_commit, b) if to_move == 0 else (a, new_commit)
                node.children[act] = build_round(
                    round_idx, 1 - to_move, True, raises + 1, na, nb,
                    acted_this_round + [act], hist2, c1_card, c2_card, pubr)
        return node

    def _end_round(round_idx, a, b, history, c1_card, c2_card, pubr,
                   build_round_fn, showdown_fn):
        if round_idx == 0:
            # reveal a public card from the 4 remaining, each with prob 1/4
            remaining = [c for c in range(NUM_CARDS) if c not in (c1_card, c2_card)]
            children = []
            for pub in remaining:
                sub = build_round_fn(1, 0, False, 0, a, b, [], history,
                                     c1_card, c2_card, _rank(pub))
                children.append((1.0 / len(remaining), sub))
            return Chance(children=children)
        else:
            return Terminal(payoff=showdown_fn(_rank(c1_card), _rank(c2_card), pubr, a))

    # Root chance over private deals (ordered, distinct), prob 1/30 each.
    deals = [(c1, c2) for c1 in range(NUM_CARDS) for c2 in range(NUM_CARDS) if c1 != c2]
    root_children = []
    for (c1_card, c2_card) in deals:
        sub = build_round(0, 0, False, 0, 1.0, 1.0, [], [], c1_card, c2_card, -1)
        root_children.append((1.0 / len(deals), sub))
    root = Chance(children=root_children)

    n0, n1 = len(info_meta[0]), len(info_meta[1])
    legal0 = _legal_mask(info_meta[0])
    legal1 = _legal_mask(info_meta[1])
    features0 = _features(info_meta[0])
    features1 = _features(info_meta[1])
    tensors = _flatten_terminals(root, n0, n1)
    return Game(root=root, n0=n0, n1=n1, legal0=legal0, legal1=legal1,
                features0=features0, features1=features1, **tensors)


# --------------------------------------------------------------------------- #
# Infoset masks and features.
# --------------------------------------------------------------------------- #
def _legal_mask(meta):
    mask = jnp.zeros((len(meta), NUM_ACTIONS))
    rows = []
    for (_, _, _, legal) in meta:
        row = [1.0 if a in legal else 0.0 for a in range(NUM_ACTIONS)]
        rows.append(row)
    return jnp.array(rows)


def _features(meta):
    """Unique per-infoset features: own rank, public rank, and padded action history."""
    max_hist = max((len(h) for (_, _, h, _) in meta), default=0)
    rows = []
    for (own_rank, pub_rank, hist, _legal) in meta:
        own_oh = [0.0] * NUM_RANKS
        own_oh[own_rank] = 1.0
        pub_oh = [0.0] * (NUM_RANKS + 1)        # index 0 = "no public card yet"
        pub_oh[0 if pub_rank < 0 else pub_rank + 1] = 1.0
        hist_feat = []
        for i in range(max_hist):
            slot = [0.0] * (NUM_ACTIONS + 1)    # last index = "no action in this slot"
            slot[hist[i] if i < len(hist) else NUM_ACTIONS] = 1.0
            hist_feat += slot
        rows.append(own_oh + pub_oh + hist_feat)
    return jnp.array(rows)


# --------------------------------------------------------------------------- #
# Tensorized terminals for differentiable EV.
# --------------------------------------------------------------------------- #
def _flatten_terminals(root, n0, n1):
    records = []  # (chance, payoff, p0_decisions, p1_decisions)

    def walk(node, chance, p0_dec, p1_dec):
        if isinstance(node, Terminal):
            records.append((chance, node.payoff, list(p0_dec), list(p1_dec)))
        elif isinstance(node, Chance):
            for prob, ch in node.children:
                walk(ch, chance * prob, p0_dec, p1_dec)
        else:  # Decision
            for act, ch in node.children.items():
                dec = (node.local_id, act)
                if node.player == 0:
                    walk(ch, chance, p0_dec + [dec], p1_dec)
                else:
                    walk(ch, chance, p0_dec, p1_dec + [dec])

    walk(root, 1.0, [], [])

    k0 = max(len(r[2]) for r in records)
    k1 = max(len(r[3]) for r in records)
    R = len(records)

    def pack(dec_lists, k):
        idx = [[0] * k for _ in range(R)]
        act = [[0] * k for _ in range(R)]
        mask = [[0.0] * k for _ in range(R)]
        for r, dec in enumerate(dec_lists):
            for j, (i, a) in enumerate(dec):
                idx[r][j] = i
                act[r][j] = a
                mask[r][j] = 1.0
        return jnp.array(idx), jnp.array(act), jnp.array(mask)

    p0_idx, p0_act, p0_mask = pack([r[2] for r in records], k0)
    p1_idx, p1_act, p1_mask = pack([r[3] for r in records], k1)
    return dict(
        payoffs=jnp.array([r[1] for r in records]),
        chance=jnp.array([r[0] for r in records]),
        p0_idx=p0_idx, p0_act=p0_act, p0_mask=p0_mask,
        p1_idx=p1_idx, p1_act=p1_act, p1_mask=p1_mask,
    )


def ev(game: Game, s0: jax.Array, s1: jax.Array) -> jax.Array:
    """Expected payoff to player 0 under behavioral strategies `s0`, `s1`.

    `s0` is `(n0, 3)`, `s1` is `(n1, 3)`; both row-stochastic over legal actions.
    Differentiable in both.
    """
    g0 = s0[game.p0_idx, game.p0_act]
    g0 = jnp.where(game.p0_mask > 0, g0, 1.0)
    reach0 = jnp.prod(g0, axis=1)
    g1 = s1[game.p1_idx, game.p1_act]
    g1 = jnp.where(game.p1_mask > 0, g1, 1.0)
    reach1 = jnp.prod(g1, axis=1)
    return jnp.sum(game.chance * reach0 * reach1 * game.payoffs)


# --------------------------------------------------------------------------- #
# Exact best response (counterfactual traversal) and exploitability.
# --------------------------------------------------------------------------- #
def _compute_best_response(game: Game, responder: int, opp_strat):
    """Exact best response to `responder` vs fixed `opp_strat`, in O(tree) time.

    Returns `(value, br)` where `value` is the best-response value in the responder's
    own payoff units and `br` is a dict `local_id -> chosen action`. Algorithm: one
    counterfactual-reach pass groups responder nodes by infoset; infosets are then
    decided deepest-first using a memoized subtree value, so each node is evaluated
    once. (The slow O(tree x infosets) version re-traversed the tree per infoset.)
    """
    import numpy as np
    from collections import defaultdict

    opp = np.asarray(opp_strat)
    opp_player = 1 - responder
    sign = 1.0 if responder == 0 else -1.0  # payoffs stored from player 0's view
    legal_all = np.asarray(game.legal0 if responder == 0 else game.legal1)

    occ = defaultdict(list)  # infoset id -> [(node, counterfactual_reach)]
    depth = {}

    def reach_pass(node, cf, d):
        if isinstance(node, Terminal):
            return
        if isinstance(node, Chance):
            for prob, ch in node.children:
                reach_pass(ch, cf * prob, d + 1)
        elif node.player == opp_player:
            for a, ch in node.children.items():
                reach_pass(ch, cf * opp[node.local_id, a], d + 1)
        else:  # responder node: record, then descend with reach unchanged (counterfactual)
            occ[node.local_id].append((node, cf))
            depth[node.local_id] = max(depth.get(node.local_id, 0), d)
            for _, ch in node.children.items():
                reach_pass(ch, cf, d + 1)

    reach_pass(game.root, 1.0, 0)

    br = {}
    memo = {}

    def sval(node):
        """Subtree value to responder; responder plays decided `br`, opp plays `opp`."""
        key = id(node)
        if key in memo:
            return memo[key]
        if isinstance(node, Terminal):
            v = sign * node.payoff
        elif isinstance(node, Chance):
            v = sum(prob * sval(ch) for prob, ch in node.children)
        elif node.player == opp_player:
            v = sum(opp[node.local_id, a] * sval(ch) for a, ch in node.children.items())
        else:
            v = sval(node.children[br[node.local_id]])
        memo[key] = v
        return v

    for local_id in sorted(occ, key=lambda i: -depth[i]):  # deepest first
        best_a, best_v = None, None
        for a in range(NUM_ACTIONS):
            if legal_all[local_id, a] <= 0:
                continue
            v = sum(cf * sval(node.children[a]) for node, cf in occ[local_id])
            if best_v is None or v > best_v:
                best_a, best_v = a, v
        br[local_id] = best_a

    return float(sval(game.root)), br


def exploitability(game: Game, s0, s1) -> float:
    """Exact NashConv: `BR0(s1) + BR1(s0)`, `>= 0`, `= 0` at Nash."""
    v0, _ = _compute_best_response(game, 0, s1)
    v1, _ = _compute_best_response(game, 1, s0)
    return v0 + v1


def quantal_best_response(game: Game, s0, s1, tau: float):
    """Entropy-regularized counterfactual best response for both players.

    For each information set, computes the reach-normalized counterfactual value of
    every action under the current strategies `(s0, s1)`, then returns the per-infoset
    softmax at temperature `tau`. This is the correct extensive-form analog of the
    matrix smoothed best-response map `G`/`F`: unlike a raw EV gradient (which scales
    each infoset by its reach, starving deep infosets), this gives a proper local best
    response at every infoset regardless of how often it is reached. Returned arrays are
    plain NumPy and serve as stop-gradient targets for the CGSP cycle penalty.
    """
    import numpy as np
    s = [np.asarray(s0), np.asarray(s1)]
    legal = [np.asarray(game.legal0), np.asarray(game.legal1)]
    n = [game.n0, game.n1]
    cfv = [np.zeros((n[p], NUM_ACTIONS)) for p in (0, 1)]
    reach_sum = [np.zeros(n[p]) for p in (0, 1)]

    def traverse(node, r0, r1, rc):
        if isinstance(node, Terminal):
            return node.payoff  # value to player 0
        if isinstance(node, Chance):
            return sum(prob * traverse(ch, r0, r1, rc * prob) for prob, ch in node.children)
        p = node.player
        strat = s[p][node.local_id]
        node_val = 0.0
        child_vals = {}
        for a, ch in node.children.items():
            if p == 0:
                cv = traverse(ch, r0 * strat[a], r1, rc)
            else:
                cv = traverse(ch, r0, r1 * strat[a], rc)
            child_vals[a] = cv
            node_val += strat[a] * cv
        cf = (r1 if p == 0 else r0) * rc            # counterfactual reach (opp x chance)
        reach_sum[p][node.local_id] += cf
        for a, cv in child_vals.items():
            val_to_p = cv if p == 0 else -cv         # value in the acting player's units
            cfv[p][node.local_id, a] += cf * val_to_p
        return node_val

    traverse(game.root, 1.0, 1.0, 1.0)

    out = []
    for p in (0, 1):
        rs = np.maximum(reach_sum[p], 1e-12)[:, None]
        q = cfv[p] / rs                              # reach-normalized counterfactual value
        q = np.where(legal[p] > 0, q / tau, -1e9)
        q = q - q.max(axis=1, keepdims=True)
        e = np.exp(q) * legal[p]
        out.append(e / e.sum(axis=1, keepdims=True))
    return out[0], out[1]


def counterfactual_values(game: Game, s0, s1):
    """Reach-normalized counterfactual value of every action at every infoset.

    Returns `(q0, q1)` with `q_p` of shape `(n_p, 3)` in player `p`'s *own* units (each
    player maximizes its `q`). This is the local payoff-gradient used by the tabular
    mirror-ascent self-play methods (GARIP / R-NaD / MMD) on the Leduc tree — the exact
    analog of `A y` / `xᵀA` in matrix games.
    """
    import numpy as np
    s = [np.asarray(s0), np.asarray(s1)]
    legal = [np.asarray(game.legal0), np.asarray(game.legal1)]
    n = [game.n0, game.n1]
    cfv = [np.zeros((n[p], NUM_ACTIONS)) for p in (0, 1)]
    reach_sum = [np.zeros(n[p]) for p in (0, 1)]

    def traverse(node, r0, r1, rc):
        if isinstance(node, Terminal):
            return node.payoff
        if isinstance(node, Chance):
            return sum(prob * traverse(ch, r0, r1, rc * prob) for prob, ch in node.children)
        p = node.player
        strat = s[p][node.local_id]
        node_val = 0.0
        child_vals = {}
        for a, ch in node.children.items():
            if p == 0:
                cv = traverse(ch, r0 * strat[a], r1, rc)
            else:
                cv = traverse(ch, r0, r1 * strat[a], rc)
            child_vals[a] = cv
            node_val += strat[a] * cv
        cf = (r1 if p == 0 else r0) * rc
        reach_sum[p][node.local_id] += cf
        for a, cv in child_vals.items():
            cfv[p][node.local_id, a] += cf * (cv if p == 0 else -cv)
        return node_val

    traverse(game.root, 1.0, 1.0, 1.0)
    out = []
    for p in (0, 1):
        rs = np.maximum(reach_sum[p], 1e-12)[:, None]
        out.append(np.where(legal[p] > 0, cfv[p] / rs, 0.0))
    return out[0], out[1]


def counterfactual_regrets(game: Game, s0, s1):
    """Instantaneous reach-weighted counterfactual regrets for both players.

    For each information set I and action a, returns
    `r(I,a) = sum_h cf_reach(h) * (v(h,a) - v(h))` in the acting player's payoff units,
    where `cf_reach` is the opponent x chance reach, `v(h,a)` is the value of taking `a`
    then following the current strategies, and `v(h)` is the strategy's expected value.
    These are exactly the regrets CFR accumulates; summing them over training and
    applying regret matching gives a no-regret (CFR-style) cycle target with no
    temperature bias. Returned as plain NumPy.
    """
    import numpy as np
    s = [np.asarray(s0), np.asarray(s1)]
    n = [game.n0, game.n1]
    reg = [np.zeros((n[p], NUM_ACTIONS)) for p in (0, 1)]

    def traverse(node, r0, r1, rc):
        if isinstance(node, Terminal):
            return node.payoff  # value to player 0
        if isinstance(node, Chance):
            return sum(prob * traverse(ch, r0, r1, rc * prob) for prob, ch in node.children)
        p = node.player
        strat = s[p][node.local_id]
        node_val = 0.0
        child_vals = {}
        for a, ch in node.children.items():
            if p == 0:
                cv = traverse(ch, r0 * strat[a], r1, rc)
            else:
                cv = traverse(ch, r0, r1 * strat[a], rc)
            child_vals[a] = cv
            node_val += strat[a] * cv
        cf = (r1 if p == 0 else r0) * rc
        sgn = 1.0 if p == 0 else -1.0          # value to the acting player
        for a, cv in child_vals.items():
            reg[p][node.local_id, a] += cf * sgn * (cv - node_val)
        return node_val

    traverse(game.root, 1.0, 1.0, 1.0)
    return reg[0], reg[1]


def regret_matching(cumulative_regret, legal):
    """CFR regret matching: strategy proportional to positive cumulative regret."""
    import numpy as np
    legal = np.asarray(legal)
    pos = np.maximum(cumulative_regret, 0.0) * legal
    total = pos.sum(axis=1, keepdims=True)
    unif = legal / legal.sum(axis=1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), unif)


def best_response_strategy(game: Game, responder: int, opp_strat):
    """Return responder's exact pure best response as a one-hot `(n, 3)` array."""
    import numpy as np
    _, br = _compute_best_response(game, responder, opp_strat)
    legal_all = np.asarray(game.legal0 if responder == 0 else game.legal1)
    n = game.n0 if responder == 0 else game.n1
    strat = np.zeros((n, NUM_ACTIONS))
    for i in range(n):
        a = br.get(i, None)
        if a is None:  # unreached infoset -> uniform over legal actions
            legal = [a for a in range(NUM_ACTIONS) if legal_all[i, a] > 0]
            for a in legal:
                strat[i, a] = 1.0 / len(legal)
        else:
            strat[i, a] = 1.0
    return jnp.asarray(strat)


# --------------------------------------------------------------------------- #
# Vanilla CFR (NumPy) -- engine validation and a strong reference baseline.
# --------------------------------------------------------------------------- #
def solve_cfr(game: Game, iterations: int = 2000):
    """Run vanilla CFR; return the average strategies `(s0_avg, s1_avg)`.

    Used both to validate the engine (exploitability must fall toward 0) and as a
    gold-standard reference curve in the experiments.
    """
    import numpy as np
    legal = [np.asarray(game.legal0), np.asarray(game.legal1)]
    n = [game.n0, game.n1]
    regret = [np.zeros((n[p], NUM_ACTIONS)) for p in (0, 1)]
    strat_sum = [np.zeros((n[p], NUM_ACTIONS)) for p in (0, 1)]

    def current_strategy(p):
        pos = np.maximum(regret[p], 0.0)
        total = pos.sum(axis=1, keepdims=True)
        unif = legal[p] / legal[p].sum(axis=1, keepdims=True)
        strat = np.where(total > 0, pos / np.where(total > 0, total, 1.0), unif)
        return strat * legal[p]  # zero illegal actions

    def cfr(node, reach):
        # reach = [reach_p0, reach_p1, reach_chance]
        if isinstance(node, Terminal):
            return np.array([node.payoff, -node.payoff])
        if isinstance(node, Chance):
            out = np.zeros(2)
            for prob, ch in node.children:
                r2 = [reach[0], reach[1], reach[2] * prob]
                out += prob * cfr(ch, r2)
            return out
        p = node.player
        strat = current_strategy(p)[node.local_id]
        node_util = np.zeros(2)
        child_util = {}
        for a, ch in node.children.items():
            r2 = list(reach)
            r2[p] = reach[p] * strat[a]
            cu = cfr(ch, r2)
            child_util[a] = cu
            node_util += strat[a] * cu
        # counterfactual reach of opponent * chance
        cf_reach = reach[1 - p] * reach[2]
        opp_reach_for_avg = reach[p]
        for a in node.children:
            regret[p][node.local_id, a] += cf_reach * (child_util[a][p] - node_util[p])
        strat_sum[p][node.local_id] += opp_reach_for_avg * strat
        return node_util

    for _ in range(iterations):
        cfr(game.root, [1.0, 1.0, 1.0])

    avg = []
    for p in (0, 1):
        total = strat_sum[p].sum(axis=1, keepdims=True)
        unif = legal[p] / legal[p].sum(axis=1, keepdims=True)
        a = np.where(total > 0, strat_sum[p] / np.where(total > 0, total, 1.0), unif)
        avg.append(jnp.asarray(a * legal[p]))
    return avg[0], avg[1]


# Build the canonical game once at import.
GAME = build_game()
