"""Turn-based PPO self-play for pgx zero-sum board games (e.g. Connect Four).

Board games are the clean second deep-RL testbed the simultaneous-move envs (Coin Game,
STORM) could not provide a faithful metric for: they are symmetric, strictly zero-sum
(win/lose/draw), the policy's action distribution *is* its strategy (so the magnet
regularizes exactly what is measured), and -- crucially -- every game terminates with a
result, so a policy cannot be "robust" by refusing to play (the inertness loophole that
broke STORM). Robustness is measured by training a fresh **best-response** policy against
the frozen policy and reporting its win-rate (higher = more exploitable).

We reduce turn-based self-play to a single-agent MDP from the *learner's* view: each
learner step plays the learner's move (params) and then the opponent's reply
(`opp_params`), returning the learner's reward and next learner-perspective observation.
`opp_params in {current, average}` plus the magnet-KL term realize the same
GARIP/R-NaD/MMD/naive ablation as the Coin Game trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import pgx


@dataclass(frozen=True)
class PgxConfig:
    num_envs: int = 128
    rollout_len: int = 32        # learner decision points per update
    epochs: int = 4
    num_minibatches: int = 8
    clip_eps: float = 0.2
    gamma: float = 1.0           # board games: undiscounted terminal reward
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    polyak: float = 0.01
    hidden: int = 128


class BoardActorCritic(nn.Module):
    hidden: int = 128
    num_actions: int = 7

    @nn.compact
    def __call__(self, x):
        a = nn.relu(nn.Dense(self.hidden)(x))
        a = nn.relu(nn.Dense(self.hidden)(a))
        logits = nn.Dense(self.num_actions)(a)
        v = nn.relu(nn.Dense(self.hidden)(x))
        v = nn.relu(nn.Dense(self.hidden)(v))
        return logits, nn.Dense(1)(v).squeeze(-1)


class Transition(NamedTuple):
    obs: jax.Array
    legal: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def _masked_logits(logits, legal):
    return jnp.where(legal, logits, -1e9)


def make_pgx_components(game: str, config: PgxConfig):
    env = pgx.make(game)
    num_actions = env.num_actions
    obs_dim = int(jnp.prod(jnp.array(env.init(jax.random.PRNGKey(0)).observation.shape)))
    net = BoardActorCritic(hidden=config.hidden, num_actions=num_actions)
    N = config.num_envs
    vinit = jax.vmap(env.init)
    vstep = jax.vmap(env.step)

    def _flat_obs(state):
        return state.observation.reshape(N, -1).astype(jnp.float32)

    def _act(params, state, rng):
        logits, value = net.apply(params, _flat_obs(state))
        logits = _masked_logits(logits, state.legal_action_mask)
        a = jax.random.categorical(rng, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(N), a]
        return a, logp, value

    def _reset_with_color(rng):
        """Init N games and assign each a random learner colour; if the learner is
        player 1, let the opponent (player 0) make the first move so that on return it
        is always the learner's turn."""
        rng, ki, kc, ko = jax.random.split(rng, 4)
        state = vinit(jax.random.split(ki, N))
        learner_color = jax.random.randint(kc, (N,), 0, 2)
        # opponent-first move where learner is player 1
        return state, learner_color, rng

    def _opp_move(state, learner_color, opp_params, rng):
        """Advance any env whose turn belongs to the opponent by one opponent ply."""
        opp_turn = (state.current_player != learner_color) & (~state.terminated)
        a, _, _ = _act(opp_params, state, rng)
        stepped = vstep(state, a)
        return jax.tree.map(lambda s2, s1: jnp.where(opp_turn.reshape((-1,) + (1,) * (s2.ndim - 1)),
                                                     s2, s1), stepped, state)

    def learner_step(carry, params, opp_params, rng):
        """One learner decision point (+ opponent reply), batched. Returns transition."""
        state, learner_color = carry
        rng, kp, ko1, kr, kc = jax.random.split(rng, 5)
        obs = _flat_obs(state)
        legal = state.legal_action_mask
        a, logp, value = _act(params, state, kp)
        state1 = vstep(state, a)
        r1 = state1.rewards[jnp.arange(N), learner_color]
        done1 = state1.terminated
        # opponent reply where the game continues
        state2 = _opp_move(state1, learner_color, opp_params, ko1)
        r2 = state2.rewards[jnp.arange(N), learner_color] - r1  # incremental
        reward = r1 + jnp.where(done1, 0.0, r2)
        done = state2.terminated | done1
        # auto-reset finished games (fresh colour); else continue
        fresh, fresh_color, _ = _reset_with_color(kr)
        # for fresh games where the new learner is player 1, opponent moves first
        fresh = _opp_move(fresh, fresh_color, opp_params, kc)
        next_state = jax.tree.map(
            lambda f, c: jnp.where(done.reshape((-1,) + (1,) * (f.ndim - 1)), f, c), fresh, state2)
        next_color = jnp.where(done, fresh_color, learner_color)
        tr = Transition(obs, legal, a, logp, value, reward, done.astype(jnp.float32))
        return (next_state, next_color), tr

    return env, net, N, obs_dim, num_actions, _reset_with_color, _opp_move, learner_step, _flat_obs


def make_pgx_trainer(game: str, config: PgxConfig, opponent_mode: str = "average",
                     magnet_mode: str = "moving", lam: float = 0.0, reset_every: int = 200):
    (env, net, N, obs_dim, num_actions, reset_with_color, opp_move,
     learner_step, flat_obs) = make_pgx_components(game, config)
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.lr))

    def gae(traj, last_val):
        def f(carry, x):
            adv, nv = carry
            r, v, d = x
            delta = r + config.gamma * nv * (1 - d) - v
            adv = delta + config.gamma * config.gae_lambda * (1 - d) * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(f, (jnp.zeros(N), last_val),
                              (traj.reward, traj.value, traj.done), reverse=True)
        return adv, adv + traj.value

    def update(params, opt_state, magnet, lam, traj, adv, ret, rng):
        b = config.rollout_len * N
        flat = jax.tree.map(lambda x: x.reshape((b,) + x.shape[2:]), traj)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(b)
        ret = ret.reshape(b)
        mb = b // config.num_minibatches

        def loss_fn(params, d):
            tr, a, r = d
            logits, value = net.apply(params, tr.obs)
            logits = _masked_logits(logits, tr.legal)
            logp = jax.nn.log_softmax(logits)[jnp.arange(tr.obs.shape[0]), tr.action]
            ratio = jnp.exp(logp - tr.logp)
            pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * a).mean()
            vl = 0.5 * ((value - r) ** 2).mean()
            probs = jax.nn.softmax(logits)
            ent = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
            mlogits, _ = net.apply(jax.lax.stop_gradient(magnet), tr.obs)
            mlogits = _masked_logits(mlogits, tr.legal)
            kl = (probs * (jax.nn.log_softmax(logits) - jax.nn.log_softmax(mlogits))).sum(-1).mean()
            return pg + config.vf_coef * vl - config.ent_coef * ent + lam * kl

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, b)
            sh = jax.tree.map(lambda x: x[perm], (flat, adv, ret))

            def mbf(carry, i):
                params, opt_state = carry
                d = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, i * mb, mb), sh)
                g = jax.grad(loss_fn)(params, d)
                u, opt_state = optimizer.update(g, opt_state)
                return (optax.apply_updates(params, u), opt_state), 0
            (params, opt_state), _ = jax.lax.scan(mbf, (params, opt_state), jnp.arange(config.num_minibatches))
            return (params, opt_state, rng), 0
        (params, opt_state, rng), _ = jax.lax.scan(epoch, (params, opt_state, rng), None, length=config.epochs)
        return params, opt_state, rng

    def init(key):
        k1, k2 = jax.random.split(key)
        params = net.init(k1, jnp.zeros((1, obs_dim)))
        opt_state = optimizer.init(params)
        state, color, _ = reset_with_color(k2)
        # opponent-first for player-1 learners at the very start
        state = opp_move(state, color, params, k2)
        return params, opt_state, params, params, params, jnp.array(0.0), (state, color), key

    def one_update(carry, _):
        params, opt_state, avg, fixed, periodic, t, runner, rng = carry
        opp = avg if opponent_mode == "average" else jax.lax.stop_gradient(params)
        magnet = fixed if magnet_mode == "fixed" else (periodic if magnet_mode == "periodic" else avg)
        rng, kr = jax.random.split(rng)

        def step_fn(c, _):
            runner, rng = c
            rng, ks = jax.random.split(rng)
            runner, tr = learner_step(runner, params, opp, ks)
            return (runner, rng), tr
        (runner, rng), traj = jax.lax.scan(step_fn, (runner, kr), None, length=config.rollout_len)
        _, last_val = net.apply(params, flat_obs(runner[0]))
        adv, ret = gae(traj, last_val)
        params, opt_state, rng = update(params, opt_state, magnet, lam, traj, adv, ret, rng)
        avg = jax.tree.map(lambda a, p: (1 - config.polyak) * a + config.polyak * p, avg, params)
        t = t + 1.0
        reset = jnp.mod(t, reset_every) < 0.5
        periodic = jax.tree.map(lambda m, p: jnp.where(reset, jax.lax.stop_gradient(p), m), periodic, params)
        return (params, opt_state, avg, fixed, periodic, t, runner, rng), traj.reward.sum()

    @partial(jax.jit, static_argnums=1)
    def train_chunk(carry, num_updates):
        carry, rew = jax.lax.scan(one_update, carry, None, length=num_updates)
        return carry, rew.mean()

    return net, init, train_chunk, obs_dim, num_actions


def head_to_head(game: str, net, params_a, params_b, key, n: int = 1024,
                 n_open: int = 4, temp: float = 1.0):
    """Win/lose/draw of A vs B, averaged over both colours. Actions are *sampled* (temp)
    and the first `n_open` plies are uniformly random, so games are diverse and the
    win-rate grades exploitability rather than collapsing to one deterministic line."""
    env = pgx.make(game)
    vinit, vstep = jax.vmap(env.init), jax.vmap(env.step)

    def run(a_color, key):
        key, ki = jax.random.split(key)
        st = vinit(jax.random.split(ki, n))
        a_is = jnp.full(n, a_color, jnp.int32)

        def body(c):
            st, R, done, t, key = c
            key, ka, kb, kr = jax.random.split(key, 4)
            obs = st.observation.reshape(n, -1).astype(jnp.float32)
            la, _ = net.apply(params_a, obs); lb, _ = net.apply(params_b, obs)
            la = jnp.where(st.legal_action_mask, la / temp, -1e9)
            lb = jnp.where(st.legal_action_mask, lb / temp, -1e9)
            act = jnp.where(st.current_player == a_is,
                            jax.random.categorical(ka, la), jax.random.categorical(kb, lb))
            rand_act = jax.random.categorical(kr, jnp.where(st.legal_action_mask, 0.0, -1e9))
            act = jnp.where(t < n_open, rand_act, act)               # random opening
            st2 = vstep(st, act)
            R = R + jnp.where(~done, st2.rewards[jnp.arange(n), a_is], 0.0)
            done2 = done | st2.terminated
            st2 = jax.tree.map(lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), y, x), st2, st)
            return st2, R, done2, t + 1, key
        _, R, _, _, _ = jax.lax.while_loop(
            lambda c: ~c[2].all(), body, (st, jnp.zeros(n), jnp.zeros(n, bool), 0, key))
        return R
    k0, k1 = jax.random.split(key)
    R = jnp.concatenate([run(0, k0), run(1, k1)])
    return float((R > 0).mean()), float((R < 0).mean()), float((R == 0).mean())


def best_response_winrate(game: str, config: PgxConfig, frozen_params, key,
                          br_updates: int = 300):
    """Train a fresh best-response learner against the frozen policy, then report the BR's
    win-rate against it (averaged over colours). Higher = frozen policy more exploitable."""
    (env, net, N, obs_dim, num_actions, reset_with_color, opp_move,
     learner_step, flat_obs) = make_pgx_components(game, config)
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.lr))

    def gae(traj, last_val):
        def f(carry, x):
            adv, nv = carry; r, v, d = x
            delta = r + config.gamma * nv * (1 - d) - v
            adv = delta + config.gamma * config.gae_lambda * (1 - d) * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(f, (jnp.zeros(N), last_val), (traj.reward, traj.value, traj.done), reverse=True)
        return adv, adv + traj.value

    def update(params, opt_state, traj, adv, ret, rng):
        b = config.rollout_len * N
        flat = jax.tree.map(lambda x: x.reshape((b,) + x.shape[2:]), traj)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(b); ret = ret.reshape(b)
        mb = b // config.num_minibatches

        def loss_fn(params, d):
            tr, a, r = d
            logits, value = net.apply(params, tr.obs)
            logits = _masked_logits(logits, tr.legal)
            logp = jax.nn.log_softmax(logits)[jnp.arange(tr.obs.shape[0]), tr.action]
            ratio = jnp.exp(logp - tr.logp)
            pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * a).mean()
            vl = 0.5 * ((value - r) ** 2).mean()
            probs = jax.nn.softmax(logits)
            ent = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
            return pg + config.vf_coef * vl - config.ent_coef * ent

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, pk = jax.random.split(rng)
            sh = jax.tree.map(lambda x: x[jax.random.permutation(pk, b)], (flat, adv, ret))

            def mbf(c, i):
                params, opt_state = c
                d = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, i * mb, mb), sh)
                g = jax.grad(loss_fn)(params, d)
                u, opt_state = optimizer.update(g, opt_state)
                return (optax.apply_updates(params, u), opt_state), 0
            (params, opt_state), _ = jax.lax.scan(mbf, (params, opt_state), jnp.arange(config.num_minibatches))
            return (params, opt_state, rng), 0
        (params, opt_state, rng), _ = jax.lax.scan(epoch, (params, opt_state, rng), None, length=config.epochs)
        return params, opt_state, rng

    k1, k2, k3 = jax.random.split(key, 3)
    params = net.init(k1, jnp.zeros((1, obs_dim)))
    opt_state = optimizer.init(params)
    state, color, _ = reset_with_color(k2)
    state = opp_move(state, color, frozen_params, k2)
    runner = (state, color)

    def one(carry, _):
        params, opt_state, runner, rng = carry
        rng, kr = jax.random.split(rng)

        def step_fn(c, _):
            runner, rng = c; rng, ks = jax.random.split(rng)
            runner, tr = learner_step(runner, params, frozen_params, ks)
            return (runner, rng), tr
        (runner, rng), traj = jax.lax.scan(step_fn, (runner, kr), None, length=config.rollout_len)
        _, last_val = net.apply(params, flat_obs(runner[0]))
        adv, ret = gae(traj, last_val)
        params, opt_state, rng = update(params, opt_state, traj, adv, ret, rng)
        return (params, opt_state, runner, rng), 0

    (params, opt_state, runner, rng), _ = jax.lax.scan(
        jax.jit(one), (params, opt_state, runner, k3), None, length=br_updates)
    w, l, d = head_to_head(game, net, params, frozen_params, rng)
    return w  # BR win-rate vs the frozen policy
