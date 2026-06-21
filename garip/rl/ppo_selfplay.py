"""Compact PPO self-play for the Coin Game, with the CGSP cycle term.

A single shared actor-critic plays the learner (player 0). The *opponent* (player 1) is
either the learner's current parameters (naive self-play) or a Polyak running-average of
them (the CGSP / fictitious-self-play "best-respond to the average opponent" mechanism).
CGSP additionally applies a cycle-consistency penalty `lam * KL(pi_theta || avg policy)`
that damps the policy from oscillating away from its own running average -- the deep-RL
analog of the cycle term pulling toward a self-consistent fixed point.

Ablation realized by `(opponent_mode, lam)`:
    ("current", 0)  -> naive self-play
    ("average", 0)  -> fictitious self-play
    ("average", >0) -> CGSP
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 64
    rollout_len: int = 64
    epochs: int = 4
    num_minibatches: int = 8
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    polyak: float = 0.01
    hidden: int = 64
    episode_length: int = 16


class ActorCritic(nn.Module):
    hidden: int = 64
    num_actions: int = 5

    @nn.compact
    def __call__(self, x):
        a = nn.tanh(nn.Dense(self.hidden)(x))
        a = nn.tanh(nn.Dense(self.hidden)(a))
        logits = nn.Dense(self.num_actions)(a)
        v = nn.tanh(nn.Dense(self.hidden)(x))
        v = nn.tanh(nn.Dense(self.hidden)(v))
        value = nn.Dense(1)(v)
        return logits, value.squeeze(-1)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def _logp_of(logits, action):
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(action.shape[0]), action]


def make_ppo(env, config: PPOConfig):
    """Returns reusable (net, rollout, gae, update) closures bound to `env`/`config`."""
    net = ActorCritic(hidden=config.hidden, num_actions=env.num_actions)
    vstep = jax.vmap(env.step)
    vreset = jax.vmap(env.reset)
    N = config.num_envs

    def init_runner(key):
        keys = jax.random.split(key, N)
        (obs0, obs1), env_state = vreset(keys)
        return env_state, obs0, obs1

    def rollout(params, opp_params, runner, rng):
        env_state, obs0, obs1 = runner

        def step_fn(carry, _):
            env_state, obs0, obs1, rng = carry
            logits0, value0 = net.apply(params, obs0)
            rng, k0 = jax.random.split(rng)
            a0 = jax.random.categorical(k0, logits0)
            logp0 = _logp_of(logits0, a0)
            logits1, _ = net.apply(opp_params, obs1)
            rng, k1 = jax.random.split(rng)
            a1 = jax.random.categorical(k1, logits1)
            rng, ks = jax.random.split(rng)
            keys = jax.random.split(ks, N)
            (no0, no1), nstate, (r0, _r1), done, _info = vstep(keys, env_state, a0, a1)
            tr = Transition(obs0, a0, logp0, value0, r0, done.astype(jnp.float32))
            return (nstate, no0, no1, rng), tr

        (env_state, obs0, obs1, rng), traj = jax.lax.scan(
            step_fn, (env_state, obs0, obs1, rng), None, length=config.rollout_len)
        _, last_val = net.apply(params, obs0)
        return traj, last_val, (env_state, obs0, obs1), rng

    def gae(traj: Transition, last_val):
        def scan_fn(carry, x):
            adv, next_val = carry
            reward, value, done = x
            delta = reward + config.gamma * next_val * (1.0 - done) - value
            adv = delta + config.gamma * config.gae_lambda * (1.0 - done) * adv
            return (adv, value), adv

        _, advantages = jax.lax.scan(
            scan_fn, (jnp.zeros(config.num_envs), last_val),
            (traj.reward, traj.value, traj.done), reverse=True)
        return advantages, advantages + traj.value

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.lr))

    def update(params, opt_state, avg_params, lam, traj, advantages, returns, rng):
        b = config.rollout_len * config.num_envs
        flat = jax.tree.map(lambda x: x.reshape((b,) + x.shape[2:]), traj)
        adv = advantages.reshape(b)
        ret = returns.reshape(b)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb_size = b // config.num_minibatches

        def loss_fn(params, mb):
            tr, a, r = mb
            logits, value = net.apply(params, tr.obs)
            logp = _logp_of(logits, tr.action)
            ratio = jnp.exp(logp - tr.logp)
            pg1 = ratio * a
            pg2 = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * a
            pg_loss = -jnp.minimum(pg1, pg2).mean()
            v_loss = 0.5 * ((value - r) ** 2).mean()
            probs = jax.nn.softmax(logits)
            entropy = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
            # cycle-consistency: pull toward the running-average policy
            avg_logits, _ = net.apply(jax.lax.stop_gradient(avg_params), tr.obs)
            kl = (probs * (jax.nn.log_softmax(logits)
                           - jax.nn.log_softmax(avg_logits))).sum(-1).mean()
            loss = (pg_loss + config.vf_coef * v_loss
                    - config.ent_coef * entropy + lam * kl)
            return loss, kl

        def epoch_fn(carry, _):
            params, opt_state, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, b)
            sh = jax.tree.map(lambda x: x[perm], (flat, adv, ret))

            def mb_fn(carry, i):
                params, opt_state = carry
                start = i * mb_size
                mb = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, mb_size), sh)
                (loss, kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), kl

            (params, opt_state), kls = jax.lax.scan(
                mb_fn, (params, opt_state), jnp.arange(config.num_minibatches))
            return (params, opt_state, rng), kls.mean()

        (params, opt_state, rng), kls = jax.lax.scan(
            epoch_fn, (params, opt_state, rng), None, length=config.epochs)
        return params, opt_state, rng, kls.mean()

    return net, init_runner, rollout, gae, update, optimizer


def make_selfplay_trainer(env, config: PPOConfig, opponent_mode: str,
                          magnet_mode: str = "moving", lam: float = 0.0,
                          reset_every: int = 200, extrap_gain: float = 1.0):
    """Build init/train-chunk closures for one self-play method.

    `opponent_mode`: "current" (latest self) or "average" (Polyak running average).
    `magnet_mode`: cycle-KL target --
        "none"     : no magnet (lam ignored);
        "fixed"    : frozen initial policy (MMD-style fixed magnet);
        "periodic" : snapshot of the current policy reset every `reset_every` updates
                     (R-NaD's moving reference);
        "moving"   : Polyak running-average policy (GARIP's continuous moving magnet);
        "extrap"   : double-EMA *anticipatory* magnet (1+g)*avg - g*avg2, avg2 the EMA
                     of avg, gain g=`extrap_gain` -- a negative-weight reference that
                     leads the policy (lower/negative effective lag; tests whether the
                     causal-average optimality of Prop. 1 can be beaten by extrapolation).
    """
    net, init_runner, rollout, gae, update, optimizer = make_ppo(env, config)

    def init(key):
        k1, k2 = jax.random.split(key)
        params = net.init(k1, jnp.zeros((1, env.obs_dim)))
        opt_state = optimizer.init(params)
        runner = init_runner(k2)
        # carry: (params, opt_state, avg, avg2, fixed_magnet, periodic_magnet, t, runner, rng)
        return params, opt_state, params, params, params, params, jnp.array(0.0), runner, key

    def one_update(carry, _):
        params, opt_state, avg_params, avg2_params, fixed_magnet, periodic_magnet, t, runner, rng = carry
        opp = avg_params if opponent_mode == "average" else jax.lax.stop_gradient(params)
        if magnet_mode == "fixed":
            magnet = fixed_magnet
        elif magnet_mode == "periodic":
            magnet = periodic_magnet
        elif magnet_mode == "extrap":
            magnet = jax.tree.map(
                lambda a, a2: (1.0 + extrap_gain) * a - extrap_gain * a2, avg_params, avg2_params)
        else:
            magnet = avg_params
        traj, last_val, runner, rng = rollout(params, opp, runner, rng)
        advantages, returns = gae(traj, last_val)
        params, opt_state, rng, kl = update(
            params, opt_state, magnet, lam, traj, advantages, returns, rng)
        avg_params = jax.tree.map(
            lambda a, p: (1 - config.polyak) * a + config.polyak * p, avg_params, params)
        avg2_params = jax.tree.map(
            lambda a2, a: (1 - config.polyak) * a2 + config.polyak * a, avg2_params, avg_params)
        t = t + 1.0
        # R-NaD periodic snapshot: reset the reference to the current policy every K updates.
        reset = jnp.mod(t, reset_every) < 0.5
        periodic_magnet = jax.tree.map(
            lambda m, p: jnp.where(reset, jax.lax.stop_gradient(p), m), periodic_magnet, params)
        return (params, opt_state, avg_params, avg2_params, fixed_magnet, periodic_magnet, t, runner, rng), \
            traj.reward.mean()

    @partial(jax.jit, static_argnums=1)
    def train_chunk(carry, num_updates):
        carry, rewards = jax.lax.scan(one_update, carry, None, length=num_updates)
        return carry, rewards.mean()

    return net, init, train_chunk
