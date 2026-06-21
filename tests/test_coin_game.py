"""Tests for the vendored Coin Game env and the PPO self-play machinery."""
import jax
import jax.numpy as jnp

from garip.envs import CoinGame
from garip.rl.ppo_selfplay import PPOConfig, make_selfplay_trainer
from garip.rl.exploitability import approx_exploitability


def test_reset_and_obs_shapes():
    env = CoinGame(episode_length=8)
    (o0, o1), state = env.reset(jax.random.PRNGKey(0))
    assert o0.shape == (36,) and o1.shape == (36,)
    # 4 one-hot channels (self, opp, self-coin, opp-coin) -> exactly 4 active cells
    assert float(o0.sum()) == 4.0 and float(o1.sum()) == 4.0
    assert env.num_actions == 5 and env.obs_dim == 36


def test_step_is_zero_sum():
    env = CoinGame(episode_length=8, zero_sum=True)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    (_, _), state = env.reset(key)
    for i in range(30):
        key, k = jax.random.split(key)
        (_o0, _o1), state, (r0, r1), done, info = step(k, state, jnp.array(i % 5), jnp.array((i + 2) % 5))
        assert abs(float(r0) + float(r1)) < 1e-6   # strictly zero-sum


def test_episode_terminates_and_resets():
    env = CoinGame(episode_length=4)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(1)
    (_, _), state = env.reset(key)
    dones = []
    for _ in range(8):
        key, k = jax.random.split(key)
        (_o, _o1), state, _r, done, _i = step(k, state, jnp.array(4), jnp.array(4))
        dones.append(bool(done))
    # done fires exactly on the episode-length boundary (steps 4 and 8)
    assert dones[3] and dones[7]
    assert not dones[0] and not dones[2]


def test_step_is_deterministic_under_fixed_key():
    env = CoinGame(episode_length=8)
    key = jax.random.PRNGKey(2)
    (_, _), state = env.reset(key)
    out1 = env.step(key, state, jnp.array(0), jnp.array(1))
    out2 = env.step(key, state, jnp.array(0), jnp.array(1))
    assert jnp.allclose(out1[2][0], out2[2][0])
    assert jnp.allclose(out1[1].red_pos, out2[1].red_pos)


def test_ppo_self_play_runs_and_reduces_exploitability():
    # A short CGSP self-play run should make the policy meaningfully less exploitable
    # than a fresh (near-random) policy, whose best responder scores a large return.
    env = CoinGame(episode_length=16, zero_sum=True)
    cfg = PPOConfig()
    net, init, train_chunk = make_selfplay_trainer(env, cfg, "average", "moving", 0.5)
    carry = init(jax.random.PRNGKey(0))
    e0 = approx_exploitability(env, cfg, carry[0], jax.random.PRNGKey(1), br_updates=60)
    carry, _ = train_chunk(carry, 300)
    e1 = approx_exploitability(env, cfg, carry[0], jax.random.PRNGKey(1), br_updates=60)
    assert e0 > 5.0          # fresh policy is highly exploitable
    assert e1 < e0 - 3.0     # training makes it substantially less exploitable
