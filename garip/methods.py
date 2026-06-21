"""Learning dynamics for two-player zero-sum games.

Every algorithm exposes the same tiny interface so the training loop can treat them
uniformly:

    init(game, key)          -> state      (a JAX pytree)
    step(state, game)        -> state      (pure, jit/scan-friendly)
    strategies(state)        -> (x, y)     (current mixed strategies, for metrics)

Methods:
    * CGSP (ours)  -- simultaneous gradient self-play + cycle-consistency penalty.
    * SGA          -- the same with lambda = 0 (naive self-play gradient ascent).
    * Fictitious play.
    * Mirror descent / multiplicative weights (optionally optimistic).
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from garip.games import ZeroSumGame
from garip.strategies import (
    to_simplex,
    col_best_response,
    row_best_response,
    hard_col_best_response,
    hard_row_best_response,
)


class Algorithm(NamedTuple):
    """A self-play learning dynamic with a uniform init/step/strategies interface."""

    name: str
    init: Callable[[ZeroSumGame, jax.Array], object]
    step: Callable[[object, ZeroSumGame], object]
    strategies: Callable[[object], tuple]


def _init_logits(game: ZeroSumGame, key: jax.Array, scale: float = 1.0):
    """Shared random start. All methods begin from the *same* strategy pair per seed.

    We start away from the uniform strategy on purpose: on RPS the uniform point is
    the Nash equilibrium, so a uniform start would make even naive dynamics look
    'converged'. A random non-uniform start exposes the cycling pathology.
    """
    kx, ky = jax.random.split(key)
    theta_x = scale * jax.random.normal(kx, (game.num_row_actions,))
    theta_y = scale * jax.random.normal(ky, (game.num_col_actions,))
    return theta_x, theta_y


# --------------------------------------------------------------------------- #
# CGSP (ours) and SGA (lambda = 0 ablation): simultaneous gradient self-play.
# --------------------------------------------------------------------------- #
class GradState(NamedTuple):
    theta_x: jax.Array
    theta_y: jax.Array
    opt_state: optax.OptState
    t: jax.Array  # step counter, drives the temperature schedule


def make_gradient_method(
    name: str,
    lam: float,
    tau_init: float = 0.1,
    tau_final: float = 0.1,
    anneal_steps: int = 1,
    lr: float = 0.05,
) -> Algorithm:
    """Simultaneous-gradient self-play.

    `lam = 0` is naive self-play gradient ascent (SGA). `lam > 0` adds the
    cycle-consistency penalty `||F(G(x)) - x||^2 + ||G(F(y)) - y||^2`, which is the
    novel CGSP term. The row player minimizes `-V + lam*cyc`, the column player
    minimizes `+V + lam*cyc`; selecting gradients by argnums implements the
    stop-gradient through the opponent's parameters that makes this *simultaneous*
    self-play rather than joint minimization of a single objective.

    The best-response maps are entropy-regularized with temperature `tau`. Annealing
    `tau` linearly from `tau_init` down to `tau_final` over `anneal_steps` shrinks the
    quantal-response bias, so the fixed point of the cycle term approaches the *exact*
    Nash equilibrium rather than a softened one. With `tau_init == tau_final` the
    temperature is constant.
    """
    optimizer = optax.adam(lr)

    def tau_at(t: jax.Array) -> jax.Array:
        frac = jnp.clip(t / anneal_steps, 0.0, 1.0)
        return tau_init + (tau_final - tau_init) * frac

    def cycle_loss(theta_x, theta_y, payoff, tau):
        x = to_simplex(theta_x)
        y = to_simplex(theta_y)
        fg_x = row_best_response(payoff, col_best_response(payoff, x, tau), tau)
        gf_y = col_best_response(payoff, row_best_response(payoff, y, tau), tau)
        return jnp.sum((fg_x - x) ** 2) + jnp.sum((gf_y - y) ** 2)

    def row_loss(theta_x, theta_y, payoff, tau):
        x = to_simplex(theta_x)
        y = to_simplex(theta_y)
        value = x @ payoff @ y
        return -value + lam * cycle_loss(theta_x, theta_y, payoff, tau)

    def col_loss(theta_x, theta_y, payoff, tau):
        x = to_simplex(theta_x)
        y = to_simplex(theta_y)
        value = x @ payoff @ y
        return value + lam * cycle_loss(theta_x, theta_y, payoff, tau)

    def init(game: ZeroSumGame, key: jax.Array) -> GradState:
        theta_x, theta_y = _init_logits(game, key)
        opt_state = optimizer.init((theta_x, theta_y))
        return GradState(theta_x, theta_y, opt_state, jnp.array(0.0))

    def step(state: GradState, game: ZeroSumGame) -> GradState:
        payoff = game.payoff
        tau = tau_at(state.t)
        grad_x = jax.grad(row_loss, argnums=0)(state.theta_x, state.theta_y, payoff, tau)
        grad_y = jax.grad(col_loss, argnums=1)(state.theta_x, state.theta_y, payoff, tau)
        updates, opt_state = optimizer.update((grad_x, grad_y), state.opt_state)
        theta_x, theta_y = optax.apply_updates((state.theta_x, state.theta_y), updates)
        return GradState(theta_x, theta_y, opt_state, state.t + 1.0)

    def strategies(state: GradState):
        return to_simplex(state.theta_x), to_simplex(state.theta_y)

    return Algorithm(name=name, init=init, step=step, strategies=strategies)


def cgsp(
    lam: float = 1.0,
    tau_init: float = 0.5,
    tau_final: float = 0.08,
    anneal_steps: int = 4000,
    lr: float = 0.05,
) -> Algorithm:
    """CycleGAN self-play (ours), with temperature annealing toward exact Nash."""
    return make_gradient_method(
        "cgsp", lam=lam, tau_init=tau_init, tau_final=tau_final,
        anneal_steps=anneal_steps, lr=lr,
    )


def sga(lr: float = 0.05) -> Algorithm:
    """Naive self-play gradient ascent (the cycling baseline).

    `lam = 0` means the cycle term (and hence `tau`) is never used, so this is exactly
    simultaneous gradient play regardless of the temperature settings.
    """
    return make_gradient_method("sga", lam=0.0, lr=lr)


# --------------------------------------------------------------------------- #
# CGSP-quantal: the unified formulation that scales to imperfect-information games
# (Kuhn, Leduc). The cycle target is the smoothed best response to the *running-average*
# opponent (the fictitious-play insight), and the reported strategy is the average.
# --------------------------------------------------------------------------- #
class QuantalState(NamedTuple):
    theta_x: jax.Array
    theta_y: jax.Array
    opt_state: optax.OptState
    avg_x: jax.Array
    avg_y: jax.Array
    t: jax.Array


def cgsp_quantal(
    lam: float = 1.0,
    tau_init: float = 0.5,
    tau_final: float = 0.08,
    anneal_steps: int = 4000,
    lr: float = 0.05,
) -> Algorithm:
    """CGSP with the averaged-opponent quantal best-response target.

    Identical in spirit to the method used on Kuhn and Leduc: each player is pulled
    toward its (entropy-regularized) best response to the opponent's *average* strategy
    while still descending the adversarial game value. The slowly-moving averaged target
    is what makes this stable at scale. The reported strategy is the running average (the
    apples-to-apples object versus fictitious play / CFR), so its exploitability is what
    the curves show.
    """
    optimizer = optax.adam(lr)

    def tau_at(t):
        frac = jnp.clip(t / anneal_steps, 0.0, 1.0)
        return tau_init + (tau_final - tau_init) * frac

    def row_loss(theta_x, theta_y, payoff, target_x, tau):
        x = to_simplex(theta_x)
        y = to_simplex(theta_y)
        return -(x @ payoff @ y) + lam * jnp.sum((target_x - x) ** 2)

    def col_loss(theta_x, theta_y, payoff, target_y, tau):
        x = to_simplex(theta_x)
        y = to_simplex(theta_y)
        return (x @ payoff @ y) + lam * jnp.sum((target_y - y) ** 2)

    def init(game, key):
        theta_x, theta_y = _init_logits(game, key)
        x, y = to_simplex(theta_x), to_simplex(theta_y)
        opt_state = optimizer.init((theta_x, theta_y))
        return QuantalState(theta_x, theta_y, opt_state, x, y, jnp.array(1.0))

    def step(state: QuantalState, game) -> QuantalState:
        payoff = game.payoff
        tau = tau_at(state.t)
        # cycle targets = smoothed best response to the *average* opponent (stop-grad)
        target_x = jax.lax.stop_gradient(row_best_response(payoff, state.avg_y, tau))
        target_y = jax.lax.stop_gradient(col_best_response(payoff, state.avg_x, tau))
        grad_x = jax.grad(row_loss, argnums=0)(state.theta_x, state.theta_y, payoff, target_x, tau)
        grad_y = jax.grad(col_loss, argnums=1)(state.theta_x, state.theta_y, payoff, target_y, tau)
        updates, opt_state = optimizer.update((grad_x, grad_y), state.opt_state)
        theta_x, theta_y = optax.apply_updates((state.theta_x, state.theta_y), updates)
        x, y = to_simplex(theta_x), to_simplex(theta_y)
        t1 = state.t + 1.0
        avg_x = state.avg_x + (x - state.avg_x) / t1
        avg_y = state.avg_y + (y - state.avg_y) / t1
        return QuantalState(theta_x, theta_y, opt_state, avg_x, avg_y, t1)

    def strategies(state: QuantalState):
        return state.avg_x, state.avg_y  # report the average strategy

    return Algorithm("cgsp_quantal", init, step, strategies)


# --------------------------------------------------------------------------- #
# Fictitious play.
# --------------------------------------------------------------------------- #
class FPState(NamedTuple):
    x_avg: jax.Array
    y_avg: jax.Array
    t: jax.Array  # number of best responses accumulated so far


def fictitious_play() -> Algorithm:
    """Each player best-responds to the opponent's empirical average strategy.

    Robinson (1951): the time-average converges to Nash in zero-sum games. We report
    the running averages as the current strategies.
    """

    def init(game: ZeroSumGame, key: jax.Array) -> FPState:
        theta_x, theta_y = _init_logits(game, key)
        return FPState(to_simplex(theta_x), to_simplex(theta_y), jnp.array(1.0))

    def step(state: FPState, game: ZeroSumGame) -> FPState:
        payoff = game.payoff
        br_x = hard_row_best_response(payoff, state.y_avg)
        br_y = hard_col_best_response(payoff, state.x_avg)
        t1 = state.t + 1.0
        x_avg = state.x_avg + (br_x - state.x_avg) / t1
        y_avg = state.y_avg + (br_y - state.y_avg) / t1
        return FPState(x_avg, y_avg, t1)

    def strategies(state: FPState):
        return state.x_avg, state.y_avg

    return Algorithm("fictitious_play", init, step, strategies)


# --------------------------------------------------------------------------- #
# Mirror descent / multiplicative weights (optionally optimistic).
# --------------------------------------------------------------------------- #
class MDState(NamedTuple):
    x: jax.Array
    y: jax.Array
    g_x_prev: jax.Array
    g_y_prev: jax.Array


def mirror_descent(eta: float = 0.1, optimistic: bool = True) -> Algorithm:
    """Entropic mirror descent (multiplicative weights) on the bilinear game.

    Row ascends its payoff gradient `A y`, column descends `x^T A`. The optimistic
    variant predicts the next gradient as `2 g_t - g_{t-1}`, which yields last-iterate
    convergence in zero-sum games; the vanilla variant only converges on average.
    """
    name = "optimistic_md" if optimistic else "mirror_descent"

    def init(game: ZeroSumGame, key: jax.Array) -> MDState:
        theta_x, theta_y = _init_logits(game, key)
        x = to_simplex(theta_x)
        y = to_simplex(theta_y)
        g_x = game.payoff @ y
        g_y = x @ game.payoff
        return MDState(x, y, g_x, g_y)

    def step(state: MDState, game: ZeroSumGame) -> MDState:
        payoff = game.payoff
        g_x = payoff @ state.y          # row payoff gradient (maximize)
        g_y = state.x @ payoff          # column payoff gradient (minimize)
        if optimistic:
            pred_x = 2.0 * g_x - state.g_x_prev
            pred_y = 2.0 * g_y - state.g_y_prev
        else:
            pred_x, pred_y = g_x, g_y
        x = state.x * jnp.exp(eta * pred_x)
        x = x / jnp.sum(x)
        y = state.y * jnp.exp(-eta * pred_y)
        y = y / jnp.sum(y)
        return MDState(x, y, g_x, g_y)

    def strategies(state: MDState):
        return state.x, state.y

    return Algorithm(name, init, step, strategies)


# --------------------------------------------------------------------------- #
# GARIP (Generative Adversarial Reciprocal Iterative Play) -- the novel method.
#
# Regularized self-play (MMD, NeuRD) anchors each iterate to a *fixed* magnet and must
# shrink the regularization to 0 to reach exact Nash, which is unstable. GARIP instead
# anchors to a *moving, self-consistent* target -- the running-average iterate (the
# reciprocal cycle-consistency point) -- combined with optimism. Because the anchor
# tracks the equilibrium as it forms, the last iterate converges to *exact* Nash with a
# *constant* anchor strength (no annealing).
# --------------------------------------------------------------------------- #
class GaripState(NamedTuple):
    x: jax.Array
    y: jax.Array
    avg_x: jax.Array
    avg_y: jax.Array
    g_x_prev: jax.Array
    g_y_prev: jax.Array
    t: jax.Array


def garip(eta: float = 0.3, beta: float = 0.02) -> Algorithm:
    """GARIP: optimistic mirror ascent/descent with a Halpern anchor to the running
    average. `beta` is the (constant) anchor strength; `eta` the step size."""

    def init(game: ZeroSumGame, key: jax.Array) -> GaripState:
        theta_x, theta_y = _init_logits(game, key)
        x, y = to_simplex(theta_x), to_simplex(theta_y)
        g_x = game.payoff @ y
        g_y = x @ game.payoff
        return GaripState(x, y, x, y, g_x, g_y, jnp.array(1.0))

    def step(state: GaripState, game: ZeroSumGame) -> GaripState:
        payoff = game.payoff
        g_x = payoff @ state.y          # row ascends
        g_y = state.x @ payoff          # col descends
        pred_x = 2.0 * g_x - state.g_x_prev   # optimistic prediction
        pred_y = 2.0 * g_y - state.g_y_prev
        x_half = state.x * jnp.exp(eta * pred_x)
        x_half = x_half / jnp.sum(x_half)
        y_half = state.y * jnp.exp(-eta * pred_y)
        y_half = y_half / jnp.sum(y_half)
        # Halpern anchor toward the (moving, self-consistent) running average.
        x = (1.0 - beta) * x_half + beta * state.avg_x
        y = (1.0 - beta) * y_half + beta * state.avg_y
        t1 = state.t + 1.0
        avg_x = state.avg_x + (x - state.avg_x) / t1
        avg_y = state.avg_y + (y - state.avg_y) / t1
        return GaripState(x, y, avg_x, avg_y, g_x, g_y, t1)

    def strategies(state: GaripState):
        return state.x, state.y  # the LAST iterate -- GARIP's headline object

    return Algorithm("garip", init, step, strategies)


# --------------------------------------------------------------------------- #
# Magnetic Mirror Descent (Sokota et al. 2023) -- the key regularized baseline.
# --------------------------------------------------------------------------- #
class MMDState(NamedTuple):
    x: jax.Array
    y: jax.Array


def mmd(eta: float = 0.1, alpha: float = 0.1) -> Algorithm:
    """MMD with a uniform magnet. Last-iterate converges to the alpha-regularized
    (quantal-response) equilibrium; reaching exact Nash needs alpha -> 0."""

    def init(game: ZeroSumGame, key: jax.Array) -> MMDState:
        theta_x, theta_y = _init_logits(game, key)
        return MMDState(to_simplex(theta_x), to_simplex(theta_y))

    def step(state: MMDState, game: ZeroSumGame) -> MMDState:
        payoff = game.payoff
        g_x = payoff @ state.y
        g_y = state.x @ payoff
        c = 1.0 / (1.0 + eta * alpha)
        mag_x = jnp.ones_like(state.x) / state.x.shape[0]
        mag_y = jnp.ones_like(state.y) / state.y.shape[0]
        # Closed-form KL-proximal MMD update toward the magnet.
        x = (state.x ** c) * (mag_x ** (1.0 - c)) * jnp.exp(eta * c * g_x)
        x = x / jnp.sum(x)
        y = (state.y ** c) * (mag_y ** (1.0 - c)) * jnp.exp(-eta * c * g_y)
        y = y / jnp.sum(y)
        return MMDState(x, y)

    def strategies(state: MMDState):
        return state.x, state.y

    return Algorithm("mmd", init, step, strategies)


# --------------------------------------------------------------------------- #
# R-NaD (Regularized Nash Dynamics, Perolat et al. 2022 -> DeepNash) -- the key
# moving-reference baseline. Same KL-proximal update as MMD, but the magnet is a
# *periodic snapshot* of the current policy (reset every K steps), not fixed. The
# sequence of regularized fixed points provably converges to Nash with constant alpha
# (no annealing) -- the direct competitor to GARIP's running-average anchor.
# --------------------------------------------------------------------------- #
class RNaDState(NamedTuple):
    x: jax.Array
    y: jax.Array
    magnet_x: jax.Array
    magnet_y: jax.Array
    t: jax.Array


def rnad(eta: float = 0.1, alpha: float = 1.0, reset_every: int = 200) -> Algorithm:
    """R-NaD: MMD-style KL-proximal updates toward a magnet that is reset to the current
    policy every `reset_every` steps (the periodic-snapshot moving reference)."""

    def init(game: ZeroSumGame, key: jax.Array) -> RNaDState:
        theta_x, theta_y = _init_logits(game, key)
        x, y = to_simplex(theta_x), to_simplex(theta_y)
        return RNaDState(x, y, x, y, jnp.array(1.0))

    def step(state: RNaDState, game: ZeroSumGame) -> RNaDState:
        payoff = game.payoff
        g_x = payoff @ state.y
        g_y = state.x @ payoff
        c = 1.0 / (1.0 + eta * alpha)
        x = (state.x ** c) * (state.magnet_x ** (1.0 - c)) * jnp.exp(eta * c * g_x)
        x = x / jnp.sum(x)
        y = (state.y ** c) * (state.magnet_y ** (1.0 - c)) * jnp.exp(-eta * c * g_y)
        y = y / jnp.sum(y)
        t1 = state.t + 1.0
        reset = (jnp.mod(t1, reset_every) < 0.5)   # periodic snapshot of the current policy
        magnet_x = jnp.where(reset, x, state.magnet_x)
        magnet_y = jnp.where(reset, y, state.magnet_y)
        return RNaDState(x, y, magnet_x, magnet_y, t1)

    def strategies(state: RNaDState):
        return state.x, state.y  # last iterate

    return Algorithm("rnad", init, step, strategies)


def all_methods() -> list[Algorithm]:
    """The standard suite compared in the experiments."""
    return [
        garip(),
        rnad(alpha=0.5, reset_every=300),
        mmd(),
        cgsp(),
        sga(),
        fictitious_play(),
        mirror_descent(optimistic=True),
    ]
