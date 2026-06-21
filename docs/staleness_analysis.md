# Why a periodic-snapshot magnet collapses and a running-average magnet does not

An informal analysis of the empirical finding (see the README): at matched peak performance,
**R-NaD collapses to an exploitable policy on ~32% of hyperparameters while GARIP collapses on ~7%**,
with the collapse concentrated in R-NaD's high-`λ` / large-`K` corner. We give the mechanism and a
falsifiable prediction, then confirm it. This is a *proof sketch*, not a theorem — assumptions are
flagged, and the deep-RL experiments are the real evidence.

## Setup

Regularized self-play. At step `t` each player updates its strategy/policy `θ_t` by trading off the
adversarial game signal against a KL pull toward a **magnet** `m_t`:

```
θ_{t+1} = argmax_π   ⟨ game-improvement vs. opponent ν_t , π ⟩  −  λ · KL(π ‖ m_t)
```

The three methods differ only in `(ν_t, m_t)`:

| method | opponent `ν_t` | magnet `m_t` |
|--------|----------------|--------------|
| MMD    | current `θ_t`  | **fixed** `m_0` (e.g. uniform) |
| R-NaD  | current `θ_t`  | **periodic snapshot** `θ_{r(t)}`, `r(t)=K⌊t/K⌋` |
| GARIP  | running avg `φ_t` | running avg `φ_t`, `φ_t=(1−ρ)φ_{t−1}+ρθ_t` |

Let `D := sup_t ‖θ_{t+1} − θ_t‖` be the maximum per-step policy change (TV distance on strategies, or
parameter norm). Assume `D` is bounded (true under a fixed learning rate).

## 1. Magnet lag

Define a magnet's **lag** `L_t := ‖θ_t − m_t‖`: how far the anchor is from the current policy.

**Claim 1 (R-NaD lag grows with `K`).** Since the snapshot is held for up to `K` steps,
```
L_t^{R-NaD} = ‖θ_t − θ_{r(t)}‖ ≤ Σ_{s=r(t)}^{t-1} ‖θ_{s+1}−θ_s‖ ≤ (t−r(t))·D ≤ K·D,
```
and the bound is tight just before a reset (`t−r(t)=K−1`). So `sup_t L_t^{R-NaD} = Θ(K·D)` — it
**sawtooths** between `0` (just after a reset) and `≈ K·D` (just before), and is **unbounded in `K`**.

**Claim 2 (GARIP lag is bounded, independent of horizon).** With `φ_t = ρ Σ_{s≤t}(1−ρ)^{t−s}θ_s`,
```
L_t^{GARIP} = ‖θ_t − φ_t‖ = ‖ρ Σ_s (1−ρ)^{t−s}(θ_t−θ_s)‖
            ≤ ρ Σ_s (1−ρ)^{t−s} (t−s) D = D·ρ·(1−ρ)/ρ² = D(1−ρ)/ρ ≤ D/ρ.
```
So `sup_t L_t^{GARIP} ≤ D/ρ`, **constant in `t`** and never reached as a stale single point — the
magnet is always a *valid mixture of recent policies*.

For a matched effective horizon (`K ≈ 1/ρ`) the worst-case lags are comparable; the difference that
matters is **what the lag is to**: R-NaD's lag is to one *stale* policy that the magnet snaps to and
holds; GARIP's is to a *smooth recent average*.

## 2. Opponent–magnet consistency (the sharper mechanism)

What actually destabilizes a regularized best response is a **mismatch** between the opponent it must
respond to and the magnet it is pulled toward:
```
Δ_t := ‖ ν_t − m_t ‖.
```

- **GARIP is reciprocally consistent:** `ν_t = m_t = φ_t` (it plays *and* anchors to the same average),
  so `Δ_t = 0`. The KL pull and the game signal point at the *same* reference. (This is the "reciprocal"
  in GARIP — the cycle-consistency `F(G(σ))≈σ` is exactly "be a best response to the thing you anchor to.")
- **R-NaD is mismatched:** `ν_t = θ_t` (current self) but `m_t = θ_{r(t)}` (a stale snapshot), so
  `Δ_t = L_t^{R-NaD} ≤ K·D`. The policy is asked to beat a *current, improving* opponent while being
  pulled back toward an *old* version of itself.

## 3. Collapse condition

The regularized fixed point biases toward the magnet with a force `∝ λ·Δ_t` (the KL gradient scales
with `λ` and with how far the magnet is from where the game wants the policy to be). Collapse — the
policy becoming a poor response to the current opponent, i.e. **exploitable** — occurs once this stale
force overwhelms the equilibrium-tracking signal `G`:
```
λ · Δ_t  ≳  G   ⟹   collapse.
```
Substituting the two designs:

- **R-NaD:** `λ · Δ_t ≈ λ·K·D`. Collapse predicted once **`λ·K ≳ G/D`** — a threshold on the
  *product* `λK`. So collapse should appear along high-`λ`/large-`K` contours and *worsen monotonically
  with `λK`*.
- **GARIP:** `Δ_t = 0` (consistency) and lag bounded by `D/ρ`. There is no stale force, so no
  `λ`-driven collapse region; the only failure is the generic too-aggressive-step regime.

## 4. Prediction vs. evidence

The prediction — **R-NaD collapses on `λK`, GARIP does not** — matches the deep-RL Coin Game sweep
([`results/coin_sensitivity.png`](../results/coin_sensitivity.png), 10 seeds):

| R-NaD config (λ, K) | λ·K | exploit return |
|---------------------|-----|----------------|
| (1, 100) | 100 | −11.0 (robust) |
| (1, 200) | 200 | −6.1 |
| (0.5, 800) | 400 | −1.5 |
| (1, 400) | 400 | **+2.3 (collapsed)** |
| (1, 800) | 800 | **+7.3** |
| (2, 400) | 800 | **+9.2** |
| (2, 800) | 1600 | **+12.0** |

Collapse (exploit return `> 0`) appears once `λK ≳ 400` and grows monotonically with `λK`, exactly as
predicted. GARIP's grid has no such region — only one isolated too-aggressive corner cell — giving the
~7% vs ~32% collapse rates. The matrix-game sweep ([`results/sensitivity_heatmaps.png`](../results/sensitivity_heatmaps.png))
shows the same shape (R-NaD only converges in the low-`α` band).

## Honest caveats

- **Informal.** This is a mechanism + scaling argument, not a convergence theorem; the `∝ λ·Δ`
  force and the collapse threshold are heuristic. A real result would bound the regularized
  best-response bias as a function of `λ·Δ` and the game's payoff range, and prove a basin-of-no-collapse.
- **Opponent vs. magnet confound.** GARIP differs from R-NaD in *both* opponent (`average` vs `current`)
  and magnet (`average` vs `snapshot`). The consistency argument (§2) says the magnet matters via the
  *mismatch* `Δ`, but a clean ablation — R-NaD's snapshot magnet with an `average` opponent — is needed
  to fully isolate it. (Predicted: still collapses, because `Δ = ‖φ_t − θ_{r(t)}‖` is still `Θ(KD)`.)
- **Scope.** The proximal-point theory underlying R-NaD guarantees tabular convergence *in the limit of
  inner convergence per phase*; the collapse here is a finite-`K`, finite-step, stochastic phenomenon
  the asymptotic theory does not cover.
