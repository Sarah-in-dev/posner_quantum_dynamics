# Cross-Domain Spec — Expected-Free-Energy Objective over the Sheaf Substrate

**Status: SPEC (math/framing locked; build open).** 2026-07-27.
**Owner of this artifact:** the math/framing (Claude + Sarah, quantum-side). **Built/grounded by:** the TALON
side, against the real substrate (murmur repo — not visible from this posner repo). **Meeting point: the Q(s)
definition (§1).**

**Scope.** One objective layer, two substrates. TALON (a sheaf over PAUL's operational graph) and Model 6 (the
entanglement-partition substrate) are the *same mathematical object* per `talon-architectural-north-star` §"The
thesis". This spec defines the expected-free-energy (active-inference) functional over that object so it ports
both ways. **Strategic:** obstruction + per-edge precision are already live in TALON (see below), so TALON may
instantiate this objective *before* Model 6; write once, port back.

**Grounding (what this is built on, tagged).**
- `[skill]` `talon-architectural-north-star` §"The active-inference reach": the functional is "descriptively
  accurate but operationally absent… hold as north-star, not a build target… lift becomes possible once the
  substrate is rich enough." This spec *is* the "rich enough" definition.
- `[TALON code, verified by Sarah 2026-07-27]` `consistency_energy.compute_energy` → **E(c)=‖δc‖²=⟨c, L_F c⟩**
  (H⁰ obstruction energy); `sheaf_consistency.semantic_gluing` → H⁰ + a **located** obstruction list. Per-edge
  precision **w_e = 1/σ_e²**, σ_e = robust MAD noise scale from the BaselinePass (landed 2026-07-13). Two of the
  three blockers the May-18 skill named are therefore stale; the one genuine gate is the cross-layer join (§1).
- `[data]` PO-11 (this repo): the within-condition partial-correlation readout — a **precision-matrix estimate**
  — recovers co-membership leak-immune. This is the empirical validation of the Q(s) coupling estimator (§5).

---

## 0. The insight that collapses the problem

`compute_energy` returns E(c)=⟨c, L_F c⟩ with **L_F = δ*Wδ** the W-weighted sheaf Laplacian, W=diag(w_e),
w_e=1/σ_e². That quadratic form is minus-twice the log-density of a Gaussian sheaf field:

> **P(s) ∝ exp(−½⟨s, L_F s⟩) is a GMRF whose precision matrix is exactly L_F.**

You are already computing the negative-log-posterior. **Q(s) is the posterior it defines.** Obstruction *is* the
prior precision; `w_e` *are* the couplings. The only missing piece is fusing the four layers' evidence into the
same precision — that is Q(s).

## 1. Q(s) — the stalk posterior (KEYSTONE, the meeting point)

**Q(s) = 𝒩(μ, Λ⁻¹)**,  μ = Λ⁻¹h,  h = h_obs + εμ₀, with precision assembled additively:

> **Λ = L_F(W) + D_obs + εI**

- **L_F(W)** — consistency prior, *already live* (`consistency_energy`). Equality edges → weighted graph
  Laplacian on stalk space; general edges → δ*Wδ with the actual restriction maps R_{e,v}.
- **εI** — ridge; keeps un-observed, un-coupled stalks proper (ε = inverse prior variance).
- **D_obs** — observation precision, block-diagonal per node. **The cross-layer join:**

> **D_obs[v] = Σ_ℓ β_ℓ · T_v^ℓ**,  **h_obs[v] = Σ_ℓ β_ℓ · T_v^ℓ · y_v^ℓ**,  ℓ ∈ {struct, stat, ts, sem}

- **T_v^ℓ** = per-layer precision (already have): stat = 1/σ² (BaselinePass MAD); ts = 1/forecast-var;
  struct = invariant/check precision; sem = 1/embedding-uncertainty (retrieval distance or LLM confidence).
- **β_ℓ** = **the one new learned quantity** — a per-layer calibration scalar putting the four raw precisions on
  one honest scale before they add. Without it, an overconfident layer dominates the fusion. **This is the whole
  risk surface, and where "calibrated" is earned or lost.**

**Fitting β_ℓ.** On held-out outcomes from the Layer-F calibration loop, choose β_ℓ so the fused z-scores
z_v=(s_v−μ_v)/√Σ_vv have unit variance (≡ minimize the negative log-predictive density of observed node values
under Q). Practically β_ℓ ∝ claimed-precision / empirical-residual-precision — a per-layer reliability rescaling.
A handful of scalars; convex given fixed structure. **If a layer will not calibrate, set β_ℓ→0 (drop it) rather
than trust it.** (Closed-form calibration objective = open next step (a), §7.)

**Non-Gaussian stalks.** Exact for continuous layers. Semantic stalks (embeddings/categoricals) enter via a
Laplace approximation — Λ-block = Hessian of that layer's log-likelihood at the MAP — joining the same precision
sum. Heavy-tailed discordance (real drift = outliers): keep MAD robustness, or Student-t likelihood with
IRLS-reweighted w_e.

## 2. Pragmatic term — expected obstruction resolved by probe π

Preference C = a consistent substrate: **P(s|C) ∝ exp(−½ γ ⟨s, L_F s⟩)**, γ = how much you want consistency.

> **G_prag(π) = γ · E_{Q(s|π)}[⟨s, L_F s⟩] = γ ( ⟨μ_π, L_F μ_π⟩ + tr(L_F Σ_π) )**

⟨μ, L_F μ⟩ *is* `compute_energy(μ)`; tr(L_F Σ) is the uncertainty contribution. **Build-today form** over the
located H⁰ list (per-edge r_e = w_e‖δμ|_e‖²): score a probe by the incident obstruction it positions you to
resolve,

> **G_prag(probe v) = −γ Σ_{e∋v} r_e**   ("probe where the located obstruction is")

The two agree *for policy selection*: total expected obstruction = const − (reduction from π), so
argmin_π of the energy form = argmax of the incident-obstruction form. Build with the located list; keep the
energy form as the principled definition.

## 3. Epistemic term — expected uncertainty reduction (info gain)

Probing v observes s_v with the probe's precision τ_π (rank-1 update Λ ↦ Λ + τ_π e_v e_vᵀ). Exact
whole-substrate Gaussian info gain (coupling propagates it beyond v):

> **G_epist(probe v) = −½ log(1 + τ_π · Σ_vv)**,  Σ_vv = [Λ⁻¹]_vv

High when v is uncertain (Σ_vv large) AND the probe is precise (τ_π large). Σ_vv is the covariance diagonal from
the **same Λ** — via **selected inverse (Takahashi recursion) on the sparse L_F+D_obs, O(nnz); never
dense-invert.** (Recursion = open next step (b), §7.)

## 4. Policy + stopping

> **G(π) = G_prag(π) + G_epist(π) [+ cost(π)]**

- **Agenda:** π* = argmin_π G(π) — probe where obstruction is high AND uncertainty is high AND the probe is
  precise/cheap. Rank candidates by G, run the top; re-solve Λ; repeat.
- **Stopping (principled — replaces the manual thresholds):** stop when **min_π G(π) ≥ −ε_stop** — no available
  probe is expected to drop free energy by more than ε_stop. Autonomy stops probing exactly when EFE stops
  dropping.

## 5. Port both ways — and PO-11 is already the validation

The identity is exact, not analogy:
- **L_F(W) precision ↔ Model 6's Werner-thresholded bond graph** (P_S weights); connected components of L_F ↔
  Model 6's partition (the coherent-update regions).
- **PO-11's readout IS a precision estimate.** Within-condition partial correlation = **−Λ_ij / √(Λ_ii Λ_jj)**,
  the normalized precision off-diagonal. What PO-11 proved leak-immune — recover co-membership from the
  *precision*, not from magnitudes/abundance — is **exactly the guarantee the coupling block of Q(s) needs.**
  PO-11 is the empirical validation of the Q(s) coupling estimator, on the Model-6 substrate.
- **Pragmatic (resolve obstruction) ↔ Model 6's reward-directed valence** (which partition to strengthen) — both
  are the "to what end." **Epistemic (sample to reduce stalk uncertainty) ↔ per-cluster stochastic measurement**
  (sample the partition).
- **Port direction:** TALON first (L_F + w_e live; only β_ℓ missing). Build β_ℓ + the Λ assembly TALON-side; port
  the same Λ/EFE structure to Model 6, where "substrate rich enough" is the loop-engagement problem audited
  2026-07-27 (substrate barely engages under navigation drive; partition is a blob, not structured). One object,
  two stalk-fillings.

## 6. Own these before building
1. **β_ℓ calibration is the whole game** — uncalibrated per-layer precision → a confidently-wrong posterior.
   Needs real outcome data; drop layers that will not calibrate.
2. **Restriction maps** — equality edges free (identity restriction); non-equality edges (derives-from + a
   transform) need R_{e,v} specified or L_F is wrong.
3. **Laplace approximation** for semantic stalks is local — watch multimodality.
4. **Selected inverse, not dense**, for Σ_vv at substrate scale.

## 7. Open next math steps
- **(a) β_ℓ calibration objective in closed form** — the reliability fit with the unit-z-variance / coverage
  constraint. **RECOMMENDED FIRST: it is the keystone/correctness step; the whole posterior's calibration rides
  on it, and everything in §2–4 inherits its miscalibration if it is wrong.**
- **(b) Takahashi selected-inverse recursion** against a sparse L_F+D_obs so Σ_vv is cheap at scale. **An
  efficiency step, needed only at scale — a small prototype can dense-invert.** Do after (a).

## 8. β_ℓ calibration — the closed form (delivers step (a), 2026-07-27)

**Target.** When a probe/outcome reveals a node's true value s_v*, the posterior's standardized residual is
z_v = (s_v* − μ_v)/√Σ_vv. Calibrated ⟺ E[z]=0 and **Var[z]=1** (coverage form: |z_v| ≤ Φ⁻¹((1+α)/2) at rate α).
Var[z]>1 ⇒ overconfident (Σ too small); <1 ⇒ underconfident.

**Per-layer closed form (the reliability fit).** Fit each layer against its own observation error. Over the
calibration nodes layer ℓ observed, its claimed-precision-weighted mean squared error (its *reduced χ²*) is

  **χ̄²_ℓ = (1/N_ℓ) Σ_{v: ℓ observes v} T_v^ℓ (s_v* − y_v^ℓ)²**

and the unit-z-variance constraint gives, in closed form,

  **β_ℓ = 1 / χ̄²_ℓ.**

Read it directly: calibrated layer ⇒ χ̄²_ℓ≈1 ⇒ β_ℓ≈1; overconfident (errors exceed claimed precision) ⇒
χ̄²_ℓ>1 ⇒ β_ℓ<1 (down-weight); underconfident ⇒ β_ℓ>1. The §1 reliability rescaling made exact — no optimization.

**Robust / coverage variant** (heavy-tailed discordance = real drift): match the *median* of the per-node terms
to the χ²₁ median 0.4549 — **β_ℓ = 0.4549 / median_v[ T_v^ℓ (s_v* − y_v^ℓ)² ]**. Consistent with the MAD scale
already in w_e; use when outlier discordances would inflate the mean.

**Online form (plugs into the Layer-F loop).** One EWMA scalar per layer; on each revealed outcome s_v*, for
each layer ℓ that observed v:

  χ̄²_ℓ ← (1−ρ) χ̄²_ℓ + ρ · T_v^ℓ (s_v* − y_v^ℓ)²,   β_ℓ ← 1 / χ̄²_ℓ.

One scalar per layer, updated per outcome — no batch refit. ρ = the loop's forgetting rate.

**Guards.**
- Non-negativity automatic (β_ℓ = 1/χ̄²_ℓ > 0).
- **Small N_ℓ:** shrink toward calibrated — χ̄²_ℓ ← (N_ℓ χ̄²_ℓ + κ)/(N_ℓ + κ) ⇒ β_ℓ→1 until a layer earns
  otherwise (κ = pseudo-count).
- **Won't-calibrate:** if χ̄²_ℓ stays large and unstable, set β_ℓ→0 (drop the layer) — the §1 rule.
- **Scale anchor:** w_e and the stat-layer T both come from the BaselinePass MAD → one shared scale; anchor
  β_stat there and calibrate the other three relative to it (resolves β-vs-L_F identifiability).

**Exact fused refinement (when the per-layer fit leaves the *fused* z-variance off 1).** The per-layer form
calibrates each layer in isolation; it does not see the fusion + L_F prior interacting. Close it by maximizing
the held-out fused predictive log-density Σ_v log 𝒩(s_v*; μ_v, Σ_vv) over β≥0. Derivatives are closed-form
(∂Λ/∂β_ℓ = T^ℓ):

  **∂μ/∂β_ℓ = Σ T^ℓ (y^ℓ − μ),   ∂Σ_vv/∂β_ℓ = −[ Σ T^ℓ Σ ]_vv**,

so a few Newton steps warm-started at β_ℓ = 1/χ̄²_ℓ converge fast (reuse the §3 selected-inverse for the Σ
blocks). Use held-out / LOO nodes so the posterior does not grade its own homework.

**Model-6 port.** Same fit; the "layers" are the readout arms. PO-11's leak-immunity is the guarantee that what
gets calibrated is the *coupling precision* (partial correlation = precision off-diagonal), not the magnitudes —
so the β_ℓ fit rescales honest precisions, not the abundance channel that leaks.

## Boundary
Math/framing is owned here (this doc). The TALON side builds and grounds Q(s) — the four T_v^ℓ and the β_ℓ fit —
against the real substrate, which is not visible from the posner repo. **We meet at §1.**
