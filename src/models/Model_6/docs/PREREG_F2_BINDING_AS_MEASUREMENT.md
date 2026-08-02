# PRE-REGISTRATION — F2: ground the MEASUREMENT trigger (binding-as-measurement)

**Written 2026-08-02, BEFORE the model wiring change.** Second quantum pillar from the `ISO-1`
reorder: the collapse currently fires on an INVENTED boolean `dopamine_present =
stimulus.get('reward', False)` (`multi_synapse_network.py:1776`; per-dimer twin gated on
`dopamine_read and calcium_elevated`, `model6_core.py:635`). "Dopamine" appears in NONE of the source
physics. This grounds the trigger on Fisher's real readout — a spin-selective molecular BINDING event —
and demotes reward/dopamine to a SEPARATE learning signal. The derived rate module `posner_binding.py`
(committed alongside this doc, `__main__` PASSES) is the F1-style "derive → reproduce the literature"
half; this doc pre-registers the wiring and its acceptance. **Emergent-only** (Sarah, ratified): every
constant on the causal path is cited/derived; none tuned to a learning result.

## The mechanism (cited, not invented)
- **Fisher 2015** (arXiv 1508.05929, abstract): *"Quantum measurements can occur when a pair of Posner
  molecules chemically BIND and subsequently MELT, releasing a shower of intra-cellular calcium ions."*
- **Fisher & Radzihovsky 2018** (PNAS 115(20):E4551, arXiv 1707.05320): **Quantum Dynamical Selection** —
  bond processes are precluded "from orbitally non-symmetric molecular states"; the reactive channel is
  the total **spin-0 (SINGLET)**.
- **Frontiers Pharmacol. 2026** (10.3389/fphar.2026.1777613): the coherent singlet preserves the symmetry
  (suppresses *premature* dissolution); two clusters sharing a singlet dissolve **correlated in time** —
  i.e. joint per-connected-component collapse, which `perform_quantum_measurement` already implements.
- **Straub, Patel, Fisher et al., PNAS 122(10) e2423211122 (2025):** ⁷Li promotes a GREATER abundance of
  observable Ca-phosphate particles than ⁶Li — the isotope observable F2 must reproduce in DIRECTION.

## What is already correct in the code (SHOWN — do not rebuild)
- **Spin-selective dissolution EXISTS:** `ca_triphosphate_complex.py:418` — `k_diss = k_classical *
  (1.0 - singlet_excess) * template_enhancement`. Higher singlet ⇒ slower *thermal* dissolution
  (metastability; Agarwal; the coherence window). This is a DIFFERENT channel from the QDS bind-melt and
  must not be conflated. **Footgun:** the QDS *measurement* rate rides on P_S (spin-0 is reactive); the
  *thermal* dissolution rides on (1−P_S). Both are grounded and both are true.
- **Ca-return-on-melt EXISTS:** `model6_core.py:480-485` (6 Ca/dimer, 0.01 µm³ AZ) and `:777-781`.
- **Joint per-component collapse EXISTS:** `perform_quantum_measurement` (`multi_synapse_network.py:931`).
- So F2 is NOT a from-scratch chemistry build — the melt, the Ca shower, and the joint collapse are all
  present. **The single phenomenological thing is the TRIGGER** (dopamine boolean instead of the binding
  event) and the coupling of reward INTO the measurement.

## The derived rate (in `posner_binding.py`, `__main__` PASSES)
    k_encounter = 4π(2D)(2a)N_A = 9.97e9 M⁻¹s⁻¹        [Smoluchowski; D via Stokes-Einstein, a=0.5 nm]
    k_measure(i,j) = k_encounter · productive_fraction · (P_S_i·P_S_j)   [productive_fraction=0.01 REUSED]
    λ_measure = k_measure · [dimer]_local ;  P(melt|dt) = 1 − exp(−λ·dt)
- **QDS selectivity (derived):** coherent/floor rate ratio = (1/0.25)² = **16×**, monotone in P_S.
- **Isotope consilience (derived, ties to F1 + PNAS 2025):** ⁷Li kills ³¹P coherence (F1: 216→13.8 s) ⇒
  P_S floors faster ⇒ singlet-reactive channel open LESS ⇒ cumulative bind-melt weight ⁷Li/⁶Li = **0.19**
  (⁷Li melts LESS ⇒ MORE particles persist) = the measured PNAS direction. **Not fitted.**
- **HONEST FINDING (surface, do not tune):** at 1000 local dimers λ_measure ≈ 1.6e4 s⁻¹ ⇒ P(melt|5 ms)≈1.
  The bind-melt is **effectively instantaneous once a coherent singlet cluster exists** at physiological
  density. The measurement is therefore gated by *the existence of a coherent cluster*, not by a slow
  bimolecular clock. This shapes the wiring (below) and is a result, not a knob to slow down.

## THE DESIGN FORK (surfaced by grounding — Sarah's call; recommendation stated)
The handoff assumed F2 must "build a Posner-Posner binding model from scratch." Grounding the code shows
the melt/Ca/joint-collapse machinery already exists, which opens two wirings:

- **Design A — new bimolecular binding clock.** Add an explicit dimer-dimer binding process (rate
  `λ_measure`) whose stochastic firing triggers the per-component collapse. Faithful to Fisher's literal
  "bind then melt," but adds a second dissolution channel parallel to the existing thermal one, and the
  "instantaneous" finding means the clock adds little dynamical range.
- **Design B — re-route the collapse onto the spin-selective events the model already computes
  (RECOMMENDED).** The symmetry-breaking dissolution the model ALREADY tracks (`get_dissolved_count`,
  the `k_diss ∝ (1−singlet)` line) IS the melt; the Frontiers framing is that a shared-singlet cluster's
  dissolution is *correlated* → fire the joint per-component collapse when a cluster's dissolution/binding
  event occurs, weighting the commit by the spin-0 projection `P_S` (via `posner_binding.k_measure`),
  and DELETE the `dopamine_present` trigger. Minimal, uses only existing machinery, no duplicated channel.

**Recommendation: B** — most emergent (no new phenomenological clock), most surgical, and consistent with
the "measurement = existence of a coherent cluster" finding. A is the fallback if B cannot cleanly express
the spin-0 weighting at the component level. **This fork is the one thing worth a nod before wiring**, since
each Model-6 validation draw is ~47 min of compute and wiring the wrong design is expensive.

## LOCKED acceptance (fixed before wiring)
1. **Module reproduces the literature** (already PASSING, `posner_binding.py` `__main__`): diffusion-limited
   encounter 1e9–2e10 M⁻¹s⁻¹; QDS 16× monotone selectivity; ⁷Li bind-melt < ⁶Li (PNAS direction); the rate
   carries NO reward/dopamine term.
2. **Reward is DECOUPLED from the measurement.** After wiring, the collapse fires with `reward` held False
   (a coherent cluster + calcium is sufficient); reward/dopamine only modulates the SEPARATE learning
   signal (`apply_reward_correlated` / DDSC). A run with `reward=False` throughout must still measure.
3. **Undoped is BIT-IDENTICAL** to the pre-change default on the NON-reward-timing behaviour (regression
   fingerprint: mean_P_S + committed-count at fixed seed/steps), OR any change is explained by the trigger
   move alone (not a silent physics change).
4. **The measurement reads the CLEAN partition** (STEP2's collapse-timing result must still hold: partition
   `[(0,1),(2,3)]` at fire time), i.e. re-routing the trigger does not blob the cluster-quotient.

## Control ladder (pre-registered)
- **C0 undoped, reward-present** (baseline reproduces STEP2's calcium-dominated commitment).
- **C1 undoped, reward-ABSENT** (acceptance-2: measurement still fires — proves decoupling).
- **C2 ⁷Li vs ⁶Li, matched drive** (isotope moves the measurement via the coherence window, DERIVED — the
  F1-follow-on redesign: span the window / delayed readout, NOT `edges=[]`-on-contact).
- **C3 shuffle control:** randomize which dimers carry high P_S — selectivity must vanish (guards against a
  magnitude/abundance leak, per STEP2's leak diagnosis).

## Verdict on the wiring itself
PASS iff: acceptance-1 stays PASS; reward-absent still measures (acceptance-2); undoped fingerprint
bit-identical or trigger-move-explained (acceptance-3); partition stays clean (acceptance-4); the model
imports and runs a short probe without error. Otherwise revert and report. The honest NEGATIVE (e.g.
"decoupling reward changes nothing downstream because commitment is calcium-dominated" — STEP2's washout)
is as publishable as a positive and must be reported, not engineered away.
