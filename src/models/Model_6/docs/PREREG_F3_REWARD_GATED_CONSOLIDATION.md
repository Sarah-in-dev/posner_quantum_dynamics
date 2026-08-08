# PRE-REGISTRATION — F3: reward-gated consolidation (windowed three-factor rule)

> **READOUT CORRECTION 2026-08-08 (grounding `RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08.md`).** Wherever
> this doc says the readout is "dopamine-triggered decoherence / dopamine reads out the tag by collapsing it,"
> that framing is SUPERSEDED and was physically unsupported. Decoherence is PASSIVE (the tag leaks coherence on
> its own clock); no mechanism exists for a neuromodulator collapsing a nuclear spin. The GROUNDED readout is
> the opposite polarity: **dopamine never touches the spin — it GATES/TIMES Fisher's spin-selective Posner
> BINDING-MELT** (the F2 readout, `posner_binding.py`; Ca²⁺/pH-enabled, so dopamine's D1/D2→cAMP/PKA→local
> Ca²⁺/pH gates it). This UNIFIES F2 (binding-melt = readout) + F3 (dopamine = reward timing). The FUNCTIONAL
> credit rule and results (F3-b delayed-credit, F3-c isotope) are UNCHANGED — they rest on the coherence gate +
> reward timing, both preserved. Read every "decoherence readout" below as "dopamine-gated binding-melt readout."

**Written 2026-08-06, BEFORE any model change. SUPERSEDES the mechanism of `PREREG_STEP3_VALENCE_INTO_
CONSOLIDATION.md` (2026-07-30) — that doc's Task-0 grounding discipline and control-ladder spirit STAND;
its cross-synapse-partition mechanism and P31/P32 lever are corrected by F1/F2-c/F2-e + the dopamine
research grounding (`RESEARCH_DOPAMINE_REWARD_SIGNAL_2026-08-06.md`).** Emergent-only (Sarah, ratified):
every constant on the causal path cited/derived before the scored run; nothing dialed to a result.

## What changed since PREREG_STEP3 (why this supersedes its mechanism)
- **ISO-1:** the isotope lever is ⁶Li/⁷Li dopant (F1 `nuclear_relaxation`), NOT P31/P32. All isotope arms use `environment.dopant`.
- **F2-c:** the measurement fires local/early (~88 ms), BEFORE the cross-synapse partition forms. So eligibility is NOT the cross-synapse partition — it is the **per-synapse local trace (P_S / the coherence window)**. PREREG_STEP3's "the partition supplies which synapses share a sign" is retired.
- **F2-e:** reward is inert because it is disconnected (`apply_reward_correlated` = dead stub) AND commitment is calcium-fast, consuming the trace before any reward read. Not a bug — the seam was never built.
- **Dopamine grounding (cited, Yagishita window VERIFIED-at-source):** per-synapse SPECIFICITY = the local eligibility trace, not dopamine (volume-transmitted, near-global). Dopamine supplies **sign + timing + delayed-credit-via-a-train**, gating the trace in a **0.3–2 s window** after the eligibility event (Yagishita 2014 Science).

## REFRAME 2026-08-08 — coherence-window trace + dopamine-decoherence readout (SUPERSEDES the fixed-window mechanism below)
The deep TCA-mechanism grounding (`docs/RESEARCH_TCA_MECHANISM_2026-08-08.md`, Yagishita + Shindou verified
at source) reframes this. The established biology is the dopamine READOUT (D1/cAMP/PKA/PDE, the 0.3–2 s
window); the UNKNOWN — and the field's hardest open problem — is the eligibility TRACE's molecular identity
and lifetime: every measured trace is ~0.3–5 s, far too short for the seconds-to-minutes gap behaviour needs,
and no single synapse is known to hold a mark across that gap. **Model-6's thesis is a candidate answer to
exactly that gap:** the trace = the coherent P_S tag (lifetime ~100 s), read out by dopamine (decoherence)
at ANY delay while still coherent. So the eligibility window is the **COHERENCE LIFETIME**, not a fixed
0.3–2 s — that fixed window is biology's CLASSICAL short trace, retained only as the **baseline arm** the
quantum tag must beat. **My earlier fixed-0.3–2 s implementation (F3-a) was the error the grounding
corrects.** Reframed mechanism (`reward_gating.py`, `__main__` 7/7):
- **quantum_credit** = `eligibility_weight(P_S_at_reward) · DA_sign`, gated on `P_S > Werner floor` (still
  coherent) — no fixed time window; the P_S decay IS the window. Readout = decoherence (consumes the tag).
- **classical_credit** = same, but gated on `t_since ∈ [0.3, 2.0] s` (biology; the baseline).
- Coherence gate = Werner bound 1/√2 [Werner 1989]; sign = burst/dip [Reynolds&Wickens; Yagishita];
  no tuned strength constant. Readable lifetime: undoped **107 s**, ⁷Li **6.8 s**, classical **1.0 s**.
- **Experiment = delayed-credit sweep** (`sweep/f3_delayed_credit_probe.py`): reward at delay ∈ {1,2,5,10,30}
  s × mode ∈ {quantum, classical}. Prediction: quantum commits up to ~the coherence lifetime; classical dies
  past ~2 s; the quantum−classical gap at ≥5 s IS the result. Isotope (⁶Li≫⁷Li) is the physical lever on the
  trace lifetime (Phase C). **Premise flagged: the ~100 s coherence is our hypothesis, not measured biology;
  the readout (dopamine-decoherence) must COEXIST with — not replace — the classical D1/cAMP/PKA cascade. A
  null (quantum also dies ~2 s) is a real result about our substrate, reported not engineered.**
- **What survives from F3-a:** the seam (dopamine gates the P_S tag, sign-correct, writes the durable channel,
  behind the flag) and the discrimination-probe evidence that the machinery works. What changed: the window.

## The mechanism (grounded three-factor rule) — SUPERSEDED by the reframe above (kept for lineage)
    durable_Δ_i  =  eligibility_i(t)  ×  DA_sign·DA_gate(t; window)          per synapse i
- **eligibility_i** = the synapse's local singlet trace P_S(t) (already emergent; ~100 s coherence window).
- **DA_gate(t; window)** = 1 only when a dopamine transient falls in the **0.3–2 s window AFTER** the eligibility (binding) event [Yagishita 2014 — VERIFIED]; else 0. Fed from `dopamine_system.py`'s existing phasic/tonic DA(t).
- **DA_sign** = +1 for a phasic burst (LTP), −1 for a dip below tonic (LTD) [Reynolds & Wickens 2002; Frémaux & Gerstner 2016; Bayer/Lau/Glimcher 2007 asymmetric — note CONTESTED Hart 2014 symmetric-release].
- Writes the **durable** channel (spine/`actin_stable`), NOT the inert `_committed_memory_level`. **No new free strength constant** — magnitude from the grounded Ca-return stoichiometry (per STEP3), not a fitted rate.

## The build (three phases — A/B are the cheap first payoff; C needs Task 0)
- **Phase A — build the seam.** Rebuild `apply_reward_correlated` → a windowed three-factor conversion:
  reads P_S eligibility, gates on DA(t) in the 0.3–2 s window with sign, writes durable consolidation, and
  **WAITS for the windowed DA signal instead of racing ahead on calcium** (the F2-e fix). Wire `dopamine_system`
  back in as DA(t) (its only reader was removed in F2). No new fitted constant; the window (0.3–2 s) and sign are the cited inputs.
- **Phase B — the discrimination probe (THE minimal first experiment; cheap, reward-absent-style short draws).**
  Two synapses, both driven to eligibility (both hold a P_S trace). Deliver ONE phasic DA transient **inside
  the 0.3–2 s window of synapse A's eligibility and OUTSIDE it for synapse B** (B's eligibility event offset so
  the same transient lands >2 s late or before its window). **Test: does A selectively consolidate (durable
  actin_stable) while B does not?** This is the direct test that windowed-DA-gating-on-a-trace produces
  timing-based credit discrimination — the thing F2-e showed we lack.
- **Phase C — the attributed ladder (needs PREREG_STEP3 Task 0: ground/replace `field_threshold_kT=20`,
  `mean_eligibility>0.3`, `learning_rate=0.05`, `molecular_memory>0.5`, `n_dimer_threshold=50`).** Adds the
  isotope arm (⁶Li vs ⁷Li) + membership/timing scramble + the full verdict. Deferred behind A/B.

## Phase-B acceptance (pre-registered, ONE grounded parameter set — no per-condition retuning)
1. **In-window A consolidates > out-of-window B** — durable `actin_stable(A) − actin_stable(B) > 0`, permutation null-p < 0.05.
2. **No-DA control:** neither consolidates (DA is necessary).
3. **Both-in-window control:** both consolidate (the effect is the WINDOW, not synapse identity).
4. **Dip (negative-RPE) control:** a dip in A's window depresses / fails to potentiate A (sign works).
5. **Window-timing scramble:** DA delivered at a random offset → A vs B separation vanishes (null-p > 0.5).
A result that lifts A **only by also lifting the controls** is an artifact ⇒ report the negative.

## The novel extension (the coherence-window payoff — run after Phase B passes/fails honestly)
Biology's eligibility trace is ~0.3–2 s; ours is ~100 s. **Delayed-credit test:** deliver the DA transient at
increasing delays (2 s, 10 s, 30 s) after the eligibility event and measure whether the synapse still credits.
Prediction under the Fisher long-trace bet: our long P_S trace credits at delays where a biological ~2 s trace
cannot. **Honesty flag (carried):** the ~100 s trace is our program's PREMISE, not established biology — the
window/sign/structure are grounded; the long trace is the hypothesis this tests. A null (credit dies by ~2 s
like biology) is a real result about our substrate, reported not engineered.

## Discipline (locked) + verdict
- Constants cited/derived BEFORE the scored run; no constant adjusted after seeing the outcome; one parameter set fixed in advance.
- **POSITIVE:** windowed DA gating on the P_S trace produces timing-discriminated durable consolidation at grounded params (Phase B ladder passes); the delayed-credit extension shows the long-trace advantage.
- **NEGATIVE (fully on the table):** the seam produces no timing discrimination, or calcium still dominates — the model's durable learning is classical/calcium-driven. Publishable per attribution §6.1.
- **Limits:** controlled Model-6-internal probe; a positive strengthens the discrimination case but does not measure nature (§5). The isotope remains the one real attribution lever (Phase C).

## Artifacts (to produce)
- Phase A: the rebuilt windowed-three-factor seam (module + wiring); `dopamine_system` reconnected.
- Phase B: `sweep/f3_discrimination_probe.py` (two-synapse windowed-DA design) + `results/f3_discrimination/`.
- Phase C: STEP3 Task-0 grounding amendment + isotope/scramble ladder.
