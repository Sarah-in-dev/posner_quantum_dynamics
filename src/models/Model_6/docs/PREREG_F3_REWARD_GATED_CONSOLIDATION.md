# PRE-REGISTRATION — F3: reward-gated consolidation (windowed three-factor rule)

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

## The mechanism (grounded three-factor rule)
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
