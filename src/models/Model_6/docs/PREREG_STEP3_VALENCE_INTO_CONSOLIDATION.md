# PRE-REGISTRATION — Step-3: emergent valence into the consolidation arm

**Written 2026-07-30, BEFORE any model change.** Thread #2, Step 3 — the payoff, and the point where the
program's "quantum-driven durable learning" claim either lands or honestly resolves to classical. Sarah's
ruling for this step is a hard constraint: **EMERGENT ONLY — no tuned constants.** Every parameter on the
causal path must be literature/physics-grounded and cited BEFORE the run; nothing may be dialed to make
P31 beat P32.

## What Step 2 established (the precise starting point)
The quantum partition is measured **correctly** (collapse fires on the clean {AB}|{CD} structure — STEP2
research-log row + `step2_collapse_timing_probe.py`), but its per-component credit is **lost downstream**:
commitment is calcium-dominated (~7/8 fire regardless of the collapse coin), so durable `actin_stable`
tracks abundance, not the partition. Step 3 must make the partition's credit **gate which spines durably
consolidate** — emergently.

## The reordering that emergent-only forces (the load-bearing finding)
You cannot wire an emergent valence coupling on top of a commitment path that already carries **fitted
parameters** — any resulting P31≫P32 would be unattributable (could be the knobs). The code itself admits
these on the exact causal path:
- `field_threshold_kT = 20.0` — "~20 kT thermal noise", **no citation** (`multi_synapse_network.py:1651`, per its own docstring).
- `mean_eligibility > 0.3` — "a chosen cut with no derivation" (`:1653`).
- `apply_reward_correlated(..., learning_rate=0.05)` — a fitted rate, and it writes `_committed_memory_level`
  (not a loop-closing durable weight) (`:1611-1642`).
- CaMKII commitment cut `molecular_memory > 0.5` (`model6-commitment-pathway`) — verify grounded vs chosen.
- `n_dimer_threshold = 50` cited as "Fisher's prediction" then re-derived from the above ⇒ **circular** (`:1654-1656`).

**Therefore Step 3, Task 0 (before any coupling): GROUND OR REPLACE these.** Each constant on the
measurement→commitment→consolidation path must get a citation, a derivation from cited physics, or be
removed. Where no grounding exists, that is itself a **result** (the mechanism was fitted), reported, not
patched over. Only once the path is emergent can a valence coupling's effect be attributed to the partition.

## The mechanism (candidate, grounded form — to be built ONLY after Task 0)
A **three-factor rule**: eligibility (the entanglement partition, already emergent) × dopamine (reward) →
**sign/direction** of consolidation, per connected component. Grounded lineage: dopamine-gated plasticity
sign (Reynolds & Wickens 2002; Frémaux & Gerstner 2016 three-factor rules) — dopamine converts the
eligibility trace into LTP vs LTD. The partition supplies *which synapses share a sign* (the per-component
collapse coin, already computed by `perform_quantum_measurement`); dopamine supplies *the sign*; the
consolidation arm (`actin_stable` via confinement) reads it. **No new free strength constant** — the
coupling magnitude must derive from the existing (grounded) calcium-return stoichiometry, not a fitted rate.
The orphaned `apply_reward_correlated` is the intended seam but must be rebuilt to (a) write durable
consolidation, not `_committed_memory_level`, and (b) carry no fitted rate.

## The control ladder (pre-registered — the guard against tuning-to-result)
The mechanism is accepted ONLY if, at ONE grounded parameter set (no per-condition retuning), ALL hold:
1. **P31 RECOVERS** the co-active pairing on the leak-immune within-condition partial correlation (null-p<0.05).
2. **P32 at chance** (isotope kills it — the physical control).
3. **Partition-membership SCRAMBLE at chance** (same sizes, permuted membership — kills the "it's just
   abundance/grouping" explanation; the `scramble` arm from Unit C).
4. **λ-short / bindoff at chance** (no domains → no credit).
A mechanism that lifts P31 **only by also lifting the controls** is tuning ⇒ report the negative.

## PRE-REGISTERED VERDICT
- **POSITIVE (quantum-driven durable learning):** the full ladder passes at grounded, cited parameters.
- **NEGATIVE (classical at the learning level — stays fully on the table):** at grounded parameters the
  partition's credit does not survive into durable consolidation (P31 ≈ controls, or only the abundance leak
  moves). This is a real result — the model's durable learning is calcium-dominated — NOT a bug to engineer
  away. Per attribution §6.1, a negative here is as publishable as a positive.
- **BLOCKED:** if Task 0 finds the commitment path cannot be grounded without inventing constants, Step 3
  cannot be run honestly until that is resolved — report that state rather than proceeding on fitted knobs.

## Discipline (locked)
- All constants cited/derived BEFORE the scored run; the citation list is part of this prereg's amendment.
- No constant is adjusted after seeing P31/P32. One parameter set, fixed in advance.
- Emergent-physics principle (`quantum-computation-and-attribution` §6.1) governs; "what value makes it come
  out right" is the named failure mode and is forbidden.

## Limits
Controlled probe, Model-6-internal (simulated) — a positive strengthens the discrimination case but does not
measure nature (attribution §5). The isotope (P31/P32) remains the one real attribution lever.

## Artifacts (to be produced)
- Task 0 grounding audit → amendment to this doc (constant-by-constant citation/derivation/removal).
- `sweep/step3_*` harness (reuses the Step-2 rig + scorer; adds the dopamine-sign three-factor seam).
- Results `results/step3_*/`; scorer `sweep/po11_valence_score.py` (+ scramble/bindoff arms).
