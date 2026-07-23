# PRE-REGISTRATION — PO-10 Unit C: the first WEIGHT-LEVEL keystone (matched-marginal pairing)

**Status: DESIGN-REGISTERED, not yet scorable.** Registered 2026-07-20 BEFORE any scored run;
append-only from here. Scorability is gated on the two pilots in §Preconditions passing — until they
do, nothing is scored.

Everything in the program to date (`components`, `cross_w`, `domains`, `fidelity`) is measured at the
**substrate/graph** level. Unit C is the first measurement at the **weight** level: an input goes in, a
synaptic weight-change vector Δw comes out, and we ask whether Δw carries input structure a scalar
mechanism cannot carry. "Computation for a purpose" = input → weight pattern → useful in a way something
simpler can't reproduce.

Builds on: `L·PO9-2` (the partition performs a graded computation over temporal overlap) and
`PO10_LOCALIZATION_ESTIMATE_2026-07-20.md` (the cross-synapse bus is delocalized over 15 µm — the
channel exists, unconditionally). Design authored with the advisor.

---

## The task — matched-marginal temporal structure (the design that makes binding NECESSARY)

Four synapse clusters A, B, C, D in **one dendritic compartment**, all mutually within λ_F (so distance
permits any pairing; **timing** selects it).

- **Trial type 1:** A,B co-active; C,D co-active → domains form as **{AB}, {CD}**.
- **Trial type 2:** A,C co-active; B,D co-active → domains form as **{AC}, {BD}**.

**Marginals uninformative BY CONSTRUCTION:** every cluster is active the same duration at the same
intensity in both types; total drive, per-synapse activity, and the calcium integral are identical. The
ONLY difference is *which pairs coincide*. Any mechanism that accumulates per-synapse statistics returns
chance; the information lives only in the pairing. "Which of these inputs go together" is the fundamental
thing a learning system must extract from a stream where many things are on at once — not a contrived
problem.

Then **dopamine at a fixed delay → collapse → measure the resulting Δw vector.**

## Why the classical baseline fails — stated correctly (the naive version is wrong)

Classical mechanisms *can* detect co-activation (BTSP does, via a shared postsynaptic plateau). So
"classical can't detect coincidence" is FALSE and must not be claimed. The correct argument is about
**what the shared signal can resolve**: classical co-activation detection is mediated by a **scalar**
(dendritic voltage / compartment calcium). With all four clusters active, that scalar is **identical
under both trial types** — it reports *how much* co-activation, not *which subset*. To distinguish
{AB},{CD} from {AC},{BD} classically you would need the groupings to align with separate compartments so
the scalar becomes a vector. Putting all four in one compartment makes the classical channel **provably a
scalar and provably at chance.**

## The measure — decoding accuracy on Δw (NOT cross_w)

Train a **linear classifier to predict trial type from the Δw vector**, cross-validated across
free-running draws; report accuracy with a null from **shuffled labels**. This is a computation-level
measure: it asks whether the information about input structure actually **reached the synaptic state** —
the only place downstream machinery can read it.

- **Prediction if binding works:** near-ceiling decoding from **one presentation per type** (domain
  membership is written in a single collapse event; sign shared across a domain).
- **Prediction for the classical arm:** **chance at any number of presentations** (the discriminating
  information never enters the scalar).
- The asymmetry is **one-shot vs never**, a stronger claim than one-shot-vs-many — and the one this
  architecture actually supports.

### Sign handling (registered — the readout is a signed group update with arbitrary per-domain sign)
Each domain collapses to a shared **random ±1**; Δw is a signed group update whose sign is arbitrary.
Decoding must therefore be **sign-invariant**: the classifier operates on the **pairwise sign-agreement
structure** (which synapses co-commit with the same sign), NOT raw signed Δw. The agreement pattern is
the invariant that differs between {AB},{CD} and {AC},{BD} under arbitrary per-domain sign flips.
Concrete feature set: the pairwise co-assignment / sign-agreement matrix over the 4 clusters (register
the exact feature map before scoring; demonstrate it is invariant under per-domain sign flips on
synthetic data). **NB — decoding sign-invariance means Δw is INFORMATIVE, not yet consistently USEFUL:
a learning rule needs a direction, and nothing in the current architecture resolves the sign. That is
the open architectural question immediately behind Unit C — flagged, not solved here.**

## The construct-validity choice for the readout (registered before build)
The readout must be the **joint-collapse committed structure** → Δw (shared random sign per connected
component; `multi_synapse_network.py` collapse path `:935–992`). It must NOT default to the existing
`apply_reward_correlated()` path (`:1611–1642`), which signs Δw by a shared scalar `reward` and carries
structure through non-negative correlated eligibility *magnitudes* — a different object than the
collapse-coin mechanism this experiment is about. Using the eligibility proxy would make the result about
the proxy, not the physics. The joint-collapse→Δw readout is built **additively / opt-in, off-path
bit-identical** with defaults, and does NOT modify the DO-NOT-MODIFY surface `spine_plasticity_module.py`.

---

## The ablation ladder (four arms; arm 3 is the one people skip)
1. **Full.** Binding on, dopamine on.
2. **Binding off (the load-bearing control).** Cross-bonds / provenance disabled, classical plasticity
   path only (`collapse_independent`, `:998` — bonds do not force joint collapse). Prediction: chance.
3. **Binding on, domains scrambled.** Same number and size of domains, membership randomly reassigned
   before Δw. Separates "grouped updates help" from "grouped updates BY COINCIDENCE help." Prediction:
   chance. Without this a reviewer says any grouping would do.
4. **λ short.** λ_F = 5 µm → cross-cluster binding below the Werner floor. Should collapse to arm 2, and
   ties the computation-level result back to the physics locked in `PO10_LOCALIZATION_ESTIMATE`.

---

## Preconditions (INVALID + unscored if any fails) — the guard that makes or breaks this

The whole result rests on two empirical claims about OUR OWN model, not assumptions. Both are pilots run
and reported BEFORE any scored Δw decoding, per the "demonstrate the failure mode before it passes"
discipline.

1. **PILOT A — the partition actually distinguishes the pairings in the chosen geometry.** Within
   λ_met = 5 µm, ignition is **branch-global** (L·ETA-4: silent-synapse `E_invasion` measured identical
   to driven to four decimals; "when r crosses 1 they cross together"). `L·PO9-2` deliberately used 15 µm
   separation to escape this. One compartment risks branch-global co-ignition **washing out the pairing**
   → chance even WITH binding (a false negative that masquerades as the null). **Show, at the SUBSTRATE
   level (cross_w block structure), that trial type 1 → {AB},{CD} and type 2 → {AC},{BD}** in the chosen
   geometry. If the partition cannot separate the pairings, FIX GEOMETRY before running — do not score.
2. **PILOT B — the classical channel is genuinely blind.** Measure the **calcium spatial profile under
   both trial types** and show it is matched (so no classical scalar/vector can resolve the pairs). If
   compartment calcium is fast and spatially structured enough to resolve the pairings, the classical arm
   PASSES and the result is VOID. Fix geometry first.
   - **The two pilots pull in opposite directions** (spacing that preserves the pairing may let calcium
     resolve it; spacing that blinds calcium may branch-globalize ignition). The geometry must thread:
     all four mutually within λ_F = 214 µm, spaced > λ_met = 5 µm for per-cluster ignition, inside one
     compartment for scalar-blindness. Whether such a spacing exists in our calcium model is exactly what
     these pilots decide.
3. **Offpath digest `515772101786800` reproduces** (all edits off-path bit-identical; re-run immediately
   before scoring).
4. **Ignition confirmed each scored draw** (peak η > 0 at every cluster that should ignite) — else the
   channel is dead and the draw is void.
5. **Sign-invariant decoder validated on synthetic data** (returns chance when only per-domain signs
   differ; returns ceiling when the agreement pattern differs).

## Sweep / draws
- Arms 1–4 as above; trial types 1 and 2; **≥12 free-running draws per (arm × type). NO SEEDING.** Drive
  via `net.step` per-synapse only.
- Readout delay: a FIXED delay where cross structure still exists (≤40 s, before the ~57 s Werner-floor
  crossing), chosen by the `L·PO9-2` rule, recorded before scoring — NOT the delay that maximizes decoding.

## Outcome statements (both are results)
- **POSITIVE:** arm 1 decodes trial type from Δw well above the shuffle null (near-ceiling, ~one-shot),
  while arms 2, 3, 4 sit at chance ⇒ the partition is **readable at the weight level and carries input
  structure a scalar mechanism cannot** — the keystone, finally at the computation level. Round-4's
  density confound and Round-6's write-time-only caveat both dissolve (activity matched by construction;
  measurement at readout).
- **NULL:** arm 1 sits at chance, or arms 2/3/4 also decode ⇒ the weight-level readout does not carry the
  pairing (or carries it for a non-coincidence reason). Reported honestly, with which arm broke it.

## What this does and does NOT establish (state it before a reviewer does)
- **Does:** the partition is readable at the weight level and carries input structure a scalar mechanism
  cannot carry.
- **Does NOT:** establish one-shot LEARNING in the behavioral sense (no behavior, no policy, no task
  performance); does NOT close the attribution gap (canonical §6); does NOT resolve the Δw **sign** into a
  consistent direction (the open architectural question behind this one).

---

## AMENDMENT 1 — 2026-07-21 · counterbalance the pair ordering (batch 1 diagnosed the early-pair-decay confound BEFORE this re-run scored)

**Batch 1 (base design, forward order only, n≈3/cell, delay 20 s) — reported, then amended.** The three
controls behaved: `bindoff` decode 0.29, `scramble` 0.50, `lamshort` 0.14 — all at chance (null p95 ≥ 0.86,
0.50, 0.86). **But `full` decoded at 0.667 = exactly the null p95 — AT CHANCE.** Not a keystone.

**Diagnosis (from the per-draw agreement patterns, not from tuning).** The signal reached the weights but
only through the LATE pair: in every pair1 draw `agree(CD)=+1`; in every pair2 draw `agree(BD)=+1`. The
EARLY pair's agreement was noise (`agree(AB)`/`agree(AC)` scattered ±1) because the early pair's
cross-bond had **decayed below the Werner floor by the 20 s readout** (early pair ~50–65 s old, past the
~57 s Werner crossing, `L·PO9-2` clock), so its two synapses are no longer co-membered and their shared
sign is lost. So only 2 of 6 agreement features carried reliable signal, and the registered decoder over
all six (with the early-pair noise, small n, and forward-order-only positional structure) could not clear
the null. This is the early/late asymmetry + positional fixity (A always early, D always late) flagged in
the original §design notes, now confirmed material.

**What is NOT done:** the registered decoder was NOT swapped for a late-pair-only variant to force a pass
(the `L·PO5-13` goalpost-move the discipline forbids). Batch 1's null stands as recorded.

**The fix — counterbalance the ordering (design change, registered here BEFORE the re-run scores).** Each
PAIRING is now run in BOTH orderings (`--order fwd|rev`): the two co-active pairs swap early/late window.
So across a class, each pair is late-position (hence reliably co-signed) in half its trials, and early
(decayed) in the other half. Consequences:
- Both of a pairing's pairs get a reliable agreement reading (each in half the trials) → all four
  discriminative features are populated, not two.
- "Which pair fired late" is now balanced across classes → the decode must ride the PAIRING (which pairs
  co-sign), not absolute timing/position. This simultaneously fixes the decay asymmetry AND the positional
  confound.
- Readout delay kept at 20 s (the persistence claim is preserved); the fix is symmetry, not a shorter clock.

**Re-run design (tags `ucB_<arm>_<mode><order>`; batch-1 `uc_` files superseded, not deleted).** Label =
PAIRING (pair1={AB,CD} → 0, pair2={AC,BD} → 1), collapsing fwd/rev. Primary contrast **`full` vs
`bindoff`** (arm 1 vs the load-bearing classical-collapse control) at **n=6 per (pairing × order) = 12 per
class**, counterbalanced. `scramble`/`lamshort` at chance are already established (batch 1) and re-confirmed
if the primary contrast passes. Decoder unchanged (sign-invariant pairwise agreement, LOO-CV vs 2000-shuffle
null); relabelled by pairing. Verdict rule unchanged: POSITIVE iff `full` clears null p95 while `bindoff`
does not; NULL otherwise (reported honestly).

---

## RESULT — 2026-07-22 (scored against the registered verdict rule)

**Verdict: POSITIVE (modest but robust).** Per the registered rule (POSITIVE iff `full` clears null p95
while `bindoff` does not), counterbalanced, delay 20:

| arm | n/cell | decode | null p95 | verdict |
|---|---|---|---|---|
| full | 6 | 0.750 | 0.708 | DECODES |
| **full** | **12** | **0.767** | **0.674** | **DECODES (robust)** |
| bindoff | 6 | 0.458 | 0.708 | chance |
| scramble | 6 | 0.000 | 0.583 | chance |
| lamshort | 6 | 0.500 | 0.708 | chance |

- **Pilots A & B passed** before scoring (partition tracks pairing with hard zeros; compartment calcium
  scalar matched 1.2%, only cross-synapse channel is the entanglement partition).
- **Batch 1 (forward-order) returned the registered NULL** (`full` 0.667); diagnosed as early-pair decay
  past the Werner floor; **Amendment 1 (counterbalance)** registered before re-scoring fixed it. The
  decoder was NOT tuned post-hoc.
- **`full` decodes; all three controls at chance** — every alternative explanation (independent collapse,
  any-grouping, sub-Werner binding) closed.
- **Robustness confirmed:** n=12/cell held the decode (0.767) with a tighter null (p95 0.674) → wider
  margin; not a marginal fluke. (`bindoff` n=12 completing for symmetry; chance at n=6.)

**Honest limits (as pre-registered):** modest effect (~0.77, not the predicted near-ceiling); the delay-5
"strengthening" was a registered NULL (0.667 — does not sharpen at shorter readout, consistent with
decay-driven self-cleaning); the Δw **sign is unresolved** (informative, not yet consistently useful);
still the (A) common-cause reading (no non-classicality claimed). Does NOT establish behavioral one-shot
learning and does NOT close the attribution gap.

Data `results/po10_unitC/` (force-added); research log `L·PO10-2`; advisor writeup
`PO10_ADVISOR_UPDATE_UNIT_C_2026-07-22.md`.
