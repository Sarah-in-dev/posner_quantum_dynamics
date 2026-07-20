# PRE-REGISTRATION — PO-9 Unit B: the readout-time input-selectivity keystone

**Status: SCORABLE.** Promotes `PREREG_PO8_UNIT_B_READOUT_KEYSTONE.md` (skeleton). The three slots
that skeleton gated on Sarah's rulings are now resolved and filled below. Registered 2026-07-20
BEFORE any scored run; append-only from here. The design invariants, the synapse-level scoring, and
the decomposition-null guard are inherited verbatim from the skeleton — read it too.

## What changed since the skeleton (the three gates, resolved)
1. **Q1 release rate** — answered: the bond release rate is NOT the lever (PO-8). Not used here.
2. **Q2 per-synapse stimuli on `net.step`** — BUILT (`4fce3cd`): `stimulus['per_synapse']` = one
   dict per synapse, inside `net.step` so `_update_backbone_field` still sets `_backbone_eta`.
   Verified: driven synapses reach 345/195/98 µM, undriven ~0.1; default path bit-identical.
3. **Q3 λ** — DECIDED (`L·PO9-1`): the metabolic length (λ_met, `coupling_length_um`) and the
   entanglement fidelity length (λ_F, `fidelity_length_um`) are DECOUPLED. λ_F is UNMEASURED and is
   this experiment's **independent variable**, NOT asserted at 214 µm. So the readout is reported as
   a FUNCTION of λ_F, and the one-vs-two-timescale question is answered empirically, not assumed.

Also new since the skeleton: the **confined-niche dissolution fix** (`L·PO9-1`, `b30c59c`). Without
it the substrate dissolved at ~25 s and the readout object was erased before it could be scored
(PO-8's blocker). With it the substrate persists to 120 s+, so the keystone is runnable. N=12
established the λ=5 µm baseline: multi-synapse partition collapses at a ~40 s cliff (cross-bond
coherence, F=P_S²·W crossing the Werner floor); per-synapse cores hold to 120 s+.

---

## The question (unchanged, on the right object, at the right time)
Does **input identity determine which synapses share a correlated domain AT READOUT**, beyond what
active-region **density** explains, beyond **geometry** (spatial locality), and **as a function of
λ_F**? Object = the synapse-level correlated-domain partition over d(u,v)=Σ(−ln p_e), p_e=(4F−1)/3
(reuse `po7_unit18_correlation_domains`). Time = a readout delay where cross-synapse structure still
exists (chosen by the rule below, not to maximise the score).

## Input conditions (density matched BY CONSTRUCTION; on/off banned)
7 synapses, `pattern="linear"`, ALL driven above dimer-forming threshold in every condition (the
HIGH-vs-LOW-drive fix, never ON-vs-OFF — inactive synapses make no dimers and confound identity with
density; that sank Unit-2). Two neighbour GROUPS, e.g. {0,1,2} and {4,5,6} (synapse 3 the spacer).
The input IDENTITY is carried by the **coincidence timing** via the per-synapse drive:
- **Condition SYNC** — both groups driven simultaneously (all groups coincident).
- **Condition STAGGER** — group A driven in an early window, group B in a later window (within-group
  coincident, across-group not), same total drive per synapse, same density.

**Geometry control (orthogonalise identity from space):** also run **interleaved** grouping
({0,2,4} vs {1,3,5}) so a positive cannot be re-explained as {0,1,2}=the spatial half.

## Scoring (synapse level — the dimer-level trap is documented in the skeleton)
Build the 7-node SYNAPSE correlation graph: inter-synapse weight = aggregate effective correlation
Σ e^{−d} across cross-bridges above the Werner bound. Intra cliques excluded (carry no cross info).
**Scored statistic: Q_act = Newman modularity of that graph against the input-group partition.**

### The falsifier / guard (registered BEFORE scoring — the decomposition null)
Input does **NOT** structure the readout if Q_act for the TRUE input-group labelling lies within the
null band of Q_act against **random equal-size synapse relabellings** (≥200 shuffles; band = 5th–95th
pct). Report the z-score of true-vs-null per (condition, λ_F, delay).

### Verdict function demonstrated FAILING first (mandatory, before any real score)
Run the scorer on a **same-input control**: both "conditions" identical (SYNC vs SYNC), identity
removed. The verdict MUST return NULL (z within band). Only after that passes on the null do I trust
a positive on the real SYNC-vs-STAGGER contrast. A clean decomposition null on the real contrast is a
REAL, reportable negative (`L·PO7-5`: either answer is a result). Input-*located* ≠ input-*computing*.

## Sweep (kept small on purpose — compute is ~4 min/draw)
- **λ_F ∈ {5, 30, 214} µm** — localised / intermediate / ballistic. The key axis.
- **readout delay:** chosen per λ_F by a FIXED rule from a short B0 characterisation — the largest
  delay at which median cross-bond count is still ≥ 25% of its write-time value (i.e. structure still
  exists to partition). NOT the delay that maximises Q_act. Recorded per λ_F before scoring.
- **conditions:** SYNC, STAGGER × {contiguous, interleaved} grouping.
- **≥8 free-running draws per cell. NO SEEDING.** Drive via `net.step` per-synapse only.

## Preconditions (INVALID + unscored if any fails)
1. Offpath digest `515772101786800` reproduces (all PO-9 edits in place, flags at default). ✅ (re-run
   immediately before scoring).
2. Ignition confirmed each scored draw (peak η > 0) — else the cross channel is dead and the draw is void.
3. `fidelity_weights` (λ_F) reaches `tracker.step` — assert non-empty η-graph at write.
4. The same-input verdict-failing-first control returns NULL. If it does not, the scorer is broken and
   nothing is scored.

## Outcome statements (both are results)
- **POSITIVE:** at some λ_F, Q_act(true) exceeds the null band AND tracks the SYNC/STAGGER contrast,
  orthogonal to geometry ⇒ input structures the readout partition; report the λ_F/delay envelope.
- **NULL:** the partition splits per-synapse / by geometry independent of input identity ⇒
  input-located but not input-computing. Report it, with the λ_F range over which it holds.

---

## AMENDMENT 1 — 2026-07-20 · primary statistic changed BEFORE scoring (the failing-first check caught the confound)

**Geometry:** rebuilt as two SEPARATED clusters of 4 (15 µm apart), not a linear array — because (a) a
tight cluster is the minimum that ignites, and (b) adjacent linear groups sit inside each other's
~5 µm metabolic-aggregation range, so driving one ignites both (branch-global, L·ETA-4) and STAGGER
could not create group-local structure. Verified: driving cluster A ignites A, leaves B dark. The
15 µm gap makes λ_F decisive: cross-cluster w=exp(−15/λ_F) is 0.05 at λ_F=5, 0.93 at λ_F=214.

**Why Q_act is retired as the scored statistic:** the failing-first control DID ITS JOB. First draws
(N=8, all 4 cells) gave Q_act z≈3.5–4.9 in EVERY cell — **including SYNC, which must be null.** Two
spatial clusters are trivially modular against a grouping that equals them, at every λ_F and both
inputs. So Q_act measures geometry, not input (the spatial-half trap, L·PO5-13). It cannot be scored.

**Amended primary statistic: `cross_w`** = total weight of the A–B block of the synapse correlation
matrix (Σ p_e over cross-CLUSTER bridges). Physically = do the two clusters bind into one domain.
`cross_frac = cross_w/(within_w+cross_w)`. Prediction, registered before scoring the amended metric:
`cross_w` is **> 0 only for SYNC at λ_F=214** — clusters must be CO-ACTIVE (SYNC not STAGGER, so an
A–B bridge can form at all) AND λ_F long (so the 15 µm bridge clears the Werner floor). STAGGER at any
λ_F → cross_w≈0 (never co-active). SYNC at λ_F=5 → cross_w≈0 (bridge below floor).

**Guard (the decomposition null):** input does NOT structure the readout if `cross_w` for SYNC λ_F=214
is NOT distinguishable from STAGGER λ_F=214 (co-activity fails to bind the clusters even when the
coherence length permits it). ≥8 free draws/cell; report cross_w distributions, SYNC−STAGGER at each
λ_F, and whether the effect is λ_F-gated.

---

## AMENDMENT 2 — 2026-07-20 · the graded-overlap experiment (detector vs computation) — advisor point 4

**Why (accepted from advisor review).** The SYNC-vs-STAGGER contrast has STAGGER's cross_w=0 arising
from *material absence* (clusters never co-exist, so no A–B bond CAN form), not from the partition
computing differently. That is the Round-4 density confound in a temporal costume: activation and
material remain inseparable, now along the time axis. A system that binds when things co-exist and
not when they don't is a **coincidence/presence detector** (an AND gate), not a computation over a
continuous input property.

**The design.** Cluster A driven [0, W]; cluster B driven [offset, offset+W]; W=20 s. Each cluster
gets identical total drive; only the temporal offset varies. Overlap fraction φ = (W−offset)/W:
- offset ∈ {0, 5, 10, 15, 20} s → φ ∈ {1.00, 0.75, 0.50, 0.25, 0.00}.
- λ_F = 214 µm (the delocalized regime — the ONLY one with a cross-channel; per advisor point 1 the
  short-λ regime has no cross-channel to grade). Readout at the delay where structure exists (≤40 s,
  before the Werner-floor crossing at ~57 s). ≥6 free draws/cell, NO SEED.

**Scored statistic:** cross_w (readout, delay=20 s) as a function of overlap fraction φ.

### The step-function null (registered BEFORE scoring — the thing to beat)
Input is a **PRESENCE DETECTOR** (not a computation) if cross_w(φ) is a STEP: ≈0 at φ=0 and
≈saturated for ALL φ>0 (any overlap, however brief, produces full binding because bonds, once formed,
persist). Operationally: cross_w(φ=0.25) is statistically indistinguishable from cross_w(φ=1.0).

### The alternative (a computation over a continuous input quantity)
The partition ENCODES a continuous property of the input if cross_w(φ) is **GRADED** — monotonically
increasing in φ with a characteristic width set by the dimer/bond lifetime (more co-active time →
more cross-bonds accumulate before decay). Operationally: cross_w(φ=0.25) < cross_w(φ=0.5) <
cross_w(φ=1.0), a resolved monotone trend beyond draw-to-draw scatter.

**Verdict rule:** fit cross_w vs φ; report (a) the φ=0 vs φ>0 step, and (b) the slope/monotonicity
across φ>0 with per-φ scatter. Graded ⇒ the first result in this sub-programme that encodes a
continuous input quantity in the readout partition (the §8 bar). Step ⇒ a presence detector, reported
honestly as such. Either is a result; the step-function null is the pre-registered default.

**Framing corrections also adopted (advisor points 1–3), to be applied to the packet + L·PO9-2:**
- λ is not a two-value sweep; W=exp(−d/λ) is lossy-propagation form, wrong for a MEDIATED channel.
  The honest object is flat-then-cutoff; the question is binary (mode delocalized vs Anderson-
  localized), settled by a localization-length estimate, not a sweep. λ_F=214 cell = the delocalized
  case; λ_F=5 cell = the localized limit where the cross-channel does NOT EXIST (analytic, not a datum).
- The 2×2 is ONE measurement (SYNC, delocalized: clusters merge + persist to the Werner-floor crossing)
  + THREE analytic predictions confirmed. Do not inflate to a 4-cell finding.
- Clock reconciled: binding dies at the Werner-floor crossing F_cross=P_S²·W<½ at delay≈57 s, with
  T_eff≈158 s (NOT T_singlet 216 s — analytical_gap applies spread_factor·template_factor) and the
  decay origin at write-end (P_S=0.943 at delay 0). Measured crash 40→60 s matches.
