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
