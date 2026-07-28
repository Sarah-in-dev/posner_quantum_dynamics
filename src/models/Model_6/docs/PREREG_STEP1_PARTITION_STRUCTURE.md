# PRE-REGISTRATION — Step-1 partition-structure probe (structured vs blob)

**Written 2026-07-28, BEFORE the scored run.** Thread #2 gate: does the spatial-discovery
loop's cross-synapse entanglement partition carry INPUT structure, or is it an
input-independent blob? This decides whether Step 2 (the P31/P32 selective-consolidation
isotope discrimination) is worth the loop compute.

Companion skills grounded first: `session-discipline`, `agent-grounding-protocol`,
`entanglement-topology-measurement` (A7b: the Werner cut is a coherence-set distance graph
`bond iff d < d* = 5·ln(P_product/0.5)`, `d*max = 3.47µm` at P_S=1),
`quantum-computation-and-attribution` (§6.3 discrimination; §7 #1 gate- vs pair-selectivity),
`coherence-gated-learning` (primitive 1: the topology IS the eligibility trace),
`quantum-system-canonical` (§4.3 eta is a GATE not a selectivity channel — FALSIFIED 2026-07-18).

## The question, made measurable

The spatial-discovery Problem-1 fear (`spatial-discovery-experiment` skill): *"The entanglement
tracker produces a complete graph regardless of input pattern. No structure = no computation."*
Step 1 tests exactly this: **does the cross-synapse partition depend on the input, or not?**

Prior sessions saw a "blob" in the loop. Grounding (SUPER-1, GAP-2, §4.3 falsification) shows
why the loop's naive drive gets there: (a) the nav drive never pushes `E_invasion` over
`invasion_threshold`, so eta=0 and NO cross-bonds form (attempt-1's inconclusive result);
(b) when engaged, close-spaced co-active synapses saturate. So the probe must GUARANTEE
engagement and then vary the input while holding geometry fixed.

## Design (controlled, tractable — NOT the live nav loop)

- **Geometry HELD FIXED:** n=8 synapses, 1D linear, spacing 1.0µm (positions 0..7µm), the
  loop's clustered spacing. Two INTERLEAVED classes at matched distances: `C={0,2,4,6}`,
  `G={1,3,5,7}`. Nearest neighbour = 1µm (always C-G); next = 2µm (always within-class).
  Both < d*max 3.47µm, so a geometry-only (all-co-active) graph connects C and G ALIKE — any
  separation of C from G is therefore INPUT, not geometry.
- **Engagement guaranteed (controlled IC, `set_arm`-style — the established pattern for this
  regime, GAP-1/GAP-2):** every synapse has `actin_enlargement`/`E_invasion` clamped high, so
  `_update_backbone_field` computes eta>0 from real metabolic power during drive. This scopes
  the probe to *"given engagement, is the partition structured?"* — NOT *"does the nav drive
  reach engagement"* (GAP-2 already answered that: no).
- **Only the INPUT TIMING varies** (two conditions):
  - **SYNC**: C and G driven by the SAME theta-burst (in phase). Positive control that the
    geometry CAN blob.
  - **STAGGER**: C on theta, G offset by THETA/2 — C and G never co-burst. Cross-bond formation
    needs both endpoints' eta>0 at the same tracker step, and eta only rises DURING a burst.
- **Dimer-count sweep** (drive duration): the structure-vs-dimer-count axis the gate needs.
- **Seeds:** 5 per (condition, duration), via `Model6Parameters` presynaptic seed + build seed.
  Report mean ± spread; the SYNC/STAGGER separation must exceed seed noise.

## Metrics (per condition × duration × seed), read on the synapse-QUOTIENT graph

`compute_synapse_quotient_betti` (the honest cross-synapse lens). Quotient edge (i,j) exists iff
any cross-dimer bond between synapses i,j clears the Werner bound.

- **CPEF** = cross-phase edge fraction = (C-G quotient edges) / (all quotient edges).
  Geometry/blob ⇒ nearest-neighbour C-G dominates ⇒ CPEF high. Timing-structured (off-phase
  suppressed) ⇒ C-G edges absent ⇒ CPEF low.
- **largest_frac** = largest quotient component / N. Blob ⇒ 1.0; fragmented ⇒ < 1.
- **betti0**, **n_edges** (secondary).
- **Δ(dimers)** = CPEF_SYNC − CPEF_STAGGER at matched dimer bins — the structure-vs-dimer curve.

## PRE-REGISTERED VERDICT (thresholds fixed before the scored run)

Let low-dimer = total dimers ≲ 1500; high-dimer = total dimers ≳ 3000.

1. **STRUCTURED / FIXABLE** (⇒ Step 2 IS worth the compute) iff:
   - `Δ ≥ 0.40` at low-dimer (SYNC blobs, STAGGER suppresses off-phase bonds — input matters), AND
   - `Δ ≤ 0.15` at high-dimer (the suppression is lost — the structure SWAMPS as dimers pile up).
   Report the **swamp threshold** = dimer count where Δ first drops below 0.20. Interpretation:
   the partition carries input structure but only below the swamp threshold; the loop must keep
   dimers under it (read-early / decay-clean) to keep the partition informative.
2. **IRREDUCIBLE / BLOB** (⇒ Step 2 NOT worth it; the (A) classical reading stands; pivot) iff:
   - `Δ ≤ 0.15` at ALL dimer counts — co-activation timing never separates the partition; it is
     the geometric blob regardless of input.
3. **INPUT-INSENSITIVE positive-control failure**: if SYNC itself never blobs
   (`CPEF_SYNC < 0.5` at high-dimer), the rig is broken — do not score; investigate.

Positive control that must hold for any verdict: **CPEF_SYNC ≥ 0.6 at high-dimer** (the geometry
blobs when input is synchronous). Engagement control: **mean peak eta > 0** on driven synapses.

## What this DOES and does NOT establish (honesty, per attribution §5/§6.3)

- Establishes: whether the partition topology is INPUT-DEPENDENT (Step-1's whole question).
- Does NOT establish quantumness. A classical coincidence-gated trace would ALSO bond co-active
  and suppress off-phase, and would also saturate. Step-1 = "is there structure to discriminate?"
  Step-2 (P31/P32 isotope) = "is the structure quantum?" A STRUCTURED verdict only makes Step 2
  WORTH RUNNING; it does not pre-empt it.
- Scope limit (stated, like GAP-2): controlled engagement IC + controlled interleaved geometry;
  the live nav-drive path is not exercised and (GAP-2) does not reach engagement on its own.

## Falsified if

- STAGGER shows the SAME CPEF as SYNC at low dimers (timing doesn't gate cross-bonds) — then the
  "structure" claim dies and the verdict is BLOB.
- The separation does not exceed seed spread.
- SYNC fails to blob at high dimers (rig broken).

## Artifacts

- Scored probe: `sweep/step1_partition_structure_probe.py` (this repo).
- Result JSON: `results/step1_partition/` (force-added for provenance; results/ gitignored).
- Exploration that motivated the thresholds (transient, scratchpad): `probe_v2.py`, `diag_gg.py`
  — SYNC blobs at all dimer counts; STAGGER CPEF=0 up to ~2013 dimers, →blob by ~3080; the eight
  synapses symmetric in drive/eta/P_S at 0.5s (G not under-driven).
