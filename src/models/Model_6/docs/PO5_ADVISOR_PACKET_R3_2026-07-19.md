# Advisor packet — round 3: the provenance build, and a scale wall
**PO-5 · 2026-07-19 · after building Fisher's actual mechanism**

Two rounds ago you diagnosed the disease (the model has no representation of the entangling origin)
and named the fix (provenance-based bonding). We built it. This packet reports what it did, discloses
two false positives PO-5 caught, and puts ONE architectural question to you.

---

## 1. SCORECARD ON ROUND-2 DIRECTIVES

| directive | outcome |
|---|---|
| **"Provenance is the fix; build dimer↔event↔dimer"** | **BUILT and it works.** Fisher's actual mechanism: events at Ca-elevated cells (2 phosphate slots each), a newborn dimer claims its ≤2 nearest events, two dimers bond iff they share one. Opt-in, provenance-off **bit-identical** (`1034/369740/0.991922159684`). **First non-blob graph in the whole investigation:** 459,889 bonds→~500, largest_frac 1.000→0.05. |
| **"Lean on the classical common-cause reading"** | **Adopted** — Player & Hore kills the quantum payload; shared origin is the justification. |
| **"Don't revert to Ca9(PO4)6"** — *(this one is PO-5's correction of the deep research, not yours)* | The research agent flagged the model's Ca6(PO4)4 as "wrong, fix to Ca9(PO4)6." **That contradicts LOCKED `quantum-system-canonical:43` (Agarwal: the dimer is the qubit, the trimer is inert).** Built on the dimer (K=2 events), correctly. Flagged as a PO-5 process failure — the physics was already documented and the workflow mis-corrected it. |

---

## 2. THE MEASUREMENTS

### 2.1 The mechanism produces a sparse, pairwise, input-LOCATED graph (works)

| | bonds | components | largest_frac |
|---|---|---|---|
| provenance OFF (clique + EM) | 459,889 | 1 | 1.000 |
| **provenance ON** | ~500 | ~700 | **0.05** |

Events are placed where calcium is elevated (`atp_system.py:90`'s own rule), so event locations are
input-correlated in space. Each event yields ≤1 edge (2 daughters). This is the sparse RIG Unit 15
showed CAN carry input-dependent partition beyond density.

### 2.2 ⚠ The computation test is a FALSE POSITIVE — the probe said SUCCESS, PO-5 overrode it

The probe's verdict function printed *"CARRIES INPUT-DEPENDENT PARTITION."* **Wrong.** It scored
Newman modularity `Q(input)=0.15 >> Q(shuffled)=0` — but the "input label" was **which spatial half a
dimer sits in.** So Q detects **spatial locality** (dimers bond to neighbours via nearest-event
claiming) — GEOMETRY, which §8 explicitly rules insufficient. The decisive quantity is the actual
input contrast:

**COND-A vs COND-B (sustained vs pulsed drive), 5 seeds: component effect size d = 0.02 — flat null.
Changing the input does not change the partition.**

(This is the third criterion-mis-registration PO-5 has caught in its own probes this session —
Units 8, 9, 13. Disclosed, not buried.)

### 2.3 Why the null — and the wall it exposes

Provenance's assortativity is **spatial** (nearest-event claiming). At **single-synapse scale** the
pulsed-vs-sustained contrast does not produce meaningfully different *spatial* calcium patterns — the
same peak-saturated, weak-contrast wall that nulled Units 9/10/13/14. So the mechanism carries
structure; the *input* just has no spatial purchase on it at this scale.

Unit 15 (abstract) proved the RIG carries input structure **IF event-sharing is assortative-by-INPUT
(Q=0.65 beyond density)**. The build produced assortativity-by-**SPACE**. The gap between those two is
the whole question.

---

## 3. THE ONE ARCHITECTURAL QUESTION FOR YOU

**Does the pair-level input channel require the MULTI-SYNAPSE scale — contradicting the program's
standing claim that §8's keystone is "single-synapse-scale, needs no backbone"
(`quantum-computation-and-attribution` §7 #1)?**

The measured tension:
- §5 [LOCKED]: one synapse = one nanodomain = one component; the "meaningful, input-dependent
  partition is over *which synapses* condense." I.e. §5 already puts input-dependence at the
  NETWORK scale.
- §7 #1: the keystone is "single-synapse-scale."
- **PO-5's measurement:** provenance carries input structure only if input varies the spatial pattern,
  and a single nanodomain gives input no spatial degrees of freedom to vary. **So the build works, and
  the scale it needs is the one §7 #1 says it doesn't.**

**Three readings, and it's your call which:**
- **(a)** §7 #1 "single-synapse-scale" is wrong/stale, and the keystone genuinely lives at the
  multi-synapse scale §5 already names. Then PO-5's whole single-synapse scope was mis-set — and the
  next test is provenance across N synapses with input selecting WHICH synapses' events overlap.
- **(b)** The single-synapse channel is real but PO-5's INPUT is too weak — a spatial input contrast
  (not temporal pulsed-vs-sustained) at one synapse would show it. Then the fix is the input design,
  not the scale.
- **(c)** The single-synapse pair-level channel does not exist, and "scalar as computation" is the
  right answer at one synapse — the computation is inherently a network-scale partition.

## 4. WHAT PO-5 IS NOT CLAIMING

- Not that provenance failed — the mechanism is faithful and produces the first non-blob graph.
- Not that the keystone is disproven — the single-synapse null may be an input-design or scale
  artifact (readings b/a), not a physics verdict.
- Not that the rate constants (event_rate, age window) are physical — swept for regime, not certified.

## 5. THE DELIVERABLE STATE

Provenance bonding is committed, gated, bit-identical-off, and produces a sparse graph. Everything
else in the investigation (§8 not supported at single-synapse scale; the mechanism inventory;
the LOCC finding; the RIG de-risk) is in `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entries
`L·PO5-1` … `L·PO5-13`. The single decision that gates the next unit is §3 above.
