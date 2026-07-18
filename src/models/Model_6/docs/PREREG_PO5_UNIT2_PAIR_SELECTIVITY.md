# PRE-REGISTRATION — PO-5 UNIT 2 · pathway provenance, and does the bond set depend on INPUT at pair resolution?

**Registered:** 2026-07-18, BEFORE any scored run. **PO:** po5-selectivity.
**Probe:** `src/models/Model_6/sweep/po5_unit2_pair_selectivity.py`.
**Charter:** `quantum-system-canonical` §8 Keystone #1. **Rulings applied:** `mo-ruling-001` §3
(Pathway 1 in scope), `mo-ruling-010` (the saturation finding stays in PO-5's acceptance; do not
engineer around it).

**Read this first:** Unit 1 (`L·PO5-1`) established `g` is LIVE (`D = 33.5`) and that the intra
graph is 0.61–0.94 saturated in one component. Per `L·PO5-1` CORRECTION 1, one component at a single
synapse is **correct physics** (`quantum-system-canonical:139` LOCKED) and is **not** read as a
finding here.

---

## 1. The two questions, kept separate

**Q-A (provenance).** Which mechanism produces the realised bond set — birth-inheritance, Pathway 1,
or Pathway 2? Unit 1 left this UNVERIFIED and refused to guess.

**Q-B (the keystone).** Does **which** dimers bond depend on **input**, at pair resolution, beyond
what active-region density alone explains?

Q-A is descriptive and cheap. Q-B is the keystone. They are scored separately and a result on one is
never reported as a result on the other.

## 2. Q-A — provenance, recovered WITHOUT editing the physics

Three bond-creation sites exist:

| site | `file:line` | when |
|---|---|---|
| birth inheritance | `dimer_particles.py:218-228` | inside `step_population` |
| Pathway 1 (burst) | `dimer_particles.py:443-444` | inside `step_entanglement` |
| Pathway 2 (EM) | `dimer_particles.py:457-458` | inside `step_entanglement` |

**The classification is exact, not inferred.** `dimer_particles.py:439` sets
`p1 = both_ent & same_burst & both_tmpl & ~has_bond`, and `:450` sets `p2 = both_ent & ~p1`. So
within `step_entanglement`, a **newly** formed bond took Pathway 1 **iff** `same_burst & both_tmpl`,
and Pathway 2 otherwise. No RNG replay is needed and no branch is guessed.

**Instrumentation method — zero source edits.** The probe wraps three methods **on the instance**
(`step_population`, `step_entanglement`, `_create_bond`), each wrapper calling through to the
original. `step_population`/`step_entanglement` set a phase flag; `_create_bond` records the origin
against the same `(min_id, max_id)` key the model uses. **No physics is altered, no RNG draw is
consumed or reordered, and `dimer_particles.py` is not modified** — deliberate, because four POs
share this tree.

**Registered validation of the instrument, run before any provenance number is reported:**
1. **Conservation** — every key in the provenance map exists in `_bond_lookup`, and every key in
   `_bond_lookup` has a provenance entry. Any mismatch ⇒ report `INSTRUMENT_INVALID` and stop.
2. **Non-perturbation** — an instrumented run and an uninstrumented run at the same seed must agree
   **bit-for-bit** on `n_dimers`, `n_bonds` and `mean P_S` at every sample. Any disagreement ⇒
   `INSTRUMENT_INVALID` and stop. (Method borrowed from PO-1/PO-6a's `po1-6a-002` bit-identity check.)

## 3. Q-B — the discriminating statistic, and why it is pair-level and not gate-level

**The problem:** dimer identities are not comparable across runs — different births, different IDs.
So "which dimers bond" cannot be compared run-to-run at the level of dimer identity.

**The statistic — a density-normalised, spatially-binned bond probability.** Bin the synapse volume
into fixed cells. For cells `a`, `b`, define

```
P_bond(a,b) = (bonds with one endpoint in a and the other in b) / (n_a * n_b)      a != b
P_bond(a,a) = (bonds with both endpoints in a)                 / C(n_a, 2)
```

**Dividing by the available pair count is the load-bearing step**, and it is chosen to answer §8's
sentence directly: *"the partition carries no more than active-region density."* A raw bond count
between two busy regions rises with density alone — that is **gate-level**. `P_bond(a,b)` is a
per-available-pair bonding probability, so **density is divided out by construction** and what
remains is pair-level: how likely *these two locations* are to bond, given how many candidates each
holds.

**The distance control, because `g` is geometry not input.** `P_bond` will depend on `|a-b|` through
`g` (Unit 1: `D = 33.5`). That dependence is **not** input-selectivity. So the scored quantity is the
**residual after regressing out distance**:

```
R(a,b) = P_bond(a,b) - f_hat(|a - b|)
```

where `f_hat` is an isotonic/binned fit of `P_bond` against pair separation, **fitted per run on that
run's own data** (so no cross-condition information leaks into the control).

## 4. Conditions, and the null — which is seed-only, NOT an activation floor

Three arms, all fully live. **No arm attempts to be "silent"** — this PO is forbidden an
activation-floor null (`BASELINE_RATE_HZ = 0.5`, `sweep/presynaptic_release.py:65`), and the standing
scar is that three probes on this board used a control assumed silent that was not.

| arm | what differs | purpose |
|---|---|---|
| **INPUT-A** | one spatial input pattern, seeds S1..S3 | condition 1 |
| **INPUT-B** | a different spatial input pattern, matched gross drive, seeds S1..S3 | condition 2 |
| **NULL (seeds-only)** | INPUT-A, seeds S4..S6 | how much `R` moves with RNG alone |

**The null is the seed-only arm.** It cannot show an input effect because no input differs across it
— the only varying quantity is the RNG stream. This is the shape the MO accepted for PO-4
(*"NULL seeds-only dV = +0.0001 → 8339× smaller than the effect"*).

**Registered matching requirement:** INPUT-A and INPUT-B must be matched on **total** drive, so that
any difference is in *pattern*, not amount. The probe reports total integrated drive per arm and
**flags the comparison INVALID if they differ by more than 5%** — otherwise a gross-drive difference
would masquerade as pair-level selectivity, which is the gate-level confound §8 warns about.

## 5. The verdict function — registered, with FALSIFIED reachable

Let `d_input` = mean pairwise distance between the `R` matrices of INPUT-A and INPUT-B runs, and
`d_null` = mean pairwise distance between `R` matrices within the seed-only arm. Distance is
Frobenius norm over cells with sufficient occupancy (`n_a >= MIN_OCC`, registered = 5).

```
ratio = d_input / d_null
```

| verdict | condition |
|---|---|
| **CONFIRMED** | `ratio >= 3.0` AND drive-matching valid AND instrument valid |
| **FALSIFIED** | `ratio <= 1.5` — the bond set moves no more with input than with RNG ⇒ **pair-flat with respect to input** |
| **INCONCLUSIVE** | `1.5 < ratio < 3.0`, or too few occupied cells (`< 10`), or drive mismatch |
| **INSTRUMENT_INVALID** | either §2 validation fails |

**Thresholds are fixed here and do not move after the run.**

**FALSIFIED is a real, expected-possible outcome and is reported as a finding, not a protocol
problem** — per the kickoff and `board.md:933-936`: if bonding is pair-flat with respect to input,
*"graph as computation"* collapses to *"scalar as computation."*

## 6. Demonstrating the verdict can fail before it is allowed to pass

Before any model is constructed, the probe scores **synthetic** `R` matrices with known answers and
must produce all four labels:

| synthetic case | required verdict |
|---|---|
| A and B drawn from the same distribution as the null | `FALSIFIED` |
| A and B separated by a large fixed offset, null tight | `CONFIRMED` |
| A and B separated by ~2× the null spread | `INCONCLUSIVE` |
| provenance map deliberately desynchronised from `_bond_lookup` | `INSTRUMENT_INVALID` |

**If any required label is not produced, the probe aborts and reports nothing else.**

## 7. Compute

Q-A validation and the smoke test are cheap (single short run). **The full Q-B matrix is 9 runs and
is NOT started without a slot** — requested in `queue/po5-selectivity.md`. Runs are backgrounded with
per-sample progress, results persisted incrementally, never piped through `tail`.

## 8. Stated limits, registered in advance

- Single synapse. **This is deliberate** — §7 #1 calls the keystone *"single-synapse-scale — needs no
  backbone"* — but see the tension raised in `requests/model6-mo/po5-selectivity-002.md` §2: §5 LOCKS
  the meaningful input-dependent partition as **cross-synapse**. **If the MO rules that reading (b),
  this unit is measuring the wrong layer and should be stopped.** Registered as a known risk, in
  advance, rather than discovered afterwards.
- A `CONFIRMED` here establishes input-dependence of the **bond set**, not of the **partition** —
  the intra partition is one component by correct physics, so this unit cannot and does not speak to
  partition-level selectivity.
- Cell binning is a coarse-graining: it can only detect pair structure at or above the cell scale.
  A `FALSIFIED` is therefore *"pair-flat at or above the cell scale under these conditions"*, and the
  cell size is reported with the verdict.
