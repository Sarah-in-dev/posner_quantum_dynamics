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

---

## AMENDMENT A2.2 — registered 2026-07-18, BEFORE the Q-B run, per MO ruling 019

Ruling 019 adopts PO-5's recommendation: **whole-set primary, verdict additionally split by
provenance** — and requires **both pre-registered so the split cannot rescue a whole-set null after
the fact.** Registering the split and its precedence now.

**PRIMARY (decides the verdict):** `ratio = d_input / d_null` computed over the **whole realised
bond set**, scored exactly as §5 already registers. **This is the verdict.**

**SECONDARY (reported always, decides nothing):** the same `ratio` computed three more times, over
the bond sub-sets partitioned by provenance — **P0 birth-inheritance**, **P1 burst**, **P2 EM** —
using the Unit 2 Q-A instrument, which has passed both its registered gates.

**The precedence rule, registered so it cannot be reinterpreted after seeing the numbers:**

> **A `FALSIFIED` or `INCONCLUSIVE` on the whole set is the verdict, regardless of what any
> provenance sub-set shows.** A `CONFIRMED` in the P2 (17%) sub-set alongside a whole-set
> `FALSIFIED` is reported as **"pair-flat overall, with a signal confined to the 17% minority
> mechanism"** — it is **NOT** reported as the keystone confirmed, and **NOT** promoted to the
> headline. The split exists to stop a whole-set null *hiding* structure, never to rescue one.

**Sub-set sample-size guard:** a provenance sub-set with fewer than **1000 bonds** at the scored
sample, or fewer than **10** occupied cells, is reported as `INSUFFICIENT` rather than given a
verdict. From Q-A, **P1 is expected to trip this** (22 bonds) — registered in advance so its absence
is not read as a result.

## AMENDMENT A2.3 — the `_remove_dimer` guard, registered 2026-07-18

Ruling 019: *"if Q-B's protocol exercises dimer removal, stop and say so first."*

**Q-B DOES exercise dimer removal** — heavily and unavoidably (see
`requests/model6-mo/po5-selectivity-005.md`). **But it does not exercise the defective function.**
The death path at `dimer_particles.py:239` calls `_remove_all_bonds_for_dimer` (`:245`), which
correctly pops `_bond_lookup`. The defective `_remove_dimer` (`:252-261`) has **no call sites**.

**Rather than rely on that reading, the probe enforces it.** `_remove_dimer` is wrapped on the
instance with a counter. **If it is ever called, the run aborts immediately and reports
`INSTRUMENT_INVALID — _remove_dimer called`**, and no verdict is produced. The wrapper calls through
to the original and alters nothing; it is a tripwire, not a patch. **PO-5 does not fix the defect** —
it is routed to PO-7.

## AMENDMENT A2.4 — cell size corrected 40 nm → 8 nm, registered 2026-07-18 BEFORE any scoring

**The first Q-B launch was KILLED by PO-5 after 3 of 9 runs (~11 min), unscored.** Reason: with
`CELL_NM = 40.0` the probe reported **`cells = 4`** occupied cells against a registered
`MIN_CELLS = 10`, so `classify()` would have returned **`INCONCLUSIVE` regardless of the physics.**
That is a verdict that cannot distinguish its outcomes — the exact failure `MO_MODEL6.md` §2.3 and
the `683b82f` scar exist to prevent — and it would have consumed ~50 min of the exclusive slot to
produce a guaranteed non-result.

**The new value is derived from Unit 1's independently-measured geometry, NOT from any Q-B outcome.**
`L·PO5-1`, four samples: `r_p10 = 3.71`, `r_p50 = 9.78`, `r_p90 = 16.18`, **`r_max = 36.45 nm`** —
the entire intra-synapse dimer cloud spans ~36 nm, so a 40 nm cell is larger than the whole object
being binned. `CELL_NM = 8.0` is chosen because it sits **above `r_p10`** (so a cell is not dominated
by a single close pair) and **below `r_p50`** (so within-cell and between-cell are distinguishable),
giving ~4–5 cells per axis across the cloud and an expected ~20 occupied cells at `MIN_OCC = 5`.

**No verdict threshold moves.** `RATIO_CONFIRM = 3.0`, `RATIO_FALSIFY = 1.5`, `MIN_OCC = 5`,
`MIN_CELLS = 10` and the A2.2 precedence are all unchanged. `CELL_NM` is a **resolution** parameter,
not a scoring threshold, and it is corrected because at 40 nm the instrument had no resolution at
all. **Nothing was scored at 40 nm; there is no result to compare against and none is being
discarded.**

**Disclosure standard applied:** this is registered here, before the re-run, rather than noted
afterwards — and the killed run's log is preserved at `results/po5/unit2_qb_KILLED_cell40.log`.

**A2.5 — a pre-flight gate is added so this class of waste cannot recur.** Before the 9-run matrix
starts, the probe runs a **1 s single-arm pre-flight** and asserts the occupied-cell count clears
`MIN_CELLS`. If it does not, the probe **aborts before consuming the slot** and reports
`PREFLIGHT_FAIL` with the observed cell count. A resolution failure is now caught in ~1 minute
instead of ~50.

## AMENDMENT A2.6 — cell size 8 nm → 6 nm, and the pre-flight now tests the SCORED condition

**A2.5's pre-flight fired on its first use and saved the slot: `cells = 9` against `MIN_CELLS = 10`,
detected in 57 seconds.** The gate worked. Two corrections follow from it.

**(i) The pre-flight was testing the wrong condition.** It ran 1 s while the scored sample is at 5 s
— a different system state, with fewer dimers and therefore fewer occupied cells. A gate must test
the condition it gates. **The pre-flight now runs to the scored duration (5 s) and asserts on the
scored sample.** It costs ~5 min against a ~50 min matrix.

**(ii) 8 nm sits too close to the boundary to be a sound operating point.** At `cells = 9` versus a
threshold of 10, the instrument would be one marginal cell away from a structural `INCONCLUSIVE` on
every run — fragile by construction, independent of any result.

**The selection rule is stated in advance and applied mechanically — it is not a search for a
passing number.** `CELL_NM` must satisfy, in order: **(1)** above `r_p10 = 3.71 nm`, so a cell is not
dominated by a single close pair; **(2)** below `r_p50 = 9.78 nm`, so within-cell and between-cell
remain distinguishable; **(3)** yield comfortably more than `MIN_CELLS = 10` occupied cells at the
scored sample. All three bounds come from **Unit 1's** independently measured geometry. 8 nm passes
(1) and (2) and fails (3) empirically at 9 cells; **6 nm passes all three** (~37 cells geometrically
available across a 36 nm cloud) and is adopted.

**The integrity constraint, stated plainly: no verdict has been computed at ANY cell size.** The
40 nm run was killed unscored, the 8 nm run aborted at pre-flight before the matrix, and no
`ratio`, `d_input` or `d_null` has been evaluated on model data at any setting. **It is therefore
not possible for this choice to have been selected for an outcome** — there is no outcome to select
for. Every verdict threshold (`RATIO_CONFIRM = 3.0`, `RATIO_FALSIFY = 1.5`, `MIN_OCC = 5`,
`MIN_CELLS = 10`) and the A2.2 precedence remain untouched.

**If 6 nm also fails pre-flight, PO-5 stops and reports rather than continuing to step the value
down.** A third adjustment would stop being a geometric derivation and start being a search, and the
honest report at that point is *"the instrument cannot resolve pair structure in this geometry at the
registered `MIN_CELLS`"* — which is itself a finding about the measurement, and is reported as one.

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
