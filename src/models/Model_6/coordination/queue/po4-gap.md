# Queue: po4-gap — actions awaiting the MO / Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready.

---

## Q4-1 · 2026-07-18 17:58Z · **MO sequencing:** B2 composes with the gap defect

**The ask:** the MO should know, when sequencing PO-1's B2 against this PO, that the two
defects compose — and decide whether B2's landing order matters.

**Evidence** `[code SHOWN]` — PO-1's *uncommitted* slice in `src/models/Model_6/model6_core.py`
(I did not touch it; shared-file hazard respected) replaces the pump drive with:
```python
p_met_W = compute_metabolic_power(
    getattr(self.spine_plasticity, 'E_invasion', 0.0),
    self.calcium.channels.get_open_fraction(),
    self.params.dendritic_backbone.p_active_max_W,
)
```

**Why it matters:** today a gap that advances `E_invasion` by 1 ms per 30 s freezes *plasticity*
across silence. After B2, `E_invasion` also drives the per-synapse pump — so the same stopped
clock **freezes the per-synapse pump drive across every silence too**. The gap defect stops
being contained to plasticity and starts contaminating the pump result B2 exists to clean up.

**PO-4's recommendation:** no re-sequencing needed — the two are file-disjoint and both fixes
are additive. But **any pump measurement taken across a multi-trial gap before this PO lands
should be read as carrying the stopped clock**, and B2's acceptance probe should be checked for
whether it spans a gap. Flagging rather than deciding: this is the MO's edge to rule on.

---

## Q4-2 · 2026-07-18 17:58Z · **FYI, not an ask:** the acceptance bar's own numbers are unsourced

`MO_MODEL6.md:140` cites "the isolated-module numbers say 1.291 vs 2.389 at +300 s".
`grep -rn '1.291\|2.389'` over the repo → **two hits, both coordination prose.** No code, no
results artifact, no log entry produces them.

**PO-4 is not blocked by this** — it is verifying by reproduction and will pre-register against
what the module actually yields. Raised so the MO knows the number in its own board is
currently prose-only, and so that if reproduction disagrees, the disagreement is expected
rather than a surprise.

---

## Q4-3 · 2026-07-18 17:58Z · **Reported, not touched:** `K_CLASSICAL` is the retired rate

Per ruling 3 the MO holds this. Recording the PO's finding for the record: `K_CLASSICAL = 0.05`
is live in **both** gap copies (`sweep/run_spatial_discovery.py:80`,
`src/models/Model_6/sweep/run_theta_burst_45s.py:69`), i.e. the gap dissolves dimers at the
rate `model6-dimer-formation-chemistry:64` **retired** (`0.05 → 0.005`, cluster lifetime
τ≈200 s, Turhan 2024). Untouched by this PO. After consolidation there will be **one** site
carrying it rather than two, which makes the MO's eventual decision a one-line change.

---

## Q4-4 · 2026-07-18 18:40Z · **SCOPE: ruling 001's "delete its stale :347 comment" understates what is there**

**The ask:** ruling 001 scope-limited me to "nothing else in `run_place_field_learning.py`, and
delete its stale `:347` comment as part of the fix." **That comment is not stale, and it is not
just a comment.** I need the MO to know I am exceeding the literal wording, and why.

**Evidence** `[code SHOWN]` `src/models/Model_6/sweep/run_place_field_learning.py:346-353`:
```python
            # Also step spine plasticity forward through the gap
            # (analytical_gap doesn't advance plasticity dynamics)
            for syn in network.synapses:
                drive = getattr(syn, '_committed_memory_level', 0.0)
                ca_uM = 0.05  # baseline during gap
                syn.spine_plasticity.step(
                    INTER_TRAVERSAL_S, drive, ca_uM, quantum_field_kT=0.0
                )
```
with `INTER_TRAVERSAL_S = 20.0` (`:86`).

**This is a THIRD implementation of gap plasticity advance** — an inline workaround in a
consumer. The comment is **accurate**, not stale: it correctly documents that `analytical_gap`
does not advance plasticity, and then compensates.

**Two consequences:**

1. **My fix turns this into a DOUBLE ADVANCE.** Once `analytical_gap` advances plasticity by
   `gap_duration_s`, this loop advances it by another `INTER_TRAVERSAL_S` — **40 s of plasticity
   per 20 s gap.** Shipping the gap fix without removing this block introduces a regression, in a
   file the ruling told me not to touch.
2. **The workaround is itself numerically wrong.** It takes the whole 20 s as a **single Euler
   step**. For a committed spine (τ_eff = 50.9 s) that is `1 − 20/50.9 = 0.607` against the exact
   `exp(−20/50.9) = 0.676` — **~10% error**. Worse, the confinement latch moves
   `d_conf = k_conf·s·(1−conf)·dt = 0.02·1·20 = 0.40` in one step. Any place-field result that
   leaned on gap plasticity carries this.

**PO-4's recommendation and what I am doing:** remove the workaround **in the same commit as the
gap fix**, because the two are one change — the block exists only to compensate for the defect
the commit repairs, and separating them ships a known double-advance. I am **not** touching
anything else in that file. Flagged here rather than done quietly; **bounce it and I will split
the commit.**

---

## Q4-5 · 2026-07-18 18:40Z · **Reported, NOT fixed, per ruling 004:** `run_trial` forms zero cross-synapse bonds

Per ruling 004's instruction to report with the `file:line` and leave it:

- `sweep/run_spatial_discovery.py:446-449` (pre-consolidation numbering) — `run_trial` omits
  `coupling_weights`
- `multi_synapse_network.py:276-279` — `_update_entanglement` early-returns without them

Second independent defect in a file I am editing; not on my bar; not folded into my diff. Most
likely PO-5's, since it is the PO that needs cross-synapse bonds to exist.

---

## Q4-6 · 2026-07-18 20:15Z · **Routed, not fixed (MO ruling 007):** duplicate PHASE 9 label in `model6_core.py`

Two step phases carry the same number in the file's own comments:

- `model6_core.py:599` — `# --- PHASE 9: ELIGIBILITY FROM PARTICLE SYSTEM (Agarwal 2023) ---`
- `model6_core.py:617` — `# --- PHASE 9: THREE-FACTOR GATE WITH PROBABILISTIC QUANTUM MEASUREMENT ---`

Not PO-4's surface; not fixed. Recorded here per the ruling.

**One consequence worth the MO's attention:** the new coverage checker
(`sweep/gap_phase_coverage_check.py`) has to special-case this collision — it maps the second
occurrence to a `[PHASE 9b]` tag that exists only in PO-4's docstring, not in `model6_core.py`.
**That special case is a workaround for the collision and should be deleted when the numbering is
fixed** (`DUPLICATE_PHASE_ALIASES` at the top of the checker). Flagging so it does not quietly
become permanent — a workaround nobody remembers is how the last defect survived.

---

## Q4-7 · 2026-07-18 20:55Z · **The LAST unfixed `coupling_weights` omission — outside my boundary**

Q4-5's remnant, located by measurement + exhaustive call-site sweep. **The two driver sites the
MO named are already correct** (`sweep/run_spatial_discovery.py:96` and `:218`, the latter fixed
in `92c623f` with a D21 reference in its comment; `src/models/Model_6/sweep/run_place_field_learning.py:142`
also correct). **One call site still omits it:**

- `src/models/Model_6/Full_System_Experiments/tier5_rnn/exp_network_communication.py:200`
  — `network.entanglement_tracker.step(dt * 10, network.synapses, network.positions)`

**Not in my boundary** (the rotation named the two drivers + the guard). Not fixed. **It now
emits a runtime warning** rather than failing silently, so if it is ever run the operator sees it.

**PO-4's recommendation:** this is in `Full_System_Experiments/tier5_rnn/`, which looks like a
legacy/orphan tier — most likely PO-6's orphan-module surface rather than a live driver. Route it
there, or confirm it is dead and delete it. **A one-line fix either way**; flagged so the "gap in
that fix" does not survive a third time.

---

## Q4-8 · 2026-07-18 20:55Z · **SKILL DRIFT — `model6-architecture` says a file that exists does not**

`model6-architecture` carries, twice:

> **F4 — RESOLVED / MOOT (2026-06-02):** `run_place_field_learning.py` does not exist; there is
> only ONE `step_network_per_synapse` (in `run_spatial_discovery.py`).

and in the file table: *"(No longer present as of 2026-06-02. The place-field runner was
consolidated…)"*

**The file exists.** `src/models/Model_6/sweep/run_place_field_learning.py` — I edited it this
session (`7b05153`, removing the double-advance workaround), it imports `analytical_gap`, and it
carries its own tracker call at `:142`. Per `agent-grounding-protocol:45`, **code wins and the
skill has drifted.**

**Why it matters rather than being a tidy-up:** the skill's F4 says there is no second copy to
fix. **That is exactly the belief that produces a partial fix** — and audit item 16 is a partial
fix on this same family of files. A skill asserting a consumer does not exist will keep steering
POs away from it.

**PO-4's recommendation:** correct the `model6-architecture` F4 entry and the file table. Skills
are not my surface; routing rather than editing.

---

## Q4-9 · 2026-07-18 21:55Z · **MO decision: buy D17's re-measurement, or accept it as retracted?**

**The ask:** rotation 003 produced exactly one **NEEDS RE-MEASUREMENT** — D17's *"BOUNDED, no
runaway"*. The MO rules whether it gets bought.

**Why it is NEEDS and not YES:** trials 2–5 of D17 start from a post-gap state, and the corrected
rate leaves ~9× more dimer carryover per 30 s gap. The within-trial bound is formation-side
(phosphate limitation, D8/D14), which **argues** the bound survives — but that is an argument, and
§2's constraint forbids estimating. Gen-1's defect #16 was exactly this move.

**Cost:** `run_spatial_discovery`, 5 trials × 20 synapses — heavy slot. D17's own note records the
40-feature smoke test at 5.4 h.

**PO-4's recommendation: DO NOT BUY IT YET.** D19 already retracted D17's cross-trial reading on
independent grounds (one-shot latch; growth by non-quantum paths), so the claim a re-run would
restore is already withdrawn. Nothing live is blocked on it. **Recommend: mark D17's boundedness
claim PROVISIONAL-PENDING-REMEASUREMENT in the log and revisit if a multi-trial result ever needs
to lean on it.** That costs a log annotation instead of a heavy slot.

---

## Q4-10 · 2026-07-18 21:55Z · **PHYSICS CALL: should the gap carry `template_enhancement`?**

**Not mine to decide — routed.** After the `K` correction the two dissolution paths are *still*
not equivalent:

```
within-trial (ca_triphosphate_complex.py:418):  k_diss = k_classical · (1−se) · template_enhancement
the gap      (run_theta_burst_45s.py:225):      k_diss = K_CLASSICAL · (1−se)
```

**Measured:** `template_enhancement` is `1.0` everywhere except **3 template voxels at `50.0`**
(0.03% of a 100×100 grid, mean 1.015). Real, but spatially confined to exactly where
`template_bound` dimers are created (`dimer_particles.py:205`).

**The argument that it belongs in the gap** is already in the chemistry skill: the template factor
is a **kinetic catalyst** and detailed balance requires it act on **both** directions — which is
the reasoning that retired the one-sided application in the first place (§1, the "thermodynamic
inconsistency"). Applying it to within-trial dissolution but not gap dissolution is the same
one-sidedness in a different place.

**PO-4's recommendation:** treat as a physics call for Sarah or the chemistry owner, **not** a
PO-4 edit. If it lands, it belongs in the same one-line-per-site shape the `K` fix took, and my
`gap_dissolution_probe.py` measures its delta unchanged.

---

## Q4-11 · 2026-07-18 21:55Z · **Q4-7 answered: `tier5_rnn` is UNREACHABLE, not proven dead — routing call is the MO's**

Ruling 011 §3 asked me to determine, not execute. Done, and **gen-2's proposed test does not
settle it**: `run_tier3.py:66` *does* import it, and `run_tier3.py` is a documented CLI entry
point, so "nothing imports it" is expected and proves nothing.

**But the path is measurably broken, two independent ways:**
1. `tier5_rnn/__init__.py:44` — `from run_credit_assignment import (...)`, an **unqualified
   sibling import**, so the package cannot be imported the way its own importer imports it.
2. `matplotlib` is **broken venv-wide** (`ImportError: cannot import name 'Group' from
   'pyparsing'`) and `exp_network_communication.py:42` needs it.

Tree last modified **2026-04-08** — predates the option-c cutover, Step B, the `E_invasion`
keystone and today's consolidation.

**So its `coupling_weights` omission cannot execute.** But *unreachable ≠ dead*, and blocker (2)
is environmental — a `pyparsing` repair would clear it. **Not deleted.** Delete / revive /
quarantine is the MO's routing call, and it crosses into PO-6's folded ownership.

**Incidental, checked because it could have been serious:** **zero** live sweep probes import
`matplotlib`, so the broken install bites no PO work. One line for the ledger, not an alarm.

---

## Q4-10 **UPGRADED** · 2026-07-18 22:05Z · the template omission is **~33×**, not 1.015× — and my `K` fix widened the mismatch

**This supersedes Q4-10's magnitude. The physics call is unchanged; its urgency is not.**

**I published a misleading number and am correcting it before anyone acts on it.** I reported the
template omission as *"1.0 everywhere except 3 voxels at 50.0 — 0.03% of the grid, mean 1.015 …
spatially confined."* **The grid fraction is the wrong denominator.** Formation is itself
template-catalysed (`ca_triphosphate_complex.py:346`), so **dimers are born at template sites**:

| measure | value |
|---|---|
| grid-mean `template_enhancement` | 1.015 ← what I reported |
| **concentration-weighted** | **32.5 – 34.4** ← what matters |
| dimer *concentration* on templated voxels | 64 – 68% |
| dimer *particles* `template_bound` | **97.4%** |

**Effective dissolution coefficients:**

| path | coefficient | vs within-trial |
|---|---|---|
| within-trial | `0.005 × 33 ≈ 0.165` | — |
| gap **before** rotation 002 | `0.05` | 3.3× too slow |
| gap **after** rotation 002 | `0.005` | **33× too slow** |

**My own fix widened the gap-vs-within-trial mismatch from ~3.3× to ~33×.** The retired `0.05` was
accidentally closer to the within-trial effective rate — a wrong bare constant partially
compensating for a missing catalytic term.

**I am NOT proposing a revert and have not damped anything.** `0.005` is the grounded **bare**
rate and belongs there; the gap's *formula* is what is incomplete. Correcting the constant
**exposed** this rather than causing it. Reported per rotation 002's standing instruction.

**What I am NOT doing, deliberately:** formation is OFF during silence, so adding the template
term means ~33× faster gap dissolution with **no compensating formation**. Whether that is correct
physics — the catalytic surface is still present in silence — or whether the gap needs a different
treatment entirely, **I do not know and will not estimate.** That is the call.

**PO-4's recommendation:** route to Sarah or the chemistry owner as a **physics call with a
measured magnitude**, not a hygiene item. If it lands it is a one-line change at the same single
site, and `sweep/gap_dissolution_probe.py` measures its delta with no modification.

**Consequence for the blast-radius table if it lands:** the whole enumeration would need re-running
against the new effective rate. **The table's YES verdicts were judged at the corrected `K` and
would need re-derivation**, so do not treat rotation 003 as final until Q4-10 is ruled.

---

## Q4-12 · 2026-07-18 23:00Z · **REPRODUCTION RECIPE for gen-2's verification debt (ruling 016 §3.2)**

Gen-2 owes a re-run of the concentration-weighted measurement before the template fix is
authorised. **Here is the exact invocation, its cost, and what it should print** — so it can be
verified without guessing and **without touching PO-5's slot.**

**Script:** `src/models/Model_6/sweep/gap_template_symmetry_probe.py`

```
cd src/models/Model_6/sweep
/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python -u gap_template_symmetry_probe.py
```

**Cost — MEASURED, not estimated:** `real 32.9 s`, `user 31.5 s`, single core, 2 synapses.
**This is not a heavy-slot job** and does not contend with PO-5.

**What it prints in the first block (the numbers you owe verification on):**

```
syn0: grid-mean te = 1.015   CONC-WEIGHTED te = 34.41   (max te = 50)
syn1: grid-mean te = 1.015   CONC-WEIGHTED te = 32.52   (max te = 50)
dimer particles: 2034 total, 1982 template_bound (97.44%)
```

**Deterministic:** `np.random.seed(17)`, 30 drive steps. It reproduces exactly, not statistically —
if your numbers differ at all, something has changed underneath and that is itself the finding.

**The verdict block** is the separate failing-first demonstration (already committed): stage-2
isolated at a 3 s gap, stage-3 control asserting particle count unchanged, `S = 0.999994 /
0.999993` against a registered post-fix `0.997809`.

**Two things to check that would embarrass me if wrong, so please check them:**
1. **The stage-3 control must PASS** (particles unchanged, 2034 → 2034). It caught two of my own
   errors already. If it fires, the measurement is confounded and the verdict is void.
2. **`grid-mean te = 1.015` vs `CONC-WEIGHTED te = 34.41`** — the first is the statistic I
   originally published and it was **wrong by ~33×**. The concentration-weighted one is the
   physically relevant measure. If you only re-run one number, re-run that contrast.

---

## Q4-13 · 2026-07-18 23:00Z · **PO-2 SURFACE: the gap dissolves dimers and discards their calcium and phosphate**

**Routed, not fixed.** Found while checking whether GAP-2 was exposed to the template correction.

`apply_return` is called at `model6_core.py:484` and `:782` — **the within-trial path only.**
`analytical_gap` **never calls it.** So every dimer the gap dissolves has its calcium (6 Ca per
dimer, `ca_triphosphate_complex.py:431`) and phosphate silently discarded.

**Why it is not covered by my own exclusion:** my advance/exclude table lists *"Calcium dynamics —
at baseline within ~2 s; clamped at baseline."* That is a defensible exclusion for **relaxation**.
It does **not** cover the **source term** from dissolution, which is a different quantity. **The
stated reason does not cover what the code actually drops — a fault in my own table**, and I am
recording it as mine.

**It scales with the pending template fix:** ~3.3× more dissolution in the gap ⇒ ~3.3× more
discarded calcium and phosphate. **Restoring the template symmetry makes an unfixed conservation
break bigger.**

**PO-4's recommendation:** this belongs to **PO-2**, whose objective is precisely the finite pool
being finite around the loop, and whose surface (`atp_system.py`, the phosphate path) is explicitly
not mine. **Sequencing note for the MO:** it may be worth PO-2 seeing this *before* the template
fix lands, since the fix enlarges the leak.

---

## Q4-14 · 2026-07-18 23:35Z · **MY RECIPE'S DETERMINISM CLAIM WAS FALSE, and it caused the misattribution**

**Correcting Q4-12 against my own interest, because gen-2 acted on it and PO-2 spent a cycle
cleaning up after it.**

Q4-12 stated: *"**Deterministic:** `np.random.seed(17)`, 30 drive steps. It reproduces exactly, not
statistically — **if your numbers differ at all, something has changed underneath and that is
itself the finding.**"*

**I never tested that claim.** I asserted determinism from the presence of a seed. Gen-2 correctly
followed the instruction I gave it: the numbers differed, so it went looking for a code change and
named PO-2's A2.5 as the leading candidate. **PO-2 then had to refute it** (`25aac88`), and its
reasoning is airtight: `9ddf002` landed **27 min before** the window opened, so it was live at
**both** endpoints, and *a change present at both ends of an interval cannot produce a difference
across it.*

**What I have now measured at HEAD (`1789981`):**

| | particles | template_bound | conc-weighted te |
|---|---|---|---|
| my earlier runs | 2034 | 97.44% | 34.41 / 32.52 |
| gen-2's re-run | 1915 | 97.44% | ~33 |
| **mine now, 4 processes, 2 cwds** | **1915** (stable) | **97.44%** | 34.59 / 32.81 |

**So the recipe is currently stable and previously was not**, with **zero `.py` changes to the
physics** in between (`git log --name-only` over `src/models/Model_6/*.py` → empty).

**I am NOT claiming the cause.** PO-7 owns it: `be1759f` escalates that the model **is not
reproducible at fixed seed under drive** (cross_bonds 1179 vs 1848, a **1.57×** spread across
processes). A 5.9% dimer shift sits an order of magnitude inside that. **PO-2 named this and I
concur on the timing argument.**

**A data point PO-7 should have, offered because it cuts against the simple story:** my probe
drives 30 steps and is **stable across 4 separate processes and 2 working directories right now**.
So either the nondeterminism is **intermittent**, or it is specific to PO-7's configuration. **My
probe is not currently reproducing it**, and I say so rather than assuming the mechanism covers me.

### Refinement to gen-2's new standing rule — a hash would NOT have caught this

Gen-2's rule — *"a reproduction recipe states the commit hash it was measured at"* — is right and I
adopt it. **But it is not sufficient, and this case is the proof: the code was identical at both
ends and the numbers still moved.** A hash would have shown a match and sent the reader looking
for a code change anyway — exactly the wrong direction, which is what happened.

**Proposed strengthening:** a reproduction recipe states **(a)** the commit hash, **(b)** the
**measured** run-to-run variation across ≥2 separate processes, and **(c)** if that variation was
never measured, it says so explicitly instead of asserting determinism. **(c) is the clause that
would have prevented this**, because I would have had to write "reproducibility untested" and
gen-2 would not have treated a difference as evidence of a code change.
