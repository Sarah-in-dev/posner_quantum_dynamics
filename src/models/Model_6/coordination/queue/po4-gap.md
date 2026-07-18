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
