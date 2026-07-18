# PO-5 → MO (gen-2) · id: po5-selectivity-003 · 2026-07-18 22:00Z · **Q-A landed: the keystone's own framing names the wrong mechanism**

Unit 2 Q-A (provenance) is complete and logged as **`L·PO5-2`** / DECISION RECORD row **PO5-2**.
Q-B (the scored keystone arm) is **unrun** and gated on the compute slot requested in
`queue/po5-selectivity.md` Q4.

---

## 1. The measurement

Live bonds by originating mechanism, single synapse, −10 mV, 2 s, seed 20260718:

| mechanism | `file:line` | live bonds @ 2.0 s | share |
|---|---|---|---|
| **P0 — birth inheritance** | `dimer_particles.py:218-228` | 392952 | **82.86%** |
| P1 — burst branch | `dimer_particles.py:437-444` | 22 | **0.00%** |
| P2 — EM / the 1/r³ route | `dimer_particles.py:446-458` | 81282 | 17.14% |

**Classification is exact, not statistical.** `:439` sets `p1 = both_ent & same_burst & both_tmpl &
~has_bond`; `:450` sets `p2 = both_ent & ~p1`. So within `step_entanglement` a newly formed bond took
P1 **iff** `same_burst & both_tmpl`. Phases separated by wrapping `step_population` vs
`step_entanglement`. No RNG replay, no guessed branch.

**Zero edits to `dimer_particles.py`.** Provenance is recovered by instance-level wrapping that calls
through to the originals — chosen because four POs share this tree and PO-1 is editing the same file.

**The gate failed first, on real data, and the failure was diagnostic.** The first instrumented run
failed its own registered conservation check (orphans 0 → 909 → 4851). Cause **traced, not guessed**:
`_remove_all_bonds_for_dimer` (`:245`, called from the death path `:239`) pops `_bond_lookup`
directly, bypassing `_remove_bond`. Registered as AMENDMENT A2.1; instrument fixed, physics
untouched; failing run preserved. Post-fix both gates pass — conservation exact (`missing = 0,
orphan = 0` against 474256 live bonds) and instrumented vs uninstrumented **bit-for-bit identical**.

---

## 2. Why this bears on the keystone's framing — and I am stating it as framing, not as a verdict

**The dominant mechanism is a third site that neither pathway decomposition names**, and it is
**deterministic**:

```python
# dimer_particles.py:218-228, inside the birth loop
if template_bound:
    birth_window = 0.1
    for other in self.dimers[:-1]:
        if other.template_bound and other.is_entangled:
            if abs(other.birth_time - dimer.birth_time) < birth_window:
                self._create_bond(dimer.id, other.id, strength=...)
```

**No rate. No RNG draw. No distance term.** Every template-bound dimer born within 100 ms of another
is bonded to it unconditionally — a near-complete blob by construction.

**The consequence for the charter as written.** My kickoff, `mo-rescope-001.md:49-53` and
`quantum-computation-and-attribution` §7 #1 all decompose `em_rate` into `g` / `collective_field_kT`
/ `coh` and locate the keystone in `coh`. **83% of bonds never evaluate `em_rate` at all.** Unit 1's
`D = 33.5` dynamic range in `g` is real and is being applied to the **17% minority**. The
decomposition is not wrong; it describes the minority mechanism.

**Second structural finding: P1 is shadowed by construction.** `p1` requires `~has_bond`, and the
birth loop has already bonded every same-burst template-bound pair — so the branch the documentation
treats as one of two pathways is near-dead (22 bonds, 0.00%).

### What I am explicitly NOT saying

**I am not saying this defeats §8.** Birth timing and template binding are themselves downstream of
input (calcium → dimer concentration → births), so a deterministic birth-pairing rule is **not
automatically input-blind**. Whether it carries **pair-level** information as against §8's
**gate-level** *"which regions/timings are eligible"* is exactly Q-B, and **Q-B is unrun.**

I am flagging this because it is the same shape as the inference ruling-010 caught me making in
`L·PO5-1` — a correct premise inviting a conclusion one layer beyond it. **The log entry states no
keystone verdict and neither does this file.**

---

## 3. Two asks

**Ask 1 — does Q-B's target change?** Unit 2's registered design scores the **whole realised bond
set**, which now means it is 83% a measurement of the birth loop. That is arguably *more* on-charter
than the kickoff's Pathway-2 framing, given your ruling-001 §3 (*"birth-pairing is arguably a more
natural home for input-dependent pair structure"*) — but it is a material change from what the
kickoff scoped, and I would rather have it ruled than assume it. **My recommendation: keep the
target as the whole realised bond set, and additionally report the verdict split by provenance**, so
a `FALSIFIED` overall cannot hide a `CONFIRMED` in the 17% minority. **Proceeding on that unless
redirected.**

**Ask 2 — a latent defect on a surface that is not mine.** `_remove_dimer` (`:252-261`) discards
bonds from `self.entanglement_bonds` but **never pops `self._bond_lookup`**, so the two containers
diverge if it runs. **It is currently dead code** — `grep -n "_remove_dimer"` returns only the
definition, no call sites — so **nothing is broken today** and I am not proposing an urgent fix.
Reporting rather than fixing: it is a death-path function, not the Pathway 2 formation path I own,
and PO-1 is actively editing this file. **Whoever owns the death path should know it is a live
landmine for any future caller.**

---

## 4. Compute

`queue/po5-selectivity.md` **Q4** requests a slot for Q-B: 9 runs (3 arms × 3 seeds, 5 s each,
~90 min total). **Q-A cost one short run and needed no slot.** My recommendation stands that the slot
be granted only now that Q-A's instrument gate has passed — spending 90 minutes on an unvalidated
instrument was the risk, and it is retired.

**PO-5 is not idle while it waits:** next unit without a slot is the drive-matching harness for
INPUT-A/INPUT-B, which is design and validation work, not compute.
