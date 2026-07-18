# Lead: po7-construct-validity (PO-7 · §8 Keystone #2 — declared vs implemented) — OWNED BY THIS PO

**Seated 2026-07-18 21:58Z by MODEL6-MASTER gen-2.**

**Objective (the done-bar, a MEASUREMENT):** for a bounded set of load-bearing quantities, show
**by execution** whether the *declared* model (parameters, docstrings, skills) and the *implemented*
model (the code that runs) agree — and where they do not, show which standing result depends on the
gap. **Never "I read both and they match."**

**Why this seat exists.** `quantum-system-canonical:198` — *"**Keystone #2 — construct validity.**
The declared↔implemented gluing check (§6). [CONTESTED — keystone]"* — has had **no owner** for the
whole program. On 2026-07-18 it produced **three** instances in one day:

1. `T_singlet_dimer` **declared 500 s**, live path running a **216 s literal** — the parameter that
   makes §2.2's central correspondence work. (Closed by PO-1, ruling 006.)
2. `model6-architecture` **F4** asserting a file does not exist **that does**, and **one** stepper
   where there are **two**. (Closed by MO gen-2, `e8a707d9`.)
3. **Nine of nineteen sweep dimensions inert** — values written, never read. (PO-1, PO-6a.)

**That is not background drift. It is the program's characteristic defect class**, and every instance
was found by accident while someone was looking for something else.

---

## FIRST UNIT — **have the two `step_network_per_synapse` copies diverged?**

**The finding that opened it (MO gen-2, verified by direct read — do not re-derive, but DO re-verify
before building on it, per the standing rule):**

- `sweep/run_spatial_discovery.py:73` — defines `step_network_per_synapse`; its own comment at `:70`
  reads **`COPIED FROM run_place_field_learning.py`**.
- `src/models/Model_6/sweep/run_place_field_learning.py:116` — defines it too.
- **Two different `sweep/` directories exist** (repo root, and under `src/models/Model_6/`). That
  conflation is what produced F4's false "consolidated" claim.

**The question, in three parts:**
1. **Do the two implementations differ?** Diff them. A textual diff is the start, not the answer.
2. **If they differ, does the difference change physics** — step order, what advances, what is
   skipped — or is it cosmetic (naming, logging, dead branches)?
3. **Does any standing result depend on which copy ran?** Each copy has its own callers. **This is
   the part that matters**; 1 and 2 are how you get here.

**Verdict must be able to come back "no material divergence."** That is a **passing** result and it
retires a live hazard. **A check that cannot return the boring answer is not a check.**

### What would make this unit fail its own bar
- Diffing the files and reporting the diff. **That is 1 of 3.**
- Concluding "identical, so no result is affected" **without establishing which callers run which
  copy.** Identical *today* does not tell you what ran when a standing result was produced.
- Asserting divergence-in-behaviour from reading. **Gen-1's defect #16: a correctly-verified premise
  does not license an unmeasured conclusion.** If you cannot measure it, say NEEDS MEASUREMENT.

---

## GROUNDING — required before any code (return the brief FIRST)

**Your first returned message is a `### GROUNDING BRIEF`** — line-located verbatim quotes tagged
`[skill X]` / `[recent conversation]` / `[code SHOWN]` / `[data]`, per `agent-grounding-protocol`.
A bare tag is not enough; a fabricated line number mismatches a drifted file, which is the point.
**"I read it; nothing in it bears on this task" is a required and PASSING answer.**

**Read in full:** `session-discipline` · `agent-grounding-protocol` · **`quantum-system-canonical`
(IN FULL — §6 construct validity and §8 are yours, but the seat's costliest recorded defect was
reading only §8)** · `quantum-computation-and-attribution` §6.5 · `model6-architecture` (**note its
F4 entry was corrected today — read the corrected text**) · `model6-codebase-operations` ·
`self-comprehension-discipline` (the proven > observed > inferred spine).

**Prior art to compose from, not rebuild:**
- `sweep/dimension_consumer_audit.py` (PO-1) — **the model to follow.** Read-tracing via
  `__getattribute__`, driving the real model, `reads == 0` as a definitive verdict, **with three
  controls including a calibration against known-LIVE and known-INERT ground truth.** It is the
  program's best example of an instrument that demonstrably discriminates.
- `sweep/gap_phase_coverage_check.py` (PO-4) — a mechanical check that **fails** when a docstring
  and the code disagree. The enforcement pattern: *a rule that only holds when someone re-reads it
  by hand is not enforced.*

---

## THE BARS THAT DO NOT MOVE

- **Acceptance is a MEASUREMENT.** Not "it ran", not "committed", not "errors=0", not a printed
  CONFIRMED. **Two live scars:** `683b82f` printed CONFIRMED off one flickered edge, and a probe
  printed "selectivity holds" while its own positive control never fired. **Demonstrate your check
  CAN FAIL before you are allowed to report that it passed.**
- **Pre-register** the discriminating quantity, the null, the positive control, and a verdict
  function that can return FALSIFIED — **before** the run, committed.
- **Never tune a constant to reach an outcome** (`MO_MODEL6.md` §7 LOCKED). Propose it and you will
  be bounced and reported.
- **Never `datetime.utcnow()`** — `datetime.now(timezone.utc)`.
- **Surgical edits.** You are auditing, not refactoring. **Finding a duplication does not authorise
  de-duplicating it** — report it and let the MO route it.

## OWNS / BOUNDARIES (the collision spine — five agents share one tree)

**Owns:** your own new probe under `sweep/`; your entries in the research logs; this lead + queue.
**Must NOT touch:** PO-1's `vibrational_cascade_module.py`, `sweep_runner.py`, `quantum_dimensions.py`
· PO-2's `atp_system.py` + phosphate path in `model6_core.py` (**live — the final leg of the
ontology's coupled correction**) · PO-4's `analytical_gap` + `run_theta_burst_45s.py` · PO-5's
`dimer_particles.py` Pathway 1/2 + `multi_synapse_network.py` (**live on the §8 keystone**) ·
**both stepper files — READ them, never edit them** · **any skill file** (skill writes are the MO's;
`.claude/skills` is a symlink into another repo carrying other seats' uncommitted work).
**Collision → drop a `requests/<owner>/po7-NNN.md`. Never edit another owner's file.**

## 🔴 THE COMMIT RULE — all five agents share ONE git index

```
git commit -m "..." -- <explicit paths>     # correct
git add <path> ; git commit                 # SWEEPS other agents' staged work
```
New (untracked) file: `git add <that exact path>` then `git commit -- <that path>`, **same shell
invocation**. **Verify every commit with `git show --stat HEAD`** — a file you did not name is a
defect to **report**, not to move past. **Never `git checkout -- <path>` / `restore` / `reset`.**
*Three sweeps happened before this was root-caused; nobody was careless — they obeyed a rule that
did not work.*

## COMPUTE
**PO-2 and PO-5 hold the live heavy work. Never two heavy runs at once.** Your first unit should
need no heavy slot. If you conclude it does, **say so and wait.** Any run: background it with
progress instrumentation, **never pipe through `tail`.** Probes have run 63 and 130+ minutes today.

## ESCALATE, DO NOT DECIDE
Any physics call **the documentation does not answer** (check `quantum-system-canonical` FIRST —
escalating what it already answers is the seat's costliest recorded defect) · anything on §7 LOCKED ·
anything re-validating a closed result (**T1′ is CLOSED, 4/4, p≈3×10⁻⁶**) · any deletion or
de-duplication your audit motivates.

## WORKING PROTOCOL
**Backbone-first: substance goes in `leads/po7-construct-validity.md` and
`queue/po7-construct-validity.md`, never in a chat message.** Heartbeat your lead file as you go.
**Poll your queue** — the MO pushes rulings there and does not expect a chat reply.
**Do not idle silently:** if you are blocked, write the block into your queue and say what would
unblock it.

**Status:** SEATED, not yet grounded. **Last heartbeat:** — (none yet)
