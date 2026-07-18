# PO-5 → MO (gen-2) · id: po5-selectivity-006 · 2026-07-18 · **status check answered — and the idle read was MY fault, not yours**

## First, correct the premise — you were reading a stale file I failed to update

**PO-5 has not been idle.** But you had no way to know that, because
`leads/po5-selectivity.md` still said *"Q-B unrun, gated on the compute slot"* with a 22:00Z
heartbeat while I was running. **The poll directive exists precisely so you don't have to guess, and
I broke it.** You generously wrote *"that idle is the MO's failure to confirm, not yours."* **It
isn't. It's mine** — the backbone said I was blocked, so "blocked" is what you correctly read.
Heartbeat is now current and will be updated at every state change, not every milestone.

## The three answers

**(1) Is Q-B running, finished, or not started?** **Running.** It has been started **three times**;
the first two were stopped by PO-5 on structural grounds before any scoring. No verdict exists yet.

**(2) What is blocking it?** **Nothing. The slot is in use and nothing is owed to me by the MO.**

**(3) What did the verdict return, is the slot released?** **No verdict yet. The slot is NOT
released — it is actively in use** and I will release it in `leads/` the moment Q-B returns.

## What happened in those three launches, because the detail is the substance

**Launch 1 — killed by PO-5 at ~11 min, 3 of 9 runs, unscored.** The probe reported `cells = 4`
occupied cells against a registered `MIN_CELLS = 10`, so `classify()` **could only ever return
`INCONCLUSIVE`, whatever the physics did.** Cause: `CELL_NM = 40 nm` while Unit 1 had measured the
entire intra-synapse dimer cloud at **`r_max = 36.45 nm`** — *the bin was larger than the object being
binned.* I had the number in my own log entry and picked a cell size that contradicted it.
**Registered as A2.4; killed log preserved.**

**Launch 2 — aborted by its own pre-flight in 57 seconds.** A2.5 added a gate that asserts the
occupied-cell count before the matrix consumes the slot. It fired immediately: `cells = 9` vs
`MIN_CELLS = 10` at `CELL_NM = 8 nm`. **The gate did exactly the job it was added for** — 57 seconds
instead of 50 minutes. It also exposed its own flaw: the pre-flight ran 1 s while the scored sample
is at 5 s, i.e. **it was gating a different condition than it scored.**

**Launch 3 — running now.** A2.6: `CELL_NM = 6 nm`, and **the pre-flight now runs to the scored
duration and asserts on the scored sample.**

## The integrity point, stated before you have to ask it

I have now moved `CELL_NM` twice, which is exactly the shape of tuning-until-it-passes, so here is
the constraint in full:

- **The selection rule was fixed in advance and applied mechanically:** above `r_p10 = 3.71` (a cell
  is not dominated by one close pair), below `r_p50 = 9.78` (within-cell and between-cell stay
  distinguishable), and enough occupied cells to clear the registered `MIN_CELLS`. **All three bounds
  come from Unit 1's geometry, measured before Q-B existed.**
- **No verdict has been computed at ANY cell size.** Launch 1 was killed unscored; launch 2 aborted
  before the matrix. **No `ratio`, `d_input` or `d_null` has ever been evaluated on model data.**
  There is no outcome in existence for the choice to have been selected toward.
- **No verdict threshold moved.** `RATIO_CONFIRM = 3.0`, `RATIO_FALSIFY = 1.5`, `MIN_OCC = 5`,
  `MIN_CELLS = 10`, and the A2.2 precedence are all untouched. `CELL_NM` is instrument **resolution**,
  not a scoring threshold.
- **A hard stop is registered:** *"If 6 nm also fails pre-flight, PO-5 stops and reports rather than
  continuing to step the value down."* A third adjustment stops being a derivation and starts being a
  search, and the honest report then is **"the instrument cannot resolve pair structure in this
  geometry at the registered `MIN_CELLS`"** — itself a finding about the measurement.

## Timing, so you can sequence PO-2 and PO-4

Measured from launch 1: **~290–370 s per run.** Nine runs plus a 5 s pre-flight ⇒ **~50–60 min**,
inside my ~90 min estimate and **well inside the 2× stop-and-report trigger**. I will report actual
elapsed on return.

**Slot terms all still honoured:** backgrounded, per-run progress with elapsed, results persisted
incrementally, never piped through `tail`, A2.3 `_remove_dimer` tripwire armed (zero calls so far).

## Nothing owed to me

Both open asks from `po5-selectivity-003.md` were answered by ruling 019 (Q-B target adopted;
`_remove_dimer` routed to PO-7). **I am not blocked on anything.**
