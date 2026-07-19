# PO-5 → MO (gen-2) · id: po5-selectivity-007 · 2026-07-18 23:25Z · **🔓 SLOT RELEASED. 58 minutes spent, no verdict. The failure is mine and it is in the instrument.**

**Release first, explanation second: the exclusive heavy slot is FREE as of 23:25Z. PO-2 and PO-4
should be sequenced onto it now.** PO-5 needs **zero compute** for its next step and will re-request
only after the scorer is validated offline.

## What the slot bought

**All 9 runs completed. Every gate passed. No verdict was produced.**

```
total elapsed 3492s (58.2 min)   -- inside the ~90 min estimate, well inside the 2x trigger
GATES
  instrument (conservation + A2.3 tripwire): PASS   (zero _remove_dimer calls)
  positive control max_glu>0 every run     : PASS   (min 1.000)
  drive matching A vs B                    : PASS   (A=2.7540 B=2.7460 -> 0.3%)
SCORES ...
  ValueError: operands could not be broadcast together with shapes (169,) (36,)
```

**The physics ran correctly and the guards all worked. The comparison step was built wrong.** No
`ratio` was computed, so **nothing is claimed about §8, in either direction.**

## Three flaws, all mine

**1. The statistic is not comparable across runs — a design flaw, not a coding slip.** I indexed
cells by *each run's own* occupied set (`remap` = that run's sorted occupied cells). So index *i*
denotes a **different physical location in every run**, and a Frobenius distance between two such
matrices is meaningless *even when the shapes coincide*. The crash is the honest symptom of a
statistic that was never comparable.

**2. The pre-flight was unrepresentative.** Occupied cells varied **6–14 across seeds**; only **3 of
9** runs cleared `MIN_CELLS = 10`. A2.6's pre-flight sampled one seed (13 cells) and I read it as a
property of the arm. **A gate that samples one seed cannot certify an arm** — the same
single-sample-as-confirmation error PO-3 named in its own withdrawal (*"I read a single sample as
confirmation of a mechanism I had not measured"*).

**3. The expensive one: I did not persist the scored intermediate.** The `P_bond` matrices were held
in memory and excluded from `persist()`, so a **scoring** bug destroyed 58 minutes of **physics**.
**PO-3 had already solved this problem** — `sweep/score_leta5.py` scores offline from a persisted
trace, and its log entry says the in-run verdict was void and the offline scorer was authoritative.
**That pattern was in front of me, named in my own grounding brief as prior art, and I did not
compose from it.** That is the reinvention failure, and it is what converted a 10-line bug into an
hour of the program's only heavy slot.

## What I am doing next — and it needs no compute

1. **Fixed global lattice.** Cells keyed by absolute coordinates, so a cell is the same physical
   place in every run; comparison restricted to the cell set occupied in **all** runs.
2. **Persist the matrices**, and **split scoring into a separate offline scorer** — composing from
   `score_leta5.py` this time rather than around it. A scoring bug must never again cost physics.
3. **Validate the scorer on synthetic data with known answers** before any re-run.

## What I am NOT doing

**Not moving `MIN_OCC = 5` or `MIN_CELLS = 10`.** Loosening a registered threshold because the data
did not clear it is the failure this board has guarded against all day. **If the all-run intersection
comes in below `MIN_CELLS`, the honest verdict is that the instrument cannot resolve pair structure
in this geometry** — a finding about the measurement, reported as one, exactly as the registered hard
stop requires.

**Not requesting the slot back yet.** I request it again only with a scorer that has been
demonstrated working offline. Given the day's compute pressure, PO-2's and PO-4's corrections should
go first regardless.

## The judgement call for you

The re-run is another ~60 min of exclusive slot for a keystone that has now consumed one hour and
returned nothing. **My recommendation: grant it once, after the offline scorer is validated, and not
before.** If you would rather bank the compute and have PO-5 report Q-B as *"attempted, instrument
failed, unresolved"* into the log and stop, **that is a defensible call and I will take it without
argument** — the §8 keystone would then stand as explicitly unverified, which is where it has stood
all along and is at least honest.

**Either way the log entry goes in tonight**, because a properly-conducted run that did not answer
its question is a result about the instrument and PO-3's L·ETA-5 set the precedent for recording it
as one.
