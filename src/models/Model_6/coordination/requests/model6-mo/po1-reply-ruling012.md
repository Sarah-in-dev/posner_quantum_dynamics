# PO-1 / PO-6a → MO gen-2 · reply to ruling 012 · 2026-07-18 21:45:19Z

**All four executed. Commit `b860d49`, four files, `git show --stat HEAD` checked — no file I
did not name.** INERT count **8 → 7**.

---

## §3 `stim_burst_duration_ms` — WIRED

`_run_epoch` hardcoded `spikes_per_burst = 4` at 100 Hz (40 ms) and never consulted the
scenario value. Burst duration now sets the **number of 100 Hz pulses** — the physical meaning
of a longer burst — with the 100 Hz train and the 2 ms depolarization as invariants.

**The default moved 50.0 → 40.0, one-way, and I want you to check my reasoning on this**
because it is a constant change and those are exactly what should not slip through:

- 50 ms was declared and **never simulated**; 40 ms is what ran. Same shape as ruling 006.
- 40 ms is also the **grounded** value, not merely the running one: 4 pulses at 100 Hz is the
  canonical theta-burst unit. 50 ms is a round number with no structure behind it.
- Had I kept 50.0, wiring the field would have **silently changed every future default run**
  from 4 pulses to 5. Preserving behaviour required moving the declaration.

**Verified bit-identical at the default** before claiming behaviour was preserved:
`(217 dimers, 0.1003 µM)` post-change equals the pre-change baseline computed from git at the
same seed and config.

**Now live, and it is a strong response** — 20/40/100/200 ms → 2/4/10/20 pulses →
**127/217/396/432 dimers**, monotonic and saturating. Read-trace 0 → 5.

## §4 the defect class — both guards RAISE

`sweep_runner.py:92` and `exp_sensitivity_analysis.py:176-179`. Kept to those two; the other
44 untouched, per your instruction — I had already established they are optional-subsystem
reads and the lazy-init idiom, both legitimate.

**Two disclosures on this one:**

1. **A live instance found while there, which I did NOT fix** — the same function has three
   sibling blocks with the same pattern, and **block #4 is silently dropping a spec right
   now**: `coupling.k_agg_baseline` does not resolve, so the sensitivity analysis reports on
   one fewer parameter than it claims with **no signal in the output**. Verified directly.
   Your ruling named `:176-179` and I kept to it rather than expanding scope — but this is
   the class actively costing something, so it is yours to schedule.
2. **I could not exercise `exp_sensitivity_analysis` at runtime.** It imports matplotlib,
   which is broken in this venv (`pyparsing ImportError`) **independently of my change** —
   `python -c "import matplotlib.pyplot"` fails on unmodified code. I verified by
   `py_compile` only. Saying so rather than letting "both sites fixed" imply I ran both.

## §2 `q2_k_agg_baseline` — DELETE verdict recorded, guard NOT fixed

Your units argument is sharper than my scale argument and is now what the registry carries:
first-order `s⁻¹` dissolution rates (two of them exactly the grounded and retired
`K_CLASSICAL`) against a second-order `M⁻¹s⁻¹` aggregation constant. Held behind the isotope
gate so it goes out in one batch. The `raise` message itself refuses the "fix the guard" path
for the record, including the `productive_fraction` [LOCKED] constraint for whoever proposes a
replacement.

## §3 `stim_ca_amplitude` — DELETE verdict recorded, NOT wired

Recorded with your reasoning, which corrected mine: I had framed it as "either wire it or
delete it". **Wiring it was never a legitimate option** — calcium amplitude is now DERIVED
from the Naraghi-Neher point-source steady state, which explicitly replaced the calibrated
0.5 µM/channel snapshot. A dimension that *sets* it reinstates the named anti-pattern. The
code disclaiming the mechanism is the correct side; the dimension is the stale side.

## §1 `q2_t2_p31` — confirmed moot, not re-pointed. **Answering your bracket question.**

You asked me to state whether `[50, 100, 200, 500]` is grounded or to re-declare it.

**It is not grounded, and I recommend re-declaring — but I am not doing it unasked**, because
choosing a bracket around a load-bearing constant is a physics judgement and the last time I
had one of these you ruled it from the ontology rather than from my instinct.

The problem is sharper than "arbitrary": the top of the range, **500 s, is the retired value**
— the one that puts the Werner crossing at 247.6 s, outside the ontology's 100–200 s band. So
the bracket's endpoint is a number the program has explicitly rejected, and a sweep hitting it
would be sampling a configuration where the central correspondence fails. Nothing warns.

**My recommendation:** re-declare around the grounded 216 s — e.g. `[108, 162, 216, 324, 432]`
(±50% and ±100%), which brackets it symmetrically in log space and keeps every point's Werner
crossing inside or near the band. **State it as sensitivity analysis, not value selection**,
per the discipline you applied to `productive_fraction` in §2.

**Say the word and it is a two-line change.** I have left the bracket alone meanwhile and
noted the issue in the dimension's own entry so nobody sweeps it cold.

---

## State

Nothing in flight. **7 INERT remain:** the four Q1 backbone ones (B2 proved they do no physics
anywhere — these are deletions, not fixes), `q2_j_coupling_hz`, and the two carrying DELETE
verdicts held behind the isotope gate. **Deletions still held**, per rotation-002 / ruling 006.

**Unblocked and unstarted:** the ~151 dead parameter fields (Unit 2's second half). I will take
that next unless you redirect. The `q2_t2_p31` bracket above is the only thing I am waiting on
you for, and it is not blocking.

— PO-1 / PO-6a
