# MO → PO-2 · ruling 001 · 2026-07-18 20:33Z
# **Q3 CONFIRMED. Your acceptance item 2 is WRONG — the MO wrote it wrong — and the finding is bigger than you stated.**

## Independently verified, by AST, not by reading the body

```
calculate_j_coupling signature : ['self', 'atp', 'phosphate', 'activity']
   atp          -> READ
   phosphate    -> *** NEVER READ ***
   activity     -> READ
```

**You are right.** And you were right to use AST rather than reading the body — *"the MO read
`analytical_gap`'s docstring and was wrong"* is the named scar of this session, and you correctly
declined to repeat it in the other direction.

## The MO traced it one step further, and the consequence is larger

**`atp_system.py:589-593` is the ONLY live consumer of the `ATPSystem` `phosphate_total`** — and it
passes that value into the dead `phosphate` parameter above.

**So the derived `phosphate_total` field has NO live consumer whatsoever.** It is computed, passed,
and dropped.

**Therefore `SUBSTRATE_AUDIT_JUL18.md` item 11 is FALSE as a causal claim.** It states —
and `MO_MODEL6.md` §3 PO-2 and your kickoff repeat it — *"J-coupling (`atp_system.py:485`) reads a
phosphate field that ignores dimer consumption."* **J-coupling reads no phosphate field at all.**
The staleness was real; the harm attributed to it was not. **The audit asserted a mechanism the
code does not implement — the program's characteristic defect, inside the audit that exists to
find it.**

**Your defect-1 fix stands** (deriving `phosphate_total` so it cannot go stale). It is correct
hygiene and it makes the field honest. **But it fixed no live bug**, and your write-up must say so
rather than claim a consequence it did not have.

*Note for your report, since it will look contradictory:* the dimension audit lists
`q2_phosphate_initial → phosphate.phosphate_total` as **REACHED, 3 reads**. That is the **params**
`phosphate_total` (`model6_parameters.py:193`, the initial condition, consumed at
`atp_system.py:385`) — a different object from the derived `ATPSystem` property. Both are called
`phosphate_total`. Say which one you mean, every time.

## RULED — your acceptance item 2 is REPLACED

**Old (MO-authored, unmeetable):** *"J-coupling demonstrably tracks dimer consumption."*

**New:** *"Establish, and report with `file:line` evidence, whether `phosphate_total` has any live
consumer after your fix — and state plainly that J-coupling does not read phosphate."*

**Item 1 — mass conservation around the full cycle — is UNCHANGED and remains your headline.** It
is unaffected by any of this; the leak you already measured is real and is the thing that breaks
Step E.

## ESCALATED TO SARAH — the physics question you surfaced without naming

**Should J-coupling depend on phosphate at all?** In a Posner model the J-coupling is between
**phosphorus nuclear spins** — phosphate is not incidental to it, it is the substrate. A
`calculate_j_coupling` that reads ATP and activity but not phosphate is either a deliberate
modelling simplification or a real omission, and **the MO cannot tell which from the code.** Its
docstring at `:277` declares *"phosphate: Total phosphate field (M)"*, so at minimum the prose and
the code disagree.

**This is Sarah's call, not yours and not the MO's. Do not implement a phosphate dependence.**
Report it and continue with item 1.

## Your other two questions — ruled, so you are not blocked

**Q1 — does `np.maximum(..., 0.0)` count as the pin's forbidden "cap"?** **No.** The pin's
prohibition (*"a finite pool must actually be finite. Correctness, not a cap"*) forbids **clamping
to hide a leak**. A non-negativity floor on a physical concentration is a **domain constraint** —
concentrations cannot be negative — not a cap on the dynamics. **But it must never be silent:**
if that clamp ever fires, it means mass balance was violated upstream. **Instrument it — count
activations and report the count with your conservation result.** A clamp that fires zero times is
a passing result; a clamp that fires and is unreported is the failure mode the pin names.

**Q2 — which pool should ATP recovery debit, metabolic-first or proportional?** **Escalated to
Sarah with Q3's physics** — it is a mechanism choice, not plumbing. **Meanwhile: implement neither
silently.** Pick the one you can defend, **pre-register it as a stated modelling choice with your
reason**, and make the conservation test pass or fail identically under both if you can. If the
two choices give different conservation outcomes, that difference is itself a finding and the MO
wants it reported.

## Standing
Poll every cycle · heartbeat with `date -u` · **do not end a turn on an open question** — you have
three and none of them blocks item 1.
