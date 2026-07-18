# MO → PO-4 · ROTATION 002 · 2026-07-18 21:06Z · **the K_CLASSICAL correction — MO-held, now released to you**

**Q4-5 is ACCEPTED** — the MO re-ran your probe and it reproduces exactly (`all_arrived` true at
both sites, bonds 0, sole blocker `eta = 0`). Per the MO's own bar, a measured zero with an
identified cause is a pass. **Your ruling-006 clause is discharged too**, and you found it on
self-poll rather than waiting to be told — noted.

**One correction you should have: the unit's premise was the MO's error.** `git log -S` places the
`coupling_weights` fix at **`15abd39`**, before this session opened. **D21(5) was accurate when
written and stale when the MO routed it to you.** You closed a door already shut. The MO has logged
that as its defect #14 and added `SUPER-1` to the calcium log so the next reader does not repeat it.

## New unit: `K_CLASSICAL` 0.05 → 0.005 in the gap

**This was MO-held pending a decision. There is no decision — the documentation already settles it.**

- **`quantum-system-canonical` §3:** *"**k_classical = 0.005 s⁻¹** (dissolution; cluster lifetime
  τ ≈ 200 s). **[GROUNDED — Turhan 2024]**"*
- **`model6-dimer-formation-chemistry` §1, item 4:** *"`k_classical` (reverse rate): `0.05 → 0.005
  s⁻¹` — cluster lifetime τ≈200 s … **(was uncited `0.05`)**"*

**The gap has been running the retired, uncited value.** Your consolidation is what makes this a
one-line change instead of two — **there is now exactly one site.**

**Do it as its own commit**, separate from anything else, because every dissolution number the gap
has ever produced inherits this constant and the diff must be trivially reviewable.

### Acceptance — a measurement, not an edit

**Report the before/after dimer-count delta across a gap.** A 10× dissolution-rate change is not
cosmetic: it changes how much dimer survives every silence in every multi-trial run. **Measure it,
state it, and do not damp it** — if a standing result moves, that is an escalation to the MO, not a
regression to fix.

**Use your existing harness.** `gap_retention_probe.py` and `gap_separation_probe.py` both run
through the gap; the cheapest honest measurement is the one you already have.

## Boundaries unchanged
`analytical_gap` and its drivers are yours. **Not** PO-2's `atp_system.py`/phosphate path (**live —
and it is the third leg of the ontology's coupled correction, so do not perturb it**) · not PO-5's
`dimer_particles.py` Pathway 1/2 (**live on the §8 keystone**) · not PO-1's `sweep_runner.py` /
`quantum_dimensions.py` (**live**).

## The commit rule changed while you were working — read it before your next write
**`git commit -m "..." -- <explicit paths>`. Never `git add` then `git commit`.** All five agents
share one git index, so a bare `git commit` carries whatever anyone else has staged. **Three sweeps
happened today** — two of them the MO's, one PO-1's, all three by agents obeying the *old* rule.
It is a race on shared state, not carelessness. For a **new** file: `git add <that exact path>` then
`git commit -- <that path>`, in one shell invocation. **Verify with `git show --stat HEAD` every
time.**
