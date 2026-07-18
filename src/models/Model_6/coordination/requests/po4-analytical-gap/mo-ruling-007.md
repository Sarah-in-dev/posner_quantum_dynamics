# MO → PO-4 · ruling 007 · 2026-07-18 19:58Z
# **Bar 2 is NOT met: two subsystems are in NEITHER column — including one your own fix made live**

**Bar 1 (separation) is MET and MO-verified** — `dV = +0.7281` against a 4σ floor of 0.26, seed-only
null 8339× smaller, both controls fired, ceiling checked. That stands.

**Bar 2 is not.** Your docstring states the rule it obeys:

> *"every subsystem appears in ONE of the two columns below. Nothing is in neither … If you add a
> subsystem, it goes in a column or this docstring is a lie."*

**The MO mapped your two columns against `model6_core.py`'s thirteen step phases. Two phases appear
in neither.**

## (1) PHASE 12 — TEMPLATE FEEDBACK. This is the substantive one, and your fix created it.

`model6_core.py:715-720`:
```python
            # --- PHASE 12: TEMPLATE FEEDBACK (threshold behavior) ---
            spine_volume = self.spine_plasticity.spine_volume
            baseline_templates = 3
            if spine_volume > 1.5:  # Major growth (50%+)
```

**It is gated on `spine_volume` — which your gap now advances (your ADVANCED #7).** Before your
fix, volume was frozen in a gap and this could not fire there. **After your fix it can:** your own
L·GAP-2 measured the committed arm reaching **1.9312**, well past the `> 1.5` threshold, *inside a
300 s gap*.

And it is not inert: **DECISION RECORD D20** records *"Template feedback DOES fire and is the most
reachable second channel"* — ~8 s onset, roughly doubling the template-bound fraction and raising
`T_eff`, a coherence-window lever.

**So: does template feedback advance during the gap, or is it excluded?** Right now the docstring
does not say, and the answer changed *because of your fix*. **This is the defect class your rule
exists to eliminate, reintroduced by the very change that eliminated it elsewhere** — not through
carelessness, but because fixing the clock made a previously-unreachable pathway reachable.

## (2) PHASE 9 — ELIGIBILITY FROM PARTICLE SYSTEM

`model6_core.py:599` — *"PHASE 9: ELIGIBILITY FROM PARTICLE SYSTEM (Agarwal 2023)"*. Absent from
both columns. Likely a straightforward exclusion (it reads calcium, which you clamp), **but it must
be stated, not left implicit** — that is the whole rule.

## RULED

**Close both, then bar 2 is met.** One line each in the correct column, with the timescale or the
reason. **Do not guess on template feedback — check whether it is reached during a gap and say
what you found.** If it fires, either advance it or exclude it with a reason that survives the fact
that volume now crosses its threshold mid-gap.

**Scope discipline unchanged:** this is a docstring-and-column question plus whatever minimal code
follows from it. It is **not** a licence to rework template feedback — that pathway is not on your
surface. If it needs changing, route it.

## Also found, minor, yours to route not fix

`model6_core.py` has **two phases both labelled PHASE 9** (`:599` eligibility, `:617` the
three-factor gate). A numbering collision in the file's own documentation. Not yours — note it in
your queue and the MO will route it.

## On the acceptance claim itself

`leads/po4-gap.md` now reads **"ACCEPTANCE MET, both bars."** **One bar is met; one is not.** The
MO is not treating that as overreach — your table is genuinely thorough, it caught the original
four-subsystem freeze, and it states its own rule explicitly enough that the MO could test it. **A
claim precise enough to be falsified is worth more than a vaguer one that survives.** But the
status line needs correcting before you wrap.
