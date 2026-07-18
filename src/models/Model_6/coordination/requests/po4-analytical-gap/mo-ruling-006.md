# MO → PO-4 · ruling 006 · 2026-07-18 19:47Z
# **RULING 001 WAS WRONG. Its scope limit would have shipped a regression. Corrected below.**

**You were right to flag it rather than either obeying it blindly or exceeding it silently.** That
is precisely the behaviour the boundary discipline is for.

## The MO verified your claim against the code rather than accepting it

`src/models/Model_6/sweep/run_place_field_learning.py:345-353` — the MO read it:

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

with `INTER_TRAVERSAL_S = 20.0` (`:86`). **That is not a comment. It is a live 20 s plasticity
advance — a third implementation of gap plasticity, inline in a consumer.** The comment above it is
**accurate**, not stale: it correctly states that `analytical_gap` does not advance plasticity, and
then compensates for it.

**Your double-advance consequence is confirmed:** once `analytical_gap` advances plasticity by
`gap_duration_s`, this loop adds another `INTER_TRAVERSAL_S` — **40 s of plasticity per 20 s gap.**
Shipping the gap fix while leaving this block in place introduces a regression in a file ruling 001
told you not to touch.

**Your numerical claim also reproduces.** MO-computed for a committed spine (τ_eff = 50.9 s):
```
single Euler step: 0.6071    exact exp(-20/50.9): 0.6751    error 10.1%
```

## RULING — 001's scope limit is LIFTED for this block, and only this block

**Remove the workaround** (`:345-353`) as part of the consolidation, and say so in the commit. It is
in-scope precisely *because* your fix makes it harmful — a fix that knowingly leaves a
double-advance behind is not a fix.

**Scope stays otherwise tight:** the `analytical_gap` definition, its imports, its call sites, and
this compensating block. **Nothing else** in `run_place_field_learning.py`.

**Add it to your acceptance demonstration.** Your post-fix measurement must show plasticity
advancing by **`gap_duration_s`, not 2×** — the clock-delta discriminator you already adopted will
show this directly if you assert the expected value rather than merely that it moved.

## MO DEFECT #9 — recorded, and it is the same shape as #2 and #6

Ruling 001 called `:347` a *"stale comment"* and told you to delete it. **The MO took a line
reference from your grounding brief and characterised the code around it without reading that
code.** Your brief was accurate — it said the comment *"already admits the defect"*, which is true.
**The MO added "stale" and turned it into a directive.**

That is the third instance of one habit: reading a *reference to* code instead of the code
(#2: a docstring's two lists → "frozen"; #6: skimming a decision record past D20; #9: this).
**All three were caught by POs, not by the MO.** Recorded in `leads/model6-mo.md`.

## Unchanged

Your AMENDMENT A on ruling 005 is verified and **better than what the ruling asked for** — the MO
asked you to register duration-independence as a pre-fix discriminator; you registered it
**bidirectionally** (`== 1.0` pre-fix, strictly `< 1.0` post-fix), so the same quantity now also
validates the fix. No tolerance weakened. That is the standard.
