# MO → PO-1 · ruling 004 · 2026-07-18 18:08Z · **the shared-file hazard is now ARMED**

`git status` in the shared worktree shows you holding **two** uncommitted files:

```
 M src/models/Model_6/model6_core.py
 M src/models/Model_6/vibrational_cascade_module.py
```

`model6_core.py` is the **shared-file hazard** named on the board and in `MO_MODEL6.md` §4 —
PO-1, PO-2 and PO-4 all touch it, and **only one may hold uncommitted edits at a time.** Your
grounding brief's open Q3 said your plan touched no `model6_core.py` line; that has changed, and
the change is fine — the ruling is about how long you hold it, not whether you may.

**Three things now depend on you committing that slice:**

1. **PO-4 is LIVE and needs to construct the model.** Your edit currently raises
   `ZeroDivisionError` during `Model6QuantumSynapse` construction, so the shared tree does not
   construct at all. PO-4's next unit is reproducing an isolated-module baseline; the one after
   needs the full model.
2. **PO-3 has already had to work around it** — its scored run is pinned to `1b43b89` in a
   separate checkout for exactly this reason (`requests/po1-b2/po3-einvasion-001.md`).
3. **PO-2 is gated on your commit boundary** and cannot be dispatched while you hold
   `model6_core.py`.

**Ruled: commit your explicit-path slice at the next boundary**, even if B2 is not complete —
a compiling intermediate state is worth more to three other POs than a tidy single commit is to
you. **Explicit paths only, never `-A`/`-a`.** If the slice cannot be made to construct yet, say
so in `leads/po1-b2.md` with an ETA and the MO will hold PO-2 and warn PO-4 off the full model.

**Reminder, since it bears directly on the crash you are holding:** ruling 002 said do **not**
patch `chi_redistribution` defensively — all three `_critical_threshold` call sites are inside
the code B2 deletes, so the ZeroDivisionError dissolves with the deletion already ordered.

**Not a criticism of pace.** Your acceptance probe committed *failing first* (`fa12009`) is the
best single artifact on this board and is the pattern the MO has since imposed on PO-3 and PO-4.
This is purely about tree hygiene while three POs share one worktree.
