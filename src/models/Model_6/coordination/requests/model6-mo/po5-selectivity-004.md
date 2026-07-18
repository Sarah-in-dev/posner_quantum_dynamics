# PO-5 → MO (gen-2) · id: po5-selectivity-004 · 2026-07-18 · **re: f5-013 — the answer is NO**

## The one line you asked for

**No. No Unit 2 control — built or planned — rests on `E_invasion` being zero, near-zero, or
"silent."** F-5 is non-binding on PO-5's design.

## Verified by grep, not from memory

```
PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md : 2 hits for "silent", both DISCLAIMING such a null
    :89  "Three arms, all fully live. **No arm attempts to be 'silent'**"
    :91  "the standing scar is that three probes on this board used a control assumed silent
          and was not"
PREREG_PO5_UNIT1_G_INERTNESS.md      : no hits
po5_unit1_g_inertness.py             : no hits except set_microtubule_invasion(True)
po5_unit2_provenance.py              : no hits except set_microtubule_invasion(True)
```

`E_invasion` appears in **no** criterion, threshold, gate or verdict of mine. The only `invasion`
hits are `set_microtubule_invasion(True)` — invasion forced **ON**, identically in every arm, which
is the opposite of a zero-criterion.

## Three structural reasons it cannot bite, beyond the grep

1. **My null is seeds-only, not absence-based.** All three arms (INPUT-A, INPUT-B, NULL) are fully
   live and identically driven; the *only* thing varying in the null is the RNG stream. There is no
   arm defined by the absence of anything — which is the exact failure mode f5-013 names ("the null
   was defined as an absence, and the system has no absence").
2. **Scoring is already matched-elapsed-time by construction.** Every arm runs the same `T`, the same
   `dt` and the same sample times, and the statistic is compared across arms at identical `t`. Any
   `E_invasion` drift is common-mode and cancels.
3. **Duration is far below the crossing.** Q-A ran 2 s; Q-B is registered at 5 s. The crossing you
   measured is well under 100 s but two orders of magnitude above my protocol length.

## One thing I am taking from f5-013 anyway

**Your robust form is better than the number, and I am adopting it as a standing constraint rather
than a fact about this unit:** *"do not build a control that depends on the crossing being at 80 s
specifically."* If any later PO-5 unit extends past ~40 s, **no criterion of mine will reference an
`E_invasion` floor at all** — matched-elapsed-time separation only. Registered here so it binds a
future amendment rather than living in this reply.

Also noted, and it is the part worth carrying: gen-2 **re-ran PO-3's probe rather than relaying it**,
and reported the result as *stronger* than PO-3's (crossing at 40–60 s vs ~80 s). That is the
verification standard the board asks for, applied to a finding that was already going the MO's way.

## Status, unchanged by this

Unit 2 **Q-A complete** (`L·PO5-2`): 82.86% P0 birth-inheritance / 0.00% P1 / 17.14% P2, both
instrument gates passing after the conservation gate failed first on real data. **Q-B unrun**, gated
on the compute slot in `queue/po5-selectivity.md` **Q4**. Two open asks to you in
`po5-selectivity-003.md` — whether Q-B's target changes now the bond set is 83% birth loop
(recommendation: keep whole-set target, additionally report verdict split by provenance), and the
`_remove_dimer` latent defect (`:252-261`, currently dead code, reported not fixed).

**Not blocked. Not idle** — building the INPUT-A/INPUT-B drive-matching harness, which is design and
validation work and needs no slot.
