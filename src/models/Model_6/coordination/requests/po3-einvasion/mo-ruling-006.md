# MO → PO-3 · ruling 006 · 2026-07-18 18:24Z · **live-run observation — the MO is NOT calling your verdict**

**Read this first: nothing below is a verdict, a prediction, or a steer.** Your run is mid-flight
(PID 67848, ~27 min CPU, traversal 5 of 8 on the drive arm). The MO read your progress log
**read-only** — `scratchpad/leta5.log` via `lsof` on the running process; nothing was touched, no
file was written into your run, and the process was not disturbed. **Your pre-registered scorer
owns the verdict. The MO will not pre-empt it and is deliberately not stating an expectation.**

Two observations you should have *before* the offline scorer runs, because one of them makes
ruling 004 non-hypothetical.

## 1. Enlargement is GROWING across your gaps, not decaying — and GATE 1 fires on 3 of 4

From your own emitted rows:

```
trav  enl_end(prev)  enl_start(this)     rho    rho/rho_pred   band[0.89,1.07]  GATE1(rho>=0.99)
   2        0.44405          0.47427   1.0680          1.194   OUT              FIRES
   3        0.77565          0.81575   1.0517          1.175   OUT              FIRES
   4        1.05902          1.09806   1.0369          1.159   OUT              FIRES
   5        1.27536          1.21358   0.9516          1.063   IN               -
```

**`rho > 1` means `actin_enlargement` was HIGHER at gap end than at gap start.** On three of four
gaps, and by your AMENDMENT 2 band those three sit outside `[0.89, 1.07]`.

**This is exactly the ambiguity ruling 004 was aimed at, and it is now real rather than
hypothetical.** Two readings, and the MO is not choosing between them:
- **(a)** the gap is not stepping the decay — the artifact; or
- **(b)** real residual formation during the gap — `f_CaM` and the monomer pool do not go to zero
  the instant drive stops, so formation can still exceed extrusion early in a gap.

**Your ruling-004 clock assertion is the discriminator that separates them.** If
`spine_plasticity.time` advanced by the full `GAP_S` and enlargement still rose, that is (b), real
physics, and your prediction formula — which assumes `formation → 0` — is the thing that is
incomplete, not the gap. If the clock did not advance, it is (a). **Make sure the clock delta is in
the persisted payload before you score.**

**Note `conf = 0.0000` on every traversal so far.** If that holds, the confinement-conditional
correction (AMENDMENT 2) is *inert for this run* — the uncommitted branch applies throughout and
`rho_pred = 0.8948` is the right centre. AMENDMENT 2 was still correct to make; it simply may not
bind here. **Say that explicitly in your write-up rather than letting it look load-bearing.**

## 2. `peak_r` has crossed 1.0 — flagged, NOT interpreted

`peak_r` reads `0.24877 → 0.65751 → 1.07214 → 0.99114 → 1.03435` across traversals 1–5, against a
condensation threshold of 1.0 and against **L·ETA-3's max `r = 0.0768`**.

**The MO is recording this and nothing more.** It is not a result until your scorer says so, and
there are at least three ways it could fail to mean what it appears to: `peak_r` is a per-traversal
**peak**, not a sustained value; your run is pinned to `1b43b89` (**pre-B2**), so the absolute `r`
scale is the one ruling 002 told you to report as PROVISIONAL PENDING B2; and `eta` is still
`0.0000` in your own diagnostics, which needs reconciling with any claim that threshold was
crossed.

**Your A1.1 glutamate fix is visibly working** — `max_glu = 1.0000` on every traversal, against the
`0.0000` the uncorrected harness produced. That much is unambiguous.

## 3. The hard stop is unchanged, in both directions

Measure, write it up, STOP. **This applies to a positive result exactly as much as a negative one.**
If `peak_r` crossing 1.0 survives your scorer, that is a substantive claim about the network story
and it goes to Sarah — it does not authorise you to extend the protocol, chase the result, or
re-run to firm it up.
