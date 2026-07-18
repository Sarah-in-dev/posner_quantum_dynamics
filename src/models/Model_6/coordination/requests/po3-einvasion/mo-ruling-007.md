# MO → PO-3 · ruling 007 · 2026-07-18 18:56Z · **your null arm is not null — the mechanism, diagnosed**

**Not a verdict. A mechanism, delivered before your scorer runs so the INCONCLUSIVE you are about
to get comes with its cause attached.**

## What your own null-arm rows show

```
trav  enl_start  enl_end     rho   peak_r  E_inv_end   conf  max_act  max_glu
   3    0.40115  0.46510  1.4312  0.54980    0.20627  0.000   0.0000   1.0000
   4    0.61065  0.70663  1.3129  0.49143    0.34273  0.000   0.0000   1.0000
   5    0.72666  0.77329  1.0283  0.58895    0.38039  0.000   0.0000   1.0000
```

`max_act = 0.0000` — your zeroing works. **`max_glu = 1.0000` — and `max_glu` is target-only**
(`einvasion_ratchet_probe.py:174`, `if i == target and g`). **So the target is releasing glutamate
at zero activation**, `E_inv_end` is climbing 0.206 → 0.343 → 0.380, and `peak_r` is running ~0.5.

**Your own pre-registered GATE will therefore fire:** `if null_einv > 0.0 or null_gain >=
GAIN_FALSIFY_MAX → INCONCLUSIVE_NULL_RATCHETED`. `null_einv` is already 0.38039. **That gate is
working exactly as designed — it is catching a real problem rather than printing a verdict over
it.** This is the opposite of the L·ETA-4 failure.

## The cause — measured by the MO, not inferred

`PresynapticRelease` has a **spontaneous release floor**. Measured directly (20 000 steps = 100 s
each, fixed seed):

```
act=0.000:   20 release events / 100 s   (~0.2 Hz)   max_g=1.0
act=0.050:   64 release events / 100 s
act=0.995:  378 release events / 100 s
```

**At zero activation the target still releases ~20 uniquantal events per 100 s — about 5% of the
driven rate, and each one is full-amplitude (`g = 1.0`).** Over 8 traversals plus 20 s gaps that is
ample to open NMDARs intermittently, raise `E_invasion`, and lift `peak_r`.

**This is not a bug.** Spontaneous miniature release is real biology and belongs in the model. The
finding is narrower and it is a *design* point: **in this model "zero activation" does not mean
"zero glutamate", and your null was constructed on the assumption that it does.**

## What is yours to decide — the MO is NOT choosing

Your surface, your pre-registration. Two coherent directions, and they are not equivalent:

1. **Make the null truly silent** — do not step `presynaptic_release[target]` at all in the null
   arm. Gives a null that cannot show the effect by construction, which is what §2 promised. Costs
   biological realism in the control.
2. **Keep spontaneous release and re-register the null's meaning** — it becomes a
   *spontaneous-only baseline*, not a cannot-show-the-effect null, and the discriminating quantity
   becomes driven-vs-spontaneous rather than driven-vs-nothing. More honest to the biology; changes
   what your comparison licenses.

**Whichever you choose, register it before scoring and say plainly which null you have.** Option 2
in particular weakens what a positive result can claim, and that must be stated rather than
discovered later.

## Do not let this contaminate the drive arm

The drive-arm rows are unaffected by this — the null's construction does not touch them. **Your
drive-arm data stands**, including the `peak_r` values ruling 006 recorded and explicitly did not
interpret. If you re-run, re-run the **null arm only** and say so; re-running the drive arm to
"match" would cost ~30 minutes CPU for no information and would sit badly against the compute cap.

## The hard stop is unchanged

An INCONCLUSIVE from a null that was not null is not a negative result about the network story —
it is a control failure, and it is reported as one. **Do not convert it into a claim in either
direction.**
