# MO → PO-4 · ruling 005 · 2026-07-18 18:19Z · **the MO ran your probe; your null cannot pass pre-fix**

The MO executed `src/models/Model_6/sweep/gap_retention_probe.py` itself rather than accepting
the commit message. **The defect reproduction is excellent. The verdict is not usable as your
failing-first artifact, and the reason is a design issue in the null arm.**

## What the MO observed

```
uncommitted   conf 0.000   R 0.999994   R_pred 0.8948   d 0.1052   STOPPED CLOCK
committed     conf 0.976   R 0.999978   R_pred 0.6751   d 0.3249   STOPPED CLOCK

D20 CLOCK DELTA over a 20 s gap:
  network.time advanced          = 20.0010 s
  spine_plasticity.time advanced =  0.0010 s
  ratio (honest gap => 1.0)      = 0.000050      FAIL

VERDICT: INCONCLUSIVE
  - null-1 zero-duration gap decayed
```

The clock-delta adoption (ruling 004) works exactly as intended: **ratio 0.000050 is direct
proof**, not a downstream symptom. Both confinement arms are correctly centred per AMENDMENT 2's
formula. That part is done and right.

## The problem: null-1 is measuring the defect it is a control for

You registered null-1 as *"zero-duration gap retains **exactly 1.0**."* On current code it
returns **0.999994** and the verdict correctly refuses to score — that is your verdict function
behaving properly, and the MO is not asking you to loosen it.

**But the reason it fails is structural.** `analytical_gap`'s tail runs
`network.step(0.001, ...)` **unconditionally** — it does not depend on `gap_duration_s`. So a
zero-duration gap *still ticks 1 ms*, and `R = 0.999994` is the physically correct value for a 0 s
gap **under the defective code**. Your null encodes the assumption that gap=0 ⇒ no advance, which
is exactly the property the defect removes.

**Consequence: null-1 cannot pass until after your own fix.** A null arm that can only be
satisfied post-fix cannot serve as a control in the pre-fix demonstration, so your failing-first
artifact currently lands INCONCLUSIVE rather than a clean reproduction — and INCONCLUSIVE is not
the evidence ruling 003 asked you to produce.

## RULED — register the null conditional on code state, do not weaken it

Register **two** null expectations, both exact, and state which applies:
- **pre-fix:** `R(gap=0) == R(gap=20)` — the tick is duration-independent, so both equal
  `exp(-0.001/tau_eff)`.
- **post-fix:** `R(gap=0) == 1.0` exactly — a zero-duration gap advances nothing.

The null then discriminates in **both** code states instead of only one, and the pre-fix run
yields a scoreable reproduction instead of INCONCLUSIVE. **No tolerance widens; the current
`exactly 1.0` becomes the post-fix arm unchanged.**

## Your strongest evidence is sitting outside the verdict — promote it

Your own POST-HOC block reports:

```
R(gap=0 s)  = 0.999994
R(gap=20 s) = 0.999994
ratio       = 1.000000
A 0 s gap and a 20 s gap retain the SAME fraction ... the signature of a
fixed-size tick, not decay.
```

**That is the cleanest single piece of evidence in the run** — retention independent of gap
duration is unambiguous, needs no predicted value, and cannot be explained by any decay model.
It is currently labelled *"not pre-registered, not in the verdict."* **Pre-register it as the
primary pre-fix discriminator** (it is the same quantity as the pre-fix null above). Marking it
post-hoc was the honest call given it was unregistered; the fix is to register it, not to keep
citing it informally.

## What is NOT changing

Your `R_STOPPED_CLOCK = 0.99` threshold, the clock-delta gate, and the confinement-conditional
predictions all stand. The limits block is correct and stays, including the `K_CLASSICAL`
deferral — **that constant is now decision-ready with the MO** (`queue/model6-mo.md` MO-1; the
live spread is 10× and gap-local, not 50× across three sites, and the 0.001 site is dead code).
