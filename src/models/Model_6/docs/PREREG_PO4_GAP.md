# PRE-REGISTRATION — PO-4 · the analytical gap · L·GAP-1

**Registered:** 2026-07-18, BEFORE the measurement is run and BEFORE any physics change.
**Author:** PO-4 (the analytical gap). **Method:** `MO_MODEL6.md` §2.4, `experiment-design-patterns`.
**Precedent followed:** `fa12009` (PO-1 committed its acceptance probe *failing* on current code
before touching physics) and `2084960` (PO-3 registered a *retention fraction*, not a direction,
so that a suspiciously perfect result reads as a red flag).

---

## 0. The trap this registration exists to defeat

**A stopped clock and a real memory effect both produce "the gap preserved state."** If I
register "state survives the gap" as my prediction, both outcomes confirm it and the
measurement cannot fail. That is the `683b82f` shape.

**The discriminator is therefore not survival but the RETENTION FRACTION**, registered as a
number in advance. A stopped clock and honest physics predict *different numbers*, and the
stopped clock's number is the suspiciously perfect one:

| | retention across a 20 s gap |
|---|---|
| current code (1 ms per gap) | **0.999994** |
| honest gap, never-committed spine | **0.8948** |
| honest gap, committed spine | **0.6751** |

**A result near 1.0 is a RED FLAG, not a success.** This inverts the naive reading, which is
the entire point of registering it in advance.

---

## 1. The discriminating quantity (fixed now, not after)

**PRIMARY — `R`, the `E_invasion` retention fraction across one gap:**

```
R = E_invasion(immediately after gap) / E_invasion(immediately before gap)
```

measured **in the full model** (`MultiSynapseNetwork`, via `analytical_gap`), reported
**separately per confinement arm**, with `self.confinement` logged per synapse per gap.

Registered predictions, derived from the code's own constants (`spine_plasticity_module.py:388-390`,
`:412`, `:109`, `:113-114`) and NOT tuned:

- `R_uncommitted(g) = exp(-g/180)` → **0.8948** at g=20 s, **0.8465** at 30 s, **0.7788** at 45 s
- `R_committed(g)  = exp(-(k_stab·conf + (1-conf)/180)·g)`, conf_ss = 0.9756
  → **0.6751** at g=20 s, **0.5546** at 30 s, **0.4131** at 45 s

**Why per-arm and not one number:** `E_invasion` reads `actin_enlargement` only, and commitment
redirects that pool from extrusion (τ=180 s) into stabilization (τ≈51 s). A single registered
number would be wrong for one of the two arms. See `requests/po3-einvasion/po4-conf-001.md`.

**Tolerance:** `|R_measured − R_predicted| ≤ 0.02` absolute. Rationale: the volume update carries
thermal noise (`_update_volume`, `thermal_fluctuation_amplitude`), but `E_invasion` is computed
from `actin_enlargement`, which is **not** noise-injected — so the tolerance covers Euler
integration error at `dt_sub`, not stochasticity. Measured spread across 5 isolated-module reps
was **±0.0000** on `E_invasion`, confirming the deterministic path.

**HEADLINE (the board's acceptance) — `ΔV`, committed-vs-uncommitted spine-volume separation
across an honest gap.** Registered as the *existence and sign* of a separation that the current
code cannot produce, **not** against a target value — see §4 on why the board's quoted numbers
are not usable as a target.

---

## 1b. AMENDMENT A — 2026-07-18 18:25Z, on MO ruling 005, BEFORE the post-fix run

**Registered before the fix exists, and it does not weaken any tolerance.** The MO ran the probe
itself and found a design fault in the null arm, not in the reproduction.

**The fault:** §2 null 1 registered *"zero-duration gap ⇒ R exactly 1.0."* That encodes the
assumption **gap=0 ⇒ no advance — which is precisely the property the defect removes.**
`analytical_gap`'s tail runs `network.step(0.001, ...)` unconditionally, so under the defective
code a 0 s gap still ticks 1 ms and `R = 0.999994` is the *correct* value for that code. A null
that can only be satisfied after my own fix cannot serve as a control in the pre-fix
demonstration — it forces INCONCLUSIVE where a clean reproduction was wanted.

**AMENDED — the null is now registered conditional on code state, both branches exact:**

- **PRE-FIX:** `R(gap=0) == R(gap=20)`, both `= exp(-0.001/tau_eff)`. The tick is
  duration-independent, so the two must be equal.
- **POST-FIX:** `R(gap=0) == 1.0` exactly. A zero-duration gap advances nothing.
  *(This is §2 null 1 unchanged, now scoped to the post-fix arm.)*

**PROMOTED — duration-independence becomes the PRIMARY PRE-FIX DISCRIMINATOR.** In the first run
this quantity sat outside the verdict labelled post-hoc:

```
R(gap=0 s) = 0.999994 ;  R(gap=20 s) = 0.999994 ;  ratio = 1.000000
```

**Registering it rather than continuing to cite it informally.** It needs no predicted value and
no decay model can produce it: retention independent of gap duration is the signature of a
fixed-size tick. Registered as `ratio == 1.0` to within 1e-5 pre-fix, and **strictly < 1.0**
post-fix (a 20 s gap must retain strictly less than a 0 s gap once the clock runs).

**Unchanged by this amendment:** `R_STOPPED_CLOCK = 0.99`, the clock-delta gate, the
confinement-conditional predictions, the ±0.02 tolerance, and the §6 limits including the
`K_CLASSICAL` deferral.

**Note on provenance:** ruling 005 refers to "AMENDMENT 2's formula" for the confinement arms.
No AMENDMENT 2 exists in this document — this is AMENDMENT A, the first. The confinement formula
is §1's, authored here. Recorded because a citation to an artifact that does not exist is the
same defect class this PO was dispatched to fix, and it should be caught in both directions.

## 1c. AMENDMENT B — 2026-07-18 18:55Z · **the registered prediction was WRONG, and the verdict caught it**

**The post-fix run returned FALSIFIED against §1's formula. That verdict stands as recorded.**
The failure is in **my registered prediction**, not in the physics or the fix — and the
distinction has to be earned, not asserted, because "the physics is fine, my formula was wrong"
is exactly what someone says when tuning to rescue a result.

**The error.** §1 registered `R = exp(-rate·g)` — the decay factor of `actin_enlargement`. But
`E_invasion` is **affine** in enlargement, not proportional (`spine_plasticity_module.py:412`):

```python
self.E_invasion = np.clip((self.actin_enlargement - a.invasion_threshold) /
                          max(1e-6, a.E_ref - a.invasion_threshold), 0.0, 1.0)
```

With a threshold offset, `R_E = (E0·f − thr)/(E0 − thr)`, **not** `f`. I registered the decay of
the wrong variable.

| arm | registered §1 | corrected | measured |
|---|---|---|---|
| uncommitted | 0.8948 | **0.8832** | **0.882920** (Δ 0.00028) |
| committed | 0.6751 | **0.6390** | **0.636079** (Δ 0.0029) |

**Why this is a derivation fix and not a curve fit:** the corrected expression is read off line
412 — an algebraic consequence of code I can point at, with no free parameter and nothing fitted
to the observed values. But that argument is available to anyone rationalising a miss, so it is
**not accepted on its own.**

### REGISTERED NOW, BEFORE RUNNING: the out-of-sample test

The corrected formula has seen `g = 20 s`. It is therefore scored on durations it has **never
been evaluated at**, registered here before execution:

- `g = 30 s`: uncommitted **0.8290**, committed **0.4779**
- `g = 45 s`: uncommitted **0.7542**, committed **0.3009**

Same tolerance, unchanged: **|R_measured − R_predicted| ≤ 0.02**, both arms, both durations.

**If any of those four misses, the corrected formula is wrong too** and the result is FALSIFIED
again — no third formula will be registered. Four out-of-sample points against a zero-parameter
expression is a real test; passing at 20 s alone would not have been.

**Recorded plainly:** the 20 s result is now IN-SAMPLE for the corrected formula and is reported
as such. The acceptance rests on the out-of-sample points.

## 2. The null that cannot show the effect

Three, and all three must behave as registered or the run is INCONCLUSIVE:

1. **Zero-duration gap.** `analytical_gap(network, 0.0)` ⇒ `R = 1.000` exactly, both arms.
   A null that structurally cannot decay. If this shows decay, the harness is wrong.
2. **Pre-gap `E_invasion = 0`.** A synapse never driven above `invasion_threshold = 0.1` has
   nothing to retain; `R` is undefined and that synapse is **excluded from the primary**, declared
   now rather than after seeing the data.
3. **Volume-null.** Both arms uncommitted ⇒ `ΔV` must be ≈ 0 (within thermal noise, ±0.07 from
   the 5-rep isolated-module spread) **even on the honest gap**. If uncommitted arms separate,
   the separation is not commitment-driven and the headline is FALSIFIED.

## 3. The positive control that MUST fire (the L·ETA-4 scar)

**`_camkii_committed == True` in the committed arm, asserted and printed, before `R` is
interpreted.** The L·ETA-4 probe printed "selectivity holds" while its own positive control
never fired. If commitment does not fire, the committed arm does not exist and the verdict is
**INCONCLUSIVE — POSITIVE CONTROL DEAD**, never CONFIRMED and never FALSIFIED.

Second control: **`confinement > 0.5` in the committed arm.** Commitment without confinement
would leave both arms on the same 180 s branch and the two registered numbers would collapse
into one, silently.

## 4. Why the board's own target numbers are NOT the registered target

`MO_MODEL6.md:140` states the isolated-module numbers are **"1.291 vs 2.389 at +300 s."**
`grep -rn '1.291\|2.389'` over the repo returns **two hits, both coordination prose** — no code,
no results artifact, no log entry produces them. **They are unsourced.**

My reproduction (isolated module, dt=0.005, 300 s, 5 reps) does not recover them:

| arm | spine_volume @300 s | E_invasion @300 s |
|---|---|---|
| committed (drive=1.0, Ca=1.0 µM) | **3.7031 ± 0.0649** | 0.0313 |
| uncommitted (drive=0.0, Ca=1.0 µM) | **3.0432 ± 0.0572** | 0.8222 |
| uncommitted (drive=0.0, Ca=0.1 µM) | **0.9609 ± 0.0181** | 0.0000 |

I did **not** search for parameters that reproduce 1.291/2.389 — doing so would be tuning to a
target, which `MO_MODEL6.md` §7 LOCKED forbids. **Registered instead:** the headline is scored on
whether a separation *exists with the registered sign* on an honest gap and *does not exist* on
the current code. The unsourced pair is reported to the MO as a finding (`queue/po4-gap.md` Q4-2)
and is not used as an acceptance number.

## 5. The verdict function — it can return FALSIFIED and INCONCLUSIVE

```
INCONCLUSIVE if:
    - positive control dead (_camkii_committed never True in the committed arm), OR
    - confinement <= 0.5 in the committed arm, OR
    - zero-duration null shows R != 1.000, OR
    - fewer than 2 synapses clear the E_invasion > invasion_threshold entry bar

FALSIFIED if:
    - |R_measured - R_predicted| > 0.02 in EITHER arm on the honest gap, OR
    - uncommitted arms separate in volume beyond thermal noise (null 3 fires), OR
    - the honest gap yields R >= 0.99 (i.e. the fix did not actually start the clock)

CONFIRMED only if ALL hold:
    - current code reproduces R >= 0.999 (the stopped clock, demonstrated FIRST), AND
    - honest gap reproduces R within 0.02 of prediction in BOTH arms, AND
    - all three nulls behave as registered, AND
    - both positive controls fire, AND
    - committed and uncommitted spine volume separate with the registered sign
```

**Note the first CONFIRMED clause.** The verdict is not reachable without first demonstrating
the defect on unmodified code. A passing run alone cannot satisfy this registration.

## 6. Registered scope limits (stated now, so they are not "limits discovered after")

- `dt_sub` for the plasticity advance is **not** asserted correct. Per MO ruling 2, a
  dt-convergence check runs against the existing 5 s full-physics validator
  (`run_theta_burst_45s.py:405-415`). DECISION RECORD `dt-1` covers `P_S`/edges, **not**
  transient-phase counts; it is not assumed to transfer.
- `K_CLASSICAL = 0.05` (the retired rate) remains live in the gap during this measurement.
  MO-held. **Every dissolution number this measurement produces inherits it** and must be read
  that way. Stated in advance, not as a caveat afterwards.
- This measures the **gap**, not the drive. Nothing here validates the physics *during* a
  traversal.
- Two synapses, one network. No claim about scaling.
