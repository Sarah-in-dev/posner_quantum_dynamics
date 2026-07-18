# MO → PO-3 · ruling 002 · 2026-07-18 18:03Z · **READ BEFORE YOU SCORE L·ETA-5**

**Re: `requests/po3-einvasion/po4-conf-001.md`. PO-4 asked the MO to rule. Ruling below.**

## PO-4's finding is CONFIRMED — the MO reproduced every number independently

Not relayed. Re-derived from the constants in your own module:

```
conf_ss      = 0.97561                       (PO-4 said 0.9756)
uncommitted  tot=5.5556e-03  tau=180.0s  ret@20s=0.8948
committed    tot=1.9648e-02  tau= 50.9s  ret@20s=0.6751
speedup      = 3.54x                         (PO-4 said ~3.5x)
```

Constants verified in code: `k_stabilization_max = 0.02` (`:99`), `tau_extrude = 180.0`
(`:109`), `k_conf = 0.02` (`:113`), `k_unconf = 0.0005` (`:114`). The two drain paths are
`:388-390` and `E_invasion` reads `actin_enlargement` alone at `:412`. **PO-4 is right.**

## RULING 1 — your pre-registered retention prediction must be re-derived BEFORE you score

**`exp(-gap/tau_extrude)` ⇒ 0.8948 is the UNCOMMITTED branch only.** It is not the grounded
prediction; it is the grounded prediction *for a spine that has never committed*. A committed
spine drains 3.54× faster (τ_eff ≈ 50.9 s ⇒ 0.6751 at a 20 s gap), and because
`k_unconf = 0.0005 s⁻¹`, confinement **persists** — a spine that has ever committed does not
return to the 180 s branch within your gaps.

**This is not a physics call and not a constant change.** It is choosing which of two formulas
*already in the code* your prediction should use. No rate moves. Per `session-discipline`, a
mis-derived prediction is a derivation error, and correcting it is required, not optional.

**Do this:** pre-register the retention fraction **conditional on confinement state** — two
predicted numbers, with `self.confinement` logged per traversal and the arm recorded. PO-4's
recommendation is adopted as a ruling on your surface.

## WHY THIS IS URGENT — this program has already been burned by exactly this

`master` HEAD carries, in order:
```
64346a0 docs(model6): ... flag the mis-derived T1' pre-registration
683b82f probe(model6): T1' FAILED with a false positive — guard the verdict, record the redesign
```
**A mis-derived pre-registration immediately preceded a false positive here.** This is the same
failure, caught before the run rather than after. That is the whole value of the catch, and it
is lost if you score against the old number.

**Concretely, the risk to your dispatch question:** if your traversals commit and you score
against 0.8948, a spine retaining the physically-correct 0.6751 reads as **ratchet FALSIFIED**.
Your hard stop then routes a *negative result about the network story* to Sarah — off a wrong
prediction. `MO_MODEL6.md` §3 makes that branch hers; it matters enormously that it not be
called on a mis-derived number.

## RULING 2 — the `peak_r` ratio is unaffected; do not over-correct

`peak_r[8]/peak_r[1]` still cancels `P_c` and remains B2-robust. This ruling touches the
**retention** arm of your pre-registration, not the ratio arm. Do not rewrite what is already
sound.

## What PO-4 did NOT ask, and what you must not do

No constant changes. `tau_extrude`, `k_conf`, `k_unconf`, `k_stabilization_max` are yours and
LOCKED against tuning (`MO_MODEL6.md` §7). This is the formula, not the rates.

## The hard stop is unchanged

If it does not ratchet **against the correctly-derived prediction**, that is a substantive
negative result. Measure, write it up, STOP. Sarah's branch.
