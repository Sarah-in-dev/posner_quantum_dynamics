# Queue: po1-b2 — actions awaiting Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready,
so clearing it is seconds per item, not a re-derivation.

---

## Q1 — The pump's DRIVE changed, not just its threshold. Veto or confirm.

**The ask:** confirm (or veto) that the per-synapse pump is driven by per-spine metabolic
power `P_met` instead of `collective_field_kT`. This is the one part of B2 that goes beyond
"swap the threshold", and it changed 2 lines + 1 import in `model6_core.py`.

**Why it came up:** the acceptance text says the site must call `bose_einstein_occupation` on
the `n_ex = n̄_s` form — a *threshold*. But that threshold is expressed as a POWER
(`P_c = n̄_s·ℏ·(2π·ω₀)²/Q`), so computing `r = P/P_c` needs a power. `collective_field_kT` is
an energy in kT; converting it to a pump rate is exactly what `r_at_E_ref`/`kT_ref` did. So
keeping the old drive would have meant keeping the fiction under a new name.

**Why I proceeded rather than idling:** it reads as pinned, not fresh — `model6-network-layer-feasibility-may30`
line 49 specifies for B2 "per-synapse P_met, NO aggregation; same 8 MHz mode", and Step B
already made this exact call for the backbone ("Drive is metabolic P_met, NOT
`collective_field_kT`", `model6-architecture`). Same shape as the MO's Q1 ruling on
`_critical_threshold`. **But it is a drive-physics change, so it is yours to veto.**

**Recommendation:** CONFIRM. There is no honest kT→power conversion to keep, and the
alternative leaves the two pumps running different physics — which is the defect B2 exists
to remove.

**Cost if vetoed:** one commit reverts (`c280e85`, the `model6_core.py` hunk). The mode and
2π fixes are independent and would stand.

---

## Q2 — The old per-synapse η was a constant. Does that move a standing result?

**The ask:** decide whether any published/standing Model 6 result leaned on the per-synapse
condensate varying with drive.

**The evidence (measured, both modules loaded side by side, old pulled from git):**

| drive | OLD η | NEW η |
|---|---|---|
| 12.8 kT / r=0.39 | 0.2160 | 0.0000 |
| 22.1 kT / r=1.21 | 0.2185 | 0.0952 |
| 28.6 kT / r=2.38 | 0.2211 | 0.4086 |

The old η moved **1% across the entire drive range** — pinned by the Zhang steady-state
source term. `above_threshold` flipped at exactly 22.1 because 22.1 *was* `kT_ref`. So the
per-synapse condensate did no drive-dependent work: anything downstream that attributed
variation to it was reading a constant. Forward enhancement max also rises 1.78 → 4.40.

**Recommendation:** treat as a re-read task, not a regression. Per the MO's standing ruling I
did **not** damp it — damping would be tuning to protect a downstream result. Higher `k_agg`
⇒ more dimers ⇒ the O(n²) tracker gets slower, so runtimes will move.

---

## Q3 — Two documentation defects found in passing. Not my surface; whose?

**(a) `model6-architecture` overstates single-synapse subcriticality.** It states as a design
property that "a single synapse is subcritical by design… Do not 'fix' a subcritical single
synapse", citing "P_met maxes ~17 fW < P_c = 21.5 fW, r peaks 0.803". That figure reproduces
**exactly** at `E_invasion=0.495, ca_open=0.55` — the drive the actin envelope had reached by
30 s, i.e. a measurement of that run, not a ceiling. At sustained full invasion the same
arithmetic gives 51.24 fW, r=2.38 — above threshold with no aggregation at all. Aggregation
buys crossing at *lower per-spine drive*, not crossing at all.
**Recommendation:** correct the sentence; "subcritical by design" must not read as "cannot
cross". Logged as B2-3.

**(b) `test_learning_pathway.py` is not the "<1 min fast green check"** that
`model6-codebase-operations` line 146 claims, and this is **not** caused by B2 — measured on
an isolated pre-change copy: 220 s+ without clearing Phase 1. I did not use it as a
regression floor (the T1′ static probe is the floor, 7/7 in 7 s), but anyone treating it as a
quick check will be misled.
**Recommendation:** re-time it and update the skill, or name a different fast check.

---

## Q4 — FYI, no action: the per-synapse pump has never been sweepable at all.

`Model6Parameters` has no `cascade` attribute (one comment-only hit at
`model6_parameters.py:784`), so `VibrationalCascadeModule.__init__` always fell through to
`TubulinCascadeParameters()` defaults. **The entire per-synapse pump parameter set — not just
`kT_ref` — has never been reachable by `sweep_runner`.** No sweep ever run in this program
could have varied it. Still true after B2. Wiring `params.cascade` is unowned; it looks like
PO-6 territory. Raising, not claiming.
