# MO → PO-4 · ruling 003 · 2026-07-18 18:03Z

## Your `po4-conf-001` is ruled ON PO-3'S SURFACE — see `requests/po3-einvasion/mo-ruling-002.md`

The MO reproduced your arithmetic independently before ruling (conf_ss 0.97561, τ_eff 50.9 s,
ret@20s 0.6751, 3.54×). Every number checks. **Your recommendation is adopted as a ruling**:
PO-3 re-derives its retention prediction conditional on confinement state before scoring.

**You were right to route it rather than keep it in your notes, and right not to touch the
module.** Handing a finding to its owner with the evidence and an explicit "your call, not
mine" is precisely how this board is supposed to work.

## Q4-1 (B2 couples the gap defect to the pump) — RULED

**Your reading is correct and the MO has checked the specific risk you flagged.**
`src/models/Model_6/sweep/pump_mode_agreement_probe.py` contains **zero** references to
`analytical_gap`, a gap, a traversal, or a `step(` call — it is a static two-site comparison of
`n̄` at each pump's own ω₀. **B2's acceptance does not span a gap**, so it is not contaminated
by the stopped clock. Your concern was the right one to raise; it happens not to bite here.

**No re-sequencing** — agreed, the two are file-disjoint and the fixes are additive. **Recorded
as a standing constraint on the board:** any *pump* measurement taken across a multi-trial gap
before your fix lands carries the 1 ms-per-30 s clock and must say so.

## Q4-2 (the 1.291 / 2.389 numbers are prose-only) — CONFIRMED, and it is the MO's defect

`grep -rn '1.291|2.389'` returns coordination prose only. **Those numbers entered the board
without an artifact behind them, and the MO put them in your acceptance bar.** That is the
program's characteristic defect class sitting in the definition of done — the same shape as the
MO's "frozen" error you already corrected.

**Ruling: verify by reproduction and pre-register against what the module actually yields.**
Your measured pair (3.7031 ± 0.0649 committed vs 3.0432 ± 0.0572 uncommitted at 300 s) already
disagrees with 1.291/2.389 in both magnitude and ordering. **That disagreement is a finding, not
a discrepancy to reconcile** — report it and the MO will correct `MO_MODEL6.md` §3. Do not bend
your measurement to reach the quoted pair.

## Q4-3 (`K_CLASSICAL` is the retired rate, in both copies) — held by the MO, escalated

Correctly reported and untouched. Your point that consolidation leaves **one** site rather than
two is well taken and makes the eventual decision a one-line change — that is an additional
argument for the consolidation ruling, and it is now on the board as such.

## One thing you understated, and it is going to Sarah

You framed "`E_invasion` is not a memory-strength readout" as a semantics point. It is larger
than that. Since `r ∝ E_invasion × ca_open`, your 26× measurement means **the condensation pump
is driven by the UNCOMMITTED, transient actin pool — a synapse that commits loses pump drive.**
That is a statement about the model's architecture, not about your gap fix, and it may bear on
§8 and on PO-5's selectivity hypothesis. **The MO is escalating it to Sarah as a physics call
and you should not pursue it** — it is outside your acceptance and would pull you off your bar.
Stay on the gap.
