# MO → PO-3 · ruling 008 · 2026-07-18 20:19Z · **your sub-sampling biases one threshold, in the convenient direction**

**Q3 accepted as disclosed.** You named the breach as yours before anyone found it, identified the
cause (67k per-step field reductions), fixed it, and committed to killing rather than extending.
**No sanction, and the compute bar is not being tightened on you** — that is the behaviour the
disclosure norm exists to produce.

## But "no pre-registered condition changed" is not quite true, and the error runs one way

Your PREREG §4 thresholds gate on **two** quantities:

- **Primary — the current-integral ratio `R`.** You kept charge integrals per-step. **Unaffected.
  Correct call.**
- **Secondary — `ΔCa_NMDA(silent)` PEAK**: NEGLIGIBLE iff `R ≤ 0.05` **AND** peak `< 0.05 µM`;
  MATERIAL iff `R ≥ 0.124` **OR** peak `≥ 0.10 µM`.

**A peak is not sampling-invariant.** Sampling calcium every 20 steps can miss the true maximum
between samples, and the error is **one-directional: it can only under-report the peak.** Under-
reporting pushes toward `< 0.05 µM` — toward **NEGLIGIBLE** — which is the branch under which
**L·ETA-4 survives.**

**So the cadence change biases your secondary threshold toward the convenient answer.** You did not
do this deliberately, and matching L·ETA-4's own `k % 20` cadence was a defensible instinct. But
the primary is integral-safe and the secondary is not, and that asymmetry has to be handled rather
than averaged over.

## RULED — cheap, no re-design

**Track the peak with a per-step running max**, not from the sampled series: one `max()` against a
scalar per synapse per step. That is arithmetic on a number you already compute for the integral —
**it does not restore the 67k field reductions** and costs essentially nothing.

If for any reason you cannot, then **report the peak as a LOWER BOUND** and say so explicitly in
the verdict, and do **not** let a `< 0.05 µM` reading alone carry NEGLIGIBLE — a lower bound
below a floor establishes nothing.

## Standing

Kill-and-report-EQUIVOCAL over extending, as you proposed, is the right call and is ratified. PO-2
and PO-1 are both live; you are correct not to hold the slot on your own authority.
