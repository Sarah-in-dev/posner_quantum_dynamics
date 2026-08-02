# PRE-REGISTRATION — F1: wire the emergent ⁶Li/⁷Li isotope lever into the model

**Written 2026-07-30, BEFORE the model change.** Reorder from `ISO-1`: the P31/P32 lever was circular
(hardcoded `T_singlet` 216↔0.4). This wires the DERIVED replacement (`nuclear_relaxation.py`, committed
`5a0d5a9`, validated). ³¹P stays the qubit; Li is a dopant (Li⁺ for Ca²⁺) that decoheres ³¹P via scalar
relaxation of the 2nd kind (Abragam; arXiv 2310.13484 eq. 9). Emergent-only: the ⁶Li/⁷Li contrast is
parameter-free from tabulated γ/I/Q; only the absolute scale rides on one J calibration.

## LOCKED acceptance (the emergent discipline — fixed before wiring)
1. **Reproduce the literature** (already PASSING, `nuclear_relaxation.py` `__main__`): ω = 5416/5199/1970 rad/s;
   scalar ⁶Li/⁷Li ratio log₁₀ = 5.0 ("five orders"); ⁷Li T2_scalar ≈ 15 s ("seconds"); ⁶Li ≈ 17 days.
2. **Undoped is BIT-IDENTICAL** to the pre-change default: `dopant=None ⇒ T2_observed(None,216)=216 s`, exactly
   the old `fraction_P31=1.0` value. A regression fingerprint (mean_P_S at fixed seed/steps) must match the
   pre-change baseline for the undoped case, or the wiring changed unintended behaviour.
3. **⁷Li is a real, gentle kill:** observable T2 ≈ 14 s (vs undoped 216 s) — a ~15× reduction over the
   coherence window, NOT the old instant 0.4 s. This is the physically-derived magnitude and is NOT to be
   made more aggressive to match the old kill (that would be tuning-to-result).

## The calibrated inputs (declared, cited, NOT tuned to a downstream result)
- `J_LI7 = 18 Hz` — the single absolute-scale input; the paper's anchor ("~2 orders above" Swift's 0.178 Hz
  P–P coupling). Calibrated so ⁷Li → "seconds" (the cited target), NOT to make any learning experiment pass.
- `τ_Li` (⁶Li 300 s / ⁷Li 10 s), `B0 = 50 µT` — [ASSUMED-in-source] from Fisher/2310.13484; direction grounded
  by |Q(⁷Li)/Q(⁶Li)| ≈ 50. `T_intrinsic = 216 s` — the kept Agarwal-band dimer number (undoped coherence).
- **If any of these is later changed to move a model RESULT, that is the emergent-physics violation (§6.1).**

## The wiring (surgical)
- Add `environment.dopant ∈ {None,'Li6','Li7'}` (default None).
- In `dimer_particles.py` and `quantum_coherence.py`, replace the `fraction_P31·216 + (1−fraction_P31)·0.4`
  blend with: `dopant is not None → T2_observed(dopant, t_intrinsic=T_singlet_dimer)`; else the old
  `fraction_P31` blend (backward-compat; undoped bit-identical; P32 path retained-but-DEPRECATED per ISO-1).
- ³¹P qubit dynamics, the partition, the measurement: UNCHANGED.

## Experiment consequence (pre-registered, for the eventual re-run)
The derived kill is gentle (14 s vs 216 s), so the ⁶Li/⁷Li contrast lives on the COHERENCE WINDOW
(cross-trial persistence / delayed readout), NOT immediate partition formation. Any isotope discrimination
experiment must span the window (delayed/multi-trial), not the old `edges=[]`-on-contact check. Designing
that is F1's follow-on, separate from this wiring.

## Verdict on the wiring itself
PASS iff: acceptance-1 still passes post-wire; undoped fingerprint bit-identical (acceptance-2); ⁷Li gives
≈14 s in the live model (acceptance-3); the model imports and runs a short probe without error. Otherwise
revert and report.
