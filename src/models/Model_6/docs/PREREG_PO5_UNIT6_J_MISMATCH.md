# PRE-REGISTRATION — PO-5 UNIT 6 · does J-coupling mismatch, as a dissolution term, fragment the graph?

**Registered:** 2026-07-19, BEFORE the code change is written. **Authorised by Sarah.**
**This is a TIER-3 change: it alters the model's rules, not the instrumentation.**

## The claim being tested, stated as a physics claim

> Two entangled dimers whose **engaged J-coupling channels are detuned** lose the bond faster than
> two whose channels match.

**Grounding, stated honestly.** The functional shape has a real basis: resonant coherence transfer
between coupled spin systems degrades with detuning (Hartmann–Hahn matching in cross-polarisation
NMR is the canonical case). **What is NOT established** is that *intra*-dimer J-couplings are the
right quantity to gate *inter*-dimer, EM-mediated entanglement in this model. **That is an
assumption, and it is the thing this unit puts at risk.** It is labelled UNVERIFIED and is not to be
cited as grounded physics on the strength of this run.

## Implementation — off by default, so nothing standing is invalidated

Two new attributes on `DimerParticleSystem`, both defaulting to the current behaviour:

```
self.j_mismatch_dissolution = False   # master flag; False == today's physics exactly
self.j_mismatch_scramble    = False   # the control arm (see below)
self.j_mismatch_scale       = 0.15    # Hz
```

When the flag is on, the existing dissolution rate is multiplied by a detuning factor:

```
k_disentangle *= (1 + (Delta / j_mismatch_scale)**2)
Delta = | J_i[k(i,e)] - J_j[k(j,e)] |
```

`k(v,e)` is the channel selected by bond direction — the same rule as Unit 5, so the two units are
describing the same object.

**`j_mismatch_scale = 0.15` Hz is NOT tuned.** It is the standard deviation of the model's own
Agarwal-DFT J distribution (`dimer_particles.py:49`, `np.random.normal(0.15, 0.15, size=6)`), i.e.
the intrinsic spread of the quantity being compared. Sensitivity to it is reported, not hidden.

**Mandatory safety check, before any arm is scored:** with the flag OFF, a run must be
**bit-for-bit identical** to the pre-change code at the same seed (`n_dimers`, `n_bonds`, `mean P_S`).
If it is not, the edit is not off-by-default and the unit is INVALID.

## The three arms — and arm C exists because of PO-5's own conflict of interest

| arm | rule | purpose |
|---|---|---|
| **A — OFF** | current physics | control; the baseline everything standing was measured against |
| **B — REAL** | dissolution scaled by true J-mismatch | the hypothesis |
| **C — SCRAMBLED** | dissolution scaled by the **same Δ values randomly permuted across pairs** | **the decisive control** |

**Why C is the one that matters.** PO-5 proposed this mechanism *while predicting it would fragment
the graph* — the outcome PO-5 has been arguing the model needs. That is a bad position from which to
add physics. Arm C has the **identical dissolution magnitude and distribution** but no correspondence
between Δ and the actual pair. So:

- **B ≈ C** ⇒ the effect is *"PO-5 added dissolution"*, not *"J-structure organises the graph."*
  **The hypothesis is NOT supported**, whatever the component count does.
- **B ≠ C** ⇒ the pairing structure is doing work beyond the added dissolution rate.

## Pre-registered outcomes

| verdict | condition |
|---|---|
| **INVALID** | flag-off is not bit-identical to pre-change code |
| **SUPPORTED** | B fragments more than A **and** B differs from C beyond seed spread |
| **NOT SUPPORTED — mechanism is generic** | B fragments more than A but **B ≈ C** |
| **NOT SUPPORTED — no effect** | B ≈ A (the rule changes nothing) |

**"NOT SUPPORTED" is a real and expected-possible outcome and is reported as a finding, not
worked around.** If it lands there, the honest conclusion is that PO-5's proposed mechanism is
wrong and the flag should be deleted rather than kept and tuned.

**No threshold in this file moves after the run.** If the rule needs a different `j_mismatch_scale`
to produce an effect, that is reported as *"the effect requires a scale the data does not supply"* —
it is not re-registered at a friendlier value.

## Measured per arm

Ordinary components, cycle rank, bond count, saturation, sheaf H0_engaged (Unit 5's machinery,
already validated against the constant-sheaf known case), at bus values spanning Unit 4's range.

## Limits

Single synapse, 1 s, seeds stated in the result. This tests whether a *proposed* mechanism has a
*structural* effect; it does not establish that the mechanism is physically correct, and a
SUPPORTED verdict would raise the question, not settle it.
