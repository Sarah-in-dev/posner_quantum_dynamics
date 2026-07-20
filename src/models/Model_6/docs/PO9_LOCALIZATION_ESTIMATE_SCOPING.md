# PO-9 — scoping the Anderson-localization estimate that replaces the λ sweep (advisor point 1)

**2026-07-20. This is a SCOPING doc, not a result.** It sets up the calculation the advisor asked for
and names the inputs that are contested (and are the advisor's / a literature task to pin). It does
NOT assert a value.

## The question (restated to be answerable)
The cross-synapse channel is mediated by a **delocalized collective mode** (the acoustic Fröhlich
condensate on the microtubule lattice). The correct coupling form is **flat-then-cutoff**, not
exp(−d/λ): W(d) ≈ O(1) for d inside the mode's coherence domain, ≈ 0 outside. So the whole question of
whether a 15 µm cross-synapse bridge exists reduces to a **binary**:

  Is the mediating mode **delocalized** over ≥ 15 µm, or **Anderson-localized** to ξ_loc < 15 µm?

The comparison is **ξ_loc vs L_coh**, where L_coh = v·τ ≈ 214 µm (v = 1074 m/s, τ = Q/(2πω₀), Q=10) is
the lifetime-limited coherence length the program already uses. If ξ_loc ≫ 15 µm the channel exists
(our λ_F=214 cell is the physical case); if ξ_loc ≲ a few µm it does not exist (not "is weak").

## The calculation (1D, weak-disorder)
Phonons along a protofilament are effectively 1D. In 1D, all eigenstates are Anderson-localized for any
disorder > 0, but the localization length can be ≫ system size (effectively delocalized). The
weak-disorder localization length for a 1D chain with on-site (mass) disorder is

    ξ_loc(ω) ≈ 8 (v_g / ω)² / (a · ⟨(δm/m)²⟩)          [Thouless / weak-scattering, order-of-magnitude]

equivalently ξ_loc ≈ ℓ (the elastic mean free path) up to O(1), with ℓ set by Rayleigh scattering off
mass/spring disorder. The dimensionless control parameter is

    D ≡ (disorder strength) / (bandwidth) ,     ξ_loc / a  ~  D^(−2)   (weak disorder).

- a = tubulin monomer spacing ≈ 8 nm (longitudinal) → ξ_loc = 15 µm needs ξ_loc/a ≈ 1900, i.e. D ≈ 0.023.
- So the mode is delocalized past 15 µm iff the fractional disorder seen by the mode is **≲ ~2%**.

## The inputs that decide it (contested — advisor's / literature's to pin)
1. **Acoustic bandwidth** of the relevant Fröhlich/phonon branch (sets ω-dependence and D denominator).
   Needs the microtubule elastic dispersion for the mode carrying the condensate.
2. **Effective mechanical disorder** ⟨(δm/m)²⟩ (and spring-constant disorder) at 310 K: GTP-vs-GDP
   tubulin heterogeneity, isotopic/mass variance, lattice defects (B-lattice seams), and thermal
   softening. This is the number that decides D against the ~2% threshold above.

## Why the optical ~1 µm number does NOT port over (important)
The external lit pass found the OPTICAL channel (Trp superradiance) localizes at ~1 µm because
electronic/excitonic disorder (~200 cm⁻¹) is ~20× its critical value (~10 cm⁻¹). That is a **different
mode with different disorder**: the acoustic Fröhlich mode has its own bandwidth and its own (mechanical)
disorder, generally weaker relative to bandwidth than excitonic disorder is. **So "optical localizes at
1 µm" is NOT evidence the acoustic mode localizes at 1 µm.** Conflating them is the same category error
as reusing λ_met for λ_F. The acoustic estimate must be done on acoustic numbers.

## What this returns, and how it closes the loop
- If D ≲ 2% for the acoustic mode → ξ_loc ≳ 15 µm → **delocalized** → cross-channel exists → our SYNC
  λ_F=214 measurement is the physical case, and the readout carries the cross-synapse coincidence.
- If D ≫ 2% → ξ_loc ≪ 15 µm → **localized** → the cross-synapse channel does not exist at 15 µm, and the
  network-scale partition cannot carry sub-branch input identity (the decomposition null, now with a
  mechanism). The SYNC λ_F=214 cell would then be un-physical (an artifact of assuming delocalization).

This is a **calculation returning delocalized-or-not**, exactly as the advisor framed it — not a
parameter to sweep. It is the single load-bearing open question for the whole input-selectivity claim.

## Recommended next step
A focused literature/estimate task (candidate for `deep-research` or the advisor directly): (a) the
acoustic dispersion / bandwidth of the microtubule mode that carries the Fröhlich condensate; (b) the
effective 310 K mechanical disorder ⟨(δm/m)²⟩ + spring disorder; (c) plug into ξ_loc/a ~ D^(−2) and
compare to 15 µm / a ≈ 1900. Report delocalized-or-not with the disorder value that flips it.
