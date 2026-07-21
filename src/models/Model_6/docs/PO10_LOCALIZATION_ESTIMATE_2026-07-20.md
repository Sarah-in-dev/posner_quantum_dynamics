# PO-10 — the localization estimate: does the cross-synapse mediating mode reach 15 µm?

**2026-07-20 (advisor-reviewed).** This is the RESULT the scoping doc (`PO9_LOCALIZATION_ESTIMATE_SCOPING.md`)
set up. It removes the single load-bearing conditional on the `L·PO9-2` graded-computation keystone: is the
collective mode that mediates cross-synapse entanglement **delocalized** over the ~15 µm between synapse
clusters, or confined to a shorter length?

No simulation was run and no code was changed. Every physical number carries an epistemic tag
(`[PROVEN]` literature/algebra · `[GROUNDED]` tied to a named measurement or our own committed constant ·
`[MODELED]` defensible choice · `[INFERRED]` follows from the model · `[CONTESTED]` unsettled bet · `[LOCKED]`
settled) and a real source. A number with no source is not permitted here.

**The system this is about (three layers — canonical §2, §4; `MICROTUBULERESEARCH.md`):**
1. **The condensate BUS** — the shared dendritic microtubule backbone condenses (Fröhlich, metabolism-driven)
   into the collective ~8 MHz mode; that mode is the bus. `L_coh = v·τ ≈ 214 µm` is its coherence length.
2. **MT network + nodes** — each synapse couples to the bus by transient, Ca-gated MT invasion into the spine;
   the invaded MT + Q1 superradiance field electromagnetically communicate with that spine's dimer network.
3. **The dimer computational program** — the Ca₆(PO₄)₄ ³¹P-spin qubits form the entanglement graph; the
   computation is its Werner-gated partition.

Cross-cluster binding is **dimer-net A ↔ (invasion+EM) ↔ bus ↔ (invasion+EM) ↔ dimer-net B** — both nodes tap
the *same* coherent bus domain. It is **not** a phonon launched from A and received at B. "Delocalized over
15 µm" therefore means: **does the shared condensate bus form one coherent domain spanning both clusters?**

---

## VERDICT

> **DELOCALIZED over 15 µm.** The bus spans the gap on **both** possible localizers, each checked:
>
> - **Disorder (§3–5):** mass/spring disorder cannot Anderson-localize the 8 MHz bus below 15 µm —
>   ξ_loc(ω₀) ≫ L_coh ≈ 214 µm > 15 µm by ≥ 5 orders of magnitude, for *any* attainable disorder.
> - **Structural continuity (§6):** individual dendritic MTs are only a few µm, but the bus lives on the
>   **staggered bundle** (5–15 MTs/cross-section), where a single MT terminus is a ~1/N ≈ 10–20% local
>   stiffness dip — *weak disorder, not a wall* — so it too reduces to the delocalized case. The only true
>   wall is a **backbone discontinuity (branch point / segment end)**, and the continuous unbranched segment
>   (tens of µm) exceeds 15 µm.
>
> **Binding constraint = the Q = 10 mode lifetime (214 µm) and backbone segment length (tens of µm), NOT
> disorder and NOT individual MT length.** The `L·PO9-2` measurement (SYNC, λ_F = 214) is therefore the
> physical case, and the graded readout computation stands as a computation, not a precondition.
>
> **Mode identity — CONFIRMED by the advisor, mechanistically (§7), not merely assumed.** The mediating mode
> is the band-bottom (~8 MHz) mode *by construction*: Fröhlich condensation funnels quanta to the lowest mode,
> so "the condensing mode" and "the lowest mode" are the same statement. A high-frequency (GHz) mediator is
> structurally impossible for a condensate and would also collapse L_coh. **Verdict locked; proceed to Unit C.**
>
> **Testable prediction:** cross-cluster binding requires both clusters on the **same unbranched backbone
> segment** — robust to individual MT turnover (a bundle property), but failing across a branch point.

---

## 1. The inputs, pinned

| symbol | value | tag | source |
|---|---|---|---|
| operating mode freq. f₀ | 8 MHz (ω₀ = 2πf₀ = 5.03×10⁷ rad/s) | `[GROUNDED]` | our code `model6_parameters.py:908` `omega_0 = 8.0e6`, comment "ω₀/2π = 8 MHz"; canonical §4.3 "collective MHz microtubule mode" |
| quality factor Q | 10 | `[GROUNDED]`/`[CONTESTED]` as a physical bet | our code `model6_parameters.py:909`; canonical §4.3, §6.1 (the AMRIS-class bet) |
| longitudinal sound speed v | 1074 m/s | `[GROUNDED]` | program-committed (`scoping:16`); corroborated below |
| — corroboration: E, ρ | E ≈ 1.2–1.9 GPa, ρ ≈ 1.4 g/cm³ ⇒ v = √(E/ρ) ≈ 930–1165 m/s | `[GROUNDED]` | Deriu 2010; MD of complete MT (Sept–MacKintosh 2010) — Sources [1,2] |
| tubulin monomer spacing a | 8 nm | `[PROVEN]` | tubulin 8 nm dimer repeat; `scoping:31` |
| temperature T | 310.15 K | `[GROUNDED]` | our code `model6_parameters.py:926` |
| MTs per backbone cross-section N | 5–15 | `[GROUNDED]` | Harris 2022 (spine density ∝ MT number); our model default 5 — Sources [5] |
| individual dendritic MT length | ~1 to >100 µm; "generally do not traverse the whole process" | `[GROUNDED]` | Baas; dendritic MT morphology — Sources [10] |
| unbranched dendritic segment length | order tens of µm (≳ 15 µm) | `[GROUNDED, order-of-magnitude]` | CA1 pyramidal morphometry — Sources [11]; precise per-compartment mean in their supp. tables |
| mass disorder δm/m (GTP/GDP) | 0.07–0.15% | `[PROVEN]` | one ~80 Da phosphate lost on GTP→GDP at β E-site, over ~110 kDa dimer / ~55 kDa monomer — Sources [4,5b] |
| stiffness disorder δK/K (GTP↔GDP) | up to ~O(1) (≈2× modulus, 2.3% compaction); effective variance unpinned | `[GROUNDED]` contrast / `[CONTESTED]` effective σ | Alushin 2014; nucleotide-stiffness — Sources [6,7] |
| ξ(ω) ∝ ω⁻² in 1D (low-f, weak disorder) | theorem | `[PROVEN]` | Monthus–Garel PRB 2010; Ishii 1973 — Sources [3,8] |

**Reconciliation note (repo rule: flag, don't silently substitute).** External MT-vibration literature reports
phonon speeds as low as **200–600 m/s** for some branches (Sources [9]). Those are the softer
radial/torsional/helical modes; the **longitudinal** (protofilament-axis) speed relevant to axial propagation
is v = √(E/ρ) ≈ 930–1165 m/s, which is what the program's 1074 m/s represents. Kept at 1074; §5 shows the
verdict survives even at 300 m/s.

---

## 2. Derived lengths (arithmetic)

From f₀ = 8 MHz, v = 1074 m/s, Q = 10, ω₀ = 2πf₀ = 5.027×10⁷ rad/s.

- **Wavelength:** λ = v/f₀ = 1074 / 8×10⁶ = **1.343×10⁻⁴ m = 134 µm**   `[INFERRED]`
- **Reduced wavelength / wavevector:** k = ω₀/v = 4.68×10⁴ m⁻¹ ; 1/k = v/ω₀ = **21.4 µm**   `[INFERRED]`
- **Lifetime-limited coherence length:** τ = Q/ω₀ = 1.99×10⁻⁷ s (199 ns); L_coh = v·τ = **214 µm**
  `[GROUNDED]` — reproduces the committed value. (Advisor independently verified λ = 134 µm ⇒ v = 1072 m/s,
  consistent with 1074; and n̄ = 8.07×10⁵ reproduces the pinned n̄_s = 8.074×10⁵, confirming the 2π/`hf`
  convention at the site in use.)

**Immediate observation.** The mode's own wavelength (134 µm) is **~9× the 15 µm cluster gap**, and even its
reduced wavelength 1/k = 21.4 µm exceeds 15 µm. A wave cannot be confined below ~its own reduced wavelength —
the **Ioffe–Regel floor**: ξ_min ≈ 1/k ≈ **21.4 µm > 15 µm**, reached only in the strongest-scattering limit.
So even *maximal* disorder cannot localize this mode below the cluster gap. `[PROVEN]` (Ioffe–Regel 1960.)

---

## 3. The localization length, at the operating frequency

Scoping doc's weak-scattering formula (`scoping:24`), v_g = v at long wavelength (linear acoustic branch ~5
decades below the band edge):

    ξ_loc(ω) ≈ 8 (v/ω)² / (a · σ²),      σ² ≡ effective fractional mechanical disorder variance

The prefactor "8" is convention-dependent (O(1)) — irrelevant at these margins. Plugging in:

    (v/ω₀)² = (2.137×10⁻⁵ m)² = 4.567×10⁻¹⁰ m²
    ξ_loc(ω₀) = 8 × 4.567×10⁻¹⁰ / (8×10⁻⁹ · σ²) = **0.457 / σ² metres**      `[INFERRED]`

| disorder channel | σ | σ² | ξ_loc(ω₀) = 0.457/σ² | vs 15 µm | tag |
|---|---|---|---|---|---|
| GTP/GDP mass (dimer basis) | 7.3×10⁻⁴ | 5.3×10⁻⁷ | **860 km** | ×5.7×10¹⁰ | `[PROVEN]` |
| GTP/GDP mass (β-monomer basis) | 1.5×10⁻³ | 2.25×10⁻⁶ | **203 km** | ×1.4×10¹⁰ | `[PROVEN]` |
| stiffness contrast, generous | 0.30 | 0.09 | **5.1 m** | ×3.4×10⁵ | `[CONTESTED]` |
| stiffness contrast, extreme (σ→1) | 1.0 | 1.0 | **0.46 m** | ×3.0×10⁴ | `[CONTESTED]` |

**Disorder required to localize at exactly 15 µm:** σ² = 0.457 / 1.5×10⁻⁵ = 3.05×10⁴ ⇒ **σ ≈ 175 (17,500%)** —
impossible (σ ≤ 1; even σ = 1 gives 46 cm). `[INFERRED]` So no attainable disorder localizes the bus below
15 µm; the mode is **lifetime-limited (214 µm), not disorder-limited**, and 214 µm > 15 µm.

---

## 4. Reconciling with the scoping doc's "≲ 2%" threshold

The scoping's "ξ_loc/a ~ D⁻², delocalized iff D ≲ 2.3%" is the localization length of a **generic
zone-boundary mode** (k ~ π/a, wavelength ~2a): ξ/a ~ 1/σ² ⇒ 15 µm needs σ ~ √(a/15 µm) = √(5.3×10⁻⁴) = 2.3%.
Internally correct **for a band-top mode.** But our mediating mode sits near the **bottom** of the MT
vibrational band (~0.1 MHz–GHz–THz; Sources [9,10]), and ξ(ω) ∝ ω⁻² in 1D `[PROVEN, Sources 3,8]`, so at
8 MHz the localization length is larger by

    (ω_band / ω₀)²  ~  (1–20 GHz / 8 MHz)²  ~  10⁵ – 10⁷        `[INFERRED]`

That factor is the entire discrepancy between the 2.3% threshold and §3. The honest refinement of the
scoping's line: *delocalized past 15 µm iff effective disorder ≲ ~10⁴* — i.e. always. This is the standard
Goldstone/acoustic protection that keeps long-wavelength sound extended despite defects; a theorem, not a
coincidence.

---

## 5. Robustness to the two things we could not pin (disorder branch)

**(a) Sound speed.** At the lowest literature value v = 300 m/s (Sources [9]): 1/k = 5.97 µm (Ioffe–Regel floor
now below 15 µm), but realistic-disorder ξ = 0.0356/σ² is still enormous — σ = 1.5×10⁻³ → **15.8 km**; extreme
σ = 0.3 → **0.40 m**. L_coh(300) = 60 µm > 15 µm. **Verdict unchanged.** `[INFERRED]`

**(b) Effective 310 K disorder.** Largest channel is GTP↔GDP spring-constant contrast (≈2× modulus). But a
mature lattice is nearly uniformly GDP (low variance — disorder needs a *mixture*), and the effective variance
is bounded well below the σ = 0.3–1.0 rows, all of which already give ξ ≫ 15 µm. The exact 310 K σ needs an MD
estimate of GTP/GDP mixing fraction + per-site spring variance — **but it is not load-bearing** (verdict
survives σ = 1). `[CONTESTED]` input, `[INFERRED]` conclusion.

**Dimensionality.** The 1D chain is a simplification (MT = 13-protofilament quasi-1D tube); higher-D
localization is only *weaker* (2D exponentially long ξ; 3D mobility edge). Strengthens the verdict. `[PROVEN]`

---

## 6. Structural continuity — the second localizer, and the one the advisor flagged

Disorder is a perturbation on a continuous medium; §3–5 dispatch it. A **structural terminus** is different —
a bare acoustic phonon on a single MT reflects near-totally at the MT end, with **no ω⁻² rescue**, because a
wave does not average over a discontinuity the way it averages over mass fluctuations. Dendritic MTs are
individually only a **few µm** with mixed polarity and frequent ends (Sources [10]), so a naïve single-MT
picture would confine the mode to a segment and **fail** at 15 µm. This is the real threat, and it is resolved
by the bus architecture — not waved away.

**The resolving fact: the bus lives on the staggered bundle, where a terminus is not a wall.**

- The dendritic backbone threads **N = 5–15 MTs per cross-section** (Sources [5]), ~35 nm apart, MAP-crosslinked,
  with individual MTs **staggered** (ends distributed, not aligned) — canonical §2.1/§4.3's "shared, continuous
  infrastructure," `MICROTUBULERESEARCH.md §1`.
- At the plane where one MT ends, the **other N−1 MTs carry straight through.** So a single terminus removes
  only ~**1/N ≈ 7–20%** of the local cross-sectional stiffness — a *dip*, not a break. A true wall (total
  reflection) requires **all N MTs to terminate at the same plane**, which staggering specifically avoids
  everywhere except a genuine backbone discontinuity.
- Therefore the advisor's "wall" **reduces to weak disorder** with σ ~ 1/N ~ 0.1–0.2 — and that case is already
  in §3: σ = 0.3 → ξ ≈ **5 m**; σ = 0.1 → ξ ≈ **46 m**. Both ≫ 15 µm. The ω⁻² protection *does* apply, because
  a staggered terminus is a sub-wavelength (few µm ≪ 134 µm) stiffness fluctuation. `[INFERRED]`

**Two independent reinforcements:**
1. **λ ≫ MT length.** 134 µm vs ~3 µm — the mode homogenizes over ~45 segments per wavelength; it cannot
   resolve individual ends (effective-medium limit). `[INFERRED]`
2. **The condensate is EM-active.** It *is* a coherent-field generator (that is how it couples to the dimers,
   canonical §4.1; Pokorný antenna model, `MICROTUBULERESEARCH.md §5`, Sources [12]). The field phase-locks the
   condensate across the overlapping bundle, bridging the µm gaps that would reflect a bare mechanical phonon.
   `[MODELED]`

**Where a true wall does live — and whether 15 µm clears it.** The only total boundary is where the **whole
bundle** breaks: a **branch point** or the segment end. So:

> **Reach of the cross-synapse bus = distance between backbone discontinuities (branch points), not individual
> MT length.**

Continuous unbranched dendritic segments run **order tens of µm** (Sources [11]; ≳ 15 µm for CA1 apical-oblique
and basal segments). So **two clusters 15 µm apart on the same unbranched segment are spanned**; a pair
straddling a branch point is not. `[GROUNDED, order-of-magnitude — INFERRED conclusion]`

**Testable prediction (the advisor's, sharpened).** Cross-cluster binding requires both clusters on the **same
unbranched backbone segment**. It should be **robust to individual MT turnover** (a bundle property, consistent
with MTs being dynamic few-minute invaders) but **fail across a branch point / bundle discontinuity** — a
spine-placement-and-branch-topology claim, more informative than a decay constant.

---

## 7. Mode identity — CONFIRMED (advisor, mechanistic)

The whole estimate evaluates localization **at ω₀ = 8 MHz**, the frequency the program assigns the mediating
mode. The advisor confirmed this is right, and for a stronger reason than the self-consistency check PO-10
first offered:

- **Mechanistic (primary).** Fröhlich condensation is *downward funnelling*: nonlinear mode–mode coupling
  channels quanta toward the band bottom faster than the bath drains them, and they accumulate in the lowest
  mode. So "the condensing mode" and "the lowest mode of the band" are the **same statement** — the mediating
  mode is the band-bottom mode **by construction** (canonical §4.1, "vibrational energy concentrates into the
  lowest collective mode"). A GHz mediator is not merely inconsistent with L_coh — it is **structurally
  impossible**, because a condensate is *defined* by accumulation at the bottom. `[PROVEN — Fröhlich; INFERRED
  for our system]`
- **Occupation (independent).** At 310 K, thermal occupation n̄ = 1/(e^{hf/kT} − 1): at 8 MHz,
  x = hf/kT = 1.24×10⁻⁶ → n̄ = 8.07×10⁵; at 1 GHz, x = 1.55×10⁻⁴ → n̄ = 6.5×10³. Mediation strength scales with
  occupation, so the low mode dominates by **~124×** even before condensation adds anything. `[PROVEN]`
- **Self-consistency (kept as a third line).** A GHz mediator gives τ = Q/ω → L_coh ~ 0.17 µm, which cannot
  reach 15 µm on lifetime grounds either — so it is internally inconsistent with the program's committed
  coherence length. `[INFERRED]`

The optical ~1 µm localization (Trp superradiance) is **not** ported: that is a *high-frequency excitonic* mode
(disorder ~20× critical), a different mode with different disorder — exactly the category error the scoping doc
warned against (`scoping:41-47`; Babcock 2024, Sources [11b]). §3–4 show why: the acoustic mode is 3–7 decades
lower in frequency, so its ξ is 10⁵–10⁷× longer.

---

## 8. What this closes, and the honest residuals

**Closes:** the `L·PO9-2 §5` load-bearing caveat. The shared condensate bus is delocalized over ≥ 15 µm — on
disorder (orders of margin) *and* structural continuity (staggered bundle → 1/N weak disorder; reach set by
inter-branch segment length > 15 µm). So the SYNC/λ_F = 214 cell is the physical case, and the graded readout
computation (ρ = +0.936, step null rejected t = 24.25) is a real computation over a channel that exists. The
advisor's "settle it by an Anderson calculation, not a sweep" is discharged, and the mode-identity assumption
is confirmed mechanistically. **Verdict locked; Unit C may proceed.**

**Honest residuals (none verdict-changing):**
1. **Model default vs experiment geometry.** `N_backbone` encodes a **10 µm** segment
   (`= 5 MTs × 100 dimers/µm × 8 Trp × 10 µm`, `MICROTUBULERESEARCH.md:178`) — *below* the experiment's 15 µm
   cluster gap. This is a parameter default, not a physics limit (the param is sweepable to longer segments),
   but **if a 15 µm cross-cluster reach is quoted anywhere external, set the backbone segment ≥ 15 µm and note
   the default was under-scoped for this geometry.** Flag for Sarah. `[GROUNDED — code]`
2. **Inter-branch segment length** is asserted as "tens of µm" at order-of-magnitude; the precise
   per-compartment CA1 mean sits in Sources [11]'s supplementary tables and should be pinned before the
   segment-length claim is quoted as a number.
3. **Exact 310 K mechanical disorder σ** unpinned (needs MD) — shown not load-bearing (§5b).
4. **Q = 10** remains the program's standing `[CONTESTED]` AMRIS-class bet (canonical §6.1); this doc assumes
   it, as the rest of the program does. If Q ≪ 10, L_coh shrinks (∝ Q) and the *lifetime* limit, not disorder
   or structure, would eventually bind — but that is the pre-existing backbone bet, not something this estimate
   introduces.

**No code changes** are made or implied here. The one code-relevant action is residual #1 (backbone segment
length for the 15 µm geometry), which is Sarah's to ratify.

---

## Sources

1. Deriu M.A. et al., *Anisotropic elastic network modeling of entire microtubules*, Biophys. J. 99 (2010) 2190. DOI: 10.1016/j.bpj.2010.06.070. (MT effective Young's modulus; PMC2931733.)
2. Sept D. & MacKintosh F.C., *Mechanical Properties of a Complete Microtubule Revealed through Molecular Dynamics Simulation*, Biophys. J. 99 (2010) 2049. DOI: 10.1016/j.bpj.2010.07.008. (Longitudinal modulus 0.3–1.9 GPa; PMC2905083.)
3. Monthus C. & Garel T., *Anderson localization of phonons in dimension d = 1, 2, 3*, Phys. Rev. B 81 (2010) 224208. DOI: 10.1103/PhysRevB.81.224208. arXiv:1003.5988. (1D low-frequency ξ ∝ 1/ω².)
4. *Unveiling the catalytic mechanism of GTP hydrolysis in microtubules*, PNAS (2023). DOI: 10.1073/pnas.2305899120. (β E-site γ-phosphate loss on GTP→GDP; ~80 Da.)
5. Harris K.M. et al. (2022), *Dendritic spine density scales with microtubule number in rat hippocampal dendrites*, Neuroscience. PMC9038701. (MT number per dendrite 5–15+; continuous dendritic infrastructure; ~35 nm inter-MT spacing.)
5b. Tubulin biochemistry: αβ-heterodimer ≈ 100–110 kDa, ~50 kDa/monomer; β E-site GTP→GDP. J. Cell Sci. 106 (1993) 627, PMID 8282766.
6. Alushin G.M. et al., *High-resolution microtubule structures reveal the structural transitions in αβ-tubulin upon GTP hydrolysis*, Cell 157 (2014) 1117. DOI: 10.1016/j.cell.2014.03.053. (2.3–2.4% axial compaction GTP→GDP; PMC4054694.)
7. *Nucleotide-dependent stiffness suggests role of interprotofilament bonds in microtubule assembly*, bioRxiv (2017) 098608. DOI: 10.1101/098608. (GDP vs GTP-analog elastic-modulus contrast ~2×.)
8. Ishii K., *Localization of eigenstates and transport phenomena in the one-dimensional disordered system*, Prog. Theor. Phys. Suppl. 53 (1973) 77. DOI: 10.1143/PTPS.53.77. (Mass-disorder 1D localization; acoustic ω⁻² law.)
9. *Dynamics of exciton polaron in microtubule*, Heliyon (2022). DOI: 10.1016/j.heliyon.2022.e08897. (MT phonon speeds 200–600 m/s for radial/torsional/helical branches.)
10. Baas P.W. et al., stability & polarity of neuronal microtubules; *Cytoskeleton* 73 (2016) 442, DOI: 10.1002/cm.21286; and Baas 1988 (PNAS 85:8335, dendritic mixed polarity). (Dendritic MT lengths few µm to >100 µm; individual MTs do not traverse the whole process; mixed polarity.)
11. *Differential Structure of Hippocampal CA1 Pyramidal Neurons in the Human and Mouse*, Cereb. Cortex 30 (2020) 730. DOI: 10.1093/cercor/bhz125. (Dendritic segment/branch morphometry; per-compartment segment length in supp. tables.)
11b. Babcock N.S. et al., *Ultraviolet Superradiance from Mega-Networks of Tryptophan in Biological Architectures*, J. Phys. Chem. B 128 (2024) 4035. DOI: 10.1021/acs.jpcb.3c07936. (The OPTICAL/excitonic ~1 µm mode — NOT ported to the acoustic mode.)
12. Pokorný J. et al. (2021), *Generation of Electromagnetic Field by Microtubules*, Int. J. Mol. Sci. 22:8215. PMC8348406. (MT as coherent antenna; condensate field propagates coherently along the backbone — the EM bridging in §6.)
13. Ioffe A.F. & Regel A.R., Prog. Semicond. 4 (1960) 237. (Ioffe–Regel bound: ξ_min ~ 1/k.)
