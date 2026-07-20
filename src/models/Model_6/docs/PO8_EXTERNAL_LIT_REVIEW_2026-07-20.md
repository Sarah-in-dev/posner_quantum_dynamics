# PO-8 external literature investigation — timescales & coherence length for the eligibility trace
**2026-07-20. Fan-out web search + primary-source fetch, adversarially cross-checked. Confidence reflects how well MEASURED data (not theory) supports each claim.**

> ## ⚠ AUTHOR CORRECTION 2026-07-20 — Q3's CONCLUSION WAS WRONG; the DATA stands, the INTERPRETATION does not.
> I concluded from BTSP (~2 s) and CaMKII (~20–40 s) that the trace "only needs ~2–40 s" and the
> model's ~200 s "over-provisions 5–100×." **That inverts the program's thesis.** Our own research is
> explicit that τ≈2 s is the CLASSICAL trace that "dies after seconds" — the thing being BEATEN — not
> the requirement (`coherence-gated-learning:63`; `quantum-biology-primer:17,21,49,51`). The quantum
> trace's target IS the **60–100 s (up to 100–200 s) coherence window**; a trace that only reached
> ~40 s would carry NO quantum advantage (CaMKII already does 40 s). So:
> - **The behavioural numbers are the classical BASELINE, not the target.** The advisor's Q4 74 s/107 s
>   (T₂=216 s, λ=5) is exactly matched to the 60–100 s window and is CORRECT.
> - **Q1/Q2 (τ_dimer, T₂ theory-only/disputed) remain valid as caveats** — but the trace NEEDS the
>   long timescale, so a disputed-low coherence is a RISK TO THE THESIS, not a reason to shorten the
>   assumption.
> - **Q4/λ:** the literature points I raised are real, but given Sarah's instruction to follow the
>   advisor's direction, λ is the advisor's call; I overreached in treating a subagent lit-take as
>   overriding it. Recorded as an open tension, NOT a recommendation to flip λ.
> The rest of this doc is the raw investigation; read it through this correction.

Commissioned to answer "what do we have vs what are we missing" for the eligibility trace, against
the model's assumed timescales (τ_dimer≈200 s, T_singlet=216 s) and the λ=5 vs 214 µm question.

## Q1 — Dimer chemical lifetime (model assumes τ_dimer ≈ 200 s)
**~200 s is NOT supported by any measurement; it is a thermodynamic-stability placeholder, and the
Ca₆(PO₄)₄ dimer as a discrete species has never been observed.** Agarwal/Kattnig/Aiello/Banerjee
2023 (JPCL 13:2673; arXiv:2210.14812) derive stability from formation energy, not kinetics, and
state the dimer "does not appear to have been observed experimentally." Precursor kinetics
(Habraken 2013, Nat Commun 4:1507 — the solution species is [Ca(HPO₄)₃]⁴⁻, dynamically aggregating)
span **seconds → hours** depending on Ca/Pᵢ. **Confidence 200 s is empirically grounded: LOW.**

## Q2 — ³¹P singlet coherence (model uses T_singlet = 216 s)
**The most contested number in the program — literature spans ~1 s to ~10⁶ s, none measured.**
Fisher 2015 (Ann Phys 362:593; arXiv:1508.05929): ~10³ s, "potentially days" (theory). Player & Hore
2018 (J R Soc Interface 15:20180494; arXiv:1807.06339): rigorous upper bound **37 min**, but
**"a few seconds" more likely** realistically — the principal refutation. Agarwal 2023: trimer
Bell-state decays **sub-second**; the DIMER singlet ~10³ s (DFT+spin-dynamics of an unobserved
molecule, no physiological decoherence bath). **Confidence 216 s reflects real physics:
LOW-to-MODERATE (theory-only, disputed by ~10³).**

## Q3 — Behavioural / eligibility-trace window (THE solid result)
**What the trace must bridge is SECONDS to a few TENS of seconds — NOT minutes. Best-measured of
the four.** Bittner/Milstein/Magee 2017 (Science 357:1033, BTSP): plasticity kernel **~2 s** (full
kernel ~±4 s), in vivo. Jain 2024 (Nature 634; bioRxiv 2023.08.01.549180, DDSC/CaMKII): molecular
eligibility fires **20–40 s** post-induction; blocking CaMKII at 30 s abolishes potentiation.
⇒ Required window **~2 s to ~40 s.** Model's ~200 s **over-provisions by 5–100×.** The 200/216 s
values are NOT pinned by the behavioural requirement and could drop to ~40–60 s with no behavioural
cost — moving them out of the most-disputed coherence range. **Confidence: HIGH.**

## Q4 — Condensate coherence length / "unity amplitude across the domain" (the biggest risk)
**Unity-amplitude remote entanglement is the WEAKEST assumption in the model, and the external
literature INVERTS the advisor's Q3 push toward λ=214 µm.** Two mechanisms are being conflated:
- **Optical Trp superradiance** (Babcock 2024 JPCB 128:4035; Celardo/Kurian 2019 NJP 21:023005):
  coherent network scales are **sub-µm to ~1 µm, not 214 µm**; the delocalised superradiant exciton
  **localises/fragments at critical disorder ~10 cm⁻¹**, while physiological disorder is ~200 cm⁻¹
  (≈20× above critical). What survives to high disorder is the **emission-rate/quantum-yield**
  enhancement — a DIFFERENT observable from the entanglement-amplitude one the model needs.
- **Acoustic Fröhlich phonon** (the model's own L_coh = v·τ ≈ 214 µm): a MECHANICAL length, a
  different mechanism from optical superradiance — mixing them is a category error. Reimers 2009
  (PNAS 106:4219): only the **weak** (classical-rate) Fröhlich condensate is biologically feasible;
  **coherent** condensation is "extremely fragile" / inaccessible — exactly the regime unity-across-
  domain needs. **⇒ The SHORT falloff (≈5 µm) is the conservative, literature-consistent choice.
  Remote should NOT be treated as ≈ local. Confidence unity-amplitude is justified: LOW.**

## BOTTOM LINE (what we have / what we're missing)
1. **The trace target is modest and well-established: ~2–40 s** (Q3). The substrate's ~200 s is more
   than enough; the long timescales are not the risk.
2. **Most-likely-wrong, ranked:** (i) the remote-entanglement / coherence-length assumption (Q4) —
   external lit favours the SHORT 5 µm falloff and contradicts the advisor's λ=214 µm push;
   (ii) τ_dimer=200 s (Q1, unobserved, unanchored); (iii) T_singlet=216 s (Q2, theory-only, disputed
   ×10³).
3. **Consequence for the trace measurement:** the eligibility trace should be judged against
   survival to **~2–40 s**, NOT ~200 s. A domain that lives ~tens of seconds is ON TARGET for
   BTSP/CaMKII, not a failure. And λ=5 µm (short falloff) is now the better-supported arm — the two-
   timescale trace (cross ~74 s, intra ~107 s) is the more defensible reading, not the λ=214 µm
   one-timescale collapse.

### Sourcing caveats
Three PDFs (Babcock arXiv, review 1910.08423, Zhang/Scully) returned binary and were read via HTML
mirror / snippet, cross-checked against abstract or secondary source. Player&Hore 37-min bound,
Jain 20–40 s, Bittner ~2 s, and the Agarwal dimer/trimer split each corroborated by ≥2 independent
retrievals.
