# Dopamine reinforces CaMKII through DARPP-32/PP1 — it does NOT bypass it (readout grounding)

**2026-08-09. Cited physics/biology assessment for the reward-signed readout at the network level (Phase C).
BLUNT VERDICT: the F3 reward-gated consolidation as currently coded COMMITS by BYPASSING CaMKII
(`model6_core.py:724`, commit on `credit` instead of `molecular_memory`), which is biologically UNSUPPORTED.
The grounded mechanism is that dopamine REINFORCES/DISINHIBITS CaMKII via the D1→PKA→DARPP-32(Thr34)→PP1
cascade; commitment remains CaMKII-gated (the LOCKED commitment pathway is correct and is REINFORCED, not
overturned). The synaptic LTP/LTD SIGN is EMERGENT from PP1 activity, not an imposed ±1.**

This doc grounds the correction and its re-validation. Approved by Sarah 2026-08-09 (unlock `camkii_module.py`;
fix + re-validate; maximize biological fidelity; document the research).

## The problem this fixes
`RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08` established the READOUT is Fisher's spin-selective binding-melt
(Ca²⁺ shower → glutamate → plasticity), reward-TIMED by dopamine. But the F3 *commitment* code took a shortcut:
it commits directly on `quantum_credit` (a `da_sign × eligibility_weight(P_S)` scalar), bypassing the CaMKII
`molecular_memory > 0.5` gate that `model6-commitment-pathway` (LOCKED, May 12) requires. Biologically, dopamine
has no path to commit plasticity that goes around CaMKII — it acts THROUGH it.

## The biology (grounded, both directions)

**CaMKII is the obligatory integrator; Hebbian Ca²⁺ alone is not enough.** Yagishita 2014 (Science): concurrent
glutamate + postsynaptic APs give Ca²⁺→CaMKII, but this "is not sufficient to fully activate CaMKII; rather, it
requires reinforcement by PKA–DARPP-32–PP1 signaling." PKA is active ONLY in a narrow window (0.3–2 s) set by high
phosphodiesterase, and "promoted spine enlargement via CaMKII." CaMKII-T286 autophosphorylation is REQUIRED for
behavioral-timescale plasticity (Xiao 2023, Sci Adv), and the commitment is dendritic/delayed/stochastic (Jain
2024, DDSC — the basis of the LOCKED commitment pathway).

**The molecular link is PP1 acting on CaMKII-pThr286.** PP1 dephosphorylates CaMKII-Thr286 (and AMPAR Ser831/845).
So the control variable dopamine reaches is **PP1 activity = the CaMKII `k_dephosphorylation`**.

**LTP arm (burst → potentiation):** D1 → AC5 → cAMP → PKA → phosphorylates **DARPP-32 at Thr34** →
phospho-Thr34-DARPP-32 **inhibits PP1** → CaMKII-pT286 is no longer stripped → pT286 accumulates → potentiation.

**LTD arm (dip / weak-Ca → depression):** low PKA + Ca²⁺→**PP2B (calcineurin)** dephosphorylates DARPP-32-Thr34 →
**PP1 disinhibited (active)** → strips CaMKII-pT286 → depression. Reinforced by Cdk5 → **DARPP-32-Thr75** →
inhibits PKA (dominant at weak Ca). Strong Ca → **PP2A** → dephosphorylates Thr75 → disinhibits PKA → back toward LTP.

**So the sign is EMERGENT from PP1, not imposed:** PP1 inhibited → LTP; PP1 active → LTD.

## The quantitative scheme (Nakano 2010, PLOS Comput Biol; the field's canonical kinetic model)
- Cascade: D1→AC5→cAMP→PKA→DARPP-32-Thr34⊣PP1⊣(CaMKII-pT286, AMPAR); Ca→PP2B⊣Thr34; Ca→Cdk5→Thr75⊣PKA;
  strong-Ca→PP2A⊣Thr75.
- **Bistable switch:** the PKA–PP2A–DARPP-32(Thr75) positive (double-negative) feedback loop gives 3 fixed points
  (2 stable, 1 unstable) with hysteresis — a threshold on/off in PKA activity vs cAMP.
- **Directionality by Ca amplitude:** weak Ca (< ~0.5 µM) → LTD (Cdk5-Thr75 dominant); strong Ca (> ~1 µM) → LTP
  (PP2A-Thr75 dominant).
- **Timing:** LTP most effective when **dopamine follows calcium**, ~500 ms critical window; elevated *basal*
  dopamine disrupts it (even strong Ca → LTD).
- Timescales: Ca α-function τ ≈ 46.7 ms; dopamine τ ≈ 80 ms; spine volume ~1 µm³.
- Scale/honesty: 72 reactions, 132 parameters (83 literature-derived, **49 hand-tuned**). We ground the STRUCTURE
  and the literature-derived rates; any constant we cannot source is flagged `[MODELED]`, never tuned to our outcome.

## How this maps onto Model 6 (the correction)
1. **New signaling module** (faithful DARPP-32/PP1 cascade): dopamine (D1 occupancy, `dopamine_system.get_d1_occupancy`)
   + calcium → PKA / PP2B / Cdk5 / PP2A → DARPP-32 phospho-state (Thr34, Thr75) → **PP1 activity**. Grounded rates
   from Nakano 2010 / Fernandez 2006; MODELED constants flagged.
2. **`camkii_module.py`** (UNLOCKED, approved): make PP1 dephosphorylation dopamine-dependent — `k_dephos_effective =
   k_dephos_base × PP1_activity` from the cascade (default `PP1_activity = 1.0` ⇒ **bit-identical when off**). Burst →
   PP1↓ → pT286 accumulates (LTP); dip/weak-Ca → PP1↑ → pT286 stripped (LTD). The flagged ungrounded barrier
   constants (`barrier_total_kT=23`, the 40/30/15/15 split) are OUT OF SCOPE — not touched.
3. **`model6_core.py:724`** — REPLACE the F3 credit-bypass: keep the coherence gate (spin-selective binding-melt
   delivers Ca²⁺ only while P_S > Werner), feed the dopamine→PP1 factor into `camkii.step`, and let commitment fire
   via **`molecular_memory > threshold`** (the DDSC lock). The LTP/LTD sign emerges from PP1, replacing `_reward_sign`.
4. **Re-validate** single-synapse F3 (delayed-credit, isotope) under the corrected mechanism, THEN extend to the
   multi-synapse network (the original Phase-C objective).

## Build status — the DARPP-32/PP1 module is built + validated (2026-08-09)
`darpp32_pp1_module.py` implements the cascade (Thr34, Thr75 phospho-states; PKA/PP2B/PP2A/Cdk5; PP1 activity),
grounded-structure with `[MODELED]`-flagged rates. Its `__main__` acceptance PASSES 5/5: the LTP/LTD sign
**emerges** (burst → `pp1_factor` 0.12 = LTP; dip → 1.07 = LTD; ordering burst < tonic < dip; the Ca-amplitude
switch orders weak-Ca above strong-Ca). **FINDING (grounded, cited):** the LTP/LTD range is **ASYMMETRIC** — the
resting striatal state is PKA-suppressed / PP1-ACTIVE (Thr75-P by Cdk5 holds PKA off until dopamine arrives;
Svenningsson/Greengard, Nishi), so dopamine drives strong LTP from a resting point where PP1 is already active and
the dip→LTD headroom is inherently small. This asymmetry is biology, not a bug, and was **confirmed by external
search** rather than tuned away (an initial symmetric-magnitude check FAILED; grounding the resting operating point
showed the check was wrong, not the model). `pp1_factor` is normalized to tonic = 1.0 ⇒ bit-identical when unwired.

## Emergent-physics discipline (LOCKED)
The PP1 modulation and the DARPP-32 cascade rates are grounded from the cited kinetic literature; NONE is tuned to
make the readout decode. Ca-amplitude directionality and the DA-follows-Ca timing window are cited, not fitted. If
the grounded cascade does not produce the delayed-credit / isotope signature, that is a FINDING (the mechanism does
not do what F3 claimed), recorded — not a license to tune PP1.

## Re-validation acceptance (the honest cost of the fix)
The F3-b (delayed-credit 3/3) and F3-c (isotope) results were obtained on the BYPASS. Under the corrected
through-CaMKII mechanism they must be re-run. Expected/required: (a) a still-coherent tag (P_S > Werner) at a
dopamine BURST reinforces PP1-inhibition → CaMKII commits (LTP) across delays where the classical ~2 s trace is
dead; (b) a dip (or decohered tag) → PP1 active / no readout → no potentiation; (c) ⁶Li (long coherence) credits at
long delay, ⁷Li (short) does not. If any fails, report it — do not retune.

## Sources (verified this session)
- Yagishita et al. 2014, Science 345:1616 — dopamine window, PKA→CaMKII reinforcement.
- Nakano et al. 2010, PLoS Comput Biol 6:e1000670 — the kinetic DA/Ca striatal plasticity model (scheme + thresholds).
- Fernandez et al. 2006, PLoS Comput Biol 2:e176 — DARPP-32 as a robust DA/glutamate integrator (Thr34/Thr75).
- Svenningsson/Greengard — the DARPP-32/PP-1 cascade (Thr34⊣PP1; Thr75⊣PKA; PP2B/PP2A phosphatase roles).
- Xiao et al. 2023, Sci Adv — CaMKII autophosphorylation required for BTSP.
- Jain et al. 2024 (DDSC) — dendritic/delayed/stochastic CaMKII underlies BTSP (the LOCKED commitment pathway).
