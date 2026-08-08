# Does "dopamine-triggered decoherence of the ³¹P tag" have a physical basis? — grounding

**2026-08-08. Cited physics/chemistry assessment. BLUNT VERDICT: the "dopamine directly decoheres/collapses
the ³¹P tag" framing is NOT physically supported and MISIDENTIFIES the readout. The grounded mechanism is the
OPPOSITE polarity — dopamine never touches the spin; it GATES/TIMES Fisher's Ca/pH-dependent spin-selective
binding-melt readout.** This UNIFIES F2 (binding-melt = the physical readout) and F3 (dopamine = the reward
timing/gate), on established physics rather than a new exotic mechanism. Verified at source: Agarwal 2022
(2210.14812) — the DIMER preserves entangled ³¹P spins for "hundreds of seconds" (trimer sub-second) → our
~100 s premise is grounded to the dimer, which the model already uses.

## Why "dopamine decoheres the spin" fails
It conflates two things the source theory keeps separate: the QUANTUM READOUT (Fisher: spin-dependent Posner
**binding then melting** → Ca²⁺ release → glutamate; a Ca/pH/proton event with **NO dopamine** — Fisher 2015
1508.05929, verified in F2) and DECOHERENCE (a passive environmental process, not a signal). The direct-radical
route is real physics but implausible biology:
- **Physics is real:** an unpaired electron is a potent nuclear-spin relaxer (PRE ∝ r⁻⁶); dopamine is a catechol
  that can form a semiquinone RADICAL (paramagnetic) + chelate metals + make ROS.
- **Biology fails on every axis:** dopamine acts by volume transmission onto GPCRs at DOPAMINERGIC terminals,
  not inside the glutamatergic spine where the Posner readout lives; the semiquinone is a rare, slow (min–hr),
  off-pathway autoxidation intermediate, not the sub-second phasic burst; a diffusing radical is indiscriminate
  (can't address one tag); and dissolved O₂ already sets a paramagnetic background the sparse radical can't beat.
  Also: singlet-stored order is ~2.7× LESS sensitive to paramagnetic relaxation — the part Fisher relies on is
  relatively protected. **Verdict: speculation with no worked mechanism.**
- **No precedent:** NO proposal exists for a neuromodulator (or any classical biochemical signal) collapsing a
  nuclear-spin state. The nearest literature (Frontiers 2026 5-HT2A→Ca/phosphate→symmetry-gated Posner
  dissolution) uses the GATING framing — neuromodulator supplies the ionic/metabolic context, the spin state
  gates the chemistry, the drug does NOT collapse the spin. No dopamine in it.

## The grounded readout (ranked most→least physical)
1. **Dopamine GATES Fisher's Ca/pH binding-melt, TIMED by reward — MOST grounded, no new physics.** Fisher's
   readout is Ca²⁺- and pH/proton-dependent; dopamine's real signaling (D1/D2→cAMP/PKA→DARPP-32→local Ca²⁺/pH,
   channels/pumps/exchangers) modulates exactly those on the phasic timescale. Dopamine never touches the spin;
   it sets the Ca/pH conditions under which the SPIN-SELECTIVE binding-melt (QDS; F2's `posner_binding.py`)
   proceeds — thereby timing the readout to reward. This IS the three-factor rule on grounded physics.
2. Endogenous paramagnet / Ca-pH transient as the actual decoherer, dopamine merely correlated — partially grounded.
3. Dopamine-radical DIRECT decoherence — least grounded; speculation.

## THE CORRECTION TO CARRY INTO THE MODEL
Replace *"dopamine triggers decoherence/collapse of the ³¹P tag"* with
**"dopamine signaling (D1/D2→cAMP/PKA→local Ca²⁺/pH) GATES and TIMES Fisher's spin-selective Posner
binding-melt readout of the coherent tag."** Same reward-timing FUNCTION; consistent with every primary source.

**What this means for F1/F2/F3 (unification, not demolition):**
- The **readout mechanism = Fisher's spin-selective binding-melt** (F2, `posner_binding.py`) — NOT a new
  dopamine-decoherence. The binding-melt is spin-selective (QDS: only the coherent/singlet channel binds-melts),
  so the coherence-gating we already have (credit only if the tag is still coherent, P_S > Werner floor) is
  PRESERVED and physically correct — a decohered tag can't undergo the spin-selective readout.
- **Dopamine's role (F3) = gate/time the binding-melt** by modulating Ca/pH at reward time. So F3's functional
  results HOLD (delayed-credit F3-b; isotope signature F3-c) — reward-timed readout of a coherence-gated tag —
  because the mechanism (reward times a coherence-gated readout) is unchanged; only the physical story of HOW
  dopamine enables the readout changes from "decoheres" (wrong) to "gates the Ca/pH binding-melt" (grounded).
- This **resolves the F2-vs-F3 tension**: one readout (binding-melt), reward-timed by dopamine. F2 and F3 unify.

**Model edits owed (framing + reconnection, not new results):**
- `reward_gating.py` / `model6_core` / PREREG_F3 / research log: rename the readout from "dopamine-decoherence"
  to "dopamine-gated binding-melt"; state the corrected physics; keep the coherence-gate (Werner) and the
  reward-timing function. Reconnect the readout to F2's `posner_binding` (the spin-selective binding-melt).
- Keep flagged: the Posner substrate itself is unproven (Agarwal dimer-vs-trimer; Player&Hore ~37 min vs Fisher
  ~1 day lifetime dispute); the "dopamine's Ca/pH modulation reaches the Posner microdomain" step is unproven
  but not implausible. Emergent-only; the readout is now grounded, the substrate premise stays a premise.

Sources: Fisher 2015 (1508.05929) [readout=binding-melt, verified F2]; Fisher & Radzihovsky 2018 (PNAS E4551,
QDS); Swift/Van de Walle/Fisher 2018 (1711.05899, water-proton decoherence); Player & Hore 2018 (1807.06339,
~37 min); **Agarwal 2022 (2210.14812) — dimer "hundreds of seconds", trimer sub-second [VERIFIED]**; Frontiers
Pharmacol. 2026 (fphar.2026.1777613, 5-HT2A gating not collapse); Zadeh-Haghighi & Simon 2021 (radical-pair
lithium); PNAS 2025 (2423211122, Li-isotope on ACP formation); dopamine redox: Segura-Aguilar 2014; Muñoz 2012;
PRE physics (paramagnetic-NMR literature).
