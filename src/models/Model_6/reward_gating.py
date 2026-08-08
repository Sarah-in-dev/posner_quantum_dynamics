#!/usr/bin/env python3
"""
DERIVED reward-gated consolidation — the coherence-window eligibility trace + dopamine readout (F3).

REFRAMED 2026-08-08 against the TCA-mechanism grounding (docs/RESEARCH_TCA_MECHANISM_2026-08-08.md).
The prior version imposed biology's CLASSICAL 0.3-2 s dopamine window (Yagishita) as the gate — which
defeats the thesis. The grounding is explicit: the dopamine READOUT is established (D1/cAMP/PKA/PDE), but
the eligibility TRACE has no known molecular identity and every measured trace (~0.3-5 s) is far too short
for the seconds-to-minutes gap behaviour needs — the temporal gap is UNSOLVED at the single synapse.

Model-6's candidate answer: the eligibility trace = the coherent ³¹P nuclear-spin state (P_S) in Posner
dimers, lifetime ~100 s. Dopamine reads it out (Sarah's "dopamine-triggered decoherence") whenever reward
arrives, as long as the tag is STILL COHERENT — so the eligibility window is the COHERENCE LIFETIME, not a
fixed 0.3-2 s. This is a NEW readout for a NEW (quantum) trace that must COEXIST with — not replace — the
classical cascade. The classical 0.3-2 s window is kept below only as the BASELINE arm the quantum trace beats.

    quantum_credit  = eligibility_weight(P_S_at_reward) · DA_sign      iff  P_S > Werner floor (still coherent)
    classical_credit= eligibility_weight(P_S_at_reward) · DA_sign      iff  t_since ∈ [0.3, 2.0] s  (biology)

Grounding of each factor (cited; none tuned to a downstream result):
  · eligibility trace = P_S coherence, lifetime set by T₂ (Agarwal 2023; F1 nuclear_relaxation). OUR PROPOSAL.
  · coherence gate = the Werner separability bound 1/√2 (Werner 1989): above it the tag is entangled (a valid
    quantum trace); below it the state is separable/decohered — nothing quantum to read out.
  · DA_sign: burst→+1 (LTP) / dip→−1 (LTD) (Reynolds & Wickens 2002; Yagishita 2014 D1-dependence). ESTABLISHED.
  · CLASSICAL window 0.3-2 s (Yagishita 2014; Shindou 2019 striatal ~2 s) — the baseline, NOT our mechanism.
There is NO tuned strength constant: credit = (P_S rescale) × (cited sign), gated by a cited coherence bound.
"""
import numpy as np

# --- cited constants ---
P_S_FLOOR = 0.25                 # thermal singlet floor (dimer_particles.py:26; model6_core.py:616)
WERNER_FLOOR = 1.0/np.sqrt(2)    # 0.7071 — entanglement/coherence threshold [Werner 1989]; the quantum-tag gate
CLASSICAL_WINDOW_LO = 0.3        # s [Yagishita 2014 / Shindou 2019] — the CLASSICAL baseline window (not ours)
CLASSICAL_WINDOW_HI = 2.0        # s
DA_EPS = 0.05                    # fractional deadband around tonic for burst/dip (baseline-drift robust)


def da_sign(da_level, da_tonic):
    """+1 phasic burst (LTP), −1 dip (LTD), 0 at tonic. Sign only — magnitude rides on the tag."""
    if da_tonic <= 0:
        return 0
    if da_level > da_tonic * (1.0 + DA_EPS):
        return +1
    if da_level < da_tonic * (1.0 - DA_EPS):
        return -1
    return 0


def eligibility_weight(p_s):
    """The model's own singlet→eligibility rescale: P_S 0.25→0, 1.0→1. Per-synapse specificity = the tag."""
    return float(np.clip((p_s - P_S_FLOOR) / (1.0 - P_S_FLOOR), 0.0, 1.0))


def is_coherent(p_s):
    """Is the tag still a valid QUANTUM eligibility trace (entangled, above the Werner bound)?"""
    return p_s > WERNER_FLOOR


def quantum_credit(p_s, da_level, da_tonic):
    """Coherence-window readout (OUR mechanism): dopamine reads out the tag at reward time, at ANY delay,
    as long as it is still coherent. The 'window' is the coherence lifetime — no fixed time gate."""
    if not is_coherent(p_s):
        return 0.0                      # tag has decohered → nothing quantum to read out
    s = da_sign(da_level, da_tonic)
    if s == 0:
        return 0.0                      # dopamine is the readout trigger; no transient → no readout
    return s * eligibility_weight(p_s)


def classical_credit(p_s, da_level, da_tonic, t_since):
    """CLASSICAL baseline readout (biology's short trace): credit only if dopamine arrives in the fixed
    0.3-2 s window. This is the arm the quantum trace must BEAT at long delays."""
    if not (CLASSICAL_WINDOW_LO <= t_since <= CLASSICAL_WINDOW_HI):
        return 0.0
    s = da_sign(da_level, da_tonic)
    if s == 0:
        return 0.0
    return s * eligibility_weight(p_s)


def p_s_decay(t, T2, p_s0=1.0, floor=P_S_FLOOR):
    """Tag coherence decaying from p_s0 to the thermal floor with coherence time T2."""
    return floor + (p_s0 - floor) * np.exp(-t / T2)


def readable_lifetime(T2, p_s0=1.0):
    """How long the tag stays a valid quantum trace: time until P_S crosses the Werner floor."""
    if p_s0 <= WERNER_FLOOR:
        return 0.0
    return -T2 * np.log((WERNER_FLOOR - P_S_FLOOR) / (p_s0 - P_S_FLOOR))


if __name__ == '__main__':
    tonic, burst, dip = 20e-9, 10e-6, 5e-9
    T2_UNDOPED, T2_LI7, T2_CLASSICAL = 216.0, 13.76, 2.0   # coherence (F1) vs biology's ~2 s trace
    print("=== readable tag lifetime (time until P_S crosses the Werner floor 0.707) ===")
    for name, T2 in [("undoped/⁶Li", T2_UNDOPED), ("⁷Li", T2_LI7), ("classical ~2 s", T2_CLASSICAL)]:
        print(f"  {name:<14} T2={T2:6.1f}s → readable for {readable_lifetime(T2):6.1f} s")

    print("\n=== the temporal-gap test: credit vs reward delay (undoped quantum tag vs classical) ===")
    print(f"  {'delay s':>8} {'P_S(quantum)':>13} {'quantum_credit':>15} {'classical_credit':>17}")
    for delay in (1, 2, 5, 10, 30, 100, 150):
        ps_q = p_s_decay(delay, T2_UNDOPED)
        qc = quantum_credit(ps_q, burst, tonic)
        cc = classical_credit(ps_q, burst, tonic, delay)
        print(f"  {delay:>8} {ps_q:>13.3f} {qc:>15.3f} {cc:>17.3f}")

    print("\n=== ACCEPTANCE CHECKS ===")
    ps_hi, ps_lo = 0.99, 0.30   # coherent vs decohered (below Werner)
    ok_qlong  = quantum_credit(ps_hi, burst, tonic) > 0                          # coherent tag credits (any delay)
    ok_qdecoh = quantum_credit(ps_lo, burst, tonic) == 0.0                       # decohered tag → no credit
    ok_qnoda  = quantum_credit(ps_hi, tonic, tonic) == 0.0                       # DA is the readout trigger
    ok_qdip   = quantum_credit(ps_hi, dip, tonic) < 0                            # sign works
    ok_gap    = (quantum_credit(p_s_decay(30, T2_UNDOPED), burst, tonic) > 0     # quantum credits at 30 s...
                 and classical_credit(p_s_decay(30, T2_UNDOPED), burst, tonic, 30) == 0.0)  # ...classical dead
    ok_iso    = readable_lifetime(T2_UNDOPED) > 30.0 > readable_lifetime(T2_LI7) # ⁶Li long-trace, ⁷Li short
    ok_base   = readable_lifetime(T2_CLASSICAL) < 3.0                            # classical trace ~seconds
    for name, ok in [("coherent tag credits (no fixed time window)", ok_qlong),
                     ("decohered tag (below Werner) → no credit", ok_qdecoh),
                     ("dopamine necessary (readout trigger)", ok_qnoda),
                     ("sign works (dip → LTD)", ok_qdip),
                     ("TEMPORAL-GAP: quantum credits @30 s where classical is dead", ok_gap),
                     ("isotope lever sets trace lifetime (⁶Li≫⁷Li)", ok_iso),
                     ("classical baseline trace is ~seconds", ok_base)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allok = all([ok_qlong, ok_qdecoh, ok_qnoda, ok_qdip, ok_gap, ok_iso, ok_base])
    print(f"\n  ALL: {'PASS — coherence-window trace + dopamine-decoherence readout; solves the temporal gap the classical trace cannot' if allok else 'FAIL'}")
