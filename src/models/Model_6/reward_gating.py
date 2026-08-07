#!/usr/bin/env python3
"""
DERIVED windowed three-factor reward gate — the emergent reward→consolidation seam (F3, Phase A).

The F2-e null showed reward is inert because it reaches nothing and calcium commits before any reward
read. The dopamine grounding (docs/RESEARCH_DOPAMINE_REWARD_SIGNAL_2026-08-06.md) says the durable
conversion is a THREE-FACTOR rule: a synapse-local eligibility trace, gated by dopamine arriving in a
narrow COINCIDENCE WINDOW after the eligibility event, with the SIGN set by burst-vs-dip.

    durable_drive_i = eligibility_weight(P_S_i) · DA_sign · [t_since_elig ∈ WINDOW]

Cited inputs (NOT tuned to a downstream result):
  · WINDOW = 0.3–2.0 s AFTER the eligibility (glutamate/binding) event.
      Yagishita, Hayashi-Takagi, Ellis-Davies, Urakubo, Ishii & Kasai 2014, Science 345:1616
      (PubMed 25258080), abstract VERIFIED at source: "dopamine promoted spine enlargement only during a
      narrow time window (0.3 to 2 seconds) after the glutamatergic inputs." Molecular basis: compartment-
      alized cAMP/PKA gated by phosphodiesterase.
  · SIGN: phasic burst (DA above tonic) → potentiation (+1); dip (DA below tonic) → depression (−1).
      Reynolds & Wickens 2002; Frémaux & Gerstner 2016 (three-factor); Bayer/Lau/Glimcher 2007 (positive RPE
      in rate, negative in pause) — CONTESTED symmetry: Hart et al. 2014 (symmetric NAc release).
  · eligibility_weight(P_S) = (P_S − 0.25)/0.75, clipped [0,1] — the model's OWN singlet→eligibility rescale
      (model6_core.py:616), NOT a new constant. Per-synapse SPECIFICITY lives here (the local trace), not in DA.

There is NO new free strength constant: the drive is (grounded eligibility rescale) × (cited sign) inside the
(cited) window. Magnitude of the resulting consolidation comes from the existing spine_plasticity dynamics.
"""
import numpy as np

# --- cited constants ---
DA_WINDOW_LO = 0.3   # s, Yagishita 2014 [VERIFIED-at-source]
DA_WINDOW_HI = 2.0   # s, Yagishita 2014 [VERIFIED-at-source]
P_S_FLOOR = 0.25     # model-native eligibility floor (dimer_particles.py:26; model6_core.py:616)
DA_EPS = 0.05        # fractional deadband around tonic for burst/dip classification (avoids tonic-noise sign flips)


def in_window(t_since_elig):
    """Is the current time inside the dopamine coincidence window after the eligibility event?"""
    return DA_WINDOW_LO <= t_since_elig <= DA_WINDOW_HI


def da_sign(da_level, da_tonic):
    """+1 phasic burst (LTP), −1 dip (LTD), 0 at tonic. Sign only — magnitude rides on eligibility."""
    if da_tonic <= 0:
        return 0
    if da_level > da_tonic * (1.0 + DA_EPS):
        return +1
    if da_level < da_tonic * (1.0 - DA_EPS):
        return -1
    return 0


def eligibility_weight(p_s):
    """The model's own singlet→eligibility rescale: P_S 0.25→0, 1.0→1. Carries per-synapse specificity."""
    return float(np.clip((p_s - P_S_FLOOR) / (1.0 - P_S_FLOOR), 0.0, 1.0))


def conversion_drive(p_s, da_level, da_tonic, t_since_elig):
    """Signed durable-consolidation drive for one synapse this step. 0 unless DA is in-window AND off-tonic."""
    if not in_window(t_since_elig):
        return 0.0
    s = da_sign(da_level, da_tonic)
    if s == 0:
        return 0.0
    return s * eligibility_weight(p_s)


if __name__ == '__main__':
    tonic = 20.0   # arbitrary tonic units for the test
    burst, dip = 60.0, 5.0
    print("=== window bounds (Yagishita 0.3–2.0 s) ===")
    for t in (0.1, 0.3, 1.0, 2.0, 3.0):
        print(f"  t_since={t:>4} s : in_window={in_window(t)}")
    print("\n=== sign (burst/dip/tonic) ===")
    print(f"  burst {burst} vs tonic {tonic}: sign={da_sign(burst,tonic):+d}")
    print(f"  dip   {dip} vs tonic {tonic}: sign={da_sign(dip,tonic):+d}")
    print(f"  tonic {tonic} vs tonic {tonic}: sign={da_sign(tonic,tonic):+d}")
    print("\n=== conversion drive (P_S sweep, in-window burst) ===")
    for ps in (0.25, 0.5, 1/np.sqrt(2), 1.0):
        print(f"  P_S={ps:.3f}: drive(in-window,burst)={conversion_drive(ps,burst,tonic,1.0):+.3f}")

    print("\n=== ACCEPTANCE CHECKS ===")
    ps = 1.0
    ok_win_burst = conversion_drive(ps, burst, tonic, 1.0) > 0                       # in-window burst → +
    ok_out_early = conversion_drive(ps, burst, tonic, 0.1) == 0.0                    # before window → 0
    ok_out_late  = conversion_drive(ps, burst, tonic, 3.0) == 0.0                    # after window → 0
    ok_dip       = conversion_drive(ps, dip, tonic, 1.0) < 0                         # in-window dip → −
    ok_noda      = conversion_drive(ps, tonic, tonic, 1.0) == 0.0                    # at tonic (no transient) → 0
    ok_mono      = (conversion_drive(1.0,burst,tonic,1.0) > conversion_drive(0.5,burst,tonic,1.0)
                    > conversion_drive(P_S_FLOOR,burst,tonic,1.0))                   # monotone in P_S (specificity)
    ok_bounds    = in_window(0.3) and in_window(2.0) and not in_window(0.29) and not in_window(2.01)
    for name, ok in [("in-window burst → potentiation (+)", ok_win_burst),
                     ("before 0.3 s → no conversion", ok_out_early),
                     ("after 2.0 s → no conversion", ok_out_late),
                     ("in-window dip → depression (−)", ok_dip),
                     ("at tonic (no transient) → no conversion (DA necessary)", ok_noda),
                     ("drive monotone in P_S (specificity = the trace)", ok_mono),
                     ("window bounds exactly 0.3–2.0 s (Yagishita)", ok_bounds)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allok = all([ok_win_burst, ok_out_early, ok_out_late, ok_dip, ok_noda, ok_mono, ok_bounds])
    print(f"\n  ALL: {'PASS — windowed three-factor gate is cited/emergent (no tuned strength constant)' if allok else 'FAIL'}")
