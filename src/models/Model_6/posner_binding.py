#!/usr/bin/env python3
"""
DERIVED spin-selective Posner/dimer binding-melt rate — the EMERGENT measurement trigger (F2).

Replaces the phenomenological collapse trigger (a raw boolean `dopamine_present =
stimulus.get('reward', False)`, `multi_synapse_network.py:1776`; per-dimer twin gated on
`dopamine_read and calcium_elevated`, `model6_core.py:635`). "Dopamine" appears in NONE of the
source physics — Fisher's readout is a spin-selective molecular BINDING event:

    Fisher 2015 (arXiv 1508.05929), abstract:
      "Quantum measurements can occur when a pair of Posner molecules chemically BIND and
       subsequently MELT, releasing a shower of intra-cellular calcium ions."

The binding is spin-SELECTIVE by Quantum Dynamical Selection (QDS):

    Fisher & Radzihovsky 2018 (PNAS 115(20):E4551, arXiv 1707.05320):
      QDS precludes chemical bond processes "from orbitally non-symmetric molecular states."
      For the pseudospin composite, the reactive channel is the total spin-0 (SINGLET) state.

    Frontiers Pharmacol. 2026 (10.3389/fphar.2026.1777613):
      "A coherent singlet state preserves the structural symmetry ... whereas decoherence into
       the triplet manifold breaks this symmetry"; "when two entangled clusters share a common
       singlet state, their ... dissolution is CORRELATED in time" — i.e. joint per-component
       collapse, exactly what `perform_quantum_measurement` already does per connected component.

So the measurement rate rides on the SINGLET PROJECTION the model already carries (`P_S`), gated
by a diffusion-limited encounter and the SAME bounded productive fraction the formation chemistry
already uses. NOTHING here is tuned to a downstream learning result.

    k_encounter = 4π (D_i+D_j)(a_i+a_j) N_A                 [Smoluchowski, M⁻¹s⁻¹; PROVEN]
    k_measure(i,j) = k_encounter · productive_fraction · (P_S_i · P_S_j)   [M⁻¹s⁻¹]
                     └ diffusion ─┘  └ reused 1% bound ─┘   └ QDS spin-0 joint projection ┘
    λ_measure = k_measure · [dimer]_local                  [s⁻¹, pseudo-first-order]
    P(bind-melt in dt) = 1 − exp(−λ_measure · dt)

ISOTOPE CONSILIENCE (ties F2 to F1 and to the 2025 experiment, no new constant):
  Under ⁷Li doping the DERIVED ³¹P coherence collapses (F1 `nuclear_relaxation`: 216 s → ~14 s), so
  P_S falls to the thermal floor FASTER → the singlet-reactive channel is open for LESS of the
  window → cumulative bind-melt is LOWER → clusters are consumed less → MORE persist. That is the
  measured direction: Straub, Patel, Fisher et al., PNAS 122(10) e2423211122 (2025) — ⁷Li promotes
  a GREATER abundance of observable Ca-phosphate particles than ⁶Li. Direction is DERIVED, not fitted.

FOOTGUNS:
  1. The spin factor is the SINGLET probability P_S (reactive = spin-0), NOT (1−P_S). (1−P_S) is the
     THERMAL single-cluster dissolution the model already has, singlet-PROTECTED
     (`ca_triphosphate_complex.py:418`, k_diss ∝ (1−singlet_excess)) — a DIFFERENT channel. Do not conflate.
  2. No `reward`/`dopamine` term appears anywhere in this module BY DESIGN — that decoupling is F2.
  3. Constants are cited/derived. If any is later moved to make a learning result come out, that is
     the emergent-physics violation (`quantum-computation-and-attribution` §6.1).
"""
import numpy as np

# --- physical constants ---
KB = 1.380649e-23     # J/K            [PROVEN — CODATA]
N_A = 6.022e23        # 1/mol          [PROVEN — CODATA]
T_BODY = 310.15       # K (37 °C)      [model-native: ca_triphosphate_complex.py:133]
ETA_WATER = 6.9e-4    # Pa·s @ 37 °C   [STANDARD-TABLE — cytosol ≈ water at body temp]

# --- model-native quantities (all cited to their source line; NONE tuned here) ---
P_S_FLOOR = 0.25              # thermal singlet-probability floor  [dimer_particles.py:26,62]
WERNER_FLOOR = 1.0/np.sqrt(2) # entanglement separability bound    [Werner 1989; canonical §5]
T_SINGLET = 216.0             # s, undoped Ca₆(PO₄)₄ dimer coherence [Agarwal 2023 band; F1]
PRODUCTIVE_FRACTION = 0.01    # per-encounter reaction probability  [ca_triphosphate_complex.py:155 — the ONE bounded param, REUSED not re-tuned]
A_DIMER = 0.5e-9              # m, Ca₆(PO₄)₄ effective radius        [ca_triphosphate_complex.py:142 — same compact-cluster size]
AZ_VOLUME_L = 1e-17          # L, active-zone volume (0.01 µm³)     [model6_core.py:483]
CA_PER_DIMER = 6             # Ca²⁺ released per dimer on melt       [model6_core.py:483]


def D_stokes_einstein(radius_m=A_DIMER, T=T_BODY, eta=ETA_WATER):
    """Translational diffusion coefficient of a dimer (Stokes–Einstein). [GROUNDED]"""
    return KB * T / (6.0 * np.pi * eta * radius_m)


def k_encounter(radius_m=A_DIMER, T=T_BODY, eta=ETA_WATER):
    """Diffusion-limited bimolecular encounter rate for two identical dimers (Smoluchowski).

    k = 4π (D_i+D_j)(a_i+a_j) N_A, i=j ⇒ 4π (2D)(2a) N_A.  Returns M⁻¹s⁻¹.  [PROVEN]
    """
    D = D_stokes_einstein(radius_m, T, eta)
    k_m3 = 4.0 * np.pi * (2.0 * D) * (2.0 * radius_m)   # m³/s per pair
    return k_m3 * N_A * 1e3                              # ×1e3 L/m³ → M⁻¹s⁻¹


def k_measure(p_s_i, p_s_j, productive_fraction=PRODUCTIVE_FRACTION, **enc):
    """Spin-selective bind-melt rate constant (QDS spin-0 joint projection). M⁻¹s⁻¹.

    The reactive channel is the singlet; the joint reactive projection is P_S_i·P_S_j — the SAME
    product form as the model's Werner cross-bond fidelity F = P_S_i·P_S_j·w_spatial (canonical §5).
    """
    return k_encounter(**enc) * productive_fraction * (p_s_i * p_s_j)


def lambda_measure(p_s_i, p_s_j, n_dimers_local, az_volume_L=AZ_VOLUME_L, **kw):
    """Pseudo-first-order bind-melt rate (s⁻¹) at the local dimer concentration."""
    conc_M = (n_dimers_local / N_A) / az_volume_L
    return k_measure(p_s_i, p_s_j, **kw) * conc_M


def p_bind_melt(p_s_i, p_s_j, n_dimers_local, dt, **kw):
    """Probability a spin-correlated pair binds-and-melts (the measurement fires) within dt."""
    return 1.0 - np.exp(-lambda_measure(p_s_i, p_s_j, n_dimers_local, **kw) * dt)


def p_s_decay(t, T2, p_s0=1.0, floor=P_S_FLOOR):
    """Singlet probability decaying from p_s0 to the thermal floor with coherence time T2."""
    return floor + (p_s0 - floor) * np.exp(-t / T2)


def cumulative_melt_weight(T2, window_s, p_s0=1.0, n_pts=2000):
    """∫₀^window P_S(t)² dt — the pairwise reactive-channel exposure over the coherence window.

    Monotone in how long the pair stays in the singlet (reactive) channel; the isotope observable.
    """
    t = np.linspace(0.0, window_s, n_pts)
    ps = p_s_decay(t, T2, p_s0)
    _trap = getattr(np, 'trapezoid', np.trapz)  # np≥2 renamed trapz→trapezoid
    return _trap(ps * ps, t)


if __name__ == '__main__':
    import inspect
    print("=== derived intermediate quantities ===")
    D = D_stokes_einstein()
    k_enc = k_encounter()
    print(f"  D(dimer, Stokes-Einstein)   = {D:.3e} m²/s   (small cluster, ~half free-Ca 2.2e-10)")
    print(f"  k_encounter (Smoluchowski)  = {k_enc:.3e} M⁻¹s⁻¹")
    print(f"  productive_fraction (reused)= {PRODUCTIVE_FRACTION}")

    print("\n=== QDS spin selectivity (reactive channel = singlet) ===")
    k_hi = k_measure(1.0, 1.0)          # fully coherent pair
    k_lo = k_measure(P_S_FLOOR, P_S_FLOOR)  # thermal-floor pair
    k_wr = k_measure(WERNER_FLOOR, WERNER_FLOOR)  # at the entanglement bound
    print(f"  k_measure(P_S=1.00) = {k_hi:.3e} M⁻¹s⁻¹")
    print(f"  k_measure(P_S=0.71) = {k_wr:.3e} M⁻¹s⁻¹   (Werner floor)")
    print(f"  k_measure(P_S=0.25) = {k_lo:.3e} M⁻¹s⁻¹   (thermal floor)")
    print(f"  selectivity coherent/floor = {k_hi/k_lo:.1f}×  (= (1/0.25)² = 16, pairwise spin-0 projection)")

    print("\n=== a concrete measurement timescale (1000 dimers, 5 ms step) ===")
    for ps in (1.0, WERNER_FLOOR, P_S_FLOOR):
        lam = lambda_measure(ps, ps, 1000)
        print(f"  P_S={ps:.2f}: λ_measure = {lam:8.2f} s⁻¹   P(melt|5ms) = {p_bind_melt(ps,ps,1000,5e-3):.4f}")

    print("\n=== F1 isotope consilience (⁶Li vs ⁷Li, DERIVED — ties to 2025 PNAS) ===")
    try:
        from nuclear_relaxation import T2_observed
        T2_undoped = T2_observed(None)   # 216 s
        T2_li6 = T2_observed('Li6')      # ~216 s (dopant negligible)
        T2_li7 = T2_observed('Li7')      # ~14 s (real kill)
    except Exception as e:
        print(f"  (nuclear_relaxation import failed: {e}; using F1 handoff values)")
        T2_undoped, T2_li6, T2_li7 = 216.0, 216.0, 13.76
    WINDOW = 200.0  # s, the coherence/eligibility window (canonical §2.2)
    melt6 = cumulative_melt_weight(T2_li6, WINDOW)
    melt7 = cumulative_melt_weight(T2_li7, WINDOW)
    print(f"  T2: undoped={T2_undoped:.1f}s  ⁶Li={T2_li6:.1f}s  ⁷Li={T2_li7:.1f}s")
    print(f"  cumulative bind-melt weight over {WINDOW:.0f}s window:  ⁶Li={melt6:.1f}   ⁷Li={melt7:.1f}")
    print(f"  ⁷Li/⁶Li melt ratio = {melt7/melt6:.3f}  (<1 ⇒ ⁷Li melts LESS ⇒ MORE particles persist)")

    print("\n=== ACCEPTANCE CHECKS ===")
    ok_diff   = 1e9 <= k_enc <= 2e10                       # diffusion-limited magnitude
    ok_qds    = k_hi > k_wr > k_lo and abs(k_hi/k_lo - 16.0) < 0.5   # monotone spin-0 selectivity
    ok_iso    = melt7 < melt6                              # ⁷Li melts less → 2025 PNAS direction
    src = inspect.getsource(k_measure) + inspect.getsource(lambda_measure) + inspect.getsource(p_bind_melt)
    ok_nodopa = ('reward' not in src) and ('dopamine' not in src)    # measurement is decoupled from reward
    print(f"  [{'PASS' if ok_diff else 'FAIL'}] encounter rate is diffusion-limited (1e9–2e10 M⁻¹s⁻¹)")
    print(f"  [{'PASS' if ok_qds else 'FAIL'}] QDS: rate monotone in P_S, coherent/floor = 16× (spin-0 projection)")
    print(f"  [{'PASS' if ok_iso else 'FAIL'}] isotope consilience: ⁷Li bind-melt < ⁶Li (⇒ ⁷Li more particles, PNAS 2025)")
    print(f"  [{'PASS' if ok_nodopa else 'FAIL'}] measurement rate carries NO reward/dopamine term (F2 decoupling)")
    allok = all([ok_diff, ok_qds, ok_iso, ok_nodopa])
    print(f"\n  ALL: {'PASS — spin-selective binding-melt is emergent; reproduces the literature' if allok else 'FAIL'}")
