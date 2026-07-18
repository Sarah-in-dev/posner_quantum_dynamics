#!/usr/bin/env python3
"""
DO THE TWO PUMP SITES AGREE ON THE SAME MODE? — the B2 acceptance measurement (PO-1).

WHY THIS EXISTS
---------------
Model 6 drives Frohlich condensation at TWO sites:

  backbone     multi_synapse_network._update_backbone_field, parameterised by
               DendriticBackboneParameters (model6_parameters.py). Rebuilt in Step B
               (2026-06-02) onto the reference-free quantum-pump threshold n_ex = n_bar_s
               (Wang/Wang 2022), using bose_einstein_occupation at model6_parameters.py:46.

  per-synapse  vibrational_cascade_module.py, parameterised by TubulinCascadeParameters.

Per the May-30 pin these are NOT two different physical systems. They are two segments of
ONE lattice driven at ONE collective mode:

    "omega_0/2pi = 8 MHz -- the collective MT condensing mode"
    "The old 40 GHz/10 GHz were the tubulin protein modes (the conflation bug), not the
     condensing mode."
        -- model6-network-layer-feasibility-may30, lines 130 and 158

Before B2 the two sites disagreed twice over: a different mode (40 GHz vs 8 MHz) AND a
different Planck convention (hbar*f vs h*f, the factor-of-2pi error). This probe measures
whether they now agree, on both axes independently.

WHAT IS MEASURED
----------------
  A1  MODE AGREEMENT       per-synapse omega_0  ==  backbone omega_0
  A2  CONVENTION AGREEMENT the thermal occupation each site's OWN code computes at its own
                           mode  ==  the independently recomputed h*f closed form

A2's reference is recomputed here from CODATA constants -- it does NOT call
bose_einstein_occupation. If it did, then after B2 (where the shipped code calls that
function) A2 would be comparing the function to itself and could never fail. The reference
is physics; the shipped code is what is on trial.

WHY THIS VERDICT CAN FAIL (the part that matters)
-------------------------------------------------
MO_MODEL6.md section 2.3: "A verdict that cannot distinguish its outcomes is not a result."
Two live scars stand behind that line -- commit 683b82f printed CONFIRMED off a single
flickered edge, and the L.ETA-4 probe printed "selectivity holds" while its own positive
control never fired.

So this probe carries TWO POSITIVE CONTROLS, and reports PASS only if the controls FIRE:

  C1  mode-conflation control   re-run A1 with the retired 40 GHz mode injected.
                               A1's comparator MUST report disagreement.
  C2  2pi-convention control    re-run A2 against the hbar*f shortcut.
                               A2's comparator MUST report disagreement, at ratio ~= 2pi.

If a control does not fire, the comparator has stopped discriminating and the verdict is
INVALID -- not PASS. That is the failure mode L.ETA-4 shipped and this probe refuses to.

LIMITS (stated, per the acceptance bar)
---------------------------------------
This measures that the two sites SHARE a mode and a convention. It does NOT establish that
8 MHz is the correct mode for the biology -- that is the May-30 pin's standing bet
(Q >~ 10, Pokorny slip-layer vs Foster/Baish overdamped, skill line 131: "Committed as
hypothesis; not cranked later to save a result"). If that bet is wrong, both sites are
wrong together, and this probe would still report PASS. Agreement is the claim; correctness
of the pinned mode is not.

Read-only. Runs in seconds. Nothing here is tuned; it only reads and compares.
"""
import sys, os
import numpy as np
import logging

logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(HERE)
sys.path.insert(0, MODEL6_DIR)

from model6_parameters import DendriticBackboneParameters
from vibrational_cascade_module import TubulinCascadeParameters, FrohlichCondensation

# --- CODATA 2018, spelled out locally so the reference is independent of the code on trial
H_PLANCK = 6.62607015e-34   # J.s   (exact, SI 2019)
K_B = 1.380649e-23          # J/K   (exact, SI 2019)
TWO_PI = 2.0 * np.pi

REL_TOL = 1e-6


def n_thermal_reference(f_hz: float, T: float) -> float:
    """Bose-Einstein occupation from the h*f closed form. The reference A2 is scored on.

    n_bar = 1 / (exp(h*f / kT) - 1), with f a LINEAR frequency in Hz.
    Uses expm1 for accuracy in the small-x regime (at 8 MHz, x ~ 1.2e-6).
    """
    x = H_PLANCK * f_hz / (K_B * T)
    if x > 500.0:
        return 0.0
    return 1.0 / np.expm1(x)


def n_thermal_hbar_shortcut(f_hz: float, T: float) -> float:
    """The WRONG form -- hbar*f on a linear frequency, dropping the 2pi. Control C2 only."""
    hbar = H_PLANCK / TWO_PI
    x = hbar * f_hz / (K_B * T)
    if x > 500.0:
        return 0.0
    return 1.0 / np.expm1(x)


def agrees(observed: float, expected: float, rel_tol: float = REL_TOL) -> bool:
    if expected == 0.0:
        return observed == 0.0
    return abs(observed - expected) / abs(expected) <= rel_tol


def ratio(observed: float, expected: float) -> float:
    return float('inf') if expected == 0.0 else observed / expected


def site_n_bar_per_synapse(params: TubulinCascadeParameters) -> float:
    """The occupation the PER-SYNAPSE site computes, using its own code, at its own mode.

    Driven at pump_rate = 0.0 so the returned n_bar isolates the Planck convention from
    the pump path entirely.
    """
    return FrohlichCondensation(params).calculate_steady_state(0.0)['n_bar']


def main() -> int:
    cascade = TubulinCascadeParameters()          # per-synapse site
    backbone = DendriticBackboneParameters()      # backbone site
    T = cascade.T_body

    f_syn = cascade.omega_0
    f_bb = backbone.omega_0

    print("=" * 78)
    print("B2 ACCEPTANCE MEASUREMENT -- do the two pump sites agree on the same mode?")
    print("=" * 78)
    print(f"  per-synapse  TubulinCascadeParameters.omega_0    = {f_syn:.6e} Hz")
    print(f"  backbone     DendriticBackboneParameters.omega_0 = {f_bb:.6e} Hz")
    print(f"  T_body = {T:.1f} K")
    print()

    # ---------------- A1: MODE AGREEMENT ----------------
    a1_pass = agrees(f_syn, f_bb)
    print("--- A1  MODE AGREEMENT ---")
    print(f"  per-synapse / backbone mode ratio = {ratio(f_syn, f_bb):.6g}")
    print(f"  A1: {'PASS' if a1_pass else 'FAIL'}  (require ratio == 1 within {REL_TOL:g})")
    print()

    # ---------------- A2: CONVENTION AGREEMENT ----------------
    n_bar_site = site_n_bar_per_synapse(cascade)
    n_bar_ref = n_thermal_reference(f_syn, T)
    a2_pass = agrees(n_bar_site, n_bar_ref)
    print("--- A2  CONVENTION AGREEMENT (per-synapse site vs independent h*f reference) ---")
    print(f"  n_bar computed BY THE SITE      = {n_bar_site:.6e}")
    print(f"  n_bar reference (h*f, CODATA)   = {n_bar_ref:.6e}")
    print(f"  site / reference                = {ratio(n_bar_site, n_bar_ref):.6f}")
    print(f"  A2: {'PASS' if a2_pass else 'FAIL'}  (require ratio == 1 within {REL_TOL:g})")
    print()

    # ---------------- C1: mode-conflation positive control ----------------
    RETIRED_MODE_HZ = 40.0e9   # the tubulin protein mode the May-30 pin retired
    c1_fired = not agrees(RETIRED_MODE_HZ, f_bb)
    print("--- C1  POSITIVE CONTROL: mode-conflation (inject the retired 40 GHz) ---")
    print(f"  injected mode {RETIRED_MODE_HZ:.6e} Hz vs backbone {f_bb:.6e} Hz")
    print(f"  ratio = {ratio(RETIRED_MODE_HZ, f_bb):.6g}")
    print(f"  C1 {'FIRED (comparator rejects it -- good)' if c1_fired else 'DID NOT FIRE -- COMPARATOR IS BLIND'}")
    print()

    # ---------------- C2: 2pi-convention positive control ----------------
    n_bar_wrong = n_thermal_hbar_shortcut(f_syn, T)
    c2_fired = not agrees(n_bar_wrong, n_bar_ref)
    c2_ratio = ratio(n_bar_wrong, n_bar_ref)
    c2_is_2pi = abs(c2_ratio - TWO_PI) / TWO_PI < 0.01
    print("--- C2  POSITIVE CONTROL: 2pi convention (score the hbar*f shortcut) ---")
    print(f"  n_bar via hbar*f shortcut       = {n_bar_wrong:.6e}")
    print(f"  n_bar reference (h*f)           = {n_bar_ref:.6e}")
    print(f"  ratio = {c2_ratio:.6f}   (2pi = {TWO_PI:.6f}; matches 2pi: {c2_is_2pi})")
    print(f"  C2 {'FIRED (comparator rejects it -- good)' if c2_fired else 'DID NOT FIRE -- COMPARATOR IS BLIND'}")
    print()

    # ---------------- VERDICT ----------------
    controls_ok = c1_fired and c2_fired
    print("=" * 78)
    if not controls_ok:
        verdict = "INVALID"
        why = "a positive control did not fire -- the comparator cannot discriminate"
    elif a1_pass and a2_pass:
        verdict = "PASS"
        why = "both sites share one mode and one Planck convention"
    else:
        verdict = "FAIL"
        failed = ", ".join(n for n, ok in (("A1 mode", a1_pass), ("A2 convention", a2_pass)) if not ok)
        why = f"the two sites disagree: {failed}"
    print(f"VERDICT: {verdict} -- {why}")
    print("  LIMIT: this proves the sites AGREE; it does not prove 8 MHz is the right mode.")
    print("         That remains the May-30 pin's standing bet (Q >~ 10, slip-layer).")
    print("=" * 78)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
