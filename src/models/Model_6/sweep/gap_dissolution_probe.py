#!/usr/bin/env python3
"""
L·GAP-3 — dimer survival across a gap, as a function of K_CLASSICAL.  PO-4.
===========================================================================
MO rotation 002 acceptance: "report the before/after dimer-count delta across a
gap. A 10x dissolution-rate change is not cosmetic. Measure it, state it, and do
NOT damp it -- if a standing result moves, that is an escalation, not a
regression."

Pre-registered in docs/PREREG_PO4_GAP.md AMENDMENT E BEFORE the constant changed.

WHAT IS MEASURED
  S = n_dimers(after gap) / n_dimers(before gap), at the same driven state.

THE REGISTERED BRACKET (derived from the code, script-computed, zero free params)
  The gap computes k_diss = K*(1 - se), se = (P_S - 0.25)/0.75. P_S decays toward
  0.25 across the gap, so se: se0 -> 0 and k_diss: K(1-se0) -> K. Survival is
  therefore bracketed by its two endpoints:
      exp(-K*g)  <=  S  <=  exp(-K*g*(1-se0))
  Falling OUTSIDE the bracket falsifies the model of the dissolution path -- and
  in that case the constant change is NOT the explanation for what is seen.

Run this once before the constant change and once after; the two runs are the
before/after delta the acceptance asks for. The live K_CLASSICAL is read out of
the module source and printed, so each run is self-labelling.
"""
import os
import re
import sys
import json
import logging

import numpy as np

logging.disable(logging.INFO)
HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(M6)))
sys.path.insert(0, M6)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'sweep'))
for _n in ['model6_core', 'multi_synapse_network', 'dimer_particles',
           'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
           'quantum_coherence', 'pH_dynamics', 'dopamine_system', 'em_tryptophan_module',
           'em_coupling_module', 'local_dimer_tubulin_coupling', 'camkii_module',
           'spine_plasticity_module', 'photon_emission_module', 'photon_receiver_module',
           'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(_n).setLevel(logging.ERROR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
import run_spatial_discovery as RSD
from run_theta_burst_45s import analytical_gap

GAPS = (20.0, 45.0)
N_SYN = 2
DRIVE_STEPS = 30


def fix_is_present():
    """Which code state is under test. AMENDMENT E's bracket was derived for the
    NO-TEMPLATE formula; after ruling 016 that premise is void, so a probe that
    cannot tell the states apart prints a misleading FALSIFIED. Same guard as
    gap_template_symmetry_probe: match executable code, not the docstring."""
    lines = open(os.path.join(HERE, 'run_theta_burst_45s.py')).read().split('\n')
    for i, ln in enumerate(lines):
        if (ln.lstrip().startswith('k_diss') and '=' in ln
                and 'K_CLASSICAL' in ln and ln.startswith(' ' * 8)):
            # the assignment WRAPS -- template_enhancement sits on the continuation
            # line. Checking only `ln` reported ABSENT on post-fix code.
            return 'template_enhancement' in ' '.join(lines[i:i + 4])
    raise RuntimeError("cannot locate the gap's k_diss assignment; refusing to guess")


def live_K():
    """Read the constant actually in force, so the run labels itself."""
    src = open(os.path.join(HERE, 'run_theta_burst_45s.py')).read()
    m = re.search(r'^\s*K_CLASSICAL\s*=\s*([0-9.eE+-]+)', src, re.M)
    return float(m.group(1)) if m else float('nan')


def make_network(n=N_SYN):
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=n, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def count(net):
    return sum(len(s.dimer_particles.dimers) for s in net.synapses)


def mean_ps(net):
    v = [d.singlet_probability for s in net.synapses for d in s.dimer_particles.dimers]
    return float(np.mean(v)) if v else 0.25


def driven(seed):
    np.random.seed(seed)
    net = make_network()
    stim = [{"voltage": -10e-3, "reward": False} for _ in net.synapses]
    for _ in range(DRIVE_STEPS):
        RSD.step_network_per_synapse(net, 0.005, stim)
    return net


def main():
    K = live_K()
    print("=" * 78)
    print(f"L·GAP-3 — dimer survival across a gap    [PO-4]")
    print(f"LIVE K_CLASSICAL = {K}   (read from run_theta_burst_45s.py)")
    print("pre-registered: PREREG_PO4_GAP.md AMENDMENT E, before the constant changed")
    print("=" * 78)

    FIXED = fix_is_present()
    print(f"CODE STATE: template factor {'PRESENT (post-ruling-016)' if FIXED else 'ABSENT'}")
    rows, fails = [], []
    print(f"\n{'gap':>6}{'n before':>11}{'n after':>10}{'survival':>11}"
          f"{'bracket lo':>13}{'bracket hi':>12}   verdict")
    print("-" * 78)
    for g in GAPS:
        net = driven(seed=17)
        n0 = count(net)
        se0 = max(0.0, (mean_ps(net) - 0.25) / 0.75)
        # concentration-weighted te -- the physically relevant factor (the grid mean
        # understates it ~33x; that was PO-4's own denominator error)
        c0 = np.asarray(net.synapses[0].ca_phosphate.dimerization.dimer_concentration)
        tef = np.asarray(net.synapses[0].ca_phosphate.template_enhancement)
        te_eff = float((c0 * tef).sum() / c0.sum()) if c0.sum() > 0 else 1.0
        analytical_gap(net, g, dt_sub=1.0)
        n1 = count(net)
        S = n1 / n0 if n0 else float('nan')
        # AMENDMENT E's bracket, extended for the template factor when present.
        # te enters k_diss multiplicatively, so it scales both bracket endpoints.
        te = te_eff if FIXED else 1.0
        lo, hi = np.exp(-K * te * g), np.exp(-K * te * g * (1 - se0))
        inside = (lo - 1e-9) <= S <= (hi + 1e-9)
        if n0 < 2:
            fails.append(f"gap {g:.0f}s: n_before={n0} < 2 -- survival undefined")
            v = "INCONCLUSIVE"
        else:
            v = "in bracket" if inside else "OUTSIDE BRACKET"
            if not inside:
                fails.append(f"gap {g:.0f}s: S={S:.4f} outside [{lo:.4f}, {hi:.4f}]")
        print(f"{g:>6.0f}{n0:>11d}{n1:>10d}{S:>11.4f}{lo:>13.4f}{hi:>12.4f}   {v}")
        rows.append(dict(gap_s=g, n_before=n0, n_after=n1, survival=float(S),
                         se0=float(se0), bracket=[float(lo), float(hi)],
                         inside=bool(inside)))

    print("\n" + "=" * 78)
    if fails:
        print("VERDICT: FALSIFIED / INCONCLUSIVE")
        for f in fails:
            print(f"  - {f}")
        print("  NOTE: outside-bracket means the DISSOLUTION MODEL is wrong for the code")
        print("  state named above -- the change under test must NOT be credited with it.")
    else:
        print("VERDICT: survival inside the registered bracket at every gap length")
        for r in rows:
            print(f"  - gap {r['gap_s']:.0f}s: {r['n_before']} -> {r['n_after']} dimers "
                  f"(S = {r['survival']:.4f})")
    print("=" * 78)
    print("LIMITS: 2 synapses, 30 drive steps. Measures the gap's DISSOLUTION path only.")
    print("Formation is excluded during silence, so this is 'what survives a silence',")
    print("NOT a steady-state dimer population. Delta reported, never damped.")

    out = os.path.join(HERE, f'gap_dissolution_probe_K{K}.json')
    with open(out, 'w') as f:
        json.dump(dict(K_CLASSICAL=K, rows=rows, fails=fails), f, indent=2)
    print(f"\npersisted -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
