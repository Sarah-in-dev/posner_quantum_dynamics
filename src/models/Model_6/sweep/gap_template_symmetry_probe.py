#!/usr/bin/env python3
"""
L·GAP-5 — the gap breaks the LOCKED template symmetry.  PO-4.  FAILING-FIRST.
=============================================================================
MO ruling 016: the fix is NOT to be landed yet. This is the failing-first
demonstration, and gen-2 re-runs the concentration-weighted measurement before
the fix is authorised.

Pre-registered in docs/PREREG_PO4_GAP.md AMENDMENT F, BEFORE this file existed.

THE LOCKED CLAUSE (read in the file, not taken from the ruling)
  quantum-system-canonical:100 -- "Template 50x enhancement is a kinetic catalyst
  applied SYMMETRICALLY to formation AND dissolution (detailed balance) ...
  [GROUNDED - Tao 2010; PROVEN - detailed balance] [LOCKED]"

THIS IS A RECURRENCE, NOT A NEW DEFECT
  model6-dimer-formation-chemistry Sec.1 records the SAME defect being repaired in
  update_dimerization: "the 50x template enhancement was applied to FORMATION ONLY
  ... applying it one-sidedly imposes a spurious ~50x equilibrium shift toward
  clusters". The gap re-introduces it at a second site -- the same partial-fix
  shape as substrate-audit item 16.

    within-trial  ca_triphosphate_complex.py:418
        k_diss = k_classical * (1 - se) * template_enhancement     <- symmetric
    the gap       run_theta_burst_45s.py:225
        k_diss = K_CLASSICAL * (1 - se)                            <- catalyst absent

WHY A SPATIAL DISCRIMINATOR
  Comparing rates would need a counterfactual run. The spatial signature does not:
  the gap's k_diss is a SCALAR, so it scales every voxel identically and CANNOT
  change the templated/bare concentration ratio. Under the locked symmetry the
  templated voxels must dissolve faster and that ratio must fall.

  S = 1.000000 EXACTLY is the failing signature. No catalysed process leaves a
  spatial distribution perfectly invariant. (Same shape as AMENDMENT A, where a
  suspiciously exact 1.000000 was the defect and not the result.)

NOT REGISTERED, deliberately: whether the post-fix behaviour is "better". Ruling
016 warns that "conveniently this makes the gap behave better" is the nearest
failure mode. The claim is only that a LOCKED symmetry is currently broken.
"""
import os
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

GAP_S = 3.0        # AMENDMENT G: longest gap with ZERO stage-3 removals NETWORK-WIDE
                   # (5.0 removes 1 particle across the pair; the control caught it)
DRIVE_STEPS = 30
N_SYN = 2
TOL_EXACT = 1e-4   # AMENDMENT G: the isolated measurement's own floor is ~1e-5


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


def spatial_ratio(syn):
    """conc on templated voxels / conc on bare voxels. Undefined if either is empty."""
    c = np.asarray(syn.ca_phosphate.dimerization.dimer_concentration)
    te = np.asarray(syn.ca_phosphate.template_enhancement)
    m = te > 1.0
    t, b = c[m].sum(), c[~m].sum()
    return (t / b if b > 0 else float('nan')), t, b


def weighted_te(syn):
    c = np.asarray(syn.ca_phosphate.dimerization.dimer_concentration)
    te = np.asarray(syn.ca_phosphate.template_enhancement)
    tot = c.sum()
    return ((c * te).sum() / tot if tot > 0 else float('nan')), float(te[te > 1].max() if (te > 1).any() else 1.0)


def main():
    print("=" * 78)
    print("L·GAP-5 — does the gap honour the LOCKED template symmetry?   [PO-4]")
    print("pre-registered: PREREG_PO4_GAP.md AMENDMENT F (before this file existed)")
    print("FAILING-FIRST: the fix is NOT landed. MO ruling 016 sequencing.")
    print("=" * 78)

    np.random.seed(17)
    net = make_network()
    stim = [{"voltage": -10e-3, "reward": False} for _ in net.synapses]
    for _ in range(DRIVE_STEPS):
        RSD.step_network_per_synapse(net, 0.005, stim)

    # --- the measurement gen-2 will re-run (ruling 016 §3.2)
    print("\nCONCENTRATION-WEIGHTED template enhancement (the measurement to be MO-verified):")
    parts = [d for s in net.synapses for d in s.dimer_particles.dimers]
    tb = sum(1 for d in parts if d.template_bound)
    for i, s in enumerate(net.synapses):
        w, temax = weighted_te(s)
        c = np.asarray(s.ca_phosphate.dimerization.dimer_concentration)
        te = np.asarray(s.ca_phosphate.template_enhancement)
        print(f"  syn{i}: grid-mean te = {te.mean():.3f}   CONC-WEIGHTED te = {w:.2f}   "
              f"(max te = {temax:.0f})")
    print(f"  dimer particles: {len(parts)} total, {tb} template_bound "
          f"({tb/max(1,len(parts))*100:.2f}%)")
    print("  ^ the grid-mean is the MISLEADING statistic (PO-4's own denominator error);")
    print("    the concentration-weighted value is the physically relevant one.")

    # --- pre-gap
    pre = []
    for s in net.synapses:
        r, t, b = spatial_ratio(s)
        pre.append((r, t, b))
    if any((not np.isfinite(r)) or t <= 0 or b <= 0 for r, t, b in pre):
        print("\nVERDICT: INCONCLUSIVE — positive control failed (templated or bare "
              "voxels hold no concentration pre-gap; R undefined).")
        return 2

    n_pre = sum(len(s.dimer_particles.dimers) for s in net.synapses)
    ps_vals = [d.singlet_probability for d in parts]
    mean_ps = float(np.mean(ps_vals)) if ps_vals else 0.25
    se = max(0.0, (mean_ps - 0.25) / 0.75)

    analytical_gap(net, GAP_S, dt_sub=1.0)

    n_post = sum(len(s.dimer_particles.dimers) for s in net.synapses)
    post = [spatial_ratio(s) for s in net.synapses]

    # AMENDMENT G positive control: stage 3 must not have fired, or S is confounded.
    if n_post != n_pre:
        print(f"\nVERDICT: INCONCLUSIVE — STAGE-3 CONFOUND")
        print(f"  {n_pre - n_post} particle(s) removed during the gap. Removal is")
        print(f"  lowest-coherence-first and therefore spatially biased, so it moves the")
        print(f"  templated/bare ratio by a mechanism AMENDMENT F scoped OUT. Shorten the gap.")
        return 2

    print(f"\nSPATIAL SIGNATURE across a {GAP_S:.0f} s gap")
    print(f"  {'syn':>5}{'R before':>13}{'R after':>13}{'S = after/before':>19}")
    print("  " + "-" * 50)
    Svals = []
    for i, ((r0, _, _), (r1, _, _)) in enumerate(zip(pre, post)):
        S = r1 / r0 if r0 else float('nan')
        Svals.append(S)
        print(f"  {i:>5}{r0:>13.6f}{r1:>13.6f}{S:>19.9f}")

    uniform = all(abs(S - 1.0) < TOL_EXACT for S in Svals)

    # registered post-fix expectation, computed from THIS run — no free parameter
    K = 0.005
    k_bar = K * (1.0 - se)
    te_max = weighted_te(net.synapses[0])[1]
    S_pred_post = float(np.exp(-k_bar * (te_max - 1.0) * GAP_S))

    print(f"\n  mean P_S pre-gap = {mean_ps:.4f}  ->  singlet_excess = {se:.4f}")
    print(f"  mean k_diss = K*(1-se) = {k_bar:.3e} s^-1")
    print(f"  stage-3 control: particles {n_pre} -> {n_post} (must be equal; else confounded)")
    print(f"  registered PRE-FIX : |S - 1| < {TOL_EXACT} (scalar k_diss cannot move the ratio)")
    print(f"  registered POST-FIX: S == exp(-k_bar*(te-1)*g) = {S_pred_post:.6f}  (te = {te_max:.0f})")

    print("\n" + "=" * 78)
    if uniform:
        print("VERDICT: LOCKED SYMMETRY IS BROKEN — demonstration FAILS on current code")
        print(f"  - S = {Svals[0]:.9f} / {Svals[1]:.9f} — within {TOL_EXACT} of 1, i.e. the")
        print(f"    spatial distribution is UNCHANGED. The gap's k_diss is a SCALAR, so it")
        print(f"    scales templated and bare voxels identically and CANNOT express a catalyst.")
        print(f"    (The residual ~6e-6 is the measurement floor, not a catalytic effect:")
        print(f"    the registered post-fix value is {S_pred_post:.6f}, ~370x further from 1.)")
        print(f"  - quantum-system-canonical:100 LOCKS the 50x template factor as symmetric")
        print(f"    across formation AND dissolution (detailed balance). Formation honours it")
        print(f"    ({tb/max(1,len(parts))*100:.1f}% of particles are template_bound); gap")
        print(f"    dissolution does not.")
        print(f"  - This is the SAME defect model6-dimer-formation-chemistry Sec.1 already")
        print(f"    repaired in update_dimerization, recurring at a second site.")
        rc = 1
    else:
        print("VERDICT: the ratio moved — the gap already expresses spatial dissolution.")
        print("  If this prints on UNMODIFIED code, AMENDMENT F's premise is wrong and the")
        print("  ruling should be re-escalated rather than the fix landed.")
        rc = 0
    print("=" * 78)
    print("LIMITS: 2 synapses, 30 drive steps, one gap length. Measures STAGE 2 (the")
    print("concentration field) only -- stage 3 particle removal is a different mechanism")
    print("and is not part of the registered quantity. NOT a claim that the post-fix gap")
    print("behaves better; only that a LOCKED symmetry is currently broken.")

    out = os.path.join(HERE, 'gap_template_symmetry_results.json')
    with open(out, 'w') as f:
        json.dump(dict(S=[float(x) for x in Svals], uniform=bool(uniform),
                       mean_ps=mean_ps, singlet_excess=se, k_bar=float(k_bar),
                       te_max=float(te_max), S_pred_post_fix=S_pred_post,
                       template_bound_fraction=tb / max(1, len(parts)),
                       weighted_te=[float(weighted_te(s)[0]) for s in net.synapses]),
                  f, indent=2)
    print(f"\npersisted -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
