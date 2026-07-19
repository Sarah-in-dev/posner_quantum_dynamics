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

IS THE PREDICTION CIRCULAR? No -- MO verification-025 asked, and it is answered by
measurement, not assertion (queue Q4-15):
  * se is computed BELOW at lines ~172-173, and analytical_gap runs at ~175. The
    drive phase is step_network_per_synapse -- the WITHIN-TRIAL path, which never
    touches the gap's k_diss. se is an input off the pre-gap state.
  * MEASURED bit-identical in both code states: importing the pre-fix gap module
    vs the post-fix one and running the identical drive gives
    mean_ps=0.9976548679, se=0.9968731573, n=1915 in BOTH, to the last digit.
  * OUT-OF-SAMPLE the formula DEVIATES, monotonically with g: 1.25e-06 at 0.02 s,
    5.80e-06 at 0.05, 2.23e-05 at the scored 0.10, 4.90e-05 at 0.15, 8.54e-05 at
    0.20. A circular check would agree to ~0 everywhere.

LIMITATION of this formula, stated because it is now on the record: S_pred treats
te as the scalar MAX (50). Real dissolution is a spatial MIXTURE (50 on template
voxels, 1.0 on bare), so the single exponential is an approximation that degrades
with g -- good to 2.2e-5 at the scored gap, 8.5e-5 by g=0.2 and worsening. Use the
concentration-weighted mixture, not this closed form, at longer gaps.
"""
import os
import sys
import json
import logging

import re

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

GAP_S = 0.1        # AMENDMENT G/H: longest gap with ZERO stage-3 removals.
                   # PRE-fix this was 3.0 s; POST-fix dissolution is ~33x faster so
                   # stage 3 fires by 0.25 s and the control (correctly) voided the
                   # 3.0 s run. Note the PRE-fix verdict is gap-INDEPENDENT: a scalar
                   # k_diss cannot move the spatial ratio at ANY gap length.
DRIVE_STEPS = 30
N_SYN = 2
TOL_EXACT = 1e-4   # AMENDMENT G: the isolated measurement's own floor is ~1e-5


def fix_is_present():
    """Read the code under test and report WHICH STATE it is in, rather than
    assuming. The verdict differs by state, so a probe that cannot tell which
    code it measured prints misleading prose -- the defect class this PO exists
    to remove."""
    lines = open(os.path.join(HERE, 'run_theta_burst_45s.py')).read().split('\n')
    for i, ln in enumerate(lines):
        # `in_code` guard: line 62 of that file is a DOCSTRING line reading
        # "k_diss = K_CLASSICAL*(1 - singlet_excess)". Matching it made this
        # detector report PRE-FIX on post-fix code -- i.e. it read PROSE and
        # reported it as code state, which is precisely the defect this probe
        # exists to catch. Require the executable form (leading indent + assignment).
        if (ln.lstrip().startswith('k_diss') and '=' in ln
                and 'K_CLASSICAL' in ln and ln.startswith(' ' * 8)):
            # the assignment may wrap; inspect it and its continuation lines
            block = ' '.join(lines[i:i + 4])
            return 'template_enhancement' in block
    raise RuntimeError("could not locate the gap's k_diss assignment -- "
                       "the probe cannot report code state and must not guess")


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

    analytical_gap(net, GAP_S, dt_sub=min(1.0, GAP_S))

    n_post = sum(len(s.dimer_particles.dimers) for s in net.synapses)
    post = [spatial_ratio(s) for s in net.synapses]

    # AMENDMENT G positive control: stage 3 must not have fired, or S is confounded.
    if n_post != n_pre:
        print(f"\nVERDICT: INCONCLUSIVE — STAGE-3 CONFOUND")
        print(f"  {n_pre - n_post} particle(s) removed during the gap. Removal is")
        print(f"  lowest-coherence-first and therefore spatially biased, so it moves the")
        print(f"  templated/bare ratio by a mechanism AMENDMENT F scoped OUT. Shorten the gap.")
        return 2

    print(f"\nSPATIAL SIGNATURE across a {GAP_S:g} s gap")
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
    fixed = fix_is_present()
    print(f"\n  CODE STATE (read from run_theta_burst_45s.py): "
          f"{'template factor PRESENT (post-fix)' if fixed else 'template factor ABSENT (pre-fix)'}")

    if not fixed and uniform:
        print("\nVERDICT: LOCKED SYMMETRY IS BROKEN — demonstration FAILS on pre-fix code")
        print(f"  - S = {Svals[0]:.9f} / {Svals[1]:.9f} — within {TOL_EXACT} of 1: the spatial")
        print(f"    distribution is UNCHANGED. k_diss is a SCALAR and cannot express a catalyst.")
        print(f"  - quantum-system-canonical:100 LOCKS the 50x factor as symmetric across")
        print(f"    formation AND dissolution. Formation honours it ({tb/max(1,len(parts))*100:.1f}%")
        print(f"    template_bound); gap dissolution does not.")
        rc = 1
    elif not fixed and not uniform:
        print("\nVERDICT: PREMISE WRONG — re-escalate, do not land the fix")
        print("  Pre-fix code moved the spatial ratio, which AMENDMENT F says it cannot.")
        rc = 3
    elif fixed and not uniform:
        d = abs(Svals[0] - S_pred_post)
        ok = d <= 5e-5
        print(f"\nVERDICT: {'SYMMETRY RESTORED' if ok else 'RESTORED BUT OFF PREDICTION'}")
        print(f"  - S = {Svals[0]:.9f} / {Svals[1]:.9f}, i.e. STRICTLY < 1: templated voxels now")
        print(f"    dissolve faster, which is what a symmetric catalyst does.")
        print(f"  - registered post-fix {S_pred_post:.6f}; |diff| = {d:.2e} "
              f"({'within' if ok else 'OUTSIDE'} 5e-5)")
        print(f"  - stage-3 control PASSED ({n_pre} -> {n_post}), so this is stage 2 alone.")
        rc = 0 if ok else 1
    else:
        print("\nVERDICT: FIX PRESENT BUT RATIO UNMOVED — the fix is not doing anything")
        rc = 1
    print("=" * 78)
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
