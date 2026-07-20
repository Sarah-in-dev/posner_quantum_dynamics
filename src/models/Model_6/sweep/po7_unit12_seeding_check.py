#!/usr/bin/env python3
"""
PO-7 UNIT 12 — is the 7-synapse rig reproducible at a fixed seed?

THE GAP THIS FILLS
------------------
HANDOFF_SARAH_2026-07-19_AM.md §1.1 records that the multi-synapse rig is NOT
reproducible under drive: cross_bonds 1179 vs 1848 at the same seed, and eta_max
across four nominally identical driven runs came out 0.0 / 0.0709 / 0.0940 /
0.1069 — i.e. WHETHER THE BACKBONE CONDENSES AT ALL was not reproducible. The
named cause was three `np.random.default_rng()` calls constructed with no seed
argument (camkii_module, spine_plasticity_module, multi_synapse_network), which
draw from OS entropy and ignore any caller seed.

Those three now take an optional seed, and MultiSynapseNetwork(seed=...) spawns
independent deterministic child streams for them. This probe MEASURES whether
that is sufficient, at the data level:

  A. run the rig TWICE at the same seed -> trajectories must be identical
  B. run it once at a DIFFERENT seed    -> trajectory must differ

Rig is copied from po7_unit8_eta2_partition.py (7 synapses, linear, 1um, dt=5e-3,
-40mV sustained + glutamate via PresynapticRelease, MT invasion on, auto
commitment off). Default 10 s, because the known divergence appeared by t=8.7 s.

Reported per sample: n_dimers, n_cross_bonds, n_components, eta_max.
"""
import sys, os, json, logging, time
from datetime import datetime, timezone
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

from model6_parameters import (Model6Parameters, compute_metabolic_power, P_BASAL_W,
                               bose_einstein_occupation, hbar)

N_SYN, SPACING = 7, 1.0
DT = 5e-3
T_SIM = float(os.environ.get("PO7_U12_SECONDS", "10.0"))
VOLT = -40e-3
PATTERN = "linear"
TRACKER_EVERY = 10
SEED_A = int(os.environ.get("PO7_U12_SEED_A", "4242"))
SEED_B = int(os.environ.get("PO7_U12_SEED_B", "9137"))

# Fields compared sample-by-sample. eta_max is the one the handoff showed
# flipping between "condenses" and "does not condense".
COMPARE = ('n_dimers', 'n_cross_bonds', 'n_components', 'eta_max')


def pc(bp):
    return bose_einstein_occupation(bp.omega_0) * hbar * (2 * np.pi * bp.omega_0) ** 2 / bp.Q


def build(seed):
    """po7_unit8_eta2_partition.build(), plus the seed thread under test."""
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern=PATTERN, spacing_um=SPACING,
                              seed=seed)
    net.initialize(Model6QuantumSynapse, params)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def eta_max_of(net, P_c):
    """probe_r() from po7_unit8, reduced to the eta_native maximum."""
    bp = net.params.dendritic_backbone
    e = np.array([getattr(s.spine_plasticity, 'E_invasion', 0.0) for s in net.synapses])
    ca = np.array([s.calcium.channels.get_open_fraction() for s in net.synapses])
    pm = np.array([compute_metabolic_power(e[i], ca[i], bp.p_active_max_W)
                   for i in range(len(net.synapses))])
    r = (P_BASAL_W + net.coupling_weights @ (pm - P_BASAL_W)) / P_c
    return float(max((x - 1) / (x + 1) if x >= 1.0 else 0.0 for x in r))


def run(seed, label):
    """One full trajectory. Returns list of per-sample dicts."""
    from presynaptic_release import PresynapticRelease

    # The rig's own global-RNG seeding, kept identical across runs so that any
    # residual divergence is attributable to the three module RNGs under test.
    np.random.seed(seed)
    net = build(seed)
    P_c = pc(net.params.dendritic_backbone)
    tr = net.entanglement_tracker
    rel = PresynapticRelease(seed=seed)

    samples = []
    n_steps = int(round(T_SIM / DT))
    t0 = time.time()
    for k in range(n_steps):
        g = rel.step(0.95, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        if (k + 1) % TRACKER_EVERY:
            continue
        # NOTE: net.step already invokes tracker.step(dt, synapses, positions,
        # coupling_weights=self.coupling_weights) on its own every-10-steps
        # counter (multi_synapse_network.py:1294). Calling it again here would
        # double-advance the tracker, so we only READ it — exactly as
        # po7_unit8_eta2_partition.py does.
        xb = sum(1 for (a, b), f in tr.cross_synapse_bonds.items()
                 if f > tr.WERNER_ENTANGLEMENT_BOUND and a[0] != b[0])
        comps = tr._find_all_clusters()
        samples.append({
            't': round((k + 1) * DT, 4),
            'n_dimers': len(tr.all_dimers),
            'n_cross_bonds': int(xb),
            'n_components': len(comps),
            'eta_max': eta_max_of(net, P_c),
        })
        if len(samples) % 20 == 0:
            s = samples[-1]
            print(f"  [{label}] t={s['t']:6.3f} dimers={s['n_dimers']:6d} "
                  f"xbond={s['n_cross_bonds']:5d} comps={s['n_components']:5d} "
                  f"eta={s['eta_max']:.6f}  ({(time.time()-t0)/(k+1):.3f}s/step)")
            sys.stdout.flush()
    return samples


def first_diff(a, b):
    """Index of the first sample where any compared field differs; None if identical."""
    for i in range(min(len(a), len(b))):
        for key in COMPARE:
            if a[i][key] != b[i][key]:
                return i, key, a[i][key], b[i][key]
    if len(a) != len(b):
        return min(len(a), len(b)), 'n_samples', len(a), len(b)
    return None


def report(tag, d, n):
    if d is None:
        print(f"  {tag}: IDENTICAL across all {n} samples "
              f"(fields {', '.join(COMPARE)})")
        return True
    i, key, va, vb = d
    print(f"  {tag}: DIFFERS first at sample {i} (t={i*TRACKER_EVERY*DT:.3f}s) "
          f"on '{key}': {va} vs {vb}")
    return False


def main():
    print("PO-7 UNIT 12 — fixed-seed reproducibility of the 7-synapse rig")
    print(f"  N={N_SYN} {PATTERN} spacing={SPACING}um dt={DT} T={T_SIM}s "
          f"drive {VOLT*1e3:.0f}mV + glutamate")
    print(f"  seeds: A={SEED_A} (x2), B={SEED_B}\n")

    a1 = run(SEED_A, "A1")
    a2 = run(SEED_A, "A2")
    b1 = run(SEED_B, "B1")

    print(f"\nRESULT ({len(a1)} samples each)")
    d_same = first_diff(a1, a2)
    d_diff = first_diff(a1, b1)
    same_ok = report(f"same seed  ({SEED_A} vs {SEED_A})", d_same, len(a1))
    diff_seen = not report(f"diff seed  ({SEED_A} vs {SEED_B})", d_diff, len(a1))

    verdict = "PASS" if (same_ok and diff_seen) else "FAIL"
    print(f"\n  REPRODUCIBILITY: {verdict}"
          f"  (same-seed identical: {same_ok}; different-seed differs: {diff_seen})")
    if same_ok and not diff_seen:
        print("  NOTE: different seeds produced identical trajectories — the seed is "
              "not reaching a stochastic path, or this rig is deterministic here.")

    out = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'config': {'n_syn': N_SYN, 'spacing_um': SPACING, 'pattern': PATTERN, 'dt': DT,
                   't_sim': T_SIM, 'volt_mV': VOLT * 1e3, 'tracker_every': TRACKER_EVERY,
                   'seed_a': SEED_A, 'seed_b': SEED_B, 'compare_fields': list(COMPARE)},
        'n_samples': len(a1),
        'same_seed_identical': bool(same_ok),
        'same_seed_first_diff': (None if d_same is None else
                                 {'sample': d_same[0], 'field': d_same[1],
                                  'a': d_same[2], 'b': d_same[3]}),
        'diff_seed_differs': bool(diff_seen),
        'diff_seed_first_diff': (None if d_diff is None else
                                 {'sample': d_diff[0], 'field': d_diff[1],
                                  'a': d_diff[2], 'b': d_diff[3]}),
        'verdict': verdict,
        'final_samples': {'A1': a1[-1] if a1 else None, 'A2': a2[-1] if a2 else None,
                          'B1': b1[-1] if b1 else None},
        'trajectories': {'A1': a1, 'A2': a2, 'B1': b1},
    }
    path = os.path.join(SWEEP_DIR, 'po7_unit12_seeding_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
