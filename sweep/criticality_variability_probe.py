#!/usr/bin/env python3
"""
Near-critical variability probe (D17 characterization)
======================================================
Isolates the single-synapse stochastic dimer-nucleation dynamics and samples the
end-of-episode dimer-count DISTRIBUTION across N independent replicates, to test
whether nucleation is near-critical (the place-cell all-or-none signature).

WHY a dedicated probe (not run_spatial_discovery):
  The validated 5-trial run (9/1/31/23/22 dimers) conflates three sources of
  spread: (a) genuine per-traversal stochastic nucleation, (b) evolving
  structural/learned-heading state across trials, (c) re-randomized start
  positions -> different features visited. A criticality claim is a DISTRIBUTION
  claim and needs many independent samples under controlled drive. This probe
  holds a SINGLE synapse at a FIXED drive and varies only the random stream.

STOCHASTIC SOURCES (all global np.random -> controlled by np.random.seed):
  - channel gating CTMC          analytical_calcium_system.py:132  (the near-critical driver:
                                                                     coincident openings cluster Ca)
  - dissolution / stochastic formation  ca_triphosphate_complex.py:405,410
  - singlet-collapse measurement        model6_core.py:579
  Glutamate is held CONSTANT (sustained agonist) so NMDAR occupancy is pinned and
  the control parameter is the drive voltage alone. (Presynaptic-release
  stochasticity is a SEPARATE layer, deliberately excluded here.)

ORDER PARAMETER / GATE (ca_triphosphate_complex.py:399):
  S = (ca^3 * po4_trivalent^2 / 1e-26)^0.2 ; formation gated on S > 1.
  Peak nanodomain [Ca] is the recorded proxy for distance to the S=1 gate.

CRITICALITY SIGNATURES TESTED:
  1. fraction-nucleated is a SHARP sigmoid in drive (the order parameter)
  2. Fano factor (var/mean of dimer count) PEAKS at the critical drive (susceptibility)
  3. dimer-count distribution is BIMODAL near S~1 (all-or-none), unimodal away from it

Findings-first: prints a per-drive table and writes JSON for plotting. No model edits.
"""

import sys
import os
import json
import time
import argparse
import logging
import numpy as np

# --- Suppress noise (copied from run_spatial_discovery.py) ---
logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from presynaptic_release import PresynapticRelease


def make_synapse():
    """A single synapse configured exactly as run_spatial_discovery's network synapses:
    EM coupling on, P31, feedback OFF (baseline), microtubule invasion on."""
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    params.spine_calcium_feedback = False
    syn = Model6QuantumSynapse(params)
    syn.set_microtubule_invasion(True)
    return syn


def act_to_voltage(act):
    """Identical mapping to run_spatial_discovery.activations_to_stimuli:
    act in [0,1] -> subthreshold band -70 mV (rest) .. -40 mV (peak)."""
    return -70e-3 + act * 30e-3


def run_episode(act, duration_s, physics_dt, glutamate, seed, presynaptic=False):
    """One fixed-drive stimulation episode. Returns per-replicate metrics.

    presynaptic=False (default): glutamate held CONSTANT (isolates channel gating).
    presynaptic=True (control b): glutamate is the stochastic cleft event stream from
      PresynapticRelease.step(act, dt) — the real input layer, an INDEPENDENT RNG
      stream (its own default_rng(seed)), added back on top of channel gating.
    """
    np.random.seed(seed)
    syn = make_synapse()
    voltage = act_to_voltage(act)
    release = PresynapticRelease(seed=seed) if presynaptic else None
    stim = {'voltage': voltage, 'glutamate': glutamate, 'reward': False}

    n_steps = int(round(duration_s / physics_dt))
    peak_ca_uM = 0.0
    peak_dimers = 0
    for _ in range(n_steps):
        if release is not None:
            stim['glutamate'] = release.step(act, physics_dt)
        syn.step(physics_dt, stim)
        ca = float(np.max(syn.calcium.get_concentration())) * 1e6
        if ca > peak_ca_uM:
            peak_ca_uM = ca
        nd = len(syn.dimer_particles.dimers)
        if nd > peak_dimers:
            peak_dimers = nd

    return {
        'final_dimers': len(syn.dimer_particles.dimers),
        'peak_dimers': peak_dimers,
        'peak_ca_uM': peak_ca_uM,
    }


def fano(xs):
    xs = np.asarray(xs, dtype=float)
    m = xs.mean()
    return float(xs.var() / m) if m > 0 else 0.0


def run_sweep(acts, reps, duration_s, physics_dt, glutamate, base_seed, presynaptic=False):
    results = {}
    print(f"# duration={duration_s}s dt={physics_dt}s glutamate={glutamate} reps={reps} "
          f"base_seed={base_seed} presynaptic={presynaptic}")
    print(f"{'act':>5} {'V_mV':>6} | {'finalDimers mean/std':>22} {'Fano':>6} "
          f"{'fracNuc':>8} | {'peakCa_uM mean':>14} {'peakDimers mean':>15} | {'wall_s':>7}")
    print("-" * 100)
    for act in acts:
        t0 = time.time()
        final, peak, ca, nuc = [], [], [], []
        for r in range(reps):
            seed = base_seed + int(round(act * 1000)) * 100000 + r
            m = run_episode(act, duration_s, physics_dt, glutamate, seed, presynaptic)
            final.append(m['final_dimers'])
            peak.append(m['peak_dimers'])
            ca.append(m['peak_ca_uM'])
            nuc.append(1.0 if m['peak_dimers'] > 0 else 0.0)
        wall = time.time() - t0
        rec = {
            'act': act, 'voltage_mV': act_to_voltage(act) * 1e3,
            'final_dimers': final, 'peak_dimers': peak, 'peak_ca_uM': ca,
            'final_mean': float(np.mean(final)), 'final_std': float(np.std(final)),
            'final_fano': fano(final),
            'frac_nucleated': float(np.mean(nuc)),
            'peak_ca_mean': float(np.mean(ca)), 'peak_dimers_mean': float(np.mean(peak)),
            'wall_s': wall,
        }
        results[f"{act:.3f}"] = rec
        print(f"{act:5.2f} {rec['voltage_mV']:6.1f} | "
              f"{rec['final_mean']:9.2f} / {rec['final_std']:<8.2f} {rec['final_fano']:6.2f} "
              f"{rec['frac_nucleated']:8.2f} | {rec['peak_ca_mean']:14.1f} "
              f"{rec['peak_dimers_mean']:15.2f} | {wall:7.1f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', type=str, default='0.3,0.6,1.0',
                    help='comma-separated activation levels in [0,1]')
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--duration', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--glutamate', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--presynaptic', action='store_true',
                    help='control b: drive glutamate via stochastic PresynapticRelease')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    acts = [float(x) for x in args.acts.split(',')]
    t0 = time.time()
    results = run_sweep(acts, args.reps, args.duration, args.dt, args.glutamate, args.seed,
                        args.presynaptic)
    total = time.time() - t0
    print("-" * 100)
    print(f"# total wall {total:.1f}s")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'results': results}, f, indent=2)
        print(f"# wrote {args.out}")


if __name__ == '__main__':
    main()
