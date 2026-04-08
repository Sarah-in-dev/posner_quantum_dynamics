#!/usr/bin/env python3
"""
Linear Track Place Field Learning Experiment
=============================================
Tests whether the spine→calcium feedback loop produces compounding
potentiation across repeated traversals of a place field.

Setup:
  - 5 synapses = 5 locations on a 150cm linear track
  - Animal runs at ~30 cm/s, each 30cm place field active for ~1s
  - Synapses activate in sequence: 0→1→2→3→4
  - Dopamine delivered at t=3.5s (midway through synapse 3 = reward location)
  - 10 traversals, 20s inter-traversal gaps (analytical)
  - spine_calcium_feedback = True (AMPAR→voltage + volume→VGCC)

Prediction:
  - Synapse 3 (reward location) gets strongest potentiation
  - Synapses 0-2 (before reward) show graded potentiation
  - Synapse 4 (after reward) shows minimal potentiation
  - Feedback loop amplifies differences across traversals
"""

import sys
import os
import time
import logging
import numpy as np

# Suppress noise
logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

MODEL6_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL6_DIR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork

# Import analytical_gap from the interval sweep module
from sweep.run_theta_burst_45s import analytical_gap


# =============================================================================
# TRACK GEOMETRY
# =============================================================================

N_SYNAPSES = 5
PLACE_FIELD_DURATION_S = 1.0   # each synapse active for 1s (30cm / 30cm/s)
DOPAMINE_TIME_S = 3.5          # reward at midway through synapse 3
END_SILENCE_S = 3.0            # 5.0s total - 5.0s stim = rest at end of track
TRAVERSAL_DURATION_S = N_SYNAPSES * PLACE_FIELD_DURATION_S + END_SILENCE_S  # 8s

# Theta burst parameters (same as interval sweep)
THETA_PERIOD_S = 0.125         # 8 Hz theta
SPIKE_PERIOD = 0.010           # 100 Hz within burst
DEPOL_DURATION = 0.002         # 2ms depolarization
SPIKES_PER_BURST = 4
BURST_ACTIVE_S = SPIKES_PER_BURST * SPIKE_PERIOD  # 40ms

DT = 0.001
N_TRAVERSALS = 10
INTER_TRAVERSAL_S = 20.0


# =============================================================================
# HELPERS
# =============================================================================

def make_network(feedback_enabled=True):
    """Create 5-synapse network with feedback loop."""
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    params.spine_calcium_feedback = feedback_enabled

    network = MultiSynapseNetwork(
        n_synapses=N_SYNAPSES, pattern="clustered", spacing_um=1.0,
    )
    network.initialize(Model6QuantumSynapse, params)
    for syn in network.synapses:
        syn.set_microtubule_invasion(True)
    network.disable_auto_commitment = True
    return network


def step_network_per_synapse(network, dt, per_syn_stimuli):
    """
    Step each synapse with its own stimulus, then run network-level
    coordination (entanglement tracker + commitment gate).

    per_syn_stimuli: list of dicts, one per synapse.
    The network-level reward flag is True if ANY synapse has reward=True.
    """
    # Step each synapse with individual stimulus
    # (each syn.step() internally tracks _peak_calcium_uM)
    for i, syn in enumerate(network.synapses):
        syn.step(dt, per_syn_stimuli[i])

    # Network-level entanglement (every 10 steps)
    if not hasattr(network, '_entanglement_step_counter'):
        network._entanglement_step_counter = 0
    network._entanglement_step_counter += 1
    if network._entanglement_step_counter % 10 == 0:
        network._network_entanglement = network.entanglement_tracker.step(
            dt, network.synapses, network.positions
        )

    # Coordinated gate on reward steps
    any_reward = any(s.get('reward', False) for s in per_syn_stimuli)
    if any_reward:
        network._evaluate_coordinated_gate({'reward': True})
        # Propagate commitment
        if not network.network_committed:
            if any(getattr(s, '_camkii_committed', False) for s in network.synapses):
                network.network_committed = True

    network.time += dt


def snapshot_per_synapse(network):
    """Capture per-synapse state."""
    rows = []
    for i, syn in enumerate(network.synapses):
        n_dimers = len(syn.dimer_particles.dimers)
        n_bonds = len(syn.dimer_particles.entanglement_bonds)
        ps_mean = syn.get_mean_singlet_probability()
        spine_vol = syn.spine_plasticity.spine_volume
        ampar = syn.spine_plasticity.AMPAR_count
        drive = getattr(syn, '_committed_memory_level', 0.0)
        committed = getattr(syn, '_camkii_committed', False)
        count = getattr(syn, '_committed_dimer_count', 0)
        rows.append({
            'synapse': i,
            'dimers': n_dimers,
            'bonds': n_bonds,
            'ps_mean': ps_mean,
            'spine_vol': spine_vol,
            'ampar': ampar,
            'drive': drive,
            'committed': committed,
            'committed_count': count,
        })
    return rows


def snapshot_network(network):
    """Capture network-level topology."""
    net = network.get_experimental_metrics()
    return {
        'total_dimers': net.get('total_dimers', 0.0),
        'n_entangled': net.get('n_entangled_network', 0),
        'total_bonds': net.get('total_bonds', 0),
        'cross_bonds': net.get('cross_synapse_bonds', 0),
        'n_clusters': net.get('n_clusters', 0),
    }


# =============================================================================
# SINGLE TRAVERSAL
# =============================================================================

def run_traversal(network, traversal_idx):
    """
    Run one traversal of the linear track.

    Synapses activate sequentially (0→4), each for PLACE_FIELD_DURATION_S.
    Dopamine delivered at DOPAMINE_TIME_S.
    3s silence at end of track.
    """
    t = 0.0
    total_steps = int(TRAVERSAL_DURATION_S / DT)
    dopamine_delivered = False

    for step in range(total_steps):
        t = step * DT

        # Which synapse is active? (0-indexed, None if in end silence)
        active_syn = int(t / PLACE_FIELD_DURATION_S)
        if active_syn >= N_SYNAPSES:
            active_syn = None

        # Dopamine at t=3.5s
        is_reward = (not dopamine_delivered and
                     t >= DOPAMINE_TIME_S and t < DOPAMINE_TIME_S + DT * 2)
        if is_reward:
            dopamine_delivered = True

        # Build per-synapse stimuli
        stimuli = []
        for i in range(N_SYNAPSES):
            if i == active_syn:
                # Theta burst voltage for active synapse
                t_in_field = t - i * PLACE_FIELD_DURATION_S
                # Which theta cycle?
                cycle_phase = t_in_field % THETA_PERIOD_S
                if cycle_phase < BURST_ACTIVE_S:
                    t_in_spike = cycle_phase % SPIKE_PERIOD
                    voltage = -10e-3 if t_in_spike < DEPOL_DURATION else -70e-3
                else:
                    voltage = -70e-3
            else:
                voltage = -70e-3

            stimuli.append({'voltage': voltage, 'reward': is_reward})

        step_network_per_synapse(network, DT, stimuli)

    return dopamine_delivered


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(feedback_enabled=True, n_traversals=None):
    if n_traversals is None:
        n_traversals = N_TRAVERSALS
    label = "FEEDBACK ON" if feedback_enabled else "FEEDBACK OFF"
    print(f"\n{'='*110}", flush=True)
    print(f"LINEAR TRACK PLACE FIELD LEARNING — {label}", flush=True)
    print(f"{'='*110}", flush=True)
    print(f"  {n_traversals} traversals, {INTER_TRAVERSAL_S}s gaps, "
          f"dopamine at t={DOPAMINE_TIME_S}s (synapse 3)", flush=True)
    print(f"  Per traversal: {TRAVERSAL_DURATION_S}s "
          f"({N_SYNAPSES}x{PLACE_FIELD_DURATION_S}s fields + {END_SILENCE_S}s silence)",
          flush=True)
    print(f"  Feedback: spine_calcium_feedback={feedback_enabled}", flush=True)
    print(flush=True)

    network = make_network(feedback_enabled=feedback_enabled)

    all_snapshots = []   # per-traversal snapshots
    wall_start = time.time()

    for trav in range(n_traversals):
        t0 = time.time()

        # Reset gate state so each traversal gets a fresh quantum measurement.
        # Spine structural state (volume, AMPAR, actin) is NOT reset —
        # only the commitment flags that gate re-evaluation.
        network._network_measurement_performed = False
        network.network_committed = False
        for syn in network.synapses:
            syn._camkii_committed = False

        run_traversal(network, trav)
        t_trav = time.time() - t0

        # Snapshot after traversal
        syn_snap = snapshot_per_synapse(network)
        net_snap = snapshot_network(network)
        all_snapshots.append({
            'traversal': trav,
            'synapses': syn_snap,
            'network': net_snap,
        })

        # Print per-synapse state
        print(f"  Trav {trav:2d} ({t_trav:5.1f}s wall) | "
              f"net: dim={net_snap['total_dimers']:.0f} "
              f"ent={net_snap['n_entangled']} "
              f"bonds={net_snap['total_bonds']} "
              f"cross={net_snap['cross_bonds']}", flush=True)
        print(f"    {'syn':>3}  {'dim':>5}  {'bonds':>5}  {'P_S':>6}  "
              f"{'spine':>6}  {'AMPAR':>6}  {'drive':>6}  {'count':>5}  {'comm':>4}",
              flush=True)
        for s in syn_snap:
            print(f"    {s['synapse']:>3}  {s['dimers']:>5}  {s['bonds']:>5}  "
                  f"{s['ps_mean']:>6.3f}  {s['spine_vol']:>6.3f}  "
                  f"{s['ampar']:>6.1f}  {s['drive']:>6.3f}  "
                  f"{s['committed_count']:>5}  "
                  f"{'Y' if s['committed'] else 'N':>4}", flush=True)

        # Inter-traversal gap (analytical)
        if trav < n_traversals - 1:
            t0 = time.time()
            analytical_gap(network, INTER_TRAVERSAL_S, dt_sub=1.0)
            t_gap = time.time() - t0

            # Also step spine plasticity forward through the gap
            # (analytical_gap doesn't advance plasticity dynamics)
            for syn in network.synapses:
                drive = getattr(syn, '_committed_memory_level', 0.0)
                ca_uM = 0.05  # baseline during gap
                syn.spine_plasticity.step(
                    INTER_TRAVERSAL_S, drive, ca_uM, quantum_field_kT=0.0
                )

            print(f"    Gap: {t_gap:.1f}s wall | "
                  f"spine_vol=[{', '.join(f'{s.spine_plasticity.spine_volume:.3f}' for s in network.synapses)}]",
                  flush=True)
        print(flush=True)

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    wall_total = time.time() - wall_start
    print(f"{'='*110}", flush=True)
    print(f"SUMMARY — {label}", flush=True)
    print(f"{'='*110}", flush=True)

    final_syn = all_snapshots[-1]['synapses']

    print(f"  {'Syn':>3}  {'Location':>12}  {'spine_vol':>9}  {'AMPAR':>6}  "
          f"{'drive':>6}  {'count':>5}  {'dimers':>6}  {'bonds':>5}  {'comm':>4}",
          flush=True)
    print(f"  {'-'*80}", flush=True)

    locations = ['start', 'early', 'mid', 'REWARD', 'post-reward']
    for s in final_syn:
        print(f"  {s['synapse']:>3}  {locations[s['synapse']]:>12}  "
              f"{s['spine_vol']:>9.3f}  {s['ampar']:>6.1f}  "
              f"{s['drive']:>6.3f}  {s['committed_count']:>5}  "
              f"{s['dimers']:>6}  {s['bonds']:>5}  "
              f"{'Y' if s['committed'] else 'N':>4}", flush=True)
    print(f"  {'-'*80}", flush=True)

    # Learning curve: drive across traversals for each synapse
    print(f"\n  Drive across traversals:", flush=True)
    print(f"  {'Trav':>4}  " + "  ".join(f"{'syn'+str(i):>7}" for i in range(N_SYNAPSES)),
          flush=True)
    print(f"  {'-'*50}", flush=True)
    for snap in all_snapshots:
        trav = snap['traversal']
        drives = [s['drive'] for s in snap['synapses']]
        print(f"  {trav:>4}  " + "  ".join(f"{d:>7.3f}" for d in drives), flush=True)
    print(f"  {'-'*50}", flush=True)

    # Spine volume across traversals
    print(f"\n  Spine volume across traversals:", flush=True)
    print(f"  {'Trav':>4}  " + "  ".join(f"{'syn'+str(i):>7}" for i in range(N_SYNAPSES)),
          flush=True)
    print(f"  {'-'*50}", flush=True)
    for snap in all_snapshots:
        trav = snap['traversal']
        vols = [s['spine_vol'] for s in snap['synapses']]
        print(f"  {trav:>4}  " + "  ".join(f"{v:>7.3f}" for v in vols), flush=True)
    print(f"  {'-'*50}", flush=True)

    print(f"\n  Total wall clock: {wall_total:.0f}s ({wall_total/60:.1f} min)", flush=True)
    print(f"{'='*110}", flush=True)

    return all_snapshots


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--traversals", type=int, default=N_TRAVERSALS)
    args = parser.parse_args()
    run_experiment(feedback_enabled=not args.no_feedback, n_traversals=args.traversals)
