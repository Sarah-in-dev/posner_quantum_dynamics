#!/usr/bin/env python3
"""
Observe cross-synapse bond partition under selective drive.

Two spatial clusters of 4 synapses, separated by 15 um (>> coupling_length 5 um)
so backbone aggregation is LOCAL within each cluster. All synapses mt_invaded=True
to bypass the invasion hard-gate; eta (Frohlich threshold) is the ONLY selectivity.

Three conditions (each a fresh network, 30 s, no reward):
  (A) Drive cluster A only (syn 0-3 at -10 mV, syn 4-7 at -70 mV)
  (B) Drive cluster B only (syn 4-7 at -10 mV, syn 0-3 at -70 mV)
  (BOTH) Drive all 8 at -10 mV

Per-step: step each synapse individually, then backbone update, then tracker step.
Log every 5 s: per-synapse r/eta, bond counts, synapse-level components.
"""

import sys, os
import logging
import numpy as np

# Suppress model logging
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

from model6_parameters import Model6Parameters, P_BASAL_W, compute_metabolic_power, bose_einstein_occupation
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork

hbar = 1.0545718e-34  # J·s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_network(params):
    """Create fresh 8-synapse network with two-cluster geometry."""
    network = MultiSynapseNetwork(
        n_synapses=8, pattern="clustered", spacing_um=1.0,
        coupling_length_um=5.0,
    )
    network.initialize(Model6QuantumSynapse, params)

    # Custom two-cluster geometry: 0-3 at x~0, 4-7 at x~15 um
    positions = np.zeros((8, 3))
    for i in range(4):
        positions[i] = [0.0 + np.random.randn() * 0.3,
                        np.random.randn() * 0.3,
                        np.random.randn() * 0.3]
    for i in range(4, 8):
        positions[i] = [15.0 + np.random.randn() * 0.3,
                        np.random.randn() * 0.3,
                        np.random.randn() * 0.3]

    network.positions = positions
    network.distances = network._compute_distances()
    network.coupling_weights = network._compute_coupling_weights()

    # All invaded — bypass invasion hard-gate so eta is the only selectivity
    for s in network.synapses:
        s.set_microtubule_invasion(True)

    return network


def get_synapse_components(tracker):
    """Extract synapse-index connected components from the tracker."""
    clusters = tracker._find_all_clusters()
    syn_components = []
    for cluster in clusters:
        syn_indices = {gid[0] for gid in cluster}
        syn_components.append(syn_indices)
    # Deduplicate (multiple dimer-level components within one synapse merge)
    # Actually each component is already a connected component; just return as-is
    return syn_components


def count_bond_types(tracker):
    """Count cross-synapse vs intra-synapse bonds."""
    n_cross = len(tracker.cross_synapse_bonds)
    n_intra = len(tracker.intra_synapse_bonds_cache)
    return n_cross, n_intra


def compute_r_eta(network):
    """Compute per-synapse r and eta (same logic as _update_backbone_field)."""
    bp = network.params.dendritic_backbone
    omega_ang = 2.0 * np.pi * bp.omega_0
    n_bar = bose_einstein_occupation(bp.omega_0)
    P_c = n_bar * hbar * omega_ang**2 / bp.Q

    p_met = np.array([
        compute_metabolic_power(
            getattr(s.spine_plasticity, 'E_invasion', 0.0),
            s.calcium.channels.get_open_fraction(),
            bp.p_active_max_W,
        ) for s in network.synapses
    ])
    p_active = p_met - P_BASAL_W
    p_met_agg = P_BASAL_W + network.coupling_weights @ p_active

    rs = p_met_agg / P_c
    etas = np.array([(r - 1.0) / (r + 1.0) if r >= 1.0 else 0.0 for r in rs])
    return rs, etas, P_c


def run_condition(label, params, stimuli, T_total=30.0, dt=0.005, log_interval=5.0):
    """Run one condition and return final partition."""
    network = make_network(params)
    tracker = network.entanglement_tracker

    # Print coupling weights once (first condition only prints)
    if label == "A":
        print("Coupling weights (intra ~0.9, inter ~0.05):")
        w = network.coupling_weights
        for i in range(8):
            print("  " + "  ".join(f"{w[i,j]:.3f}" for j in range(8)))
        print()

    print(f"{'='*100}")
    print(f"CONDITION {label}: {['syn 0-3 driven, 4-7 rest',
                                   'syn 0-3 rest, 4-7 driven',
                                   'all 8 driven'][['A','B','BOTH'].index(label)]}")
    print(f"{'='*100}")

    header = (f"{'t':>5s}  "
              f"{'r0':>5s} {'r1':>5s} {'r2':>5s} {'r3':>5s} "
              f"{'r4':>5s} {'r5':>5s} {'r6':>5s} {'r7':>5s}  "
              f"{'n_cond':>6s}  "
              f"{'cross':>5s} {'intra':>5s}  "
              f"{'comps':>5s} {'lg':>3s}  {'partition':s}")
    print(header)
    print("-" * len(header))

    steps = int(round(T_total / dt))
    next_log = 0.0
    t = 0.0
    final_partition = []

    for step_i in range(steps):
        # Step each synapse with its own stimulus
        for i, syn in enumerate(network.synapses):
            syn.step(dt, stimuli[i])

        # Backbone update (sets per-synapse eta)
        if network.params is not None and hasattr(network.params, 'dendritic_backbone') and network.params.dendritic_backbone.enabled:
            network._update_backbone_field()

        # Entanglement tracker (every 10 steps, matching network.step pattern)
        if not hasattr(network, '_ent_counter'):
            network._ent_counter = 0
        network._ent_counter += 1
        if network._ent_counter % 10 == 0:
            tracker.step(dt * 10, network.synapses, network.positions,
                         coupling_weights=network.coupling_weights)

        t += dt
        network.time = t

        if t >= next_log - dt / 2:
            rs, etas, P_c = compute_r_eta(network)
            n_condensed = int(np.sum(etas > 0))
            n_cross, n_intra = count_bond_types(tracker)

            # Ensure collect_dimers is fresh for component extraction
            tracker.collect_dimers(network.synapses, network.positions)
            syn_comps = get_synapse_components(tracker)

            # Merge components that share synapse indices (shouldn't happen, but safe)
            n_comps = len(syn_comps)
            largest = max((len(c) for c in syn_comps), default=0)

            # Format partition as sorted list of sorted sets
            partition_str = str(sorted([sorted(c) for c in syn_comps]))
            if not syn_comps:
                partition_str = "[]"

            r_strs = " ".join(f"{r:5.3f}" for r in rs)
            print(f"{t:5.1f}  {r_strs}  {n_condensed:6d}  "
                  f"{n_cross:5d} {n_intra:5d}  "
                  f"{n_comps:5d} {largest:3d}  {partition_str}")

            final_partition = syn_comps
            next_log += log_interval

    print()
    return final_partition


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
params = Model6Parameters()
params.em_coupling_enabled = True

STIM_DRIVE = {'voltage': -10e-3, 'reward': False}
STIM_REST  = {'voltage': -70e-3, 'reward': False}

print()
print("#" * 100)
print("# NETWORK PARTITION OBSERVATION: two-cluster geometry, selective drive")
print("# 8 synapses: cluster A = {0,1,2,3} at x~0, cluster B = {4,5,6,7} at x~15 um")
print("# coupling_length = 5.0 um => intra-cluster w~0.9, inter-cluster w~0.05")
print("# All mt_invaded=True; eta is the only selectivity for cross-synapse bonds")
print("#" * 100)
print()

# Condition A: drive cluster A only
stimuli_A = [STIM_DRIVE]*4 + [STIM_REST]*4
part_A = run_condition("A", params, stimuli_A)

# Condition B: drive cluster B only
stimuli_B = [STIM_REST]*4 + [STIM_DRIVE]*4
part_B = run_condition("B", params, stimuli_B)

# Condition BOTH: drive all
stimuli_BOTH = [STIM_DRIVE]*8
part_BOTH = run_condition("BOTH", params, stimuli_BOTH)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def fmt_partition(syn_comps):
    if not syn_comps:
        return "[] (no cross-synapse bonds)"
    return str(sorted([sorted(c) for c in syn_comps]))

print()
print("=" * 100)
print("FINAL PARTITIONS (synapse-index connected components)")
print("=" * 100)
print(f"  (A) Cluster A driven:  {fmt_partition(part_A)}")
print(f"  (B) Cluster B driven:  {fmt_partition(part_B)}")
print(f"  (BOTH) All driven:     {fmt_partition(part_BOTH)}")
print()
print("EXPECTED if selective: A has components within {0,1,2,3} only,")
print("  B within {4,5,6,7} only, BOTH has components spanning both clusters.")
print()
print("Done.")
