#!/usr/bin/env python3
"""
Observe readout (reward/measurement) after selective drive.

Two spatial clusters of 3 synapses, separated by 15 um (>> coupling_length 5 um)
so backbone aggregation is LOCAL within each cluster. All synapses mt_invaded=True
to bypass the invasion hard-gate; eta (Frohlich threshold) is the ONLY selectivity.

Two conditions (each a fresh network):
  (A)    Drive cluster A only (syn 0-2 at -10 mV, syn 3-5 at -70 mV) for 15 s
  (BOTH) Drive all 6 at -10 mV for 15 s

After 15 s of drive, a one-shot reward/readout block fires:
  - One extra step under the same stimulus (keeps calcium elevated)
  - Snapshot partition_before (connected components just before collapse)
  - _evaluate_coordinated_gate({'reward': True}) — quantum measurement
  - Record which synapses' gates opened, dimer counts, cluster stats

Per-step during drive: step each synapse individually, then backbone update,
then tracker step (every 10 steps). Log partition every 5 s.
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

N_SYN = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_network(params):
    """Create fresh 6-synapse network with two-cluster geometry."""
    network = MultiSynapseNetwork(
        n_synapses=N_SYN, pattern="clustered", spacing_um=1.0,
        coupling_length_um=5.0,
    )
    network.initialize(Model6QuantumSynapse, params)

    # Custom two-cluster geometry: 0-2 at x~0, 3-5 at x~15 um
    positions = np.zeros((N_SYN, 3))
    for i in range(3):
        positions[i] = [0.0 + np.random.randn() * 0.3,
                        np.random.randn() * 0.3,
                        np.random.randn() * 0.3]
    for i in range(3, N_SYN):
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


def run_condition(label, params, stimuli, T_drive=15.0, dt=0.005, log_interval=5.0):
    """Run one condition: drive phase + one-shot reward readout."""
    network = make_network(params)
    tracker = network.entanglement_tracker

    # Print coupling weights once (first condition only)
    if label == "A":
        print("Coupling weights (intra ~0.9, inter ~0.05):")
        w = network.coupling_weights
        for i in range(N_SYN):
            print("  " + "  ".join(f"{w[i,j]:.3f}" for j in range(N_SYN)))
        print()

    print(f"{'='*110}")
    print(f"CONDITION {label}: {{'A': 'syn 0-2 driven, 3-5 rest', 'BOTH': 'all 6 driven'}[label]}")
    print(f"{'='*110}")

    header = (f"{'t':>5s}  "
              + "  ".join(f"{'r'+str(i):>5s}" for i in range(N_SYN))
              + f"  {'n_cond':>6s}  "
              + f"{'cross':>5s} {'intra':>5s}  "
              + f"{'comps':>5s} {'lg':>3s}  {'partition':s}")
    print(header)
    print("-" * len(header))

    # =========================================================================
    # DRIVE PHASE (15 s)
    # =========================================================================
    steps = int(round(T_drive / dt))
    next_log = 0.0
    t = 0.0

    for step_i in range(steps):
        # Step each synapse with its own stimulus
        for i, syn in enumerate(network.synapses):
            syn.step(dt, stimuli[i])

        # Backbone update (sets per-synapse eta)
        if (network.params is not None
                and hasattr(network.params, 'dendritic_backbone')
                and network.params.dendritic_backbone.enabled):
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

            tracker.collect_dimers(network.synapses, network.positions)
            syn_comps = get_synapse_components(tracker)

            n_comps = len(syn_comps)
            largest = max((len(c) for c in syn_comps), default=0)

            partition_str = str(sorted([sorted(c) for c in syn_comps]))
            if not syn_comps:
                partition_str = "[]"

            r_strs = " ".join(f"{r:5.3f}" for r in rs)
            print(f"{t:5.1f}  {r_strs}  {n_condensed:6d}  "
                  f"{n_cross:5d} {n_intra:5d}  "
                  f"{n_comps:5d} {largest:3d}  {partition_str}")

            next_log += log_interval

    # =========================================================================
    # REWARD / READOUT BLOCK (one-shot, after drive)
    # =========================================================================
    print()
    print(f"--- REWARD READOUT (t={t:.1f} s) ---")

    # (3a) One extra step under drive stimulus — keeps calcium elevated
    for i, syn in enumerate(network.synapses):
        syn.step(dt, stimuli[i])
    if (network.params is not None
            and hasattr(network.params, 'dendritic_backbone')
            and network.params.dendritic_backbone.enabled):
        network._update_backbone_field()
    tracker.step(dt, network.synapses, network.positions,
                 coupling_weights=network.coupling_weights)
    t += dt
    network.time = t

    # (3b) Snapshot partition before collapse
    tracker.collect_dimers(network.synapses, network.positions)
    partition_before = get_synapse_components(tracker)

    # _peak_calcium_uM is a running max set by syn.step() (model6_core.py:327).
    # After 15 s of drive at -10 mV it will be well above 0.5 µM for driven
    # synapses. No explicit wiring needed — just verify and report.
    print("  Per-synapse _peak_calcium_uM:")
    for i, syn in enumerate(network.synapses):
        pca = getattr(syn, '_peak_calcium_uM', 0.0)
        print(f"    syn {i}: {pca:.3f} uM"
              f"{'  (BELOW 0.5 threshold!)' if pca <= 0.5 else ''}")

    # (3c) Allow a fresh measurement
    network._network_measurement_performed = False

    # (3d) Fire the coordinated gate — quantum measurement + gate evaluation
    network._evaluate_coordinated_gate({'reward': True})

    # (3e) Read commit pattern
    commit_pattern = []
    for i, syn in enumerate(network.synapses):
        opened = bool(getattr(syn, '_measurement_gate_opened', False))
        dimer_count = int(getattr(syn, '_measurement_dimer_count', 0))
        commit_pattern.append((i, opened, dimer_count))

    last_measurement = getattr(tracker, '_last_measurement', {})

    # (3f) Return all three
    return partition_before, commit_pattern, last_measurement


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
params = Model6Parameters()
params.em_coupling_enabled = True

STIM_DRIVE = {'voltage': -10e-3, 'reward': False}
STIM_REST  = {'voltage': -70e-3, 'reward': False}

print()
print("#" * 110)
print("# READOUT-LOOP OBSERVATION: two-cluster geometry, selective drive + one-shot reward")
print(f"# {N_SYN} synapses: cluster A = {{0,1,2}} at x~0, cluster B = {{3,4,5}} at x~15 um")
print("# coupling_length = 5.0 um => intra-cluster w~0.9, inter-cluster w~0.05")
print("# All mt_invaded=True; eta is the only selectivity for cross-synapse bonds")
print("# Drive 15 s, then one-shot reward/readout via _evaluate_coordinated_gate")
print("#" * 110)
print()

# Condition A: drive cluster A only
stimuli_A = [STIM_DRIVE]*3 + [STIM_REST]*3
part_before_A, commit_A, meas_A = run_condition("A", params, stimuli_A)

print()

# Condition BOTH: drive all
stimuli_BOTH = [STIM_DRIVE]*N_SYN
part_before_BOTH, commit_BOTH, meas_BOTH = run_condition("BOTH", params, stimuli_BOTH)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def fmt_partition(syn_comps):
    if not syn_comps:
        return "[] (no cross-synapse bonds)"
    return str(sorted([sorted(c) for c in syn_comps]))


def fmt_commit(commit_pattern):
    lines = []
    for (i, opened, dc) in commit_pattern:
        tag = "OPENED" if opened else "closed"
        lines.append(f"    syn {i}: gate={tag}, committed_dimers={dc}")
    return "\n".join(lines)


print()
print("=" * 110)
print("FINAL SUMMARY")
print("=" * 110)

for label, part_before, commit, meas in [
    ("A", part_before_A, commit_A, meas_A),
    ("BOTH", part_before_BOTH, commit_BOTH, meas_BOTH),
]:
    print(f"\n  CONDITION {label}:")
    print(f"    Partition before collapse: {fmt_partition(part_before)}")
    print(f"    Gate commit pattern:")
    print(fmt_commit(commit))
    n_clust = meas.get('n_clusters_measured', '?')
    n_clust_s = meas.get('n_clusters_singlet', '?')
    cc = meas.get('committed_counts', None)
    cc_str = [f"{c:.0f}" for c in cc] if cc is not None else '?'
    print(f"    Measurement: {n_clust} clusters measured, "
          f"{n_clust_s} collapsed to singlet")
    print(f"    committed_counts per synapse: {cc_str}")

print()
print("EXPECTED (condition A): partition_before has components within {0,1,2} only;")
print("  gate opens for driven synapses (0,1,2) if they have committed dimers + calcium.")
print("  Resting synapses (3,4,5) should have gate=closed (no calcium / no dimers).")
print()
print("EXPECTED (condition BOTH): partition may span both clusters;")
print("  all 6 synapses could have gate open if they have dimers + calcium.")
print()
print("Done.")
