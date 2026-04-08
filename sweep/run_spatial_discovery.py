#!/usr/bin/env python3
"""
Spatial Discovery Experiment
============================
Tests whether quantum-synapse networks can learn goal locations in a 2D
spatial environment through dopamine-gated reinforcement.

Setup:
  - 40 synapses = 40 spatial features (clustered in a 10x10 arena)
  - Agent navigates with heading biased by synaptic strengths
  - Feature activations → graded voltage drive (Gaussian receptive fields)
  - Dopamine delivered on goal arrival
  - Inter-trial gaps advanced analytically
  - spine_calcium_feedback = False (baseline, no compounding)

Prediction:
  - Features near the goal accumulate larger spine volumes
  - Agent trajectory becomes more directed across trials
"""

import sys
import os
import time
import json
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

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from spatial_environment import SpatialEnvironment, Agent


# =============================================================================
# COPIED FROM run_theta_burst_45s.py — analytical_gap
# =============================================================================

def analytical_gap(network, gap_duration_s, dt_sub=1.0, diagnostics=False):
    """
    Advance the network through a silent gap analytically.

    Physics during silence (V = -70 mV, no reward):
      1. P_S decoherence: per-dimer exponential decay toward 0.25
      2. Dissolution: concentration decays at k_diss = k_classical*(1 - singlet_excess)
      3. Particle removal: track concentration, remove lowest-coherence particles
      4. Bond cleanup: remove bonds involving P_S < 0.5 dimers
      5. Stochastic disentanglement: k_decohere = 0.01*(1 - P_Si*P_Sj)

    NOT computed (negligible during silence):
      - Calcium dynamics (already at baseline within ~2s)
      - ATP hydrolysis / phosphate production
      - EM field / tryptophan superradiance
      - New dimer formation (no calcium)
      - New bond formation (no EM field)
      - Network entanglement tracker O(n^2) recalc

    Sub-interval dt_sub controls accuracy of the coupling between
    P_S decay and dissolution rate. 1s is fine for 45s gaps.

    If diagnostics=True, returns a dict with per-stage counts.
    """
    P_THERMAL = 0.25
    K_CLASSICAL = 0.05  # s^-1, bare dissolution rate

    n_subs = int(np.ceil(gap_duration_s / dt_sub))
    actual_dt = gap_duration_s / n_subs

    # Diagnostics counters
    diag = None
    if diagnostics:
        pre_dimers = sum(len(s.dimer_particles.dimers) for s in network.synapses)
        pre_per_syn_bonds = sum(len(s.dimer_particles.entanglement_bonds)
                                for s in network.synapses)
        pre_net_bonds = len(network.entanglement_tracker.entanglement_bonds)
        # P_S distribution of all dimers pre-gap
        pre_ps = []
        for syn in network.synapses:
            for d in syn.dimer_particles.dimers:
                pre_ps.append(d.singlet_probability)
        # Count per-synapse bonds per dimer (connectivity)
        pre_bond_counts = {}
        for syn in network.synapses:
            for b in syn.dimer_particles.entanglement_bonds:
                pre_bond_counts[b.dimer_i] = pre_bond_counts.get(b.dimer_i, 0) + 1
                pre_bond_counts[b.dimer_j] = pre_bond_counts.get(b.dimer_j, 0) + 1

        diag = {
            "pre_dimers": pre_dimers,
            "pre_per_syn_bonds": pre_per_syn_bonds,
            "pre_net_bonds": pre_net_bonds,
            "pre_ps_min": float(np.min(pre_ps)) if pre_ps else 0,
            "pre_ps_max": float(np.max(pre_ps)) if pre_ps else 0,
            "pre_ps_mean": float(np.mean(pre_ps)) if pre_ps else 0,
            "total_dimers_dissolved": 0,
            "dissolved_bond_counts": [],  # per-syn bond count of each dissolved dimer
            "dissolved_ps_values": [],    # P_S of each dissolved dimer at removal
            "bonds_removed_step3": 0,     # per-syn bonds removed with dissolved dimers
            "bonds_removed_step4": 0,     # per-syn bonds removed for P_S < 0.5
            "bonds_removed_step5": 0,     # per-syn bonds removed stochastically
        }

    for sub_idx in range(n_subs):

        # ------------------------------------------------------------------
        # 1. Decay singlet probability per dimer (matches step_coherence)
        # ------------------------------------------------------------------
        for syn in network.synapses:
            frac_P31 = getattr(syn.params.environment, 'fraction_P31', 1.0)
            T_base = frac_P31 * 216.0 + (1.0 - frac_P31) * 0.4

            for dimer in syn.dimer_particles.dimers:
                j_spread = np.std(dimer.j_couplings_intra)
                j_mean = np.abs(np.mean(dimer.j_couplings_intra))
                spread_factor = 1.0 + 2.0 * j_spread / (j_mean + 0.1)
                template_factor = 0.7 if dimer.template_bound else 1.0
                T_eff = max(T_base / (spread_factor * template_factor), 0.1)

                decay = np.exp(-actual_dt / T_eff)
                noise = 1.0 + 0.01 * np.sqrt(actual_dt) * np.random.randn()
                P_excess = dimer.singlet_probability - P_THERMAL
                dimer.singlet_probability = float(np.clip(
                    P_THERMAL + P_excess * decay * noise, P_THERMAL, 1.0
                ))

        # ------------------------------------------------------------------
        # 2. Dissolution of concentration field
        # ------------------------------------------------------------------
        for syn in network.synapses:
            ps_vals = [d.singlet_probability for d in syn.dimer_particles.dimers]
            mean_ps = np.mean(ps_vals) if ps_vals else P_THERMAL

            # Update the chemistry module's knowledge of P_S
            syn.ca_phosphate.dimerization.set_mean_singlet_probability(mean_ps)

            singlet_excess = max(0.0, (mean_ps - P_THERMAL) / 0.75)
            k_diss = K_CLASSICAL * (1.0 - singlet_excess)

            # Exponential decay of concentration
            decay = np.exp(-k_diss * actual_dt)
            syn.ca_phosphate.dimerization.dimer_concentration *= decay
            syn.ca_phosphate.dimerization.trimer_concentration *= np.exp(
                -k_diss * 10.0 * actual_dt
            )
            # Clamp
            syn.ca_phosphate.dimerization.dimer_concentration = np.maximum(
                syn.ca_phosphate.dimerization.dimer_concentration, 0
            )
            syn.ca_phosphate.dimerization.trimer_concentration = np.maximum(
                syn.ca_phosphate.dimerization.trimer_concentration, 0
            )

        # ------------------------------------------------------------------
        # 3. Remove particles that exceed new concentration target
        # ------------------------------------------------------------------
        for syn in network.synapses:
            az_volume_L = 1e-17  # 0.01 um^3
            N_A = 6.022e23
            peak_conc = float(np.max(
                syn.ca_phosphate.dimerization.dimer_concentration
            ))
            target_count = int(round(peak_conc * az_volume_L * N_A))
            current_count = len(syn.dimer_particles.dimers)

            if current_count > target_count:
                n_remove = current_count - target_count
                sorted_dimers = sorted(
                    syn.dimer_particles.dimers,
                    key=lambda d: d.singlet_probability
                )
                for i in range(min(n_remove, len(sorted_dimers))):
                    d = sorted_dimers[i]
                    if diag is not None:
                        diag["total_dimers_dissolved"] += 1
                        diag["dissolved_ps_values"].append(d.singlet_probability)
                        # Count per-syn bonds this dimer participates in
                        n_bonds_this = sum(
                            1 for b in syn.dimer_particles.entanglement_bonds
                            if b.dimer_i == d.id or b.dimer_j == d.id
                        )
                        diag["dissolved_bond_counts"].append(n_bonds_this)
                        bonds_before = len(syn.dimer_particles.entanglement_bonds)
                    syn.dimer_particles.dimers.remove(d)
                    syn.dimer_particles._remove_all_bonds_for_dimer(d.id)
                    if diag is not None:
                        bonds_after = len(syn.dimer_particles.entanglement_bonds)
                        diag["bonds_removed_step3"] += (bonds_before - bonds_after)
                        bonds_before = bonds_after  # for next iteration

        # ------------------------------------------------------------------
        # 4. Remove bonds for decoherent dimers (P_S < 0.5)
        # ------------------------------------------------------------------
        if diag is not None:
            bonds_before_step4 = sum(len(s.dimer_particles.entanglement_bonds)
                                     for s in network.synapses)
        for syn in network.synapses:
            for dimer in syn.dimer_particles.dimers:
                if dimer.singlet_probability < 0.5:
                    syn.dimer_particles._remove_all_bonds_for_dimer(dimer.id)
        if diag is not None:
            bonds_after_step4 = sum(len(s.dimer_particles.entanglement_bonds)
                                    for s in network.synapses)
            diag["bonds_removed_step4"] += (bonds_before_step4 - bonds_after_step4)

        # ------------------------------------------------------------------
        # 5. Stochastic disentanglement (no EM protection at rest)
        # ------------------------------------------------------------------
        if diag is not None:
            bonds_before_step5 = sum(len(s.dimer_particles.entanglement_bonds)
                                     for s in network.synapses)
        for syn in network.synapses:
            # Build dimer lookup for this synapse
            dimer_map = {d.id: d for d in syn.dimer_particles.dimers}
            to_remove = []
            for bond in list(syn.dimer_particles.entanglement_bonds):
                di = dimer_map.get(bond.dimer_i)
                dj = dimer_map.get(bond.dimer_j)
                if di is None or dj is None:
                    to_remove.append((bond.dimer_i, bond.dimer_j))
                    continue
                coh_factor = di.singlet_probability * dj.singlet_probability
                k_decohere = 0.01 * (1.0 - coh_factor)
                p_disentangle = 1.0 - np.exp(-k_decohere * actual_dt)
                if np.random.random() < p_disentangle:
                    to_remove.append((bond.dimer_i, bond.dimer_j))

            for id_i, id_j in to_remove:
                syn.dimer_particles._remove_bond(id_i, id_j)
        if diag is not None:
            bonds_after_step5 = sum(len(s.dimer_particles.entanglement_bonds)
                                    for s in network.synapses)
            diag["bonds_removed_step5"] += (bonds_before_step5 - bonds_after_step5)

    # Refresh network entanglement tracker after dissolutions
    if diag is not None:
        post_dimers = sum(len(s.dimer_particles.dimers) for s in network.synapses)
        post_per_syn_bonds = sum(len(s.dimer_particles.entanglement_bonds)
                                 for s in network.synapses)
        net_bonds_before_refresh = len(network.entanglement_tracker.entanglement_bonds)
        diag["post_dimers"] = post_dimers
        diag["post_per_syn_bonds"] = post_per_syn_bonds
        diag["net_bonds_before_refresh"] = net_bonds_before_refresh

    # Rebuild all_dimers from current (post-dissolution) per-synapse dimer lists
    # With stable IDs (syn_idx, dimer.id), the bond set correctly identifies
    # which dimers still exist — dissolved dimers' IDs are simply absent.
    network.entanglement_tracker.collect_dimers(network.synapses, network.positions)

    if diag is not None:
        new_ids = {d['global_id'] for d in network.entanglement_tracker.all_dimers}
        surviving = {b for b in network.entanglement_tracker.entanglement_bonds
                     if b[0] in new_ids and b[1] in new_ids}
        diag["net_bonds_surviving_id_check"] = len(surviving)
        diag["net_bonds_pruned_by_dissolution"] = net_bonds_before_refresh - len(surviving)

    # Prune stale bonds + check coherence/coupling thresholds (dt=0 -> no new bonds)
    network.entanglement_tracker._update_entanglement(0)

    if diag is not None:
        diag["net_bonds_after_update_ent"] = len(network.entanglement_tracker.entanglement_bonds)

    # Rebuild cluster metrics and update the cache that get_experimental_metrics() reads
    largest_cluster = network.entanglement_tracker._find_largest_cluster()
    entangled_ids = set()
    for bond in network.entanglement_tracker.entanglement_bonds:
        entangled_ids.add(bond[0])
        entangled_ids.add(bond[1])
    n = len(network.entanglement_tracker.all_dimers)
    network._network_entanglement = {
        'n_total_dimers': n,
        'n_entangled_network': len(largest_cluster),
        'n_bonds': len(network.entanglement_tracker.entanglement_bonds),
        'f_entangled': len(entangled_ids) / n if n > 0 else 0.0,
    }

    if diag is not None:
        diag["net_bonds_final_cache"] = network._network_entanglement['n_bonds']

    # Advance network time
    network.time += gap_duration_s

    # One full step to sync all internal state (calcium baseline, etc.)
    network.step(0.001, {"voltage": -70e-3, "reward": False})

    if diag is not None:
        diag["net_bonds_after_final_step"] = (
            network._network_entanglement.get('n_bonds', '?'))
        return diag


# =============================================================================
# COPIED FROM run_place_field_learning.py — step_network_per_synapse
# =============================================================================

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


# =============================================================================
# HELPERS
# =============================================================================

def activations_to_stimuli(activations, reward=False):
    """Convert feature activations to per-synapse stimulus dicts."""
    stimuli = []
    for act in activations:
        if act > 0.05:
            voltage = -70e-3 + act * 60e-3  # 0->-70mV, 1.0->-10mV
        else:
            voltage = -70e-3
        stimuli.append({'voltage': voltage, 'reward': reward})
    return stimuli


def get_synaptic_strengths(network):
    """Extract synaptic strengths from spine volumes (baseline-subtracted)."""
    strengths = np.array([syn.spine_plasticity.spine_volume for syn in network.synapses])
    return np.clip(strengths - 1.0, 0, None)


def make_network(n_synapses=40, feedback_enabled=False):
    """Create n-synapse network for spatial discovery."""
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    params.spine_calcium_feedback = feedback_enabled
    network = MultiSynapseNetwork(
        n_synapses=n_synapses, pattern="clustered", spacing_um=1.0,
    )
    network.initialize(Model6QuantumSynapse, params)
    for syn in network.synapses:
        syn.set_microtubule_invasion(True)
    network.disable_auto_commitment = True
    return network


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_trial(network, env, agent, trial_num, agent_dt=0.5,
              trial_time_budget=90.0, physics_dt=0.005):
    """
    Run one navigation trial.

    Agent moves through the environment; feature activations drive
    synapse voltage; dopamine is delivered on goal arrival.
    """
    physics_steps_per_agent_step = int(agent_dt / physics_dt)
    trajectory = []
    trial_time = 0.0
    found_goal = False

    # Reset per-trial commitment state (same pattern as place field runner)
    for syn in network.synapses:
        syn._camkii_committed = False
    network.network_committed = False

    while trial_time < trial_time_budget:
        # Record position
        trajectory.append(agent.position.copy())

        # Get activations from current position
        activations = env.get_activations(agent.position)

        # Build stimuli and step physics — only active synapses
        stimuli = activations_to_stimuli(activations)
        active_mask = activations > 0.05

        for _ in range(physics_steps_per_agent_step):
            # Step only active synapses
            for i, syn in enumerate(network.synapses):
                if active_mask[i]:
                    syn.step(physics_dt, stimuli[i])

            # Entanglement tracker still runs on all synapses (every 10th step)
            if not hasattr(network, '_entanglement_step_counter'):
                network._entanglement_step_counter = 0
            network._entanglement_step_counter += 1
            if network._entanglement_step_counter % 10 == 0:
                network._network_entanglement = network.entanglement_tracker.step(
                    physics_dt, network.synapses, network.positions
                )
            network.time += physics_dt

        # Move agent
        strengths = get_synaptic_strengths(network)
        agent.step(agent_dt, env, strengths)

        # Check goal
        if env.check_goal(agent.position):
            # Deliver dopamine — one reward step
            reward_activations = env.get_activations(agent.position)
            reward_stimuli = activations_to_stimuli(reward_activations, reward=True)
            step_network_per_synapse(network, physics_dt, reward_stimuli)
            found_goal = True
            trajectory.append(agent.position.copy())
            break

        trial_time += agent_dt

    # Collect per-synapse state
    dimer_counts = [len(syn.dimer_particles.dimers) for syn in network.synapses]
    spine_volumes = [syn.spine_plasticity.spine_volume for syn in network.synapses]

    return {
        'trial': trial_num,
        'found_goal': found_goal,
        'time_to_goal': trial_time if found_goal else None,
        'trajectory': [(p[0], p[1]) for p in trajectory],
        'dimer_counts': dimer_counts,
        'spine_volumes': spine_volumes,
        'total_dimers': sum(dimer_counts),
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(n_trials=25, seed=42):
    """Run the full spatial discovery experiment."""
    rng = np.random.default_rng(seed)
    env = SpatialEnvironment(seed=seed)
    network = make_network(n_synapses=env.n_features)
    agent = Agent()

    # Print environment summary
    feature_goal_dists = env.get_feature_goal_distances()
    print(f"Environment: {env.n_features} features, "
          f"goal at ({env.goal_center[0]:.2f}, {env.goal_center[1]:.2f})")
    print(f"Nearest feature to goal: {feature_goal_dists.min():.2f} units")
    print(f"Farthest feature from goal: {feature_goal_dists.max():.2f} units")
    print()

    all_trials = []

    for trial in range(n_trials):
        t_start = time.time()

        # Reset agent position
        agent.reset(env.size, rng)

        # Run trial
        result = run_trial(network, env, agent, trial)
        wall_time = time.time() - t_start

        status = (f"FOUND t={result['time_to_goal']:.1f}s"
                  if result['found_goal'] else "FAILED")
        print(f"Trial {trial:2d}: {status:20s} | "
              f"dimers={result['total_dimers']:4d} | "
              f"spines=[{min(result['spine_volumes']):.2f}, "
              f"{max(result['spine_volumes']):.2f}] | "
              f"wall={wall_time:.0f}s")

        # Inter-trial gap
        analytical_gap(network, 30.0)

        all_trials.append(result)

    # Final summary: top 10 features by spine volume
    final_volumes = np.array(all_trials[-1]['spine_volumes'])
    top10 = np.argsort(final_volumes)[-10:][::-1]
    print(f"\nTop 10 features by spine volume (with distance to goal):")
    for idx in top10:
        print(f"  Feature {idx:2d}: vol={final_volumes[idx]:.3f}, "
              f"dist_to_goal={feature_goal_dists[idx]:.2f}")

    return all_trials, env


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == '__main__':
    # Start with just 3 trials to verify the plumbing works
    all_trials, env = run_experiment(n_trials=3, seed=42)
