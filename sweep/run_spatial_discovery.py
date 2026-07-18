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
from presynaptic_release import PresynapticRelease


# =============================================================================
# analytical_gap — IMPORTED, not copied (MO ruling 001, 2026-07-18)
# =============================================================================
# This file previously carried its own 252-line copy of analytical_gap, marked
# "COPIED FROM run_theta_burst_45s.py". A difflib comparison of the two bodies
# found them byte-identical but for one Unicode arrow in a comment -- there was
# no divergence to preserve, only a second place for a fix to miss.
#
# That miss already happened once: substrate-audit item 16 records "the
# 2026-07-18 fix covered sweep/run_spatial_discovery.py only -- a gap in that
# fix". Consolidating to ONE definition is what makes that shape structurally
# unable to recur, which is why the MO preferred it to patching both copies.
#
# The surviving definition is src/models/Model_6/sweep/run_theta_burst_45s.py.
# run_place_field_learning.py already imported it from there; this file now does
# too, so there are three consumers and one definition.
from sweep.run_theta_burst_45s import analytical_gap

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

    # Backbone condensation update — must run after per-synapse step
    # (so calcium/MT state is fresh) and before the entanglement tracker.
    if network.params is not None and hasattr(network.params, 'dendritic_backbone') and network.params.dendritic_backbone.enabled:
        network._update_backbone_field()

    # Network-level entanglement (every 10 steps)
    if not hasattr(network, '_entanglement_step_counter'):
        network._entanglement_step_counter = 0
    network._entanglement_step_counter += 1
    if network._entanglement_step_counter % 10 == 0:
        network._network_entanglement = network.entanglement_tracker.step(
            dt, network.synapses, network.positions,
            coupling_weights=getattr(network, 'coupling_weights', None),
        )

    # Coordinated gate — called EVERY step, not only reward steps. The gate does its
    # own reward check; it must also see the falling edge of reward to re-arm its
    # one-shot measurement latch, otherwise the measurement fires once per experiment
    # rather than once per reward episode (D19).
    any_reward = any(s.get('reward', False) for s in per_syn_stimuli)
    network._evaluate_coordinated_gate({'reward': any_reward})
    if any_reward:
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
            # Synaptic-input depolarization is SUBTHRESHOLD. Established BTSP
            # grounding (input-engine sessions): local synaptic regime ~-60..-40 mV;
            # the plateau (~-20 mV, a SEPARATE instructive event) is not this knob.
            # Peak activation maps to the top of the subthreshold band, -40 mV.
            # Was -10 mV, which illegally merged the plateau into the synaptic knob.
            voltage = -70e-3 + act * 30e-3  # 0->-70mV (rest), 1.0->-40mV (subthreshold peak)
        else:
            voltage = -70e-3
        stimuli.append({'voltage': voltage, 'reward': reward})
    return stimuli


def get_synaptic_strengths(network):
    """Extract synaptic strengths from spine volumes (baseline-subtracted)."""
    strengths = np.array([syn.spine_plasticity.spine_volume for syn in network.synapses])
    return np.clip(strengths - 1.0, 0, None)


def make_network(n_synapses=40, feedback_enabled=False, seed=0):
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
    # Per-synapse presynaptic stochastic release (stimulus-construction layer).
    # Independent reproducible RNG stream per synapse derived from `seed`.
    _release_seeds = np.random.SeedSequence(seed).spawn(n_synapses)
    network.presynaptic_release = [PresynapticRelease(seed=s) for s in _release_seeds]
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
            # Step presynaptic release for ALL synapses so RRP/facilitation recover
            # even while a synapse is silent; inject the cleft event as glutamate for
            # active synapses, and step only active synapses.
            for i, syn in enumerate(network.synapses):
                glu_event = network.presynaptic_release[i].step(activations[i], physics_dt)
                if active_mask[i]:
                    stimuli[i]['glutamate'] = glu_event
                    syn.step(physics_dt, stimuli[i])

            # Entanglement tracker still runs on all synapses (every 10th step)
            if not hasattr(network, '_entanglement_step_counter'):
                network._entanglement_step_counter = 0
            network._entanglement_step_counter += 1
            if network._entanglement_step_counter % 10 == 0:
                # coupling_weights MUST be passed: _update_entanglement early-returns
                # without it (multi_synapse_network.py ~:277), so omitting it meant NO
                # cross-synapse bonds formed during a trial at all — the entanglement
                # topology, which is supposed to BE the eligibility trace, was built
                # only on the single reward step that routes through
                # step_network_per_synapse. See research log D21 (construct-validity #5).
                network._network_entanglement = network.entanglement_tracker.step(
                    physics_dt, network.synapses, network.positions,
                    coupling_weights=getattr(network, 'coupling_weights', None)
                )
            network.time += physics_dt

        # Move agent
        strengths = get_synaptic_strengths(network)
        agent.step(agent_dt, env, strengths)

        step_count = int(trial_time / agent_dt)
        if step_count % 10 == 0:
            n_active = int(np.sum(activations > 0.05))
            total_dimers = sum(len(syn.dimer_particles.dimers) for syn in network.synapses)
            print(f"  step {step_count:3d} t={trial_time:5.1f}s | pos=({agent.position[0]:.1f},{agent.position[1]:.1f}) | active={n_active} | dimers={total_dimers}", flush=True)

        # Check goal
        if env.check_goal(agent.position):
            # Deliver dopamine — one reward step
            reward_activations = env.get_activations(agent.position)
            reward_stimuli = activations_to_stimuli(reward_activations, reward=True)
            for i in range(len(network.synapses)):
                reward_stimuli[i]['glutamate'] = network.presynaptic_release[i].step(
                    reward_activations[i], physics_dt)
                # WIRE 3 (2026-07-18): the dendrite-wide instructive PLATEAU, delivered
                # to EVERY synapse — not only the active ones — because it is a dendritic
                # event, not a synaptic one (model6-input-engine BTSP section; Jain 2024
                # measures it as dendritic and non-synapse-specific). Read at
                # model6_core.py:647 to trigger DDSC. No driver had ever set it, so the
                # DDSC path — the model's whole delayed-commitment mechanism — had never
                # fired in any learning run.
                reward_stimuli[i]['plateau_potential'] = True
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

def run_experiment(n_trials=25, seed=42, n_features=40, trial_time_budget=90.0):
    """Run the full spatial discovery experiment."""
    rng = np.random.default_rng(seed)
    env = SpatialEnvironment(n_features=n_features, seed=seed)
    network = make_network(n_synapses=env.n_features, seed=seed)
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
        result = run_trial(network, env, agent, trial, trial_time_budget=trial_time_budget)
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
        # Advance presynaptic release through the inter-trial silence (RRP recovers,
        # facilitation relaxes) — the analytical gap does not touch these modules.
        for _r in network.presynaptic_release:
            _r.advance_silent(30.0)

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
    all_trials, env = run_experiment(n_trials=5, n_features=20, trial_time_budget=60.0, seed=42)
