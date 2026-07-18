#!/usr/bin/env python3
"""
DIAGNOSTIC PROBE v2 — force goal arrival on every trial so the reward path
actually fires, then count per-trial: quantum measurements, latch early-returns,
gate_opened / committed / molecular_memory / spine_volume per synapse.

Deviations from the real run (stated in the report):
  - 4 synapses (real: 20), 3 trials (real: 5), trial budget 12s (real: 60s),
    inter-trial gap 5s (real: 30s).
  - Agent walks deterministically straight at the goal (real: stochastic
    heading). Navigation is not what's under test; reward delivery is.
  - Features placed ON the approach path so calcium/dimers build before reward.
"""
import sys, os, json
import numpy as np

WT = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6"
sys.path.insert(0, os.path.join(WT, "src", "models", "Model_6"))
sys.path.insert(0, os.path.join(WT, "sweep"))

import logging
logging.disable(logging.INFO)

import run_spatial_discovery as RSD
from spatial_environment import SpatialEnvironment, Agent

C = {}
def reset_counters(t):
    C.clear(); C.update(dict(trial=t, gate_calls=0, gate_latch_early_return=0,
                             gate_body_ran=0, pqm_calls=0, gate_opened_newly_set=0))

def install(network):
    Tr = type(network.entanglement_tracker)
    orig_pqm = Tr.perform_quantum_measurement
    def pqm(self, syns):
        C['pqm_calls'] += 1
        return orig_pqm(self, syns)
    Tr.perform_quantum_measurement = pqm

    N = type(network)
    orig_gate = N._evaluate_coordinated_gate
    def gate(self, stim):
        C['gate_calls'] += 1
        if not stim.get('reward', False):
            return orig_gate(self, stim)
        if getattr(self, '_coordinated_measurement_performed', False):
            C['gate_latch_early_return'] += 1
            return orig_gate(self, stim)
        C['gate_body_ran'] += 1
        pre = [getattr(s, '_measurement_gate_opened', False) for s in self.synapses]
        r = orig_gate(self, stim)
        post = [getattr(s, '_measurement_gate_opened', False) for s in self.synapses]
        C['gate_opened_newly_set'] += sum(1 for a, b in zip(pre, post) if (not a) and b)
        return r
    N._evaluate_coordinated_gate = gate


class StraightAgent(Agent):
    def __init__(self, goal, start_offset=2.4):
        super().__init__()
        self.goal = np.asarray(goal, float)
        self.start_offset = start_offset
    def reset(self, env_size, rng):
        self.position = self.goal - np.array([self.start_offset, 0.0])
        self.heading = 0.0
    def step(self, dt, env, strengths):
        d = self.goal - self.position
        n = np.linalg.norm(d)
        if n > 1e-9:
            self.position = self.position + (d / n) * self.speed * dt
        return self.position.copy()


def snap(network, tag):
    return dict(tag=tag,
        net_latch=bool(network._coordinated_measurement_performed),
        net_time=round(float(network.time), 3),
        syn=[dict(i=i,
             gate_opened=bool(getattr(s, '_measurement_gate_opened', False)),
             meas_time=getattr(s, '_measurement_time', None),
             committed=bool(getattr(s, '_camkii_committed', False)),
             mol_mem=round(float(s.camkii.molecular_memory), 4),
             plast_gate=bool(getattr(s, '_plasticity_gate', False)),
             peak_ca=round(float(getattr(s, '_peak_calcium_uM', 0.0)), 4),
             n_dimers=len(s.dimer_particles.dimers),
             spine_vol=round(float(s.spine_plasticity.spine_volume), 4),
             sp_time=round(float(s.spine_plasticity.time), 4),
             sp_tat=round(float(s.spine_plasticity.time_above_threshold), 4),
        ) for i, s in enumerate(network.synapses)])


def main():
    N_SYN, N_TRIALS, BUDGET, GAP, SEED = 4, 3, 14.0, 5.0, 42
    env = SpatialEnvironment(n_features=N_SYN, seed=SEED, size=10.0,
                             goal_center=(5.0, 5.0), goal_radius=0.35)
    # put features on the straight approach path (x from 2.6 -> 5.0 at y=5)
    env.feature_positions = np.array([[3.0, 5.0], [3.6, 5.0],
                                      [4.2, 5.0], [4.8, 5.0]])
    network = RSD.make_network(n_synapses=N_SYN, seed=SEED)
    agent = StraightAgent(env.goal_center, start_offset=2.4)
    install(network)

    log = []
    for trial in range(N_TRIALS):
        reset_counters(trial)
        agent.reset(env.size, None)
        res = RSD.run_trial(network, env, agent, trial, trial_time_budget=BUDGET)
        rec = dict(counters=dict(C), found_goal=res['found_goal'],
                   snap=snap(network, f"END_TRIAL_{trial}"))
        log.append(rec)
        print("### TRIAL", trial, json.dumps(rec), flush=True)
        RSD.analytical_gap(network, GAP)
        for r in network.presynaptic_release:
            r.advance_silent(GAP)
        print("### AFTER_GAP", trial, json.dumps(snap(network, f"AFTER_GAP_{trial}")), flush=True)

    print("\n=== FINAL ===")
    print("network.time =", network.time)
    for i, s in enumerate(network.synapses):
        print(f"syn{i}: sp.time={s.spine_plasticity.time:.4f} vol={s.spine_plasticity.spine_volume:.4f} "
              f"gate_opened={getattr(s,'_measurement_gate_opened',False)} "
              f"committed={getattr(s,'_camkii_committed',False)} mol_mem={s.camkii.molecular_memory:.4f}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe2_out.json'), 'w') as f:
        json.dump(log, f, indent=1)

if __name__ == '__main__':
    main()
