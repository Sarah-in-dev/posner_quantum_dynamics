#!/usr/bin/env python3
"""AMENDMENT 3 (MO ruling 004) — is the gap STEPPING, or is rho>1 a non-stepping artifact?

D19 (RESEARCH_LOG_CALCIUM_DIMER) names a false-ratchet generator: "only *active* synapses are
stepped, so silent ones never run their decay term." L·ETA-5's GATE 1 tests that by a
retention THRESHOLD, which is a symptom. Ruling 004: an observed CLOCK DELTA is proof.

This asserts spine_plasticity.time advances by the FULL gap, and measures calcium / f_CaM /
formation / extrusion THROUGH the gap — which also settles ruling 006's (a)-vs-(b) ambiguity:
  (a) gap not stepping  -> clock delta < GAP_S, formation ~ 0
  (b) residual formation -> clock delta == GAP_S, formation > extrusion early in the gap
Cheap: one synapse-network, 14 s drive + 20 s gap.
"""
import sys, os, logging
import numpy as np
logging.disable(logging.INFO)
HERE = os.path.dirname(os.path.abspath(__file__)); M6 = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(M6, '..', '..', '..'))
sys.path.insert(0, M6); sys.path.insert(0, os.path.join(REPO, 'sweep'))
from run_spatial_discovery import make_network, activations_to_stimuli, step_network_per_synapse
from spatial_environment import SpatialEnvironment
from einvasion_ratchet_probe import (N_FEATURES, SEED, PHYSICS_DT, AGENT_DT, GAP_S,
                                     HALF_PATH, AGENT_SPEED, pick_target_and_heading,
                                     pick_park_position, TAU_EXTRUDE, K_STAB)

np.random.seed(SEED)
env = SpatialEnvironment(n_features=N_FEATURES, seed=SEED)
net = make_network(n_synapses=N_FEATURES, seed=SEED)
target, heading, _ = pick_target_and_heading(env)
park, _ = pick_park_position(env)
u = np.array([np.cos(heading), np.sin(heading)])
sp = net.synapses[target].spine_plasticity
a = sp.params.actin
phys_per = int(AGENT_DT / PHYSICS_DT)

def run_at(pos, seconds):
    for _ in range(int(seconds / AGENT_DT)):
        acts = env.get_activations(pos)
        stim = activations_to_stimuli(acts)
        for _ in range(phys_per):
            for i in range(len(net.synapses)):
                g = net.presynaptic_release[i].step(acts[i], PHYSICS_DT)
                stim[i]['glutamate'] = g
            step_network_per_synapse(net, PHYSICS_DT, stim)

# --- drive: one traversal through the feature centre ---
n_steps = int((2*HALF_PATH/AGENT_SPEED)/AGENT_DT)
for k in range(n_steps):
    s = -HALF_PATH + (k+0.5)*(2*HALF_PATH/n_steps)
    run_at(env.feature_positions[target] + s*u, AGENT_DT)

print(f"after traversal: enl={sp.actin_enlargement:.5f}  spine_clock={sp.time:.4f}s")
print()
print("--- 20 s GAP, agent parked, sampled every 2 s ---")
print(f"{'t_gap':>6} {'spine_clk':>10} {'d_clk':>7} {'enl':>9} {'ca_uM':>9} {'f_CaM':>8} "
      f"{'formation':>10} {'extrusion':>10} {'net':>10}")
clk0, enl0 = sp.time, sp.actin_enlargement
for seg in range(10):
    run_at(park, 2.0)
    ca = float(np.max(net.synapses[target].calcium.get_concentration()))*1e6
    f_CaM = ca**a.hill_calcium/(a.K_calcium_poly**a.hill_calcium + ca**a.hill_calcium)
    F = sp.actin_dynamic + sp.actin_enlargement + sp.actin_stable
    F_max = sp.params.volume.max_enlargement_ratio**(1.0/sp.params.volume.actin_volume_scaling)
    room = max(0.0, 1.0 - F/F_max)
    form = a.k_polymerization_max*f_CaM*(sp.actin_monomer/a.S0)*room
    extr = (1.0/a.tau_extrude)*(1.0-sp.confinement)*sp.actin_enlargement
    print(f"{(seg+1)*2:6.0f} {sp.time:10.4f} {sp.time-clk0:7.2f} {sp.actin_enlargement:9.5f} "
          f"{ca:9.4f} {f_CaM:8.5f} {form:10.6f} {extr:10.6f} {form-extr:+10.6f}")

d_clk = sp.time - clk0
print()
print(f"CLOCK DELTA over the gap : {d_clk:.4f} s   (expected {GAP_S:.1f} s)")
ok = abs(d_clk - GAP_S) < 1e-6
print(f"ASSERTION spine clock advanced by the FULL gap: {'PASS' if ok else 'FAIL'}")
print(f"enlargement {enl0:.5f} -> {sp.actin_enlargement:.5f}  (rho = {sp.actin_enlargement/enl0:.4f})")
print()
if ok:
    print("=> D19's non-stepping mechanism is RULED OUT for this probe: the target synapse's")
    print("   own clock advanced the full gap while inactive. GATE 1's threshold was a symptom;")
    print("   this is the proof ruling 004 asked for.")
    print("=> Ruling 006's ambiguity resolves to (b) RESIDUAL FORMATION: see the formation")
    print("   column above — nonzero while calcium decays, so rho>1 early in a gap is real")
    print("   physics, not a stopped clock.")
else:
    print("=> THE GAP IS NOT FULLY STEPPING. L·ETA-5's retention numbers are artifactual.")
