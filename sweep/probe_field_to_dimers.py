#!/usr/bin/env python3
"""
Field-only probe: traces the full chain
  E_invasion -> eta -> collective_field_kT -> intra Pathway-2 (bonds + P_S)

Config: N_SYN=8, 30 s sustained -10 mV, cross-synapse tracker disabled.
The cross-synapse entanglement_tracker.step is monkeypatched to a no-op
so that ONLY intra-synapse dimer_particles.step (Pathway-2 at :404/:422)
runs.  This isolates the field -> dimer path from the O(n^2) tracker.
"""

import sys, os
import logging

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

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
params = Model6Parameters()
params.em_coupling_enabled = True   # needed: gates tryptophan module -> collective_field_kT

N_SYN = 8
T_total = 30.0   # 30 s

network = MultiSynapseNetwork(
    n_synapses=N_SYN, pattern="clustered", spacing_um=1.0,
)
network.initialize(Model6QuantumSynapse, params)
for s in network.synapses:
    s.set_microtubule_invasion(True)

# --- Confirm backbone is live before driving ---
assert network.params is not None, "network.params is None -- initialize() did not set it"
assert network.params.dendritic_backbone.enabled, "dendritic_backbone.enabled is False"

# --- Disable cross-synapse tracker (no-op returning stale/empty dict) ---
network.entanglement_tracker.step = lambda *a, **k: getattr(network, '_network_entanglement', {})

print(f"[startup] backbone LIVE, cross-tracker DISABLED, N_syn={N_SYN}")

dt = 0.005          # 5 ms -- standard physics timestep
log_interval = 5.0  # log every 5 s

stimulus = {'voltage': -10e-3, 'reward': False}  # sustained depolarization

# ---------------------------------------------------------------------------
# Drive loop
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("FIELD-ONLY PROBE: E_invasion -> eta -> collective_field_kT -> intra Pathway-2")
print("=" * 90)
print(f"  N_synapses={N_SYN}  dt={dt}  T={T_total}s  voltage={stimulus['voltage']*1e3:.0f}mV")
print()

header = (f"{'t(s)':>6s}  {'E_inv[0]':>8s}  {'E_inv_m':>7s}  "
          f"{'ca_op[0]':>8s}  {'eta[0]':>8s}  {'eta_max':>8s}  {'eta_m':>7s}  "
          f"{'fld_kT[0]':>9s}  {'bnd[0]':>6s}  {'bnd_m':>6s}  "
          f"{'P_S[0]':>6s}  {'P_S_m':>6s}")
print(header)
print("-" * len(header))

log = []
live_ok = True

steps = int(round(T_total / dt))
next_log = 0.0
t = 0.0

for i in range(steps):
    try:
        network.step(dt, stimulus)
    except AttributeError as e:
        print(f"\n[FAIL] AttributeError at t={t:.2f}s: {e}")
        live_ok = False
        break

    t += dt

    if t >= next_log - dt / 2:
        s0 = network.synapses[0]
        E_inv_0 = s0.spine_plasticity.E_invasion
        ca_open_0 = s0.calcium.channels.get_open_fraction()
        eta_0 = s0._backbone_eta
        field_kT_0 = s0._collective_field_kT
        bonds_0 = len(s0.dimer_particles.entanglement_bonds)
        PS_0 = s0.get_mean_singlet_probability()

        E_inv_mean = sum(s.spine_plasticity.E_invasion for s in network.synapses) / N_SYN
        eta_max = max(s._backbone_eta for s in network.synapses)
        eta_mean = sum(s._backbone_eta for s in network.synapses) / N_SYN
        bonds_mean = sum(len(s.dimer_particles.entanglement_bonds) for s in network.synapses) / N_SYN
        PS_mean = sum(s.get_mean_singlet_probability() for s in network.synapses) / N_SYN

        row = (t, E_inv_0, E_inv_mean, ca_open_0, eta_0, eta_max, eta_mean,
               field_kT_0, bonds_0, bonds_mean, PS_0, PS_mean)
        log.append(row)
        print(f"{t:6.1f}  {E_inv_0:8.4f}  {E_inv_mean:7.4f}  "
              f"{ca_open_0:8.4f}  {eta_0:8.4f}  {eta_max:8.4f}  {eta_mean:7.4f}  "
              f"{field_kT_0:9.4f}  {bonds_0:6d}  {bonds_mean:6.0f}  "
              f"{PS_0:6.4f}  {PS_mean:6.4f}")

        next_log += log_interval

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 90)
if not live_ok:
    print("ABORTED: AttributeError")
    sys.exit(1)

final = log[-1]
print(f"FINAL STATE (t={final[0]:.1f}s):")
print(f"  E_invasion[0]={final[1]:.4f}  mean={final[2]:.4f}")
print(f"  eta[0]={final[4]:.4f}  max={final[5]:.4f}  mean={final[6]:.4f}")
print(f"  field_kT[0]={final[7]:.4f}")
print(f"  intra bonds[0]={final[8]:.0f}  mean={final[9]:.0f}")
print(f"  P_S[0]={final[10]:.4f}  mean={final[11]:.4f}")

field_active = final[7] > 0
bonds_present = final[9] > 0
print()
print(f"  field reached dimers: {'YES' if field_active else 'NO'} (field_kT > 0)")
print(f"  intra bonds formed:  {'YES' if bonds_present else 'NO'} (mean bonds > 0)")
print("=" * 90)
sys.exit(0)
