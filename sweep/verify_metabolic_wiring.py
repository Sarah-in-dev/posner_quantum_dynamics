#!/usr/bin/env python3
"""
Verify metabolic-power wiring: E_invasion (continuous 0-1) replaces
binary mt_invaded in compute_metabolic_power.

Checks:
  0. SETUP-SANITY:  E_invasion ramped (mean > 0.5 by end)
  1. LIVE:          No AttributeError on s.spine_plasticity.E_invasion
  2. CONDENSATION:  eta_max > 0 at some point (any synapse crossed r >= 1)
  3. TRACKING:      eta_mean and field_kT rise as E_invasion rises
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

N_SYN = 1
T_total = 30.0   # 30 s -- single-synapse fast probe

network = MultiSynapseNetwork(
    n_synapses=N_SYN, pattern="clustered", spacing_um=1.0,
)
network.initialize(Model6QuantumSynapse, params)
for s in network.synapses:
    s.set_microtubule_invasion(True)

# --- Confirm backbone is live before driving ---
assert network.params is not None, "network.params is None -- initialize() did not set it"
assert network.params.dendritic_backbone.enabled, "dendritic_backbone.enabled is False"
print(f"[startup] backbone LIVE: dendritic_backbone.enabled={network.params.dendritic_backbone.enabled}, "
      f"network.params set, N_syn={N_SYN}")

dt = 0.005          # 5 ms -- standard physics timestep
log_interval = 5.0  # log every 5 s

stimulus = {'voltage': -10e-3, 'reward': False}  # sustained depolarization

# ---------------------------------------------------------------------------
# Drive loop
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("VERIFY METABOLIC WIRING: E_invasion -> compute_metabolic_power")
print("=" * 80)
print(f"  N_synapses={N_SYN}  dt={dt}  T={T_total}s  voltage={stimulus['voltage']*1e3:.0f}mV")
print()

header = (f"{'t(s)':>6s}  {'E_inv[0]':>8s}  {'E_inv_mean':>10s}  "
          f"{'ca_open[0]':>10s}  {'eta[0]':>8s}  {'eta_max':>8s}  "
          f"{'eta_mean':>8s}  {'field_kT[0]':>11s}")
print(header)
print("-" * len(header))

# log rows: (t, E_inv_0, E_inv_mean, ca_open_0, eta_0, eta_max, eta_mean, field_kT_0)
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

        E_inv_mean = sum(s.spine_plasticity.E_invasion for s in network.synapses) / N_SYN
        eta_max = max(s._backbone_eta for s in network.synapses)
        eta_mean = sum(s._backbone_eta for s in network.synapses) / N_SYN

        row = (t, E_inv_0, E_inv_mean, ca_open_0, eta_0, eta_max, eta_mean, field_kT_0)
        log.append(row)
        print(f"{t:6.1f}  {E_inv_0:8.4f}  {E_inv_mean:10.4f}  "
              f"{ca_open_0:10.4f}  {eta_0:8.4f}  {eta_max:8.4f}  "
              f"{eta_mean:8.4f}  {field_kT_0:11.4f}")

        next_log += log_interval

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("CHECKS")
print("=" * 80)

results = {}

# 1. LIVE -- no AttributeError
results["LIVE"] = live_ok
print(f"\n[{'PASS' if live_ok else 'FAIL'}] LIVE: "
      f"{'run completed, s.spine_plasticity.E_invasion reachable' if live_ok else 'AttributeError encountered'}")

if not live_ok:
    print(f"\nSUMMARY: 0/3 -- LIVE failed, skipping remaining checks")
    sys.exit(1)

# 0. SETUP-SANITY -- E_invasion actually ramped
final_E_inv_mean = log[-1][2]  # E_inv_mean from last row
setup_ok = final_E_inv_mean > 0.5
if not setup_ok:
    print(f"\n[SETUP FAIL] drive not reaching spine module -- E_invasion flat")
    print(f"  final E_inv_mean={final_E_inv_mean:.4f} (need > 0.5)")
    print(f"  CONDENSATION/TRACKING checks are meaningless; exiting.")
    sys.exit(1)
print(f"[PASS] SETUP-SANITY: final E_inv_mean={final_E_inv_mean:.4f} (> 0.5)")

# 2. CONDENSATION -- eta_max > 0 at some point (any synapse crossed r >= 1)
run_eta_max = max(row[5] for row in log)  # max of eta_max across all rows
condensation_ok = run_eta_max > 0
results["CONDENSATION"] = condensation_ok
print(f"[{'PASS' if condensation_ok else 'FAIL'}] CONDENSATION: "
      f"max(eta_max)={run_eta_max:.4f} ({'> 0, backbone crossed threshold' if condensation_ok else '== 0, graded drive stayed subcritical'})")

# 3. TRACKING -- eta_mean and field_kT rise as E_invasion rises
first = log[0]
peak_row = max(log, key=lambda r: r[1])  # row with highest E_inv_0

E_inv_rose = peak_row[1] > first[1] + 0.01
eta_mean_rose = peak_row[6] > first[6]    # eta_mean (network-wide)
field_rose = peak_row[7] > first[7]        # field_kT[0]
tracking_ok = E_inv_rose and eta_mean_rose and field_rose
results["TRACKING"] = tracking_ok
print(f"[{'PASS' if tracking_ok else 'FAIL'}] TRACKING: "
      f"E_inv {first[1]:.4f}->{peak_row[1]:.4f} "
      f"({'rose' if E_inv_rose else 'FLAT'}), "
      f"eta_mean {first[6]:.4f}->{peak_row[6]:.4f} "
      f"({'rose' if eta_mean_rose else 'FLAT'}), "
      f"field_kT {first[7]:.4f}->{peak_row[7]:.4f} "
      f"({'rose' if field_rose else 'FLAT'})")

# ---------------------------------------------------------------------------
n_pass = sum(results.values())
n_total = len(results)
all_pass = n_pass == n_total
print(f"\n{'=' * 80}")
print(f"SUMMARY: {n_pass}/{n_total} checks passed -- "
      f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=" * 80)

sys.exit(0 if all_pass else 1)
