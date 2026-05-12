#!/usr/bin/env python3
"""
Minimal unit test: does the learning pathway produce spine volume changes?

Phases:
  1. Strong voltage drive (5 s) — build calcium / dimers
  2. Dopamine reward (1 s)     — trigger commitment gate
  3. Silent period (5 s)       — let consolidation run
"""

import sys
import os
import logging
import numpy as np

# ── Logging suppression (same as run_spatial_discovery.py) ────────────────────
logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

# ── Path setup (same as run_spatial_discovery.py) ─────────────────────────────
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)

from run_spatial_discovery import make_network, step_network_per_synapse

# ── Create network ────────────────────────────────────────────────────────────
network = make_network(n_synapses=2)

dt = 0.005

def spine_volumes():
    return [syn.spine_plasticity.spine_volume for syn in network.synapses]

def dimer_counts():
    return [len(syn.dimer_particles.dimers) for syn in network.synapses]

def total_bonds():
    return sum(len(syn.dimer_particles.entanglement_bonds) for syn in network.synapses)

def print_gate_status():
    for i, syn in enumerate(network.synapses):
        print(f"  syn {i}: _measurement_gate_opened={getattr(syn, '_measurement_gate_opened', False)}", flush=True)

def print_camkii_status():
    for i, syn in enumerate(network.synapses):
        cs = syn.camkii.get_state()
        print(f"  syn {i} CaMKII: molecular_memory={cs['molecular_memory']:.4f}, "
              f"pT286={cs['pT286']:.4f}, CaCaM={cs['CaCaM_bound']:.4f}", flush=True)

def print_calcium():
    for i, syn in enumerate(network.synapses):
        ca_conc = syn.calcium.get_concentration()
        print(f"  syn {i} calcium_uM={float(np.max(ca_conc))*1e6:.3f}", flush=True)

print("=== INITIAL STATE ===", flush=True)
print(f"  spine volumes: {spine_volumes()}", flush=True)

# ── PHASE 1: Strong voltage, no reward (5 s) ─────────────────────────────────
print("\n=== PHASE 1: Strong voltage drive (5 s) ===", flush=True)
t = 0.0
while t < 5.0:
    stim = [{'voltage': -10e-3, 'reward': False},
            {'voltage': -10e-3, 'reward': False}]
    step_network_per_synapse(network, dt, stim)
    t += dt

print(f"  spine volumes: {spine_volumes()}", flush=True)
print(f"  dimer counts:  {dimer_counts()}", flush=True)
print(f"  total bonds:   {total_bonds()}", flush=True)
print_gate_status()
print_camkii_status()

# ── PHASE 2: Dopamine reward (1 s) ───────────────────────────────────────────
print("\n=== PHASE 2: Dopamine + voltage (1 s) ===", flush=True)
t = 0.0
committed_printed = False
network_committed_printed = False
total_dissolved = 0
while t < 1.0:
    stim = [{'voltage': -10e-3, 'reward': True},
            {'voltage': -10e-3, 'reward': True}]
    step_network_per_synapse(network, dt, stim)
    t += dt

    total_dissolved += sum(syn.dimer_particles.get_dissolved_count() for syn in network.synapses)

    if not committed_printed:
        for i, syn in enumerate(network.synapses):
            if getattr(syn, '_camkii_committed', False):
                print(f"  *** synapse {i} _camkii_committed at t={t:.3f}s", flush=True)
                committed_printed = True

    if not network_committed_printed and network.network_committed:
        print(f"  *** network_committed at t={t:.3f}s", flush=True)
        network_committed_printed = True

print(f"  spine volumes:  {spine_volumes()}", flush=True)
print(f"  dimer counts:   {dimer_counts()}", flush=True)
print(f"  committed:      {[getattr(s, '_camkii_committed', False) for s in network.synapses]}", flush=True)
print(f"  network_committed: {network.network_committed}", flush=True)
print(f"  total dissolved during phase 2: {total_dissolved}", flush=True)
print_gate_status()
print_camkii_status()
print_calcium()

# ── PHASE 3: Silent period (5 s) ─────────────────────────────────────────────
print("\n=== PHASE 3: Silent period (5 s) ===", flush=True)
t = 0.0
last_print = 0.0
while t < 5.0:
    stim = [{'voltage': -70e-3, 'reward': False},
            {'voltage': -70e-3, 'reward': False}]
    step_network_per_synapse(network, dt, stim)
    t += dt

    # Print CaMKII state every 1 second
    if t - last_print >= 1.0:
        print(f"  --- t={t:.1f}s ---", flush=True)
        print_camkii_status()
        print(f"  spine volumes: {spine_volumes()}", flush=True)
        if not committed_printed:
            for i, syn in enumerate(network.synapses):
                if getattr(syn, '_camkii_committed', False):
                    print(f"  *** synapse {i} _camkii_committed at t={t:.3f}s (phase 3)", flush=True)
                    committed_printed = True
        last_print = t

print(f"  final spine volumes: {spine_volumes()}", flush=True)

# ── Verdict ───────────────────────────────────────────────────────────────────
v0 = spine_volumes()
changed = any(abs(v - 1.0) > 1e-6 for v in v0)
print(f"\n=== VERDICT: spine volume changed = {changed} ===", flush=True)
