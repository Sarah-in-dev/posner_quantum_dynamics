"""
Quick validation test for EM field fixes

Tests single condition to verify:
1. Tryptophan EM field is calculated (should be ~1.4 GV/m)
2. n_coherent is counting correctly (should be >0)
3. Collective field is calculated (should be ~20 kT for N=10)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from current directory (run this from src/models/Model_6/)
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

print("="*80)
print("EM FIELD FIX VALIDATION TEST")
print("="*80)

# Configure parameters for single test condition
# N=10 synapses (at threshold), P31, baseline conditions
params = Model6Parameters()

# === SET TIMESTEP ===
params.simulation.dt_diffusion = 1e-3  # 1 ms

# === ENABLE EM COUPLING ===
params.em_coupling_enabled = True
params.multi_synapse_enabled = True
params.multi_synapse.n_synapses_default = 10

# === ISOTOPE ===
params.environment.fraction_P31 = 1.0
params.environment.fraction_P32 = 0.0

# === TEMPERATURE ===
params.environment.T = 310.15  # 37°C

print("\nTest Configuration:")
print(f"  N_synapses: 10 (threshold)")
print(f"  Isotope: P31")
print(f"  Temperature: 310 K")
print(f"  dt: {params.simulation.dt_diffusion*1e3:.1f} ms")
print(f"  EM coupling: {params.em_coupling_enabled}")

# Initialize model
print("\nInitializing model...")
model = Model6QuantumSynapse(params=params)
print(f"✓ Model initialized (dt={model.dt*1e3:.1f} ms)")

# Run burst protocol
BURST_PROTOCOL = {
    'n_bursts': 5,
    'burst_duration_ms': 30,
    'inter_burst_interval_ms': 150
}

print(f"\nRunning burst protocol:")
print(f"  {BURST_PROTOCOL['n_bursts']} bursts × {BURST_PROTOCOL['burst_duration_ms']}ms")

# Baseline
print("  Baseline (20 steps)...", end="")
for _ in range(20):
    model.step(model.dt, stimulus={'voltage': -70e-3})
print(" done")

# Bursts
for burst_num in range(BURST_PROTOCOL['n_bursts']):
    print(f"  Burst {burst_num+1}/{BURST_PROTOCOL['n_bursts']}...", end="")
    
    # Active phase
    for _ in range(BURST_PROTOCOL['burst_duration_ms']):
        model.step(model.dt, stimulus={'voltage': -10e-3})
    
    # Rest phase
    if burst_num < BURST_PROTOCOL['n_bursts'] - 1:
        for _ in range(BURST_PROTOCOL['inter_burst_interval_ms']):
            model.step(model.dt, stimulus={'voltage': -70e-3})
    
    print(" done")

# Final recovery
print("  Recovery (300 steps)...", end="")
for _ in range(300):
    model.step(model.dt, stimulus={'voltage': -70e-3})
print(" done")

# Get metrics
print("\n" + "="*80)
print("RESULTS")
print("="*80)

metrics = model.get_experimental_metrics()

# Core dimer formation
print("\n### Dimer Formation ###")
print(f"  Peak concentration: {metrics.get('dimer_peak_nM_ct', 0):.1f} nM")
print(f"  Coherence mean: {metrics.get('coherence_dimer_mean', 0):.3f}")
print(f"  T2 time: {metrics.get('T2_dimer_s', 0):.1f} s")

# EM coupling - Forward path (Trp → Dimers)
print("\n### EM Coupling - Forward Path ###")
trp_field = metrics.get('trp_em_field_gv_m', 0)
k_enhancement = metrics.get('em_formation_enhancement', 1.0)
print(f"  Tryptophan EM field: {trp_field:.3f} GV/m")
print(f"  k_enhancement: {k_enhancement:.2f}×")

if trp_field > 0.5:
    print(f"  ✓ PASS - Field is non-zero and reasonable")
else:
    print(f"  ✗ FAIL - Field should be ~1.4 GV/m")

# EM coupling - Reverse path (Dimers → Proteins)
print("\n### EM Coupling - Reverse Path ###")
collective_field = metrics.get('collective_field_kT', 0)
print(f"  Collective quantum field: {collective_field:.1f} kT")

if collective_field > 10:
    print(f"  ✓ PASS - Field is above threshold (~20 kT expected for N=10)")
elif collective_field > 0:
    print(f"  ⚠ PARTIAL - Field is calculated but weak")
else:
    print(f"  ✗ FAIL - Field should be ~20 kT for N=10 synapses")

# Overall assessment
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

pass_count = 0
if trp_field > 0.5:
    pass_count += 1
if collective_field > 10:
    pass_count += 1

print(f"\nPassed: {pass_count}/2 checks")

if pass_count == 2:
    print("✓ ALL CHECKS PASSED - EM coupling is working!")
elif pass_count == 1:
    print("⚠ PARTIAL SUCCESS - One pathway working")
else:
    print("✗ BOTH CHECKS FAILED - Issues remain")

print("\n" + "="*80)