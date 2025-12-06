"""
POC TEST: Tryptophan Geometry Integration
==========================================

Based on test_em_field_fix.py working pattern.

Tests whether Model 6 still produces correct outputs when we replace:
  FITTED: spatial_factor = 0.5 (hardcoded)
  WITH: spatial_factor from real PDB 1JFF geometry
"""

import numpy as np
import sys
from pathlib import Path

# Add Model 6 to path (same as working test)
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

print("="*80)
print("POC TEST: TRYPTOPHAN GEOMETRY INTEGRATION")
print("="*80)

# =============================================================================
# CONFIGURE PARAMETERS (exact pattern from test_em_field_fix.py)
# =============================================================================

params = Model6Parameters()

# === SET TIMESTEP (CRITICAL) ===
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

# =============================================================================
# INITIALIZE MODEL
# =============================================================================

print("\nInitializing model...")
model = Model6QuantumSynapse(params=params)
print(f"✓ Model initialized (dt={model.dt*1e3:.1f} ms)")

# =============================================================================
# RUN BURST PROTOCOL (exact pattern from test_em_field_fix.py)
# =============================================================================

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

# =============================================================================
# GET RESULTS
# =============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

metrics = model.get_experimental_metrics()

# Core dimer formation
print("\n### Dimer Formation ###")
dimer_peak = metrics.get('dimer_peak_nM_ct', 0)
dimer_mean = metrics.get('dimer_mean_nM_ct', 0)
print(f"  Peak concentration: {dimer_peak:.1f} nM")
print(f"  Mean concentration: {dimer_mean:.1f} nM")
print(f"  Total dimers (N=10): {dimer_peak * 10:.0f}")
print(f"  Per synapse: {dimer_peak:.1f} nM = {dimer_peak * 10 / 10:.1f} dimers/synapse")

# Quantum metrics
print("\n### Quantum Coherence ###")
T2 = metrics.get('T2_dimer_s', 0)
coherence = metrics.get('coherence_dimer_mean', 0)
print(f"  T2 time: {T2:.1f} s")
print(f"  Coherence fraction: {coherence:.3f}")

# EM coupling metrics
print("\n### EM Coupling ###")
print(f"  EM coupling enabled: {metrics.get('em_coupling_enabled', False)}")

if metrics.get('em_coupling_enabled', False):
    trp_field = metrics.get('trp_em_field_gv_m', 0)
    em_enhancement = metrics.get('em_formation_enhancement', 1.0)
    collective_field = metrics.get('collective_field_kT', 0)
    
    print(f"  Tryptophan field: {trp_field:.2f} GV/m")
    print(f"  Formation enhancement: {em_enhancement:.2f}x")
    print(f"  Collective field: {collective_field:.1f} kT")
    
    # Check if EM is working
    if trp_field > 0.5:
        print(f"  ✓ Tryptophan field being calculated")
    else:
        print(f"  ✗ WARNING: Tryptophan field too low or not calculated")
    
    if collective_field > 10:
        print(f"  ✓ Collective quantum field present")
    else:
        print(f"  ✗ WARNING: Collective field too low")
else:
    print("  ✗ EM coupling not enabled!")

# =============================================================================
# SUCCESS CRITERIA
# =============================================================================

print("\n" + "="*80)
print("SUCCESS CRITERIA CHECK")
print("="*80)

dimers_per_syn = dimer_peak
target_dimers = (4, 5)
target_T2 = 50
target_coherence = 0.3
target_energy = (15, 25)

# Check dimers
if target_dimers[0] <= dimers_per_syn <= target_dimers[1]:
    print(f"✓ Dimers per synapse: {dimers_per_syn:.1f} (target: {target_dimers[0]}-{target_dimers[1]})")
else:
    print(f"✗ Dimers per synapse: {dimers_per_syn:.1f} (target: {target_dimers[0]}-{target_dimers[1]})")

# Check T2
if T2 >= target_T2:
    print(f"✓ T2 coherence time: {T2:.1f} s (target: >{target_T2}s)")
else:
    print(f"✗ T2 coherence time: {T2:.1f} s (target: >{target_T2}s)")

# Check coherence
if coherence >= target_coherence:
    print(f"✓ Coherence fraction: {coherence:.3f} (target: >{target_coherence})")
else:
    print(f"✗ Coherence fraction: {coherence:.3f} (target: >{target_coherence})")

# Check energy scale (if EM enabled)
if metrics.get('em_coupling_enabled', False):
    collective_field = metrics.get('collective_field_kT', 0)
    if target_energy[0] <= collective_field <= target_energy[1]:
        print(f"✓ Energy scale: {collective_field:.1f} kT (target: {target_energy[0]}-{target_energy[1]} kT)")
    else:
        print(f"✗ Energy scale: {collective_field:.1f} kT (target: {target_energy[0]}-{target_energy[1]} kT)")

# =============================================================================
# NEXT STEPS
# =============================================================================

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("\nThis is the BASELINE test using fitted parameters.")
print("Current em_coupling_module.py uses: spatial_factor = 0.5 (hardcoded)")
print("\nTo test with real geometry:")
print("  1. Edit src/models/Model_6/em_coupling_module.py")
print("  2. Replace: spatial_factor = 0.5")
print("  3. With: spatial_factor = self._calculate_spatial_averaging()")
print("  4. Add the _calculate_spatial_averaging() method")
print("  5. Rerun this script and compare results")

print("\n" + "="*80)