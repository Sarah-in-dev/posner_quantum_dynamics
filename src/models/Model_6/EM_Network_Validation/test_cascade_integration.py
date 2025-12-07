"""
Test cascade integration in Model 6
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

np.random.seed(42)  # Reproducibility for testing

print("="*70)
print("CASCADE ARCHITECTURE VALIDATION")
print("="*70)

# Setup parameters
params = Model6Parameters()
params.em_coupling_enabled = True
params.multi_synapse_enabled = True
params.multi_synapse.n_synapses_default = 10

model = Model6QuantumSynapse(params)

print("\n1. Cascade modules initialized:")
print(f"   local_dimer_coupling: {model.local_dimer_coupling is not None}")
print(f"   network_integrator: {model.network_integrator is not None}")

# Run simulation
print("\n2. Running simulation (1000 steps with activity)...")
for i in range(1000):
    model.step(0.001, {'voltage': -20e-3, 'activity_level': 0.8})

print("\n3. CASCADE RESULTS:")
print(f"   network_modulation: {model._network_modulation:.2f}")
print(f"   collective_field_kT: {model._collective_field_kT:.1f}")

# Check threshold
if model._network_modulation >= 5.0:
    print(f"\n   ✓ ABOVE SUPERRADIANCE THRESHOLD ({model._network_modulation:.1f} >= 5.0)")
else:
    print(f"\n   ✗ Below threshold ({model._network_modulation:.2f} / 5.0)")

# Get metrics
metrics = model.get_experimental_metrics()
print("\n4. EXPERIMENTAL METRICS:")
print(f"   dimer_peak_nM: {metrics.get('dimer_peak_nM_ct', 0):.1f}")
print(f"   T2_dimer_s: {metrics.get('T2_dimer_s', 0):.1f}")

# Validate against targets
print("\n5. VALIDATION:")
dimer_nM = metrics.get('dimer_peak_nM_ct', 0)
dimers_per_synapse = dimer_nM * 0.006 / 10  # 10 synapses
print(f"   dimers per synapse: {dimers_per_synapse:.1f} (target: 4-6)")

if 3 <= dimers_per_synapse <= 8:
    print("   ✓ Dimer count in range")
else:
    print("   ✗ Dimer count out of range")

if model._collective_field_kT >= 20:
    print(f"   ✓ Collective field >= 20 kT ({model._collective_field_kT:.1f})")
else:
    print(f"   ✗ Collective field below 20 kT ({model._collective_field_kT:.1f})")

T2 = metrics.get('T2_dimer_s', 0)
if T2 >= 50:
    print(f"   ✓ T2 >= 50s ({T2:.1f})")
else:
    print(f"   ✗ T2 below 50s ({T2:.1f})")

print("\n" + "="*70)
print("CASCADE VALIDATION COMPLETE")
print("="*70)