"""
Debug: Trace the cascade values step by step
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

params = Model6Parameters()
params.em_coupling_enabled = True
params.multi_synapse_enabled = True
params.multi_synapse.n_synapses_default = 10

model = Model6QuantumSynapse(params)

# Run to build up dimers
print("Running 100 steps...")
for i in range(100):
    model.step(0.001, {'voltage': -20e-3, 'activity_level': 0.8})

print("\n" + "="*70)
print("TRACING CASCADE VALUES")
print("="*70)

# Now manually trace what should happen
print("\n1. Check n_synapses:")
n_synapses = params.multi_synapse.n_synapses_default if params.multi_synapse_enabled else 1
print(f"   multi_synapse_enabled: {params.multi_synapse_enabled}")
print(f"   n_synapses_default: {params.multi_synapse.n_synapses_default}")
print(f"   n_synapses used: {n_synapses}")

print("\n2. Calculate n_dimers (with correct units):")
coherent_mask = (model.quantum.coherence > 0.5) & (model.quantum.dimer_concentration > 0)
cleft_width = 20e-9
volume_per_voxel_m3 = (model.dx)**2 * cleft_width
volume_per_voxel_L = volume_per_voxel_m3 * 1000  # m³ to L

n_dimers_single = np.sum(
    model.quantum.dimer_concentration[coherent_mask] * volume_per_voxel_L * 6.022e23
)
mean_coherence = float(np.mean(model.quantum.coherence[coherent_mask])) if np.any(coherent_mask) else 0.0

print(f"   volume_per_voxel_m3: {volume_per_voxel_m3:.3e}")
print(f"   volume_per_voxel_L: {volume_per_voxel_L:.3e}")
print(f"   coherent voxels: {np.sum(coherent_mask)}")
print(f"   sum(concentration): {np.sum(model.quantum.dimer_concentration[coherent_mask]):.6f} M")
print(f"   n_dimers_single: {n_dimers_single:.3f}")
print(f"   mean_coherence: {mean_coherence:.3f}")

print("\n3. Calculate local modulation:")
local_mod = model.local_dimer_coupling.calculate_local_modulation(
    n_dimers=n_dimers_single,
    mean_coherence=mean_coherence
)
print(f"   local_mod result: {local_mod}")

print("\n4. Network integration:")
synapse_modulations = [local_mod['modulation_strength']] * n_synapses
print(f"   synapse_modulations list: {synapse_modulations}")
print(f"   list length: {len(synapse_modulations)}")

network_result = model.network_integrator.integrate_network(synapse_modulations)
print(f"   network_result: {network_result}")

print("\n5. Compare with model state:")
print(f"   model._network_modulation: {model._network_modulation:.3f}")
print(f"   model._collective_field_kT: {model._collective_field_kT:.2f}")

print("\n" + "="*70)