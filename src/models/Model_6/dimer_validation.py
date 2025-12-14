# test_dimer_emergence.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

# Single synapse test
params = Model6Parameters()
params.em_coupling_enabled = True
params.multi_synapse_enabled = False  # Single synapse

model = Model6QuantumSynapse(params)

# Run with stimulation
print("Running stimulation...")
for i in range(500):
    model.step(0.001, {'voltage': -20e-3, 'activity_level': 0.8})

# Check what emerged
metrics = model.get_experimental_metrics()
print("\n=== DIMER EMERGENCE CHECK ===")
print(f"dimer_peak_nM_ct (from ca_phosphate): {metrics.get('dimer_peak_nM_ct', 0):.1f} nM")
print(f"dimer_mean_nM_ct (from ca_phosphate): {metrics.get('dimer_mean_nM_ct', 0):.1f} nM")

# Check quantum system
print(f"\nquantum.dimer_concentration max: {np.max(model.quantum.dimer_concentration)*1e9:.1f} nM")
print(f"quantum.dimer_concentration mean: {np.mean(model.quantum.dimer_concentration)*1e9:.3f} nM")

# Check _previous_dimer_count
print(f"\n_previous_dimer_count: {model._previous_dimer_count:.1f}")

# Expected calculation
print("\n=== EXPECTED (from Oct 7 formula) ===")
dimer_nM = metrics.get('dimer_peak_nM_ct', 0)
expected_dimers = dimer_nM * 0.006
print(f"{dimer_nM:.1f} nM × 0.006 = {expected_dimers:.1f} dimers per synapse")