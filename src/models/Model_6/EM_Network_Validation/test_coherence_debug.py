import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_parameters import Model6Parameters
from quantum_coherence import QuantumCoherenceSystem

# Initialize
params = Model6Parameters()
params.em_coupling_enabled = True

grid_shape = (50, 50)
qc = QuantumCoherenceSystem(grid_shape, params.quantum, isotope_P31_fraction=1.0)

# Simulate dimer formation
dimer_conc = np.zeros(grid_shape)
dimer_conc[20:30, 20:30] = 500e-9  # 500 nM dimers

# Strong J-coupling
j_coupling = np.ones(grid_shape) * 0.2
j_coupling[20:30, 20:30] = 18.0  # ATP active

print("=== INITIAL STATE ===")
print(f"Dimer concentration: {np.max(dimer_conc)*1e9:.1f} nM")
print(f"J-coupling: {np.max(j_coupling):.1f} Hz")

# Run one timestep
dt = 0.001  # 1 ms
qc.step(dt, dimer_conc, j_coupling)

print("\n=== AFTER ONE STEP ===")
metrics = qc.get_experimental_metrics()
print("Available metrics keys:", list(metrics.keys()))
print(f"Coherence mean: {metrics.get('coherence_mean', 'NOT FOUND')}")
print(f"T2 effective: {metrics.get('T2_dimer_s', 'NOT FOUND')} s")