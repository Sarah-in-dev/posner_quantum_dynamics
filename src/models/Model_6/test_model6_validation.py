"""
Model 6 Validation Test Script
===============================
Tests whether Model 6 properly implements:
1. Calcium triphosphate (Ca3(PO4)2) formation
2. Posner molecule (Ca9(PO4)6) formation  
3. Dimer formation with entanglement
4. Quantum coherence with spatial/temporal dynamics

This script answers: "Do we have the physics we claim to have?"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_parameters import Model6Parameters
from calcium_system import CalciumSystem
from atp_system import ATPSystem
from pnc_formation import PNCFormationSystem
from posner_system import PosnerSystem
from pH_dynamics import pHDynamics

print("="*70)
print("MODEL 6 PHYSICS VALIDATION TEST")
print("="*70)

# =============================================================================
# TEST 1: Do we form Posner molecules (Ca9(PO4)6)?
# =============================================================================

print("\n### TEST 1: Posner Molecule Formation ###")

params = Model6Parameters()
grid_shape = (50, 50)
dx = 2 * params.spatial.active_zone_radius / grid_shape[0]

# Create PNC and Posner systems
pnc_system = PNCFormationSystem(grid_shape, dx, params)
posner_system = PosnerSystem(grid_shape, params)

# Simulate conditions that should form Posners
print("Setting up supersaturated conditions...")

dt = 1e-4  # 100 μs
ca_high = np.ones(grid_shape) * 100e-9
ca_high[20:30, 20:30] = 10e-6  # 10 μM

po4_high = np.ones(grid_shape) * 1e-3  # 1 mM

# Run PNC formation
print("Running PNC formation...")
for i in range(1000):
    pnc_system.step(dt, ca_high, po4_high)

pnc_metrics = pnc_system.get_experimental_metrics()
print(f"\nPNC Results:")
print(f"  Total PNC: {pnc_metrics['pnc_peak_nM']:.1f} nM")
print(f"  Average size: {pnc_metrics['pnc_size_avg_Ca']:.1f} Ca atoms")
print(f"  Supersaturation max: {pnc_metrics['supersaturation_max']:.2e}")

# Check if we have large PNCs (>30 Ca)
pnc_large = pnc_system.get_large_pncs()
print(f"  Large PNCs (>30 Ca): {np.sum(pnc_large > 0)} sites")

if np.max(pnc_large) > 0:
    print("✓ PNC formation successful")
else:
    print("✗ NO PNC formation - check supersaturation")

# =============================================================================
# TEST 2: Do PNCs convert to Posner molecules?
# =============================================================================

print("\n### TEST 2: Posner Formation from PNCs ###")

# Need J-coupling and dopamine
j_coupling = np.ones(grid_shape) * 0.2
j_coupling[20:30, 20:30] = 18.0  # High J-coupling from ATP

dopamine_d2 = np.ones(grid_shape) * 0.5  # Moderate D2 activation

print("Running Posner formation...")
for i in range(1000):
    posner_system.step(dt, pnc_large, ca_high, j_coupling, dopamine_d2)
    
    if i % 200 == 0:
        metrics = posner_system.get_experimental_metrics()
        print(f"  t={i*dt*1e3:.0f} ms: "
              f"Dimers={metrics['dimer_peak_nM']:.1f} nM, "
              f"Trimers={metrics['trimer_peak_nM']:.1f} nM")

metrics = posner_system.get_experimental_metrics()
print(f"\nPosner Results:")
print(f"  Dimers formed: {metrics['dimer_peak_nM']:.1f} nM")
print(f"  Trimers formed: {metrics['trimer_peak_nM']:.1f} nM")
print(f"  Dimer/Trimer ratio: {metrics['dimer_trimer_ratio']:.2f}")

if metrics['dimer_peak_nM'] > 0:
    print("✓ Posner dimers formed")
else:
    print("✗ NO Posner formation")

# =============================================================================
# TEST 3: Do dimers have quantum coherence?
# =============================================================================

print("\n### TEST 3: Quantum Coherence ###")

print(f"Coherence (dimers): {metrics['coherence_dimer_mean']:.3f}")
print(f"Coherence (trimers): {metrics['coherence_trimer_mean']:.3f}")
print(f"T2 (dimers): {metrics['T2_dimer_s']:.1f} s")
print(f"T2 (trimers): {metrics['T2_trimer_s']:.1f} s")

if metrics['coherence_dimer_mean'] > 0:
    print("✓ Quantum coherence present in dimers")
else:
    print("✗ NO quantum coherence")

# =============================================================================
# TEST 4: Check for entanglement (MISSING?)
# =============================================================================

print("\n### TEST 4: Quantum Entanglement ###")

# Check if we have entanglement calculation
try:
    # This should exist if we properly implement entanglement
    entanglement_field = posner_system.quantum.get_entanglement_field()
    print(f"Entanglement field exists: {entanglement_field.shape}")
    print(f"Max entanglement: {np.max(entanglement_field):.3f}")
    print("✓ Entanglement tracking implemented")
except AttributeError:
    print("✗ MISSING: Entanglement field not implemented")
    print("   Need to add spatial entanglement tracking!")

# =============================================================================
# TEST 5: Spatial/Temporal dynamics
# =============================================================================

print("\n### TEST 5: Spatial/Temporal Dynamics ###")

# Check if dimers form in specific locations
dimer_field = posner_system.get_dimer_concentration()
spatial_variance = np.var(dimer_field)

print(f"Spatial variance in dimers: {spatial_variance*1e18:.2e} nM²")

if spatial_variance > 0:
    # Find hotspots
    hotspots = np.where(dimer_field > np.mean(dimer_field) + 2*np.std(dimer_field))
    print(f"Dimer hotspots: {len(hotspots[0])} locations")
    print("✓ Spatial heterogeneity present")
else:
    print("✗ NO spatial structure - dimers uniform everywhere")

# =============================================================================
# TEST 6: Isotope dependence
# =============================================================================

print("\n### TEST 6: Isotope Dependence (KEY PREDICTION) ###")

# Test P31 vs P32
print("\nRunning P31 (natural)...")
params_P31 = Model6Parameters()
posner_P31 = PosnerSystem(grid_shape, params_P31)

for i in range(500):
    posner_P31.step(dt, pnc_large, ca_high, j_coupling, dopamine_d2)

metrics_P31 = posner_P31.get_experimental_metrics()

print("\nRunning P32 (substituted)...")
params_P32 = Model6Parameters()
params_P32.environment.fraction_P31 = 0.0
params_P32.environment.fraction_P32 = 1.0
posner_P32 = PosnerSystem(grid_shape, params_P32)

for i in range(500):
    posner_P32.step(dt, pnc_large, ca_high, j_coupling, dopamine_d2)

metrics_P32 = posner_P32.get_experimental_metrics()

print(f"\nIsotope Comparison:")
print(f"  P31 T2: {metrics_P31['T2_dimer_s']:.1f} s")
print(f"  P32 T2: {metrics_P32['T2_dimer_s']:.1f} s")

fold_change = metrics_P31['T2_dimer_s'] / metrics_P32['T2_dimer_s']
print(f"  Fold change: {fold_change:.2f}x")
print(f"  Expected: 10x (Fisher 2015)")

if fold_change > 5:
    print("✓ Isotope effect confirmed")
else:
    print("⚠ Isotope effect weaker than expected")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

tests = [
    ("PNC Formation", np.max(pnc_large) > 0),
    ("Posner Formation", metrics['dimer_peak_nM'] > 0),
    ("Quantum Coherence", metrics['coherence_dimer_mean'] > 0),
    ("Spatial Dynamics", spatial_variance > 0),
    ("Isotope Dependence", fold_change > 2),
]

print("\nPhysics Implementation Status:")
for test_name, passed in tests:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")

# Critical missing features
print("\nCRITICAL MISSING FEATURES:")
missing = []

try:
    posner_system.quantum.get_entanglement_field()
except AttributeError:
    missing.append("Spatial entanglement between dimers")

if len(missing) == 0:
    print("  None - all critical features implemented!")
else:
    for feature in missing:
        print(f"  ✗ {feature}")

print("\n" + "="*70)

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# PNC concentration
im1 = axes[0, 0].imshow(pnc_large * 1e9, cmap='hot')
axes[0, 0].set_title('Large PNC Concentration (nM)')
plt.colorbar(im1, ax=axes[0, 0])

# Dimer concentration
im2 = axes[0, 1].imshow(dimer_field * 1e9, cmap='viridis')
axes[0, 1].set_title('Dimer Concentration (nM)')
plt.colorbar(im2, ax=axes[0, 1])

# Coherence
coherence_field = posner_system.get_coherence_dimer()
im3 = axes[0, 2].imshow(coherence_field, cmap='plasma', vmin=0, vmax=1)
axes[0, 2].set_title('Dimer Coherence')
plt.colorbar(im3, ax=axes[0, 2])

# J-coupling field
im4 = axes[1, 0].imshow(j_coupling, cmap='coolwarm')
axes[1, 0].set_title('J-Coupling (Hz)')
plt.colorbar(im4, ax=axes[1, 0])

# Calcium
im5 = axes[1, 1].imshow(ca_high * 1e6, cmap='Reds')
axes[1, 1].set_title('Calcium (μM)')
plt.colorbar(im5, ax=axes[1, 1])

# Isotope comparison
axes[1, 2].bar(['P31', 'P32'], [metrics_P31['T2_dimer_s'], metrics_P32['T2_dimer_s']])
axes[1, 2].set_ylabel('T2 Coherence Time (s)')
axes[1, 2].set_title('Isotope Dependence')
axes[1, 2].axhline(y=100, color='r', linestyle='--', label='P31 expected')
axes[1, 2].axhline(y=10, color='b', linestyle='--', label='P32 expected')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('model6_validation.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: model6_validation.png")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)