"""
Model 6 System Validation Test
===============================
Tests that the INTEGRATED model6_core.py system works correctly

Validates the complete biological process:
1. Calcium channel opening → Ca2+ spike
2. ATP hydrolysis → Phosphate release → J-coupling
3. Ca + PO4 → CaHPO4 complex formation
4. Complex → PNC nucleation at templates
5. PNC → Posner molecule (Ca9(PO4)6) formation
6. Posner → Dimer aggregation (2 Posners = Ca18(PO4)12)
7. Quantum coherence in formed dimers
8. Dopamine modulation of dimer/trimer ratio

This test focuses on P31 only (natural human phosphorus).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

print("="*70)
print("MODEL 6 SYSTEM VALIDATION")
print("Testing complete integrated quantum synapse")
print("="*70)

# =============================================================================
# SETUP
# =============================================================================

print("\n### INITIALIZATION ###")

# Use default parameters (P31 = 100%, natural conditions)
params = Model6Parameters()
params.environment.T = 310.15  # 37°C

# Create integrated model
model = Model6QuantumSynapse(params=params)

print(f"✓ Model initialized")
print(f"  Grid: {model.grid_shape[0]}x{model.grid_shape[1]}")
print(f"  Spatial resolution: {model.dx*1e9:.1f} nm")
print(f"  Channels: {len(model.calcium.channels.positions)}")
print(f"  Temperature: {params.environment.T:.1f} K")

# =============================================================================
# TEST 1: Baseline State (No Activity)
# =============================================================================

print("\n### TEST 1: BASELINE (No Stimulus) ###")
print("Checking resting conditions...")

# Run 100ms at rest
for _ in range(100):  # 100 steps x 1ms
    model.step(model.dt, stimulus={'voltage': -70e-3})  # Resting potential

baseline_metrics = model.get_experimental_metrics()

print(f"\nBaseline Concentrations:")
print(f"  Calcium: {baseline_metrics['calcium_peak_uM']:.3f} μM")
print(f"  ATP: {baseline_metrics['atp_mean_mM']:.2f} mM")
print(f"  PNC: {baseline_metrics['pnc_peak_nM']:.3f} nM")
print(f"  Dimers: {baseline_metrics['dimer_peak_nM']:.3f} nM")
print(f"  Trimers: {baseline_metrics['trimer_peak_nM']:.3f} nM")

# Validate resting state
assert baseline_metrics['calcium_peak_uM'] < 0.5, "Calcium too high at rest"
assert baseline_metrics['atp_mean_mM'] > 2.0, "ATP depleted at rest"
print("✓ Baseline state correct")

# =============================================================================
# TEST 2: Calcium Spike (Channel Opening)
# =============================================================================

print("\n### TEST 2: CALCIUM DYNAMICS ###")
print("Opening channels to trigger Ca2+ spike...")

# Depolarize to trigger channels
ca_before = model.get_experimental_metrics()['calcium_peak_uM']

# Run 50ms with depolarization
for i in range(200):
    model.step(model.dt, stimulus={'voltage': -10e-3})  # Depolarized

ca_after = model.get_experimental_metrics()['calcium_peak_uM']

print(f"\nCalcium Response:")
print(f"  Before: {ca_before:.3f} μM")
print(f"  After: {ca_after:.2f} μM")
print(f"  Increase: {(ca_after/ca_before):.1f}x")

assert ca_after > ca_before * 5, "Calcium spike too small"
assert ca_after < 100, "Calcium unrealistically high"
print("✓ Calcium spike generated")

# =============================================================================
# TEST 3: ATP Hydrolysis & J-Coupling
# =============================================================================

print("\n### TEST 3: ATP & J-COUPLING ###")

atp_metrics = model.get_experimental_metrics()
j_coupling_max = atp_metrics['j_coupling_max_Hz']

print(f"\nATP System:")
print(f"  ATP concentration: {atp_metrics['atp_mean_mM']:.2f} mM")
print(f"  Max J-coupling: {j_coupling_max:.1f} Hz")
print(f"  Baseline J: {params.atp.J_PO_free:.1f} Hz")

# J-coupling should increase with ATP hydrolysis
assert j_coupling_max > params.atp.J_PO_free, "No J-coupling enhancement"
assert j_coupling_max > 5.0, "J-coupling too weak"
print("✓ ATP hydrolysis and J-coupling working")

# =============================================================================
# TEST 4: PNC Formation
# =============================================================================

print("\n### TEST 4: PNC FORMATION ###")
print("Checking prenucleation cluster formation...")

pnc_metrics = model.get_experimental_metrics()
pnc_concentration = pnc_metrics['pnc_peak_nM']

print(f"\nPNC Formation:")
print(f"  Peak PNC: {pnc_concentration:.2f} nM")
print(f"  Baseline: {baseline_metrics['pnc_peak_nM']:.3f} nM")
print(f"  Increase: {(pnc_concentration/baseline_metrics['pnc_peak_nM']):.1f}x")

# Get spatial distribution
pnc_field = model.pnc.get_pnc_concentration()
pnc_hotspots = np.sum(pnc_field > np.mean(pnc_field) * 2)

print(f"  Spatial hotspots: {pnc_hotspots} sites")

assert pnc_concentration > baseline_metrics['pnc_peak_nM'] * 2, "PNC formation insufficient"
assert pnc_hotspots > 0, "No spatial localization of PNCs"
print("✓ PNC formation occurring")

# =============================================================================
# TEST 5: Posner Dimer Formation
# =============================================================================

print("\n### TEST 5: POSNER DIMER FORMATION ###")
print("Checking if PNCs convert to Posner dimers...")

# Continue simulation to allow Posner formation
print("Running extended simulation (200ms more)...")
for i in range(200):
    model.step(model.dt, stimulus={'voltage': -70e-3, 'reward': False})
    
    if i % 50 == 0:
        m = model.get_experimental_metrics()
        print(f"  t={i}ms: Dimers={m['dimer_peak_nM']:.2f} nM, "
              f"Trimers={m['trimer_peak_nM']:.2f} nM")

final_metrics = model.get_experimental_metrics()

print(f"\nFinal Posner Concentrations:")
print(f"  Dimers: {final_metrics['dimer_peak_nM']:.2f} nM")
print(f"  Trimers: {final_metrics['trimer_peak_nM']:.2f} nM")
print(f"  Dimer/Trimer ratio: {final_metrics['dimer_trimer_ratio']:.2f}")

# CRITICAL: Must actually form dimers
if final_metrics['dimer_peak_nM'] > 0.01:  # At least 10 pM
    print("✓ Posner dimers formed successfully")
    dimer_formed = True
else:
    print("✗ CRITICAL: NO Posner dimer formation!")
    print("   PNCs are present but not converting to Posners")
    dimer_formed = False

# =============================================================================
# TEST 6: Quantum Coherence
# =============================================================================

print("\n### TEST 6: QUANTUM COHERENCE ###")

if dimer_formed:
    print("Checking quantum coherence in formed dimers...")
    
    coherence_dimer = final_metrics['coherence_dimer_mean']
    coherence_trimer = final_metrics['coherence_trimer_mean']
    T2_dimer = final_metrics['T2_dimer_s']
    T2_trimer = final_metrics['T2_trimer_s']
    
    print(f"\nCoherence Properties:")
    print(f"  Dimer coherence: {coherence_dimer:.3f}")
    print(f"  Trimer coherence: {coherence_trimer:.3f}")
    print(f"  Dimer T2: {T2_dimer:.1f} s")
    print(f"  Trimer T2: {T2_trimer:.1f} s")
    
    assert coherence_dimer > 0, "No quantum coherence in dimers"
    assert T2_dimer > T2_trimer, "Dimers should have longer T2 than trimers"
    print("✓ Quantum coherence present in dimers")
else:
    print("⚠ Skipping coherence test - no dimers formed")

# =============================================================================
# TEST 7: Dopamine Modulation (If Available)
# =============================================================================

print("\n### TEST 7: DOPAMINE MODULATION ###")

if model.dopamine is not None:
    print("Testing dopamine's effect on dimer/trimer ratio...")
    
    # Run with reward signal
    model2 = Model6QuantumSynapse(params=params)
    
    # Activity + reward
    for i in range(50):
        model2.step(model.dt, stimulus={'voltage': -10e-3, 'reward': True})
    
    # Recovery
    for i in range(200):
        model2.step(model.dt, stimulus={'voltage': -70e-3, 'reward': False})
    
    reward_metrics = model2.get_experimental_metrics()
    
    print(f"\nDopamine Effect:")
    print(f"  No reward - Dimer/Trimer: {final_metrics['dimer_trimer_ratio']:.2f}")
    print(f"  With reward - Dimer/Trimer: {reward_metrics['dimer_trimer_ratio']:.2f}")
    print(f"  DA peak: {reward_metrics['dopamine_peak_nM']:.1f} nM")
    
    if reward_metrics['dimer_peak_nM'] > 0.01:
        ratio_change = reward_metrics['dimer_trimer_ratio'] / final_metrics['dimer_trimer_ratio']
        print(f"  Ratio change: {ratio_change:.2f}x")
        
        if ratio_change > 1.1:
            print("✓ Dopamine modulates dimer/trimer formation")
        else:
            print("⚠ Dopamine effect weak or absent")
    else:
        print("⚠ No dimers formed even with dopamine")
else:
    print("⚠ Dopamine system not available")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

tests = {
    "Baseline State": baseline_metrics['calcium_peak_uM'] < 0.5,
    "Calcium Spike": ca_after > ca_before * 5,
    "J-Coupling Enhancement": j_coupling_max > 5.0,
    "PNC Formation": pnc_concentration > baseline_metrics['pnc_peak_nM'] * 2,
    "Posner Dimer Formation": dimer_formed,
    "Quantum Coherence": dimer_formed and coherence_dimer > 0,
    "Spatial Localization": pnc_hotspots > 0,
}

print("\nCore Physics Tests:")
for test_name, passed in tests.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")

# Critical failure analysis
critical_fails = []
if not tests["Posner Dimer Formation"]:
    critical_fails.append("NO DIMER FORMATION - Check posner_system.py formation rates")
if not tests["PNC Formation"]:
    critical_fails.append("NO PNC FORMATION - Check supersaturation conditions")
if not tests["Calcium Spike"]:
    critical_fails.append("NO CALCIUM RESPONSE - Check calcium_system.py channels")

if critical_fails:
    print("\n⚠ CRITICAL ISSUES:")
    for issue in critical_fails:
        print(f"  • {issue}")
else:
    print("\n✓ ALL CORE SYSTEMS FUNCTIONAL")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n### GENERATING VISUALIZATION ###")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model 6 System Validation Results', fontsize=14, fontweight='bold')

# Get final spatial fields
ca_field = model.calcium.get_concentration()
pnc_field = model.pnc.get_pnc_concentration()
dimer_field = model.posner.get_dimer_concentration()
j_coupling_field = model.atp.get_j_coupling()

# 1. Calcium
im1 = axes[0, 0].imshow(ca_field * 1e6, cmap='Reds')
axes[0, 0].set_title('Calcium (μM)')
axes[0, 0].set_xlabel('Position (grid units)')
axes[0, 0].set_ylabel('Position (grid units)')
plt.colorbar(im1, ax=axes[0, 0])

# Mark channel positions
for pos in model.calcium.channels.positions:
    axes[0, 0].plot(pos[1], pos[0], 'wo', markersize=8, markeredgecolor='black')

# 2. PNC concentration
im2 = axes[0, 1].imshow(pnc_field * 1e9, cmap='viridis')
axes[0, 1].set_title('PNC Concentration (nM)')
axes[0, 1].set_xlabel('Position (grid units)')
plt.colorbar(im2, ax=axes[0, 1])

# 3. Dimer concentration
im3 = axes[0, 2].imshow(dimer_field * 1e9, cmap='plasma')
axes[0, 2].set_title('Posner Dimers (nM)')
axes[0, 2].set_xlabel('Position (grid units)')
plt.colorbar(im3, ax=axes[0, 2])

# 4. J-coupling field
im4 = axes[1, 0].imshow(j_coupling_field, cmap='hot')
axes[1, 0].set_title('J-Coupling (Hz)')
axes[1, 0].set_xlabel('Position (grid units)')
axes[1, 0].set_ylabel('Position (grid units)')
plt.colorbar(im4, ax=axes[1, 0])

# 5. Time series of key metrics
if len(model.history['time']) > 0:
    time_ms = np.array(model.history['time']) * 1000
    
    axes[1, 1].plot(time_ms, model.history['calcium_peak'], 'r-', label='Calcium')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Calcium (μM)', color='r')
    axes[1, 1].tick_params(axis='y', labelcolor='r')
    
    ax_twin = axes[1, 1].twinx()
    ax_twin.plot(time_ms, model.history['dimer_concentration'], 'b-', label='Dimers')
    ax_twin.set_ylabel('Dimers (nM)', color='b')
    ax_twin.tick_params(axis='y', labelcolor='b')
    axes[1, 1].set_title('Time Evolution')

# 6. Summary metrics
axes[1, 2].axis('off')
summary_text = f"""
VALIDATION RESULTS

Baseline:
  Ca: {baseline_metrics['calcium_peak_uM']:.3f} μM
  ATP: {baseline_metrics['atp_mean_mM']:.1f} mM

After Stimulation:
  Ca peak: {ca_after:.1f} μM ({ca_after/ca_before:.0f}x)
  J-coupling: {j_coupling_max:.1f} Hz
  PNC: {pnc_concentration:.2f} nM
  
Formation:
  Dimers: {final_metrics['dimer_peak_nM']:.2f} nM
  Trimers: {final_metrics['trimer_peak_nM']:.2f} nM
  Ratio: {final_metrics['dimer_trimer_ratio']:.2f}

Quantum:
  Coherence: {coherence_dimer:.3f}
  T2: {T2_dimer:.1f} s
  
Status: {'✓ PASS' if all(tests.values()) else '✗ FAIL'}
"""
axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig('model6_system_validation.png', dpi=150, bbox_inches='tight')
print("Visualization saved: model6_system_validation.png")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)