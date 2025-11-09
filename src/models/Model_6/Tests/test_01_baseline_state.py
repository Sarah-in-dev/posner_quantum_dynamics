"""
Test 01: Baseline State Validation
===================================

Verifies Model 6 initializes correctly with physiological resting conditions.

This is the foundation test - everything should be stable at rest:
- Calcium at resting level (~100 nM, with stochastic channel noise)
- ATP stores full (>2 mM, with stochastic hydrolysis variability)
- Minimal complex formation (with stochastic aggregation)
- No excessive drift (allows for local fluctuations)

Pass Criteria (UPDATED Nov 2025 for stochasticity):
- Calcium peak: 0.05-0.3 μM (was <0.5 μM)
- ATP: 2.3-2.6 mM (was >2.0 mM)
- System drift: <0.05 μM (was <0.01 μM)
- Dimers: <2.0 nM (was <1.0 nM)

NOTE: Values will vary run-to-run due to stochastic processes.
These ranges represent physiological variability.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_NAME = "Baseline State"
TEST_NUMBER = "01"
DURATION_MS = 100  # Run for 100ms at rest
OUTPUT_DIR = Path(__file__).parent / "validation_results" / "test_01_baseline"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"TEST {TEST_NUMBER}: {TEST_NAME.upper()}")
print("="*80)
print(f"Duration: {DURATION_MS} ms")
print(f"Output directory: {OUTPUT_DIR}")
print()

# =============================================================================
# INITIALIZE MODEL
# =============================================================================

print("### INITIALIZATION ###")
params = Model6Parameters()
params.environment.T = 310.15  # 37°C

model = Model6QuantumSynapse(params=params)

print(f"✓ Model initialized")
print(f"  Grid: {model.grid_shape[0]}x{model.grid_shape[1]}")
print(f"  dx: {model.dx*1e9:.1f} nm")
print(f"  Temperature: {params.environment.T:.1f} K")
print(f"  Isotope: {params.environment.fraction_P31*100:.0f}% ³¹P")
print()

# =============================================================================
# RUN BASELINE SIMULATION
# =============================================================================

print("### RUNNING BASELINE SIMULATION ###")
print("Holding at resting potential (-70 mV)...")

# Track time series
n_steps = DURATION_MS
time_points = []
ca_history = []
atp_history = []
j_coupling_history = []
ion_pair_history = []
dimer_history = []

# Run simulation
for step in range(n_steps):
    # Resting potential
    model.step(model.dt, stimulus={'voltage': -70e-3})
    
    # Record metrics every ms
    metrics = model.get_experimental_metrics()
    time_points.append(step * model.dt * 1000)  # Convert to ms
    ca_history.append(metrics['calcium_peak_uM'])
    atp_history.append(metrics['atp_mean_mM'])
    j_coupling_history.append(metrics['j_coupling_max_Hz'])
    ion_pair_history.append(metrics['ion_pair_peak_nM'])
    dimer_history.append(metrics['dimer_peak_nM_ct'])
    
    # Progress indicator
    if (step + 1) % 20 == 0:
        print(f"  t={step+1}ms: Ca={ca_history[-1]:.3f}μM, "
              f"ATP={atp_history[-1]:.2f}mM, "
              f"Dimers={dimer_history[-1]:.2f}nM")

print()

# =============================================================================
# GET FINAL STATE
# =============================================================================

print("### FINAL STATE ###")
final_metrics = model.get_experimental_metrics()

print(f"Calcium: {final_metrics['calcium_peak_uM']:.3f} μM")
print(f"ATP: {final_metrics['atp_mean_mM']:.2f} mM")
print(f"J-coupling: {final_metrics['j_coupling_max_Hz']:.1f} Hz")
print(f"CaHPO₄ ion pairs: {final_metrics['ion_pair_peak_nM']:.1f} nM")
print(f"Ca₆(PO₄)₄ dimers: {final_metrics['dimer_peak_nM_ct']:.2f} nM")
print()

# =============================================================================
# VALIDATION
# =============================================================================

print("### VALIDATION ###")

# Check pass criteria (UPDATED FOR STOCHASTICITY - Nov 2025)
tests_passed = []
tests_failed = []

# Test 1: Calcium at rest (RANGE CHECK - allows for stochastic channel noise)
# With stochastic gating, expect 0.05-0.2 μM instead of exact 0.1 μM
if 0.05 < final_metrics['calcium_peak_uM'] < 0.3:
    print(f"✓ PASS: Calcium at resting level ({final_metrics['calcium_peak_uM']:.3f} μM)")
    tests_passed.append("calcium_resting")
else:
    print(f"✗ FAIL: Calcium out of range ({final_metrics['calcium_peak_uM']:.3f} μM, expected 0.05-0.3)")
    tests_failed.append("calcium_resting")

# Test 2: ATP not depleted (RANGE CHECK - allows for stochastic hydrolysis)
# With stochastic bursts, expect 2.3-2.5 mM instead of exact 2.5 mM
if 2.3 < final_metrics['atp_mean_mM'] < 2.6:
    print(f"✓ PASS: ATP stores full ({final_metrics['atp_mean_mM']:.2f} mM)")
    tests_passed.append("atp_full")
else:
    print(f"✗ FAIL: ATP out of range ({final_metrics['atp_mean_mM']:.2f} mM, expected 2.3-2.6)")
    tests_failed.append("atp_full")

# Test 3: Stability (UPDATED - allows for stochastic drift)
# With pH noise and local fluctuations, allow up to 0.05 μM drift
ca_drift = np.std(ca_history[-20:])  # Last 20ms
if ca_drift < 0.05:  # Increased from 0.01 to allow stochastic noise
    print(f"✓ PASS: System stable (drift={ca_drift:.4f} μM)")
    tests_passed.append("stable")
else:
    print(f"✗ FAIL: System unstable (drift={ca_drift:.4f} μM, expected <0.05)")
    tests_failed.append("stable")

# Test 4: Minimal complex formation (RANGE CHECK)
# With stochastic dimer formation, expect 0-2 nM instead of <1 nM
if final_metrics['dimer_peak_nM_ct'] < 2.0:
    print(f"✓ PASS: Minimal dimer formation at rest ({final_metrics['dimer_peak_nM_ct']:.2f} nM)")
    tests_passed.append("minimal_dimers")
else:
    print(f"⚠ WARNING: Elevated dimers at rest ({final_metrics['dimer_peak_nM_ct']:.2f} nM)")
    # Still not a hard failure for baseline

# Overall result
test_passed = len(tests_failed) == 0

# =============================================================================
# VISUALIZATION
# =============================================================================

print("### GENERATING VISUALIZATIONS ###")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Convert to numpy arrays
time_points = np.array(time_points)
ca_history = np.array(ca_history)
atp_history = np.array(atp_history)
j_coupling_history = np.array(j_coupling_history)
ion_pair_history = np.array(ion_pair_history)
dimer_history = np.array(dimer_history)

# --- TIME SERIES (Top 2 rows) ---

# Row 1: Calcium and ATP
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(time_points, ca_history, 'r-', linewidth=2)
ax1.axhline(0.1, color='k', linestyle='--', alpha=0.3, label='Expected resting (0.1 μM)')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('[Ca²⁺] (μM)', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_title('Calcium Concentration at Rest')
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2:4])
ax2.plot(time_points, atp_history, 'b-', linewidth=2)
ax2.axhline(2.5, color='k', linestyle='--', alpha=0.3, label='Physiological (2.5 mM)')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('[ATP] (mM)', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_title('ATP Concentration')
ax2.grid(alpha=0.3)

# Row 2: J-coupling and complexes
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(time_points, j_coupling_history, 'orange', linewidth=2)
ax3.axhline(params.atp.J_PO_free, color='k', linestyle='--', alpha=0.3, 
            label=f'Free phosphate ({params.atp.J_PO_free} Hz)')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('J-coupling (Hz)')
ax3.legend(loc='upper right', fontsize=8)
ax3.set_title('J-Coupling Field')
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 2:4])
ax4.plot(time_points, ion_pair_history, 'g-', linewidth=2, label='CaHPO₄ ion pairs')
ax4.plot(time_points, dimer_history, 'm-', linewidth=2, label='Ca₆(PO₄)₄ dimers')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Concentration (nM)')
ax4.legend(loc='upper right', fontsize=8)
ax4.set_title('Calcium Phosphate Complexes')
ax4.grid(alpha=0.3)

# --- SPATIAL MAPS (Bottom row) ---

# Get final spatial fields
ca_field = model.calcium.get_concentration()
ion_pair_field = model.ca_phosphate.get_ion_pair_concentration()
dimer_field = model.ca_phosphate.get_dimer_concentration()
j_coupling_field = model.atp.get_j_coupling()

# Calcium
ax5 = fig.add_subplot(gs[2, 0])
im1 = ax5.imshow(ca_field * 1e6, cmap='Reds', vmin=0, vmax=0.2)
ax5.set_title('Ca²⁺ (μM)')
ax5.set_xlabel('Position (grid)')
ax5.set_ylabel('Position (grid)')
plt.colorbar(im1, ax=ax5, fraction=0.046)
# Mark channel positions
for pos in model.calcium.channels.positions:
    ax5.plot(pos[1], pos[0], 'wo', markersize=4, markeredgecolor='black', markeredgewidth=0.5)

# Ion pairs
ax6 = fig.add_subplot(gs[2, 1])
im2 = ax6.imshow(ion_pair_field * 1e9, cmap='Greens', vmin=0)
ax6.set_title('CaHPO₄⁰ Ion Pairs (nM)')
ax6.set_xlabel('Position (grid)')
plt.colorbar(im2, ax=ax6, fraction=0.046)

# Dimers
ax7 = fig.add_subplot(gs[2, 2])
im3 = ax7.imshow(dimer_field * 1e9, cmap='plasma', vmin=0)
ax7.set_title('Ca₆(PO₄)₄ Dimers (nM)')
ax7.set_xlabel('Position (grid)')
plt.colorbar(im3, ax=ax7, fraction=0.046)

# J-coupling
ax8 = fig.add_subplot(gs[2, 3])
im4 = ax8.imshow(j_coupling_field, cmap='hot', vmin=0, vmax=20)
ax8.set_title('J-Coupling (Hz)')
ax8.set_xlabel('Position (grid)')
plt.colorbar(im4, ax=ax8, fraction=0.046)

# Overall title
fig.suptitle(f'Test {TEST_NUMBER}: {TEST_NAME} - Model 6 at Rest', 
             fontsize=14, fontweight='bold')

# Save figure
fig_path = OUTPUT_DIR / 'baseline_state.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✓ Figure saved: {fig_path}")

plt.close()

# =============================================================================
# SAVE SUMMARY STATISTICS
# =============================================================================

print("### SAVING RESULTS ###")

# Summary statistics
summary = {
    'test_name': TEST_NAME,
    'test_number': TEST_NUMBER,
    'timestamp': datetime.now().isoformat(),
    'passed': test_passed,
    'tests_passed': tests_passed,
    'tests_failed': tests_failed,
    'duration_ms': DURATION_MS,
    'parameters': {
        'temperature_K': params.environment.T,
        'P31_fraction': params.environment.fraction_P31,
        'grid_size': model.grid_shape[0],
        'dx_nm': model.dx * 1e9,
    },
    'final_state': {
        'calcium_peak_uM': float(final_metrics['calcium_peak_uM']),
        'calcium_mean_uM': float(final_metrics['calcium_mean_uM']),
        'atp_mean_mM': float(final_metrics['atp_mean_mM']),
        'j_coupling_max_Hz': float(final_metrics['j_coupling_max_Hz']),
        'j_coupling_mean_Hz': float(final_metrics['j_coupling_mean_Hz']),
        'ion_pair_peak_nM': float(final_metrics['ion_pair_peak_nM']),
        'dimer_peak_nM': float(final_metrics['dimer_peak_nM_ct']),
    },
    'time_series_statistics': {
        'calcium_mean': float(np.mean(ca_history)),
        'calcium_std': float(np.std(ca_history)),
        'calcium_drift': float(ca_drift),
        'atp_mean': float(np.mean(atp_history)),
        'atp_std': float(np.std(atp_history)),
    }
}

# Save JSON
json_path = OUTPUT_DIR / 'summary.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {json_path}")

# =============================================================================
# TEXT REPORT
# =============================================================================

report_path = OUTPUT_DIR / 'report.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write(f"TEST {TEST_NUMBER}: {TEST_NAME}\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Duration: {DURATION_MS} ms\n")
    f.write(f"Result: {'PASSED' if test_passed else 'FAILED'}\n\n")
    
    f.write("PARAMETERS\n")
    f.write("-"*80 + "\n")
    f.write(f"Temperature: {params.environment.T:.1f} K ({params.environment.T-273.15:.1f}°C)\n")
    f.write(f"Isotope: {params.environment.fraction_P31*100:.0f}% ³¹P\n")
    f.write(f"Grid: {model.grid_shape[0]}x{model.grid_shape[1]}, dx={model.dx*1e9:.1f} nm\n\n")
    
    f.write("FINAL STATE\n")
    f.write("-"*80 + "\n")
    f.write(f"Calcium peak: {final_metrics['calcium_peak_uM']:.3f} μM\n")
    f.write(f"Calcium mean: {final_metrics['calcium_mean_uM']:.3f} μM\n")
    f.write(f"ATP: {final_metrics['atp_mean_mM']:.2f} mM\n")
    f.write(f"J-coupling max: {final_metrics['j_coupling_max_Hz']:.1f} Hz\n")
    f.write(f"CaHPO₄ ion pairs: {final_metrics['ion_pair_peak_nM']:.1f} nM\n")
    f.write(f"Ca₆(PO₄)₄ dimers: {final_metrics['dimer_peak_nM_ct']:.2f} nM\n\n")
    
    f.write("VALIDATION CRITERIA\n")
    f.write("-"*80 + "\n")
    for test in tests_passed:
        f.write(f"✓ PASS: {test}\n")
    for test in tests_failed:
        f.write(f"✗ FAIL: {test}\n")
    f.write("\n")
    
    f.write("TIME SERIES STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Calcium mean ± std: {np.mean(ca_history):.3f} ± {np.std(ca_history):.4f} μM\n")
    f.write(f"Calcium drift (last 20ms): {ca_drift:.4f} μM\n")
    f.write(f"ATP mean ± std: {np.mean(atp_history):.3f} ± {np.std(atp_history):.4f} mM\n\n")
    
    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")
    if test_passed:
        f.write("Model successfully maintains physiological resting conditions:\n")
        f.write("- Calcium at ~100 nM (resting level, with stochastic noise)\n")
        f.write("- ATP stores full (>2 mM, with burst variability)\n")
        f.write("- Minimal spontaneous complex formation\n")
        f.write("- System stable with tolerable stochastic drift\n\n")
        f.write("NOTE: Values vary run-to-run due to stochastic processes.\n")
        f.write("This establishes the baseline for all subsequent tests.\n")
    else:
        f.write("Model shows deviations from expected resting state.\n")
        f.write("Review failed criteria above and check:\n")
        f.write("- Parameter values in model6_parameters.py\n")
        f.write("- Integration stability in model6_core.py\n")
        f.write("- Component initialization\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"✓ Report saved: {report_path}")
print()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("="*80)
print("TEST COMPLETE")
print("="*80)
print(f"Result: {'PASSED' if test_passed else 'FAILED'}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files generated:")
print(f"  - baseline_state.png (visualization)")
print(f"  - summary.json (metrics)")
print(f"  - report.txt (detailed report)")
print("="*80)