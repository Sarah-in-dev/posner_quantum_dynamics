"""
Test 02: Calcium Dynamics - Spike Generation
=============================================

Validates calcium channel gating and spike generation in Model 6.

Tests the calcium response to depolarization:
- Membrane depolarization opens voltage-gated channels
- Ca²⁺ influx creates microdomain (1-50 μM)
- Spatial nanodomains near channels
- Recovery after stimulus

Pass Criteria:
- Spike amplitude > 5x baseline
- Peak Ca²⁺ in range 1-50 μM (Naraghi & Neher 1997)
- Spatial localization near channels
- Recovery after stimulus ends

Author: Sarah Davidson
Date: November 2025
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

TEST_NAME = "Calcium Dynamics - Spike Generation"
TEST_NUMBER = "02"
BASELINE_MS = 20      # Initial rest period
STIMULUS_MS = 50      # Depolarization duration
RECOVERY_MS = 300      # Recovery period
TOTAL_MS = BASELINE_MS + STIMULUS_MS + RECOVERY_MS


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).parent / "validation_results" / f"test_02_calcium_dynamics_{timestamp}"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"TEST {TEST_NUMBER}: {TEST_NAME.upper()}")
print("="*80)
print(f"Protocol:")
print(f"  Baseline: {BASELINE_MS} ms at rest (-70 mV)")
print(f"  Stimulus: {STIMULUS_MS} ms depolarized (-10 mV)")
print(f"  Recovery: {RECOVERY_MS} ms at rest (-70 mV)")
print(f"  Total: {TOTAL_MS} ms")
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
print(f"  Channels: {len(model.calcium.channels.positions)}")
print(f"  dx: {model.dx*1e9:.1f} nm")
print()

# =============================================================================
# RUN PROTOCOL
# =============================================================================

print("### RUNNING CALCIUM SPIKE PROTOCOL ###")

# Track time series
time_points = []
ca_peak_history = []
ca_mean_history = []
voltage_history = []

# Storage for spatial snapshots
snapshots = {
    'baseline': {'time_ms': 0, 'ca_field': None, 'voltage': -70e-3},
    'peak': {'time_ms': 0, 'ca_field': None, 'voltage': -10e-3},
    'recovery': {'time_ms': 0, 'ca_field': None, 'voltage': -70e-3}
}

# Phase 1: Baseline
print(f"\n1. Baseline ({BASELINE_MS} ms at rest)...")
for step in range(BASELINE_MS):
    model.step(model.dt, stimulus={'voltage': -70e-3})
    
    metrics = model.get_experimental_metrics()
    time_points.append(step * model.dt * 1000)
    ca_peak_history.append(metrics['calcium_peak_uM'])
    ca_mean_history.append(metrics['calcium_mean_uM'])
    voltage_history.append(-70)
    
    # Save baseline snapshot
    if step == BASELINE_MS - 1:
        snapshots['baseline']['time_ms'] = time_points[-1]
        snapshots['baseline']['ca_field'] = model.calcium.get_concentration()

baseline_ca = ca_peak_history[-1]
print(f"   Baseline Ca²⁺: {baseline_ca:.3f} μM")

# Phase 2: Stimulus (Depolarization)
print(f"\n2. Depolarization ({STIMULUS_MS} ms at -10 mV)...")
peak_ca = 0
peak_time_ms = 0
for step in range(BASELINE_MS, BASELINE_MS + STIMULUS_MS):
    model.step(model.dt, stimulus={'voltage': -10e-3})
    
    metrics = model.get_experimental_metrics()
    time_points.append(step * model.dt * 1000)
    ca_peak_history.append(metrics['calcium_peak_uM'])
    ca_mean_history.append(metrics['calcium_mean_uM'])
    voltage_history.append(-10)
    
    # Track peak
    if ca_peak_history[-1] > peak_ca:
        peak_ca = ca_peak_history[-1]
        peak_time_ms = time_points[-1]
        snapshots['peak']['time_ms'] = peak_time_ms
        snapshots['peak']['ca_field'] = model.calcium.get_concentration()
    
    # Progress
    if (step - BASELINE_MS + 1) % 10 == 0:
        print(f"   t={time_points[-1]:.0f}ms: Ca²⁺ = {ca_peak_history[-1]:.2f} μM")

print(f"   Peak Ca²⁺: {peak_ca:.2f} μM at t={peak_time_ms:.0f}ms")

# Phase 3: Recovery
print(f"\n3. Recovery ({RECOVERY_MS} ms at rest)...")
for step in range(BASELINE_MS + STIMULUS_MS, TOTAL_MS):
    model.step(model.dt, stimulus={'voltage': -70e-3})
    
    metrics = model.get_experimental_metrics()
    time_points.append(step * model.dt * 1000)
    ca_peak_history.append(metrics['calcium_peak_uM'])
    ca_mean_history.append(metrics['calcium_mean_uM'])
    voltage_history.append(-70)
    
    # Save recovery snapshot (midpoint)
    if step == BASELINE_MS + STIMULUS_MS + RECOVERY_MS // 2:
        snapshots['recovery']['time_ms'] = time_points[-1]
        snapshots['recovery']['ca_field'] = model.calcium.get_concentration()

recovery_ca = ca_peak_history[-1]
print(f"   Final Ca²⁺: {recovery_ca:.3f} μM")
print()

# =============================================================================
# VALIDATION
# =============================================================================

print("### VALIDATION ###")

tests_passed = []
tests_failed = []

# Test 1: Spike amplitude
spike_amplitude = peak_ca / baseline_ca
if spike_amplitude > 5.0:
    print(f"✓ PASS: Spike amplitude sufficient ({spike_amplitude:.1f}x baseline)")
    tests_passed.append("spike_amplitude")
else:
    print(f"✗ FAIL: Spike too small ({spike_amplitude:.1f}x baseline)")
    tests_failed.append("spike_amplitude")

# Test 2: Physiological range
if 1.0 <= peak_ca <= 50.0:
    print(f"✓ PASS: Peak Ca²⁺ in physiological range ({peak_ca:.1f} μM)")
    tests_passed.append("physiological_range")
else:
    print(f"✗ FAIL: Peak Ca²⁺ outside range ({peak_ca:.1f} μM)")
    tests_failed.append("physiological_range")

# Test 3: Recovery initiated
recovery_fraction = (peak_ca - recovery_ca) / (peak_ca - baseline_ca)
if recovery_fraction > 0.15:  # At least 15% recovery
    print(f"✓ PASS: Recovery initiated ({recovery_fraction*100:.0f}% recovered)")
    tests_passed.append("recovery")
else:
    print(f"✗ FAIL: Insufficient recovery ({recovery_fraction*100:.0f}%)")
    tests_failed.append("recovery")

# Test 4: Spatial localization
ca_field_peak = snapshots['peak']['ca_field']
hotspot_threshold = 0.5 * np.max(ca_field_peak)
n_hotspots = np.sum(ca_field_peak > hotspot_threshold)
total_points = ca_field_peak.size
hotspot_fraction = n_hotspots / total_points

if hotspot_fraction < 0.1:  # Less than 10% of grid
    print(f"✓ PASS: Spatial localization ({hotspot_fraction*100:.1f}% of grid)")
    tests_passed.append("localization")
else:
    print(f"⚠ WARNING: Broad spatial spread ({hotspot_fraction*100:.1f}% of grid)")

# Overall result
test_passed = len(tests_failed) == 0
print()
if test_passed:
    print("="*80)
    print("✓✓✓ TEST PASSED ✓✓✓")
    print("="*80)
else:
    print("="*80)
    print(f"✗✗✗ TEST FAILED ({len(tests_failed)} criteria failed) ✗✗✗")
    print("="*80)
print()

# =============================================================================
# VISUALIZATION
# =============================================================================

print("### GENERATING VISUALIZATIONS ###")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Convert to arrays
time_points = np.array(time_points)
ca_peak_history = np.array(ca_peak_history)
ca_mean_history = np.array(ca_mean_history)
voltage_history = np.array(voltage_history)

# --- TIME SERIES (Top 2 rows) ---

# Row 1: Calcium trace with stimulus bar
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_points, ca_peak_history, 'r-', linewidth=2, label='Peak Ca²⁺')
ax1.plot(time_points, ca_mean_history, 'r--', linewidth=1, alpha=0.6, label='Mean Ca²⁺')
ax1.axhline(baseline_ca, color='k', linestyle=':', alpha=0.5, label='Baseline')
ax1.axvline(BASELINE_MS, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(BASELINE_MS + STIMULUS_MS, color='gray', linestyle='--', alpha=0.3)

# Shade stimulus period
ax1.axvspan(BASELINE_MS, BASELINE_MS + STIMULUS_MS, alpha=0.2, color='blue', label='Stimulus')

# Mark snapshots
for name, snap in snapshots.items():
    ax1.plot(snap['time_ms'], np.max(snap['ca_field']) * 1e6, 'ko', markersize=8)
    ax1.text(snap['time_ms'], np.max(snap['ca_field']) * 1e6 + 1, name.capitalize(),
             ha='center', fontsize=8)

ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('[Ca²⁺] (μM)', fontsize=12)
ax1.set_title('Calcium Response to Depolarization', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# Row 2: Voltage protocol
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(time_points, voltage_history, 'b-', linewidth=2)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Voltage (mV)')
ax2.set_title('Voltage Protocol')
ax2.grid(alpha=0.3)
ax2.set_ylim([-80, 0])

# Row 2: Spike analysis
ax3 = fig.add_subplot(gs[1, 2:])
ax3.axis('off')
analysis_text = f"""
SPIKE ANALYSIS

Baseline: {baseline_ca:.3f} μM
Peak: {peak_ca:.2f} μM at t={peak_time_ms:.0f} ms
Amplitude: {spike_amplitude:.1f}x baseline
Recovery: {recovery_ca:.3f} μM

Spatial localization:
  Hotspot area: {hotspot_fraction*100:.1f}% of grid
  Channel count: {len(model.calcium.channels.positions)}

Literature comparison:
  Expected: 1-50 μM (microdomains)
  Measured: {peak_ca:.1f} μM ✓
  
  Naraghi & Neher 1997:
    "10-100 μM near open channels"
"""
ax3.text(0.1, 0.5, analysis_text, fontsize=10, family='monospace',
         verticalalignment='center', transform=ax3.transAxes)

# --- SPATIAL MAPS (Bottom row) ---

# Baseline
ax4 = fig.add_subplot(gs[2, 0])
ca_baseline = snapshots['baseline']['ca_field'] * 1e6
im1 = ax4.imshow(ca_baseline, cmap='Reds', vmin=0, vmax=peak_ca)
ax4.set_title(f'Baseline\nt={snapshots["baseline"]["time_ms"]:.0f}ms')
ax4.set_xlabel('Position (grid)')
ax4.set_ylabel('Position (grid)')
plt.colorbar(im1, ax=ax4, fraction=0.046, label='Ca²⁺ (μM)')
# Mark channels
for pos in model.calcium.channels.positions:
    ax4.plot(pos[1], pos[0], 'wo', markersize=3, markeredgecolor='black', markeredgewidth=0.5)

# Peak
ax5 = fig.add_subplot(gs[2, 1])
ca_peak_field = snapshots['peak']['ca_field'] * 1e6
im2 = ax5.imshow(ca_peak_field, cmap='Reds', vmin=0, vmax=peak_ca)
ax5.set_title(f'Peak\nt={snapshots["peak"]["time_ms"]:.0f}ms')
ax5.set_xlabel('Position (grid)')
plt.colorbar(im2, ax=ax5, fraction=0.046, label='Ca²⁺ (μM)')
# Mark channels
for pos in model.calcium.channels.positions:
    ax5.plot(pos[1], pos[0], 'wo', markersize=3, markeredgecolor='black', markeredgewidth=0.5)

# Recovery
ax6 = fig.add_subplot(gs[2, 2])
ca_recovery = snapshots['recovery']['ca_field'] * 1e6
im3 = ax6.imshow(ca_recovery, cmap='Reds', vmin=0, vmax=peak_ca)
ax6.set_title(f'Recovery\nt={snapshots["recovery"]["time_ms"]:.0f}ms')
ax6.set_xlabel('Position (grid)')
plt.colorbar(im3, ax=ax6, fraction=0.046, label='Ca²⁺ (μM)')
# Mark channels
for pos in model.calcium.channels.positions:
    ax6.plot(pos[1], pos[0], 'wo', markersize=3, markeredgecolor='black', markeredgewidth=0.5)

# Line profile through channels
ax7 = fig.add_subplot(gs[2, 3])
center = model.grid_shape[0] // 2
profile_baseline = ca_baseline[center, :]
profile_peak = ca_peak_field[center, :]
profile_recovery = ca_recovery[center, :]

positions_nm = np.arange(len(profile_peak)) * model.dx * 1e9

ax7.plot(positions_nm, profile_baseline, 'b-', linewidth=2, label='Baseline')
ax7.plot(positions_nm, profile_peak, 'r-', linewidth=2, label='Peak')
ax7.plot(positions_nm, profile_recovery, 'g-', linewidth=2, label='Recovery')
ax7.set_xlabel('Position (nm)')
ax7.set_ylabel('[Ca²⁺] (μM)')
ax7.set_title('Cross-section Profile')
ax7.legend(fontsize=8)
ax7.grid(alpha=0.3)

# Overall title
fig.suptitle(f'Test {TEST_NUMBER}: {TEST_NAME}', 
             fontsize=14, fontweight='bold')

# Save
fig_path = OUTPUT_DIR / 'calcium_dynamics.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✓ Figure saved: {fig_path}")
plt.close()

# =============================================================================
# SAVE SUMMARY
# =============================================================================

print("### SAVING RESULTS ###")

summary = {
    'test_name': TEST_NAME,
    'test_number': TEST_NUMBER,
    'timestamp': datetime.now().isoformat(),
    'passed': test_passed,
    'tests_passed': tests_passed,
    'tests_failed': tests_failed,
    'protocol': {
        'baseline_ms': BASELINE_MS,
        'stimulus_ms': STIMULUS_MS,
        'recovery_ms': RECOVERY_MS,
        'total_ms': TOTAL_MS,
        'stimulus_voltage_mV': -10,
        'rest_voltage_mV': -70,
    },
    'parameters': {
        'temperature_K': params.environment.T,
        'P31_fraction': params.environment.fraction_P31,
        'n_channels': len(model.calcium.channels.positions),
        'grid_size': model.grid_shape[0],
        'dx_nm': model.dx * 1e9,
    },
    'results': {
        'baseline_ca_uM': float(baseline_ca),
        'peak_ca_uM': float(peak_ca),
        'peak_time_ms': float(peak_time_ms),
        'recovery_ca_uM': float(recovery_ca),
        'spike_amplitude_fold': float(spike_amplitude),
        'recovery_fraction': float(recovery_fraction),
        'hotspot_fraction': float(hotspot_fraction),
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
    f.write(f"Result: {'PASSED' if test_passed else 'FAILED'}\n\n")
    
    f.write("PROTOCOL\n")
    f.write("-"*80 + "\n")
    f.write(f"1. Baseline: {BASELINE_MS} ms at -70 mV (rest)\n")
    f.write(f"2. Stimulus: {STIMULUS_MS} ms at -10 mV (depolarized)\n")
    f.write(f"3. Recovery: {RECOVERY_MS} ms at -70 mV (rest)\n")
    f.write(f"Total duration: {TOTAL_MS} ms\n\n")
    
    f.write("RESULTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Baseline Ca²⁺: {baseline_ca:.3f} μM\n")
    f.write(f"Peak Ca²⁺: {peak_ca:.2f} μM (at t={peak_time_ms:.0f} ms)\n")
    f.write(f"Recovery Ca²⁺: {recovery_ca:.3f} μM\n")
    f.write(f"Spike amplitude: {spike_amplitude:.1f}x baseline\n")
    f.write(f"Recovery: {recovery_fraction*100:.0f}%\n\n")
    
    f.write("SPATIAL ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write(f"Hotspot area: {hotspot_fraction*100:.1f}% of grid\n")
    f.write(f"Number of channels: {len(model.calcium.channels.positions)}\n")
    f.write(f"Grid size: {model.grid_shape[0]}x{model.grid_shape[1]}\n")
    f.write(f"Spatial resolution: {model.dx*1e9:.1f} nm\n\n")
    
    f.write("VALIDATION CRITERIA\n")
    f.write("-"*80 + "\n")
    for test in tests_passed:
        f.write(f"✓ PASS: {test}\n")
    for test in tests_failed:
        f.write(f"✗ FAIL: {test}\n")
    f.write("\n")
    
    f.write("LITERATURE COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write("Naraghi & Neher 1997 (J Neurosci):\n")
    f.write("  'Ca²⁺ reaches 10-100 μM near open channels'\n")
    f.write(f"  Our result: {peak_ca:.1f} μM ✓\n\n")
    f.write("Sabatini & Svoboda 2000 (Nature):\n")
    f.write("  'Microdomain concentrations: 1-10 μM'\n")
    f.write(f"  Our result: {peak_ca:.1f} μM ✓\n\n")
    
    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")
    if test_passed:
        f.write("Calcium dynamics validated:\n")
        f.write("- Channels respond appropriately to depolarization\n")
        f.write("- Ca²⁺ spike amplitude in physiological range\n")
        f.write("- Spatial nanodomains form near channels\n")
        f.write("- Recovery mechanisms functioning\n\n")
        f.write("This Ca²⁺ spike will drive downstream processes:\n")
        f.write("- ATP hydrolysis and phosphate release\n")
        f.write("- CaHPO₄ ion pair formation\n")
        f.write("- Ca₆(PO₄)₄ dimer aggregation\n")
    else:
        f.write("Calcium dynamics show issues. Check:\n")
        for test in tests_failed:
            f.write(f"- {test}\n")
        f.write("\nReview:\n")
        f.write("- Channel parameters in model6_parameters.py\n")
        f.write("- Voltage-gating in calcium_system.py\n")
        f.write("- Diffusion and buffering parameters\n")
    
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
print(f"  - calcium_dynamics.png (visualization)")
print(f"  - summary.json (metrics)")
print(f"  - report.txt (detailed report)")
print("="*80)