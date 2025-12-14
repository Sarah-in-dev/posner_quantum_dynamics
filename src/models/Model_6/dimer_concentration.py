#!/usr/bin/env python3
"""
DIMER FORMATION DIAGNOSTIC
==========================
Traces through every step of dimer calculation to find the discrepancy.

VALIDATED REFERENCE (October 7):
- Active zone volume: 0.01 μm³ = 10⁻¹⁷ L  
- Concentration → molecules: [nM] × 0.006 = dimers
- Target: 741 nM → 4.45 dimers per synapse

CURRENT OBSERVATION:
- 23,658 nM → 142 dimers per synapse
- This is 32× too high

This script traces each step to find where the calculation diverges.
"""

import sys
from pathlib import Path

# Add model path
MODEL_PATH = Path(__file__).resolve().parent.parent
if str(MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(MODEL_PATH))

import numpy as np

print("="*80)
print("DIMER FORMATION DIAGNOSTIC")
print("="*80)

# =============================================================================
# PART 1: FUNDAMENTAL CONSTANTS
# =============================================================================
print("\n" + "="*80)
print("PART 1: FUNDAMENTAL CONSTANTS")
print("="*80)

# Physical constants
N_A = 6.022e23  # Avogadro's number

# Active zone geometry (literature values)
az_diameter = 200e-9  # 200 nm (Harris & Weinberg 2012)
az_height = 50e-9     # 50 nm
az_volume_m3 = np.pi * (az_diameter/2)**2 * az_height
az_volume_L = az_volume_m3 * 1000  # m³ to L

print(f"\nActive Zone Geometry:")
print(f"  Diameter: {az_diameter*1e9:.0f} nm")
print(f"  Height: {az_height*1e9:.0f} nm")
print(f"  Volume: {az_volume_m3:.2e} m³")
print(f"  Volume: {az_volume_L:.2e} L")
print(f"  Volume: {az_volume_m3*1e18:.2f} μm³")

# Conversion factor
conversion_factor = az_volume_L * N_A * 1e-9  # [nM] → molecules
print(f"\nConversion factor: {conversion_factor:.4f}")
print(f"  [nM] × {conversion_factor:.4f} = molecules")
print(f"  741 nM × {conversion_factor:.4f} = {741 * conversion_factor:.2f} molecules")

# =============================================================================
# PART 2: MODEL GRID VS ACTIVE ZONE
# =============================================================================
print("\n" + "="*80)
print("PART 2: MODEL GRID VS ACTIVE ZONE")
print("="*80)

try:
    from model6_parameters import Model6Parameters
    params = Model6Parameters()
    
    # Grid parameters
    grid_size = params.spatial.grid_size
    dx = 4e-9  # 4 nm resolution (typical)
    
    grid_side = grid_size * dx
    grid_area = grid_side**2
    grid_volume_2D = grid_area  # For 2D simulation
    
    print(f"\nModel Grid:")
    print(f"  Size: {grid_size} × {grid_size}")
    print(f"  dx: {dx*1e9:.1f} nm")
    print(f"  Physical side: {grid_side*1e9:.0f} nm")
    print(f"  Physical area: {grid_area*1e18:.2f} nm²")
    
    # Is model 2D or 3D?
    print(f"\n  Grid is 2D - concentrations are 2D averages")
    print(f"  Need to interpret as volume concentration")
    
except Exception as e:
    print(f"Could not load parameters: {e}")

# =============================================================================
# PART 3: TRACK DIMER FORMATION IN MODEL
# =============================================================================
print("\n" + "="*80)
print("PART 3: TRACK DIMER FORMATION STEP BY STEP")
print("="*80)

try:
    from model6_core import Model6QuantumSynapse
    from model6_parameters import Model6Parameters
    
    # Initialize with EM coupling (as experiments use)
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = False
    
    model = Model6QuantumSynapse(params)
    
    # Get initial k_base
    k_base = model.ca_phosphate.dimerization.k_base
    k_diss = model.ca_phosphate.dimerization.k_dissociation
    print(f"\nRate Constants:")
    print(f"  k_base (aggregation): {k_base:.2e} M⁻¹s⁻¹")
    print(f"  k_diss (dissociation): {k_diss:.2e} s⁻¹")
    print(f"  Dimer half-life: {np.log(2)/k_diss:.0f} seconds")
    
    # Track formation over simulation
    print(f"\n--- Running Simulation (500 steps, 1ms each) ---")
    print(f"{'Step':<8} {'Ca(μM)':<10} {'IonPair(nM)':<14} {'Dimer(nM)':<12} {'k_enh':<8} {'Field(kT)':<10}")
    print("-"*70)
    
    dt = 0.001  # 1 ms
    for i in range(500):
        model.step(dt, {'voltage': -20e-3, 'activity_level': 0.8})
        
        if i % 100 == 0 or i == 499:
            metrics = model.get_experimental_metrics()
            ca_uM = metrics.get('calcium_peak_uM', 0)
            ion_pair_nM = metrics.get('ion_pair_peak_nM', 0)
            dimer_nM = metrics.get('dimer_peak_nM_ct', 0)
            k_enh = getattr(model, '_k_enhancement', 1.0)
            field_kT = getattr(model, '_collective_field_kT', 0)
            
            print(f"{i:<8} {ca_uM:<10.2f} {ion_pair_nM:<14.1f} {dimer_nM:<12.1f} {k_enh:<8.2f} {field_kT:<10.1f}")
    
    # Final analysis
    print("\n--- Final Analysis ---")
    final_metrics = model.get_experimental_metrics()
    dimer_peak = final_metrics.get('dimer_peak_nM_ct', 0)
    dimer_mean = final_metrics.get('dimer_mean_nM_ct', 0)
    
    print(f"\nFinal Dimer Concentration:")
    print(f"  Peak: {dimer_peak:.1f} nM")
    print(f"  Mean: {dimer_mean:.1f} nM")
    
    # Convert to molecules
    dimers_using_peak = dimer_peak * conversion_factor
    dimers_using_mean = dimer_mean * conversion_factor
    
    print(f"\nConverted to Molecules (active zone volume):")
    print(f"  Using peak: {dimer_peak:.1f} nM × {conversion_factor:.4f} = {dimers_using_peak:.1f} dimers")
    print(f"  Using mean: {dimer_mean:.1f} nM × {conversion_factor:.4f} = {dimers_using_mean:.1f} dimers")
    
    # What model uses
    model_dimer_count = getattr(model, '_previous_dimer_count', 0)
    print(f"\nModel's internal dimer count: {model_dimer_count:.1f}")
    
    # Check volume used in model
    print("\n--- Model's Volume Calculation ---")
    # This is what model6_core.py uses:
    processing_volume_m3 = 5.2e-22  # From code
    processing_volume_L = processing_volume_m3 * 1000
    n_templates = 3
    model_formula = dimer_peak * 1e-9 * processing_volume_L * n_templates * N_A
    
    print(f"  model6_core uses processing_volume = {processing_volume_m3:.2e} m³")
    print(f"  With n_templates = {n_templates}")
    print(f"  Formula: {dimer_peak:.1f} nM × {processing_volume_L:.2e} L × {n_templates} × N_A")
    print(f"  Result: {model_formula:.1f} dimers")
    
    # Correct volume
    print(f"\n  Should use active_zone_volume = {az_volume_m3:.2e} m³")
    correct_formula = dimer_peak * 1e-9 * az_volume_L * N_A
    print(f"  Correct: {dimer_peak:.1f} nM × {az_volume_L:.2e} L × N_A")
    print(f"  Result: {correct_formula:.1f} dimers")
    
except Exception as e:
    import traceback
    print(f"Error running model: {e}")
    traceback.print_exc()

# =============================================================================
# PART 4: COMPARE TO VALIDATED OCT 18 RESULTS
# =============================================================================
print("\n" + "="*80)
print("PART 4: COMPARISON TO VALIDATED RESULTS")
print("="*80)

print("""
VALIDATED (October 18, 2025):
-----------------------------
  Ion pairs: 5,690 nM
  Dimers: 740.71 nM  
  Template enhancement: 699.2×
  
CURRENT OBSERVATION:
-------------------
  Ion pairs: ??? nM
  Dimers: 23,658 nM (from your test)
  
RATIO: 23,658 / 741 = 31.9× too high

POSSIBLE CAUSES:
----------------
1. Simulation duration: Oct 18 ran for X ms, current runs for Y ms
   → Dimers accumulate (k_diss is very slow)
   
2. Template enhancement changed:
   → Oct 18 used 699×, current might be different
   
3. EM coupling enhancement stacking:
   → k_agg × em_enhancement × template_enhancement = too high
   
4. PNC calculation changed:
   → Stochastic nucleation adding extra dimers
""")

# =============================================================================
# PART 5: ISOLATION TEST - NO EM COUPLING
# =============================================================================
print("\n" + "="*80)
print("PART 5: ISOLATION TEST - NO EM COUPLING")
print("="*80)

try:
    # Run without EM coupling to see baseline
    params_no_em = Model6Parameters()
    params_no_em.em_coupling_enabled = False
    params_no_em.multi_synapse_enabled = False
    
    model_no_em = Model6QuantumSynapse(params_no_em)
    
    print("Running 500ms without EM coupling...")
    for i in range(500):
        model_no_em.step(0.001, {'voltage': -20e-3, 'activity_level': 0.8})
    
    metrics_no_em = model_no_em.get_experimental_metrics()
    dimer_no_em = metrics_no_em.get('dimer_peak_nM_ct', 0)
    
    print(f"\nWithout EM coupling:")
    print(f"  Dimer peak: {dimer_no_em:.1f} nM")
    print(f"  Molecules: {dimer_no_em * conversion_factor:.1f} dimers")
    
    print(f"\nWith EM coupling (from Part 3):")
    print(f"  Dimer peak: {dimer_peak:.1f} nM")
    print(f"  Molecules: {dimers_using_peak:.1f} dimers")
    
    em_enhancement_actual = dimer_peak / dimer_no_em if dimer_no_em > 0 else float('inf')
    print(f"\nEM coupling effect: {em_enhancement_actual:.1f}×")
    
except Exception as e:
    print(f"Error: {e}")

# =============================================================================
# PART 6: RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("PART 6: DIAGNOSTIC SUMMARY")
print("="*80)

print("""
NEXT STEPS:
-----------
1. If dimer concentration is ~741 nM (validated):
   → Problem is ONLY in molecule count formula
   → Fix: Use active zone volume (10⁻¹⁷ L) not processing volume (5.2e-19 L)

2. If dimer concentration is ~23,000 nM:
   → Problem is in dimer FORMATION (too fast) or ACCUMULATION (not decaying)
   → Check: Template enhancement factor
   → Check: Simulation duration
   → Check: k_dissociation rate
   
3. Run this script and report back which values you see
""")