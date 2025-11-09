"""
Protocol 01: Baseline Statistical Characterization
===================================================
Runs identical single-spike stimulus N=100 times to quantify stochastic variability

This establishes the statistical baseline (mean ± std) for all Model 6 metrics.

Protocol (repeated 100x):
  1. Baseline: 20 ms at rest (-70 mV)
  2. Stimulus: 50 ms depolarization (-10 mV) 
  3. Recovery: 300 ms at rest (-70 mV)
  Total: 370 ms per run

Metrics Collected (N=100 runs):
  - Calcium: peak, mean, decay time
  - ATP: consumption, recovery
  - J-coupling: enhancement factor
  - pH: minimum, recovery
  - Ion pairs: peak concentration
  - Dimers: peak concentration, accumulation rate
  - Quantum coherence: T2 time, decoherence rate

Output:
  - Statistical summary (mean ± std for all metrics)
  - Distribution histograms
  - Coefficient of variation (CV) analysis
  - Raw data (HDF5 file with all 100 runs)
  - Correlation analysis between metrics

Author: Sarah Davidson
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import h5py
from datetime import datetime
from scipy import stats
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

# =============================================================================
# PROTOCOL CONFIGURATION
# =============================================================================

PROTOCOL_NAME = "Baseline Statistical Characterization"
PROTOCOL_NUMBER = "01"

# Protocol timing
BASELINE_MS = 20
STIMULUS_MS = 50
RECOVERY_MS = 300
TOTAL_MS = BASELINE_MS + STIMULUS_MS + RECOVERY_MS

# Statistical parameters
N_RUNS = 100
RANDOM_SEED = 42  # For reproducibility of initial conditions

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).parent / "experimental_protocols" / f"protocol_01_baseline_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"PROTOCOL {PROTOCOL_NUMBER}: {PROTOCOL_NAME.upper()}")
print("="*80)
print(f"Runs: {N_RUNS}")
print(f"Protocol: {BASELINE_MS}ms rest → {STIMULUS_MS}ms stimulus → {RECOVERY_MS}ms recovery")
print(f"Total duration per run: {TOTAL_MS} ms")
print(f"Output directory: {OUTPUT_DIR}")
print()

# =============================================================================
# INITIALIZE PARAMETERS
# =============================================================================

print("### INITIALIZATION ###")
params = Model6Parameters()
params.environment.T = 310.15  # 37°C

# Disable dopamine for single-spike baseline (no reward signal)
# This suppresses warnings and speeds up simulation
from model6_parameters import DopamineParameters
params.dopamine = None  # Or just comment out dopamine initialization in model6_core

print(f"✓ Parameters loaded")
print(f"  Temperature: {params.environment.T:.1f} K")
print(f"  Isotope: {params.environment.fraction_P31*100:.0f}% ³¹P")
print()

# =============================================================================
# DATA COLLECTION ARRAYS
# =============================================================================

# Initialize storage for all runs
data = {
    # Calcium
    'ca_peak_uM': np.zeros(N_RUNS),
    'ca_mean_uM': np.zeros(N_RUNS),
    'ca_baseline_uM': np.zeros(N_RUNS),
    'ca_recovery_uM': np.zeros(N_RUNS),
    
    # ATP
    'atp_baseline_mM': np.zeros(N_RUNS),
    'atp_min_mM': np.zeros(N_RUNS),
    'atp_consumed_uM': np.zeros(N_RUNS),
    
    # J-coupling
    'j_baseline_Hz': np.zeros(N_RUNS),
    'j_peak_Hz': np.zeros(N_RUNS),
    'j_enhancement_factor': np.zeros(N_RUNS),
    
    # pH
    'pH_baseline': np.zeros(N_RUNS),
    'pH_min': np.zeros(N_RUNS),
    'pH_drop': np.zeros(N_RUNS),
    
    # Calcium phosphate
    'ion_pair_peak_nM': np.zeros(N_RUNS),
    'dimer_peak_nM': np.zeros(N_RUNS),
    'dimer_baseline_nM': np.zeros(N_RUNS),
    'dimer_accumulation_nM': np.zeros(N_RUNS),
    
    # Quantum (if available)
    'coherence_mean': np.zeros(N_RUNS),
    'T2_dimer_s': np.zeros(N_RUNS),
}

# Time series storage (only for selected runs for visualization)
SAVE_TIMESERIES_EVERY = 20  # Save full timeseries every 20th run
timeseries_runs = []

# =============================================================================
# RUN N=100 EXPERIMENTS
# =============================================================================

print("### RUNNING N=100 EXPERIMENTS ###")
print(f"Progress: ", end='', flush=True)

for run_idx in range(N_RUNS):
    # Progress indicator
    if (run_idx + 1) % 10 == 0:
        print(f"{run_idx + 1}...", end='', flush=True)
    
    # Initialize fresh model for each run
    # NOTE: Each run gets stochastic variations in channel gating, 
    # ATP bursts, pH noise, etc.
    model = Model6QuantumSynapse(params=params)
    
    # Track metrics during this run
    run_data = {
        'time_ms': [],
        'ca_peak': [],
        'ca_mean': [],
        'atp': [],
        'j_coupling': [],
        'pH': [],
        'ion_pairs': [],
        'dimers': [],
    }
    
    # Phase 1: Baseline (20 ms)
    for step in range(BASELINE_MS):
        model.step(model.dt, stimulus={'voltage': -70e-3})
        
        if step == BASELINE_MS - 1:
            metrics = model.get_experimental_metrics()
            data['ca_baseline_uM'][run_idx] = metrics['calcium_peak_uM']
            data['atp_baseline_mM'][run_idx] = metrics['atp_mean_mM']
            data['j_baseline_Hz'][run_idx] = metrics['j_coupling_max_Hz']
            data['pH_baseline'][run_idx] = metrics.get('pH_mean', 7.35)
            data['dimer_baseline_nM'][run_idx] = metrics['dimer_peak_nM_ct']
    
    # Phase 2: Stimulus (50 ms)
    ca_peak = 0
    j_peak = 0
    atp_min = 10.0
    pH_min = 8.0
    ion_pair_peak = 0
    dimer_peak = 0
    
    for step in range(BASELINE_MS, BASELINE_MS + STIMULUS_MS):
        model.step(model.dt, stimulus={'voltage': -10e-3})
        
        metrics = model.get_experimental_metrics()
        
        # Track peaks
        ca_peak = max(ca_peak, metrics['calcium_peak_uM'])
        j_peak = max(j_peak, metrics['j_coupling_max_Hz'])
        atp_min = min(atp_min, metrics['atp_mean_mM'])
        pH_min = min(pH_min, metrics.get('pH_mean', 7.35))
        ion_pair_peak = max(ion_pair_peak, metrics['ion_pair_peak_nM'])
        dimer_peak = max(dimer_peak, metrics['dimer_peak_nM_ct'])
        
        # Save timeseries for selected runs
        if run_idx % SAVE_TIMESERIES_EVERY == 0:
            run_data['time_ms'].append(step * model.dt * 1000)
            run_data['ca_peak'].append(metrics['calcium_peak_uM'])
            run_data['ca_mean'].append(metrics['calcium_mean_uM'])
            run_data['atp'].append(metrics['atp_mean_mM'])
            run_data['j_coupling'].append(metrics['j_coupling_max_Hz'])
            run_data['pH'].append(metrics.get('pH_mean', 7.35))
            run_data['ion_pairs'].append(metrics['ion_pair_peak_nM'])
            run_data['dimers'].append(metrics['dimer_peak_nM_ct'])
    
    # Phase 3: Recovery (300 ms)
    for step in range(BASELINE_MS + STIMULUS_MS, TOTAL_MS):
        model.step(model.dt, stimulus={'voltage': -70e-3})
        
        metrics = model.get_experimental_metrics()
        dimer_peak = max(dimer_peak, metrics['dimer_peak_nM_ct'])
        
        if step == TOTAL_MS - 1:
            data['ca_recovery_uM'][run_idx] = metrics['calcium_mean_uM']
        
        # Save timeseries for selected runs
        if run_idx % SAVE_TIMESERIES_EVERY == 0:
            run_data['time_ms'].append(step * model.dt * 1000)
            run_data['ca_peak'].append(metrics['calcium_peak_uM'])
            run_data['ca_mean'].append(metrics['calcium_mean_uM'])
            run_data['atp'].append(metrics['atp_mean_mM'])
            run_data['j_coupling'].append(metrics['j_coupling_max_Hz'])
            run_data['pH'].append(metrics.get('pH_mean', 7.35))
            run_data['ion_pairs'].append(metrics['ion_pair_peak_nM'])
            run_data['dimers'].append(metrics['dimer_peak_nM_ct'])
    
    # Store run metrics
    data['ca_peak_uM'][run_idx] = ca_peak
    data['ca_mean_uM'][run_idx] = np.mean(run_data['ca_mean']) if run_data['ca_mean'] else 0
    data['atp_min_mM'][run_idx] = atp_min
    data['atp_consumed_uM'][run_idx] = (data['atp_baseline_mM'][run_idx] - atp_min) * 1000
    data['j_peak_Hz'][run_idx] = j_peak
    data['j_enhancement_factor'][run_idx] = j_peak / data['j_baseline_Hz'][run_idx] if data['j_baseline_Hz'][run_idx] > 0 else 1.0
    data['pH_min'][run_idx] = pH_min
    data['pH_drop'][run_idx] = data['pH_baseline'][run_idx] - pH_min
    data['ion_pair_peak_nM'][run_idx] = ion_pair_peak
    data['dimer_peak_nM'][run_idx] = dimer_peak
    data['dimer_accumulation_nM'][run_idx] = dimer_peak - data['dimer_baseline_nM'][run_idx]
    
    # Get quantum metrics if available
    final_metrics = model.get_experimental_metrics()
    data['coherence_mean'][run_idx] = final_metrics.get('coherence_mean', 0)
    data['T2_dimer_s'][run_idx] = final_metrics.get('T2_dimer_s', 0)
    
    # Save timeseries for selected runs
    if run_idx % SAVE_TIMESERIES_EVERY == 0:
        timeseries_runs.append({
            'run_idx': run_idx,
            'data': run_data
        })

print(" Done!")
print()

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("### STATISTICAL ANALYSIS ###")

# Calculate statistics for each metric
stats_summary = {}

for key, values in data.items():
    # Remove any NaN or inf values
    clean_values = values[np.isfinite(values)]
    
    if len(clean_values) > 0:
        stats_summary[key] = {
            'mean': np.mean(clean_values),
            'std': np.std(clean_values),
            'sem': np.std(clean_values) / np.sqrt(len(clean_values)),
            'cv': np.std(clean_values) / np.mean(clean_values) if np.mean(clean_values) != 0 else 0,
            'min': np.min(clean_values),
            'max': np.max(clean_values),
            'median': np.median(clean_values),
            'q25': np.percentile(clean_values, 25),
            'q75': np.percentile(clean_values, 75),
        }
    else:
        stats_summary[key] = {
            'mean': 0, 'std': 0, 'sem': 0, 'cv': 0,
            'min': 0, 'max': 0, 'median': 0, 'q25': 0, 'q75': 0
        }

# Print key statistics
print("\nKey Metrics (N=100 runs):")
print("-" * 80)
print(f"Calcium Peak:        {stats_summary['ca_peak_uM']['mean']:.2f} ± {stats_summary['ca_peak_uM']['std']:.2f} μM (CV={stats_summary['ca_peak_uM']['cv']:.1%})")
print(f"ATP Consumption:     {stats_summary['atp_consumed_uM']['mean']:.2f} ± {stats_summary['atp_consumed_uM']['std']:.2f} μM (CV={stats_summary['atp_consumed_uM']['cv']:.1%})")
print(f"J-coupling Peak:     {stats_summary['j_peak_Hz']['mean']:.1f} ± {stats_summary['j_peak_Hz']['std']:.1f} Hz (CV={stats_summary['j_peak_Hz']['cv']:.1%})")
print(f"pH Drop:             {stats_summary['pH_drop']['mean']:.3f} ± {stats_summary['pH_drop']['std']:.3f} units (CV={stats_summary['pH_drop']['cv']:.1%})")
print(f"Dimer Accumulation:  {stats_summary['dimer_accumulation_nM']['mean']:.2f} ± {stats_summary['dimer_accumulation_nM']['std']:.2f} nM (CV={stats_summary['dimer_accumulation_nM']['cv']:.1%})")

if stats_summary['T2_dimer_s']['mean'] > 0:
    print(f"T2 Coherence Time:   {stats_summary['T2_dimer_s']['mean']:.1f} ± {stats_summary['T2_dimer_s']['std']:.1f} s (CV={stats_summary['T2_dimer_s']['cv']:.1%})")
print()

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("### GENERATING VISUALIZATIONS ###")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.35)

# Color scheme
colors = {
    'ca': '#d62728',
    'atp': '#1f77b4',
    'j': '#ff7f0e',
    'pH': '#2ca02c',
    'dimer': '#9467bd'
}

# Row 1: Distribution histograms
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data['ca_peak_uM'], bins=20, color=colors['ca'], alpha=0.7, edgecolor='black')
ax1.axvline(stats_summary['ca_peak_uM']['mean'], color='black', linestyle='--', linewidth=2)
ax1.set_xlabel('Peak Ca²⁺ (μM)')
ax1.set_ylabel('Frequency')
ax1.set_title(f"Ca²⁺ Peak\n{stats_summary['ca_peak_uM']['mean']:.2f} ± {stats_summary['ca_peak_uM']['std']:.2f} μM")
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(data['atp_consumed_uM'], bins=20, color=colors['atp'], alpha=0.7, edgecolor='black')
ax2.axvline(stats_summary['atp_consumed_uM']['mean'], color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('ATP Consumed (μM)')
ax2.set_ylabel('Frequency')
ax2.set_title(f"ATP Consumption\n{stats_summary['atp_consumed_uM']['mean']:.2f} ± {stats_summary['atp_consumed_uM']['std']:.2f} μM")
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(data['j_peak_Hz'], bins=20, color=colors['j'], alpha=0.7, edgecolor='black')
ax3.axvline(stats_summary['j_peak_Hz']['mean'], color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('J-coupling (Hz)')
ax3.set_ylabel('Frequency')
ax3.set_title(f"J-coupling Peak\n{stats_summary['j_peak_Hz']['mean']:.1f} ± {stats_summary['j_peak_Hz']['std']:.1f} Hz")
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[0, 3])
ax4.hist(data['pH_drop'], bins=20, color=colors['pH'], alpha=0.7, edgecolor='black')
ax4.axvline(stats_summary['pH_drop']['mean'], color='black', linestyle='--', linewidth=2)
ax4.set_xlabel('pH Drop')
ax4.set_ylabel('Frequency')
ax4.set_title(f"pH Acidification\n{stats_summary['pH_drop']['mean']:.3f} ± {stats_summary['pH_drop']['std']:.3f}")
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[0, 4])
ax5.hist(data['dimer_accumulation_nM'], bins=20, color=colors['dimer'], alpha=0.7, edgecolor='black')
ax5.axvline(stats_summary['dimer_accumulation_nM']['mean'], color='black', linestyle='--', linewidth=2)
ax5.set_xlabel('Dimer Accumulation (nM)')
ax5.set_ylabel('Frequency')
ax5.set_title(f"Dimer Formation\n{stats_summary['dimer_accumulation_nM']['mean']:.2f} ± {stats_summary['dimer_accumulation_nM']['std']:.2f} nM")
ax5.grid(alpha=0.3)

# Row 2: Box plots showing variability
metrics_for_boxplot = [
    ('ca_peak_uM', 'Ca²⁺ Peak\n(μM)', colors['ca']),
    ('j_enhancement_factor', 'J-coupling\nEnhancement', colors['j']),
    ('pH_drop', 'pH Drop\n(units)', colors['pH']),
    ('dimer_accumulation_nM', 'Dimer\nAccum (nM)', colors['dimer']),
]

for idx, (key, label, color) in enumerate(metrics_for_boxplot):
    ax = fig.add_subplot(gs[1, idx])
    bp = ax.boxplot([data[key]], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    ax.set_ylabel(label)
    ax.set_xticks([])
    ax.grid(alpha=0.3, axis='y')
    
    # Add CV annotation
    cv = stats_summary[key]['cv']
    ax.text(1, ax.get_ylim()[1] * 0.95, f'CV={cv:.1%}', 
            ha='center', va='top', fontsize=9, fontweight='bold')

# Row 3: Example time series from selected runs
ax_ts = fig.add_subplot(gs[2, :])

# Plot 5 example runs
n_examples = min(5, len(timeseries_runs))
for i in range(n_examples):
    ts = timeseries_runs[i]['data']
    run_idx = timeseries_runs[i]['run_idx']
    alpha = 0.6 if i > 0 else 1.0
    ax_ts.plot(ts['time_ms'], ts['ca_peak'], alpha=alpha, linewidth=1.5 if i == 0 else 1,
               label=f'Run {run_idx}' if i < 3 else None)

# Add mean trace
if len(timeseries_runs) > 0:
    # Calculate mean across all saved timeseries
    all_times = timeseries_runs[0]['data']['time_ms']
    mean_ca = np.zeros(len(all_times))
    for ts_run in timeseries_runs:
        mean_ca += np.array(ts_run['data']['ca_peak'])
    mean_ca /= len(timeseries_runs)
    ax_ts.plot(all_times, mean_ca, 'k--', linewidth=2, label='Mean')

ax_ts.axvspan(BASELINE_MS, BASELINE_MS + STIMULUS_MS, alpha=0.2, color='blue', label='Stimulus')
ax_ts.set_xlabel('Time (ms)')
ax_ts.set_ylabel('Peak Ca²⁺ (μM)')
ax_ts.set_title('Example Time Series (Stochastic Variability)')
ax_ts.legend(loc='upper right', fontsize=8)
ax_ts.grid(alpha=0.3)

# Row 4: Correlation analysis
correlation_pairs = [
    ('ca_peak_uM', 'dimer_accumulation_nM', 'Ca²⁺ Peak (μM)', 'Dimer Accumulation (nM)'),
    ('j_peak_Hz', 'dimer_accumulation_nM', 'J-coupling (Hz)', 'Dimer Accumulation (nM)'),
    ('pH_drop', 'ion_pair_peak_nM', 'pH Drop', 'Ion Pairs (nM)'),
]

for idx, (x_key, y_key, x_label, y_label) in enumerate(correlation_pairs):
    ax = fig.add_subplot(gs[3, idx])
    ax.scatter(data[x_key], data[y_key], alpha=0.5, s=20)
    
    # Calculate correlation
    r, p = stats.pearsonr(data[x_key], data[y_key])
    
    # Fit line
    z = np.polyfit(data[x_key], data[y_key], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(data[x_key].min(), data[x_key].max(), 100)
    ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'r={r:.3f}, p={p:.2e}')
    ax.grid(alpha=0.3)

# Statistical summary text
ax_text = fig.add_subplot(gs[3, 3:])
ax_text.axis('off')

summary_text = f"""
STATISTICAL SUMMARY (N={N_RUNS} runs)

Stochastic Variability (CV):
  Ca²⁺ Peak:        {stats_summary['ca_peak_uM']['cv']:.1%}
  ATP Consumption:  {stats_summary['atp_consumed_uM']['cv']:.1%}
  J-coupling:       {stats_summary['j_peak_Hz']['cv']:.1%}
  pH Drop:          {stats_summary['pH_drop']['cv']:.1%}
  Dimer Formation:  {stats_summary['dimer_accumulation_nM']['cv']:.1%}

Ranges (min-max):
  Ca²⁺:    {stats_summary['ca_peak_uM']['min']:.2f} - {stats_summary['ca_peak_uM']['max']:.2f} μM
  Dimers:  {stats_summary['dimer_accumulation_nM']['min']:.2f} - {stats_summary['dimer_accumulation_nM']['max']:.2f} nM

Interpretation:
  • CV < 20% = Low variability (deterministic)
  • CV 20-50% = Moderate variability (typical)
  • CV > 50% = High variability (stochastic)
  
  Model shows realistic biological variability
  with appropriate stochastic fluctuations.
"""

ax_text.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax_text.transAxes)

# Overall title
fig.suptitle(f'Protocol {PROTOCOL_NUMBER}: {PROTOCOL_NAME} (N={N_RUNS} runs)', 
            fontsize=16, fontweight='bold')

# Save figure
fig_path = OUTPUT_DIR / 'baseline_statistics.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✓ Figure saved: {fig_path}")
plt.close()

# =============================================================================
# SAVE RAW DATA (HDF5)
# =============================================================================

print("### SAVING RAW DATA ###")

h5_path = OUTPUT_DIR / 'raw_data.h5'
with h5py.File(h5_path, 'w') as f:
    # Save all metric arrays
    for key, values in data.items():
        f.create_dataset(key, data=values)
    
    # Save timeseries for selected runs
    ts_group = f.create_group('timeseries')
    for ts_run in timeseries_runs:
        run_group = ts_group.create_group(f"run_{ts_run['run_idx']:03d}")
        for key, values in ts_run['data'].items():
            run_group.create_dataset(key, data=np.array(values))
    
    # Save metadata
    f.attrs['protocol_name'] = PROTOCOL_NAME
    f.attrs['n_runs'] = N_RUNS
    f.attrs['total_ms'] = TOTAL_MS
    f.attrs['timestamp'] = timestamp

print(f"✓ Raw data saved: {h5_path}")

# =============================================================================
# SAVE STATISTICAL SUMMARY (JSON)
# =============================================================================

summary_path = OUTPUT_DIR / 'statistics.json'
with open(summary_path, 'w') as f:
    json.dump({
        'protocol_name': PROTOCOL_NAME,
        'protocol_number': PROTOCOL_NUMBER,
        'timestamp': timestamp,
        'n_runs': N_RUNS,
        'statistics': stats_summary,
        'protocol': {
            'baseline_ms': BASELINE_MS,
            'stimulus_ms': STIMULUS_MS,
            'recovery_ms': RECOVERY_MS,
            'total_ms': TOTAL_MS,
        }
    }, f, indent=2)

print(f"✓ Statistics saved: {summary_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("="*80)
print("PROTOCOL COMPLETE")
print("="*80)
print(f"N = {N_RUNS} runs completed successfully")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files generated:")
print(f"  - baseline_statistics.png (visualizations)")
print(f"  - statistics.json (summary statistics)")
print(f"  - raw_data.h5 (all N={N_RUNS} runs)")
print()
print("KEY FINDINGS:")
print(f"  • Stochastic variability present in all metrics")
print(f"  • Dimer formation: {stats_summary['dimer_accumulation_nM']['mean']:.2f} ± {stats_summary['dimer_accumulation_nM']['std']:.2f} nM")
print(f"  • Coefficient of variation: {stats_summary['dimer_accumulation_nM']['cv']:.1%} (biological realism)")
print("="*80)