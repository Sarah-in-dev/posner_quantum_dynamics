"""
Experimental Paradigms
=======================

Complete set of experiments for characterizing the quantum-classical cascade.

Experiments:
1. Network Size Threshold - Critical N for collective ignition
2. MT Invasion - Tryptophan network effects  
3. UV Wavelength - Optimal superradiance activation
4. Anesthetic Dose-Response - Selective Q1 blockade
5. Stimulation Intensity - Input-output curves
6. Dopamine Timing - Eligibility decay (T2 validation)
7. Temperature - Coherence and kinetic effects
8. Pharmacological Dissection - Separate Q1, Q2, Classical
9. Isotope Comparison - Definitive quantum test
10. Spatial Clustering - Geometry effects on cooperativity
11. Consolidation Kinetics - Classical cascade dynamics
12. Three-Factor Gate - Requirement for all three signals
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import time

from core import (
    ExperimentConfig, ExperimentCondition, ExperimentResult,
    compute_summary_statistics, fit_exponential_decay, fit_hill_equation,
    compute_cooperativity, print_section_header, print_subsection,
    create_progress_bar, save_results
)

from runner import run_protocol, run_trials_parallel


# =============================================================================
# EXPERIMENT 1: NETWORK SIZE THRESHOLD
# =============================================================================

def experiment_network_threshold(config: ExperimentConfig, 
                                  verbose: bool = True) -> ExperimentResult:
    """
    How does network size affect cascade ignition?
    
    Tests the hypothesis that ~50 dimers (from ~10 synapses) are needed
    for collective quantum state formation.
    
    Predictions:
    - Sharp threshold around N=10 synapses
    - Superlinear scaling (cooperativity > 1)
    - Below threshold: no commitment
    - Above threshold: full commitment
    """
    
    if verbose:
        print_section_header("EXPERIMENT 1: NETWORK SIZE THRESHOLD")
        print("Question: What's the critical N for cascade ignition?")
        print("Prediction: Sharp transition at N≈10 (50 dimer threshold)")
    
    # Define conditions
    n_values = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30]
    if config.quick_mode:
        n_values = [1, 5, 10, 15, 20]
    
    conditions = []
    for n in n_values:
        conditions.append(ExperimentCondition(
            name=f"N={n}",
            description=f"Network of {n} synapses",
            n_synapses=n,
            mt_invaded=True,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    # Run trials
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    # Aggregate results
    result = ExperimentResult(
        experiment_name="network_threshold",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    # Compute statistics for each N
    summary = {}
    for n in n_values:
        n_trials = [t for t in trials if t.condition.n_synapses == n]
        
        strengths = [t.peak_values.get('final_strength', 1.0) for t in n_trials]
        dimers = [t.peak_values.get('peak_dimers', 0) for t in n_trials]
        fields = [t.peak_values.get('peak_em_field', 0) for t in n_trials]
        committed = [t.peak_values.get('committed_level', 0) for t in n_trials]
        
        summary[n] = {
            'strength': compute_summary_statistics(strengths),
            'dimers': compute_summary_statistics(dimers),
            'field': compute_summary_statistics(fields),
            'commitment': compute_summary_statistics(committed),
        }
        
        if verbose:
            s = summary[n]
            print(f"  N={n:2d}: Strength={s['strength']['mean']:.2f}±{s['strength']['sem']:.2f}, "
                  f"Dimers={s['dimers']['mean']:.1f}, Committed={s['commitment']['mean']:.2f}")
    
    result.summary_stats = summary
    
    # Fit Hill equation to find threshold
    x = np.array(n_values)
    y = np.array([summary[n]['strength']['mean'] for n in n_values])
    
    hill_fit = fit_hill_equation(x, y)
    result.fitted_params['hill_fit'] = hill_fit
    
    # Compute cooperativity
    y_dimers = np.array([summary[n]['dimers']['mean'] for n in n_values])
    coop = compute_cooperativity(x, y_dimers)
    result.fitted_params['cooperativity'] = coop
    
    if verbose:
        print(f"\n  Hill fit: EC50 = {hill_fit['EC50']:.1f} synapses, n = {hill_fit['hill_n']:.2f}")
        print(f"  Cooperativity coefficient: {coop:.2f}")
        if coop > 1:
            print(f"    → SUPERLINEAR: Collective effects present!")
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 2: MT INVASION
# =============================================================================

def experiment_mt_invasion(config: ExperimentConfig,
                           verbose: bool = True) -> ExperimentResult:
    """
    Effect of microtubule invasion on tryptophan network
    
    MT+ spine: ~1200 tryptophans (organized)
    MT- spine: ~200 tryptophans (random)
    
    Predictions:
    - √6 ≈ 2.4× enhancement in collective field
    - Stronger forward coupling to dimers
    """
    
    if verbose:
        print_section_header("EXPERIMENT 2: MICROTUBULE INVASION")
        print("Comparing MT+ (1200 Trp) vs MT- (200 Trp)")
    
    conditions = [
        ExperimentCondition(
            name="MT-",
            description="No MT invasion, disordered tryptophans",
            n_synapses=10,
            mt_invaded=False,
            consolidation_duration_s=config.consolidation_duration
        ),
        ExperimentCondition(
            name="MT+",
            description="MT invasion, organized tryptophan lattice",
            n_synapses=10,
            mt_invaded=True,
            consolidation_duration_s=config.consolidation_duration
        ),
    ]
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="mt_invasion",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    # Aggregate
    summary = {}
    for label in ["MT-", "MT+"]:
        label_trials = [t for t in trials if t.condition.name == label]
        
        summary[label] = {
            'n_trp': compute_summary_statistics([t.peak_values.get('peak_n_trp', 0) for t in label_trials]),
            'field': compute_summary_statistics([t.peak_values.get('peak_em_field', 0) for t in label_trials]),
            'k_enhancement': compute_summary_statistics([t.peak_values.get('peak_k_enhancement', 1) for t in label_trials]),
            'peak_dimers': compute_summary_statistics([t.peak_values.get('peak_dimers', 0) for t in label_trials]),
            'peak_eligibility': compute_summary_statistics([t.peak_values.get('peak_eligibility', 0) for t in label_trials]),
            'committed': compute_summary_statistics([1 if t.peak_values.get('committed', False) else 0 for t in label_trials]),
            'committed_level': compute_summary_statistics([t.peak_values.get('committed_level', 0) for t in label_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in label_trials]),
            'peak_coherence': compute_summary_statistics([t.peak_values.get('peak_coherence', 0) for t in label_trials]),
        }
        
        if verbose:
            s = summary[label]
            print(f"  {label}: N_trp={s['n_trp']['mean']:.0f}, Field={s['field']['mean']:.1f} kT, "
                  f"Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    
    # Compute enhancement ratio
    mt_minus_field = summary["MT-"]['field']['mean']
    mt_plus_field = summary["MT+"]['field']['mean']
    enhancement = mt_plus_field / max(mt_minus_field, 0.1)
    result.fitted_params['field_enhancement'] = enhancement
    
    if verbose:
        print(f"\n  Field enhancement: {enhancement:.2f}×")
        print(f"  Theory predicts: √6 ≈ 2.45×")
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 3: UV WAVELENGTH
# =============================================================================

def experiment_uv_wavelength(config: ExperimentConfig,
                              verbose: bool = True) -> ExperimentResult:
    """
    Optimal UV wavelength for tryptophan superradiance
    
    Tryptophan absorption peak: 280 nm
    """
    
    if verbose:
        print_section_header("EXPERIMENT 3: UV WAVELENGTH OPTIMIZATION")
    
    wavelengths = [
        (None, 0, "Metabolic"),
        (220, 1.0, "220nm"),
        (250, 1.0, "250nm"),
        (280, 1.0, "280nm (peak)"),
        (310, 1.0, "310nm"),
        (340, 1.0, "340nm"),
    ]
    if config.quick_mode:
        wavelengths = [(None, 0, "Metabolic"), (280, 1.0, "280nm")]
    
    conditions = []
    for wl, intensity, label in wavelengths:
        conditions.append(ExperimentCondition(
            name=label,
            n_synapses=10,
            mt_invaded=True,
            uv_wavelength_nm=wl,
            uv_intensity_mW=intensity,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="uv_wavelength",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for cond in conditions:
        label = cond.name
        label_trials = [t for t in trials if t.condition.name == label]
        
        summary[label] = {
            'field': compute_summary_statistics([t.peak_values.get('peak_em_field', 0) for t in label_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in label_trials]),
        }
        
        if verbose:
            s = summary[label]
            print(f"  {label:15s}: Field={s['field']['mean']:.1f} kT, Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 4: ANESTHETIC DOSE-RESPONSE
# =============================================================================

def experiment_anesthetic(config: ExperimentConfig,
                          verbose: bool = True) -> ExperimentResult:
    """
    Isoflurane dose-response curve
    
    Anesthetics disrupt tryptophan dipole coupling
    Should show dose-dependent elimination of quantum contribution
    """
    
    if verbose:
        print_section_header("EXPERIMENT 4: ANESTHETIC DOSE-RESPONSE")
        print("Isoflurane selectively blocks Q1 (tryptophan coupling)")
    
    concentrations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if config.quick_mode:
        concentrations = [0.0, 0.3, 0.6, 0.9]
    
    conditions = []
    for conc in concentrations:
        conditions.append(ExperimentCondition(
            name=f"{int(conc*100)}%",
            description=f"Isoflurane {conc*100:.0f}% block",
            n_synapses=10,
            mt_invaded=True,
            anesthetic_concentration=conc,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="anesthetic",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for conc in concentrations:
        label = f"{int(conc*100)}%"
        label_trials = [t for t in trials if t.condition.name == label]
        
        summary[conc] = {
            'field': compute_summary_statistics([t.peak_values.get('peak_em_field', 0) for t in label_trials]),
            'dimers': compute_summary_statistics([t.peak_values.get('peak_dimers', 0) for t in label_trials]),
            'committed': compute_summary_statistics([t.peak_values.get('committed_level', 0) for t in label_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in label_trials]),
        }
        
        if verbose:
            s = summary[conc]
            print(f"  {int(conc*100):3d}% block: Field={s['field']['mean']:.1f} kT, "
                  f"Dimers={s['dimers']['mean']:.1f}, Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    
    # Fit IC50
    x = np.array(concentrations)
    y = np.array([summary[c]['strength']['mean'] for c in concentrations])
    
    # Inverse Hill for inhibition
    hill_fit = fit_hill_equation(1 - x + 0.01, y)  # Invert x for IC50
    result.fitted_params['ic50_fit'] = hill_fit
    
    if verbose and hill_fit['EC50'] > 0:
        print(f"\n  IC50 estimate: {(1 - hill_fit['EC50'])*100:.0f}%")
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 5: STIMULATION INTENSITY
# =============================================================================

def experiment_stim_intensity(config: ExperimentConfig,
                               verbose: bool = True) -> ExperimentResult:
    """
    Input-output relationship
    
    Voltage → Calcium → Dimers → Strength
    """
    
    if verbose:
        print_section_header("EXPERIMENT 5: STIMULATION INTENSITY")
    
    voltages_mV = [-70, -60, -50, -40, -30, -20, -10, 0, 10]
    if config.quick_mode:
        voltages_mV = [-60, -30, 0]
    
    conditions = []
    for v in voltages_mV:
        conditions.append(ExperimentCondition(
            name=f"{v}mV",
            n_synapses=10,
            mt_invaded=True,
            stim_voltage_mV=v,
            stim_duration_s=0.05,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="stim_intensity",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for v in voltages_mV:
        label = f"{v}mV"
        v_trials = [t for t in trials if t.condition.name == label]
        
        summary[v] = {
            'dimers': compute_summary_statistics([t.peak_values.get('peak_dimers', 0) for t in v_trials]),
            'eligibility': compute_summary_statistics([t.peak_values.get('peak_eligibility', 0) for t in v_trials]),
            'committed': compute_summary_statistics([t.peak_values.get('committed_level', 0) for t in v_trials]),
            'field': compute_summary_statistics([t.peak_values.get('peak_em_field', 0) for t in v_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in v_trials]),
        }
        
        if verbose:
            s = summary[v]
            print(f"  {v:4d} mV: Dimers={s['dimers']['mean']:.1f}, "
                  f"Committed={s['committed']['mean']:.2f}, "
                  f"Eligibility={s['eligibility']['mean']:.2f}")
            
    result.summary_stats = summary
    
    # Fit sigmoid
    x = np.array(voltages_mV)
    y = np.array([summary[v]['strength']['mean'] for v in voltages_mV])
    hill_fit = fit_hill_equation(x + 80, y)  # Shift x to positive
    result.fitted_params['io_curve'] = hill_fit
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 6: DOPAMINE TIMING
# =============================================================================

def experiment_dopamine_timing(config: ExperimentConfig,
                                verbose: bool = True) -> ExperimentResult:
    """
    Eligibility decay with dopamine delay
    
    This is THE key test of T2 coherence time
    P31: T2 ≈ 67s → eligibility persists
    P32: T2 ≈ 0.3s → eligibility decays rapidly
    """
    
    if verbose:
        print_section_header("EXPERIMENT 6: DOPAMINE TIMING")
        print("Testing eligibility trace decay (T2 = 67s prediction)")
    
    delays = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    if config.quick_mode:
        delays = [0, 15, 30, 60]
    
    conditions = []
    for delay in delays:
        conditions.append(ExperimentCondition(
            name=f"Delay={delay}s",
            n_synapses=10,
            mt_invaded=True,
            dopamine_delay_s=delay,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="dopamine_timing",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for delay in delays:
        label = f"Delay={delay}s"
        d_trials = [t for t in trials if t.condition.name == label]
        
        # Get eligibility at moment of dopamine
        pre_da_elig = []
        for t in d_trials:
            if t.pre_dopamine:
                pre_da_elig.append(t.pre_dopamine.eligibility)
            else:
                pre_da_elig.append(0)
        
        summary[delay] = {
            'pre_da_eligibility': compute_summary_statistics(pre_da_elig),
            'committed': compute_summary_statistics([t.peak_values.get('committed_level', 0) for t in d_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in d_trials]),
        }
        
        if verbose:
            s = summary[delay]
            print(f"  Delay={delay:3d}s: Eligibility={s['pre_da_eligibility']['mean']:.3f}, "
                  f"Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    
    # Fit exponential decay to extract T2
    x = np.array(delays)
    y = np.array([summary[d]['pre_da_eligibility']['mean'] for d in delays])
    
    decay_fit = fit_exponential_decay(x, y)
    result.fitted_params['decay_fit'] = decay_fit
    
    if verbose and decay_fit['tau'] > 0:
        print(f"\n  Fitted T2: {decay_fit['tau']:.1f} s")
        print(f"  Theory: 67 s (P31 nuclear spin)")
        print(f"  R² = {decay_fit['r_squared']:.3f}")
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 7: TEMPERATURE
# =============================================================================

def experiment_temperature(config: ExperimentConfig,
                           verbose: bool = True) -> ExperimentResult:
    """
    Temperature effects on coherence and kinetics
    """
    
    if verbose:
        print_section_header("EXPERIMENT 7: TEMPERATURE DEPENDENCE")
    
    temperatures_C = [25, 30, 33, 35, 37, 39, 41, 43]
    if config.quick_mode:
        temperatures_C = [30, 37, 42]
    
    conditions = []
    for T in temperatures_C:
        conditions.append(ExperimentCondition(
            name=f"{T}°C",
            n_synapses=10,
            mt_invaded=True,
            temperature_C=T,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="temperature",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for T in temperatures_C:
        label = f"{T}°C"
        t_trials = [t for t in trials if t.condition.name == label]
        
        summary[T] = {
            'coherence': compute_summary_statistics([t.peak_values.get('peak_coherence', 0) for t in t_trials]),
            'dimers': compute_summary_statistics([t.peak_values.get('peak_dimers', 0) for t in t_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in t_trials]),
        }
        
        if verbose:
            s = summary[T]
            print(f"  {T}°C: Coherence={s['coherence']['mean']:.3f}, Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 8: PHARMACOLOGICAL DISSECTION
# =============================================================================

def experiment_pharmacology(config: ExperimentConfig,
                            verbose: bool = True) -> ExperimentResult:
    """
    Dissect cascade using selective blockers
    
    - Control: Full cascade
    - APV: Block NMDA → No calcium → No Q2
    - Nocodazole: Destroy MT → No organized Trp → Weak Q1
    - Isoflurane: Block Trp coupling → Weak Q1
    """
    
    if verbose:
        print_section_header("EXPERIMENT 8: PHARMACOLOGICAL DISSECTION")
        print("Separating Q1, Q2, and Classical contributions")
    
    conditions = [
        ExperimentCondition(
            name="Control",
            description="Full cascade",
            n_synapses=10,
            mt_invaded=True,
            consolidation_duration_s=config.consolidation_duration
        ),
        ExperimentCondition(
            name="APV",
            description="NMDA blocker - no calcium",
            n_synapses=10,
            mt_invaded=True,
            apv_applied=True,
            consolidation_duration_s=config.consolidation_duration
        ),
        ExperimentCondition(
            name="Nocodazole",
            description="MT disruptor - disorganized Trp",
            n_synapses=10,
            mt_invaded=True,
            nocodazole_applied=True,
            consolidation_duration_s=config.consolidation_duration
        ),
        ExperimentCondition(
            name="Isoflurane",
            description="90% Trp coupling block",
            n_synapses=10,
            mt_invaded=True,
            anesthetic_concentration=0.9,
            consolidation_duration_s=config.consolidation_duration
        ),
    ]
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="pharmacology",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for cond in conditions:
        c_trials = [t for t in trials if t.condition.name == cond.name]
        
        summary[cond.name] = {
            'field': compute_summary_statistics([t.peak_values.get('peak_em_field', 0) for t in c_trials]),
            'dimers': compute_summary_statistics([t.peak_values.get('peak_dimers', 0) for t in c_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in c_trials]),
            'committed': compute_summary_statistics([1 if t.peak_values.get('committed', False) else 0 for t in c_trials]),
        }
        
        if verbose:
            s = summary[cond.name]
            print(f"  {cond.name:12s}: Q1={s['field']['mean']:.1f} kT, Q2={s['dimers']['mean']:.1f}, "
                  f"Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 9: ISOTOPE COMPARISON
# =============================================================================

def experiment_isotope(config: ExperimentConfig,
                       verbose: bool = True) -> ExperimentResult:
    """
    P31 vs P32 - THE definitive test
    
    P31: I=1/2, T2 ≈ 67s
    P32: I=1, T2 ≈ 0.3s (due to quadrupolar relaxation)
    
    Prediction: P32 should show no benefit from delay > 1s
    """
    
    if verbose:
        print_section_header("EXPERIMENT 9: ISOTOPE COMPARISON")
        print("P31 (T2≈67s) vs P32 (T2≈0.3s)")
    
    # P31 with various delays
    conditions = []
    p31_delays = [0, 15, 30, 60]
    for delay in p31_delays:
        conditions.append(ExperimentCondition(
            name=f"P31-{delay}s",
            n_synapses=10,
            mt_invaded=True,
            isotope="P31",
            dopamine_delay_s=delay,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    # P32 - only short delays make sense
    p32_delays = [0, 0.5, 1.0, 2.0]
    for delay in p32_delays:
        conditions.append(ExperimentCondition(
            name=f"P32-{delay}s",
            n_synapses=10,
            mt_invaded=True,
            isotope="P32",
            dopamine_delay_s=delay,
            consolidation_duration_s=config.consolidation_duration
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="isotope",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {'P31': {}, 'P32': {}}
    
    for cond in conditions:
        c_trials = [t for t in trials if t.condition.name == cond.name]
        
        isotope = cond.isotope
        delay = cond.dopamine_delay_s
        
        pre_da_elig = []
        for t in c_trials:
            if t.pre_dopamine:
                pre_da_elig.append(t.pre_dopamine.eligibility)
            else:
                pre_da_elig.append(0)
        
        summary[isotope][delay] = {
            'pre_da_eligibility': compute_summary_statistics(pre_da_elig),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in c_trials]),
        }
        
        if verbose:
            s = summary[isotope][delay]
            print(f"  {isotope} delay={delay:4.1f}s: Elig={s['pre_da_eligibility']['mean']:.3f}, "
                  f"Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    
    # Fit T2 for each isotope
    for iso, delays_list in [('P31', p31_delays), ('P32', p32_delays)]:
        x = np.array(delays_list)
        y = np.array([summary[iso][d]['pre_da_eligibility']['mean'] for d in delays_list])
        
        decay_fit = fit_exponential_decay(x, y)
        result.fitted_params[f'{iso}_decay'] = decay_fit
        
        if verbose:
            print(f"\n  {iso} fitted T2: {decay_fit['tau']:.1f} s (R²={decay_fit['r_squared']:.3f})")
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 10: SPATIAL CLUSTERING
# =============================================================================

def experiment_spatial_clustering(config: ExperimentConfig,
                                   verbose: bool = True) -> ExperimentResult:
    """
    Effect of synapse spatial arrangement on cooperativity
    
    Clustered: All within ~2 µm (high coupling)
    Distributed: Spread across ~20 µm (low coupling)
    """
    
    if verbose:
        print_section_header("EXPERIMENT 10: SPATIAL CLUSTERING")
        print("Does synapse geometry affect collective quantum effects?")
    
    patterns = ["clustered", "linear", "distributed"]
    n_values = [5, 10, 15, 20]
    if config.quick_mode:
        patterns = ["clustered", "distributed"]
        n_values = [5, 15]
    
    conditions = []
    for pattern in patterns:
        for n in n_values:
            conditions.append(ExperimentCondition(
                name=f"{pattern}-N{n}",
                n_synapses=n,
                spatial_pattern=pattern,
                mt_invaded=True,
                consolidation_duration_s=config.consolidation_duration
            ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="spatial_clustering",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {p: {} for p in patterns}
    
    for cond in conditions:
        c_trials = [t for t in trials if t.condition.name == cond.name]
        
        pattern = cond.spatial_pattern
        n = cond.n_synapses
        
        summary[pattern][n] = {
            'dimers': compute_summary_statistics([t.peak_values.get('peak_dimers', 0) for t in c_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in c_trials]),
        }
        
        if verbose:
            s = summary[pattern][n]
            print(f"  {pattern:12s} N={n:2d}: Dimers={s['dimers']['mean']:.1f}, "
                  f"Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    
    # Compute cooperativity for each pattern
    for pattern in patterns:
        x = np.array([n for n in n_values if n in summary[pattern]])
        y = np.array([summary[pattern][n]['dimers']['mean'] for n in x])
        
        if len(x) > 1:
            coop = compute_cooperativity(x, y)
            result.fitted_params[f'{pattern}_cooperativity'] = coop
            
            if verbose:
                print(f"\n  {pattern} cooperativity: {coop:.2f}")
    
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 11: CONSOLIDATION KINETICS
# =============================================================================

def experiment_consolidation(config: ExperimentConfig,
                              verbose: bool = True) -> ExperimentResult:
    """
    How does the classical cascade unfold over time?
    
    Tracks CaMKII → Actin → AMPARs → Strength
    """
    
    if verbose:
        print_section_header("EXPERIMENT 11: CONSOLIDATION KINETICS")
        print("Tracking classical cascade dynamics")
    
    # Single condition, many timepoints
    timepoints = [1, 5, 10, 30, 60, 120, 300, 600]  # seconds
    if config.quick_mode:
        timepoints = [1, 10, 60]
    
    conditions = []
    for t in timepoints:
        conditions.append(ExperimentCondition(
            name=f"t={t}s",
            n_synapses=10,
            mt_invaded=True,
            consolidation_duration_s=t
        ))
    
    start = time.time()
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="consolidation",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for t in timepoints:
        label = f"t={t}s"
        t_trials = [tr for tr in trials if tr.condition.name == label]
        
        summary[t] = {
            'pT286': compute_summary_statistics([tr.final.camkii_pT286 if tr.final else 0 for tr in t_trials]),
            'mol_memory': compute_summary_statistics([tr.final.molecular_memory if tr.final else 0 for tr in t_trials]),
            'spine_volume': compute_summary_statistics([tr.final.spine_volume if tr.final else 1 for tr in t_trials]),
            'AMPAR': compute_summary_statistics([tr.final.AMPAR_count if tr.final else 80 for tr in t_trials]),
            'strength': compute_summary_statistics([tr.final.synaptic_strength if tr.final else 1 for tr in t_trials]),
        }
        
        if verbose:
            s = summary[t]
            print(f"  t={t:3d}s: pT286={s['pT286']['mean']:.3f}, Spine={s['spine_volume']['mean']:.2f}×, "
                  f"Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# EXPERIMENT 12: THREE-FACTOR GATE
# =============================================================================

def experiment_three_factor_gate(config: ExperimentConfig,
                                  verbose: bool = True) -> ExperimentResult:
    """
    Test that all three factors are required:
    1. Eligibility (dimer coherence)
    2. Dopamine (reward signal)
    3. Calcium (activity)
    
    Remove each systematically
    """
    
    if verbose:
        print_section_header("EXPERIMENT 12: THREE-FACTOR GATE")
        print("Testing requirement for eligibility + dopamine + calcium")
    
    conditions = [
        # Full: all three
        ExperimentCondition(
            name="Full (E+D+Ca)",
            n_synapses=10,
            mt_invaded=True,
            consolidation_duration_s=config.consolidation_duration
        ),
        # No eligibility: use P32 with long delay
        ExperimentCondition(
            name="No Elig (P32+delay)",
            n_synapses=10,
            mt_invaded=True,
            isotope="P32",
            dopamine_delay_s=5.0,  # Long for P32
            consolidation_duration_s=config.consolidation_duration
        ),
        # No dopamine: omit reward signal
        ExperimentCondition(
            name="No DA",
            description="Eligibility but no dopamine",
            n_synapses=10,
            mt_invaded=True,
            dopamine_delay_s=120,  
            consolidation_duration_s=config.consolidation_duration
        ),
        # No calcium: block NMDA
        ExperimentCondition(
            name="No Ca (APV)",
            n_synapses=10,
            mt_invaded=True,
            apv_applied=True,
            consolidation_duration_s=config.consolidation_duration
        ),
    ]
    
    start = time.time()
    
    # Special handling for "No DA" - need custom protocol
    # For now run with standard protocol
    trials = run_trials_parallel(conditions, config)
    
    result = ExperimentResult(
        experiment_name="three_factor_gate",
        conditions=conditions,
        trials=trials,
        timestamp=datetime.now().isoformat(),
        config=config
    )
    
    summary = {}
    for cond in conditions:
        c_trials = [t for t in trials if t.condition.name == cond.name]
        
        summary[cond.name] = {
            'committed': compute_summary_statistics([1 if t.peak_values.get('committed', False) else 0 for t in c_trials]),
            'strength': compute_summary_statistics([t.peak_values.get('final_strength', 1) for t in c_trials]),
        }
        
        if verbose:
            s = summary[cond.name]
            commit_pct = s['committed']['mean'] * 100
            print(f"  {cond.name:20s}: Committed={commit_pct:.0f}%, Strength={s['strength']['mean']:.2f}×")
    
    result.summary_stats = summary
    result.runtime_seconds = time.time() - start
    
    return result


# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================

def run_all_experiments(config: ExperimentConfig,
                        verbose: bool = True) -> Dict[str, ExperimentResult]:
    """
    Run complete experiment suite
    
    Returns dict of experiment_name → ExperimentResult
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("QUANTUM CASCADE COMPLETE EXPERIMENTAL SUITE".center(70))
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Workers: {config.n_workers}")
        print(f"  Trials per condition: {config.n_trials}")
        print(f"  Quick mode: {config.quick_mode}")
        print(f"  Output directory: {config.output_dir}")
    
    experiments = [
        ("1. Network Threshold", experiment_network_threshold),
        ("2. MT Invasion", experiment_mt_invasion),
        ("3. UV Wavelength", experiment_uv_wavelength),
        ("4. Anesthetic", experiment_anesthetic),
        ("5. Stim Intensity", experiment_stim_intensity),
        ("6. Dopamine Timing", experiment_dopamine_timing),
        ("7. Temperature", experiment_temperature),
        ("8. Pharmacology", experiment_pharmacology),
        ("9. Isotope", experiment_isotope),
        ("10. Spatial Clustering", experiment_spatial_clustering),
        ("11. Consolidation", experiment_consolidation),
        ("12. Three-Factor Gate", experiment_three_factor_gate),
    ]
    
    results = {}
    total_start = time.time()
    
    for name, func in experiments:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running: {name}")
        
        try:
            result = func(config, verbose=verbose)
            results[result.experiment_name] = result
            
            # Save intermediate results
            save_results(result, config.output_dir)
            
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT SUITE COMPLETE".center(70))
        print("=" * 70)
        print(f"\nTotal runtime: {total_time/60:.1f} minutes")
        print(f"Results saved to: {config.output_dir}")
    
    return results