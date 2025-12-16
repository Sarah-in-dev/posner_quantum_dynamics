"""
Model Runner - Interface between experiments and Model 6
=========================================================

Handles:
- Model configuration based on experimental conditions
- Protocol execution with state recording
- Single trial running (for parallel execution)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import time

# Import from core
from core import (
    ExperimentConfig, ExperimentCondition, SystemState, TrialResult,
    generate_synapse_positions, compute_distance_matrix, compute_network_coupling_factor
)

MODEL6_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL6_PATH))

# Import Model 6
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

from multi_synapse_network import MultiSynapseNetwork

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def configure_model(condition: ExperimentCondition):
    """
    Configure Model 6 based on experimental condition
    
    Returns configured model ready for simulation
    """
    
    
    params = Model6Parameters()
    
    # === Core Settings ===
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.simulation.dt_diffusion = 1e-4
    
    # === Network Architecture ===
    params.multi_synapse.n_synapses_default = condition.n_synapses
    
    # === Isotope ===
    if condition.isotope == "P31":
        params.environment.fraction_P31 = 1.0
    elif condition.isotope == "P32":
        params.environment.fraction_P31 = 0.0
    else:
        params.environment.fraction_P31 = 1.0
    
    # === Temperature ===
    params.environment.T = condition.temperature_C + 273.15  # Convert to Kelvin
    
    # === UV Illumination ===
    if condition.uv_wavelength_nm is not None and condition.uv_intensity_mW > 0:
        params.metabolic_uv.external_uv_illumination = True
        params.metabolic_uv.external_uv_wavelength = condition.uv_wavelength_nm * 1e-9
        params.metabolic_uv.external_uv_intensity = condition.uv_intensity_mW * 1e-3
    
    # === Anesthetic ===
    if condition.anesthetic_concentration > 0:
        params.tryptophan.anesthetic_applied = True
        params.tryptophan.anesthetic_blocking_factor = condition.anesthetic_concentration

    # === APV (NMDA blocker) ===
    if condition.apv_applied:
        params.calcium.nmda_blocked = True
    
    # === Create Model ===
    model = Model6QuantumSynapse(params=params)
    
    # === Post-creation Configuration ===
    
    # Microtubule invasion
    if condition.nocodazole_applied:
        # Nocodazole destroys MT network
        model.set_microtubule_invasion(False)
        model.n_tryptophans = 50  # Severely reduced
    else:
        model.set_microtubule_invasion(condition.mt_invaded)
    
    # === SINGLE SYNAPSE ===
    if condition.n_synapses == 1:
        return model
    
    # === MULTI-SYNAPSE NETWORK ===
    # Create network with N independent Model6 instances
    network = MultiSynapseNetwork(
        n_synapses=condition.n_synapses,
        spacing_um=condition.synapse_spacing_um,
        pattern=condition.spatial_pattern,
        field_threshold_kT=20.0
    )
    
    # Initialize with the same params we configured
    network.initialize(Model6QuantumSynapse, params)
    
    # Apply condition settings to all synapses
    if condition.nocodazole_applied:
        for synapse in network.synapses:
            synapse.set_microtubule_invasion(False)
            synapse.n_tryptophans = 50
    else:
        network.set_microtubule_invasion(condition.mt_invaded)
    
    # Mark that this is a network (for run_protocol to detect)
    network._is_network = True
    
    return network


def measure_state(model, time: float, phase: str, network=None) -> SystemState:
    """
    Capture complete system state from model
    
    Extracts all relevant variables for analysis
    """
    
    state = SystemState()
    state.time = time
    state.phase = phase
    
    # === QUANTUM SYSTEM 1: Tryptophan ===
    state.n_tryptophans = getattr(model, 'n_tryptophans', 0)
    state.em_field_trp = getattr(model, '_em_field_trp', 0.0)
    state.collective_field_kT = getattr(model, '_collective_field_kT', 0.0)
    state.superradiance_active = state.collective_field_kT > 5.0
    
    
    # === QUANTUM SYSTEM 2: Dimers ===
    # Get dimer concentration from proper source
    metrics = model.get_experimental_metrics()
    dimer_conc_nM = metrics.get('dimer_peak_nM_ct', 0.0)
    
    # Convert concentration to molecule count per synapse
    # Spine volume ≈ 0.1 µm³ = 1e-16 L
    # n = C × V × N_A where C in M, V in L
    spine_volume_L = 1e-16
    N_A = 6.022e23
    dimers_per_synapse = dimer_conc_nM * 1e-9 * spine_volume_L * N_A
    
    # Network parameters
    if hasattr(model.params, 'multi_synapse') and hasattr(model.params.multi_synapse, 'n_synapses_default'):
        n_synapses = model.params.multi_synapse.n_synapses_default
    else:
        n_synapses = 1
    coupling_factor = getattr(model, '_spatial_coupling_factor', 1.0)
    
    # Total coherent dimers in network
    # Coupling factor accounts for spatial clustering benefits
    total_network_dimers = dimers_per_synapse * n_synapses * coupling_factor
    
    # Store all metrics
    state.dimer_conc_nM = dimer_conc_nM
    state.dimers_per_synapse = dimers_per_synapse
    state.total_network_dimers = total_network_dimers
    state.dimer_count = total_network_dimers  # Legacy compatibility
    
    state.dimer_coherence = getattr(model, '_previous_coherence', 0.0)
    # Use the proper eligibility getter (matches test_eligibility_decay.py)
    if hasattr(model, 'eligibility'):
        state.eligibility = model.eligibility.get_eligibility()
    else:
        state.eligibility = 0.0
    state.network_modulation = getattr(model, '_network_modulation', 0.0)

    # Get peak calcium for diagnostics
    if hasattr(model, 'get_experimental_metrics'):
        metrics = model.get_experimental_metrics()
        state.peak_calcium_uM = metrics.get('calcium_peak_uM', 0.0)
        
    # === COUPLING ===
    k_agg = getattr(model, '_k_agg_for_next_step', None)
    state.k_agg_enhanced = k_agg if k_agg is not None else 0.0
    state.k_enhancement = getattr(model, '_k_enhancement', 1.0)
    state.forward_coupling_active = state.k_enhancement > 1.05
    state.reverse_coupling_active = state.network_modulation > 0.1
    
    # === PLASTICITY GATE ===
    state.plasticity_gate = getattr(model, '_plasticity_gate', False)
    state.committed = getattr(model, '_camkii_committed', False)
    state.committed_level = getattr(model, '_committed_memory_level', 0.0)
    
    # Gate components (if tracked)
    state.gate_eligibility = state.eligibility > 0.3
    state.gate_calcium = True  # Would need calcium tracking
    state.gate_dopamine = getattr(model, '_dopamine_above_read_threshold', lambda: False)()
    
    # === CLASSICAL BRIDGE: CaMKII ===
    if hasattr(model, 'camkii'):
        state.camkii_pT286 = getattr(model.camkii, 'pT286', 0.0)
        state.camkii_active = getattr(model.camkii, 'active_fraction', 0.0)
        state.molecular_memory = getattr(model.camkii, 'molecular_memory', 0.0)
    
    # === CLASSICAL OUTPUT: Spine Plasticity ===
    if hasattr(model, 'spine_plasticity'):
        state.spine_volume = model.spine_plasticity.spine_volume
        state.AMPAR_count = model.spine_plasticity.AMPAR_count
        state.synaptic_strength = model.spine_plasticity.get_synaptic_strength()
        state.plasticity_phase = model.spine_plasticity.phase
    
    # === FEEDBACK ===
    if hasattr(model, 'ca_phosphate') and hasattr(model.ca_phosphate, 'templates'):
        state.n_templates = int(np.sum(model.ca_phosphate.templates.template_field))
    state.template_feedback_active = state.n_templates > 3
    
    # === NETWORK OVERRIDES ===
    if network is not None:
        total_dimers = 0.0
        total_eligibility = 0.0
        weighted_coherence = 0.0
        
        for synapse in network.synapses:
            # Get CURRENT dimer concentration (not peak)
            dimer_nM = getattr(synapse, '_previous_dimer_count', 0.0)
            # Convert nM to molecule count: dimers = nM × 0.006
            dimers_this_synapse = dimer_nM * 0.006
            total_dimers += dimers_this_synapse
            
            # Get coherence
            coherence = getattr(synapse, '_previous_coherence', 0.0)
            weighted_coherence += coherence * dimers_this_synapse
            
            # Get eligibility from proper module
            if hasattr(synapse, 'eligibility'):
                total_eligibility += synapse.eligibility.get_eligibility()
        
        state.dimer_count = total_dimers
        state.total_network_dimers = total_dimers
        state.eligibility = total_eligibility / len(network.synapses)
        
        # Weighted coherence
        if total_dimers > 0:
            state.dimer_coherence = weighted_coherence / total_dimers
        else:
            state.dimer_coherence = 0.0
        
        # USE NETWORK COMMITMENT (not individual synapse)
        state.committed = network.network_committed
        state.committed_level = network.network_commitment_level
        
        # Synaptic strength scales with commitment
        if state.committed:
            state.synaptic_strength = 1.0 + 0.5 * state.committed_level
        else:
            state.synaptic_strength = 1.0
    
    return state


# =============================================================================
# PROTOCOL EXECUTION
# =============================================================================

def run_protocol(condition: ExperimentCondition, 
                 config: ExperimentConfig,
                 trial_id: int = 0,
                 verbose: bool = False) -> TrialResult:
    """
    Execute complete experimental protocol
    
    Protocol phases:
    1. Baseline (500ms)
    2. Stimulation (variable)
    3. Delay (variable)
    4. Dopamine READ (300ms)
    5. Consolidation (variable)
    
    Returns TrialResult with complete state traces
    """
    
    start_time = time.time()
    
    # Configure model (may return single model or network)
    model_or_network = configure_model(condition)
    dt = config.dt
    
    # Detect if this is a network
    is_network = getattr(model_or_network, '_is_network', False)
    
    if is_network:
        network = model_or_network
        model = network.synapses[0]  # Reference synapse for single-model properties
    else:
        network = None
        model = model_or_network
    
    # Initialize result
    result = TrialResult(condition=condition, trial_id=trial_id)
    
    t = 0.0
    step = 0
    
    def record(phase: str):
        """Record current state"""
        state = measure_state(model, t, phase)
        result.traces.append(state)
        return state
    
    # =========================================================================
    # PHASE 1: BASELINE
    # =========================================================================
    n_steps = int(config.baseline_duration / dt)
    
    if verbose:
        print(f"  Phase 1: Baseline ({config.baseline_duration}s)")
    
    for i in range(n_steps):
        if is_network:
            network.step(dt, {'voltage': -70e-3, 'reward': False})
        else:
            model.step(dt, {'voltage': -70e-3, 'reward': False})
        
        if step % config.record_interval == 0:
            record("baseline")
        
        t += dt
        step += 1
    
    result.baseline = measure_state(model, t, "baseline", network=network)
    
    # =========================================================================
    # PHASE 2: STIMULATION
    # =========================================================================
    stim_duration = condition.stim_duration_s
    n_steps = int(stim_duration / dt)
    
    if verbose:
        print(f"  Phase 2: Stimulation ({stim_duration}s)")
    
    for i in range(n_steps):
        # APV blocks NMDA - no calcium influx
        if condition.apv_applied:
            voltage = -70e-3
        else:
            voltage = condition.stim_voltage_mV * 1e-3
        
        if is_network:
            network.step(dt, {'voltage': voltage, 'reward': False})
        else:
            model.step(dt, {'voltage': voltage, 'reward': False})
        
        if step % config.record_interval == 0:
            record("stimulation")
        
        t += dt
        step += 1
    
    result.post_stim = measure_state(model, t, "post_stim", network=network)
    
    if verbose:
        print(f"    Dimers: {result.post_stim.dimer_count:.1f}")
        print(f"    Eligibility: {result.post_stim.eligibility:.3f}")
    
    # =========================================================================
    # =========================================================================
    # PHASE 3: DELAY (if any)
    # =========================================================================
    if condition.dopamine_delay_s > 0:
        delay_duration = condition.dopamine_delay_s
        
        if verbose:
            print(f"  Phase 3: Delay ({delay_duration}s)")
        
        # Use coarse timesteps during delay (10ms instead of 1ms)
        dt_coarse = dt * 10  # 10ms steps
        n_steps_coarse = int(delay_duration / dt_coarse)

        for i in range(n_steps_coarse):
            if is_network:
                network.step(dt_coarse, {'voltage': -70e-3, 'reward': False})
            else:
                model.step(dt_coarse, {'voltage': -70e-3, 'reward': False})
            
            # Record less frequently during long delays
            interval = config.record_interval * 10
            if step % interval == 0:
                record("delay")
            
            t += dt_coarse  # Also fix: should be dt_coarse, not dt
            step += 1
    
    # =========================================================================
    # PHASE 4: DOPAMINE READ
    # =========================================================================
    da_duration = condition.dopamine_duration_s
    n_steps = int(da_duration / dt)
    
    if verbose:
        print(f"  Phase 4: Dopamine READ ({da_duration}s)")
    
    for i in range(n_steps):
        if is_network:
            network.step(dt, {'voltage': -70e-3, 'reward': True})
        else:
            model.step(dt, {'voltage': -70e-3, 'reward': True})
        
        if step % config.record_interval == 0:
            record("dopamine")
        
        t += dt
        step += 1
    
    result.post_dopamine = measure_state(model, t, "post_dopamine", network=network)
    
    if verbose:
        print(f"    Committed: {result.post_dopamine.committed}")
        print(f"    Committed level: {result.post_dopamine.committed_level:.3f}")
    
    # =========================================================================
    # PHASE 5: CONSOLIDATION
    # =========================================================================
    consol_duration = condition.consolidation_duration_s
    n_steps = int(consol_duration / dt)
    
    if verbose:
        print(f"  Phase 5: Consolidation ({consol_duration}s)")
    
    # Use coarse timesteps during consolidation (10ms instead of 1ms)
    dt_coarse = dt * 10  # 10ms steps
    n_steps_coarse = int(consol_duration / dt_coarse)

    for i in range(n_steps_coarse):
        if is_network:
            network.step(dt_coarse, {'voltage': -70e-3, 'reward': False})
        else:
            model.step(dt, {'voltage': -70e-3, 'reward': False})
        
        # Record infrequently during consolidation
        if step % (config.record_interval * 100) == 0:
            record("consolidation")
        
        t += dt_coarse  # Not dt
        step += 1
    
    result.final = measure_state(model, t, "final", network=network)
    
    if verbose:
        print(f"\n  === FINAL STATE ===")
        print(f"  Q1 (Trp field): {result.final.collective_field_kT:.1f} kT")
        print(f"  Q2 (Dimers): {result.final.dimer_count:.1f}")
        print(f"  Classical (Strength): {result.final.synaptic_strength:.2f}×")
    
    # =========================================================================
    # COMPUTE PEAK VALUES
    # =========================================================================
    result.peak_values = {
        # Quantum System 1
        'peak_em_field': max((s.collective_field_kT for s in result.traces), default=0),
        'peak_n_trp': max((s.n_tryptophans for s in result.traces), default=0),
        
        # Quantum System 2
        'peak_dimers': max((s.dimer_count for s in result.traces), default=0),
        'peak_coherence': max((s.dimer_coherence for s in result.traces), default=0),
        'peak_eligibility': max((s.eligibility for s in result.traces), default=0),
        
        # Coupling
        'peak_k_enhancement': max((s.k_enhancement for s in result.traces), default=1),
        'peak_network_mod': max((s.network_modulation for s in result.traces), default=0),
        
        # Classical Bridge
        'peak_pT286': max((s.camkii_pT286 for s in result.traces), default=0),
        'peak_mol_memory': max((s.molecular_memory for s in result.traces), default=0),
        
        # Classical Output
        'final_spine_volume': result.final.spine_volume if result.final else 1.0,
        'final_AMPAR': result.final.AMPAR_count if result.final else 80,
        'final_strength': result.final.synaptic_strength if result.final else 1.0,
        'committed': result.final.committed if result.final else False,
        'committed_level': result.final.committed_level if result.final else 0,
    }
    
    result.runtime_seconds = time.time() - start_time
    
    return result


# =============================================================================
# PARALLEL EXECUTION WRAPPER
# =============================================================================

def run_single_trial(args: Tuple) -> TrialResult:
    """
    Wrapper for parallel execution
    
    Args is tuple of (condition, config, trial_id)
    """
    condition, config, trial_id = args
    
    try:
        result = run_protocol(condition, config, trial_id, verbose=False)
        return result
    except Exception as e:
        print(f"Error in trial {trial_id}: {e}")
        # Return empty result on error
        return TrialResult(condition=condition, trial_id=trial_id)


def run_trials_parallel(conditions: List[ExperimentCondition],
                        config: ExperimentConfig,
                        n_trials: int = None) -> List[TrialResult]:
    """
    Run multiple trials in parallel
    
    Returns list of TrialResults
    """
    
    if n_trials is None:
        n_trials = config.n_trials
    
    # Build task list
    tasks = []
    for condition in conditions:
        for trial_id in range(n_trials):
            tasks.append((condition, config, trial_id))
    
    results = []
    
    if config.use_multiprocessing and config.n_workers > 1:
        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = [executor.submit(run_single_trial, task) for task in tasks]
            
            for future in futures:
                try:
                    result = future.result(timeout=600)  # 10 min timeout per trial
                    results.append(result)
                except Exception as e:
                    print(f"Trial failed: {e}")
    else:
        # Sequential execution
        for task in tasks:
            result = run_single_trial(task)
            results.append(result)
    
    return results