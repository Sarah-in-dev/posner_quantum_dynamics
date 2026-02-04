#!/usr/bin/env python3
"""
Coordinated Decoherence Experiment
==================================

Demonstrates that entanglement-based coordination produces correlated
commitment patterns using the full Model 6 physics.

HYPOTHESIS:
-----------
When eligibility is near threshold at reward time:
- COORDINATED: Entangled synapses pass/fail TOGETHER (correlated measurement)
- INDEPENDENT: Synapses pass/fail based on their individual states

PHYSICS:
--------
- P31 isotope: T₂ = 216s (the real physics)
- Eligibility decays as: P_S(t) = 0.25 + 0.75 × exp(-t/216)
- Gate threshold: P_S > 0.5 (from Agarwal 2023)
- Time to reach P_S = 0.5: ~238s

PROTOCOL:
---------
1. Stimulate all synapses (theta-burst, 1s) → dimers form, entanglement builds
2. Delay (configurable, default 150s) → eligibility decays toward threshold  
3. Apply reward → triggers three-factor gate evaluation
4. Record commitment pattern

EFFICIENCY:
-----------
- 1ms timesteps during stimulation (calcium dynamics matter)
- 1s timesteps during delay (only decoherence matters)
- 100ms timesteps during reward

Author: Sarah Davidson
University of Florida
Date: February 2026
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter
import time
import logging
import json
from datetime import datetime

# Setup logging - reduce noise but keep important info
logging.basicConfig(level=logging.WARNING)
for logger_name in ['model6_core', 'multi_synapse_network', 'dimer_particles', 
                    'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
                    'quantum_coherence', 'pH_dynamics', 'dopamine_system',
                    'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
                    'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
                    'photon_receiver_module']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Find Model 6 path
def find_model6_path():
    """Search for Model 6 in common locations"""
    possible_paths = [
        Path(__file__).parent.parent,  # From Full_System_Experiments -> Model_6
        Path(__file__).parent,
        Path.cwd().parent,  # If cwd is Full_System_Experiments
        Path.cwd(),
        Path(__file__).parent / "src" / "models" / "Model_6",
        Path(__file__).parent.parent / "src" / "models" / "Model_6",
    ]
    
    for path in possible_paths:
        if (path / "model6_core.py").exists():
            return path
    
    raise FileNotFoundError("Could not find Model 6. Please run from the Model_6 directory.")

MODEL6_PATH = find_model6_path()
sys.path.insert(0, str(MODEL6_PATH))
print(f"Using Model 6 from: {MODEL6_PATH}")

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single trial"""
    trial_id: int
    condition: str  # 'coordinated' or 'independent'
    
    # Pre-reward state
    n_dimers: int
    eligibilities: List[float]
    mean_eligibility: float
    correlation_matrix: np.ndarray
    mean_correlation: float
    
    # Outcome
    committed: List[bool]
    n_committed: int
    
    @property
    def outcome_type(self) -> str:
        """Classify as 'all', 'none', or 'partial'"""
        if self.n_committed == 0:
            return "none"
        elif self.n_committed == len(self.committed):
            return "all"
        else:
            return "partial"


@dataclass 
class ExperimentResult:
    """Complete experiment results"""
    n_synapses: int
    n_trials_per_condition: int
    delay_s: float
    stim_duration_s: float
    
    # Results by condition
    coordinated_trials: List[TrialResult] = field(default_factory=list)
    independent_trials: List[TrialResult] = field(default_factory=list)
    
    # Timing
    start_time: str = ""
    runtime_s: float = 0.0
    
    def get_outcome_counts(self, condition: str) -> Dict[str, int]:
        """Count outcomes for a condition"""
        trials = self.coordinated_trials if condition == 'coordinated' else self.independent_trials
        counts = Counter(t.outcome_type for t in trials)
        return {"all": counts.get("all", 0), 
                "none": counts.get("none", 0), 
                "partial": counts.get("partial", 0)}
    
    def get_commitment_distribution(self, condition: str) -> Dict[int, int]:
        """Distribution of number committed"""
        trials = self.coordinated_trials if condition == 'coordinated' else self.independent_trials
        return dict(Counter(t.n_committed for t in trials))


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_trial(
    trial_id: int,
    condition: str,
    n_synapses: int,
    stim_duration_s: float,
    delay_s: float,
    reward_duration_s: float = 0.5,
    verbose: bool = False
) -> TrialResult:
    """
    Run a single trial of the coordination experiment.
    
    Parameters
    ----------
    trial_id : int
        Trial number for logging
    condition : str
        'coordinated' or 'independent'
    n_synapses : int
        Number of synapses in network
    stim_duration_s : float
        Duration of theta-burst stimulation
    delay_s : float
        Delay between stimulation and reward
    reward_duration_s : float
        Duration of reward signal
    verbose : bool
        Print detailed progress
    """
    use_correlated = (condition == 'coordinated')
    
    # === SETUP ===
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = 1.0  # Real P31 physics, T₂ = 216s
    
    network = MultiSynapseNetwork(
        n_synapses=n_synapses,
        params=params,
        pattern='clustered',
        spacing_um=2.0,
        use_correlated_sampling=use_correlated
    )
    network.initialize(Model6QuantumSynapse, params)
    network.set_microtubule_invasion(True)
    
    # Enable coordination mode (disables auto-commitment)
    network.set_coordination_mode(use_correlated)
    # Ensure auto-commitment is disabled for both conditions
    network.disable_auto_commitment = True
    network.use_correlated_sampling = use_correlated
    
    if verbose:
        print(f"  Trial {trial_id} ({condition}): initialized {n_synapses} synapses")
    
    # === PHASE 1: STIMULATION (theta-burst) ===
    # Use 1ms timesteps during spikes (calcium dynamics matter)
    # Use 10ms timesteps during inter-burst intervals (faster)
    dt_spike = 0.001
    dt_rest = 0.01
    
    # Theta burst: 5 bursts, 4 spikes each, 200ms between bursts
    for burst in range(5):
        for spike in range(4):
            # 2ms depolarization (2 steps at 1ms)
            for _ in range(2):
                network.step(dt_spike, {'voltage': -10e-3, 'reward': False})
            # 8ms rest within burst (8 steps at 1ms)
            for _ in range(8):
                network.step(dt_spike, {'voltage': -70e-3, 'reward': False})
        # Inter-burst interval: ~160ms (16 steps at 10ms)
        for _ in range(16):
            network.step(dt_rest, {'voltage': -70e-3, 'reward': False})
    
    if verbose:
        dimer_counts = [len(s.dimer_particles.dimers) for s in network.synapses]
        print(f"    After stim: dimers per synapse = {dimer_counts}")
    
    # === PHASE 2: DELAY ===
    # During delay, only singlet probability decay matters
    # Skip full step() and directly decay P_S for efficiency
    # Physics: P_S(t) = 0.25 + (P_S(0) - 0.25) * exp(-t/T2)
    # T2 = 216s for P31
    
    T2 = 216.0  # seconds for P31
    P_thermal = 0.25
    
    if verbose:
        # Get initial singlet probs
        initial_ps = []
        for syn in network.synapses:
            for d in syn.dimer_particles.dimers:
                initial_ps.append(d.singlet_probability)
        if initial_ps:
            print(f"    Initial mean P_S: {np.mean(initial_ps):.3f}")
    
    # Decay singlet probability for all dimers directly
    decay_factor = np.exp(-delay_s / T2)
    
    for syn in network.synapses:
        for dimer in syn.dimer_particles.dimers:
            P_excess = dimer.singlet_probability - P_thermal
            dimer.singlet_probability = P_thermal + P_excess * decay_factor
    
    # Update network time
    network.time += delay_s
    
    if verbose:
        final_ps = []
        for syn in network.synapses:
            for d in syn.dimer_particles.dimers:
                final_ps.append(d.singlet_probability)
        if final_ps:
            print(f"    After {delay_s}s delay: mean P_S = {np.mean(final_ps):.3f}")
    
    # === MEASURE PRE-REWARD STATE ===
    # Do one real step to sync internal state after manual decay
    network.step(0.1, {'voltage': -70e-3, 'reward': False})
    
    eligibilities = [s.get_eligibility() for s in network.synapses]
    mean_eligibility = np.mean(eligibilities)
    
    # Count dimers
    n_dimers = sum(len(s.dimer_particles.dimers) for s in network.synapses)
    
    # Get correlation matrix
    network.entanglement_tracker.collect_dimers(network.synapses, network.positions)
    C = network.entanglement_tracker.get_synapse_correlation_matrix(network.synapses)
    
    # Mean off-diagonal correlation
    if n_synapses > 1:
        off_diag = C[np.triu_indices(n_synapses, k=1)]
        mean_correlation = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
    else:
        mean_correlation = 0.0
    
    if verbose:
        print(f"    Before reward: elig={[f'{e:.3f}' for e in eligibilities]}")
        print(f"    Mean eligibility: {mean_eligibility:.3f}, correlation: {mean_correlation:.3f}")
    
    # === PHASE 3: REWARD ===
    # Single reward step for clean commitment decision
    # Multiple steps would give multiple chances to commit, washing out correlation effects
    network.step(0.1, {'voltage': -70e-3, 'reward': True})
    
    # === MEASURE OUTCOME ===
    committed = [getattr(s, '_camkii_committed', False) for s in network.synapses]
    n_committed = sum(committed)
    
    if verbose:
        print(f"    After reward: committed = {committed}")
    
    return TrialResult(
        trial_id=trial_id,
        condition=condition,
        n_dimers=n_dimers,
        eligibilities=eligibilities,
        mean_eligibility=mean_eligibility,
        correlation_matrix=C.copy(),
        mean_correlation=mean_correlation,
        committed=committed,
        n_committed=n_committed
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(
    n_trials: int = 20,
    n_synapses: int = 4,
    stim_duration_s: float = 1.0,
    delay_s: float = 150.0,
    verbose: bool = True,
    save_results: bool = True,
    output_dir: Optional[Path] = None
) -> ExperimentResult:
    """
    Run the complete coordinated decoherence experiment.
    
    Parameters
    ----------
    n_trials : int
        Number of trials per condition
    n_synapses : int
        Number of synapses in network
    stim_duration_s : float
        Stimulation duration in seconds
    delay_s : float
        Delay before reward in seconds
    verbose : bool
        Print progress
    save_results : bool
        Save results to JSON
    output_dir : Path, optional
        Directory for output files
    """
    start_time = time.time()
    
    print("=" * 70)
    print("COORDINATED DECOHERENCE EXPERIMENT")
    print("=" * 70)
    print(f"Testing whether entanglement produces correlated commitment patterns")
    print()
    print("Parameters:")
    print(f"  N synapses: {n_synapses}")
    print(f"  N trials per condition: {n_trials}")
    print(f"  Stimulation: {stim_duration_s}s theta-burst")
    print(f"  Delay: {delay_s}s (for eligibility decay)")
    print(f"  Isotope: P31 (T₂ = 216s)")
    print()
    print("Physics prediction:")
    p_s_at_delay = 0.25 + 0.75 * np.exp(-delay_s / 216)
    print(f"  Expected P_S after {delay_s}s delay: {p_s_at_delay:.3f}")
    print(f"  Gate threshold: eligibility > 0.33")
    print(f"  (Delay guide: 150s→0.62, 300s→0.44, 400s→0.37, 450s→0.34)")
    if p_s_at_delay > 0.5:
        print(f"  → Eligibility well ABOVE threshold (all synapses pass)")
    elif p_s_at_delay > 0.33:
        print(f"  → Eligibility NEAR threshold (differentiation possible)")
    else:
        print(f"  → Eligibility BELOW threshold (most synapses fail)")
    print()
    
    result = ExperimentResult(
        n_synapses=n_synapses,
        n_trials_per_condition=n_trials,
        delay_s=delay_s,
        stim_duration_s=stim_duration_s,
        start_time=datetime.now().isoformat()
    )
    
    # === RUN COORDINATED TRIALS ===
    print("-" * 70)
    print("COORDINATED CONDITION (use_correlated_sampling=True)")
    print("-" * 70)
    
    for trial in range(n_trials):
        if verbose:
            print(f"\nTrial {trial + 1}/{n_trials}:")
        else:
            print(f"  Trial {trial + 1}/{n_trials}...", flush=True)
        
        tr = run_trial(
            trial_id=trial,
            condition='coordinated',
            n_synapses=n_synapses,
            stim_duration_s=stim_duration_s,
            delay_s=delay_s,
            verbose=verbose
        )
        result.coordinated_trials.append(tr)
        
        if not verbose:
            print(f"    → {tr.outcome_type} ({tr.n_committed}/{n_synapses} committed)", flush=True)
    
    coord_outcomes = result.get_outcome_counts('coordinated')
    print(f"\nCoordinated results:")
    print(f"  All commit:  {coord_outcomes['all']:3d} ({100*coord_outcomes['all']/n_trials:5.1f}%)")
    print(f"  None commit: {coord_outcomes['none']:3d} ({100*coord_outcomes['none']/n_trials:5.1f}%)")
    print(f"  Partial:     {coord_outcomes['partial']:3d} ({100*coord_outcomes['partial']/n_trials:5.1f}%)")
    
    # === RUN INDEPENDENT TRIALS ===
    print()
    print("-" * 70)
    print("INDEPENDENT CONDITION (use_correlated_sampling=False)")
    print("-" * 70)
    
    for trial in range(n_trials):
        if verbose:
            print(f"\nTrial {trial + 1}/{n_trials}:")
        else:
            print(f"  Trial {trial + 1}/{n_trials}...", flush=True)
        
        tr = run_trial(
            trial_id=trial,
            condition='independent',
            n_synapses=n_synapses,
            stim_duration_s=stim_duration_s,
            delay_s=delay_s,
            verbose=verbose
        )
        result.independent_trials.append(tr)
        
        if not verbose:
            print(f"    → {tr.outcome_type} ({tr.n_committed}/{n_synapses} committed)", flush=True)
    
    indep_outcomes = result.get_outcome_counts('independent')
    print(f"\nIndependent results:")
    print(f"  All commit:  {indep_outcomes['all']:3d} ({100*indep_outcomes['all']/n_trials:5.1f}%)")
    print(f"  None commit: {indep_outcomes['none']:3d} ({100*indep_outcomes['none']/n_trials:5.1f}%)")
    print(f"  Partial:     {indep_outcomes['partial']:3d} ({100*indep_outcomes['partial']/n_trials:5.1f}%)")
    
    result.runtime_s = time.time() - start_time
    
    # === SUMMARY ===
    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\nCommitment Pattern Distribution:")
    print(f"{'':20} {'ALL':>10} {'NONE':>10} {'PARTIAL':>10}")
    print(f"{'Coordinated':20} {coord_outcomes['all']:>10} {coord_outcomes['none']:>10} {coord_outcomes['partial']:>10}")
    print(f"{'Independent':20} {indep_outcomes['all']:>10} {indep_outcomes['none']:>10} {indep_outcomes['partial']:>10}")
    
    # Key comparison: extreme (all-or-none) vs partial
    coord_extreme = coord_outcomes['all'] + coord_outcomes['none']
    indep_extreme = indep_outcomes['all'] + indep_outcomes['none']
    coord_partial = coord_outcomes['partial']
    indep_partial = indep_outcomes['partial']
    
    print(f"\nAll-or-None (extreme) outcomes:")
    print(f"  Coordinated: {coord_extreme}/{n_trials} ({100*coord_extreme/n_trials:.1f}%)")
    print(f"  Independent: {indep_extreme}/{n_trials} ({100*indep_extreme/n_trials:.1f}%)")
    
    print(f"\nPartial outcomes:")
    print(f"  Coordinated: {coord_partial}/{n_trials} ({100*coord_partial/n_trials:.1f}%)")
    print(f"  Independent: {indep_partial}/{n_trials} ({100*indep_partial/n_trials:.1f}%)")
    
    # Hypothesis test
    print(f"\n" + "-" * 70)
    print("HYPOTHESIS TEST")
    print("-" * 70)
    
    if coord_extreme > indep_extreme and coord_partial < indep_partial:
        print("✓ CONFIRMED: Coordinated shows MORE all-or-none, FEWER partial")
        print("  This supports entanglement-based coordination")
    elif coord_extreme == indep_extreme and coord_partial == indep_partial:
        print("? INCONCLUSIVE: No difference between conditions")
        print("  May need longer delays or more trials")
    else:
        print("✗ NOT CONFIRMED: Pattern does not match prediction")
        print("  Coordinated should have more extreme, fewer partial outcomes")
    
    # Mean eligibility comparison
    coord_mean_elig = np.mean([t.mean_eligibility for t in result.coordinated_trials])
    indep_mean_elig = np.mean([t.mean_eligibility for t in result.independent_trials])
    coord_mean_corr = np.mean([t.mean_correlation for t in result.coordinated_trials])
    indep_mean_corr = np.mean([t.mean_correlation for t in result.independent_trials])
    
    print(f"\nMean eligibility at reward:")
    print(f"  Coordinated: {coord_mean_elig:.3f}")
    print(f"  Independent: {indep_mean_elig:.3f}")
    
    print(f"\nMean correlation at reward:")
    print(f"  Coordinated: {coord_mean_corr:.3f}")
    print(f"  Independent: {indep_mean_corr:.3f}")
    
    print(f"\nRuntime: {result.runtime_s:.1f}s ({result.runtime_s/60:.1f} min)")
    
    # === SAVE RESULTS ===
    if save_results:
        if output_dir is None:
            output_dir = Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'experiment': 'coordinated_decoherence',
            'timestamp': result.start_time,
            'parameters': {
                'n_synapses': n_synapses,
                'n_trials': n_trials,
                'stim_duration_s': stim_duration_s,
                'delay_s': delay_s,
                'isotope': 'P31',
                'T2_s': 216
            },
            'results': {
                'coordinated': {
                    'outcomes': coord_outcomes,
                    'mean_eligibility': coord_mean_elig,
                    'mean_correlation': coord_mean_corr,
                    'commitment_distribution': result.get_commitment_distribution('coordinated')
                },
                'independent': {
                    'outcomes': indep_outcomes,
                    'mean_eligibility': indep_mean_elig,
                    'mean_correlation': indep_mean_corr,
                    'commitment_distribution': result.get_commitment_distribution('independent')
                }
            },
            'runtime_s': result.runtime_s
        }
        
        json_path = output_dir / f'coordinated_decoherence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {json_path}")
    
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Coordinated Decoherence Experiment')
    parser.add_argument('--trials', type=int, default=10, help='Trials per condition')
    parser.add_argument('--synapses', type=int, default=4, help='Number of synapses')
    parser.add_argument('--delay', type=float, default=400.0, help='Delay in seconds')
    parser.add_argument('--stim', type=float, default=1.0, help='Stimulation duration')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    result = run_experiment(
        n_trials=args.trials,
        n_synapses=args.synapses,
        stim_duration_s=args.stim,
        delay_s=args.delay,
        verbose=args.verbose,
        save_results=True,
        output_dir=Path(args.output)
    )