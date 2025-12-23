#!/usr/bin/env python3
"""
Experiment: Feedback Loop Characterization (TIER 4)
=====================================================

Tests the bidirectional quantum-classical feedback loop.

Scientific basis:
The system contains TWO coupled feedback loops:

LOOP 1 - Fast (Forward Coupling):
  Tryptophan superradiance → EM field → Enhanced dimer formation
  Timescale: milliseconds to seconds

LOOP 2 - Slow (Reverse Coupling):
  Dimer coherent fields → Tubulin conformational modulation → Better Trp coupling
  Timescale: seconds to minutes

Together, these create:
  - Positive feedback amplification (gain > 1)
  - Bistability (low/high states)
  - Hysteresis (history dependence)
  - Self-organization over multiple activity cycles

Predictions:
1. Positive feedback: Gain > 1 when both loops active
2. Bistability: Two stable states emerge with sufficient stimulation
3. Hysteresis: Different thresholds for switching up vs down
4. Multi-cycle enhancement: Each stimulation cycle should increase subsequent response

Protocol:
1. Test feedback amplification (compare with/without coupling)
2. Test bistability (partial stimulation should show two outcomes)
3. Test hysteresis (ramp up vs ramp down stimulation)
4. Test multi-cycle self-organization (repeated stimulation)

Success criteria:
- Measurable positive feedback (>1.5× amplification)
- Evidence of bistability (bimodal outcome distribution)
- Hysteresis in state transitions
- Progressive enhancement over cycles

References:
- Our hierarchical model architecture discussions
- Bistability in biological systems (Ferrell & Bhalla, 2002)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


@dataclass
class FeedbackCondition:
    """Single experimental condition"""
    test_type: str  # 'amplification', 'bistability', 'hysteresis', 'multicycle'
    coupling_enabled: bool = True  # Enable bidirectional coupling
    n_synapses: int = 10
    
    # Test-specific parameters
    stim_intensity: float = 1.0  # Relative stimulation strength
    n_cycles: int = 1  # For multicycle test
    
    @property
    def name(self) -> str:
        coupling = "+coupling" if self.coupling_enabled else "-coupling"
        return f"{self.test_type}_{coupling}"


@dataclass 
class FeedbackTrialResult:
    """Results from single trial"""
    condition: FeedbackCondition
    trial_id: int
    
    # Amplification metrics
    dimer_formation_rate: float = 0.0  # With coupling
    baseline_formation_rate: float = 0.0  # Without coupling (if measured)
    amplification_factor: float = 1.0
    
    # State metrics
    final_state: str = "unknown"  # 'low', 'high', or 'intermediate'
    em_field_mean_kT: float = 0.0
    dimer_coherence_mean: float = 0.0
    
    # Cycle-by-cycle data (for multicycle)
    cycle_responses: List[float] = field(default_factory=list)
    cycle_em_fields: List[float] = field(default_factory=list)
    
    # Timeline
    time_points: List[float] = field(default_factory=list)
    em_field_trace: List[float] = field(default_factory=list)
    dimer_trace: List[float] = field(default_factory=list)
    
    # Output
    committed: bool = False
    final_strength: float = 1.0
    
    runtime_s: float = 0.0


@dataclass
class FeedbackResult:
    """Complete experiment results"""
    conditions: List[FeedbackCondition]
    trials: List[FeedbackTrialResult]
    
    # Summary by test type
    summary: Dict = field(default_factory=dict)
    
    # Key findings
    amplification_factor: float = 1.0  # Mean amplification with coupling
    bistability_detected: bool = False
    hysteresis_detected: bool = False
    multicycle_enhancement: float = 1.0  # Enhancement over cycles
    
    # State distribution
    low_state_fraction: float = 0.0
    high_state_fraction: float = 0.0
    
    timestamp: str = ""
    runtime_s: float = 0.0


class FeedbackLoopExperiment:
    """
    Feedback loop characterization experiment (Tier 4)
    
    Tests bidirectional quantum-classical coupling and emergent bistability.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        if quick_mode:
            self.n_trials = 3
            self.n_synapses = 5
            self.n_cycles = 3
            self.stim_intensities = [0.5, 1.0]
            self.consolidation_s = 0.5
        else:
            self.n_trials = 10  # More trials for bistability statistics
            self.n_synapses = 10
            self.n_cycles = 5
            self.stim_intensities = [0.3, 0.5, 0.7, 1.0]
            self.consolidation_s = 2.0
        
        # State thresholds
        self.high_state_threshold = 18  # kT - above this is "high state"
        self.low_state_threshold = 8  # kT - below this is "low state"
    
    def _create_network(self, coupling_enabled: bool = True) -> MultiSynapseNetwork:
        """Create network with or without bidirectional coupling"""
        params = Model6Parameters()
        params.em_coupling_enabled = coupling_enabled
        params.multi_synapse_enabled = True
        
        # Enable or disable feedback coupling
        if hasattr(params, 'feedback'):
            params.feedback.bidirectional_enabled = coupling_enabled
            params.feedback.dimer_to_trp_coupling = coupling_enabled
            params.feedback.trp_to_dimer_coupling = coupling_enabled
        
        network = MultiSynapseNetwork(
            n_synapses=self.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _run_stimulation_cycle(self, network: MultiSynapseNetwork, 
                               intensity: float = 1.0, 
                               reward: bool = True) -> Tuple[float, float, int]:
        """
        Run one stimulation cycle and return metrics
        
        Returns: (peak_em_field, dimer_formation_rate, peak_dimers)
        """
        dt = 0.001
        
        em_fields = []
        dimer_counts = []
        start_dimers = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # Scale voltage by intensity
        stim_voltage = -70e-3 + intensity * 60e-3  # -70 to -10 mV range
        
        # Theta-burst stimulation
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):
                    network.step(dt, {"voltage": stim_voltage, "reward": reward})
                for _ in range(8):
                    network.step(dt, {"voltage": -70e-3, "reward": reward})
            
            # Record metrics
            metrics = network.get_experimental_metrics()
            em_fields.append(metrics.get('mean_field_kT', 0))
            
            dimers = sum(
                len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                for s in network.synapses
            )
            dimer_counts.append(dimers)
            
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": reward})
        
        # Brief rest
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": reward})
        
        peak_field = max(em_fields) if em_fields else 0
        peak_dimers = max(dimer_counts) if dimer_counts else 0
        dimer_rate = (peak_dimers - start_dimers) / 1.0  # Per second
        
        return peak_field, dimer_rate, peak_dimers
    
    def _test_amplification(self, trial_id: int) -> FeedbackTrialResult:
        """Test feedback amplification by comparing with/without coupling"""
        result = FeedbackTrialResult(
            condition=FeedbackCondition(test_type='amplification', coupling_enabled=True),
            trial_id=trial_id
        )
        
        # Run WITH coupling
        network_coupled = self._create_network(coupling_enabled=True)
        
        # Baseline
        dt = 0.001
        for _ in range(100):
            network_coupled.step(dt, {"voltage": -70e-3, "reward": False})
        
        # Stimulate
        peak_field_coupled, rate_coupled, dimers_coupled = self._run_stimulation_cycle(
            network_coupled, intensity=1.0
        )
        
        # Run WITHOUT coupling
        network_uncoupled = self._create_network(coupling_enabled=False)
        
        for _ in range(100):
            network_uncoupled.step(dt, {"voltage": -70e-3, "reward": False})
        
        peak_field_uncoupled, rate_uncoupled, dimers_uncoupled = self._run_stimulation_cycle(
            network_uncoupled, intensity=1.0
        )
        
        # Compute amplification
        result.dimer_formation_rate = rate_coupled
        result.baseline_formation_rate = rate_uncoupled
        
        if rate_uncoupled > 0:
            result.amplification_factor = rate_coupled / rate_uncoupled
        else:
            result.amplification_factor = 1.0
        
        result.em_field_mean_kT = peak_field_coupled
        
        # Determine state
        if peak_field_coupled > self.high_state_threshold:
            result.final_state = 'high'
        elif peak_field_coupled < self.low_state_threshold:
            result.final_state = 'low'
        else:
            result.final_state = 'intermediate'
        
        return result
    
    def _test_bistability(self, intensity: float, trial_id: int) -> FeedbackTrialResult:
        """Test for bistability at intermediate stimulation"""
        result = FeedbackTrialResult(
            condition=FeedbackCondition(
                test_type='bistability', 
                coupling_enabled=True,
                stim_intensity=intensity
            ),
            trial_id=trial_id
        )
        
        network = self._create_network(coupling_enabled=True)
        
        dt = 0.001
        
        # Baseline
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # Stimulate at intermediate intensity (where bistability should emerge)
        peak_field, dimer_rate, peak_dimers = self._run_stimulation_cycle(
            network, intensity=intensity
        )
        
        result.em_field_mean_kT = peak_field
        result.dimer_formation_rate = dimer_rate
        
        # Brief consolidation
        n_consol = int(self.consolidation_s / dt)
        for _ in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # Final state measurement
        metrics = network.get_experimental_metrics()
        final_field = metrics.get('mean_field_kT', 0)
        
        if final_field > self.high_state_threshold:
            result.final_state = 'high'
        elif final_field < self.low_state_threshold:
            result.final_state = 'low'
        else:
            result.final_state = 'intermediate'
        
        result.committed = network.network_committed
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        return result
    
    def _test_multicycle(self, trial_id: int) -> FeedbackTrialResult:
        """Test enhancement over multiple stimulation cycles"""
        result = FeedbackTrialResult(
            condition=FeedbackCondition(
                test_type='multicycle',
                coupling_enabled=True,
                n_cycles=self.n_cycles
            ),
            trial_id=trial_id
        )
        
        network = self._create_network(coupling_enabled=True)
        
        dt = 0.001
        
        # Baseline
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # Run multiple cycles with rest between
        for cycle in range(self.n_cycles):
            peak_field, dimer_rate, peak_dimers = self._run_stimulation_cycle(
                network, intensity=0.7  # Moderate intensity to see enhancement
            )
            
            result.cycle_responses.append(dimer_rate)
            result.cycle_em_fields.append(peak_field)
            
            # Rest period between cycles
            rest_steps = int(2.0 / dt)  # 2 second rest
            for _ in range(rest_steps):
                network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # Compute enhancement
        if len(result.cycle_responses) >= 2 and result.cycle_responses[0] > 0:
            result.amplification_factor = result.cycle_responses[-1] / result.cycle_responses[0]
        
        result.em_field_mean_kT = np.mean(result.cycle_em_fields) if result.cycle_em_fields else 0
        
        # Final state
        metrics = network.get_experimental_metrics()
        final_field = metrics.get('mean_field_kT', 0)
        
        if final_field > self.high_state_threshold:
            result.final_state = 'high'
        elif final_field < self.low_state_threshold:
            result.final_state = 'low'
        else:
            result.final_state = 'intermediate'
        
        result.committed = network.network_committed
        
        return result
    
    def run(self) -> FeedbackResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        if self.verbose:
            print(f"\nRunning feedback loop experiment (Tier 4)...")
            print(f"  Tests: amplification, bistability, multicycle")
            print(f"  Trials per condition: {self.n_trials}")
        
        trials = []
        conditions = []
        
        # === TEST 1: AMPLIFICATION ===
        if self.verbose:
            print(f"\n  Amplification test: ", end='', flush=True)
        
        for trial_id in range(self.n_trials):
            trial = self._test_amplification(trial_id)
            trials.append(trial)
            if self.verbose:
                print(".", end='', flush=True)
        
        if self.verbose:
            amp_results = [t.amplification_factor for t in trials if t.condition.test_type == 'amplification']
            mean_amp = np.mean(amp_results)
            print(f" amplification={mean_amp:.2f}×")
        
        # === TEST 2: BISTABILITY ===
        if self.verbose:
            print(f"  Bistability test:", flush=True)
        
        for intensity in self.stim_intensities:
            if self.verbose:
                print(f"    intensity={intensity:.1f}: ", end='', flush=True)
            
            for trial_id in range(self.n_trials):
                trial = self._test_bistability(intensity, trial_id)
                trials.append(trial)
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                bi_trials = [t for t in trials 
                            if t.condition.test_type == 'bistability' 
                            and t.condition.stim_intensity == intensity]
                states = [t.final_state for t in bi_trials]
                high_frac = states.count('high') / len(states) if states else 0
                print(f" high_state={high_frac:.0%}")
        
        # === TEST 3: MULTICYCLE ===
        if self.verbose:
            print(f"  Multicycle test: ", end='', flush=True)
        
        for trial_id in range(self.n_trials):
            trial = self._test_multicycle(trial_id)
            trials.append(trial)
            if self.verbose:
                print(".", end='', flush=True)
        
        if self.verbose:
            multi_results = [t for t in trials if t.condition.test_type == 'multicycle']
            enhancements = [t.amplification_factor for t in multi_results]
            mean_enh = np.mean(enhancements)
            print(f" enhancement={mean_enh:.2f}×")
        
        # Build result
        result = FeedbackResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Analyze findings
        self._analyze_findings(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[FeedbackTrialResult]) -> Dict:
        """Compute summary statistics"""
        summary = {}
        
        # Amplification summary
        amp_trials = [t for t in trials if t.condition.test_type == 'amplification']
        if amp_trials:
            summary['amplification'] = {
                'mean_amplification': np.mean([t.amplification_factor for t in amp_trials]),
                'std_amplification': np.std([t.amplification_factor for t in amp_trials]),
                'mean_coupled_rate': np.mean([t.dimer_formation_rate for t in amp_trials]),
                'mean_uncoupled_rate': np.mean([t.baseline_formation_rate for t in amp_trials]),
                'n_trials': len(amp_trials)
            }
        
        # Bistability summary by intensity
        for intensity in self.stim_intensities:
            bi_trials = [t for t in trials 
                        if t.condition.test_type == 'bistability' 
                        and t.condition.stim_intensity == intensity]
            
            if bi_trials:
                states = [t.final_state for t in bi_trials]
                summary[f'bistability_{intensity}'] = {
                    'intensity': intensity,
                    'high_fraction': states.count('high') / len(states),
                    'low_fraction': states.count('low') / len(states),
                    'intermediate_fraction': states.count('intermediate') / len(states),
                    'mean_field': np.mean([t.em_field_mean_kT for t in bi_trials]),
                    'n_trials': len(bi_trials)
                }
        
        # Multicycle summary
        multi_trials = [t for t in trials if t.condition.test_type == 'multicycle']
        if multi_trials:
            # Extract cycle-by-cycle enhancement
            all_responses = [t.cycle_responses for t in multi_trials]
            mean_by_cycle = np.mean(all_responses, axis=0) if all_responses else []
            
            summary['multicycle'] = {
                'mean_enhancement': np.mean([t.amplification_factor for t in multi_trials]),
                'std_enhancement': np.std([t.amplification_factor for t in multi_trials]),
                'responses_by_cycle': list(mean_by_cycle),
                'n_cycles': self.n_cycles,
                'n_trials': len(multi_trials)
            }
        
        return summary
    
    def _analyze_findings(self, result: FeedbackResult):
        """Analyze key findings"""
        # Amplification
        amp_data = result.summary.get('amplification', {})
        result.amplification_factor = amp_data.get('mean_amplification', 1.0)
        
        # Bistability - look for bimodal distribution
        # At intermediate intensity, should see mix of high/low states
        for intensity in self.stim_intensities:
            key = f'bistability_{intensity}'
            if key in result.summary:
                data = result.summary[key]
                high = data.get('high_fraction', 0)
                low = data.get('low_fraction', 0)
                
                # Bistability = both states populated
                if high > 0.2 and low > 0.2:
                    result.bistability_detected = True
                    break
        
        # Multicycle enhancement
        multi_data = result.summary.get('multicycle', {})
        result.multicycle_enhancement = multi_data.get('mean_enhancement', 1.0)
        
        # State fractions (from all bistability trials)
        all_bi_trials = [t for t in result.trials if t.condition.test_type == 'bistability']
        if all_bi_trials:
            states = [t.final_state for t in all_bi_trials]
            result.high_state_fraction = states.count('high') / len(states)
            result.low_state_fraction = states.count('low') / len(states)
    
    def print_summary(self, result: FeedbackResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("FEEDBACK LOOP CHARACTERIZATION RESULTS (TIER 4)")
        print("="*70)
        
        # Amplification
        print("\n--- AMPLIFICATION TEST ---")
        amp = result.summary.get('amplification', {})
        print(f"  With coupling:    {amp.get('mean_coupled_rate', 0):.1f} dimers/s")
        print(f"  Without coupling: {amp.get('mean_uncoupled_rate', 0):.1f} dimers/s")
        print(f"  Amplification:    {result.amplification_factor:.2f}×")
        
        if result.amplification_factor > 1.5:
            print("  ✓ POSITIVE FEEDBACK CONFIRMED")
        else:
            print("  ⚠ Weak feedback (< 1.5×)")
        
        # Bistability
        print("\n--- BISTABILITY TEST ---")
        print(f"  {'Intensity':<12} {'High State':<12} {'Low State':<12} {'Intermediate':<12}")
        print("-"*50)
        
        for intensity in self.stim_intensities:
            key = f'bistability_{intensity}'
            if key in result.summary:
                data = result.summary[key]
                high = data.get('high_fraction', 0)
                low = data.get('low_fraction', 0)
                inter = data.get('intermediate_fraction', 0)
                print(f"  {intensity:<12.1f} {high:<12.0%} {low:<12.0%} {inter:<12.0%}")
        
        if result.bistability_detected:
            print("\n  ✓ BISTABILITY DETECTED")
            print("    System shows two stable states at intermediate stimulation")
        else:
            print("\n  ⚠ No clear bistability (single stable state)")
        
        # Multicycle
        print("\n--- MULTICYCLE ENHANCEMENT ---")
        multi = result.summary.get('multicycle', {})
        responses = multi.get('responses_by_cycle', [])
        
        if responses:
            print(f"  Cycle  Response (dimers/s)")
            print("-"*30)
            for i, resp in enumerate(responses):
                print(f"  {i+1}      {resp:.1f}")
        
        print(f"\n  Enhancement over cycles: {result.multicycle_enhancement:.2f}×")
        
        if result.multicycle_enhancement > 1.3:
            print("  ✓ PROGRESSIVE ENHANCEMENT CONFIRMED")
            print("    Each cycle potentiates the next")
        else:
            print("  ⚠ Limited cycle-to-cycle enhancement")
        
        # Overall conclusions
        print("\n" + "="*70)
        print("FEEDBACK LOOP CONCLUSIONS")
        print("="*70)
        
        findings = []
        if result.amplification_factor > 1.5:
            findings.append("✓ Positive feedback amplification")
        if result.bistability_detected:
            findings.append("✓ Bistability (two stable states)")
        if result.multicycle_enhancement > 1.3:
            findings.append("✓ Progressive enhancement over cycles")
        
        if len(findings) >= 2:
            print("\n  FEEDBACK LOOP VALIDATED")
            for f in findings:
                print(f"    {f}")
            print("\n  The quantum-classical system shows emergent self-organization")
        else:
            print("\n  PARTIAL VALIDATION")
            print("  Some feedback effects present but not all predicted behaviors")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: FeedbackResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # === Panel A: Amplification ===
        ax1 = axes[0, 0]
        
        amp = result.summary.get('amplification', {})
        labels = ['Without\nCoupling', 'With\nCoupling']
        values = [amp.get('mean_uncoupled_rate', 0), amp.get('mean_coupled_rate', 0)]
        colors = ['#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
        
        # Annotate amplification
        if values[0] > 0:
            amp_factor = values[1] / values[0]
            ax1.annotate(f'{amp_factor:.1f}× amplification',
                        xy=(1, values[1]), xytext=(0.5, values[1] * 1.1),
                        fontsize=11, ha='center',
                        arrowprops=dict(arrowstyle='->', color='black'))
        
        ax1.set_ylabel('Dimer Formation Rate (per second)', fontsize=11)
        ax1.set_title('A. Feedback Amplification', fontweight='bold')
        
        # === Panel B: Bistability ===
        ax2 = axes[0, 1]
        
        intensities = []
        high_fracs = []
        low_fracs = []
        
        for intensity in self.stim_intensities:
            key = f'bistability_{intensity}'
            if key in result.summary:
                data = result.summary[key]
                intensities.append(intensity)
                high_fracs.append(data.get('high_fraction', 0))
                low_fracs.append(data.get('low_fraction', 0))
        
        x = np.arange(len(intensities))
        width = 0.35
        
        ax2.bar(x - width/2, high_fracs, width, label='High State', color='#2ca02c', alpha=0.8)
        ax2.bar(x + width/2, low_fracs, width, label='Low State', color='#d62728', alpha=0.8)
        
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Stimulation Intensity', fontsize=11)
        ax2.set_ylabel('Fraction of Trials', fontsize=11)
        ax2.set_title('B. Bistability: State Distribution', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{i:.1f}' for i in intensities])
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(0, 1.1)
        
        # === Panel C: Multicycle Enhancement ===
        ax3 = axes[1, 0]
        
        multi = result.summary.get('multicycle', {})
        responses = multi.get('responses_by_cycle', [])
        
        if responses:
            cycles = list(range(1, len(responses) + 1))
            ax3.plot(cycles, responses, 'o-', color='#1f77b4', markersize=10, linewidth=2)
            
            # Fit trend line
            if len(responses) > 2:
                z = np.polyfit(cycles, responses, 1)
                p = np.poly1d(z)
                ax3.plot(cycles, p(cycles), '--', color='gray', alpha=0.5, label='Trend')
            
            ax3.set_xlabel('Stimulation Cycle', fontsize=11)
            ax3.set_ylabel('Response (dimers/s)', fontsize=11)
            ax3.set_title('C. Progressive Enhancement', fontweight='bold')
            ax3.set_xticks(cycles)
        else:
            ax3.text(0.5, 0.5, 'No multicycle data', ha='center', va='center',
                    transform=ax3.transAxes)
        
        # === Panel D: Summary Diagram ===
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Draw feedback loop diagram
        diagram_text = """
        QUANTUM-CLASSICAL FEEDBACK LOOP
        
        ┌─────────────────────────────────────┐
        │                                     │
        │  Tryptophan Superradiance (Q1)      │
        │  ↓ EM field (20 kT)                 │
        │  ↓                                  │
        │  Ca-Phosphate Dimer Formation (Q2)  │
        │  ↓ Coherent field (30 kT)           │
        │  ↓                                  │
        │  Tubulin Conformations              │
        │  ↓                                  │
        │  Better Trp Network Coupling        │
        │  └──────────────────────────────────┘
        │           (positive feedback)
        └─────────────────────────────────────┘
        """
        
        # Summary text
        summary = f"""
RESULTS SUMMARY

Amplification: {result.amplification_factor:.1f}×
Bistability: {'YES' if result.bistability_detected else 'NO'}
Multicycle Enhancement: {result.multicycle_enhancement:.1f}×

High State Fraction: {result.high_state_fraction:.0%}
Low State Fraction: {result.low_state_fraction:.0%}
        """
        
        ax4.text(0.5, 0.5, summary, fontsize=11, ha='center', va='center',
                transform=ax4.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('D. Summary', fontweight='bold')
        
        plt.suptitle('Feedback Loop: Bidirectional Quantum-Classical Coupling', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'feedback_loop.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: FeedbackResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'feedback_loop',
            'tier': 4,
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'amplification_factor': result.amplification_factor,
            'bistability_detected': result.bistability_detected,
            'multicycle_enhancement': result.multicycle_enhancement,
            'high_state_fraction': result.high_state_fraction,
            'low_state_fraction': result.low_state_fraction,
            'summary': result.summary,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses,
            'n_cycles': self.n_cycles
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: FeedbackResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'amplification_factor': result.amplification_factor,
            'bistability_detected': result.bistability_detected,
            'multicycle_enhancement': result.multicycle_enhancement,
            'high_state_fraction': result.high_state_fraction,
            'feedback_validated': (result.amplification_factor > 1.5 and 
                                  (result.bistability_detected or result.multicycle_enhancement > 1.3)),
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running feedback loop experiment (quick mode)...")
    exp = FeedbackLoopExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()