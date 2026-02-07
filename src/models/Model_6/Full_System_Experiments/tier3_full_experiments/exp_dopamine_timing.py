#!/usr/bin/env python3
"""
Experiment: Dopamine as Quantum Readout Operator
==================================================

Demonstrates that dopamine is the measurement signal that collapses quantum
eligibility into classical synaptic strengthening. The isotope experiment
shows coherence persists; THIS experiment shows what happens when you read
it out — and what happens when you don't.

Scientific basis:
- Eligibility trace = mean singlet probability of dimer nuclear spins
- Dopamine triggers the three-factor gate: eligibility × dopamine × calcium
- Without dopamine, quantum memory exists but is never converted to plasticity
- Without prior activity, dopamine has nothing to read out

Protocol (stim_plus_dopamine condition):
1. Baseline (100ms): resting, no reward
2. Theta-burst stimulation: reward=FALSE (critical fix — no premature gate firing)
3. Variable delay (0–200s): resting, no reward
4. Measure eligibility at dopamine onset
5. Dopamine readout (300ms): reward=TRUE — gate evaluates here
6. Consolidation (1s): no reward
7. Measure final synaptic strength

Three conditions:
- stim_plus_dopamine: Activity → delay → dopamine readout
- stim_no_dopamine:   Activity → delay → NO readout (quantum memory unused)
- dopamine_only:      No activity → delay → dopamine (nothing to read)

Success criteria:
- stim_plus_dopamine: Δstrength > 0, decaying with T₂
- stim_no_dopamine: Δstrength ≈ 0 at all delays (memory exists but unread)
- dopamine_only: Δstrength ≈ 0 at all delays (nothing to read)
- Fitted T₂ from strength decay consistent with isotope experiment

References:
- Yagishita et al. (2014): Dopamine timing window for spine enlargement
- Fisher (2015): Quantum cognition hypothesis
- Agarwal et al. (2023): Ca₆(PO₄)₄ dimer coherence times

Author: Sarah Davidson
University of Florida
Date: February 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimingCondition:
    """Single experimental condition"""
    condition_type: str  # 'stim_plus_dopamine', 'stim_no_dopamine', 'dopamine_only'
    dopamine_delay_s: float
    n_synapses: int = 10
    
    @property
    def name(self) -> str:
        return f"{self.condition_type}_delay_{self.dopamine_delay_s:.1f}s"


@dataclass 
class TimingTrialResult:
    """Results from single trial"""
    condition: TimingCondition
    trial_id: int
    
    # Key measurements at dopamine onset (or equivalent time point)
    eligibility_at_readout: float = 0.0
    mean_singlet_prob: float = 0.0
    dimer_count: int = 0
    
    # Post-stimulation measurements (before delay)
    eligibility_post_stim: float = 0.0
    dimers_post_stim: int = 0
    peak_calcium_uM: float = 0.0
    peak_em_field_kT: float = 0.0
    
    # Outcome after dopamine readout + consolidation
    committed: bool = False
    commitment_level: float = 0.0
    final_strength: float = 1.0
    delta_strength: float = 0.0  # final_strength - 1.0
    
    # Per-synapse commitment (for later coordinated decoherence analysis)
    n_synapses_committed: int = 0
    
    runtime_s: float = 0.0


@dataclass
class TimingResult:
    """Complete experiment results"""
    conditions: List[TimingCondition]
    trials: List[TimingTrialResult]
    
    # Summary by condition type and delay
    summary: Dict = field(default_factory=dict)
    
    # Fitted parameters (from stim_plus_dopamine condition)
    fitted_T2: float = 0.0
    fitted_A: float = 0.0  # Initial amplitude
    r_squared: float = 0.0
    
    # Theory comparison
    theory_T2: float = 100.0  # seconds
    
    timestamp: str = ""
    runtime_s: float = 0.0


# =============================================================================
# EXPERIMENT CLASS
# =============================================================================

class DopamineTimingExperiment:
    """
    Dopamine timing experiment — demonstrates dopamine as quantum readout operator.
    
    Three conditions show that:
    1. Quantum memory + dopamine readout = synaptic strengthening (delay-dependent)
    2. Quantum memory without readout = no change (memory wasted)
    3. Readout without prior memory = no change (nothing to read)
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        if quick_mode:
            self.n_trials = 2
            self.n_synapses = 5
            self.stim_duration_s = 0.5
            self.consolidation_s = 1.0
            # Fewer delays for quick testing
            self.main_delays = [0, 10, 30, 60, 120]
            self.control_delays = [0, 30, 120]
        else:
            self.n_trials = 10
            self.n_synapses = 10
            self.stim_duration_s = 1.0
            self.consolidation_s = 1.0
            # Extended delays matching isotope experiment range
            self.main_delays = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200]
            # Controls at subset of delays (flat lines need fewer points)
            self.control_delays = [0, 15, 30, 60, 120, 200]
    
    def _create_network(self) -> MultiSynapseNetwork:
        """Create standard P31 network"""
        params = Model6Parameters()
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        network = MultiSynapseNetwork(
            n_synapses=self.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _run_trial(self, condition: TimingCondition, trial_id: int) -> TimingTrialResult:
        """
        Execute single trial with corrected protocol.
        
        CRITICAL FIX: reward=False during stimulation phase.
        Dopamine only delivered in Phase 5 (readout), not during theta-burst.
        """
        start_time = time.time()
        
        result = TimingTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        network = self._create_network()
        network.set_coordination_mode(True)
        dt = 0.001  # 1 ms timestep
        
        # Determine what this condition does
        do_stimulate = condition.condition_type in ('stim_plus_dopamine', 'stim_no_dopamine')
        do_dopamine = condition.condition_type in ('stim_plus_dopamine', 'dopamine_only')
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (theta-burst, NO DOPAMINE) ===
        if do_stimulate:
            # Theta-burst: 5 bursts × 4 spikes at 100 Hz, 200ms inter-burst
            # CRITICAL: reward=False during all stimulation
            for burst in range(5):
                for spike in range(4):
                    # 2ms depolarization (spike)
                    for _ in range(2):
                        network.step(dt, {"voltage": -10e-3, "reward": False})
                    # 8ms rest between spikes
                    for _ in range(8):
                        network.step(dt, {"voltage": -70e-3, "reward": False})
                # 160ms inter-burst interval
                for _ in range(160):
                    network.step(dt, {"voltage": -70e-3, "reward": False})
        else:
            # dopamine_only condition: sit at rest for equivalent duration
            # Theta-burst total: 5 × (4×10ms + 160ms) = 1000ms = 1s
            for _ in range(1000):
                network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # Record post-stimulation state
        result.eligibility_post_stim = np.mean([s.get_eligibility() for s in network.synapses])
        result.dimers_post_stim = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        # Track peak calcium and EM field achieved during stimulation
        result.peak_calcium_uM = max(
            getattr(s, '_peak_calcium_uM', 0.0) for s in network.synapses
        )
        result.peak_em_field_kT = max(
            getattr(s, '_collective_field_kT', 0.0) for s in network.synapses
        )
        
        # === PHASE 3: DELAY (no reward, resting potential) ===
        if condition.dopamine_delay_s > 0:
            # Use coarser timestep for long delays (performance)
            dt_delay = 0.1 if condition.dopamine_delay_s > 5 else 0.05
            n_delay = int(condition.dopamine_delay_s / dt_delay)
            
            for i in range(n_delay):
                network.step(dt_delay, {'voltage': -70e-3, 'reward': False})
        
        # === PHASE 4: MEASURE AT READOUT ONSET ===
        # Snapshot eligibility just before dopamine arrives (or equivalent time)
        eligibilities = [s.get_eligibility() for s in network.synapses]
        result.eligibility_at_readout = np.mean(eligibilities)
        
        # Dimer metrics at readout time
        all_ps = []
        total_dimers = 0
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                total_dimers += len(s.dimer_particles.dimers)
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        
        result.mean_singlet_prob = np.mean(all_ps) if all_ps else 0.0
        result.dimer_count = total_dimers
        
        # === PHASE 5: DOPAMINE READOUT (300ms) ===
        if do_dopamine:
            for _ in range(300):
                network.step_with_coordination(dt, {"voltage": -70e-3, "reward": True})
        else:
            # stim_no_dopamine: equivalent time passes with no reward
            for _ in range(300):
                network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 6: CONSOLIDATION (1s) ===
        dt_consol = 0.01  # 10ms timestep for consolidation
        n_consol = int(self.consolidation_s / dt_consol)
        for _ in range(n_consol):
            network.step(dt_consol, {"voltage": -70e-3, "reward": False})
        
        # === FINAL MEASUREMENTS ===
        result.committed = network.network_committed
        result.commitment_level = network.network_commitment_level
        
        # Count per-synapse commitments
        result.n_synapses_committed = sum(
            1 for s in network.synapses if getattr(s, '_camkii_committed', False)
        )
        
        # Synaptic strength
        if result.committed:
            result.final_strength = 1.0 + 0.5 * result.commitment_level
        else:
            result.final_strength = 1.0
        
        result.delta_strength = result.final_strength - 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> TimingResult:
        """Execute complete experiment with all three conditions"""
        start_time = time.time()
        
        # Build all conditions
        conditions = []
        
        # Main condition: stim + dopamine at variable delays
        for delay in self.main_delays:
            conditions.append(TimingCondition(
                condition_type='stim_plus_dopamine',
                dopamine_delay_s=delay,
                n_synapses=self.n_synapses
            ))
        
        # Control 1: stim but NO dopamine
        for delay in self.control_delays:
            conditions.append(TimingCondition(
                condition_type='stim_no_dopamine',
                dopamine_delay_s=delay,
                n_synapses=self.n_synapses
            ))
        
        # Control 2: dopamine only, no prior stimulation
        for delay in self.control_delays:
            conditions.append(TimingCondition(
                condition_type='dopamine_only',
                dopamine_delay_s=delay,
                n_synapses=self.n_synapses
            ))
        
        # Run trials
        trials = []
        n_control_trials = max(2, self.n_trials // 2)  # Controls need fewer trials
        
        for cond in conditions:
            n = self.n_trials if cond.condition_type == 'stim_plus_dopamine' else n_control_trials
            
            if self.verbose:
                print(f"  {cond.name}: ", end='', flush=True)
            
            for trial_id in range(n):
                trial_result = self._run_trial(cond, trial_id)
                trials.append(trial_result)
                
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                cond_trials = [t for t in trials if t.condition.name == cond.name]
                mean_ds = np.mean([t.delta_strength for t in cond_trials])
                mean_elig = np.mean([t.eligibility_at_readout for t in cond_trials])
                print(f" Δstr={mean_ds:.3f}, elig={mean_elig:.3f}", flush=True)
        
        # Build result
        result = TimingResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Fit decay curve from stim_plus_dopamine condition
        self._fit_strength_decay(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[TimingTrialResult]) -> Dict:
        """Compute summary statistics by condition type and delay"""
        summary = {}
        
        for ctype in ['stim_plus_dopamine', 'stim_no_dopamine', 'dopamine_only']:
            ctype_trials = [t for t in trials if t.condition.condition_type == ctype]
            delays = sorted(set(t.condition.dopamine_delay_s for t in ctype_trials))
            
            summary[ctype] = {}
            
            for delay in delays:
                d_trials = [t for t in ctype_trials if t.condition.dopamine_delay_s == delay]
                
                if d_trials:
                    ds = [t.delta_strength for t in d_trials]
                    eligs = [t.eligibility_at_readout for t in d_trials]
                    commits = [1 if t.committed else 0 for t in d_trials]
                    dimers = [t.dimer_count for t in d_trials]
                    n_syn_committed = [t.n_synapses_committed for t in d_trials]
                    
                    summary[ctype][delay] = {
                        'delta_strength_mean': float(np.mean(ds)),
                        'delta_strength_std': float(np.std(ds)),
                        'eligibility_mean': float(np.mean(eligs)),
                        'eligibility_std': float(np.std(eligs)),
                        'commit_rate': float(np.mean(commits)),
                        'dimer_count_mean': float(np.mean(dimers)),
                        'n_synapses_committed_mean': float(np.mean(n_syn_committed)),
                        'n_trials': len(d_trials)
                    }
        
        return summary
    
    def _fit_strength_decay(self, result: TimingResult):
        """Fit exponential decay to Δstrength from stim_plus_dopamine condition"""
        spd = result.summary.get('stim_plus_dopamine', {})
        
        if len(spd) < 3:
            return
        
        delays = sorted(spd.keys())
        strengths = [spd[d]['delta_strength_mean'] for d in delays]
        
        x = np.array(delays, dtype=float)
        y = np.array(strengths, dtype=float)
        
        # Only fit if there's actual signal
        if max(y) < 0.01:
            return
        
        # Fit: Δstrength = A × exp(-t / T₂)
        try:
            def decay_func(t, A, T2):
                return A * np.exp(-t / T2)
            
            popt, pcov = curve_fit(
                decay_func, x, y,
                p0=[max(y), 100.0],
                bounds=([0, 1], [2.0, 1000]),
                maxfev=5000
            )
            
            result.fitted_A = popt[0]
            result.fitted_T2 = popt[1]
            
            # R²
            y_pred = decay_func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
        except (RuntimeError, ValueError) as e:
            if self.verbose:
                print(f"  Warning: Curve fit failed: {e}")
    
    def print_summary(self, result: TimingResult):
        """Print formatted summary"""
        print("\n" + "=" * 70)
        print("DOPAMINE TIMING: QUANTUM READOUT EXPERIMENT")
        print("=" * 70)
        
        for ctype, label in [
            ('stim_plus_dopamine', 'STIM + DOPAMINE (main)'),
            ('stim_no_dopamine', 'STIM ONLY (no readout)'),
            ('dopamine_only', 'DOPAMINE ONLY (nothing to read)')
        ]:
            print(f"\n--- {label} ---")
            print(f"{'Delay (s)':<12} {'Δ Strength':<18} {'Eligibility':<18} {'Commit%':<10} {'Dimers':<8}")
            print("-" * 70)
            
            cdata = result.summary.get(ctype, {})
            for delay in sorted(cdata.keys()):
                s = cdata[delay]
                print(f"{delay:<12.1f} "
                      f"{s['delta_strength_mean']:.3f} ± {s['delta_strength_std']:.3f}    "
                      f"{s['eligibility_mean']:.3f} ± {s['eligibility_std']:.3f}    "
                      f"{s['commit_rate']:<10.0%} "
                      f"{s['dimer_count_mean']:<8.0f}")
        
        print("\n" + "=" * 70)
        print("STRENGTH DECAY FIT (stim_plus_dopamine)")
        print("=" * 70)
        
        if result.fitted_T2 > 0:
            print(f"  Fitted T₂:     {result.fitted_T2:.1f} s")
            print(f"  Initial Δstr:   {result.fitted_A:.3f}")
            print(f"  R²:             {result.r_squared:.3f}")
            print(f"  Theory T₂:     {result.theory_T2:.1f} s")
            print(f"  Ratio:          {result.fitted_T2/result.theory_T2:.2f}× theory")
        else:
            print("  Fit not available — check if stim_plus_dopamine produced signal")
        
        # Key biological conclusion
        spd = result.summary.get('stim_plus_dopamine', {})
        sno = result.summary.get('stim_no_dopamine', {})
        dop = result.summary.get('dopamine_only', {})
        
        spd_max = max((s['delta_strength_mean'] for s in spd.values()), default=0)
        sno_max = max((s['delta_strength_mean'] for s in sno.values()), default=0)
        dop_max = max((s['delta_strength_mean'] for s in dop.values()), default=0)
        
        print(f"\n  Max Δstrength (stim+DA):     {spd_max:.3f}")
        print(f"  Max Δstrength (stim only):   {sno_max:.3f}")
        print(f"  Max Δstrength (DA only):     {dop_max:.3f}")
        
        if spd_max > 0.05 and sno_max < 0.01 and dop_max < 0.01:
            print("\n  ✓ DOPAMINE READOUT VALIDATED")
            print("    → Quantum memory requires dopamine to convert to structural change")
            print("    → Dopamine alone insufficient without prior quantum memory")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: TimingResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """
        Single-panel figure: Dopamine as quantum readout operator.
        
        Shows Δ synaptic strength vs. dopamine delay for all three conditions.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        
        # Colors
        color_main = '#2166AC'      # Blue for stim + dopamine
        color_no_da = '#999999'     # Gray for stim only
        color_da_only = '#B2182B'   # Red for dopamine only
        color_fit = '#2166AC'       # Match main color for fit
        
        spd = result.summary.get('stim_plus_dopamine', {})
        sno = result.summary.get('stim_no_dopamine', {})
        dop = result.summary.get('dopamine_only', {})
        
        # === BEHAVIORAL LEARNING WINDOW ===
        ax.axvspan(0, 60, alpha=0.06, color='green', zorder=0)
        ax.text(30, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 0.48,
                'Behavioral learning window\n(Yagishita 2014, Bittner 2017)',
                ha='center', va='bottom', fontsize=8, color='green', alpha=0.7,
                style='italic')
        
        # === SCATTER: INDIVIDUAL TRIALS ===
        # stim_plus_dopamine trials
        spd_trials = [t for t in result.trials if t.condition.condition_type == 'stim_plus_dopamine']
        if spd_trials:
            jitter = np.random.normal(0, 0.8, len(spd_trials))
            ax.scatter(
                [t.condition.dopamine_delay_s + j for t, j in zip(spd_trials, jitter)],
                [t.delta_strength for t in spd_trials],
                c=color_main, alpha=0.25, s=20, zorder=3, edgecolors='none'
            )
        
        # stim_no_dopamine trials
        sno_trials = [t for t in result.trials if t.condition.condition_type == 'stim_no_dopamine']
        if sno_trials:
            jitter = np.random.normal(0, 0.8, len(sno_trials))
            ax.scatter(
                [t.condition.dopamine_delay_s + j for t, j in zip(sno_trials, jitter)],
                [t.delta_strength for t in sno_trials],
                c=color_no_da, alpha=0.25, s=15, zorder=3, edgecolors='none',
                marker='s'
            )
        
        # dopamine_only trials
        dop_trials = [t for t in result.trials if t.condition.condition_type == 'dopamine_only']
        if dop_trials:
            jitter = np.random.normal(0, 0.8, len(dop_trials))
            ax.scatter(
                [t.condition.dopamine_delay_s + j for t, j in zip(dop_trials, jitter)],
                [t.delta_strength for t in dop_trials],
                c=color_da_only, alpha=0.25, s=15, zorder=3, edgecolors='none',
                marker='^'
            )
        
        # === MEAN LINES ===
        # stim_plus_dopamine means with error bars
        if spd:
            delays_spd = sorted(spd.keys())
            means_spd = [spd[d]['delta_strength_mean'] for d in delays_spd]
            stds_spd = [spd[d]['delta_strength_std'] for d in delays_spd]
            ax.errorbar(delays_spd, means_spd, yerr=stds_spd,
                       fmt='o-', color=color_main, markersize=6, linewidth=1.5,
                       capsize=3, capthick=1, zorder=5,
                       label='Activity + Dopamine')
        
        # stim_no_dopamine means
        if sno:
            delays_sno = sorted(sno.keys())
            means_sno = [sno[d]['delta_strength_mean'] for d in delays_sno]
            stds_sno = [sno[d]['delta_strength_std'] for d in delays_sno]
            ax.errorbar(delays_sno, means_sno, yerr=stds_sno,
                       fmt='s--', color=color_no_da, markersize=5, linewidth=1.2,
                       capsize=2, capthick=0.8, zorder=4,
                       label='Activity only (no readout)')
        
        # dopamine_only means
        if dop:
            delays_dop = sorted(dop.keys())
            means_dop = [dop[d]['delta_strength_mean'] for d in delays_dop]
            stds_dop = [dop[d]['delta_strength_std'] for d in delays_dop]
            ax.errorbar(delays_dop, means_dop, yerr=stds_dop,
                       fmt='^--', color=color_da_only, markersize=5, linewidth=1.2,
                       capsize=2, capthick=0.8, zorder=4,
                       label='Dopamine only (no memory)')
        
        # === FITTED DECAY CURVE ===
        if result.fitted_T2 > 0 and result.fitted_A > 0:
            t_smooth = np.linspace(0, max(self.main_delays) + 10, 200)
            y_fit = result.fitted_A * np.exp(-t_smooth / result.fitted_T2)
            ax.plot(t_smooth, y_fit, '-', color=color_fit, alpha=0.4, linewidth=2,
                   zorder=2)
        
        # === T₂ ANNOTATION BOX ===
        if result.fitted_T2 > 0:
            textstr = (f'Fitted T₂ = {result.fitted_T2:.0f} s\n'
                      f'R² = {result.r_squared:.3f}\n'
                      f'n = {self.n_trials} trials/delay')
            props = dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor=color_main, alpha=0.9)
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=props, zorder=10)
        
        # === ANNOTATION: WHAT CONTROLS SHOW ===
        # Find a good position for annotation
        max_delay = max(self.main_delays)
        if spd:
            max_str = max(spd[d]['delta_strength_mean'] for d in spd)
        else:
            max_str = 0.5
            
        ax.annotate('Quantum memory exists\nbut never read out →\nno structural change',
                    xy=(max_delay * 0.7, 0.005), fontsize=8, color=color_no_da,
                    style='italic', ha='center', va='bottom')
        
        # === FORMATTING ===
        ax.set_xlabel('Dopamine Delay (s)', fontsize=12)
        ax.set_ylabel('Δ Synaptic Strength', fontsize=12)
        ax.set_title('Dopamine as Quantum Readout: Timing Determines Plasticity',
                     fontsize=13, fontweight='bold')
        
        ax.set_xlim(-5, max(self.main_delays) + 10)
        # Set y limits to show zero baseline clearly
        if spd:
            y_max = max(spd[d]['delta_strength_mean'] + spd[d]['delta_strength_std'] 
                       for d in spd) * 1.3
            ax.set_ylim(-0.05, max(0.6, y_max))
        else:
            ax.set_ylim(-0.05, 0.6)
        
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'dopamine_timing.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            if self.verbose:
                print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: TimingResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'dopamine_timing_readout',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'fitted_T2': result.fitted_T2,
            'fitted_A': result.fitted_A,
            'r_squared': result.r_squared,
            'theory_T2': result.theory_T2,
            'summary': result.summary,
            'main_delays': self.main_delays,
            'control_delays': self.control_delays,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: TimingResult) -> dict:
        """Get summary as dictionary for master results"""
        spd = result.summary.get('stim_plus_dopamine', {})
        sno = result.summary.get('stim_no_dopamine', {})
        
        return {
            'fitted_T2_s': result.fitted_T2,
            'theory_T2_s': result.theory_T2,
            'ratio_to_theory': result.fitted_T2 / result.theory_T2 if result.theory_T2 > 0 else 0,
            'r_squared': result.r_squared,
            'max_delta_strength_with_DA': max((s['delta_strength_mean'] for s in spd.values()), default=0),
            'max_delta_strength_no_DA': max((s['delta_strength_mean'] for s in sno.values()), default=0),
            'readout_validated': (
                max((s['delta_strength_mean'] for s in spd.values()), default=0) > 0.05 and
                max((s['delta_strength_mean'] for s in sno.values()), default=0) < 0.01
            ),
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running dopamine timing experiment (quick mode)...")
    exp = DopamineTimingExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()