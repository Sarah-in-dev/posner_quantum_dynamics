#!/usr/bin/env python3
"""
Experiment: Quantum → Classical Cascade
=========================================

THE CORE DEMONSTRATION: Shows how quantum coherence gates classical plasticity
through protein conformational changes.

THEORETICAL FRAMEWORK:
─────────────────────
The quantum layer (Q1 tryptophan + Q2 dimer coherence) generates a 20-40 kT
electromagnetic field that modulates protein activation barriers:

    QUANTUM                 BARRIER                 PROTEIN                 OUTPUT
    ───────                 ───────                 ───────                 ──────
    Q1: Trp EM field ─┐                            
                      ├→ 20-40 kT field → ~1.5 kT → CaMKII pT286 → Molecular → DDSC →
    Q2: Dimer coh. ───┘    (combined)    barrier↓   GluN2B bound   Memory     Trigger
                                                                                 ↓
                                                                         Projected
                                                                         Plasticity

THE KEY CLAIM:
─────────────
Without the quantum field, CaMKII autophosphorylation is too slow - the calcium
signal dissipates before commitment occurs. With the 20-40 kT field reducing
the barrier by ~1.5 kT, the rate increases by ~4.5×, allowing commitment within
the ~100s coherence window. This is the quantum advantage.

WHAT WE TRACK:
─────────────
1. QUANTUM LAYER: Q1 field, Q2 field, combined field, coherence
2. BARRIER MODULATION: Reduction (kT), rate enhancement factor
3. CaMKII STATE: CaCaM_bound, active, pT286, GluN2B_bound, molecular_memory
4. CASCADE: Eligibility, DDSC triggered, structural_drive, projected_strength

CONDITIONS:
──────────
• Full (+EM, +MT): Maximum quantum effect - both Q1 and Q2 active
• Reduced Q1 (-MT): No microtubule invasion → fewer tryptophans
• No Q2 (APV): NMDA blocked → no calcium → no dimers
• Control (-EM): EM coupling disabled → no quantum modulation

SUCCESS CRITERIA:
────────────────
1. +EM shows higher field (20-40 kT) than -EM (~0 kT)
2. +EM shows barrier reduction (~1-2 kT) while -EM shows none
3. +EM shows faster/higher pT286 than -EM
4. +EM shows molecular_memory → DDSC trigger → projected plasticity
5. -EM shows slow/incomplete pT286 → no DDSC → no plasticity

REFERENCES:
──────────
- Fisher 2015: Quantum cognition hypothesis
- Agarwal 2023: Dimer coherence ~100s, field 20-40 kT
- Nature 2024: DDSC as instructive signal
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
class CascadeCondition:
    """Experimental condition"""
    name: str
    em_enabled: bool = True
    mt_invaded: bool = True
    apv_applied: bool = False  # Blocks NMDA → no calcium → no dimers
    n_synapses: int = 10
    
    @property 
    def description(self) -> str:
        if not self.em_enabled:
            return "No EM coupling (control)"
        elif self.apv_applied:
            return "APV blocks Q2 (no dimers)"
        elif not self.mt_invaded:
            return "No MT invasion (reduced Q1)"
        else:
            return "Full EM (Q1 + Q2)"


@dataclass
class CascadeSnapshot:
    """State at a single timepoint"""
    time_s: float = 0.0
    
    # Quantum layer
    n_dimers: int = 0
    mean_singlet_prob: float = 0.25
    n_entangled: int = 0
    q1_field_kT: float = 0.0  # Tryptophan contribution
    q2_field_kT: float = 0.0  # Dimer contribution (estimated)
    combined_field_kT: float = 0.0
    
    # Barrier modulation
    barrier_reduction_kT: float = 0.0
    rate_enhancement: float = 1.0
    
    # CaMKII state
    CaCaM_bound: float = 0.0
    CaMKII_active: float = 0.0
    pT286: float = 0.0
    GluN2B_bound: float = 0.0
    molecular_memory: float = 0.0
    
    # Cascade output
    eligibility: float = 0.0
    ddsc_triggered: bool = False
    ddsc_current: float = 0.0
    ddsc_integrated: float = 0.0
    structural_drive: float = 0.0


@dataclass 
class CascadeTrialResult:
    """Results from single trial"""
    condition: CascadeCondition
    trial_id: int
    
    # Time series
    snapshots: List[CascadeSnapshot] = field(default_factory=list)
    
    # Peak values
    peak_field_kT: float = 0.0
    peak_barrier_reduction_kT: float = 0.0
    peak_pT286: float = 0.0
    peak_molecular_memory: float = 0.0
    
    # Final outcomes
    final_eligibility: float = 0.0
    ddsc_triggered: bool = False
    final_structural_drive: float = 0.0
    projected_strength: float = 1.0
    
    # Timing
    time_to_pT286_half: float = 0.0  # Time to reach 50% of peak pT286
    time_to_memory: float = 0.0  # Time molecular_memory > 0.1
    
    runtime_s: float = 0.0


@dataclass
class CascadeResult:
    """Complete experiment results"""
    conditions: List[CascadeCondition]
    trials: List[CascadeTrialResult]
    
    summary: Dict = field(default_factory=dict)
    
    # Key findings
    field_difference_confirmed: bool = False  # +EM > -EM field?
    barrier_modulation_confirmed: bool = False  # +EM shows barrier reduction?
    camkii_enhancement_confirmed: bool = False  # +EM faster pT286?
    plasticity_gating_confirmed: bool = False  # Only +EM triggers DDSC?
    
    timestamp: str = ""
    runtime_s: float = 0.0


class QuantumClassicalCascadeExperiment:
    """
    Demonstrates the complete quantum → classical cascade.
    
    Tracks all stages from quantum field through CaMKII conformational
    changes to plasticity outcome.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        if quick_mode:
            self.n_trials = 2
            self.n_synapses = 5
            self.observation_s = 60  # Watch cascade for 60s
            self.sample_interval_s = 2.0
        else:
            self.n_trials = 5
            self.n_synapses = 10
            self.observation_s = 120  # Full DDSC window
            self.sample_interval_s = 1.0
        
        # Experimental conditions
        self.conditions = [
            CascadeCondition("full_em", em_enabled=True, mt_invaded=True, apv_applied=False),
            CascadeCondition("no_mt", em_enabled=True, mt_invaded=False, apv_applied=False),
            CascadeCondition("apv", em_enabled=True, mt_invaded=True, apv_applied=True),
            CascadeCondition("no_em", em_enabled=False, mt_invaded=True, apv_applied=False),
        ]
        
        if quick_mode:
            # Just compare +EM vs -EM
            self.conditions = [
                CascadeCondition("full_em", em_enabled=True, mt_invaded=True, apv_applied=False),
                CascadeCondition("no_em", em_enabled=False, mt_invaded=True, apv_applied=False),
            ]
    
    def _create_network(self, condition: CascadeCondition) -> MultiSynapseNetwork:
        """Create network for specified condition"""
        params = Model6Parameters()
        
        # EM coupling
        params.em_coupling_enabled = condition.em_enabled
        params.multi_synapse_enabled = True
        
        # APV blocks NMDA (no calcium influx)
        if condition.apv_applied:
            params.calcium.nmda_blocked = True
        
        network = MultiSynapseNetwork(
            n_synapses=condition.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        
        # MT invasion affects tryptophan count
        network.set_microtubule_invasion(condition.mt_invaded)
        
        return network
    
    def _get_snapshot(self, network: MultiSynapseNetwork, current_time: float) -> CascadeSnapshot:
        """Capture current state across all cascade stages"""
        snap = CascadeSnapshot(time_s=current_time)
        
        # Aggregate across synapses
        n_dimers_list = []
        singlet_probs = []
        n_entangled_list = []
        q1_fields = []
        combined_fields = []
        
        cacam_list = []
        camkii_active_list = []
        pt286_list = []
        glun2b_list = []
        memory_list = []
        
        eligibility_list = []
        ddsc_triggered_list = []
        ddsc_current_list = []
        ddsc_integrated_list = []
        structural_drive_list = []
        
        barrier_reductions = []
        rate_enhancements = []
        
        for s in network.synapses:
            # Quantum layer
            if hasattr(s, 'dimer_particles'):
                n_dimers_list.append(len(s.dimer_particles.dimers))
                for d in s.dimer_particles.dimers:
                    singlet_probs.append(d.singlet_probability)
                n_entangled_list.append(sum(1 for d in s.dimer_particles.dimers if d.is_entangled))
            
            # EM fields
            if hasattr(s, '_em_field_trp'):
                q1_fields.append(s._em_field_trp)
            if hasattr(s, '_collective_field_kT'):
                combined_fields.append(s._collective_field_kT)
            
            # CaMKII state
            if hasattr(s, 'camkii'):
                cacam_list.append(s.camkii.CaCaM_bound)
                camkii_active_list.append(s.camkii.CaMKII_active)
                pt286_list.append(s.camkii.pT286)
                glun2b_list.append(s.camkii.GluN2B_bound)
                memory_list.append(s.camkii.molecular_memory)
            
            if hasattr(s, 'em_coupling') and s.em_coupling is not None:
                n_coh = len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                mod = s.em_coupling.reverse.calculate_protein_modulation(n_coh, 'camkii')
                barrier_reductions.append(mod.get('barrier_reduction_kT', 0))
                rate_enhancements.append(mod.get('rate_enhancement', 1))
            
            # DDSC cascade
            if hasattr(s, 'get_eligibility'):
                eligibility_list.append(s.get_eligibility())
            if hasattr(s, 'ddsc'):
                ddsc_triggered_list.append(s.ddsc.triggered)
                ddsc_current_list.append(s.ddsc.current_ddsc)
                ddsc_integrated_list.append(s.ddsc.integrated_ddsc)
                structural_drive_list.append(s.ddsc.get_structural_drive())
        
        # Aggregate
        snap.n_dimers = sum(n_dimers_list) if n_dimers_list else 0
        snap.mean_singlet_prob = np.mean(singlet_probs) if singlet_probs else 0.25
        snap.n_entangled = sum(n_entangled_list) if n_entangled_list else 0
        snap.q1_field_kT = np.mean(q1_fields) if q1_fields else 0.0
        snap.combined_field_kT = np.mean(combined_fields) if combined_fields else 0.0
        
        # Estimate Q2 contribution (combined - Q1, roughly)
        snap.q2_field_kT = max(0, snap.combined_field_kT - snap.q1_field_kT * 0.5)
        
        snap.barrier_reduction_kT = np.mean(barrier_reductions) if barrier_reductions else 0.0
        snap.rate_enhancement = np.mean(rate_enhancements) if rate_enhancements else 1.0
        
        snap.CaCaM_bound = np.mean(cacam_list) if cacam_list else 0.0
        snap.CaMKII_active = np.mean(camkii_active_list) if camkii_active_list else 0.0
        snap.pT286 = np.mean(pt286_list) if pt286_list else 0.0
        snap.GluN2B_bound = np.mean(glun2b_list) if glun2b_list else 0.0
        snap.molecular_memory = np.mean(memory_list) if memory_list else 0.0
        
        snap.eligibility = np.mean(eligibility_list) if eligibility_list else 0.0
        snap.ddsc_triggered = any(ddsc_triggered_list) if ddsc_triggered_list else False
        snap.ddsc_current = np.mean(ddsc_current_list) if ddsc_current_list else 0.0
        snap.ddsc_integrated = np.mean(ddsc_integrated_list) if ddsc_integrated_list else 0.0
        snap.structural_drive = np.mean(structural_drive_list) if structural_drive_list else 0.0
        
        return snap
    
    def _get_projected_strength(self, network: MultiSynapseNetwork) -> float:
        """Get projected final strength"""
        drives = []
        for s in network.synapses:
            if hasattr(s, 'ddsc'):
                drives.append(s.ddsc.get_structural_drive())
        
        if not drives:
            return 1.0
        
        mean_drive = np.mean(drives)
        max_strength_increase = 0.35
        return 1.0 + mean_drive * max_strength_increase
    
    def _run_trial(self, condition: CascadeCondition, trial_id: int) -> CascadeTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = CascadeTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        network = self._create_network(condition)
        
        dt = 0.001  # 1 ms timestep
        current_time = 0.0
        
        # Take initial snapshot
        result.snapshots.append(self._get_snapshot(network, current_time))
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
            current_time += dt
        
        result.snapshots.append(self._get_snapshot(network, current_time))
        
        # === PHASE 2: STIMULATION (theta-burst with reward) ===
        stim_start = current_time
        
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):  # 2ms depolarization
                    network.step(dt, {"voltage": -10e-3, "reward": True})
                    current_time += dt
                for _ in range(8):  # 8ms rest
                    network.step(dt, {"voltage": -70e-3, "reward": True})
                    current_time += dt
            
            # Inter-burst interval (160ms)
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": True})
                current_time += dt
            
            # Snapshot after each burst
            result.snapshots.append(self._get_snapshot(network, current_time))
        
        # === PHASE 3: PLATEAU DELIVERY ===
        # This is the gate-check moment for DDSC
        plateau_time = current_time
        
        for _ in range(50):  # 50ms plateau
            network.step(dt, {
                "voltage": 20e-3,
                "reward": True,
                "plateau_potential": True
            })
            current_time += dt
        
        result.snapshots.append(self._get_snapshot(network, current_time))
        
        # === PHASE 4: OBSERVE CASCADE ===
        observation_start = current_time
        dt_obs = 0.01  # 10ms steps for efficiency
        
        last_sample_time = current_time
        
        while current_time - observation_start < self.observation_s:
            network.step(dt_obs, {
                "voltage": -70e-3,
                "reward": True,
                "plateau_potential": False
            })
            current_time += dt_obs
            
            # Sample at intervals
            if current_time - last_sample_time >= self.sample_interval_s:
                result.snapshots.append(self._get_snapshot(network, current_time))
                last_sample_time = current_time
        
        # === COMPUTE SUMMARY METRICS ===
        
        # Peak values
        result.peak_field_kT = max(s.combined_field_kT for s in result.snapshots)
        result.peak_barrier_reduction_kT = max(s.barrier_reduction_kT for s in result.snapshots)
        result.peak_pT286 = max(s.pT286 for s in result.snapshots)
        result.peak_molecular_memory = max(s.molecular_memory for s in result.snapshots)
        
        # Final state
        final_snap = result.snapshots[-1]
        result.final_eligibility = final_snap.eligibility
        result.ddsc_triggered = any(s.ddsc_triggered for s in result.snapshots)
        result.final_structural_drive = final_snap.structural_drive
        result.projected_strength = self._get_projected_strength(network)
        
        # Timing metrics
        peak_pT286 = result.peak_pT286
        if peak_pT286 > 0.01:
            half_pT286 = peak_pT286 / 2
            for snap in result.snapshots:
                if snap.pT286 >= half_pT286:
                    result.time_to_pT286_half = snap.time_s
                    break
        
        for snap in result.snapshots:
            if snap.molecular_memory > 0.1:
                result.time_to_memory = snap.time_s
                break
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> CascadeResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("QUANTUM → CLASSICAL CASCADE EXPERIMENT")
            print(f"{'='*80}")
            print(f"\nConditions: {[c.name for c in self.conditions]}")
            print(f"Trials per condition: {self.n_trials}")
            print(f"Observation window: {self.observation_s}s")
        
        trials = []
        
        for cond in self.conditions:
            cond.n_synapses = self.n_synapses
            
            if self.verbose:
                print(f"\n  {cond.name} ({cond.description}): ", end='', flush=True)
            
            for trial_id in range(self.n_trials):
                trial_result = self._run_trial(cond, trial_id)
                trials.append(trial_result)
                
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                cond_trials = [t for t in trials if t.condition.name == cond.name]
                mean_field = np.mean([t.peak_field_kT for t in cond_trials])
                mean_barrier = np.mean([t.peak_barrier_reduction_kT for t in cond_trials])
                mean_pT286 = np.mean([t.peak_pT286 for t in cond_trials])
                mean_memory = np.mean([t.peak_molecular_memory for t in cond_trials])
                ddsc_rate = np.mean([1 if t.ddsc_triggered else 0 for t in cond_trials])
                mean_proj = np.mean([t.projected_strength for t in cond_trials])
                
                print(f"\n      Field: {mean_field:.1f} kT | Barrier↓: {mean_barrier:.2f} kT | "
                      f"pT286: {mean_pT286:.3f} | Memory: {mean_memory:.3f}")
                print(f"      DDSC: {ddsc_rate:.0%} | Projected: {mean_proj:.2f}×")
        
        # Build result
        result = CascadeResult(
            conditions=self.conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Check key findings
        self._check_findings(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[CascadeTrialResult]) -> Dict:
        """Compute summary statistics"""
        summary = {}
        
        for cond in self.conditions:
            cond_trials = [t for t in trials if t.condition.name == cond.name]
            
            if cond_trials:
                summary[cond.name] = {
                    'description': cond.description,
                    'em_enabled': cond.em_enabled,
                    'mt_invaded': cond.mt_invaded,
                    'apv_applied': cond.apv_applied,
                    
                    # Quantum layer
                    'peak_field_mean': np.mean([t.peak_field_kT for t in cond_trials]),
                    'peak_field_std': np.std([t.peak_field_kT for t in cond_trials]),
                    
                    # Barrier
                    'barrier_reduction_mean': np.mean([t.peak_barrier_reduction_kT for t in cond_trials]),
                    'barrier_reduction_std': np.std([t.peak_barrier_reduction_kT for t in cond_trials]),
                    
                    # CaMKII
                    'peak_pT286_mean': np.mean([t.peak_pT286 for t in cond_trials]),
                    'peak_pT286_std': np.std([t.peak_pT286 for t in cond_trials]),
                    'peak_memory_mean': np.mean([t.peak_molecular_memory for t in cond_trials]),
                    'peak_memory_std': np.std([t.peak_molecular_memory for t in cond_trials]),
                    'time_to_pT286_half': np.mean([t.time_to_pT286_half for t in cond_trials if t.time_to_pT286_half > 0]) or 0,
                    
                    # Cascade output
                    'ddsc_trigger_rate': np.mean([1 if t.ddsc_triggered else 0 for t in cond_trials]),
                    'structural_drive_mean': np.mean([t.final_structural_drive for t in cond_trials]),
                    'projected_strength_mean': np.mean([t.projected_strength for t in cond_trials]),
                    'projected_strength_std': np.std([t.projected_strength for t in cond_trials]),
                    
                    'n_trials': len(cond_trials)
                }
        
        return summary
    
    def _check_findings(self, result: CascadeResult):
        """Check if key experimental predictions are confirmed"""
        
        full_em = result.summary.get('full_em', {})
        no_em = result.summary.get('no_em', {})
        
        # 1. Field difference: +EM should have much higher field than -EM
        if full_em and no_em:
            result.field_difference_confirmed = (
                full_em.get('peak_field_mean', 0) > no_em.get('peak_field_mean', 0) + 5
            )
        
        # 2. Barrier modulation: +EM should show barrier reduction
        if full_em:
            result.barrier_modulation_confirmed = full_em.get('barrier_reduction_mean', 0) > 0.1
        
        # 3. CaMKII enhancement: +EM should have higher pT286
        if full_em and no_em:
            result.camkii_enhancement_confirmed = (
                full_em.get('peak_pT286_mean', 0) > no_em.get('peak_pT286_mean', 0) * 1.2
            )
        
        # 4. Plasticity gating: +EM should trigger DDSC, -EM should not (or less)
        if full_em and no_em:
            result.plasticity_gating_confirmed = (
                full_em.get('ddsc_trigger_rate', 0) > no_em.get('ddsc_trigger_rate', 1) + 0.3
            )
    
    def print_summary(self, result: CascadeResult):
        """Print formatted summary"""
        print("\n" + "="*90)
        print("QUANTUM → CLASSICAL CASCADE RESULTS")
        print("="*90)
        
        # Header
        print("\n" + "-"*90)
        print(f"{'Condition':<12} {'Field':<12} {'Barrier↓':<12} {'pT286':<12} "
              f"{'Memory':<12} {'DDSC%':<8} {'Proj.Str':<12}")
        print(f"{'':12} {'(kT)':12} {'(kT)':12} {'':12} {'':12} {'':8} {'':12}")
        print("-"*90)
        
        for name in [c.name for c in self.conditions]:
            stats = result.summary.get(name, {})
            field = stats.get('peak_field_mean', 0)
            field_std = stats.get('peak_field_std', 0)
            barrier = stats.get('barrier_reduction_mean', 0)
            pt286 = stats.get('peak_pT286_mean', 0)
            memory = stats.get('peak_memory_mean', 0)
            ddsc_rate = stats.get('ddsc_trigger_rate', 0)
            proj = stats.get('projected_strength_mean', 1)
            proj_std = stats.get('projected_strength_std', 0)
            
            print(f"{name:<12} {field:>5.1f}±{field_std:<4.1f}  {barrier:>5.2f}        "
                  f"{pt286:>6.3f}       {memory:>6.3f}       {ddsc_rate:>5.0%}   "
                  f"{proj:.2f}±{proj_std:.2f}×")
        
        # Key findings
        print("\n" + "="*90)
        print("KEY FINDINGS: QUANTUM → CLASSICAL HANDOFF")
        print("="*90)
        
        full_em = result.summary.get('full_em', {})
        no_em = result.summary.get('no_em', {})
        
        print(f"\n  1. QUANTUM FIELD GENERATION:")
        print(f"     +EM: {full_em.get('peak_field_mean', 0):.1f} kT | "
              f"-EM: {no_em.get('peak_field_mean', 0):.1f} kT")
        if result.field_difference_confirmed:
            print(f"     ✓ Quantum field requires EM coupling")
        else:
            print(f"     ✗ Field difference not significant")
        
        print(f"\n  2. BARRIER MODULATION:")
        print(f"     +EM: {full_em.get('barrier_reduction_mean', 0):.2f} kT reduction | "
              f"-EM: {no_em.get('barrier_reduction_mean', 0):.2f} kT")
        if result.barrier_modulation_confirmed:
            enhancement = np.exp(full_em.get('barrier_reduction_mean', 0))
            print(f"     ✓ Barrier lowered → {enhancement:.1f}× rate enhancement")
        else:
            print(f"     ✗ No significant barrier modulation")
        
        print(f"\n  3. CaMKII CONFORMATIONAL CHANGE:")
        print(f"     +EM pT286: {full_em.get('peak_pT286_mean', 0):.3f} | "
              f"-EM pT286: {no_em.get('peak_pT286_mean', 0):.3f}")
        print(f"     +EM memory: {full_em.get('peak_memory_mean', 0):.3f} | "
              f"-EM memory: {no_em.get('peak_memory_mean', 0):.3f}")
        if result.camkii_enhancement_confirmed:
            print(f"     ✓ Quantum field accelerates CaMKII commitment")
        else:
            print(f"     ✗ CaMKII enhancement not significant")
        
        print(f"\n  4. PLASTICITY GATING:")
        print(f"     +EM DDSC: {full_em.get('ddsc_trigger_rate', 0):.0%} | "
              f"-EM DDSC: {no_em.get('ddsc_trigger_rate', 0):.0%}")
        print(f"     +EM strength: {full_em.get('projected_strength_mean', 1):.2f}× | "
              f"-EM strength: {no_em.get('projected_strength_mean', 1):.2f}×")
        if result.plasticity_gating_confirmed:
            print(f"     ✓ Quantum coherence gates plasticity induction")
        else:
            print(f"     ✗ Plasticity gating not confirmed")
        
        # Overall verdict
        n_confirmed = sum([
            result.field_difference_confirmed,
            result.barrier_modulation_confirmed,
            result.camkii_enhancement_confirmed,
            result.plasticity_gating_confirmed
        ])
        
        print(f"\n  {'='*60}")
        if n_confirmed >= 3:
            print(f"  ✓ CASCADE CONFIRMED: {n_confirmed}/4 key findings validated")
            print(f"    The quantum field modulates protein barriers, accelerating")
            print(f"    CaMKII commitment and gating downstream plasticity.")
        else:
            print(f"  ⚠ CASCADE INCOMPLETE: Only {n_confirmed}/4 findings validated")
            print(f"    Check model parameters or extend observation window.")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: CascadeResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        colors = {
            'full_em': '#2ca02c',
            'no_mt': '#1f77b4', 
            'apv': '#ff7f0e',
            'no_em': '#d62728'
        }
        
        # Get representative trials for time series
        rep_trials = {}
        for cond in self.conditions:
            cond_trials = [t for t in result.trials if t.condition.name == cond.name]
            if cond_trials:
                rep_trials[cond.name] = cond_trials[0]
        
        # === Panel A: Quantum Field over Time ===
        ax1 = axes[0, 0]
        for name, trial in rep_trials.items():
            times = [s.time_s for s in trial.snapshots]
            fields = [s.combined_field_kT for s in trial.snapshots]
            ax1.plot(times, fields, '-', color=colors.get(name, 'gray'), 
                    linewidth=2, label=name, alpha=0.8)
        
        ax1.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Combined Field (kT)', fontsize=11)
        ax1.set_title('A. Quantum Field', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        
        # === Panel B: CaMKII pT286 over Time ===
        ax2 = axes[0, 1]
        for name, trial in rep_trials.items():
            times = [s.time_s for s in trial.snapshots]
            pt286 = [s.pT286 for s in trial.snapshots]
            ax2.plot(times, pt286, '-', color=colors.get(name, 'gray'),
                    linewidth=2, label=name, alpha=0.8)
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('pT286 Level', fontsize=11)
        ax2.set_title('B. CaMKII Autophosphorylation', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        
        # === Panel C: Molecular Memory over Time ===
        ax3 = axes[0, 2]
        for name, trial in rep_trials.items():
            times = [s.time_s for s in trial.snapshots]
            memory = [s.molecular_memory for s in trial.snapshots]
            ax3.plot(times, memory, '-', color=colors.get(name, 'gray'),
                    linewidth=2, label=name, alpha=0.8)
        
        ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='DDSC Threshold')
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Molecular Memory', fontsize=11)
        ax3.set_title('C. Molecular Memory (pT286 × GluN2B)', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        
        # === Panel D: Barrier Reduction Summary ===
        ax4 = axes[1, 0]
        cond_names = [c.name for c in self.conditions if c.name in result.summary]
        barrier_vals = [result.summary[c]['barrier_reduction_mean'] for c in cond_names]
        barrier_stds = [result.summary[c]['barrier_reduction_std'] for c in cond_names]
        bar_colors = [colors.get(c, 'gray') for c in cond_names]
        
        x = np.arange(len(cond_names))
        bars = ax4.bar(x, barrier_vals, yerr=barrier_stds, capsize=4, 
                      color=bar_colors, alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(cond_names, rotation=20, ha='right')
        ax4.set_ylabel('Barrier Reduction (kT)', fontsize=11)
        ax4.set_title('D. Barrier Modulation', fontweight='bold')
        
        # === Panel E: Projected Strength Summary ===
        ax5 = axes[1, 1]
        proj_vals = [result.summary[c]['projected_strength_mean'] for c in cond_names]
        proj_stds = [result.summary[c]['projected_strength_std'] for c in cond_names]
        
        bars = ax5.bar(x, proj_vals, yerr=proj_stds, capsize=4,
                      color=bar_colors, alpha=0.8)
        ax5.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax5.set_xticks(x)
        ax5.set_xticklabels(cond_names, rotation=20, ha='right')
        ax5.set_ylabel('Projected Strength (×)', fontsize=11)
        ax5.set_title('E. Projected Plasticity', fontweight='bold')
        ax5.set_ylim(0.9, max(proj_vals) * 1.1 + 0.1)
        
        # === Panel F: Cascade Summary ===
        ax6 = axes[1, 2]
        
        # Create cascade flow diagram as text
        ax6.axis('off')
        
        full = result.summary.get('full_em', {})
        no = result.summary.get('no_em', {})
        
        cascade_text = f"""
CASCADE COMPARISON

                    +EM (Full)      -EM (Control)
                    ──────────      ─────────────
Quantum Field:      {full.get('peak_field_mean', 0):>6.1f} kT        {no.get('peak_field_mean', 0):>6.1f} kT
        ↓
Barrier Reduction:  {full.get('barrier_reduction_mean', 0):>6.2f} kT        {no.get('barrier_reduction_mean', 0):>6.2f} kT
        ↓
CaMKII pT286:       {full.get('peak_pT286_mean', 0):>6.3f}           {no.get('peak_pT286_mean', 0):>6.3f}
        ↓
Molecular Memory:   {full.get('peak_memory_mean', 0):>6.3f}           {no.get('peak_memory_mean', 0):>6.3f}
        ↓
DDSC Triggered:     {full.get('ddsc_trigger_rate', 0)*100:>5.0f}%            {no.get('ddsc_trigger_rate', 0)*100:>5.0f}%
        ↓
Projected LTP:      {full.get('projected_strength_mean', 1):>6.2f}×          {no.get('projected_strength_mean', 1):>6.2f}×
"""
        
        ax6.text(0.1, 0.95, cascade_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_title('F. Cascade Summary', fontweight='bold')
        
        plt.suptitle('Quantum → Classical Cascade: From Coherence to Plasticity', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'quantum_classical_cascade.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: CascadeResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'quantum_classical_cascade',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            
            # Key findings
            'field_difference_confirmed': result.field_difference_confirmed,
            'barrier_modulation_confirmed': result.barrier_modulation_confirmed,
            'camkii_enhancement_confirmed': result.camkii_enhancement_confirmed,
            'plasticity_gating_confirmed': result.plasticity_gating_confirmed,
            
            'summary': result.summary,
            'conditions': [c.name for c in self.conditions],
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses,
            'observation_s': self.observation_s
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: CascadeResult) -> dict:
        """Get summary as dictionary for master results"""
        full = result.summary.get('full_em', {})
        return {
            'field_difference_confirmed': result.field_difference_confirmed,
            'barrier_modulation_confirmed': result.barrier_modulation_confirmed,
            'camkii_enhancement_confirmed': result.camkii_enhancement_confirmed,
            'plasticity_gating_confirmed': result.plasticity_gating_confirmed,
            'peak_field_kT': full.get('peak_field_mean', 0),
            'barrier_reduction_kT': full.get('barrier_reduction_mean', 0),
            'projected_strength': full.get('projected_strength_mean', 1),
            'runtime_s': result.runtime_s
        }
# Alias for backward compatibility with run_tier3.py
ConsolidationKineticsExperiment = QuantumClassicalCascadeExperiment

if __name__ == "__main__":
    print("="*80)
    print("QUANTUM → CLASSICAL CASCADE EXPERIMENT")
    print("="*80)
    
    print("\nThis experiment demonstrates the complete handoff from quantum")
    print("coherence through protein conformational changes to plasticity.")
    
    print("\nRunning in quick mode...")
    exp = QuantumClassicalCascadeExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    
    print("\nGenerating plots...")
    fig = exp.plot(result)
    plt.show()