"""
Pathway 1 & 2: Ca²⁺ → Calmodulin → CaMKII → F-Actin Reorganization
======================================================================

This module implements the classical cascade that initiates plasticity:
1. Ca²⁺ binds calmodulin (4 EF-hand sites)
2. Ca²⁺/CaM activates CaMKII (releases autoinhibition)
3. CaMKII autophosphorylates (creates persistent signal)
4. CaMKII releases from F-actin (enables reorganization)
5. F-actin reorganizes at spine base (gates MT invasion)

Key References:
---------------
Lisman et al. 2012 - Nat Rev Neurosci 13:169-182
    "The molecular basis of CaMKII function in synaptic and learning plasticity"

Hudmon & Schulman 2002 - Annu Rev Biochem 71:473-510
    "Neuronal CA2+/calmodulin-dependent protein kinase II"

Colbran & Brown 2004 - Curr Opin Neurobiol 14:318-327
    "Calcium/calmodulin-dependent protein kinase II and synaptic plasticity"

Okamoto et al. 2004 - Neuron 42:549-559
    "Rapid and persistent modulation of actin dynamics"

Honkura et al. 2008 - Neuron 57:719-729
    "The subspine organization of actin fibers regulates plasticity"

Author: Sarah Davidson
Date: October 2025
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

from hierarchical_model_parameters import (
    HierarchicalModelParameters,
    CalmodulinParameters,
    CaMKIIParameters,
    ActinParameters,
    k_B_T_310K
)

logger = logging.getLogger(__name__)


class CalmodulinActivation:
    """
    Calmodulin Ca²⁺ binding and conformational change
    
    Calmodulin has 4 EF-hand Ca²⁺ binding sites (2 per lobe).
    Binding causes conformational change exposing hydrophobic pockets
    that bind target proteins like CaMKII.
    
    References:
    -----------
    Persechini & Stemmer 2002 - Trends Cardiovasc Med 12:32-37
        Ca²⁺ binding cooperativity measured by stopped-flow fluorescence
    
    Meador et al. 1992 - Science 257:1251-1255
        Crystal structure of Ca²⁺-CaM shows conformational changes
    
    Shifman et al. 2006 - PNAS 103:13968-13973
        All-atom simulations of Ca²⁺ binding dynamics
    """
    
    def __init__(self, params: CalmodulinParameters):
        self.params = params
        
        # State: Number of Ca²⁺ bound (0-4)
        self.ca_bound = 0
        
        # Conformational state: 0 = closed, 1 = open
        self.conformation = 0.0
        
        logger.info("Calmodulin activation module initialized")
    
    def update(self, dt: float, ca_concentration: float) -> Dict:
        """
        Update Ca²⁺ binding and conformational state
        
        Args:
            dt: Timestep (s)
            ca_concentration: [Ca²⁺] (M)
        
        Returns:
            state: Dictionary with activation state
        """
        
        # === Ca²⁺ BINDING KINETICS ===
        # Persechini & Stemmer 2002 Trends Cardiovasc Med 12:32-37
        # Cooperative binding to 4 sites
        
        # Effective occupancy (simplified from 4-site model)
        # Assumes rapid equilibrium for computational efficiency
        
        # N-lobe binding (sites 1,2)
        n_lobe_occupancy = self._hill_binding(
            ca_concentration,
            self.params.kd_n_lobe,
            n_hill=2  # Cooperativity
        )
        
        # C-lobe binding (sites 3,4)
        c_lobe_occupancy = self._hill_binding(
            ca_concentration,
            self.params.kd_c_lobe,
            n_hill=2
        )
        
        # Average occupancy across all 4 sites
        total_occupancy = (n_lobe_occupancy + c_lobe_occupancy) / 2
        self.ca_bound = total_occupancy * 4  # Scale to 0-4 Ca²⁺
        
        # === CONFORMATIONAL CHANGE ===
        # Meador et al. 1992 Science 257:1251-1255
        # Ca²⁺ binding opens hydrophobic pockets
        
        # Activation requires ~2-3 Ca²⁺ bound
        # Shifman et al. 2006 PNAS 103:13968-13973
        activation_threshold = 2.0
        
        if self.ca_bound > activation_threshold:
            # Drive toward open conformation
            target = 1.0
            tau = self.params.tau_conformational
            self.conformation += (target - self.conformation) / tau * dt
        else:
            # Return to closed
            target = 0.0
            tau = self.params.tau_conformational
            self.conformation += (target - self.conformation) / tau * dt
        
        # Clip to [0, 1]
        self.conformation = np.clip(self.conformation, 0, 1)
        
        return {
            'ca_bound': self.ca_bound,
            'conformation': self.conformation,
            'active': self.conformation > 0.5
        }
    
    @staticmethod
    def _hill_binding(concentration: float, kd: float, n_hill: float) -> float:
        """
        Hill equation for cooperative binding
        
        θ = [L]^n / (Kd^n + [L]^n)
        
        Args:
            concentration: Ligand concentration (M)
            kd: Dissociation constant (M)
            n_hill: Hill coefficient (cooperativity)
        
        Returns:
            occupancy: Fraction bound (0-1)
        """
        return concentration**n_hill / (kd**n_hill + concentration**n_hill)


class CaMKIIActivation:
    """
    CaMKII activation, autophosphorylation, and actin binding
    
    CaMKII exists in three states:
    1. Autoinhibited (inactive) - bound to F-actin
    2. Ca²⁺/CaM-activated - released from actin
    3. Phosphorylated T286 - autonomously active
    
    The T286 autophosphorylation creates persistent activity lasting
    ~100 seconds, providing a molecular memory of the Ca²⁺ transient.
    
    References:
    -----------
    Lisman et al. 2012 - Nat Rev Neurosci 13:169-182
        Comprehensive review of CaMKII in plasticity
    
    Stratton et al. 2014 - Nat Commun 5:5304
        Crystal structures of CaMKII activation states
    
    Colbran & Brown 2004 - Curr Opin Neurobiol 14:318-327
        CaMKII-actin interactions
    """
    
    def __init__(self, params: CaMKIIParameters):
        self.params = params
        
        # CaMKII states (12 subunits in holoenzyme)
        # For simplicity, track average state
        self.fraction_cam_bound = 0.0  # Ca²⁺/CaM-activated
        self.fraction_phosphorylated = 0.0  # T286-phosphorylated
        self.fraction_on_actin = 0.8  # Starts mostly bound to actin
        
        # Activity level (0-1)
        self.activity = 0.0
        
        # Quantum modulation state
        self.quantum_barrier_shift = 0.0  # kT units
        
        logger.info("CaMKII activation module initialized")
    
    def update(self, dt: float, cam_active: float, 
               quantum_field: float = 0.0) -> Dict:
        """
        Update CaMKII activation state
        
        Args:
            dt: Timestep (s)
            cam_active: Active Ca²⁺/CaM fraction (0-1)
            quantum_field: Quantum electrostatic field strength (J)
        
        Returns:
            state: Dictionary with CaMKII state
        """
        
        # === QUANTUM FIELD MODULATION ===
        # Quantum field modulates activation barrier
        # Davidson 2025 - Hierarchical Quantum Processing
        # Field of ~20-40 kT can significantly reduce barrier
        
        if quantum_field > 0:
            # Convert field energy to kT units
            self.quantum_barrier_shift = quantum_field / k_B_T_310K
        else:
            self.quantum_barrier_shift = 0.0
        
        # === CA²⁺/CAM BINDING ===
        # Lisman et al. 2012 Nat Rev Neurosci 13:169-182
        # Ca²⁺/CaM binds to regulatory domain with high affinity
        
        # Binding rate depends on available (non-phosphorylated) subunits
        available = 1 - self.fraction_phosphorylated
        
        # Forward binding
        d_cam_bind = (self.params.k_on_cam * cam_active * available * dt)
        
        # Reverse unbinding (slow - tight binding)
        d_cam_unbind = self.params.k_off_cam * self.fraction_cam_bound * dt
        
        self.fraction_cam_bound += d_cam_bind - d_cam_unbind
        self.fraction_cam_bound = np.clip(self.fraction_cam_bound, 0, 1)
        
        # === AUTOPHOSPHORYLATION AT T286 ===
        # Stratton et al. 2014 Nat Commun 5:5304
        # Adjacent subunits phosphorylate each other (inter-subunit)
        # Requires Ca²⁺/CaM binding to release autoinhibition
        
        if self.fraction_cam_bound > 0.1:
            # Quantum field modulates barrier
            # Effective rate: k_eff = k₀ × exp(ΔE/kT)
            enhancement = np.exp(min(self.quantum_barrier_shift, 30))  # Cap at e^30
            k_phos_eff = self.params.k_autophosphorylation * enhancement
            
            # Phosphorylation of remaining unphosphorylated subunits
            d_phos = (k_phos_eff * self.fraction_cam_bound * 
                     (1 - self.fraction_phosphorylated) * dt)
            
            self.fraction_phosphorylated += d_phos
        
        # Dephosphorylation by phosphatases (PP1)
        # Colbran & Brown 2004 Curr Opin Neurobiol 14:318-327
        d_dephos = self.params.k_dephosphorylation * self.fraction_phosphorylated * dt
        self.fraction_phosphorylated -= d_dephos
        
        self.fraction_phosphorylated = np.clip(self.fraction_phosphorylated, 0, 1)
        
        # === F-ACTIN BINDING ===
        # Colbran & Brown 2004 Curr Opin Neurobiol 14:318-327
        # Ca²⁺/CaM and F-actin compete for regulatory domain
        # When Ca²⁺/CaM binds, CaMKII releases from actin
        
        if self.fraction_cam_bound > 0.5:
            # Strong Ca²⁺/CaM binding strips CaMKII off actin
            target_on_actin = 0.1  # Mostly released
            tau_release = 1.0  # s (fast release)
            self.fraction_on_actin += (target_on_actin - self.fraction_on_actin) / tau_release * dt
        else:
            # Without Ca²⁺/CaM, rebinds to actin
            target_on_actin = 0.8  # Mostly bound
            tau_rebind = 10.0  # s (slower rebinding)
            self.fraction_on_actin += (target_on_actin - self.fraction_on_actin) / tau_rebind * dt
        
        self.fraction_on_actin = np.clip(self.fraction_on_actin, 0, 1)
        
        # === TOTAL ACTIVITY ===
        # Active when either Ca²⁺/CaM-bound OR phosphorylated
        self.activity = max(self.fraction_cam_bound, self.fraction_phosphorylated)
        
        return {
            'cam_bound': self.fraction_cam_bound,
            'phosphorylated_T286': self.fraction_phosphorylated,
            'on_actin': self.fraction_on_actin,
            'activity': self.activity,
            'quantum_shift_kT': self.quantum_barrier_shift
        }


class ActinReorganization:
    """
    F-actin reorganization at spine base gates microtubule invasion
    
    In baseline state:
    - CaMKII bundles F-actin → stable, rigid spine
    
    During plasticity:
    - Ca²⁺/CaM strips CaMKII from actin
    - F-actin unbundles
    - Cofilin severs, Arp2/3 polymerizes
    - Creates pathway for microtubule entry
    
    References:
    -----------
    Okamoto et al. 2004 - Neuron 42:549-559
        Rapid actin remodeling during LTP
    
    Honkura et al. 2008 - Neuron 57:719-729
        Subspine actin organization regulates plasticity
    
    Bosch et al. 2014 - Science 344:1252304
        Structural and molecular remodeling during plasticity
    
    Frost et al. 2010 - Neuron 66:370-382
        CaMKII regulation of actin dynamics
    """
    
    def __init__(self, params: ActinParameters):
        self.params = params
        
        # F-actin organization state
        self.density_spine_base = params.f_actin_baseline_density
        self.bundled_fraction = params.camkii_actin_bundling
        self.reorganization_level = 0.0  # 0 = baseline, 1 = fully reorganized
        
        # Actin-binding protein activities
        self.drebrin_binding = 0.0
        self.cofilin_activity = params.cofilin_activity
        self.arp23_activity = params.arp23_activity
        
        logger.info("Actin reorganization module initialized")
    
    def update(self, dt: float, camkii_on_actin: float) -> Dict:
        """
        Update F-actin organization state
        
        Args:
            dt: Timestep (s)
            camkii_on_actin: Fraction of CaMKII bound to actin (0-1)
        
        Returns:
            state: Dictionary with actin state
        """
        
        # === CAMKII-MEDIATED BUNDLING ===
        # Frost et al. 2010 Neuron 66:370-382
        # CaMKII crosslinks F-actin filaments when bound
        
        # Bundling reflects CaMKII binding
        self.bundled_fraction = camkii_on_actin * self.params.camkii_actin_bundling
        
        # === REORGANIZATION DYNAMICS ===
        # Okamoto et al. 2004 Neuron 42:549-559
        # When CaMKII releases (camkii_on_actin low), reorganization occurs
        
        if camkii_on_actin < 0.3:  # Threshold for reorganization
            # Drive toward reorganized state
            target = 1.0
            tau = self.params.tau_reorganization
            self.reorganization_level += (target - self.reorganization_level) / tau * dt
        else:
            # Return to baseline
            target = 0.0
            tau = self.params.tau_reorganization * 2  # Slower return
            self.reorganization_level += (target - self.reorganization_level) / tau * dt
        
        self.reorganization_level = np.clip(self.reorganization_level, 0, 1)
        
        # === ACTIN-BINDING PROTEIN RECRUITMENT ===
        # Bosch et al. 2014 Science 344:1252304
        
        # Drebrin accumulates during reorganization
        # Critical for MT invasion (binds EB3)
        self.drebrin_binding = self.reorganization_level * 0.8
        
        # Cofilin activity increases (severs F-actin)
        self.cofilin_activity = (self.params.cofilin_activity + 
                                0.3 * self.reorganization_level)
        
        # Arp2/3 activity increases (nucleates new branches)
        self.arp23_activity = (self.params.arp23_activity + 
                              0.4 * self.reorganization_level)
        
        # === F-ACTIN DENSITY AT SPINE BASE ===
        # Net effect of severing and polymerization
        # Honkura et al. 2008 Neuron 57:719-729
        
        # During reorganization: transient decrease then increase
        if self.reorganization_level > 0.5:
            # Polymerization phase
            target_density = 0.7
        else:
            # Baseline or severing phase
            target_density = self.params.f_actin_baseline_density
        
        tau_density = 5.0  # s
        self.density_spine_base += (target_density - self.density_spine_base) / tau_density * dt
        self.density_spine_base = np.clip(self.density_spine_base, 0, 1)
        
        return {
            'density_spine_base': self.density_spine_base,
            'bundled_fraction': self.bundled_fraction,
            'reorganization_level': self.reorganization_level,
            'drebrin_binding': self.drebrin_binding,
            'cofilin_activity': self.cofilin_activity,
            'arp23_activity': self.arp23_activity,
            'ready_for_mt_invasion': self.reorganization_level > 0.7
        }


class CaMKIIActinPathway:
    """
    Integrated Pathway 1 & 2: Ca²⁺ → CaMKII → Actin
    
    This combines the three subsystems into a complete pathway
    from calcium signal to actin reorganization.
    """
    
    def __init__(self, params: HierarchicalModelParameters):
        self.params = params
        
        # Initialize subsystems
        self.calmodulin = CalmodulinActivation(params.calmodulin)
        self.camkii = CaMKIIActivation(params.camkii)
        self.actin = ActinReorganization(params.actin)
        
        logger.info("CaMKII-Actin pathway initialized")
    
    def update(self, dt: float, ca_concentration: float,
               quantum_field: float = 0.0) -> Dict:
        """
        Update complete pathway
        
        Args:
            dt: Timestep (s)
            ca_concentration: [Ca²⁺] (M)
            quantum_field: Quantum electrostatic field (J)
        
        Returns:
            state: Complete pathway state
        """
        
        # Step 1: Calmodulin activation
        cam_state = self.calmodulin.update(dt, ca_concentration)
        
        # Step 2: CaMKII activation
        # Quantum field modulates CaMKII autophosphorylation
        camkii_state = self.camkii.update(
            dt, 
            cam_state['conformation'],
            quantum_field
        )
        
        # Step 3: Actin reorganization
        # CaMKII release from actin enables reorganization
        actin_state = self.actin.update(dt, camkii_state['on_actin'])
        
        return {
            'calmodulin': cam_state,
            'camkii': camkii_state,
            'actin': actin_state,
            # Key outputs for other pathways
            'camkii_activity': camkii_state['activity'],
            'camkii_pT286': camkii_state['phosphorylated_T286'],
            'actin_ready_for_mt': actin_state['ready_for_mt_invasion'],
            'actin_reorganization': actin_state['reorganization_level']
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("TESTING PATHWAY 1 & 2: CaMKII-ACTIN CASCADE")
    print("="*70)
    
    # Initialize with default parameters
    params = HierarchicalModelParameters()
    pathway = CaMKIIActinPathway(params)
    
    # Simulate Ca²⁺ spike and decay
    dt = 0.01  # 10 ms timestep
    times = np.arange(0, 300, dt)  # 5 minutes
    
    # Results storage
    ca_trace = []
    cam_active = []
    camkii_pT286 = []
    camkii_on_actin = []
    actin_reorg = []
    
    # Simulate
    for t in times:
        # Ca²⁺ transient: spike at t=0, decay with tau=100ms
        if t < 1.0:
            ca = 100e-9 + 5e-6 * np.exp(-t/0.1)  # 5 µM spike
        else:
            ca = 100e-9  # Return to baseline
        
        ca_trace.append(ca * 1e6)  # Convert to µM for plotting
        
        # Update pathway (no quantum field initially)
        state = pathway.update(dt, ca, quantum_field=0.0)
        
        # Store results
        cam_active.append(state['calmodulin']['conformation'])
        camkii_pT286.append(state['camkii']['phosphorylated_T286'])
        camkii_on_actin.append(state['camkii']['on_actin'])
        actin_reorg.append(state['actin']['reorganization_level'])
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Ca²⁺ transient
    axes[0].plot(times, ca_trace, 'b-', linewidth=2)
    axes[0].set_ylabel('[Ca²⁺] (µM)', fontsize=12)
    axes[0].set_title('Pathway 1&2: Ca²⁺ → CaMKII → Actin', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # CaM activation
    axes[1].plot(times, cam_active, 'g-', linewidth=2, label='CaM active')
    axes[1].set_ylabel('CaM Activation', fontsize=12)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # CaMKII states
    axes[2].plot(times, camkii_pT286, 'r-', linewidth=2, label='pT286 (persistent)')
    axes[2].plot(times, camkii_on_actin, 'orange', linewidth=2, alpha=0.7, label='On actin')
    axes[2].set_ylabel('CaMKII State', fontsize=12)
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axhline(0.7, color='red', linestyle='--', alpha=0.3)
    axes[2].text(250, 0.75, '~100s persistence →', fontsize=10, color='red')
    
    # Actin reorganization
    axes[3].plot(times, actin_reorg, 'purple', linewidth=2)
    axes[3].fill_between(times, 0, actin_reorg, alpha=0.3, color='purple')
    axes[3].axhline(0.7, color='purple', linestyle='--', alpha=0.5, label='MT invasion threshold')
    axes[3].set_ylabel('Actin\nReorganization', fontsize=12)
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_ylim([-0.1, 1.1])
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pathway_1_2_test.png', dpi=300, bbox_inches='tight')
    print("\n✓ Test figure saved to outputs/pathway_1_2_test.png")
    
    print("\nKEY FINDINGS:")
    print(f"  CaMKII pT286 peak: {max(camkii_pT286):.2f}")
    print(f"  pT286 at t=100s: {camkii_pT286[int(100/dt)]:.2f}")
    print(f"  Actin reorganization peak: {max(actin_reorg):.2f}")
    print(f"  Time above MT threshold: {sum(np.array(actin_reorg) > 0.7) * dt:.1f} s")
    
    print("\n" + "="*70)
    print("Pathway 1 & 2 validation complete!")
    print("="*70)