"""
Calcium Triphosphate Complex Formation Module
==============================================
Replaces pnc_formation.py with correct chemistry from Habraken et al. 2013
and Agarwal et al. 2023.

KEY CHEMISTRY:
- Ca²⁺ + 3 HPO₄²⁻ ⇌ Ca(HPO₄)₃⁴⁻ (monomer, instant equilibrium)
- 2 Ca(HPO₄)₃⁴⁻ → [Ca(HPO₄)₃]₂⁸⁻ (dimer, slow aggregation)
- Dimers (4 ³¹P nuclei) maintain entanglement for 100+ seconds

Citations:
- Habraken et al. 2013 Nature Commun: PNC structure
- Agarwal et al. 2023 J Phys Chem Lett: Dimers not trimers
- Mancardi et al. 2016 Cryst Growth Des: Formation constants
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional, List
import logging

from model6_parameters import Model6Parameters

logger = logging.getLogger(__name__)


class CalciumTriphosphateFormation:
    """
    Formation of calcium triphosphate complexes via equilibrium
    
    These are the "prenucleation clusters" identified by Habraken et al. (2013).
    They form INSTANTLY via ion association - no nucleation barrier!
    """
    
    def __init__(self, params: Model6Parameters, temperature: float = 310.15):
        self.params = params
        self.T = temperature  # K
        
        # Formation constant for Ca(HPO4)3^4-
        # From Mancardi et al. 2016 and simulation studies
        # K = [Ca(HPO4)3^4-] / ([Ca²⁺] * [HPO4²⁻]³)
        self.K_triphosphate = 1e6  # M⁻² (adjusted for physiological conditions)
        
        logger.info(f"Initialized calcium triphosphate formation (K={self.K_triphosphate:.2e} M⁻²)")
        
    def calculate_monomer_concentration(self, ca_conc: np.ndarray, 
                                       hpo4_conc: np.ndarray) -> np.ndarray:
        """
        Calculate Ca(HPO4)3^4- concentration from equilibrium
        
        This is INSTANTANEOUS - no nucleation, no barrier, pure equilibrium!
        
        Args:
            ca_conc: Free calcium concentration (M)
            hpo4_conc: HPO4²⁻ concentration (M)
            
        Returns:
            Monomer concentration (M)
        """
        # Equilibrium: Ca²⁺ + 3 HPO₄²⁻ ⇌ Ca(HPO₄)₃⁴⁻
        monomer_conc = self.K_triphosphate * ca_conc * (hpo4_conc ** 3)
        
        # Physical limit: can't exceed total calcium
        monomer_conc = np.minimum(monomer_conc, ca_conc)
        
        return monomer_conc


class CalciumTriphosphateDimerization:
    """
    Dimerization of calcium triphosphate complexes
    
    This is the SLOW step and creates the actual quantum qubit!
    2 Ca(HPO4)3^4- → [Ca(HPO4)3]2^8- (dimer with 4 ³¹P)
    
    Agarwal et al. 2023: Dimers maintain entanglement for 100+ seconds
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float):
        self.grid_shape = grid_shape
        self.dx = dx
        
        # Dimer concentration field
        self.dimer_concentration = np.zeros(grid_shape)
        
        # Dimerization rate constant
        # This is diffusion-limited aggregation (Smoluchowski)
        # k = 4π * D * R
        D_monomer = 1e-10  # m²/s (diffusion coefficient)
        R_monomer = 1e-9   # m (molecular radius ~1 nm)
        k_smoluchowski = 4 * np.pi * D_monomer * R_monomer * 1000  # m³/s
        
        # Convert to concentration units: M⁻¹s⁻¹
        N_A = 6.022e23  # mol⁻¹
        self.k_dimerization = k_smoluchowski * N_A  # M⁻¹s⁻¹
        
        # Dissociation rate (much slower than formation)
        self.k_dissociation = 0.01  # s⁻¹ (dimers are stable)
        
        logger.info(f"Initialized dimerization (k_dimer={self.k_dimerization:.2e} M⁻¹s⁻¹)")
        
    def update_dimerization(self, dt: float, monomer_conc: np.ndarray,
                           template_enhancement: np.ndarray) -> None:
        """
        Update dimer concentration via second-order kinetics
        
        d[Dimer]/dt = k_dimer * [Monomer]² - k_dissoc * [Dimer]
        
        Templates enhance dimerization by 100x (surface aggregation)
        
        Args:
            dt: Time step (s)
            monomer_conc: Ca(HPO4)3^4- concentration (M)
            template_enhancement: Enhancement factor at templates (0-100)
        """
        # Effective rate with template enhancement
        k_eff = self.k_dimerization * (1.0 + template_enhancement)
        
        # Formation: second-order in monomer
        formation_rate = k_eff * monomer_conc ** 2
        
        # Dissociation: first-order in dimer
        dissociation_rate = self.k_dissociation * self.dimer_concentration
        
        # Update
        d_dimer_dt = formation_rate - dissociation_rate
        self.dimer_concentration += d_dimer_dt * dt
        
        # Physical limits
        self.dimer_concentration = np.maximum(self.dimer_concentration, 0)
        
        # Can't exceed monomer supply (stoichiometry)
        max_dimer = monomer_conc / 2.0  # Need 2 monomers per dimer
        self.dimer_concentration = np.minimum(self.dimer_concentration, max_dimer)


class TemplateEffects:
    """
    Template (protein) effects on calcium triphosphate aggregation
    
    Templates don't PULL complexes in - they provide SURFACES for aggregation!
    Tao et al. 2010: "PNCs aggregate close to the surface"
    """
    
    def __init__(self, grid_shape: Tuple[int, int],
                 template_positions: Optional[List[Tuple[int, int]]] = None):
        self.grid_shape = grid_shape
        
        # Template field (binary: 0 or 1)
        self.template_field = np.zeros(grid_shape)
        
        if template_positions:
            for x, y in template_positions:
                self.template_field[x, y] = 1.0
            logger.info(f"Initialized {len(template_positions)} template sites")
        
        # Enhancement factor field (0-100x)
        self.enhancement_field = np.zeros(grid_shape)
        
    def calculate_surface_enhancement(self) -> np.ndarray:
        """
        Calculate aggregation enhancement near template surfaces
        
        Enhancement decays with distance from template:
        - At template: 100x faster
        - 1 nm away: ~50x faster
        - >5 nm away: ~1x (no effect)
        
        Returns:
            Enhancement factor (0-100)
        """
        if np.sum(self.template_field) == 0:
            return np.zeros(self.grid_shape)
        
        # Distance from nearest template
        distance = ndimage.distance_transform_edt(1 - self.template_field)
        
        # Convert to nanometers (assuming dx is in meters)
        # For 100x100 grid over 400 nm → dx = 4 nm
        distance_nm = distance * 4.0  # Approximate
        
        # Enhancement: exponential decay with distance
        # 100x at surface, 1x at 10 nm
        enhancement = 100 * np.exp(-distance_nm / 3.0)
        
        return enhancement


class CalciumTriphosphateSystem:
    """
    Complete system for calcium triphosphate complex formation and dimerization
    
    This replaces PNCFormationSystem with correct chemistry.
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float,
                 params: Model6Parameters,
                 template_positions: Optional[List[Tuple[int, int]]] = None):
        
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params
        
        # Components
        self.formation = CalciumTriphosphateFormation(params, params.environment.T)
        self.dimerization = CalciumTriphosphateDimerization(grid_shape, dx)
        self.templates = TemplateEffects(grid_shape, template_positions)
        
        # State fields
        self.monomer_concentration = np.zeros(grid_shape)  # Ca(HPO4)3^4-
        self.dimer_concentration = np.zeros(grid_shape)     # [Ca(HPO4)3]2^8-
        
        # Precompute template enhancement
        self.template_enhancement = self.templates.calculate_surface_enhancement()
        
        logger.info("Initialized calcium triphosphate system")
        
    def step(self, dt: float, ca_conc: np.ndarray, po4_conc: np.ndarray) -> None:
        """
        Update calcium triphosphate system
        
        Step 1: Calculate monomer concentration (instant equilibrium)
        Step 2: Update dimer concentration (slow aggregation)
        
        Args:
            dt: Time step (s)
            ca_conc: Free calcium concentration (M)
            po4_conc: Total phosphate concentration (M)
        """
        # Step 1: Monomers form instantly via equilibrium
        # Ca²⁺ + 3 HPO₄²⁻ ⇌ Ca(HPO₄)₃⁴⁻
        # Note: We assume all phosphate is HPO4²⁻ at pH 7.3
        self.monomer_concentration = self.formation.calculate_monomer_concentration(
            ca_conc, po4_conc
        )
        
        # Step 2: Dimers form slowly via aggregation
        # 2 Ca(HPO₄)₃⁴⁻ → [Ca(HPO₄)₃]₂⁸⁻
        self.dimerization.update_dimerization(
            dt,
            self.monomer_concentration,
            self.template_enhancement
        )
        
        # Update dimer field reference
        self.dimer_concentration = self.dimerization.dimer_concentration
        
    def get_monomer_concentration(self) -> np.ndarray:
        """Get Ca(HPO4)3^4- concentration (M)"""
        return self.monomer_concentration
    
    def get_dimer_concentration(self) -> np.ndarray:
        """Get [Ca(HPO4)3]2^8- concentration (M) - the quantum qubit!"""
        return self.dimer_concentration
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return metrics for validation
        
        Returns:
            Dictionary with concentrations in nM
        """
        return {
            'monomer_mean_nM': float(np.mean(self.monomer_concentration) * 1e9),
            'monomer_peak_nM': float(np.max(self.monomer_concentration) * 1e9),
            'dimer_mean_nM': float(np.mean(self.dimer_concentration) * 1e9),
            'dimer_peak_nM': float(np.max(self.dimer_concentration) * 1e9),
            'dimer_at_templates_nM': float(
                np.mean(self.dimer_concentration[self.templates.template_field > 0.5]) * 1e9
                if np.sum(self.templates.template_field) > 0 else 0
            ),
        }


# ===========================================================================
# VALIDATION TEST
# ===========================================================================

if __name__ == "__main__":
    print("="*70)
    print("CALCIUM TRIPHOSPHATE SYSTEM VALIDATION")
    print("="*70)
    
    from model6_parameters import Model6Parameters
    
    params = Model6Parameters()
    grid_shape = (50, 50)
    dx = 4e-9  # 4 nm
    
    # Template at center
    center = grid_shape[0] // 2
    template_positions = [(center, center)]
    
    # Create system
    system = CalciumTriphosphateSystem(grid_shape, dx, params, template_positions)
    
    print(f"\nSetup: {grid_shape[0]}x{grid_shape[1]} grid, {len(template_positions)} templates")
    
    # Test 1: Resting conditions (low Ca)
    print("\n### Test 1: Resting (Ca=100 nM, PO4=1 mM) ###")
    ca_rest = np.ones(grid_shape) * 100e-9  # 100 nM
    po4_rest = np.ones(grid_shape) * 1e-3   # 1 mM
    
    system.step(0.001, ca_rest, po4_rest)
    metrics = system.get_experimental_metrics()
    
    print(f"  Monomers: {metrics['monomer_peak_nM']:.2f} nM")
    print(f"  Dimers: {metrics['dimer_peak_nM']:.4f} nM")
    print(f"  Expected: Very low (< 1 nM)")
    
    # Test 2: Calcium spike
    print("\n### Test 2: Calcium Spike (Ca=12 μM, PO4=1 mM) ###")
    ca_spike = np.ones(grid_shape) * 100e-9
    ca_spike[20:30, 20:30] = 12e-6  # 12 μM in center
    
    # Run for 100 ms
    for i in range(100):
        system.step(0.001, ca_spike, po4_rest)
        
        if i % 25 == 0:
            m = system.get_experimental_metrics()
            print(f"  t={i}ms: Monomers={m['monomer_peak_nM']:.1f} nM, "
                  f"Dimers={m['dimer_peak_nM']:.2f} nM")
    
    final_metrics = system.get_experimental_metrics()
    print(f"\n  Final dimers: {final_metrics['dimer_peak_nM']:.2f} nM")
    print(f"  At templates: {final_metrics['dimer_at_templates_nM']:.2f} nM")
    
    if final_metrics['dimer_peak_nM'] > 1.0:
        print("\n  ✓ Dimers formed successfully!")
    else:
        print("\n  ⚠ Dimer formation may be too slow")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("  Monomers (Ca(HPO4)3^4-) form INSTANTLY from equilibrium")
    print("  Dimers ([Ca(HPO4)3]2^8-) form SLOWLY via aggregation")
    print("  Dimers are the quantum qubits (4 ³¹P, long coherence)")
    print("="*70)