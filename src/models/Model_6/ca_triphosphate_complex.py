"""
Calcium Phosphate Ion Pair Formation - CORRECTED CHEMISTRY
===========================================================

CRITICAL CORRECTION:
Previous model used Ca(HPO₄)₃⁴⁻ which is NOT the dominant species at pH 7.3!

CORRECT CHEMISTRY (McDonogh et al. 2024, Moreno & Brown 1966):
At pH 6-10, the dominant species is CaHPO₄⁰ (1:1 ion pair)

Ca²⁺ + HPO₄²⁻ ⇌ [CaHPO₄]⁰

Step 1: Ca²⁺ + HPO₄²⁻ ⇌ CaHPO₄⁰ (K=588 M⁻¹, instant)
Step 2: 6 × CaHPO₄ → Ca₆(PO₄)₄ (k=1e6 M⁻¹s⁻¹, slow)
         ↑ This is the "dimer" - 4 ³¹P nuclei!
         
Alternative: 9 × CaHPO₄ → Ca₉(PO₄)₆ ("trimer", 6 ³¹P nuclei)
             ↑ Classical Posner, forms under different conditions

Then these aggregate to form the clusters Habraken described:
n × [CaHPO₄]⁰ → [Ca_n(HPO₄)_n] → Ca₆(PO₄)₄ (Agarwal's dimer)

Key Citations:
- McDonogh et al. 2024 Nat Commun: "CaHPO₄ dominates pH 6-10"
- Moreno & Brown 1966: K = 588 M⁻¹ at 37.5°C for [CaHPO₄]⁰
- Habraken et al. 2013: PNCs are aggregates of CaHPO₄ units
- Agarwal et al. 2023: Ca₆(PO₄)₄ dimers maintain coherence
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional, List
import logging

from model6_parameters import Model6Parameters

logger = logging.getLogger(__name__)


class CalciumPhosphateIonPair:
    """
    Formation of CaHPO₄⁰ ion pairs via 1:1 association
    
    This is the CORRECT dominant species at physiological pH!
    
    McDonogh et al. 2024 Nat Commun 15:3359:
    "CaHPO₄⁰ dominates the aqueous CaP speciation between pH ~6–10"
    "ΔG° = -14.0 kJ/mol for [CaHPO₄]⁰"
    
    Moreno & Brown 1966 J Res Natl Bur Stand:
    "Stability constant for [CaHPO₄]⁰ = 5.88 × 10² L/mol at 37.5°C"
    """
    
    def __init__(self, params: Model6Parameters, temperature: float = 310.15):
        self.params = params
        self.T = temperature  # K
        
        # Formation constant for CaHPO₄⁰ ion pair
        # Moreno & Brown 1966: K = 588 M⁻¹ at 37.5°C
        # McDonogh et al. 2024: ΔG° = -14.0 kJ/mol → K ≈ 300 M⁻¹
        # Using experimental value at body temperature
        self.K_CaHPO4 = 588.0  # M⁻¹
        
        logger.info(f"Initialized CaHPO₄⁰ ion pair formation (K={self.K_CaHPO4:.0f} M⁻¹)")
        
    def calculate_ion_pair_concentration(self, ca_conc: np.ndarray, 
                                        hpo4_conc: np.ndarray) -> np.ndarray:
        """
        Calculate [CaHPO₄]⁰ concentration from 1:1 binding
        
        This is INSTANTANEOUS equilibrium - no barrier!
        
        Reaction: Ca²⁺ + HPO₄²⁻ ⇌ [CaHPO₄]⁰
        K = [CaHPO₄] / ([Ca²⁺][HPO₄²⁻])
        
        Args:
            ca_conc: Free calcium concentration (M)
            hpo4_conc: HPO₄²⁻ concentration (M)
            
        Returns:
            CaHPO₄⁰ concentration (M)
        """
        # Simple 1:1 binding equilibrium
        # [CaHPO₄] = K × [Ca²⁺] × [HPO₄²⁻]
        ion_pair_conc = self.K_CaHPO4 * ca_conc * hpo4_conc
        
        # Physical limits
        ion_pair_conc = np.minimum(ion_pair_conc, ca_conc)  # Can't exceed Ca
        ion_pair_conc = np.minimum(ion_pair_conc, hpo4_conc)  # Can't exceed PO4
        
        return ion_pair_conc


class CalciumPhosphateDimerization:
    """
    Aggregation of CaHPO₄ units to form Ca₆(PO₄)₄ dimers
    
    This is Agarwal's quantum qubit!
    
    Habraken et al. 2013 Nat Commun 4:1507:
    "PNCs form via aggregation of ion pairs"
    
    Agarwal et al. 2023 J Phys Chem Lett 14:2518:
    "Ca₆(PO₄)₄ dimers preserve entangled nuclear spins for hundreds of seconds"
    "4 ³¹P nuclei in dimer structure"
    
    Formation pathway:
    6 × CaHPO₄⁰ → Ca₆(HPO₄)₆ → Ca₆(PO₄)₄ + 2H₃PO₄ (deprotonation)
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float):
        self.grid_shape = grid_shape
        self.dx = dx
        
        # Dimer concentration field
        self.dimer_concentration = np.zeros(grid_shape)
        
        # Aggregation is diffusion-limited
        # BUT: 6 units must come together (higher order than 2)
        # Use effective rate for 6-body collision
        
        # Base rate from Smoluchowski for pairwise collision
        D_ion_pair = 5e-10  # m²/s (small neutral complex, faster than charged)
        R_ion_pair = 0.5e-9  # m (compact neutral ion pair)
        k_pairwise = 4 * np.pi * D_ion_pair * R_ion_pair  # m³/s
        
        # Convert to M⁻¹s⁻¹
        N_A = 6.022e23
        k_pairwise_M = k_pairwise * N_A
        
        # For 6-body aggregation, effective rate is much lower
        # Estimated from nucleation theory: k_eff ≈ k₂ × ([CaHPO₄]/C₀)⁴
        # where C₀ is a reference concentration (1 mM)
        # This gives concentration-dependent rate
        
        # Use 2nd order rate with tuning factor for biological timescale
        # Tuned to give ~1-10 nM in 100-200ms with μM ion pairs
        self.k_base = 1e6  # M⁻¹s⁻¹ (tuned for biological timescale)
        
        
        # Dissociation (very slow - dimers are stable)
        self.k_dissociation = 0.001  # s⁻¹
        
        logger.info(f"Initialized Ca₆(PO₄)₄ dimer aggregation (k_agg={self.k_base:.2e} M⁻¹s⁻¹)")
        
    def update_dimerization(self, dt: float, ion_pair_conc: np.ndarray,
                        template_enhancement: np.ndarray,
                        ca_conc: np.ndarray) -> None:
        """
        Update dimer concentration via STEPWISE aggregation
    
        Growth is 2nd order (add one unit at a time), NOT 6th order!
    
        CaHPO₄ + CaHPO₄ → (CaHPO₄)₂
        (CaHPO₄)₂ + CaHPO₄ → (CaHPO₄)₃
        ... → Ca₆(PO₄)₄ dimer
    
        Effective rate for reaching hexamer: k_eff × [CaHPO₄]²
        """
        # Effective rate with template enhancement
        # Templates provide 2D surface → 100-1000x enhancement
        k_eff = self.k_base * template_enhancement
    
        # Formation: SECOND ORDER (stepwise growth)
        # This is the rate-limiting step for cluster formation
        formation_rate = k_eff * (ion_pair_conc ** 2)
    
        # Dissociation: first order
        dissociation_rate = self.k_dissociation * self.dimer_concentration
    
        # Update
        d_dimer_dt = formation_rate - dissociation_rate
        self.dimer_concentration += d_dimer_dt * dt
    
        # Physical limits
        self.dimer_concentration = np.maximum(self.dimer_concentration, 0)
    
        # Stoichiometry: need 6 Ca per dimer
        max_dimer_from_ca = ca_conc / 6.0
        self.dimer_concentration = np.minimum(self.dimer_concentration, max_dimer_from_ca)


class TemplateEffects:
    """
    Template (protein) catalysis of aggregation
    
    Tao et al. 2010 Cryst Growth Des:
    "Surfaces reduce dimensionality: 3D→2D aggregation is 100-1000x faster"
    """
    
    def __init__(self, grid_shape: Tuple[int, int],
                 template_positions: Optional[List[Tuple[int, int]]] = None):
        self.grid_shape = grid_shape
        self.template_field = np.zeros(grid_shape)
        
        if template_positions:
            for x, y in template_positions:
                self.template_field[x, y] = 1.0
            logger.info(f"Initialized {len(template_positions)} template sites")
        
        self.enhancement_field = np.zeros(grid_shape)
        
    def calculate_surface_enhancement(self) -> np.ndarray:
        """
        2D surface aggregation is 1000x faster than 3D
        
        Enhancement decays with distance from template surface
        """
        if np.sum(self.template_field) == 0:
            return np.zeros(self.grid_shape)
        
        distance = ndimage.distance_transform_edt(1 - self.template_field)
        distance_nm = distance * 4.0  # Assuming 4nm grid spacing
        
        # SHARPER decay: 1000x at surface, 1x at ~5-8nm
        # Decay length = 1.5 nm (tighter confinement to template surface)
        enhancement = 1000 * np.exp(-distance_nm / 1.5)

        # Only count enhancement > 10x as "template effect"
        enhancement = np.where(enhancement > 10.0, enhancement, 1.0)
        
        return enhancement


class CaHPO4DimerSystem:
    """
    CORRECTED calcium phosphate system using proper 1:1 ion pair chemistry
    
    Step 1: Ca²⁺ + HPO₄²⁻ → CaHPO₄⁰ (instant, K=588 M⁻¹)
    Step 2: 6 × CaHPO₄ → Ca₆(PO₄)₄ dimer (slow aggregation)
    Step 3: Dimers are quantum qubits (4 ³¹P, T2=100s)
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float,
                 params: Model6Parameters,
                 template_positions: Optional[List[Tuple[int, int]]] = None):
        
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params
        
        # Components
        self.ion_pair_formation = CalciumPhosphateIonPair(params, params.environment.T)
        self.dimerization = CalciumPhosphateDimerization(grid_shape, dx)
        self.templates = TemplateEffects(grid_shape, template_positions)
        
        # State fields
        self.ion_pair_concentration = np.zeros(grid_shape)  # CaHPO₄⁰
        self.dimer_concentration = np.zeros(grid_shape)      # Ca₆(PO₄)₄
        
        # Precompute template enhancement
        self.template_enhancement = self.templates.calculate_surface_enhancement()
        
        logger.info("Initialized CaHPO₄→Ca₆(PO₄)₄ dimer system with CORRECT chemistry")
        
    def step(self, dt: float, ca_conc: np.ndarray, po4_conc: np.ndarray) -> None:
        """
        Update system with correct 1:1 ion pair chemistry
        
        Args:
            dt: Time step (s)
            ca_conc: Free calcium concentration (M)
            po4_conc: HPO₄²⁻ concentration (M) at pH 7.3
        """
        # Step 1: CaHPO₄ ion pairs form instantly (equilibrium)
        self.ion_pair_concentration = self.ion_pair_formation.calculate_ion_pair_concentration(
            ca_conc, po4_conc
        )
        
        # Step 2: Dimers form via 6-body aggregation (slow)
        self.dimerization.update_dimerization(
            dt,
            self.ion_pair_concentration,
            self.template_enhancement,
            ca_conc
        )
        
        # Update reference
        self.dimer_concentration = self.dimerization.dimer_concentration
        
    def get_ion_pair_concentration(self) -> np.ndarray:
        """Get CaHPO₄⁰ concentration (M) - the monomer units"""
        return self.ion_pair_concentration
    
    def get_dimer_concentration(self) -> np.ndarray:
        """Get Ca₆(PO₄)₄ concentration (M) - the quantum qubit!"""
        return self.dimer_concentration
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """Return metrics in nM for validation"""
        return {
            'ion_pair_mean_nM': float(np.mean(self.ion_pair_concentration) * 1e9),
            'ion_pair_peak_nM': float(np.max(self.ion_pair_concentration) * 1e9),
            'dimer_mean_nM': float(np.mean(self.dimer_concentration) * 1e9),
            'dimer_peak_nM': float(np.max(self.dimer_concentration) * 1e9),
            'dimer_at_templates_nM': float(
                np.max(self.dimer_concentration[self.templates.template_field > 0.5]) * 1e9
                if np.sum(self.templates.template_field) > 0 else 0
            ),
        }