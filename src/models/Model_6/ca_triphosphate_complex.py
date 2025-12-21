"""
Calcium Phosphate Ion Pair Formation - CORRECTED CHEMISTRY
===========================================================

CRITICAL CORRECTION:
Previous model used Ca(HPO₄)₃⁴⁻ which is NOT the dominant species at pH 7.3!

CORRECT CHEMISTRY (McDonogh et al. 2024, Moreno & Brown 1966):
At pH 6-10, the dominant species is CaHPO₄⁰ (1:1 ion pair)

Ca²⁺ + HPO₄²⁻ ⇌ [CaHPO₄]⁰

Step 1: Ca²⁺ + HPO₄²⁻ ⇌ CaHPO₄⁰ (K=588 M⁻¹, instant)
Step 2: 6 × CaHPO₄ → Ca₆(PO₄)₄ (k=8e5 M⁻¹s⁻¹, slow)
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

        # NEW: Trimer concentration field (for Ca/P selectivity testing)
        self.trimer_concentration = np.zeros(grid_shape)
        
        # NEW: PNC concentration field (Turhan et al. 2024)
        self.pnc_concentration = np.zeros(grid_shape)
    
        # NEW: PNC formation parameters
        self.pnc_binding_fraction = 0.5  # 50% from Turhan 2024
        self.n_ion_pairs_per_pnc = 3  # Ca(HPO₄)₃⁴⁻ structure
    
        # NEW: PNC lifetime tracking (optional, for diagnostics)
        self.pnc_lifetime = np.zeros(grid_shape)
        
        
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
        
        # Recalibrated for PNC aggregation (from spec doc)
        # Target: 741 nM dimers in 100-200 ms with template enhancement
        # [PNC] ~ few μM (from 50% binding)
        # k_agg ≈ 8×10⁵ M⁻¹s⁻¹ for bulk aggregation
        self.k_base = 8e5  # M⁻¹s⁻¹ (recalibrated for PNC aggregation)
        
        
        # Dissociation (very slow - dimers are stable)
        self.k_dissociation = 0.001  # s⁻¹

        # Stochastic parameters
        self.nucleation_probability = 0.01  # Probability per timestep at templates
        self.dissolution_noise_sigma = 0.1  # Relative noise in dissolution
        
        logger.info(f"Initialized Ca₆(PO₄)₄ dimer aggregation (k_agg={self.k_base:.2e} M⁻¹s⁻¹)")
        
    def calculate_pnc_equilibrium(self, 
                                ion_pair_conc: np.ndarray,
                                ca_conc: np.ndarray) -> np.ndarray:
        """
        Calculate PNC concentration from fast equilibrium with ion pairs
    
        Based on Turhan et al. 2024:
        - 50-57% of Ca binds into PNCs immediately
        - Forms within milliseconds (treat as equilibrium)
    
        Reaction: n × CaHPO₄ ⇌ [Ca(HPO₄)₃]⁴⁻
    
        Args:
            ion_pair_conc: CaHPO₄ concentration (M)
            ca_conc: Total calcium concentration (M)
    
        Returns:
            PNC concentration (M)
        """
        # Strategy: Use binding fraction to determine PNC concentration
        # If 50% of Ca is in PNCs, then [Ca_PNC] = 0.5 × [Ca_total]
    
        # Each PNC contains 3 Ca atoms (from Ca(HPO₄)₃ structure)
        # So [PNC] = [Ca_PNC] / 3 = 0.5 × [Ca_total] / 3
    
        ca_in_pncs = self.pnc_binding_fraction * ca_conc
        pnc_conc = ca_in_pncs / 3.0  # 3 Ca per PNC
    
        # Physical limits
        # Can't exceed ion pair concentration (stoichiometry)
        max_pnc_from_ion_pairs = ion_pair_conc / self.n_ion_pairs_per_pnc
        pnc_conc = np.minimum(pnc_conc, max_pnc_from_ion_pairs)
    
        # Can't exceed calcium availability
        max_pnc_from_ca = ca_conc / 3.0
        pnc_conc = np.minimum(pnc_conc, max_pnc_from_ca)
    
        return pnc_conc
    
    
    def calculate_pnc_lifetime(self, ca_conc: np.ndarray, 
                            po4_conc: np.ndarray) -> np.ndarray:
        """
        Calculate PNC lifetime as function of Ca/P ratio
    
        From Turhan et al. 2024:
        χ = 0.50: τ = minutes
        χ = 0.375: τ = tens of minutes    
        χ = 0.25: τ = hours
    
        Also includes 17 kJ/mol dissociation barrier (Garcia 2019)
    
        Args:
            ca_conc: Calcium concentration (M)
            po4_conc: Phosphate concentration (M)
    
        Returns:
            PNC lifetime (s)
        """
        # Calculate Ca/P ratio
        ca_p_ratio = ca_conc / (po4_conc + 1e-10)  # Avoid division by zero
    
        # Empirical relationship from Turhan et al. 2024
        # Lower Ca/P → longer lifetime
        # τ ∝ 1 / χ^α (power law)
    
        # Fit to Turhan data:
        # χ = 0.50 → τ = 60 s (minutes)
        # χ = 0.25 → τ = 3600 s (hours)
        # log(3600/60) = log(60) = 4.09
        # log(0.5/0.25) = log(2) = 0.69
        # α = 4.09 / 0.69 ≈ 6
    
        tau_reference = 60.0  # seconds at χ = 0.5
        chi_reference = 0.5
        alpha = 6.0
    
        tau = tau_reference * (chi_reference / ca_p_ratio)**alpha
    
        # Physical limits
        tau = np.clip(tau, 60, 7200)  # 1 min to 2 hours
    
        return tau
    
    def calculate_dimer_fraction(self, ca_conc: np.ndarray,
                                po4_conc: np.ndarray) -> np.ndarray:
        """
        Calculate fraction of aggregates that form dimers vs trimers
    
        From Garcia et al. 2019:
        "Dimerization is favorable up to a Ca/HPO₄ ratio of 1:2"
    
        Interpretation:
        - Ca/P < 0.5: Strongly favors dimers
        - Ca/P ≈ 0.5: Transition region
        - Ca/P > 0.5: Favors trimers (and larger)
    
        Args:
            ca_conc: Calcium concentration (M)
            po4_conc: Phosphate concentration (M)
    
        Returns:
            Fraction forming dimers (0-1)
        """
        ca_p_ratio = ca_conc / (po4_conc + 1e-10)
    
        # Sigmoid centered at 0.5
        # Steep transition (factor of 10 controls steepness)
        dimer_fraction = 1.0 / (1.0 + np.exp(10 * (ca_p_ratio - 0.5)))
    
        return dimer_fraction


    
    def update_dimerization(self, 
                        dt: float,
                        ion_pair_conc: np.ndarray,
                        template_enhancement: np.ndarray,
                        ca_conc: np.ndarray,
                        po4_conc: np.ndarray = None) -> None:
        """
        Update dimer concentration via TWO-STEP STOCHASTIC kinetics
    
        STEP 1: Fast PNC equilibrium (milliseconds)
        CaHPO₄ ⇌ PNCs (treated as instantaneous)
    
        STEP 2: Slow aggregation (seconds, rate-limiting)
        PNCs → Ca₆(PO₄)₄ dimers (template-enhanced, STOCHASTIC)
    
        Changes from old version:
        - PNC concentration calculated from equilibrium (NEW)
        - Aggregation uses PNC concentration, not ion pairs (NEW)
        - Ca/P ratio controls dimer vs trimer formation (NEW)
        - Template enhancement ONLY affects aggregation step
        - Keeps stochastic nucleation and formation events (PRESERVED)
        """
    
        # =====================================================================
        # STEP 1: CALCULATE PNC CONCENTRATION (FAST EQUILIBRIUM)
        # =====================================================================
    
        # PNCs form from ion pairs in milliseconds
        # Treat as instantaneous equilibrium
        self.pnc_concentration = self.calculate_pnc_equilibrium(
            ion_pair_conc,
            ca_conc
        )
    
        # Validation: Check 50% binding (for diagnostics)
        # ca_in_pncs = self.pnc_concentration * 3.0  # 3 Ca per PNC
        # binding_fraction = ca_in_pncs / (ca_conc + 1e-12)
        # This should be ~0.5 everywhere

        # =====================================================================
        # NEW: CALCIUM THRESHOLD FOR FORMATION
        # =====================================================================
        # Only form NEW dimers when calcium is elevated
        # Below threshold, existing dimers persist but no new ones form
        ca_formation_threshold = 1e-6  # 1 µM - need calcium spike to form dimers
        
        if np.max(ca_conc) < ca_formation_threshold:
            # No new formation at rest - only dissociation
            dissolution_noise = 1.0 + np.random.normal(0, self.dissolution_noise_sigma, self.grid_shape)
            dissolution_noise = np.maximum(dissolution_noise, 0.1)
            
            self.dimer_concentration -= self.k_dissociation * self.dimer_concentration * dissolution_noise * dt
            self.trimer_concentration -= self.k_dissociation * 10.0 * self.trimer_concentration * dissolution_noise * dt
            
            self.dimer_concentration = np.maximum(self.dimer_concentration, 0)
            self.trimer_concentration = np.maximum(self.trimer_concentration, 0)
            return  # Skip formation steps
    
        # =====================================================================
        # STEP 2: SLOW PNC AGGREGATION (RATE-LIMITING, STOCHASTIC)
        # =====================================================================
    
        # Template enhancement applies ONLY to aggregation
        k_eff = self.k_base * template_enhancement
    
        # === STOCHASTIC FORMATION ===
        # Instead of deterministic rate, use probabilistic events
    
        # 1. Calculate formation probability per voxel
        # NOW USES PNC CONCENTRATION (not ion pairs!)
        formation_probability = k_eff * (self.pnc_concentration ** 2) * dt
        formation_probability = np.clip(formation_probability, 0, 1)  # Keep as probability
    
        # 2. Stochastic nucleation at high-enhancement sites (templates)
        template_sites = template_enhancement > 100  # Strong template regions
        nucleation_events = np.random.rand(*self.grid_shape) < (self.nucleation_probability * dt)
        nucleation_events = nucleation_events & template_sites & (self.pnc_concentration > 1e-7)  # Need PNC substrate
    
        # 3. Ca/P ratio determines dimer vs trimer formation
        # If po4_conc not provided, assume all form dimers (backward compatibility)
        if po4_conc is not None:
            dimer_fraction = self.calculate_dimer_fraction(ca_conc, po4_conc)
        else:
            dimer_fraction = np.ones_like(ca_conc)
    
        # 4. Combine deterministic growth + stochastic nucleation
        # Deterministic component (averaged behavior) - NOW USES PNCs
        deterministic_formation = k_eff * (self.pnc_concentration ** 2) * dt * 0.5  # Reduced weight
    
        # Stochastic component (random events)
        random_roll = np.random.rand(*self.grid_shape)
        stochastic_formation = np.where(
            random_roll < formation_probability,
            self.pnc_concentration * 0.1,  # Each event forms ~10% of available PNC substrate
            0
        )
    
        # Nucleation adds large burst at templates
        nucleation_contribution = np.where(
            nucleation_events,
            1e-10,  # 0.1 nM burst per nucleation event (reduced for stability)
            0
        )
    
        # Apply dimer fraction to split formation between dimers and trimers
        dimer_formation = (deterministic_formation + stochastic_formation + nucleation_contribution) * dimer_fraction
        trimer_formation = (deterministic_formation + stochastic_formation + nucleation_contribution) * (1 - dimer_fraction)

        # === CALCIUM-DEPENDENT SCALING (Fix for voltage sensitivity) ===
        # Scale formation by calcium level to create proper IO curve
        # Without this, template enhancement causes saturation at any calcium level
        ca_peak_uM = np.max(ca_conc) * 1e6  # Convert to μM
        ca_half_max_uM = 8.0  # Half-maximal calcium (tune for IO curve steepness)
        calcium_scaling = ca_peak_uM / (ca_half_max_uM + ca_peak_uM)  # Michaelis-Menten
        dimer_formation = dimer_formation * calcium_scaling
        trimer_formation = trimer_formation * calcium_scaling

        # === STOCHASTIC DISSOCIATION ===
        # Add noise to dissolution rate
        dissolution_noise = 1.0 + np.random.normal(0, self.dissolution_noise_sigma, self.grid_shape)
        dissolution_noise = np.maximum(dissolution_noise, 0.1)  # Keep positive

        dimer_dissociation = self.k_dissociation * self.dimer_concentration * dissolution_noise
        trimer_dissociation = self.k_dissociation * 10.0 * self.trimer_concentration * dissolution_noise  # Trimers less stable

        # Update both species
        d_dimer_dt = dimer_formation - dimer_dissociation  # (you modified this earlier)
        d_trimer_dt = trimer_formation - trimer_dissociation  # (you added this earlier)

        self.dimer_concentration += d_dimer_dt
        self.trimer_concentration += d_trimer_dt

        # Physical limits  ← YOUR NEW CODE STARTS HERE
        self.dimer_concentration = np.maximum(self.dimer_concentration, 0)
        self.trimer_concentration = np.maximum(self.trimer_concentration, 0)

        # Stoichiometry: need 6 Ca per dimer, 9 Ca per trimer
        # max_dimer_from_ca = ca_conc / 6.0
        # max_trimer_from_ca = ca_conc / 9.0
        # self.dimer_concentration = np.minimum(self.dimer_concentration, max_dimer_from_ca)
        # self.trimer_concentration = np.minimum(self.trimer_concentration, max_trimer_from_ca)

    
        # =====================================================================
        # STEP 3: PNC LIFETIME TRACKING (OPTIONAL, FOR VALIDATION)
        # =====================================================================
    
        if po4_conc is not None:
            dimer_fraction = self.calculate_dimer_fraction(ca_conc, po4_conc)
    
        else:
            dimer_fraction = np.ones_like(ca_conc)


    def get_pnc_metrics(self) -> Dict[str, float]:
        """
        Get PNC-specific metrics for validation
    
        Returns:
            Dictionary with PNC statistics
        """
        ca_in_pncs = self.pnc_concentration * 3.0  # 3 Ca per PNC
    
        # Need to handle case where we don't have ca_total stored
        # Use mean PNC concentration as proxy
        total_ca_estimated = np.mean(ca_in_pncs) / 0.5  # Assume 50% binding
    
        return {
            'pnc_peak_nM': float(np.max(self.pnc_concentration) * 1e9),
            'pnc_mean_nM': float(np.mean(self.pnc_concentration) * 1e9),
            'pnc_binding_fraction': 0.5,  # By design from Turhan 2024
            'pnc_lifetime_mean_s': float(np.mean(self.pnc_lifetime)) if np.any(self.pnc_lifetime > 0) else 0,
            'pnc_lifetime_max_s': float(np.max(self.pnc_lifetime)) if np.any(self.pnc_lifetime > 0) else 0,
        }

    def get_dimer_trimer_ratio(self) -> float:
        """
        Get ratio of dimers to trimers
    
        Returns:
            Dimer/trimer ratio (currently only tracks dimers, so returns inf)
        """
        dimer_total = np.sum(self.dimer_concentration)
        trimer_total = np.sum(self.trimer_concentration)
    
        if trimer_total > 0:
            return dimer_total / trimer_total
        elif dimer_total > 0:
            return np.inf  # All dimers
        else:
            return 0.0



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
        enhancement = 50 * np.exp(-distance_nm / 1.5)

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
        # DIAGNOSTIC: Check what phosphate we're actually getting
        #if np.random.rand() < 0.0001:
            #print(f"CaHPO4DimerSystem.step: po4_conc = {np.mean(po4_conc)*1e3:.2f} mM (mean), {np.max(po4_conc)*1e3:.2f} mM (max)")
        
        
        # Step 1: CaHPO₄ ion pairs form instantly (equilibrium)
        self.ion_pair_concentration = self.ion_pair_formation.calculate_ion_pair_concentration(
            ca_conc, po4_conc
        )
        
        # Step 2: PNCs form (fast equilibrium), then aggregate to dimers (slow)
        self.dimerization.update_dimerization(
            dt,
            self.ion_pair_concentration,
            self.template_enhancement,
            ca_conc,
            po4_conc  # NEW: Pass phosphate for Ca/P ratio calculations
        )
        
        # Update reference
        self.dimer_concentration = self.dimerization.dimer_concentration
        
    def get_ion_pair_concentration(self) -> np.ndarray:
        """Get CaHPO₄⁰ concentration (M) - the monomer units"""
        return self.ion_pair_concentration
    
    def get_dimer_concentration(self) -> np.ndarray:
        """Get Ca₆(PO₄)₄ concentration (M) - the quantum qubit!"""
        return self.dimer_concentration
    
    def get_formation_rate_field(self) -> np.ndarray:
        """
        Get dimer formation rate field for particle system
        
        Returns formation rate in M/s at each grid point
        """
        # Formation rate = k_eff × [PNC]²
        # This is what feeds the particle birth process
        k_eff = self.dimerization.k_base * self.template_enhancement
        formation_rate = k_eff * (self.dimerization.pnc_concentration ** 2)
        return formation_rate
    
    def set_n_templates(self, n_templates: int):
        """
        Update number of template sites (for plasticity feedback)
        
        As spine grows, more scaffold proteins become available,
        providing more nucleation sites for dimer formation.
        
        Args:
            n_templates: New number of template sites (typically 3-10)
        """
        n_templates = max(1, min(n_templates, 15))  # Clamp to reasonable range
        
        # Get current template count
        current_count = int(np.sum(self.templates.template_field))
        
        if n_templates == current_count:
            return  # No change needed
        
        # Get grid center (where channels are)
        center = self.grid_shape[0] // 2
        
        if n_templates > current_count:
            # Add more templates near center
            added = 0
            radius = 1
            while added < (n_templates - current_count) and radius < 10:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        x, y = center + dx, center + dy
                        if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                            if self.templates.template_field[x, y] == 0:
                                self.templates.template_field[x, y] = 1.0
                                added += 1
                                if added >= (n_templates - current_count):
                                    break
                    if added >= (n_templates - current_count):
                        break
                radius += 1
        else:
            # Remove templates (furthest from center first)
            positions = np.argwhere(self.templates.template_field > 0)
            if len(positions) > 0:
                # Sort by distance from center (furthest first)
                distances = np.sqrt((positions[:, 0] - center)**2 + (positions[:, 1] - center)**2)
                sorted_idx = np.argsort(-distances)  # Descending
                
                to_remove = current_count - n_templates
                for i in range(min(to_remove, len(sorted_idx))):
                    x, y = positions[sorted_idx[i]]
                    self.templates.template_field[x, y] = 0.0
        
        # Recalculate enhancement field
        self.template_enhancement = self.templates.calculate_surface_enhancement()
    
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """Return metrics in nM for validation (includes PNC data)"""
        metrics = {
            'ion_pair_mean_nM': float(np.mean(self.ion_pair_concentration) * 1e9),
            'ion_pair_peak_nM': float(np.max(self.ion_pair_concentration) * 1e9),
            'dimer_mean_nM': float(np.mean(self.dimer_concentration) * 1e9),
            'dimer_peak_nM': float(np.max(self.dimer_concentration) * 1e9),
            'trimer_mean_nM': float(np.mean(self.dimerization.trimer_concentration) * 1e9),  # NEW
            'trimer_peak_nM': float(np.max(self.dimerization.trimer_concentration) * 1e9),  # NEW
            'dimer_trimer_ratio': float(self.dimerization.get_dimer_trimer_ratio()),  # NEW
            'dimer_at_templates_nM': float(
                np.max(self.dimer_concentration[self.templates.template_field > 0.5]) * 1e9
                if np.sum(self.templates.template_field) > 0 else 0
            ),
        }
    
        # Add PNC metrics
        metrics.update(self.dimerization.get_pnc_metrics())
    
        return metrics  # <-- Return AFTER adding PNC metrics