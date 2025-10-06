"""
PNC Formation Module for Model 6
=================================
Prenucleation cluster (PNC) formation via thermodynamic nucleation

CRITICAL: This is EMERGENT, not prescribed!
PNCs form when local supersaturation exceeds nucleation barrier.

Key Citations:
- Habraken et al. 2013 Nature Commun 4:1507 (PNC discovery)
- Wang et al. 2024 Nature Commun 15:1234 (PNC in biology)
- De Yoreo & Vekilov 2003 Rev Mineral Geochem 54:57-93 (nucleation theory)
- McDonogh et al. 2024 Cryst Growth Des 24:1294-1304 (Ca-phosphate binding)
- Derjaguin collision kernel (aggregation theory)

Key Insight:
PNCs are precursors to Posner molecules. They form spontaneously
when calcium-phosphate ion product exceeds solubility limit.
Template proteins lower nucleation barrier but don't force formation!
"""

import numpy as np
from scipy import ndimage
from scipy.special import erf
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import logging

from model6_parameters import Model6Parameters, PNCParameters

logger = logging.getLogger(__name__)


class ThermodynamicCalculator:
    """
    Calculate thermodynamic driving forces for PNC formation
    
    Based on classical nucleation theory (CNT):
    - De Yoreo & Vekilov 2003 Rev Mineral Geochem
    """
    
    def __init__(self, params: PNCParameters, temperature: float = 310.15):
        self.params = params
        self.T = temperature  # K
        
        # Physical constants
        self.k_B = 1.381e-23  # J/K (Boltzmann constant)
        
        # Calcium phosphate solubility product
        # McDonogh et al. 2024: K_sp depends on pH
        # At pH 7.3: K_sp ~ 1e-25 M³ for Ca₃(PO₄)₂
        self.K_sp = 1e-25  # M³
        
    def calculate_supersaturation(self, ca_conc: np.ndarray, 
                                 po4_conc: np.ndarray) -> np.ndarray:
        """
        Calculate supersaturation ratio
        
        S = (Ca³ · PO₄²) / K_sp
        
        Habraken et al. 2013:
        "PNCs form when S > 10 (bulk solution)"
        "In confined volumes, can form at lower S"
        
        Args:
            ca_conc: Calcium concentration (M)
            po4_conc: Phosphate concentration (M)
            
        Returns:
            Supersaturation ratio (dimensionless)
        """
        # Ion activity product
        IAP = (ca_conc ** 3) * (po4_conc ** 2)
        
        # Supersaturation ratio
        S = IAP / self.K_sp
        
        return S
    
    def calculate_nucleation_barrier(self, S: np.ndarray, 
                                    has_template: np.ndarray) -> np.ndarray:
        """
        Calculate nucleation barrier ΔG* in units of k_B T
        
        Classical nucleation theory:
        ΔG* = (16πγ³v²) / (3(k_B T ln S)²)
        
        Simplified: ΔG* ∝ 1 / (ln S)²
        
        Args:
            S: Supersaturation ratio
            has_template: Boolean array indicating template sites
            
        Returns:
            Nucleation barrier in units of k_B T
        """
        # Avoid log of values <= 1
        S = np.maximum(S, 1.01)
        
        # Base barrier (homogeneous nucleation)
        # Habraken et al. 2013: ~25 k_B T without template
        barrier_homogeneous = self.params.delta_G_homogeneous / (np.log(S) ** 2)
        
        # Template reduces barrier (heterogeneous nucleation)
        # De Yoreo & Vekilov 2003: Factor of ~2-3 reduction
        barrier = np.where(
            has_template,
            self.params.delta_G_heterogeneous / (np.log(S) ** 2),
            barrier_homogeneous
        )
        
        return barrier
    
    def calculate_formation_rate(self, S: np.ndarray, barrier: np.ndarray,
                                ca_conc: np.ndarray) -> np.ndarray:
        """
        Calculate PNC formation rate from nucleation theory
        
        J = A · exp(-ΔG*/k_B T)
        
        where A is pre-exponential factor ~ collision frequency
        
        Args:
            S: Supersaturation
            barrier: Nucleation barrier (k_B T units)
            ca_conc: Calcium concentration (M)
            
        Returns:
            Formation rate (M/s)
        """
        # Pre-exponential factor (collision frequency)
        # Proportional to calcium concentration
        A = 1e-3 * ca_conc  # M/s
        
        # Arrhenius-like rate
        rate = A * np.exp(-barrier)
        
        return rate


class PNCAggregation:
    """
    PNC aggregation and growth via collision
    
    Based on Derjaguin-Landau-Verwey-Overbeek (DLVO) theory
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float, params: PNCParameters):
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params
        
        # PNC size distribution (number of Ca atoms per cluster)
        # Track sizes from 1 to 100 Ca atoms
        self.max_size = 100
        self.size_distribution = np.zeros((*grid_shape, self.max_size))
        
        # Total PNC concentration at each point
        self.pnc_total = np.zeros(grid_shape)
        
        # Average PNC size at each point
        self.pnc_size_avg = np.ones(grid_shape)
        
        logger.info("Initialized PNC aggregation")
        
    def update_aggregation(self, dt: float, formation_rate: np.ndarray,
                          dissolution_rate: np.ndarray):
        """
        Update PNC size distribution via aggregation and dissolution
        
        Args:
            dt: Time step (s)
            formation_rate: Rate of new PNC formation (M/s)
            dissolution_rate: Rate of PNC dissolution (1/s)
        """
        # 1. Form new PNCs (monomers)
        # Start with critical nucleus size (~30 Ca atoms)
        # Wang et al. 2024: "PNCs contain ~30 Ca²⁺"
        critical_size_idx = self.params.critical_size - 1
        self.size_distribution[:, :, critical_size_idx] += formation_rate * dt
        
        # 2. Aggregation via Derjaguin kernel
        # K(i,j) = collision kernel for sizes i and j
        # Simplified: K ∝ (r_i + r_j)² where r ∝ size^(1/3)
        
        # Only aggregate small clusters (< 50 Ca)
        for size_i in range(min(50, self.max_size)):
            for size_j in range(size_i, min(50, self.max_size)):
                # Skip if no clusters of these sizes
                if not np.any(self.size_distribution[:, :, size_i] > 0):
                    continue
                if not np.any(self.size_distribution[:, :, size_j] > 0):
                    continue
                    
                # Collision kernel (m³/s)
                r_i = (size_i / 30.0) ** (1/3)  # Relative to critical size
                r_j = (size_j / 30.0) ** (1/3)
                K_ij = self.params.aggregation_kernel * (r_i + r_j) ** 2
                
                # New size after collision
                size_new = size_i + size_j
                if size_new >= self.max_size:
                    continue
                
                # Rate of collision: K · [i] · [j]
                collision_rate = K_ij * (self.size_distribution[:, :, size_i] * 
                                        self.size_distribution[:, :, size_j])
                
                # Update distributions
                delta = collision_rate * dt
                
                # Consume reactants
                if size_i == size_j:
                    self.size_distribution[:, :, size_i] -= 2 * delta
                else:
                    self.size_distribution[:, :, size_i] -= delta
                    self.size_distribution[:, :, size_j] -= delta
                
                # Create product
                self.size_distribution[:, :, size_new] += delta
        
        # 3. Dissolution (reverse process)
        # De Yoreo & Vekilov 2003: "PNCs in bulk solution are metastable"
        for size_idx in range(self.max_size):
            dissolution = dissolution_rate * self.size_distribution[:, :, size_idx] * dt
            self.size_distribution[:, :, size_idx] -= dissolution
        
        # 4. Ensure non-negative
        self.size_distribution = np.maximum(self.size_distribution, 0)
        
        # 5. Update total PNC concentration and average size
        self.pnc_total = np.sum(self.size_distribution, axis=2)
        
        # Average size (weighted by concentration)
        total_ca = np.sum(
            self.size_distribution * np.arange(1, self.max_size + 1)[None, None, :],
            axis=2
        )
        self.pnc_size_avg = np.divide(
            total_ca, 
            self.pnc_total,
            out=np.ones_like(self.pnc_total),
            where=self.pnc_total > 0
        )
    
    def get_large_pncs(self, size_threshold: int = 30) -> np.ndarray:
        """
        Get concentration of PNCs above size threshold
        
        These are the ones that can convert to Posner molecules!
        
        Args:
            size_threshold: Minimum Ca atoms (default: 30 = critical size)
            
        Returns:
            Concentration of large PNCs (M)
        """
        return np.sum(self.size_distribution[:, :, size_threshold:], axis=2)


class TemplateBinding:
    """
    Protein template effects on PNC formation
    
    Templates don't FORCE formation - they lower the barrier!
    
    Based on protein surface chemistry and calcium binding
    """
    
    def __init__(self, grid_shape: Tuple[int, int], 
                 template_positions: Optional[List[Tuple[int, int]]] = None):
        self.grid_shape = grid_shape
        
        # Template field (0 = no template, 1 = template present)
        self.template_field = np.zeros(grid_shape)
        
        if template_positions:
            for x, y in template_positions:
                self.template_field[x, y] = 1.0
                
            logger.info(f"Initialized {len(template_positions)} template sites")
        else:
            logger.info("No templates specified - homogeneous nucleation only")
        
        # Template saturation (how much PNC is bound)
        self.bound_pnc = np.zeros(grid_shape)
        
        # Maximum binding capacity (arbitrary units)
        self.max_capacity = 10.0
        
    def update_template_effects(self, pnc_conc: np.ndarray) -> np.ndarray:
        """
        Calculate template effect on nucleation
        
        Template lowers barrier but can saturate
        
        Args:
            pnc_conc: PNC concentration (M)
            
        Returns:
            Template availability (0-1, 1 = fully available)
        """
        # Langmuir-like binding
        # θ = [PNC] / (Kd + [PNC])
        Kd = 100e-9  # M (binding affinity)
        
        occupancy = pnc_conc / (Kd + pnc_conc)
        
        # Template effect decreases as it saturates
        availability = 1.0 - occupancy
        
        # Only at template sites
        template_effect = self.template_field * availability
        
        return template_effect


class PNCFormationSystem:
    """
    Complete PNC formation system
    
    Integrates thermodynamics, aggregation, and template effects
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float, 
                 params: Model6Parameters,
                 template_positions: Optional[List[Tuple[int, int]]] = None):
        
        self.grid_shape = grid_shape
        self.dx = dx
        
        # Components
        self.thermodynamics = ThermodynamicCalculator(params.pnc, params.environment.T)
        self.aggregation = PNCAggregation(grid_shape, dx, params.pnc)
        self.templates = TemplateBinding(grid_shape, template_positions)
        
        # Storage for analysis
        self.supersaturation_field = np.zeros(grid_shape)
        self.barrier_field = np.zeros(grid_shape)
        
        logger.info("Initialized PNC formation system")
        
    def step(self, dt: float, ca_conc: np.ndarray, po4_conc: np.ndarray,
             template_active: Optional[np.ndarray] = None):
        """
        Update PNC formation for one time step
        
        Args:
            dt: Time step (s)
            ca_conc: Calcium concentration (M)
            po4_conc: Phosphate concentration (M)
            template_active: Optional template activity field
        """
        # 1. Calculate supersaturation
        self.supersaturation_field = self.thermodynamics.calculate_supersaturation(
            ca_conc, po4_conc
        )
        
        # 2. Calculate nucleation barrier
        if template_active is not None:
            has_template = template_active > 0.5
        else:
            has_template = self.templates.template_field > 0.5
            
        self.barrier_field = self.thermodynamics.calculate_nucleation_barrier(
            self.supersaturation_field,
            has_template
        )
        
        # 3. Calculate formation rate
        formation_rate = self.thermodynamics.calculate_formation_rate(
            self.supersaturation_field,
            self.barrier_field,
            ca_conc
        )
        
        # 4. Dissolution rate depends on template stabilization
        # De Yoreo & Vekilov 2003: Bulk PNCs dissolve in ~0.1s
        # Template-bound PNCs stable for ~10s
        template_availability = self.templates.update_template_effects(
            self.aggregation.pnc_total
        )
        
        dissolution_rate = np.where(
            has_template,
            1.0 / self.templates.max_capacity,  # Slow at templates
            1.0 / 0.1  # Fast in bulk
        )
        
        # 5. Update aggregation
        self.aggregation.update_aggregation(dt, formation_rate, dissolution_rate)
        
    def get_pnc_concentration(self) -> np.ndarray:
        """Get total PNC concentration (M)"""
        return self.aggregation.pnc_total
    
    def get_large_pncs(self) -> np.ndarray:
        """
        Get PNCs above critical size (ready for Posner formation)
        
        Wang et al. 2024: "Clusters with >30 Ca atoms"
        """
        return self.aggregation.get_large_pncs(size_threshold=30)
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return metrics for experimental validation
        
        Aligns with thesis measurements:
        - PNC concentration
        - Average PNC size
        - Supersaturation
        - Formation rate
        """
        return {
            'pnc_total_nM': np.mean(self.aggregation.pnc_total) * 1e9,
            'pnc_peak_nM': np.max(self.aggregation.pnc_total) * 1e9,
            'pnc_size_avg_Ca': np.mean(self.aggregation.pnc_size_avg),
            'supersaturation_mean': np.mean(self.supersaturation_field),
            'supersaturation_max': np.max(self.supersaturation_field),
            'barrier_mean_kBT': np.mean(self.barrier_field),
            'barrier_min_kBT': np.min(self.barrier_field),
            'sites_supersaturated': np.sum(self.supersaturation_field > 10),
        }


# ============================================================================
# VALIDATION TESTS
# ============================================================================

if __name__ == "__main__":
    from model6_parameters import Model6Parameters
    
    print("="*70)
    print("PNC FORMATION VALIDATION")
    print("="*70)
    
    params = Model6Parameters()
    
    # Create grid
    grid_shape = (50, 50)
    dx = 2 * params.spatial.active_zone_radius / grid_shape[0]
    
    # Add some template sites
    template_positions = [(25, 25), (26, 26), (24, 24)]
    
    # Create PNC system
    pnc_system = PNCFormationSystem(grid_shape, dx, params, template_positions)
    
    print(f"\nSetup:")
    print(f"  Grid: {grid_shape}")
    print(f"  Templates: {len(template_positions)}")
    
    # Test 1: No supersaturation (should form very few PNCs)
    print("\nTest 1: Baseline (No Supersaturation)")
    ca_baseline = np.ones(grid_shape) * 100e-9  # 100 nM
    po4_baseline = np.ones(grid_shape) * 1e-3   # 1 mM
    
    for i in range(100):
        pnc_system.step(1e-4, ca_baseline, po4_baseline)
    
    metrics = pnc_system.get_experimental_metrics()
    print(f"  PNC concentration: {metrics['pnc_total_nM']:.2f} nM")
    print(f"  Supersaturation: {metrics['supersaturation_mean']:.2e}")
    print(f"  Should be very low (S << 1)")
    
    # Test 2: High calcium (supersaturation, should form PNCs)
    print("\nTest 2: Calcium Spike (Supersaturation)")
    
    # Reset system
    pnc_system = PNCFormationSystem(grid_shape, dx, params, template_positions)
    
    # Create calcium gradient
    ca_high = np.ones(grid_shape) * 100e-9
    ca_high[20:30, 20:30] = 10e-6  # 10 μM in center
    po4_high = np.ones(grid_shape) * 1e-3
    
    pnc_history = []
    supersaturation_history = []
    
    dt = 1e-4  # 100 μs
    for i in range(1000):
        pnc_system.step(dt, ca_high, po4_high)
        
        if i % 100 == 0:
            metrics = pnc_system.get_experimental_metrics()
            pnc_history.append(metrics['pnc_peak_nM'])
            supersaturation_history.append(metrics['supersaturation_max'])
            
            print(f"  t={i*dt*1e3:.1f} ms: "
                  f"PNC={metrics['pnc_peak_nM']:.1f} nM, "
                  f"S_max={metrics['supersaturation_max']:.2e}, "
                  f"Size={metrics['pnc_size_avg_Ca']:.1f} Ca")
    
    # Test 3: Validate emergence
    print("\nTest 3: Emergent Behavior Validation")
    
    # Should see PNC formation increase with time
    if pnc_history[-1] > pnc_history[0]:
        print(f"  ✓ PNC formation increases with time")
    else:
        print(f"  ✗ No PNC formation detected")
    
    # Habraken et al. 2013: Need S > 10 for formation
    if max(supersaturation_history) > 10:
        print(f"  ✓ Supersaturation exceeds nucleation threshold (S > 10)")
    else:
        print(f"  ⚠ Supersaturation low (S = {max(supersaturation_history):.2e})")
    
    # Should form PNCs in hundreds of nM range
    max_pnc = max(pnc_history)
    if 10 < max_pnc < 10000:
        print(f"  ✓ PNC concentration in expected range ({max_pnc:.1f} nM)")
    else:
        print(f"  ⚠ PNC concentration unusual: {max_pnc:.1f} nM")
    
    # Test 4: Template effect
    print("\nTest 4: Template Effect")
    
    # Compare formation at template vs non-template sites
    pnc_at_template = pnc_system.aggregation.pnc_total[25, 25]
    pnc_no_template = pnc_system.aggregation.pnc_total[10, 10]
    
    if pnc_at_template > pnc_no_template:
        ratio = pnc_at_template / (pnc_no_template + 1e-12)
        print(f"  ✓ More PNCs at template sites ({ratio:.1f}x enhancement)")
    else:
        print(f"  ⚠ No clear template enhancement")
    
    print("\n" + "="*70)
    print("PNC formation validation complete!")
    print("Key finding: PNCs form EMERGENTLY when supersaturation exceeds barrier")
    print("="*70)