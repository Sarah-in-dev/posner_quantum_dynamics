"""
Dopamine System Module for Model 6
===================================
Complete biophysically-constrained dopamine dynamics.

Merged from dopamine_system.py and dopamine_biophysics.py to provide
full spatial dynamics, receptor binding, and quantum modulation effects.

Key Citations:
- Pothos et al. 2000 J Neurosci (vesicular content)
- Garris et al. 1994 J Neurochem (concentration ranges)
- Rice & Cragg 2008 Brain Res Rev (diffusion, tortuosity)
- Cragg 2000 J Neurosci (DAT kinetics)
- Hernandez-Lopez et al. 2000 J Neurosci (D2 effects on calcium)
- Yagishita et al. 2014 Science (dopamine timing window)

KEY FINDING FROM MODEL 5:
Dopamine biases toward dimers (long coherence) over trimers (short coherence)
This is the quantum switch for learning!
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import ndimage
import logging

from model6_parameters import Model6Parameters, DopamineParameters

logger = logging.getLogger(__name__)


class DopamineSystemAdapter:
    """
    Biophysically-constrained dopamine system for Model 6
    
    Handles:
    - Stochastic vesicular release with Gaussian spatial spread
    - Diffusion with tortuosity correction
    - Michaelis-Menten DAT uptake
    - D1 and D2 receptor binding dynamics
    - Quantum modulation effects (dimer/trimer bias)
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float,
                 params: Model6Parameters,
                 release_sites: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize dopamine system
        
        Args:
            grid_shape: Spatial grid shape
            dx: Grid spacing (m)
            params: Model 6 parameters
            release_sites: Optional list of (x, y) dopamine release positions
        """
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params.dopamine
        
        # Default release site at center if not specified
        if release_sites is None:
            center = grid_shape[0] // 2
            release_sites = [(center, center)]
        
        self.release_sites = release_sites
        
        # === CONCENTRATION FIELDS ===
        self.dopamine_concentration = np.ones(grid_shape) * self.params.dopamine_tonic
        
        # === RECEPTOR OCCUPANCY FIELDS ===
        self.d1_occupancy = np.zeros(grid_shape)
        self.d2_occupancy = np.zeros(grid_shape)
        
        # === DIFFUSION PARAMETERS ===
        # Tortuosity correction (Rice & Cragg 2008)
        self.tortuosity = 1.6
        self.D_effective = self.params.D_dopamine / (self.tortuosity ** 2)
        
        # === RELEASE PARAMETERS ===
        self.release_spread_sigma = 1e-6  # 1 µm spread from release site
        self.cleft_width = 20e-9  # 20 nm synaptic cleft
        
        # === STOCHASTIC PARAMETERS ===
        self.vesicle_release_probability_base = self.params.release_probability
        self.vesicle_content_mean = self.params.molecules_per_vesicle
        self.vesicle_content_std = self.params.molecules_per_vesicle * 0.2  # 20% CV
        
        # === STATE TRACKING ===
        self.release_timer = 0.0
        self.vesicles_released = 0
        self.total_molecules_released = 0
        self.burst_duration = 0.1  # 100 ms phasic burst
        
        # === QUANTUM MODULATION PARAMETERS ===
        self.da_quantum_threshold = 100e-9  # 100 nM for quantum effects
        self.da_protection_threshold = 50e-9  # 50 nM starts protecting coherence
        self.dimer_enhancement_max = 10.0
        self.trimer_suppression_max = 0.1
        self.coherence_protection_max = 3.0
        self.d2_quantum_coupling = 10e-9  # Kd for quantum effects via D2
        
        # === TIMING PARAMETERS (Yagishita 2014) ===
        self.da_calcium_delay = 0.050  # 50 ms optimal delay
        self.quantum_window = 0.200  # 200 ms window for coincidence
        
        logger.info(f"Dopamine system initialized: {grid_shape}, "
                   f"{len(release_sites)} release sites, D_eff={self.D_effective:.2e} m²/s")
    
    # =========================================================================
    # MAIN UPDATE
    # =========================================================================
    
    def step(self, dt: float, reward_signal: bool = False):
        """
        Update dopamine system for one timestep
        
        Args:
            dt: Time step (s)
            reward_signal: Whether reward is present (triggers phasic release)
        """
        # 1. Handle release
        if reward_signal:
            self._phasic_release(dt)
        else:
            self._tonic_release(dt)
            self.release_timer = 0
        
        # 2. Diffusion with tortuosity
        # self._apply_diffusion(dt)
        
        # 3. DAT uptake (Michaelis-Menten)
        self._apply_uptake(dt)
        
        # 4. Update receptor occupancy
        self._update_receptor_occupancy()
        
        # 5. Enforce constraints
        self._enforce_constraints()
    
    # =========================================================================
    # RELEASE DYNAMICS
    # =========================================================================
    
    def _phasic_release(self, dt: float):
        """Phasic burst release during reward"""
        if self.release_timer < self.burst_duration:
            for (i, j) in self.release_sites:
                # During burst, release is more reliable
                if self.release_timer < 0.1:  # First 100ms of burst
                    vesicles_per_pulse = 5  # Multiple vesicles for µM concentrations
                    self._release_vesicles(i, j, vesicles_per_pulse)
            
            self.release_timer += dt
    
    def _tonic_release(self, dt: float):
        """Tonic release during normal activity"""
        tonic_firing_rate = 4.0  # Hz
        
        for (i, j) in self.release_sites:
            p_release = self.vesicle_release_probability_base * tonic_firing_rate * dt
            
            if np.random.random() < p_release:
                self._release_vesicles(i, j, 1)
    
    def _release_vesicles(self, i: int, j: int, n_vesicles: int):
        """Release vesicles at specified location with Gaussian spatial spread"""
        # Stochastic vesicle content
        molecules_per_vesicle = max(100, np.random.normal(
            self.vesicle_content_mean, self.vesicle_content_std
        ))
        total_molecules = n_vesicles * molecules_per_vesicle
        
        self.total_molecules_released += total_molecules
        self.vesicles_released += n_vesicles
        
        # Calculate concentration increase
        sigma = self.release_spread_sigma
        effective_radius = 2 * sigma  # 2 sigma captures ~95%
        effective_area = np.pi * effective_radius**2
        effective_volume = effective_area * self.cleft_width
        
        avogadro = 6.022e23
        moles = total_molecules / avogadro
        concentration_increase = moles / effective_volume
        
        # Apply with Gaussian spread
        self._add_with_gaussian_spread(i, j, concentration_increase)
    
    def _add_with_gaussian_spread(self, i: int, j: int, concentration: float):
        """Add concentration with Gaussian spatial spread"""
        sigma_grid = self.release_spread_sigma / self.dx
        
        # Create kernel
        kernel_size = min(int(6 * sigma_grid) + 1, self.grid_shape[0])
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        
        # Generate normalized Gaussian
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma_grid**2 + 1e-10))
        kernel = kernel / np.sum(kernel)
        
        # Apply to field with boundary handling
        half = kernel_size // 2
        i_min = max(0, i - half)
        i_max = min(self.grid_shape[0], i + half + 1)
        j_min = max(0, j - half)
        j_max = min(self.grid_shape[1], j + half + 1)
        
        ki_min = half - (i - i_min)
        ki_max = half + (i_max - i)
        kj_min = half - (j - j_min)
        kj_max = half + (j_max - j)
        
        self.dopamine_concentration[i_min:i_max, j_min:j_max] += \
            concentration * kernel[ki_min:ki_max, kj_min:kj_max]
    
    # =========================================================================
    # DIFFUSION AND UPTAKE
    # =========================================================================
    
    def _apply_diffusion(self, dt: float):
        """Apply diffusion with Laplacian and tortuosity correction"""
        # Laplacian kernel
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64)
        
        laplacian = ndimage.convolve(self.dopamine_concentration, kernel, mode='nearest')
        laplacian = laplacian / (self.dx * self.dx)
        
        self.dopamine_concentration += self.D_effective * laplacian * dt
    
    def _apply_uptake(self, dt: float):
        """Apply Michaelis-Menten uptake via DAT"""
        # Only uptake above tonic level
        excess = np.maximum(0, self.dopamine_concentration - self.params.dopamine_tonic)
        
        # Michaelis-Menten kinetics with stochastic noise
        uptake_noise = np.random.normal(1.0, 0.1, self.grid_shape)
        uptake_noise = np.maximum(uptake_noise, 0.1)
        
        uptake_rate = self.params.dat_vmax * excess / (self.params.dat_km + excess)
        self.dopamine_concentration -= uptake_rate * dt * uptake_noise
    
    def _update_receptor_occupancy(self):
        """Update D1 and D2 receptor occupancy (Hill equation)"""
        # D1: High affinity (~1 nM)
        kd_d1 = 1e-9
        self.d1_occupancy = self.dopamine_concentration / (self.dopamine_concentration + kd_d1)
        
        # D2: Lower affinity (~10 nM) - main quantum modulator
        self.d2_occupancy = self.dopamine_concentration / (self.dopamine_concentration + self.params.d2_kd)
        
        # Clip to valid range
        self.d1_occupancy = np.clip(self.d1_occupancy, 0, 1)
        self.d2_occupancy = np.clip(self.d2_occupancy, 0, 1)
    
    def _enforce_constraints(self):
        """Enforce biological constraints"""
        # Minimum at tonic level
        self.dopamine_concentration = np.maximum(
            self.dopamine_concentration, self.params.dopamine_tonic
        )
        
        # Maximum reasonable concentration
        max_possible = 10e-6  # 10 µM
        self.dopamine_concentration = np.minimum(self.dopamine_concentration, max_possible)
    
    # =========================================================================
    # ACCESSORS
    # =========================================================================
    
    def get_dopamine_concentration(self) -> np.ndarray:
        """Get current dopamine concentration field (M)"""
        return self.dopamine_concentration.copy()
    
    def get_d1_occupancy(self) -> np.ndarray:
        """Get D1 receptor occupancy field (0-1)"""
        return self.d1_occupancy.copy()
    
    def get_d2_occupancy(self) -> np.ndarray:
        """Get D2 receptor occupancy field (0-1)"""
        return self.d2_occupancy.copy()
    
    def get_calcium_modulation(self) -> np.ndarray:
        """
        Get calcium channel modulation factor
        
        Hernandez-Lopez et al. 2000:
        D2 activation reduces Ca²⁺ current by 30%
        
        Returns:
            Modulation factor (0.7 = 30% reduction at full D2 activation)
        """
        modulation = 1.0 - (self.params.ca_channel_inhibition * self.d2_occupancy)
        return modulation
    
    def get_dimer_bias(self) -> np.ndarray:
        """
        Calculate bias toward dimer formation
        
        MODEL 5 KEY FINDING:
        High dopamine (D2 activation) favors dimers over trimers
        
        Returns:
            Dimer bias factor (0-1, where 1 = maximum dimer preference)
        """
        # At 0% D2: prefer trimers (bias = 0.3)
        # At 100% D2: prefer dimers (bias = 0.8)
        bias = 0.3 + 0.5 * self.d2_occupancy
        return bias
    
    # =========================================================================
    # QUANTUM MODULATION
    # =========================================================================
    
    def get_quantum_modulation(self, i: int, j: int) -> Dict[str, float]:
        """
        Calculate quantum modulation factors at a specific location.
        
        Returns factors for dimer enhancement, trimer suppression, 
        and coherence protection.
        
        Based on:
        - Agarwal et al. 2023: Dimers superior for coherence
        - D2 receptor activation curve
        """
        da_local = self.dopamine_concentration[i, j]
        
        # D2 receptor activation for quantum effects
        d2_activation = da_local / (da_local + self.d2_quantum_coupling)
        
        factors = {
            'dimer_enhancement': 1.0 + (self.dimer_enhancement_max - 1.0) * d2_activation,
            'trimer_suppression': 1.0 - (1.0 - self.trimer_suppression_max) * d2_activation,
            'coherence_protection': 1.0 + (self.coherence_protection_max - 1.0) * d2_activation,
            'd2_occupancy': d2_activation,
            'above_quantum_threshold': da_local > self.da_quantum_threshold
        }
        
        return factors
    
    def get_coincidence_factor(self, calcium_spike_time: float, 
                               current_time: float) -> float:
        """
        Calculate coincidence factor for STDP-like learning.
        
        Based on Yagishita et al. 2014 Science:
        Dopamine must arrive within specific window after calcium.
        
        Args:
            calcium_spike_time: Time of calcium spike (s)
            current_time: Current simulation time (s)
            
        Returns:
            Coincidence factor (0-1)
        """
        time_diff = current_time - calcium_spike_time
        
        if time_diff < 0:
            # Dopamine before calcium - no learning
            return 0.0
        elif time_diff < self.da_calcium_delay:
            # Too early - dopamine before optimal window
            return 0.5
        elif time_diff <= self.quantum_window:
            # Optimal window for quantum coherence
            return 1.0
        else:
            # Too late - coherence already decayed
            return np.exp(-(time_diff - self.quantum_window) / 0.1)
    
    def calculate_emergent_selectivity(self, i: int, j: int, 
                                       ca_local: float, po4_local: float) -> Dict[str, float]:
        """
        Calculate emergent dimer/trimer selectivity based on dopamine's 
        physical effects on the local environment.
        
        Key mechanisms:
        1. D2 activation reduces calcium influx
        2. Changes Ca/P ratio, affecting which phase forms
        3. Local ionic strength changes affect nucleation
        
        Args:
            i, j: Grid position
            ca_local: Local calcium concentration (M)
            po4_local: Local phosphate concentration (M)
            
        Returns:
            Dictionary with selectivity factors
        """
        da_local = self.dopamine_concentration[i, j]
        d2_occ = self.d2_occupancy[i, j]
        
        # 1. D2 reduces calcium (via Gi/o → decreased Ca channel opening)
        ca_effective = ca_local * (1 - 0.3 * d2_occ)  # 30% reduction at full D2
        
        # 2. Calculate feasibility based on stoichiometry
        # Dimers need Ca6(PO4)4, Trimers need Ca9(PO4)6
        if d2_occ > 0.5:
            # With less calcium, smaller clusters (dimers) are favored
            dimer_feasibility = min(ca_effective / (6 * 50e-9), 1.0)
            trimer_feasibility = min(ca_effective / (9 * 50e-9), 1.0)
        else:
            dimer_feasibility = 1.0
            trimer_feasibility = 1.0
        
        # 3. Ionic strength effects
        # Higher ionic strength reduces Debye length → affects larger clusters more
        ionic_strength_factor = 1 + 0.5 * d2_occ
        
        # Trimers have more P-P interactions, more affected by screening
        dimer_ionic_penalty = 1.0 / (1 + 0.1 * ionic_strength_factor)
        trimer_ionic_penalty = 1.0 / (1 + 0.3 * ionic_strength_factor)
        
        # 4. Combine effects
        dimer_factor = dimer_feasibility * dimer_ionic_penalty
        trimer_factor = trimer_feasibility * trimer_ionic_penalty
        
        return {
            'dimer_factor': dimer_factor,
            'trimer_factor': trimer_factor,
            'ca_effective': ca_effective,
            'd2_occupancy': d2_occ,
            'emergent_selectivity': dimer_factor / trimer_factor if trimer_factor > 0 else 10.0
        }
    
    def calculate_spatiotemporal_gradient(self) -> np.ndarray:
        """
        Calculate spatial gradient of dopamine for directional effects.
        High gradients might indicate learning-relevant locations.
        
        Returns:
            Gradient magnitude field
        """
        grad_x = np.gradient(self.dopamine_concentration, axis=1) / self.dx
        grad_y = np.gradient(self.dopamine_concentration, axis=0) / self.dx
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude
    
    # =========================================================================
    # METRICS AND RESET
    # =========================================================================
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """Return metrics for experimental validation"""
        return {
            'dopamine_mean_nM': np.mean(self.dopamine_concentration) * 1e9,
            'dopamine_max_nM': np.max(self.dopamine_concentration) * 1e9,
            'dopamine_peak_nM': np.max(self.dopamine_concentration) * 1e9,
            'd1_occupancy_mean': np.mean(self.d1_occupancy),
            'd1_occupancy_peak': np.max(self.d1_occupancy),
            'd2_occupancy_mean': np.mean(self.d2_occupancy),
            'd2_occupancy_peak': np.max(self.d2_occupancy),
            'calcium_modulation_mean': np.mean(self.get_calcium_modulation()),
            'dimer_bias_mean': np.mean(self.get_dimer_bias()),
            'vesicles_released': self.vesicles_released,
            'total_molecules': self.total_molecules_released,
        }
    
    def reset(self):
        """Reset to baseline state"""
        self.dopamine_concentration[:] = self.params.dopamine_tonic
        self.d1_occupancy[:] = 0
        self.d2_occupancy[:] = 0
        self.release_timer = 0
        self.vesicles_released = 0
        self.total_molecules_released = 0


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_dopamine_system(grid_shape: Tuple[int, int], dx: float,
                           params: Model6Parameters,
                           release_sites: Optional[List[Tuple[int, int]]] = None) -> DopamineSystemAdapter:
    """
    Create a dopamine system with Model 6 parameters.
    
    Args:
        grid_shape: Grid dimensions
        dx: Grid spacing in meters
        params: Model6Parameters instance
        release_sites: Optional release site positions
        
    Returns:
        Configured DopamineSystemAdapter instance
    """
    return DopamineSystemAdapter(grid_shape, dx, params, release_sites)