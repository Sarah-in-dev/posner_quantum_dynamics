# src/models/dynamic_nanoreactor_model.py
# Complete implementation with all dependencies

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
import logging
from scipy.ndimage import maximum_filter
from datetime import datetime
from pathlib import Path
import json
import h5py


logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Container for simulation results"""
    model_version: str = "4.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict = field(default_factory=dict)
    
    # Time series data
    time: np.ndarray = None
    calcium_map: np.ndarray = None
    posner_map: np.ndarray = None
    pnc_map: np.ndarray = None  # NEW: PNC tracking
    channel_states: np.ndarray = None
    
    # Summary metrics
    peak_posner: float = 0.0
    mean_posner: float = 0.0
    peak_pnc: float = 0.0
    mean_pnc: float = 0.0
    hotspot_lifetime: float = 0.0
    spatial_heterogeneity: float = 0.0
    
    # Mass balance metrics
    calcium_conservation: float = 1.0
    phosphate_conservation: float = 1.0
    
    # Template metrics
    templates_occupied: int = 0
    fusion_events: int = 0
    
    def save(self, filepath: Path):
        """Save results to HDF5 and JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save arrays to HDF5
        with h5py.File(f"{filepath}.h5", 'w') as f:
            f.attrs['model_version'] = self.model_version
            f.attrs['timestamp'] = self.timestamp
            
            if self.time is not None:
                f.create_dataset('time', data=self.time, compression='gzip')
            if self.calcium_map is not None:
                f.create_dataset('calcium_map', data=self.calcium_map, compression='gzip')
            if self.posner_map is not None:
                f.create_dataset('posner_map', data=self.posner_map, compression='gzip')
            if self.pnc_map is not None:
                f.create_dataset('pnc_map', data=self.pnc_map, compression='gzip')
            if self.channel_states is not None:
                f.create_dataset('channel_states', data=self.channel_states, compression='gzip')
        
        # Save metadata to JSON
        metadata = {
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': {
                'peak_posner': float(self.peak_posner),
                'mean_posner': float(self.mean_posner),
                'peak_pnc': float(self.peak_pnc),
                'mean_pnc': float(self.mean_pnc),
                'hotspot_lifetime': float(self.hotspot_lifetime),
                'spatial_heterogeneity': float(self.spatial_heterogeneity),
                'calcium_conservation': float(self.calcium_conservation),
                'phosphate_conservation': float(self.phosphate_conservation),
                'templates_occupied': int(self.templates_occupied),
                'fusion_events': int(self.fusion_events)
            }
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path):
        """Load results from saved files"""
        filepath = Path(filepath)
        
        # Load metadata
        with open(f"{filepath}.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        results = cls(
            model_version=metadata['model_version'],
            timestamp=metadata['timestamp'],
            parameters=metadata['parameters']
        )
        
        # Load metrics
        for key, value in metadata['metrics'].items():
            setattr(results, key, value)
        
        # Load arrays
        with h5py.File(f"{filepath}.h5", 'r') as f:
            if 'time' in f:
                results.time = f['time'][:]
            if 'calcium_map' in f:
                results.calcium_map = f['calcium_map'][:]
            if 'posner_map' in f:
                results.posner_map = f['posner_map'][:]
            if 'pnc_map' in f:
                results.pnc_map = f['pnc_map'][:]
            if 'channel_states' in f:
                results.channel_states = f['channel_states'][:]
        
        return results

# ============================================================================
# CHANNEL DYNAMICS 
# ============================================================================

class ChannelDynamics:
    """Markov model for stochastic calcium channel gating"""
    
    # Channel states
    CLOSED = 0
    OPEN = 1
    INACTIVATED = 2
    
    def __init__(self, n_channels: int = 6, 
                 open_rate: float = 100.0,  # Hz
                 close_rate: float = 50.0,   # Hz
                 inactivation_rate: float = 10.0,  # Hz
                 recovery_rate: float = 20.0):  # Hz
        
        self.n_channels = n_channels
        
        # Transition rates (Hz)
        self.rates = {
            'open': open_rate,
            'close': close_rate,
            'inactivate': inactivation_rate,
            'recover': recovery_rate
        }
        
        # Build transition probability matrix
        self.build_transition_matrix()
        
        # Initialize all channels as closed
        self.states = np.zeros(n_channels, dtype=int)
        
    def build_transition_matrix(self):
        """Build the transition probability matrix for dt=1ms"""
        dt = 0.001  # 1 ms timestep
        
        # Q-matrix (rate matrix)
        Q = np.zeros((3, 3))
        
        # From CLOSED
        Q[self.CLOSED, self.OPEN] = self.rates['open']
        Q[self.CLOSED, self.CLOSED] = -(self.rates['open'])
        
        # From OPEN
        Q[self.OPEN, self.CLOSED] = self.rates['close']
        Q[self.OPEN, self.INACTIVATED] = self.rates['inactivate']
        Q[self.OPEN, self.OPEN] = -(self.rates['close'] + self.rates['inactivate'])
        
        # From INACTIVATED
        Q[self.INACTIVATED, self.CLOSED] = self.rates['recover']
        Q[self.INACTIVATED, self.INACTIVATED] = -self.rates['recover']
        
        # Convert to probability matrix: P = I + Q*dt
        self.P = np.eye(3) + Q * dt
        
        # Ensure probabilities are valid
        self.P = np.clip(self.P, 0, 1)
        
        # Normalize rows
        for i in range(3):
            row_sum = self.P[i].sum()
            if row_sum > 0:
                self.P[i] /= row_sum
    
    def update(self, dt: float, depolarized: bool = True) -> np.ndarray:
        """Update channel states stochastically"""
        
        if not depolarized:
            # All channels close during hyperpolarization
            self.states[:] = self.CLOSED
            return self.states
        
        # Update each channel independently
        for i in range(self.n_channels):
            current_state = self.states[i]
            
            # Sample next state from transition probabilities
            probs = self.P[current_state]
            self.states[i] = np.random.choice(3, p=probs)
        
        return self.states
    
    def get_open_channels(self) -> np.ndarray:
        """Return boolean array of open channels"""
        return self.states == self.OPEN

# ============================================================================
# COMPLETE PARAMETERS
# ============================================================================

@dataclass
class DynamicParameters:
    """
    Complete parameters for PNC-based Posner formation
    """
    # ============= SPATIAL PARAMETERS =============
    grid_size: int = 50  # Reduced for faster testing
    active_zone_radius: float = 200e-9  # 200 nm
    cleft_width: float = 20e-9  # 20 nm
    
    # ============= CHANNEL PARAMETERS =============
    n_channels: int = 6
    channel_current: float = 0.3e-12  # 0.3 pA
    channel_open_rate: float = 100.0  # Hz
    channel_close_rate: float = 50.0  # Hz
    
    # ============= BASELINE CONCENTRATIONS =============
    ca_baseline: float = 100e-9  # 100 nM resting
    po4_baseline: float = 1e-3  # 1 mM
    
    # ============= DIFFUSION COEFFICIENTS =============
    D_calcium: float = 220e-12  # m²/s
    D_phosphate: float = 280e-12  # m²/s
    D_pnc: float = 100e-12  # m²/s (slower than ions)
    D_posner: float = 50e-12  # m²/s (larger, slower)
    
    # ============= PNC DYNAMICS =============
    # Complex formation (Ca + HPO4 ⇌ CaHPO4)
    k_complex_formation: float = 1e8  # M⁻¹s⁻¹ (fast) (was 1e7)
    k_complex_dissociation: float = 100.0  # s⁻¹ (was 1e3)
    
    # PNC formation/dissolution
    k_pnc_formation: float = 1000.0  # s⁻¹ (complex → PNC)(was 1e3)
    k_pnc_dissolution: float = 10.0  # s⁻¹ (gives ~50ms lifetime)(was 20)
    pnc_size: int = 30  # Ca atoms per PNC
    pnc_max_concentration: float = 1e-6  # M saturation
    
    # ============= TEMPLATE PARAMETERS =============
    templates_per_synapse: int = 500  # Total synaptotagmin molecules
    n_binding_sites: int = 3  # PNC binding sites per template
    k_pnc_binding: float = 1e8  # M⁻¹s⁻¹ (was 1e6)
    k_pnc_unbinding: float = 1.0  # s⁻¹ (was 10)
    
    # Fusion kinetics (SLOW!)
    k_fusion_attempt: float = 10.0  # s⁻¹ attempt rate (was 10.0)
    fusion_probability: float = 0.1  # 1% success per attempt (was 0.01)
    
    # ============= STOICHIOMETRY =============
    ca_per_posner: int = 9  # Ca₉(PO₄)₆
    po4_per_posner: int = 6
    pnc_per_posner: int = 3  # Approximate
    
    # ============= OTHER PARAMETERS =============
    kr_posner: float = 0.5  # s⁻¹ Posner dissolution
    f_hpo4_ph73: float = 0.61  # Fraction as HPO4²⁻
    kappa_buffering: float = 20  # Ca buffering
    max_calcium: float = 500e-6  # 500 μM max
    max_posner: float = 1e-3  # 1 mM max
    
    # Template factors
    template_factor: float = 10.0
    template_range: float = 10e-9  # 10 nm
    
    # Enhancement factors
    confinement_factor: float = 5.0
    electrostatic_factor: float = 3.0
    
    # Temporal
    dt: float = 0.0001  # 100 μs
    save_interval: int = 10
    noise_amplitude: float = 0.01

# ============================================================================
# COMPLETE DYNAMIC NANOREACTOR
# ============================================================================

class DynamicNanoreactor:
    """
    Complete implementation with PNC dynamics and mass balance
    """
    
    def __init__(self, params: DynamicParameters = None):
        self.params = params or DynamicParameters()
        
        # Setup spatial grid
        self.setup_spatial_grid()
        
        # Initialize channel dynamics
        self.channels = ChannelDynamics(
            n_channels=self.params.n_channels,
            open_rate=self.params.channel_open_rate,
            close_rate=self.params.channel_close_rate
        )
        
        # Position elements
        self.position_channels()
        self.position_templates()
        
        # Initialize all chemical fields
        self.initialize_fields()
        
        logger.info(f"Initialized complete PNC dynamics model")
    
    def setup_spatial_grid(self):
        """Create spatial discretization"""
        n = self.params.grid_size
        r = self.params.active_zone_radius
        
        # Create grid
        x = np.linspace(-r, r, n)
        y = np.linspace(-r, r, n)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Grid spacing
        self.dx = 2 * r / (n - 1)
        
        # Mask for active zone (circular)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.active_mask = self.R <= r
        
        self.grid_shape = (n, n)
    
    def position_channels(self):
        """Position channels in hexagonal array"""
        n = self.params.n_channels
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = self.params.active_zone_radius * 0.5
        
        self.channel_positions = []
        self.channel_indices = []
        
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            self.channel_positions.append((x, y))
            
            # Find nearest grid point
            i = np.argmin(np.abs(self.X[0, :] - x))
            j = np.argmin(np.abs(self.Y[:, 0] - y))
            self.channel_indices.append((j, i))
    
    def position_templates(self):
        """Position templates between channels"""
        n = self.params.n_channels
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n
        radius = self.params.active_zone_radius * 0.4
        
        self.template_positions = []
        self.template_indices = []
        
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            self.template_positions.append((x, y))
            
            i = np.argmin(np.abs(self.X[0, :] - x))
            j = np.argmin(np.abs(self.Y[:, 0] - y))
            self.template_indices.append((j, i))
    
    def initialize_fields(self):
        """Initialize all chemical concentration fields"""
        # Ion fields
        self.calcium_field = np.ones(self.grid_shape) * self.params.ca_baseline
        self.phosphate_field = np.ones(self.grid_shape) * self.params.po4_baseline
        
        # Intermediate species
        self.complex_field = np.zeros(self.grid_shape)  # CaHPO4
        self.pnc_field = np.zeros(self.grid_shape)  # Prenucleation clusters
        
        # Final product
        self.posner_field = np.zeros(self.grid_shape)
        
        # Template tracking
        self.template_occupancy = np.zeros(self.grid_shape)
        self.template_pnc_bound = np.zeros(self.grid_shape)
        self.template_fusion_timer = np.zeros(self.grid_shape)

        # CRITICAL: Set initial PNC concentration to kickstart the system
        # In reality, there's always some baseline PNC concentration
        self.pnc_field = np.ones(self.grid_shape) * 1e-10  # 0.1 nM baseline
        
        # For mass balance
        self.total_ca_initial = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        self.total_po4_initial = np.sum(self.phosphate_field * self.active_mask) * self.dx * self.dx
    
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """Calculate calcium field from open channels"""
        ca_field = np.ones(self.grid_shape) * self.params.ca_baseline
        
        for idx, (j, i) in enumerate(self.channel_indices):
            if channel_open[idx]:
                # Distance from channel
                dist = np.sqrt((self.X - self.X[j, i])**2 + (self.Y - self.Y[j, i])**2)
                dist = np.maximum(dist, self.dx/100)  # Avoid singularity, smaller minimum for stronger peak
                
                # Buffered diffusion
                kappa = 5 # Effective buffering factor
                D_eff = self.params.D_calcium / (1 + kappa)
                
                # Point source solution
                i_channel = self.params.channel_current
                F = 96485  # Faraday constant
                flux = i_channel / (2 * F)  # Ca²⁺ flux
                
                # Concentration profile
                ca_contribution = 2 * flux / (4 * np.pi * D_eff * dist) # Added factor of 2 for Ca²⁺
                ca_field += ca_contribution * self.active_mask
        
        # Apply maximum
        ca_field = np.minimum(ca_field, self.params.max_calcium)
        return ca_field
    
    # Add these methods to the DynamicNanoreactor class:

    def get_template_enhancement(self, position: Tuple[int, int]) -> float:
        """
        Check if position is near a template protein.
    
        Args:
            position: (i, j) grid indices
        
        Returns:
            Enhancement factor (template_factor if near template, 1.0 otherwise)
       """
        i, j = position
    
        # Check distance to each template
        for ti, tj in self.template_indices:
            # Calculate distance in real space
            dist = np.sqrt((self.X[i, j] - self.X[ti, tj])**2 + 
                          (self.Y[i, j] - self.Y[ti, tj])**2)
        
            # If within template range, return enhancement factor
            if dist < self.params.template_range:
                return self.params.template_factor  # Typically 10×
    
        return 1.0  # No enhancement if not near template

    def is_near_membrane(self, position: Tuple[int, int]) -> bool:
        """
        Check if position is near the membrane for electrostatic enhancement.
    
        Args:
            position: (i, j) grid indices
        
        Returns:
            True if within Debye length of membrane edge
        """
        i, j = position
    
        # Distance from center
        r = np.sqrt(self.X[i, j]**2 + self.Y[i, j]**2)
    
        # Distance from edge of active zone
        membrane_distance = self.params.active_zone_radius - r
    
        # Debye length in physiological conditions ~1 nm
        debye_length = 1e-9
    
        # Near membrane if within Debye length but still in active zone
        return 0 < membrane_distance < debye_length

    def calculate_posner_formation_rate(self, position: Tuple[int, int]) -> float:
        """Calculate formation rate with debug output"""
        i, j = position
    
        # Get local concentrations
        ca_local = self.calcium_field[i, j]
        po4_local = self.phosphate_field[i, j] * self.params.f_hpo4_ph73
    
        # Check template
        template_enh = self.get_template_enhancement(position)
    
        # Calculate complex
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
        c_complex = K_eq * ca_local * po4_local / (1 + K_eq * (ca_local + po4_local))
    
        # Calculate PNC concentration
        c_pnc = self.pnc_field[i, j]  # Use actual PNC field value
    
        # Debug output
        if template_enh > 1.0:
            print(f"  Complex: {c_complex*1e9:.3f} nM")
            print(f"  PNC: {c_pnc*1e9:.3f} nM")
            print(f"  Template bound: {self.template_pnc_bound[i, j]:.2f}/{self.params.n_binding_sites}")
    
        if template_enh <= 1.0:
            return 0.0  # No formation without template
    
        # Check if template has bound PNCs
        if self.template_pnc_bound[i, j] >= self.params.n_binding_sites * 0.5:
            # Can form Posner
            rate = self.params.k_fusion_attempt * self.params.fusion_probability
        
            # Convert to M/s
            volume = self.dx * self.dx * self.params.cleft_width
            rate_M_s = rate / (6.022e23 * volume * 1e3)
        
            # Apply enhancements
            rate_M_s *= self.params.confinement_factor
            if self.is_near_membrane(position):
                rate_M_s *= self.params.electrostatic_factor
        
            return rate_M_s
    
        return 0.0

    def calculate_metrics(self, posner_history: List[np.ndarray], 
                         channel_history: List[np.ndarray]) -> Dict:
        """
        Calculate summary metrics from simulation history.
    
        Args:
             posner_history: List of Posner concentration snapshots
             channel_history: List of channel state snapshots
        
         Returns:
            Dictionary of metrics
        """
        posner_arr = np.array(posner_history) if len(posner_history) > 0 else np.array([0])
        channel_arr = np.array(channel_history) if len(channel_history) > 0 else np.array([[0]])
    
        # Calculate metrics
        peak_posner = np.max(posner_arr) * 1e9  # Convert to nM
        mean_posner = np.mean(posner_arr[posner_arr > 0]) * 1e9 if np.any(posner_arr > 0) else 0
    
        # Spatial heterogeneity
        if len(posner_arr.shape) > 2:  # If we have spatial data
            final_field = posner_arr[-1] if len(posner_arr) > 0 else np.zeros(self.grid_shape)
            if np.mean(final_field) > 0:
                spatial_heterogeneity = np.std(final_field) / np.mean(final_field)
            else:
                spatial_heterogeneity = 0
        else:
            spatial_heterogeneity = 0
    
        # Channel statistics
        if len(channel_arr.shape) > 1:
            channel_open_fraction = np.mean(channel_arr == 1)
        else:
            channel_open_fraction = 0
    
        return {
            'peak_posner': peak_posner,
            'mean_posner': mean_posner,
            'spatial_heterogeneity': spatial_heterogeneity,
            'channel_open_fraction': channel_open_fraction,
            'templates_occupied': np.sum(self.template_pnc_bound > 0),
            'pnc_peak': np.max(self.pnc_field) * 1e9,
            'pnc_mean': np.mean(self.pnc_field[self.pnc_field > 0]) * 1e9 if np.any(self.pnc_field > 0) else 0
    }
    
    
    def calculate_complex_equilibrium(self):
        """Fast equilibrium for CaHPO4 complex formation"""
        # pH-corrected phosphate
        po4_eff = self.phosphate_field * self.params.f_hpo4_ph73
        
        # Equilibrium constant
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
        
        # Simple approximation (can be refined)
        self.complex_field = K_eq * self.calcium_field * po4_eff / (1 + K_eq * (self.calcium_field + po4_eff))
        self.complex_field *= self.active_mask
    
    def update_pnc_dynamics(self, dt: float):
        """Update PNC field with formation, dissolution, and binding"""
        
        # Calculate complex concentration first
        self.calculate_complex_equilibrium()

        # Formation from complexes
        formation = self.params.k_pnc_formation * self.complex_field
        
        # Saturation
        #saturation = 1 - self.pnc_field / self.params.pnc_max_concentration
        #saturation = np.clip(saturation, 0, 1)
        #formation *= saturation
        
        # Dissolution
        dissolution = self.params.k_pnc_dissolution * self.pnc_field
        
        # Template binding at template locations
        binding = np.zeros(self.grid_shape)
        for ti, tj in self.template_indices:
            # Always allow some binding if PNCs are present
            if self.pnc_field[ti, tj] > 1e-12:  # If there are PNCs
                # Simple binding kinetics
                bind_rate = self.params.k_pnc_binding * self.pnc_field[ti, tj] * 1e-7  # Scale Factor
                binding[ti, tj] = bind_rate
                # Accumulate bound PNCs
                self.template_pnc_bound[ti, tj] += bind_rate * dt

                # Cap at maximum binding sites
            self.template_pnc_bound[ti, tj] = min(
                self.template_pnc_bound[ti, tj], 
                self.params.n_binding_sites
            )
        
        # Update PNC field
        dpnc_dt = formation - dissolution - binding
        self.pnc_field += dpnc_dt * dt * self.active_mask
        self.pnc_field = np.maximum(self.pnc_field, 0)

        # Add small diffusion
        laplacian = self.calculate_laplacian_neumann(self.pnc_field)
        self.pnc_field += self.params.D_pnc * laplacian * dt
    
    def update_posner_formation(self, dt: float):
        """Slow Posner formation at occupied templates"""
        for ti, tj in self.template_indices:
            # Check occupancy
            if self.template_pnc_bound[ti, tj] >= self.params.n_binding_sites:
                # Increment timer
                self.template_fusion_timer[ti, tj] += dt
                
                # Attempt fusion
                if self.template_fusion_timer[ti, tj] > 1.0 / self.params.k_fusion_attempt:
                    self.template_fusion_timer[ti, tj] = 0
                    
                    # Stochastic fusion
                    if np.random.random() < self.params.fusion_probability:
                        # Form Posner!
                        volume = self.dx * self.dx * self.params.cleft_width
                        conc_increase = 1 / (6.022e23 * volume * 1e3)
                        self.posner_field[ti, tj] += conc_increase
                        
                        # Clear template
                        self.template_pnc_bound[ti, tj] = 0
                        
                        # Consume PNCs
                        self.pnc_field[ti, tj] -= conc_increase * self.params.pnc_per_posner
                        self.pnc_field[ti, tj] = max(0, self.pnc_field[ti, tj])
    
    def calculate_laplacian_neumann(self, field: np.ndarray) -> np.ndarray:
        """Laplacian with no-flux boundaries"""
        laplacian = np.zeros_like(field)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # Boundaries (no flux)
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
        
        return laplacian
    
    def update_fields(self, dt: float, channel_open: np.ndarray):
        """Main update routine"""
        # Update calcium from channels
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        
        # Complex equilibrium
        self.calculate_complex_equilibrium()
        
        # PNC dynamics
        self.update_pnc_dynamics(dt)
        
        # Posner formation
        self.update_posner_formation(dt)
        
        # Posner dissolution
        dissolution = self.params.kr_posner * self.posner_field * dt
        self.posner_field -= dissolution
        self.posner_field = np.maximum(self.posner_field, 0)
        
        # Posner diffusion
        posner_laplacian = self.calculate_laplacian_neumann(self.posner_field)
        self.posner_field += self.params.D_posner * posner_laplacian * dt
        
        # Apply constraints
        self.posner_field *= self.active_mask
        self.posner_field = np.minimum(self.posner_field, self.params.max_posner)
    
    def simulate(self, duration: float = 1.0, stimulus_times: List[float] = None) -> SimulationResults:
        """Run simulation and return SimulationResults object"""
        n_steps = int(duration / self.params.dt)
        save_steps = list(range(0, n_steps, self.params.save_interval))
        n_saves = len(save_steps)
    
        # Initialize results
        results = SimulationResults(
            parameters=asdict(self.params)
        )
    
        # Allocate storage
        results.time = np.array(save_steps) * self.params.dt
        results.posner_map = np.zeros((n_saves, *self.grid_shape))
        results.pnc_map = np.zeros((n_saves, *self.grid_shape))
        results.calcium_map = np.zeros((n_saves, *self.grid_shape))
        results.channel_states = np.zeros((n_saves, self.params.n_channels))
    
        # Track fusion events
        fusion_count = 0
    
        # Determine when channels are open
        if stimulus_times is None:
            depolarized = np.ones(n_steps, dtype=bool)
        else:
            depolarized = np.zeros(n_steps, dtype=bool)
            for t in stimulus_times:
                idx = int(t / self.params.dt)
                depolarized[idx:idx+int(0.005/self.params.dt)] = True
    
        # Main simulation loop
        save_idx = 0
        for step in range(n_steps):
            # Update channels
            self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
        
            # Update fields
            self.update_fields(self.params.dt, channel_open)
        
            # Save periodically
            if step in save_steps:
                results.posner_map[save_idx] = self.posner_field
                results.pnc_map[save_idx] = self.pnc_field
                results.calcium_map[save_idx] = self.calcium_field
                results.channel_states[save_idx] = self.channels.states
                save_idx += 1
            
                if step % 1000 == 0:
                    logger.info(f"Step {step}: Posner={np.max(self.posner_field)*1e9:.1f} nM, "
                               f"PNC={np.max(self.pnc_field)*1e9:.1f} nM")
    
        # Calculate final metrics
        results.peak_posner = np.max(results.posner_map) * 1e9  # Convert to nM
        results.mean_posner = np.mean(results.posner_map[results.posner_map > 0]) * 1e9
        results.peak_pnc = np.max(results.pnc_map) * 1e9
        results.mean_pnc = np.mean(results.pnc_map[results.pnc_map > 0]) * 1e9
        results.templates_occupied = np.sum(self.template_pnc_bound > 0)
        results.fusion_events = fusion_count
    
        # Check mass balance
        final_ca_total = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        results.calcium_conservation = final_ca_total / self.total_ca_initial if self.total_ca_initial > 0 else 1.0
    
        logger.info(f"Simulation complete. Peak Posner: {results.peak_posner:.1f} nM")
    
        return results
    
# ============================================================================
# ANALYSIS FUNCTIONS (module level, outside classes)
# ============================================================================

def analyze_spatial_pattern(results: SimulationResults) -> Dict:
    """Analyze spatial patterns in Posner distribution"""
    
    # Get final Posner field
    final_posner = results.posner_map[-1] if len(results.posner_map.shape) > 2 else results.posner_map
    
    # Find hotspots
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(final_posner, size=5)
    hotspots = (final_posner == local_max) & (final_posner > 0.5 * np.max(final_posner))
    
    n_hotspots = np.sum(hotspots)
    
    # Calculate radial profile
    n = final_posner.shape[0]
    center = n // 2
    Y, X = np.ogrid[:n, :n]
    r = np.sqrt((X - center)**2 + (Y - center)**2)
    
    # Bin by radius
    r_bins = np.linspace(0, n//2, 20)
    radial_profile = []
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(mask):
            radial_profile.append(np.mean(final_posner[mask]))
        else:
            radial_profile.append(0)
    
    return {
        'n_hotspots': n_hotspots,
        'radial_profile': radial_profile,
        'max_location': np.unravel_index(np.argmax(final_posner), final_posner.shape)
    }

def analyze_temporal_dynamics(results: SimulationResults) -> Dict:
    """Analyze temporal dynamics of Posner formation"""
    
    # Handle different data shapes
    if len(results.posner_map.shape) == 3:
        # 3D: [time, x, y]
        total_posner = np.sum(results.posner_map, axis=(1, 2))
    else:
        # Assume 1D time series
        total_posner = results.posner_map
    
    # Find rise time
    max_val = np.max(total_posner)
    if max_val > 0:
        t10 = np.where(total_posner > 0.1 * max_val)[0]
        t90 = np.where(total_posner > 0.9 * max_val)[0]
        
        if len(t10) > 0 and len(t90) > 0:
            rise_time = (t90[0] - t10[0]) * 0.001  # Convert to seconds
        else:
            rise_time = 0
    else:
        rise_time = 0
    
    # Find peak time
    peak_idx = np.argmax(total_posner) if len(total_posner) > 0 else 0
    
    return {
        'rise_time': rise_time,
        'peak_time': peak_idx * 0.001,  # Convert to seconds
        'total_accumulated': np.sum(total_posner)
    }