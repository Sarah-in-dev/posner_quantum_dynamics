# src/models/dynamic_nanoreactor_model.py
"""
Model 4: Dynamic Nanoreactor with Stochastic Channel Gating
Builds on Model 3 by adding temporal dynamics and stochasticity

CORRECTED VERSION - Key changes marked with # CHANGED
"""

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
import json
import h5py
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# RESULTS CONTAINER - NO CHANGES NEEDED
# ============================================================================

@dataclass
class SimulationResults:
    """Standardized results container for Model 4"""
    model_version: str = "4.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict = field(default_factory=dict)
    
    # Time series data
    time: np.ndarray = None
    calcium_map: np.ndarray = None  # 3D: time x space x space
    posner_map: np.ndarray = None   # 3D: time x space x space
    channel_states: np.ndarray = None  # 2D: time x n_channels
    
    # Summary metrics
    peak_posner: float = 0.0
    mean_posner: float = 0.0
    hotspot_lifetime: float = 0.0
    spatial_heterogeneity: float = 0.0
    
    # Quantum metrics
    coherence_time: float = 0.0
    entanglement_range: float = 0.0
    
    # Stochastic metrics
    channel_open_fraction: float = 0.0
    mean_burst_duration: float = 0.0
    
    def save(self, filepath: Path):
        """Save results to HDF5 and JSON"""
        filepath = Path(filepath)
        
        # HDF5 for arrays
        with h5py.File(f"{filepath}.h5", 'w') as f:
            f.attrs['model_version'] = self.model_version
            f.attrs['timestamp'] = self.timestamp
            
            if self.time is not None:
                f.create_dataset('time', data=self.time, compression='gzip')
            if self.calcium_map is not None:
                f.create_dataset('calcium_map', data=self.calcium_map, compression='gzip')
            if self.posner_map is not None:
                f.create_dataset('posner_map', data=self.posner_map, compression='gzip')
            if self.channel_states is not None:
                f.create_dataset('channel_states', data=self.channel_states, compression='gzip')
        
        # JSON for metadata and scalars
        metadata = {
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': {
                'peak_posner': float(self.peak_posner),
                'mean_posner': float(self.mean_posner),
                'hotspot_lifetime': float(self.hotspot_lifetime),
                'spatial_heterogeneity': float(self.spatial_heterogeneity),
                'coherence_time': float(self.coherence_time),
                'entanglement_range': float(self.entanglement_range),
                'channel_open_fraction': float(self.channel_open_fraction),
                'mean_burst_duration': float(self.mean_burst_duration)
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
            if 'channel_states' in f:
                results.channel_states = f['channel_states'][:]
        
        return results

# ============================================================================
# CHANNEL DYNAMICS - NO CHANGES NEEDED
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
        
        # Track statistics
        self.open_times = []
        self.burst_durations = []
    
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
            self.P[i] /= self.P[i].sum()
    
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
    
    def get_statistics(self) -> Dict:
        """Calculate channel statistics"""
        return {
            'n_open': np.sum(self.states == self.OPEN),
            'n_closed': np.sum(self.states == self.CLOSED),
            'n_inactivated': np.sum(self.states == self.INACTIVATED),
            'open_fraction': np.mean(self.states == self.OPEN)
        }

# ============================================================================
# SPATIOTEMPORAL DYNAMICS - KEEP ALL PARAMETERS
# ============================================================================

@dataclass
class DynamicParameters:
    """
    Parameters for dynamic nanoreactor model
    All physics-based, no empirical rate constants!
    """
    
    # [KEEP ALL YOUR EXISTING PARAMETERS - they're already correct!]
    # ============= SPATIAL PARAMETERS =============
    grid_size: int = 100  # Grid points in each dimension
    active_zone_radius: float = 200e-9  # 200 nm
    cleft_width: float = 20e-9  # 20 nm (Zuber et al., 2005)
    
    # ============= PHYSICAL CONSTANTS =============
    temperature: float = 310  # K (37°C)
    kB: float = 1.38e-23  # Boltzmann constant (J/K)
    N_A: float = 6.022e23  # Avogadro's number (mol⁻¹)
    F: float = 96485  # Faraday constant (C/mol)
    R: float = 8.314  # Gas constant (J/mol·K)
    
    # ============= ION PARAMETERS =============
    # Diffusion coefficients (measured in cytoplasm)
    D_calcium: float = 220e-12  # m²/s (Allbritton et al., 1992)
    D_phosphate: float = 280e-12  # m²/s
    D_posner: float = 50e-12  # m²/s (larger molecule, slower)
    
    # Ionic radii for encounter distance
    r_calcium: float = 1.0e-10  # m
    r_phosphate: float = 2.4e-10  # m
    
    # Baseline concentrations
    ca_baseline: float = 100e-9  # 100 nM (resting)
    po4_baseline: float = 1e-3  # 1 mM (physiological)
    
    # ============= POSNER CHEMISTRY =============
    # Solubility product for Ca₉(PO₄)₆
    Ksp_posner: float = 1e-58  # (Posner & Betts, 1975)
    
    # Surface tension for nucleation
    gamma_interface: float = 0.06  # J/m² (calcium phosphate/water)
    
    # Molar volume of Posner
    V_molar: float = 2.8e-4  # m³/mol
    
    # Dissolution/degradation rate
    kr_posner: float = 0.5  # s⁻¹ (2s lifetime)
    
    # ============= CHANNEL PARAMETERS =============
    n_channels: int = 6  # Hexagonal array
    
    # Single channel properties (measured)
    channel_current: float = 0.3e-12  # 0.3 pA (Schneggenburger & Neher, 2000)
    channel_conductance: float = 12e-12  # 12 pS
    
    # Gating kinetics (from patch clamp data)
    channel_open_rate: float = 100.0  # Hz
    channel_close_rate: float = 50.0  # Hz
    inactivation_rate: float = 10.0  # Hz  
    recovery_rate: float = 20.0  # Hz
    
    # ============= ENHANCEMENT FACTORS =============
    # All from literature or Model 3 calculations
    template_factor: float = 10.0  # Synaptotagmin nucleation (Hunter et al., 1996)
    confinement_factor: float = 5.0  # 2D vs 3D kinetics (Erdemir et al., 2009)
    electrostatic_factor: float = 3.0  # Membrane surface charge concentration
    
    # ============= QUANTUM PARAMETERS =============
    isotope: str = 'P31'  # 'P31' or 'P32'
    
    # Base coherence times (Fisher, 2015; Swift et al., 2018)
    t2_base_p31: float = 1.0  # seconds for ³¹P
    t2_base_p32: float = 0.1  # seconds for ³²P
    
    # Quantum thresholds
    critical_conc_nM: float = 100  # nM - concentration for significant decoherence
    entanglement_radius: float = 50e-9  # 50 nm
    max_enhancement: float = 2.0  # Maximum learning enhancement from quantum effects
    
    # Synaptic protection factor
    synaptic_protection: float = 1.5  # Protection from decoherence in cleft
    
    # ============= CLEARANCE/PUMPING =============
    k_clearance: float = 10.0  # s⁻¹ - calcium clearance rate
    k_pump: float = 5.0  # s⁻¹ - active pumping rate
    
    # ============= SAFETY LIMITS =============
    max_calcium: float = 500e-6  # 500 μM (toxic above this)
    max_posner: float = 1e-3  # 1 mM (precipitation limit)
    min_channel_distance: float = 10e-9  # 10 nm minimum separation
    
    # ============= TEMPORAL PARAMETERS =============
    dt: float = 0.0001  # 100 μs timestep
    save_interval: int = 10  # Save every 10 timesteps (1 ms)
    
    # ============= STOCHASTIC PARAMETERS =============
    noise_amplitude: float = 0.01  # Thermal noise amplitude

    # ============= PNC PHYSICS PARAMETERS =============
    # Ion association (from literature)
    K_association: float = 1e4  # M^-1, CaHPO4 association constant
    
    # PNC formation efficiency
    f_pnc_baseline: float = 0.1  # Fraction of complexes forming PNCs
    
    # Template binding
    k_pnc_binding: float = 1e8  # M^-1 s^-1, PNC binding to template
    k_pnc_fusion: float = 1e3  # s^-1, fusion rate of bound PNCs
    n_pnc_per_posner: int = 3  # PNCs needed to form one Posner
    
    # Template parameters
    template_binding_sites: int = 6  # Binding sites per template
    template_range: float = 10e-9  # 10 nm effective range
    K_half_pnc: float = 1e-8  # M, half-saturation for PNC binding
    hill_coefficient: float = 2.5  # Cooperativity of PNC binding
    
    # pH correction
    f_hpo4_ph73: float = 0.61  # Fraction of phosphate as HPO4^2- at pH 7.3
    
    # Buffering
    kappa_buffering: float = 20  # Calcium buffering capacity
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving"""
        return asdict(self)
    
    def get_effective_t2_base(self) -> float:
        """Get isotope-specific base coherence time"""
        if self.isotope == 'P31':
            return self.t2_base_p31
        elif self.isotope == 'P32':
            return self.t2_base_p32
        else:
            raise ValueError(f"Unknown isotope: {self.isotope}")
    
    def calculate_encounter_rate(self) -> float:
        """
        Calculate diffusion-limited encounter rate using Smoluchowski equation
        Returns rate constant in M⁻¹s⁻¹
        """
        # Sum of diffusion coefficients
        D_sum = self.D_calcium + self.D_phosphate
        
        # Encounter distance (sum of ionic radii)
        r_encounter = self.r_calcium + self.r_phosphate
        
        # Smoluchowski rate (m³/s)
        k_diff = 4 * np.pi * D_sum * r_encounter
        
        # Convert to M⁻¹s⁻¹
        k_encounter = k_diff * self.N_A
        
        return k_encounter
    
    def calculate_nucleation_barrier(self, supersaturation: float) -> float:
        """
        Calculate the nucleation energy barrier using classical nucleation theory
        
        Args:
            supersaturation: S = IAP/Ksp (dimensionless)
            
        Returns:
            Energy barrier in units of kT
        """
        if supersaturation <= 1:
            return np.inf  # No nucleation if undersaturated
        
        # Critical nucleus size
        ln_S = np.log(supersaturation)
        
        # Energy barrier (classical nucleation theory)
        # ΔG* = (16π γ³ v²) / (3 (kT ln S)²)
        # where v is molecular volume
        
        # Convert molar volume to molecular volume
        v_molecule = self.V_molar / self.N_A
        
        # Calculate barrier in Joules
        numerator = 16 * np.pi * (self.gamma_interface ** 3) * (v_molecule ** 2)
        denominator = 3 * ((self.kB * self.temperature * ln_S) ** 2)
        
        barrier_J = numerator / denominator
        
        # Convert to units of kT
        barrier_kT = barrier_J / (self.kB * self.temperature)
        
        return barrier_kT
    
    def calculate_nucleation_probability(self, supersaturation: float) -> float:
        """
        Calculate the probability of nucleation given supersaturation
        
        Args:
            supersaturation: S = IAP/Ksp
            
        Returns:
            Probability between 0 and 1
        """
        if supersaturation <= 1:
            return 0.0
        
        # Get energy barrier
        barrier_kT = self.calculate_nucleation_barrier(supersaturation)
        
        # Boltzmann factor
        if barrier_kT > 100:  # Avoid numerical underflow
            return 0.0
        else:
            return np.exp(-barrier_kT)
    
    def validate(self) -> bool:
        """
        Validate that all parameters are physically reasonable
        """
        checks = [
            self.grid_size > 0,
            self.active_zone_radius > 0,
            self.temperature > 0,
            self.D_calcium > 0,
            self.D_phosphate > 0,
            self.Ksp_posner > 0,
            self.n_channels > 0,
            self.dt > 0,
            self.isotope in ['P31', 'P32'],
            self.max_calcium > self.ca_baseline,
            self.critical_conc_nM > 0
        ]
        
        if not all(checks):
            raise ValueError("Invalid parameters detected")
        
        return True

# ============================================================================
# MAIN MODEL - THIS IS WHERE THE KEY CHANGES ARE
# ============================================================================

class DynamicNanoreactor:
    """
    Model 4: Dynamic nanoreactor with stochastic channel gating
    and spatiotemporal Posner dynamics
    """
    
    def __init__(self, params: DynamicParameters = None):
        self.params = params or DynamicParameters()
        
        # Initialize spatial grid
        self.setup_spatial_grid()
        
        # Initialize channel dynamics
        self.channels = ChannelDynamics(
            n_channels=self.params.n_channels,
            open_rate=self.params.channel_open_rate,
            close_rate=self.params.channel_close_rate
        )
        
        # Position channels in hexagonal array
        self.position_channels()
        
        # Position template proteins
        self.position_templates()
        
        # Initialize fields
        self.calcium_field = np.ones(self.grid_shape) * self.params.ca_baseline
        self.posner_field = np.zeros(self.grid_shape)
        
        logger.info(f"Initialized DynamicNanoreactor with {self.params.n_channels} channels")
    
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
        radius = self.params.active_zone_radius * 0.5  # Half radius
        
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
        """Position template proteins between channels"""
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
    
    # Add these methods to your DynamicNanoreactor class in dynamic_nanoreactor_model.py
    # Place them after the position_templates method and before update_fields

    def get_template_enhancement(self, position: Tuple[int, int]) -> float:
        """
        Check if position is near a template protein.
        Templates (e.g., synaptotagmin) are essential for Posner formation.
    
        Args:
            position: (i, j) grid indices
        
        Returns:
            Enhancement factor (>1 if near template, 1 otherwise)
        """
        i, j = position
    
        # Check distance to each template
        for ti, tj in self.template_indices:
            # Distance in real space
            dist = np.sqrt((self.X[i, j] - self.X[ti, tj])**2 + 
                      (self.Y[i, j] - self.Y[ti, tj])**2)
        
            # Templates have effective range ~10 nm
            if dist < self.params.template_range:
                return self.params.template_factor  # ~10×
    
        return 1.0  # No enhancement

    def is_near_membrane(self, position: Tuple[int, int]) -> bool:
        """
        Check if position is near the membrane (for electrostatic enhancement).
    
        Args:
            position: (i, j) grid indices
        
        Returns:
            True if within Debye length of membrane
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

    def calculate_laplacian_neumann(self, field: np.ndarray) -> np.ndarray:
        """
        Calculate Laplacian with Neumann (no-flux) boundary conditions.
        This is physically correct for the closed synaptic cleft.
    
        Args:
            field: 2D concentration field
        
        Returns:
            Laplacian of the field
        """
        laplacian = np.zeros_like(field)
    
        # Interior points: standard 5-point stencil
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] +      # bottom
            field[:-2, 1:-1] +     # top  
            field[1:-1, 2:] +      # right
            field[1:-1, :-2] -     # left
            4 * field[1:-1, 1:-1]  # center
        ) / (self.dx ** 2)
    
        # Boundaries: mirror condition (∂C/∂n = 0)
    
        # Top boundary (i=0)
        laplacian[0, 1:-1] = (
            2 * field[1, 1:-1] +   # use interior twice
            field[0, 2:] +         # right
            field[0, :-2] -        # left
            4 * field[0, 1:-1]     # center
        ) / (self.dx ** 2)
    
        # Bottom boundary (i=-1)
        laplacian[-1, 1:-1] = (
            2 * field[-2, 1:-1] +  # use interior twice
            field[-1, 2:] +        # right
            field[-1, :-2] -       # left
            4 * field[-1, 1:-1]    # center
        ) / (self.dx ** 2)
    
        # Left boundary (j=0)
        laplacian[1:-1, 0] = (
            field[2:, 0] +         # bottom
            field[:-2, 0] +        # top
            2 * field[1:-1, 1] -   # use interior twice
            4 * field[1:-1, 0]     # center
        ) / (self.dx ** 2)
    
        # Right boundary (j=-1)
        laplacian[1:-1, -1] = (
            field[2:, -1] +        # bottom
            field[:-2, -1] +       # top
            2 * field[1:-1, -2] -  # use interior twice
            4 * field[1:-1, -1]    # center
        ) / (self.dx ** 2)
    
        # Corners: average adjacent boundaries
        laplacian[0, 0] = (laplacian[0, 1] + laplacian[1, 0]) / 2
        laplacian[0, -1] = (laplacian[0, -2] + laplacian[1, -1]) / 2
        laplacian[-1, 0] = (laplacian[-1, 1] + laplacian[-2, 0]) / 2
        laplacian[-1, -1] = (laplacian[-1, -2] + laplacian[-2, -1]) / 2
    
        return laplacian


    # CHANGED: Fixed calcium microdomains to include buffering
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """Calculate calcium concentration field from open channels"""
        ca_field = np.ones(self.grid_shape) * self.params.ca_baseline
        
        for idx, (j, i) in enumerate(self.channel_indices):
            if channel_open[idx]:
                # Distance from this channel to all points
                dist = np.sqrt((self.X - self.X[j, i])**2 + 
                              (self.Y - self.Y[j, i])**2)
                
                # Avoid singularity
                dist = np.maximum(dist, self.dx/10)  # CHANGED: smaller minimum
                
                # CHANGED: Include buffering in effective diffusion
                kappa_B = self.params.kappa_buffering  # ~20
                D_eff = self.params.D_calcium / (1 + kappa_B)  # Reduced by buffering
                
                # Steady-state concentration from point source
                i_channel = self.params.channel_current
                F = 96485  # Faraday constant
                z_ca = 2  # Calcium charge
                
                # Add contribution from this channel
                flux = i_channel / (z_ca * F)
                ca_contribution = flux / (4 * np.pi * D_eff * dist)  # CHANGED: use D_eff
                ca_field += ca_contribution * self.active_mask
                
        # Apply maximum cap
        ca_field = np.minimum(ca_field, self.params.max_calcium)
        
        return ca_field
    
    # CHANGED: This now takes a position tuple, not the whole field!
    def calculate_posner_formation_rate(self, position: Tuple[int, int]) -> float:
        """
        Calculate PNC-based Posner formation at a SINGLE position.
        
        Args:
            position: (i, j) grid indices
            
        Returns:
            Formation rate in M/s at this position
        """
        i, j = position
        
        # Get local calcium concentration
        ca_local = self.calcium_field[i, j]
        
        # Effective phosphate (pH corrected)
        po4_eff = self.params.po4_baseline * self.params.f_hpo4_ph73
        
        # Step 1: CaHPO4 complex formation
        c_complex = self.params.K_association * ca_local * po4_eff
        
        # Step 2: PNC formation
        if ca_local > 1e-8:
            if ca_local > 1e-5:
                f_pnc = self.params.f_pnc_baseline * np.exp(-(ca_local - 1e-5)/1e-5)
            else:
                f_pnc = self.params.f_pnc_baseline
        else:
            f_pnc = 0
        
        c_pnc = c_complex * f_pnc
        
        # Step 3: Check for template
        template_enhancement = self.get_template_enhancement(position)
        
        if template_enhancement <= 1.0:
            return 0.0  # No template = no Posner
        
        # Step 4: Template-directed assembly
        if c_pnc > 1e-10:
            c_pnc_local = c_pnc * self.params.template_binding_sites
            K_half = self.params.K_half_pnc
            n_hill = self.params.hill_coefficient
            
            theta = (c_pnc_local**n_hill) / (K_half**n_hill + c_pnc_local**n_hill)
            rate = self.params.k_pnc_fusion * theta
        else:
            rate = 0
        
        # Step 5: Apply enhancements
        rate *= self.params.confinement_factor
        
        if self.is_near_membrane(position):
            rate *= self.params.electrostatic_factor
        
        # Convert to M/s
        volume_element = self.dx * self.dx * self.params.cleft_width
        volume_element_L = volume_element * 1000
        rate_M_s = rate / (6.022e23 * volume_element_L)
        
        return rate_M_s
    
    # CHANGED: Complete replacement of update_fields
    def update_fields(self, dt: float, channel_open: np.ndarray):
        """Update calcium and Posner fields with correct spatial iteration"""
        
        # Step 1: Update calcium field
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        
        # Step 2: Calculate formation rate at EACH grid point
        formation_rate_field = np.zeros(self.grid_shape)
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                if self.active_mask[i, j]:
                    formation_rate_field[i, j] = self.calculate_posner_formation_rate((i, j))
        
        # Step 3: Calculate diffusion with proper boundaries
        laplacian = self.calculate_laplacian_neumann(self.posner_field)
        
        # Step 4: Reaction-diffusion update
        dPosner_dt = (
            self.params.D_posner * laplacian +
            formation_rate_field -  # Now a 2D field!
            self.params.kr_posner * self.posner_field
        )
        
        # Step 5: Update
        self.posner_field += dPosner_dt * dt
        
        # Step 6: Add noise if needed
        if self.params.noise_amplitude > 0:
            noise = np.random.normal(0, self.params.noise_amplitude * np.sqrt(dt), 
                                   self.grid_shape)
            self.posner_field += noise * self.active_mask
        
        # Step 7: Apply constraints
        self.posner_field = np.maximum(self.posner_field, 0)
        self.posner_field = np.minimum(self.posner_field, self.params.max_posner)
        self.posner_field *= self.active_mask
    
    # NEW METHOD: Added for template checking
    def get_template_enhancement(self, position: Tuple[int, int]) -> float:
        """Check if position is near a template"""
        i, j = position
        
        for ti, tj in self.template_indices:
            dist = np.sqrt((self.X[i, j] - self.X[ti, tj])**2 + 
                          (self.Y[i, j] - self.Y[ti, tj])**2)
            
            if dist < self.params.template_range:
                return self.params.template_factor
        
        return 1.0
    
    # NEW METHOD: Added for membrane proximity
    def is_near_membrane(self, position: Tuple[int, int]) -> bool:
        """Check if near membrane for electrostatic enhancement"""
        i, j = position
        r = np.sqrt(self.X[i, j]**2 + self.Y[i, j]**2)
        membrane_distance = self.params.active_zone_radius - r
        debye_length = 1e-9
        return 0 < membrane_distance < debye_length
    
    # NEW METHOD: Added for proper boundary conditions
    def calculate_laplacian_neumann(self, field: np.ndarray) -> np.ndarray:
        """Calculate Laplacian with Neumann boundary conditions"""
        laplacian = np.zeros_like(field)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # Boundaries (no-flux)
        laplacian[0, 1:-1] = (2*field[1, 1:-1] + field[0, 2:] + field[0, :-2] - 4*field[0, 1:-1]) / (self.dx**2)
        laplacian[-1, 1:-1] = (2*field[-2, 1:-1] + field[-1, 2:] + field[-1, :-2] - 4*field[-1, 1:-1]) / (self.dx**2)
        laplacian[1:-1, 0] = (field[2:, 0] + field[:-2, 0] + 2*field[1:-1, 1] - 4*field[1:-1, 0]) / (self.dx**2)
        laplacian[1:-1, -1] = (field[2:, -1] + field[:-2, -1] + 2*field[1:-1, -2] - 4*field[1:-1, -1]) / (self.dx**2)
        
        # Corners
        laplacian[0, 0] = (laplacian[0, 1] + laplacian[1, 0]) / 2
        laplacian[0, -1] = (laplacian[0, -2] + laplacian[1, -1]) / 2
        laplacian[-1, 0] = (laplacian[-1, 1] + laplacian[-2, 0]) / 2
        laplacian[-1, -1] = (laplacian[-1, -2] + laplacian[-2, -1]) / 2
        
        return laplacian
    
    # [KEEP ALL OTHER METHODS: calculate_metrics, find_bursts, simulate, etc.]
    # They don't need changes!
    
    def calculate_metrics(self, posner_history: List[np.ndarray], 
                         channel_history: List[np.ndarray]) -> Dict:
        """Calculate summary metrics from simulation"""
        # [Keep existing implementation - no changes needed]
        posner_arr = np.array(posner_history)
        channel_arr = np.array(channel_history)
        
        peak_posner = np.max(posner_arr) * 1e9  # Convert to nM
        mean_posner = np.mean(posner_arr[posner_arr > 0]) * 1e9
        
        if mean_posner > 0:
            spatial_heterogeneity = np.std(posner_arr[-1]) / np.mean(posner_arr[-1])
        else:
            spatial_heterogeneity = 0
        
        threshold = 0.5 * np.max(posner_arr)
        above_threshold = np.any(posner_arr > threshold, axis=(1, 2))
        if np.any(above_threshold):
            hotspot_lifetime = np.sum(above_threshold) * self.params.dt * self.params.save_interval
        else:
            hotspot_lifetime = 0
        
        channel_open_fraction = np.mean(channel_arr == 1)
        
        open_any = np.any(channel_arr == 1, axis=1)
        bursts = self.find_bursts(open_any)
        if bursts:
            mean_burst_duration = np.mean([b[1] - b[0] for b in bursts]) * self.params.dt * self.params.save_interval
        else:
            mean_burst_duration = 0
        
        if peak_posner > 0:
            coherence_time = 1.0 / (1 + peak_posner / 100)
            entanglement_range = 50e-9 * np.exp(-peak_posner / 50)
        else:
            coherence_time = 0
            entanglement_range = 0
        
        return {
            'peak_posner': peak_posner,
            'mean_posner': mean_posner,
            'spatial_heterogeneity': spatial_heterogeneity,
            'hotspot_lifetime': hotspot_lifetime,
            'channel_open_fraction': channel_open_fraction,
            'mean_burst_duration': mean_burst_duration,
            'coherence_time': coherence_time,
            'entanglement_range': entanglement_range
        }
    
    def find_bursts(self, binary_signal: np.ndarray) -> List[Tuple[int, int]]:
        """Find burst start and end times in binary signal"""
        # [Keep existing implementation - no changes needed]
        bursts = []
        in_burst = False
        start = 0
        
        for i, val in enumerate(binary_signal):
            if val and not in_burst:
                start = i
                in_burst = True
            elif not val and in_burst:
                bursts.append((start, i))
                in_burst = False
        
        if in_burst:
            bursts.append((start, len(binary_signal)))
        
        return bursts
    
    def simulate(self, duration: float = 1.0, 
                stimulus_times: List[float] = None) -> SimulationResults:
        """Run full simulation with stochastic channel dynamics"""
        # [Keep existing implementation - no changes needed]
        n_steps = int(duration / self.params.dt)
        time = np.arange(n_steps) * self.params.dt
        
        save_times = list(range(0, n_steps, self.params.save_interval))
        n_saves = len(save_times)
        
        calcium_history = np.zeros((n_saves, *self.grid_shape))
        posner_history = np.zeros((n_saves, *self.grid_shape))
        channel_history = np.zeros((n_saves, self.params.n_channels), dtype=int)
        
        if stimulus_times is None:
            depolarized = np.ones(n_steps, dtype=bool)
        else:
            depolarized = np.zeros(n_steps, dtype=bool)
            for stim_time in stimulus_times:
                stim_idx = int(stim_time / self.params.dt)
                depolarized[stim_idx:stim_idx + int(0.005 / self.params.dt)] = True
        
        save_idx = 0
        for step in range(n_steps):
            channel_states = self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
            
            self.update_fields(self.params.dt, channel_open)
            
            if step in save_times:
                calcium_history[save_idx] = self.calcium_field
                posner_history[save_idx] = self.posner_field
                channel_history[save_idx] = channel_states
                save_idx += 1
            
            if step % 1000 == 0:
                logger.debug(f"Step {step}/{n_steps}, "
                           f"Max Posner: {np.max(self.posner_field)*1e9:.1f} nM")
        
        metrics = self.calculate_metrics(posner_history, channel_history)
        
        results = SimulationResults(
            model_version="4.0",
            parameters=self.params.to_dict(),
            time=time[save_times],
            calcium_map=calcium_history,
            posner_map=posner_history,
            channel_states=channel_history,
            **metrics
        )
        
        logger.info(f"Simulation complete. Peak Posner: {metrics['peak_posner']:.1f} nM")
        
        return results

# [KEEP ALL ANALYSIS FUNCTIONS AS-IS]
# analyze_spatial_pattern, analyze_temporal_dynamics don't need changes

# [KEEP THE TEST BLOCK AS-IS]
if __name__ == "__main__":
    # Your existing test code is fine
    logging.basicConfig(level=logging.INFO)
    
    params = DynamicParameters(
        n_channels=6,
        grid_size=50,
        dt=0.0001,
        save_interval=10
    )
    
    model = DynamicNanoreactor(params)
    
    print("Running test simulation...")
    results = model.simulate(duration=0.1, stimulus_times=[0.01, 0.05])
    
    test_path = Path("test_results/model4_test")
    test_path.parent.mkdir(exist_ok=True)
    results.save(test_path)
    
    print(f"Peak Posner: {results.peak_posner:.2f} nM")
    print(f"Coherence time: {results.coherence_time:.3f} s")
    print(f"Spatial heterogeneity: {results.spatial_heterogeneity:.3f}")
    print(f"Channel open fraction: {results.channel_open_fraction:.3f}")
    
    loaded = SimulationResults.load(test_path)
    print(f"Successfully loaded results with {len(loaded.time)} timepoints")