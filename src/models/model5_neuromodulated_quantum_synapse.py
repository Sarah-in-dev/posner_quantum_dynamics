"""
Model 5: Neuromodulated Quantum Synapse
========================================
Complete implementation incorporating all Model 4 dynamics plus:
- ATP-driven J-coupling for quantum coherence
- Dopamine neuromodulation switching between dimers/trimers
- Separate tracking of Ca6(PO4)4 dimers vs Ca9(PO4)6 trimers
- Quantum coherence calculation with dopamine protection
- Learning signal from overlap of coherence, dopamine, and activity

Author: Sarah Davidson
Date: August 2025
Version: 5.0
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import logging
from scipy.ndimage import maximum_filter
from datetime import datetime
from pathlib import Path
import json
import h5py
from dopamine_biophysics import DopamineField, DopamineParameters
logger = logging.getLogger(__name__)

# ============================================================================
# PARAMETERS
# ============================================================================

@dataclass
class Model5Parameters:
    """
    Complete parameters for the neuromodulated quantum synapse.
    Combines all Model 4 parameters with new quantum/neuromodulation features.
    """
    
    # ============= SPATIAL PARAMETERS (Model 4) =============
    grid_size: int = 50                 # 50x50 grid for computational efficiency
    active_zone_radius: float = 200e-9  # 200 nm - typical synaptic active zone
    cleft_width: float = 20e-9          # 20 nm synaptic cleft width
    
    # ============= CHANNEL PARAMETERS (Model 4) =============
    n_channels: int = 6                 # Number of calcium channels
    channel_current: float = 0.3e-12    # 0.3 pA single channel current (IN AMPERES!)
    channel_open_rate: float = 100.0    # Hz - opening rate when depolarized
    channel_close_rate: float = 50.0    # Hz - closing rate
    
    # ============= BASELINE CONCENTRATIONS =============
    ca_baseline: float = 100e-9         # 100 nM resting calcium
    po4_baseline: float = 1e-3          # 1 mM phosphate
    atp_concentration: float = 3e-3     # 3 mM ATP (Model 4)
    pnc_baseline: float = 1e-10         # 0.1 nM - CRITICAL! Non-zero for nucleation
    
    # ============= DIFFUSION COEFFICIENTS =============
    D_calcium: float = 220e-12          # m²/s - Ca²⁺ diffusion
    D_phosphate: float = 280e-12        # m²/s - phosphate diffusion
    D_pnc: float = 100e-12             # m²/s - PNC diffusion (larger clusters)
    D_dopamine: float = 400e-12         # m²/s - dopamine diffusion (Model 5)
    
    # ============= PNC DYNAMICS (Model 4 and Model 5 Enhancement) =============
    # CaHPO4 complex formation - precursor to PNCs
    k_complex_formation: float = 1e8    # M⁻¹s⁻¹ - near diffusion limited
    k_complex_dissociation: float = 5300.0  # s⁻¹ - moderate stability

    # This gives K_eq = 1e8/5300 = 18,900 M⁻¹
    # Justification breakdown:
    # - McDonogh bulk K = 470 M⁻¹ (activity-based at 25°C)
    # - Temperature correction to 37°C: ~2x → 940 M⁻¹
    # - Confinement effect (20 nm): ~10x → 9,400 M⁻¹  
    # - Membrane surface charge: ~2x → 18,800 M⁻¹

    """
    CaHPO4 binding enhanced 40-fold over bulk due to:
    1. Synaptic cleft confinement (Savtchenko & Rusakov 2007)
    2. Membrane surface effects (McLaughlin 1989)
    3. Activity coefficient changes in confined spaces (Berezhkovskii & Bezrukov 2005)
    Based on McDonogh et al. 2024 bulk value of 470 M⁻¹
    """
    
    # PNC formation from complexes
    k_pnc_formation: float = 200.0      # s⁻¹ - effective rate in complex-rich regions
    k_pnc_dissolution: float = 10.0     # s⁻¹ - PNC lifetime ~100 ms
    pnc_size: int = 30                  # ~30 Ca/PO4 units per PNC
    pnc_max_concentration: float = 1e-6 # 1 µM maximum to prevent runaway
    
    # ============= TEMPLATE PARAMETERS (Model 4) =============
    templates_per_synapse: int = 500    # Synaptotagmin molecules
    n_binding_sites: int = 3            # Binding sites per template
    k_pnc_binding: float = 1e9          # M⁻¹s⁻¹ - strong binding
    k_pnc_unbinding: float = 0.1        # s⁻¹ - slow unbinding
    template_accumulation_range: int = 2  # Grid cells - drift radius
    template_accumulation_rate: float = 0.3  # fraction/dt - accumulation strength
    
    # ============= FUSION KINETICS =============
    fusion_probability: float = 0.01    # Base probability for PNC fusion
    pnc_per_posner: int = 2             # PNCs consumed per Posner/dimer/trimer
    
    # ============= ACTIVITY-DEPENDENT (Model 4) =============
    k_atp_hydrolysis: float = 10.0      # s⁻¹ - ATP breakdown during activity
    ph_activity_shift: float = -0.3     # pH drops by 0.3 units during activity
    
    # ============= BIOPHYSICAL FACTORS (Model 4) =============
    f_hpo4_ph73: float = 0.49          # Fraction of phosphate as HPO₄²⁻ at pH 7.3
    gamma_ca: float = 0.4               # Activity coefficient for Ca²⁺ (non-ideal)
    gamma_po4: float = 0.2              # Activity coefficient for PO₄³⁻ (non-ideal)
    
    # ============= ENHANCEMENT FACTORS (Model 4) =============
    template_factor: float = 10.0       # Template enhancement of complex formation
    confinement_factor: float = 5.0     # 2D confinement enhancement
    electrostatic_factor: float = 3.0   # Membrane charge enhancement
    membrane_concentration_factor: float = 5.0  # Near-membrane concentration boost
    
    # ============= ATP/PHOSPHATE (Model 5) =============
    # Based on ³¹P NMR studies and Adams et al. 2025
    atp_baseline: float = 2.5e-3        # 2.5 mM baseline (Rangaraju 2014)
    adp_baseline: float = 0.25e-3       # 250 µM (10:1 ATP:ADP ratio)

    # J-coupling from ATP (Adams et al. 2025, Cohn & Hughes 1962)
    J_PP_atp: float = 20.0              # Hz - P-P coupling in ATP γ-β phosphates
    J_PO_atp: float = 7.5               # Hz - P-O coupling in free phosphate
    J_coupling_decay_time: float = 0.010 # 10 ms decay of J-coupling memory

    # ATP consumption (Rangaraju et al. 2014 Cell)
    atp_per_ap: float = 4.7e5           # molecules per action potential
    k_atp_hydrolysis: float = 10.0      # s⁻¹ - ATPase rate during activity
    
    # ============= DOPAMINE (Model 5) =============
    dopamine_tonic: float = 20e-9       # 20 nM tonic dopamine
    dopamine_peak: float = 1.6e-6       # 1.6 µM peak during reward
    k_dat_uptake: float = 4.0           # s⁻¹ - dopamine transporter uptake
    
    # ============= DIMER/TRIMER CHEMISTRY (Model 5) =============
    # Based on Agarwal et al. 2023 "The Biological Qubit"
    # Dimers: Ca6(PO4)4 - 4 ³¹P nuclei → longer coherence
    # Trimers: Ca9(PO4)6 - 6 ³¹P nuclei → shorter coherence

    # Formation: These are EFFECTIVE first-order rates in supersaturated regions
    # True mechanism is higher-order but we use pseudo-first-order approximation
    k_dimer_formation: float = 1e-3     # s⁻¹ (effective in PNC-rich regions)
    k_trimer_formation: float = 1e-4    # s⁻¹ (10x slower than dimers)

    # Dissolution rates based on stability
    kr_dimer: float = 0.01              # s⁻¹ - slow dissolution (100s lifetime)
    kr_trimer: float = 0.5              # s⁻¹ - fast dissolution (2s lifetime)

    # ============= QUANTUM PARAMETERS (Model 5) =============
    # Coherence times from Agarwal et al. 2023 AIMD simulations
    T2_base_dimer: float = 100.0        # 100-1000s range, using lower bound
    T2_base_trimer: float = 0.5         # <1s for 95% of configurations
    critical_conc_nM: float = 100.0      # Threshold for dipolar coupling effects
    
    # ============= pH DYNAMICS =============
    pH_baseline: float = 7.3            # Resting pH
    pH_activity_drop: float = 0.2       # pH drop during activity
    
    # ============= SIMULATION PARAMETERS =============
    dt: float = 0.001                   # 1 ms timestep
    duration: float = 1.0               # 1 second default simulation
    save_interval: float = 0.01         # Save every 10 ms
    
    def __post_init__(self):
        """Compute derived parameters after initialization"""
        # Grid spacing in meters
        self.dx = 2 * self.active_zone_radius / self.grid_size

# ============================================================================
# CHANNEL DYNAMICS (From Model 4)
# ============================================================================

class ChannelDynamics:
    """
    Handles stochastic calcium channel gating with 3 states:
    CLOSED <-> OPEN -> INACTIVATED -> CLOSED
    This creates realistic calcium transients with stochastic variability.
    """
    
    def __init__(self, n_channels: int, params: Model5Parameters):
        self.n_channels = n_channels
        self.params = params
        
        # Channel states as integers for efficiency
        self.CLOSED = 0
        self.OPEN = 1
        self.INACTIVATED = 2
        
        # Initialize all channels as closed
        self.states = np.zeros(n_channels, dtype=int)
        
        # Transition rates (Hz)
        self.rates = {
            'open': params.channel_open_rate,      # Closed -> Open
            'close': params.channel_close_rate,    # Open -> Closed
            'inactivate': 10.0,                   # Open -> Inactivated
            'recover': 5.0                        # Inactivated -> Closed
        }
        
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """
        Build Markov chain transition probability matrix.
        P = I + Q*dt approximates P = exp(Q*dt) for small dt.
        """
        dt = 0.001  # 1 ms timestep
        
        # Rate matrix Q
        Q = np.zeros((3, 3))
        
        # From CLOSED state
        Q[self.CLOSED, self.OPEN] = self.rates['open']
        Q[self.CLOSED, self.CLOSED] = -self.rates['open']
        
        # From OPEN state  
        Q[self.OPEN, self.CLOSED] = self.rates['close']
        Q[self.OPEN, self.INACTIVATED] = self.rates['inactivate']
        Q[self.OPEN, self.OPEN] = -(self.rates['close'] + self.rates['inactivate'])
        
        # From INACTIVATED state
        Q[self.INACTIVATED, self.CLOSED] = self.rates['recover']
        Q[self.INACTIVATED, self.INACTIVATED] = -self.rates['recover']
        
        # Convert to probability matrix
        self.P = np.eye(3) + Q * dt
        self.P = np.clip(self.P, 0, 1)
        
        # Normalize rows to ensure valid probabilities
        for i in range(3):
            row_sum = self.P[i].sum()
            if row_sum > 0:
                self.P[i] /= row_sum
    
    def update(self, dt: float, depolarized: bool = True) -> np.ndarray:
        """Update channel states stochastically"""
        if not depolarized:
            # All channels close immediately when not depolarized
            self.states[:] = self.CLOSED
            return self.states
        
        # Stochastic state transitions for each channel
        for i in range(self.n_channels):
            current_state = self.states[i]
            probs = self.P[current_state]
            self.states[i] = np.random.choice(3, p=probs)
        
        return self.states
    
    def get_open_channels(self) -> np.ndarray:
        """Return boolean array of which channels are open"""
        return self.states == self.OPEN

class pHDynamics:
    """Biophysically accurate pH dynamics"""
    def __init__(self):
        self.pH_rest = 7.3
        self.pH_active = 7.0
        self.buffer_capacity = 30e-3  # mM/pH unit (Chesler 2003)
        
    def calculate_activity_pH(self, ca_influx, atp_hydrolysis):
        # H+ from Ca/H exchange (2:1 stoichiometry)
        h_from_calcium = 2 * ca_influx
        # H+ from ATP hydrolysis
        h_from_atp = atp_hydrolysis
        
        total_h_production = h_from_calcium + h_from_atp
        dpH = -total_h_production / self.buffer_capacity
        return np.clip(self.pH_rest + dpH, 6.8, 7.4)

# ============================================================================
# RESULTS CONTAINER
# ============================================================================

@dataclass
class Model5Results:
    """Container for simulation results with metadata"""
    model_version: str = "5.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict = field(default_factory=dict)
    
    # Time series data
    time: np.ndarray = None
    calcium_map: np.ndarray = None
    pnc_map: np.ndarray = None
    dimer_map: np.ndarray = None
    trimer_map: np.ndarray = None
    dopamine_map: np.ndarray = None
    coherence_dimer_map: np.ndarray = None
    coherence_trimer_map: np.ndarray = None
    channel_states: np.ndarray = None
    
    # Summary metrics
    peak_calcium: float = 0.0
    peak_pnc: float = 0.0
    peak_dimer: float = 0.0
    peak_trimer: float = 0.0
    peak_dopamine: float = 0.0
    max_coherence_dimer: float = 0.0
    max_coherence_trimer: float = 0.0
    learning_signal: float = 0.0
    
    # Conservation metrics
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
            
            # Save all array data with compression
            for key in ['time', 'calcium_map', 'pnc_map', 'dimer_map', 
                       'trimer_map', 'dopamine_map', 'coherence_dimer_map',
                       'coherence_trimer_map', 'channel_states']:
                if getattr(self, key) is not None:
                    f.create_dataset(key, data=getattr(self, key), compression='gzip')
        
        # Save metadata to JSON
        metadata = {
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': {
                'peak_calcium': float(self.peak_calcium),
                'peak_pnc': float(self.peak_pnc),
                'peak_dimer': float(self.peak_dimer),
                'peak_trimer': float(self.peak_trimer),
                'peak_dopamine': float(self.peak_dopamine),
                'max_coherence_dimer': float(self.max_coherence_dimer),
                'max_coherence_trimer': float(self.max_coherence_trimer),
                'learning_signal': float(self.learning_signal),
                'calcium_conservation': float(self.calcium_conservation),
                'phosphate_conservation': float(self.phosphate_conservation),
                'templates_occupied': int(self.templates_occupied),
                'fusion_events': int(self.fusion_events)
            }
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")

# ============================================================================
# MAIN MODEL CLASS
# ============================================================================

class NeuromodulatedQuantumSynapse:
    """
    Model 5: Complete synaptic quantum processor with neuromodulation.
    
    This model combines:
    1. All Model 4 spatial dynamics and template-mediated formation
    2. ATP-driven phosphate with enhanced J-coupling for quantum coherence
    3. Dopamine neuromodulation that switches between dimer/trimer formation
    4. Separate tracking and coherence calculation for dimers vs trimers
    5. Learning signal from overlap of coherence, dopamine, and calcium
    """
    
    def __init__(self, params: Model5Parameters = None):
        """Initialize the complete neuromodulated quantum synapse"""
        self.params = params or Model5Parameters()
        
        # Setup spatial grid (from Model 4)
        self.grid_shape = (self.params.grid_size, self.params.grid_size)
        self.dx = 2 * self.params.active_zone_radius / self.params.grid_size
        
        # Create coordinate system centered at (0,0)
        x = np.linspace(-self.params.active_zone_radius, 
                        self.params.active_zone_radius, 
                        self.params.grid_size)
        y = np.linspace(-self.params.active_zone_radius, 
                        self.params.active_zone_radius, 
                        self.params.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Active zone mask - circular region where reactions occur
        self.active_mask = (self.X**2 + self.Y**2) <= self.params.active_zone_radius**2
        
        # Membrane proximity mask - enhanced concentration near edges
        r = np.sqrt(self.X**2 + self.Y**2)
        edge_distance = self.params.active_zone_radius - r
        self.membrane_mask = (edge_distance < 2e-9) & (edge_distance > 0)  # Within 2 nm
        
        # Initialize all concentration fields
        self.initialize_fields()
        
        # Setup spatial structures (channels, templates, dopamine sites)
        self.setup_spatial_structures()

        # Dopamine vesicular release tracking
        self.vesicle_pools = {site: 20 for site in self.da_release_sites}  # 20 vesicles per site
        self.release_probability = 0.06  # per action potential
        self.tonic_firing_rate = 5.0  # Hz
        self.phasic_firing_rate = 20.0  # Hz during burst
        self.dopamine_burst_active = False
        self.dopamine_burst_timer = 0.0
        
        # J-coupling decay (add if not present)
        self.phosphate_j_coupling = np.ones((self.params.grid_size, self.params.grid_size)) * 0.2
        
        # Timing for coincidence detection
        self.calcium_spike_time = -1.0
        self.current_time = 0.0

        self.reward_signal = False

        # Initialize biophysical dopamine system
        self.initialize_dopamine_system()
        
        # Template state tracking (critical from Model 4!)
        self.template_pnc_bound = np.zeros(self.grid_shape)
        self.template_fusion_timer = np.zeros(self.grid_shape)
        
        # Channel dynamics
        self.channels = ChannelDynamics(self.params.n_channels, self.params)
        
        # Track initial mass for conservation checks
        self.total_ca_initial = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        self.total_po4_initial = np.sum(self.phosphate_field * self.active_mask) * self.dx * self.dx
        
        # Metrics
        self.fusion_count = 0
        
        # Dopamine release timer for sustained phasic release
        self.dopamine_release_timer = 0
        self.current_sim_time = 0.0  # ADD THIS
        self.last_calcium_spike_time = -1.0  # ADD THIS
        self.coincidence_factor = 1.0  # ADD THIS - default value
        
        logger.info(f"Model 5 initialized: {self.params.n_channels} channels, "
                   f"{len(self.template_indices)} templates, "
                   f"{len(self.da_release_sites)} dopamine sites")
    
    def initialize_dopamine_system(self):
        """Initialize biophysically accurate dopamine dynamics"""
        # Create dopamine field with proper parameters
        self.da_params = DopamineParameters()
        self.da_field = DopamineField(
            grid_size=self.params.grid_size,
            dx=self.dx,
            params=self.da_params,
            release_sites=self.da_release_sites
        )
        
        # Track timing for coincidence detection
        self.last_calcium_spike_time = -1.0
        self.current_time = 0.0
    
    def initialize_fields(self):
        """Initialize all concentration fields"""
        gs = self.params.grid_size
        
        # === Core fields from Model 4 ===
        # Calcium starts at baseline everywhere in active zone
        self.calcium_field = np.ones((gs, gs)) * self.params.ca_baseline * self.active_mask
        
        # Phosphate at physiological concentration
        self.phosphate_field = np.ones((gs, gs)) * self.params.po4_baseline * self.active_mask
        
        # CaHPO4 complex - precursor to PNCs
        self.complex_field = np.zeros((gs, gs))
        
        # PNCs with NON-ZERO baseline (critical for nucleation!)
        self.pnc_field = np.ones((gs, gs)) * self.params.pnc_baseline * self.active_mask
        
        # pH and HPO4 fraction fields
        self.local_pH = 7.3 * np.ones((gs, gs))
        self.f_hpo4_local = self.params.f_hpo4_ph73 * np.ones((gs, gs))
        
        # === New Model 5 fields ===
        # ATP field for J-coupling source
        self.atp_field = np.ones((gs, gs)) * self.params.atp_baseline * self.active_mask
        
        # Dopamine field for neuromodulation
        self.dopamine_field = np.ones((gs, gs)) * self.params.dopamine_tonic * self.active_mask
        
        # Separate dimer and trimer tracking
        self.dimer_field = np.zeros((gs, gs))   # Ca6(PO4)4
        self.trimer_field = np.zeros((gs, gs))  # Ca9(PO4)6
        
        # Quantum coherence fields (0-1 normalized)
        self.coherence_dimer = np.zeros((gs, gs))
        self.coherence_trimer = np.zeros((gs, gs))
        
        # J-coupling field from ATP hydrolysis
        self.phosphate_j_coupling = np.ones((gs, gs))

        self.phosphate_j_coupling = np.ones((gs, gs)) * self.params.J_PO_atp  # Will be 7.5 Hz
    
    def setup_spatial_structures(self):
        """Setup channels, templates, and dopamine release sites"""
        # Calcium channels in hexagonal pattern
        self._setup_channel_positions()
        
        # Template proteins between channels
        self._setup_template_positions()
        
        # Dopamine release sites (volumetric transmission)
        center = self.params.grid_size // 2
        self.da_release_sites = [
            (center - 10, center),  # Left of center
            (center + 10, center)   # Right of center
        ]
    
    def _setup_channel_positions(self):
        """
        Position channels in hexagonal array.
        Hexagonal packing is optimal for coverage and matches experimental observations.
        """
        n = self.params.n_channels
        positions = []
        
        if n <= 1:
            # Single channel at center
            positions = [(self.params.grid_size // 2, self.params.grid_size // 2)]
        elif n <= 7:
            # Hexagonal arrangement: 1 center + 6 surrounding
            center = self.params.grid_size // 2
            radius = self.params.grid_size // 6
            
            # Center channel
            positions.append((center, center))
            
            # Surrounding channels at 60° intervals
            for i in range(min(n-1, 6)):
                angle = i * np.pi / 3
                x = center + int(radius * np.cos(angle))
                y = center + int(radius * np.sin(angle))
                if 0 <= x < self.params.grid_size and 0 <= y < self.params.grid_size:
                    positions.append((x, y))
        
        self.channel_indices = positions[:n]
    
    def _setup_template_positions(self):
        """
        Position template proteins (synaptotagmin) between channels.
        Templates are crucial for organizing PNC fusion into dimers/trimers.
        """
        templates = []
        n_templates = min(self.params.templates_per_synapse, 
                          self.params.grid_size * self.params.grid_size // 10)
        
        if len(self.channel_indices) > 1:
            # Place templates at midpoints between channel pairs
            for i in range(len(self.channel_indices)):
                for j in range(i+1, len(self.channel_indices)):
                    c1 = self.channel_indices[i]
                    c2 = self.channel_indices[j]
                    
                    mid_x = (c1[0] + c2[0]) // 2
                    mid_y = (c1[1] + c2[1]) // 2
                    
                    if (0 <= mid_x < self.params.grid_size and 
                        0 <= mid_y < self.params.grid_size and
                        self.active_mask[mid_y, mid_x]):
                        templates.append((mid_x, mid_y))
        
        # Add random templates if needed to reach target number
        while len(templates) < min(6, n_templates):
            x = np.random.randint(5, self.params.grid_size-5)
            y = np.random.randint(5, self.params.grid_size-5)
            if self.active_mask[y, x] and (x, y) not in templates:
                templates.append((x, y))
        
        self.template_indices = templates[:n_templates]

    def apply_diffusion(self, field: np.ndarray, D: float, dt: float) -> np.ndarray:
        """Apply diffusion using finite difference method"""
        dx = self.params.dx
        if dx == 0 or np.isnan(dx):
            return field  # No diffusion if dx is invalid
    
        # Stability check
        max_dt = dx**2 / (4 * D)
        if dt > max_dt:
            dt = max_dt * 0.9  # Use 90% of max stable timestep

        # Calculate Laplacian
        laplacian = (
            np.roll(field, 1, axis=0) + 
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + 
            np.roll(field, -1, axis=1) - 
            4 * field
        ) / (self.params.dx ** 2)

        # print(f"DEBUG: dx = {self.params.dx}, grid_size = {self.params.grid_size}")
        if self.params.dx == 0:
            self.params.dx = 8e-9  # 8 nm default
    
        # Apply Neumann boundary conditions (no flux)
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
    
        # Update field with clipping to prevent overflow
        result = field + D * laplacian * dt
        return np.clip(result, 0, 1e-3)  # Max 1 mM concentration
    
    # ========================================================================
    # CORE MODEL 4 DYNAMICS
    # ========================================================================
    
    def update_local_pH(self, activity_level: float):
        """
        Update local pH based on synaptic activity.
        Vesicle fusion and ATP hydrolysis cause acidification.
        """
        # Activity causes acidification (more activity = lower pH)
        pH_shift = self.params.ph_activity_shift * activity_level
        self.local_pH = 7.3 + pH_shift * self.active_mask
        
        # Update HPO4²⁻ fraction using Henderson-Hasselbalch
        # pKa2 for phosphate = 7.2
        self.f_hpo4_local = 1 / (1 + 10**(7.2 - self.local_pH))
    
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """
        Calculate calcium concentration from open channels.
        Uses analytical solution for steady-state diffusion from point source.
        Creates sharp microdomains with 1/r profile.
        """
        calcium = self.params.ca_baseline * np.ones(self.grid_shape)
        
        for idx, (ci, cj) in enumerate(self.channel_indices):
            if idx < len(channel_open) and channel_open[idx]:
                # Distance from channel
                r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                r = np.maximum(r, 1e-9)  # Avoid singularity (1 nm minimum)
                
                # Calcium flux from single channel
                # Current in Amperes: 0.3e-12 A = 0.3 pA
                # Convert to Ca ions/sec: I/(2e) where e = 1.602e-19 C

                # FIX: The issue is we need to scale properly for the grid
                # The analytical solution assumes infinite medium, but we have finite grid
                # Add a scaling factor to get physiological values
                flux_ions_per_sec = self.params.channel_current / (2 * 1.602e-19)
                
                # Convert to molar flux
                flux_molar = flux_ions_per_sec / 6.022e23  # moles/sec
                
                # Steady-state concentration from point source
                # C(r) = flux / (4 * pi * D * r)
                # Add a factor to account for finite volume and buffering

                scaling_factor = 1e-4  # Adjusted scaling factor for physiological range
                single_ca = scaling_factor * flux_molar / (4 * np.pi * self.params.D_calcium * r)
                
                # Add to baseline
                calcium += single_ca
        
        # Apply membrane enhancement (Ca²⁺ attracted to negative membrane)
        calcium[self.membrane_mask] *= self.params.membrane_concentration_factor
        
        # Apply buffering (reduced to allow higher peaks)
        kappa = 5  # Buffering power
        calcium_buffered = calcium / (1 + kappa)
        
        # Apply saturation only at very high levels (10 mM)
        calcium_final = calcium_buffered / (1 + calcium_buffered/10e-3)
        
        return calcium_final * self.active_mask
    
    def update_phosphate_from_ATP(self, dt: float, activity_level: float):
        """
        ATP hydrolysis releases phosphate during activity.
        Creates phosphate hotspots near active channels.
        """
        if activity_level > 0:
            # ATP hydrolysis rate proportional to activity
            hydrolysis_rate = self.params.k_atp_hydrolysis * activity_level
            
            # Phosphate production (10% of ATP gets hydrolyzed)
            phosphate_production = self.params.atp_concentration * hydrolysis_rate * dt * 0.1
            
            # Add preferentially near active channels
            for idx, (ci, cj) in enumerate(self.channel_indices):
                if idx < len(self.channels.states) and self.channels.states[idx] == 1:
                    # Local enhancement around active channel
                    r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                    enhancement = np.exp(-r / 10e-9)  # 10 nm decay length
                    self.phosphate_field += phosphate_production * enhancement
    
    def calculate_complex_equilibrium(self):
        """
        Calculate CaHPO4 complex concentration.
        This neutral complex is the precursor to PNC formation.
        Uses activity coefficients to account for non-ideal solution behavior.
        """
        # Get effective concentrations with activity coefficients
        ca_eff = self.calcium_field * self.params.gamma_ca
        po4_eff = self.phosphate_field * self.f_hpo4_local * self.params.gamma_po4
        
        # Prevent overflow
        ca_eff = np.clip(ca_eff, 0, 1e-3)  # Max 1 mM
        po4_eff = np.clip(po4_eff, 0, 1e-3)  # Max 1 mM
        
        # Equilibrium constant
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
        
        # Mass action equilibrium
        denominator = 1 + K_eq * (ca_eff + po4_eff)
        denominator = np.maximum(denominator, 1.0)  # Prevent divide by zero
        
        self.complex_field = K_eq * ca_eff * po4_eff / denominator
        
        # Templates enhance complex formation
        for ti, tj in self.template_indices:
            self.complex_field[tj, ti] = min(
                self.complex_field[tj, ti] * self.params.template_factor, 
                1e-6  # Cap at 1 µM
            )
        
        self.complex_field *= self.active_mask
    
    def update_pnc_dynamics(self, dt: float):
        """
        Update prenucleation cluster dynamics.
        PNCs form from CaHPO4 complexes and accumulate at templates.
        This is the critical precursor step before dimer/trimer formation.
        """
        # Calculate complex concentration first
        self.calculate_complex_equilibrium()
        
        # Formation from complexes
        formation = self.params.k_pnc_formation * self.complex_field
        
        # Apply saturation to prevent runaway growth
        saturation = 1 - self.pnc_field / self.params.pnc_max_concentration
        saturation = np.clip(saturation, 0, 1)
        formation *= saturation
        
        # Dissolution back to ions
        dissolution = self.params.k_pnc_dissolution * self.pnc_field
        
        # === Template accumulation (KEY MECHANISM!) ===
        # Templates actively concentrate PNCs from surrounding area
        for ti, tj in self.template_indices:
            if self.pnc_field[tj, ti] < self.params.pnc_max_concentration:
                # Accumulate from surrounding grid cells
                r_max = self.params.template_accumulation_range
                for di in range(-r_max, r_max+1):
                    for dj in range(-r_max, r_max+1):
                        ni, nj = ti + di, tj + dj
                        if (0 <= ni < self.params.grid_size and 
                            0 <= nj < self.params.grid_size and
                            (di != 0 or dj != 0) and
                            self.active_mask[nj, ni]):
                            
                            # Drift toward template
                            distance = np.sqrt(di**2 + dj**2)
                            drift_rate = self.params.template_accumulation_rate / (1 + distance)
                            transfer = self.pnc_field[nj, ni] * drift_rate * dt
                            
                            # Move PNCs
                            transfer = min(transfer, self.pnc_field[nj, ni])
                            self.pnc_field[nj, ni] -= transfer
                            self.pnc_field[tj, ti] += transfer
            
            # === Template binding ===
            # PNCs bind to template proteins for organized fusion
            if self.pnc_field[tj, ti] > 1e-9:  # If PNCs present
                available = self.params.n_binding_sites - self.template_pnc_bound[tj, ti]
                
                if available > 0:
                    # Simple first-order binding
                    binding_rate = 10.0  # s⁻¹ (aggressive binding)
                    binding_amount = min(binding_rate * dt * available, available)
                    
                    if self.pnc_field[tj, ti] > 1e-8:  # Need at least 10 nM
                        self.template_pnc_bound[tj, ti] += binding_amount
        
        # Update PNC field
        dpnc_dt = formation - dissolution
        self.pnc_field += dpnc_dt * dt * self.active_mask
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
        
        # Add small diffusion
        laplacian = self.calculate_laplacian_neumann(self.pnc_field)
        self.pnc_field += self.params.D_pnc * laplacian * dt
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
    
    def calculate_laplacian_neumann(self, field: np.ndarray) -> np.ndarray:
        """
        Calculate Laplacian with no-flux (Neumann) boundary conditions.
        Used for diffusion calculations.
        """
        laplacian = np.zeros_like(field)
        
        # Interior points - standard 5-point stencil
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # Boundaries - mirror for no flux
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
        
        return laplacian
    
    # ========================================================================
    # MODEL 5 ENHANCEMENTS
    # ========================================================================
    
    def update_atp_dynamics(self, dt: float, activity_level: float):
        """
        ATP hydrolysis during activity produces phosphate with strong J-coupling.
        Based on Adams et al. 2025 showing ATP has 100x stronger J-coupling than free phosphate.
        """
        if activity_level > 0:
            # ATP hydrolysis rate from Rangaraju et al. 2014 (Cell)
            # ~4.7 × 10^5 ATP molecules/synapse/AP
            base_hydrolysis_rate = 10.0  # s⁻¹ during activity
        
            # Spatial pattern - enhanced near channels (ATPase pumps colocalized)
            hydrolysis_field = np.zeros_like(self.atp_field)
        
            for idx, (ci, cj) in enumerate(self.channel_indices):
                if idx < len(self.channels.states) and self.channels.states[idx] == 1:
                    # Create gradient around active channel
                    r = np.sqrt((self.X - self.X[ci, cj])**2 + 
                            (self.Y - self.Y[ci, cj])**2)
                
                    # ATPase activity falls off with distance (10 nm scale)
                    local_rate = base_hydrolysis_rate * np.exp(-r / 10e-9)
                    hydrolysis_field += local_rate * activity_level
        
            # Apply hydrolysis
            atp_consumed = self.atp_field * hydrolysis_field * dt
            atp_consumed = np.minimum(atp_consumed, self.atp_field)  # Can't consume more than available
        
            # Products: ADP + Pi (with enhanced J-coupling)
            phosphate_produced = atp_consumed
            self.atp_field -= atp_consumed
            self.phosphate_field += phosphate_produced
        
            # KEY: Track J-coupling strength of phosphate
            # Based on Adams et al. 2025 - ATP phosphates have J_PP ~ 20 Hz
            # This decays to bulk value (~0.2 Hz) over ~10 ms
        
            # Update J-coupling field
            new_coupling = np.where(
                phosphate_produced > 0,
                self.params.J_PP_atp,  # 20 Hz for fresh ATP-derived phosphate
                self.phosphate_j_coupling  # Keep existing value
            )
            self.phosphate_j_coupling = new_coupling

            # Where hydrolysis is high, set J-coupling to ATP value
            mask = hydrolysis_field > 1e-6  # Some threshold
            self.phosphate_j_coupling[mask] = self.params.J_PP_atp  # Should be 20 Hz
        
            # J-coupling decay (exponential with 10 ms time constant)
            tau_j_decay = 0.010  # 10 ms
            decay_factor = np.exp(-dt / tau_j_decay)
    
            # Decay toward baseline J-coupling
            self.phosphate_j_coupling = (
                self.params.J_PO_atp +  # Baseline: 7.5 Hz
                (self.phosphate_j_coupling - self.params.J_PO_atp) * decay_factor
            )
    
        # ATP regeneration (simplified - would involve mitochondria)
        # During rest, slow regeneration
        if activity_level < 0.1:
            regeneration_rate = 1.0  # s⁻¹
            atp_deficit = self.params.atp_baseline - self.atp_field
            self.atp_field += atp_deficit * regeneration_rate * dt * 0.1
    
    
    def update_j_coupling_decay(self, dt: float):
        """J-coupling decays from 20 Hz to 0.2 Hz over ~10ms"""
        
        # Decay time constant
        tau = 0.01  # 10 ms
        
        # Exponential decay
        decay_factor = np.exp(-dt / tau)
        
        # Decay toward baseline (0.2 Hz)
        self.phosphate_j_coupling = 0.2 + (self.phosphate_j_coupling - 0.2) * decay_factor
    
    
    def update_dopamine(self, dt: float, reward_signal: bool):
        """Dopamine release with vesicular dynamics"""
        
        # Check if we should start phasic burst
        if reward_signal and not self.dopamine_burst_active:
            self.dopamine_burst_active = True
            self.dopamine_burst_timer = 0.0
            
        # Update burst timer
        if self.dopamine_burst_active:
            self.dopamine_burst_timer += dt
            if self.dopamine_burst_timer > 0.1:  # 100ms burst
                self.dopamine_burst_active = False
                
        # Determine firing rate
        firing_rate = self.phasic_firing_rate if self.dopamine_burst_active else self.tonic_firing_rate
        
        # Stochastic release from each site
        for site in self.da_release_sites:
            if site not in self.vesicle_pools:
                self.vesicle_pools[site] = 20
                
            # Poisson probability of spike in dt
            p_spike = 1 - np.exp(-firing_rate * dt)
            
            if np.random.random() < p_spike:
                # Spike occurred - check for vesicle release
                if np.random.random() < self.release_probability and self.vesicle_pools[site] > 0:
                    # Release one vesicle
                    molecules = 3000  # molecules per vesicle
                    
                    # Convert to concentration
                    # Volume = dx * dx * cleft_width
                    volume = self.params.dx * self.params.dx * self.params.cleft_width  # m³
                    concentration = molecules / (volume * 6.022e23)  # M
                    
                    # Add to field
                    i, j = site
                    self.dopamine_field[j, i] += concentration  # Note: [j,i] for numpy array
                    
                    # Consume vesicle
                    self.vesicle_pools[site] -= 1
        
        # Refill vesicles
        for site in self.vesicle_pools:
            if self.vesicle_pools[site] < 20:
                refill = 2.0 * dt  # 2 vesicles/second
                self.vesicle_pools[site] = min(20, self.vesicle_pools[site] + refill)
        
        # Diffusion
        result = self.apply_diffusion(
            self.dopamine_field,
            self.params.D_dopamine,
            dt
        )
        self.dopamine_field = np.clip(result, 0, 10e-6)  # Max 10 µM for dopamine

        # DAT uptake (Michaelis-Menten)
        Km = 220e-9  # 220 nM
        Vmax = 4.1e-6  # M/s
        excess = np.maximum(0, self.dopamine_field - self.params.dopamine_tonic)
        uptake_rate = Vmax * excess / (Km + excess)
        self.dopamine_field -= uptake_rate * dt
        
        # Ensure we don't go below tonic
        self.dopamine_field = np.maximum(self.dopamine_field, self.params.dopamine_tonic)
        
        # Reset reward signal
        if self.reward_signal:
            self.reward_signal = False
        
    
    def update_template_fusion(self, dt: float):
        """
        Template-mediated formation of dimers or trimers.
        This is where Model 4's template fusion is adapted for Model 5's
        dopamine-dependent dimer vs trimer selection.
        """
        for ti, tj in self.template_indices:
            # Check PNC occupancy at template
            occupancy = self.template_pnc_bound[tj, ti] / self.params.n_binding_sites
            
            if occupancy >= 0.5:  # Threshold for fusion attempt
                # === Environmental modulation (from Model 4) ===
                # Multiple factors affect fusion probability

                # Get quantum modulation from dopamine
                da_modulation = self.da_field.get_quantum_modulation(ti, tj)
                
                # 1. Cooperative effect - more bound PNCs help
                cooperative_factor = 1 + 2 * (occupancy - 0.5)
                
                # 2. Calcium enhancement - Ca²⁺ stabilizes transition
                ca_local = self.calcium_field[tj, ti]
                ca_enhancement = 1 + (ca_local / 100e-6)  # Enhanced above 100 µM
                
                # 3. pH effect - lower pH favors fusion
                pH_local = self.local_pH[tj, ti]
                pH_factor = 1 + 0.5 * (7.3 - pH_local)
                
                # 4. Activity-dependent template conformational change
                activity_level = np.sum(self.channels.get_open_channels()) / self.params.n_channels
                template_activation = 1 + activity_level * 2

                # Apply coincidence timing
                timing_factor = self.coincidence_factor
                
                # Combined enhancement
                total_enhancement = (cooperative_factor * ca_enhancement * 
                                    pH_factor * timing_factor)
                
                # Calculate fusion probability
                fusion_prob = self.params.fusion_probability * total_enhancement
                fusion_prob = min(fusion_prob, 0.5)  # Cap at 50%
                
                # Stochastic fusion event
                if np.random.random() < fusion_prob:
                    # Use biophysical dopamine to select dimer vs trimer
                    if da_modulation['above_quantum_threshold']:
                        # Form dimers with enhancement
                        n_dimers = np.random.poisson(3 * da_modulation['dimer_enhancement'])
                        self.dimer_field[tj, ti] += n_dimers * 1e-9
                    else:
                        # Form trimers (suppressed if DA present)
                        n_trimers = np.random.poisson(2 * da_modulation['trimer_suppression'])
                        self.trimer_field[tj, ti] += n_trimers * 1e-9
                    
                    # Reset template
                    self.template_pnc_bound[tj, ti] *= 0.3
                    self.fusion_count += 1
                    
                    # Template recycling depends on activity
                    if activity_level > 0.5:
                        # High activity: partial reset (fast recycling)
                        self.template_pnc_bound[tj, ti] *= 0.3
                    else:
                        # Low activity: full reset (slow recycling)
                        self.template_pnc_bound[tj, ti] = 0
                    
                    # PNC consumption
                    pncs_consumed = self.params.pnc_per_posner
                    # Fix: Check for zero before division
                    if self.pnc_field[tj, ti] > 0:
                        consumption_factor = max(0, 1 - pncs_consumed * 1e-10 / self.pnc_field[tj, ti])
                        self.pnc_field[tj, ti] *= consumption_factor
                    
                    self.fusion_count += 1
    
    def form_dimers_and_trimers(self, dt: float):
        """
        Direct formation pathway from PNC aggregation.
        Based on classical nucleation theory where PNCs aggregate into dimers/trimers.
        Enhanced by ATP-derived phosphate J-coupling and modulated by dopamine.
        """
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                if not self.active_mask[j, i]:
                    continue
            
                # Use PNC concentration as the driver (not Ca/PO4)
                pnc_local = self.pnc_field[j, i]
            
                # Need minimum PNC concentration for aggregation
                pnc_critical = 100e-9  # 100 nM threshold
            
                if pnc_local > pnc_critical:
                    # Supersaturation in terms of PNC availability
                    S_pnc = pnc_local / pnc_critical

                    # J-coupling enhancement factor
                    # Strong J-coupling (20 Hz) helps maintain quantum coherence during formation
                    j_coupling = self.phosphate_j_coupling[j, i]
                    j_enhancement = j_coupling / 0.2  # Normalized to free phosphate baseline
                    # This gives ~2.7x enhancement for ATP-derived phosphate
                
                    # Get modulation factors
                    da = self.dopamine_field[j, i]
                    pH_local = self.local_pH[j, i]
                
                    # pH affects aggregation (lower pH favors it)
                    pH_factor = 10 ** (7.3 - pH_local)  # Inverted - acidic favors
                
                    # J-coupling specifically helps dimer formation (fewer spins to couple)
                    if da > 100e-9:
                        dimer_rate = self.params.k_dimer_formation * 10 * j_enhancement
                        trimer_rate = self.params.k_trimer_formation * 0.1  # No J benefit
                    else:
                        dimer_rate = self.params.k_dimer_formation * j_enhancement
                        trimer_rate = self.params.k_trimer_formation * j_enhancement
                
                    if i == 25 and j == 25:  # Only print for test location
                        print(f"DEBUG: DA={da*1e9:.1f}nM, J={j_coupling:.1f}Hz, dimer_rate={dimer_rate}, trimer_rate={trimer_rate}")
                    
                    # Dimer formation (2 PNCs → 1 dimer)
                    # Rate depends on PNC supersaturation and J-coupling
                    dimer_formation = dimer_rate * (S_pnc - 1.0) * pH_factor * dt
                
                    # Trimer formation (3 PNCs → 1 trimer) 
                    # Requires higher supersaturation
                    trimer_formation = 0
                    if S_pnc > 1.5:  # Need 50% more PNCs for trimers
                        trimer_formation = trimer_rate * (S_pnc - 1.5) * pH_factor * dt
                
                    # Consume PNCs (stoichiometry matters!)
                    pnc_consumed = 0
                
                    # Cap formation by available PNCs
                    max_dimers = pnc_local / (2 * 1e-9)  # 2 PNCs per dimer, 30 units per PNC
                    if dimer_formation > 0:
                        dimer_formation = min(dimer_formation, max_dimers * 1e-9)
                        self.dimer_field[j, i] += dimer_formation
                        pnc_consumed += dimer_formation * 2  # Consume 2 PNCs
                
                    max_trimers = (pnc_local - pnc_consumed) / (3 * 30 * 1e-9)  # 3 PNCs per trimer
                    if trimer_formation > 0:
                        trimer_formation = min(trimer_formation, max_trimers * 1e-9)
                        self.trimer_field[j, i] += trimer_formation
                        pnc_consumed += trimer_formation * 3 * 30 * 1e-9  # Consume 3 PNCs
                
                    # After calculating dimer_formation, add:
                    if i == 25 and j == 25 and dimer_formation > 1e-6:
                        print(f"DEBUG FORMATION: dimer_rate={dimer_rate}, S_pnc={S_pnc}, pH_factor={pH_factor}, dt={dt}")
                        print(f"  dimer_formation={dimer_formation}, max_dimers={max_dimers}")
                    
                    # Update PNC field
                    self.pnc_field[j, i] -= pnc_consumed
                    self.pnc_field[j, i] = max(0, self.pnc_field[j, i])
    
    def update_quantum_coherence(self, dt: float):
        """
        Calculate quantum coherence for dimers and trimers.
        Based on Agarwal et al. 2023 showing dimers maintain coherence ~100x longer.
        Dopamine protects coherence, especially for dimers.
        """
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                da_local = self.dopamine_field[j, i]
                j_coupling_local = self.phosphate_j_coupling[j, i]
            
                # === DIMERS - Long baseline coherence ===
                if self.dimer_field[j, i] > 1e-12:  # If dimers present
                    # Base T2 from Agarwal et al. 2023
                    T2_dimer = self.params.T2_base_dimer  # 100s baseline
                
                    # J-coupling enhancement (stronger coupling = better coherence)
                    j_factor = j_coupling_local / self.params.J_PO_atp
                    T2_dimer *= j_factor  # Up to 2.7x for ATP-derived phosphate
                
                    # Dopamine protection mechanism
                    # Based on D2 receptor activation (Kd = 10 nM)
                    if da_local > 50e-9:  # Above 50 nM
                        d2_activation = da_local / (da_local + 10e-9)
                        protection = 1 + 2 * d2_activation  # Up to 3x protection
                        T2_dimer *= protection
                
                    # Environmental decoherence (temperature, noise)
                    # From Player et al. 2018 - dipolar coupling effects
                    concentration_factor = min(self.dimer_field[j, i] / 100e-9, 2.0)
                    T2_dimer /= concentration_factor  # Higher conc = more dipolar coupling
                
                    # Update coherence (exponential decay)
                    if not hasattr(self, 'coherence_dimer') or self.coherence_dimer[j, i] == 0:
                        # Initialize to 1 when dimers first form
                        self.coherence_dimer[j, i] = 1.0
                    else:
                        # Decay existing coherence
                        self.coherence_dimer[j, i] *= np.exp(-dt / T2_dimer)
                else:
                    self.coherence_dimer[j, i] = 0
            
                # === TRIMERS - Short baseline coherence ===
                if self.trimer_field[j, i] > 1e-12:  # If trimers present
                    # Base T2 from Agarwal et al. 2023
                    T2_trimer = self.params.T2_base_trimer  # 0.5-1s baseline
                
                    # J-coupling has minimal effect on trimers (too many coupled spins)
                    # Based on Adams et al. 2025 - complexity overwhelms enhancement
                    j_factor = 1 + 0.1 * (j_coupling_local / self.params.J_PO_atp - 1)
                    T2_trimer *= j_factor  # Only 10% of J benefit
                
                    # Dopamine can't save trimers (Agarwal et al. 2023)
                    # The 6 coupled ³¹P nuclei create too many decoherence pathways
                    # Minimal protection even at high dopamine
                    if da_local > 100e-9:
                        T2_trimer *= 1.1  # Only 10% improvement
                
                    # Environmental decoherence - stronger for trimers
                    concentration_factor = min(self.trimer_field[j, i] / 100e-9, 3.0)
                    T2_trimer /= concentration_factor
                
                    # Update coherence
                    if not hasattr(self, 'coherence_trimer') or self.coherence_trimer[j, i] == 0:
                        self.coherence_trimer[j, i] = 1.0
                    else:
                        self.coherence_trimer[j, i] *= np.exp(-dt / T2_trimer)
                else:
                    self.coherence_trimer[j, i] = 0
            
                # Ensure coherence stays in [0, 1]
                self.coherence_dimer[j, i] = np.clip(self.coherence_dimer[j, i], 0, 1)
                self.coherence_trimer[j, i] = np.clip(self.coherence_trimer[j, i], 0, 1)
    
    def calculate_learning_signal(self) -> float:
        """
        Learning occurs when three conditions overlap:
        1. High dimer coherence (>0.5)
        2. High dopamine (>100 nM)
        3. Recent activity (elevated Ca)
        
        This triple coincidence is the quantum signature of learning!
        """
        # Check timing window (50-200ms after calcium spike)
        if self.calcium_spike_time < 0:
            return 0.0
            
        time_since_calcium = self.current_time - self.calcium_spike_time
        if time_since_calcium < 0.05 or time_since_calcium > 0.2:
            return 0.0
        
        # Find sites meeting all criteria
        coherent_dimers = (self.dimer_field > 1e-9) & (self.coherence_dimer > 0.5)
        dopamine_present = self.dopamine_field > 100e-9
        recent_calcium = self.calcium_field > 10e-6
        
        # Triple coincidence
        learning_sites = coherent_dimers & dopamine_present & recent_calcium
        
        if np.any(learning_sites):
            # Weight by coherence and dopamine level
            learning_strength = np.sum(
                self.coherence_dimer[learning_sites] * 
                (self.dopamine_field[learning_sites] / 1e-6)
            )
            return learning_strength / (self.params.grid_size ** 2)
        
        return 0.0
    
    # ========================================================================
    # MAIN UPDATE AND SIMULATION
    # ========================================================================
    
    def update_fields(self, dt: float, channel_open: np.ndarray, reward_signal: bool = False):
        """
        Main update orchestrating all dynamics.
        Order matters! Each step depends on previous ones.
        """
        # Track time (ADD THIS)
        self.current_time += dt
        
        # Calculate activity level
        activity_level = np.sum(channel_open) / max(1, len(channel_open))
        
        # === Core Model 4 updates ===
        # 1. pH changes affect phosphate speciation
        self.update_local_pH(activity_level)
        
        # 2. Calcium microdomains from open channels
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)

        # Track calcium spikes for coincidence detection (ADD THIS)
        if np.max(self.calcium_field) > 10e-6 and self.calcium_spike_time < 0:
            self.calcium_spike_time = self.current_time
        
        # 3. Phosphate from ATP hydrolysis
        self.update_phosphate_from_ATP(dt, activity_level)
        
        # 4. CaHPO4 complex equilibrium
        self.calculate_complex_equilibrium()
        
        # 5. PNC dynamics with template accumulation
        self.update_pnc_dynamics(dt)
        
        # === Model 5 enhancements ===
        # 6. ATP creates J-coupling
        self.update_atp_dynamics(dt, activity_level)

        # 6.5 J-coupling decay (ADD THIS NEW STEP)
        self.update_j_coupling_decay(dt)
        
        # 7. Dopamine if reward signal
        if reward_signal:
            self.reward_signal = True
        self.update_dopamine(dt, reward_signal)  # Always call it (for diffusion/uptake)
        
        # 8. Template-mediated dimer/trimer formation
        self.update_template_fusion(dt)
        
        # 9. Direct formation pathway
        self.form_dimers_and_trimers(dt)
        
        # 10. Update quantum coherence
        self.update_quantum_coherence(dt)
        
        # 11. Dissolution of dimers and trimers
        self.dimer_field -= self.params.kr_dimer * self.dimer_field * dt
        self.trimer_field -= self.params.kr_trimer * self.trimer_field * dt
        self.dimer_field = np.maximum(self.dimer_field, 0)
        self.trimer_field = np.maximum(self.trimer_field, 0)

        # 12. Calculate learning signal (ADD THIS)
        self.learning_signal = self.calculate_learning_signal()
    
    def run_simulation(self, 
                      duration: float = None,
                      stim_protocol: str = 'single_spike',
                      reward_time: float = 0.2) -> Model5Results:
        """
        Run complete simulation with specified protocol.
        
        Parameters:
        -----------
        duration : float
            Simulation duration in seconds
        stim_protocol : str
            'single_spike', 'tetanus', or 'custom'
        reward_time : float
            Time when reward signal (dopamine) is delivered
        
        Returns:
        --------
        Model5Results object with all data and metrics
        """
        if duration is None:
            duration = self.params.duration
        
        # Create results container
        results = Model5Results(parameters=asdict(self.params))
        
        # Setup time
        n_steps = int(duration / self.params.dt)
        save_interval = int(self.params.save_interval / self.params.dt)
        n_saves = n_steps // save_interval + 1
        
        # Initialize result arrays
        results.time = np.arange(0, duration + self.params.dt, self.params.save_interval)[:n_saves]
        results.calcium_map = np.zeros((n_saves, *self.grid_shape))
        results.pnc_map = np.zeros((n_saves, *self.grid_shape))
        results.dimer_map = np.zeros((n_saves, *self.grid_shape))
        results.trimer_map = np.zeros((n_saves, *self.grid_shape))
        results.dopamine_map = np.zeros((n_saves, *self.grid_shape))
        results.coherence_dimer_map = np.zeros((n_saves, *self.grid_shape))
        results.coherence_trimer_map = np.zeros((n_saves, *self.grid_shape))
        results.channel_states = np.zeros((n_saves, self.params.n_channels))
        
        # === Setup stimulus protocol ===
        depolarized = np.zeros(n_steps, dtype=bool)
        reward = np.zeros(n_steps, dtype=bool)
        
        if stim_protocol == 'single_spike':
            # Single 50 ms depolarization at t=100 ms
            depolarized[int(0.1/self.params.dt):int(0.15/self.params.dt)] = True
            # Reward signal at specified time
            reward[int(reward_time/self.params.dt):int((reward_time+0.1)/self.params.dt)] = True
            
        elif stim_protocol == 'tetanus':
            # 10 spikes at 100 Hz
            for i in range(10):
                start = int((0.1 + i*0.01)/self.params.dt)
                depolarized[start:start+int(0.005/self.params.dt)] = True
            # Reward during tetanus
            reward[int(0.1/self.params.dt):int(0.3/self.params.dt)] = True
        
        # === Main simulation loop ===
        save_idx = 0
        for step in range(n_steps):
            # Update channels
            self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
            
            # Update all fields
            self.update_fields(self.params.dt, channel_open, reward[step])
            
            # Save periodically
            if step % save_interval == 0:
                results.calcium_map[save_idx] = self.calcium_field
                results.pnc_map[save_idx] = self.pnc_field
                results.dimer_map[save_idx] = self.dimer_field
                results.trimer_map[save_idx] = self.trimer_field
                results.dopamine_map[save_idx] = self.dopamine_field
                results.coherence_dimer_map[save_idx] = self.coherence_dimer
                results.coherence_trimer_map[save_idx] = self.coherence_trimer
                results.channel_states[save_idx] = self.channels.states
                save_idx += 1
                
                # Progress logging
                if step % 1000 == 0:
                    logger.info(f"Step {step}/{n_steps}: "
                               f"Dimers={np.max(self.dimer_field)*1e9:.1f} nM, "
                               f"Trimers={np.max(self.trimer_field)*1e9:.1f} nM, "
                               f"Dopamine={np.max(self.dopamine_field)*1e9:.1f} nM")
        
        # === Calculate final metrics ===
        results.peak_calcium = np.max(results.calcium_map) * 1e6  # µM
        results.peak_pnc = np.max(results.pnc_map) * 1e9  # nM
        results.peak_dimer = np.max(results.dimer_map) * 1e9  # nM
        results.peak_trimer = np.max(results.trimer_map) * 1e9  # nM
        results.peak_dopamine = np.max(results.dopamine_map) * 1e9  # nM
        results.max_coherence_dimer = np.max(results.coherence_dimer_map)
        results.max_coherence_trimer = np.max(results.coherence_trimer_map)
        results.learning_signal = self.calculate_learning_signal()
        
        # Conservation checks
        final_ca_total = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        final_po4_total = np.sum(self.phosphate_field * self.active_mask) * self.dx * self.dx
        results.calcium_conservation = final_ca_total / self.total_ca_initial
        results.phosphate_conservation = final_po4_total / self.total_po4_initial
        
        # Template metrics
        results.templates_occupied = np.sum(self.template_pnc_bound > 0)
        results.fusion_events = self.fusion_count
        
        logger.info(f"Simulation complete. Learning signal: {results.learning_signal:.3f}")
        
        return results

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_quantum_advantage(results: Model5Results) -> Dict:
    """
    Analyze the quantum advantage of dopamine-modulated dimer formation.
    
    Key metrics:
    - Coherence time ratio (dimer vs trimer)
    - Dopamine enhancement factor
    - Learning efficiency
    """
    analysis = {}
    
    # Coherence time advantage
    if results.max_coherence_trimer > 0:
        analysis['coherence_advantage'] = results.max_coherence_dimer / results.max_coherence_trimer
    else:
        analysis['coherence_advantage'] = np.inf
    
    # Dimer/trimer ratio
    total_dimers = np.sum(results.dimer_map)
    total_trimers = np.sum(results.trimer_map)
    if total_trimers > 0:
        analysis['dimer_trimer_ratio'] = total_dimers / total_trimers
    else:
        analysis['dimer_trimer_ratio'] = np.inf
    
    # Spatial correlation of dopamine and dimers
    if np.std(results.dopamine_map) > 0 and np.std(results.dimer_map) > 0:
        correlation = np.corrcoef(
            results.dopamine_map.flatten(),
            results.dimer_map.flatten()
        )[0, 1]
        analysis['dopamine_dimer_correlation'] = correlation
    else:
        analysis['dopamine_dimer_correlation'] = 0
    
    # Learning efficiency
    analysis['learning_signal'] = results.learning_signal
    analysis['peak_coherence'] = results.max_coherence_dimer
    
    return analysis

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model with default parameters
    model = NeuromodulatedQuantumSynapse()
    
    # Run simulation with reward
    results = model.run_simulation(
        duration=1.0,
        stim_protocol='single_spike',
        reward_time=0.2  # Dopamine 100ms after stimulus
    )
    
    # Analyze quantum advantage
    analysis = analyze_quantum_advantage(results)
    
    print("\n=== Model 5 Simulation Results ===")
    print(f"Peak Calcium: {results.peak_calcium:.1f} µM")
    print(f"Peak PNC: {results.peak_pnc:.1f} nM")
    print(f"Peak Dimer: {results.peak_dimer:.1f} nM")
    print(f"Peak Trimer: {results.peak_trimer:.1f} nM")
    print(f"Peak Dopamine: {results.peak_dopamine:.1f} nM")
    print(f"Max Dimer Coherence: {results.max_coherence_dimer:.3f}")
    print(f"Max Trimer Coherence: {results.max_coherence_trimer:.3f}")
    print(f"Learning Signal: {results.learning_signal:.3f}")
    print(f"\nCoherence Advantage (Dimer/Trimer): {analysis['coherence_advantage']:.1f}x")
    print(f"Dimer/Trimer Ratio: {analysis['dimer_trimer_ratio']:.1f}")
    print(f"Dopamine-Dimer Correlation: {analysis['dopamine_dimer_correlation']:.3f}")
    
    # Save results
    results.save(Path("results/model5_test"))