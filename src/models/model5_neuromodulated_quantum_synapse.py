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

Update September 13, 2025:
Based on Agarwal et al. 2023 and calcium phosphate chemistry:
1. Dimers (Ca6(PO4)4) maintain coherence for ~100s
2. Formation requires supersaturation and nucleation
3. Dopamine modulates selectivity, not total amount
4. Substrate depletion provides natural feedback

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.gridspec as gridspec
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
    pnc_baseline: float = 1e-15         # 0.1 nM - CRITICAL! Non-zero for nucleation
    ca_microdomain_peak: float = 100e-6  # 100 µM in microdomains
    
    # ============= DIFFUSION COEFFICIENTS =============
    D_calcium: float = 220e-12          # m²/s - Ca²⁺ diffusion
    D_phosphate: float = 280e-12        # m²/s - phosphate diffusion
    D_pnc: float = 100e-12             # m²/s - PNC diffusion (larger clusters)
    D_dopamine: float = 400e-12         # m²/s - dopamine diffusion (Model 5)
    
    # ============= PNC DYNAMICS (Model 4 and Model 5 Enhancement) =============
    # CaHPO4 complex formation - precursor to PNCs
    k_complex_formation: float = 1e6    # M⁻¹s⁻¹ - near diffusion limited
    k_complex_dissociation: float = 5300.0  # s⁻¹ - moderate stability
    # This gives K_eq = 189 M⁻¹ (closer to McDonogh's 470 M⁻¹)

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
    
    # ============= PNC DYNAMICS (RATE-LIMITING STEP) =============
    # PNCs are the bottleneck - they form slowly!
    k_pnc_formation: float = 0.01     # REDUCED from 10.0 (100x reduction)
    k_pnc_dissolution: float = 100.0     # s⁻¹ - PNC lifetime ~100 ms
    pnc_size: int = 30                  # ~30 Ca/PO4 units per PNC
    pnc_max_concentration: float = 1e-9 # 1 µM maximum to prevent runaway
    pnc_critical: float = 10e-9        # 10 nM instead of 50 nM
    
    # ============= TEMPLATE PARAMETERS (Model 4) =============
    templates_per_synapse: int = 100    # Synaptotagmin molecules
    n_binding_sites: int = 3            # Binding sites per template
    k_pnc_binding: float = 1e6          # M⁻¹s⁻¹ - strong binding
    k_pnc_unbinding: float = 1.0        # s⁻¹ - slow unbinding
    template_accumulation_range: int = 2  # Grid cells - drift radius
    template_accumulation_rate: float = 0.0003  # 0.03% per ms = 30% per second
    
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
    template_factor: float = 2.0       # Template enhancement of complex formation
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
    k_dimer_formation: float = 100.0     # s⁻¹ maximum rate (was 1e-3)
    k_trimer_formation: float = 50.0    # s⁻¹ maximum rate (was 1e-4)

    # Dissolution rates based on stability
    kr_dimer: float = 0.01             # s⁻¹ → 100s lifetime (CORRECT!)
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
        
        self.params = params
    
        # Define grid_shape early, before using it
        self.grid_shape = (self.params.grid_size, self.params.grid_size)
        
        # Time tracking for coincidence detection
        self.current_time = 0.0
        self.calcium_spike_time = -1.0  # -1 means no recent spike

        # Coherence arrays
        self.coherence_dimer = np.zeros(self.grid_shape)
        self.coherence_trimer = np.zeros(self.grid_shape)
        
        
        self.n_channels = n_channels

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
        Fixed version that maintains baseline calcium
        """
        # Start with baseline everywhere
        calcium = self.params.ca_baseline * np.ones(self.grid_shape)
    
        for idx, (ci, cj) in enumerate(self.channel_indices):
            if idx < len(channel_open) and channel_open[idx]:
                r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                r = np.maximum(r, 1e-9)
            
                flux_ions_per_sec = self.params.channel_current / (2 * 1.602e-19)
                flux_molar = flux_ions_per_sec / 6.022e23
            
                # REDUCED scaling factor to prevent depletion
                scaling_factor = 1e-5  # Was 1e-4
                single_ca = scaling_factor * flux_molar / (4 * np.pi * self.params.D_calcium * r)
            
                # ADD to baseline, don't replace
                calcium += single_ca
    
        # Less aggressive membrane enhancement
        calcium[self.membrane_mask] *= 2.0  # Was membrane_concentration_factor
    
        # Gentler buffering
        kappa = 2  # Was 5
        calcium_buffered = calcium / (1 + kappa)
    
        # Higher saturation limit
        calcium_final = calcium_buffered / (1 + calcium_buffered/100e-3)  # Was 10e-3
    
        # CRITICAL: Ensure minimum baseline is maintained
        calcium_final = np.maximum(calcium_final, self.params.ca_baseline)
    
        return calcium_final * self.active_mask
    
    def calculate_complex_equilibrium(self):
        """
        Calculate CaHPO4 complex concentration.
        This neutral complex is the precursor to PNC formation.
        Uses activity coefficients to account for non-ideal solution behavior.
        """
        # Get effective concentrations with activity coefficients
        ca_eff = self.calcium_field * self.params.gamma_ca
        po4_eff = self.phosphate_field * self.f_hpo4_local * self.params.gamma_po4
        
        # CRITICAL FIX: Only form complex above baseline calcium
        # This prevents PNC formation at rest
        ca_threshold = 1e-6  # 1 µM threshold for complex formation
        ca_above_threshold = np.maximum(0, ca_eff - ca_threshold)
        
        
        # Prevent overflow
        ca_above_threshold = np.clip(ca_above_threshold, 0, 100e-6)
        po4_eff = np.clip(po4_eff, 0, 100e-6)
        
        # Equilibrium constant
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
        
        # Mass action equilibrium - using only calcium above threshold
        # denominator = 1 + K_eq * (ca_above_threshold + po4_eff)
        # denominator = np.maximum(denominator, 1.0)
        
        self.complex_field = K_eq * ca_above_threshold * po4_eff
        
        # Cap it
        self.complex_field = np.minimum(self.complex_field, 1e-6)  # Cap at 1 µM
        
        # Templates enhance complex formation but with strict cap
        for ti, tj in self.template_indices:
            self.complex_field[tj, ti] *= self.params.template_factor
                
        self.complex_field *= self.active_mask
    
    def update_pnc_dynamics(self, dt: float):
        """
        Fixed version with proper saturation and activity dependence
        """
        # Calculate complex concentration first
        self.calculate_complex_equilibrium()
    
        # CRITICAL: Only form PNC when there's significant complex
        complex_threshold = 1e-12  # 1 pM threshold
        active_complex = np.maximum(0, self.complex_field - complex_threshold)
    
        # Formation from complexes - only where complex exists
        formation = self.params.k_pnc_formation * active_complex
    
        # FIXED saturation - ensure it works properly
        current_saturation = self.pnc_field / self.params.pnc_max_concentration
        saturation_factor = np.exp(-current_saturation * 10)  # Exponential decay
        formation *= saturation_factor
    
        # Increased dissolution to prevent accumulation
        dissolution = self.params.k_pnc_dissolution * self.pnc_field
    
        # Update PNC field with formation/dissolution
        dpnc_dt = formation - dissolution
        self.pnc_field += dpnc_dt * dt * self.active_mask
    
        # Hard cap to prevent explosion
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
    
        # Reduced diffusion to prevent spreading
        laplacian = self.calculate_laplacian_neumann(self.pnc_field)
        self.pnc_field += 0.1 * self.params.D_pnc * laplacian * dt  # Reduce diffusion by 10x
    
        # Template binding - but only if activity is present
        mean_j_coupling = np.mean(self.phosphate_j_coupling)
        if mean_j_coupling > 1.0:  # Only bind during activity
            for ti, tj in self.template_indices:
                if self.pnc_field[tj, ti] > 1e-12:  # If PNCs present
                    available = self.params.n_binding_sites - self.template_pnc_bound[tj, ti]
                
                    if available > 0:
                        binding_rate = self.params.k_pnc_binding
                        binding = binding_rate * self.pnc_field[tj, ti] * available * dt
                        binding = min(binding, available * 0.01, self.pnc_field[tj, ti])  # Limit binding
                    
                        self.template_pnc_bound[tj, ti] += binding
                        self.pnc_field[tj, ti] -= binding
    
        # Final safety check
        self.pnc_field = np.maximum(self.pnc_field, 0)
        self.template_pnc_bound = np.maximum(self.template_pnc_bound, 0)
    
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
    
    def update_atp_dynamics(self, dt: float, activity_level: float, channel_open: np.ndarray):
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
    
            # Count active channels
            active_channels = 0
            for idx, (i, j) in enumerate(self.channel_indices):
                if idx < len(channel_open) and channel_open[idx]:
                    active_channels += 1
            
                    # ATP → ADP + Pi at this location
                    local_atp = self.atp_field[i, j]
                    hydrolysis = self.params.k_atp_hydrolysis * local_atp * dt
            
                    self.atp_field[i, j] -= hydrolysis
                    self.phosphate_field[i, j] += hydrolysis
            
                    # Set high J-coupling at active sites
                    self.phosphate_j_coupling[i, j] = self.params.J_PP_atp  # 20 Hz
    
            if active_channels > 0:
                print(f"  Active channels: {active_channels}")
                print(f"  Max J-coupling: {np.max(self.phosphate_j_coupling):.1f} Hz")
    
            # Apply hydrolysis
            atp_consumed = self.atp_field * hydrolysis_field * dt
            atp_consumed = np.minimum(atp_consumed, self.atp_field)  # Can't consume more than available

            # Check if any ATP was consumed
            total_consumed = np.sum(atp_consumed)
            print(f"  ATP consumed: {total_consumed:.2e} M")
    
            # Products: ADP + Pi (with enhanced J-coupling)
            phosphate_produced = atp_consumed
            self.atp_field -= atp_consumed
            self.phosphate_field += phosphate_produced
    
            # KEY: Set J-coupling to 20 Hz where ATP was just hydrolyzed
            # This is the ONLY place we set high J-coupling
            high_hydrolysis_mask = phosphate_produced > 1e-10  # Wherever phosphate was produced
            n_high = np.sum(high_hydrolysis_mask)
        
            # Debug output to verify
            if n_high > 0:
                self.phosphate_j_coupling[high_hydrolysis_mask] = self.params.J_PP_atp
                max_j = np.max(self.phosphate_j_coupling)
                print(f"  Set J-coupling to {self.params.J_PP_atp}Hz at {n_high} locations")
                print(f"  Max J-coupling now: {max_j:.1f}Hz")
    
            # ATP regeneration (simplified - would involve mitochondria)
            # During rest, slow regeneration
            if activity_level < 0.1:
                regeneration_rate = 1.0  # s⁻¹
                atp_deficit = self.params.atp_baseline - self.atp_field
                self.atp_field += atp_deficit * regeneration_rate * dt * 0.1

    def update_j_coupling_decay(self, dt: float):
        """
        J-coupling decays from 20 Hz to 0.2 Hz over ~10ms
        This should be called AFTER update_atp_dynamics in the main loop
        """
        # Decay time constant
        tau = self.params.J_coupling_decay_time  # Should be 0.01 (10 ms)
    
        # Exponential decay
        decay_factor = np.exp(-dt / tau)
    
        # Decay toward baseline (0.2 Hz for free phosphate)
        baseline = 0.2
        self.phosphate_j_coupling = baseline + (self.phosphate_j_coupling - baseline) * decay_factor
    
        # Debug: Check if we still have high J-coupling
        max_j = np.max(self.phosphate_j_coupling)
        if max_j > 10.0:
            print(f"DEBUG J-DECAY: Max J-coupling after decay: {max_j:.1f} Hz")
    
    
    def update_dopamine(self, dt: float, reward_signal: bool):
        """Use the biophysical dopamine model to update dopamine field"""
        
        if not hasattr(self, 'da_field'):
            # Initialize dopamine field if not exists
            from dopamine_biophysics import DopamineField, DopamineParameters
            params = DopamineParameters()
            self.da_field = DopamineField(
                grid_size=self.params.grid_size,
                dx=self.dx,
                params=params,
                release_sites=[(25, 25)]  # Or your release sites
            )
    
        # Update the field
        self.da_field.update(dt, reward=reward_signal)
    
        # Copy to local array for compatibility
        self.dopamine_field = self.da_field.field.copy()
        
    
    def update_template_fusion(self, dt: float):
        """
        Template-mediated formation with proper PNC consumption.
        Templates organize PNC into dimers/trimers but still consume PNC!
        """
        for ti, tj in self.template_indices:
            # Check template occupancy
            occupancy = self.template_pnc_bound[tj, ti] / self.params.n_binding_sites
        
            if occupancy >= 0.8:  # 80% threshold for fusion
                # Get local conditions
                pnc_local = self.pnc_field[tj, ti]
            
                # Need PNC for fusion even at templates
                if pnc_local < 100e-9:  # Not enough PNC
                    continue
            
                # Calculate all enhancement factors
                cooperative_factor = 1 + 2 * occupancy
            
                ca_local = self.calcium_field[tj, ti]
                ca_enhancement = 1.0
                if ca_local > 1e-6:
                    ca_enhancement = 1 + np.log10(ca_local / 1e-6)
            
                pH_local = self.local_pH[tj, ti]
                pH_factor = 10 ** (7.3 - pH_local)
            
                activity_level = np.sum(self.channels.get_open_channels()) / self.params.n_channels
                template_activation = 1 + activity_level * 2
            
                da_modulation = self.da_field.get_quantum_modulation(ti, tj)
            
                coincidence_factor = self.da_field.get_coincidence_factor(
                    calcium_spike_time=self.calcium_spike_time,
                    current_time=self.current_time
                )
            
                total_enhancement = (cooperative_factor * ca_enhancement * 
                                    pH_factor * coincidence_factor * template_activation)
            
                fusion_prob = self.params.fusion_probability * total_enhancement
                fusion_prob = min(fusion_prob, 0.5)
            
                if np.random.random() < fusion_prob:
                    # FIXED: Calculate formation based on available PNC
                    if da_modulation['above_quantum_threshold']:
                        # Dimers preferred
                        n_dimers = min(3, int(pnc_local / (2 * 50e-9)))  # Max 3, limited by PNC
                        if n_dimers > 0:
                            dimer_conc = n_dimers * 50e-9  # 50 nM per dimer
                            pnc_consumed = 2 * dimer_conc  # Stoichiometry
                        
                            self.dimer_field[tj, ti] += dimer_conc
                            self.pnc_field[tj, ti] -= pnc_consumed
                    else:
                        # Trimers
                        n_trimers = min(2, int(pnc_local / (3 * 50e-9)))  # Max 2, limited by PNC
                        if n_trimers > 0:
                            trimer_conc = n_trimers * 50e-9
                            pnc_consumed = 3 * trimer_conc
                        
                            self.trimer_field[tj, ti] += trimer_conc
                            self.pnc_field[tj, ti] -= pnc_consumed
                
                    # Template recycling
                    if activity_level > 0.5:
                        self.template_pnc_bound[tj, ti] *= 0.3
                    else:
                        self.template_pnc_bound[tj, ti] = 0
                
                    self.fusion_count += 1
    
    def form_dimers_and_trimers(self, dt: float):
        """Form dimers and trimers with proper calcium and dopamine gating"""
    
        # No formation without activity
        calcium_peak = np.max(self.calcium_field)
        if calcium_peak < 500e-9:
            return
    
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                ca_local = self.calcium_field[j, i]
                pnc_local = self.pnc_field[j, i]
                po4_local = self.phosphate_field[j, i]
            
                if ca_local < 500e-9 or pnc_local < self.params.pnc_critical:
                    continue
            
                # USE DOPAMINE FIELD METHODS HERE!
                if hasattr(self, 'da_field') and self.da_field is not None:
                    # Get emergent selectivity
                    selectivity = self.da_field.calculate_emergent_selectivity(
                        i, j, ca_local, po4_local
                    )
                
                    # Get quantum modulation
                    quantum_mod = self.da_field.get_quantum_modulation(i, j)
                
                    # Get timing coincidence
                    coincidence = self.da_field.get_coincidence_factor(
                        self.calcium_spike_time,
                        self.current_time
                    )
                
                    dimer_factor = selectivity['dimer_factor'] * quantum_mod['dimer_enhancement']
                    trimer_factor = selectivity['trimer_factor'] * quantum_mod['trimer_suppression']
                    d2_occupancy = selectivity['d2_occupancy']
                else:
                    # Fallback - minimal formation without dopamine
                    dimer_factor = 1.0
                    trimer_factor = 1.0
                    d2_occupancy = 0
                    coincidence = 1.0
            
                # Calculate supersaturation
                S_pnc = np.log10(pnc_local / self.params.pnc_critical + 1)
                S_pnc = np.clip(S_pnc, 0, 2)
            
                # Calculate formation rates using the factors from dopamine field
                k_dimer = self.params.k_dimer_formation * S_pnc * dimer_factor * coincidence
                k_trimer = self.params.k_trimer_formation * S_pnc * trimer_factor * coincidence
            
                # pH modulation
                pH_local = self.local_pH[j, i]
                if pH_local > 7.3:
                    k_trimer *= 1.5  # Trimers need higher pH
            
                # Formation amounts
                dimer_formation = k_dimer * pnc_local * dt
                trimer_formation = k_trimer * pnc_local * dt
            
                # Stoichiometric constraints
                max_conversion = pnc_local * 0.01
                dimer_formation = min(dimer_formation, max_conversion / 2)
                trimer_formation = min(trimer_formation, max_conversion / 3)
            
                # Apply formation
                self.dimer_field[j, i] += dimer_formation
                self.trimer_field[j, i] += trimer_formation
            
                # Consume PNC
                self.pnc_field[j, i] -= (2 * dimer_formation + 3 * trimer_formation)
                self.pnc_field[j, i] = max(0, self.pnc_field[j, i])

                # After forming dimers/trimers, initialize their coherence
                if dimer_formation > 0 and self.coherence_dimer[j, i] == 0:
                    self.coherence_dimer[j, i] = 1.0  # Start with full coherence
    
                if trimer_formation > 0 and self.coherence_trimer[j, i] == 0:
                    self.coherence_trimer[j, i] = 1.0  # Start with full coherence
    
    def update_quantum_coherence(self, dt: float):
        """
        Calculate quantum coherence for dimers and trimers.
        Based on Agarwal et al. 2023 showing dimers maintain coherence ~100x longer.
        Dopamine protects coherence, especially for dimers.
        """
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                da_local = self.dopamine_field[j, i] if hasattr(self, 'dopamine_field') else self.dopamine_field[j, i]
                j_coupling_local = self.phosphate_j_coupling[j, i] if hasattr(self, 'phosphate_j_coupling') else self.phosphate_j_coupling[j, i]
            
                # === DIMERS - Long baseline coherence ===
                if self.dimer_field[j, i] > 1e-12:  # If dimers present
                    # Base T2 from Agarwal et al. 2023
                    T2_dimer = self.params.T2_base_dimer if hasattr(self.params, 'T2_base_dimer') else 100.0  # 100s baseline
                    self.coherence_dimer *= np.exp(-dt / T2_dimer)

                    # J-coupling enhancement (stronger coupling = better coherence)
                    j_factor = j_coupling_local / 0.27 if j_coupling_local > 0 else 0.5  # Normalized to ATP J-coupling
                    T2_dimer *= max(j_factor, 0.5)  # At least 50% of base
                
                    # Dopamine protection mechanism
                    # Based on D2 receptor activation (Kd = 10 nM)
                    if da_local > 50e-9:  # Above 50 nM
                        d2_activation = da_local / (da_local + 10e-9)
                        protection = 1 + 2 * d2_activation  # Up to 3x protection
                        T2_dimer *= protection
                
                    # Environmental decoherence (temperature, noise)
                    concentration_factor = min(self.dimer_field[j, i] / 100e-9, 2.0)
                    T2_dimer /= max(concentration_factor, 1.0)  # Higher conc = more dipolar coupling
                
                    # Update coherence (exponential decay)
                    if not hasattr(self, 'coherence_dimer'):
                        self.coherence_dimer = np.ones((self.params.grid_size, self.params.grid_size))
                
                    if self.coherence_dimer[j, i] == 0:
                        # Initialize to 1 when dimers first form
                        self.coherence_dimer[j, i] = 1.0
                    else:
                        # Decay existing coherence
                        decay_rate = dt / T2_dimer
                        self.coherence_dimer[j, i] *= np.exp(-decay_rate)
                else:
                    if hasattr(self, 'coherence_dimer'):
                        self.coherence_dimer[j, i] = 0
            
                # === TRIMERS - Short baseline coherence ===
                if self.trimer_field[j, i] > 1e-12:  # If trimers present
                    # Base T2 from Agarwal et al. 2023
                    T2_trimer = self.params.T2_base_trimer if hasattr(self.params, 'T2_base_trimer') else 1.0  # 1s baseline (100x shorter!)
                    self.coherence_trimer *= np.exp(-dt / T2_trimer)

                    # J-coupling has minimal effect on trimers (too many coupled spins)
                    j_factor = 1 + 0.1 * (j_coupling_local / 0.27 - 1) if j_coupling_local > 0 else 1.0
                    T2_trimer *= j_factor  # Only 10% of J benefit
                
                    # Dopamine can't save trimers (Agarwal et al. 2023)
                    if da_local > 100e-9:
                        T2_trimer *= 1.1  # Only 10% improvement even at high dopamine
                
                    # Environmental decoherence - stronger for trimers
                    concentration_factor = min(self.trimer_field[j, i] / 100e-9, 3.0)
                    T2_trimer /= max(concentration_factor, 1.0)
                
                    # Update coherence with faster decay
                    if not hasattr(self, 'coherence_trimer'):
                        self.coherence_trimer = np.ones((self.params.grid_size, self.params.grid_size))
                    
                    if self.coherence_trimer[j, i] == 0:
                        self.coherence_trimer[j, i] = 1.0
                    else:
                        # Trimers decay much faster
                        decay_rate = dt / T2_trimer
                        self.coherence_trimer[j, i] *= np.exp(-decay_rate)
                else:
                    if hasattr(self, 'coherence_trimer'):
                        self.coherence_trimer[j, i] = 0
            
        # Ensure coherence stays in [0, 1]
        if hasattr(self, 'coherence_dimer'):
            self.coherence_dimer = np.clip(self.coherence_dimer, 0, 1)
        if hasattr(self, 'coherence_trimer'):
            self.coherence_trimer = np.clip(self.coherence_trimer, 0, 1)
    
    def calculate_learning_signal(self) -> float:
        """
        Learning occurs when three conditions overlap:
        1. High dimer coherence (>0.5)
        2. High dopamine (>100 nM)
        3. Recent activity (elevated Ca)
        
        This triple coincidence is the quantum signature of learning!
        The coincidence factor from dopamine biophysics ensures proper STDP timing.
        """
        if not hasattr(self, 'da_field'):
            return 0.0
    
        # Triple coincidence detection
        high_coherence = self.coherence_dimer > 0.5
        high_dopamine = self.da_field.D2_occupancy > 0.5
        high_calcium = self.calcium_field > 50e-6
    
        # Get timing factor
        coincidence = self.da_field.get_coincidence_factor(
            self.calcium_spike_time,
            self.current_time
        )
    
        # Spatial integral
        overlap = high_coherence * high_dopamine * high_calcium * coincidence
        learning_signal = np.sum(overlap) * self.dx * self.dx
    
        return learning_signal
    
    def check_mass_conservation(self):
        """
        Verify phosphate mass is conserved.
        Total P = free PO4 + complex + PNC + dimers + trimers
        """
        # Calculate total phosphate in each form
        free_po4 = np.sum(self.phosphate_field * self.active_mask)
    
        # 1 P per complex
        in_complex = np.sum(self.complex_field * self.active_mask)
    
        # Assume 4 P per PNC (simplified)
        in_pnc = 4 * np.sum(self.pnc_field * self.active_mask)
    
        # 4 P per dimer, 6 P per trimer
        in_dimers = 4 * np.sum(self.dimer_field * self.active_mask)
        in_trimers = 6 * np.sum(self.trimer_field * self.active_mask)
    
        total_phosphate = free_po4 + in_complex + in_pnc + in_dimers + in_trimers
    
        # Convert to total moles
        volume_element = self.dx * self.dx * 20e-9  # dx² * cleft_width
        total_moles = total_phosphate * volume_element
    
        return {
            'free_po4': free_po4,
            'complex': in_complex,
            'pnc': in_pnc,
            'dimers': in_dimers,
            'trimers': in_trimers,
            'total': total_phosphate,
            'total_moles': total_moles
        }

    def diagnose_formation(self):
        """Print diagnostic info about formation dynamics"""
        total_dimers = np.sum(self.dimer_field) * 1e9  # nM
        total_trimers = np.sum(self.trimer_field) * 1e9
        mean_ca = np.mean(self.calcium_field[self.active_mask]) * 1e9
        mean_po4 = np.mean(self.phosphate_field[self.active_mask]) * 1e6  # µM
        mean_pnc = np.mean(self.pnc_field[self.active_mask]) * 1e9
        mean_da = np.mean(self.dopamine_field[self.active_mask]) * 1e9
        
        print(f"\n=== Formation Diagnostics ===")
        print(f"Dimers: {total_dimers:.1f} nM")
        print(f"Trimers: {total_trimers:.1f} nM")
        print(f"Ratio: {total_dimers/total_trimers if total_trimers > 0 else 'inf':.2f}")
        print(f"Mean Ca: {mean_ca:.1f} nM")
        print(f"Mean PO4: {mean_po4:.1f} µM")
        print(f"Mean PNC: {mean_pnc:.1f} nM")
        print(f"Mean DA: {mean_da:.1f} nM")
        
        # Check if substrates are depleted
        if mean_ca < 10:  # Less than 10 nM
            print("WARNING: Calcium depleted!")
        if mean_po4 < 100:  # Less than 100 µM
            print("WARNING: Phosphate depleted!")
        if mean_pnc < 1:  # Less than 1 nM
            print("WARNING: PNC depleted!")
            
        # Check for runaway accumulation
        if total_dimers > 1000:  # More than 1 µM
            print("WARNING: Excessive dimer accumulation!")
        if total_trimers > 1000:
            print("WARNING: Excessive trimer accumulation!")

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

         # 3. ATP creates J-coupling
        self.update_atp_dynamics(dt, activity_level, channel_open)
        
        # 3.5 J-coupling decay (ADD THIS NEW STEP)
        self.update_j_coupling_decay(dt)

        # DEBUG: Check J-coupling after decay
        max_j_after_decay = np.max(self.phosphate_j_coupling)
        if max_j_after_decay > 1.0:
            print(f"DEBUG after decay: Max J-coupling = {max_j_after_decay:.1f} Hz")

        # 4. CaHPO4 complex equilibrium
        self.calculate_complex_equilibrium()
        
        # 5. PNC dynamics with template accumulation
        self.update_pnc_dynamics(dt)
        
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
        

        # Periodic mass check (every 100 steps)
        if hasattr(self, 'step_count'):
            self.step_count += 1
            if self.step_count % 100 == 0:
                mass_balance = self.check_mass_conservation()
                if self.step_count % 1000 == 0:  # Less frequent printing
                    print(f"Mass balance check: Total P = {mass_balance['total']:.2e} M")
        else:
            self.step_count = 0




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

    def analyze_dopamine_gradients(self):
        """Analyze spatial gradients for directional learning effects"""
        gradient_mag = self.da_field.calculate_spatiotemporal_gradient()
    
        # High gradients indicate boundaries between high/low DA regions
        # These might be important for spatial learning
        return gradient_mag

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

    def get_simulation_metrics(self) -> Dict[str, float]:
        """
        Get current state metrics for analysis.
        Enhanced with gradient analysis.
        """
        # Existing metrics
        metrics = {
            'mean_calcium': np.mean(self.calcium_field[self.active_mask]),
            'mean_phosphate': np.mean(self.phosphate_field[self.active_mask]),
            'mean_pnc': np.mean(self.pnc_field[self.active_mask]),
            'mean_dopamine': np.mean(self.dopamine_field[self.active_mask]),
            'total_dimers': np.sum(self.dimer_field[self.active_mask]),
            'total_trimers': np.sum(self.trimer_field[self.active_mask]),
            'mean_coherence_dimer': np.mean(self.coherence_dimer[self.active_mask]),
            'mean_coherence_trimer': np.mean(self.coherence_trimer[self.active_mask]),
            'learning_signal': getattr(self, 'learning_signal', 0.0),
            'fusion_events': self.fusion_count
        }
    
        # Add gradient analysis
        gradient_mag = self.analyze_dopamine_gradients()
        metrics['max_da_gradient'] = np.max(gradient_mag)
        metrics['mean_da_gradient'] = np.mean(gradient_mag[self.active_mask])
    
        return metrics

    def plot_dopamine_analysis(self):
        """Visualize dopamine field and gradients"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        # Dopamine concentration
        im1 = ax1.imshow(self.dopamine_field * 1e9, cmap='viridis')
        ax1.set_title('Dopamine Concentration (nM)')
        plt.colorbar(im1, ax=ax1)
    
        # Dopamine gradients
        gradient_mag = self.analyze_dopamine_gradients()
        im2 = ax2.imshow(gradient_mag * 1e9, cmap='hot')
        ax2.set_title('Dopamine Gradient Magnitude (nM/µm)')
        plt.colorbar(im2, ax=ax2)
    
        plt.tight_layout()
        return fig

    def debug_dopamine_system(self):
            """Debug the dopamine system to find why vesicles aren't releasing"""
            print("\n=== DOPAMINE SYSTEM DEBUG ===")
        
            # Check basic parameters
            print(f"DopamineField parameters:")
            print(f"  Burst frequency: {self.da_field.params.burst_frequency} Hz")
            print(f"  Release probability: {self.da_field.params.release_probability}")
            print(f"  Vesicles per bouton: {self.da_field.params.vesicles_per_bouton}")
            print(f"  Molecules per vesicle: {self.da_field.params.quantal_size}")
        
            # Check release sites
            print(f"\nRelease sites: {self.da_field.release_sites}")
        
            # Manually test release probability
            dt = 0.001
            p_spike = 1 - np.exp(-self.da_field.params.burst_frequency * dt)
            print(f"\nWith dt={dt}s and burst_frequency={self.da_field.params.burst_frequency}Hz:")
            print(f"  P(spike) = {p_spike:.4f}")
            print(f"  P(release|spike) = {self.da_field.params.release_probability}")
            print(f"  P(release) = {p_spike * self.da_field.params.release_probability:.6f}")
        
            # Check if _phasic_release is being called
            print(f"\nTesting manual phasic release:")
            initial_max = np.max(self.da_field.field)
            self.da_field._phasic_release(0.01)  # 10ms timestep
            final_max = np.max(self.da_field.field)
            print(f"  Before: {initial_max*1e9:.1f} nM")
            print(f"  After: {final_max*1e9:.1f} nM")
            print(f"  Change: {(final_max-initial_max)*1e9:.1f} nM")
    
    def diagnose_mass_flow(self):
        """Check where all the phosphate is going"""
        # Total phosphate in system
        total_P_free = np.sum(self.phosphate_field)
        total_P_complex = np.sum(self.complex_field)
        total_P_pnc = 4 * np.sum(self.pnc_field)  # Assume 4 P per PNC
        total_P_dimer = 4 * np.sum(self.dimer_field)
        total_P_trimer = 6 * np.sum(self.trimer_field)
    
        total_P = total_P_free + total_P_complex + total_P_pnc + total_P_dimer + total_P_trimer
    
        print(f"\nPhosphate distribution:")
        print(f"  Free PO4: {total_P_free*1e3:.2f} mM")
        print(f"  In complex: {total_P_complex*1e6:.2f} µM")
        print(f"  In PNC: {total_P_pnc*1e6:.2f} µM")
        print(f"  In dimers: {total_P_dimer*1e6:.2f} µM")
        print(f"  In trimers: {total_P_trimer*1e6:.2f} µM")
        print(f"  TOTAL: {total_P*1e3:.2f} mM")
    
        # Initial phosphate was 1 mM
        if total_P > 2e-3:  # More than 2 mM
            print("ERROR: Creating phosphate from nothing!")

def analyze_quantum_advantage(results):
            """
            Analyze quantum advantage metrics from simulation results.
            Returns a dictionary with coherence advantage, dimer/trimer ratio, and dopamine-dimer correlation.
            """
            coherence_advantage = (results.max_coherence_dimer / results.max_coherence_trimer
                                if getattr(results, 'max_coherence_trimer', 0) else 0)
            dimer_trimer_ratio = (results.peak_dimer / results.peak_trimer
                                if getattr(results, 'peak_trimer', 0) else 0)
            dopamine_dimer_correlation = (
                np.corrcoef(results.dopamine_map.flatten(), results.dimer_map.flatten())[0, 1]
                if hasattr(results, 'dopamine_map') and hasattr(results, 'dimer_map') else 0
            )

            return {
                'coherence_advantage': coherence_advantage,
                'dimer_trimer_ratio': dimer_trimer_ratio,
                'dopamine_dimer_correlation': dopamine_dimer_correlation
            }



def create_actual_data_visualization(results, save_path='model5_actual_data.png'):
    """
    Create a comprehensive visualization of ACTUAL simulation data over time
    Shows real values from the simulation, not conceptual diagrams
    """
    
    # Create figure with proper layout for time series data
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.8], 
                          hspace=0.3, wspace=0.3)
    
    # Convert time to milliseconds
    time_ms = results.time * 1000
    
    # Color scheme
    colors = {
        'calcium': '#3498db',
        'phosphate': '#f39c12', 
        'pnc': '#e67e22',
        'dopamine': '#e74c3c',
        'dimer': '#9b59b6',
        'trimer': '#95a5a6',
        'coherence_d': '#1abc9c',
        'coherence_t': '#16a085',
        'learning': '#2ecc71'
    }
    
    # === PANEL 1: Calcium Dynamics ===
    ax1 = fig.add_subplot(gs[0, 0])
    # Get peak calcium at each timepoint (max across spatial dimensions)
    ca_peak = np.max(results.calcium_map, axis=(1, 2)) * 1e6  # Convert to µM
    ca_mean = np.mean(results.calcium_map, axis=(1, 2)) * 1e6
    
    ax1.plot(time_ms, ca_peak, color=colors['calcium'], linewidth=2, label='Peak')
    ax1.plot(time_ms, ca_mean, color=colors['calcium'], linewidth=1, alpha=0.5, linestyle='--', label='Mean')
    ax1.fill_between(time_ms, 0, ca_peak, alpha=0.2, color=colors['calcium'])
    
    ax1.set_title('Calcium Concentration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[Ca²⁺] (µM)', fontsize=11)
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add annotations for key events
    max_ca_idx = np.argmax(ca_peak)
    if ca_peak[max_ca_idx] > 0.1:
        ax1.annotate(f'Peak: {ca_peak[max_ca_idx]:.1f} µM', 
                    xy=(time_ms[max_ca_idx], ca_peak[max_ca_idx]),
                    xytext=(time_ms[max_ca_idx] + 50, ca_peak[max_ca_idx]),
                    arrowprops=dict(arrowstyle='->', color=colors['calcium'], alpha=0.5),
                    fontsize=9)
    
    # === PANEL 2: PNC Formation ===
    ax2 = fig.add_subplot(gs[0, 1])
    pnc_total = np.sum(results.pnc_map, axis=(1, 2)) * 1e9  # Total PNC in nM
    pnc_peak = np.max(results.pnc_map, axis=(1, 2)) * 1e9   # Peak PNC in nM
    
    ax2.plot(time_ms, pnc_total, color=colors['pnc'], linewidth=2, label='Total')
    ax2.plot(time_ms, pnc_peak, color=colors['pnc'], linewidth=1, alpha=0.5, linestyle='--', label='Peak')
    ax2.fill_between(time_ms, 0, pnc_total, alpha=0.2, color=colors['pnc'])
    
    ax2.set_title('PNC (Prenucleation Clusters)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('[PNC] (nM)', fontsize=11)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # === PANEL 3: Dopamine Release ===
    ax3 = fig.add_subplot(gs[1, 0])
    da_peak = np.max(results.dopamine_map, axis=(1, 2)) * 1e9  # Peak dopamine in nM
    da_mean = np.mean(results.dopamine_map, axis=(1, 2)) * 1e9
    
    ax3.plot(time_ms, da_peak, color=colors['dopamine'], linewidth=2, label='Peak')
    ax3.plot(time_ms, da_mean, color=colors['dopamine'], linewidth=1, alpha=0.5, linestyle='--', label='Mean')
    ax3.fill_between(time_ms, 0, da_peak, alpha=0.2, color=colors['dopamine'])
    
    # Add threshold lines
    ax3.axhline(y=100, color='red', linestyle=':', alpha=0.5, label='Quantum threshold')
    ax3.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='D2 Kd')
    
    ax3.set_title('Dopamine Concentration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('[DA] (nM)', fontsize=11)
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_ylim(1, 20000)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # === PANEL 4: Dimer vs Trimer Formation ===
    ax4 = fig.add_subplot(gs[1, 1])
    dimer_total = np.sum(results.dimer_map, axis=(1, 2)) * 1e9  # Total dimers in nM
    trimer_total = np.sum(results.trimer_map, axis=(1, 2)) * 1e9  # Total trimers in nM
    
    ax4.plot(time_ms, dimer_total, color=colors['dimer'], linewidth=2, label='Dimers (Ca₆(PO₄)₄)')
    ax4.plot(time_ms, trimer_total, color=colors['trimer'], linewidth=2, label='Trimers (Ca₉(PO₄)₆)')
    
    ax4.set_title('Quantum State Formation', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Concentration (nM)', fontsize=11)
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    # Add ratio annotation
    if np.max(trimer_total) > 0:
        final_ratio = dimer_total[-1] / trimer_total[-1] if trimer_total[-1] > 0 else np.inf
        ax4.text(0.7, 0.95, f'Final D/T Ratio: {final_ratio:.1f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # === PANEL 5: Quantum Coherence ===
    ax5 = fig.add_subplot(gs[2, 0])
    coherence_dimer = np.max(results.coherence_dimer_map, axis=(1, 2))
    coherence_trimer = np.max(results.coherence_trimer_map, axis=(1, 2))
    
    ax5.plot(time_ms, coherence_dimer, color=colors['coherence_d'], linewidth=2, label='Dimer coherence')
    ax5.plot(time_ms, coherence_trimer, color=colors['coherence_t'], linewidth=2, label='Trimer coherence')
    ax5.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.3, label='1/e threshold')
    
    ax5.set_title('Quantum Coherence', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Coherence (0-1)', fontsize=11)
    ax5.set_xlabel('Time (ms)', fontsize=11)
    ax5.set_ylim(0, 1.1)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')
    
    # === PANEL 6: Channel States ===
    ax6 = fig.add_subplot(gs[2, 1])
    # Count open channels at each timepoint
    open_channels = np.sum(results.channel_states == 1, axis=1)  # Assuming 1 = open
    
    ax6.plot(time_ms, open_channels, color='navy', linewidth=2, drawstyle='steps-post')
    ax6.fill_between(time_ms, 0, open_channels, alpha=0.3, color='navy', step='post')
    
    ax6.set_title('Active Calcium Channels', fontsize=12, fontweight='bold')
    ax6.set_ylabel('# Open Channels', fontsize=11)
    ax6.set_xlabel('Time (ms)', fontsize=11)
    ax6.set_ylim(0, results.channel_states.shape[1] + 1)
    ax6.grid(True, alpha=0.3)
    
    # === PANEL 7: Energy/ATP (if available) ===
    ax7 = fig.add_subplot(gs[3, 0])
    # This would show ATP/energy if tracked - placeholder for now
    ax7.text(0.5, 0.5, 'ATP/Energy dynamics\n(data not tracked in current results)', 
            ha='center', va='center', fontsize=11, alpha=0.5)
    ax7.set_title('Metabolic State', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Time (ms)', fontsize=11)
    ax7.set_xlim(0, time_ms[-1])
    
    # === PANEL 8: Learning Signal ===
    ax8 = fig.add_subplot(gs[3, 1])
    # Calculate learning signal over time (simplified - would need actual calculation)
    # For now, approximate as product of key factors
    ca_norm = ca_peak / (np.max(ca_peak) + 1e-10)
    da_norm = da_peak / (np.max(da_peak) + 1e-10)
    dimer_norm = dimer_total / (np.max(dimer_total) + 1e-10)
    learning_signal = ca_norm * da_norm * dimer_norm * coherence_dimer
    
    ax8.plot(time_ms, learning_signal, color=colors['learning'], linewidth=2.5)
    ax8.fill_between(time_ms, 0, learning_signal, alpha=0.3, color=colors['learning'])
    
    ax8.set_title('Learning Signal (Calculated)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Signal Strength', fontsize=11)
    ax8.set_xlabel('Time (ms)', fontsize=11)
    ax8.set_ylim(0, 1.1)
    ax8.grid(True, alpha=0.3)
    
    # === PANEL 9: Summary Statistics ===
    ax9 = fig.add_subplot(gs[4, :])
    ax9.axis('off')
    
    # Calculate key metrics
    metrics_text = f"""
    KEY METRICS FROM SIMULATION:
    
    Calcium:          Peak = {np.max(ca_peak):.2f} µM at t = {time_ms[np.argmax(ca_peak)]:.1f} ms
    Dopamine:         Peak = {np.max(da_peak):.0f} nM at t = {time_ms[np.argmax(da_peak)]:.1f} ms  
    PNC:              Max = {np.max(pnc_total):.1f} nM
    Dimers:           Max = {np.max(dimer_total):.2f} nM, Final = {dimer_total[-1]:.2f} nM
    Trimers:          Max = {np.max(trimer_total):.3f} nM, Final = {trimer_total[-1]:.3f} nM
    Dimer/Trimer:     Final ratio = {(dimer_total[-1]/trimer_total[-1] if trimer_total[-1] > 0 else 0):.1f}
    Coherence:        Dimer max = {np.max(coherence_dimer):.3f}, Trimer max = {np.max(coherence_trimer):.3f}
    Learning Signal:  Peak = {np.max(learning_signal):.3f}
    """
    
    ax9.text(0.5, 0.5, metrics_text, fontsize=11, family='monospace',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))
    
    # Main title
    fig.suptitle('Model 5: Actual Simulation Data Over Time', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved actual data visualization to '{save_path}'")
    return fig

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model with default parameters
    model = NeuromodulatedQuantumSynapse()
    
    # Run simulation
    results = model.run_simulation(
        duration=1.0,
        stim_protocol='single_spike',
        reward_time=0.2
    )
    
    # Analyze quantum advantage
    analysis = analyze_quantum_advantage(results)
    
    # Print results
    print("\n=== Model 5 Simulation Results ===")
    print(f"Peak Calcium: {results.peak_calcium:.1f} µM")
    print(f"Peak Dimer: {results.peak_dimer:.1f} nM")
    print(f"Peak Trimer: {results.peak_trimer:.1f} nM")
    print(f"Peak Dopamine: {results.peak_dopamine:.1f} nM")
    print(f"Learning Signal: {results.learning_signal:.3f}")
    print(f"Coherence Advantage: {analysis['coherence_advantage']:.1f}x")
    
    # Create visualization of actual data
    fig = create_actual_data_visualization(results)
    plt.show()
    
    # Save results
    results.save(Path("results/model5_test"))