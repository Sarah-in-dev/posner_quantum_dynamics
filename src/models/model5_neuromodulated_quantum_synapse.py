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
    
    # ============= PNC DYNAMICS (Model 4) =============
    # CaHPO4 complex formation - precursor to PNCs
    k_complex_formation: float = 1e8    # M⁻¹s⁻¹ - near diffusion limited
    k_complex_dissociation: float = 100.0  # s⁻¹ - moderate stability
    
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
    atp_baseline: float = 2.5e-3        # 2.5 mM baseline ATP
    J_PP_atp: float = 20.0              # Hz - P-P J-coupling in ATP (KEY!)
    J_PO_atp: float = 7.5               # Hz - P-O coupling
    
    # ============= DOPAMINE (Model 5) =============
    dopamine_tonic: float = 20e-9       # 20 nM tonic dopamine
    dopamine_peak: float = 1.6e-6       # 1.6 µM peak during reward
    k_dat_uptake: float = 4.0           # s⁻¹ - dopamine transporter uptake
    
    # ============= DIMER/TRIMER CHEMISTRY (Model 5) =============
    # Dimers: Ca6(PO4)4 - fewer spins, longer coherence
    # Reduce formation rates for dimers and trimers
    k_dimer_formation: float = 1e-6     # s⁻¹ - preferred with dopamine
    kr_dimer: float = 0.01              # s⁻¹ - slow dissolution (100s lifetime)
    
    # Trimers: Ca9(PO4)6 - more spins, shorter coherence  
    k_trimer_formation: float = 1e-7    # s⁻¹ - slower formation
    kr_trimer: float = 0.5              # s⁻¹ - fast dissolution (2s lifetime)
    
    # ============= QUANTUM PARAMETERS (Model 5) =============
    T2_base_dimer: float = 100.0        # 100s coherence for dimers (4 ³¹P nuclei)
    T2_base_trimer: float = 1.0         # 1s coherence for trimers (6 ³¹P nuclei)
    critical_conc_nM: float = 100.0     # nM threshold for quantum effects
    
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
        
        logger.info(f"Model 5 initialized: {self.params.n_channels} channels, "
                   f"{len(self.template_indices)} templates, "
                   f"{len(self.da_release_sites)} dopamine sites")
    
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
        This J-coupling (20 Hz) is the key quantum resource!
        """
        if activity_level > 0:
            # Activity-dependent ATP hydrolysis
            hydrolysis_rate = self.params.k_atp_hydrolysis * self.atp_field * activity_level
            
            # Enhanced hydrolysis near open channels
            for idx, (ci, cj) in enumerate(self.channel_indices):
                if idx < len(self.channels.states) and self.channels.states[idx] == 1:
                    # 10x enhancement at active channels
                    hydrolysis_rate[cj, ci] *= 10
            
            # Update ATP and phosphate
            self.atp_field -= hydrolysis_rate * dt
            self.phosphate_field += hydrolysis_rate * dt
            
            # KEY INSIGHT: Phosphate from ATP has enhanced J-coupling!
            # This is what makes it quantum-mechanically special
            self.phosphate_j_coupling = np.where(
                hydrolysis_rate > 0,
                self.params.J_PP_atp,  # 20 Hz coupling from ATP
                1.0                    # Weak coupling otherwise
            )
    
    def update_dopamine(self, dt: float, reward_signal: bool):
        """Dopamine dynamics with sustained phasic release"""
        if reward_signal:
            release_duration = 0.1  # 100 ms
            if self.dopamine_release_timer < release_duration:
                for (i, j) in self.da_release_sites:  # More explicit naming
                    before = self.dopamine_field[i, j]  # Get before value FIRST
                    release_per_timestep = self.params.dopamine_peak * dt / release_duration
                    self.dopamine_field[i, j] += release_per_timestep  # [row, col] = [y, x]
                    after = self.dopamine_field[i, j]
                
                    if self.dopamine_release_timer < 0.010:
                        print(f"Site ({i},{j}): {before*1e9:.1f} → {after*1e9:.1f} nM (added {release_per_timestep*1e9:.3f})")
                self.dopamine_release_timer += dt
        else:
            self.dopamine_release_timer = 0
    
        # Clip before diffusion
        self.dopamine_field = np.clip(self.dopamine_field, 0, 2e-6)  # Max 2 µM
    
        # Diffusion
        laplacian = self.calculate_laplacian_neumann(self.dopamine_field)
        laplacian = np.clip(laplacian, -1e10, 1e10)
        self.dopamine_field += self.params.D_dopamine * laplacian * dt
    
        # DAT reuptake
        uptake = self.params.k_dat_uptake * np.maximum(0, self.dopamine_field - self.params.dopamine_tonic)
        self.dopamine_field -= uptake * dt
        self.dopamine_field = np.maximum(self.dopamine_field, 0)
        
    
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
                
                # Combined enhancement
                total_enhancement = (cooperative_factor * ca_enhancement * 
                                    pH_factor * template_activation)
                
                # Calculate fusion probability
                fusion_prob = self.params.fusion_probability * total_enhancement
                fusion_prob = min(fusion_prob, 0.5)  # Cap at 50%
                
                # Stochastic fusion event
                if np.random.random() < fusion_prob:
                    # === DOPAMINE DETERMINES DIMER VS TRIMER ===
                    da_local = self.dopamine_field[tj, ti]
                    
                    if da_local > 100e-9:  # High dopamine (>100 nM)
                        # Form DIMERS - Ca6(PO4)4
                        # Fewer spins = longer coherence time
                        n_dimers = np.random.poisson(3)  # Average 3 dimers
                        conc_increase = n_dimers * 1e-9  # nM
                        self.dimer_field[tj, ti] += conc_increase
                    else:
                        # Form TRIMERS - Ca9(PO4)6
                        # More spins = shorter coherence time
                        n_trimers = np.random.poisson(2)  # Average 2 trimers
                        conc_increase = n_trimers * 1e-9  # nM
                        self.trimer_field[tj, ti] += conc_increase
                    
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
        Direct formation pathway based on supersaturation.
        Supplements template-mediated formation.
        """
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                if not self.active_mask[j, i]:
                    continue
                
                # Get local concentrations
                ca = self.calcium_field[j, i]
                po4 = self.phosphate_field[j, i]
                da = self.dopamine_field[j, i]
                j_coupling = self.phosphate_j_coupling[j, i]
                pH_local = self.local_pH[j, i]

                # Fix: Prevent overflow in supersaturation calculation
                max_formation = self.pnc_field[j, i] * 0.1 # Max 1% of PNCs can form dimers/trimers per timestep
                # Add absolute cap  
                max_formation = min(max_formation, 1e-7)  # Max 100 nM per timestep
                
                # Critical concentrations for formation
                ca_critical = 10e-6   # 10 µM
                po4_critical = 100e-6  # 100 µM
                
                # Calculate supersaturation
                S = (ca/ca_critical) * (po4/po4_critical)
                
                if S > 1.0:  # Supersaturated
                    # pH modulation
                    pH_factor = 10 ** (pH_local - 7.3)
                    
                    # Dopamine modulates dimer vs trimer preference
                    if da > 100e-9:  # High dopamine
                        # Debug: count how many sites have high dopamine
                        if not hasattr(self, 'da_debug_count'):
                            self.da_debug_count = 0
                        self.da_debug_count += 1

                        dimer_rate = self.params.k_dimer_formation * 10  # Enhanced
                        trimer_rate = self.params.k_trimer_formation * 0.1  # Suppressed
                    else:
                        dimer_rate = self.params.k_dimer_formation
                        trimer_rate = self.params.k_trimer_formation
                    
                    # Form dimers (enhanced by J-coupling)
                    formation = dimer_rate * (S - 1.0) * j_coupling * pH_factor * dt
                    # Apply saturation limit
                    formation = min(formation, max_formation)
                    self.dimer_field[j, i] += formation
                    
                    # Form trimers (need higher supersaturation)
                    if S > 2.0:
                        formation = trimer_rate * (S - 2.0) * pH_factor * dt
                        # Apply saturation limit
                        formation = min(formation, max_formation)
                        self.trimer_field[j, i] += formation
    
    def update_quantum_coherence(self, dt: float):
        """
        Calculate quantum coherence for dimers and trimers.
        Dopamine protects coherence, especially for dimers.
        """
        for i in range(self.params.grid_size):
            for j in range(self.params.grid_size):
                da_local = self.dopamine_field[j, i]
                
                # === DIMERS - Long baseline coherence ===
                if self.dimer_field[j, i] > 1e-12:  # If dimers present
                    T2_dimer = self.params.T2_base_dimer  # 100s baseline
                    
                    # Dopamine protection mechanism
                    if da_local > 50e-9:  # Above 50 nM
                        # Dopamine extends coherence time
                        protection = 1 + (da_local / 100e-9)  # Up to 2x protection
                        T2_dimer *= protection
                    
                    # Exponential decay of coherence
                    self.coherence_dimer[j, i] = np.exp(-dt / T2_dimer)
                else:
                    self.coherence_dimer[j, i] = 0
                
                # === TRIMERS - Short baseline coherence ===
                if self.trimer_field[j, i] > 1e-12:  # If trimers present
                    T2_trimer = self.params.T2_base_trimer  # 1s baseline
                    
                    # Dopamine can't save trimers (too many coupled spins)
                    # This is the key difference - dimers benefit from dopamine, trimers don't
                    
                    # Exponential decay
                    self.coherence_trimer[j, i] = np.exp(-dt / T2_trimer)
                else:
                    self.coherence_trimer[j, i] = 0
    
    def calculate_learning_signal(self) -> float:
        """
        Learning occurs when three conditions overlap:
        1. High dimer coherence (>0.5)
        2. High dopamine (>100 nM)
        3. Recent activity (elevated Ca)
        
        This triple coincidence is the quantum signature of learning!
        """
        learning_sites = (
            (self.coherence_dimer > 0.5) &      # Quantum coherence
            (self.dopamine_field > 100e-9) &    # Reward signal
            (self.calcium_field > 10e-6)        # Recent activity
        )
        
        # Fraction of synaptic area showing learning signature
        return np.sum(learning_sites) / (self.params.grid_size ** 2)
    
    # ========================================================================
    # MAIN UPDATE AND SIMULATION
    # ========================================================================
    
    def update_fields(self, dt: float, channel_open: np.ndarray, reward_signal: bool = False):
        """
        Main update orchestrating all dynamics.
        Order matters! Each step depends on previous ones.
        """
        # Calculate activity level
        activity_level = np.sum(channel_open) / max(1, len(channel_open))
        
        # === Core Model 4 updates ===
        # 1. pH changes affect phosphate speciation
        self.update_local_pH(activity_level)
        
        # 2. Calcium microdomains from open channels
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        
        # 3. Phosphate from ATP hydrolysis
        self.update_phosphate_from_ATP(dt, activity_level)
        
        # 4. CaHPO4 complex equilibrium
        self.calculate_complex_equilibrium()
        
        # 5. PNC dynamics with template accumulation
        self.update_pnc_dynamics(dt)
        
        # === Model 5 enhancements ===
        # 6. ATP creates J-coupling
        self.update_atp_dynamics(dt, activity_level)
        
        # 7. Dopamine if reward signal
        self.update_dopamine(dt, reward_signal)
        
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