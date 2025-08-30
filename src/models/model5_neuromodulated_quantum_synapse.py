# src/models/model5_neuromodulated_quantum_synapse.py
"""
Model 5: Neuromodulated Quantum Synapse
Builds upon Model 4's successful dynamic nanoreactor implementation
Adds: Dopamine modulation, ATP J-coupling, Dimer/Trimer pathways, Quantum coherence
"""

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

# ============================================================================
# SIMULATION RESULTS CONTAINER (Enhanced from Model 4)
# ============================================================================

@dataclass
class SimulationResults:
    """Container for simulation results - expanded from Model 4"""
    model_version: str = "5.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict = field(default_factory=dict)
    
    # Time series data (from Model 4)
    time: np.ndarray = None
    calcium_map: np.ndarray = None
    posner_map: np.ndarray = None
    pnc_map: np.ndarray = None
    channel_states: np.ndarray = None
    
    # NEW Model 5 time series
    dimer_map: np.ndarray = None
    trimer_map: np.ndarray = None
    dopamine_map: np.ndarray = None
    coherence_dimer_map: np.ndarray = None
    coherence_trimer_map: np.ndarray = None
    j_coupling_map: np.ndarray = None
    learning_signal: np.ndarray = None
    
    # Summary metrics (from Model 4)
    peak_posner: float = 0.0
    mean_posner: float = 0.0
    peak_pnc: float = 0.0
    mean_pnc: float = 0.0
    hotspot_lifetime: float = 0.0
    spatial_heterogeneity: float = 0.0
    calcium_conservation: float = 1.0
    phosphate_conservation: float = 1.0
    templates_occupied: int = 0
    fusion_events: int = 0
    
    # NEW Model 5 metrics
    peak_dimer: float = 0.0
    peak_trimer: float = 0.0
    mean_coherence_dimer: float = 0.0
    mean_coherence_trimer: float = 0.0
    max_j_coupling: float = 0.0
    peak_dopamine: float = 0.0
    learning_events: int = 0
    
    def save(self, filepath: Path):
        """Save results to HDF5 and JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save arrays to HDF5
        with h5py.File(f"{filepath}.h5", 'w') as f:
            f.attrs['model_version'] = self.model_version
            f.attrs['timestamp'] = self.timestamp
            
            # Save all array fields
            for field_name in ['time', 'calcium_map', 'posner_map', 'pnc_map', 
                             'channel_states', 'dimer_map', 'trimer_map', 
                             'dopamine_map', 'coherence_dimer_map', 
                             'coherence_trimer_map', 'j_coupling_map', 
                             'learning_signal']:
                data = getattr(self, field_name, None)
                if data is not None:
                    f.create_dataset(field_name, data=data, compression='gzip')
        
        # Save metadata to JSON
        metadata = {
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': {
                # Model 4 metrics
                'peak_posner': float(self.peak_posner),
                'mean_posner': float(self.mean_posner),
                'peak_pnc': float(self.peak_pnc),
                'mean_pnc': float(self.mean_pnc),
                'hotspot_lifetime': float(self.hotspot_lifetime),
                'spatial_heterogeneity': float(self.spatial_heterogeneity),
                'calcium_conservation': float(self.calcium_conservation),
                'phosphate_conservation': float(self.phosphate_conservation),
                'templates_occupied': int(self.templates_occupied),
                'fusion_events': int(self.fusion_events),
                # Model 5 metrics
                'peak_dimer': float(self.peak_dimer),
                'peak_trimer': float(self.peak_trimer),
                'mean_coherence_dimer': float(self.mean_coherence_dimer),
                'mean_coherence_trimer': float(self.mean_coherence_trimer),
                'max_j_coupling': float(self.max_j_coupling),
                'peak_dopamine': float(self.peak_dopamine),
                'learning_events': int(self.learning_events)
            }
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved Model 5 results to {filepath}")

# ============================================================================
# PARAMETERS (Model 4 + Model 5 enhancements)
# ============================================================================

@dataclass
class Model5Parameters:
    """
    Complete parameters for Model 5
    Includes all Model 4 parameters plus quantum/neuromodulation additions
    """
    # ============= SPATIAL PARAMETERS (from Model 4) =============
    grid_size: int = 50
    active_zone_radius: float = 200e-9  # 200 nm
    cleft_width: float = 20e-9
    
    # ============= CHANNEL PARAMETERS (from Model 4) =============
    n_channels: int = 6
    channel_current: float = 1.0e-12  # 1.0 pA
    channel_open_rate: float = 100.0
    channel_close_rate: float = 50.0
    
    # ============= BASELINE CONCENTRATIONS (from Model 4) =============
    ca_baseline: float = 100e-9  # 100 nM resting
    po4_baseline: float = 1e-3  # 1 mM
    atp_concentration: float = 3e-3  # 3 mM
    pnc_baseline: float = 1e-10  # 0.1 nM baseline (FIX #1)
    
    # ============= DIFFUSION COEFFICIENTS (from Model 4) =============
    D_calcium: float = 220e-12
    D_phosphate: float = 280e-12
    D_pnc: float = 100e-12
    D_posner: float = 50e-12
    
    # ============= PNC DYNAMICS (from Model 4) =============
    k_complex_formation: float = 1e8  # M⁻¹s⁻¹
    k_complex_dissociation: float = 100.0  # s⁻¹
    k_pnc_formation: float = 200.0  # s⁻¹
    k_pnc_dissolution: float = 10.0  # s⁻¹
    pnc_size: int = 30
    pnc_max_concentration: float = 1e-6
    
    # ============= TEMPLATE PARAMETERS (from Model 4) =============
    templates_per_synapse: int = 500
    n_binding_sites: int = 3
    k_pnc_binding: float = 1e9  # M⁻¹s⁻¹
    k_pnc_unbinding: float = 0.1  # s⁻¹
    template_accumulation_range: int = 2
    template_accumulation_rate: float = 0.3
    
    # Fusion kinetics (from Model 4)
    k_fusion_attempt: float = 10.0  # s⁻¹
    fusion_probability: float = 0.01
    pnc_per_posner: int = 2
    
    # ============= ACTIVITY-DEPENDENT (from Model 4) =============
    k_atp_hydrolysis: float = 10.0  # s⁻¹ during activity
    ph_activity_shift: float = -0.3  # pH units during activity
    
    # ============= BIOPHYSICAL FACTORS (from Model 4) =============
    f_hpo4_ph73: float = 0.49  # HPO₄²⁻ fraction at pH 7.3
    gamma_ca: float = 0.4  # Activity coefficient for Ca²⁺
    gamma_po4: float = 0.2  # Activity coefficient for HPO₄²⁻
    
    # Enhancement factors (from Model 4)
    template_factor: float = 10.0
    confinement_factor: float = 5.0
    electrostatic_factor: float = 3.0
    membrane_concentration_factor: float = 5.0
    
    # ============= NEW MODEL 5: ATP J-COUPLING =============
    J_PP_atp: float = 20.0  # Hz - ATP-derived phosphate coupling
    J_PO_atp: float = 7.5  # Hz
    D_atp: float = 150e-12  # m²/s
    
    # ============= NEW MODEL 5: DOPAMINE SYSTEM =============
    dopamine_baseline: float = 20e-9  # 20 nM tonic
    dopamine_peak: float = 1.6e-6  # 1.6 μM phasic
    D_dopamine: float = 400e-12  # m²/s
    k_dat_uptake: float = 4.0  # s⁻¹ DAT reuptake
    dopamine_threshold: float = 100e-9  # 100 nM for effect
    
    # ============= NEW MODEL 5: DIMER vs TRIMER =============
    k_dimer_formation: float = 1e-4  # Ca6(PO4)4
    k_trimer_formation: float = 1e-5  # Ca9(PO4)6 - 10x slower
    
    # ============= NEW MODEL 5: QUANTUM COHERENCE =============
    T2_base_dimer: float = 100.0  # 100s for dimers
    T2_base_trimer: float = 1.0  # 1s for trimers
    coherence_threshold: float = 0.5  # For learning signal
    
    # ============= SIMULATION PARAMETERS (from Model 4) =============
    dt: float = 0.001  # 1 ms
    duration: float = 1.0  # 1 second
    save_interval: float = 0.01  # Save every 10 ms
    
    # ============= DISSOLUTION RATES (from Model 4) =============
    kr_posner: float = 0.5  # s⁻¹

# ============================================================================
# CHANNEL DYNAMICS (Directly from Model 4)
# ============================================================================

class ChannelDynamics:
    """Handles stochastic calcium channel gating - EXACTLY from Model 4"""
    
    def __init__(self, n_channels: int, params: Model5Parameters):
        self.n_channels = n_channels
        self.params = params
        
        # Channel states
        self.CLOSED = 0
        self.OPEN = 1
        self.INACTIVATED = 2
        
        self.states = np.zeros(n_channels, dtype=int)
        
        # Transition rates
        self.rates = {
            'open': params.channel_open_rate,
            'close': params.channel_close_rate,
            'inactivate': 10.0,
            'recover': 5.0
        }
        
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """Build transition probability matrix for dt=1ms"""
        dt = 0.001
        
        Q = np.zeros((3, 3))
        
        Q[self.CLOSED, self.OPEN] = self.rates['open']
        Q[self.CLOSED, self.CLOSED] = -self.rates['open']
        
        Q[self.OPEN, self.CLOSED] = self.rates['close']
        Q[self.OPEN, self.INACTIVATED] = self.rates['inactivate']
        Q[self.OPEN, self.OPEN] = -(self.rates['close'] + self.rates['inactivate'])
        
        Q[self.INACTIVATED, self.CLOSED] = self.rates['recover']
        Q[self.INACTIVATED, self.INACTIVATED] = -self.rates['recover']
        
        self.P = np.eye(3) + Q * dt
        self.P = np.clip(self.P, 0, 1)
        
        for i in range(3):
            row_sum = self.P[i].sum()
            if row_sum > 0:
                self.P[i] /= row_sum
    
    def update(self, dt: float, depolarized: bool = True) -> np.ndarray:
        """Update channel states stochastically"""
        if not depolarized:
            self.states[:] = self.CLOSED
            return self.states
        
        for i in range(self.n_channels):
            current_state = self.states[i]
            probs = self.P[current_state]
            self.states[i] = np.random.choice(3, p=probs)
        
        return self.states
    
    def get_open_channels(self) -> np.ndarray:
        """Return boolean array of open channels"""
        return self.states == self.OPEN

# ============================================================================
# MAIN MODEL CLASS
# ============================================================================

class NeuromodulatedQuantumSynapse:
    """
    Model 5: Complete implementation building on Model 4's foundation
    """
    
    def __init__(self, params: Model5Parameters):
        self.params = params
        
        # Setup spatial grid (from Model 4)
        self.grid_shape = (params.grid_size, params.grid_size)
        self.dx = 2 * params.active_zone_radius / params.grid_size
        
        # Create coordinate system (from Model 4)
        x = np.linspace(-params.active_zone_radius, params.active_zone_radius, params.grid_size)
        y = np.linspace(-params.active_zone_radius, params.active_zone_radius, params.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Active zone mask (from Model 4)
        self.active_mask = (self.X**2 + self.Y**2) <= params.active_zone_radius**2
        
        # Membrane proximity mask (from Model 4)
        r = np.sqrt(self.X**2 + self.Y**2)
        edge_distance = params.active_zone_radius - r
        self.membrane_mask = (edge_distance < 2e-9) & (edge_distance > 0)
        
        # Initialize all fields
        self._initialize_fields()
        
        # Setup spatial structures
        self._setup_channel_positions()
        self._setup_template_positions()
        self._setup_atp_sources()  # NEW Model 5
        self._setup_dopamine_sites()  # NEW Model 5
        
        # Initialize channel dynamics (from Model 4)
        self.channels = ChannelDynamics(params.n_channels, params)
        
        # Template state tracking (from Model 4)
        self.template_pnc_bound = np.zeros(self.grid_shape)
        self.template_fusion_timer = np.zeros(self.grid_shape)
        
        # Track initial mass for conservation
        self.total_ca_initial = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        self.total_po4_initial = np.sum(self.phosphate_field * self.active_mask) * self.dx * self.dx
        
        # Metrics
        self.fusion_count = 0
        self.learning_event_count = 0
        
        logger.info(f"Model 5 initialized: {params.n_channels} channels, "
                   f"{len(self.template_indices)} templates, "
                   f"{len(self.atp_sources)} ATP sources")
    
    def _initialize_fields(self):
        """Initialize all concentration fields - FIXED version"""
        gs = self.params.grid_size
    
        # ========== Model 4 Fields ==========
        # Initialize with baseline everywhere, then mask
        self.calcium_field = np.ones(self.grid_shape) * self.params.ca_baseline
        self.calcium_field[~self.active_mask] = 0  # Zero outside active zone
    
        self.phosphate_field = np.ones(self.grid_shape) * self.params.po4_baseline
        self.phosphate_field[~self.active_mask] = 0
    
        self.complex_field = np.zeros(self.grid_shape)
    
        # PNC with baseline (FIX #1)
        self.pnc_field = np.ones(self.grid_shape) * self.params.pnc_baseline
        self.pnc_field[~self.active_mask] = 0
    
        # Posner field
        self.posner_field = np.zeros(self.grid_shape)
    
        # Activity-dependent fields
        self.local_pH = np.ones(self.grid_shape) * 7.3
        self.f_hpo4_local = np.ones(self.grid_shape) * self.params.f_hpo4_ph73
    
        # ========== NEW Model 5 Fields ==========
        # ATP
        self.atp_field = np.ones(self.grid_shape) * self.params.atp_concentration
        self.atp_field[~self.active_mask] = 0
    
        self.j_coupling_field = np.ones(self.grid_shape) * 1.0  # Baseline weak coupling
    
        # Dopamine
        self.dopamine_field = np.ones(self.grid_shape) * self.params.dopamine_baseline
        self.dopamine_field[~self.active_mask] = 0
    
        # Separate dimer and trimer tracking
        self.dimer_field = np.zeros(self.grid_shape)
        self.trimer_field = np.zeros(self.grid_shape)
    
        # Quantum coherence
        self.coherence_dimer = np.zeros(self.grid_shape)
        self.coherence_trimer = np.zeros(self.grid_shape)

    
    def _setup_channel_positions(self):
        """Position channels in hexagonal array (from Model 4)"""
        n = self.params.n_channels
        positions = []
        
        if n <= 1:
            positions = [(self.params.grid_size // 2, self.params.grid_size // 2)]
        elif n <= 7:
            center = self.params.grid_size // 2
            radius = self.params.grid_size // 6
            
            positions.append((center, center))
            for i in range(min(n-1, 6)):
                angle = i * np.pi / 3
                x = center + int(radius * np.cos(angle))
                y = center + int(radius * np.sin(angle))
                if 0 <= x < self.params.grid_size and 0 <= y < self.params.grid_size:
                    positions.append((x, y))
        
        self.channel_indices = positions[:n]
    
    def _setup_template_positions(self):
        """Position templates between channels (from Model 4)"""
        templates = []
        n_templates = min(self.params.templates_per_synapse, 
                          self.params.grid_size * self.params.grid_size // 10)
        
        if len(self.channel_indices) > 1:
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
        
        # Add random templates if needed
        while len(templates) < min(6, n_templates):
            x = np.random.randint(5, self.params.grid_size-5)
            y = np.random.randint(5, self.params.grid_size-5)
            if self.active_mask[y, x] and (x, y) not in templates:
                templates.append((x, y))
        
        self.template_indices = templates[:n_templates]
    
    def _setup_atp_sources(self):
        """Position mitochondrial ATP sources (NEW Model 5)"""
        center = self.params.grid_size // 2
        sources = []
        
        # Ring of mitochondria around active zone
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 sources
        radius = self.params.grid_size // 3
        
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            if (0 <= x < self.params.grid_size and 
                0 <= y < self.params.grid_size and
                self.active_mask[y, x]):
                sources.append((x, y))
        
        self.atp_sources = sources
    
    def _setup_dopamine_sites(self):
        """Position dopamine release sites (NEW Model 5)"""
        center = self.params.grid_size // 2
        
        # Two release sites on opposite sides
        self.dopamine_sites = [
            (max(0, center - 10), center),
            (min(self.params.grid_size-1, center + 10), center)
        ]
    
    # ========================================================================
    # CORE MODEL 4 METHODS (Preserved exactly)
    # ========================================================================
    
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """FIXED: Calculate calcium concentration with proper amplitudes"""
        # Start with baseline
        calcium = np.ones(self.grid_shape) * self.params.ca_baseline
    
        for idx, (ci, cj) in enumerate(self.channel_indices):
            if idx < len(channel_open) and channel_open[idx]:
                # Add a strong local calcium influx at the channel
                # Direct increase at channel location
                calcium[ci, cj] += 100e-6  # Add 100 µM at channel
            
                # Create microdomain around channel
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < self.params.grid_size and 
                            0 <= nj < self.params.grid_size and
                            (di != 0 or dj != 0)):
                        
                            dist = np.sqrt(di**2 + dj**2)
                            # Exponential decay from channel
                            calcium[ni, nj] += 100e-6 * np.exp(-dist/2)
    
        # Apply membrane enhancement
        calcium[self.membrane_mask] *= self.params.membrane_concentration_factor
    
        # Mask to active zone
        calcium[~self.active_mask] = 0
    
        return calcium
    
    def update_local_pH(self, activity_level: float):
        """Update local pH based on activity (FIX #3 from Model 4)"""
        pH_shift = self.params.ph_activity_shift * activity_level
        self.local_pH = 7.3 + pH_shift * self.active_mask
        
        # Update HPO4 fraction
        self.f_hpo4_local = 1 / (1 + 10**(7.2 - self.local_pH))
    
    def update_phosphate_from_ATP(self, dt: float, activity_level: float):
        """ATP hydrolysis releases phosphate during activity (FIX #5 from Model 4)"""
        if activity_level > 0:
            hydrolysis_rate = self.params.k_atp_hydrolysis * activity_level
            phosphate_production = self.params.atp_concentration * hydrolysis_rate * dt * 0.1
            
            for idx, (ci, cj) in enumerate(self.channel_indices):
                if idx < len(self.channels.states) and self.channels.states[idx] == 1:
                    r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                    enhancement = np.exp(-r / 10e-9)
                    self.phosphate_field += phosphate_production * enhancement
    
    def calculate_complex_equilibrium(self):
        """Calculate CaHPO4 complex with activity coefficients (FIX #2 from Model 4)"""
        ca_eff = self.calcium_field * self.params.gamma_ca
        po4_eff = self.phosphate_field * self.f_hpo4_local * self.params.gamma_po4
        
        ca_eff = np.clip(ca_eff, 0, 1e-3)
        po4_eff = np.clip(po4_eff, 0, 1e-3)
        
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
        
        denominator = 1 + K_eq * (ca_eff + po4_eff)
        denominator = np.maximum(denominator, 1.0)
        
        self.complex_field = K_eq * ca_eff * po4_eff / denominator
        
        # Templates enhance complex formation
        for ti, tj in self.template_indices:
            self.complex_field[tj, ti] = min(self.complex_field[tj, ti] * self.params.template_factor, 1e-6)
        
        self.complex_field *= self.active_mask
    
    def update_pnc_dynamics(self, dt: float):
        """FIXED: Update PNC field with proper scaling"""
        self.calculate_complex_equilibrium()
        
        # Formation from complexes - SCALE DOWN to prevent saturation
        j_enhancement = self.j_coupling_field / 20.0  # Normalize properly
        formation = self.params.k_pnc_formation * self.complex_field * j_enhancement * 0.01  # Scale down
        
        # Apply saturation
        saturation = 1 - self.pnc_field / self.params.pnc_max_concentration
        saturation = np.clip(saturation, 0, 1)
        formation *= saturation
        
        # Dissolution
        dissolution = self.params.k_pnc_dissolution * self.pnc_field
        
        # Template accumulation (keep from Model 4)
        for ti, tj in self.template_indices:
            if self.pnc_field[tj, ti] < self.params.pnc_max_concentration * 0.5:  # Don't oversaturate
                r_max = self.params.template_accumulation_range
                for di in range(-r_max, r_max+1):
                    for dj in range(-r_max, r_max+1):
                        ni, nj = ti + di, tj + dj
                        if (0 <= ni < self.params.grid_size and 
                            0 <= nj < self.params.grid_size and
                            (di != 0 or dj != 0) and
                            self.active_mask[nj, ni]):
                            
                            distance = np.sqrt(di**2 + dj**2)
                            drift_rate = self.params.template_accumulation_rate / (1 + distance)
                            transfer = self.pnc_field[nj, ni] * drift_rate * dt * 0.1  # Scale down
                            transfer = min(transfer, self.pnc_field[nj, ni] * 0.1)
                            
                            self.pnc_field[nj, ni] -= transfer
                            self.pnc_field[tj, ti] += transfer
                
                # Template binding
                if self.pnc_field[tj, ti] > 1e-9:
                    available = self.params.n_binding_sites - self.template_pnc_bound[tj, ti]
                    
                    if available > 0:
                        binding_rate = 1.0  # Reduced from 10.0
                        binding_amount = min(binding_rate * dt * available * 0.1, available)
                        
                        if self.pnc_field[tj, ti] > 1e-8:
                            self.template_pnc_bound[tj, ti] += binding_amount
        
        # Update PNC field
        dpnc_dt = formation - dissolution
        self.pnc_field += dpnc_dt * dt
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
        
        # Add diffusion
        laplacian = self.calculate_laplacian_neumann(self.pnc_field)
        self.pnc_field += self.params.D_pnc * laplacian * dt
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)

    
    def update_posner_formation(self, dt: float):
        """FIXED: Posner formation at templates with proper scaling"""
        for ti, tj in self.template_indices:
            occupancy = self.template_pnc_bound[tj, ti] / self.params.n_binding_sites
            
            if occupancy >= 0.5:
                # Environmental factors
                cooperative_factor = 1 + (occupancy - 0.5)
                ca_local = self.calcium_field[tj, ti]
                ca_enhancement = 1 + min(ca_local / 100e-6, 10)  # Cap enhancement
                pH_local = self.local_pH[tj, ti]
                pH_factor = 1 + 0.5 * (7.3 - pH_local)
                activity_level = np.sum(self.channels.get_open_channels()) / len(self.channels.states)
                template_activation = 1 + activity_level
                
                # Dopamine enhancement
                da_local = self.dopamine_field[tj, ti]
                da_enhancement = 1.0
                if da_local > self.params.dopamine_threshold:
                    da_enhancement = 1.5  # Reduced from 2.0
                
                total_enhancement = (cooperative_factor * ca_enhancement * 
                                   pH_factor * template_activation * da_enhancement)
                
                self.template_fusion_timer[tj, ti] += dt * total_enhancement
                
                if self.template_fusion_timer[tj, ti] > 1.0 / self.params.k_fusion_attempt:
                    self.template_fusion_timer[tj, ti] = 0
                    
                    fusion_prob = self.params.fusion_probability * min(total_enhancement, 10)
                    fusion_prob = min(fusion_prob, 0.5)
                    
                    if np.random.random() < fusion_prob:
                        # FUSION EVENT - create reasonable amounts
                        n_posner_formed = np.random.poisson(2)  # Reduced from 5
                        
                        # Avoid division by zero
                        if self.pnc_field[tj, ti] > 0:
                            # Decide between dimer and trimer
                            if da_local > self.params.dopamine_threshold:
                                # Dopamine favors dimers
                                self.dimer_field[tj, ti] += n_posner_formed * 1e-9  # 1 nM per event
                            else:
                                # Without dopamine, mostly trimers
                                self.trimer_field[tj, ti] += n_posner_formed * 0.5e-9
                            
                            # Update total Posner
                            self.posner_field[tj, ti] += n_posner_formed * 1e-9
                            
                            # Spatial spread (reduced)
                            spread_radius = 1  # Reduced from 2
                            for di in range(-spread_radius, spread_radius+1):
                                for dj in range(-spread_radius, spread_radius+1):
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = ti + di, tj + dj
                                    if (0 <= ni < self.params.grid_size and 
                                        0 <= nj < self.params.grid_size and
                                        self.active_mask[nj, ni]):
                                        distance = np.sqrt(di**2 + dj**2)
                                        spread_factor = np.exp(-distance**2)
                                        amount = n_posner_formed * 0.1e-9 * spread_factor
                                        if da_local > self.params.dopamine_threshold:
                                            self.dimer_field[nj, ni] += amount
                                        else:
                                            self.trimer_field[nj, ni] += amount
                                        self.posner_field[nj, ni] += amount
                            
                            # Reset template
                            self.template_pnc_bound[tj, ti] *= 0.5
                            
                            # PNC consumption (avoid division by zero)
                            pncs_consumed = n_posner_formed * 2
                            if self.pnc_field[tj, ti] > 1e-12:
                                consumption_factor = max(0, 1 - pncs_consumed * 1e-10 / self.pnc_field[tj, ti])
                                self.pnc_field[tj, ti] *= consumption_factor
                            
                            self.fusion_count += 1
    # ========================================================================
    # NEW MODEL 5 METHODS
    # ========================================================================
    
    def update_atp_dynamics(self, dt: float, activity_level: float):
        """
        NEW Model 5: ATP hydrolysis produces phosphate with strong J-coupling
        """
        # Base hydrolysis
        hydrolysis_rate = self.params.k_atp_hydrolysis * self.atp_field
        
        # Activity-dependent hydrolysis near channels
        for i, (x, y) in enumerate(self.channel_indices):
            if i < len(self.channels.states) and self.channels.states[i] == 1:
                dist = np.sqrt((self.X - self.X[x, y])**2 + (self.Y - self.Y[x, y])**2)
                enhancement = np.exp(-dist**2 / (2 * (5*self.dx)**2))
                hydrolysis_rate += enhancement * 10 * self.params.k_atp_hydrolysis
        
        # Update ATP and phosphate
        atp_consumed = hydrolysis_rate * dt
        self.atp_field -= atp_consumed
        self.phosphate_field += atp_consumed  # 1:1 stoichiometry
        
        # Critical: Update J-coupling field based on ATP-derived phosphate
        self.j_coupling_field = np.where(
            hydrolysis_rate > 1e-6,
            self.params.J_PP_atp,  # Strong coupling (20 Hz) from ATP
            1.0  # Weak background coupling
        )
        
        # Replenish ATP from mitochondria
        for (x, y) in self.atp_sources:
            self.atp_field[x, y] = self.params.atp_concentration
    
    def update_dopamine(self, dt: float, reward_signal: bool = False):
        """
        NEW Model 5: Phasic dopamine release and reuptake
        """
        # Release at dopamine sites if reward signal
        if reward_signal:
            for (x, y) in self.dopamine_sites:
                if (0 <= x < self.params.grid_size and 
                    0 <= y < self.params.grid_size):
                    self.dopamine_field[x, y] += self.params.dopamine_peak * dt
        
        # DAT reuptake (first-order kinetics)
        excess = self.dopamine_field - self.params.dopamine_baseline
        uptake = self.params.k_dat_uptake * np.maximum(excess, 0) * dt
        self.dopamine_field -= uptake
        
        # Ensure baseline is maintained
        self.dopamine_field = np.maximum(self.dopamine_field, self.params.dopamine_baseline)
    
    def form_dimers_and_trimers(self, dt: float):
        """FIXED: Form dimers or trimers with reasonable rates"""
        # Get local concentrations
        ca = self.calcium_field
        po4 = self.phosphate_field
        da = self.dopamine_field
        pH = self.local_pH
        j_coupling = self.j_coupling_field

        # Only form where calcium is elevated
        ca_elevated = ca > 10e-6  # 10 µM threshold

        # pH effect on formation
        pH_factor = 10 ** (7.0 - pH)  # Inverted - lower pH favors formation

        # Dopamine modulation
        dimer_enhancement = np.where(da > self.params.dopamine_threshold, 2.0, 1.0)  # Reduced from 10
        trimer_suppression = np.where(da > self.params.dopamine_threshold, 0.5, 1.0)  # Less suppression

        # J-coupling enhancement
        coupling_factor = j_coupling / 20.0  # Normalize to max value

        # Form dimers: Ca6(PO4)4 - but with more reasonable kinetics
        # Use linear dependence on Ca and PO4 instead of high powers
        dimer_rate = np.where(ca_elevated,
                              self.params.k_dimer_formation *
                              ca * po4 *  # Linear, not 6th and 4th power
                              pH_factor * coupling_factor * dimer_enhancement,
                              0)

        self.dimer_field += dimer_rate * dt

        # Form trimers: Ca9(PO4)6 - also simplified
        trimer_rate = np.where(ca_elevated,
                               self.params.k_trimer_formation *
                               ca * po4 *  # Linear
                               pH_factor * trimer_suppression,
                               0)

        self.trimer_field += trimer_rate * dt

        # Apply saturation
        max_posner = 100e-9  # 100 nM max
        self.dimer_field = np.minimum(self.dimer_field, max_posner)
        self.trimer_field = np.minimum(self.trimer_field, max_posner)

        # Update total Posner
        self.posner_field = self.dimer_field + self.trimer_field

        # Mask to active zone
        self.dimer_field[~self.active_mask] = 0
        self.trimer_field[~self.active_mask] = 0
        self.posner_field[~self.active_mask] = 0
    
    def update_quantum_coherence(self, dt: float):
        """FIXED: Calculate quantum coherence avoiding NaN"""
        da = self.dopamine_field
        
        # Dimers - inherently longer coherence
        dimer_present = self.dimer_field > 1e-12
        
        if np.any(dimer_present):
            # Initialize coherence for new dimers
            new_dimers = dimer_present & (self.coherence_dimer == 0)
            self.coherence_dimer[new_dimers] = 1.0
            
            # Calculate T2 with dopamine protection
            T2_dimer = self.params.T2_base_dimer
            protection = np.where(da > self.params.dopamine_threshold,
                                  1 + (da - self.params.dopamine_threshold) / 1e-6,
                                  1.0)
            T2_dimer_protected = T2_dimer * protection
            
            # Decay coherence
            decay_rate = 1.0 / np.maximum(T2_dimer_protected, 1.0)  # Avoid division by zero
            self.coherence_dimer *= np.exp(-dt * decay_rate)
            self.coherence_dimer[~dimer_present] = 0
        
        # Trimers - short coherence, no protection
        trimer_present = self.trimer_field > 1e-12
        
        if np.any(trimer_present):
            # Initialize coherence for new trimers
            new_trimers = trimer_present & (self.coherence_trimer == 0)
            self.coherence_trimer[new_trimers] = 1.0
            
            # Decay (no dopamine protection for trimers)
            self.coherence_trimer *= np.exp(-dt / self.params.T2_base_trimer)
            self.coherence_trimer[~trimer_present] = 0
    
    def calculate_learning_signal(self) -> float:
        """
        NEW Model 5: Learning requires AND logic of:
        1. Coherent dimers (not trimers)
        2. Dopamine above threshold
        3. Recent calcium activity
        """
        # Identify learning-competent sites
        learning_sites = (
            (self.coherence_dimer > self.params.coherence_threshold) &
            (self.dopamine_field > self.params.dopamine_threshold) &
            (self.calcium_field > 10e-6)  # Recent activity
        )
        
        # Learning signal strength
        if np.any(learning_sites):
            signal = np.mean(
                self.coherence_dimer[learning_sites] * 
                self.dopamine_field[learning_sites] / self.params.dopamine_peak
            )
            
            # Track learning events
            if signal > 0.1:  # Threshold for counting as event
                self.learning_event_count += 1
        else:
            signal = 0.0
        
        return signal
    
    def calculate_laplacian_neumann(self, field: np.ndarray) -> np.ndarray:
        """Laplacian with no-flux boundaries (from Model 4)"""
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
    
    def update_fields(self, dt: float, channel_open: np.ndarray, reward_signal: bool = False):
        """
        Main update routine integrating Model 4 and Model 5 dynamics
        """
        # Calculate activity level
        activity_level = np.sum(channel_open) / max(1, len(channel_open))
        
        # Model 4 updates
        self.update_local_pH(activity_level)
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        self.update_phosphate_from_ATP(dt, activity_level)
        self.calculate_complex_equilibrium()
        
        # Model 5: ATP dynamics affect J-coupling
        self.update_atp_dynamics(dt, activity_level)
        
        # Model 5: Dopamine if reward present
        self.update_dopamine(dt, reward_signal)
        
        # PNC dynamics (Model 4 with Model 5 J-coupling)
        self.update_pnc_dynamics(dt)
        
        # Posner formation (Model 4 method creates dimers/trimers in Model 5)
        self.update_posner_formation(dt)
        
        # Model 5: Direct dimer/trimer formation
        self.form_dimers_and_trimers(dt)
        
        # Model 5: Update quantum coherence
        self.update_quantum_coherence(dt)
        
        # Posner dissolution
        dissolution = self.params.kr_posner * self.posner_field
        self.posner_field -= dissolution * dt
        self.posner_field = np.maximum(self.posner_field, 0)
        
        # Also apply to dimers and trimers
        self.dimer_field -= self.params.kr_posner * self.dimer_field * dt
        self.trimer_field -= self.params.kr_posner * self.trimer_field * dt
        self.dimer_field = np.maximum(self.dimer_field, 0)
        self.trimer_field = np.maximum(self.trimer_field, 0)
    
    def run_simulation(self, duration: float = None, 
                      stim_protocol: str = 'single_spike',
                      reward_protocol: str = None) -> SimulationResults:
        """
        Run complete simulation with stimulus and reward protocols
        """
        if duration is None:
            duration = self.params.duration
        
        # Create results container
        results = SimulationResults(parameters=asdict(self.params))
        
        # Setup time
        n_steps = int(duration / self.params.dt)
        save_interval = int(self.params.save_interval / self.params.dt)
        n_saves = n_steps // save_interval + 1
        
        # Initialize arrays
        results.time = np.arange(0, duration + self.params.dt, self.params.save_interval)[:n_saves]
        results.posner_map = np.zeros((n_saves, *self.grid_shape))
        results.pnc_map = np.zeros((n_saves, *self.grid_shape))
        results.calcium_map = np.zeros((n_saves, *self.grid_shape))
        results.channel_states = np.zeros((n_saves, self.params.n_channels))
        
        # Model 5 specific arrays
        results.dimer_map = np.zeros((n_saves, *self.grid_shape))
        results.trimer_map = np.zeros((n_saves, *self.grid_shape))
        results.dopamine_map = np.zeros((n_saves, *self.grid_shape))
        results.coherence_dimer_map = np.zeros((n_saves, *self.grid_shape))
        results.coherence_trimer_map = np.zeros((n_saves, *self.grid_shape))
        results.j_coupling_map = np.zeros((n_saves, *self.grid_shape))
        results.learning_signal = np.zeros(n_saves)
        
        # Stimulus protocol
        depolarized = np.zeros(n_steps, dtype=bool)
        if stim_protocol == 'single_spike':
            depolarized[int(0.1/self.params.dt):int(0.15/self.params.dt)] = True
        elif stim_protocol == 'tetanus':
            for i in range(10):
                start = int((0.1 + i*0.05)/self.params.dt)
                depolarized[start:start+int(0.005/self.params.dt)] = True
        elif stim_protocol == 'continuous':
            depolarized[:] = True
        
        # Reward protocol (Model 5)
        reward_signal = np.zeros(n_steps, dtype=bool)
        if reward_protocol == 'delayed':
            # Reward 200ms after stimulus
            reward_signal[int(0.3/self.params.dt):int(0.35/self.params.dt)] = True
        elif reward_protocol == 'coincident':
            # Reward with stimulus
            reward_signal = depolarized.copy()
        elif reward_protocol == 'continuous':
            reward_signal[:] = True
        
        # Reset counters
        self.fusion_count = 0
        self.learning_event_count = 0
        
        # Main simulation loop
        save_idx = 0
        for step in range(n_steps):
            # Update channels
            self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
            
            # Update all fields
            self.update_fields(self.params.dt, channel_open, reward_signal[step])
            
            # Calculate learning signal
            learning = self.calculate_learning_signal()
            
            # Save periodically
            if step % save_interval == 0:
                results.posner_map[save_idx] = self.posner_field
                results.pnc_map[save_idx] = self.pnc_field
                results.calcium_map[save_idx] = self.calcium_field
                results.channel_states[save_idx] = self.channels.states
                
                # Model 5 specific
                results.dimer_map[save_idx] = self.dimer_field
                results.trimer_map[save_idx] = self.trimer_field
                results.dopamine_map[save_idx] = self.dopamine_field
                results.coherence_dimer_map[save_idx] = self.coherence_dimer
                results.coherence_trimer_map[save_idx] = self.coherence_trimer
                results.j_coupling_map[save_idx] = self.j_coupling_field
                results.learning_signal[save_idx] = learning
                
                save_idx += 1
                
                if step % 1000 == 0:
                    logger.info(f"Step {step}: Posner={np.max(self.posner_field)*1e9:.1f} nM, "
                               f"Dimers={np.max(self.dimer_field)*1e9:.1f} nM, "
                               f"Learning={learning:.3f}")
        
        # Calculate final metrics
        results.peak_posner = np.max(results.posner_map) * 1e9
        results.mean_posner = np.mean(results.posner_map[results.posner_map > 0]) * 1e9 if np.any(results.posner_map > 0) else 0
        results.peak_pnc = np.max(results.pnc_map) * 1e9
        results.mean_pnc = np.mean(results.pnc_map[results.pnc_map > 0]) * 1e9 if np.any(results.pnc_map > 0) else 0
        
        # Model 5 metrics
        results.peak_dimer = np.max(results.dimer_map) * 1e9
        results.peak_trimer = np.max(results.trimer_map) * 1e9
        results.mean_coherence_dimer = np.mean(results.coherence_dimer_map[results.coherence_dimer_map > 0]) if np.any(results.coherence_dimer_map > 0) else 0
        results.mean_coherence_trimer = np.mean(results.coherence_trimer_map[results.coherence_trimer_map > 0]) if np.any(results.coherence_trimer_map > 0) else 0
        results.max_j_coupling = np.max(results.j_coupling_map)
        results.peak_dopamine = np.max(results.dopamine_map) * 1e9
        results.learning_events = self.learning_event_count
        
        results.templates_occupied = np.sum(self.template_pnc_bound > 0)
        results.fusion_events = self.fusion_count
        
        # Check mass balance
        final_ca_total = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        results.calcium_conservation = final_ca_total / self.total_ca_initial if self.total_ca_initial > 0 else 1.0
        
        logger.info(f"Simulation complete. Peak Dimers: {results.peak_dimer:.2f} nM, "
                   f"Peak Trimers: {results.peak_trimer:.2f} nM, "
                   f"Learning events: {results.learning_events}")
        
        return results

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_spatial_pattern(results: SimulationResults) -> Dict:
    """Analyze spatial distribution of Posner formation"""
    analysis = {}
    
    # Use final timepoint for spatial analysis
    final_posner = results.posner_map[-1] if results.posner_map is not None else np.zeros((50, 50))
    final_dimer = results.dimer_map[-1] if results.dimer_map is not None else np.zeros((50, 50))
    final_trimer = results.trimer_map[-1] if results.trimer_map is not None else np.zeros((50, 50))
    
    if np.any(final_posner > 0):
        # Identify peaks
        local_maxima = (final_posner == maximum_filter(final_posner, size=3))
        
        analysis['n_hotspots'] = np.sum(local_maxima)
        analysis['hotspot_positions'] = np.argwhere(local_maxima)
        analysis['hotspot_concentrations'] = final_posner[local_maxima] * 1e9
        
        # Spatial heterogeneity
        if np.mean(final_posner) > 0:
            analysis['spatial_cv'] = np.std(final_posner) / np.mean(final_posner)
        else:
            analysis['spatial_cv'] = 0
        
        # Dimer vs Trimer distribution
        analysis['dimer_sites'] = np.sum(final_dimer > 1e-12)
        analysis['trimer_sites'] = np.sum(final_trimer > 1e-12)
        analysis['dimer_fraction'] = np.sum(final_dimer) / (np.sum(final_dimer) + np.sum(final_trimer) + 1e-10)
    else:
        analysis['n_hotspots'] = 0
        analysis['hotspot_positions'] = []
        analysis['hotspot_concentrations'] = []
        analysis['spatial_cv'] = 0
        analysis['dimer_sites'] = 0
        analysis['trimer_sites'] = 0
        analysis['dimer_fraction'] = 0
    
    return analysis

def analyze_temporal_dynamics(results: SimulationResults) -> Dict:
    """Analyze temporal evolution of Model 5 dynamics"""
    analysis = {}
    
    if results.time is not None:
        # Peak concentrations over time
        posner_peaks = np.max(results.posner_map.reshape(len(results.time), -1), axis=1) * 1e9
        dimer_peaks = np.max(results.dimer_map.reshape(len(results.time), -1), axis=1) * 1e9
        trimer_peaks = np.max(results.trimer_map.reshape(len(results.time), -1), axis=1) * 1e9
        
        # Formation kinetics
        analysis['posner_rise_time'] = results.time[np.argmax(posner_peaks > 1.0)] if np.any(posner_peaks > 1.0) else np.inf
        analysis['dimer_rise_time'] = results.time[np.argmax(dimer_peaks > 0.1)] if np.any(dimer_peaks > 0.1) else np.inf
        
        # Coherence dynamics
        analysis['max_coherence_dimer'] = np.max(results.coherence_dimer_map)
        analysis['coherence_lifetime'] = np.sum(results.coherence_dimer_map > 0.5) * results.time[1]
        
        # Learning signal analysis
        analysis['learning_peaks'] = np.sum(results.learning_signal > 0.1)
        analysis['max_learning_signal'] = np.max(results.learning_signal)
    else:
        analysis['posner_rise_time'] = np.inf
        analysis['dimer_rise_time'] = np.inf
        analysis['max_coherence_dimer'] = 0
        analysis['coherence_lifetime'] = 0
        analysis['learning_peaks'] = 0
        analysis['max_learning_signal'] = 0
    
    return analysis

def analyze_quantum_metrics(results: SimulationResults) -> Dict:
    """Analyze quantum-specific metrics for Model 5"""
    analysis = {}
    
    if results.coherence_dimer_map is not None:
        # Average coherence over space and time
        analysis['mean_dimer_coherence'] = np.mean(results.coherence_dimer_map[results.coherence_dimer_map > 0])
        analysis['mean_trimer_coherence'] = np.mean(results.coherence_trimer_map[results.coherence_trimer_map > 0])
        
        # Coherence protection by dopamine
        high_da_mask = results.dopamine_map > 100e-9
        analysis['coherence_with_da'] = np.mean(results.coherence_dimer_map[high_da_mask])
        analysis['coherence_without_da'] = np.mean(results.coherence_dimer_map[~high_da_mask])
        
        # J-coupling effectiveness
        analysis['mean_j_coupling'] = np.mean(results.j_coupling_map)
        analysis['max_j_coupling'] = np.max(results.j_coupling_map)
        
        # Learning competence
        analysis['learning_events'] = results.learning_events
        analysis['peak_learning_signal'] = np.max(results.learning_signal)
    else:
        analysis = {key: 0 for key in ['mean_dimer_coherence', 'mean_trimer_coherence',
                                       'coherence_with_da', 'coherence_without_da',
                                       'mean_j_coupling', 'max_j_coupling',
                                       'learning_events', 'peak_learning_signal']}
    
    return analysis