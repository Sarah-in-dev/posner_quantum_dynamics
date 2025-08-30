# src/models/model5_complete.py
"""
Model 5: Neuromodulated Quantum Synapse
Builds on Model 4's successful dynamic nanoreactor with:
- ATP J-coupling (100x stronger than isolated Posner)
- Dimer vs Trimer pathways (Agarwal et al. 2023)
- Dopamine modulation of quantum coherence
- Literature-based parameters
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
# ENHANCED RESULTS CONTAINER
# ============================================================================

@dataclass
class Model5Results:
    """Extended container for Model 5 simulation results"""
    model_version: str = "5.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict = field(default_factory=dict)
    
    # Time series from Model 4
    time: np.ndarray = None
    calcium_map: np.ndarray = None
    phosphate_map: np.ndarray = None
    pnc_map: np.ndarray = None
    channel_states: np.ndarray = None
    
    # NEW Model 5 time series
    atp_map: np.ndarray = None
    dopamine_map: np.ndarray = None
    ph_map: np.ndarray = None
    dimer_map: np.ndarray = None
    trimer_map: np.ndarray = None
    coherence_dimer_map: np.ndarray = None
    coherence_trimer_map: np.ndarray = None
    j_coupling_map: np.ndarray = None
    learning_signal: np.ndarray = None
    
    # Metrics from Model 4
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
    dimer_trimer_ratio: float = 0.0
    mean_coherence_dimer: float = 0.0
    mean_coherence_trimer: float = 0.0
    max_j_coupling: float = 0.0
    peak_dopamine: float = 0.0
    learning_events: int = 0
    quantum_advantage: float = 0.0  # Ratio of quantum to classical learning rate
    
    def save(self, filepath: Path):
        """Save results to HDF5 and JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save arrays to HDF5
        with h5py.File(f"{filepath}.h5", 'w') as f:
            f.attrs['model_version'] = self.model_version
            f.attrs['timestamp'] = self.timestamp
            
            # Save all array fields
            array_fields = ['time', 'calcium_map', 'phosphate_map', 'pnc_map', 
                          'channel_states', 'atp_map', 'dopamine_map', 'ph_map',
                          'dimer_map', 'trimer_map', 'coherence_dimer_map',
                          'coherence_trimer_map', 'j_coupling_map', 'learning_signal']
            
            for field_name in array_fields:
                data = getattr(self, field_name, None)
                if data is not None:
                    f.create_dataset(field_name, data=data, compression='gzip')
        
        # Save metadata to JSON
        metadata = {
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': {key: float(getattr(self, key)) for key in dir(self) 
                       if not key.startswith('_') and isinstance(getattr(self, key), (int, float))}
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved Model 5 results to {filepath}")

# ============================================================================
# COMPREHENSIVE PARAMETERS
# ============================================================================

@dataclass
class Model5Parameters:
    """
    Complete parameters incorporating Model 4 successes and literature findings
    """
    # ============= SPATIAL (from Model 4) =============
    grid_size: int = 50
    active_zone_radius: float = 200e-9  # 200 nm
    cleft_width: float = 20e-9
    
    # ============= CHANNELS (from Model 4) =============
    n_channels: int = 6
    channel_current: float = 0.3e-12  # 0.3 pA (realistic single channel)
    channel_open_rate: float = 100.0  # s⁻¹
    channel_close_rate: float = 50.0   # s⁻¹
    
    # ============= CALCIUM (validated in Model 4) =============
    ca_baseline: float = 100e-9  # 100 nM
    ca_peak: float = 300e-6  # 300 µM at channel mouth
    D_calcium: float = 220e-12  # m²/s
    
    # ============= ATP/PHOSPHATE (literature-based) =============
    atp_baseline: float = 2.5e-3  # 2.5 mM (Magistretti & Allaman, 2015)
    atp_synaptic: float = 5e-3  # 5 mM near mitochondria
    k_atp_hydrolysis: float = 0.1  # s⁻¹ (Campanella et al., 2009)
    
    # CRITICAL: J-coupling from ATP is 100x stronger!
    J_PP_atp: float = 20.0  # Hz in ATP (Jung et al., 1997)
    J_PP_isolated: float = 0.2  # Hz in isolated Posner (theoretical)
    
    po4_baseline: float = 1e-3  # 1 mM
    D_phosphate: float = 280e-12  # m²/s
    D_atp: float = 150e-12  # m²/s
    
    # ============= PNC (from Model 4, adjusted) =============
    k_complex_formation: float = 1e8  # M⁻¹s⁻¹
    k_complex_dissociation: float = 100.0  # s⁻¹
    k_pnc_formation_base: float = 10.0  # s⁻¹ (reduced from Model 4)
    k_pnc_dissolution: float = 1.0  # s⁻¹
    pnc_baseline: float = 1e-10  # 0.1 nM
    pnc_max_concentration: float = 100e-9  # 100 nM (reduced from 1000)
    
    # ============= TEMPLATES (from Model 4) =============
    templates_per_synapse: int = 15  # Reduced for Model 5
    n_binding_sites: int = 3
    k_pnc_binding: float = 1e8  # M⁻¹s⁻¹
    k_pnc_unbinding: float = 0.1  # s⁻¹
    template_accumulation_range: int = 2
    template_accumulation_rate: float = 0.3
    k_fusion_attempt: float = 10.0  # s⁻¹
    fusion_probability: float = 0.01
    
    # ============= DOPAMINE (NEW - literature) =============
    dopamine_baseline: float = 20e-9  # 20 nM tonic (Garris et al., 1994)
    dopamine_peak: float = 1.6e-6  # 1.6 µM phasic (Rice & Cragg, 2008)
    D_dopamine: float = 400e-12  # m²/s (Ford, 2014)
    k_dat_uptake: float = 4.0  # s⁻¹ DAT reuptake
    dopamine_threshold: float = 50e-9  # 50 nM for quantum effects
    dopamine_D1_Kd: float = 1e-9  # 1 nM (Richfield et al., 1989)
    dopamine_D2_Kd: float = 10e-9  # 10 nM
    
    # ============= DIMER vs TRIMER (NEW - Agarwal et al.) =============
    # Formation based on supersaturation, not raw powers
    k_dimer_formation: float = 1e-3  # s⁻¹ (adjusted for supersaturation)
    k_trimer_formation: float = 1e-4  # s⁻¹ (10x slower)
    Ksp_dimer: float = 1e-36  # Solubility product Ca₆(PO₄)₄
    Ksp_trimer: float = 1e-54  # Solubility product Ca₉(PO₄)₆
    
    # ============= QUANTUM COHERENCE (NEW - Agarwal et al.) =============
    T2_dimer_base: float = 100.0  # 100-1000s for dimers!
    T2_trimer_base: float = 0.8  # <1s for trimers
    coherence_threshold: float = 0.5
    
    # ============= pH (from Model 4 + literature) =============
    pH_baseline: float = 7.3
    pH_cleft: float = 7.2
    pH_activity_drop: float = 0.2  # During vesicle fusion
    pH_recovery_tau: float = 0.5  # seconds
    f_hpo4_ph73: float = 0.49  # HPO₄²⁻ fraction at pH 7.3
    
    # ============= ACTIVITY FACTORS (from Model 4) =============
    gamma_ca: float = 0.4  # Activity coefficient Ca²⁺
    gamma_po4: float = 0.2  # Activity coefficient HPO₄²⁻
    template_factor: float = 10.0
    membrane_concentration_factor: float = 5.0
    
    # ============= DISSOLUTION =============
    kr_dimer: float = 0.01  # s⁻¹ - slow for stable dimers
    kr_trimer: float = 1.0  # s⁻¹ - fast for unstable trimers
    
    # ============= SIMULATION =============
    dt: float = 0.001  # 1 ms
    duration: float = 1.0  # 1 second
    save_interval: float = 0.01  # Save every 10 ms

# ============================================================================
# CHANNEL DYNAMICS (from Model 4)
# ============================================================================

class ChannelDynamics:
    """Stochastic calcium channel gating - directly from Model 4"""
    
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
        """Build transition probability matrix"""
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
        
        # Spatial setup from Model 4
        self.grid_shape = (params.grid_size, params.grid_size)
        self.dx = 2 * params.active_zone_radius / params.grid_size
        
        # Coordinate system
        x = np.linspace(-params.active_zone_radius, params.active_zone_radius, params.grid_size)
        y = np.linspace(-params.active_zone_radius, params.active_zone_radius, params.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Masks
        self.active_mask = (self.X**2 + self.Y**2) <= params.active_zone_radius**2
        r = np.sqrt(self.X**2 + self.Y**2)
        edge_distance = params.active_zone_radius - r
        self.membrane_mask = (edge_distance < 2e-9) & (edge_distance > 0)
        
        # Initialize all fields
        self._initialize_fields()
        
        # Setup structures
        self._setup_channel_positions()
        self._setup_template_positions()
        self._setup_atp_sources()
        self._setup_dopamine_sites()
        
        # Initialize dynamics
        self.channels = ChannelDynamics(params.n_channels, params)
        
        # Template tracking
        self.template_pnc_bound = np.zeros(self.grid_shape)
        self.template_fusion_timer = np.zeros(self.grid_shape)
        
        # Mass conservation tracking
        self.total_ca_initial = np.sum(self.calcium_field[self.active_mask]) * self.dx * self.dx
        self.total_po4_initial = np.sum(self.phosphate_field[self.active_mask]) * self.dx * self.dx
        
        # Counters
        self.fusion_count = 0
        self.learning_event_count = 0
        
        logger.info(f"Model 5 initialized: {params.n_channels} channels, "
                   f"{len(self.template_indices)} templates")
    
    def _initialize_fields(self):
        """Initialize all concentration fields"""
        # Model 4 fields
        self.calcium_field = np.ones(self.grid_shape) * self.params.ca_baseline
        self.calcium_field[~self.active_mask] = 0
        
        self.phosphate_field = np.ones(self.grid_shape) * self.params.po4_baseline
        self.phosphate_field[~self.active_mask] = 0
        
        self.complex_field = np.zeros(self.grid_shape)
        
        self.pnc_field = np.ones(self.grid_shape) * self.params.pnc_baseline
        self.pnc_field[~self.active_mask] = 0
        
        # pH and activity
        self.local_pH = np.ones(self.grid_shape) * self.params.pH_baseline
        self.f_hpo4_local = np.ones(self.grid_shape) * self.params.f_hpo4_ph73
        
        # Model 5 additions
        self.atp_field = np.ones(self.grid_shape) * self.params.atp_baseline
        self.atp_field[~self.active_mask] = 0
        
        self.dopamine_field = np.ones(self.grid_shape) * self.params.dopamine_baseline
        self.dopamine_field[~self.active_mask] = 0
        
        self.j_coupling_field = np.ones(self.grid_shape) * self.params.J_PP_isolated
        
        # Separate dimer/trimer tracking
        self.dimer_field = np.zeros(self.grid_shape)
        self.trimer_field = np.zeros(self.grid_shape)
        
        # Quantum coherence
        self.coherence_dimer = np.zeros(self.grid_shape)
        self.coherence_trimer = np.zeros(self.grid_shape)
    
    def _setup_channel_positions(self):
        """Hexagonal channel array from Model 4"""
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
        """Templates between channels from Model 4"""
        templates = []
        n_templates = self.params.templates_per_synapse
        
        # Between channel pairs
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
        """Mitochondrial ATP sources - NEW"""
        center = self.params.grid_size // 2
        sources = []
        
        # Ring around active zone
        angles = np.linspace(0, 2*np.pi, 9)[:-1]
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
        """Dopamine release sites - NEW"""
        center = self.params.grid_size // 2
        self.dopamine_sites = [
            (max(0, center - 10), center),
            (min(self.params.grid_size-1, center + 10), center)
        ]
    
    # ========================================================================
    # CORE DYNAMICS FROM MODEL 4
    # ========================================================================
    
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """Calculate calcium with realistic microdomains - from Model 4"""
        calcium = np.ones(self.grid_shape) * self.params.ca_baseline
        
        for idx, (ci, cj) in enumerate(self.channel_indices):
            if idx < len(channel_open) and channel_open[idx]:
                # Flux calculation from Model 4
                r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                r = np.maximum(r, 1e-9)
                
                flux_ions_per_sec = self.params.channel_current / (2 * 1.602e-19)
                flux_molar = flux_ions_per_sec / 6.022e23
                
                # Steady-state from point source
                single_ca = flux_molar / (4 * np.pi * self.params.D_calcium * r)
                calcium += single_ca
        
        # Membrane enhancement
        calcium[self.membrane_mask] *= self.params.membrane_concentration_factor
        
        # Buffering
        kappa = 5
        calcium_buffered = calcium / (1 + kappa)
        calcium_final = calcium_buffered / (1 + calcium_buffered/10e-3)
        
        return calcium_final * self.active_mask
    
    def update_local_pH(self, activity_level: float):
        """pH dynamics from Model 4"""
        pH_shift = self.params.pH_activity_drop * activity_level
        self.local_pH = self.params.pH_baseline - pH_shift * self.active_mask
        
        # Update phosphate fraction
        self.f_hpo4_local = 1 / (1 + 10**(7.2 - self.local_pH))
    
    def calculate_complex_equilibrium(self):
        """CaHPO4 complex from Model 4"""
        ca_eff = self.calcium_field * self.params.gamma_ca
        po4_eff = self.phosphate_field * self.f_hpo4_local * self.params.gamma_po4
        
        ca_eff = np.clip(ca_eff, 0, 1e-3)
        po4_eff = np.clip(po4_eff, 0, 1e-3)
        
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
        
        denominator = 1 + K_eq * (ca_eff + po4_eff)
        denominator = np.maximum(denominator, 1.0)
        
        self.complex_field = K_eq * ca_eff * po4_eff / denominator
        
        # Template enhancement
        for ti, tj in self.template_indices:
            self.complex_field[tj, ti] = min(
                self.complex_field[tj, ti] * self.params.template_factor, 1e-6
            )
        
        self.complex_field *= self.active_mask
    
    def update_pnc_dynamics(self, dt: float):
        """PNC dynamics with J-coupling enhancement - MODIFIED for Model 5"""
        self.calculate_complex_equilibrium()
        
        # Formation enhanced by J-coupling from ATP
        j_enhancement = self.j_coupling_field / self.params.J_PP_isolated
        formation = self.params.k_pnc_formation_base * self.complex_field * j_enhancement
        
        # Saturation
        saturation = 1 - self.pnc_field / self.params.pnc_max_concentration
        saturation = np.clip(saturation, 0, 1)
        formation *= saturation
        
        # Dissolution
        dissolution = self.params.k_pnc_dissolution * self.pnc_field
        
        # Template accumulation from Model 4
        for ti, tj in self.template_indices:
            if self.pnc_field[tj, ti] < self.params.pnc_max_concentration * 0.8:
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
                            transfer = self.pnc_field[nj, ni] * drift_rate * dt * 0.1
                            transfer = min(transfer, self.pnc_field[nj, ni] * 0.1)
                            
                            self.pnc_field[nj, ni] -= transfer
                            self.pnc_field[tj, ti] += transfer
            
            # Binding
            if self.pnc_field[tj, ti] > 1e-9:
                available = self.params.n_binding_sites - self.template_pnc_bound[tj, ti]
                if available > 0:
                    binding_rate = self.params.k_pnc_binding * self.pnc_field[tj, ti]
                    binding_amount = min(binding_rate * dt * available, available)
                    
                    if self.pnc_field[tj, ti] > 1e-8:
                        self.template_pnc_bound[tj, ti] += binding_amount
        
        # Update field
        dpnc_dt = formation - dissolution
        self.pnc_field += dpnc_dt * dt * self.active_mask
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
        
        # Diffusion
        laplacian = self.calculate_laplacian_neumann(self.pnc_field)
        self.pnc_field += self.params.D_phosphate * laplacian * dt
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
    
    # ========================================================================
    # NEW MODEL 5 DYNAMICS
    # ========================================================================
    
    def update_atp_dynamics(self, dt: float, activity_level: float):
        """ATP hydrolysis creates phosphate with strong J-coupling"""
        # Base hydrolysis
        hydrolysis_rate = self.params.k_atp_hydrolysis * self.atp_field
        
        # Enhanced near active channels
        for i, (x, y) in enumerate(self.channel_indices):
            if i < len(self.channels.states) and self.channels.states[i] == 1:
                dist = np.sqrt((self.X - self.X[x, y])**2 + (self.Y - self.Y[x, y])**2)
                enhancement = np.exp(-dist**2 / (2 * (5*self.dx)**2))
                hydrolysis_rate += enhancement * 10 * self.params.k_atp_hydrolysis
        
        # Update fields
        atp_consumed = hydrolysis_rate * dt
        self.atp_field -= atp_consumed
        self.phosphate_field += atp_consumed  # 1:1 stoichiometry
        
        # CRITICAL: Update J-coupling based on ATP
        self.j_coupling_field = np.where(
            hydrolysis_rate > 1e-6,
            self.params.J_PP_atp,  # Strong coupling from ATP
            self.params.J_PP_isolated  # Weak otherwise
        )
        
        # Replenish from mitochondria
        for (x, y) in self.atp_sources:
            self.atp_field[x, y] = self.params.atp_synaptic
    
    def update_dopamine(self, dt: float, reward_signal: bool = False):
        """Dopamine release and reuptake"""
        if reward_signal:
            for (x, y) in self.dopamine_sites:
                self.dopamine_field[x, y] += self.params.dopamine_peak * dt
        
        # DAT reuptake
        excess = self.dopamine_field - self.params.dopamine_baseline
        uptake = self.params.k_dat_uptake * np.maximum(excess, 0) * dt
        self.dopamine_field -= uptake
        
        # Ensure baseline
        self.dopamine_field = np.maximum(self.dopamine_field, self.params.dopamine_baseline)
    
    def form_dimers_and_trimers(self, dt: float):
        """Form dimers or trimers based on supersaturation and dopamine"""
        ca = self.calcium_field
        po4 = self.phosphate_field
        da = self.dopamine_field
        pH = self.local_pH
        j_coupling = self.j_coupling_field
        
        # Calculate supersaturation (not raw powers!)
        # Using effective concentrations with activity coefficients
        ca_eff = ca * self.params.gamma_ca
        po4_eff = po4 * self.f_hpo4_local * self.params.gamma_po4
        
        # Critical concentrations
        ca_crit_dimer = (self.params.Ksp_dimer / (po4_eff**4 + 1e-50))**(1/6)
        ca_crit_trimer = (self.params.Ksp_trimer / (po4_eff**6 + 1e-50))**(1/9)
        
        # Supersaturation
        super_dimer = np.maximum(0, (ca_eff - ca_crit_dimer) / (ca_crit_dimer + 1e-9))
        super_trimer = np.maximum(0, (ca_eff - ca_crit_trimer) / (ca_crit_trimer + 1e-9))
        
        # pH effect
        pH_factor = 10 ** (7.0 - pH)  # Lower pH favors formation
        
        # Dopamine modulation (key finding!)
        dimer_enhancement = np.where(da > self.params.dopamine_threshold, 10.0, 1.0)
        trimer_suppression = np.where(da > self.params.dopamine_threshold, 0.1, 1.0)
        
        # J-coupling enhancement
        coupling_factor = j_coupling / self.params.J_PP_isolated
        
        # Formation rates based on supersaturation
        dimer_rate = (self.params.k_dimer_formation * super_dimer * 
                     pH_factor * coupling_factor * dimer_enhancement)
        
        trimer_rate = (self.params.k_trimer_formation * super_trimer * 
                      pH_factor * trimer_suppression)
        
        # Update fields
        self.dimer_field += dimer_rate * dt * self.active_mask
        self.trimer_field += trimer_rate * dt * self.active_mask
        
        # Apply dissolution
        self.dimer_field -= self.params.kr_dimer * self.dimer_field * dt
        self.trimer_field -= self.params.kr_trimer * self.trimer_field * dt
        
        # Prevent negative
        self.dimer_field = np.maximum(self.dimer_field, 0)
        self.trimer_field = np.maximum(self.trimer_field, 0)
        
        # Cap at reasonable concentrations
        self.dimer_field = np.minimum(self.dimer_field, 100e-9)
        self.trimer_field = np.minimum(self.trimer_field, 100e-9)
    
    def update_quantum_coherence(self, dt: float):
        """Calculate coherence with dopamine protection"""
        da = self.dopamine_field
        
        # Dimers - long coherence
        dimer_present = self.dimer_field > 1e-12
        
        if np.any(dimer_present):
            # Initialize new dimers
            new_dimers = dimer_present & (self.coherence_dimer == 0)
            self.coherence_dimer[new_dimers] = 1.0
            
            # Calculate T2 with dopamine protection
            T2_dimer = self.params.T2_dimer_base
            protection = np.where(
                da > self.params.dopamine_threshold,
                1 + np.log10((da / self.params.dopamine_threshold).clip(1, 100)),
                1.0
            )
            T2_protected = T2_dimer * protection
            
            # Decay
            decay_rate = 1.0 / np.maximum(T2_protected, 1.0)
            self.coherence_dimer *= np.exp(-dt * decay_rate)
            self.coherence_dimer[~dimer_present] = 0
        
        # Trimers - short coherence
        trimer_present = self.trimer_field > 1e-12
        
        if np.any(trimer_present):
            new_trimers = trimer_present & (self.coherence_trimer == 0)
            self.coherence_trimer[new_trimers] = 1.0
            
            # No dopamine protection for trimers
            self.coherence_trimer *= np.exp(-dt / self.params.T2_trimer_base)
            self.coherence_trimer[~trimer_present] = 0
    
    def calculate_learning_signal(self) -> float:
        """Learning requires AND logic: coherence + dopamine + activity"""
        learning_sites = (
            (self.coherence_dimer > self.params.coherence_threshold) &
            (self.dopamine_field > self.params.dopamine_threshold) &
            (self.calcium_field > 10e-6)
        )
        
        if np.any(learning_sites):
            signal = np.mean(
                self.coherence_dimer[learning_sites] * 
                np.log10((self.dopamine_field[learning_sites] / 
                         self.params.dopamine_threshold).clip(1, 100))
            )
            
            if signal > 0.1:
                self.learning_event_count += 1
        else:
            signal = 0.0
        
        return signal
    
    def update_template_fusion(self, dt: float):
        """Template-mediated dimer/trimer formation - MODIFIED for Model 5"""
        for ti, tj in self.template_indices:
            occupancy = self.template_pnc_bound[tj, ti] / self.params.n_binding_sites
            
            if occupancy >= 0.5:
                # Environmental factors
                ca_local = self.calcium_field[tj, ti]
                da_local = self.dopamine_field[tj, ti]
                pH_local = self.local_pH[tj, ti]
                activity_level = np.sum(self.channels.get_open_channels()) / len(self.channels.states)
                
                # Calculate enhancement
                cooperative = 1 + 2 * (occupancy - 0.5)
                ca_enhance = 1 + min(ca_local / 100e-6, 10)
                pH_enhance = 1 + 0.5 * (7.3 - pH_local)
                activity_enhance = 1 + activity_level
                
                # Dopamine determines dimer vs trimer preference
                if da_local > self.params.dopamine_threshold:
                    da_enhance = 2.0
                    prefer_dimer = True
                else:
                    da_enhance = 1.0
                    prefer_dimer = False
                
                total_enhancement = (cooperative * ca_enhance * 
                                   pH_enhance * activity_enhance * da_enhance)
                
                self.template_fusion_timer[tj, ti] += dt * total_enhancement
                
                if self.template_fusion_timer[tj, ti] > 1.0 / self.params.k_fusion_attempt:
                    self.template_fusion_timer[tj, ti] = 0
                    
                    fusion_prob = self.params.fusion_probability * min(total_enhancement, 10)
                    fusion_prob = min(fusion_prob, 0.5)
                    
                    if np.random.random() < fusion_prob:
                        # FUSION EVENT
                        n_formed = np.random.poisson(2)
                        
                        if prefer_dimer:
                            self.dimer_field[tj, ti] += n_formed * 1e-9
                        else:
                            self.trimer_field[tj, ti] += n_formed * 0.5e-9
                        
                        # Reset template
                        self.template_pnc_bound[tj, ti] *= 0.5
                        
                        # Consume PNC
                        if self.pnc_field[tj, ti] > 1e-12:
                            self.pnc_field[tj, ti] *= 0.8
                        
                        self.fusion_count += 1
    
    def calculate_laplacian_neumann(self, field: np.ndarray) -> np.ndarray:
        """Laplacian with no-flux boundaries from Model 4"""
        laplacian = np.zeros_like(field)
        
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # No-flux boundaries
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
        
        return laplacian
    
    def update_fields(self, dt: float, channel_open: np.ndarray, reward_signal: bool = False):
        """Main update orchestrating all dynamics"""
        # Activity level
        activity_level = np.sum(channel_open) / max(1, len(channel_open))
        
        # Core Model 4 updates
        self.update_local_pH(activity_level)
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        
        # Model 5: ATP creates J-coupling
        self.update_atp_dynamics(dt, activity_level)
        
        # Model 5: Dopamine if reward
        self.update_dopamine(dt, reward_signal)
        
        # PNC with J-coupling enhancement
        self.update_pnc_dynamics(dt)
        
        # Template fusion
        self.update_template_fusion(dt)
        
        # Model 5: Dimer/trimer formation
        self.form_dimers_and_trimers(dt)
        
        # Model 5: Quantum coherence
        self.update_quantum_coherence(dt)
    
    def run_simulation(self, duration: float = None, 
                      stim_protocol: str = 'single_spike',
                      reward_protocol: str = None) -> Model5Results:
        """Run complete simulation"""
        if duration is None:
            duration = self.params.duration
        
        # Create results container
        results = Model5Results(parameters=asdict(self.params))
        
        # Setup time
        n_steps = int(duration / self.params.dt)
        save_interval = int(self.params.save_interval / self.params.dt)
        n_saves = n_steps // save_interval + 1
        
        # Initialize arrays
        results.time = np.arange(0, duration + self.params.dt, self.params.save_interval)[:n_saves]
        
        # Allocate storage
        array_shape = (n_saves, *self.grid_shape)
        results.calcium_map = np.zeros(array_shape)
        results.phosphate_map = np.zeros(array_shape)
        results.pnc_map = np.zeros(array_shape)
        results.channel_states = np.zeros((n_saves, self.params.n_channels))
        
        # Model 5 arrays
        results.atp_map = np.zeros(array_shape)
        results.dopamine_map = np.zeros(array_shape)
        results.ph_map = np.zeros(array_shape)
        results.dimer_map = np.zeros(array_shape)
        results.trimer_map = np.zeros(array_shape)
        results.coherence_dimer_map = np.zeros(array_shape)
        results.coherence_trimer_map = np.zeros(array_shape)
        results.j_coupling_map = np.zeros(array_shape)
        results.learning_signal = np.zeros(n_saves)
        
        # Stimulus protocol
        depolarized = np.zeros(n_steps, dtype=bool)
        if stim_protocol == 'single_spike':
            depolarized[int(0.1/self.params.dt):int(0.15/self.params.dt)] = True
        elif stim_protocol == 'tetanus':
            for i in range(10):
                start = int((0.1 + i*0.05)/self.params.dt)
                depolarized[start:start+int(0.005/self.params.dt)] = True
        
        # Reward protocol
        reward_signal = np.zeros(n_steps, dtype=bool)
        if reward_protocol == 'delayed':
            reward_signal[int(0.3/self.params.dt):int(0.35/self.params.dt)] = True
        elif reward_protocol == 'coincident':
            reward_signal = depolarized.copy()
        
        # Reset counters
        self.fusion_count = 0
        self.learning_event_count = 0
        
        # Main loop
        save_idx = 0
        for step in range(n_steps):
            # Update channels
            self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
            
            # Update all fields
            self.update_fields(self.params.dt, channel_open, reward_signal[step])
            
            # Calculate learning
            learning = self.calculate_learning_signal()
            
            # Save periodically
            if step % save_interval == 0:
                results.calcium_map[save_idx] = self.calcium_field
                results.phosphate_map[save_idx] = self.phosphate_field
                results.pnc_map[save_idx] = self.pnc_field
                results.channel_states[save_idx] = self.channels.states
                
                results.atp_map[save_idx] = self.atp_field
                results.dopamine_map[save_idx] = self.dopamine_field
                results.ph_map[save_idx] = self.local_pH
                results.dimer_map[save_idx] = self.dimer_field
                results.trimer_map[save_idx] = self.trimer_field
                results.coherence_dimer_map[save_idx] = self.coherence_dimer
                results.coherence_trimer_map[save_idx] = self.coherence_trimer
                results.j_coupling_map[save_idx] = self.j_coupling_field
                results.learning_signal[save_idx] = learning
                
                save_idx += 1
                
                if step % 1000 == 0:
                    logger.info(f"Step {step}: Dimers={np.max(self.dimer_field)*1e9:.1f} nM, "
                               f"Trimers={np.max(self.trimer_field)*1e9:.1f} nM, "
                               f"Learning={learning:.3f}")
        
        # Calculate metrics
        results.peak_pnc = np.max(results.pnc_map) * 1e9
        results.mean_pnc = np.mean(results.pnc_map[results.pnc_map > 0]) * 1e9 if np.any(results.pnc_map > 0) else 0
        
        results.peak_dimer = np.max(results.dimer_map) * 1e9
        results.peak_trimer = np.max(results.trimer_map) * 1e9
        
        if results.peak_trimer > 0:
            results.dimer_trimer_ratio = results.peak_dimer / results.peak_trimer
        else:
            results.dimer_trimer_ratio = np.inf
        
        results.mean_coherence_dimer = np.mean(results.coherence_dimer_map[results.coherence_dimer_map > 0]) if np.any(results.coherence_dimer_map > 0) else 0
        results.mean_coherence_trimer = np.mean(results.coherence_trimer_map[results.coherence_trimer_map > 0]) if np.any(results.coherence_trimer_map > 0) else 0
        
        results.max_j_coupling = np.max(results.j_coupling_map)
        results.peak_dopamine = np.max(results.dopamine_map) * 1e9
        results.learning_events = self.learning_event_count
        results.templates_occupied = np.sum(self.template_pnc_bound > 0)
        results.fusion_events = self.fusion_count
        
        # Conservation check
        final_ca_total = np.sum(self.calcium_field[self.active_mask]) * self.dx * self.dx
        results.calcium_conservation = final_ca_total / self.total_ca_initial if self.total_ca_initial > 0 else 1.0
        
        logger.info(f"Simulation complete. Peak Dimers: {results.peak_dimer:.2f} nM, "
                   f"Peak Trimers: {results.peak_trimer:.2f} nM, "
                   f"D/T Ratio: {results.dimer_trimer_ratio:.1f}, "
                   f"Learning events: {results.learning_events}")
        
        return results

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_quantum_advantage(results: Model5Results) -> Dict:
    """Analyze quantum vs classical processing metrics"""
    analysis = {}
    
    # Check for quantum signatures
    if results.learning_signal is not None:
        # Quantum advantage = learning with coherence vs without
        quantum_learning = np.sum(results.learning_signal > 0.1)
        baseline_learning = 5  # Expected classical learning events
        
        analysis['quantum_advantage'] = quantum_learning / baseline_learning if baseline_learning > 0 else 0
        analysis['quantum_events'] = quantum_learning
        
        # Check for sub-Poisson statistics (would need spike data)
        # Fano factor < 1 indicates quantum noise suppression
        
        # Temperature independence (Q10 < 1.2 for quantum)
        # Would need temperature sweep data
    
    # Dimer dominance (indicates quantum pathway)
    if results.dimer_trimer_ratio > 10:
        analysis['quantum_pathway'] = 'dimer_dominant'
    elif results.dimer_trimer_ratio > 1:
        analysis['quantum_pathway'] = 'mixed'
    else:
        analysis['quantum_pathway'] = 'trimer_dominant'
    
    return analysis