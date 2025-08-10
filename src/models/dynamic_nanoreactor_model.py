# src/models/dynamic_nanoreactor_model.py
# Complete implementation with all fixes applied

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
    model_version: str = "4.1"  # Updated version
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict = field(default_factory=dict)
    
    # Time series data
    time: np.ndarray = None
    calcium_map: np.ndarray = None
    posner_map: np.ndarray = None
    pnc_map: np.ndarray = None
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

@dataclass
class DynamicParameters:
    """
    Complete parameters for PNC-based Posner formation with fixes
    """
    # ============= SPATIAL PARAMETERS =============
    grid_size: int = 50
    active_zone_radius: float = 200e-9
    cleft_width: float = 20e-9
    
    # ============= CHANNEL PARAMETERS =============
    n_channels: int = 6
    channel_current: float = 1.0e-12  # # 1.0 pA (increased for stronger microdomains)
    channel_open_rate: float = 100.0
    channel_close_rate: float = 50.0
    
    # ============= BASELINE CONCENTRATIONS =============
    ca_baseline: float = 100e-9  # 100 nM resting
    po4_baseline: float = 1e-3  # 1 mM
    atp_concentration: float = 3e-3  # 3 mM - NEW
    
    # ============= DIFFUSION COEFFICIENTS =============
    D_calcium: float = 220e-12
    D_phosphate: float = 280e-12
    D_pnc: float = 100e-12
    D_posner: float = 50e-12
    
    # ============= PNC DYNAMICS (UPDATED) =============
    # Complex formation
    k_complex_formation: float = 1e8  # M⁻¹s⁻¹
    k_complex_dissociation: float = 100.0  # s⁻¹
    
    # PNC formation/dissolution
    k_pnc_formation: float = 200.0  # s⁻¹ - More reasonable rate
    k_pnc_dissolution: float = 10.0  # s⁻¹
    pnc_baseline: float = 1e-10  # 0.1 nM baseline - NEW
    pnc_size: int = 30
    pnc_max_concentration: float = 1e-6
    
    # ============= TEMPLATE PARAMETERS (UPDATED) =============
    templates_per_synapse: int = 500
    n_binding_sites: int = 3
    k_pnc_binding: float = 1e9  # M⁻¹s⁻¹ - INCREASED
    k_pnc_unbinding: float = 0.1  # s⁻¹ - DECREASED
    template_accumulation_range: int = 2  # Grid cells - NEW
    template_accumulation_rate: float = 0.3  # fraction/dt - NEW
    
    # Fusion kinetics
    k_fusion_attempt: float = 10.0  # s⁻¹
    fusion_probability: float = 0.01  # Base probability
    pnc_per_posner: int = 2  # PNCs consumed per Posner
    
    # ============= ACTIVITY-DEPENDENT (NEW) =============
    k_atp_hydrolysis: float = 10.0  # s⁻¹ during activity
    ph_activity_shift: float = -0.3  # pH units during activity
    
    # ============= BIOPHYSICAL FACTORS =============
    f_hpo4_ph73: float = 0.49  # HPO₄²⁻ fraction at pH 7.3
    gamma_ca: float = 0.4  # Activity coefficient for Ca²⁺ - NEW
    gamma_po4: float = 0.2  # Activity coefficient for HPO₄²⁻ - NEW
    
    # Enhancement factors
    template_factor: float = 10.0
    confinement_factor: float = 5.0
    electrostatic_factor: float = 3.0
    membrane_concentration_factor: float = 5.0  # NEW - near membrane
    
    # ============= SIMULATION PARAMETERS =============
    dt: float = 0.001  # 1 ms
    duration: float = 1.0  # 1 second
    save_interval: float = 0.01  # Save every 10 ms
    
    # ============= DISSOLUTION RATES =============
    kr_posner: float = 0.5  # s⁻¹

class ChannelDynamics:
    """Handles stochastic calcium channel gating"""
    
    def __init__(self, n_channels: int, params: DynamicParameters):
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

class DynamicNanoreactor:
    """
    Complete PNC-based dynamic nanoreactor with all fixes
    """
    
    def __init__(self, params: DynamicParameters):
        self.params = params
        
        # Setup spatial grid
        self.grid_shape = (params.grid_size, params.grid_size)
        self.dx = 2 * params.active_zone_radius / params.grid_size
        
        # Create coordinate system
        x = np.linspace(-params.active_zone_radius, params.active_zone_radius, params.grid_size)
        y = np.linspace(-params.active_zone_radius, params.active_zone_radius, params.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Active zone mask
        self.active_mask = (self.X**2 + self.Y**2) <= params.active_zone_radius**2
        
        # Membrane proximity mask (NEW)
        r = np.sqrt(self.X**2 + self.Y**2)
        edge_distance = params.active_zone_radius - r
        self.membrane_mask = (edge_distance < 2e-9) & (edge_distance > 0)  # Within 2 nm of edge
        
        # Initialize fields
        self.calcium_field = np.ones(self.grid_shape) * params.ca_baseline * self.active_mask
        self.phosphate_field = np.ones(self.grid_shape) * params.po4_baseline * self.active_mask
        self.complex_field = np.zeros(self.grid_shape)
        
        # Initialize PNC with baseline (FIX #1)
        self.pnc_field = np.ones(self.grid_shape) * params.pnc_baseline * self.active_mask
        
        self.posner_field = np.zeros(self.grid_shape)
        
        # Activity-dependent fields (NEW)
        self.local_pH = 7.3 * np.ones(self.grid_shape)
        self.f_hpo4_local = params.f_hpo4_ph73 * np.ones(self.grid_shape)
        
        # Setup channels and templates
        self._setup_channel_positions()
        self._setup_template_positions()
        
        # Template state tracking
        self.template_pnc_bound = np.zeros(self.grid_shape)
        self.template_fusion_timer = np.zeros(self.grid_shape)
        
        # Channel dynamics
        self.channels = ChannelDynamics(params.n_channels, params)
        
        # Track initial mass
        self.total_ca_initial = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        
        # Metrics
        self.fusion_count = 0
        
        logger.info(f"DynamicNanoreactor initialized: {params.n_channels} channels, "
                   f"{len(self.template_indices)} templates")
    
    def _setup_channel_positions(self):
        """Position channels in hexagonal array"""
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
        """Position templates between channels"""
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
    
    def update_local_pH(self, activity_level: float):
        """Update local pH based on activity (FIX #3)"""
        # Activity causes acidification
        pH_shift = self.params.ph_activity_shift * activity_level
        self.local_pH = 7.3 + pH_shift * self.active_mask
        
        # Update HPO4 fraction
        # Henderson-Hasselbalch with pKa2 = 7.2
        self.f_hpo4_local = 1 / (1 + 10**(7.2 - self.local_pH))

    def update_phosphate_from_ATP(self, dt: float, activity_level: float):
        """ATP hydrolysis releases phosphate during activity (FIX #5)"""
        if activity_level > 0:
            # ATP hydrolysis rate proportional to activity
            hydrolysis_rate = self.params.k_atp_hydrolysis * activity_level
        
            # Phosphate production (assuming 10% of ATP gets hydrolyzed)
            phosphate_production = self.params.atp_concentration * hydrolysis_rate * dt * 0.1
        
            # Add preferentially near active channels
            for idx, (ci, cj) in enumerate(self.channel_indices):
                if idx < len(self.channels.states) and self.channels.states[idx] == 1:
                    # Local enhancement around active channel
                    r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                    enhancement = np.exp(-r / 10e-9)  # 10 nm decay length
                    self.phosphate_field += phosphate_production * enhancement
    
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """Calculate calcium concentration with proper amplitudes (FIX #6)"""
        calcium = self.params.ca_baseline * np.ones(self.grid_shape)
    
        for idx, (ci, cj) in enumerate(self.channel_indices):
            if idx < len(channel_open) and channel_open[idx]:
                # Distance from channel
                r = np.sqrt((self.X - self.X[ci, cj])**2 + (self.Y - self.Y[ci, cj])**2)
                r = np.maximum(r, 1e-9)  # Avoid singularity (1 nm minimum)
            
                # Calcium flux from single channel
                # 0.3 pA current = 0.3e-12 A / (2 * 1.602e-19 C/Ca) = 9.36e5 Ca ions/sec
                flux_ions_per_sec = self.params.channel_current / (2 * 1.602e-19)
            
                # Convert to molar flux (moles/sec)
                flux_molar = flux_ions_per_sec / 6.022e23  # moles/sec
            
                # Steady-state concentration from point source
                # C(r) = flux / (4 * pi * D * r)
                # This gives concentration in M
                single_ca = flux_molar / (4 * np.pi * self.params.D_calcium * r)
            
                # Add to baseline (no saturation yet, to see true values)
                calcium += single_ca
    
        # Apply membrane enhancement (FIX #3B)
        calcium[self.membrane_mask] *= self.params.membrane_concentration_factor
    
        # Apply buffering (reduced to allow higher peaks)
        kappa = 5  # Reduced buffering power for higher peaks
        calcium_buffered = calcium / (1 + kappa)
    
        # Apply saturation only at very high levels (10 mM)
        calcium_final = calcium_buffered / (1 + calcium_buffered/10e-3)
    
        return calcium_final * self.active_mask
    
    def calculate_complex_equilibrium(self):
        """Calculate CaHPO4 complex with activity coefficients (FIX #2)"""
        # Get effective concentrations with activity coefficients
        ca_eff = self.calcium_field * self.params.gamma_ca
        po4_eff = self.phosphate_field * self.f_hpo4_local * self.params.gamma_po4
    
        # Prevent overflow by capping concentrations
        ca_eff = np.clip(ca_eff, 0, 1e-3)  # Max 1 mM
        po4_eff = np.clip(po4_eff, 0, 1e-3)  # Max 1 mM
    
        # Equilibrium constant
        K_eq = self.params.k_complex_formation / self.params.k_complex_dissociation
    
        # Mass action with safety checks
        denominator = 1 + K_eq * (ca_eff + po4_eff)
        denominator = np.maximum(denominator, 1.0)  # Prevent divide by zero
    
        self.complex_field = K_eq * ca_eff * po4_eff / denominator
    
        # Templates enhance complex formation (but with limits)
        for ti, tj in self.template_indices:
            self.complex_field[tj, ti] = min(self.complex_field[tj, ti] * self.params.template_factor, 1e-6)
    
        self.complex_field *= self.active_mask
    
    def update_pnc_dynamics(self, dt: float):
        """Update PNC field with accumulation at templates (FIX #4)"""
        # Calculate complex concentration first
        self.calculate_complex_equilibrium()
    
        # Formation from complexes (with saturation)
        formation = self.params.k_pnc_formation * self.complex_field
    
        # Apply saturation to prevent runaway growth
        saturation = 1 - self.pnc_field / self.params.pnc_max_concentration
        saturation = np.clip(saturation, 0, 1)
        formation *= saturation
    
        # Dissolution
        dissolution = self.params.k_pnc_dissolution * self.pnc_field
    
        # Template accumulation (NEW - FIX #4)
        for ti, tj in self.template_indices:
            # Only accumulate if not saturated
            if self.pnc_field[tj, ti] < self.params.pnc_max_concentration:
                # Accumulate PNCs from surrounding area
                r_max = self.params.template_accumulation_range
                for di in range(-r_max, r_max+1):
                    for dj in range(-r_max, r_max+1):
                        ni, nj = ti + di, tj + dj
                        if (0 <= ni < self.params.grid_size and 
                            0 <= nj < self.params.grid_size and
                            (di != 0 or dj != 0) and
                            self.active_mask[nj, ni]):  # Check active mask
                        
                            # Drift toward template
                            distance = np.sqrt(di**2 + dj**2)
                            drift_rate = self.params.template_accumulation_rate / (1 + distance)
                            transfer = self.pnc_field[nj, ni] * drift_rate * dt
                        
                            # Prevent negative values
                            transfer = min(transfer, self.pnc_field[nj, ni])
                        
                            self.pnc_field[nj, ni] -= transfer
                            self.pnc_field[tj, ti] += transfer
        
            # FIXED BINDING: Simplified and more aggressive
            if self.pnc_field[tj, ti] > 1e-9:  # If PNCs present (1 nM threshold)
                # Available binding sites
                available = self.params.n_binding_sites - self.template_pnc_bound[tj, ti]
            
                if available > 0:
                    # Simple first-order binding
                    # Binding rate proportional to PNC concentration and available sites
                    binding_rate = 10.0  # s⁻¹ (aggressive binding)
                    binding_amount = min(binding_rate * dt * available, available)
                
                    # Only bind if enough PNCs available
                    if self.pnc_field[tj, ti] > 1e-8:  # Need at least 10 nM to bind
                        self.template_pnc_bound[tj, ti] += binding_amount
                    
                        # Don't remove PNCs from field (they can bind multiple templates)
                        # This represents PNCs associating with template, not being consumed
    
        # Update PNC field with bounds
        dpnc_dt = formation - dissolution
        self.pnc_field += dpnc_dt * dt * self.active_mask
    
        # Enforce bounds
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
    
        # Add small diffusion
        laplacian = self.calculate_laplacian_neumann(self.pnc_field)
        self.pnc_field += self.params.D_pnc * laplacian * dt
    
        # Final bounds check
        self.pnc_field = np.clip(self.pnc_field, 0, self.params.pnc_max_concentration)
    
    def update_posner_formation(self, dt: float):
        """Posner formation with realistic physics and concentrations"""
        for ti, tj in self.template_indices:
            # Check occupancy
            occupancy = self.template_pnc_bound[tj, ti] / self.params.n_binding_sites
        
            if occupancy >= 0.5:  # Threshold for attempting fusion
                # Environmental factors affect fusion probability
            
                # 1. Cooperative enhancement (multiple PNCs help)
                cooperative_factor = 1 + 2 * (occupancy - 0.5)
            
                # 2. Local calcium enhancement (Ca stabilizes transition state)
                ca_local = self.calcium_field[tj, ti]
                ca_enhancement = 1 + (ca_local / 100e-6)  # Enhancement above 100 µM
            
                # 3. pH effect (lower pH favors fusion)
                pH_local = self.local_pH[tj, ti]
                pH_factor = 1 + 0.5 * (7.3 - pH_local)  # Better at lower pH
            
                # 4. Template conformation (activity-dependent)
                activity_level = np.sum(self.channels.get_open_channels()) / len(self.channels.states)
                template_activation = 1 + activity_level
            
                # Combined enhancement
                total_enhancement = cooperative_factor * ca_enhancement * pH_factor * template_activation
            
                # Increment fusion timer with enhancements
                self.template_fusion_timer[tj, ti] += dt * total_enhancement
            
                # Attempt fusion when timer exceeds threshold
                if self.template_fusion_timer[tj, ti] > 1.0 / self.params.k_fusion_attempt:
                    self.template_fusion_timer[tj, ti] = 0
                
                    # Stochastic fusion with environmental modulation
                    fusion_prob = self.params.fusion_probability * total_enhancement
                    fusion_prob = min(fusion_prob, 0.5)  # Cap at 50%
                
                    if np.random.random() < fusion_prob:
                        # FUSION EVENT!
                    
                        # Realistic concentration increase
                        # Each fusion creates a "burst" of Posner molecules
                        # that represents multiple PNCs combining
                        n_posner_formed = np.random.poisson(5)  # Average 5 Posner per event
                    
                        # Concentration in nM (direct, avoiding numerical issues)
                        conc_increase_nM = n_posner_formed * 2.0  # Each Posner adds ~2 nM locally
                    
                        # Add to formation site (hotspot)
                        self.posner_field[tj, ti] += conc_increase_nM * 1e-9
                    
                        # Spatial spread (representing immediate local diffusion)
                        spread_radius = 2  # Spread over 2 grid points
                        for di in range(-spread_radius, spread_radius+1):
                            for dj in range(-spread_radius, spread_radius+1):
                                if di == 0 and dj == 0:
                                    continue  # Skip center (already added)
                                ni, nj = ti + di, tj + dj
                                if (0 <= ni < self.params.grid_size and 
                                    0 <= nj < self.params.grid_size and
                                    self.active_mask[nj, ni]):
                                    # Gaussian-like spread
                                    distance = np.sqrt(di**2 + dj**2)
                                    spread_factor = np.exp(-distance**2 / 2)
                                    self.posner_field[nj, ni] += conc_increase_nM * 1e-9 * spread_factor * 0.2
                    
                        # Template dynamics (activity-dependent reset)
                        if activity_level > 0.5:
                            # High activity: partial reset (fast recycling)
                            self.template_pnc_bound[tj, ti] *= 0.3
                        else:
                            # Low activity: full reset (slow recycling)
                            self.template_pnc_bound[tj, ti] = 0
                    
                        # PNC consumption (stoichiometric)
                        pncs_consumed = n_posner_formed * 2  # 2 PNCs per Posner
                        # Remove PNCs proportionally
                        consumption_factor = max(0, 1 - pncs_consumed * 1e-10 / self.pnc_field[tj, ti])
                        self.pnc_field[tj, ti] *= consumption_factor
                    
                        self.fusion_count += 1
    
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
        """Main update routine with all fixes integrated"""
        # Calculate activity level
        activity_level = np.sum(channel_open) / max(1, len(channel_open))
        
        # Update pH based on activity (FIX #3)
        self.update_local_pH(activity_level)
        
        # Update calcium from channels (FIX #6)
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        
        # Update phosphate from ATP (FIX #5)
        self.update_phosphate_from_ATP(dt, activity_level)
        
        # Complex equilibrium (FIX #2)
        self.calculate_complex_equilibrium()
        
        # PNC dynamics with accumulation (FIX #4)
        self.update_pnc_dynamics(dt)
        
        # Posner formation with cooperation
        self.update_posner_formation(dt)
        
        # Posner dissolution
        dissolution = self.params.kr_posner * self.posner_field
        self.posner_field -= dissolution * dt
        self.posner_field = np.maximum(self.posner_field, 0)
    
    def run_simulation(self, duration: float = None, 
                      stim_protocol: str = 'single_spike') -> SimulationResults:
        """Run complete simulation with stimulus protocol"""
        if duration is None:
            duration = self.params.duration
        
        # Create results container
        results = SimulationResults(parameters=asdict(self.params))
        
        # Setup time
        n_steps = int(duration / self.params.dt)
        save_interval = int(self.params.save_interval / self.params.dt)
        n_saves = n_steps // save_interval + 1
        
        results.time = np.arange(0, duration + self.params.dt, self.params.save_interval)[:n_saves]
        results.posner_map = np.zeros((n_saves, *self.grid_shape))
        results.pnc_map = np.zeros((n_saves, *self.grid_shape))
        results.calcium_map = np.zeros((n_saves, *self.grid_shape))
        results.channel_states = np.zeros((n_saves, self.params.n_channels))
        
        # Stimulus protocol
        depolarized = np.zeros(n_steps, dtype=bool)
        if stim_protocol == 'single_spike':
            depolarized[int(0.1/self.params.dt):int(0.15/self.params.dt)] = True
        elif stim_protocol == 'tetanus':
            for i in range(10):
                start = int((0.1 + i*0.05)/self.params.dt)
                depolarized[start:start+int(0.005/self.params.dt)] = True
        
        # Reset fusion counter
        self.fusion_count = 0
        
        # Main simulation loop
        save_idx = 0
        for step in range(n_steps):
            # Update channels
            self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
            
            # Update fields
            self.update_fields(self.params.dt, channel_open)
            
            # Save periodically
            if step % save_interval == 0:
                results.posner_map[save_idx] = self.posner_field
                results.pnc_map[save_idx] = self.pnc_field
                results.calcium_map[save_idx] = self.calcium_field
                results.channel_states[save_idx] = self.channels.states
                save_idx += 1
                
                if step % 1000 == 0:
                    logger.info(f"Step {step}: Posner={np.max(self.posner_field)*1e9:.1f} nM, "
                               f"PNC={np.max(self.pnc_field)*1e9:.1f} nM")
        
        # Calculate final metrics
        results.peak_posner = np.max(results.posner_map) * 1e9
        results.mean_posner = np.mean(results.posner_map[results.posner_map > 0]) * 1e9 if np.any(results.posner_map > 0) else 0
        results.peak_pnc = np.max(results.pnc_map) * 1e9
        results.mean_pnc = np.mean(results.pnc_map[results.pnc_map > 0]) * 1e9 if np.any(results.pnc_map > 0) else 0
        results.templates_occupied = np.sum(self.template_pnc_bound > 0)
        results.fusion_events = self.fusion_count
        
        # Check mass balance
        final_ca_total = np.sum(self.calcium_field * self.active_mask) * self.dx * self.dx
        results.calcium_conservation = final_ca_total / self.total_ca_initial if self.total_ca_initial > 0 else 1.0
        
        logger.info(f"Simulation complete. Peak Posner: {results.peak_posner:.2f} nM, "
                   f"Fusion events: {results.fusion_events}")
        
        return results

# Analysis functions
def analyze_spatial_pattern(results: SimulationResults) -> Dict:
    """Analyze spatial distribution of Posner formation"""
    analysis = {}
    
    # Find hotspots
    final_posner = results.posner_map[-1] if results.posner_map is not None else np.zeros((50, 50))
    
    if np.any(final_posner > 0):
        # Identify peaks
        from scipy.ndimage import maximum_filter
        local_maxima = (final_posner == maximum_filter(final_posner, size=3))
        
        analysis['n_hotspots'] = np.sum(local_maxima)
        analysis['hotspot_positions'] = np.argwhere(local_maxima)
        analysis['hotspot_concentrations'] = final_posner[local_maxima] * 1e9  # nM
        
        # Spatial heterogeneity (coefficient of variation)
        if np.mean(final_posner) > 0:
            analysis['spatial_cv'] = np.std(final_posner) / np.mean(final_posner)
        else:
            analysis['spatial_cv'] = 0
    else:
        analysis['n_hotspots'] = 0
        analysis['hotspot_positions'] = []
        analysis['hotspot_concentrations'] = []
        analysis['spatial_cv'] = 0
    
    return analysis

def analyze_temporal_dynamics(results: SimulationResults) -> Dict:
    """Analyze temporal evolution of Posner and PNC"""
    analysis = {}
    
    if results.time is not None and results.posner_map is not None:
        # Peak concentrations over time
        posner_peaks = np.max(results.posner_map.reshape(len(results.time), -1), axis=1) * 1e9
        pnc_peaks = np.max(results.pnc_map.reshape(len(results.time), -1), axis=1) * 1e9
        
        # Formation kinetics
        analysis['posner_rise_time'] = results.time[np.argmax(posner_peaks > 1.0)] if np.any(posner_peaks > 1.0) else np.inf
        analysis['pnc_rise_time'] = results.time[np.argmax(pnc_peaks > 1.0)] if np.any(pnc_peaks > 1.0) else np.inf
        
        # Steady-state levels
        analysis['posner_steady_state'] = np.mean(posner_peaks[-10:]) if len(posner_peaks) >= 10 else posner_peaks[-1]
        analysis['pnc_steady_state'] = np.mean(pnc_peaks[-10:]) if len(pnc_peaks) >= 10 else pnc_peaks[-1]
    else:
        analysis['posner_rise_time'] = np.inf
        analysis['pnc_rise_time'] = np.inf
        analysis['posner_steady_state'] = 0
        analysis['pnc_steady_state'] = 0
    
    return analysis