# src/models/dynamic_nanoreactor_model.py
"""
Model 4: Dynamic Nanoreactor with Stochastic Channel Gating
Builds on Model 3 by adding temporal dynamics and stochasticity
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
# RESULTS CONTAINER
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
# SPATIOTEMPORAL DYNAMICS
# ============================================================================

@dataclass
class DynamicParameters:
    """Parameters for dynamic nanoreactor model"""
    
    # Spatial parameters
    grid_size: int = 100  # Grid points in each dimension
    active_zone_radius: float = 200e-9  # 200 nm
    cleft_width: float = 20e-9  # 20 nm
    
    # Channel parameters
    n_channels: int = 6
    channel_current: float = 0.3e-12  # 0.3 pA
    channel_open_rate: float = 100.0  # Hz
    channel_close_rate: float = 50.0  # Hz
    
    # Chemical parameters
    ca_baseline: float = 1e-7  # 100 nM
    po4_baseline: float = 1e-3  # 1 mM
    
    # Posner formation (from Model 3)
    kf_posner_base: float = 2e-3
    kr_posner: float = 0.5  # 2s lifetime
    
    # Enhancement factors (from Model 3)
    microdomain_factor: float = 776.9
    template_factor: float = 10.0
    electrostatic_factor: float = 3.0
    confinement_factor: float = 5.0
    
    # Diffusion coefficients
    D_calcium: float = 220e-12  # m²/s
    D_posner: float = 50e-12  # m²/s (larger molecule, slower)
    
    # Temporal parameters
    dt: float = 0.0001  # 100 μs
    save_interval: int = 10  # Save every 10 timesteps (1 ms)
    
    # Stochastic parameters
    noise_amplitude: float = 0.01  # Thermal noise

    # Add maximum calcium concentration (physiological limit)
    max_calcium: float = 500e-6  # 500 μM max (measured experimentally)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return asdict(self)

# ============================================================================
# MAIN MODEL
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
    
    def calculate_calcium_microdomains(self, channel_open: np.ndarray) -> np.ndarray:
        """Calculate calcium concentration field from open channels"""
        ca_field = np.ones(self.grid_shape) * self.params.ca_baseline
        
        for idx, (j, i) in enumerate(self.channel_indices):
            if channel_open[idx]:
                # Distance from this channel to all points
                dist = np.sqrt((self.X - self.X[j, i])**2 + 
                              (self.Y - self.Y[j, i])**2)
                
                # Avoid singularity
                dist = np.maximum(dist, 1e-9)
                
                # Steady-state concentration from point source
                i_channel = self.params.channel_current
                D = self.params.D_calcium
                F = 96485  # Faraday constant
                
                # Add contribution from this channel
                ca_contribution = i_channel / (4 * np.pi * D * dist * F * 2)
                ca_field += ca_contribution * self.active_mask
                ca_field = np.minimum(ca_field, self.params.max_calcium)

        return ca_field
    
    def calculate_posner_formation_rate(self, ca_field: np.ndarray) -> np.ndarray:
        """Calculate spatially-resolved Posner formation rate"""
        rate_field = np.zeros(self.grid_shape)
        
        # Base formation rate
        kf = self.params.kf_posner_base
        
        # Phosphate with electrostatic enhancement
        po4_enhanced = self.params.po4_baseline * self.params.electrostatic_factor
        
        # Calculate rate at each point
        for j, i in np.ndindex(self.grid_shape):
            if not self.active_mask[j, i]:
                continue
            
            # Local calcium (already includes microdomain enhancement)
            ca_local = ca_field[j, i]
            
            # Check for template enhancement
            template_enhancement = 1.0
            for tj, ti in self.template_indices:
                dist = np.sqrt((self.X[j, i] - self.X[tj, ti])**2 + 
                              (self.Y[j, i] - self.Y[tj, ti])**2)
                if dist < 10e-9:  # Within 10 nm
                    template_enhancement = self.params.template_factor
                    break
            
            # Total rate with all enhancements
            rate = (kf * template_enhancement * self.params.confinement_factor *
                   ca_local * po4_enhanced)
            
            rate_field[j, i] = rate
        
        return rate_field
    
    def update_fields(self, dt: float, channel_open: np.ndarray):
        """Update calcium and Posner fields"""
        
        # Update calcium field based on open channels
        self.calcium_field = self.calculate_calcium_microdomains(channel_open)
        
        # Calculate Posner formation rate
        formation_rate = self.calculate_posner_formation_rate(self.calcium_field)
        
        # Update Posner concentration (reaction-diffusion)
        # dP/dt = D∇²P + R - λP
        
        # Diffusion term (simple finite difference)
        laplacian = np.zeros_like(self.posner_field)
        laplacian[1:-1, 1:-1] = (
            self.posner_field[2:, 1:-1] + self.posner_field[:-2, 1:-1] +
            self.posner_field[1:-1, 2:] + self.posner_field[1:-1, :-2] -
            4 * self.posner_field[1:-1, 1:-1]
        ) / self.dx**2
        
        # Apply reaction-diffusion equation
        dPosner_dt = (
            self.params.D_posner * laplacian +  # Diffusion
            formation_rate -  # Formation
            self.params.kr_posner * self.posner_field  # Degradation
        )
        
        # Update with Euler method
        self.posner_field += dPosner_dt * dt
        
        # Add thermal noise
        noise = np.random.normal(0, self.params.noise_amplitude * np.sqrt(dt), 
                                self.grid_shape)
        self.posner_field += noise * self.active_mask
        
        # Ensure non-negative
        self.posner_field = np.maximum(self.posner_field, 0)
    
    def calculate_metrics(self, posner_history: List[np.ndarray], 
                         channel_history: List[np.ndarray]) -> Dict:
        """Calculate summary metrics from simulation"""
        
        # Convert to arrays
        posner_arr = np.array(posner_history)
        channel_arr = np.array(channel_history)
        
        # Peak and mean Posner
        peak_posner = np.max(posner_arr) * 1e9  # Convert to nM
        mean_posner = np.mean(posner_arr[posner_arr > 0]) * 1e9
        
        # Spatial heterogeneity (coefficient of variation)
        if mean_posner > 0:
            spatial_heterogeneity = np.std(posner_arr[-1]) / np.mean(posner_arr[-1])
        else:
            spatial_heterogeneity = 0
        
        # Hotspot lifetime (time above 50% of peak)
        threshold = 0.5 * np.max(posner_arr)
        above_threshold = np.any(posner_arr > threshold, axis=(1, 2))
        if np.any(above_threshold):
            hotspot_lifetime = np.sum(above_threshold) * self.params.dt * self.params.save_interval
        else:
            hotspot_lifetime = 0
        
        # Channel statistics
        channel_open_fraction = np.mean(channel_arr == 1)
        
        # Burst analysis
        open_any = np.any(channel_arr == 1, axis=1)
        bursts = self.find_bursts(open_any)
        if bursts:
            mean_burst_duration = np.mean([b[1] - b[0] for b in bursts]) * self.params.dt * self.params.save_interval
        else:
            mean_burst_duration = 0
        
        # Quantum metrics (simplified - would need full quantum model)
        # Estimate based on concentration
        if peak_posner > 0:
            coherence_time = 1.0 / (1 + peak_posner / 100)  # Simplified model
            entanglement_range = 50e-9 * np.exp(-peak_posner / 50)  # nm
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
        """
        Run full simulation with stochastic channel dynamics
        
        Args:
            duration: Simulation duration in seconds
            stimulus_times: Times to trigger depolarization (if None, continuous)
        
        Returns:
            SimulationResults object with all data and metrics
        """
        
        # Setup time
        n_steps = int(duration / self.params.dt)
        time = np.arange(n_steps) * self.params.dt
        
        # Storage (save subset to manage memory)
        save_times = list(range(0, n_steps, self.params.save_interval))
        n_saves = len(save_times)
        
        calcium_history = np.zeros((n_saves, *self.grid_shape))
        posner_history = np.zeros((n_saves, *self.grid_shape))
        channel_history = np.zeros((n_saves, self.params.n_channels), dtype=int)
        
        # Determine when to depolarize
        if stimulus_times is None:
            # Continuous stimulation
            depolarized = np.ones(n_steps, dtype=bool)
        else:
            # Pulsed stimulation
            depolarized = np.zeros(n_steps, dtype=bool)
            for stim_time in stimulus_times:
                stim_idx = int(stim_time / self.params.dt)
                # 5 ms depolarization
                depolarized[stim_idx:stim_idx + int(0.005 / self.params.dt)] = True
        
        # Main simulation loop
        save_idx = 0
        for step in range(n_steps):
            
            # Update channel states
            channel_states = self.channels.update(self.params.dt, depolarized[step])
            channel_open = self.channels.get_open_channels()
            
            # Update fields
            self.update_fields(self.params.dt, channel_open)
            
            # Save if needed
            if step in save_times:
                calcium_history[save_idx] = self.calcium_field
                posner_history[save_idx] = self.posner_field
                channel_history[save_idx] = channel_states
                save_idx += 1
            
            # Progress
            if step % 1000 == 0:
                logger.debug(f"Step {step}/{n_steps}, "
                           f"Max Posner: {np.max(self.posner_field)*1e9:.1f} nM")
        
        # Calculate metrics
        metrics = self.calculate_metrics(posner_history, channel_history)
        
        # Create results object
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

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_spatial_pattern(results: SimulationResults) -> Dict:
    """Analyze spatial patterns in Posner distribution"""
    
    # Get final Posner field
    final_posner = results.posner_map[-1]
    
    # Find hotspots (local maxima)
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
    
    # Total Posner over time
    total_posner = np.sum(results.posner_map, axis=(1, 2))
    
    # Find rise time (10% to 90% of max)
    max_val = np.max(total_posner)
    t10 = np.where(total_posner > 0.1 * max_val)[0]
    t90 = np.where(total_posner > 0.9 * max_val)[0]
    
    if len(t10) > 0 and len(t90) > 0:
        rise_time = (t90[0] - t10[0]) * results.time[1]
    else:
        rise_time = 0
    
    # Find decay time (90% to 10% after peak)
    peak_idx = np.argmax(total_posner)
    decay_portion = total_posner[peak_idx:]
    t90_decay = np.where(decay_portion < 0.9 * max_val)[0]
    t10_decay = np.where(decay_portion < 0.1 * max_val)[0]
    
    if len(t90_decay) > 0 and len(t10_decay) > 0:
        decay_time = (t10_decay[0] - t90_decay[0]) * results.time[1]
    else:
        decay_time = 0
    
    return {
        'rise_time': rise_time,
        'decay_time': decay_time,
        'peak_time': results.time[peak_idx],
        'total_posner_trajectory': total_posner
    }

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    params = DynamicParameters(
        n_channels=6,
        grid_size=50,  # Smaller for testing
        dt=0.0001,
        save_interval=10
    )
    
    model = DynamicNanoreactor(params)
    
    # Run short simulation
    print("Running test simulation...")
    results = model.simulate(duration=0.1, stimulus_times=[0.01, 0.05])
    
    # Save results
    test_path = Path("test_results/model4_test")
    test_path.parent.mkdir(exist_ok=True)
    results.save(test_path)
    
    print(f"Peak Posner: {results.peak_posner:.2f} nM")
    print(f"Coherence time: {results.coherence_time:.3f} s")
    print(f"Spatial heterogeneity: {results.spatial_heterogeneity:.3f}")
    print(f"Channel open fraction: {results.channel_open_fraction:.3f}")
    
    # Test loading
    loaded = SimulationResults.load(test_path)
    print(f"Successfully loaded results with {len(loaded.time)} timepoints")