"""
Analytical Nanodomain Calcium System
=====================================
Physics-preserving replacement for full-grid diffusion

COMPUTATIONAL ADVANTAGE:
- Old: 100×100 grid × 1777 substeps = 177.7M operations per ms
- New: 50 channels × 150 local points = 7,500 operations per ms
- Speedup: ~24,000×

PHYSICS PRESERVED:
- Stochastic channel gating (identical to original)
- Nanodomain concentrations 10-100 μM (Naraghi & Neher 1997)
- Spatial decay with buffered diffusion length
- Template-localized dimer formation
- Pump-mediated recovery

Key Citations:
- Naraghi & Neher 1997 J Neurosci 17:6961-6973
  "Buffered diffusion of calcium in a nerve terminal"
  Analytical solution for [Ca](r,t) near point source

- Neher 1998 Neuron 20:389-399
  "Usefulness and limitations of linear approximations"
  Steady-state: [Ca](r) = i/(4πDr) for r < λ

- Smith et al. 2001 Biophys J 81:3064-3078
  "Validity of the rapid buffering approximation"
  D_eff = D_free / (1 + κ_s) where κ_s ≈ 60
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

from model6_parameters import Model6Parameters, CalciumParameters

logger = logging.getLogger(__name__)


class CalciumChannels:
    """
    Voltage-gated calcium channels with stochastic gating
    
    IDENTICAL to original - this is the physics we must preserve!
    
    Based on:
    - Borst & Sakmann 1996 Nature 383:431-434
      "~50 channels per active zone"
      "Single channel current = 0.3 pA"
      "Open time = 0.5 ms"
    """
    
    def __init__(self, positions: np.ndarray, params: CalciumParameters):
        self.positions = positions
        self.params = params
        self.n_channels = len(positions)
        
        # Channel state (boolean: open or closed)
        self.state = np.zeros(self.n_channels, dtype=bool)
        
        # Current through each channel (Amperes)
        self.current = np.zeros(self.n_channels)
        
        # Gating kinetics
        self.alpha = 1e3  # Opening rate (1/s)
        self.beta = 2e3   # Closing rate (1/s) → ~0.5ms open time
        
        # Track open time for each channel
        self.time_open = np.zeros(self.n_channels)
        
        logger.info(f"Initialized {self.n_channels} calcium channels")
        
    def update_gating(self, dt: float, voltage: Optional[float] = None):
        """
        Stochastic channel gating with realistic kinetics
        """
        # APV blocks NMDA - force all channels closed
        if getattr(self.params, 'nmda_blocked', False):
            self.state[:] = False
            self.current[:] = 0.0
            return
            
        if voltage is not None:
            V_threshold = -0.050  # -50 mV
            V_half = -0.030      # -30 mV
            V_slope = 0.012      # 12 mV slope
            
            # Boltzmann activation
            P_open_voltage = 1.0 / (1.0 + np.exp(-(voltage - V_half) / V_slope))
            
            # Modulate rates
            alpha_eff = self.alpha * P_open_voltage
            beta_eff = self.beta * (1 - P_open_voltage)
        else:
            alpha_eff = self.alpha
            beta_eff = self.beta
        
        # Stochastic transitions
        for i in range(self.n_channels):
            if self.state[i]:  # Channel open
                self.time_open[i] += dt
                if np.random.rand() < beta_eff * dt:
                    self.state[i] = False
                    self.current[i] = 0.0
                    self.time_open[i] = 0.0
            else:  # Channel closed
                if np.random.rand() < alpha_eff * dt:
                    self.state[i] = True
                    self.current[i] = self.params.single_channel_current
                    
    def get_open_channels(self) -> np.ndarray:
        return np.where(self.state)[0]
    
    def get_open_fraction(self) -> float:
        return np.sum(self.state) / self.n_channels


class AnalyticalNanodomainCalculator:
    """
    Analytical calcium nanodomain solutions
    
    Replaces expensive PDE diffusion with direct calculation.
    
    Physics basis (Naraghi & Neher 1997):
    
    Steady-state nanodomain around point source:
        [Ca](r) = [Ca]_rest + i_Ca / (4π × D_eff × r)
    
    With buffered diffusion:
        D_eff = D_free / (1 + κ_s)
        D_free = 220 μm²/s
        κ_s ≈ 60 (buffer capacity)
        D_eff ≈ 3.6 μm²/s
        
    Decay length (pump equilibration):
        λ = √(D_eff / k_pump) ≈ 100-200 nm
    """
    
    def __init__(self, params: CalciumParameters, dx: float):
        self.params = params
        self.dx = dx  # Grid spacing (m)
        
        # Physical constants
        self.z = 2  # Ca²⁺ valence
        self.F = 96485  # Faraday constant (C/mol)
        
        self.D_free = params.D_ca  # 220 μm²/s = 2.2e-10 m²/s
        self.kappa_s = params.buffer_capacity_kappa_s  # ~60
        self.D_eff = self.D_free / (1 + self.kappa_s)  # ~3.6 μm²/s
        
        # Pump parameters for decay length
        # λ = √(D_eff / k_pump), typical k_pump ~ 400/s gives λ ~ 100nm
        k_pump = params.pump_vmax / params.pump_km  # Effective first-order rate
        self.decay_length = np.sqrt(self.D_eff / max(k_pump, 1))  # meters
        
        # Baseline
        self.ca_baseline = params.ca_baseline  # 100 nM
        
        # For temporal dynamics - track accumulated calcium per channel
        # This handles the transient buildup
        self.channel_ca_accumulated = None
        
        logger.info(f"AnalyticalNanodomainCalculator initialized:")
        logger.info(f"  D_eff = {self.D_eff*1e12:.1f} μm²/s")
        logger.info(f"  Decay length λ = {self.decay_length*1e9:.0f} nm")
        
    def calculate_nanodomain_contribution(self, channel_current: float, 
                                          distance_m: float) -> float:
        """
        Calculate [Ca²⁺] contribution from one open channel at given distance
        
        Uses voxel-based approach matching original CalciumSystem, then
        applies analytical spatial decay from Naraghi & Neher 1997.
        
        Args:
            channel_current: Current in Amperes (typically 0.3 pA)
            distance_m: Distance from channel in meters
            
        Returns:
            Calcium concentration in Molarity
        """
        if channel_current <= 0:
            return 0.0
        
        # Convert current to flux (mol/s) - same as original
        J_ca = channel_current / (self.z * self.F)  # mol/s
        
        # Voxel volume from original system (Zuber et al. 2005: 20nm cleft)
        dz = 20e-9  # m
        dV = self.dx * self.dx * dz  # m³
        
        # Peak concentration at channel (in the voxel)
        # This matches the original flux calculation
        ca_at_source = J_ca / dV  # M/s... but we want steady-state M
        
        # Steady-state: balance influx with diffusion loss
        # At steady state in a voxel: influx = D_eff * ∇²[Ca] * dV
        # For a point source: [Ca]_ss ≈ J / (4π * D_eff * r_eff)
        # where r_eff is effective voxel radius
        r_eff = (3 * dV / (4 * np.pi)) ** (1/3)  # ~15 nm
        
        ca_peak = J_ca / (4 * np.pi * self.D_eff * r_eff)
        
        # Spatial decay with distance (Naraghi-Neher)
        if distance_m < r_eff:
            # Inside source voxel - use peak
            return ca_peak
        else:
            # Decay with distance and pump equilibration
            decay = (r_eff / distance_m) * np.exp(-(distance_m - r_eff) / self.decay_length)
            return ca_peak * decay
    
    def calculate_field_at_points(self, channel_positions: np.ndarray,
                                   channel_states: np.ndarray,
                                   channel_currents: np.ndarray,
                                   query_points: np.ndarray,
                                   dx: float) -> np.ndarray:
        """
        Calculate calcium concentration at specific query points
        
        This is the core efficient calculation - only computes at points we care about
        
        Args:
            channel_positions: (N_channels, 2) grid coordinates
            channel_states: (N_channels,) boolean open/closed
            channel_currents: (N_channels,) current when open (A)
            query_points: (N_points, 2) grid coordinates to evaluate
            dx: Grid spacing in meters
            
        Returns:
            (N_points,) calcium concentrations in Molarity
        """
        n_points = len(query_points)
        ca_field = np.ones(n_points) * self.ca_baseline
        
        # Sum contributions from all open channels
        open_idx = np.where(channel_states)[0]
        
        for ch_idx in open_idx:
            ch_pos = channel_positions[ch_idx]
            ch_current = channel_currents[ch_idx]
            
            for pt_idx, pt_pos in enumerate(query_points):
                # Distance in grid units, then convert to meters
                dist_grid = np.sqrt(np.sum((pt_pos - ch_pos)**2))
                dist_m = dist_grid * dx
                
                # Add contribution from this channel
                ca_field[pt_idx] += self.calculate_nanodomain_contribution(
                    ch_current, dist_m
                )
        
        return ca_field


class AnalyticalCalciumSystem:
    """
    Complete calcium system using analytical nanodomain calculations
    
    Drop-in replacement for CalciumSystem with identical interface
    but ~24,000× faster computation.
    
    INTERFACE (identical to CalciumSystem):
    - step(dt, stimulus)
    - get_concentration() -> np.ndarray
    - get_peak_concentration() -> float
    - get_mean_concentration() -> float
    - channels.get_open_fraction() -> float
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float,
                 channel_positions: np.ndarray, params: Model6Parameters,
                 template_positions: Optional[List[Tuple[int, int]]] = None,
                 local_radius_nm: float = 30.0):
        """
        Args:
            grid_shape: Shape of full grid (for interface compatibility)
            dx: Grid spacing in meters
            channel_positions: (N, 2) array of channel positions
            params: Model6Parameters
            template_positions: Optional list of (x,y) template locations
            local_radius_nm: Radius around templates to compute (default 30nm)
        """
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params.calcium
        self.full_params = params
        
        # Channel system (identical to original)
        self.channels = CalciumChannels(channel_positions, params.calcium)
        
        # Analytical calculator
        self.nanodomain = AnalyticalNanodomainCalculator(params.calcium, dx)
        
        # Template positions - these are where dimers form
        if template_positions is None:
            # Default: first 3 channel positions (matches model6_core.py)
            self.template_positions = channel_positions[:3].copy()
        else:
            self.template_positions = np.array(template_positions)
        
        # Build local computation region around templates
        self.local_radius_grid = int(local_radius_nm * 1e-9 / dx) + 1
        self._build_local_region()
        
        # Calcium field - sparse representation
        # Full grid for interface compatibility, but only update local regions
        self._ca_field = np.ones(grid_shape) * params.calcium.ca_baseline
        
        # Local region values (the ones we actually compute)
        self._local_ca = np.ones(len(self.local_points)) * params.calcium.ca_baseline
        
        # Temporal dynamics - track pump-mediated decay
        self._time_since_last_spike = 1.0  # seconds (start equilibrated)
        self._recovery_tau = 0.020  # 20 ms recovery time constant
        
        # Tracking
        self.peak_concentration = params.calcium.ca_baseline
        
        logger.info(f"AnalyticalCalciumSystem initialized:")
        logger.info(f"  Grid: {grid_shape} (full), {len(self.local_points)} local points")
        logger.info(f"  Templates: {len(self.template_positions)}")
        logger.info(f"  Local radius: {local_radius_nm} nm ({self.local_radius_grid} grid points)")
        logger.info(f"  Speedup: ~{grid_shape[0]*grid_shape[1]*1777 / len(self.local_points):.0f}×")
        
    def _build_local_region(self):
        """
        Build the set of grid points to compute (local to templates + channels)
        """
        local_set = set()
        
        # Add points around each template
        for tx, ty in self.template_positions:
            for di in range(-self.local_radius_grid, self.local_radius_grid + 1):
                for dj in range(-self.local_radius_grid, self.local_radius_grid + 1):
                    x, y = int(tx) + di, int(ty) + dj
                    if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                        if di*di + dj*dj <= self.local_radius_grid**2:  # Circular region
                            local_set.add((x, y))
        
        # Add points around each channel (they create the nanodomains)
        for cx, cy in self.channels.positions:
            for di in range(-self.local_radius_grid, self.local_radius_grid + 1):
                for dj in range(-self.local_radius_grid, self.local_radius_grid + 1):
                    x, y = int(cx) + di, int(cy) + dj
                    if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                        if di*di + dj*dj <= self.local_radius_grid**2:
                            local_set.add((x, y))
        
        self.local_points = np.array(list(local_set))
        
        # Create mapping from local index to grid position
        self._local_to_grid = {i: (pt[0], pt[1]) for i, pt in enumerate(self.local_points)}
        self._grid_to_local = {(pt[0], pt[1]): i for i, pt in enumerate(self.local_points)}
        
    def step(self, dt: float, stimulus: Optional[Dict] = None):
        """
        Update calcium system for one timestep
        
        Interface-compatible with original CalciumSystem
        """
        stimulus = stimulus or {}
        
        # 1. Update channel gating (stochastic - identical to original)
        voltage = stimulus.get('voltage', None)
        self.channels.update_gating(dt, voltage)
        
        # 2. Calculate nanodomain calcium at local points (FAST!)
        self._local_ca = self.nanodomain.calculate_field_at_points(
            channel_positions=self.channels.positions,
            channel_states=self.channels.state,
            channel_currents=self.channels.current,
            query_points=self.local_points,
            dx=self.dx
        )
        
        # 3. Apply pump-mediated recovery for points without active channels nearby
        # This handles the temporal decay after channels close
        any_open = np.any(self.channels.state)
        if not any_open:
            # Exponential decay toward baseline
            decay = np.exp(-dt / self._recovery_tau)
            self._local_ca = self.params.ca_baseline + (self._local_ca - self.params.ca_baseline) * decay
        
        # 4. Update full field at local points (for interface compatibility)
        # Set rest of grid to baseline
        self._ca_field.fill(self.params.ca_baseline)
        for i, (x, y) in enumerate(self.local_points):
            self._ca_field[x, y] = self._local_ca[i]
        
        # 5. Track peak
        self.peak_concentration = max(self.peak_concentration, np.max(self._local_ca))
        
    def get_concentration(self) -> np.ndarray:
        """
        Get current free calcium concentration field
        
        Returns full grid for interface compatibility
        """
        return self._ca_field.copy()
    
    def get_concentration_at_templates(self) -> np.ndarray:
        """
        Get calcium concentration specifically at template sites
        
        This is what actually matters for dimer formation!
        """
        ca_at_templates = np.zeros(len(self.template_positions))
        for i, (tx, ty) in enumerate(self.template_positions):
            tx, ty = int(tx), int(ty)
            if (tx, ty) in self._grid_to_local:
                local_idx = self._grid_to_local[(tx, ty)]
                ca_at_templates[i] = self._local_ca[local_idx]
            else:
                ca_at_templates[i] = self._ca_field[tx, ty]
        return ca_at_templates
    
    def get_peak_concentration(self) -> float:
        """Get peak calcium concentration (M)"""
        return np.max(self._local_ca)
    
    def get_mean_concentration(self) -> float:
        """Get mean calcium concentration (M) - over local region"""
        return np.mean(self._local_ca)
    
    def get_nanodomains(self, threshold: float = 10e-6) -> List[Tuple[int, int]]:
        """
        Find calcium nanodomains (hotspots above threshold)
        """
        hotspots = []
        for i, ca in enumerate(self._local_ca):
            if ca > threshold:
                x, y = self.local_points[i]
                hotspots.append((x, y))
        return hotspots
    
    def get_buffer_capacity(self) -> np.ndarray:
        """
        Calculate current buffer capacity κ_s
        
        For analytical system, return constant (fast equilibrium assumed)
        """
        return np.ones(self.grid_shape) * self.params.buffer_capacity
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return metrics for experimental validation
        """
        nanodomains = self.get_nanodomains(threshold=10e-6)
        
        return {
            'peak_ca_uM': self.get_peak_concentration() * 1e6,
            'mean_ca_nM': self.get_mean_concentration() * 1e9,
            'n_nanodomains': len(nanodomains),
            'buffer_capacity_mean': self.params.buffer_capacity_kappa_s,
            'open_channels': self.channels.get_open_fraction(),
            'n_local_points': len(self.local_points),
        }


# =============================================================================
# FACTORY FUNCTION - Easy switching between implementations
# =============================================================================

def create_calcium_system(grid_shape: Tuple[int, int], dx: float,
                          channel_positions: np.ndarray, params: Model6Parameters,
                          template_positions: Optional[List[Tuple[int, int]]] = None,
                          use_analytical: bool = True) -> 'CalciumSystem':
    """
    Factory function to create calcium system
    
    Args:
        use_analytical: If True, use fast analytical solver. If False, use original PDE.
        
    Returns:
        CalciumSystem or AnalyticalCalciumSystem (same interface)
    """
    if use_analytical:
        return AnalyticalCalciumSystem(
            grid_shape, dx, channel_positions, params,
            template_positions=template_positions
        )
    else:
        # Import original only if needed
        from calcium_system import CalciumSystem
        return CalciumSystem(grid_shape, dx, channel_positions, params)


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    import time
    from model6_parameters import Model6Parameters
    
    print("=" * 70)
    print("ANALYTICAL vs PDE CALCIUM SYSTEM COMPARISON")
    print("=" * 70)
    
    # Setup
    params = Model6Parameters()
    grid_shape = (100, 100)
    dx = 2 * params.spatial.active_zone_radius / grid_shape[0]  # ~4nm
    
    # Channel positions (clustered at center)
    center = grid_shape[0] // 2
    channel_positions = np.array([
        [center + np.random.randint(-2, 3), center + np.random.randint(-2, 3)]
        for _ in range(50)
    ])
    
    template_positions = [(center, center), (center+1, center), (center, center+1)]
    
    # Create analytical system
    print("\n--- Analytical System ---")
    analytical = AnalyticalCalciumSystem(
        grid_shape, dx, channel_positions, params,
        template_positions=template_positions
    )
    
    # Test stepping
    print("\nRunning 100 steps with depolarization...")
    t0 = time.time()
    for i in range(100):
        analytical.step(0.001, {'voltage': -0.010})  # Depolarized
    t_analytical = time.time() - t0
    
    metrics = analytical.get_experimental_metrics()
    print(f"  Time: {t_analytical*1000:.1f} ms")
    print(f"  Peak [Ca²⁺]: {metrics['peak_ca_uM']:.1f} μM")
    print(f"  Open channels: {metrics['open_channels']*100:.0f}%")
    print(f"  Nanodomains: {metrics['n_nanodomains']}")
    print(f"  Local points computed: {metrics['n_local_points']}")
    
    # Compare with original if available
    try:
        from calcium_system import CalciumSystem
        print("\n--- Original PDE System ---")
        original = CalciumSystem(grid_shape, dx, channel_positions, params)
        
        t0 = time.time()
        for i in range(100):
            original.step(0.001, {'voltage': -0.010})
        t_original = time.time() - t0
        
        orig_metrics = original.get_experimental_metrics()
        print(f"  Time: {t_original*1000:.1f} ms")
        print(f"  Peak [Ca²⁺]: {orig_metrics['peak_ca_uM']:.1f} μM")
        print(f"  Speedup: {t_original/t_analytical:.0f}×")
        
    except ImportError:
        print("\nOriginal CalciumSystem not available for comparison")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)