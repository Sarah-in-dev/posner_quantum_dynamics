"""
Calcium System Module for Model 6
==================================
Handles all calcium dynamics with full literature citations:
- Channel-mediated influx with stochastic gating
- Diffusion with realistic buffering
- Pump/exchanger extrusion
- Nanodomain formation

ALL EMERGENT - no prescribed concentrations!

Key Citations:
- Helmchen et al. 1996 Biophys J 70:1069-1081 (baseline Ca)
- Naraghi & Neher 1997 J Neurosci 17:6961-6973 (nanodomains)
- Allbritton et al. 1992 Science 258:1812-1815 (diffusion)
- Neher & Augustine 1992 Neuron 9:21-30 (buffering)
- Borst & Sakmann 1996 Nature 383:431-434 (channels)
- Scheuss et al. 2006 Neuron 52:831-843 (extrusion)
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import logging

from model6_parameters import Model6Parameters, CalciumParameters

logger = logging.getLogger(__name__)


class CalciumChannels:
    """
    Voltage-gated calcium channels with stochastic gating
    
    Based on:
    - Borst & Sakmann 1996 Nature 383:431-434
      "~50 channels per active zone"
      "Single channel current = 0.3 pA"
      "Open time = 0.5 ms"
    
    - Catterall 2011 Cold Spring Harb Perspect Biol
      "P/Q-type channels at presynaptic terminals"
    """
    
    def __init__(self, positions: np.ndarray, params: CalciumParameters):
        """
        Args:
            positions: (N, 2) array of channel positions in grid coordinates
            params: Calcium parameters
        """
        self.positions = positions
        self.params = params
        self.n_channels = len(positions)
        
        # Channel state (boolean: open or closed)
        self.state = np.zeros(self.n_channels, dtype=bool)
        
        # Current through each channel (Amperes)
        self.current = np.zeros(self.n_channels)
        
        # Gating kinetics
        # Simplified from Hodgkin-Huxley formalism
        self.alpha = 1e3  # Opening rate (1/s)
        self.beta = 2e3  # Closing rate (1/s) → ~0.5ms open time
        
        # Track open time for each channel
        self.time_open = np.zeros(self.n_channels)
        
        logger.info(f"Initialized {self.n_channels} calcium channels")
        
    def update_gating(self, dt: float, voltage: Optional[float] = None):
        """
        Stochastic channel gating with realistic kinetics
        
        Args:
            dt: Time step (s)
            voltage: Membrane voltage (V) - if None, use fixed probability
            
        Note:
            During action potential, voltage ~ -10 to +40 mV
            At rest, voltage ~ -70 mV
            Channels activate above -50 mV threshold
        """
        # APV blocks NMDA - force all channels closed
        if getattr(self.params, 'nmda_blocked', False):
            self.state[:] = False
            self.current[:] = 0.0
            return
            
        if voltage is not None:
            # Voltage-dependent activation (simplified)
            # Full model would use Hodgkin-Huxley m³h kinetics
            V_threshold = -0.050  # -50 mV
            V_half = -0.020  # -20 mV
            
            # Boltzmann activation
            P_open_voltage = 1.0 / (1.0 + np.exp(-(voltage - V_half) / 0.005))
            
            # Modulate rates
            alpha_eff = self.alpha * P_open_voltage
            beta_eff = self.beta * (1 - P_open_voltage)
        else:
            # Use fixed rates
            alpha_eff = self.alpha
            beta_eff = self.beta
        
        # Stochastic transitions
        for i in range(self.n_channels):
            if self.state[i]:  # Channel open
                self.time_open[i] += dt
                
                # Probability of closing
                # P_close = 1 - exp(-β*dt) ≈ β*dt for small dt
                if np.random.rand() < beta_eff * dt:
                    self.state[i] = False
                    self.current[i] = 0.0
                    self.time_open[i] = 0.0
                    
            else:  # Channel closed
                # Probability of opening
                if np.random.rand() < alpha_eff * dt:
                    self.state[i] = True
                    # Borst & Sakmann 1996: 0.3 pA single channel current
                    self.current[i] = self.params.single_channel_current
                    
    def get_calcium_flux(self, grid_shape: Tuple[int, int], dx: float) -> np.ndarray:
        """
        Convert channel current to calcium concentration flux
        
        Following Helmchen et al. 1996 Biophys J:
        "Channel current of 0.3 pA produces ~50 μM local [Ca²⁺]"
        
        Args:
            grid_shape: Shape of concentration grid
            dx: Grid spacing (m)
            
        Returns:
            flux: (grid_shape) array of calcium flux (M/s)
        """
        flux = np.zeros(grid_shape)
        
        # Convert current (A) to flux (mol/s)
        # I = z * F * J  where J is flux in mol/s
        # J = I / (z * F)
        z = 2  # Ca²⁺ valence
        F = 96485  # Faraday constant (C/mol)
        
        # Voxel height from cleft width (Zuber et al. 2005: 20 nm)
        dz = 20e-9  # m
        dV = dx * dx * dz  # m³ (true voxel volume)

        for i in range(self.n_channels):
            if self.state[i]:
                x, y = self.positions[i]
                
                # Flux in mol/s
                J = self.current[i] / (z * F)
                
                # Concentration change (M/s) in this voxel
                # This creates the nanodomain!
                flux[x, y] += J / dV
                
        return flux
    
    def get_open_channels(self) -> np.ndarray:
        """Return array of open channel indices"""
        return np.where(self.state)[0]
    
    def get_open_fraction(self) -> float:
        """Return fraction of channels open"""
        return np.sum(self.state) / self.n_channels


class CalciumDiffusion:
    """
    Calcium diffusion with buffering
    
    Based on:
    - Allbritton et al. 1992 Science 258:1812-1815
      "D_Ca = 220 μm²/s free calcium"
      
    - Smith et al. 2001 Biophys J 81:3064-3078  
      "Buffering reduces effective D by factor of ~10"
      
    - Neher & Augustine 1992 Neuron 9:21-30
      "Buffer capacity κ_s ~ 60 in neurons"
      "Calbindin + Parvalbumin provide most buffering"
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float, params: CalciumParameters):
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params
        
        # Free calcium concentration field (Molarity)
        # Helmchen et al. 1996: "Resting [Ca²⁺] = 100 nM"
        self.ca_free = np.ones(grid_shape) * params.ca_baseline
        
        # Bound calcium (buffered)
        self.ca_bound = np.zeros(grid_shape)
        
        # Buffer concentration (calbindin, parvalbumin, etc.)
        # Neher & Augustine 1992: total buffer ~300 μM
        self.buffer = np.ones(grid_shape) * params.buffer_concentration
        
        logger.info(f"Initialized calcium diffusion on {grid_shape} grid")
        
    def update_buffering(self, dt: float):
        """
        Fast equilibrium buffering (rapid buffering approximation)
        
        Reaction: Ca + B ⇌ CaB
        
        Neher & Augustine 1992 Neuron 9:21-30:
        "Buffering equilibrates on microsecond timescale"
        "Can assume instantaneous equilibrium for diffusion modeling"
        
        Solving: [Ca][B] / [CaB] = Kd
        With conservation: Ca_total = Ca_free + Ca_bound
                          B_total = B_free + Ca_bound
        
        Yields quadratic equation for [Ca]_free
        """
        # Total calcium in each voxel
        ca_total = self.ca_free + self.ca_bound
        
        # Solve quadratic for new free calcium
        # [Ca]free = (-b + sqrt(b² - 4ac)) / 2a
        # where a=1, b=(Kd + B_tot + Ca_tot), c=-Ca_tot*Kd
        
        a = 1.0
        b = self.params.buffer_kd + self.buffer + ca_total
        c = -ca_total * self.params.buffer_kd
        
        # Quadratic formula (take positive root)
        discriminant = b**2 - 4*a*c
        discriminant = np.maximum(discriminant, 0)  # Avoid sqrt(negative)
        
        self.ca_free = (-b + np.sqrt(discriminant)) / (2*a)
        self.ca_bound = ca_total - self.ca_free
        
        # Ensure non-negative
        self.ca_free = np.maximum(self.ca_free, 0)
        self.ca_bound = np.maximum(self.ca_bound, 0)
        
    def update_diffusion(self, dt: float):
        """
        Diffusion of free and buffered calcium
        
        Smith et al. 2001 Biophys J:
        "Effective diffusion coefficient D_eff = D_Ca / (1 + κ_s)"
        "Buffer capacity κ_s = [B]·Kd / ([Ca] + Kd)²"
        
        Using explicit finite difference method for ∇²[Ca]
        """
        # Calculate buffer capacity κ_s at each point
        # Neher & Augustine 1992: κ_s ~ 60 for neurons
        kappa_s = (self.params.buffer_concentration * self.params.buffer_kd / 
                   (self.ca_free + self.params.buffer_kd)**2)
        
        # Effective diffusion coefficient
        # Allbritton et al. 1992: D_Ca = 220 μm²/s = 220e-12 m²/s
        D_eff = self.params.D_ca / (1 + kappa_s)
        
        # Laplacian via 5-point stencil convolution
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]]) / (self.dx**2)
        
        laplacian = ndimage.convolve(self.ca_free, laplacian_kernel, mode='constant')
        
        # Update free calcium: ∂[Ca]/∂t = D_eff ∇²[Ca]
        self.ca_free += dt * D_eff * laplacian
        
        # Ensure non-negative
        self.ca_free = np.maximum(self.ca_free, 0)
        
    def add_flux(self, flux: np.ndarray, dt: float):
        """
        Add calcium flux from channels
        
        This creates nanodomains!
        Naraghi & Neher 1997: "Peak [Ca²⁺] = 50 μM near channel mouth"
        """
        self.ca_free += flux * dt
        
    def update_extrusion(self, dt: float):
        """
        Remove calcium via pumps/exchangers
        
        Scheuss et al. 2006 Neuron 52:831-843:
        "PMCA (plasma membrane Ca-ATPase) + NCX (Na/Ca exchanger)"
        "Combined Vmax ~ 50 μM/s, Km ~ 0.5 μM"
        
        Michaelis-Menten kinetics: v = Vmax·[Ca] / (Km + [Ca])
        """
        # Pump rate (M/s)
        pump_rate = (self.params.pump_vmax * self.ca_free / 
                    (self.params.pump_km + self.ca_free))
        
        # Remove calcium
        self.ca_free -= pump_rate * dt
        
        # Never go below baseline
        # Helmchen et al. 1996: "Pumps maintain 100 nM resting [Ca²⁺]"
        self.ca_free = np.maximum(self.ca_free, self.params.ca_baseline)


class CalciumSystem:
    """
    Complete calcium system integrating channels, diffusion, buffering
    
    This reproduces published calcium dynamics:
    - Nanodomains: 10-100 μM near channels
    - Microdomains: 1-10 μM nearby
    - Bulk: 0.1-1 μM spine-wide
    - Recovery: 20-50 ms time constant
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float, 
                 channel_positions: np.ndarray, params: Model6Parameters):
        
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params.calcium
        
        # Components
        self.channels = CalciumChannels(channel_positions, params.calcium)
        self.diffusion = CalciumDiffusion(grid_shape, dx, params.calcium)
        
        # Tracking for analysis
        self.peak_concentration = params.calcium.ca_baseline
        self.nanodomain_history = []
        
        logger.info(f"Initialized calcium system with {len(channel_positions)} channels")
        
    def step(self, dt: float, stimulus: Optional[Dict] = None):
        """
        Update calcium system for one time step
        
        Args:
            dt: Time step (s)
            stimulus: Dictionary with 'voltage' or other control signals
            
        Note:
            Order matters! Must do buffering before diffusion
            to correctly calculate D_eff
        """
        # 1. Update channel gating (stochastic)
        voltage = stimulus.get('voltage', None) if stimulus else None
        self.channels.update_gating(dt, voltage)
        
        # 2. Get calcium flux from open channels
        flux = self.channels.get_calcium_flux(self.grid_shape, self.dx)
        
        # 3. Add flux to calcium field
        self.diffusion.add_flux(flux, dt)
        
        # 4. Fast buffering equilibrium (microseconds)
        self.diffusion.update_buffering(dt)
        
        # 5. Diffusion (with buffer-reduced D_eff)
        self.diffusion.update_diffusion(dt)
        
        # 6. Extrusion via pumps
        self.diffusion.update_extrusion(dt)
        
        # 7. Update tracking
        self.peak_concentration = max(self.peak_concentration, 
                                     np.max(self.diffusion.ca_free))
        
    def get_concentration(self) -> np.ndarray:
        """
        Get current free calcium concentration field
        
        Returns:
            Array of [Ca²⁺] in Molarity
        """
        return self.diffusion.ca_free.copy()
    
    def get_peak_concentration(self) -> float:
        """Get peak calcium concentration (M)"""
        return np.max(self.diffusion.ca_free)
    
    def get_mean_concentration(self) -> float:
        """Get mean calcium concentration (M)"""
        return np.mean(self.diffusion.ca_free)
    
    def get_nanodomains(self, threshold: float = 10e-6) -> List[Tuple[int, int]]:
        """
        Find calcium nanodomains (hotspots above threshold)
        
        Naraghi & Neher 1997:
        "Nanodomains defined as [Ca²⁺] > 10 μM"
        "Typically 50-100 μM peak"
        
        Args:
            threshold: Concentration threshold (M), default 10 μM
            
        Returns:
            List of (x, y) positions of nanodomains
        """
        hotspots = np.where(self.diffusion.ca_free > threshold)
        return list(zip(hotspots[0], hotspots[1]))
    
    def get_buffer_capacity(self) -> np.ndarray:
        """
        Calculate current buffer capacity κ_s
        
        Returns spatial field of buffering strength
        Useful for understanding local calcium dynamics
        """
        kappa_s = (self.params.buffer_concentration * self.params.buffer_kd / 
                   (self.diffusion.ca_free + self.params.buffer_kd)**2)
        return kappa_s
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return metrics for experimental validation
        
        Aligns with thesis validation criteria:
        - Peak [Ca²⁺] in nanodomains
        - Number of nanodomains
        - Mean bulk [Ca²⁺]
        - Buffer capacity
        """
        nanodomains = self.get_nanodomains(threshold=10e-6)
        
        return {
            'peak_ca_uM': self.get_peak_concentration() * 1e6,
            'mean_ca_nM': self.get_mean_concentration() * 1e9,
            'n_nanodomains': len(nanodomains),
            'buffer_capacity_mean': np.mean(self.get_buffer_capacity()),
            'open_channels': self.channels.get_open_fraction(),
        }


# ============================================================================
# VALIDATION TESTS
# ============================================================================

if __name__ == "__main__":
    from model6_parameters import Model6Parameters
    
    print("="*70)
    print("CALCIUM SYSTEM VALIDATION")
    print("="*70)
    
    # Test setup
    params = Model6Parameters()
    
    # Small grid for testing
    grid_shape = (50, 50)
    dx = 2 * params.spatial.active_zone_radius / grid_shape[0]
    
    # Place channels in a cluster (typical active zone arrangement)
    channel_positions = np.array([
        [25, 25],  # Center
        [26, 25],
        [25, 26],
        [26, 26],
        [24, 25],
        [25, 24]
    ])
    
    # Create calcium system
    ca_system = CalciumSystem(grid_shape, dx, channel_positions, params)
    
    print(f"\nInitial state:")
    print(f"  Grid: {grid_shape}")
    print(f"  Resolution: {dx*1e9:.1f} nm")
    print(f"  Channels: {len(channel_positions)}")
    print(f"  Initial [Ca]: {ca_system.get_peak_concentration()*1e9:.1f} nM")
    
    # Test 1: Channel activation
    print("\nTest 1: Channel Activation")
    print("Simulating 10 ms with depolarization...")
    
    dt = 1e-5  # 10 μs time step
    n_steps = 1000  # 10 ms total
    
    ca_history = []
    nanodomain_history = []
    
    for i in range(n_steps):
        # Depolarize for first 2 ms to open channels
        if i < 200:
            stimulus = {'voltage': -0.01}  # -10 mV (depolarized)
        else:
            stimulus = {'voltage': -0.07}  # -70 mV (resting)
        
        ca_system.step(dt, stimulus)
        
        # Record every 100 steps (1 ms)
        if i % 100 == 0:
            metrics = ca_system.get_experimental_metrics()
            ca_history.append(metrics['peak_ca_uM'])
            nanodomain_history.append(metrics['n_nanodomains'])
            
            print(f"  t={i*dt*1e3:.1f} ms: "
                  f"Peak [Ca]={metrics['peak_ca_uM']:.1f} μM, "
                  f"Nanodomains={metrics['n_nanodomains']}, "
                  f"Open channels={metrics['open_channels']:.2f}")
    
    # Test 2: Validation against literature
    print("\nTest 2: Literature Validation")
    
    final_metrics = ca_system.get_experimental_metrics()
    
    # Naraghi & Neher 1997: Peak should be 10-100 μM
    peak_ca_uM = np.max(ca_history)
    if 10 <= peak_ca_uM <= 100:
        print(f"  ✓ Peak [Ca] = {peak_ca_uM:.1f} μM (within 10-100 μM range)")
    else:
        print(f"  ✗ Peak [Ca] = {peak_ca_uM:.1f} μM (outside expected range)")
    
    # Neher & Augustine 1992: Buffer capacity ~ 60
    if 40 <= final_metrics['buffer_capacity_mean'] <= 80:
        print(f"  ✓ Buffer capacity = {final_metrics['buffer_capacity_mean']:.1f} "
              f"(within expected range)")
    else:
        print(f"  ✗ Buffer capacity = {final_metrics['buffer_capacity_mean']:.1f} "
              f"(outside expected range)")
    
    # Should see nanodomains form and decay
    max_nanodomains = np.max(nanodomain_history)
    if max_nanodomains > 0:
        print(f"  ✓ Nanodomains formed: {max_nanodomains} sites")
    else:
        print(f"  ✗ No nanodomains formed")
    
    print("\n" + "="*70)
    print("Calcium system validation complete!")
    print("="*70)