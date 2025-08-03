"""
Enhanced Working Posner Model with Nanoreactor Features
Incorporates microdomains, templates, and all enhancement mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

@dataclass
class NanoreactorParameters:
    """Parameters for the synaptic nanoreactor"""
    # Original parameters
    cleft_width: float = 20e-9  # 20 nm
    active_zone_radius: float = 200e-9  # 200 nm
    
    # Chemical parameters - keep original
    ca_baseline: float = 1e-7  # 100 nM
    po4_baseline: float = 1e-3  # 1 mM
    kf_posner: float = 2e-3  # Base formation rate (will be enhanced locally)
    kr_posner: float = 0.5   # Dissolution rate (2s lifetime)
    
    # Environmental conditions
    pH: float = 7.3
    temperature: float = 310  # K
    
    # Quantum parameters
    t2_base_p31: float = 1.0  # Base T2 for ³¹P (s)
    t2_base_p32: float = 0.1  # Base T2 for ³²P (s)
    critical_conc_nM: float = 100  # Concentration where T2 halves
    
    # Stimulation parameters
    spike_amplitude: float = 300e-6  # 300 μM calcium spike
    spike_duration: float = 0.001  # 1 ms decay
    spike_rise_time: float = 0.0001  # 0.1 ms rise
    
    # NEW: Nanoreactor features
    channel_positions: List[Tuple[float, float]] = None  # (x, y) in nm
    template_positions: List[Tuple[float, float]] = None  # Nucleation sites
    
    # Microdomain parameters
    channel_current: float = 0.3e-12  # 0.3 pA single channel
    ca_diffusion_coeff: float = 220e-12  # m²/s for Ca²⁺
    
    # Enhancement factors
    template_enhancement: float = 10.0  # 10× at template sites
    electrostatic_enhancement: float = 3.0  # 3× for phosphate
    confinement_enhancement: float = 5.0  # 2D vs 3D
    
    def __post_init__(self):
        """Initialize channel and template positions"""
        if self.channel_positions is None:
            # Default: 6 channels in hexagonal pattern
            n_channels = 6
            angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
            radius = 100e-9  # 100 nm from center
            self.channel_positions = [
                (radius * np.cos(a), radius * np.sin(a)) 
                for a in angles
            ]
        
        if self.template_positions is None:
            # Templates between channels
            n_templates = 6
            angles = np.linspace(0, 2*np.pi, n_templates, endpoint=False) + np.pi/6
            radius = 80e-9  # Slightly inner ring
            self.template_positions = [
                (radius * np.cos(a), radius * np.sin(a))
                for a in angles
            ]
    
    @property
    def volume(self) -> float:
        """Synaptic cleft volume in L"""
        return np.pi * self.active_zone_radius**2 * self.cleft_width * 1000


class NanoreactorPosnerModel:
    """
    Enhanced model incorporating all nanoreactor features
    """
    
    def __init__(self, isotope: str = 'P31', params: NanoreactorParameters = None):
        self.isotope = isotope
        self.params = params or NanoreactorParameters()
        
        # Set isotope-specific quantum parameters
        if isotope == 'P31':
            self.t2_base = self.params.t2_base_p31
        else:  # P32
            self.t2_base = self.params.t2_base_p32
        
        # Include calcium buffers
        self.buffers = {
            'calmodulin': {'total': 10e-6, 'Kd': 2e-6, 'n_sites': 4},
            'calbindin': {'total': 40e-6, 'Kd': 0.5e-6, 'n_sites': 4},
            'parvalbumin': {'total': 100e-6, 'Kd': 0.05e-6, 'n_sites': 2},
            'ATP': {'total': 2e-3, 'Kd': 150e-6, 'n_sites': 1}
        }
        
        logger.info(f"Initialized nanoreactor model for {isotope}")
    
    def microdomain_concentration(self, distance_from_channel: float) -> float:
        """Calculate steady-state [Ca²⁺] in microdomain"""
        r = max(distance_from_channel, 1e-9)  # Avoid singularity
        
        # Steady-state solution for point source
        i = self.params.channel_current
        D = self.params.ca_diffusion_coeff
        F = 96485  # Faraday constant
        
        C_channel = i / (4 * np.pi * D * r * F * 2)  # Factor of 2 for Ca²⁺
        C_bulk = self.params.ca_baseline
        
        return C_channel + C_bulk
    
    def local_calcium_concentration(self, position: Tuple[float, float], 
                                  bulk_ca: float, channel_open: List[bool]) -> float:
        """Calculate local [Ca²⁺] at given position considering all open channels"""
        max_ca = bulk_ca
        
        for i, (ch_x, ch_y) in enumerate(self.params.channel_positions):
            if i < len(channel_open) and channel_open[i]:
                distance = np.sqrt((position[0] - ch_x)**2 + (position[1] - ch_y)**2)
                local_ca = self.microdomain_concentration(distance)
                max_ca = max(max_ca, local_ca)
        
        return max_ca
    
    def template_enhancement_factor(self, position: Tuple[float, float]) -> float:
        """Calculate formation enhancement from nearby templates"""
        for (t_x, t_y) in self.params.template_positions:
            distance = np.sqrt((position[0] - t_x)**2 + (position[1] - t_y)**2)
            if distance < 10e-9:  # Within 10 nm of template
                return self.params.template_enhancement
        return 1.0
    
    def calculate_free_calcium(self, ca_total: float) -> float:
        """Account for calcium buffering"""
        ca_free = ca_total  # Initial guess
        
        for _ in range(50):  # Iterate to convergence
            ca_bound = 0
            
            for buffer_name, buffer in self.buffers.items():
                occupancy = (ca_free / buffer['Kd']) ** buffer['n_sites']
                fraction_bound = occupancy / (1 + occupancy)
                ca_bound += buffer['total'] * buffer['n_sites'] * fraction_bound
            
            ca_free_new = max(1e-12, ca_total - ca_bound)
            
            if abs(ca_free_new - ca_free) / ca_free < 1e-6:
                break
            
            ca_free = ca_free_new
        
        return ca_free
    
    def nanoreactor_formation_rate(self, position: Tuple[float, float],
                                 bulk_ca: float, po4_conc: float,
                                 channel_open: List[bool]) -> float:
        """Calculate Posner formation rate at specific position in nanoreactor"""
        
        # 1. Get local calcium (including microdomains)
        ca_local = self.local_calcium_concentration(position, bulk_ca, channel_open)
        
        # 2. Account for buffering at this high local concentration
        ca_free = self.calculate_free_calcium(ca_local)
        
        # 3. Electrostatic enhancement of phosphate
        po4_local = po4_conc * self.params.electrostatic_enhancement
        
        # 4. Template enhancement
        template_factor = self.template_enhancement_factor(position)
        
        # 5. 2D confinement enhancement
        confinement_factor = self.params.confinement_enhancement
        
        # 6. Calculate formation rate with all enhancements
        rate = (self.params.kf_posner * template_factor * confinement_factor *
                ca_free * po4_local)
        
        return rate
    
    def simulate_nanoreactor_spike_train(self, n_spikes: int = 20, 
                                       frequency: float = 50.0,
                                       n_positions: int = 10) -> Dict:
        """Simulate Posner formation at multiple nanoreactor positions"""
        
        duration = max(3.0, n_spikes/frequency + 1.0)
        dt = 0.0001
        time = np.arange(0, duration, dt)
        
        # Generate spike times
        spike_times = np.arange(0, n_spikes) / frequency
        
        # Sample positions in active zone
        angles = np.linspace(0, 2*np.pi, n_positions, endpoint=False)
        radii = np.random.uniform(0, self.params.active_zone_radius, n_positions)
        positions = [(r*np.cos(a), r*np.sin(a)) for r, a in zip(radii, angles)]
        
        # Track Posner at each position
        posner_positions = np.zeros((len(time), n_positions))
        
        # Track bulk calcium
        ca_bulk = np.zeros_like(time)
        ca_bulk[0] = self.params.ca_baseline
        
        # Simulate
        for i in range(1, len(time)):
            # Update bulk calcium
            ca_input = self.params.ca_baseline
            for spike_time in spike_times:
                if 0 <= time[i] - spike_time < 0.005:
                    spike_phase = (time[i] - spike_time) / self.params.spike_duration
                    ca_input += self.params.spike_amplitude * np.exp(-spike_phase)
            
            ca_bulk[i] = ca_input
            
            # Determine which channels are open (simplified)
            channel_open = [any(abs(time[i] - st) < 0.001 for st in spike_times) 
                          for _ in self.params.channel_positions]
            
            # Update Posner at each position
            for j, pos in enumerate(positions):
                # Formation rate at this position
                formation_rate = self.nanoreactor_formation_rate(
                    pos, ca_bulk[i], self.params.po4_baseline, channel_open
                )
                
                # Dissolution
                dissolution_rate = self.params.kr_posner * posner_positions[i-1, j]
                
                # Update
                dposner = (formation_rate - dissolution_rate) * dt
                posner_positions[i, j] = max(0, posner_positions[i-1, j] + dposner)
        
        # Calculate ensemble statistics
        posner_mean = np.mean(posner_positions, axis=1)
        posner_max = np.max(posner_positions, axis=1)
        
        # Find hotspots
        final_posner = posner_positions[-1, :]
        hotspot_indices = np.argsort(final_posner)[-3:]  # Top 3 positions
        
        return {
            'time': time,
            'ca_bulk': ca_bulk,
            'posner_mean': posner_mean,
            'posner_max': posner_max,
            'posner_positions': posner_positions,
            'positions': positions,
            'hotspot_indices': hotspot_indices,
            'max_posner_nM': np.max(posner_max) * 1e9,
            'mean_posner_nM': np.max(posner_mean) * 1e9,
            'spike_times': spike_times
        }
    
    def visualize_nanoreactor_results(self, results: Dict):
        """Visualize spatial and temporal Posner formation"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Temporal dynamics
        ax = axes[0, 0]
        ax.plot(results['time'], results['posner_mean'] * 1e9, 'b-', 
                linewidth=2, label='Mean')
        ax.plot(results['time'], results['posner_max'] * 1e9, 'r-', 
                linewidth=2, label='Max')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('[Posner] (nM)')
        ax.set_title('Posner Formation Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Spatial distribution (final)
        ax = axes[0, 1]
        positions = np.array(results['positions'])
        final_posner = results['posner_positions'][-1, :] * 1e9
        
        scatter = ax.scatter(positions[:, 0]*1e9, positions[:, 1]*1e9, 
                           c=final_posner, s=200, cmap='hot', 
                           edgecolors='black', linewidth=2)
        
        # Mark channels and templates
        for ch_pos in self.params.channel_positions:
            ax.plot(ch_pos[0]*1e9, ch_pos[1]*1e9, 'bs', markersize=10)
        for t_pos in self.params.template_positions:
            ax.plot(t_pos[0]*1e9, t_pos[1]*1e9, 'g^', markersize=8)
        
        ax.set_xlabel('X position (nm)')
        ax.set_ylabel('Y position (nm)')
        ax.set_title('Spatial Distribution of Posner')
        plt.colorbar(scatter, ax=ax, label='[Posner] (nM)')
        
        # Draw active zone boundary
        circle = plt.Circle((0, 0), self.params.active_zone_radius*1e9, 
                          fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        
        # 3. Hotspot analysis
        ax = axes[1, 0]
        hotspot_data = []
        for idx in results['hotspot_indices']:
            pos = results['positions'][idx]
            max_posner = np.max(results['posner_positions'][:, idx]) * 1e9
            
            # Find nearest channel
            channel_dists = [np.sqrt((pos[0]-ch[0])**2 + (pos[1]-ch[1])**2) 
                           for ch in self.params.channel_positions]
            nearest_channel_dist = min(channel_dists) * 1e9
            
            # Find nearest template
            template_dists = [np.sqrt((pos[0]-t[0])**2 + (pos[1]-t[1])**2)
                            for t in self.params.template_positions]
            nearest_template_dist = min(template_dists) * 1e9
            
            hotspot_data.append({
                'Position': f"({pos[0]*1e9:.0f}, {pos[1]*1e9:.0f})",
                'Max Posner (nM)': f"{max_posner:.1f}",
                'Channel dist (nm)': f"{nearest_channel_dist:.0f}",
                'Template dist (nm)': f"{nearest_template_dist:.0f}"
            })
        
        # Create text summary
        text = "Hotspot Analysis:\n" + "-"*40 + "\n"
        for i, data in enumerate(hotspot_data):
            text += f"Hotspot {i+1}:\n"
            for key, value in data.items():
                text += f"  {key}: {value}\n"
            text += "\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.axis('off')
        
        # 4. Enhancement breakdown
        ax = axes[1, 1]
        
        # Calculate typical enhancements
        ca_microdomain = self.microdomain_concentration(10e-9) / self.params.ca_baseline
        
        factors = {
            'Base rate': 1,
            'Microdomain Ca²⁺': ca_microdomain,
            'Template': self.params.template_enhancement,
            'Electrostatic': self.params.electrostatic_enhancement,
            'Confinement': self.params.confinement_enhancement
        }
        
        # Create cumulative bar
        values = list(factors.values())
        cumulative = np.cumprod(values)
        
        x = np.arange(len(factors))
        ax.bar(x, cumulative, color='blue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (name, val) in enumerate(factors.items()):
            ax.text(i, cumulative[i]*1.1, f'{cumulative[i]:.0f}×', 
                   ha='center', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(factors.keys(), rotation=45, ha='right')
        ax.set_ylabel('Cumulative Enhancement')
        ax.set_title('Nanoreactor Enhancement Factors')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def validate_enhanced_model(self):
        """Run validation to show nanoreactor achieves physiological Posner levels"""
        
        print("NANOREACTOR MODEL VALIDATION")
        print("="*50)
        
        # Test different conditions
        conditions = {
            'Baseline (1 spike)': {'n_spikes': 1, 'frequency': 1},
            'Moderate (20 spikes @ 10 Hz)': {'n_spikes': 20, 'frequency': 10},
            'Optimal (50 spikes @ 50 Hz)': {'n_spikes': 50, 'frequency': 50},
            'Intense (100 spikes @ 100 Hz)': {'n_spikes': 100, 'frequency': 100}
        }
        
        for name, params in conditions.items():
            results = self.simulate_nanoreactor_spike_train(**params)
            print(f"\n{name}:")
            print(f"  Max [Posner]: {results['max_posner_nM']:.1f} nM")
            print(f"  Mean [Posner]: {results['mean_posner_nM']:.1f} nM")
            
            # Check if we reach physiological levels
            if results['max_posner_nM'] >= 10:
                print("  ✓ Reaches physiological range (>10 nM)")
            else:
                print("  ✗ Below physiological range")
        
        return True


# Quick test
if __name__ == "__main__":
    model = NanoreactorPosnerModel()
    results = model.simulate_nanoreactor_spike_train(n_spikes=50, frequency=50)
    
    print(f"\nWith nanoreactor enhancements:")
    print(f"Max Posner: {results['max_posner_nM']:.1f} nM")
    print(f"Mean Posner: {results['mean_posner_nM']:.1f} nM")
    
    fig = model.visualize_nanoreactor_results(results)
    plt.show()