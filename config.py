"""
Flexible configuration system for Posner Quantum Dynamics project
Supports easy experimentation and parameter sweeps
"""
import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

# Create config directory if it doesn't exist
CONFIG_DIR.mkdir(exist_ok=True)

# Physical constants (these shouldn't change)
CONSTANTS = {
    'AVOGADRO': 6.022e23,  # mol^-1
    'BOLTZMANN': 1.381e-23,  # J/K
    'PLANCK': 6.626e-34,  # J·s
    'HBAR': 1.055e-34,  # J·s
    'ELECTRON_CHARGE': 1.602e-19,  # C
    'SPEED_OF_LIGHT': 2.998e8,  # m/s
}

@dataclass
class ExperimentalConfig:
    """Configuration for a single experiment"""
    name: str = "default"
    description: str = ""
    
    # Synaptic parameters
    cleft_width: float = 20e-9  # m
    active_zone_radius: float = 200e-9  # m
    
    # Chemical parameters
    ca_baseline: float = 1e-7  # M
    po4_baseline: float = 1e-3  # M
    kf_posner: float = 2e-3  # formation rate
    kr_posner: float = 0.5  # dissolution rate
    
    # Environmental conditions
    pH: float = 7.3
    temperature: float = 310  # K
    
    # Quantum parameters
    isotope: str = 'P31'
    b_field: float = 50e-6  # T
    j_coupling_intra: float = 1.0  # Hz
    
    # Stimulation parameters
    n_spikes: int = 10
    frequency: float = 10.0  # Hz
    spike_amplitude: float = 300e-6  # M
    
    # Simulation parameters
    dt: float = 0.0001  # s
    total_time: float = 2.0  # s
    
    # Analysis parameters
    coherence_threshold: float = 0.5
    
    # Sweep parameters (for parameter studies)
    sweep_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def save(self, filename: str = None):
        """Save configuration to file"""
        if filename is None:
            filename = CONFIG_DIR / f"{self.name}.json"
        else:
            filename = Path(filename)
            
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'ExperimentalConfig':
        """Load configuration from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def copy(self, **kwargs) -> 'ExperimentalConfig':
        """Create a copy with modified parameters"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return ExperimentalConfig(**config_dict)


class ConfigurationManager:
    """Manages multiple experimental configurations"""
    
    def __init__(self):
        self.configs = {}
        self.active_config = None
        
    def create_config(self, name: str, base_config: str = None, **kwargs) -> ExperimentalConfig:
        """Create a new configuration"""
        if base_config and base_config in self.configs:
            config = self.configs[base_config].copy(name=name, **kwargs)
        else:
            config = ExperimentalConfig(name=name, **kwargs)
        
        self.configs[name] = config
        return config
    
    def load_config(self, name: str, filename: str = None) -> ExperimentalConfig:
        """Load configuration from file"""
        if filename is None:
            filename = CONFIG_DIR / f"{name}.json"
        
        config = ExperimentalConfig.load(filename)
        self.configs[name] = config
        return config
    
    def set_active(self, name: str):
        """Set active configuration"""
        if name in self.configs:
            self.active_config = self.configs[name]
        else:
            raise ValueError(f"Configuration '{name}' not found")
    
    def get_active(self) -> ExperimentalConfig:
        """Get active configuration"""
        if self.active_config is None:
            self.active_config = ExperimentalConfig()
        return self.active_config
    
    def create_parameter_sweep(self, base_name: str, param_name: str, 
                             param_values: np.ndarray) -> Dict[str, ExperimentalConfig]:
        """Create configurations for parameter sweep"""
        sweep_configs = {}
        
        base_config = self.configs.get(base_name, ExperimentalConfig())
        
        for i, value in enumerate(param_values):
            name = f"{base_name}_{param_name}_{i}"
            config = base_config.copy(
                name=name,
                description=f"Sweep {param_name} = {value}",
                **{param_name: value}
            )
            config.sweep_params = {
                'parameter': param_name,
                'value': value,
                'index': i,
                'total': len(param_values)
            }
            sweep_configs[name] = config
            
        return sweep_configs


# Predefined experimental configurations
def create_standard_configs() -> Dict[str, ExperimentalConfig]:
    """Create standard experimental configurations"""
    configs = {}
    
    # Default physiological conditions
    configs['physiological'] = ExperimentalConfig(
        name='physiological',
        description='Standard physiological conditions',
        pH=7.2,
        temperature=310,
        cleft_width=20e-9
    )
    
    # Optimal quantum conditions (based on preliminary analysis)
    configs['quantum_optimal'] = ExperimentalConfig(
        name='quantum_optimal',
        description='Optimized for quantum coherence',
        pH=7.3,
        temperature=308,  # Slightly cooler
        cleft_width=22e-9,  # Slightly wider
        frequency=15.0  # Higher frequency
    )
    
    # BCI simulation conditions
    configs['bci_learning'] = ExperimentalConfig(
        name='bci_learning',
        description='BCI learning paradigm',
        n_spikes=100,
        frequency=10.0,
        total_time=10.0,
        isotope='P31'
    )
    
    # Isotope comparison
    configs['isotope_p31'] = ExperimentalConfig(
        name='isotope_p31',
        description='Phosphorus-31 conditions',
        isotope='P31'
    )
    
    configs['isotope_p32'] = ExperimentalConfig(
        name='isotope_p32',
        description='Phosphorus-32 conditions',
        isotope='P32'
    )
    
    # Extreme conditions for testing
    configs['high_calcium'] = ExperimentalConfig(
        name='high_calcium',
        description='High calcium spike amplitude',
        spike_amplitude=1000e-6  # 1 mM
    )
    
    configs['acidic'] = ExperimentalConfig(
        name='acidic',
        description='Acidic pH conditions',
        pH=6.8
    )
    
    configs['cold'] = ExperimentalConfig(
        name='cold',
        description='Lower temperature',
        temperature=300  # 27°C
    )
    
    return configs


# Global configuration manager
config_manager = ConfigurationManager()

# Load standard configurations
for name, config in create_standard_configs().items():
    config_manager.configs[name] = config

# Set default active configuration
config_manager.set_active('physiological')


# Convenience functions for experiments
def get_config(name: str = None) -> ExperimentalConfig:
    """Get configuration by name or active config"""
    if name:
        return config_manager.configs.get(name, config_manager.get_active())
    return config_manager.get_active()


def create_sweep(param_name: str, start: float, stop: float, 
                num: int = 20, scale: str = 'linear') -> np.ndarray:
    """Create parameter sweep values"""
    if scale == 'linear':
        return np.linspace(start, stop, num)
    elif scale == 'log':
        return np.logspace(np.log10(start), np.log10(stop), num)
    else:
        raise ValueError(f"Unknown scale: {scale}")


# Example usage functions
def example_parameter_study():
    """Example of how to use configs for parameter studies"""
    
    # pH study
    ph_values = create_sweep('pH', 6.5, 7.8, num=15)
    ph_configs = config_manager.create_parameter_sweep('physiological', 'pH', ph_values)
    
    # Temperature study
    temp_values = create_sweep('temperature', 295, 315, num=10)  # 22-42°C
    temp_configs = config_manager.create_parameter_sweep('physiological', 'temperature', temp_values)
    
    # Frequency study
    freq_values = create_sweep('frequency', 1, 100, num=20, scale='log')
    freq_configs = config_manager.create_parameter_sweep('physiological', 'frequency', freq_values)
    
    return ph_configs, temp_configs, freq_configs


def example_custom_config():
    """Example of creating custom configuration"""
    
    # Create a custom config for specific experiment
    custom = config_manager.create_config(
        'my_experiment',
        base_config='physiological',
        pH=7.4,
        temperature=305,
        n_spikes=50,
        frequency=20.0,
        description='Testing higher frequency with cooler temperature'
    )
    
    # Save it for later use
    custom.save()
    
    return custom


if __name__ == "__main__":
    # Demonstrate configuration system
    print("Available configurations:")
    for name, config in config_manager.configs.items():
        print(f"  - {name}: {config.description}")
    
    print(f"\nActive configuration: {config_manager.get_active().name}")
    
    # Show how to create a parameter sweep
    print("\nCreating pH sweep...")
    ph_configs, _, _ = example_parameter_study()
    print(f"Created {len(ph_configs)} configurations for pH sweep")
    
    # Show how to access parameters
    config = get_config()
    print(f"\nCurrent pH: {config.pH}")
    print(f"Current temperature: {config.temperature - 273.15:.1f}°C")
