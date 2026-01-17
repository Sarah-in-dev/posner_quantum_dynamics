"""
Model 6 Recurrent Network
==========================

A recurrent neural network where each synapse is a FULL Model6QuantumSynapse.

This is the proper integration of Model 6 physics with network dynamics.
Unlike the simplified QuantumRNNSynapse, this preserves ALL Model 6 mechanisms:
- Calcium dynamics with voltage-gated channels
- ATP hydrolysis and phosphate release  
- Prenucleation cluster pathway
- Dimer formation with proper chemistry
- Singlet probability evolution
- Three-factor gate (eligibility + dopamine + calcium)
- CaMKII commitment and plasticity

Architecture:
    External Input → [Neurons with Model6 Synapses] → Output
    
    Each recurrent synapse (i→j) is a Model6QuantumSynapse instance.
    The synapse receives voltage based on pre/post neuron activity,
    implementing NMDA-like coincidence detection.

Key Innovation:
    - Activity → Voltage mapping captures NMDA receptor properties
    - Collective EM field computed across all synapses
    - Cross-synapse entanglement from birth correlations
    - Network learns from delayed reward using quantum eligibility

Author: Sarah Davidson
University of Florida
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import logging
import time

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NeuronConfig:
    """Configuration for rate-based neurons"""
    tau_membrane: float = 0.020      # Membrane time constant (20 ms)
    threshold: float = 1.0           # Soft threshold for sigmoid
    gain: float = 5.0                # Sigmoid steepness
    resting_potential: float = 0.0   # Resting state


@dataclass  
class NetworkConfig:
    """Configuration for Model 6 recurrent network"""
    
    # Network architecture
    n_neurons: int = 10
    connection_probability: float = 1.0  # 1.0 = all-to-all (no self)
    
    # Isotope (for quantum coherence)
    isotope: str = 'P31'  # 'P31' or 'P32'
    
    # Activity thresholds for voltage mapping
    pre_activity_threshold: float = 0.3
    post_activity_threshold: float = 0.3
    
    # Voltage levels (V)
    voltage_resting: float = -70e-3
    voltage_partial: float = -50e-3  
    voltage_depolarized: float = -10e-3
    
    # Weight extraction
    base_weight: float = 0.5
    weight_scale: float = 1.0
    
    # Collective field
    enable_collective_field: bool = True
    field_coupling_strength: float = 1.0
    
    # Neuron config
    neuron_config: NeuronConfig = field(default_factory=NeuronConfig)
    
    # Microtubule invasion (enables tryptophan EM field)
    mt_invaded: bool = True


@dataclass
class NetworkState:
    """Snapshot of network state at one timestep"""
    time: float
    
    # Neuron state
    rates: np.ndarray
    membrane_potentials: np.ndarray
    
    # Aggregate synapse state
    total_dimers: int
    mean_eligibility: float
    mean_singlet_prob: float
    n_committed: int
    
    # Collective effects
    collective_field_kT: float
    n_entangled_synapses: int
    
    # Per-synapse details (optional, for analysis)
    synapse_dimers: Optional[Dict[Tuple[int, int], int]] = None
    synapse_eligibilities: Optional[Dict[Tuple[int, int], float]] = None


# =============================================================================
# RATE NEURON (simple, as the physics is in the synapses)
# =============================================================================

class RateNeuron:
    """
    Simple rate-based neuron.
    
    The interesting physics is in the Model 6 synapses, not here.
    This just provides realistic membrane dynamics.
    """
    
    def __init__(self, config: Optional[NeuronConfig] = None, neuron_id: int = 0):
        self.config = config or NeuronConfig()
        self.neuron_id = neuron_id
        
        self.membrane_potential = self.config.resting_potential
        self.firing_rate = 0.0
        self.total_input = 0.0
    
    def step(self, dt: float, input_current: float) -> float:
        """Update neuron state"""
        self.total_input = input_current
        
        # Leaky integration
        dV = dt * (-self.membrane_potential + input_current) / self.config.tau_membrane
        self.membrane_potential += dV
        
        # Sigmoid output
        x = self.config.gain * (self.membrane_potential - self.config.threshold)
        x = np.clip(x, -500, 500)
        self.firing_rate = 1.0 / (1.0 + np.exp(-x))
        
        return self.firing_rate
    
    def reset(self):
        self.membrane_potential = self.config.resting_potential
        self.firing_rate = 0.0


# =============================================================================
# MODEL 6 RECURRENT NETWORK
# =============================================================================

class Model6RecurrentNetwork:
    """
    Recurrent neural network with Model 6 quantum synapses.
    
    This is the proper way to test quantum eligibility traces:
    - Each synapse is a FULL Model6QuantumSynapse
    - All biophysics preserved (calcium, ATP, dimers, coherence)
    - Network dynamics drive synaptic voltage
    - Collective EM field couples synapses
    
    Usage:
        # Create network
        network = Model6RecurrentNetwork(
            n_neurons=10,
            isotope='P31',
            Model6Class=Model6QuantumSynapse,
            Model6Params=Model6Parameters
        )
        
        # Run simulation
        for t in range(n_steps):
            state = network.step(dt, external_input, reward=False)
        
        # Apply reward
        for t in range(reward_steps):
            state = network.step(dt, external_input, reward=True)
    """
    
    def __init__(self,
                 n_neurons: int = 10,
                 isotope: str = 'P31',
                 Model6Class: Any = None,
                 Model6Params: Any = None,
                 config: Optional[NetworkConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize Model 6 recurrent network.
        
        Parameters
        ----------
        n_neurons : int
            Number of neurons in the recurrent circuit
        isotope : str
            'P31' (quantum, τ~100s) or 'P32' (control, τ~0.4s)
        Model6Class : class
            The Model6QuantumSynapse class to instantiate
        Model6Params : class
            The Model6Parameters class for configuration
        config : NetworkConfig, optional
            Network configuration
        seed : int, optional
            Random seed for reproducibility
        """
        # Configuration
        if config is not None:
            self.config = config
        else:
            self.config = NetworkConfig(n_neurons=n_neurons, isotope=isotope)
        
        self.n_neurons = self.config.n_neurons
        self.rng = np.random.default_rng(seed)
        
        # Store Model 6 classes
        self.Model6Class = Model6Class
        self.Model6Params = Model6Params
        
        # Create neurons
        self.neurons: List[RateNeuron] = []
        for i in range(self.n_neurons):
            neuron = RateNeuron(config=self.config.neuron_config, neuron_id=i)
            self.neurons.append(neuron)
        
        # Create synapses - each is a FULL Model6 instance
        # Dict: (pre_idx, post_idx) -> Model6QuantumSynapse
        self.synapses: Dict[Tuple[int, int], Any] = {}
        self._create_synapses()
        
        # Network state
        self.time = 0.0
        self._current_rates = np.zeros(self.n_neurons)
        
        # Collective field state
        self._collective_field_kT = 0.0
        
        # Statistics
        self.n_synapses = len(self.synapses)
        
        logger.info(f"Model6RecurrentNetwork created: {self.n_neurons} neurons, "
                   f"{self.n_synapses} synapses, isotope={self.config.isotope}")
    
    def _create_synapses(self):
        """Create Model 6 synapses for recurrent connections"""
        
        if self.Model6Class is None:
            raise ValueError("Model6Class must be provided")
        
        synapse_id = 0
        
        for pre in range(self.n_neurons):
            for post in range(self.n_neurons):
                if pre == post:
                    continue  # No self-connections
                
                # Probabilistic connectivity
                if self.rng.random() > self.config.connection_probability:
                    continue
                
                # Create Model 6 parameters
                if self.Model6Params is not None:
                    params = self.Model6Params()
                    
                    # Set isotope
                    if hasattr(params, 'environment'):
                        params.environment.fraction_P31 = 1.0 if self.config.isotope == 'P31' else 0.0
                    
                    # Enable EM coupling
                    if hasattr(params, 'em_coupling_enabled'):
                        params.em_coupling_enabled = self.config.enable_collective_field
                else:
                    params = None
                
                # Create synapse
                synapse = self.Model6Class(params=params)
                
                # Set microtubule invasion
                if hasattr(synapse, 'set_microtubule_invasion'):
                    synapse.set_microtubule_invasion(self.config.mt_invaded)
                
                # Store metadata
                synapse._network_pre = pre
                synapse._network_post = post
                synapse._network_id = synapse_id
                
                self.synapses[(pre, post)] = synapse
                synapse_id += 1
    
    def _compute_synapse_voltage(self, pre_rate: float, post_rate: float) -> float:
        """
        Map neuron activity to synaptic voltage.
        
        This implements NMDA-like coincidence detection:
        - NMDA receptors require BOTH glutamate (pre) AND depolarization (post)
        - Mg²⁺ block is removed only when postsynaptic neuron is depolarized
        
        Returns voltage that Model 6 calcium channels will respond to.
        """
        pre_active = pre_rate > self.config.pre_activity_threshold
        post_active = post_rate > self.config.post_activity_threshold
        
        if pre_active and post_active:
            # Both active → NMDA unblocked → strong calcium influx
            return self.config.voltage_depolarized
        elif pre_active:
            # Pre only → AMPA but NMDA blocked
            return self.config.voltage_partial
        else:
            # Neither or post only → resting
            return self.config.voltage_resting
    
    def _get_synapse_weight(self, synapse) -> float:
        """
        Extract effective synaptic weight from Model 6 state.
        
        Model 6 has CaMKII commitment rather than a simple weight.
        We map commitment level to effective weight.
        """
        base = self.config.base_weight
        
        # Get commitment level
        committed_level = getattr(synapse, '_committed_memory_level', 0.0)
        
        # Could also use CaMKII pT286 or eligibility
        # eligibility = getattr(synapse, '_current_eligibility', 0.0)
        
        return base + committed_level * self.config.weight_scale
    
    def _compute_collective_field(self) -> float:
        """
        Compute collective EM field from all synapses.
        
        The tryptophan network couples synapses through the dendritic
        microtubule network. Field scales as √N (superradiance).
        """
        if not self.config.enable_collective_field:
            return 0.0
        
        total_dimers = 0
        for synapse in self.synapses.values():
            if hasattr(synapse, 'dimer_particles'):
                total_dimers += len(synapse.dimer_particles.dimers)
            elif hasattr(synapse, 'get_experimental_metrics'):
                metrics = synapse.get_experimental_metrics()
                # Estimate dimer count from concentration
                # Active zone volume ~1e-17 L, so nM * 1e-17 * 6e23 ≈ dimers
                dimer_nM = metrics.get('dimer_peak_nM_ct', 0.0)
                total_dimers += int(dimer_nM * 1e-17 * 6e23 / 1e9)
        
        if total_dimers < 5:
            return 0.0
        
        # √N scaling from superradiance
        collective_enhancement = np.sqrt(total_dimers / 5)
        
        # Per-dimer field (from paper: ~0.5 kT per dimer)
        base_field = min(total_dimers, 50) * 0.5
        
        return base_field * collective_enhancement * self.config.field_coupling_strength
    
    def _distribute_collective_field(self, field_kT: float):
        """
        Distribute collective field back to individual synapses.
        
        This enables coordination: synapses see the collective state
        of the network, not just their local state.
        """
        for synapse in self.synapses.values():
            if hasattr(synapse, '_collective_field_kT'):
                synapse._collective_field_kT = field_kT
    
    def step(self, 
             dt: float, 
             external_input: np.ndarray,
             reward: bool = False) -> NetworkState:
        """
        Advance network by one timestep.
        
        Parameters
        ----------
        dt : float
            Timestep in seconds
        external_input : np.ndarray
            External input current to each neuron [n_neurons]
        reward : bool
            Whether reward/dopamine signal is present
            
        Returns
        -------
        NetworkState
            Current network state
        """
        self.time += dt
        
        # Get current rates (from previous step)
        rates = self._current_rates.copy()
        
        # Compute collective field
        self._collective_field_kT = self._compute_collective_field()
        self._distribute_collective_field(self._collective_field_kT)
        
        # Update each synapse with appropriate voltage
        recurrent_input = np.zeros(self.n_neurons)
        
        for (pre, post), synapse in self.synapses.items():
            # Compute voltage based on pre/post activity
            voltage = self._compute_synapse_voltage(rates[pre], rates[post])
            
            # Step the FULL Model 6 synapse
            stimulus = {
                'voltage': voltage,
                'reward': reward
            }
            synapse.step(dt, stimulus)
            
            # Get effective weight for recurrent input
            weight = self._get_synapse_weight(synapse)
            recurrent_input[post] += weight * rates[pre]
        
        # Update neurons
        total_input = external_input + recurrent_input
        
        for i, neuron in enumerate(self.neurons):
            neuron.step(dt, total_input[i])
            self._current_rates[i] = neuron.firing_rate
        
        # Compute network state
        return self._compute_network_state()
    
    def _compute_network_state(self) -> NetworkState:
        """Compute aggregate network state"""
        
        # Aggregate synapse metrics
        total_dimers = 0
        eligibilities = []
        singlet_probs = []
        n_committed = 0
        
        for synapse in self.synapses.values():
            # Dimer count
            if hasattr(synapse, 'dimer_particles'):
                total_dimers += len(synapse.dimer_particles.dimers)
            
            # Eligibility
            if hasattr(synapse, '_current_eligibility'):
                eligibilities.append(synapse._current_eligibility)
            elif hasattr(synapse, 'get_eligibility'):
                eligibilities.append(synapse.get_eligibility())
            
            # Singlet probability
            if hasattr(synapse, 'get_mean_singlet_probability'):
                singlet_probs.append(synapse.get_mean_singlet_probability())
            
            # Commitment
            if getattr(synapse, '_camkii_committed', False):
                n_committed += 1
        
        return NetworkState(
            time=self.time,
            rates=self._current_rates.copy(),
            membrane_potentials=np.array([n.membrane_potential for n in self.neurons]),
            total_dimers=total_dimers,
            mean_eligibility=np.mean(eligibilities) if eligibilities else 0.0,
            mean_singlet_prob=np.mean(singlet_probs) if singlet_probs else 0.25,
            n_committed=n_committed,
            collective_field_kT=self._collective_field_kT,
            n_entangled_synapses=sum(1 for e in eligibilities if e > 0.5)
        )
    
    def get_rates(self) -> np.ndarray:
        """Get current firing rates"""
        return self._current_rates.copy()
    
    def get_weight_matrix(self) -> np.ndarray:
        """Get current effective weight matrix"""
        W = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            W[pre, post] = self._get_synapse_weight(synapse)
        return W
    
    def get_eligibility_matrix(self) -> np.ndarray:
        """Get current eligibility matrix"""
        E = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            if hasattr(synapse, '_current_eligibility'):
                E[pre, post] = synapse._current_eligibility
            elif hasattr(synapse, 'get_eligibility'):
                E[pre, post] = synapse.get_eligibility()
        return E
    
    def get_commitment_matrix(self) -> np.ndarray:
        """Get commitment state matrix"""
        C = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            C[pre, post] = getattr(synapse, '_committed_memory_level', 0.0)
        return C
    
    def get_network_metrics(self) -> Dict:
        """Get comprehensive network metrics"""
        state = self._compute_network_state()
        
        return {
            'time': self.time,
            'n_neurons': self.n_neurons,
            'n_synapses': self.n_synapses,
            'total_dimers': state.total_dimers,
            'mean_eligibility': state.mean_eligibility,
            'mean_singlet_prob': state.mean_singlet_prob,
            'n_committed': state.n_committed,
            'collective_field_kT': state.collective_field_kT,
            'mean_rate': np.mean(self._current_rates),
            'active_neurons': np.sum(self._current_rates > 0.5)
        }
    
    def reset(self):
        """Reset network to initial state"""
        for neuron in self.neurons:
            neuron.reset()
        
        # Note: Model 6 synapses don't have a simple reset
        # For full reset, recreate the network
        
        self._current_rates = np.zeros(self.n_neurons)
        self.time = 0.0
    
    def __repr__(self) -> str:
        return (f"Model6RecurrentNetwork(n_neurons={self.n_neurons}, "
                f"n_synapses={self.n_synapses}, isotope={self.config.isotope})")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model6_network(
    n_neurons: int = 10,
    isotope: str = 'P31',
    mt_invaded: bool = True,
    seed: Optional[int] = None
) -> Model6RecurrentNetwork:
    """
    Factory function to create Model 6 recurrent network.
    
    Handles imports and configuration.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons
    isotope : str
        'P31' (quantum) or 'P32' (control)
    mt_invaded : bool
        Whether microtubules have invaded (enables EM coupling)
    seed : int, optional
        Random seed
        
    Returns
    -------
    Model6RecurrentNetwork
        Configured network
        
    Example
    -------
    >>> from model6_core import Model6QuantumSynapse
    >>> from model6_parameters import Model6Parameters
    >>> 
    >>> network = create_model6_network(
    ...     n_neurons=10, 
    ...     isotope='P31',
    ...     Model6Class=Model6QuantumSynapse,
    ...     Model6Params=Model6Parameters
    ... )
    """
    config = NetworkConfig(
        n_neurons=n_neurons,
        isotope=isotope,
        mt_invaded=mt_invaded
    )
    
    # These need to be imported from your actual Model 6 code
    # from model6_core import Model6QuantumSynapse
    # from model6_parameters import Model6Parameters
    
    # For now, return network configured for later injection
    return Model6RecurrentNetwork(
        config=config,
        seed=seed
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL 6 RECURRENT NETWORK")
    print("=" * 70)
    print("\nThis module requires Model 6 classes to be injected.")
    print("\nUsage:")
    print("  from model6_core import Model6QuantumSynapse")
    print("  from model6_parameters import Model6Parameters")
    print("  ")
    print("  network = Model6RecurrentNetwork(")
    print("      n_neurons=10,")
    print("      isotope='P31',")
    print("      Model6Class=Model6QuantumSynapse,")
    print("      Model6Params=Model6Parameters")
    print("  )")
    print("  ")
    print("  # Run simulation")
    print("  for t in range(1000):")
    print("      state = network.step(dt=0.001, external_input=stimulus, reward=False)")
    print("  ")
    print("  # Check metrics")
    print("  print(network.get_network_metrics())")