"""
Model 6 Core - Complete Quantum Synapse Integration
====================================================
Integrates all subsystems into a complete emergent quantum synapse model

CORRECTED ARCHITECTURE (Nov 2025):
- Calcium system (stochastic channel gating, diffusion, buffering)
- ATP system (hydrolysis, J-coupling, phosphate)
- Ca₆(PO₄)₄ dimer formation (ca_triphosphate_complex.py)
- Quantum coherence tracking (quantum_coherence_system.py - NOT posner!)
- pH dynamics (activity-dependent acidification)
- Dopamine system (reward signaling)

KEY FIX: Quantum coherence tracks dimers FROM ca_triphosphate, not from PNCs
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import h5py
from datetime import datetime

from model6_parameters import Model6Parameters, compute_metabolic_power

# =============================================================================
# PLATEAU POTENTIAL — the dendrite-wide instructive depolarization (2026-07-18)
# =============================================================================
# Applied in step() via a driving-force-scaled combination, NOT summation. See the
# block in STEP 1 for the algebra and why summation is physically impossible here.
#
# PLATEAU_VOLTAGE_V = -20 mV. Grounded three independent ways:
#   1. Local dendritic plateau measurements: 57.6 +/- 5.5 mV above rest, absolute
#      range -21 to -10 mV (Antic 2020, PMC8087381; neocortical basal dendrites,
#      voltage-imaging + model inference). CA1 plateau termination break point
#      -9.9 +/- 1 mV (J Neurosci 2026, PMC12157498). -20 mV is the CONSERVATIVE end.
#   2. This repo already named it: sweep/run_spatial_discovery.py:360-365 — "the
#      plateau (~-20 mV, a SEPARATE instructive event) is not this knob".
#   3. It coincides with the VGCC Boltzmann midpoint already in
#      analytical_calcium_system.py:80 (V_half = -0.020 V), written independently.
#
# !! SENSITIVITY WARNING !! Because V_half == PLATEAU_VOLTAGE_V, VGCC open
# probability is at MAXIMUM sensitivity to this value: -20 mV -> P_open 0.50, but
# -15 mV -> 0.70. The literature pins the plateau only to +/-5-10 mV. Any result that
# leans on plateau-driven VGCC influx MUST be swept over this value, not reported at
# a single point. Do NOT tune it to make a downstream number come out.
#
# NOTE (somatic vs dendritic): somatic recordings show only ~10-20 mV depolarization
# with a ~7-fold distance gradient. "Plateaus are 20 mV" from somatic data describes a
# DIFFERENT quantity. This constant is the LOCAL dendritic value, which is the right
# frame because the model applies it per-synapse.
PLATEAU_VOLTAGE_V: float = -20e-3
PLATEAU_V_REST: float = -70e-3
PLATEAU_E_SYN: float = 0.0        # AMPA/NMDA reversal potential; the hard bound
# Duration lives with the drivers, not here (it is a protocol property). Central
# value 0.250 s: Bittner/Milstein/Magee 2017 in vivo naturally-occurring plateaus
# ~265 ms (PMC7289271). CONTESTED: CA1 slice gives 140.2 +/- 10.2 ms (PMC12157498);
# neocortical basal 200-500 ms. The in-vivo/slice discrepancy is unexplained. Honest
# range ~80-500 ms — sweep it rather than trusting the central value.
# from calcium_system import CalciumSystem
from analytical_calcium_system import AnalyticalCalciumSystem as CalciumSystem
from atp_system import ATPSystem
from ca_triphosphate_complex import CaHPO4DimerSystem
from quantum_coherence import QuantumCoherenceSystem  # CHANGED from posner_system
from pH_dynamics import pHDynamics
from camkii_module import CaMKIIModule
from spine_plasticity_module import SpinePlasticityModule
from ddsc_module import DDSCSystem, DDSCParameters

from dimer_particles import DimerParticleSystem

# Import dopamine system
try:
    from dopamine_system import DopamineSystemAdapter
    HAS_DOPAMINE = True
except ImportError:
    HAS_DOPAMINE = False
    logging.warning("dopamine_system not found - dopamine effects disabled")

# EM coupling modules (optional)
try:
    from em_tryptophan_module import TryptophanSuperradianceModule
    from em_coupling_module import EMCouplingModule
    from vibrational_cascade_module import VibrationalCascadeModule
    from local_dimer_tubulin_coupling import LocalDimerTubulinCoupling, NetworkModulationIntegrator
    HAS_EM_COUPLING = True
except ImportError:
    HAS_EM_COUPLING = False
    logging.info("EM coupling modules not found - running Model 6 baseline")

# Cross-region modules
try:
    from photon_emission_module import PhotonEmissionTracker
    from photon_receiver_module import PhotonReceiver
    HAS_CROSS_REGION = True
except ImportError:
    HAS_CROSS_REGION = False
    logging.info("Cross-region modules not found - single-region only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model6QuantumSynapse:
    """
    Complete quantum synapse model integrating all subsystems
    
    Architecture:
    1. Calcium influx (stochastic channels)
    2. ATP hydrolysis → phosphate + J-coupling
    3. CaHPO₄ ion pairs → Ca₆(PO₄)₄ dimers (ca_triphosphate)
    4. Quantum coherence tracking of dimers (quantum_coherence_system)
    5. pH dynamics
    6. Dopamine modulation (optional)
    """
    
    def __init__(self, params: Optional[Model6Parameters] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize complete quantum synapse model
        
        Args:
            params: Model parameters (default: natural conditions)
            output_dir: Directory for saving results
        """
        # Parameters
        self.params = params or Model6Parameters()
        
        # Grid setup
        self.grid_shape = (self.params.spatial.grid_size, 
                          self.params.spatial.grid_size)
        self.dx = (2 * self.params.spatial.active_zone_radius / 
                   self.params.spatial.grid_size)
        
        # Time tracking
        self.time = 0.0
        self.dt = self.params.simulation.dt_diffusion  # Base timestep
        
        self._peak_calcium_uM = 0.0  # Track peak calcium for plasticity
        
        # Output
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subsystems
        self._initialize_subsystems()
        
        
        # Initialize EM coupling (optional)
        self.em_enabled = (self.params.em_coupling_enabled and HAS_EM_COUPLING)
        
        if self.em_enabled:
            self.tryptophan = TryptophanSuperradianceModule(self.params)
            self.em_coupling = VibrationalCascadeModule(self.params)
            self.local_dimer_coupling = LocalDimerTubulinCoupling()
            self.network_integrator = NetworkModulationIntegrator()
            self._network_modulation = 0.0
            
            logger.info("EM coupling ENABLED")
        else:
            self.tryptophan = None
            self.em_coupling = None
            self.local_dimer_coupling = None
            self.network_integrator = None
            self._network_modulation = 0.0
            logger.info("EM coupling DISABLED (Model 6 baseline)")
        
        
        self._em_field_history = []
        self._k_enhancement_history = []
        self._collective_field_history = []
        
        
        # Track tryptophan count (changes with MT invasion)
        self.n_tryptophans = self.params.tryptophan.n_trp_baseline
        self._mt_invaded = False
        self._backbone_eta = 0.0  # Backbone Fröhlich condensation ratio

        self.camkii = CaMKIIModule()
        self.spine_plasticity = SpinePlasticityModule()
        logger.info("CaMKII and spine plasticity modules initialized")

         
        self.ddsc = DDSCSystem(DDSCParameters())

        # Cross-region photon emission/reception
        self.cross_region_enabled = HAS_CROSS_REGION
        if self.cross_region_enabled:
            self.photon_emitter = PhotonEmissionTracker(synapse_id=0, mt_alignment=0.7)
            self.photon_receiver = PhotonReceiver(synapse_id=0)
            self._external_trp_boost = 1.0
            self._external_k_boost = 1.0
            logger.info("Cross-region coupling ENABLED")
        
        # Particle-based dimer tracking with emergent entanglement
        self.dimer_particles = DimerParticleSystem(
            params=self.params,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
        logger.info("Dimer particle system initialized (emergent entanglement)")
        
        # History tracking
        self.history = {
            'time': [],
            'calcium_peak': [],
            'atp_mean': [],
            'j_coupling_max': [],
            'ion_pair_total': [],
            'dimer_ct_total': [],
            'coherence_mean': [],
            'pH_min': [],
            'dopamine_max': [],
            # Add to self.history dict:
            'molecular_memory': [],
            'spine_volume': [],
            'AMPAR_count': [],
            'synaptic_strength': [],
            'eligibility': [],
            'plasticity_gate': [],
            'structural_drive': [],
            'ddsc_current': [],
            'n_entangled_network': [],
            'f_entangled': [],
            'n_entanglement_bonds': [],
            'mean_singlet_prob': [],
        }

        # Add to self.history dict (around line ~95):
        if self.em_enabled:
            self.history['trp_em_field'] = []
            self.history['k_enhancement'] = []
            self.history['collective_field_kT'] = []
        
        
        # === FEEDBACK STATE TRACKING ===
        # For closed-loop dynamics
        self._k_agg_for_next_step = None  # Enhanced rate prepared by EM coupling
        self._previous_dimer_count = 0.0
        self._previous_coherence = 0.0
        self._collective_field_kT = 0.0

        # Dopamine as READ signal
        self._dopamine_read_threshold_nM = 50.0  # Phasic burst detection

        # CaMKII commitment (once set, locked)
        self._camkii_committed = False
        self._committed_memory_level = 0.0
        self._measurement_performed = False  # One-shot quantum measurement
        self._measured_eligibility = 0.0
        
        
        logger.info("Model 6 initialized successfully")
        
    def _initialize_subsystems(self):
        """Initialize all physics subsystems with proper connectivity"""
        
        # 1. Calcium channels
        # Place channels in a cluster at center (typical active zone)
        n_channels = self.params.calcium.n_channels_per_site
        center = self.grid_shape[0] // 2
        
        # Random positions near center
        # np.random.seed(42)  # Reproducibility for channel POSITIONS only
        # NOTE: Channel gating is stochastic with NO seed
        channel_positions = []
        for _ in range(n_channels):
            x = center + np.random.randint(-2, 3)
            y = center + np.random.randint(-2, 3)
            channel_positions.append((x, y))
        channel_positions = np.array(channel_positions)
        
        # 2. Template positions (protein scaffolds for dimer nucleation)
        # Place at channel sites where high calcium occurs
        template_positions = [(x, y) for x, y in channel_positions[:3]]
        
        # Initialize all subsystems
        self.calcium = CalciumSystem(
            self.grid_shape, self.dx, channel_positions, self.params
        )
        logger.info(f"Calcium system: {n_channels} channels at center")
        
        self.atp = ATPSystem(
            self.grid_shape, self.dx, self.params
        )
        logger.info("ATP system initialized")
        
        self.ca_phosphate = CaHPO4DimerSystem(
            self.grid_shape, self.dx, self.params, template_positions
        )
        logger.info(f"Calcium phosphate system: {len(template_positions)} template sites")
        
        # CHANGED: Use QuantumCoherenceSystem instead of PosnerSystem
        self.quantum = QuantumCoherenceSystem(
            self.grid_shape, self.params.quantum, isotope_P31_fraction=self.params.environment.fraction_P31,
            dopant=getattr(self.params.environment, 'dopant', None)
        )
        logger.info("Quantum coherence system initialized")
        
        self.pH = pHDynamics(
            self.grid_shape, self.dx, self.params
        )
        logger.info("pH dynamics initialized")
       
        # Dopamine system (optional)
        if HAS_DOPAMINE and self.params.dopamine is not None:
            self.dopamine = DopamineSystemAdapter(
                grid_shape=self.grid_shape,
                dx=self.dx,
                params=self.params,
                release_sites=[(center, center)]
            )
            logger.info("Dopamine system initialized")
        else:
            self.dopamine = None
            if self.params.dopamine is None:
                logger.info("Dopamine system disabled (params.dopamine = None)")
            else:
                logger.warning("Dopamine system not available")
        
    def step(self, dt: float, stimulus: Optional[Dict] = None):
        """
        Advance simulation by one timestep
        
        This is the orchestrator that connects all components.
        
        Args:
            dt: Time step (s)
            stimulus: Dictionary with control signals
                - 'voltage': Membrane voltage (V) - controls calcium channels
                - 'reward': Reward signal (boolean) - triggers dopamine
                - 'temperature': Optional temperature override (K)
        """
        stimulus = stimulus or {}
        
        # === STEP 1: CALCIUM DYNAMICS ===
        # Channels open based on voltage, calcium diffuses
        # NOTE: Channel gating is STOCHASTIC (no seed, varies run-to-run)

        # Spine → calcium feedback (when enabled):
        #   1. AMPAR gain: more AMPARs → larger EPSP → more VGCC activation
        #   2. Channel gain: larger spine surface area → more VGCCs → more Ca²⁺
        if getattr(self.params, 'spine_calcium_feedback', False):
            voltage = stimulus.get('voltage', -70e-3) if stimulus else -70e-3
            resting_v = -70e-3
            ampar_baseline = 80.0
            ampar_gain = self.spine_plasticity.AMPAR_count / ampar_baseline
            v_eff = resting_v + (voltage - resting_v) * ampar_gain
            stimulus = dict(stimulus) if stimulus else {}
            stimulus['voltage'] = v_eff

            spine_vol = self.spine_plasticity.spine_volume
            self.calcium.channel_gain = spine_vol ** 0.67  # surface area ~ vol^(2/3)
        else:
            self.calcium.channel_gain = 1.0

        # === PLATEAU POTENTIAL -> LOCAL MEMBRANE VOLTAGE (2026-07-18) ===
        # Until now `plateau_potential` was a BOOLEAN that only gated DDSC (~L656) and
        # had NO effect on voltage. But a dendritic plateau IS primarily a voltage
        # event, and its absence is why the pump never reached threshold in a live
        # trial (research log L·ETA-3: max r = 0.077 vs 1.0, with ca_open 2.7x short).
        #
        # NOT SUMMATION. `V_syn + V_plat` is physically impossible at the top of its
        # own range: max synaptic push (+30 mV) on a -20 mV plateau gives +10 mV, ABOVE
        # the AMPA/NMDA reversal (~0 mV) that generates the EPSP. The literature's form
        # is the steady-state conductance divider
        #     V = (g_L E_L + g_syn E_syn + g_plat E_plat) / (g_L + g_syn + g_plat)
        # whose voltage-space equivalent is used here. It is sublinear by construction,
        # HARD-BOUNDED by E_syn, and degrades EXACTLY to plain summation when
        # V_plat == V_rest — so it changes nothing when no plateau is present.
        # (Driving-force reduction: Tran-Van-Minh et al. 2015 Front Cell Neurosci;
        #  above-threshold saturation into duration not amplitude: Major/Polsky/Antic
        #  2008 J Neurophysiol; EPSPs still add during a plateau: Antic 2020.)
        if stimulus.get('plateau_potential', False):
            V_syn = stimulus.get('voltage', PLATEAU_V_REST)
            scale = ((PLATEAU_E_SYN - PLATEAU_VOLTAGE_V)
                     / (PLATEAU_E_SYN - PLATEAU_V_REST))
            stimulus = dict(stimulus)
            stimulus['voltage'] = (PLATEAU_VOLTAGE_V
                                   + (V_syn - PLATEAU_V_REST) * scale)

        self.calcium.step(dt, stimulus)
        ca_conc = self.calcium.get_concentration()

        current_ca_uM = float(np.max(ca_conc)) * 1e6
        if current_ca_uM > self._peak_calcium_uM:
            self._peak_calcium_uM = current_ca_uM
        
        # Calculate calcium influx for pH (needed later)
        ca_influx = np.gradient(ca_conc, axis=0) / dt  # Rough estimate
        
        # === STEP 2: ATP & J-COUPLING ===
        # ATP hydrolysis triggered by calcium, produces phosphate and J-coupling
        pH_field = self.pH.get_pH()
        self.atp.step(dt, ca_conc, pH_field)
        
        j_coupling = self.atp.get_j_coupling()
        atp_conc = self.atp.get_atp_concentration()
        phosphate = self.atp.get_phosphate_for_posner()

        # EXPERIMENTAL OVERRIDE: Use fixed phosphate if set
        if hasattr(self, '_override_phosphate'):
            phosphate = self._override_phosphate

        # DIAGNOSTIC: Check phosphate values
        # if np.random.rand() < 0.0001:
            # print(f"model6_core.step: phosphate = {np.mean(phosphate)*1e3:.2f} mM (mean), {np.max(phosphate)*1e3:.2f} mM (max)")
        
        # === STEP 3: pH DYNAMICS ===
        # Activity acidifies the synapse
        # Need ATP hydrolysis rate for pH
        atp_prev = getattr(self, '_atp_prev', atp_conc)
        atp_hydrolysis_rate = (atp_prev - atp_conc) / dt
        atp_hydrolysis_rate = np.maximum(atp_hydrolysis_rate, 0)
        self._atp_prev = atp_conc.copy()
        
        # Activity level from channels
        activity_level = self.calcium.channels.get_open_fraction()
        activity_field = np.ones(self.grid_shape) * activity_level
        
        self.pH.step(dt, atp_hydrolysis_rate, ca_influx, activity_field)
        pH_field = self.pH.get_pH()
        
        collective_field_kT = 0.0  # Default
        
        # === EM COUPLING: COMPLETE CLOSED-LOOP ARCHITECTURE ===
        if self.em_enabled:
            
            # --- PHASE 1: DIMER FORMATION (using previous step's enhanced k_agg) ---
            # This MUST happen first so we know how many dimers exist
            
            if self._k_agg_for_next_step is not None:
                k_agg_to_use = self._k_agg_for_next_step
            else:
                k_agg_to_use = self.ca_phosphate.dimerization.k_base
            
            # Form dimers with the prepared enhancement
            k_agg_original = self.ca_phosphate.dimerization.k_base
            self.ca_phosphate.dimerization.k_base = k_agg_to_use
            ca_consumed = self.ca_phosphate.step(dt, ca_conc, phosphate)

            # Apply consumption back to calcium system
            if ca_consumed is not None:
                self.calcium.apply_consumption(ca_consumed)

            # Apply phosphate consumption back to structural pool
            po4_consumed = self.ca_phosphate.get_phosphate_consumed()
            if po4_consumed is not None:
                self.atp.phosphate.phosphate_structural = np.maximum(
                    self.atp.phosphate.phosphate_structural - po4_consumed, 0.0
                )

            self.ca_phosphate.dimerization.k_base = k_agg_original  # Restore
            
            dimer_conc = self.ca_phosphate.get_dimer_concentration()
            
            # --- PHASE 2: QUANTUM COHERENCE OF DIMERS ---
            temperature = stimulus.get('temperature', self.params.environment.T)
            # Apply temperature scaling to chemical kinetics
            self.calcium.channels.apply_temperature_scaling(temperature)
            self.ca_phosphate.dimerization.apply_temperature_scaling(temperature)
            self.quantum.step(dt, dimer_conc, j_coupling, temperature)
            
            # --- PHASE 3: PARTICLE-BASED DIMER TRACKING WITH EMERGENT ENTANGLEMENT ---
            # Get dimer concentration from ca_phosphate system
            dimer_concentration = self.ca_phosphate.get_dimer_concentration()
            
            # Step particle system - this handles population, coherence, entanglement
            particle_metrics = self.dimer_particles.step(
                dt=dt,
                dimer_concentration=dimer_concentration,
                template_field=self.ca_phosphate.templates.template_field,
                calcium_field=ca_conc,
                j_coupling_field=j_coupling,
                collective_field_kT=self._collective_field_kT
            )

            # Calcium return from dissolved dimers (DDSC mechanism)
            n_dissolved = self.dimer_particles.get_dissolved_count()
            if n_dissolved > 0:
                ca_per_dimer_M = 6.0 / (6.022e23 * 1e-17)  # 6 Ca per dimer, 0.01 µm³ AZ volume
                ca_return = np.full_like(ca_conc, ca_per_dimer_M * n_dissolved)
                self.calcium.apply_return(ca_return)

            # Update dimer count from particle system (source of truth)
            self._previous_dimer_count = particle_metrics['n_dimers']

            # EMERGENT metrics replace prescribed ones
            n_dimers_total = particle_metrics['n_dimers']
            n_entangled_network = particle_metrics['largest_cluster']  # KEY: entangled network size
            mean_coherence = particle_metrics['mean_coherence']
            f_entangled = particle_metrics['f_entangled']
            
            # Store for tracking - use ENTANGLED count for network effects
            self._previous_dimer_count = n_entangled_network  # Changed from total to entangled
            self._previous_coherence = mean_coherence
            self.ca_phosphate.dimerization.set_mean_singlet_probability(
                self.get_mean_singlet_probability()
            )
            self._n_dimers_total = n_dimers_total
            self._n_entangled_network = n_entangled_network
            self._f_entangled = f_entangled
            
            # --- PHASE 4: LOCAL DIMER → TUBULIN COUPLING ---
            local_mod = self.local_dimer_coupling.calculate_local_modulation(
                n_dimers=n_dimers_total,  # Changed from n_dimers_single
                mean_coherence=mean_coherence
            )
            
            # --- PHASE 5: NETWORK INTEGRATION (multi-synapse) ---
            n_synapses = self.params.multi_synapse.n_synapses_default if self.params.multi_synapse_enabled else 1
            synapse_modulations = [local_mod['modulation_strength']] * n_synapses
            network_result = self.network_integrator.integrate_network(synapse_modulations)
            self._network_modulation = network_result['total_modulation']
            
            # --- PHASE 6: TRYPTOPHAN SUPERRADIANCE (receives dimer feedback) ---
            metabolic_uv_flux = self.params.metabolic_uv.photon_flux_baseline
            activity_level = self.calcium.channels.get_open_fraction()
            if activity_level > 0.1:
                metabolic_uv_flux *= self.params.metabolic_uv.flux_enhancement_active
            
            if self.params.metabolic_uv.external_uv_illumination:
                wavelength = self.params.metabolic_uv.external_uv_wavelength
                intensity = self.params.metabolic_uv.external_uv_intensity
                h, c = 6.626e-34, 3e8
                psd_area = 3.14159 * (350e-9)**2
                external_flux = (intensity * psd_area) / (h * c / wavelength)
                metabolic_uv_flux += external_flux
            
            ca_spike_active = (activity_level > 0.1)
            
            n_trp_effective = self.n_tryptophans
            if self.cross_region_enabled:
                n_trp_effective = int(self.n_tryptophans * self._external_trp_boost)
            
            trp_state = self.tryptophan.update(
                dt=dt,
                photon_flux=metabolic_uv_flux,
                n_tryptophans=n_trp_effective,
                ca_spike_active=ca_spike_active,
                network_modulation=self._network_modulation,  # REVERSE COUPLING
                backbone_eta=self._backbone_eta * getattr(self.spine_plasticity, 'E_invasion', 0.0)
            )
            # Cross-region: emit photons based on tryptophan state
            if self.cross_region_enabled:
                emission_result = self.photon_emitter.step(
                    dt=dt,
                    tryptophan_state=trp_state,
                    ca_spike_active=ca_spike_active
                )


            em_field_trp = trp_state['output']['em_field_time_averaged']
            self._collective_field_kT = trp_state['output']['collective_field_kT']

            # --- PHASE 7: PREPARE k_agg FOR NEXT TIMESTEP (forward coupling) ---
            k_agg_baseline = self.ca_phosphate.dimerization.k_base
            
            # Per-synapse metabolic drive for this spine's lattice segment. Same helper
            # and same inputs the backbone pump uses (multi_synapse_network), but NOT
            # aggregated over neighbours — the backbone sums coupled spines, this site
            # sees only its own. A single spine is therefore expected to be subcritical.
            #
            # B2 (2026-07-18): this replaced `collective_field_kT=` as the drive. The old
            # path converted a kT field energy into a pump rate via r_at_E_ref/kT_ref,
            # which is the calibration fiction B2 retires; there is no honest kT→power
            # conversion to keep. Step B made the same call for the backbone in June.
            p_met_W = compute_metabolic_power(
                getattr(self.spine_plasticity, 'E_invasion', 0.0),
                self.calcium.channels.get_open_fraction(),
                self.params.dendritic_backbone.p_active_max_W,
            )

            coupling_state = self.em_coupling.update(
                p_met_W=p_met_W,
                n_coherent_dimers=n_entangled_network,
                k_agg_baseline=k_agg_baseline,
                phosphate_fraction=np.mean(phosphate) / 0.001,
            )
            
            # Store for NEXT timestep (this is the feedback delay)
            # FORWARD COUPLING: Tryptophan EM field enhances dimer formation rate
            self._k_agg_for_next_step = coupling_state['output']['k_agg_enhanced']   
            
            # Track for diagnostics
            self._em_field_trp = em_field_trp
            self._k_enhancement = coupling_state['forward']['enhancement']
            self._em_field_history.append(em_field_trp)
            self._k_enhancement_history.append(coupling_state['forward']['enhancement'])
            self._collective_field_history.append(self._collective_field_kT)
            
            # --- PHASE 8: DOPAMINE UPDATE ---
            if self.dopamine is not None:
                reward = stimulus.get('reward', False)
                self.dopamine.step(dt, reward_signal=reward)
            

            # --- PHASE 9: ELIGIBILITY FROM PARTICLE SYSTEM (Agarwal 2023) ---
            ca_conc = self.calcium.get_concentration()  # Re-read after dissolution return
            calcium_uM = float(np.max(ca_conc)) * 1e6   # nanodomain PEAK — dimers form here; measurement trigger
            # The plasticity cascade (CaMKII + DARPP-32/PP1) sits at the PSD and integrates the DIFFUSE
            # PSD-distance calcium (area-averaged over the ~180 nm PSD disk), NOT the channel-mouth nanodomain
            # peak. Feeding np.max saturated CaMKII (~100s of uM >> K_half 1 uM) so dopamine's PP1 reinforcement
            # was inert (calcium-dominated commitment; F3-e). Grounded: PSD geometry (EM) + the model's own
            # Naraghi-Neher profile. See analytical_calcium_system.get_psd_averaged_concentration.
            camkii_calcium_uM = float(self.calcium.get_psd_averaged_concentration()) * 1e6

            # Eligibility IS the singlet state - no separate module needed
            if self.dimer_particles.dimers:
                mean_P_S = np.mean([d.singlet_probability for d in self.dimer_particles.dimers])
                n_entangled = sum(1 for d in self.dimer_particles.dimers if d.is_entangled)
                n_dimers = len(self.dimer_particles.dimers)
            else:
                mean_P_S = 0.25  # Thermal equilibrium (no dimers)
                n_entangled = 0
                n_dimers = 0

            # Eligibility = rescaled singlet probability [0, 1]
            # P_S = 0.25 → eligibility = 0, P_S = 1.0 → eligibility = 1
            eligibility = (mean_P_S - 0.25) / 0.75

            # --- PHASE 9: THREE-FACTOR GATE WITH PROBABILISTIC QUANTUM MEASUREMENT ---
            # Physics: Dopamine triggers ONE-SHOT measurement of quantum state.
            # Each dimer collapses stochastically: P(singlet) = P_S × field_fidelity
            # Measured eligibility = fraction of dimers that collapsed to singlet.
            # Q1 field modulates measurement fidelity (not a binary threshold).
            
            calcium_threshold_uM = 0.5
            calcium_elevated = (calcium_uM > calcium_threshold_uM)
            # F2 (Design B): the measurement is TRIGGERED by the spin-selective binding-melt event
            # (Fisher 2015; QDS Fisher & Radzihovsky 2018 — singlet is the reactive channel), NOT by
            # dopamine. The single-synapse dimer cloud is one coherent cluster; it binds-and-melts
            # with posner_binding.p_bind_melt(mean_P_S, mean_P_S, n_dimers, dt). Reward is decoupled
            # (it survives only as the separate learning signal). `dopamine.step` still runs upstream.
            from posner_binding import p_bind_melt
            binding_fires = (n_dimers > 0 and
                             np.random.random() < p_bind_melt(mean_P_S, mean_P_S, n_dimers, dt))

            # Store diagnostics (always updated)
            self._current_eligibility = eligibility
            self._mean_singlet_prob = mean_P_S
            self._n_entangled_dimers = n_entangled

            # Measurement happens ONCE when a coherent cluster first binds-melts with calcium
            # present. After measurement, quantum state is collapsed — no re-roll.
            if (binding_fires and calcium_elevated
                    and not self._camkii_committed
                    and not getattr(self, '_network_controlled', False)
                    and not self._measurement_performed):
                
                self._measurement_performed = True
                
                if self.dimer_particles.dimers:
                    # Q1 field modulates measurement fidelity
                    # High field (22 kT) → faithful readout (fidelity ~1.0)
                    # Low field (12 kT) → degraded readout (fidelity ~0.6)
                    # No field (0 kT) → no readout (fidelity = 0)
                    field_kT = self._collective_field_kT
                    field_fidelity = min(1.0, field_kT / 20.0)
                    
                    # Probabilistic collapse of each dimer
                    n_singlet = 0
                    for dimer in self.dimer_particles.dimers:
                        p_collapse_singlet = dimer.singlet_probability * field_fidelity
                        if np.random.random() < p_collapse_singlet:
                            n_singlet += 1
                    
                    measured_elig = n_singlet / len(self.dimer_particles.dimers)
                else:
                    measured_elig = 0.0
                
                self._measured_eligibility = measured_elig
                
                # Gate opens if majority collapsed to singlet
                plasticity_gate = measured_elig > 0.5
                
                if plasticity_gate:
                    self._camkii_committed = True
                    self._committed_memory_level = measured_elig
                
                self._plasticity_gate = plasticity_gate
            else:
                self._plasticity_gate = False

            # --- PHASE 10: CaMKII WITH BARRIER MODULATION ---
            # REVERSE COUPLING: Dimer field ALWAYS modulates CaMKII barrier
            # This is the Q2 → Classical pathway (runs every step regardless of gate)
            dimer_field_kT = coupling_state['reverse']['energy_modulation_kT']
            
            # --- F3 (CORRECTED 2026-08-09): dopamine REINFORCES CaMKII via DARPP-32/PP1 — it does NOT
            # bypass it. The eligibility trace = the coherent P_S tag (~100 s), decohering passively. At reward
            # the spin-selective binding-melt reads out a STILL-COHERENT tag (delivering the Ca²⁺ shower);
            # dopamine (D1→PKA→DARPP-32-Thr34) INHIBITS PP1 → CaMKII-pT286 persists (LTP); a dip / weak Ca
            # (PP2B) disinhibits PP1 → pT286 stripped (LTD). The reinforcement channel is `pp1_factor` scaling
            # CaMKII k_dephos; commitment STILL fires via CaMKII molecular_memory (the DDSC lock is HONORED,
            # not bypassed), and the LTP/LTD sign is EMERGENT from PP1. Coherence (quantum) or the fixed 0.3-2 s
            # window (classical baseline) gates whether dopamine can reinforce at all.
            # Grounding: docs/RESEARCH_DOPAMINE_CAMKII_REINFORCEMENT_2026-08-09.md (Yagishita 2014; Nakano 2010).
            # --- GROUNDED MEMORY ARCHITECTURE (2026-08-10, from the real CaMKII biology) ---
            # CaMKII activation is TRANSIENT (~1 min; Chang 2017) — NOT a bistable switch (that model is contested
            # / non-physiological, Frontiers 2025). The PERSISTENT memory is the CaMKII–GluN2B STRUCTURAL complex
            # (autonomous, protected from phosphatases; Cell Reports 2024). The seconds-scale commitment event is
            # DDSC — a DELAYED (10–100 s) CaMKII activation driven by internal-store Ca²⁺ (Jain 2024, Nature); here
            # that delayed Ca is delivered by the dopamine-gated spin-selective BINDING-MELT of the still-coherent
            # tag (F3-d). And DAPK1 makes CaMKII–GluN2B binding LTP-SPECIFIC: it BLOCKS binding unless dopamine→PKA
            # (PP1 inhibited) suppresses it — so a reward burst is REQUIRED to commit, at ANY calcium magnitude.
            # That is why Hebbian Ca alone does not commit (Yagishita), and it is what dissolves calcium-domination.
            pp1_factor = 1.0
            reward_thr34 = 0.0
            reward_gated = getattr(self, '_reward_gated_consolidation', False)
            if reward_gated:
                if not getattr(self, '_glun2b_memory_configured', False):
                    self.camkii.params.glun2b.glun2b_memory = True     # transient pT286 + persistent GluN2B latch
                    self._glun2b_memory_configured = True
                da_tonic = (self.params.dopamine.dopamine_tonic
                            if getattr(self.params, 'dopamine', None) is not None else 20e-9)
                da_override = getattr(self, '_da_signal', None)   # probe/network may inject DA(t)
                if da_override is not None:
                    da_level = float(da_override)
                elif self.dopamine is not None:
                    da_level = float(np.mean(self.dopamine.get_dopamine_concentration()))
                else:
                    da_level = da_tonic
                KD_D1 = 1e-6                                       # D1 low-affinity Kd ~1 µM (grounded)

                # READOUT: a phasic dopamine transient GATES/TIMES the spin-selective binding-melt (F3-d) of a
                # STILL-COHERENT tag, delivering the DDSC-analog commitment calcium. Magnitude rides on the tag's
                # remaining coherence (eligibility_weight(P_S)) — so a decohered tag (⁷Li, or aged past the window)
                # delivers less: the quantum lever is preserved. Scale is low-µM [GROUNDED — DDSC internal-store Ca
                # is ~µM, near CaMKII K_half=1 µM]; the exact peak is [MODELED].
                if not self._camkii_committed and getattr(self, '_measurement_gate_opened', False):
                    from reward_gating import (is_coherent, eligibility_weight,
                                               CLASSICAL_WINDOW_LO, CLASSICAL_WINDOW_HI)
                    mode = getattr(self, '_reward_gating_mode', 'quantum')  # 'quantum' | 'classical' (baseline)
                    t_since = self.time - getattr(self, '_measurement_time', self.time)
                    if mode == 'classical':
                        readable = CLASSICAL_WINDOW_LO <= t_since <= CLASSICAL_WINDOW_HI   # fixed short window
                        expired = t_since > CLASSICAL_WINDOW_HI
                    else:
                        readable = is_coherent(self._mean_singlet_prob)                    # coherent tag readable
                        expired = not readable
                    phasic = abs(da_level - da_tonic) > 0.05 * da_tonic   # a genuine reward transient
                    # Dopamine GATES THE ONSET of the melt; it does not set its duration. Once triggered, the
                    # readout is a CHEMICAL CASCADE that runs on the DDSC timescale (delayed Ca²⁺ event, 10–100 s;
                    # Jain 2024) — far longer than the sub-second phasic burst that timed it. Tying the Ca to the
                    # burst duration (the first wiring) starved the commitment whenever the burst was brief.
                    READOUT_DURATION_S = 20.0                          # [GROUNDED-in-band: DDSC 10–100 s]
                    if readable and phasic and not getattr(self, '_readout_fired', False):
                        CA_READOUT_UM = 3.0                            # [MODELED-in-grounded-band] DDSC-scale Ca
                        self._readout_fired = True                     # one-shot: the tag is read out once
                        self._readout_until = self.time + READOUT_DURATION_S
                        self._readout_ca_uM = CA_READOUT_UM * eligibility_weight(self._mean_singlet_prob)
                    if getattr(self, '_readout_fired', False) and self.time < getattr(self, '_readout_until', 0.0):
                        camkii_calcium_uM += self._readout_ca_uM       # the DDSC-analog commitment calcium
                    elif expired and not getattr(self, '_readout_fired', False):
                        self._measurement_gate_opened = False           # tag decohered / window closed

                # CONTINUOUS DARPP-32/PP1 cascade — the physiological signalling runs EVERY step (not only in the
                # reward window), because the DAPK1 gate reads PP1 continuously: at tonic DA, PP1 is active ⇒ DAPK1
                # active ⇒ binding blocked ⇒ Hebbian activity CANNOT commit. Grounding: Nakano 2010; Fernandez 2006.
                da_occ = da_level / (da_level + KD_D1)             # D1 receptor occupancy
                if getattr(self, '_darpp32', None) is None:
                    from darpp32_pp1_module import DARPP32PP1Module
                    self._darpp32 = DARPP32PP1Module(
                        da_tonic_occupancy=da_tonic / (da_tonic + KD_D1), ca_basal_uM=0.1)
                _d32 = self._darpp32.step(dt, da_occ, camkii_calcium_uM)
                pp1_factor = _d32['pp1_factor']
                reward_thr34 = _d32['thr34']      # the PKA/reward signal that suppresses DAPK1

            # CaMKII integrates the DIFFUSE PSD calcium (not the nanodomain peak) with dopamine-reinforced PP1.
            camkii_state = self.camkii.step(dt, camkii_calcium_uM, dimer_field_kT,
                                            pp1_factor=pp1_factor, reward_thr34=reward_thr34)

            # Commitment fires via CaMKII molecular_memory (DDSC lock) — now dopamine-reinforced. CONSUME the
            # token (D19): a measurement licensed one commitment; clearing the gate stops later re-commits.
            if (not self._camkii_committed
                    and getattr(self, '_measurement_gate_opened', False)
                    and camkii_state['molecular_memory'] > 0.5):
                self._camkii_committed = True
                self._commitment_time = getattr(self, '_measurement_time', self.time)
                self._committed_memory_level = camkii_state['molecular_memory']
                self._measurement_gate_opened = False

            # --- PHASE 11: SPINE PLASTICITY ---
            if self._camkii_committed:
                spine_state = self.spine_plasticity.step(
                    dt,
                    self._committed_memory_level,
                    calcium_uM,
                    quantum_field_kT=dimer_field_kT
                )
            else:
                spine_state = self.spine_plasticity.step(
                    dt, 0.0, calcium_uM,
                    quantum_field_kT=dimer_field_kT
                )
            
            # --- PHASE 12: TEMPLATE FEEDBACK (threshold behavior) ---
            spine_volume = self.spine_plasticity.spine_volume
            baseline_templates = 3
            
            # Threshold-based template increase
            if spine_volume > 1.5:  # Major growth (50%+)
                new_templates = 6  # Double the sites
            elif spine_volume > 1.25:  # Moderate growth (25%+)
                new_templates = 5
            else:
                new_templates = baseline_templates  # No change below threshold
            
            self.ca_phosphate.set_n_templates(new_templates)

            # --- PHASE 13: DDSC TRIGGERING ---
            plateau = stimulus.get('plateau_potential', False)
            if plateau:
                # DDSC requires BOTH eligibility AND sufficient combined field (>20 kT)
                FIELD_THRESHOLD_KT = 20.0
                if self._collective_field_kT >= FIELD_THRESHOLD_KT:
                    self.ddsc.check_trigger(self._current_eligibility, self.time)
                        
            # Update DDSC if triggered
            if self.ddsc.triggered:
                self.ddsc.integrate(self.time, dt)
        
        else:
            # === NON-EM PATH ===
            self._em_field_trp = 0.0
            self._k_enhancement = 1.0
            self._collective_field_kT = 0.0
            self._network_modulation = 0.0
            
            # Dimer formation (no EM enhancement)
            ca_consumed = self.ca_phosphate.step(dt, ca_conc, phosphate)
            if ca_consumed is not None:
                self.calcium.apply_consumption(ca_consumed)

            # Apply phosphate consumption back to structural pool
            po4_consumed = self.ca_phosphate.get_phosphate_consumed()
            if po4_consumed is not None:
                self.atp.phosphate.phosphate_structural = np.maximum(
                    self.atp.phosphate.phosphate_structural - po4_consumed, 0.0
                )

            dimer_conc = self.ca_phosphate.get_dimer_concentration()
            
            # Quantum coherence tracking
            temperature = stimulus.get('temperature', self.params.environment.T)
            self.quantum.step(dt, dimer_conc, j_coupling, temperature)

            # Particle-based dimer tracking
            dimer_concentration = self.ca_phosphate.get_dimer_concentration()
            particle_metrics = self.dimer_particles.step(
                dt=dt,
                dimer_concentration=dimer_concentration,
                template_field=self.ca_phosphate.templates.template_field,
                calcium_field=ca_conc,
                j_coupling_field=j_coupling,
                collective_field_kT=self._collective_field_kT
            )

            # Calcium return from dissolved dimers (DDSC mechanism)
            n_dissolved = self.dimer_particles.get_dissolved_count()
            if n_dissolved > 0:
                ca_per_dimer_M = 6.0 / (6.022e23 * 1e-17)  # 6 Ca per dimer, 0.01 µm³ AZ volume
                ca_return = np.full_like(ca_conc, ca_per_dimer_M * n_dissolved)
                self.calcium.apply_return(ca_return)

            # Update dimer count from particle system (source of truth)
            self._previous_dimer_count = particle_metrics['n_dimers']

            # Dopamine update
            if self.dopamine is not None:
                reward = stimulus.get('reward', False)
                self.dopamine.step(dt, reward_signal=reward)

            ca_conc = self.calcium.get_concentration()  # Re-read after dissolution return
            calcium_uM = float(np.max(ca_conc)) * 1e6

            # --- ELIGIBILITY FROM PARTICLE SYSTEM (Agarwal 2023) ---
            if self.dimer_particles.dimers:
                mean_P_S = np.mean([d.singlet_probability for d in self.dimer_particles.dimers])
                n_entangled = sum(1 for d in self.dimer_particles.dimers if d.is_entangled)
                n_dimers = len(self.dimer_particles.dimers)
            else:
                mean_P_S = 0.25
                n_entangled = 0
                n_dimers = 0

            eligibility = (mean_P_S - 0.25) / 0.75
            self._current_eligibility = eligibility
            self._mean_singlet_prob = mean_P_S
            self._n_entangled_dimers = n_entangled
            self._plasticity_gate = False  # No gate in non-EM path
            
            # Non-EM path: No DDSC triggering
            # Without EM coupling, dimers cannot influence CaMKII barrier
            # Classical plasticity follows a different (slower) pathway
            # DDSC remains untriggered - the quantum-classical handoff doesn't occur
            
            # Get structural drive from DDSC
            structural_drive = self.ddsc.get_structural_drive()
            
            # CaMKII (no quantum field in non-EM path)
            camkii_state = self.camkii.step(dt, calcium_uM, 0.0)
            
            # Spine plasticity driven by DDSC
            spine_state = self.spine_plasticity.step(
                dt, 
                structural_drive=structural_drive,
                calcium=calcium_uM
            )
        
        # Update time
        self.time += dt
        
    def receive_photons(self, delivered_packets: list, current_time: float):
        """Receive photons from waveguide and update modulation"""
        if not self.cross_region_enabled:
            return
        
        result = self.photon_receiver.receive_photons(delivered_packets, current_time)
        mod = self.photon_receiver.get_modulation_factors()
        self._external_trp_boost = mod['trp_n_effective_multiplier']
        self._external_k_boost = mod['k_agg_multiplier']
    
    def get_emitted_packets(self) -> list:
        """Get photon packets ready for waveguide transmission"""
        if not self.cross_region_enabled:
            return []
        return self.photon_emitter.get_pending_packets()
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return all experimental metrics for validation
        
        Returns metrics from each subsystem aligned with experimental measurements
        """
        # Get metrics from each subsystem
        ca_metrics = self.calcium.get_experimental_metrics()
        atp_metrics = self.atp.get_experimental_metrics()
        ca_phosphate_metrics = self.ca_phosphate.get_experimental_metrics()
        quantum_metrics = self.quantum.get_experimental_metrics()  # CHANGED from posner_metrics
        pH_metrics = self.pH.get_experimental_metrics()
        
        metrics = {
            # Calcium
            'calcium_peak_uM': ca_metrics['peak_ca_uM'],
            'calcium_mean_uM': ca_metrics.get('mean_ca_nM', 0) / 1000,  # Convert nM to μM
            
            # ATP
            'atp_mean_mM': atp_metrics['atp_mean_mM'],
            'j_coupling_max_Hz': atp_metrics['j_coupling_max_Hz'],
            'j_coupling_mean_Hz': atp_metrics['j_coupling_mean_Hz'],
            
            # Calcium Phosphate (the correct Ca₆(PO₄)₄ chemistry!)
            'ion_pair_peak_nM': ca_phosphate_metrics['ion_pair_peak_nM'],
            'ion_pair_mean_nM': ca_phosphate_metrics['ion_pair_mean_nM'],
            'dimer_peak_nM_ct': ca_phosphate_metrics['dimer_peak_nM'],
            'dimer_mean_nM_ct': ca_phosphate_metrics['dimer_mean_nM'],

            'trimer_peak_nM_ct': ca_phosphate_metrics.get('trimer_peak_nM', 0),  
            'trimer_mean_nM_ct': ca_phosphate_metrics.get('trimer_mean_nM', 0), 

            'dimer_trimer_ratio': ca_phosphate_metrics.get('dimer_trimer_ratio', 0),  # ADD THIS LINE

            # PNC metrics (NEW - from Turhan et al. 2024)
            'pnc_peak_nM': ca_phosphate_metrics.get('pnc_peak_nM', 0),
            'pnc_mean_nM': ca_phosphate_metrics.get('pnc_mean_nM', 0),
            'pnc_binding_fraction': ca_phosphate_metrics.get('pnc_binding_fraction', 0),
            'pnc_lifetime_mean_s': ca_phosphate_metrics.get('pnc_lifetime_mean_s', 0),
            'pnc_lifetime_max_s': ca_phosphate_metrics.get('pnc_lifetime_max_s', 0),
            
            # Quantum Coherence (CHANGED - simpler metrics)
            'coherence_dimer_mean': quantum_metrics['coherence_mean'],
            'coherence_std': quantum_metrics['coherence_std'],
            'coherence_peak': quantum_metrics['coherence_peak'],
            'T2_dimer_s': quantum_metrics['T2_dimer_s'],
            
            # pH
            'pH_min': pH_metrics['pH_min'],
            'pH_mean': pH_metrics['pH_mean'],

            # Plasticity metrics
            'molecular_memory': self.camkii.molecular_memory,
            'pT286_fraction': self.camkii.pT286,
            'spine_volume_fold': self.spine_plasticity.spine_volume,
            'AMPAR_count': self.spine_plasticity.AMPAR_count,
            'synaptic_strength': self.spine_plasticity.get_synaptic_strength(),
            'plasticity_phase': self.spine_plasticity.phase,
        }
        
        # EM Coupling metrics
        if self.em_enabled and len(self._em_field_history) > 0 and len(self._collective_field_history) > 0:
            metrics['trp_em_field_gv_m'] = np.max(self._em_field_history) / 1e9
            metrics['em_formation_enhancement'] = np.max(self._k_enhancement_history)
            # Use last 30% of simulation (after transients settled)
            steady_state_start = int(len(self._collective_field_history) * 0.7)
            if steady_state_start < len(self._collective_field_history):
                metrics['collective_field_kT'] = np.mean(self._collective_field_history[steady_state_start:])
            else:
                metrics['collective_field_kT'] = self._collective_field_kT  # Fallback to current
        else:
            metrics['trp_em_field_gv_m'] = 0.0
            metrics['em_formation_enhancement'] = 1.0
            metrics['collective_field_kT'] = 0.0  # ADD THIS LINE
        
        
        # Dopamine (if available)
        if self.dopamine is not None:
            da_metrics = self.dopamine.get_experimental_metrics()
            metrics['dopamine_max_nM'] = da_metrics['dopamine_max_nM']
            metrics['dopamine_mean_nM'] = da_metrics['dopamine_mean_nM']
            metrics['d2_occupancy_mean'] = da_metrics['d2_occupancy_mean']
        else:
            metrics['dopamine_max_nM'] = 0
            metrics['dopamine_mean_nM'] = 0
            metrics['d2_occupancy_mean'] = 0
        
        return metrics
    
    
    def get_eligibility(self) -> float:
        """
        Eligibility = rescaled mean singlet probability from dimer particles
        
        Maps P_S range [0.25, 1.0] to eligibility range [0, 1]
        This IS the temporal credit assignment trace (Agarwal 2023)
        """
        if hasattr(self, 'dimer_particles') and self.dimer_particles.dimers:
            mean_P_S = np.mean([d.singlet_probability for d in self.dimer_particles.dimers])
            return (mean_P_S - 0.25) / 0.75
        return 0.0

    def is_eligible(self) -> bool:
        """
        Synapse is eligible when mean P_S > 0.5 (Agarwal entanglement threshold)
        """
        if hasattr(self, 'dimer_particles') and self.dimer_particles.dimers:
            mean_P_S = np.mean([d.singlet_probability for d in self.dimer_particles.dimers])
            return mean_P_S > 0.5
        return False

    def get_mean_singlet_probability(self) -> float:
        """Direct access to mean P_S for reporting"""
        if hasattr(self, 'dimer_particles') and self.dimer_particles.dimers:
            return float(np.mean([d.singlet_probability for d in self.dimer_particles.dimers]))
        return 0.25  # Thermal equilibrium
    
    def set_locked_eligibility(self, value):
        """Lock eligibility to a fixed value for the duration of an experiment"""
        self._locked_eligibility = value
        
    def clear_locked_eligibility(self):
        """Clear locked eligibility, return to dynamic calculation"""
        self._locked_eligibility = None

    
    def _record_timestep(self):
        """Record current state to history"""
        self.history['time'].append(self.time)
        
        metrics = self.get_experimental_metrics()
        
        self.history['calcium_peak'].append(metrics['calcium_peak_uM'])
        self.history['atp_mean'].append(metrics['atp_mean_mM'])
        self.history['j_coupling_max'].append(metrics['j_coupling_max_Hz'])
        self.history['ion_pair_total'].append(metrics['ion_pair_peak_nM'])
        self.history['dimer_ct_total'].append(metrics['dimer_peak_nM_ct'])
        self.history['coherence_mean'].append(metrics['coherence_mean'])
        self.history['pH_min'].append(metrics['pH_min'])

        # Add to history recording section:
        self.history['molecular_memory'].append(self.camkii.molecular_memory)
        self.history['spine_volume'].append(self.spine_plasticity.spine_volume)
        self.history['AMPAR_count'].append(self.spine_plasticity.AMPAR_count)
        self.history['synaptic_strength'].append(self.spine_plasticity.get_synaptic_strength())

        self.history['eligibility'].append(self.get_eligibility())
        self.history['mean_singlet_prob'].append(self.get_mean_singlet_probability())
        self.history['plasticity_gate'].append(self._plasticity_gate)

        self.history['structural_drive'].append(self.ddsc.get_structural_drive())
        self.history['ddsc_current'].append(self.ddsc.current_ddsc)
        
        
        if self.dopamine is not None:
            self.history['dopamine_max'].append(metrics['dopamine_max_nM'])

        if self.em_enabled:
            self.history['trp_em_field'].append(self._em_field_trp)
            self.history['k_enhancement'].append(self._k_enhancement)
            self.history['collective_field_kT'].append(self._collective_field_kT)
    
    
    def set_microtubule_invasion(self, invaded: bool):
        """
        Set microtubule invasion state

        Updates tryptophan count when MT invades spine during plasticity

        Args:
            invaded: True if microtubules have invaded
        """
        self._mt_invaded = invaded
        if invaded:
            # MT invasion brings additional tryptophans
            self.n_tryptophans = self.params.get_total_tryptophans(mt_invaded=True)
        else:
            # Just baseline PSD lattice
            self.n_tryptophans = self.params.get_total_tryptophans(mt_invaded=False)

        if self.em_enabled:
            logger.info(f"MT invasion: {invaded}, n_trp: {self.n_tryptophans}")

    def set_backbone_condensation_eta(self, eta: float):
        """Backbone Fröhlich condensation ratio.

        Modulates f_coherent of invaded spine tryptophans via cooperative
        robustness (Babcock 2024). Only affects invaded spines — the
        continuous tubulin lattice from backbone into spine enables the
        backbone's condensed mode to stabilize local coherence.
        """
        self._backbone_eta = eta
    
    
    def _dopamine_above_read_threshold(self) -> bool:
        """
        Check if dopamine is above the READ threshold (phasic burst)
        
        This is the signal that "reads" the eligibility trace and
        triggers CaMKII commitment. NOT the same as D2 modulation
        of dimer formation.
        
        Returns:
            True if phasic dopamine burst detected (>50 nM)
        """
        if self.dopamine is None:
            return False
        
        # Get current dopamine concentration
        da_conc = self.dopamine.get_dopamine_concentration()
        da_max_nM = float(np.max(da_conc)) * 1e9  # Convert to nM
        
        return da_max_nM > self._dopamine_read_threshold_nM
    
    
    def get_em_coupling_state(self) -> Dict:
        """
        Get current EM coupling state (if enabled)
        
        Returns:
            dict with EM coupling diagnostics or empty dict if disabled
        """
        if not self.em_enabled:
            return {}
        
        return {
            'em_field_trp': self._em_field_trp,
            'k_enhancement': self._k_enhancement,
            'collective_field_kT': self._collective_field_kT,
            'n_tryptophans': self.n_tryptophans
        }
    
    
    def save_results(self, filepath: Optional[str] = None):
        """
        Save simulation results to HDF5 file
        
        Args:
            filepath: Path to output file (default: auto-generated in output_dir)
        """
        if filepath is None:
            if self.output_dir is None:
                raise ValueError("No output directory specified")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"model6_results_{timestamp}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['grid_shape'] = self.grid_shape
            f.attrs['dx_nm'] = self.dx * 1e9
            f.attrs['total_time_s'] = self.time
            
            # History
            hist_group = f.create_group('history')
            for key, values in self.history.items():
                hist_group.create_dataset(key, data=values)
             
            
            # Final state
            state_group = f.create_group('final_state')
            state_group.create_dataset('calcium', data=self.calcium.get_concentration())
            state_group.create_dataset('ion_pairs', data=self.ca_phosphate.get_ion_pair_concentration())
            state_group.create_dataset('dimers', data=self.ca_phosphate.get_dimer_concentration())
            state_group.create_dataset('coherence', data=self.quantum.get_coherence_field())

            # EM coupling state (if enabled)
            if self.em_enabled:
                em_group = f.create_group('em_coupling')
                em_group.attrs['enabled'] = True
                em_group.attrs['n_tryptophans'] = self.n_tryptophans
                em_group.attrs['trp_em_field_final'] = self._em_field_trp
                em_group.attrs['k_enhancement_final'] = self._k_enhancement
                em_group.attrs['collective_field_kT_final'] = self._collective_field_kT
            else:
                em_group = f.create_group('em_coupling')
                em_group.attrs['enabled'] = False
            
        logger.info(f"Results saved to {filepath}")


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL 6 CORE ORCHESTRATOR TEST")
    print("="*70)
    
    # Create model with default parameters
    model = Model6QuantumSynapse()
    
    print("\n### Quick Integration Test ###")
    print("Running 10 steps to verify all components connect properly...")
    
    # Test baseline
    for i in range(10):
        model.step(model.dt, stimulus={'voltage': -70e-3})  # Resting
    
    print("✓ Baseline steps completed")
    
    # Test with activity
    print("\nTesting with depolarization...")
    for i in range(10):
        model.step(model.dt, stimulus={'voltage': -10e-3})  # Depolarized
    
    print("✓ Activity steps completed")
    
    # Get metrics
    metrics = model.get_experimental_metrics()
    
    print("\nFinal Metrics:")
    print(f"  Calcium: {metrics['calcium_peak_uM']:.3f} μM")
    print(f"  ATP: {metrics['atp_mean_mM']:.2f} mM")
    print(f"  J-coupling: {metrics['j_coupling_max_Hz']:.1f} Hz")
    print(f"  Ion pairs: {metrics['ion_pair_peak_nM']:.2f} nM")
    print(f"  Ca₆(PO₄)₄ dimers: {metrics['dimer_peak_nM_ct']:.2f} nM")
    print(f"  Quantum coherence: {metrics['coherence_mean']:.3f}")
    print(f"  T2 time: {metrics['T2_dimer_s']:.1f} s")
    
    print("\n" + "="*70)
    print("✓ ORCHESTRATOR FUNCTIONING CORRECTLY")
    print("  - Calcium → ATP → Dimers → Quantum coherence")
    print("  - All subsystems connected")
    print("  - Proper architecture (no redundant formation)")
    print("="*70)