"""
Spine Plasticity Module - Structural Information Storage
=========================================================

ARCHITECTURE: Follows Model 6 module pattern
- SpinePlasticityParameters: Literature-derived constants
- SpinePlasticityModule: Main class with step() and get_experimental_metrics()

INPUTS:
    - molecular_memory: CaMKII pT286 × GluN2B (from CaMKII module)
    - calcium_uM: Local calcium concentration
    
OUTPUTS:
    - spine_volume: Relative to baseline (1.0 = baseline)
    - AMPAR_count: Number of surface AMPARs
    - synaptic_strength: AMPAR / baseline (functional output)
    - actin_stable: Stable actin pool (consolidation marker)
    - phase: transient / stabilizing / consolidated

THE CENTRAL CLAIM:
    Spine morphology changes ARE the information storage.
    Volume increase = more AMPAR slots = stronger synapse.

LITERATURE SOURCES (audited 2026-07-18 — three of the five below were wrong):
    - Matsuzaki et al. 2004 (Nature 429:761): spine enlargement, ~3.9x cap. VERIFIED.
    - Honkura et al. 2008 (Neuron 57:719): three actin pools; stable pool tau ~17 min;
      enlargement pool extruded unless CaMKII-confined. VERIFIED — and note it does NOT
      support the fixed 85/15 pool split it is cited for below.
    - Bosch et al. 2014 (Neuron): actin dynamics and phases. Attributed, not verified.
    - "Makino & Bhalla 2018" — NOT VERIFIED; appears in no REFERENCES section in this
      repo and could not be resolved. Affects only the cosmetic `phase` label.
    - "Choquet & Bhalla 2025 (EMBO J)" — DOES NOT EXIST. Conflation of Nowacka, Getz,
      Bessa-Neto & Choquet, Phil Trans R Soc B 2024;379:20230220 with Hayer & Bhalla,
      PLoS Comput Biol 2005;1(2):e20.
    - "Tang & bhalla 2017" — NOT VERIFIED; resolves to nothing in this repo. The
      volume-strength relation it is invoked for is Matsuzaki et al., Nat Neurosci
      2001;4:1086.

NOTE (2026-07-18): `quantum_field_kT` is accepted by step() and read in NO code path.
Measured bit-identical spine volume for kT in {0, 1, 5, 20, 100}. The
`ActinParameters` barrier fields below are vestigial — the quantum rate-enhancement
block that used them was added in 89e62ca and DELETED in 9e4e221; the declarations
were left behind. The same mechanism is still live in camkii_module.py. Do not read
this module's docstring as describing an implemented quantum coupling.

"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETERS — MIXED PROVENANCE. See the 2026-07-18 audit (research log D20/D21).
#
# This header used to read "ALL FROM LITERATURE". That was FALSE. The ACTIN block is
# genuinely literature-grounded (Honkura 2008, Matsuzaki 2004, Lee 2009, Star 2002,
# Lei 2017 — see model6-actin-invasion-driver §5, a real provenance table). The AMPAR
# block and the two durability rates are NOT, and three of the citations attached to
# them do not exist as written. Corrected in place below; each is now labelled.
# =============================================================================

@dataclass
class ActinParameters:
    """
    Actin dynamics parameters from Bosch et al. 2014
    
    Two pools:
        - Dynamic: Rapidly turning over (branched F-actin at spine tip)
        - Stable: Slowly turning over (core structure, resists depolymerization)
    
    Transition: Dynamic → Stable via cofilin inhibition (LIMK pathway)
    """
    # Pool baselines (relative units, normalized so total = 1.0 at rest)
    # MIS-CITED (corrected 2026-07-18). Honkura 2008 does NOT report a fixed 85/15
    # split — his finding is that the STABLE POOL'S SIZE SCALES WITH SPINE VOLUME,
    # i.e. explicitly not a constant fraction. A fixed split contradicts the paper it
    # cites. The literature does not supply a single number here, because the paper's
    # result is that there isn't one. Left at 0.85/0.15 pending a decision on whether
    # to make the split volume-dependent; do NOT cite Honkura for these two values.
    dynamic_pool_baseline: float = 0.85  # UNGROUNDED as a fixed fraction (see above)
    stable_pool_baseline: float = 0.15   # UNGROUNDED as a fixed fraction (see above)
    
    # Polymerization (Arp2/3 mediated, calcium/CaMKII dependent)
    # Bosch et al. 2014: maximal polymerization ~2x baseline in first 2 min
    k_polymerization_max: float = 0.1    # s⁻¹, rate constant for new actin
    K_calcium_poly: float = 1.0          # μM, calcium EC50 for polymerization
    K_camkii_poly: float = 0.3           # molecular_memory EC50
    
    # Depolymerization (cofilin-mediated)
    k_depolymerization: float = 0.05     # s⁻¹ baseline depolymerization
    
    # Stabilization (LIMK → cofilin phosphorylation)
    # CaMKII activates LIMK which inhibits cofilin → stabilizes actin
    k_stabilization_max: float = 0.02    # s⁻¹, max rate dynamic→stable
    K_memory_stab: float = 0.5           # molecular_memory threshold
    
    # Destabilization (slow return to baseline)
    k_destabilization: float = 0.001     # s⁻¹, stable→dynamic (very slow)

    # Enlargement + shaft-coupled supply + confinement (Honkura 2008): activity/CaM-driven,
    # extruded to the dendritic shaft unless confined by CaMKII; the MT-invasion driver.
    hill_calcium: float = 4.0        # CaM cooperativity (4 Ca2+ per CaM); matches camkii hill_calcium
    # f_CaM half-activation reuses K_calcium_poly (= 1.0 uM), defined above.
    tau_extrude: float = 180.0       # s; unconfined-enlargement extrusion to shaft. Honkura band
                                     # 2-15 min (>6 min retention threshold). Pre-registered in-band.
    S0: float = 0.12                 # resting G-actin monomer fraction (~12%, Star/Honkura)
    k_exchange: float = 1.0          # s^-1; spine<->shaft G-actin exchange (fast, ~seconds). Pre-registered.
    k_conf: float = 0.02             # s^-1; confinement establishment (~CaMKII 1-2 min). Pre-registered.
    k_unconf: float = 0.0005         # s^-1; confinement decay (slow; confinement persists). Pre-registered.
    E_ref: float = 1.87              # asymptotic enlargement under sustained maximal Ca (3000 s uncommitted run).
    invasion_threshold: float = 0.1  # "sufficient F-actin reorganization" onset; soft, pre-registered.
    # E_cap removed: the ceiling is now F_max, derived in _update_actin from
    # volume.max_enlargement_ratio and volume.actin_volume_scaling (the measured 3.9x cap).

    # Quantum coupling - barrier modulation
    # Arp2/3 conformational barrier ~8 kT, ~30% electrostatic
    barrier_polymerization_kT: float = 8.0
    barrier_electrostatic_fraction_poly: float = 0.30
    
    # Cofilin binding barrier ~5 kT, ~25% electrostatic  
    barrier_depolymerization_kT: float = 5.0
    barrier_electrostatic_fraction_depoly: float = 0.25
    
    # LIMK-cofilin barrier ~6 kT, ~35% electrostatic
    barrier_stabilization_kT: float = 6.0
    barrier_electrostatic_fraction_stab: float = 0.35
    
    # Coupling efficiency (same as CaMKII)
    quantum_coupling_efficiency: float = 0.1

    # Baseline polymerization (maintains homeostasis at rest)
    # At equilibrium: k_poly_baseline = k_depoly × dynamic_pool_baseline
    # 0.05 × 0.85 = 0.0425
    k_polymerization_baseline: float = 0.0425
    
    # Chemical Langevin noise (Bhalla lab modeling convention)
    # Noise amplitude scales as √(rate × dt) for molecular reactions
    stochastic: bool = False              # Enable Chemical Langevin noise

@dataclass  
class SpineVolumeParameters:
    """
    Spine volume dynamics from Matsuzaki et al. 2004, Makino & Bhalla 2018
    
    Key findings:
        - Maximum enlargement: ~390% for small spines, ~50% for large spines
        - Transient phase: τ ≈ 60s (rapid initial swelling)
        - Stabilization: requires sustained activity (>7 min)
        - Volume tracks actin content nearly linearly
    """
    # Volume limits
    baseline_volume: float = 1.0         # Normalized
    max_enlargement_ratio: float = 3.9   # 390% from Matsuzaki (small spines)
    min_volume_ratio: float = 0.5        # LTD can shrink spines
    
    # Volume-actin relationship
    # Bosch et al.: V ∝ (total actin content)^scaling
    actin_volume_scaling: float = 1.2    # Slightly supralinear
    
    # Phase thresholds (Makino & Bhalla 2018)
    stabilization_threshold: float = 2.0  # Volume > 2x baseline → stabilizing
    consolidation_time: float = 420.0     # 7 minutes for consolidation
    
    # Timescales
    tau_transient: float = 60.0          # seconds, Matsuzaki fast phase
    tau_volume_follow_actin: float = 5.0 # seconds, volume tracks actin

    # Thermal membrane fluctuations
    # From fluctuation-dissipation: σ_V/V ~ √(kT/κA) ~ 0.01-0.02 per √s
    stochastic: bool = True
    thermal_fluctuation_amplitude: float = 0.015  # Fractional noise per √second

@dataclass
class AMPARParameters:
    """
    AMPAR trafficking parameters.

    CITATIONS CORRECTED 2026-07-18 — the two named above did not exist:
      * "Bhalla 2014" -> Hayer A & Bhalla US, PLoS Comput Biol 2005;1(2):e20.
        NOTE the 80-receptor figure is a MODELING CHOICE in that paper ("we maintained
        the total number of internal plus synaptic membrane receptors at 80 molecules"),
        not a measurement. Citing it converts an unsourced number into a sourced
        modeling assumption — better provenance, not better evidence.
      * "Choquet & Bhalla 2025" -> did not exist; it conflates Nowacka, Getz,
        Bessa-Neto & Choquet, Phil Trans R Soc B 2024;379:20230220 with the Hayer &
        Bhalla paper above. Two real papers merged into one fictional one.

    The τ ~ 5 s lateral-diffusion and τ ~ 120 s exocytosis routes below ARE consistent
    with the measured literature (Patterson et al., PNAS 2010;107:15951 — AMPARs
    recruited within ~10 s of induction, exocytosis elevated ~1 min; Nowacka/Choquet
    2024 — early LTP expressed by lateral diffusion trapping, 70-90% pre-existing
    surface receptors). They are gated out by `ampar_onset_delay` — see its note.

    Three trafficking routes:
        1. Lateral diffusion (fast, τ ~ 5s)
        2. Exocytosis (slow, τ ~ 120s)  
        3. Endocytosis (removal, τ ~ 60s)
    
    Slot model: Spine volume determines available AMPAR slots
    """
    # Baseline counts
    # Hayer & Bhalla 2005 (a modeling choice there, not a measurement — see docstring).
    # Experimental range for calibration: Matsuzaki et al., Nat Neurosci 2001;4:1086 —
    # up to ~150 AMPARs in mushroom spines, sparse-to-zero in thin spines, strongly
    # volume-dependent. ~80 is a reasonable mid-range mushroom-spine baseline; a FIXED
    # baseline across a spine population is not.
    baseline_AMPAR: float = 80.0         # [MODELED, sourced to a model not a measurement]
    
    # Trafficking rates
    k_lateral_diffusion: float = 0.2     # s⁻¹ (τ ~ 5s), extrasynaptic→synaptic
    k_exocytosis: float = 0.008          # s⁻¹ (τ ~ 120s), intracellular→surface
    k_endocytosis: float = 0.017         # s⁻¹ (τ ~ 60s), removal
    
    # Slot model
    # MIS-CITED (corrected 2026-07-18): this relation is NOT in Hayer & Bhalla 2005 —
    # that paper's slot count is set by anchor proteins with no volume coupling. The
    # V ∝ AMPAR relation is real but its source is Matsuzaki et al., Nat Neurosci 2001.
    AMPAR_per_unit_volume: float = 80.0  # At baseline volume
    
    # Activity-dependent trafficking
    K_memory_exo: float = 0.3            # molecular_memory EC50 for exocytosis
    K_memory_lat: float = 0.2            # molecular_memory EC50 for lateral diff
    
    # Saturation
    max_AMPAR_ratio: float = 1.0         # Slots scale linearly with volume

    # Stochastic trafficking (Bhalla 2014: individual receptor events)
    stochastic: bool = False              # Enable Poisson trafficking events
@dataclass
class SpinePlasticityParameters:
    """Combined parameters for spine plasticity module"""
    actin: ActinParameters = field(default_factory=ActinParameters)
    volume: SpineVolumeParameters = field(default_factory=SpineVolumeParameters)
    ampar: AMPARParameters = field(default_factory=AMPARParameters)
    
    def __post_init__(self):
        logger.info("Spine plasticity parameters initialized with literature values")


# =============================================================================
# MAIN MODULE
# =============================================================================

class SpinePlasticityModule:
    """
    Spine structural plasticity - the physical substrate of information storage
    
    This module tracks:
        1. Actin dynamics (the cytoskeleton that shapes the spine)
        2. Spine volume (the morphological change)
        3. AMPAR count (the functional readout)
    
    Architecture follows Model 6 pattern:
        - Initialize with parameters
        - step(dt, molecular_memory, calcium_uM) advances state
        - get_experimental_metrics() returns measurable outputs
    """
    
    def __init__(self, params: Optional[SpinePlasticityParameters] = None):
        """
        Initialize spine plasticity module
        
        Args:
            params: SpinePlasticityParameters (uses defaults if None)
        """
        self.params = params or SpinePlasticityParameters()

        # Use numpy's modern Generator for better statistical properties
        self.rng = np.random.default_rng()
        
        # State variables
        self._initialize_state()
        
        # Phase tracking
        self.phase = 'baseline'  # baseline → transient → stabilizing → consolidated
        self.time_above_threshold = 0.0
        
        # History for experimental analysis
        self.history = {
            'time': [],
            'actin_dynamic': [],
            'actin_stable': [],
            'actin_total': [],
            'spine_volume': [],
            'AMPAR_count': [],
            'synaptic_strength': [],
            'phase': []
        }
        
        self.time = 0.0
        
        logger.info("SpinePlasticityModule initialized")
        logger.info(f"  Baseline AMPAR: {self.params.ampar.baseline_AMPAR}")
        logger.info(f"  Max enlargement: {self.params.volume.max_enlargement_ratio}x")
        
    def _initialize_state(self):
        """Set initial state to baseline values"""
        p = self.params
        
        # Actin pools (relative units)
        self.actin_dynamic = p.actin.dynamic_pool_baseline
        self.actin_stable = p.actin.stable_pool_baseline
        self.actin_enlargement = 0.0     # zero at rest (subcritical)
        self.E_invasion = 0.0            # invasion reach 0-1 -> r in the Q1->Q2 kernel (wired later)

        self.actin_monomer = p.actin.S0   # shaft-coupled G-actin supply
        self.confinement   = 0.0          # CaMKII confinement latch (0 = unconfined)

        # Derived quantities
        self.actin_total = self.actin_dynamic + self.actin_stable + self.actin_enlargement
        self.spine_volume = p.volume.baseline_volume
        self.AMPAR_count = p.ampar.baseline_AMPAR
        self.AMPAR_surface = p.ampar.baseline_AMPAR  # Surface pool
        self.AMPAR_reserve = p.ampar.baseline_AMPAR * 0.5  # Intracellular reserve
        
    def step(self, dt: float, structural_drive: float = 0.0, calcium: float = 0.1,
             molecular_memory: float = None, calcium_uM: float = None,
             quantum_field_kT: float = 0.0) -> Dict:
        """
        Advance spine plasticity by one timestep.
        
        HANDOFF MODEL: structural_drive (0-1) from DDSC sets the TARGET state.
        Actin/AMPAR approach targets over minutes (classical consolidation).
        
        Args:
            dt: Timestep in seconds
            structural_drive: 0-1 from integrated DDSC (the instructive signal)
            calcium: Current calcium concentration in μM
            molecular_memory: DEPRECATED - use structural_drive
            calcium_uM: DEPRECATED alias for calcium
        """
        # Handle deprecated parameters
        if calcium_uM is not None:
            calcium = calcium_uM

        self.time += dt
        
        # 1. Update actin dynamics (driven by structural_drive)
        self._update_actin(dt, structural_drive, calcium, quantum_field_kT)
        
        # 2. Update spine volume (follows actin)
        self._update_volume(dt)
        
        # 3. Update AMPAR trafficking (driven by structural_drive)
        self._update_AMPAR(dt, structural_drive, quantum_field_kT)
        
        # 4. Update phase
        self._update_phase(dt)
        
        # 5. Record history
        self._record_history()
        
        return self.get_state()
        
    def _update_actin(self, dt: float, structural_drive: float, calcium: float,
                      quantum_field_kT: float = 0.0):
        """
        Shaft-reservoir + confinement-gated-extrusion actin model.
        """
        a = self.params.actin
        vol = self.params.volume

        # CaM activation by the calcium transient (uM, Hill n=4).
        ca = max(0.0, calcium)
        f_CaM = ca**a.hill_calcium / (a.K_calcium_poly**a.hill_calcium + ca**a.hill_calcium)

        s = float(np.clip(structural_drive, 0.0, 1.0))  # commitment (DDSC)

        # Confinement latch: commitment establishes a PERSISTENT lock (CaMKII cross-linking).
        d_conf = a.k_conf * s * (1.0 - self.confinement) - a.k_unconf * self.confinement
        self.confinement = float(np.clip(self.confinement + d_conf * dt, 0.0, 1.0))
        conf = self.confinement

        # Structural ceiling: formation stops as F-actin nears the spine's measured max size.
        # F_max (actin units) derived from the volume cap so the two ceilings agree.
        F = self.actin_dynamic + self.actin_enlargement + self.actin_stable
        F_max = vol.max_enlargement_ratio ** (1.0 / vol.actin_volume_scaling)   # 3.9 ** (1/1.2) ~= 3.17
        room = max(0.0, 1.0 - F / F_max)

        # Formation: Ca/CaM-driven, monomer-supply-limited (S/S0) and room-limited.
        formation = a.k_polymerization_max * f_CaM * (self.actin_monomer / a.S0) * room

        k_extrude = 1.0 / a.tau_extrude
        extrusion = k_extrude * (1.0 - conf) * self.actin_enlargement      # CLEARING: unconfined -> shaft
        retention = a.k_stabilization_max * conf * self.actin_enlargement  # confined -> stable

        # Dynamic pool: homeostatic baseline treadmilling (self-maintaining; outside the S economy).
        d_dynamic = a.k_polymerization_baseline - a.k_depolymerization * self.actin_dynamic

        d_enlarge = formation - extrusion - retention
        d_stable  = retention - a.k_destabilization * self.actin_stable

        # Monomer pool: shaft buffers it toward S0 (fast exchange); formation draws it down.
        # Extruded enlargement and destabilized stable exit to the large shaft (not tracked here).
        d_monomer = a.k_exchange * (a.S0 - self.actin_monomer) - formation

        # Euler integration. No hard clip: extrusion clears unconfined enlargement and the room
        # term stops formation at F_max.
        self.actin_dynamic     = max(0.0, self.actin_dynamic + d_dynamic * dt)
        self.actin_enlargement = max(0.0, self.actin_enlargement + d_enlarge * dt)
        self.actin_stable      = max(0.0, self.actin_stable + d_stable * dt)
        self.actin_monomer     = max(0.0, self.actin_monomer + d_monomer * dt)
        self.actin_total       = self.actin_dynamic + self.actin_enlargement + self.actin_stable

        # Invasion reach (0-1): "sufficient F-actin reorganization" (Honkura threshold).
        denom = max(1e-6, a.E_ref - a.invasion_threshold)
        self.E_invasion = float(np.clip((self.actin_enlargement - a.invasion_threshold) / denom, 0.0, 1.0))
        
    def _update_volume(self, dt: float):
        """
        Update spine volume based on actin content with thermal fluctuations
        
        Matsuzaki et al. 2004: Volume tracks actin with τ ~ 5-60s
        
        Stochastic: Thermal membrane fluctuations from fluctuation-dissipation theorem
            σ_V/V ~ √(kT / κA) where κ is membrane bending modulus, A is area
            For dendritic spines: ~1-2% fluctuation amplitude per √second
        """
        p = self.params.volume
        
        # Target volume from actin content
        baseline_actin = p.baseline_volume
        target_volume = (self.actin_total / baseline_actin) ** p.actin_volume_scaling
        
        # Clamp to physiological limits
        target_volume = np.clip(target_volume, p.min_volume_ratio, p.max_enlargement_ratio)
        
        # Volume follows actin with timescale
        tau = p.tau_volume_follow_actin
        dV = (target_volume - self.spine_volume) / tau * dt
        
        # Thermal membrane fluctuations
        if p.stochastic:
            # Fluctuation-dissipation: noise amplitude ~ √(kT/κ) 
            # Empirically ~1.5% of volume per √second for spine membranes
            thermal_noise = (p.thermal_fluctuation_amplitude * self.spine_volume 
                            * np.sqrt(dt) * self.rng.standard_normal())
            dV += thermal_noise
        
        # Update with bounds
        self.spine_volume = np.clip(self.spine_volume + dV, 
                                    p.min_volume_ratio, p.max_enlargement_ratio)
        
    def _update_AMPAR(self, dt: float, structural_drive: float,
                      quantum_field_kT: float = 0.0):
        """
        Update AMPAR count based on DDSC-derived structural drive.
        
        BIOLOGICALLY REALISTIC TIMING (Royal Society, Bhalla 2014):
        - PSD unchanged at 5-30 min despite maximal potentiation
        - PSD increases visible at 2 hours
        - AMPAR trafficking has ~30 min onset delay, ~1 hour tau
        
        Args:
            structural_drive: 0-1 from DDSC integration
            quantum_field_kT: DEPRECATED
        """
        p = self.params.ampar
        
        # Store structural_drive for projected_strength calculation
        self._current_structural_drive = structural_drive
        
        # !! CONTRADICTED BY THE LITERATURE (flagged 2026-07-18, NOT yet changed) !!
        # The stated rationale ("AMPAR changes don't begin until ~30 min after
        # structural drive") cited "Royal Society, Bhalla 2014", which is not a
        # citation and resolves to nothing in this repo. The measured onset is
        # SECONDS, not 30 minutes:
        #   Patterson et al., PNAS 2010;107:15951 — AMPARs recruited to spines within
        #     ~10 s of LTP induction, concurrent with the volume increase; exocytosis
        #     rate up ~5x, decaying to baseline within ~1 min.
        #   Nowacka/Getz/Bessa-Neto/Choquet, Phil Trans R Soc B 2024;379:20230220 —
        #     early LTP expressed by lateral diffusion trapping (immediate); 70-90%
        #     of receptors are pre-existing surface pool.
        # This gate sits ON TOP of the module's own correctly-grounded rate constants
        # (k_lateral_diffusion tau~5 s, k_exocytosis tau~120 s) and suppresses them,
        # which is why AMPAR has never changed in ANY experiment this repo has run
        # (measured duty cycle => 159-809 trials to reach onset, against n_trials=5).
        # LEFT IN PLACE pending Sarah's call: removing it changes every result, and
        # the AMPAR->voltage consumer is separately gated off by spine_calcium_feedback.
        ampar_onset_delay = 1800.0  # UNGROUNDED and literature-contradicted; see above
        
        if self.time < ampar_onset_delay:
            # No AMPAR changes yet - stay at baseline
            # (Spine volume can change, but AMPAR count doesn't)
            return
        
        # After onset delay: slow AMPAR accumulation.
        # "Literature: full expression takes ~2 hours" had no reference. Defensible as
        # the LATE/maintenance phase (Choquet 2024's exocytosis-for-maintenance limb;
        # Fukazawa et al., Neuron 2003;38:447 on F-actin and late LTP) but it is not a
        # measured tau from any source found. [MODELED, plausible, unsourced]
        tau_ampar = 3600.0
        
        # Target AMPAR scales with structural drive
        # "(literature-based)" was unsupported — no 35% figure appears in any skill,
        # research log, or handoff in this repo. Audit 2026-07-18 judges it ~3x TOO LOW:
        # single-spine uncaging LTP gives a sustained volume plateau of ~+120%
        # (Matsuzaki 2004), and AMPAR number scales with spine head volume (Matsuzaki
        # et al., Nat Neurosci 2001). Left unchanged pending the same call as the
        # onset delay above. [UNGROUNDED]
        max_ampar_increase = 0.35
        target_ampar = p.baseline_AMPAR * (1.0 + structural_drive * max_ampar_increase)
        
        # Exponential approach to target
        approach_rate = (target_ampar - self.AMPAR_count) / tau_ampar
        self.AMPAR_count += approach_rate * dt
        self.AMPAR_surface = self.AMPAR_count
        
        # Bounds
        self.AMPAR_count = np.clip(self.AMPAR_count, 
                                    p.baseline_AMPAR * 0.5,
                                    p.baseline_AMPAR * 1.5)
        self.AMPAR_surface = self.AMPAR_count
        
    def _update_phase(self, dt: float):
        """
        Track plasticity phase transitions
        
        Phases (Makino & Bhalla 2018):
            - baseline: V < 1.2, no persistent activity
            - transient: V > 1.2, < 7 min since induction
            - stabilizing: V > 2.0, sustained > 7 min
            - consolidated: stable pool dominates, irreversible
        """
        p = self.params.volume
        
        # Track time above threshold
        if self.spine_volume > p.stabilization_threshold:
            self.time_above_threshold += dt
        else:
            # Reset if volume drops
            self.time_above_threshold = max(0, self.time_above_threshold - dt * 0.5)
        
        # Determine phase
        stable_fraction = self.actin_stable / max(0.1, self.actin_total)
        
        if self.spine_volume < 1.2:
            self.phase = 'baseline'
        elif self.time_above_threshold < p.consolidation_time:
            self.phase = 'transient'
        elif stable_fraction < 0.6:
            self.phase = 'stabilizing'
        else:
            self.phase = 'consolidated'
            
    def _record_history(self):
        """Record current state to history"""
        self.history['time'].append(self.time)
        self.history['actin_dynamic'].append(self.actin_dynamic)
        self.history['actin_stable'].append(self.actin_stable)
        self.history['actin_total'].append(self.actin_total)
        self.history['spine_volume'].append(self.spine_volume)
        self.history['AMPAR_count'].append(self.AMPAR_count)
        self.history['synaptic_strength'].append(self.get_synaptic_strength())
        self.history['phase'].append(self.phase)
        
    def get_state(self) -> Dict:
        """Get current state as dictionary"""
        return {
            'actin_enlargement': self.actin_enlargement,
            'E_invasion': self.E_invasion,
            'actin_monomer': self.actin_monomer,
            'confinement': self.confinement,
            'actin_dynamic': self.actin_dynamic,
            'actin_stable': self.actin_stable,
            'actin_total': self.actin_total,
            'spine_volume': self.spine_volume,
            'AMPAR_count': self.AMPAR_count,
            'AMPAR_surface': self.AMPAR_surface,
            'AMPAR_reserve': self.AMPAR_reserve,
            'synaptic_strength': self.get_synaptic_strength(),
            'phase': self.phase,
            'time_above_threshold': self.time_above_threshold
        }
        
    def get_synaptic_strength(self) -> float:
        """
        Calculate synaptic strength (the functional output)
        
        Defined as AMPAR_count / baseline
        This is what electrophysiology measures (EPSP amplitude)
        """
        return self.AMPAR_count / self.params.ampar.baseline_AMPAR
    
    
    def get_projected_strength(self) -> float:
        """
        Project final synaptic strength from current structural drive.
        
        This allows validation of quantum gating without running 2-hour
        simulations. Based on literature showing ~35% max LTP increase.
        
        Returns:
            Projected final strength (1.0 = baseline, 1.35 = max LTP)
        """
        structural_drive = getattr(self, '_current_structural_drive', 0.0)
        # Duplicate of max_ampar_increase above — same UNGROUNDED 35%, hard-coded in a
        # second place. If that one is revised, revise this too. [UNGROUNDED]
        max_strength_increase = 0.35
        
        projected = 1.0 + structural_drive * max_strength_increase
        return projected
    
    
    def is_potentiated(self) -> bool:
        """Check if spine is in potentiated state (LTP)"""
        return self.spine_volume > 1.2 and self.get_synaptic_strength() > 1.2
    
    def get_experimental_metrics(self) -> Dict:
        """
        Get metrics that can be experimentally measured
        
        Returns:
            Dict with measurable quantities:
            - spine_volume_fold: Volume relative to baseline (2-photon microscopy)
            - synaptic_strength: EPSP amplitude / baseline (electrophysiology)
            - AMPAR_density: AMPARs per unit area (immunogold EM)
            - actin_stable_fraction: Stable / total actin (FRAP)
            - phase: Current plasticity phase
            - time_to_potentiation: Time until spine_volume > 1.5
        """
        metrics = {
            # Direct measurements
            'spine_volume_fold': self.spine_volume,
            'synaptic_strength': self.get_synaptic_strength(),
            'AMPAR_count': self.AMPAR_count,
            'AMPAR_density': self.AMPAR_count / self.spine_volume if self.spine_volume > 0 else 0,
            
            # Actin metrics (measurable by FRAP, phalloidin staining)
            'actin_total_fold': self.actin_total / (self.params.actin.dynamic_pool_baseline + 
                                                     self.params.actin.stable_pool_baseline),
            'actin_stable_fraction': self.actin_stable / max(0.1, self.actin_total),
            'actin_dynamic_fraction': self.actin_dynamic / max(0.1, self.actin_total),
            
            # Phase
            'phase': self.phase,
            'is_potentiated': self.is_potentiated(),
            
            # Kinetics (from history if available)
            'time_to_half_max_volume': self._find_time_to_threshold(
                self.history['spine_volume'], 
                0.5 * (self.params.volume.max_enlargement_ratio - 1) + 1
            ) if self.history['time'] else None,
            
            'time_to_potentiation': self._find_time_to_threshold(
                self.history['spine_volume'], 1.5
            ) if self.history['time'] else None,
        }
        
        # Add peak values from history
        if self.history['time']:
            metrics['peak_spine_volume'] = max(self.history['spine_volume'])
            metrics['peak_AMPAR_count'] = max(self.history['AMPAR_count'])
            metrics['peak_synaptic_strength'] = max(self.history['synaptic_strength'])
        
        return metrics
    
    def _find_time_to_threshold(self, values: list, threshold: float) -> Optional[float]:
        """Find first time when value exceeds threshold"""
        for i, v in enumerate(values):
            if v >= threshold:
                return self.history['time'][i]
        return None
    
    def reset(self):
        """Reset to baseline state"""
        self._initialize_state()
        self.phase = 'baseline'
        self.time_above_threshold = 0.0
        self.time = 0.0
        self.history = {k: [] for k in self.history}
        logger.info("SpinePlasticityModule reset to baseline")


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SPINE PLASTICITY MODULE - VALIDATION")
    print("="*70)
    
    # Initialize module
    module = SpinePlasticityModule()
    
    # Simulation parameters
    dt = 0.1  # 100 ms timestep
    duration = 600.0  # 10 minutes
    
    # Protocol: 60s of high activity, then monitor
    print("\nRunning protocol: 60s activity → 540s recovery")
    print("-"*70)
    
    t = 0.0
    while t < duration:
        # Activity pattern
        if t < 60.0:
            # High activity: elevated calcium and CaMKII
            calcium = 5.0  # μM
            molecular_memory = 0.8  # High CaMKII activation
        else:
            # Baseline: low calcium, decaying memory
            calcium = 0.1  # μM
            # Memory decays slowly
            molecular_memory = 0.8 * np.exp(-(t - 60.0) / 300.0)
        
        # Step
        state = module.step(dt, molecular_memory, calcium)
        
        # Print progress
        if t % 60.0 < dt:
            print(f"  t={t:5.0f}s: V={state['spine_volume']:.2f}, "
                  f"AMPAR={state['AMPAR_count']:.0f}, "
                  f"strength={state['synaptic_strength']:.2f}, "
                  f"phase={state['phase']}")
        
        t += dt
    
    # Get final metrics
    print("\n" + "="*70)
    print("EXPERIMENTAL METRICS")
    print("="*70)
    
    metrics = module.get_experimental_metrics()
    
    print(f"\nSpine morphology:")
    print(f"  Volume: {metrics['spine_volume_fold']:.2f}x baseline")
    print(f"  Peak volume: {metrics.get('peak_spine_volume', 0):.2f}x")
    print(f"  Time to 1.5x: {metrics['time_to_potentiation']:.1f}s" 
          if metrics['time_to_potentiation'] else "  Time to 1.5x: not reached")
    
    print(f"\nSynaptic strength:")
    print(f"  Current: {metrics['synaptic_strength']:.2f}x baseline")
    print(f"  Peak: {metrics.get('peak_synaptic_strength', 0):.2f}x")
    print(f"  AMPAR count: {metrics['AMPAR_count']:.0f}")
    
    print(f"\nActin dynamics:")
    print(f"  Total actin: {metrics['actin_total_fold']:.2f}x baseline")
    print(f"  Stable fraction: {metrics['actin_stable_fraction']:.1%}")
    print(f"  Dynamic fraction: {metrics['actin_dynamic_fraction']:.1%}")
    
    print(f"\nPhase: {metrics['phase']}")
    print(f"Potentiated: {metrics['is_potentiated']}")
    
    print("\n" + "="*70)
    print("✓ SPINE PLASTICITY MODULE VALIDATED")
    print("="*70)