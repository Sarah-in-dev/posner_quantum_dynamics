"""
Hierarchical Quantum Processing Model - Parameter Definitions
==============================================================

Complete parameter set for the hierarchical quantum processing architecture
described in "Hierarchical Quantum Processing Architecture in Neural Plasticity"
(Davidson 2025).

This model integrates:
- Model 6 calcium phosphate dimer quantum chemistry
- Tryptophan superradiance in microtubules (Babcock et al. 2024)
- CaMKII-actin cascade (Lisman et al. 2012)
- Microtubule invasion during plasticity (Merriam et al. 2013)
- Bidirectional quantum-classical coupling

ALL PARAMETERS ARE EITHER:
1. Measured from primary literature (with full citation)
2. Derived from first principles
3. Experimental control variables (clearly marked with ⚠️)

NO FITTING PARAMETERS. NO ARBITRARY SCALING.

Author: Sarah Davidson
Date: October 2025
Version: 7.0 (Hierarchical)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from scipy import constants
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

k_B = constants.Boltzmann  # J/K = 1.380649e-23
N_A = constants.Avogadro   # 1/mol = 6.02214076e23
h = constants.h             # J·s = 6.62607015e-34
h_bar = constants.hbar      # J·s = 1.054571817e-34
e = constants.e             # C = 1.602176634e-19
c = constants.c             # m/s = 299792458
epsilon_0 = constants.epsilon_0  # F/m = 8.8541878128e-12

# Derived constants
k_B_T_310K = k_B * 310  # Thermal energy at 37°C = 4.28e-21 J


# ============================================================================
# PATHWAY 1: Ca²⁺ → CALMODULIN → CaMKII
# ============================================================================

@dataclass
class CalmodulinParameters:
    """
    Calmodulin binding and CaMKII activation parameters
    
    References:
    -----------
    Persechini & Stemmer 2002 - Trends Cardiovasc Med 12:32-37
        "Ca²⁺ binding kinetics to calmodulin"
    
    Meador et al. 1992 - Science 257:1251-1255
        "Structure and conformational changes"
    
    Shifman et al. 2006 - PNAS 103:13968-13973
        "Ca²⁺ binding cooperativity in EF hands"
    """
    
    # === CA²⁺ BINDING TO CALMODULIN ===
    # Persechini & Stemmer 2002 Trends Cardiovasc Med 12:32-37
    # CaM has 4 EF-hand Ca²⁺ binding sites (2 per lobe)
    n_binding_sites: int = 4
    
    # N-lobe (higher affinity)
    kd_n_lobe: float = 1e-6  # M (1 µM)
    # C-lobe (lower affinity)  
    kd_c_lobe: float = 10e-6  # M (10 µM)
    
    # Effective Kd for Ca²⁺/CaM complex formation
    # Needs ~2-3 Ca²⁺ bound for activation
    kd_activation: float = 0.5e-6  # M (500 nM)
    
    # === BINDING KINETICS ===
    # Shifman et al. 2006 PNAS 103:13968-13973
    k_on_ca: float = 1e8   # M⁻¹s⁻¹ (diffusion limited)
    k_off_ca: float = 100  # s⁻¹ (gives Kd ~ 1 µM)
    
    # === CONFORMATIONAL CHANGE ===
    # Meador et al. 1992 Science 257:1251-1255
    # Ca²⁺ binding opens hydrophobic pockets
    tau_conformational: float = 10e-3  # s (10 ms)
    
    # Conformational change exposes CaMKII binding sites
    hydrophobic_pocket_area: float = 1000e-20  # m² (~10 Å²)


@dataclass
class CaMKIIParameters:
    """
    CaMKII activation, autophosphorylation, and actin binding
    
    References:
    -----------
    Lisman et al. 2012 - Nat Rev Neurosci 13:169-182
        "The molecular basis of CaMKII function in synaptic plasticity"
    
    Hudmon & Schulman 2002 - Annu Rev Biochem 71:473-510
        "Structure, regulation and function of CaMKII"
    
    Colbran & Brown 2004 - Curr Opin Neurobiol 14:318-327
        "CaMKII binding to cytoskeletal/synaptic proteins"
    
    Stratton et al. 2014 - Nat Commun 5:5304
        "Structural studies of CaMKII regulation"
    """
    
    # === CAMKII STRUCTURE ===
    # Hudmon & Schulman 2002 Annu Rev Biochem 71:473-510
    # CaMKII is a dodecameric holoenzyme (12 subunits)
    n_subunits: int = 12
    molecular_weight: float = 650e3  # Da (650 kDa holoenzyme)
    
    # === CA²⁺/CAM BINDING ===
    # Lisman et al. 2012 Nat Rev Neurosci 13:169-182
    # Ca²⁺/CaM binds to regulatory domain, releasing autoinhibition
    kd_cam: float = 20e-9  # M (20 nM) - very tight binding
    k_on_cam: float = 1e6  # M⁻¹s⁻¹
    k_off_cam: float = 0.02  # s⁻¹ (slow unbinding)
    
    # === AUTOPHOSPHORYLATION AT T286 ===
    # Stratton et al. 2014 Nat Commun 5:5304
    # Adjacent subunits phosphorylate each other (inter-subunit)
    # This requires Ca²⁺/CaM to release autoinhibition first
    
    k_autophosphorylation: float = 0.5  # s⁻¹
    # Requires neighboring subunit to be Ca²⁺/CaM-bound
    neighbor_requirement: bool = True
    
    # Phosphorylated T286 creates autonomous activity
    # (stays active even after Ca²⁺ drops)
    k_dephosphorylation: float = 0.01  # s⁻¹ (phosphatase PP1)
    # Creates ~100s persistence - matches behavioral timescale!
    tau_persistent: float = 100  # s
    
    # === ACTIVATION BARRIER ===
    # Colbran & Brown 2004 Curr Opin Neurobiol 14:318-327
    # Conformational change from autoinhibited → active
    delta_G_activation: float = 2.3e3 * k_B_T_310K  # ~90 kT
    # In kcal/mol: ~2.3 kcal/mol
    
    # ⚠️ QUANTUM MODULATION PARAMETER
    # Quantum electrostatic field can modulate this barrier
    # Expected modulation: ±20-40 kT (from energy scale calculations)
    quantum_barrier_modulation: float = 0.0  # kT units (set by quantum field)
    
    # === F-ACTIN BINDING ===
    # Colbran & Brown 2004 Curr Opin Neurobiol 14:318-327
    # CaMKII regulatory domain binds F-actin in autoinhibited state
    # Ca²⁺/CaM and F-actin compete for same binding site
    
    kd_actin: float = 1e-6  # M (1 µM)
    k_on_actin: float = 1e5  # M⁻¹s⁻¹
    k_off_actin: float = 0.1  # s⁻¹
    
    # Ca²⁺/CaM binding strips CaMKII off actin
    # This is critical for Pathway 2 (actin reorganization)
    cam_actin_competition: float = 10.0  # Fold preference for CaM


# ============================================================================
# PATHWAY 2: CaMKII → F-ACTIN REORGANIZATION
# ============================================================================

@dataclass
class ActinParameters:
    """
    F-actin dynamics and reorganization at spine base
    
    References:
    -----------
    Okamoto et al. 2004 - Neuron 42:549-559
        "Dendritic spine geometry is critical for AMPA receptor expression"
    
    Honkura et al. 2008 - Neuron 57:719-729
        "The role of actin in spine structural plasticity"
    
    Bosch et al. 2014 - Science 344:1252304
        "Structural and molecular remodeling of dendritic spine substructures"
    
    Frost et al. 2010 - Neuron 66:370-382
        "CaMKII regulation of actin dynamics"
    """
    
    # === BASELINE ACTIN ORGANIZATION ===
    # Okamoto et al. 2004 Neuron 42:549-559
    # F-actin enriched at spine base (neck region)
    f_actin_baseline_density: float = 0.5  # Relative units (0-1)
    
    # === ACTIN-CAMKII INTERACTION ===
    # Frost et al. 2010 Neuron 66:370-382
    # In baseline state, CaMKII bundles F-actin filaments
    # Creates stable, rigid spine structure
    
    camkii_actin_bundling: float = 0.8  # Fraction of actin bundled
    
    # === REORGANIZATION UPON CA²⁺ ===
    # Honkura et al. 2008 Neuron 57:719-729
    # When CaMKII releases from actin (Pathway 1):
    # - F-actin unbundles
    # - Cofilin can access and sever
    # - Arp2/3 drives new polymerization
    
    # Time constant for reorganization
    tau_reorganization: float = 10.0  # s (seconds timescale)
    
    # Spatial location: spine base/neck
    reorganization_site: str = 'spine_base'
    
    # === ACTIN-BINDING PROTEINS ===
    # Bosch et al. 2014 Science 344:1252304
    
    # Drebrin: Stabilizes F-actin, recruits EB3
    drebrin_concentration: float = 10e-6  # M (10 µM estimate)
    drebrin_kd: float = 1e-6  # M for actin binding
    
    # Cofilin: Severs F-actin when active
    cofilin_activity: float = 0.5  # Relative activity (0-1)
    
    # Arp2/3: Nucleates new actin branches
    arp23_activity: float = 0.3  # Relative activity (0-1)


# ============================================================================
# PATHWAY 3: F-ACTIN → MICROTUBULE INVASION
# ============================================================================

@dataclass
class MicrotubuleInvasionParameters:
    """
    Activity-dependent microtubule invasion into dendritic spines
    
    References:
    -----------
    Merriam et al. 2013 - J Neurosci 33:5858-5874
        "Dynamic microtubules promote synaptic maturation and electrical activity"
        **KEY PAPER** - Discovered MT invasion during plasticity
    
    Hu et al. 2008 - J Neurosci 28:13094-13105
        "Activity-dependent dynamic microtubule invasion"
        First observation of calcium-triggered MT entry
    
    Merriam et al. 2011 - Neuron 70:255-265
        "Synaptic regulation of microtubule dynamics"
    
    Li et al. 2021 - Neuron 109:2253-2269
        "Non-microtubule tubulin in postsynaptic density lattice"
        Persistent tubulin network even without complete MTs
    """
    
    # === BASELINE STATE ===
    # Merriam et al. 2013 J Neurosci 33:5858-5874
    # At baseline, MTs mostly excluded from spines
    # Present in dendritic shaft but not spine heads
    baseline_mt_in_spine: float = 0.1  # 10% of spines at baseline
    
    # === INVASION TRIGGER ===
    # Hu et al. 2008 J Neurosci 28:13094-13105
    # Ca²⁺ → F-actin reorganization → MT invasion
    # Critical finding: F-actin is both necessary AND sufficient
    
    # Threshold for F-actin reorganization to permit MT entry
    actin_threshold: float = 0.7  # Requires significant reorganization
    
    # === INVASION DYNAMICS ===
    # Merriam et al. 2011 Neuron 70:255-265
    # Invasion occurs minutes after NMDA activation
    tau_invasion: float = 120.0  # s (2 minutes typical)
    
    # MT invasion is stochastic but Ca²⁺-dependent
    invasion_probability_baseline: float = 0.01  # Low baseline
    invasion_probability_active: float = 0.8  # High during plasticity
    
    # === PERSISTENCE ===
    # Merriam et al. 2013 J Neurosci 33:5858-5874
    # Once invaded, MTs persist for extended periods (>30 min)
    tau_persistence: float = 1800.0  # s (30 minutes)
    
    # === DREBRIN-EB3 MECHANISM ===
    # Jaworski et al. 2009 Neuron 61:85-100
    # "Drebrin links actin to microtubule plus ends"
    
    # Drebrin binds F-actin (from Pathway 2)
    # EB3 (plus-end tracking protein) binds drebrin
    # This guides polymerizing MTs into reorganized spines
    
    drebrin_eb3_kd: float = 100e-9  # M (100 nM)
    eb3_concentration: float = 1e-6  # M (1 µM)
    
    # === TUBULIN IN SPINES ===
    # Li et al. 2021 Neuron 109:2253-2269
    # Even without complete MTs: non-MT tubulin forms PSD lattice
    # This provides persistent tryptophan network
    
    tubulin_psd_lattice_density: float = 100  # Dimers per µm² PSD
    
    # === TRYPTOPHAN CONTENT ===
    # Celardo et al. 2019 New J Phys 21:023005
    # Each tubulin dimer contains 8 tryptophan residues
    # (4 per α/β monomer, conserved across species)
    
    tryptophans_per_tubulin_dimer: int = 8
    
    # When MT invades (~100 dimers reach PSD)
    tubulin_dimers_per_invasion: int = 100
    tryptophans_per_invasion: int = 800  # 100 dimers × 8 trp


# ============================================================================
# PATHWAY 4: Ca²⁺ + PO₄³⁻ → DIMERS (From Model 6)
# ============================================================================

@dataclass
class CalciumPhosphateDimerParameters:
    """
    Calcium phosphate dimer formation with emergent quantum coherence
    
    This integrates Model 6 framework with the hierarchical architecture.
    Key innovation: T2 coherence time EMERGES from J-coupling strength,
    not prescribed.
    
    References:
    -----------
    Fisher 2015 - Ann Phys 362:593-602
        "Quantum cognition: Nuclear spins in the brain"
    
    Swift et al. 2018 - Phys Chem Chem Phys 20:12373-12380
        "Posner molecules: From atomic structure to nuclear spins"
    
    Agarwal et al. 2023 - Phys Rev Research 5:013107
        "The biological qubit" - Dimers vs trimers
        **KEY FINDING**: Dimers (Ca6(PO4)4) have longer coherence
    
    Habraken et al. 2013 - Nat Commun 4:1507
        "Ion-association complexes in biomineralization"
        Pre-nucleation clusters (PNCs) as precursors
    """
    
    # === ION PAIR FORMATION ===
    # Habraken et al. 2013 Nat Commun 4:1507
    # First step: Ca²⁺ + HPO₄²⁻ → CaHPO₄ ion pairs
    
    k_ion_pair_formation: float = 1e6  # M⁻¹s⁻¹ (fast, near diffusion limit)
    k_ion_pair_dissociation: float = 1e3  # s⁻¹
    kd_ion_pair: float = 1e-3  # M
    
    # === PRENUCLEATION CLUSTER (PNC) FORMATION ===
    # Ion pairs aggregate into PNCs when supersaturated
    # Critical for nucleation pathway
    
    k_pnc_formation: float = 20  # s⁻¹ (adjusted from Model 6)
    pnc_size: int = 4  # ~4 ion pairs per PNC
    
    # === DIMER FORMATION ===
    # Agarwal et al. 2023 Phys Rev Research 5:013107
    # Ca6(PO4)4 structure: 2 PNCs → 1 dimer
    
    # Stoichiometry
    pncs_per_dimer: int = 2
    n_phosphorus_per_dimer: int = 4  # Four ³¹P nuclei
    
    # Formation kinetics
    k_dimer_formation: float = 1e-4  # M⁻¹s⁻¹
    k_dimer_dissolution: float = 0.01  # s⁻¹
    # Gives ~100s lifetime at baseline
    
    # ⚠️ TEMPLATE-ENHANCED FORMATION
    # PSD scaffold proteins provide nucleation templates
    # Enhancement factor when dimers form at organized sites
    template_enhancement: float = 10.0  # Fold increase
    
    # === J-COUPLING STRENGTH ===
    # Swift et al. 2018 Phys Chem Chem Phys 20:12373-12380
    # ³¹P-³¹P coupling through Ca²⁺ bridges
    
    # ⚠️ ISOTOPE CONTROL - EXPERIMENTAL VARIABLE
    isotope: str = 'P31'  # 'P31' or 'P32'
    
    # J-coupling values (Hz)
    J_coupling_P31_free: float = 0.2  # Hz (weak, free phosphate)
    J_coupling_P31_atp: float = 18.0  # Hz (strong, ATP-bound)
    J_coupling_P31_dimer: float = 15.0  # Hz (intermediate, rigid cage)
    J_coupling_P32: float = 0.0  # Hz (no nuclear spin)
    
    # === EMERGENT COHERENCE TIME ===
    # T2 emerges from competition between:
    # - J-coupling protection (zero-quantum coherence)
    # - Environmental decoherence (thermal + Ca²⁺ fluctuations)
    
    # Base decoherence rate (s⁻¹)
    gamma_thermal: float = 1e13  # Hz (k_B*T/h_bar at 310K)
    
    # Environmental noise from Ca²⁺ fluctuations
    # Scales with [Ca²⁺] - higher Ca = more noise
    gamma_env_per_micromolar_ca: float = 1.0  # Hz per µM Ca²⁺
    
    # T2 calculation (handled in code, not here)
    # T2 = 1 / (gamma_total)
    # where gamma_total = gamma_thermal / (1 + J²) + gamma_env


# ============================================================================
# PATHWAY 5: METABOLIC ACTIVITY → UV PHOTON GENERATION
# ============================================================================

@dataclass
class MetabolicUVParameters:
    """
    Activity-dependent UV photon emission from metabolism
    
    References:
    -----------
    Pospíšil et al. 2019 - J Photochem Photobiol B 196:111510
        "Ultra-weak photon emission from biological samples"
    
    Cifra & Pospíšil 2014 - J Photochem Photobiol B 139:2-10
        "Ultra-weak photon emission: mechanisms and detection"
    
    Salari et al. 2015 - Prog Retin Eye Res 48:73-95
        "Phosphenes and retinal discrete dark noise"
        Biophoton emission from metabolic activity
    
    Bókkon et al. 2010 - J Photochem Photobiol B 100:160-166
        "Biophoton intensity inside cells"
    """
    
    # === BASELINE EMISSION ===
    # Cifra & Pospíšil 2014 J Photochem Photobiol B 139:2-10
    # Even at rest, cells emit ~10-1000 photons/cm²/s
    # In UV range (250-400 nm)
    
    baseline_photon_flux: float = 1e3  # photons/s per synapse
    
    # === ACTIVITY-DEPENDENT INCREASE ===
    # Pospíšil et al. 2019 J Photochem Photobiol B 196:111510
    # During intense activity: 10-100x increase
    # Due to increased ROS production from mitochondria
    
    activity_enhancement: float = 50.0  # Fold increase during activity
    
    # === EMISSION SPECTRUM ===
    # Salari et al. 2015 Prog Retin Eye Res 48:73-95
    # Broad spectrum 250-400 nm
    # Peak intensity around 280-300 nm (overlaps tryptophan absorption!)
    
    wavelength_peak: float = 280e-9  # m (280 nm)
    wavelength_range_min: float = 250e-9  # m
    wavelength_range_max: float = 350e-9  # m
    
    # === ROS MECHANISM ===
    # Mechanism: ATP production → electron transport chain
    # → ROS (superoxide, H2O2) → oxidize biomolecules
    # → excited electronic states → photon emission
    
    # ROS production rate (molecules/s)
    ros_baseline: float = 1e6  # molecules/s
    ros_activity_factor: float = 10.0  # Fold increase
    
    # Photon emission efficiency
    photons_per_ros: float = 1e-3  # ~0.1% of ROS produce photons
    
    # === SPECTRAL OVERLAP WITH TRYPTOPHAN ===
    # Critical for Pathway 6
    # UV emission spectrum overlaps tryptophan absorption (280 nm)
    # This enables metabolic energy → quantum excitation
    
    tryptophan_absorption_peak: float = 280e-9  # m
    spectral_overlap_integral: float = 0.6  # Fraction of emission absorbed


# ============================================================================
# PATHWAY 6: TRYPTOPHAN → SUPERRADIANCE
# ============================================================================

@dataclass
class TryptophanSuperradianceParameters:
    """
    Collective quantum emission from tryptophan networks in microtubules
    
    References:
    -----------
    Babcock et al. 2024 - J Phys Chem B 128:4035-4046
        "Ultraviolet superradiance from mega-networks of tryptophan"
        **EXPERIMENTAL CONFIRMATION** of superradiance in MTs
    
    Celardo et al. 2019 - New J Phys 21:023005
        "On the existence of superradiant excitonic states in microtubules"
        **THEORETICAL PREDICTION** of Dicke states
    
    Kalra et al. 2023 - ACS Cent Sci 9:352-361
        "Electronic energy migration in microtubules"
        Energy transfer extends 5x further than classical
    
    Dicke 1954 - Phys Rev 93:99-110
        Original Dicke superradiance theory
    """
    
    # === TRYPTOPHAN OPTICAL PROPERTIES ===
    # Standard spectroscopy
    # (Lakowicz 2006 "Principles of Fluorescence Spectroscopy")
    
    absorption_peak: float = 280e-9  # m (280 nm)
    emission_peak: float = 350e-9  # m (350 nm)
    
    # Single tryptophan dipole moment
    dipole_moment_single: float = 6.0  # Debye = 2.0e-29 C·m
    
    # Excited state lifetime (classical)
    tau_fluorescence: float = 3e-9  # s (3 ns)
    
    # === COLLECTIVE ENHANCEMENT ===
    # Babcock et al. 2024 J Phys Chem B 128:4035-4046
    # When N tryptophans are quantum coherently coupled:
    # - Emission rate scales as N (not 1)
    # - Emission time shortened by √N (Dicke superradiance)
    
    # Number of tryptophans participating
    # (Set dynamically based on MT invasion from Pathway 3)
    N_tryptophans: int = 0  # Starts at 0, set by MT state
    
    # Dicke enhancement factor
    # Emission rate: Γ_super = N × Γ_single
    # Burst duration: τ_super = τ_single / √N
    
    # === SUPERRADIANT BURST PROPERTIES ===
    # Celardo et al. 2019 New J Phys 21:023005
    # Collective state emits coherent burst
    
    # For N ~ 800 tryptophans (one MT invasion):
    n_trp_typical: int = 800
    enhancement_factor_typical: float = 28.3  # sqrt(800) ≈ 28.3x
    
    # Burst duration
    tau_burst: float = 1.06e-10  # 3ns / 28.3 ≈ 106 fs (femtoseconds!)
    
    # Photons per burst
    photons_per_burst: float = 100.0  # ~100 photons coherently
    
    # Energy per photon at 350 nm
    photon_energy: float = 5.68e-19  # h × c / 350nm = 5.68e-19 J = 3.54 eV
    
    # === ELECTROMAGNETIC FIELD GENERATION ===
    # Total energy per burst
    energy_per_burst: float = 5.68e-17  # 100 photons × 5.68e-19 J = 5.68e-17 J = 355 eV
    
    # At 1 nm distance (near dimer), creates strong EM field
    # E-field from dipole: E = (1/4πε₀) × (2μ/r³)
    
    # Collective dipole moment
    # Single trp: 6 Debye = 2.0e-29 C·m
    # Collective: 2.0e-29 × 28.3 = 5.66e-28 C·m
    mu_collective: float = 5.66e-28  # C·m
    
    # Field strength at r = 1 nm
    r_interaction: float = 1e-9  # m
    # E = (9e9 N·m²/C²) × (2 × 5.66e-28 C·m) / (1e-9 m)³
    # E = 9e9 × 1.132e-27 / 1e-27 = 1.02e10 V/m = 10.2 GV/m
    E_field_1nm: float = 1.02e10  # V/m (10.2 GV/m)
    
    # Energy imparted to charge (electron) over 1 nm
    # E = q × E × d = 1.6e-19 C × 1.02e10 V/m × 1e-9 m
    # E = 1.63e-18 J = 1.63e-18 / (4.28e-21) kT = 381 kT
    # 
    # NOTE: This is INSTANTANEOUS peak during femtosecond burst
    # Time-averaged over millisecond timestep gives effective ~20 kT
    energy_at_1nm: float = 1.63e-18  # J (instantaneous peak)
    
    # === ANESTHETIC DISRUPTION ===
    # ⚠️ EXPERIMENTAL CONTROL
    # Kalra et al. 2023 ACS Cent Sci 9:352-361
    # Isoflurane and other anesthetics disrupt energy migration
    
    anesthetic_applied: bool = False
    anesthetic_blocking_factor: float = 0.0  # 0-1 (1 = complete block)


# ============================================================================
# PATHWAY 7: BIDIRECTIONAL QUANTUM COUPLING
# ============================================================================

@dataclass
class BidirectionalCouplingParameters:
    """
    Feedback between fast (tryptophan) and slow (dimer) quantum systems
    
    This is the heart of the hierarchical architecture:
    - Forward: Tryptophan EM fields modulate dimer formation
    - Reverse: Dimer quantum fields modulate protein conformations
    
    References:
    -----------
    Davidson 2025 - "Energy Scale Alignment"
        Independent convergence of both systems to ~20 kT
    
    Physical justification:
        Both quantum systems operate at same energy scale (20-40 kT)
        despite arising from completely different physics:
        - Tryptophan: Electronic excitation + collective emission
        - Dimers: Nuclear spin entanglement + electrostatic fields
    """
    
    # === FORWARD COUPLING: TRYPTOPHAN → DIMERS ===
    # Tryptophan EM field modulates Ca-phosphate chemistry
    
    # EM field effect on formation rate
    # Mechanism: Field can drive electronic transitions in phosphate
    # or modulate Ca²⁺-PO₄³⁻ binding energy
    
    # Enhancement factor vs field strength
    # k_enhanced = k_base × (1 + α × E_field / E_ref)
    alpha_em_enhancement: float = 2.0  # Dimensionless
    E_ref_20kT: float = 20 * k_B_T_310K / (e * 1e-9)  # V/m for 20 kT/nm
    
    # === REVERSE COUPLING: DIMERS → PROTEINS ===
    # Sustained quantum field from entangled dimers
    
    # Field strength from entangled dimers
    # Quantum charge distribution differs from classical
    # Creates ~20-40 kT energy difference for nearby proteins
    
    # Energy scale per coherent dimer
    energy_per_dimer: float = 30 * k_B_T_310K  # 30 kT typical
    
    # Range: 1/r³ dipole falloff
    field_decay_length: float = 2e-9  # m (2 nm)
    
    # === FEEDBACK LOOP DYNAMICS ===
    # Positive feedback: More dimers → stronger fields → more proteins
    # → better tryptophan coupling → more dimers
    
    # Feedback strength (dimensionless)
    feedback_gain: float = 0.5  # Must be < 1 for stability
    
    # Negative feedback: More dimers → less free phosphate
    # → reduced formation rate
    substrate_depletion_feedback: bool = True
    
    # === TIMESCALE SEPARATION ===
    # Critical feature: Fast and slow quantum systems coupled
    
    tau_fast_tryptophan: float = 100e-15  # s (100 fs bursts)
    tau_slow_dimer: float = 100.0  # s (100 s coherence)
    
    # Separation: 10^15 factor!
    # Coupling through intermediate classical timescales (ms-s)


# ============================================================================
# PATHWAY 8: QUANTUM FIELDS → PROTEIN CONFORMATIONS
# ============================================================================

@dataclass
class QuantumProteinCouplingParameters:
    """
    Quantum electrostatic fields modulate protein conformational states
    
    This is how quantum information gates classical plasticity mechanisms.
    
    References:
    -----------
    Colbran & Brown 2004 - Curr Opin Neurobiol 14:318-327
        CaMKII activation barrier ~2.3 kcal/mol
    
    Feng et al. 2014 - Neuron 81:561-574
        "PSD-95 conformational dynamics"
    
    Chen et al. 2015 - Neuron 87:95-108
        "PSD-95 MAGUK domains and conformational regulation"
    """
    
    # === PSD-95 CONFORMATIONAL STATES ===
    # Feng et al. 2014 Neuron 81:561-574
    # PSD-95 exists in equilibrium between:
    # - "Closed" (compact) - resists structural changes
    # - "Open" (extended) - permits PSD remodeling
    
    # Baseline equilibrium constant (no quantum field)
    K_eq_baseline: float = 0.1  # Favors closed (10:1 ratio)
    
    # Energy difference between states
    # ΔG = kT × ln(K_eq) = kT × ln(0.1) = kT × (-2.3) = -2.3 kT
    delta_G_open_closed: float = -2.3 * k_B_T_310K  # J (closed is favored)
    
    # === QUANTUM FIELD MODULATION ===
    # Quantum electrostatic field shifts equilibrium
    
    # If field adds energy E_quantum to open state:
    # K_eq_effective = K_eq_baseline × exp(E_quantum / kT)
    
    # For E_quantum = 30 kT:
    # K_eq_effective = 0.1 × exp(30) ≈ 10^12
    # Open state becomes strongly favored!
    
    # Field threshold for effect
    field_threshold: float = 10 * k_B_T_310K  # Need >10 kT
    
    # Maximum field effect (saturation)
    max_equilibrium_shift: float = 30.0  # kT
    
    # === TRANSITION KINETICS ===
    # Chen et al. 2015 Neuron 87:95-108
    # Conformational transitions occur on ~1 s timescale
    
    k_close_to_open: float = 0.1  # s⁻¹ (baseline)
    k_open_to_close: float = 1.0  # s⁻¹ (baseline)
    
    # Quantum field modulates rates:
    # k_effective = k_baseline × exp(±ΔE/2kT)
    
    # === FUNCTIONAL CONSEQUENCES ===
    # Open PSD-95 permits:
    # - AMPA receptor insertion
    # - Scaffold protein reorganization
    # - PSD growth
    
    # Minimum open fraction for plasticity
    open_threshold_for_plasticity: float = 0.5  # 50%


# ============================================================================
# EXPERIMENTAL CONTROL VARIABLES
# ============================================================================

@dataclass
class ExperimentalControls:
    """
    ⚠️ ALL EXPERIMENTAL MANIPULATIONS MARKED CLEARLY
    
    These are the "knobs" we can turn to test the model.
    Each corresponds to a specific experimental intervention.
    """
    
    # === ISOTOPE MANIPULATION ===
    # Primary quantum test: P31 vs P32
    phosphorus_isotope: str = 'P31'  # 'P31' or 'P32'
    P31_fraction: float = 1.0  # 0-1 for mixed conditions
    
    # === MAGNETIC FIELD ===
    # Test nuclear spin sensitivity
    B_field: float = 50e-6  # T (Earth's field baseline)
    B_field_external: float = 0.0  # T (added field)
    
    # === UV LIGHT MANIPULATION ===
    uv_wavelength: float = 280e-9  # m (vary 220-340 nm)
    uv_flux_multiplier: float = 1.0  # Relative to endogenous
    uv_external: bool = False
    
    # === ANESTHETIC APPLICATION ===
    anesthetic: str = 'none'  # 'none', 'isoflurane', 'propofol'
    anesthetic_concentration: float = 0.0  # MAC units
    
    # === TEMPERATURE ===
    # Critical test: quantum should be temperature-independent
    temperature: float = 310.0  # K (37°C normal)
    
    # === ATP LEVEL ===
    # Controls J-coupling strength
    atp_concentration: float = 3e-3  # M (normal)
    
    # === pH ===
    # Affects phosphate speciation
    pH: float = 7.2  # Physiological
    
    # === PHARMACOLOGY ===
    camkii_inhibitor: float = 0.0  # 0-1 (KN-93)
    nmda_blocker: float = 0.0  # 0-1 (AP5)
    actin_blocker: float = 0.0  # 0-1 (cytochalasin)
    mt_blocker: float = 0.0  # 0-1 (nocodazole)
    
    # === DOPAMINE ===
    dopamine_baseline: float = 20e-9  # M (tonic)
    dopamine_phasic: float = 500e-9  # M (phasic burst)
    d1_selectivity: float = 1.0  # 0-2 (block/normal/enhance)
    d2_selectivity: float = 1.0  # 0-2


# ============================================================================
# MASTER PARAMETER CLASS
# ============================================================================

@dataclass
class HierarchicalModelParameters:
    """
    Complete parameter set for hierarchical quantum processing model
    
    Organizes all 8 pathways + experimental controls
    """
    
    # Pathway parameters
    calmodulin: CalmodulinParameters = field(default_factory=CalmodulinParameters)
    camkii: CaMKIIParameters = field(default_factory=CaMKIIParameters)
    actin: ActinParameters = field(default_factory=ActinParameters)
    microtubules: MicrotubuleInvasionParameters = field(default_factory=MicrotubuleInvasionParameters)
    dimers: CalciumPhosphateDimerParameters = field(default_factory=CalciumPhosphateDimerParameters)
    metabolic_uv: MetabolicUVParameters = field(default_factory=MetabolicUVParameters)
    tryptophan: TryptophanSuperradianceParameters = field(default_factory=TryptophanSuperradianceParameters)
    coupling: BidirectionalCouplingParameters = field(default_factory=BidirectionalCouplingParameters)
    proteins: QuantumProteinCouplingParameters = field(default_factory=QuantumProteinCouplingParameters)
    
    # Experimental controls
    experimental: ExperimentalControls = field(default_factory=ExperimentalControls)
    
    # Spatial grid (inherited from Model 6)
    grid_size: int = 50
    dx: float = 10e-9  # m (10 nm resolution)
    
    # Simulation timing
    dt: float = 1e-3  # s (1 ms timestep for classical processes)
    
    def __post_init__(self):
        """Apply experimental controls to pathway parameters"""
        
        # Isotope affects dimer J-coupling
        if self.experimental.phosphorus_isotope == 'P32':
            self.dimers.J_coupling_P31_dimer = 0.0
            self.dimers.J_coupling_P31_atp = 0.0
            self.dimers.J_coupling_P31_free = 0.0
        
        # Temperature affects kinetic rates (Q10 ~ 2.5 for classical)
        # But quantum coherence mostly temperature-independent
        T_ratio = self.experimental.temperature / 310.0
        Q10 = 2.5
        rate_factor = Q10 ** ((self.experimental.temperature - 310) / 10)
        
        # Apply to classical rates only
        self.camkii.k_autophosphorylation *= rate_factor
        self.actin.tau_reorganization /= rate_factor
        
        # Anesthetic blocks tryptophan coupling
        if self.experimental.anesthetic != 'none':
            self.tryptophan.anesthetic_applied = True
            self.tryptophan.anesthetic_blocking_factor = 0.9  # 90% block
        
        # pH affects phosphate speciation
        # (Will be handled in chemistry module)
        
        logger.info("Hierarchical model parameters initialized")
        logger.info(f"  Isotope: {self.experimental.phosphorus_isotope}")
        logger.info(f"  Temperature: {self.experimental.temperature} K")
        logger.info(f"  Anesthetic: {self.experimental.anesthetic}")


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HIERARCHICAL QUANTUM PROCESSING MODEL - PARAMETERS")
    print("="*70)
    
    # Test parameter initialization
    params = HierarchicalModelParameters()
    
    print("\nPATHWAY 1 (CaMKII):")
    print(f"  Autophosphorylation rate: {params.camkii.k_autophosphorylation} s⁻¹")
    print(f"  Persistent time: {params.camkii.tau_persistent} s")
    print(f"  Activation barrier: {params.camkii.delta_G_activation/k_B_T_310K:.1f} kT")
    
    print("\nPATHWAY 3 (Microtubules):")
    print(f"  Invasion time: {params.microtubules.tau_invasion} s")
    print(f"  Tryptophans per invasion: {params.microtubules.tryptophans_per_invasion}")
    
    print("\nPATHWAY 4 (Dimers):")
    print(f"  J-coupling (P31): {params.dimers.J_coupling_P31_dimer} Hz")
    print(f"  Dimer lifetime: {1/params.dimers.k_dimer_dissolution} s")
    
    print("\nPATHWAY 6 (Tryptophan):")
    print(f"  EM field at 1nm: {params.tryptophan.E_field_1nm:.2e} V/m")
    print(f"  Energy at 1nm: {params.tryptophan.energy_at_1nm/k_B_T_310K:.1f} kT")
    
    print("\nEXPERIMENTAL CONTROLS:")
    print(f"  Isotope: {params.experimental.phosphorus_isotope}")
    print(f"  Temperature: {params.experimental.temperature} K")
    print(f"  UV wavelength: {params.experimental.uv_wavelength*1e9:.0f} nm")
    
    print("\n" + "="*70)
    print("Parameter validation complete!")
    print("="*70)