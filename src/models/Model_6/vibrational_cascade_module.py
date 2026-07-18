"""
Vibrational Cascade Coupling Module
=====================================

Replaces em_coupling_module.py's linear coupling with a physically grounded
frequency cascade based on Fröhlich condensation dynamics.

PHYSICS:
--------
The coupling between tryptophan superradiance (Q1) and calcium phosphate
dimer coherence (Q2) operates through a multi-stage frequency cascade:

  UV excitation (~10^15 Hz, femtoseconds)
      ↓ Excitonic coupling (Babcock/Kurian 2024)
  Collective superradiant emission (√N enhancement)
      ↓ Metabolic power delivered to the lattice segment
  Collective microtubule mode at ω₀ = 8 MHz
      ↓ Fröhlich condensation if P_met ≥ P_c (Wang/Wang 2022)
  Condensed collective mode
      ↓ Electric field modulation at dimer sites (Chafai/Cifra 2019)
  Modified electric field gradient at P-31 sites
      ↓ NMR-like relaxation modulation
  Nuclear spin coherence dynamics (~Hz, seconds)

The critical innovation: the 20 kT threshold is NOT an energy barrier.
It's a threshold for Fröhlich condensation. Below it, energy thermalizes
normally. Above it, energy condenses into the collective mode.

THE THRESHOLD (Wang & Wang 2022, arXiv:2209.05086)
--------------------------------------------------
Reference-free — condensation begins where the source occupation reaches the
thermal occupation of the bath at the mode frequency, n_ex = n̄_s:

    P_c = n̄_s · ℏ · (2π·ω₀)² / Q
    r   = P_met / P_c
    η   = (r − 1)/(r + 1)  for r ≥ 1, else 0      (second order, β = 1)

This is the SAME physics the backbone pump runs
(multi_synapse_network._update_backbone_field). The two are two segments of ONE
microtubule lattice at ONE collective mode — not two systems. The only intended
difference is aggregation: the backbone sums over coupled spines, this site does not.
sweep/pump_mode_agreement_probe.py measures that they still agree.

RETIRED 2026-07-18 (B2) — do not restore:
  · ω₀ = 40 GHz / ω_max = 160 GHz, the tubulin PROTEIN modes (Pandey & Cifra 2024).
    A different mode family; carrying them made the two pumps disagree by 5000×.
  · φ = 10 GHz, χ = 0.05 GHz and the Zhang 2019 BSA parameter values. φ is now the
    oscillator constraint ω₀/Q; χ is rescaled to preserve χ < φ and sets slope only.
  · r_at_E_ref = 100 GHz, pump_exponent = 2.0, E_ref_pump, and the kT_ref = 22.1
    function-body literal — the calibration fiction that made the headline
    "pump exceeds threshold at MT+" (r/r_c ≈ 1.045) an arithmetic identity.
  · r_c = (φ/(D+1))(1+φ/χ), the classical critical pump: →0 in large-D, an
    artificial reference scale rather than a threshold.
  · The hand-rolled hbar = 1.0546e-34 applied to a LINEAR frequency (ℏ·f), which
    inflated n̄ by 6.3×. There is one Planck factor in Model 6:
    bose_einstein_occupation (model6_parameters.py), and it uses h·f.

The Zhang rate equations still describe the ABOVE-threshold dynamics, and φ/χ set the
slope there. What they do not supply is the threshold.

INTERFACE:
---------
update() takes metabolic power (p_met_W). It previously took em_field_trp /
collective_field_kT; converting a kT field energy to a pump rate is what required the
retired calibration constants.

LITERATURE:
----------
Wang & Wang (2022) arXiv:2209.05086 — quantum-pump threshold n_ex = n̄_s, β = 1
Zhang, Agarwal & Scully (2019) PRL 122:158101 — Fröhlich rate equations (dynamics
    above threshold; BSA parameter VALUES deliberately not adopted here)
Sahu et al.; Pokorný — measured MT resonances; 8.085 MHz, slip-layer damping
Azizi, Gori, Morzan, Hassanali & Kurian (2023) PNAS Nexus 2:pgad257
Reimers et al. (2009) PNAS 106:4219 — Weak/strong/coherent Fröhlich regimes
Chafai/Cifra et al. (2019) Sci Rep 9:10477 — Tubulin electric field response

Author: Sarah Davidson
Date: March 2026 (pump retired and rebuilt, B2, July 2026)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

# The ONE Planck factor in Model 6 (h·f, not the ℏ·f shortcut), and the CODATA hbar
# (scipy.constants.hbar). This site used to hand-roll its own hbar = 1.0546e-34 with a
# dropped 2π; B2 retired that. Do not reintroduce a local hbar here. Imported exactly as
# multi_synapse_network.py imports them, so the two pump sites share one source.
from model6_parameters import bose_einstein_occupation, hbar

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class TubulinCascadeParameters:
    """
    Physics-based parameters for the vibrational cascade.
    
    Sources are noted for each parameter. Parameters marked [ESTIMATE]
    are derived from scaling arguments and should be updated when
    tubulin-specific measurements become available.
    """
    
    # === THE COLLECTIVE CONDENSING MODE ===
    # ω₀ is a LINEAR frequency f, in Hz. It is consumed as h·f (never ℏ·f — see
    # bose_einstein_occupation, model6_parameters.py:41). The symbol is named ω₀ for
    # continuity with the Zhang/Agarwal/Scully notation, but the VALUE is f, not 2πf.
    # That naming/units mismatch is what produced the factor-of-2π error retired in B2.
    #
    # 8 MHz is the collective microtubule condensing mode (Sahu et al. measured MT
    # resonances; Pokorný 8.085 MHz), pinned 2026-05-30. It is the SAME mode the backbone
    # pump uses (DendriticBackboneParameters.omega_0, model6_parameters.py) — the two pumps
    # are two segments of one lattice, not two systems.
    #
    # RETIRED here (B2, 2026-07-18): ω₀ = 40 GHz and ω_max = 160 GHz (Pandey & Cifra 2024).
    # Those are the tubulin PROTEIN modes, a different mode family from the condensing mode.
    # Carrying them here made the two pump sites disagree by 5000× — the mode-conflation
    # bug. ω_max was declared and never read by anything; it is gone rather than rescaled.
    omega_0: float = 8.0e6            # Hz — collective MT condensing mode (linear f)
    Q: float = 10.0                   # Quality factor. Matches the backbone's Q
                                      # (model6_parameters.py) — same lattice, same damping
                                      # bet: Pokorný slip-layer/ordered-water (Q≳10) vs
                                      # Foster/Baish overdamped (Q~1). Committed as a
                                      # hypothesis on 2026-05-30, not to be cranked later
                                      # to save a result.
    D_modes: int = 20                 # Effective number of sub-THz modes participating
                                      # in the cascade to nuclear spin environment.
                                      # Tubulin has ~300 sub-THz modes total, but only
                                      # ~20 couple efficiently to the dimer formation
                                      # sites via conformational pathways.
                                      #
                                      # ⚠ POST-B2: D_modes enters NO physics here — it is
                                      # printed and nothing more (verified on executable
                                      # code, MO ruling 005 §2). It does not enter P_c;
                                      # only ω₀ and Q do.
                                      #
                                      # STATED LIMIT ON η — read this before quoting η.
                                      # η = (r−1)/(r+1) is the LARGE-D limit of the
                                      # quantum-pump treatment (Wang/Wang 2022). The pin
                                      # calls for D ≳ 200; this site runs D = 20 and the
                                      # backbone runs 50. Because D no longer enters the
                                      # formula, a small D does not change the number
                                      # computed — it bears on whether the large-D FORM is
                                      # the right one to use at all. Finite-D corrections
                                      # to the order parameter have NOT been derived or
                                      # checked here, so the adequacy of the limit at
                                      # D = 20 is UNVERIFIED and is recorded as a limit on
                                      # every per-synapse η this module reports.
                                      # D is NOT tuned to make this comfortable — raising
                                      # it to sit inside the limit would be moving a
                                      # constant to reach an outcome (MO_MODEL6 §7).
                                      # Closing it needs the finite-D expansion from the
                                      # source paper, which is a physics unit, not a
                                      # code change.
    
    # === FRÖHLICH DISSIPATION AND REDISTRIBUTION RATES ===
    #
    # ⚠ POST-B2 STATUS, VERIFIED 2026-07-18 (MO ruling 005 §2): φ, χ and D_modes are
    # **DIAGNOSTIC ONLY at this site — they enter NO physics.** Checked on executable code
    # with comments and docstrings stripped: every remaining use is a declaration, the
    # __post_init__ derivation, or a print/log. The live chain is
    #     P_c = n̄_s·ℏ·(2π·ω₀)²/Q  →  r = P_met/P_c  →  η = (r−1)/(r+1)
    # which consumes ω₀, Q and T only. An earlier draft of this comment claimed χ was
    # "kept because the steady-state solution needs a nonlinear term" — that was FALSE and
    # is corrected here: B2 deleted the Zhang steady-state quadratic, so there is no
    # nonlinear term left to feed. They are retained, not deleted, because the Zhang rate
    # equations remain the right description of above-threshold dynamics should a future
    # consumer need them — but nothing consumes them today. Do not describe them as
    # load-bearing, and do not infer a result from changing them: nothing will move.
    #
    # NO Zhang 2019 CITATION HERE — deliberately. Zhang's BSA values (φ=6 GHz, χ=0.07 GHz)
    # belong to the GHz protein-mode family retired above. Adopting them would re-import
    # the exact mode conflation B2 removes, wearing a citation.
    #
    # φ is fixed by the oscillator constraint φ = ω₀/Q, pinned 2026-05-30:
    #   "φ ≤ ω₀ (oscillator constraint; φ = ω₀/Q ≲ 0.8 MHz at Q≳10). Forces the old
    #    10 GHz φ out."  — model6-network-layer-feasibility-may30
    # DERIVED in __post_init__ so it cannot drift out of step with ω₀/Q if either is
    # swept. The old 10 GHz value was 12500× the constraint.
    phi_dissipation: float = 0.0      # Hz — DERIVED as ω₀/Q in __post_init__. Do not set.

    # χ. Its VALUE had to move: χ must satisfy χ < φ (two-phonon slower than one-phonon),
    # and the old χ = 0.05 GHz is 62.5× LARGER than the new φ = 0.8 MHz — it would have
    # inverted its own stated constraint. The χ/φ ratio is preserved from the pre-B2
    # values (0.05/10 = 0.005) rather than re-derived. A rescale to stay self-consistent,
    # NOT a new claim about the value — and, per the status note above, it currently
    # changes no computed quantity.
    chi_ratio: float = 0.005          # χ/φ, preserved from the pre-B2 values
    chi_redistribution: float = 0.0   # Hz — DERIVED as chi_ratio·φ in __post_init__.
    
    # === TEMPERATURE ===
    T_body: float = 310.0             # K — body temperature

    # === RETIRED IN B2 (2026-07-18): THE CALIBRATION FICTION ===
    # E_ref_pump = 1.4e9, r_at_E_ref = 100.0e9 and pump_exponent = 2.0 are GONE, together
    # with the kT_ref = 22.1 function-body literal that used them. They were not a
    # measurement: r_at_E_ref's own comment read "Calibrated so that full MT invasion
    # (22 kT field) produces r > r_c", and with r_c = (φ/(D+1))(1+φ/χ) that made the
    # headline r/r_c ≈ 1.045 at MT+ an ARITHMETIC IDENTITY between two numbers chosen to
    # produce it. There is no derivation to recover: r_c is the classical critical pump,
    # which →0 in large-D — an artificial reference scale (May-30 session). The threshold
    # is now the reference-free n_ex = n̄_s (Wang/Wang 2022); see PumpRateCalculator.
    # E_ref_pump was additionally self-referential ("from current em_tryptophan_module
    # typical output") while citing Azizi/Kurian 2023 for the phenomenon.

    # === CONDENSATION → PHYSICAL MODULATION ===
    # How the condensed mode affects the local environment
    modulation_coupling: float = 0.8  # Fraction of condensation energy → barrier modulation

    # === FORMATION RATE ENHANCEMENT ===
    # How condensation affects dimer formation kinetics
    # Replaces the linear alpha_em_enhancement = 2.0
    enhancement_baseline: float = 1.0 # No enhancement below threshold

    # === REVERSE COUPLING (dimers → protein modulation) ===
    # Kept from original for compatibility; these affect CaMKII gating
    energy_per_dimer_kT: float = 6.6  # From original calculations
    partial_entanglement_factor: float = 0.3
    spatial_averaging_factor: float = 0.15
    n_dimer_threshold: int = 50       # Fisher's prediction

    substrate_depletion_feedback: bool = True

    def __post_init__(self):
        """Derive φ and χ from ω₀ and Q so they cannot drift out of step.

        φ = ω₀/Q is the oscillator constraint (pinned 2026-05-30). χ = chi_ratio·φ keeps
        χ < φ. Deriving rather than declaring is deliberate: the defect B2 retires was two
        parameter sets silently disagreeing about which mode they described, and a declared
        φ swept independently of ω₀ would reopen exactly that. A caller that needs a
        different damping sets Q or chi_ratio, not φ or χ directly.
        """
        self.phi_dissipation = self.omega_0 / self.Q
        self.chi_redistribution = self.chi_ratio * self.phi_dissipation


# =============================================================================
# STAGE 1: PUMP RATE CALCULATOR
# =============================================================================

class PumpRateCalculator:
    """
    Compute how hard the collective mode is being driven, relative to the threshold
    at which it condenses.

    THE THRESHOLD IS REFERENCE-FREE (Wang & Wang 2022, arXiv:2209.05086)
    ------------------------------------------------------------------
    Condensation begins when the source occupation reaches the thermal occupation of
    the bath at the mode frequency:

        n_ex = n̄_s          (equivalently, P_met = P_c)

    with the critical power

        P_c = n̄_s · ℏ · (2π·ω₀)² / Q

    and the order parameter, a genuine second-order transition with β = 1,

        η = (r − 1)/(r + 1)   for r ≥ 1,   η = 0 otherwise,   r = P_met / P_c

    This is the SAME computation the backbone pump performs in
    multi_synapse_network._update_backbone_field — deliberately, and it is the point of
    B2. The two pumps are two segments of one microtubule lattice at one collective
    mode, so they must not run different threshold physics. The only difference is
    aggregation: the backbone sums over coupled neighbours (coupling_weights @ p_active),
    the per-synapse site does NOT aggregate — it sees its own spine's metabolic power.

    WHAT WAS RETIRED HERE, AND WHY IT WAS NOT "FIXED" (B2, 2026-07-18)
    -----------------------------------------------------------------
    This class previously computed

        r = r_at_E_ref · (collective_field_kT / kT_ref) ^ pump_exponent
        r_c = (φ/(D+1)) · (1 + φ/χ)                       [Zhang Eq. 4]

    with r_at_E_ref = 100 GHz and kT_ref = 22.1 bound as a literal INSIDE the function
    body — invisible to TubulinCascadeParameters and therefore to sweep_runner. The
    headline result "the per-synapse pump exceeds its Fröhlich threshold under MT
    invasion" (r/r_c ≈ 1.045) was an ARITHMETIC IDENTITY between two numbers chosen to
    produce it, not a measurement.

    There was no derivation to recover, so nothing was recalibrated: r_c is the CLASSICAL
    critical pump, which →0 in large-D and is only an artificial reference scale
    (May-30 session). Justifying kT_ref would have meant justifying scaffolding with
    nothing underneath it. The whole reference construction is gone, replaced by a
    threshold that needs no reference.

    _critical_threshold() (Zhang Eq. 4) is deleted for the same reason: a live
    computation of a retired artificial scale is the scaffolding, not a diagnostic.

    LITERATURE
    ----------
    Wang & Wang 2022 (arXiv:2209.05086) — quantum-pump threshold n_ex = n̄_s, β = 1
    Zhang, Agarwal & Scully 2019 (PRL 122:158101) — rate equations. Used for the
        above-threshold dynamics ONLY; its BSA parameter values are not adopted here
        (they belong to the retired GHz protein-mode family).
    """

    def __init__(self, params: TubulinCascadeParameters):
        self.params = params
        self._r_history = []

    def critical_power(self) -> float:
        """P_c = n̄_s · ℏ · (2π·ω₀)² / Q — the power at which n_ex reaches n̄_s.

        Mirrors the backbone's P_c exactly (multi_synapse_network.py). ω₀ is a LINEAR
        frequency, so the angular frequency is formed explicitly here and the thermal
        occupation is taken from bose_einstein_occupation (h·f). Both conventions are
        checked against an independent CODATA recomputation by
        sweep/pump_mode_agreement_probe.py.
        """
        p = self.params
        omega_ang = 2.0 * np.pi * p.omega_0
        n_bar_s = bose_einstein_occupation(p.omega_0, p.T_body)
        return n_bar_s * hbar * omega_ang ** 2 / p.Q

    def calculate_pump_rate(self, p_met_W: float) -> Dict:
        """Drive ratio r = P_met / P_c for THIS spine (no aggregation).

        Parameters
        ----------
        p_met_W : float
            Metabolic power delivered to this spine's lattice segment, in Watts, from
            compute_metabolic_power(E_invasion, ca_open_fraction, p_active_max_W).

        Returns
        -------
        dict with pump_power_W, critical_power_W, pump_ratio (r), above_threshold.
        """
        P_c = self.critical_power()

        if p_met_W is None or p_met_W <= 0.0 or P_c <= 0.0:
            return {
                'pump_power_W': 0.0,
                'critical_power_W': P_c,
                'pump_ratio': 0.0,
                'above_threshold': False,
            }

        r = p_met_W / P_c
        self._r_history.append(r)

        return {
            'pump_power_W': p_met_W,
            'critical_power_W': P_c,
            'pump_ratio': r,
            'above_threshold': r >= 1.0,
        }


# =============================================================================
# STAGE 2: FRÖHLICH CONDENSATION DYNAMICS
# =============================================================================

class FrohlichCondensation:
    """
    Turn the drive ratio r = P_met/P_c into the condensation order parameter η.

    η = (r − 1)/(r + 1) for r ≥ 1, else 0.

    This is the Wang/Wang 2022 quantum-pump order parameter — a genuine second-order
    transition with critical exponent β = 1 — and it is EXACTLY the expression the
    backbone pump uses (multi_synapse_network._update_backbone_field). One lattice, one
    mode, one order parameter. The substrate audit (2026-07-18) confirmed this form
    carries no fitted curve: "eta is exactly (r-1)/(r+1) with no fitted curve".

    WHAT WAS RETIRED HERE (B2, 2026-07-18)
    --------------------------------------
    This class previously solved the Zhang Eq. 14 steady-state quadratic for ⟨n₀⟩ and
    formed η = ⟨n₀⟩/N, against the classical threshold r_c = (φ/(D+1))(1+φ/χ). Two
    problems, and the second is why the code went rather than being patched:

      1. Its Planck factor used a hand-rolled hbar on a LINEAR frequency (ℏ·f, not h·f),
         inflating n̄ by 6.3×.
      2. r_c is the classical critical pump, which →0 in large-D. It is an artificial
         reference scale, not a threshold — so the quantity η was measured against did
         not mean what the surrounding prose said it meant.

    The Zhang rate equations remain the correct description of the ABOVE-threshold
    dynamics, and φ/χ still set the slope there. What they do not provide is the
    threshold, which is now reference-free.

    NOT MODELLED, deliberately: the ⟨n₀⟩ phonon count and the γ₀ coherence-lifetime
    enhancement that the old steady-state returned. Nothing outside this module read
    them (verified by grep across the tree, 2026-07-18) and reconstructing them under
    the new threshold would be inventing numbers no consumer asked for. If a future
    consumer needs ⟨n₀⟩, derive it then, from the quantum-pump treatment.

    REGIMES (Reimers et al. 2009) — classified on η, the order parameter:
    - r < 1     : 'thermal', no condensation
    - η < 0.3   : 'weak_condensate'  ← the biologically feasible regime we claim
    - η < 0.7   : 'strong_condensate'
    - else      : 'full_condensate'
    """

    def __init__(self, params: TubulinCascadeParameters):
        self.params = params
        self._eta = 0.0
        self._regime = 'thermal'

    def calculate_steady_state(self, pump_ratio: float) -> Dict:
        """
        Condensation state from the drive ratio.

        Parameters
        ----------
        pump_ratio : float
            r = P_met / P_c from PumpRateCalculator. DIMENSIONLESS — this is a power
            ratio, not the Hz pump rate the pre-B2 signature took.

        Returns
        -------
        dict with condensation_ratio (η), above_threshold, regime, and the drive
        diagnostics.
        """
        p = self.params
        r = float(pump_ratio) if pump_ratio is not None else 0.0

        above_threshold = r >= 1.0
        eta = (r - 1.0) / (r + 1.0) if above_threshold else 0.0
        eta = float(np.clip(eta, 0.0, 1.0))

        if not above_threshold:
            regime = 'thermal'
        elif eta < 0.3:
            regime = 'weak_condensate'
        elif eta < 0.7:
            regime = 'strong_condensate'
        else:
            regime = 'full_condensate'

        self._eta = eta
        self._regime = regime

        return {
            'condensation_ratio': eta,
            'pump_ratio': r,
            'above_threshold': above_threshold,
            'regime': regime,
            'n_bar': bose_einstein_occupation(p.omega_0, p.T_body),
        }


# =============================================================================
# STAGE 3: ENVIRONMENT MODULATION
# =============================================================================

class CondensationModulator:
    """
    Translate condensation state into physical effects on:
    1. Dimer formation rate (forward coupling)
    2. Protein conformational barriers (reverse coupling / CaMKII gating)
    
    PHYSICS:
    -------
    The condensed vibrational mode produces a coherent mechanical
    oscillation that modulates the local electrostatic environment.
    
    Below threshold: thermal vibrations → incoherent, weak effects
    Above threshold: condensed mode → coherent, strong effects
    
    The transition is sharp because Fröhlich condensation is a
    phase transition, not a linear scaling.
    """
    
    def __init__(self, params: TubulinCascadeParameters):
        self.params = params
        
    def calculate_modulation(self, 
                             condensation_state: Dict,
                             n_coherent_dimers: int,
                             k_agg_baseline: float,
                             phosphate_fraction: float = 1.0) -> Dict:
        """
        Calculate forward and reverse coupling effects.
        
        Parameters:
        ----------
        condensation_state : dict
            Output from FrohlichCondensation.calculate_steady_state()
        n_coherent_dimers : int
            Number of quantum coherent dimers (from Model 6)
        k_agg_baseline : float
            Baseline aggregation rate (M⁻¹s⁻¹)
        phosphate_fraction : float
            Fraction of phosphate available (0-1)
            
        Returns:
        -------
        dict with forward, reverse, and output sub-dicts
        """
        p = self.params
        eta = condensation_state['condensation_ratio']
        above_threshold = condensation_state['above_threshold']
        regime = condensation_state['regime']
        
        # === FORWARD COUPLING: condensation → dimer formation rate ===
        # Enhancement scales with condensation ratio
        # Below threshold: no enhancement (baseline chemistry)
        # Above threshold: scales with η (sigmoid provides natural saturation)
        if above_threshold:
            # Enhancement scales continuously with condensation ratio
            # No arbitrary cap — physics (substrate depletion, condensation saturation) provides natural limits
            enhancement_factor = p.enhancement_baseline + \
                eta * (1.0 / (1.0 + np.exp(-10.0 * (eta - 0.2)))) * 10.0
        else:
            enhancement_factor = p.enhancement_baseline
        
        # Apply substrate depletion
        if p.substrate_depletion_feedback:
            depletion = np.clip(phosphate_fraction, 0.1, 1.0)
            enhancement_factor *= depletion
        
        k_enhanced = k_agg_baseline * enhancement_factor
        
        forward_details = {
            'enhancement': enhancement_factor,
            'k_enhanced': k_enhanced,
            'k_baseline': k_agg_baseline,
            'em_field': 0.0,  # Backward compat — not used in cascade model
            'condensation_driven': True,
        }
        
        # === REVERSE COUPLING: dimers + condensation → protein modulation ===
        # This is the CaMKII barrier modulation pathway
        # Requires BOTH condensed mode AND sufficient dimers
        
        # n_coherent_dimers is already the entangled cluster size (largest_cluster from particle_metrics)
        # No additional discount — these dimers ARE the quantum-correlated subset
        dimer_field_kT = n_coherent_dimers * p.energy_per_dimer_kT * p.spatial_averaging_factor
        
        # Condensation amplifies dimer field — no arbitrary cap
        # Natural limit: eta ≤ 1.0, modulation_coupling ≤ 1.0, dimer_field bounded by actual dimer count
        if above_threshold:
            condensation_boost = 1.0 + eta * p.modulation_coupling
            energy_modulation_kT = dimer_field_kT * condensation_boost
        else:
            energy_modulation_kT = dimer_field_kT * 0.5  # Weak thermal contribution
        
        # Barrier reduction for CaMKII (electrostatic component ~15% of total barrier)
        barrier_reduction_kT = energy_modulation_kT * 0.15
        rate_enhancement = np.exp(min(barrier_reduction_kT, 50.0))  # Arrhenius
        
        # Above 20 kT threshold? (now emergent from condensation, not prescribed)
        above_20kT = energy_modulation_kT >= 20.0
        
        reverse_details = {
            'n_dimers': n_coherent_dimers,
            'regime': regime,
            'energy_kT_raw': dimer_field_kT,
            'energy_modulation_kT': energy_modulation_kT,
            'above_threshold': above_20kT,
            'barrier_reduction_kT': barrier_reduction_kT,
            'rate_enhancement': rate_enhancement,
            'condensation_boost': condensation_boost if above_threshold else 1.0,
        }
        
        # === FEEDBACK LOOP ===
        # Loop gain emerges from forward and reverse path physics — no prescribed damping
        forward_component = enhancement_factor - 1.0
        reverse_component = energy_modulation_kT / 20.0
        loop_gain = forward_component * reverse_component

        if p.substrate_depletion_feedback:
            loop_gain *= np.clip(phosphate_fraction, 0.1, 1.0)
        
        feedback_details = {
            'loop_gain': loop_gain,
            'stable': loop_gain < 1.0,
            'forward_component': forward_component,
            'reverse_component': reverse_component,
            'depletion_factor': np.clip(phosphate_fraction, 0.1, 1.0),
            'feedback_active': loop_gain > 0.01,
        }
        
        return {
            'forward': forward_details,
            'reverse': reverse_details,
            'feedback': feedback_details,
        }
    
    @staticmethod
    def _sigmoid(x: float, center: float = 0.5, steepness: float = 10.0) -> float:
        """Smooth sigmoid transition"""
        arg = steepness * (x - center)
        arg = np.clip(arg, -500, 500)
        return 1.0 / (1.0 + np.exp(-arg))


# =============================================================================
# INTEGRATED MODULE (drop-in replacement for EMCouplingModule)
# =============================================================================

class VibrationalCascadeModule:
    """
    Complete vibrational cascade coupling system.
    
    Drop-in replacement for EMCouplingModule with identical
    update() signature and output dict structure.
    
    INTEGRATES:
    1. PumpRateCalculator — tryptophan EM field → vibrational pump rate
    2. FrohlichCondensation — pump rate → condensation state (Zhang 2019)
    3. CondensationModulator — condensation → dimer/protein effects
    
    Usage:
    ------
    >>> module = VibrationalCascadeModule(params)
    >>> state = module.update(
    ...     em_field_trp=1.4e9,       # From tryptophan module
    ...     n_coherent_dimers=50,      # From Model 6 quantum system
    ...     k_agg_baseline=8e5,        # From Model 6 chemistry
    ...     phosphate_fraction=0.8     # From Model 6 state
    ... )
    >>> k_enhanced = state['output']['k_agg_enhanced']
    >>> protein_mod = state['output']['protein_modulation_kT']
    """
    
    def __init__(self, params):
        """
        Initialize the cascade module.
        
        Parameters:
        ----------
        params : Model6Parameters (or similar)
            If params has a 'cascade' attribute, use TubulinCascadeParameters.
            Otherwise, construct defaults. This allows backward compatibility
            with existing Model6Parameters that have em_coupling attribute.
        """
        # Extract or create cascade parameters
        if hasattr(params, 'cascade'):
            self.cascade_params = params.cascade
        else:
            self.cascade_params = TubulinCascadeParameters()
            
        # Initialize stages
        self.pump_calculator = PumpRateCalculator(self.cascade_params)
        self.condensation = FrohlichCondensation(self.cascade_params)
        self.modulator = CondensationModulator(self.cascade_params)
        
        # State tracking (backward compatible with EMCouplingModule)
        self.state = {
            'forward_enhancement': 1.0,
            'reverse_modulation_kT': 0.0,
            'loop_gain': 0.0,
            'stable': True,
        }
        
        # Cascade-specific tracking
        self.cascade_state = {
            'pump_rate': 0.0,
            'condensation_ratio': 0.0,
            'regime': 'thermal',
            'above_condensation_threshold': False,
        }
        
        logger.info("=" * 70)
        logger.info("VIBRATIONAL CASCADE MODULE (Fröhlich condensation)")
        logger.info("=" * 70)
        cp = self.cascade_params
        logger.info(f"  Collective mode ω₀={cp.omega_0/1e6:.3f} MHz (linear f), Q={cp.Q:.1f}, "
                     f"D={cp.D_modes} modes")
        logger.info(f"  Dissipation φ={cp.phi_dissipation/1e6:.3f} MHz (=ω₀/Q), "
                     f"Redistribution χ={cp.chi_redistribution/1e3:.3f} kHz (slope only)")
        logger.info(f"  Critical power P_c={self.pump_calculator.critical_power()*1e15:.2f} fW "
                     f"(threshold n_ex = n̄_s; reference-free)")
        logger.info(f"  Drive: per-spine metabolic power, NOT aggregated (backbone aggregates)")

    def update(self,
               p_met_W: float,
               n_coherent_dimers: int,
               k_agg_baseline: float,
               phosphate_fraction: float = 1.0,
               protein_type: str = 'generic') -> Dict:
        """
        Update complete cascade coupling state.

        Parameters:
        ----------
        p_met_W : float
            Metabolic power delivered to THIS spine's lattice segment, in Watts, from
            compute_metabolic_power(E_invasion, ca_open_fraction, p_active_max_W).
            This is the drive. It is NOT aggregated over neighbours — that is the one
            deliberate difference from the backbone pump, which sums coupled spines.
        n_coherent_dimers : int
            Number of quantum coherent dimers from Model 6
        k_agg_baseline : float
            Baseline aggregation rate from Model 6 (M⁻¹s⁻¹)
        phosphate_fraction : float
            Fraction of phosphate available (0-1)
        protein_type : str
            Backward compatibility (unused in cascade model)

        Returns:
        -------
        dict with:
            'forward': Forward coupling results
            'reverse': Reverse coupling results
            'feedback': Loop dynamics
            'state': Internal state
            'output': {
                'k_agg_enhanced': enhanced formation rate,
                'protein_modulation_kT': barrier modulation energy,
                'above_threshold': bool,
                'feedback_active': bool,
            }
            'cascade': Fröhlich-specific diagnostics

        RETIRED PARAMETERS (B2, 2026-07-18): `em_field_trp` and `collective_field_kT`.
        The drive is metabolic power, not the tryptophan field — the same call already
        made for the backbone in Step B ("Drive is metabolic P_met, NOT
        collective_field_kT"). Converting a kT field energy into a pump rate required the
        r_at_E_ref/kT_ref calibration constants, which is precisely the fiction B2
        retires; there is no honest kT→power conversion to keep.
        """
        # === STAGE 1: DRIVE RATIO r = P_met / P_c ===
        pump_result = self.pump_calculator.calculate_pump_rate(p_met_W)

        # === STAGE 2: FRÖHLICH CONDENSATION ===
        cond_result = self.condensation.calculate_steady_state(pump_result['pump_ratio'])

        # === STAGE 3: ENVIRONMENT MODULATION ===
        mod_result = self.modulator.calculate_modulation(
            condensation_state=cond_result,
            n_coherent_dimers=n_coherent_dimers,
            k_agg_baseline=k_agg_baseline,
            phosphate_fraction=phosphate_fraction,
        )
        
        # === UPDATE INTERNAL STATE ===
        self.state = {
            'forward_enhancement': mod_result['forward']['enhancement'],
            'reverse_modulation_kT': mod_result['reverse']['energy_modulation_kT'],
            'loop_gain': mod_result['feedback']['loop_gain'],
            'stable': mod_result['feedback']['stable'],
        }
        
        # 'n0_phonons', 'lifetime_enhancement' and 'r_c' are NOT reported any more.
        # r_c was the retired artificial reference scale; n0/lifetime came from the Zhang
        # steady-state quadratic that the quantum-pump treatment replaces. No consumer
        # outside this module ever read them (grep-verified across the tree, 2026-07-18),
        # so they are dropped rather than reconstructed as plausible-looking numbers.
        self.cascade_state = {
            'pump_power_W': pump_result['pump_power_W'],
            'critical_power_W': pump_result['critical_power_W'],
            'pump_ratio': pump_result['pump_ratio'],
            'condensation_ratio': cond_result['condensation_ratio'],
            'regime': cond_result['regime'],
            'above_condensation_threshold': cond_result['above_threshold'],
        }
        
        # === ASSEMBLE OUTPUT (backward compatible) ===
        return {
            'forward': mod_result['forward'],
            'reverse': mod_result['reverse'],
            'feedback': mod_result['feedback'],
            'state': self.state,
            'output': {
                'k_agg_enhanced': mod_result['forward']['k_enhanced'],
                'protein_modulation_kT': mod_result['reverse']['energy_modulation_kT'],
                'above_threshold': mod_result['reverse']['above_threshold'],
                'feedback_active': mod_result['feedback']['feedback_active'],
            },
            'cascade': self.cascade_state,
        }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from model6_parameters import compute_metabolic_power, P_BASAL_W, DendriticBackboneParameters

    print("=" * 80)
    print("VIBRATIONAL CASCADE MODULE — SELF-TEST")
    print("Threshold: Wang & Wang 2022 (n_ex = n̄_s). Dynamics above it: Zhang 2019.")
    print("=" * 80)

    params = TubulinCascadeParameters()
    module = VibrationalCascadeModule.__new__(VibrationalCascadeModule)
    module.cascade_params = params
    module.pump_calculator = PumpRateCalculator(params)
    module.condensation = FrohlichCondensation(params)
    module.modulator = CondensationModulator(params)
    module.state = {}
    module.cascade_state = {}

    P_c = module.pump_calculator.critical_power()

    print(f"\n--- Fröhlich Parameters (collective MT condensing mode) ---")
    print(f"  ω₀ = {params.omega_0/1e6:.3f} MHz  (linear f; consumed as h·f)")
    print(f"  Q = {params.Q:.1f}")
    print(f"  D = {params.D_modes} modes")
    print(f"  φ = {params.phi_dissipation/1e6:.3f} MHz  (= ω₀/Q)")
    print(f"  χ = {params.chi_redistribution/1e3:.3f} kHz  (= {params.chi_ratio}·φ; slope only)")
    n_bar = bose_einstein_occupation(params.omega_0, params.T_body)
    print(f"  n̄_s(T=310K) = {n_bar:.4g}")
    print(f"  P_c = {P_c*1e15:.3f} fW   (reference-free threshold, n_ex = n̄_s)")

    # === TEST 1: the order parameter through threshold ===
    print(f"\n--- TEST 1: η through threshold (η = (r−1)/(r+1)) ---")
    print(f"{'r = P/P_c':<12} {'η':<10} {'Regime':<20}")
    print("-" * 44)
    for r in [0.0, 0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]:
        cond = module.condensation.calculate_steady_state(r)
        print(f"{r:<12.2f} {cond['condensation_ratio']:<10.4f} {cond['regime']:<20}")

    # === TEST 2: does the per-synapse site agree with the backbone on P_c? ===
    # Same mode, same Q => same critical power. This is the invariant B2 exists to
    # establish, asserted here so the module's own self-test would catch a re-fork.
    print(f"\n--- TEST 2: per-synapse vs backbone critical power ---")
    bp = DendriticBackboneParameters()
    omega_ang_bb = 2.0 * np.pi * bp.omega_0
    P_c_backbone = bose_einstein_occupation(bp.omega_0) * hbar * omega_ang_bb**2 / bp.Q
    print(f"  per-synapse P_c = {P_c*1e15:.4f} fW   (ω₀={params.omega_0/1e6:.3f} MHz, Q={params.Q})")
    print(f"  backbone    P_c = {P_c_backbone*1e15:.4f} fW   (ω₀={bp.omega_0/1e6:.3f} MHz, Q={bp.Q})")
    same_mode = abs(params.omega_0 - bp.omega_0) / bp.omega_0 < 1e-9
    same_P_c = abs(P_c - P_c_backbone) / P_c_backbone < 1e-9
    print(f"  same mode: {'✓' if same_mode else '✗'}    same P_c: {'✓' if same_P_c else '✗'}")

    # === TEST 3: full integration, driven by metabolic power ===
    # Drive levels are the physical range from the Step-B calc: rest is basal only;
    # active is basal + E_invasion·ca_open·p_active_max. NOTHING here is calibrated to
    # put a particular condition above threshold — the drive is computed and the
    # verdict falls where it falls.
    print(f"\n--- TEST 3: Full Module Integration (metabolic drive) ---")
    print(f"{'condition':<34} {'P_met (fW)':<12} {'r':<8} {'η':<8} {'regime':<18} {'above?':<7}")
    print("-" * 92)

    conditions = [
        ("rest (E_inv=0, ca_open=0)",        0.0, 0.0),
        ("partial (E_inv=0.5, ca_open=0.5)", 0.5, 0.5),
        ("active (E_inv=1.0, ca_open=0.84)", 1.0, 0.84),
    ]
    for label, e_inv, ca_open in conditions:
        p_met = compute_metabolic_power(e_inv, ca_open, bp.p_active_max_W)
        st = module.update(
            p_met_W=p_met,
            n_coherent_dimers=50,
            k_agg_baseline=8e5,
            phosphate_fraction=0.8,
        )
        c = st['cascade']
        print(f"{label:<34} {p_met*1e15:<12.3f} {c['pump_ratio']:<8.3f} "
              f"{c['condensation_ratio']:<8.4f} {c['regime']:<18} "
              f"{str(c['above_condensation_threshold']):<7}")

    print(f"\n{'='*80}")
    print("NOTE ON WHAT THIS SELF-TEST NO LONGER CLAIMS")
    print(f"{'='*80}")
    print("  The pre-B2 self-test asserted 'MT+ (22.1 kT) above threshold, MT- (12.8 kT)")
    print("  below' and printed a '20 kT threshold emerges' tick. That was not a")
    print("  measurement: 22.1 was kT_ref itself, so the test restated its own input.")
    print("  It is gone, and nothing here replaces it with a pass/fail tick: TEST 3")
    print("  computes the drive and prints where the verdict falls.")
    print()
    print("  On single-synapse criticality — state this carefully. A single spine is")
    print("  subcritical AT REST (r≈0.04) and at the E_invasion a short run reaches:")
    print("  the June-7 figure 'P_met maxes ~17 fW, r peaks 0.803' reproduces exactly")
    print("  at E_inv=0.495, ca_open=0.55, which is what the envelope had climbed to by")
    print("  30 s. It is NOT a structural ceiling — at sustained full invasion the same")
    print("  arithmetic gives 51.2 fW, r=2.38, above threshold on its own. What the")
    print("  backbone's aggregation buys is crossing at LOWER per-spine drive, not")
    print("  crossing at all. Do not read 'subcritical by design' as 'cannot cross'.")
    print(f"{'='*80}")
