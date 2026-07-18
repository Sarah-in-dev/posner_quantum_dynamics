"""
Model 6 Quantum Biology — Sweep Dimensions
===========================================
Defines the non-emergent parameter space as RefinedDimension instances
for 3-wise covering array generation.

Four groups:
  Q1_DIMENSIONS       — tryptophan superradiance / backbone parameters
  Q2_DIMENSIONS       — calcium phosphate dimer / coherence parameters
  NETWORK_DIMENSIONS  — multi-synapse coordination parameters
  STIMULUS_DIMENSIONS — CA1 theta-burst input scenario parameters

Parameters that are emergent (condensation threshold, commit rate, loop gain)
are NOT included — those are outputs, not inputs.

⚠ READ INERT_DIMENSIONS BELOW BEFORE INTERPRETING ANY SWEEP RESULT ⚠
====================================================================
Measured 2026-07-18 (`sweep/dimension_consumer_audit.py`, PO-6a Unit 1): **9 of 19
read-traceable dimensions are INERT** — swept by sweep_runner.py and read by nothing.

A sweep over an inert dimension returns a FLAT RESPONSE. A flat response over a swept
parameter reads as "this parameter does not matter" — a physical null. It is not. It is a
wiring gap wearing the costume of a result. Two of the inert dimensions are declared
importance="critical", including the dimer coherence lifetime.

Each inert dimension is annotated at its definition below and registered in
INERT_DIMENSIONS with its measured reason. Do not report a flat response over any of them
as physics until it is resolved.
"""

from typing import Dict

from sweep.talon_core.permutation_engine import RefinedDimension


# ── INERT REGISTRY (measured, not assumed) ───────────────────────────────────
#
# dim_id -> why it is inert. Machine-readable on purpose: a comment can be skimmed past,
# a registry can be asserted against. sweep_runner.py warns loudly when a vector includes
# any of these.
#
# Method: read-tracing via __getattribute__ on the parameter dataclasses and the scenario,
# driving the real model. reads == 0 is definitive — nothing looked at the value. The audit
# carries three controls, including calibration against known-live/known-dead attributes,
# so the verdict demonstrably distinguishes its outcomes.
#
# THREE MECHANISMS, needing three different fixes — do not treat these as one bug:
#   (1) NO CONSUMER          nothing reads the attribute at all
#   (2) CONSUMER HARDCODED   the physics DOES use the quantity, but as a literal the
#                            parameter system cannot reach — and the literal disagrees
#                            with the declared parameter value
#   (3) OVERRIDDEN/DISCLAIMED the code explicitly implements a different mechanism
INERT_DIMENSIONS: Dict[str, str] = {
    "q1_d_modes":
        "(1) NO CONSUMER. dendritic_backbone.D_modes is a declaration with zero reads in "
        "multi_synapse_network.py. It does not enter P_c — only omega_0 and Q do (B2).",
    "q1_phi_dissipation":
        "(1) NO CONSUMER. dendritic_backbone.phi_dissipation: declaration only, zero reads.",
    "q1_chi_redistribution":
        "(1) NO CONSUMER. dendritic_backbone.chi_redistribution: declaration only, zero reads.",
    "q1_kT_per_modulation":
        "(1) NO CONSUMER. dendritic_backbone.kT_per_modulation_unit: declaration only.",
    # q2_t2_p31 — RESOLVED 2026-07-18 (MO ruling 006), no longer inert. It was
    # CONSUMER-HARDCODED: the live dimer singlet lifetime was a literal T_singlet_P31 = 216.0
    # duplicated in dimer_particles.py and quantum_coherence.py, while the parameter this
    # dimension writes (quantum.T_singlet_dimer) said 500.0 and was read only by an orphan.
    # Fixed one-way — the PARAMETER moved to the physics (500 -> 216), never the reverse —
    # and both literals now read the single field. Demonstrated live: driving
    # T_singlet_dimer 50/216/500 moves mean P_S 0.998512/0.998893/0.998949, monotonic in the
    # physically correct direction, with de-duplication verified bit-identical to pre-change.
    # 216 s is load-bearing for quantum-system-canonical §2.2 — do NOT retune it.
    "q2_j_coupling_hz":
        "(1) NO CONSUMER, and the parameter is scale-mismatched to the thing it names. "
        "quantum.J_intrinsic_dimer = 15.0 Hz has ONE write (sweep_runner.py:73) and ZERO "
        "reads anywhere in the tree. J-coupling itself IS live, but via a different route: "
        "an ATP-derived field (atp_system.py:296-339, params atp.J_PO_free / J_PP_atp) plus "
        "per-dimer j_couplings_intra drawn from a hardcoded N(0.15, 0.15) at "
        "dimer_particles.py:49 — mean 0.15, i.e. 100x below this parameter's 15.0 Hz. "
        "Re-targeting this dimension means choosing WHICH J is meant. Physics call. ROUTED.",
    "q2_k_agg_baseline":
        "*** DELETE-VERDICT (MO ruling 012 §2), HELD behind the isotope gate — goes out in "
        "one batch with the orphan deletions, not as a loose one-off. *** WRONG UNITS, not "
        "merely wrong scale: this dimension's values [0.001, 0.005, 0.01, 0.05] are "
        "FIRST-ORDER dissolution rates (s^-1) — two of them are exactly the grounded and the "
        "retired K_CLASSICAL — while the aggregation constant k_base is SECOND-ORDER "
        "(~1.9e4 M^-1 s^-1, = productive_fraction x Smoluchowski [GROUNDED]). The dimension "
        "duplicates q2_k_classical. sweep_runner.py:92 now RAISES rather than silently "
        "skipping (ruling 012 §4); 'fixing the guard' by assigning to k_base is refused for "
        "the record — it injects a dissolution rate into an aggregation constant and yields "
        "a smooth, plausible, WRONG response curve. If an aggregation sensitivity sweep is "
        "wanted later it is a NEW declaration bracketing k_base, not a repair — and whoever "
        "declares it must state up front whether it is sensitivity analysis or a value "
        "choice, because productive_fraction is [LOCKED] 'never tuned to a target dimer "
        "count'.",
    "stim_ca_amplitude":
        "*** DELETE-VERDICT (MO ruling 012 §3), HELD with the batch. DO NOT WIRE IT. *** "
        "Wiring this would reinstate a named anti-pattern. Calcium amplitude is now a "
        "DERIVED quantity — the near-mouth nanodomain peak from a closed-form point-source "
        "steady state (Naraghi & Neher 1997), which explicitly 'replaces the prior "
        "calibrated 0.5 uM/channel snapshot' (quantum-system-canonical:82). A dimension that "
        "SETS calcium amplitude directly re-introduces exactly what the calcium grounding "
        "retired. _run_epoch's docstring ('Calcium enters via voltage-gated channel physics, "
        "not direct injection') is the CORRECT side of this contradiction; the dimension is "
        "the stale side.",
    # stim_burst_duration_ms — RESOLVED 2026-07-18 (MO ruling 012 §3), no longer inert.
    # It was a HARDCODED OVERRIDE: _run_epoch fixed spikes_per_burst = 4 at 100 Hz = 40 ms
    # and never consulted the scenario value. Burst duration now sets the number of 100 Hz
    # pulses (the physical meaning of a longer burst); the 100 Hz train and 2 ms
    # depolarization are the invariants. The dataclass default moved 50.0 -> 40.0 one-way,
    # to the value that actually ran AND the grounded one (4 pulses at 100 Hz is the
    # canonical theta-burst unit; 50 ms was a round number) — same direction as ruling 006.
    # Verified bit-identical at the default before and after.
}


# ── Q1: Tryptophan superradiance & backbone ──────────────────────────────────

Q1_DIMENSIONS = [
    RefinedDimension(
        dim_id="q1_n_tryptophan",
        variable="n_tryptophan",
        category="threshold",
        values=[50, 100, 200, 500],
        value_labels=["50", "100", "200", "500"],
        source_file="model6_parameters.py",
        source_function="EMTryptophanParameters",
        source_line=0,
        condition="tryptophan lattice size — controls collective dipole scaling (√N)",
        importance="critical",
    ),
    RefinedDimension(
        dim_id="q1_f_coherent_base",
        variable="f_coherent_base",
        category="threshold",
        values=[0.04, 0.06, 0.08, 0.10],
        value_labels=["0.04", "0.06", "0.08", "0.10"],
        source_file="model6_parameters.py",
        source_function="EMTryptophanParameters",
        source_line=0,
        condition="base coherent fraction without backbone modulation (Babcock 2024 range)",
        importance="critical",
    ),
    RefinedDimension(
        dim_id="q1_d_modes",
        variable="D_modes",
        category="threshold",
        values=[20, 50, 100, 200],
        value_labels=["20", "50", "100", "200"],
        source_file="model6_parameters.py",
        source_function="DendriticBackboneParameters",
        source_line=0,
        condition="backbone lattice vibrational modes",
        importance="high",
    ),
    RefinedDimension(
        dim_id="q1_phi_dissipation",
        variable="phi_dissipation",
        category="threshold",
        values=[4.0, 8.0, 16.0, 32.0],
        value_labels=["4GHz", "8GHz", "16GHz", "32GHz"],
        source_file="model6_parameters.py",
        source_function="DendriticBackboneParameters",
        source_line=0,
        condition="backbone lattice loss rate (Zhang/Agarwal/Scully 2019)",
        importance="high",
    ),
    RefinedDimension(
        dim_id="q1_chi_redistribution",
        variable="chi_redistribution",
        category="threshold",
        values=[0.03, 0.06, 0.12, 0.24],
        value_labels=["0.03GHz", "0.06GHz", "0.12GHz", "0.24GHz"],
        source_file="model6_parameters.py",
        source_function="DendriticBackboneParameters",
        source_line=0,
        condition="backbone mode coupling — controls condensation sharpness",
        importance="high",
    ),
    RefinedDimension(
        dim_id="q1_kT_per_modulation",
        variable="kT_per_modulation_unit",
        category="threshold",
        values=[0.75, 1.5, 3.0, 6.0],
        value_labels=["0.75", "1.5", "3.0", "6.0"],
        source_file="model6_parameters.py",
        source_function="DendriticBackboneParameters",
        source_line=0,
        condition="spine-to-backbone coupling efficiency (10% default from geometry)",
        importance="high",
    ),
]


# ── Q2: Calcium phosphate dimer coherence ────────────────────────────────────

Q2_DIMENSIONS = [
    RefinedDimension(
        dim_id="q2_k_classical",
        variable="k_classical",
        category="threshold",
        values=[0.01, 0.05, 0.10, 0.20],
        value_labels=["0.01/s", "0.05/s", "0.10/s", "0.20/s"],
        source_file="ca_triphosphate_complex.py",
        source_function="CalciumPhosphateDimerization",
        source_line=0,
        condition="classical dissolution rate at P_S=0.25 (Fisher 2015)",
        importance="critical",
    ),
    # ── q2_t2_p31 — SENSITIVITY ANALYSIS, **NOT VALUE SELECTION** ───────────────────────
    #
    # READ THIS BEFORE INTERPRETING ANY SWEEP OVER THIS DIMENSION.
    #
    # The grounded value is **T_singlet_dimer = 216 s** (Agarwal; dipolar relaxation only).
    # It is LOAD-BEARING for `quantum-system-canonical` §2.2 and is NOT up for revision by a
    # sweep. This bracket exists to measure **how the model degrades away from 216 s**, not to
    # search for a better value. See the one-way-fix note at `model6_parameters.py:409-411`:
    # *"the parameter was moved to match the physics. Do NOT adjust the 216 s to match a
    # declared number — that is tuning a constant to reach an outcome."*
    # **A result from this sweep may never be cited as choosing a coherence time.**
    #
    # Bracket: symmetric in log space about 216 s (x1/2, x3/4, x1, x3/2, x2), approved
    # MO ruling 017. It REPLACES [50, 100, 200, 500], whose top arm was the **retired** 500 s
    # — a configuration the program has explicitly rejected (crossing 247.6 s), which a sweep
    # would have sampled with nothing warning.
    #
    # WHERE THE PHYSICS STOPS HOLDING — annotate, do not assume.
    # P_S(t) = 0.25 + 0.75·exp(−t/T) toward the thermal floor 0.25 (dimer_particles.py:283,
    # :323-332); a pair clears the Werner bound F = P_S² > 0.5 while P_S > 1/√2. So the
    # crossing is t = 0.49516·T. The ontology's coherence window is ~100–200 s.
    #
    #     T = 108 s  ->  crossing  53.5 s   OUTSIDE (below band)
    #     T = 162 s  ->  crossing  80.2 s   OUTSIDE (below band)
    #     T = 216 s  ->  crossing 107.0 s   INSIDE   <-- the grounded value
    #     T = 324 s  ->  crossing 160.4 s   INSIDE
    #     T = 432 s  ->  crossing 213.9 s   OUTSIDE (above band)
    #
    # **THREE of the five arms sit outside the band, not one.** Ruling 017 flagged the 432 s
    # arm; the two low arms are outside as well, below rather than above. That is correct for
    # a sensitivity sweep — sampling where the correspondence degrades is the point — but
    # only the 216 s and 324 s arms are configurations in which §2.2's central correspondence
    # holds. **Do not report an aggregate over all five arms as if it described the grounded
    # model.** Formula above is derived from the code, so re-derive rather than trust these
    # numbers if the decay law changes.
    RefinedDimension(
        dim_id="q2_t2_p31",
        variable="T_singlet_dimer",
        category="threshold",
        values=[108.0, 162.0, 216.0, 324.0, 432.0],
        value_labels=["108s (x0.5, outside)", "162s (x0.75, outside)",
                      "216s GROUNDED", "324s (x1.5, inside)", "432s (x2, outside)"],
        source_file="model6_parameters.py",
        source_function="QuantumParameters",
        source_line=409,
        condition=("dimer singlet coherence lifetime — the eligibility-trace window. "
                   "SENSITIVITY ANALYSIS ONLY about the grounded 216 s; NOT value selection. "
                   "Werner crossing = 0.4952*T; band 100-200 s; only the 216 s and 324 s arms "
                   "sit inside it. See the block comment above this declaration."),
        importance="critical",
    ),
    RefinedDimension(
        dim_id="q2_j_coupling_hz",
        variable="j_coupling_hz",
        category="threshold",
        values=[0.5, 1.0, 2.0, 5.0],
        value_labels=["0.5Hz", "1.0Hz", "2.0Hz", "5.0Hz"],
        source_file="quantum_coherence.py",
        source_function="DimerCoherenceModule",
        source_line=0,
        condition="J-coupling between P31 nuclei — controls singlet protection",
        importance="high",
    ),
    RefinedDimension(
        dim_id="q2_phosphate_initial",
        variable="phosphate_structural_initial",
        category="threshold",
        values=[0.0001, 0.001, 0.005, 0.010],
        value_labels=["0.1mM", "1mM", "5mM", "10mM"],
        source_file="atp_system.py",
        source_function="PhosphateSpeciation",
        source_line=0,
        condition="structural phosphate pool — material ceiling on dimer count",
        importance="high",
    ),
    RefinedDimension(
        dim_id="q2_k_agg_baseline",
        variable="k_agg_baseline",
        category="threshold",
        values=[0.001, 0.005, 0.01, 0.05],
        value_labels=["0.001", "0.005", "0.01", "0.05"],
        source_file="ca_triphosphate_complex.py",
        source_function="CalciumPhosphateDimerization",
        source_line=0,
        condition="baseline dimer aggregation rate constant",
        importance="high",
    ),
]


# ── Network: Multi-synapse coordination ──────────────────────────────────────

NETWORK_DIMENSIONS = [
    RefinedDimension(
        dim_id="net_n_synapses",
        variable="n_synapses",
        category="threshold",
        values=[3, 5, 10, 20],
        value_labels=["3", "5", "10", "20"],
        source_file="multi_synapse_network.py",
        source_function="MultiSynapseNetwork",
        source_line=0,
        condition="co-active synapses on shared dendritic segment",
        importance="critical",
    ),
    RefinedDimension(
        dim_id="net_mt_invaded_fraction",
        variable="mt_invaded_fraction",
        category="threshold",
        values=[0.2, 0.4, 0.6, 0.8, 1.0],
        value_labels=["20%", "40%", "60%", "80%", "100%"],
        source_file="multi_synapse_network.py",
        source_function="set_microtubule_invasion",
        source_line=0,
        condition="fraction of synapses with MT invasion (Hu 2008, activity-dependent)",
        importance="high",
    ),
    RefinedDimension(
        dim_id="net_spacing_um",
        variable="spacing_um",
        category="threshold",
        values=[0.5, 1.0, 2.0, 5.0],
        value_labels=["0.5um", "1.0um", "2.0um", "5.0um"],
        source_file="multi_synapse_network.py",
        source_function="MultiSynapseNetwork",
        source_line=0,
        condition="inter-synapse spacing along dendrite",
        importance="medium",
    ),
]


# ── Stimulus: CA1 theta-burst input scenarios ────────────────────────────────

STIMULUS_DIMENSIONS = [
    RefinedDimension(
        dim_id="stim_ca_amplitude",
        variable="ca_amplitude",
        category="threshold",
        values=[5e-6, 1e-5, 5e-5, 1e-4],
        value_labels=["5uM", "10uM", "50uM", "100uM"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="peak calcium per burst (direct injection)",
        importance="critical",
    ),
    RefinedDimension(
        dim_id="stim_theta_cycles",
        variable="theta_cycles_per_traversal",
        category="threshold",
        values=[8, 12, 16],
        value_labels=["8", "12", "16"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="theta cycles per place field traversal (8 Hz bursts)",
        importance="high",
    ),
    RefinedDimension(
        dim_id="stim_n_traversals",
        variable="n_traversals",
        category="threshold",
        values=[1, 3, 6, 10],
        value_labels=["1", "3", "6", "10"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="number of place field traversals (laps)",
        importance="critical",
    ),
    RefinedDimension(
        dim_id="stim_inter_traversal_s",
        variable="inter_traversal_interval_s",
        category="threshold",
        values=[30.0, 45.0, 60.0, 90.0],
        value_labels=["30s", "45s", "60s", "90s"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="inter-traversal interval — coherence decay window between laps",
        importance="high",
    ),
    RefinedDimension(
        dim_id="stim_burst_duration_ms",
        variable="burst_duration_ms",
        category="threshold",
        values=[20, 50, 100, 200],
        value_labels=["20ms", "50ms", "100ms", "200ms"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="duration of each calcium burst",
        importance="medium",
    ),
    RefinedDimension(
        dim_id="stim_theta_period_ms",
        variable="theta_period_ms",
        category="threshold",
        values=[100, 125, 150, 200],
        value_labels=["100ms", "125ms", "150ms", "200ms"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="theta cycle period (5-10 Hz range, CA1 canonical ~125ms)",
        importance="high",
    ),
    RefinedDimension(
        dim_id="stim_dopamine_delay",
        variable="dopamine_delay_s",
        category="threshold",
        values=[0.3, 0.5, 1.0, 2.0],
        value_labels=["0.3s", "0.5s", "1.0s", "2.0s"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="dopamine arrival delay post-burst (reward timing)",
        importance="critical",
    ),
    RefinedDimension(
        dim_id="stim_silence_duration",
        variable="silence_duration_s",
        category="threshold",
        values=[10.0, 30.0, 60.0, 120.0],
        value_labels=["10s", "30s", "60s", "120s"],
        source_file="sweep/theta_burst_scenario.py",
        source_function="ThetaBurstScenario",
        source_line=0,
        condition="post-stimulus silence window — maps to eligibility trace window",
        importance="critical",
    ),
]


# ── Full parameter space ─────────────────────────────────────────────────────

ALL_DIMENSIONS = (
    Q1_DIMENSIONS
    + Q2_DIMENSIONS
    + NETWORK_DIMENSIONS
    + STIMULUS_DIMENSIONS
)

CRITICAL_DIMENSIONS = [d for d in ALL_DIMENSIONS if d.importance == "critical"]
HIGH_DIMENSIONS     = [d for d in ALL_DIMENSIONS if d.importance in ("critical", "high")]

# Dimensions measured to reach a consumer. NOTE the standard: "reached" means the value is
# READ on the live path, which is necessary but NOT sufficient for physical effect — a read
# could be a log line. Membership here is not a claim that the dimension moves a result.
LIVE_DIMENSIONS = [d for d in ALL_DIMENSIONS if d.dim_id not in INERT_DIMENSIONS]


def inert_dims_in(dimensions) -> list:
    """Return the dimensions in `dimensions` that are known-inert."""
    return [d for d in dimensions if d.dim_id in INERT_DIMENSIONS]


def assert_no_inert(dimensions) -> None:
    """Raise if any known-inert dimension is present.

    For callers that want a hard stop rather than sweep_runner's warning — e.g. a script
    about to publish a response curve. Deliberately NOT called by default: silently
    dropping inert dimensions would hide the defect instead of surfacing it.
    """
    bad = inert_dims_in(dimensions)
    if bad:
        raise ValueError(
            "Refusing to sweep known-INERT dimensions — a flat response over these reads "
            "as a physical null but is a wiring gap:\n" +
            "\n".join(f"  {d.dim_id}: {INERT_DIMENSIONS[d.dim_id]}" for d in bad)
        )


if __name__ == "__main__":
    print(f"Total dimensions:    {len(ALL_DIMENSIONS)}")
    print(f"Critical:            {len(CRITICAL_DIMENSIONS)}")
    print(f"Critical + high:     {len(HIGH_DIMENSIONS)}")
    print()
    for group_name, group in [
        ("Q1 (tryptophan/backbone)", Q1_DIMENSIONS),
        ("Q2 (dimer/coherence)",     Q2_DIMENSIONS),
        ("Network",                  NETWORK_DIMENSIONS),
        ("Stimulus (theta-burst)",   STIMULUS_DIMENSIONS),
    ]:
        print(f"  {group_name}")
        for d in group:
            print(f"    [{d.importance:8s}] {d.dim_id:30s} {d.values}")