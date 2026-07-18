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
        "(1) NO CONSUMER via a SILENT GUARD, and the declared values are off by ~1e6. "
        "sweep_runner.py:92 guards on hasattr(dimerization,'k_agg'), which is False — the "
        "attribute is k_base — so the write never executes. But re-pointing at k_base is NOT "
        "mechanical: k_base = 18918.67 M^-1 s^-1, while this dimension's values are "
        "[0.001, 0.005, 0.01, 0.05] — which match k_classical (0.005) exactly. The values "
        "were written for a DISSOLUTION rate, duplicating q2_k_classical. ROUTED.",
    "stim_ca_amplitude":
        "(3) MECHANISM DISCLAIMED BY THE CODE. This dimension's own text says 'peak calcium "
        "per burst (direct injection)', but _run_epoch's docstring "
        "(theta_burst_scenario.py:107) states: 'Calcium enters via voltage-gated channel "
        "physics, not direct injection.' The scenario never reads ca_amplitude. Either wire "
        "an injection path or DELETE the dimension — leaving it asserts a mechanism the code "
        "explicitly disclaims.",
    "stim_burst_duration_ms":
        "(3) HARDCODED OVERRIDE. theta_burst_scenario.py:_run_epoch fixes burst length at "
        "spikes_per_burst * spike_period = 4 * 10 ms = 40 ms. The swept values "
        "[20, 50, 100, 200] ms are never consulted.",
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
    RefinedDimension(
        dim_id="q2_t2_p31",
        variable="T_singlet_dimer",
        category="threshold",
        values=[50.0, 100.0, 200.0, 500.0],
        value_labels=["50s", "100s", "200s", "500s"],
        source_file="model6_parameters.py",
        source_function="QuantumParameters",
        source_line=0,
        condition="dimer singlet coherence lifetime — controls eligibility trace window (current default 500s)",
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