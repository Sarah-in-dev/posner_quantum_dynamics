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
"""

from sweep.talon_core.permutation_engine import RefinedDimension


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