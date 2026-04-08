"""
Quantum Biology Sweep Runner
==============================
Executes TestVectors from the covering array against model6_core.py,
stores results, and produces a summary report.

Usage (smoke test — 10 vectors, critical dimensions only):
    python sweep/sweep_runner.py --mode smoke

Usage (full sweep — all dimensions, 3-wise coverage):
    python sweep/sweep_runner.py --mode full --coverage 3

Usage (critical only, 2-wise):
    python sweep/sweep_runner.py --mode critical --coverage 2

Results written to sweep/results/
"""

import sys
import os
import json
import time
import argparse
import logging
import traceback
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add model root to path (sweep/ lives inside the quantum repo)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from sweep.talon_core.permutation_engine import (
    CoveringArrayGenerator, TestVectorBuilder, TestSuite
)
from sweep.quantum_dimensions import (
    ALL_DIMENSIONS, CRITICAL_DIMENSIONS, HIGH_DIMENSIONS
)
from sweep.theta_burst_scenario import scenario_from_vector

logging.basicConfig(level=logging.WARNING)


# ── Parameter application ────────────────────────────────────────────────────

def apply_vector_to_params(params, values: Dict[str, Any]) -> None:
    """
    Apply a TestVector's values to a Model6Parameters instance.
    Note: phi_dissipation and chi_redistribution are stored in Hz
    but defined in GHz in the dimensions — multiply by 1e9 on apply.
    """
    # Q1 — tryptophan lattice
    if "q1_n_tryptophan" in values:
        params.tryptophan.n_trp_baseline = values["q1_n_tryptophan"]
    if "q1_f_coherent_base" in values:
        params.tryptophan.f_coherent = values["q1_f_coherent_base"]

    # Q1 — backbone
    if "q1_d_modes" in values:
        params.dendritic_backbone.D_modes = values["q1_d_modes"]
    if "q1_phi_dissipation" in values:
        params.dendritic_backbone.phi_dissipation = values["q1_phi_dissipation"] * 1e9
    if "q1_chi_redistribution" in values:
        params.dendritic_backbone.chi_redistribution = values["q1_chi_redistribution"] * 1e9
    if "q1_kT_per_modulation" in values:
        params.dendritic_backbone.kT_per_modulation_unit = values["q1_kT_per_modulation"]

    # Q2 — coherence
    if "q2_t2_p31" in values:
        params.quantum.T_singlet_dimer = values["q2_t2_p31"]
    if "q2_j_coupling_hz" in values:
        params.quantum.J_intrinsic_dimer = values["q2_j_coupling_hz"]

    # Q2 — phosphate pool
    if "q2_phosphate_initial" in values:
        params.phosphate.phosphate_total = values["q2_phosphate_initial"]

    # Note: q2_k_classical and q2_k_agg_baseline are set on the dimerization
    # module after model instantiation (see apply_vector_to_model below).


def apply_vector_to_model(model, values: Dict[str, Any]) -> None:
    """
    Apply parameters that must be set on the model instance after init
    (i.e. on sub-modules rather than on Model6Parameters).
    """
    if "q2_k_classical" in values:
        model.ca_phosphate.dimerization.k_classical = values["q2_k_classical"]
    if "q2_k_agg_baseline" in values:
        # k_agg may be on dimerization or a separate attribute — adjust if needed
        if hasattr(model.ca_phosphate.dimerization, 'k_agg'):
            model.ca_phosphate.dimerization.k_agg = values["q2_k_agg_baseline"]


# ── Single vector execution ──────────────────────────────────────────────────

def run_vector(vector, dt: float = 0.001) -> Dict[str, Any]:
    """
    Execute one TestVector. Returns a result dict with metrics + metadata.
    """
    # Import here so sweep_runner.py can be imported without model6 on path
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    values = vector.values
    result = {
        "vector_id":   vector.vector_id,
        "description": vector.description,
        "status":      "ok",
        "error":       None,
        "duration_s":  0.0,
    }

    t0 = time.time()
    try:
        # Build params and apply Q1/Q2/backbone dimensions
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        apply_vector_to_params(params, values)

        # Build network
        n_syn = int(values.get("net_n_synapses", 5))
        spacing = float(values.get("net_spacing_um", 1.0))
        network = MultiSynapseNetwork(
            n_synapses=n_syn,
            pattern="clustered",
            spacing_um=spacing,
        )
        network.initialize(Model6QuantumSynapse, params)

        # Apply invasion fraction
        inv_fraction = float(values.get("net_mt_invaded_fraction", 1.0))
        for i, syn in enumerate(network.synapses):
            invaded = (i / n_syn) < inv_fraction
            syn.set_microtubule_invasion(invaded)

        # Apply instance-level parameters
        for syn in network.synapses:
            apply_vector_to_model(syn, values)

        # Build and run scenario
        scenario = scenario_from_vector(values)
        scenario.run(network, dt=dt)

        # Extract metrics
        metrics = scenario.extract_metrics(network)
        result.update(metrics)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["duration_s"] = time.time() - t0
    return result


# ── Sweep orchestration ──────────────────────────────────────────────────────

def run_sweep(
    dimensions,
    coverage: int = 3,
    max_tests: int = 2000,
    dt: float = 0.001,
    output_dir: str = None,
    label: str = "sweep",
) -> List[Dict[str, Any]]:
    """
    Generate covering array and execute all vectors.

    Args:
        dimensions:  List of RefinedDimension instances to sweep
        coverage:    N-wise coverage strength (2 or 3)
        max_tests:   Hard cap on number of vectors generated
        dt:          Simulation timestep (seconds)
        output_dir:  Where to write results JSON
        label:       Run label for output files

    Returns:
        List of result dicts, one per executed vector.
    """
    output_dir = output_dir or os.path.join(
        os.path.dirname(__file__), "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  QUANTUM BIOLOGY PARAMETER SWEEP")
    print(f"{'='*65}")
    print(f"  Dimensions:   {len(dimensions)}")
    print(f"  Coverage:     {coverage}-wise")
    print(f"  Max vectors:  {max_tests}")
    print(f"  Timestep:     {dt*1000:.1f}ms")
    print(f"  Output:       {output_dir}")
    print(f"{'='*65}\n")

    # Generate covering array
    print("Generating covering array...")
    generator = CoveringArrayGenerator(
        coverage_strength=coverage,
        max_tests=max_tests,
        seed=42,
    )
    covering_rows = generator.generate(dimensions)

    builder = TestVectorBuilder(dimensions)
    vectors = builder.build("qb_sweep", "Quantum Biology Sweep", covering_rows)

    print(f"Generated {len(vectors)} test vectors\n")

    # Execute vectors
    results = []
    errors = 0

    for i, vector in enumerate(vectors):
        pct = (i + 1) / len(vectors) * 100
        print(f"[{i+1:4d}/{len(vectors)}] ({pct:5.1f}%)  {vector.description[:70]}")

        result = run_vector(vector, dt=dt)
        results.append(result)

        if result["status"] == "error":
            errors += 1
            print(f"          ⚠ ERROR: {result['error']}")
        else:
            n_snaps = len(result.get('traversal_snapshots', []))
            print(f"          field={result.get('mean_field_kT', 0):.2f}kT  "
                  f"dimers={result.get('dimers_mean_per_syn', 0):.4f}  "
                  f"P_S={result.get('ps_mean', 0):.3f}  "
                  f"committed={result.get('committed', False)}  "
                  f"snaps={n_snaps}  "
                  f"({result['duration_s']:.1f}s)")

    # Write results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    results_path = os.path.join(output_dir, f"{label}_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp":        timestamp,
                "label":            label,
                "n_dimensions":     len(dimensions),
                "coverage":         coverage,
                "n_vectors":        len(vectors),
                "n_errors":         errors,
                "dt":               dt,
            },
            "dimensions": [
                {"dim_id": d.dim_id, "variable": d.variable,
                 "values": d.values, "importance": d.importance}
                for d in dimensions
            ],
            "results": results,
        }, f, indent=2, default=str)

    print(f"\n{'='*65}")
    print(f"  SWEEP COMPLETE")
    print(f"  Vectors:  {len(vectors)}")
    print(f"  Errors:   {errors}")
    print(f"  Results:  {results_path}")
    print(f"{'='*65}\n")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model 6 Quantum Biology Parameter Sweep")
    parser.add_argument(
        "--mode", choices=["smoke", "critical", "high", "full"],
        default="smoke",
        help=(
            "smoke:    10 vectors, critical dimensions only (fast verification)\n"
            "critical: all critical dimensions, 2-wise\n"
            "high:     critical+high dimensions, 3-wise\n"
            "full:     all dimensions, 3-wise"
        ),
    )
    parser.add_argument("--coverage", type=int, choices=[2, 3], default=None,
                        help="Override coverage strength")
    parser.add_argument("--max-tests", type=int, default=None,
                        help="Override max test vectors")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Simulation timestep in seconds (default 0.001)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    mode_configs = {
        "smoke":    (CRITICAL_DIMENSIONS, 2, 10),
        "critical": (CRITICAL_DIMENSIONS, 2, 500),
        "high":     (HIGH_DIMENSIONS,     3, 1000),
        "full":     (ALL_DIMENSIONS,      3, 2000),
    }

    dimensions, default_coverage, default_max = mode_configs[args.mode]
    coverage  = args.coverage  or default_coverage
    max_tests = args.max_tests or default_max

    run_sweep(
        dimensions=dimensions,
        coverage=coverage,
        max_tests=max_tests,
        dt=args.dt,
        output_dir=args.output_dir,
        label=args.mode,
    )


if __name__ == "__main__":
    main()