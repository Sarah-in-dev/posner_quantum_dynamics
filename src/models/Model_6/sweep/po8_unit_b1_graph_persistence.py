#!/usr/bin/env python3
"""
PO-8 UNIT B1 — does the entanglement graph SURVIVE a quiescent readout delay? NO SEEDING.

THE QUESTION (a measurement, not an opinion)
--------------------------------------------
The eligibility trace must persist across the coherence window (~100-200 s) for a dopamine
readout at a realistic delay to have anything to read. But `dimer_particles.step()` separates
    "2. Population: birth/death to track concentration (FAST chemistry)"
    "3. Coherence: T2 decay for each particle (SLOW quantum)"
and `step_population` sets `target_count = peak(dimer_concentration) * az_volume * N_A`, culling
dimers (and DELETING their bonds via _remove_all_bonds_for_dimer) whenever count > target.

So the graph's lifetime may be capped by the FAST chemistry (dimers tracking calcium) rather than
by the SLOW quantum clock (T_singlet=216 s) or the bond-release rate. "Write-once" protects a bond
from DISSOLVING; it does not protect a bond whose dimer is CULLED.

WHAT THIS MEASURES
  Drive for WRITE_S (build + ignite), then go quiet, and record every second:
    peak dimer_concentration, target_count, actual n_dimers, culled-deaths/s,
    n_intra bonds, n_cross bonds, mean P_S, mean correlated-domain size.
  This separates the three candidate causes of graph loss:
    (a) CULLING      -> deaths track a collapsing target_count (fast chemistry)
    (b) BOND DEATH   -> bonds fall while dimers persist (the release-rate channel)
    (c) COHERENCE    -> mean P_S falls toward the 0.5 Werner floor (the slow quantum clock)
  Whichever dominates sets the true eligibility-trace lifetime.

Instrumentation is READ-ONLY: step_population is wrapped to record its inputs (the established
probe pattern, cf. sweep/po5_unit2_qb_selectivity.py:127). No physics is modified.

ABSOLUTE RULE: no np.random.seed(), no seed= to any constructor. Free-running draws only.
Drive via net.step() ONLY (per-synapse s.step() leaves _backbone_eta=0 and nothing ignites).
"""
import sys, os, json, math, argparse
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)
import logging
logging.disable(logging.INFO)

from po7_unit18_correlation_domains import build_weighted_graph, bounded_dijkstra, WERNER

N_SYN, SPACING = 7, 1.0
DT = 5e-3
VOLT, REST_VOLT = -40e-3, -70e-3
ACT = 0.95
WRITE_S = 20.0          # drive window (past ~15 s ignition)
QUIET_S = 60.0          # quiescent readout delay to observe
SAMPLE_S = 1.0
OUT_DIR = os.path.join(PROJECT_ROOT, "results")

AZ_VOLUME_L = 1e-17     # step_population's active-zone volume
N_A = 6.022e23


def one_run(run_id):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)  # NO seed=
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker
    rel = PresynapticRelease(None)

    # --- READ-ONLY instrumentation: capture step_population's inputs/outputs per synapse ---
    probe = {"peak_conc": 0.0, "target": 0, "deaths": 0, "births": 0}

    def wrap(ps):
        orig = ps.step_population
        def wrapped(dt, dimer_concentration, template_field):
            pc = float(np.max(dimer_concentration)) if dimer_concentration is not None else 0.0
            probe["peak_conc"] += pc
            probe["target"] += int(round(pc * AZ_VOLUME_L * N_A))
            r = orig(dt, dimer_concentration, template_field)
            probe["deaths"] += int(r.get("n_deaths", 0))
            probe["births"] += int(r.get("n_births", 0))
            return r
        ps.step_population = wrapped
    for s in net.synapses:
        wrap(s.dimer_particles)

    def domain_mean(tr):
        nodes, adj, edges, _mf = build_weighted_graph(tr)
        if not nodes:
            return 0.0
        return float(np.mean([sum(math.exp(-d) for d in bounded_dijkstra(u, adj).values())
                              for u in nodes]))

    t_end = WRITE_S + QUIET_S
    nsteps = int(round(t_end / DT))
    write_steps = int(round(WRITE_S / DT))
    sample_every = int(round(SAMPLE_S / DT))
    peak_eta = 0.0
    rows = []
    for i in range(1, nsteps + 1):
        driving = i <= write_steps
        g = rel.step(ACT, DT) if driving else 0.0
        net.step(DT, {"voltage": (VOLT if driving else REST_VOLT),
                      "reward": False, "glutamate": g})
        peak_eta = max(peak_eta, max((float(getattr(s, "_backbone_eta", 0.0))
                                      for s in net.synapses), default=0.0))
        if i % sample_every == 0:
            ndim = sum(len(s.dimer_particles.dimers) for s in net.synapses)
            ps_all = [float(d['P_S']) for d in tr.all_dimers]
            n_cross = sum(1 for k, v in tr.cross_synapse_bonds.items()
                          if float(v) > WERNER and k[0][0] != k[1][0])
            rows.append(dict(
                t=round(i * DT, 3), delay=round(i * DT - WRITE_S, 3), driving=bool(driving),
                n_dimers=int(ndim),
                target_count=int(probe["target"]),          # summed over synapses this sample
                peak_conc=float(probe["peak_conc"]),
                deaths=int(probe["deaths"]), births=int(probe["births"]),
                n_intra=int(len(tr.intra_synapse_bonds_cache)),
                n_cross=int(n_cross),
                mean_PS=float(np.mean(ps_all)) if ps_all else 0.0,
                mean_domain=domain_mean(tr),
                peak_eta=float(peak_eta)))
            probe.update({"peak_conc": 0.0, "target": 0, "deaths": 0, "births": 0})
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                write_s=WRITE_S, quiet_s=QUIET_S, peak_eta=float(peak_eta),
                ignited=bool(peak_eta > 0.0), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po8_unit_b1_worker{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"b1_w{a.worker}r{j}"
        sys.stderr.write(f"[b1 w{a.worker}] draw {j+1}/{a.n} ({rid})\n"); sys.stderr.flush()
        rec = one_run(rid)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        sys.stderr.write(f"[b1 w{a.worker}] done {rid} ignited={rec['ignited']}\n")
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
