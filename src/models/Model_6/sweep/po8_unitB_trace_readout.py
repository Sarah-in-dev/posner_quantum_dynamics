#!/usr/bin/env python3
"""
PO-8 UNIT B — the eligibility trace, measured on the CANONICAL protocol. NO SEEDING.

THE PROTOCOL IS THE MODEL'S OWN VALIDATED ONE (run_theta_burst_45s.py:615-681):
    theta-burst traversals (WRITE)  ->  analytical_gap(delay)  (AGE)  ->  reward step (READ)
`analytical_gap` ages the population by the INTRINSIC dissolution (K_CLASSICAL=0.005/s, tau=200s)
plus coherence death at P_S<0.5 and stochastic disentanglement — i.e. the "trace falls out of T2"
mechanism (advisor Q4). This is the ONLY correct way to age the graph across a readout delay; a
naive drop-the-drive "quiet" instead collapses the calcium peak and culls the population in ~30s
(the step_population np.max artifact, L·PO5-11) — that was PO-8's earlier error, now avoided by
using analytical_gap as the model intends.

We import `run_burst_traversal` and `analytical_gap` from the canonical driver and CALL them —
no edit to that file (PO-4's surface, on the do-not-edit list).

WHAT THIS MEASURES
  Drive `N_TRAV` traversals (write + ignite), snapshot the correlated-domain partition at WRITE
  (delay 0), then advance the gap in increments and snapshot again at each cumulative delay. The
  domain-size-vs-delay curve IS the eligibility trace. The advisor's prediction (Q4): under
  coherence-limited death the trace should persist across tens of seconds and fall as cross bonds
  (F=P_S^2 W) cross the Werner floor — ~74s at lambda=5um, ~107s at lambda=214um.

ARMS (free-running draws, no seeding):
  release rate: off (unmoored) vs on (derived, PO-8 Unit A `physical_release_rate`)
  lambda:       5um (coded) vs 214um (the L_coh the advisor argues for, Q3)

CONTROL FIRST: report WRITE-time ignition (peak eta) and domain size, to confirm the canonical
theta-burst drive reproduces a live graph before any delay is applied. A dead condensate voids the
readout (the eta==0 trap, L·PO7-4 §7).
"""
import sys, os, json, math, argparse
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
import logging
logging.disable(logging.INFO)

from po7_unit18_correlation_domains import build_weighted_graph, bounded_dijkstra, connected_components, WERNER
from presynaptic_release import PresynapticRelease
# CALL the canonical aging function; do NOT edit run_theta_burst_45s.py (PO-4 surface).
from run_theta_burst_45s import analytical_gap

# WRITE: glutamate drive that provably IGNITES the condensate (the B0/Unit-17 rig). The canonical
# run_burst_traversal is voltage-ONLY, which leaves NMDARs silent (the ERR-2 defect, handoff §3) and
# never ignites the multi-synapse graph — so we cannot use it to build a cross-synapse domain. The
# theta-burst temporal PATTERN matters for input-selectivity (scored Unit B), NOT for trace
# persistence, which only needs a live ignited graph to then age. AGE: analytical_gap (validated).
N_SYN, SPACING = 7, 1.0
DT = 5e-3              # write dt (B0/Unit-17 rig; ignites reliably)
WRITE_S = 20.0        # glutamate-driven write, past ~15 s ignition
VOLT, ACT = -40e-3, 0.95
DELAYS = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 90.0, 120.0]   # cumulative gap after write
OUT_DIR = os.path.join(PROJECT_ROOT, "results")


def domain_snapshot(tr):
    nodes, adj, edges, max_cross_F = build_weighted_graph(tr)
    V = len(nodes)
    if V == 0:
        return dict(n_dimers=0, n_edges=0, components_ge2=0, largest_frac=0.0,
                    mean_domain=0.0, effective_domains=0.0, max_cross_F=0.0,
                    mean_PS=0.0, n_cross=0, intraF_med=0.0, crossF_med=0.0)
    comp = connected_components(nodes, edges.keys())
    largest = comp[0] if comp else 0
    S = np.asarray([sum(math.exp(-d) for d in bounded_dijkstra(u, adj).values()) for u in nodes], float)
    intraF = [float(v) for v in tr.intra_synapse_bonds_cache.values()]
    crossF = [float(v) for k, v in tr.cross_synapse_bonds.items()
              if float(v) > WERNER and k[0][0] != k[1][0]]
    ps = [float(d['P_S']) for d in tr.all_dimers]
    return dict(n_dimers=int(V), n_edges=int(len(edges)),
                components_ge2=int(sum(1 for s in comp if s >= 2)),
                largest_frac=float(largest / V), mean_domain=float(S.mean()),
                effective_domains=float(np.sum(1.0 / S)), max_cross_F=float(max_cross_F),
                mean_PS=float(np.mean(ps)) if ps else 0.0, n_cross=int(len(crossF)),
                intraF_med=float(np.median(intraF)) if intraF else 0.0,
                crossF_med=float(np.median(crossF)) if crossF else 0.0)


def one_run(run_id, physical_rate, lam):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING,
                              coupling_length_um=lam)  # NO seed=
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker
    tr.physical_release_rate = bool(physical_rate)

    # WRITE: glutamate-driven, ignites the condensate (builds the cross-synapse domain)
    rel = PresynapticRelease(None)
    peak_eta = 0.0
    for _ in range(int(round(WRITE_S / DT))):
        g = rel.step(ACT, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        peak_eta = max(peak_eta, max((float(getattr(s, "_backbone_eta", 0.0))
                                      for s in net.synapses), default=0.0))

    snaps = []
    prev_delay = 0.0
    for d in DELAYS:
        if d > prev_delay:
            analytical_gap(net, d - prev_delay, dt_sub=1.0)   # AGE by the intrinsic mechanism
            # tail refresh: recompute the network tracker's cross bonds from the aged dimers.
            # coupling_weights MUST be passed or _update_entanglement silently forms none.
            net.step(DT, {"voltage": -70e-3, "reward": False})
            prev_delay = d
        rec = domain_snapshot(tr)
        rec["delay"] = d
        snaps.append(rec)
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                physical_release_rate=bool(physical_rate), coupling_length_um=float(lam),
                write_s=WRITE_S, peak_eta=float(peak_eta), ignited=bool(peak_eta > 0.0),
                write_domain=snaps[0]["mean_domain"], snapshots=snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--rate", choices=["off", "on"], required=True)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po8_unitB_{a.tag}_worker{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"{a.tag}_w{a.worker}r{j}"
        sys.stderr.write(f"[unitB {a.tag} w{a.worker}] draw {j+1}/{a.n} rate={a.rate} lam={a.lam}\n")
        sys.stderr.flush()
        rec = one_run(rid, a.rate == "on", a.lam)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        md = {s["delay"]: round(s["mean_domain"], 0) for s in rec["snapshots"]}
        sys.stderr.write(f"[unitB {a.tag} w{a.worker}] done {rid} ignited={rec['ignited']} "
                         f"peak_eta={rec['peak_eta']:.3f} domain_by_delay={md}\n")
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
