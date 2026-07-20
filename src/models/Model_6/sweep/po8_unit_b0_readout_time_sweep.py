#!/usr/bin/env python3
"""
PO-8 UNIT B0 — READOUT-TIME domain sweep (characterization; NOT the scored keystone). NO SEEDING.

WHAT THIS ANSWERS
-----------------
The reframe (L·PO7-5) says the computational output is the CORRELATED-DOMAIN PARTITION read at
readout (a dopamine event at a realistic delay), not the write-time graph. Unit 18 measured the
partition at WRITE time only (~20 s, P_S still ~1). The open question advisor R6 named as gating
the next phase (Q2): does the near-lossless intra-synapse core PERSIST to readout, or do domains
fragment as P_S(t) decays?

The advisor's prediction (R6 check-in, arithmetic independently reproduced by PO-8): with
    P_S(t) = 0.25 + 0.75·exp(-t/216)
intra correlation p=(4·P_S²-1)/3 stays supercritical until it crosses the percolation point at
t≈105 s, ~2 s before Werner death at 107 s. So domains should hold synapse-scale and then COLLAPSE
ABRUPTLY at end-of-life — readout-time domain size ≈ write-time domain size, right up until nothing.
The competing hypothesis: in a dense, loop-heavy intra cloud the mean-field cliff is softened and
domains DECLINE GRADUALLY with delay. This probe discriminates them.

This is INTRA-core dominated and therefore LAMBDA-INDEPENDENT (intra F = P_S_i·P_S_j carries no
spatial weight), so the result stands regardless of the open λ=5µm-vs-214µm decision. The CROSS
channel is reported alongside but its interpretation waits on that ruling.

NOT the keystone: there is NO input contrast here and NO scored verdict. It is instrument
validation + realistic-delay selection for the pre-registered Unit B. The dopamine event is the
CLOCK, not a decoherence mechanism (dopamine is state-inert in this model, verified: PO-8 grounding
brief correction 2), so a read-only snapshot of the graph AT delay t IS the readout object.

METHOD
  One free-running draw (NO seed) is stepped once to T_MAX. The correlated-domain partition is
  SNAPSHOT (read-only, non-perturbing) at each READOUT_TIME within that single run, so all delays
  share one trajectory (the fair within-draw time course). N draws give the ensemble spread.
  Metric reused VERBATIM by import from po7_unit18_correlation_domains (build_weighted_graph,
  bounded_dijkstra, connected_components): p_e=(4F-1)/3, w_e=-ln p_e, S(u)=Σ exp(-d), bounded
  Dijkstra D_MAX=8.0. No F-threshold anywhere.

USAGE (parallelised, cap 4 workers)
  run:     po8_unit_b0_readout_time_sweep.py run --worker W --n K
             K draws -> one fsync'd JSONL line per draw to results/po8_unit_b0_worker{W}.jsonl
  analyze: po8_unit_b0_readout_time_sweep.py analyze
             pools workers -> per-readout-time table + collapse-time estimate ->
             results/po8_unit_b0_results.json

ABSOLUTE RULE: no np.random.seed(), no seed= to any constructor. Free-running only.
Rig config copied from po7_unit18_correlation_domains.py (which copied po7_unit17).
"""
import sys, os, json, glob, argparse, math
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

# reuse the Unit-18 metric verbatim (single source of truth for the domain measure)
from po7_unit18_correlation_domains import (
    build_weighted_graph, bounded_dijkstra, connected_components, D_MAX, WERNER)

# --- rig constants (from po7_unit18_correlation_domains.py) ---
N_SYN, SPACING = 7, 1.0
DT = 5e-3
VOLT = -40e-3
ACT = 0.95

# readout delays (s): write-time regime (advisor's 12-20s) through the Werner-floor crossing
# (~107 s) and past it. Aligned to tracker-update steps (every 10th step = 50 ms) by construction.
READOUT_TIMES = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 110.0, 120.0]
T_MAX = READOUT_TIMES[-1]
OUT_DIR = os.path.join(PROJECT_ROOT, "results")


def snapshot(tr):
    """Read-only domain-partition snapshot of the tracker's current state."""
    nodes, adj, edges, max_cross_F = build_weighted_graph(tr)
    V = len(nodes)
    if V == 0:
        return dict(n_dimers=0, n_edges=0, components_ge2=0, largest_frac=0.0,
                    mean_domain_size=0.0, effective_domains=0.0, domain_p90=0.0,
                    max_cross_F=0.0, mean_PS=0.0, intraF_median=0.0, crossF_median=0.0,
                    n_intra_edges=0, n_cross_edges=0)
    comp_sizes = connected_components(nodes, edges.keys())
    largest = comp_sizes[0] if comp_sizes else 0
    components_ge2 = sum(1 for s in comp_sizes if s >= 2)
    S = np.asarray([sum(math.exp(-d) for d in bounded_dijkstra(u, adj).values())
                    for u in nodes], float)
    effective_domains = float(np.sum(1.0 / S)) if S.size else 0.0
    # channel split, read straight off the tracker containers
    intraF = [float(v) for v in tr.intra_synapse_bonds_cache.values()]
    crossF = [float(v) for k, v in tr.cross_synapse_bonds.items()
              if float(v) > WERNER and k[0][0] != k[1][0]]
    ps = [float(d['P_S']) for d in tr.all_dimers]
    return dict(
        n_dimers=int(V), n_edges=int(len(edges)),
        components_ge2=int(components_ge2), largest_frac=float(largest / V),
        mean_domain_size=float(S.mean()), effective_domains=effective_domains,
        domain_p90=float(np.percentile(S, 90)),
        max_cross_F=float(max_cross_F), mean_PS=float(np.mean(ps)) if ps else 0.0,
        intraF_median=float(np.median(intraF)) if intraF else 0.0,
        crossF_median=float(np.median(crossF)) if crossF else 0.0,
        n_intra_edges=int(len(intraF)), n_cross_edges=int(len(crossF)))


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
    rel = PresynapticRelease(None)   # None => OS entropy => free draw

    snap_steps = {int(round(t / DT)): t for t in READOUT_TIMES}
    nsteps = int(round(T_MAX / DT))
    peak_eta = 0.0
    snaps = []
    for i in range(1, nsteps + 1):
        g = rel.step(ACT, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        e = max((float(getattr(s, "_backbone_eta", 0.0)) for s in net.synapses), default=0.0)
        peak_eta = max(peak_eta, e)
        if i in snap_steps:
            rec = snapshot(tr)
            rec["t"] = snap_steps[i]
            rec["peak_eta_so_far"] = float(peak_eta)
            snaps.append(rec)
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                peak_eta=float(peak_eta), ignited=bool(peak_eta > 0.0),
                readout_times=READOUT_TIMES, snapshots=snaps)


def cmd_run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po8_unit_b0_worker{args.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(args.n):
        run_id = f"w{args.worker}r{j}"
        sys.stderr.write(f"[worker {args.worker}] starting draw {j+1}/{args.n} ({run_id})\n")
        sys.stderr.flush()
        rec = one_run(run_id)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        md = {s["t"]: s["mean_domain_size"] for s in rec["snapshots"]}
        sys.stderr.write(f"[worker {args.worker}] done {run_id}: ignited={rec['ignited']} "
                         f"peak_eta={rec['peak_eta']:.3f} mean_dom@{{"
                         + ", ".join(f"{t:.0f}s:{md.get(t,0):.0f}" for t in READOUT_TIMES)
                         + "}}\n")
        sys.stderr.flush()
    return 0


def _dist(a):
    a = np.asarray([x for x in a if x is not None], float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def cmd_analyze(args):
    files = sorted(glob.glob(os.path.join(OUT_DIR, "po8_unit_b0_worker*.jsonl")))
    runs = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    runs.sort(key=lambda r: r["run_id"])
    N = len(runs)
    print("=" * 120)
    print(f"PO-8 UNIT B0 — readout-time domain sweep  (N={N} free draws, NO SEEDING)")
    print("=" * 120)
    if N == 0:
        print("  no runs found.")
        return 1

    # pool snapshots by readout time
    by_t = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for s in r["snapshots"]:
            t = s["t"]
            for k in ("mean_domain_size", "effective_domains", "components_ge2", "largest_frac",
                      "max_cross_F", "mean_PS", "intraF_median", "crossF_median",
                      "n_cross_edges", "n_dimers"):
                by_t[t][k].append(s.get(k))

    ts = sorted(by_t)
    print(f"\n{'t(s)':>6} {'meanPS':>7} {'mean_dom':>9} {'eff_dom':>8} {'comps>=2':>8} "
          f"{'lg_frac':>7} {'intraF':>7} {'crossF':>7} {'n_cross':>7} {'maxF':>6}")
    med = {}
    for t in ts:
        m = {k: _dist(v)["median"] for k, v in by_t[t].items()}
        med[t] = m
        def g(k, f="{:>9.1f}"):
            return f.format(m[k]) if m[k] is not None else "  n/a"
        print(f"{t:>6.0f} {m['mean_PS']:>7.3f} {m['mean_domain_size']:>9.1f} "
              f"{m['effective_domains']:>8.1f} {m['components_ge2']:>8.1f} "
              f"{m['largest_frac']:>7.3f} {m['intraF_median']:>7.3f} "
              f"{(m['crossF_median'] or 0):>7.3f} {(m['n_cross_edges'] or 0):>7.1f} "
              f"{(m['max_cross_F'] or 0):>6.3f}")

    # collapse-time estimate: last t where median mean_domain_size is still >= 50% of its write-time
    # (t=2s) value, then the first t where it drops below 10% — brackets the collapse.
    md_series = [(t, med[t]["mean_domain_size"]) for t in ts]
    base = md_series[0][1] or 0.0
    half_hold = max((t for t, v in md_series if base and v >= 0.5 * base), default=None)
    collapsed = next((t for t, v in md_series if base and v < 0.1 * base), None)
    print(f"\n[COLLAPSE BRACKET]  write-time(t={ts[0]:.0f}s) median domain = {base:.1f} dimers")
    print(f"  domain still >=50% of write-time up to t = {half_hold}s")
    print(f"  domain <10% of write-time first at   t = {collapsed}s")
    if half_hold is not None and collapsed is not None:
        shape = ("ABRUPT (advisor): holds synapse-scale then cliffs"
                 if (collapsed - half_hold) <= 20.0
                 else "GRADUAL: domain declines smoothly across the delay window")
    else:
        shape = "domain did not collapse within the swept window (extend T_MAX)"
    print(f"  -> shape: {shape}")

    out = {"unit": "po8_unit_b0_readout_time_sweep",
           "generated": datetime.now(timezone.utc).isoformat(),
           "n_runs": N, "no_seeding": True, "d_max": D_MAX,
           "readout_times": ts,
           "config": {"n_synapses": N_SYN, "pattern": "linear", "spacing_um": SPACING,
                      "dt": DT, "t_max_s": T_MAX, "voltage": VOLT, "activation": ACT},
           "median_by_t": {str(t): med[t] for t in ts},
           "dist_by_t": {str(t): {k: _dist(v) for k, v in by_t[t].items()} for t in ts},
           "collapse_bracket": {"write_time_domain": base, "half_hold_t": half_hold,
                                "collapsed_t": collapsed, "shape": shape},
           "runs": runs}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "po8_unit_b0_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--worker", type=int, required=True)
    pr.add_argument("--n", type=int, required=True)
    sub.add_parser("analyze")
    args = ap.parse_args()
    return cmd_run(args) if args.mode == "run" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
