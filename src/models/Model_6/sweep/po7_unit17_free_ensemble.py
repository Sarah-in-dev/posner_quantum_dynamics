#!/usr/bin/env python3
"""
PO-7 UNIT 17 — free-running ensemble of the 7-synapse rig.  NO SEEDING ANYWHERE.

THE POINT
---------
Every prior cross-synapse claim in this program rests on ONE draw (Unit 16 hard-codes
seed 4242, a known igniter). But this is a STOCHASTIC quantum-biological system: its output
is a DISTRIBUTION, not a value. The randomness — vesicle release, calcium gating, dimer birth,
whether the microtubule condensate ignites — is the physics, not noise. So we run the identical
rig N=16 times, each a fresh free-running draw from OS entropy, seed NOWHERE, and report what the
free-running system actually does — including "it did not ignite," as data.

WHAT EACH RUN RECORDS
  - ignited? (peak backbone eta > 0 over the run) and the peak eta value
  - peak and final cross-synapse bond count (F>0.5 edges spanning two synapses)
  - final components / largest_frac / n_multi (clusters spanning >=2 synapses) via
    tracker._find_all_clusters()
  - full list of cross-synapse edge fidelities F at the final state (pooled across igniters)
  - bridge fidelities (cross edges whose removal increases the component count) at the final state

THE THREE DISTRIBUTIONS (analyze mode)
  1. P(ignition) and the distribution of peak eta.
  2. largest_frac across ALL runs and across IGNITED runs.
  3. THE DECISIVE ONE — pooled cross-synapse edge fidelities across igniters (min/p10/median/p90/
     max, fraction F<0.55 i.e. tangle tau=(2F-1)^2 < 0.01), AND the same for BRIDGES only.
     Near the Werner floor (F~0.5, negligible entanglement) => the blob is an unweighted-connectivity
     artifact.  Genuinely strong (F~0.9) => real structure.

USAGE (parallelised, cap 4 workers)
  run:     po7_unit17_free_ensemble.py run --worker W --n K
             does K free-running runs, appends one flushed JSONL line each to
             results/po7_unit17_worker{W}.jsonl (a crash loses at most the in-flight run).
  analyze: po7_unit17_free_ensemble.py analyze
             pools every results/po7_unit17_worker*.jsonl line, prints the three distributions,
             writes results/po7_unit17_results.json.

ABSOLUTE RULE: no np.random.seed(), no seed= passed to any constructor. Free-running only.
"""
import sys, os, json, glob, argparse
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

import logging
logging.disable(logging.INFO)

N_SYN, SPACING = 7, 1.0
DT, T_SIM = 5e-3, 20.0
VOLT = -40e-3
WERNER = 0.5
ACT = 0.95                      # sustained presynaptic activation (matches Unit 16 drive)
OUT_DIR = os.path.join(PROJECT_ROOT, "results")


# --------------------------------------------------------------------------------------------
# graph helpers (union-find INCLUDING singletons, so removing a true bridge raises the count)
# --------------------------------------------------------------------------------------------
def comps_count(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = {find(n) for n in nodes}
    return len(roots)


def cross_edges(tr):
    """Cross-synapse edges at the current tracker state: F>0.5 and spanning two synapses.
    Key k = ((syn_i,id),(syn_j,id)); k[*][0] is the synapse index."""
    return [(k, float(f)) for k, f in tr.cross_synapse_bonds.items()
            if f > WERNER and k[0][0] != k[1][0]]


# --------------------------------------------------------------------------------------------
# one free-running draw
# --------------------------------------------------------------------------------------------
def one_run(run_id):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    # --- rig config copied from po7_unit16_fidelity_of_bridges.py, seeds STRIPPED ---
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0

    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)  # NO seed=
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True   # drop clique + EM (Unit 14)
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker

    # PresynapticRelease requires a positional `seed`; None => np.random.default_rng(None),
    # i.e. a fresh draw from OS entropy. This is the free-running "no seed" form.
    rel = PresynapticRelease(None)

    peak_eta = 0.0
    peak_xb = 0
    nsteps = int(round(T_SIM / DT))
    for _ in range(nsteps):
        g = rel.step(ACT, DT)
        # net.step drives every synapse with the shared stimulus and advances the tracker
        # internally every 10 steps. Do NOT call tr.step() (double-advance -> TypeError).
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        etas = [float(getattr(s, "_backbone_eta", 0.0)) for s in net.synapses]
        e = max(etas) if etas else 0.0
        if e > peak_eta:
            peak_eta = e
        xb = len(cross_edges(tr))
        if xb > peak_xb:
            peak_xb = xb

    # --- final state graph ---
    cross = cross_edges(tr)
    final_xb = len(cross)
    nodes = [d["global_id"] for d in tr.all_dimers]
    nodeset = set(nodes)
    V = len(nodes)

    clusters = tr._find_all_clusters()          # list of sets of global_id, singletons omitted
    components = len(clusters)
    largest = max((len(c) for c in clusters), default=0)
    largest_frac = (largest / V) if V else 0.0
    n_multi = sum(1 for c in clusters if len({g[0] for g in c}) >= 2)

    cross_F = [f for _, f in cross]

    # --- bridges: cross edges whose removal increases the component count ---
    intra_keys = [k for k in tr.intra_synapse_bonds_cache
                  if k[0] in nodeset and k[1] in nodeset]
    cross_keys = [k for k, _ in cross]
    base_count = comps_count(nodes, intra_keys + cross_keys)
    bridge_F = []
    for k, f in cross:
        without = intra_keys + [kk for kk in cross_keys if kk != k]
        if comps_count(nodes, without) > base_count:
            bridge_F.append(f)

    ignited = peak_eta > 0.0
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ignited": bool(ignited),
        "peak_eta": float(peak_eta),
        "peak_xbond": int(peak_xb),
        "final_xbond": int(final_xb),
        "V": int(V),
        "components": int(components),
        "largest": int(largest),
        "largest_frac": float(largest_frac),
        "n_multi": int(n_multi),
        "n_bridges": int(len(bridge_F)),
        "cross_F": [float(x) for x in cross_F],
        "bridge_F": [float(x) for x in bridge_F],
    }


# --------------------------------------------------------------------------------------------
# run mode: K free-running draws for one worker, flushed to its own JSONL as we go
# --------------------------------------------------------------------------------------------
def cmd_run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po7_unit17_worker{args.worker}.jsonl")
    # fresh file for this worker so re-runs don't pool stale lines
    open(jsonl, "w").close()
    for j in range(args.n):
        run_id = f"w{args.worker}r{j}"
        sys.stderr.write(f"[worker {args.worker}] starting run {j+1}/{args.n} ({run_id})\n")
        sys.stderr.flush()
        rec = one_run(run_id)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        sys.stderr.write(
            f"[worker {args.worker}] done {run_id}: ignited={rec['ignited']} "
            f"peak_eta={rec['peak_eta']:.4f} final_xbond={rec['final_xbond']} "
            f"largest_frac={rec['largest_frac']:.4f} n_bridges={rec['n_bridges']}\n")
        sys.stderr.flush()
    return 0


# --------------------------------------------------------------------------------------------
# analyze mode: pool every worker JSONL, print the three distributions, write summary JSON
# --------------------------------------------------------------------------------------------
def _dist(a):
    a = np.asarray(a, float)
    return {
        "n": int(a.size),
        "min": float(a.min()) if a.size else None,
        "p10": float(np.percentile(a, 10)) if a.size else None,
        "median": float(np.median(a)) if a.size else None,
        "p90": float(np.percentile(a, 90)) if a.size else None,
        "max": float(a.max()) if a.size else None,
    }


def cmd_analyze(args):
    files = sorted(glob.glob(os.path.join(OUT_DIR, "po7_unit17_worker*.jsonl")))
    runs = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    runs.sort(key=lambda r: r["run_id"])
    N = len(runs)

    print("=" * 96)
    print(f"PO-7 UNIT 17 — free-running ensemble  (N={N} draws, NO SEEDING)")
    print("=" * 96)
    if N == 0:
        print("  no runs found — nothing to analyse.")
        return 1

    igniters = [r for r in runs if r["ignited"]]
    n_ign = len(igniters)
    p_ign = n_ign / N
    peak_etas = [r["peak_eta"] for r in runs]

    # ---- (1) P(ignition) + peak eta distribution ----
    print("\n[1] IGNITION")
    print(f"  P(ignition) = {n_ign}/{N} = {p_ign:.3f}")
    pe = np.asarray(peak_etas, float)
    print(f"  peak eta over ALL runs: min={pe.min():.4f} median={np.median(pe):.4f} "
          f"max={pe.max():.4f}")
    if n_ign:
        pei = np.asarray([r["peak_eta"] for r in igniters], float)
        print(f"  peak eta over IGNITERS:  min={pei.min():.4f} median={np.median(pei):.4f} "
              f"max={pei.max():.4f}")
    else:
        print("  HEADLINE: the cross-synapse keystone did NOT occur in any of the free draws.")

    # ---- (2) largest_frac distribution ----
    print("\n[2] LARGEST COMPONENT FRACTION")
    lf_all = np.asarray([r["largest_frac"] for r in runs], float)
    print(f"  largest_frac ALL runs:  min={lf_all.min():.4f} median={np.median(lf_all):.4f} "
          f"max={lf_all.max():.4f}")
    if n_ign:
        lf_ign = np.asarray([r["largest_frac"] for r in igniters], float)
        print(f"  largest_frac IGNITERS:  min={lf_ign.min():.4f} median={np.median(lf_ign):.4f} "
              f"max={lf_ign.max():.4f}")
    print("  (also, for reference) components / n_multi per run:")
    for r in runs:
        print(f"    {r['run_id']:>8}  ignited={str(r['ignited']):5}  V={r['V']:>4}  "
              f"comps={r['components']:>3}  largest_frac={r['largest_frac']:.4f}  "
              f"n_multi={r['n_multi']:>2}  final_xbond={r['final_xbond']:>4}  "
              f"n_bridges={r['n_bridges']:>3}")

    # ---- (3) THE DECISIVE ONE — pooled cross-edge & bridge fidelity across igniters ----
    print("\n[3] CROSS-SYNAPSE EDGE FIDELITY, pooled across IGNITERS  (the decisive distribution)")
    pooled_cross = [f for r in igniters for f in r["cross_F"]]
    pooled_bridge = [f for r in igniters for f in r["bridge_F"]]
    cross_summary = None
    bridge_summary = None
    if not pooled_cross:
        if n_ign == 0:
            print("  no igniting runs => no cross-synapse edges to pool.")
            print("  HEADLINE: P(ignition)=0; the cross-synapse keystone did not occur in "
                  f"{N} free draws.")
        else:
            print("  igniters produced NO cross-synapse edges (F>0.5) at the final state.")
    else:
        F = np.asarray(pooled_cross, float)
        tau = (2.0 * F - 1.0) ** 2
        frac_lt = float((F < 0.55).mean())
        print(f"  pooled cross edges: n={F.size}  from {n_ign} igniter(s)")
        print(f"    F: min={F.min():.4f} p10={np.percentile(F,10):.4f} "
              f"median={np.median(F):.4f} p90={np.percentile(F,90):.4f} max={F.max():.4f}")
        print(f"    tangle tau=(2F-1)^2: median={np.median(tau):.6f} max={tau.max():.6f}")
        print(f"    fraction with F < 0.55 (tau < 0.01, ~1% of maximal entanglement): "
              f"{100.0*frac_lt:.1f}%")
        cross_summary = _dist(F); cross_summary["frac_F_below_0p55"] = frac_lt
        cross_summary["tau_median"] = float(np.median(tau))

        if pooled_bridge:
            B = np.asarray(pooled_bridge, float)
            tb = (2.0 * B - 1.0) ** 2
            fb = float((B < 0.55).mean())
            print(f"  pooled BRIDGES (removal increases component count): n={B.size}")
            print(f"    F: min={B.min():.4f} p10={np.percentile(B,10):.4f} "
                  f"median={np.median(B):.4f} p90={np.percentile(B,90):.4f} max={B.max():.4f}")
            print(f"    tangle: median={np.median(tb):.6f} max={tb.max():.6f}")
            print(f"    fraction with F < 0.55: {100.0*fb:.1f}%")
            bridge_summary = _dist(B); bridge_summary["frac_F_below_0p55"] = fb
            bridge_summary["tau_median"] = float(np.median(tb))
            verdict = ("NEAR THE WERNER FLOOR (F~0.5, negligible entanglement — giant component is "
                       "an unweighted-connectivity artifact)"
                       if np.median(B) < 0.55 else
                       "GENUINELY STRONG (real structure)" if np.median(B) > 0.85 else
                       "INTERMEDIATE (neither clearly floor nor clearly strong)")
            print(f"  BRIDGE VERDICT: median bridge F = {np.median(B):.4f} -> {verdict}")
        else:
            print("  no bridges among pooled cross edges (all cross edges are redundant).")

    out = {
        "unit": "po7_unit17_free_ensemble",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_runs": N,
        "no_seeding": True,
        "config": {"n_synapses": N_SYN, "pattern": "linear", "spacing_um": SPACING,
                   "dt": DT, "t_sim_s": T_SIM, "voltage": VOLT, "activation": ACT,
                   "werner": WERNER, "fraction_P31": 1.0},
        "p_ignition": p_ign,
        "n_igniters": n_ign,
        "peak_eta_all": _dist(peak_etas),
        "peak_eta_igniters": _dist([r["peak_eta"] for r in igniters]) if n_ign else None,
        "largest_frac_all": _dist([r["largest_frac"] for r in runs]),
        "largest_frac_igniters": _dist([r["largest_frac"] for r in igniters]) if n_ign else None,
        "pooled_cross_F": cross_summary,
        "pooled_bridge_F": bridge_summary,
        "runs": runs,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "po7_unit17_results.json")
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
    if args.mode == "run":
        return cmd_run(args)
    return cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
