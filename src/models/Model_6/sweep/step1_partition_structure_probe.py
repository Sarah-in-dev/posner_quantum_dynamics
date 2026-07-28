#!/usr/bin/env python3
"""
STEP-1 PARTITION-STRUCTURE PROBE (scored) — is the cross-synapse entanglement partition
STRUCTURED (input-dependent) or a BLOB (input-independent)?  Gate for Step 2 (P31/P32).

Pre-registered in docs/PREREG_STEP1_PARTITION_STRUCTURE.md (written BEFORE this run).
Verdict thresholds are LOCKED there; this script only measures and applies them.

Design (see prereg): n=8 linear @1.0µm, interleaved C={0,2,4,6}/G={1,3,5,7} at matched
distances (all < d*max 3.47µm, so geometry alone connects C,G alike). Engagement guaranteed
by controlled IC (E_invasion clamped). Only INPUT TIMING varies: SYNC (in phase) vs STAGGER
(G offset THETA/2, never co-bursting). Sweep dimer count via duration; 5 seeds.

Read on the synapse-QUOTIENT graph:
  CPEF        = cross-phase (C-G) quotient edges / all quotient edges  (high=blob, low=structured)
  largest_frac= largest quotient component / N
  Δ(dimers)   = CPEF_SYNC - CPEF_STAGGER   (the structure-vs-dimer-count curve)

Self-daemonizes (double-fork + os.setsid; macOS has no setsid cmd) so a teardown can't kill it.
"""
import sys, os, json, logging, time
import numpy as np

M6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(M6, "sweep")
sys.path.insert(0, M6); sys.path.insert(0, SWEEP)

N, SPACING = 8, 1.0
C = [0, 2, 4, 6]; G = [1, 3, 5, 7]
DT = 0.001
SPIKE_P, DEPOL_DUR, SPIKES, THETA = 0.010, 0.002, 4, 0.125
DURATIONS = [0.05, 0.15, 0.35, 0.60]     # -> ~600, 1300, 2600, 3500 dimers (low..high)
SEEDS = [0, 1, 2, 3, 4]
# pre-registered bins/thresholds (mirror the prereg doc)
LOW_MAX, HIGH_MIN = 1500, 3000
D_LOW_STRUCT, D_HIGH_SWAMP = 0.40, 0.15
SYNC_BLOB_MIN = 0.60

def _imports():
    logging.disable(logging.INFO)
    for _n in ['model6_core','multi_synapse_network','dimer_particles','analytical_calcium_system',
               'atp_system','ca_triphosphate_complex','quantum_coherence','pH_dynamics','dopamine_system',
               'em_tryptophan_module','em_coupling_module','local_dimer_tubulin_coupling','camkii_module',
               'spine_plasticity_module','photon_emission_module','photon_receiver_module','ddsc_module',
               'vibrational_cascade_module']:
        logging.getLogger(_n).setLevel(logging.ERROR)
    global Model6Parameters, Model6QuantumSynapse, MultiSynapseNetwork, compute_synapse_quotient_betti
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from entanglement_topology import compute_synapse_quotient_betti

def build_line(seed):
    pos = np.zeros((N,3)); pos[:,0] = np.arange(N)*SPACING
    p = Model6Parameters(); p.em_coupling_enabled=True; p.multi_synapse_enabled=True; p.environment.fraction_P31=1.0
    net = MultiSynapseNetwork(n_synapses=N, pattern="linear", spacing_um=SPACING)
    net.positions = pos
    net.distances = net._compute_distances()
    net.coupling_weights = net._compute_coupling_weights()
    net.initialize(Model6QuantumSynapse, p)
    net.fidelity_weights = np.exp(-net.distances/net.fidelity_length_um); np.fill_diagonal(net.fidelity_weights,1.0)
    for s in net.synapses: s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net

def theta_v(t):
    ph = t % THETA
    return (-10e-3 if (ph < SPIKES*SPIKE_P and (ph % SPIKE_P) < DEPOL_DUR) else -70e-3)

def quotient(net):
    tr = net.entanglement_tracker
    tr.collect_dimers(net.synapses, net.positions)
    cross = tr.cross_synapse_bonds
    q = compute_synapse_quotient_betti(tr.all_dimers, cross, werner_bound=tr.WERNER_ENTANGLEMENT_BOUND)
    gid_syn = {d['global_id']: d['synapse_idx'] for d in tr.all_dimers}
    edges = set()
    for (i,j),f in cross.items():
        if f > tr.WERNER_ENTANGLEMENT_BOUND:
            a,b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a!=b: edges.add((min(a,b),max(a,b)))
    cc = sum(1 for a,b in edges if a in C and b in C)
    gg = sum(1 for a,b in edges if a in G and b in G)
    cg = sum(1 for a,b in edges if (a in C)!=(b in C))
    tot = cc+cg+gg
    return dict(b0=q.betti0, n_edges=len(edges),
                largest_frac=(max(q.component_sizes)/N if q.component_sizes else 0.0),
                CC=cc, CG=cg, GG=gg, cpef=(cg/tot if tot else 0.0),
                g_incl=len({s for e in edges for s in e} & set(G))/len(G))

def run(condition, seconds, seed):
    np.random.seed(seed)
    net = build_line(seed)
    peak_eta = np.zeros(N)
    for k in range(int(seconds/DT)):
        t = k*DT
        for i,s in enumerate(net.synapses):
            s.spine_plasticity.actin_enlargement = 2.0
            s.spine_plasticity.E_invasion = 1.0
            tt = t if (condition=="SYNC" or i in C) else t + THETA/2
            s.step(DT, {'voltage': theta_v(tt), 'reward': False})
        net._update_backbone_field()
        peak_eta = np.maximum(peak_eta, [float(getattr(s,'_backbone_eta',0.0)) for s in net.synapses])
        if k % 10 == 0:
            net._network_entanglement = net.entanglement_tracker.step(
                DT, net.synapses, net.positions, coupling_weights=net.fidelity_weights)
    nd = sum(len(s.dimer_particles.dimers) for s in net.synapses)
    q = quotient(net); q.update(dimers=nd, peak_eta=float(np.mean(peak_eta)),
                                condition=condition, seconds=seconds, seed=seed)
    return q

def score(rows):
    """Apply the pre-registered verdict to collected rows."""
    def agg(cond, lo, hi):
        vals = [r['cpef'] for r in rows if r['condition']==cond and lo <= r['dimers'] < hi]
        return float(np.mean(vals)) if vals else None
    # Δ per duration bin (matched by nominal duration)
    curve = []
    for secs in DURATIONS:
        sy = [r['cpef'] for r in rows if r['condition']=="SYNC" and r['seconds']==secs]
        st = [r['cpef'] for r in rows if r['condition']=="STAGGER" and r['seconds']==secs]
        dm = [r['dimers'] for r in rows if r['seconds']==secs]
        if sy and st:
            curve.append(dict(seconds=secs, dimers_mean=float(np.mean(dm)),
                              cpef_sync=float(np.mean(sy)), cpef_stag=float(np.mean(st)),
                              delta=float(np.mean(sy)-np.mean(st))))
    low = [c for c in curve if c['dimers_mean'] <= LOW_MAX]
    high = [c for c in curve if c['dimers_mean'] >= HIGH_MIN]
    d_low = max((c['delta'] for c in low), default=None)      # best low-dimer separation
    d_high = min((c['delta'] for c in high), default=None)    # residual high-dimer separation
    sync_blob = max((c['cpef_sync'] for c in high), default=None)
    swamp = next((c['dimers_mean'] for c in curve if c['delta'] < 0.20), None)
    peak_eta = float(np.mean([r['peak_eta'] for r in rows]))
    verdict = "INDETERMINATE"
    if sync_blob is not None and sync_blob < 0.5:
        verdict = "RIG-BROKEN (SYNC never blobs)"
    elif d_low is not None and d_high is not None:
        if d_low >= D_LOW_STRUCT and d_high <= D_HIGH_SWAMP:
            verdict = "STRUCTURED / FIXABLE — Step 2 worth it"
        elif d_high is not None and max((c['delta'] for c in curve), default=0) <= D_HIGH_SWAMP:
            verdict = "IRREDUCIBLE / BLOB — Step 2 NOT worth it; (A) classical stands"
    return dict(curve=curve, d_low=d_low, d_high=d_high, sync_blob_high=sync_blob,
                swamp_dimers=swamp, mean_peak_eta=peak_eta, verdict=verdict)

def main():
    _imports()
    t0 = time.time()
    outdir = os.path.join(M6, "results", "step1_partition"); os.makedirs(outdir, exist_ok=True)
    rows = []
    print(f"[{time.time()-t0:.0f}s] START ppid={os.getppid()} pid={os.getpid()}  "
          f"{len(SEEDS)}seeds x {len(DURATIONS)}durs x 2conds = {len(SEEDS)*len(DURATIONS)*2} runs", flush=True)
    for secs in DURATIONS:
        for cond in ["SYNC", "STAGGER"]:
            for seed in SEEDS:
                r = run(cond, secs, seed); rows.append(r)
                print(f"[{time.time()-t0:.0f}s] {cond:>8} {secs:5.3f}s seed{seed} "
                      f"dimers={r['dimers']:5d} cpef={r['cpef']:.3f} largest={r['largest_frac']:.2f} "
                      f"b0={r['b0']} pkEta={r['peak_eta']:.3f}", flush=True)
            json.dump(rows, open(os.path.join(outdir,"rows_partial.json"),"w"), indent=1)
    result = score(rows)
    json.dump({"rows":rows, "score":result}, open(os.path.join(outdir,"step1_result.json"),"w"), indent=1)
    print(f"\n[{time.time()-t0:.0f}s] ===== VERDICT: {result['verdict']} =====", flush=True)
    print(json.dumps(result, indent=1), flush=True)

def daemonize(logpath):
    if os.fork() > 0: os._exit(0)      # parent exits
    os.setsid()
    if os.fork() > 0: os._exit(0)      # give up session leadership
    sys.stdout.flush(); sys.stderr.flush()
    f = open(logpath, "a", buffering=1)
    os.dup2(f.fileno(), sys.stdout.fileno()); os.dup2(f.fileno(), sys.stderr.fileno())
    devnull = open(os.devnull, "r"); os.dup2(devnull.fileno(), sys.stdin.fileno())

if __name__ == '__main__':
    if "--daemonize" in sys.argv:
        logp = sys.argv[sys.argv.index("--daemonize")+1]
        daemonize(logp)
    main()
