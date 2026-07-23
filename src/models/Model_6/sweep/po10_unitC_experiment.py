"""
PO-10 Unit C — the WEIGHT-LEVEL keystone experiment (PREREG_PO10_UNIT_C_WEIGHT_LEVEL_KEYSTONE.md).

An input (trial type: pair1 {AB,CD} vs pair2 {AC,BD}) goes in; a SIGNED Δw vector comes out; a decoder
asks whether trial type is recoverable from Δw. This is the first measurement at the weight level.

THE READOUT (registered construct-validity choice): the SIGNED joint-collapse structure, NOT the
eligibility-covariance proxy (apply_reward_correlated). Built ENTIRELY here from the tracker's existing
_find_all_clusters() + all_dimers — NO model .py is modified, so it is off-path bit-identical by
construction. Faithful to perform_quantum_measurement (per-component commit coin ~ cluster_P_S), plus the
shared random ±1 per component (the correlated collapse DIRECTION) the pre-reg specifies:
    for each correlated domain (connected component):
        commit? ~ Bernoulli(mean P_S of the domain)         # exactly perform_quantum_measurement
        if commit: draw shared sign s in {-1,+1};  Δw[syn] += s  for each committed dimer in the domain

FOUR ARMS (registered ablation ladder):
  full     : λ_F=214, joint collapse over the real domains.                (binding on)
  bindoff  : λ_F=214, every dimer collapses INDEPENDENTLY (no domains).    (classical control -> chance)
  scramble : λ_F=214, joint collapse but domain MEMBERSHIP permuted, sizes kept. (grouping-by-coincidence control -> chance)
  lamshort : λ_F=5,   joint collapse; cross-bonds below the Werner floor so domains are per-synapse. (physics tie -> chance)

Sign handling: Δw sign is arbitrary per domain, so the decoder (separate scorer) uses the sign-INVARIANT
pairwise agreement structure. Recorded here: Δw (16), per-cluster Δw (4), the 6 pairwise cluster sign
agreements, domain sizes, and the substrate pair_w (cross-check vs Pilot A). NO SEEDING anywhere.
"""
import sys, os, json, argparse
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, SWEEP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
import logging; logging.disable(logging.INFO)
from presynaptic_release import PresynapticRelease
from run_theta_burst_45s import analytical_gap

PER, N_CLUST = 4, 4
N_SYN = PER * N_CLUST
DT = 5e-3
WRITE_S = 30.0
VOLT, ACT, REST = -40e-3, 0.95, -70e-3
WERNER = 0.5
CLUSTERS = {c: list(range(c * PER, c * PER + PER)) for c in range(N_CLUST)}
CLUSTER_NAME = {0: "A", 1: "B", 2: "C", 3: "D"}
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_NAME = {p: CLUSTER_NAME[p[0]] + CLUSTER_NAME[p[1]] for p in PAIRS}
# A pairing is a partition of the 4 clusters into 2 co-active pairs. `mode` names the PAIRING (the
# decoder's label); `order` counterbalances WHICH pair fires early vs late (Amendment 1) so that, across
# a class, each pair is late-position in half its trials — removing the early-pair-decay / positional
# confound that made batch 1 (all forward-order) decode at chance.
PAIRSETS = {"pair1": ([0, 1], [2, 3]),   # pairing {AB},{CD}
            "pair2": ([0, 2], [1, 3])}    # pairing {AC},{BD}
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "po10_unitC")


def active_synapses(t, mode, gap2_s, order):
    first, second = PAIRSETS[mode]
    if order == "rev":
        first, second = second, first
    act = set()
    if t < WRITE_S:
        for c in first:
            act |= set(CLUSTERS[c])
    lo = WRITE_S + gap2_s
    if lo <= t < lo + WRITE_S:
        for c in second:
            act |= set(CLUSTERS[c])
    return act


def synapse_corr_matrix(tr):
    W = np.zeros((N_SYN, N_SYN))
    for key, F in tr.cross_synapse_bonds.items():
        F = float(F)
        if F <= WERNER:
            continue
        si, sj = key[0][0], key[1][0]
        if si == sj:
            continue
        p = (4.0 * F - 1.0) / 3.0
        W[si, sj] += p; W[sj, si] += p
    return W


def signed_readout(tr, arm, rng, fixed_sign=False):
    """Signed Δw from the joint-collapse structure. Returns (dw[N_SYN], domain_sizes).
    fixed_sign=True: every committed domain takes s=+1 (the RECTIFIED-collapse variant — advisor Q1/Q2:
    dopamine/CaMKII biases the sign so it is not a random ±1). Tests whether sign-noise is the decode limiter."""
    dimers = tr.all_dimers
    id_to_dimer = {d['global_id']: d for d in dimers}
    dw = np.zeros(N_SYN)
    if arm == "bindoff":
        for d in dimers:                                  # independent collapse: no domains
            if rng.random() < float(d.get('P_S', 0.25)):
                dw[d['synapse_idx']] += 1.0 if fixed_sign else rng.choice([-1.0, 1.0])
        return dw, [1] * len(dimers)
    clusters = tr._find_all_clusters()                    # list of lists of global_ids (correlated domains)
    if arm == "scramble":                                 # keep domain sizes, permute membership
        sizes = [len(c) for c in clusters]
        all_ids = [g for c in clusters for g in c]
        rng.shuffle(all_ids)
        clusters, i = [], 0
        for s in sizes:
            clusters.append(all_ids[i:i + s]); i += s
    sizes = []
    for cluster_ids in clusters:
        ps = np.mean([id_to_dimer[g]['P_S'] for g in cluster_ids if g in id_to_dimer])
        sizes.append(len(cluster_ids))
        if rng.random() < ps:                             # per-domain commit coin (perform_quantum_measurement)
            s = 1.0 if fixed_sign else rng.choice([-1.0, 1.0])   # rectified (+1) vs random ±1 collapse direction
            for g in cluster_ids:
                d = id_to_dimer.get(g)
                if d is not None:
                    dw[d['synapse_idx']] += s
    return dw, sizes


def cluster_fragmentation(tr):
    """Advisor Loss-2 diagnostic: for each cluster, the fraction of its dimers in its MODAL (largest)
    real correlated domain. 1.0 = the whole cluster shares one domain (clean shared sign); < 1 dilutes
    the agreement signal toward zero. Uses the REAL (unscrambled) domains."""
    from collections import Counter
    clusters = tr._find_all_clusters()
    dom_of = {g: di for di, cl in enumerate(clusters) for g in cl}
    frac = {}
    for c in range(N_CLUST):
        syns = set(CLUSTERS[c])
        doms = [dom_of[d['global_id']] for d in tr.all_dimers
                if d['synapse_idx'] in syns and d['global_id'] in dom_of]
        frac[CLUSTER_NAME[c]] = round(max(Counter(doms).values()) / len(doms), 3) if doms else 0.0
    return frac


def one_run(run_id, mode, arm, gap_um, gap2_s, delay_s, order, fixed_sign=False):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    lam_F = 5.0 if arm == "lamshort" else 214.0
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, fidelity_length_um=lam_F)
    centers = [i * gap_um for i in range(N_CLUST)]
    xs = [centers[c] + 0.5 * k for c in range(N_CLUST) for k in range(PER)]
    net.positions = np.array([[x, 0.0, 0.0] for x in xs])
    net.distances = np.sqrt(((net.positions[:, None, :] - net.positions[None, :, :]) ** 2).sum(-1))
    net.coupling_weights = np.exp(-net.distances / 5.0); np.fill_diagonal(net.coupling_weights, 1.0)
    net.fidelity_weights = np.exp(-net.distances / lam_F); np.fill_diagonal(net.fidelity_weights, 1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker

    rels = [PresynapticRelease(None) for _ in range(N_SYN)]
    peak_eta_per = np.zeros(N_SYN)
    nsteps = int(round((2 * WRITE_S + gap2_s) / DT))
    t = 0.0
    for _ in range(nsteps):
        act = active_synapses(t, mode, gap2_s, order)
        per_syn = [{"voltage": VOLT, "glutamate": rels[i].step(ACT, DT)} if i in act
                   else {"voltage": REST, "glutamate": 0.0} for i in range(N_SYN)]
        net.step(DT, {"per_synapse": per_syn, "reward": False})
        peak_eta_per = np.maximum(peak_eta_per, [float(getattr(s, "_backbone_eta", 0.0)) for s in net.synapses])
        t += DT

    if delay_s > 0:
        analytical_gap(net, delay_s, dt_sub=1.0)
        net.step(DT, {"voltage": REST, "reward": False})

    rng = np.random.default_rng()                          # free draw, NO seed
    frag = cluster_fragmentation(tr)                        # advisor Loss-2 diagnostic (real domains)
    dw, dom_sizes = signed_readout(tr, arm, rng, fixed_sign)
    dw_cluster = {CLUSTER_NAME[c]: float(dw[np.array(CLUSTERS[c])].sum()) for c in range(N_CLUST)}
    csign = {k: int(np.sign(v)) for k, v in dw_cluster.items()}
    agree = {PAIR_NAME[p]: int(csign[CLUSTER_NAME[p[0]]] * csign[CLUSTER_NAME[p[1]]]) for p in PAIRS}
    W = synapse_corr_matrix(tr)
    pair_w = {PAIR_NAME[p]: float(W[np.ix_(np.array(CLUSTERS[p[0]]), np.array(CLUSTERS[p[1]]))].sum() * 2)
              for p in PAIRS}
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                mode=mode, arm=arm, order=order, fixed_sign=bool(fixed_sign),
                lam_F=lam_F, gap_um=float(gap_um), gap2_s=float(gap2_s),
                cluster_frag=frag,
                delay_s=float(delay_s), ignited=bool(peak_eta_per.max() > 0.0),
                dw=[round(float(x), 3) for x in dw], dw_cluster={k: round(v, 3) for k, v in dw_cluster.items()},
                cluster_sign=csign, agree=agree, n_domains=len(dom_sizes),
                domain_sizes_top=sorted(dom_sizes, reverse=True)[:6],
                pair_w={k: round(v) for k, v in pair_w.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--mode", choices=["pair1", "pair2"], required=True)
    ap.add_argument("--arm", choices=["full", "bindoff", "scramble", "lamshort"], required=True)
    ap.add_argument("--order", choices=["fwd", "rev"], default="fwd")
    ap.add_argument("--fixed_sign", action="store_true",
                    help="rectified collapse: every committed domain takes +1 (advisor Q1/Q2 test)")
    ap.add_argument("--gap_um", type=float, default=10.0)
    ap.add_argument("--gap2_s", type=float, default=0.0)
    ap.add_argument("--delay_s", type=float, default=20.0)
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po10_unitC_{a.tag}_w{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"{a.tag}_w{a.worker}r{j}"
        sys.stderr.write(f"[unitC {a.tag} w{a.worker}] draw {j+1}/{a.n} mode={a.mode} arm={a.arm}\n"); sys.stderr.flush()
        rec = one_run(rid, a.mode, a.arm, a.gap_um, a.gap2_s, a.delay_s, a.order, a.fixed_sign)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        sys.stderr.write(f"[unitC {a.tag} w{a.worker}] done {rid} ignited={rec['ignited']} "
                         f"n_dom={rec['n_domains']} agree={rec['agree']}\n"); sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
