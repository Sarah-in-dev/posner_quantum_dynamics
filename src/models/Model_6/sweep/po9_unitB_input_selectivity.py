#!/usr/bin/env python3
"""
PO-9 UNIT B — the readout-time input-selectivity keystone. NO SEEDING.

Pre-registered: docs/PREREG_PO9_UNIT_B_READOUT_KEYSTONE.md. Read it before changing anything.

QUESTION: does the correlated-domain partition AT READOUT carry WHICH INPUT was presented,
beyond density and geometry, AS A FUNCTION OF lambda_F?

PROTOCOL per draw:  per-synapse WRITE (SYNC or STAGGER, density matched, ALL driven)
                    -> analytical_gap(delay)  (AGE, the confined-niche intrinsic mechanism)
                    -> score the SYNAPSE-level correlation graph vs the input grouping.

SCORING (synapse level; the dimer-level trap is documented in the prereg): the 7-node synapse
graph has edge weight W_ij = sum over cross-bridges (i,j) with F>Werner of p_e=(4F-1)/3 (total
correlation flux between the two synapses). Q_act = Newman modularity of that weighted graph
against the input-group partition {A}{B}{spacer}. NULL BAND = Q against >=200 random equal-size
relabellings; z = (Q_true - mean_null)/std_null. Report z per (condition, lambda_F, delay).

VERDICT DEMONSTRATED FAILING FIRST: run condition SYNCxSYNC (identity removed) and confirm z is
within the null band before trusting any positive on SYNC-vs-STAGGER.

lambda decoupling (L.PO9-1): metabolic length stays 5 um; fidelity_length_um = lambda_F is swept.
Drive via net.step per-synapse ONLY (per-synapse s.step() leaves eta=0 -> nothing ignites).
"""
import sys, os, json, math, argparse
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, SWEEP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
import logging; logging.disable(logging.INFO)

from presynaptic_release import PresynapticRelease
from run_theta_burst_45s import analytical_gap   # PO-4 surface; CALLED, never edited

N_SYN, SPACING = 7, 1.0
DT = 5e-3
WRITE_S = 20.0                       # per-group write window
VOLT, ACT = -40e-3, 0.95
REST = -70e-3
WERNER = 0.5
DELAYS = [0.0, 10.0, 20.0, 30.0, 40.0, 60.0]
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "po9_unitB")

# groupings: (A, B, spacer). contiguous => A is the spatial half (geometry-confounded);
# interleaved => identity is ORTHOGONAL to space (the geometry control).
GROUPINGS = {
    "contig": ([0, 1, 2], [4, 5, 6], [3]),
    "interleave": ([0, 2, 4], [1, 3, 5], [6]),
}


def synapse_corr_matrix(tr):
    """W_ij = sum over cross-bridges between synapse i and j (F>Werner) of p_e=(4F-1)/3."""
    W = np.zeros((N_SYN, N_SYN))
    for key, F in tr.cross_synapse_bonds.items():
        F = float(F)
        if F <= WERNER:
            continue
        si, sj = key[0][0], key[1][0]
        if si == sj:
            continue
        p = (4.0 * F - 1.0) / 3.0
        W[si, sj] += p
        W[sj, si] += p
    return W


def modularity(W, labels):
    """Newman weighted modularity for the given community labels (spacer is its own community)."""
    m2 = W.sum()
    if m2 <= 0:
        return 0.0
    k = W.sum(axis=1)
    Q = 0.0
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += W[i, j] - k[i] * k[j] / m2
    return Q / m2


def score_Qact(W, A, B, spacer, n_null=200):
    labels = np.empty(N_SYN, dtype=int)
    for i in A: labels[i] = 0
    for i in B: labels[i] = 1
    for i in spacer: labels[i] = 2
    Q_true = modularity(W, labels)
    sizes = [len(A), len(B), len(spacer)]
    nulls = []
    for _ in range(n_null):
        perm = np.random.permutation(N_SYN)   # free draw; no seed
        lab = np.empty(N_SYN, dtype=int)
        lab[perm[:sizes[0]]] = 0
        lab[perm[sizes[0]:sizes[0] + sizes[1]]] = 1
        lab[perm[sizes[0] + sizes[1]:]] = 2
        nulls.append(modularity(W, lab))
    nulls = np.asarray(nulls)
    mu, sd = float(nulls.mean()), float(nulls.std())
    z = (Q_true - mu) / sd if sd > 1e-12 else 0.0
    return dict(Q_true=float(Q_true), null_mu=mu, null_sd=sd,
                null_p05=float(np.percentile(nulls, 5)), null_p95=float(np.percentile(nulls, 95)),
                z=float(z), edge_weight_total=float(W.sum()))


def active_at(t, cond, A, B, spacer):
    """Which synapses carry drive at time t. Density matched: every group synapse is driven for
    exactly WRITE_S. SYNC: A,B both in [0,WRITE_S]. STAGGER: A in [0,WRITE_S], B in [WRITE_S,2W].
    Spacer in [0,WRITE_S] in both. SYNC2 (failing-first control): identical to SYNC."""
    act = set()
    if cond in ("sync", "sync2"):
        if t < WRITE_S: act |= set(A) | set(B) | set(spacer)
    elif cond == "stagger":
        if t < WRITE_S: act |= set(A) | set(spacer)
        if WRITE_S <= t < 2 * WRITE_S: act |= set(B)
    return act


def write_end(cond):
    return 2 * WRITE_S if cond == "stagger" else WRITE_S


def one_run(run_id, lam_F, cond, grouping):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    A, B, spacer = GROUPINGS[grouping]

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING,
                              coupling_length_um=5.0, fidelity_length_um=float(lam_F))  # NO seed
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker

    rels = [PresynapticRelease(None) for _ in range(N_SYN)]   # independent free draws per synapse
    peak_eta = 0.0
    t = 0.0
    nsteps = int(round(write_end(cond) / DT))
    for _ in range(nsteps):
        act = active_at(t, cond, A, B, spacer)
        per_syn = []
        for i in range(N_SYN):
            if i in act:
                g = rels[i].step(ACT, DT)
                per_syn.append({"voltage": VOLT, "glutamate": g})
            else:
                per_syn.append({"voltage": REST, "glutamate": 0.0})
        net.step(DT, {"per_synapse": per_syn, "reward": False})
        peak_eta = max(peak_eta, max((float(getattr(s, "_backbone_eta", 0.0))
                                      for s in net.synapses), default=0.0))
        t += DT

    snaps, prev = [], 0.0
    for d in DELAYS:
        if d > prev:
            analytical_gap(net, d - prev, dt_sub=1.0)
            net.step(DT, {"voltage": REST, "reward": False})   # tail refresh (needs coupling_weights)
            prev = d
        W = synapse_corr_matrix(tr)
        sc = score_Qact(W, A, B, spacer)
        sc["delay"] = d
        sc["n_cross_syn_edges"] = int((W > 0).sum() // 2)
        snaps.append(sc)
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                lam_F=float(lam_F), cond=cond, grouping=grouping,
                peak_eta=float(peak_eta), ignited=bool(peak_eta > 0.0),
                write_end_s=write_end(cond), snapshots=snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--lamF", type=float, required=True)
    ap.add_argument("--cond", choices=["sync", "stagger", "sync2"], required=True)
    ap.add_argument("--grouping", choices=list(GROUPINGS), default="contig")
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po9_unitB_{a.tag}_w{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"{a.tag}_w{a.worker}r{j}"
        sys.stderr.write(f"[unitB {a.tag} w{a.worker}] draw {j+1}/{a.n} "
                         f"lamF={a.lamF} cond={a.cond} grp={a.grouping}\n"); sys.stderr.flush()
        rec = one_run(rid, a.lamF, a.cond, a.grouping)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        zc = {s["delay"]: round(s["z"], 1) for s in rec["snapshots"]}
        sys.stderr.write(f"[unitB {a.tag} w{a.worker}] done {rid} ignited={rec['ignited']} "
                         f"z_by_delay={zc}\n"); sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
