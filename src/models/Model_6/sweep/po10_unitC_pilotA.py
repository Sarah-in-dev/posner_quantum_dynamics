"""
PO-10 Unit C — PILOT A (substrate-level): does the partition DISTINGUISH the pairings
in a one-compartment, 4-cluster geometry, despite the branch-global-ignition risk (L·ETA-4)?

This is a GATING pilot from PREREG_PO10_UNIT_C_WEIGHT_LEVEL_KEYSTONE.md §Preconditions.1.
It does NOT touch weights / Δw — it measures the SUBSTRATE partition (the cross-cluster
correlation-block structure), exactly the L·PO9-2 machinery, extended from 2 clusters to 4.

Question: with 4 clusters A,B,C,D in one compartment (all within λ_F, spaced > λ_met so each
ignites on its own drive), does
    trial type 1  ({A,B} co-active, {C,D} co-active)  ->  partition {AB},{CD}
    trial type 2  ({A,C} co-active, {B,D} co-active)  ->  partition {AC},{BD}
i.e. does the 4x4 cluster correlation matrix put weight on the CO-ACTIVE pairs and not the others?

If YES: the pairing survives the one-compartment geometry -> Unit C is runnable.
If NO (branch-global washout, or persistence merges everything into one blob): FIX GEOMETRY/TIMING
before building the Δw readout. Either way it is a result about the instrument, reported honestly.

Nothing here modifies model .py. Reuses the po9_unitB net setup verbatim (positions, coupling_weights
on λ_met=5 µm, fidelity_weights on λ_F, provenance+spin bonding, analytical_gap for the readout delay).
NO SEEDING.
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
from run_theta_burst_45s import analytical_gap   # PO-4 surface; CALLED, never edited

PER = 4                               # synapses per cluster (ignition quorum ~4, per L·PO9-2)
N_CLUST = 4
N_SYN = PER * N_CLUST                  # 16
DT = 5e-3
WRITE_S = 30.0                        # per-window drive; > ignition latency (~13-15 s) so a pair co-ignites
VOLT, ACT = -40e-3, 0.95
REST = -70e-3
WERNER = 0.5
CLUSTERS = {c: list(range(c * PER, c * PER + PER)) for c in range(N_CLUST)}   # 0:A 1:B 2:C 3:D
CLUSTER_NAME = {0: "A", 1: "B", 2: "C", 3: "D"}
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "po10_unitC")

# Two coincidence WINDOWS. Each cluster is active in exactly ONE window (duration WRITE_S) in BOTH
# trial types -> marginals matched by construction. Only WHICH pairs share a window differs.
#   pair1: early {A,B}, late {C,D}   -> expected partition {AB},{CD}
#   pair2: early {A,C}, late {B,D}   -> expected partition {AC},{BD}
#   simul: all four in the early window (smoke test: feasibility + independent ignition + one-blob check)
EARLY = {"pair1": [0, 1], "pair2": [0, 2], "simul": [0, 1, 2, 3]}
LATE = {"pair1": [2, 3], "pair2": [1, 3], "simul": []}


def cluster_centers(gap_um):
    return [i * gap_um for i in range(N_CLUST)]


def active_synapses(t, mode, gap2_s):
    """Which synapses are driven at time t. early window [0,WRITE_S); late window [WRITE_S+gap2, ...)."""
    act = set()
    if t < WRITE_S:
        for c in EARLY[mode]:
            act |= set(CLUSTERS[c])
    lo = WRITE_S + gap2_s
    if lo <= t < lo + WRITE_S:
        for c in LATE[mode]:
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
        W[si, sj] += p
        W[sj, si] += p
    return W


def cluster_block_weights(W):
    """Aggregate the 16x16 synapse matrix to the 6 inter-cluster pair weights + within-cluster totals."""
    pair_w, within_w = {}, {}
    for c in range(N_CLUST):
        idx = np.array(CLUSTERS[c])
        within_w[CLUSTER_NAME[c]] = float(W[np.ix_(idx, idx)].sum())
    for a in range(N_CLUST):
        for b in range(a + 1, N_CLUST):
            ia, ib = np.array(CLUSTERS[a]), np.array(CLUSTERS[b])
            key = CLUSTER_NAME[a] + CLUSTER_NAME[b]
            pair_w[key] = float(W[np.ix_(ia, ib)].sum() + W[np.ix_(ib, ia)].sum())
    return pair_w, within_w


def one_run(run_id, mode, lam_F, gap_um, gap2_s, delay_s):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, fidelity_length_um=float(lam_F))  # NO seed
    centers = cluster_centers(gap_um)
    xs = [centers[c] + 0.5 * k for c in range(N_CLUST) for k in range(PER)]   # 0.5 µm within-cluster
    net.positions = np.array([[x, 0.0, 0.0] for x in xs])
    net.distances = np.sqrt(((net.positions[:, None, :] - net.positions[None, :, :]) ** 2).sum(-1))
    net.coupling_weights = np.exp(-net.distances / 5.0); np.fill_diagonal(net.coupling_weights, 1.0)
    net.fidelity_weights = np.exp(-net.distances / float(lam_F)); np.fill_diagonal(net.fidelity_weights, 1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker

    rels = [PresynapticRelease(None) for _ in range(N_SYN)]
    peak_eta_per = np.zeros(N_SYN)
    drive_end = WRITE_S if mode == "simul" else (2 * WRITE_S + gap2_s)
    nsteps = int(round(drive_end / DT))
    t = 0.0
    for _ in range(nsteps):
        act = active_synapses(t, mode, gap2_s)
        per_syn = []
        for i in range(N_SYN):
            if i in act:
                g = rels[i].step(ACT, DT)
                per_syn.append({"voltage": VOLT, "glutamate": g})
            else:
                per_syn.append({"voltage": REST, "glutamate": 0.0})
        net.step(DT, {"per_synapse": per_syn, "reward": False})
        etas = np.array([float(getattr(s, "_backbone_eta", 0.0)) for s in net.synapses])
        peak_eta_per = np.maximum(peak_eta_per, etas)
        t += DT

    # readout at a single delay
    if delay_s > 0:
        analytical_gap(net, delay_s, dt_sub=1.0)
        net.step(DT, {"voltage": REST, "reward": False})
    W = synapse_corr_matrix(tr)
    pair_w, within_w = cluster_block_weights(W)
    ps = [float(dd["P_S"]) for dd in tr.all_dimers]
    per_cluster_eta = {CLUSTER_NAME[c]: round(float(peak_eta_per[np.array(CLUSTERS[c])].mean()), 4)
                       for c in range(N_CLUST)}
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                mode=mode, lam_F=float(lam_F), gap_um=float(gap_um), gap2_s=float(gap2_s),
                delay_s=float(delay_s), n_syn=N_SYN,
                pair_w=pair_w, within_w=within_w,
                per_cluster_peak_eta=per_cluster_eta,
                ignited=bool(peak_eta_per.max() > 0.0),
                mean_PS=float(np.mean(ps)) if ps else 0.0,
                n_cross_edges=int((W > 0).sum() // 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--mode", choices=["simul", "pair1", "pair2"], required=True)
    ap.add_argument("--lamF", type=float, default=214.0)
    ap.add_argument("--gap_um", type=float, default=10.0, help="inter-cluster center spacing (µm)")
    ap.add_argument("--gap2_s", type=float, default=0.0, help="silent gap between the two drive windows (s)")
    ap.add_argument("--delay_s", type=float, default=20.0, help="readout delay (s)")
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po10_unitC_{a.tag}_w{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"{a.tag}_w{a.worker}r{j}"
        sys.stderr.write(f"[pilotA {a.tag} w{a.worker}] draw {j+1}/{a.n} mode={a.mode} "
                         f"lamF={a.lamF} gap={a.gap_um} gap2={a.gap2_s} delay={a.delay_s}\n"); sys.stderr.flush()
        rec = one_run(rid, a.mode, a.lamF, a.gap_um, a.gap2_s, a.delay_s)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        sys.stderr.write(f"[pilotA {a.tag} w{a.worker}] done {rid} ignited={rec['ignited']} "
                         f"eta={rec['per_cluster_peak_eta']} pair_w={ {k: round(v,1) for k,v in rec['pair_w'].items()} }\n")
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
