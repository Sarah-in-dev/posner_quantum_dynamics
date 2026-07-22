"""
PO-10 Unit C — PILOT B (classical-blindness): is the classical scalar channel matched between the
two trial types, so it cannot resolve the pairing?

Gating pilot from PREREG_PO10_UNIT_C_WEIGHT_LEVEL_KEYSTONE.md §Preconditions.2.

TWO-PART ARGUMENT:
 (1) CODE INSPECTION (recorded here, verified 2026-07-21): the ONLY cross-synapse channel that reaches
     the readout is the QUANTUM entanglement partition (`cross_synapse_bonds` in _update_entanglement,
     weight = P_S_i·P_S_j·w_spatial). The backbone/metabolic aggregation (p_met_agg) is a COMPARTMENT
     SCALAR (sum over all synapses' metabolic power) — blind to WHICH pairs co-occur. Calcium is
     per-synapse LOCAL (each synapse owns its CalciumSystem; ~117 nm nanodomain; no cross-synapse
     calcium coupling). So the model has NO classical second-order (pairing) channel: the only thing
     that reads "which pairs coincided" is the entanglement partition. Arm 2 (binding off) removes
     exactly that channel.
 (2) EMPIRICAL (this script): record the per-synapse calcium and the COMPARTMENT-AGGREGATE calcium
     time-course for pair1 vs pair2, and show the aggregate scalar (what a classical plateau reader
     sees) is MATCHED, and per-cluster marginals are matched — so nothing first-order leaks the pairing.
     The SPATIAL calcium DOES differ (A,B peak together vs A,C together in the early window) — that is
     the pairing-carrying signal — but it is not available to the compartment scalar the classical
     plasticity path reads.

Shortened drive (WRITE_S=15 s): the matched-ness of the compartment scalar is WRITE_S-independent (it is
2 clusters early + 2 clusters late in BOTH types), so a short run suffices and keeps compute cheap.
No readout / no entanglement scoring needed — calcium is a drive-phase quantity. NO SEEDING.
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

PER, N_CLUST = 4, 4
N_SYN = PER * N_CLUST
DT = 5e-3
WRITE_S = 15.0                        # shortened for the calcium check (matched-ness is WRITE_S-independent)
VOLT, ACT, REST = -40e-3, 0.95, -70e-3
CLUSTERS = {c: list(range(c * PER, c * PER + PER)) for c in range(N_CLUST)}
CLUSTER_NAME = {0: "A", 1: "B", 2: "C", 3: "D"}
EARLY = {"pair1": [0, 1], "pair2": [0, 2]}
LATE = {"pair1": [2, 3], "pair2": [1, 3]}
REC_EVERY = 100                       # downsample: record every 100 steps (0.5 s)
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "po10_unitC")


def active_synapses(t, mode, gap2_s):
    act = set()
    if t < WRITE_S:
        for c in EARLY[mode]:
            act |= set(CLUSTERS[c])
    lo = WRITE_S + gap2_s
    if lo <= t < lo + WRITE_S:
        for c in LATE[mode]:
            act |= set(CLUSTERS[c])
    return act


def syn_calcium_uM(s):
    try:
        return float(np.max(s.calcium.get_concentration())) * 1e6
    except Exception:
        return float("nan")


def one_run(run_id, mode, gap_um, gap2_s):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, fidelity_length_um=214.0)
    centers = [i * gap_um for i in range(N_CLUST)]
    xs = [centers[c] + 0.5 * k for c in range(N_CLUST) for k in range(PER)]
    net.positions = np.array([[x, 0.0, 0.0] for x in xs])
    net.distances = np.sqrt(((net.positions[:, None, :] - net.positions[None, :, :]) ** 2).sum(-1))
    net.coupling_weights = np.exp(-net.distances / 5.0); np.fill_diagonal(net.coupling_weights, 1.0)
    net.fidelity_weights = np.exp(-net.distances / 214.0); np.fill_diagonal(net.fidelity_weights, 1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True

    rels = [PresynapticRelease(None) for _ in range(N_SYN)]
    ts, per_syn_ca = [], []
    drive_end = 2 * WRITE_S + gap2_s
    nsteps = int(round(drive_end / DT))
    t = 0.0
    for step in range(nsteps):
        act = active_synapses(t, mode, gap2_s)
        per_syn = []
        for i in range(N_SYN):
            if i in act:
                g = rels[i].step(ACT, DT)
                per_syn.append({"voltage": VOLT, "glutamate": g})
            else:
                per_syn.append({"voltage": REST, "glutamate": 0.0})
        net.step(DT, {"per_synapse": per_syn, "reward": False})
        if step % REC_EVERY == 0:
            ts.append(round(t, 3))
            per_syn_ca.append([round(syn_calcium_uM(s), 3) for s in net.synapses])
        t += DT

    ca = np.array(per_syn_ca)                                   # [T, N_SYN]
    # per-cluster mean time-course, and the COMPARTMENT-AGGREGATE scalar (sum over all synapses)
    per_cluster = {CLUSTER_NAME[c]: ca[:, np.array(CLUSTERS[c])].mean(axis=1).round(3).tolist()
                   for c in range(N_CLUST)}
    compartment_scalar = ca.sum(axis=1).round(3).tolist()       # what a spatially-unresolved reader sees
    per_cluster_integral = {CLUSTER_NAME[c]: float(ca[:, np.array(CLUSTERS[c])].mean(axis=1).sum() * REC_EVERY * DT)
                            for c in range(N_CLUST)}
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                mode=mode, gap_um=float(gap_um), gap2_s=float(gap2_s), write_s=WRITE_S,
                t=ts, per_cluster_ca=per_cluster, compartment_scalar=compartment_scalar,
                per_cluster_integral=per_cluster_integral,
                compartment_integral=float(np.array(compartment_scalar).sum() * REC_EVERY * DT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--mode", choices=["pair1", "pair2"], required=True)
    ap.add_argument("--gap_um", type=float, default=10.0)
    ap.add_argument("--gap2_s", type=float, default=0.0)
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po10_unitC_{a.tag}_w{a.worker}.jsonl")
    rid = f"{a.tag}_w{a.worker}"
    sys.stderr.write(f"[pilotB {a.tag}] mode={a.mode} gap={a.gap_um} gap2={a.gap2_s}\n"); sys.stderr.flush()
    rec = one_run(rid, a.mode, a.gap_um, a.gap2_s)
    with open(jsonl, "w") as f:
        f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
    sys.stderr.write(f"[pilotB {a.tag}] done; per-cluster Ca integral={rec['per_cluster_integral']} "
                     f"compartment integral={round(rec['compartment_integral'],1)}\n"); sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
