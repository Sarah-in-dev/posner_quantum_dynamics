#!/usr/bin/env python3
"""
COHERENCE-RADIUS probe — is the Werner partition exactly a distance graph?

THE DERIVATION (algebra from the live code, not a hypothesis)
------------------------------------------------------------
Cross-bond fidelity is  F = P_S_i * P_S_j * w_spatial   (multi_synapse_network.py
_update_entanglement), with w_spatial = exp(-d/lambda), lambda = 5 um, and the
Werner cut counts an edge iff F > 0.5 (_find_all_clusters). Therefore two
synapses are entangled iff

    d  <  d*  =  lambda * ln(P_product / 0.5)

Two consequences fall straight out of the Werner bound, and neither is a tuned
number:

  1. HARD COHERENCE FLOOR. w <= 1, so F <= P_S^2. F > 0.5 requires
     P_S > 1/sqrt(2) = 0.7071. Below that the cross-synapse partition cannot
     exist AT ANY DISTANCE. The Werner separability bound IS a coherence
     threshold.
  2. THE RADIUS IS SET BY COHERENCE, and it shrinks as coherence decays:
     P_S=1.00 -> d*=3.47um;  0.95 -> 2.95;  0.90 -> 2.41;  0.85 -> 1.84;
     0.80 -> 1.23;  0.7071 -> 0.  So the partition fragments GEOMETRICALLY over
     the coherence window — the most distant pairs decouple FIRST, in an order
     set by d*(t). A classical scalar eligibility trace decays uniformly and has
     no spatial structure at all; this is what makes the claim discriminating.

WHAT THIS PROBE TESTS (the static half)
---------------------------------------
That the partition's edge set IS exactly the <= d* distance graph — i.e. that
d* GENERATES the partition, rather than merely correlating with it. Same shape
as the Stage 3 chain-vs-ring validation, but sharper: instead of a binary
(tree vs loop) it pre-registers a MULTI-COMPONENT partition with specific
sizes, from geometry alone.

Measured P_S at t=0.08s in this rig is ~0.998 (essentially full coherence, as
expected — the coherence window is ~100s), giving d* = 3.45um. The ladder below
straddles that with margin.

RETRODICTION (free, no run needed): d*=3.45um already explains both Stage 3
geometries exactly — chain @2.5um: nearest 2.5 < 3.45 bonds, next-nearest 5.0
does not -> the 5 observed path edges; ring hexagon @2.5um side: side 2.5 bonds,
next chord 4.33 and opposite 5.0 do not -> the 6 observed hexagon edges.

THE DYNAMIC half (partition fragments far-pairs-first as P_S decays) needs a
long run to let coherence actually fall, and is NOT tested here — at 0.08s
P_S has not moved. See the handoff for that follow-on.

PRE-REGISTERED PREDICTION (written before the run)
--------------------------------------------------
Ladder gaps:      2.0  2.5  4.5  3.0  2.0  4.5  2.8      (um, consecutive)
d* = 3.45um  =>   ok   ok   NO   ok   ok   NO   ok
In 1D, connectivity is decided entirely by consecutive gaps (if i and i+2 are
within d*, i+1 must be), so non-consecutive pairs cannot rescue a broken link.

  => quotient edges       = [(0,1),(1,2),(3,4),(4,5),(6,7)]
  => betti0_cross         = 3
  => component_sizes      = [3, 3, 2]
  => betti1_cross         = 0        (every clump is a path, not a cycle:
                                      0-2 = 4.5um and 3-5 = 5.0um both exceed d*)
  => quotient nodes/edges = 8 / 5

FALSIFIED IF: the observed edge set differs from the predicted one — either a
gap > d* bonds anyway (the rule is not the generator), or a gap < d* fails to
bond (something else gates the partition).

Pump is clamped to the ignited regime (eta=0.26), same as Stage 2/3; valid
because Stage 1 proved eta naturally reachable.
"""
import sys, os, time, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soc_topology_geometry_discriminator import build, clamp_eta
from entanglement_topology import compute_synapse_quotient_betti

LAMBDA = 5.0
GAPS = [2.0, 2.5, 4.5, 3.0, 2.0, 4.5, 2.8]
PRED_EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7)]
PRED_B0, PRED_B1 = 3, 0
PRED_SIZES = [3, 3, 2]


def ladder_positions(gaps):
    xs = np.concatenate([[0.0], np.cumsum(gaps)])
    p = np.zeros((len(xs), 3))
    p[:, 0] = xs
    return p


def d_star(p_product):
    return LAMBDA * np.log(p_product / 0.5) if p_product > 0.5 else 0.0


def main():
    t0 = time.time()
    pos = ladder_positions(GAPS)
    n = len(pos)
    print("=== COHERENCE-RADIUS PROBE (eta=0.26 ignited) ===")
    print(f"ladder x (um): {np.round(pos[:,0],2).tolist()}")
    print(f"gaps     (um): {GAPS}")

    net = build(pos)
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)

    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    dt, seconds = 0.001, 0.08
    for k in range(int(seconds / dt)):
        t = k * dt
        ph = t % theta_period
        v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
        net.step(dt, {"voltage": v, "reward": False})

    tr = net.entanglement_tracker
    tr.collect_dimers(net.synapses, net.positions)
    ps = np.array([d["P_S"] for d in tr.all_dimers])
    ps_med = float(np.median(ps))
    ds = d_star(ps_med * ps_med)
    print(f"\nmeasured P_S (median) = {ps_med:.4f}  over {len(ps)} dimers")
    print(f"implied d* = 5*ln(P^2/0.5) = {ds:.2f} um")
    print(f"coherence floor 1/sqrt2 = 0.7071 -> "
          f"{100*(ps < 0.7071).mean():.1f}% of dimers below it")

    q = compute_synapse_quotient_betti(
        tr.all_dimers, tr.cross_synapse_bonds,
        werner_bound=tr.WERNER_ENTANGLEMENT_BOUND, crosscheck=True)

    # Observed synapse-level edges
    gid_syn = {d["global_id"]: d["synapse_idx"] for d in tr.all_dimers}
    obs = set()
    for (i, j), f in tr.cross_synapse_bonds.items():
        if f > tr.WERNER_ENTANGLEMENT_BOUND:
            a, b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a != b:
                obs.add((min(a, b), max(a, b)))
    obs = sorted(obs)

    print(f"\n{'gap':>5} {'pair':>8} {'< d*?':>6} {'predicted':>10} {'observed':>9}")
    for i, g in enumerate(GAPS):
        pred = g < ds
        seen = (i, i + 1) in obs
        flag = "" if pred == seen else "   <-- MISMATCH"
        print(f"{g:5.1f} {str((i,i+1)):>8} {str(g < ds):>6} "
              f"{str(pred):>10} {str(seen):>9}{flag}")

    print(f"\nPRE-REGISTERED  edges={PRED_EDGES}")
    print(f"                betti0_cross={PRED_B0} sizes={PRED_SIZES} "
          f"betti1_cross={PRED_B1}  V/E=8/5")
    print(f"OBSERVED        edges={obs}")
    print(f"                betti0_cross={q.betti0} sizes={q.component_sizes} "
          f"betti1_cross={q.betti1}  V/E={q.n_nodes}/{q.n_edges}")
    print(f"                crosscheck_ok={q.crosscheck_ok}")

    hit = (obs == sorted(PRED_EDGES) and q.betti0 == PRED_B0
           and q.betti1 == PRED_B1 and q.component_sizes == PRED_SIZES)
    print(f"\n[{time.time()-t0:.0f}s] VERDICT: "
          + ("CONFIRMED — d* GENERATES the partition; the Werner cut IS a "
             "coherence-set distance rule." if hit else
             "FALSIFIED — the observed partition is NOT the <= d* distance "
             "graph. Report the negative; do NOT adjust d* or the Werner bound."))


if __name__ == "__main__":
    main()
