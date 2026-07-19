#!/usr/bin/env python3
"""
PO-5 UNIT 5 — a GENUINE (non-constant) cellular sheaf on the dimer graph, and its
cohomology across the Unit-4 bus sweep.

THE QUESTION
------------
`entanglement-topology-measurement` A2 rules the model is "a PLAIN WEIGHTED GRAPH, not a
sheaf", on the grounds that a non-trivial restriction map "would have to be relative phase /
holonomy", which the (A) model forecloses. **That inference is the weak link.** A cellular
sheaf over R^n with REAL linear restriction maps is entirely classical (Hansen & Ghrist 2019,
"Toward a spectral theory of cellular sheaves"). Phase is SUFFICIENT for non-triviality, not
NECESSARY. A coordinate projection R^6 -> R is non-identity, non-product, and carries no phase.

THE STALK ALREADY EXISTS AND IS DEAD
------------------------------------
`dimer_particles.py:40` gives every dimer `j_couplings_intra`: 6 real J-couplings among its
4 P31 spins (Agarwal DFT). Its ONLY use is `:310-311`, where it is collapsed to std/mean and
never seen by the entanglement layer. This unit uses it as the vertex stalk.

THE CONSTRUCTION
----------------
  vertex stalk  F(v) = R^6      (j_couplings_intra)
  edge stalk    F(e) = R^1      (the coupling channel the bond engages)
  restriction   F_{v|e} : R^6 -> R,  projection onto channel k(v,e)
  coboundary    (delta x)_e = x_v[k(v,e)] - x_u[k(u,e)]
  sheaf Laplacian L = delta^T delta ;  H0 = ker L ;  dim H1 = dim C1 - rank delta

CHANNEL RULE (a stated MODELLING CHOICE, not a derivation): the channel a bond engages is set
by the bond's direction -- which spin pair faces the partner. Direction is binned to one of 6
by dominant axis and sign. Endpoint u sees +d and v sees -d, so k(u,e) != k(v,e) in general:
the two restrictions differ, which is what makes the sheaf non-constant.

WHY THIS IS COMPUTABLE WITHOUT LINEAR ALGEBRA
---------------------------------------------
With coordinate-projection restrictions, (delta x)_e = 0 iff x_v[k_v] == x_u[k_u]. So ker delta
is exactly the set of functions on the 6V coordinate-nodes that are CONSTANT ON CONNECTED
COMPONENTS of the derived "channel graph" G', whose nodes are (dimer, channel) pairs and whose
edges join (v,k_v)--(u,k_u) for each bond. Hence
    dim H0 = #components(G')            [union-find, O(E alpha)]
    rank delta = 6V - dim H0
    dim H1 = E - rank delta = E - 6V + dim H0
No 380795 x 7092 eigendecomposition is required.

PRE-REGISTERED PREDICTIONS (before the run)
-------------------------------------------
  P1 VALIDATION (must pass or the run is INVALID): with the CONSTANT sheaf (1 channel, identity
     restrictions) dim H0 MUST equal the ordinary connected-component count at every bus value,
     and dim H1 must equal the ordinary cycle rank E - V + c. This is the known case; if the
     implementation cannot reproduce it, no other number here is admissible.
  P2 THE HYPOTHESIS: at the native operating point, where the ordinary component count is
     pinned at 1 and carries no information, the NON-CONSTANT sheaf is non-degenerate.
     *** AMENDMENT A5.1, registered BEFORE the run on real data. *** P2 is scored on
     H0_ENGAGED, not raw H0. A toy check (2 triangles) showed raw H0 = 30 on 36 coordinate
     nodes: coupling channels that NO bond engages are unconstrained and free by
     construction, so raw H0 is inflated by them and is trivially > 1 always. That would be
     an artifact, not evidence. H0_engaged counts components only among (dimer, channel)
     coordinates that at least one bond actually engages, and is the honest comparator.
  P3 If sheaf H0 also collapses to 1 wherever ordinary components = 1, the sheaf readout is
     NOT more informative here and the hypothesis FAILS. That is a real outcome and is reported
     as one.
  CONTROL: a RANDOM channel assignment arm. If geometry-derived channels and random channels
     give the same cohomology, the structure is an artifact of having 6 coordinates rather than
     of the physics, and P2 is not evidence for the physical claim.
"""

import sys, os, json, time
import logging
import numpy as np

logging.disable(logging.INFO)

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

BUS_VALUES = [0.0, 0.5, 2.0, 10.0, 20.0]     # spans Unit 4's fragmented -> blob range
T_SIM, DT = 1.0, 0.005
N_CHANNELS = 6


# ---------------------------------------------------------------------------
class UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
    def n_components(self, nodes):
        return len({self.find(n) for n in nodes})


def channel_from_direction(d, rng=None):
    """Bin a unit direction to one of 6 channels by dominant axis and sign."""
    if rng is not None:
        return int(rng.integers(0, N_CHANNELS))
    ax = int(np.argmax(np.abs(d)))
    return ax * 2 + (0 if d[ax] >= 0 else 1)


def sheaf_cohomology(dimer_ids, positions, edges, mode, seed=0):
    """dim H0 / dim H1 for the sheaf described in the module docstring.

    mode = 'constant'  -> 1 channel, identity restrictions (the KNOWN case, P1)
           'geometry'  -> channel from bond direction (the physical proposal)
           'random'    -> channel drawn at random (the CONTROL)
    """
    rng = np.random.default_rng(seed) if mode == "random" else None
    V = len(dimer_ids)
    E = len(edges)
    nch = 1 if mode == "constant" else N_CHANNELS

    uf = UF()
    nodes = {(v, k) for v in dimer_ids for k in range(nch)}
    for n in nodes:
        uf.find(n)

    engaged = set()
    for (a, b) in edges:
        if mode == "constant":
            ka = kb = 0
        else:
            d = positions[b] - positions[a]
            nrm = np.linalg.norm(d)
            if nrm == 0:
                ka = kb = 0
            else:
                d = d / nrm
                ka = channel_from_direction(d, rng)
                kb = channel_from_direction(-d, rng)
        uf.union((a, ka), (b, kb))
        engaged.add((a, ka)); engaged.add((b, kb))

    h0 = uf.n_components(nodes)
    dim_C0 = V * nch
    rank_delta = dim_C0 - h0
    h1 = E - rank_delta
    h0_engaged = uf.n_components(engaged) if engaged else 0
    return {"dim_C0": dim_C0, "dim_C1": E, "rank_delta": rank_delta,
            "H0": h0, "H1": h1, "H0_engaged": h0_engaged,
            "n_engaged_coords": len(engaged)}


def ordinary_topology(dimer_ids, edges):
    uf = UF()
    for v in dimer_ids:
        uf.find(v)
    for a, b in edges:
        uf.union(a, b)
    c = uf.n_components(dimer_ids)
    return c, len(edges) - len(dimer_ids) + c


# ---------------------------------------------------------------------------
def run(bus, seed=7777):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(seed)
    params = Model6Parameters(); params.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, params)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    dp = net.synapses[0].dimer_particles

    o_step = dp.step
    native = []

    def w_step(*a, **k):
        if "collective_field_kT" in k:
            native.append(float(k["collective_field_kT"]))
            if bus is not None:
                k["collective_field_kT"] = bus
        return o_step(*a, **k)
    dp.step = w_step

    rel = PresynapticRelease(seed=seed)
    t0 = time.time()
    for _ in range(int(round(T_SIM / DT))):
        glu = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": glu})

    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]
    idset = set(ids)
    pos = {d.id: np.asarray(d.position, float) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in idset and b in idset]

    c, cyc = ordinary_topology(ids, edges)
    out = {"bus": bus, "native_bus": float(np.mean(native)) if native else None,
           "V": len(ids), "E": len(edges),
           "ordinary_components": c, "ordinary_cycle_rank": cyc,
           "elapsed_s": time.time() - t0}
    for mode in ("constant", "geometry", "random"):
        out[mode] = sheaf_cohomology(ids, pos, edges, mode)
    return out


def main():
    print("=" * 104)
    print("PO-5 UNIT 5 — genuine (non-constant) cellular sheaf on the dimer graph")
    print("  stalk R^6 = j_couplings_intra (dead data, dimer_particles.py:40)")
    print("  restriction = coordinate projection R^6 -> R  (real, non-identity, NO phase)")
    print("  P1 VALIDATION: constant sheaf MUST reproduce ordinary components and cycle rank")
    print("  P2 HYPOTHESIS: sheaf H0 > 1 where ordinary components == 1")
    print("=" * 104, flush=True)

    rows = []
    for bus in BUS_VALUES + [None]:
        r = run(bus); rows.append(r)
        lbl = "NATIVE" if bus is None else f"{bus:.2f}"
        print(f"\nbus={lbl}  V={r['V']} E={r['E']}  ordinary: components={r['ordinary_components']} "
              f"cycle_rank={r['ordinary_cycle_rank']}  ({r['elapsed_s']:.0f}s)")
        for mode in ("constant", "geometry", "random"):
            m = r[mode]
            print(f"    {mode:9s} H0={m['H0']:6d}  H1={m['H1']:8d}  "
                  f"H0_engaged={m['H0_engaged']:6d}  rank_delta={m['rank_delta']:6d}")
        with open(os.path.join(SWEEP_DIR, "po5_unit5_sheaf_results.json"), "w") as f:
            json.dump(rows, f, indent=2)

    # ---- P1 validation ----
    bad = [r for r in rows
           if r["constant"]["H0"] != r["ordinary_components"]
           or r["constant"]["H1"] != r["ordinary_cycle_rank"]]
    print("\n" + "=" * 104)
    if bad:
        print("P1 VALIDATION: FAIL — the constant sheaf does not reproduce the known case.")
        for r in bad:
            print(f"   bus={r['bus']}: constant H0={r['constant']['H0']} vs components="
                  f"{r['ordinary_components']}; H1={r['constant']['H1']} vs cycle_rank="
                  f"{r['ordinary_cycle_rank']}")
        print("VERDICT: INVALID — no cohomology claim is made.")
        return
    print("P1 VALIDATION: PASS — constant sheaf reproduces components AND cycle rank exactly,")
    print("               at every bus value. The implementation is admissible.")

    # ---- P2 / P3 ----
    blob = [r for r in rows if r["ordinary_components"] == 1]
    print("\nP2 — where the ORDINARY readout is saturated (components == 1):")
    if not blob:
        print("   no bus value reached a single component; P2 not evaluable here.")
    for r in blob:
        lbl = "NATIVE" if r["bus"] is None else f"{r['bus']:.2f}"
        g, c_, rd = r["geometry"], r["constant"], r["random"]
        print(f"   bus={lbl}: ordinary components=1  |  SCORED geometry H0_engaged="
              f"{g['H0_engaged']}  |  control random H0_engaged={rd['H0_engaged']}"
              f"   (raw H0 {g['H0']}/{rd['H0']} — inflated by free channels, NOT scored)")
    # A5.1: score H0_engaged, not raw H0 (unengaged channels are free by construction)
    informative = [r for r in blob if r["geometry"]["H0_engaged"] > 1]
    print()
    if informative:
        print("P2 VERDICT: SUPPORTED — sheaf H0_engaged is non-degenerate where component-counting is")
        print("            pinned at 1. The sheaf readout carries information the current one")
        print("            provably cannot at this operating point.")
    elif blob:
        print("P3 VERDICT: HYPOTHESIS FAILS — sheaf H0_engaged also collapses where components==1.")
        print("            The sheaf structure buys nothing at this operating point.")
    print("\nCONTROL — geometry vs random channels: if these agree, the structure comes from")
    print("having 6 coordinates, NOT from the physics, and P2 is not evidence for the claim.")
    print("\nLIMITS: single synapse, 1 s, one seed. The channel rule is a STATED MODELLING")
    print("CHOICE, not a derivation. No dynamics enforce this sheaf's consistency condition yet")
    print("— A2's falsifier requires that, and this unit does NOT claim to have met it.")


if __name__ == "__main__":
    main()
