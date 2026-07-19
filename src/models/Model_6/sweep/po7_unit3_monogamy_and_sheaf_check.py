#!/usr/bin/env python3
"""
PO-7 UNIT 3 — verify two advisor-R4 claims against the code and the data.
Read-only analysis: no physics change, no verdict function, nothing tuned.

CLAIM A — "your graphs violate monogamy."
  A Ca6(PO4)4 dimer carries four 31P spin-1/2 nuclei. A bond consumes one spin at each
  endpoint, so a dimer's singlet-strength degree cannot exceed 4. Measure the actual degree
  distribution of the entanglement graph at the standing fingerprint operating point
  (1 synapse, 200 steps, seed 31337, MT invaded) and report the violation magnitude.

CLAIM B — "your H0_engaged is six graph Laplacians in a trench coat."
  Unit 5's restriction maps are coordinate projections R^6 -> R, which are diagonal, so the
  sheaf should decompose as a direct sum and H0 should equal the sum of per-block component
  counts. VERIFY THIS RATHER THAN ACCEPT IT -- and check the block structure, because the
  channel rule pairs each axis with its opposite sign:
      ka = channel_from_direction(+d) = 2*ax + (0 if d[ax]>=0 else 1)
      kb = channel_from_direction(-d) = 2*ax + (1 if d[ax]>=0 else 0)
  so every edge joins channel 2m to channel 2m+1 for the same dominant axis m. The channels
  therefore do NOT stay separate in 6 blocks; they fuse in PAIRS, giving 3 blocks
  ({0,1}, {2,3}, {4,5}), not 6. The decomposition claim stands; the block count differs.
  This probe checks the decomposition identity numerically.
"""
import sys, os, json, logging
from collections import Counter
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

T_SIM, DT, SEED = 1.0, 0.005, 31337
N_CHANNELS = 6
SPINS_PER_DIMER = 4          # four 31P nuclei in Ca6(PO4)4  [quantum-system-canonical:43]


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb
    def n_components(self, nodes): return len({self.find(n) for n in nodes})


def channel_from_direction(d):
    ax = int(np.argmax(np.abs(d)))
    return ax * 2 + (0 if d[ax] >= 0 else 1)


def main():
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    dp = net.synapses[0].dimer_particles
    rel = PresynapticRelease(seed=SEED)
    for _ in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})

    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]
    idset = set(ids)
    pos = {d.id: np.asarray(d.position, float) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in idset and b in idset]
    V, E = len(ids), len(edges)

    # ---------------- CLAIM A: monogamy ----------------
    deg = Counter()
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    degs = np.array([deg.get(i, 0) for i in ids], float)
    over = int((degs > SPINS_PER_DIMER).sum())
    excess_edges = float(np.clip(degs - SPINS_PER_DIMER, 0, None).sum() / 2.0)

    print("=" * 92)
    print("CLAIM A — MONOGAMY BOUND")
    print("=" * 92)
    print(f"  V (entangled dimers) = {V}    E (edges) = {E}")
    print(f"  monogamy bound: <= {SPINS_PER_DIMER} singlet-strength bonds per dimer "
          f"(4 x 31P spins per Ca6(PO4)4)")
    print(f"  mean degree   = {degs.mean():.1f}   ({degs.mean()/SPINS_PER_DIMER:.0f}x the bound)")
    print(f"  median degree = {np.median(degs):.1f}")
    print(f"  max degree    = {degs.max():.0f}   ({degs.max()/SPINS_PER_DIMER:.0f}x the bound)")
    print(f"  dimers over bound: {over}/{V}  ({100.0*over/V:.1f}%)")
    print(f"  edges in excess of the bound: {excess_edges:.0f} / {E}  "
          f"({100.0*excess_edges/E:.2f}% of the graph is physically inadmissible)")
    print(f"  max admissible E under monogamy = {V*SPINS_PER_DIMER//2} "
          f"(vs actual {E}, i.e. {E/(V*SPINS_PER_DIMER/2):.0f}x over)")

    # ---------------- CLAIM B: does the Unit-5 sheaf decompose? ----------------
    uf_full = UF()
    nodes = {(v, k) for v in ids for k in range(N_CHANNELS)}
    for n in nodes: uf_full.find(n)
    engaged, block_edges = set(), {0: [], 1: [], 2: []}
    cross_block = 0
    for (a, b) in edges:
        d = pos[b] - pos[a]
        nrm = np.linalg.norm(d)
        if nrm == 0:
            ka = kb = 0
        else:
            d = d / nrm
            ka = channel_from_direction(d); kb = channel_from_direction(-d)
        uf_full.union((a, ka), (b, kb))
        engaged.add((a, ka)); engaged.add((b, kb))
        if ka // 2 != kb // 2:
            cross_block += 1
        block_edges[ka // 2].append(((a, ka), (b, kb)))
    h0_engaged_full = uf_full.n_components(engaged)

    # Recompute per axis-pair block independently and sum.
    total_block_components, per_block = 0, {}
    for m, elist in block_edges.items():
        ufb = UF()
        eng_b = set()
        for (na, nb) in elist:
            ufb.union(na, nb); eng_b.add(na); eng_b.add(nb)
        c = ufb.n_components(eng_b) if eng_b else 0
        per_block[m] = {'n_edges': len(elist), 'n_engaged_coords': len(eng_b), 'components': c}
        total_block_components += c

    print("\n" + "=" * 92)
    print("CLAIM B — DOES THE UNIT-5 SHEAF DECOMPOSE?")
    print("=" * 92)
    print(f"  edges joining DIFFERENT axis-pair blocks: {cross_block} / {E}")
    print(f"  (channel rule pairs +axis with -axis, so edges join channel 2m <-> 2m+1;")
    print(f"   channels fuse in PAIRS => 3 blocks {{0,1}},{{2,3}},{{4,5}}, not 6 independent ones)")
    for m, r in per_block.items():
        print(f"    block {{{2*m},{2*m+1}}} (axis {m}): edges={r['n_edges']:>7} "
              f"engaged_coords={r['n_engaged_coords']:>5} components={r['components']}")
    print(f"  sum of per-block components = {total_block_components}")
    print(f"  H0_engaged computed on the full channel graph = {h0_engaged_full}")
    ok = (total_block_components == h0_engaged_full)
    print(f"  DECOMPOSITION IDENTITY HOLDS: {ok}")
    print("  => " + ("CONFIRMED: H0_engaged is a SUM of independent graph-component counts — "
                     "a direct sum of graph Laplacians, not irreducible sheaf structure."
                     if ok else
                     "NOT confirmed — the sheaf does not decompose this way; re-examine."))

    out = {'V': V, 'E': E, 'monogamy': {
                'bound_per_dimer': SPINS_PER_DIMER, 'mean_degree': float(degs.mean()),
                'median_degree': float(np.median(degs)), 'max_degree': float(degs.max()),
                'n_over_bound': over, 'frac_over_bound': float(over / V),
                'excess_edges': excess_edges,
                'max_admissible_E': int(V * SPINS_PER_DIMER // 2)},
           'sheaf': {'cross_block_edges': cross_block, 'per_block': per_block,
                     'sum_block_components': total_block_components,
                     'H0_engaged_full': h0_engaged_full,
                     'decomposition_holds': bool(ok)}}
    path = os.path.join(SWEEP_DIR, 'po7_unit3_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
