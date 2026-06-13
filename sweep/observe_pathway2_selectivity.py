#!/usr/bin/env python3
"""
Observe Pathway-2 spatial selectivity after adding 1/r^3 coupling.

Single synapse, sustained depolarisation, 30 s.
Logs intra-bond graph metrics every 5 s:
  - bonded-pair separations vs all-pair separations
  - bond count, saturation, connected components
"""

import sys, os
import logging
import numpy as np
from numpy.linalg import norm

# Suppress model logging
logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------
def graph_metrics(dp):
    """
    Compute bond-graph metrics over currently-entangled dimers.

    Returns dict with: n_entangled, n_bonds, saturation,
      bonded_r_{med,p90,max}, all_r_{med,p90,max},
      n_comp, largest_frac
    """
    entangled = [d for d in dp.dimers if d.is_entangled]
    pos_by_id = {d.id: d.position for d in entangled}
    ids = list(pos_by_id)
    n = len(ids)

    # All-pair separations (baseline)
    all_r = []
    for ai in range(n):
        for bi in range(ai + 1, n):
            all_r.append(float(norm(pos_by_id[ids[ai]] - pos_by_id[ids[bi]])))

    # Bonded-pair separations
    bonded = [(b.dimer_i, b.dimer_j) for b in dp.entanglement_bonds
              if b.dimer_i in pos_by_id and b.dimer_j in pos_by_id]
    bonded_r = [float(norm(pos_by_id[i] - pos_by_id[j])) for (i, j) in bonded]
    n_bonds = len(bonded)

    n_pairs = n * (n - 1) // 2
    saturation = n_bonds / n_pairs if n_pairs > 0 else 0.0

    # Connected components via union-find
    parent = {did: did for did in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j) in bonded:
        union(i, j)

    if n > 0:
        from collections import Counter
        comp_sizes = Counter(find(did) for did in ids)
        n_comp = len(comp_sizes)
        largest_frac = max(comp_sizes.values()) / n
    else:
        n_comp = 0
        largest_frac = 0.0

    def stats(arr):
        if not arr:
            return 'na', 'na', 'na'
        a = np.array(arr)
        return f"{np.median(a):.1f}", f"{np.percentile(a, 90):.1f}", f"{np.max(a):.1f}"

    bm, bp, bx = stats(bonded_r)
    am, ap, ax = stats(all_r)

    return {
        'n_entangled': n, 'n_bonds': n_bonds, 'saturation': saturation,
        'bonded_med': bm, 'bonded_p90': bp, 'bonded_max': bx,
        'all_med': am, 'all_p90': ap, 'all_max': ax,
        'n_comp': n_comp, 'largest_frac': largest_frac,
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
params = Model6Parameters()
params.em_coupling_enabled = True

N_SYN = 1
T_total = 30.0
dt = 0.005
log_interval = 5.0

network = MultiSynapseNetwork(
    n_synapses=N_SYN, pattern="clustered", spacing_um=1.0,
)
network.initialize(Model6QuantumSynapse, params)
for s in network.synapses:
    s.set_microtubule_invasion(True)

stimulus = {'voltage': -10e-3, 'reward': False}

# ---------------------------------------------------------------------------
# Drive loop
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("PATHWAY-2 SELECTIVITY OBSERVATION: 1/r^3 spatial coupling")
print("=" * 100)
print(f"  N_synapses={N_SYN}  dt={dt}  T={T_total}s  voltage={stimulus['voltage']*1e3:.0f}mV")
print(f"  coupling_length={network.synapses[0].dimer_particles.coupling_length} nm")
print()

header = (f"{'t':>5s}  {'n_ent':>5s}  {'bonds':>5s}  {'sat':>6s}  "
          f"{'b_med':>6s} {'b_p90':>6s} {'b_max':>6s}  "
          f"{'a_med':>6s} {'a_p90':>6s} {'a_max':>6s}  "
          f"{'comps':>5s}  {'lg_fr':>5s}")
print(header)
print("-" * len(header))

steps = int(round(T_total / dt))
next_log = 0.0
t = 0.0

for i in range(steps):
    try:
        network.step(dt, stimulus)
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__} at t={t:.2f}s: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    t += dt

    if t >= next_log - dt / 2:
        dp = network.synapses[0].dimer_particles
        m = graph_metrics(dp)
        print(f"{t:5.1f}  {m['n_entangled']:5d}  {m['n_bonds']:5d}  {m['saturation']:6.4f}  "
              f"{m['bonded_med']:>6s} {m['bonded_p90']:>6s} {m['bonded_max']:>6s}  "
              f"{m['all_med']:>6s} {m['all_p90']:>6s} {m['all_max']:>6s}  "
              f"{m['n_comp']:5d}  {m['largest_frac']:5.3f}")
        next_log += log_interval

print()
print("Done.")
