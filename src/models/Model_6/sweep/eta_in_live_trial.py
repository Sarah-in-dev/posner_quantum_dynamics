#!/usr/bin/env python3
"""
DOES eta CLEAR THRESHOLD IN A LIVE TRIAL? — the one precondition left before the §8
input-selectivity phase.

WHY THIS EXISTS
---------------
L·ETA-2 measured the pump igniting after the input engine was finished: r 0.3509 ->
1.6234, eta 0.0000 -> 0.2376. But that was a CHARACTERIZATION RIG — 7 synapses at 1 um,
all held at act=1.0 for 20 s. The live spatial-discovery experiment is nothing like it:
Gaussian feature activations mostly below 1.0, features scattered across a 10x10 arena,
and only the handful within the agent's sensory horizon active at any moment.

eta is aggregated: p_met_agg = P_BASAL + coupling_weights @ p_active. A synapse's
condensation depends on its NEIGHBOURS' drive as well as its own, so the live question is
not "can eta ignite" (answered: yes) but "does the drive an actual trial delivers, at the
actual feature geometry, aggregate to r >= 1 anywhere".

If it does not, the §8 selectivity phase is still blocked — not by the pump's capability
but by the experiment's drive regime, and the fix is a protocol question, not a physics
one.

WHAT IS MEASURED (per agent step, in a real trial)
-------------------------------------------------
  r and eta PER SYNAPSE (recomputed exactly as _update_backbone_field does)
  n_active            synapses above the 0.05 activation floor
  n_condensed         synapses with eta > 0  <-- THE ANSWER
  max_r               how close the closest synapse got
  cross bonds / betti0 of the cross-synapse partition, so we see whether eta>0 actually
                      produces topology rather than just clearing a threshold

HONEST BY CONSTRUCTION
----------------------
Reports max_r even when nothing condenses, so "it did not ignite" comes with how far
short it fell. Nothing here is tuned; it only reads.
"""
import sys, os, json
import numpy as np
import logging

logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(MODEL6_DIR, '..', '..', '..'))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(REPO, 'sweep'))

from model6_parameters import (P_BASAL_W, compute_metabolic_power,
                               bose_einstein_occupation)
from run_spatial_discovery import (make_network, activations_to_stimuli,
                                   step_network_per_synapse)
from spatial_environment import SpatialEnvironment, Agent

hbar = 1.0545718e-34

# Small but REAL: same generator as the shipped experiment, fewer features so a trial
# completes in minutes. Feature count changes the coupling row-sums, so it is reported.
N_FEATURES = 12
TRIAL_BUDGET_S = 40.0
AGENT_DT = 0.5
PHYSICS_DT = 0.005
SEED = 7


def r_eta_per_synapse(network):
    """Recompute r and eta exactly as _update_backbone_field does (read-only)."""
    bp = network.params.dendritic_backbone
    omega_ang = 2.0 * np.pi * bp.omega_0
    n_bar = bose_einstein_occupation(bp.omega_0)
    P_c = n_bar * hbar * omega_ang ** 2 / bp.Q
    p_met = np.array([
        compute_metabolic_power(getattr(s.spine_plasticity, 'E_invasion', 0.0),
                                s.calcium.channels.get_open_fraction(),
                                bp.p_active_max_W) for s in network.synapses])
    p_active = p_met - P_BASAL_W
    p_met_agg = P_BASAL_W + network.coupling_weights @ p_active
    rs = p_met_agg / P_c
    etas = np.array([(r - 1.0) / (r + 1.0) if r >= 1.0 else 0.0 for r in rs])
    # r ∝ E_invasion · ca_open. Report BOTH factors so a shortfall is attributable
    # to one of them rather than guessed at.
    e_inv = np.array([getattr(s.spine_plasticity, 'E_invasion', 0.0)
                      for s in network.synapses])
    ca_op = np.array([s.calcium.channels.get_open_fraction()
                      for s in network.synapses])
    return rs, etas, P_c, e_inv, ca_op


def cross_topology(network):
    """Cross-synapse edge count and component count, Werner-thresholded."""
    tr = network.entanglement_tracker
    bound = tr.WERNER_ENTANGLEMENT_BOUND
    n_cross_above = sum(1 for f in tr.cross_synapse_bonds.values() if f > bound)
    try:
        comps = tr._find_all_clusters()
        syn_comps = {frozenset(gid[0] for gid in c) for c in comps}
        betti0 = len(syn_comps)
    except Exception:
        betti0 = -1
    return n_cross_above, betti0


def main():
    print("=" * 100)
    print("eta IN A LIVE TRIAL — does the real drive regime reach the condensation threshold?")
    print("=" * 100)

    np.random.seed(SEED)
    env = SpatialEnvironment(n_features=N_FEATURES, seed=SEED)
    network = make_network(n_synapses=N_FEATURES, seed=SEED)
    agent = Agent(rng=np.random.default_rng(SEED))
    agent.reset(env.size, np.random.default_rng(SEED))

    rowsum = network.coupling_weights.sum(axis=1)
    print(f"  {N_FEATURES} features/synapses, budget {TRIAL_BUDGET_S}s, "
          f"agent_dt={AGENT_DT}, physics_dt={PHYSICS_DT}, seed={SEED}")
    print(f"  coupling row-sums: min {rowsum.min():.2f}  median "
          f"{np.median(rowsum):.2f}  max {rowsum.max():.2f}")
    print()
    print(f"{'t(s)':>6} {'n_act':>6} {'max_act':>8} {'max_Einv':>9} {'max_caop':>9} "
          f"{'max_r':>8} {'n_cond':>7} {'x-edges':>8}")
    print("-" * 100)

    steps = int(TRIAL_BUDGET_S / AGENT_DT)
    phys_per = int(AGENT_DT / PHYSICS_DT)
    best_r, best_eta, ever_condensed, rows = 0.0, 0.0, 0, []

    for k in range(steps):
        acts = env.get_activations(agent.position)
        stimuli = activations_to_stimuli(acts)
        for i in range(len(network.synapses)):
            g = network.presynaptic_release[i].step(acts[i], PHYSICS_DT)
            if g:
                stimuli[i]['glutamate'] = g

        for _ in range(phys_per):
            step_network_per_synapse(network, PHYSICS_DT, stimuli)

        rs, etas, P_c, e_inv, ca_op = r_eta_per_synapse(network)
        n_cond = int(np.sum(etas > 0))
        ever_condensed = max(ever_condensed, n_cond)
        best_r = max(best_r, float(rs.max()))
        best_eta = max(best_eta, float(etas.max()))
        xe, b0 = cross_topology(network)

        t = (k + 1) * AGENT_DT
        rows.append(dict(t=t, n_active=int(np.sum(acts > 0.05)),
                         max_act=float(acts.max()), max_r=float(rs.max()),
                         n_condensed=n_cond, max_eta=float(etas.max()),
                         x_edges=xe, betti0=b0,
                         max_E_invasion=float(e_inv.max()),
                         max_ca_open=float(ca_op.max())))
        if k % 4 == 0 or n_cond > 0:
            print(f"{t:6.1f} {int(np.sum(acts > 0.05)):6d} {acts.max():8.3f} "
                  f"{e_inv.max():9.4f} {ca_op.max():9.4f} "
                  f"{rs.max():8.3f} {n_cond:7d} {xe:8d}")

        agent.step(AGENT_DT, env, np.zeros(N_FEATURES))

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  max r reached in the trial : {best_r:.4f}   (threshold 1.0)")
    print(f"  max eta reached            : {best_eta:.4f}")
    print(f"  most synapses condensed at once : {ever_condensed}")
    print()
    print("  ATTRIBUTION — r is proportional to E_invasion x ca_open:")
    print(f"    max E_invasion over the trial : {max(r['max_E_invasion'] for r in rows):.4f}"
          f"   (rig reached 0.35)")
    print(f"    max ca_open over the trial    : {max(r['max_ca_open'] for r in rows):.4f}"
          f"   (rig reached ~0.38 NMDAR)")
    print()
    if ever_condensed > 0:
        print("  => eta CLEARS THRESHOLD in the live drive regime. The §8 selectivity")
        print("     precondition is MET: drive patterns the partition in a real trial.")
    else:
        print(f"  => eta DOES NOT CLEAR in the live regime (max r {best_r:.3f} vs 1.0).")
        print("     The pump is capable (L·ETA-2) but this experiment's drive does not")
        print("     aggregate to threshold. That is a PROTOCOL question, not a physics")
        print("     one: feature density / co-activation / spacing set the row-sums.")

    out = os.path.join(MODEL6_DIR, 'results', 'eta_live_trial')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'eta_live_seed{SEED}.json'), 'w') as fh:
        json.dump(dict(n_features=N_FEATURES, seed=SEED, budget_s=TRIAL_BUDGET_S,
                       rowsum_min=float(rowsum.min()),
                       rowsum_max=float(rowsum.max()),
                       max_r=best_r, max_eta=best_eta,
                       max_condensed=ever_condensed, rows=rows), fh, indent=1)
    print(f"\n  trace -> {out}/eta_live_seed{SEED}.json")


if __name__ == "__main__":
    main()
