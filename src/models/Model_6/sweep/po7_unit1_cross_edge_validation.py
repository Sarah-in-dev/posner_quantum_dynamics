#!/usr/bin/env python3
"""
PO-7 UNIT 1 — DATA-LEVEL VALIDATION of the inherited network-provenance layer.

This is NOT the keystone test. Its only job is the thing the handoff addendum says was
never done: SHOW whether _step_network_provenance actually produces cross-synapse edges,
and measure the event-pool OVERLAP fraction, before anything is built on top of it.

PREDICTION REGISTERED BEFORE RUNNING (from the geometry, not from a fit):
  Each synapse's grid is grid_shape=(100,100) x dx=4nm = 400nm across (dimer_particles.py:109),
  so a dimer/event sits within +-200nm of its synapse centre. Claim reach is 500nm
  (provenance_net_reach_nm, multi_synapse_network.py:124). Minimum possible cross-synapse
  dimer<->event distance for adjacent synapses is therefore (spacing_nm - 400).
    => cross edges are IMPOSSIBLE when spacing_nm - 400 > 500, i.e. spacing > 0.9 um.
    => the default spacing_um=2.0 gives a STRUCTURAL zero, not a physical null.
    => partial overlap (the SS8 regime) should live around spacing ~0.4-0.8 um.
  If cross edges appear above 0.9um, or never appear below it, my geometric read is wrong
  and the layer is broken -- either way this probe says so rather than guessing.

All six synapses are driven here on purpose: this unit asks "CAN the mechanism form a
cross-synapse edge at all", which is a question about geometry and reach, not about input.
The input contrast is Unit 2's job.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

T_SIM, DT = 0.4, 0.005
TRACKER_EVERY = 10                      # matches net.step()'s %10 entanglement cadence
SPACINGS_UM = [0.2, 0.4, 0.6, 0.8, 1.2, 2.0]
# Unit 1 is a GEOMETRY LOCATOR, not the scored test: 2 seeds suffice to find the
# overlap band. The >=5-seed rule binds the scored keystone test (Unit 2), not this.
SEEDS = [31337, 4242]
N_SYN = 6


def run_one(spacing_um: float, seed: int) -> dict:
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(seed)
    p = Model6Parameters()
    p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=spacing_um)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)

    tr = net.entanglement_tracker
    tr.provenance_network = True          # the opt-in under test

    rel = PresynapticRelease(seed=seed)
    peak = {'n_cross_bonds': 0, 'n_prov_bonds': 0, 'overlap_frac': 0.0, 'n_events': 0}
    n_steps = int(round(T_SIM / DT))
    for i in range(n_steps):
        g = rel.step(0.95, DT)
        for s in net.synapses:
            s.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        if (i + 1) % TRACKER_EVERY == 0:
            # coupling_weights MUST be passed or _update_entanglement forms no
            # cross-synapse eta-bonds and the graph is silently empty (the code
            # warns about exactly this). Provenance is eta-free by design, but the
            # comparison is only honest against a populated eta-path.
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))
            st = tr._prov_last_stats
            for k in peak:
                peak[k] = max(peak[k], st.get(k, 0))

    # Final-state cross/intra split, recomputed independently of the diagnostics field
    # so the probe does not simply trust _prov_last_stats.
    cross = sum(1 for (a, b) in tr._prov_bonds if a[0] != b[0])
    return {
        'spacing_um': spacing_um, 'seed': seed,
        'final_prov_bonds': len(tr._prov_bonds),
        'final_cross_bonds': cross,
        'final_intra_bonds': len(tr._prov_bonds) - cross,
        'peak_cross_bonds': peak['n_cross_bonds'],
        'peak_prov_bonds': peak['n_prov_bonds'],
        'peak_overlap_frac': peak['overlap_frac'],
        'peak_events': peak['n_events'],
    }


def main():
    rows = []
    for sp in SPACINGS_UM:
        for sd in SEEDS:
            r = run_one(sp, sd)
            rows.append(r)
            print(f"spacing={sp:>4} seed={sd:>6}  cross={r['final_cross_bonds']:>4} "
                  f"intra={r['final_intra_bonds']:>5} peak_cross={r['peak_cross_bonds']:>4} "
                  f"overlap={r['peak_overlap_frac']:.3f} events={r['peak_events']}")
            sys.stdout.flush()

    print("\n=== PER-SPACING SUMMARY (mean over seeds) ===")
    summary = []
    for sp in SPACINGS_UM:
        sub = [r for r in rows if r['spacing_um'] == sp]
        s = {
            'spacing_um': sp,
            'mean_cross': float(np.mean([r['final_cross_bonds'] for r in sub])),
            'mean_intra': float(np.mean([r['final_intra_bonds'] for r in sub])),
            'mean_peak_cross': float(np.mean([r['peak_cross_bonds'] for r in sub])),
            'mean_overlap': float(np.mean([r['peak_overlap_frac'] for r in sub])),
            'seeds_with_any_cross': int(sum(1 for r in sub if r['peak_cross_bonds'] > 0)),
        }
        summary.append(s)
        print(f"  spacing={sp:>4}um  mean_cross={s['mean_cross']:>7.1f}  "
              f"mean_peak_cross={s['mean_peak_cross']:>7.1f}  "
              f"mean_overlap={s['mean_overlap']:.3f}  "
              f"seeds_with_cross={s['seeds_with_any_cross']}/{len(SEEDS)}")

    # The registered prediction, checked mechanically.
    above = [s for s in summary if s['spacing_um'] > 0.9]
    below = [s for s in summary if s['spacing_um'] <= 0.8]
    pred_above_zero = all(s['mean_peak_cross'] == 0 for s in above)
    pred_below_any = any(s['mean_peak_cross'] > 0 for s in below)
    print("\n=== REGISTERED PREDICTION CHECK ===")
    print(f"  zero cross edges at spacing > 0.9um : {pred_above_zero}")
    print(f"  some cross edges at spacing <= 0.8um: {pred_below_any}")
    verdict = ("GEOMETRY CONFIRMED - layer forms cross-synapse edges in the predicted band"
               if (pred_above_zero and pred_below_any)
               else "PREDICTION FAILED - re-read the layer before building on it")
    print(f"  VERDICT: {verdict}")

    out = {'rows': rows, 'summary': summary,
           'prediction_above_0p9um_zero': bool(pred_above_zero),
           'prediction_below_0p8um_nonzero': bool(pred_below_any),
           'verdict': verdict,
           'config': {'n_synapses': N_SYN, 'pattern': 'linear', 't_sim_s': T_SIM,
                      'dt_s': DT, 'tracker_every': TRACKER_EVERY, 'seeds': SEEDS,
                      'reach_nm': 500.0, 'grid_span_nm': 400.0}}
    path = os.path.join(SWEEP_DIR, 'po7_unit1_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
