#!/usr/bin/env python3
"""
PO-7 UNIT 5 — measure the COINCIDENCE WINDOW before designing the coincidence test.
Diagnostic only: no verdict function, nothing scored, nothing tuned.

WHY THIS RUNS FIRST
-------------------
The synchrony test (advisor R4 step 2) contrasts synchronous vs staggered drive at matched
density. Its one free design parameter is the STAGGER OFFSET, and the right value depends on
a number nobody has measured: how long a hydrolysis event actually survives before its two
daughter slots are consumed.

Two regimes, with very different experiments:
  (A) events survive near the full age window (provenance_net_age_s = 2.0 s) before being
      claimed => a stagger must exceed ~2 s to prevent sharing => 6 synapses x >2 s per
      condition => ~18 h of compute. Infeasible here.
  (B) events are consumed within the burst (tens of ms) => a stagger of a few hundred ms
      already separates the conditions => the experiment is cheap and can use all 6 synapses.

Guessing this parameter would put the whole synchrony result on an unmeasured assumption, so
it is measured. What is reported:
  - time from event CREATION to FIRST claim
  - time from creation to SECOND claim (the cross-relevant one — the second claim is what
    creates a bond)
  - fraction of events that expire UNCLAIMED, claimed once, or fully consumed
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
TRACKER_EVERY = 10
SPACING, SEED, N_SYN = 0.2, 31337, 6


def main():
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    tr = net.entanglement_tracker
    tr.provenance_network = True

    # event id -> {'t': creation, 'first': t of 1st claim, 'second': t of 2nd claim, 'seen_last': ...}
    life = {}
    rel = PresynapticRelease(seed=SEED)
    for i in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        for s in net.synapses:
            s.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        if (i + 1) % TRACKER_EVERY == 0:
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))
            now = tr._prov_time
            for e in tr._prov_events:
                r = life.setdefault(e['id'], {'t': e['t'], 'first': None, 'second': None,
                                              'n_holders': 0})
                nh = len(e['holders'])
                if nh >= 1 and r['first'] is None:
                    r['first'] = now
                if nh >= 2 and r['second'] is None:
                    r['second'] = now
                r['n_holders'] = max(r['n_holders'], nh)

    n = len(life)
    unclaimed = sum(1 for r in life.values() if r['n_holders'] == 0)
    once = sum(1 for r in life.values() if r['n_holders'] == 1)
    full = sum(1 for r in life.values() if r['n_holders'] >= 2)
    d1 = np.array([r['first'] - r['t'] for r in life.values() if r['first'] is not None])
    d2 = np.array([r['second'] - r['t'] for r in life.values() if r['second'] is not None])

    print("=" * 88)
    print("EVENT LIFECYCLE — the coincidence window")
    print("=" * 88)
    print(f"  events observed: {n}   (age window = {tr.provenance_net_age_s} s, "
          f"slots = {tr.provenance_net_event_slots})")
    print(f"  never claimed : {unclaimed:>5}  ({100.0*unclaimed/n:.1f}%)")
    print(f"  claimed once  : {once:>5}  ({100.0*once/n:.1f}%)")
    print(f"  fully consumed: {full:>5}  ({100.0*full/n:.1f}%)  <- these are the bond-makers")
    if len(d1):
        print(f"\n  creation -> FIRST claim (s):  median={np.median(d1):.4f} "
              f"mean={d1.mean():.4f} p90={np.percentile(d1,90):.4f} max={d1.max():.4f}")
    if len(d2):
        print(f"  creation -> SECOND claim (s): median={np.median(d2):.4f} "
              f"mean={d2.mean():.4f} p90={np.percentile(d2,90):.4f} max={d2.max():.4f}")
        w = float(np.percentile(d2, 90))
        print(f"\n  => COINCIDENCE WINDOW (p90 of creation->second claim): {w:.4f} s")
        print(f"  => a stagger offset of ~{max(4*w, 0.05):.3f} s should separate the conditions")
        print(f"     (vs {tr.provenance_net_age_s} s if events survived the full age window)")
        regime = "B (cheap): events consumed fast" if w < 0.5 else "A (expensive): near full age"
        print(f"  => REGIME {regime}")
    else:
        print("\n  no event reached a second claim in this run — window not measurable here.")

    out = {'n_events': n, 'unclaimed': unclaimed, 'claimed_once': once, 'fully_consumed': full,
           'age_window_s': tr.provenance_net_age_s,
           'first_claim_s': {'median': float(np.median(d1)) if len(d1) else None,
                             'p90': float(np.percentile(d1, 90)) if len(d1) else None,
                             'max': float(d1.max()) if len(d1) else None},
           'second_claim_s': {'median': float(np.median(d2)) if len(d2) else None,
                              'p90': float(np.percentile(d2, 90)) if len(d2) else None,
                              'max': float(d2.max()) if len(d2) else None}}
    path = os.path.join(SWEEP_DIR, 'po7_unit5_event_lifetime_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
