#!/usr/bin/env python3
"""
Q4-5 — does `coupling_weights` actually REACH `_update_entanglement`?  PO-4.
============================================================================
MO acceptance: "coupling_weights reaches _update_entanglement in every driver
call site -- DEMONSTRATED BY MEASUREMENT, NOT BY GREP -- plus an explicit
statement of whether bonds now form and, if not, which blocker is responsible."

WHY A GREP IS NOT ENOUGH HERE. Every driver call site passes
`coupling_weights=getattr(network, 'coupling_weights', None)`. That READS as a
fix and silently degrades to None if the attribute is ever absent -- and
`_update_entanglement` then returns before forming anything, with no warning.
So the grep and the behaviour can disagree, which is the whole reason this
defect survived a fix that "covered" the file. This probe instruments the callee
and records what actually arrived.

THE COMPOSITION NOTE (MO, and it changes the reading). Three INDEPENDENT reasons
the live topology can be empty:
  (1) coupling_weights missing      -> early return, no formation      [this unit]
  (2) active_mask: only ACTIVE synapses are stepped (D19)              [not this unit]
  (3) eta = 0 in live trials (L·ETA-1/3) -> k_cross ~ sqrt(eta_i*eta_j) = 0
Fixing (1) is necessary and CANNOT BY ITSELF PRODUCE BONDS. A measured zero with
an identified cause is a pass, and this probe is built to say which cause.
"""
import os
import sys
import json
import logging

import numpy as np

logging.disable(logging.INFO)
HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(M6)))
sys.path.insert(0, M6)
sys.path.insert(0, os.path.join(ROOT, 'sweep'))
for _n in ['model6_core', 'multi_synapse_network', 'dimer_particles',
           'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
           'quantum_coherence', 'pH_dynamics', 'dopamine_system', 'em_tryptophan_module',
           'em_coupling_module', 'local_dimer_tubulin_coupling', 'camkii_module',
           'spine_plasticity_module', 'photon_emission_module', 'photon_receiver_module',
           'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(_n).setLevel(logging.ERROR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork, NetworkEntanglementTracker

sys.path.insert(0, os.path.join(ROOT, 'sweep'))
import run_spatial_discovery as RSD
from spatial_environment import SpatialEnvironment, Agent
from presynaptic_release import PresynapticRelease

CALLS = []          # one record per _update_entanglement invocation
_real = NetworkEntanglementTracker._update_entanglement


def instrumented(self, dt, coupling_weights=None):
    """Record what ACTUALLY arrived, then run the real thing unchanged."""
    CALLS.append(dict(
        dt=float(dt),
        got_weights=coupling_weights is not None,
        shape=(None if coupling_weights is None else list(np.shape(coupling_weights))),
        n_dimers=len(self.all_dimers),
        eta=[float(getattr(s, '_backbone_eta', 0.0)) for s in getattr(self, '_syns', [])] or None,
        cross_before=len(self.cross_synapse_bonds),
    ))
    out = _real(self, dt, coupling_weights)
    CALLS[-1]['cross_after'] = len(self.cross_synapse_bonds)
    return out


STEPS = []          # one record per tracker.step() invocation
_real_step = NetworkEntanglementTracker.step


def instrumented_step(self, dt, synapses, positions, coupling_weights=None):
    """Record the OUTER call too. tracker.step early-returns at n_dimers < 2
    (multi_synapse_network.py:194-204), UPSTREAM of the coupling_weights guard --
    so 'the guard was never reached' and 'the guard rejected us' are different
    failures and must not be conflated."""
    rec = dict(dt=float(dt), got_weights=coupling_weights is not None)
    n_before = len(CALLS)
    out = _real_step(self, dt, synapses, positions, coupling_weights)
    rec['reached_update_entanglement'] = len(CALLS) > n_before
    rec['n_dimers'] = out.get('n_total_dimers', -1) if isinstance(out, dict) else -1
    STEPS.append(rec)
    return out


NetworkEntanglementTracker.step = instrumented_step
NetworkEntanglementTracker._update_entanglement = instrumented


def make_network(n=3):
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=n, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    # run_trial reads network.presynaptic_release (run_spatial_discovery.py:202);
    # the driver's own make_network wires it at :160. Same wiring here.
    net.presynaptic_release = [PresynapticRelease(seed=sq)
                               for sq in np.random.SeedSequence(5).spawn(n)]
    return net


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    print("=" * 78)
    print("Q4-5 — does coupling_weights REACH _update_entanglement?   [PO-4]")
    print("instrumenting the callee; every driver call site exercised for real")
    print("=" * 78)

    results = {}

    # ---- SITE 1: step_network_per_synapse (the reward-step path) -----------
    section("SITE 1 — step_network_per_synapse  (run_spatial_discovery.py:96)")
    CALLS.clear(); STEPS.clear()
    np.random.seed(7)
    net = make_network(3)
    stim = [{"voltage": -10e-3, "reward": False} for _ in net.synapses]
    # The tracker runs only every 10th step, AND tracker.step() early-returns at
    # n_dimers < 2 (multi_synapse_network.py:194-204) -- a FOURTH gate, upstream of
    # the coupling_weights guard. Drive long enough to clear both.
    for _ in range(60):
        RSD.step_network_per_synapse(net, 0.005, stim)
    site1 = list(CALLS)
    for c in site1:
        print(f"  call: dt={c['dt']:<7.4f} coupling_weights={'ARRIVED' if c['got_weights'] else 'None (EARLY RETURN)'}"
              f"  shape={c['shape']}  dimers={c['n_dimers']}  cross {c['cross_before']}->{c['cross_after']}")
    ok1 = bool(site1) and all(c['got_weights'] for c in site1)
    print(f"  -> {'PASS' if ok1 else 'FAIL'} ({len(site1)} call(s))")
    results['site1_step_network_per_synapse'] = dict(calls=len(site1), all_arrived=ok1)

    # ---- SITE 2: run_trial's every-10th-step tracker call ------------------
    section("SITE 2 — run_trial every-10th-step  (run_spatial_discovery.py:218)")
    CALLS.clear(); STEPS.clear()
    np.random.seed(11)
    net2 = make_network(3)
    env = SpatialEnvironment(size=10.0, n_features=3, n_clusters=1, seed=3)
    agent = Agent()
    agent.reset(env.size, np.random.default_rng(3))
    # CONTROLLED INITIAL CONDITION, stated: park the agent ON a feature. With a
    # random start the agent sits far from every feature, active_mask makes the
    # active set empty, no dimers form, and tracker.step early-returns at n<2 --
    # so the coupling_weights guard is never reached and the measurement is vacuous.
    # That is D19's blocker (2) and it is reported separately below.
    agent.position = env.feature_positions[0].copy()
    # Smallest budget that still reaches the every-10th-step tracker call.
    RSD.run_trial(net2, env, agent, trial_num=0, agent_dt=0.5,
                  trial_time_budget=2.0, physics_dt=0.005)
    site2 = list(CALLS)
    steps2 = list(STEPS)
    arrived = sum(c['got_weights'] for c in site2)
    reached = sum(st['reached_update_entanglement'] for st in steps2)
    print(f"  tracker.step() invocations: {len(steps2)}; "
          f"with coupling_weights: {sum(st['got_weights'] for st in steps2)}; "
          f"that REACHED _update_entanglement: {reached}")
    if steps2 and reached < len(steps2):
        print(f"    ^ {len(steps2)-reached} early-returned at n_dimers < 2 -- UPSTREAM of the")
        print(f"      guard. Not a coupling_weights failure; that is blocker (2)/(0).")
    print(f"  {len(site2)} tracker call(s) during the trial; "
          f"{arrived} arrived with coupling_weights, {len(site2)-arrived} as None")
    for c in site2[:4]:
        print(f"    dt={c['dt']:<7.4f} {'ARRIVED' if c['got_weights'] else 'None (EARLY RETURN)'}"
              f"  shape={c['shape']}  dimers={c['n_dimers']}  cross {c['cross_before']}->{c['cross_after']}")
    ok2 = bool(site2) and arrived == len(site2)
    print(f"  -> {'PASS' if ok2 else 'FAIL'}")
    results['site2_run_trial'] = dict(calls=len(site2), all_arrived=ok2)

    # ---- DID BONDS FORM, AND IF NOT, WHICH BLOCKER? -----------------------
    section("BONDS: did they form, and if not, which of the three blockers?")
    allc = site1 + site2
    formed = sum(c['cross_after'] - c['cross_before'] for c in allc)
    etas = [float(getattr(s, '_backbone_eta', 0.0)) for s in net2.synapses]
    total_dimers = max((c['n_dimers'] for c in allc), default=0)
    print(f"  cross-synapse bonds formed across all instrumented calls: {formed}")
    print(f"  backbone eta per synapse at trial end: {[f'{e:.4f}' for e in etas]}")
    print(f"  peak dimers seen by the tracker: {total_dimers}")
    print()
    blockers = []
    if not (ok1 and ok2):
        blockers.append("(1) coupling_weights did NOT reach the callee")
    if all(e == 0.0 for e in etas):
        blockers.append("(3) eta = 0 at every synapse -> k_cross ~ sqrt(eta_i*eta_j) = 0")
    if total_dimers == 0:
        blockers.append("(0) NO DIMERS reached the tracker -- nothing to bond regardless")
    if formed == 0:
        print("  ZERO bonds formed. Attributed cause(s):")
        for b in blockers:
            print(f"    - {b}")
        if not blockers:
            print("    - UNATTRIBUTED: weights arrived, eta nonzero, dimers present, "
                  "yet no bonds. That is a NEW finding and must be escalated.")
    else:
        print("  Bonds DID form.")
    results['bonds'] = dict(formed=int(formed), etas=etas, peak_dimers=int(total_dimers),
                            blockers=blockers)

    # ---- FAILING-FIRST: does this probe actually DETECT the omission? ------
    # Standing instruction: demonstrate the check failing before it passes. This
    # reproduces the HISTORICAL call signature verbatim -- pre-92c623f, run_trial
    # called tracker.step(physics_dt, synapses, positions) with NO coupling_weights
    # while step_network_per_synapse already passed it. That one-site-fixed /
    # one-site-missed split IS substrate-audit item 16's "gap in that fix".
    section("FAILING-FIRST — the probe against the pre-92c623f call signature")
    CALLS.clear(); STEPS.clear()
    net3 = make_network(3)
    stim3 = [{"voltage": -10e-3, "reward": False} for _ in net3.synapses]
    for _ in range(60):
        RSD.step_network_per_synapse(net3, 0.005, stim3)
    CALLS.clear(); STEPS.clear()
    # the historical omission, verbatim:
    net3.entanglement_tracker.step(0.005, net3.synapses, net3.positions)
    hist = list(CALLS)
    hist_steps = list(STEPS)
    detected = bool(hist_steps) and not any(st['got_weights'] for st in hist_steps)
    print(f"  tracker.step() called WITHOUT coupling_weights (as the old code did)")
    print(f"  probe observed: got_weights={[st['got_weights'] for st in hist_steps]}, "
          f"reached _update_entanglement={[st['reached_update_entanglement'] for st in hist_steps]}")
    for c in hist:
        print(f"    inner call: {'ARRIVED' if c['got_weights'] else 'None (EARLY RETURN)'}"
              f"  dimers={c['n_dimers']}  cross {c['cross_before']}->{c['cross_after']}")
    print(f"  -> probe DETECTS the omission: {'YES' if detected else 'NO -- the probe is blind, do not trust its PASS'}")
    results['failing_first_detects_omission'] = bool(detected)

    # ---- VERDICT ----------------------------------------------------------
    section("VERDICT")
    v = "PASS" if (ok1 and ok2 and detected) else "FAIL"
    print(f"  probe detects the historical omission (failing-first): {'YES' if detected else 'NO'}")
    print(f"  coupling_weights reaches _update_entanglement at BOTH driver sites: {v}")
    print(f"  bonds formed: {formed}")
    if formed == 0 and blockers and v == "PASS":
        print("  -> MEASURED ZERO WITH AN IDENTIFIED CAUSE. Per the MO's acceptance this")
        print("     is a PASS: the omission is closed, and the remaining blocker(s) are")
        print("     named above and are NOT this unit's surface.")
    print("\n  LIMITS: 3 synapses, 2.0 s trial budget -- sized to reach the every-10th-step")
    print("  tracker call, NOT to reproduce a full 90 s trial. This measures the CALL PATH,")
    print("  not the long-run topology. eta is read at trial end, not integrated.")

    results['verdict'] = v
    p = os.path.join(HERE, 'coupling_weights_reach_probe_results.json')
    with open(p, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\npersisted -> {p}")
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
