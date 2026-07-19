#!/usr/bin/env python3
"""
PO-5 UNIT 4 — is `collective_field_kT` a percolation control parameter?

THE ARCHITECTURE, as implemented (not an analogy):
    tryptophan module  ->  model6_core.py:555  self._collective_field_kT = trp_state[...]
    that field         ->  dimer_particles.py:454
                           em_rate = k_base * (collective_field_kT / reference_kT) * coh * g
So the tryptophan network is the NETWORK, `collective_field_kT` is the BUS (a single global
scalar multiplying EVERY pair's bond rate), and dimers are the BITS.

WHY THIS IS THE RIGHT KNOB. L·PO5-4 showed birth-pairing (83% of bonds) builds ~6 disjoint
temporal cohorts, and the 17% distance-based EM bonds BRIDGE them into one blob. The bus sets
the density of exactly those bridging edges. It is therefore the percolation control parameter:
low bus -> cohorts survive; high bus -> single blob.

THE GAP THIS UNIT DOES NOT FIX. The condensate's only handle on the bus is
`model6_core.py:543`: backbone_eta * E_invasion -- and both factors measure 0.0000 in every
live trial. This probe OVERRIDES the bus directly to ask whether a transition exists at all.
It does NOT reconnect the condensate and makes no claim that it is connected.

PRE-REGISTERED PREDICTIONS (stated before the run):
  P1. At bus = 0 the EM rate is identically zero, so NO P2 bonds form and the component count
      must equal the P0-only (birth-cohort) count. This doubles as the override's positive
      control: if bus=0 does not zero the P2 bonds, the override is not working and the run
      is INVALID.
  P2. As bus increases, components -> 1 and largest_frac -> 1.0.
  P3. If a transition exists, there is a bus value where components is strictly between 1 and
      the P0 count. If NO swept value yields an intermediate state, report NO_TRANSITION --
      the architecture has nowhere for a modulator to act, which is a finding.

lambda_2 (algebraic connectivity) is recorded as a DIAGNOSTIC ONLY. `entanglement-topology-
measurement` A5 rules the spectrum is "a scientist's instrument for characterising the
partition's robustness", NEVER the computation. That is exactly the use here.
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

BIRTH_WINDOW = 0.1
BUS_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
T_SIM, DT = 1.0, 0.005


def instrument(dp, bus_override):
    """Wrap on the INSTANCE. Tags bond provenance AND overrides the bus.

    The override replaces the collective_field_kT kwarg on dp.step -- the single value
    model6_core passes down from the tryptophan module. Nothing else is touched, and the
    native value is recorded so we know where the system actually sits.
    """
    state = {"phase": None, "origin": {}, "native_bus": []}
    o_step = dp.step
    o_pop, o_ent = dp.step_population, dp.step_entanglement
    o_create, o_remove = dp._create_bond, dp._remove_bond
    o_remove_all = dp._remove_all_bonds_for_dimer

    def w_step(*a, **k):
        if "collective_field_kT" in k:
            state["native_bus"].append(float(k["collective_field_kT"]))
            if bus_override is not None:
                k["collective_field_kT"] = bus_override
        return o_step(*a, **k)

    def w_pop(*a, **k):
        state["phase"] = "population"
        try: return o_pop(*a, **k)
        finally: state["phase"] = None

    def w_ent(*a, **k):
        state["phase"] = "entanglement"
        try: return o_ent(*a, **k)
        finally: state["phase"] = None

    def w_create(i, j, s):
        key = (min(i, j), max(i, j)); already = key in dp._bond_lookup
        r = o_create(i, j, s)
        if not already and key in dp._bond_lookup:
            if state["phase"] == "population":
                org = "P0"
            else:
                by = {d.id: d for d in dp.dimers}
                di, dj = by.get(i), by.get(j)
                if di is None or dj is None:
                    org = "unknown"
                else:
                    sb = abs(di.birth_time - dj.birth_time) < BIRTH_WINDOW
                    bt = bool(di.template_bound) and bool(dj.template_bound)
                    org = "P1" if (sb and bt) else "P2"
            state["origin"][key] = org
        return r

    def w_remove(i, j):
        state["origin"].pop((min(i, j), max(i, j)), None)
        return o_remove(i, j)

    def w_remove_all(did):
        doomed = [k for k, b in dp._bond_lookup.items()
                  if b.dimer_i == did or b.dimer_j == did]
        r = o_remove_all(did)
        for k in doomed:
            state["origin"].pop(k, None)
        return r

    dp.step = w_step
    dp.step_population, dp.step_entanglement = w_pop, w_ent
    dp._create_bond, dp._remove_bond = w_create, w_remove
    dp._remove_all_bonds_for_dimer = w_remove_all
    return state


def comps_and_lambda2(nodes, edges, want_spectrum=True):
    idx = {n: i for i, n in enumerate(sorted(nodes))}
    n = len(idx)
    if n < 2:
        return n, None, 1.0
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    A = np.zeros((n, n)) if (want_spectrum and n <= 1600) else None
    for a, b in edges:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        ra, rb = find(ia), find(ib)
        if ra != rb:
            parent[ra] = rb
        if A is not None:
            A[ia, ib] = A[ib, ia] = 1.0
    roots = {find(i) for i in range(n)}
    sizes = {}
    for i in range(n):
        r = find(i); sizes[r] = sizes.get(r, 0) + 1
    largest_frac = max(sizes.values()) / n
    lam2 = None
    if A is not None:
        L = np.diag(A.sum(1)) - A
        ev = np.linalg.eigvalsh(L)
        lam2 = float(ev[1]) if len(ev) > 1 else None
    return len(roots), lam2, largest_frac


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
    state = instrument(dp, bus)
    rel = PresynapticRelease(seed=seed)

    t, t0 = 0.0, time.time()
    for _ in range(int(round(T_SIM / DT))):
        glu = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": glu})
        t += DT

    ent = {d.id for d in dp.dimers if d.is_entangled}
    all_e = [(a, b) for (a, b) in dp._bond_lookup if a in ent and b in ent]
    p0_e = [(a, b) for (a, b), o in state["origin"].items()
            if o == "P0" and a in ent and b in ent]
    n_p2 = sum(1 for (a, b), o in state["origin"].items()
               if o == "P2" and a in ent and b in ent)

    c_all, lam2, lfrac = comps_and_lambda2(ent, all_e)
    c_p0, _, _ = comps_and_lambda2(ent, p0_e, want_spectrum=False)
    npair = len(ent) * (len(ent) - 1) / 2

    return {"bus": bus, "n_ent": len(ent), "n_bonds": len(all_e), "n_P2": n_p2,
            "saturation": len(all_e) / npair if npair else 0.0,
            "components_all": c_all, "components_P0only": c_p0,
            "largest_frac": lfrac, "lambda2": lam2,
            "native_bus_mean": float(np.mean(state["native_bus"])) if state["native_bus"] else None,
            "elapsed_s": time.time() - t0}


def main():
    print("=" * 100)
    print("PO-5 UNIT 4 — is `collective_field_kT` (the BUS) a percolation control parameter?")
    print("  P1: bus=0 => zero P2 bonds => components == P0-only count  (override positive control)")
    print("  P2: bus up => components -> 1")
    print("  P3: no intermediate state at ANY value => NO_TRANSITION (a finding)")
    print("=" * 100, flush=True)

    print("\n[native] measuring what the tryptophan module actually produces...", flush=True)
    native = run(None)
    print(f"  NATIVE bus value = {native['native_bus_mean']:.4f} kT   "
          f"(reference_kT=20.0, FIELD_THRESHOLD_KT=20.0)")
    print(f"  -> components {native['components_all']}, saturation {native['saturation']:.4f}, "
          f"lambda2 {native['lambda2']:.4e}", flush=True)

    hdr = (f"\n{'bus':>7s} {'n_ent':>6s} {'n_bonds':>9s} {'n_P2':>8s} {'sat':>7s} "
           f"{'comps':>6s} {'P0comps':>8s} {'lgfrac':>7s} {'lambda2':>11s} {'sec':>5s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = [native]
    for bus in BUS_VALUES:
        r = run(bus); rows.append(r)
        l2 = f"{r['lambda2']:.4e}" if r["lambda2"] is not None else "n/a"
        print(f"{bus:7.2f} {r['n_ent']:6d} {r['n_bonds']:9d} {r['n_P2']:8d} "
              f"{r['saturation']:7.4f} {r['components_all']:6d} {r['components_P0only']:8d} "
              f"{r['largest_frac']:7.4f} {l2:>11s} {r['elapsed_s']:5.0f}", flush=True)
        with open(os.path.join(SWEEP_DIR, "po5_unit4_bus_percolation_results.json"), "w") as f:
            json.dump({"native": native, "sweep": rows}, f, indent=2)

    swept = [r for r in rows if r["bus"] is not None]
    zero = next((r for r in swept if r["bus"] == 0.0), None)
    p1 = (zero is not None and zero["n_P2"] == 0
          and zero["components_all"] == zero["components_P0only"])
    interm = [r for r in swept
              if 1 < r["components_all"] < max(x["components_P0only"] for x in swept)]

    print("\n" + "=" * 100)
    print(f"P1 override positive control : {'PASS' if p1 else 'FAIL'}"
          + ("" if p1 else "  -> the override did NOT zero P2; run is INVALID"))
    if not p1:
        print("VERDICT: INVALID — no percolation claim is made.")
        return
    if interm:
        print(f"P3 VERDICT: TRANSITION EXISTS — {len(interm)} bus value(s) give an intermediate")
        print("   component count. The architecture HAS an operating point for a modulator.")
        for r in interm:
            print(f"     bus={r['bus']:.2f} -> {r['components_all']} components, "
                  f"largest_frac {r['largest_frac']:.4f}")
    else:
        print("P3 VERDICT: NO_TRANSITION — every bus value gives either the birth-cohort count")
        print("   or a single blob, with nothing between. There is nowhere for a modulator to")
        print("   act, and the problem is UPSTREAM of the bus (the 100 ms all-to-all birth rule).")
    print("\nLIMITS: single synapse, 1 s, one seed per bus value. The condensate is NOT")
    print("reconnected — this overrides the bus directly and says nothing about whether")
    print("backbone_eta * E_invasion can reach it. lambda2 is a diagnostic (A5), not a readout.")


if __name__ == "__main__":
    main()
