#!/usr/bin/env python3
"""
PO-5 UNIT 3 — is the P0 bond graph an indifference graph on birth time?

Pre-registered: docs/PREREG_PO5_UNIT3_BIRTH_COHORTS.md (committed BEFORE this run).

STRUCTURAL CLAIM, from dimer_particles.py:218-228 -- a birth-bond forms iff both dimers are
template_bound, the older is_entangled, and |delta birth_time| < 0.1 s; and all dimers made in
one step_population call share birth_time (:210). So the P0 graph is a UNIT-INTERVAL
(indifference) graph on the birth-time axis, whose components are the maximal runs of birth
times with no gap > 100 ms.

PREDICTION (registered before the run):
    components(P0 graph) == 1 + count(gaps > 0.1 s)

Two arms. The PULSED arm is NOT claimed to be silent -- BASELINE_RATE_HZ = 0.5 and calcium does
not stop; whether births actually pause is the measurement.
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

BIRTH_WINDOW = 0.1     # dimer_particles.py:222


def instrument(dp):
    """Instance-level wrapping; calls through, alters nothing. Tags each bond's origin."""
    state = {"phase": None, "origin": {}}
    o_pop, o_ent = dp.step_population, dp.step_entanglement
    o_create, o_remove = dp._create_bond, dp._remove_bond
    o_remove_all = dp._remove_all_bonds_for_dimer

    def w_pop(*a, **k):
        state["phase"] = "population"
        try: return o_pop(*a, **k)
        finally: state["phase"] = None

    def w_ent(*a, **k):
        state["phase"] = "entanglement"
        try: return o_ent(*a, **k)
        finally: state["phase"] = None

    def w_create(i, j, strength):
        key = (min(i, j), max(i, j)); already = key in dp._bond_lookup
        r = o_create(i, j, strength)
        if not already and key in dp._bond_lookup:
            if state["phase"] == "population":
                org = "P0_birth_inherit"
            else:
                by = {d.id: d for d in dp.dimers}
                di, dj = by.get(i), by.get(j)
                if di is None or dj is None:
                    org = "unknown"
                else:
                    sb = abs(di.birth_time - dj.birth_time) < BIRTH_WINDOW
                    bt = bool(di.template_bound) and bool(dj.template_bound)
                    org = "P1_burst" if (sb and bt) else "P2_em"
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

    dp.step_population, dp.step_entanglement = w_pop, w_ent
    dp._create_bond, dp._remove_bond = w_create, w_remove
    dp._remove_all_bonds_for_dimer = w_remove_all
    return state


def components(nodes, edges):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return len({find(n) for n in nodes})


def analyse(dp, state):
    """The registered comparison: predicted-from-gaps vs measured P0 components."""
    tb = [d for d in dp.dimers if d.template_bound]
    if len(tb) < 2:
        return None
    ids = {d.id for d in tb}
    births = sorted({round(d.birth_time, 9) for d in tb})
    gaps = np.diff(np.array(births)) if len(births) > 1 else np.array([])
    n_big = int((gaps > BIRTH_WINDOW).sum()) if gaps.size else 0
    predicted = 1 + n_big

    p0_edges = [(a, b) for (a, b), org in state["origin"].items()
                if org == "P0_birth_inherit" and a in ids and b in ids]
    measured = components(ids, p0_edges)

    return {
        "n_template_bound": len(tb),
        "n_distinct_births": len(births),
        "n_p0_edges": len(p0_edges),
        "max_gap_s": float(gaps.max()) if gaps.size else None,
        "median_gap_s": float(np.median(gaps)) if gaps.size else None,
        "p99_gap_s": float(np.percentile(gaps, 99)) if gaps.size else None,
        "n_gaps_over_window": n_big,
        "predicted_components": predicted,
        "measured_components": measured,
        "match": predicted == measured,
        "excess": measured - predicted,
        "min_P_S": float(min(d.singlet_probability for d in dp.dimers)) if dp.dimers else None,
    }


def drive(arm, t):
    if arm == "SUSTAINED":
        return 0.95
    period, on = 0.6, 0.2          # 0.2 s on, 0.4 s off  -> drive gap 400 ms > 100 ms window
    return 0.95 if (t % period) < on else 0.0


def run(arm, seed, T=5.0, dt=0.005, samples=(1.0, 3.0, 5.0)):
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
    state = instrument(dp)
    rel = PresynapticRelease(seed=seed)

    out, t, nxt, t0 = [], 0.0, 0, time.time()
    max_glu = 0.0
    for _ in range(int(round(T / dt))):
        act = drive(arm, t)
        glu = rel.step(act, dt)                  # inside the physics loop
        max_glu = max(max_glu, glu)
        net.synapses[0].step(dt, {"voltage": -10e-3, "reward": False, "glutamate": glu})
        t += dt
        if nxt < len(samples) and t >= samples[nxt] - dt / 2:
            a = analyse(dp, state)
            if a:
                a["t"] = round(t, 3); a["arm"] = arm
                out.append(a)
                print(f"  [{arm}] t={t:.1f} tb={a['n_template_bound']:5d} "
                      f"births={a['n_distinct_births']:4d} maxgap={a['max_gap_s']:.4f}s "
                      f"gaps>{BIRTH_WINDOW}s={a['n_gaps_over_window']:3d}  "
                      f"pred={a['predicted_components']:3d} meas={a['measured_components']:3d}  "
                      f"{'MATCH' if a['match'] else 'MISMATCH (excess %d)' % a['excess']}",
                      flush=True)
            nxt += 1
    print(f"  [{arm}] done in {time.time()-t0:.0f}s, max_glu={max_glu:.3f}", flush=True)
    return out, max_glu


def main():
    print("=" * 86)
    print("PO-5 UNIT 3 — is the P0 bond graph an indifference graph on birth time?")
    print("  PREDICTION (registered): components(P0) == 1 + count(gaps > 0.1 s)")
    print("=" * 86, flush=True)

    results, glus = [], {}
    SEEDS = [4242, 4243, 4244]
    for arm in ("SUSTAINED", "PULSED"):
        for sd in SEEDS:
            r, g = run(arm, seed=sd)
            for row in r:
                row["seed"] = sd
            results += r; glus[f"{arm}_{sd}"] = g

    ok = all(r["match"] for r in results)
    any_gap = any(r["n_gaps_over_window"] > 0 for r in results)
    verdict = ("INCONCLUSIVE" if not results else
               "CONFIRMED" if ok else "FALSIFIED")

    print()
    print("=" * 86)
    print(f"STRUCTURAL VERDICT: {verdict}   "
          f"({sum(r['match'] for r in results)}/{len(results)} samples match)")
    print("=" * 86)
    for arm in ("SUSTAINED", "PULSED"):
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        mg = max(r["max_gap_s"] for r in rows)
        nb = max(r["n_gaps_over_window"] for r in rows)
        mc = rows[-1]["measured_components"]
        print(f"  {arm:10s} max birth gap {mg:.4f}s | gaps>100ms: {nb} | "
              f"final P0 components: {mc}")
    print()
    if not any_gap:
        print("  NO arm produced a birth gap > 100 ms.")
        print("  => comps=1 is fully explained by the birth mechanism, and §8's keystone")
        print("     acquires a concrete physical requirement: the input must be able to create")
        print("     >100 ms gaps in dimer FORMATION. Reported as a finding.")
    else:
        print("  At least one arm produced a birth gap > 100 ms -> formation CAN be gated")
        print("     finely enough to split the P0 graph. That is the channel §8 needs.")
    print()
    print("LIMITS: single synapse, 5 s, one seed per arm. Tests a MECHANISM, not")
    print("input-selectivity. No §8 verdict is produced or implied.")

    out = os.path.join(SWEEP_DIR, "po5_unit3_birth_cohorts_results.json")
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "samples": results, "max_glu": glus}, f, indent=2)
    print(f"\npersisted -> {out}")


if __name__ == "__main__":
    main()
