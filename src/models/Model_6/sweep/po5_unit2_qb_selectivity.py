#!/usr/bin/env python3
"""
PO-5 UNIT 2 · Q-B — does the realised bond set depend on INPUT at pair resolution?

Pre-registered: docs/PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md, incl. AMENDMENT A2.2 (the
provenance split and its precedence) and A2.3 (the _remove_dimer tripwire), both
committed BEFORE this run.

THE STATISTIC (PREREG §3). Dimer identities are not comparable across runs, so the
comparison is made on a density-normalised, spatially-binned bond probability:

    P_bond(a,b) = bonds between cell a and cell b / (n_a * n_b)      a != b
    P_bond(a,a) = bonds within cell a            / C(n_a, 2)

Dividing by the AVAILABLE pair count is the load-bearing step: it answers §8's sentence
"the partition carries no more than active-region density" by dividing density out. What
remains is pair-level -- how likely THESE TWO LOCATIONS are to bond given their candidates.

`g` is geometry not input (Unit 1: D = 33.5), so the scored quantity is the residual
after regressing P_bond on pair separation, fitted PER RUN on that run's own data.

THE NULL is seeds-only: same input, different RNG. No arm is "silent" -- this PO is
forbidden an activation-floor null (BASELINE_RATE_HZ = 0.5) and the standing scar is that
three probes on this board used a control assumed silent that was not.

INPUT-A vs INPUT-B differ in the TEMPORAL pattern of activation and are matched on total
integrated drive BY CONSTRUCTION (same values, reversed order), so any difference is in
pattern, not amount. Release is stepped INSIDE the physics loop at physics_dt, matching
the shipped run_trial pattern (sweep/run_spatial_discovery.py:197-204) per mo-f3-001's
surviving clause.
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

# ---- registered thresholds (PREREG §5, A2.2). Do not move after the run. ----
RATIO_CONFIRM = 3.0
RATIO_FALSIFY = 1.5
MIN_OCC = 5            # min dimers in a cell for it to be scored
MIN_CELLS = 10         # min occupied cells, else INCONCLUSIVE
SUBSET_MIN_BONDS = 1000   # A2.2 sub-set guard
# AMENDMENT A2.4: was 40.0, which gave cells=4 against MIN_CELLS=10 -> the verdict could
# only ever return INCONCLUSIVE. 8.0 is derived from Unit 1's measured geometry
# (r_p10=3.71 < 8.0 < r_p50=9.78; whole cloud r_max=36.45 nm), NOT from any Q-B outcome.
CELL_NM = 6.0          # A2.6: 8.0 gave cells=9 vs MIN_CELLS=10. Rule (Unit 1 geometry):
                       # above r_p10=3.71, below r_p50=9.78, and > MIN_CELLS at the scored sample.
BIRTH_WINDOW = 0.1


# ===========================================================================
# Verdict machinery
# ===========================================================================
def classify(ratio, n_cells, drive_ok, instrument_ok):
    if not instrument_ok:
        return "INSTRUMENT_INVALID"
    if not drive_ok or n_cells < MIN_CELLS or ratio is None:
        return "INCONCLUSIVE"
    if ratio >= RATIO_CONFIRM:
        return "CONFIRMED"
    if ratio <= RATIO_FALSIFY:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def demonstrate_verdict():
    """PREREG §6 — all four labels on synthetic input, before any model is built."""
    rng = np.random.default_rng(0)
    cases = []
    # same distribution as null -> FALSIFIED
    base = rng.normal(0, 1, (6, 8, 8))
    d_null = _mean_pairdist(base[:3]); d_in = _mean_pairdist_between(base[3:5], base[5:6])
    cases.append(("A,B same distribution as null", d_in / d_null, 20, True, True, "FALSIFIED"))
    # large offset -> CONFIRMED
    A = rng.normal(0, 0.01, (3, 8, 8)); B = A + 5.0
    cases.append(("A,B separated by a large offset", _mean_pairdist_between(A, B) /
                  max(_mean_pairdist(A), 1e-9), 20, True, True, "CONFIRMED"))
    # ~2x null spread -> INCONCLUSIVE
    A2 = rng.normal(0, 1, (3, 8, 8)); B2 = A2 + 2.0 * np.std(A2)
    cases.append(("A,B at ~2x the null spread", 2.0, 20, True, True, "INCONCLUSIVE"))
    # instrument desync -> INSTRUMENT_INVALID
    cases.append(("provenance desynchronised", 10.0, 20, True, False, "INSTRUMENT_INVALID"))

    print("=" * 78)
    print("VERDICT DEMONSTRATION (PREREG §6) — all four labels before the model is built")
    print("=" * 78)
    ok = True
    for label, ratio, ncell, dok, iok, required in cases:
        got = classify(ratio, ncell, dok, iok)
        status = "ok" if got == required else "MISMATCH"
        if got != required:
            ok = False
        print(f"  {label:34s} ratio={ratio:8.3f} -> {got:20s} [{status}]")
    print()
    if not ok:
        print("ABORT: the verdict function does not discriminate its outcomes. (PREREG §6)")
        sys.exit(1)
    print("All four outcomes reachable and correctly labelled. Verdict admissible.\n")


def _mean_pairdist(mats):
    m = [x[np.isfinite(x)] for x in mats]
    ds = [np.linalg.norm(m[i] - m[j]) for i in range(len(m)) for j in range(i + 1, len(m))]
    return float(np.mean(ds)) if ds else 0.0


def _mean_pairdist_between(A, B):
    ds = [np.linalg.norm(a[np.isfinite(a)] - b[np.isfinite(b)]) for a in A for b in B]
    return float(np.mean(ds)) if ds else 0.0


# ===========================================================================
# Instrumentation (Q-A's, plus the A2.3 tripwire)
# ===========================================================================
def instrument(dp):
    state = {"phase": None, "origin": {},
             "counts": {"P0_birth_inherit": 0, "P1_burst": 0, "P2_em": 0, "unknown": 0},
             "remove_dimer_calls": 0}
    o_pop, o_ent = dp.step_population, dp.step_entanglement
    o_create, o_remove = dp._create_bond, dp._remove_bond
    o_remove_all = dp._remove_all_bonds_for_dimer
    o_remove_dimer = dp._remove_dimer

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
            by = {d.id: d for d in dp.dimers}
            di, dj = by.get(i), by.get(j)
            if state["phase"] == "population":
                org = "P0_birth_inherit"
            elif state["phase"] == "entanglement" and di is not None and dj is not None:
                sb = abs(di.birth_time - dj.birth_time) < BIRTH_WINDOW
                bt = bool(di.template_bound) and bool(dj.template_bound)
                org = "P1_burst" if (sb and bt) else "P2_em"
            else:
                org = "unknown"
            state["origin"][key] = org; state["counts"][org] += 1
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

    def w_remove_dimer(dimer):
        # AMENDMENT A2.3 tripwire. Calls through; alters nothing.
        state["remove_dimer_calls"] += 1
        return o_remove_dimer(dimer)

    dp.step_population, dp.step_entanglement = w_pop, w_ent
    dp._create_bond, dp._remove_bond = w_create, w_remove
    dp._remove_all_bonds_for_dimer = w_remove_all
    dp._remove_dimer = w_remove_dimer
    return state


# ===========================================================================
# The statistic
# ===========================================================================
def persistable_cells_and_pairs(dp, state):
    """Emit ABSOLUTE-lattice cell occupancies and per-subset pair counts.

    L·PO5-3 fix: cells are keyed by absolute coordinates (floor(x/CELL_NM), floor(y/CELL_NM)),
    NOT by this run's own occupied set, so a cell denotes the same physical place in every
    run and matrices built from this are comparable across runs. Scoring is done OFFLINE by
    sweep/po5_unit2_score.py (MO ruling 028).
    """
    ent = [d for d in dp.dimers if d.is_entangled]
    if len(ent) < 2:
        return None
    pos = np.asarray([d.position for d in ent], dtype=float)
    ids = [d.id for d in ent]
    cell_of = {}
    occ = {}
    for k, i in enumerate(ids):
        c = (int(np.floor(pos[k, 0] / CELL_NM)), int(np.floor(pos[k, 1] / CELL_NM)))
        key = f"({c[0]},{c[1]})"
        cell_of[i] = key
        occ[key] = occ.get(key, 0) + 1

    pairs = {s: {} for s in ["ALL", "P0_birth_inherit", "P1_burst", "P2_em"]}
    for (a_id, b_id), bond in dp._bond_lookup.items():
        ca, cb = cell_of.get(a_id), cell_of.get(b_id)
        if ca is None or cb is None:
            continue
        k = f"{ca}|{cb}" if ca <= cb else f"{cb}|{ca}"
        org = state["origin"].get((a_id, b_id), "unknown")
        pairs["ALL"][k] = pairs["ALL"].get(k, 0) + 1
        if org in pairs:
            pairs[org][k] = pairs[org].get(k, 0) + 1
    n_occ = sum(1 for v in occ.values() if v >= MIN_OCC)
    return {"cells": occ, "pairs": pairs, "K": n_occ}


def residual(P, sep):
    """R = P - f_hat(|a-b|), f_hat a binned fit on THIS run's own data (PREREG §3)."""
    R = np.full_like(P, np.nan)
    finite = np.isfinite(P)
    if finite.sum() < 3:
        return R
    edges = np.quantile(sep[finite], np.linspace(0, 1, 9))
    edges = np.unique(edges)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = finite & (sep >= lo) & (sep <= hi)
        if m.sum() >= 2:
            R[m] = P[m] - np.nanmean(P[m])
    return R


# ===========================================================================
# Runs
# ===========================================================================
def activation_pattern(name, t, T):
    """INPUT-A and INPUT-B: same values, reversed order => identical integral."""
    hi, lo = 0.95, 0.15
    first = t < (T / 2.0)
    if name == "A":
        return hi if first else lo
    return lo if first else hi


def run_arm(arm, seed, T, dt, samples, log):
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

    integ_drive = 0.0
    max_glu = 0.0
    t = 0.0
    nxt = 0
    snaps = []
    t0 = time.time()
    for _ in range(int(round(T / dt))):
        act = activation_pattern(arm, t, T)
        glu = rel.step(act, dt)                    # inside the physics loop
        max_glu = max(max_glu, glu)
        integ_drive += act * dt
        net.synapses[0].step(dt, {"voltage": -10e-3, "reward": False, "glutamate": glu})
        t += dt
        if nxt < len(samples) and t >= samples[nxt] - dt / 2:
            m = persistable_cells_and_pairs(dp, state)
            snaps.append({"t": round(t, 3), "mats": m,
                          "n_bonds": len(dp._bond_lookup),
                          "n_dimers": len(dp.dimers)})
            log(f"    arm={arm} seed={seed} t={t:.2f} bonds={len(dp._bond_lookup)} "
                f"cells={m['K'] if m else 0} elapsed={time.time()-t0:.0f}s")
            nxt += 1
    return {"arm": arm, "seed": seed, "snaps": snaps, "integ_drive": integ_drive,
            "max_glu": max_glu, "remove_dimer_calls": state["remove_dimer_calls"],
            "elapsed_s": time.time() - t0,
            "conservation_ok": set(state["origin"]) == set(dp._bond_lookup)}


def main():
    demonstrate_verdict()

    T, dt = 5.0, 0.005
    samples = [5.0]
    SEEDS_A = [101, 102, 103]
    SEEDS_B = [101, 102, 103]
    SEEDS_NULL = [201, 202, 203]
    out_path = os.path.join(SWEEP_DIR, "po5_unit2_qb_results.json")
    results = []

    def log(msg):
        print(msg, flush=True)

    def persist():
        with open(out_path, "w") as f:
            json.dump({"thresholds": {"CONFIRM": RATIO_CONFIRM, "FALSIFY": RATIO_FALSIFY,
                                      "CELL_NM": CELL_NM, "MIN_OCC": MIN_OCC},
                       "runs": [dict(r, cells=r["snaps"][-1]["mats"]["cells"],
                                     pairs=r["snaps"][-1]["mats"]["pairs"])
                                if r["snaps"] and r["snaps"][-1]["mats"] else r
                                for r in results]}, f, indent=2)

    print("=" * 78)
    print("PO-5 UNIT 2 · Q-B — pair-level input selectivity")
    print(f"  T={T}s dt={dt} cell={CELL_NM}nm  arms: A(seeds {SEEDS_A}) B({SEEDS_B}) NULL({SEEDS_NULL})")
    print("=" * 78, flush=True)

    # ---- AMENDMENT A2.5 pre-flight: prove the instrument has resolution before
    # consuming the exclusive slot. A resolution failure now costs ~1 min, not ~50.
    print(f"PRE-FLIGHT (A2.5/A2.6): {T}s run at the scored sample, "
          f"asserting occupied cells >= MIN_CELLS", flush=True)
    pf = run_arm("A", 999, T, dt, samples, log)   # A2.6: gate the SCORED condition
    pf_cells = pf["snaps"][-1]["mats"]["K"] if pf["snaps"][-1]["mats"] else 0
    print(f"  pre-flight cells={pf_cells} (need >= {MIN_CELLS}), "
          f"elapsed={pf['elapsed_s']:.0f}s", flush=True)
    if pf_cells < MIN_CELLS:
        print(f"\nPREFLIGHT_FAIL: {pf_cells} occupied cells < MIN_CELLS={MIN_CELLS}. "
              f"The verdict could only return INCONCLUSIVE. Aborting BEFORE the matrix.",
              flush=True)
        sys.exit(1)
    print("  pre-flight PASS — the instrument has resolution. Starting matrix.\n", flush=True)

    plan = ([("A", s) for s in SEEDS_A] + [("B", s) for s in SEEDS_B] +
            [("A", s) for s in SEEDS_NULL])
    t_start = time.time()
    for n, (arm, seed) in enumerate(plan):
        label = arm if n < 6 else "NULL"
        log(f"[{n+1}/{len(plan)}] arm={label} seed={seed} starting "
            f"(total elapsed {time.time()-t_start:.0f}s)")
        r = run_arm(arm, seed, T, dt, samples, log)
        r["label"] = label
        results.append(r); persist()
        if r["remove_dimer_calls"]:
            print(f"\nABORT: _remove_dimer called {r['remove_dimer_calls']}x — "
                  f"AMENDMENT A2.3 tripwire. INSTRUMENT_INVALID; no verdict.", flush=True)
            sys.exit(1)

    total = time.time() - t_start
    print(f"\nall runs complete, total elapsed {total:.0f}s ({total/60:.1f} min)", flush=True)

    # ---- gates ----
    instrument_ok = all(r["conservation_ok"] for r in results) and \
                    all(r["remove_dimer_calls"] == 0 for r in results)
    posctl_ok = all(r["max_glu"] > 0 for r in results)
    dA = np.mean([r["integ_drive"] for r in results if r["label"] == "A"])
    dB = np.mean([r["integ_drive"] for r in results if r["label"] == "B"])
    drive_ok = abs(dA - dB) / max(dA, 1e-12) <= 0.05

    print("\nGATES")
    print(f"  instrument (conservation + A2.3 tripwire): {'PASS' if instrument_ok else 'FAIL'}")
    print(f"  positive control max_glu>0 every run     : {'PASS' if posctl_ok else 'FAIL'}"
          f"  (min {min(r['max_glu'] for r in results):.3f})")
    print(f"  drive matching A vs B                    : {'PASS' if drive_ok else 'FAIL'}"
          f"  (A={dA:.4f} B={dB:.4f})")

    # ---- NO SCORING HERE. MO ruling 028: scoring is a separate offline step. ----
    with open(out_path, "w") as f:
        json.dump({"drive_A": dA, "drive_B": dB, "instrument_ok": instrument_ok,
                   "posctl_ok": posctl_ok, "drive_ok": drive_ok,
                   "total_elapsed_s": total,
                   "thresholds": {"CELL_NM": CELL_NM, "MIN_OCC": MIN_OCC},
                   "runs": [{"arm": r["arm"], "seed": r["seed"], "label": r["label"],
                             "integ_drive": r["integ_drive"], "max_glu": r["max_glu"],
                             "elapsed_s": r["elapsed_s"],
                             "conservation_ok": r["conservation_ok"],
                             "remove_dimer_calls": r["remove_dimer_calls"],
                             "cells": r["snaps"][-1]["mats"]["cells"],
                             "pairs": r["snaps"][-1]["mats"]["pairs"]}
                            for r in results if r["snaps"] and r["snaps"][-1]["mats"]]},
                  f, indent=2)
    print(f"\npersisted scored intermediate -> {out_path}")
    print("SCORE IT OFFLINE:  python src/models/Model_6/sweep/po5_unit2_score.py "
          + out_path)
    print("(the scorer self-validates on planted-vs-flat before it will score real data)")


if __name__ == "__main__":
    main()
