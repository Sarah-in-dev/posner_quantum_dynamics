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
CELL_NM = 8.0          # cell size; reported with the verdict (PREREG §8)
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
def pbond_matrices(dp, state):
    """Return (P_bond dict by subset, cell coords, occupancy). PREREG §3."""
    ent = [d for d in dp.dimers if d.is_entangled]
    if len(ent) < 2:
        return None
    pos = np.asarray([d.position for d in ent], dtype=float)
    ids = [d.id for d in ent]
    idx = {i: k for k, i in enumerate(ids)}
    cell = np.floor(pos[:, :2] / CELL_NM).astype(int)      # 2-D binning (z span is 20 nm)
    keys = [tuple(c) for c in cell]
    uniq = sorted(set(keys))
    cidx = {c: k for k, c in enumerate(uniq)}
    occ = np.zeros(len(uniq))
    for k in keys:
        occ[cidx[k]] += 1
    keep = [k for k, c in enumerate(uniq) if occ[k] >= MIN_OCC]
    if len(keep) < 2:
        return None
    keep_set = {uniq[k] for k in keep}
    remap = {c: n for n, c in enumerate(sorted(keep_set))}
    K = len(remap)

    subsets = {"ALL": None, "P0_birth_inherit": "P0_birth_inherit",
               "P1_burst": "P1_burst", "P2_em": "P2_em"}
    counts = {s: np.zeros((K, K)) for s in subsets}
    nb = {s: 0 for s in subsets}

    for (a_id, b_id), bond in dp._bond_lookup.items():
        ka, kb = idx.get(a_id), idx.get(b_id)
        if ka is None or kb is None:
            continue
        ca, cb = keys[ka], keys[kb]
        if ca not in remap or cb not in remap:
            continue
        ia, ib = remap[ca], remap[cb]
        org = state["origin"].get((a_id, b_id), "unknown")
        for s, want in subsets.items():
            if want is None or want == org:
                counts[s][ia, ib] += 1; counts[s][ib, ia] += 1
                nb[s] += 1

    n = np.zeros(K)
    for k in keys:
        if k in remap:
            n[remap[k]] += 1
    avail = np.outer(n, n).astype(float)
    np.fill_diagonal(avail, n * (n - 1))       # counts[a,a] was double-incremented too
    avail[avail == 0] = np.nan

    cent = np.array([[c[0] + 0.5, c[1] + 0.5] for c in sorted(remap)]) * CELL_NM
    sep = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)

    out = {}
    for s in subsets:
        P = counts[s] / avail
        out[s] = {"P": P, "n_bonds": nb[s]}
    return {"subsets": out, "sep": sep, "occ": n, "K": K}


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
            m = pbond_matrices(dp, state)
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
                       "runs": [{k: v for k, v in r.items() if k != "snaps"} for r in results]},
                      f, indent=2)

    print("=" * 78)
    print("PO-5 UNIT 2 · Q-B — pair-level input selectivity")
    print(f"  T={T}s dt={dt} cell={CELL_NM}nm  arms: A(seeds {SEEDS_A}) B({SEEDS_B}) NULL({SEEDS_NULL})")
    print("=" * 78, flush=True)

    # ---- AMENDMENT A2.5 pre-flight: prove the instrument has resolution before
    # consuming the exclusive slot. A resolution failure now costs ~1 min, not ~50.
    print("PRE-FLIGHT (A2.5): 1 s run, asserting occupied cells >= MIN_CELLS", flush=True)
    pf = run_arm("A", 999, 1.0, dt, [1.0], log)
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

    # ---- score, whole set + provenance split (A2.2) ----
    print("\nSCORES  (PRIMARY = ALL; sub-sets are SECONDARY and decide nothing — A2.2)")
    verdicts = {}
    for subset in ["ALL", "P0_birth_inherit", "P1_burst", "P2_em"]:
        def Rs(lbl):
            o = []
            for r in results:
                if r["label"] != lbl:
                    continue
                m = r["snaps"][-1]["mats"]
                if m is None:
                    continue
                o.append(residual(m["subsets"][subset]["P"], m["sep"]))
            return o
        RA, RB, RN = Rs("A"), Rs("B"), Rs("NULL")
        nb = np.mean([r["snaps"][-1]["mats"]["subsets"][subset]["n_bonds"]
                      for r in results if r["snaps"][-1]["mats"]]) if results else 0
        K = results[0]["snaps"][-1]["mats"]["K"] if results[0]["snaps"][-1]["mats"] else 0
        if subset != "ALL" and nb < SUBSET_MIN_BONDS:
            print(f"  {subset:20s} INSUFFICIENT (mean {nb:.0f} bonds < {SUBSET_MIN_BONDS})")
            verdicts[subset] = "INSUFFICIENT"
            continue
        d_null = _mean_pairdist(RA + RN) if (RA and RN) else 0.0
        d_in = _mean_pairdist_between(RA, RB) if (RA and RB) else None
        ratio = (d_in / d_null) if (d_in is not None and d_null > 0) else None
        v = classify(ratio, K, drive_ok, instrument_ok and posctl_ok)
        verdicts[subset] = v
        rtxt = f"{ratio:.3f}" if ratio is not None else "n/a"
        tag = "PRIMARY" if subset == "ALL" else "secondary"
        print(f"  {subset:20s} d_input={d_in if d_in is None else round(d_in,5)} "
              f"d_null={d_null:.5f} ratio={rtxt} cells={K} -> {v}  [{tag}]")

    print("\n" + "=" * 78)
    print(f"VERDICT (PRIMARY, whole realised bond set): {verdicts.get('ALL')}")
    print("=" * 78)
    print("A2.2 precedence: the whole-set verdict stands regardless of any sub-set result.")
    print(f"LIMITS: single synapse, {T}s, 3 seeds/arm, cell={CELL_NM}nm — a FALSIFIED is")
    print("'pair-flat at or above the cell scale under these conditions'.")

    with open(out_path, "w") as f:
        json.dump({"verdicts": verdicts, "drive_A": dA, "drive_B": dB,
                   "instrument_ok": instrument_ok, "posctl_ok": posctl_ok,
                   "drive_ok": drive_ok, "total_elapsed_s": total,
                   "thresholds": {"CONFIRM": RATIO_CONFIRM, "FALSIFY": RATIO_FALSIFY,
                                  "CELL_NM": CELL_NM, "MIN_OCC": MIN_OCC},
                   "runs": [{k: v for k, v in r.items() if k != "snaps"} for r in results]},
                  f, indent=2)
    print(f"\npersisted -> {out_path}")


if __name__ == "__main__":
    main()
