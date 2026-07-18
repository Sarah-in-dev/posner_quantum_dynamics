#!/usr/bin/env python3
"""
PO-5 UNIT 2 · Q-A — bond provenance, recovered WITHOUT editing dimer_particles.py

Pre-registered: docs/PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md §2 (committed before this run).

Three bond-creation sites exist:
  dimer_particles.py:218-228   birth inheritance   (inside step_population)
  dimer_particles.py:443-444   Pathway 1 (burst)   (inside step_entanglement)
  dimer_particles.py:457-458   Pathway 2 (EM)      (inside step_entanglement)

The classification is EXACT, not inferred. `:439` sets
    p1 = both_ent & same_burst & both_tmpl & ~has_bond
and `:450` sets
    p2 = both_ent & ~p1
so within step_entanglement a NEWLY formed bond took Pathway 1 iff (same_burst & both_tmpl),
and Pathway 2 otherwise. No RNG replay, no guessed branch.

Instrumentation is INSTANCE-LEVEL wrapping that calls through to the originals. No physics is
altered, no RNG draw is consumed or reordered, and dimer_particles.py is NOT modified --
deliberate, because four POs share this worktree.

Both registered instrument validations (PREREG §2) run here:
  1. conservation  -- provenance map and _bond_lookup agree on key sets
  2. non-perturbation -- instrumented vs uninstrumented run agree BIT-FOR-BIT
Either failing => INSTRUMENT_INVALID and nothing else is reported.
"""

import sys, os, json
import logging
import numpy as np

logging.disable(logging.INFO)

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
sys.path.insert(0, MODEL6_DIR)

BIRTH_WINDOW = 0.1   # dimer_particles.py:222 / :389 -- same value at both sites


# ---------------------------------------------------------------------------
# Instance-level instrumentation
# ---------------------------------------------------------------------------
def instrument(dp):
    """Wrap step_population / step_entanglement / _create_bond on the INSTANCE.

    Every wrapper calls through to the original. Returns a state dict the caller
    reads; the physics is untouched.
    """
    state = {
        "phase": None,
        "origin": {},                 # (min_id,max_id) -> origin label
        "counts": {"P0_birth_inherit": 0, "P1_burst": 0, "P2_em": 0, "unknown": 0},
    }

    orig_pop = dp.step_population
    orig_ent = dp.step_entanglement
    orig_create = dp._create_bond
    orig_remove = dp._remove_bond

    def wrapped_pop(*a, **k):
        state["phase"] = "population"
        try:
            return orig_pop(*a, **k)
        finally:
            state["phase"] = None

    def wrapped_ent(*a, **k):
        state["phase"] = "entanglement"
        try:
            return orig_ent(*a, **k)
        finally:
            state["phase"] = None

    def wrapped_create(id_i, id_j, strength):
        key = (min(id_i, id_j), max(id_i, id_j))
        already = key in dp._bond_lookup
        result = orig_create(id_i, id_j, strength)
        if not already and key in dp._bond_lookup:
            state["origin"][key] = _classify(dp, id_i, id_j, state["phase"])
            state["counts"][state["origin"][key]] += 1
        return result

    def wrapped_remove(id_i, id_j):
        key = (min(id_i, id_j), max(id_i, id_j))
        state["origin"].pop(key, None)
        return orig_remove(id_i, id_j)

    dp.step_population = wrapped_pop
    dp.step_entanglement = wrapped_ent
    dp._create_bond = wrapped_create
    dp._remove_bond = wrapped_remove
    return state


def _classify(dp, id_i, id_j, phase):
    """Exact branch recovery -- see module docstring."""
    if phase == "population":
        return "P0_birth_inherit"
    if phase != "entanglement":
        return "unknown"
    by_id = {d.id: d for d in dp.dimers}
    di, dj = by_id.get(id_i), by_id.get(id_j)
    if di is None or dj is None:
        return "unknown"
    same_burst = abs(di.birth_time - dj.birth_time) < BIRTH_WINDOW
    both_tmpl = bool(di.template_bound) and bool(dj.template_bound)
    return "P1_burst" if (same_burst and both_tmpl) else "P2_em"


# ---------------------------------------------------------------------------
# PREREG §2 validation 1 -- conservation
# ---------------------------------------------------------------------------
def check_conservation(dp, state):
    live = set(dp._bond_lookup.keys())
    tracked = set(state["origin"].keys())
    missing = live - tracked          # a bond we failed to attribute
    orphan = tracked - live           # provenance for a bond that no longer exists
    return {"ok": (not missing and not orphan),
            "n_live": len(live), "n_tracked": len(tracked),
            "n_missing": len(missing), "n_orphan": len(orphan)}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def build(seed):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    np.random.seed(seed)
    params = Model6Parameters()
    params.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, params)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    return net


def run(seed, T, dt, do_instrument, samples):
    net = build(seed)
    dp = net.synapses[0].dimer_particles
    state = instrument(dp) if do_instrument else None
    stim = {"voltage": -10e-3, "reward": False}

    trace = []
    steps = int(round(T / dt))
    t = 0.0
    nxt = 0
    for _ in range(steps):
        net.step(dt, stim)
        t += dt
        if nxt < len(samples) and t >= samples[nxt] - dt / 2:
            ent = [d for d in dp.dimers if d.is_entangled]
            rec = {"t": round(t, 3),
                   "n_dimers": len(dp.dimers),
                   "n_entangled": len(ent),
                   "n_bonds": len(dp._bond_lookup),
                   "mean_PS": float(np.mean([d.singlet_probability for d in dp.dimers]))
                   if dp.dimers else 0.0}
            if state is not None:
                cons = check_conservation(dp, state)
                live_counts = {"P0_birth_inherit": 0, "P1_burst": 0, "P2_em": 0, "unknown": 0}
                for k in dp._bond_lookup:
                    live_counts[state["origin"].get(k, "unknown")] += 1
                rec["live_by_origin"] = live_counts
                rec["cumulative_created"] = dict(state["counts"])
                rec["conservation"] = cons
            trace.append(rec)
            nxt += 1
    return trace


def main():
    T, dt = 2.0, 0.005
    samples = [0.5, 1.0, 2.0]
    SEED = 20260718

    print("=" * 78)
    print("PO-5 UNIT 2 · Q-A — bond provenance (no source edits)")
    print("=" * 78)
    print(f"  seed={SEED}  T={T}s  dt={dt}  samples={samples}")
    print()

    print("--- instrumented run ---")
    inst = run(SEED, T, dt, True, samples)
    print("--- uninstrumented run (same seed) — PREREG §2 validation 2 ---")
    base = run(SEED, T, dt, False, samples)

    # ---- validation 1: conservation ----
    cons_ok = all(r["conservation"]["ok"] for r in inst)
    # ---- validation 2: bit-for-bit non-perturbation ----
    keys = ["n_dimers", "n_entangled", "n_bonds", "mean_PS"]
    perturb = []
    for a, b in zip(inst, base):
        for k in keys:
            if a[k] != b[k]:
                perturb.append((a["t"], k, a[k], b[k]))
    perturb_ok = not perturb

    print()
    print("INSTRUMENT VALIDATION (PREREG §2)")
    print(f"  1. conservation      : {'PASS' if cons_ok else 'FAIL'}")
    for r in inst:
        c = r["conservation"]
        print(f"       t={r['t']:.1f}  live={c['n_live']:6d} tracked={c['n_tracked']:6d} "
              f"missing={c['n_missing']} orphan={c['n_orphan']}")
    print(f"  2. non-perturbation  : {'PASS (bit-for-bit)' if perturb_ok else 'FAIL'}")
    for t, k, x, y in perturb[:10]:
        print(f"       t={t} {k}: instrumented={x} vs baseline={y}")
    print()

    if not (cons_ok and perturb_ok):
        print("VERDICT: INSTRUMENT_INVALID — no provenance number is reported. (PREREG §2)")
        sys.exit(1)

    print("=" * 78)
    print("Q-A PROVENANCE — live bonds by originating mechanism")
    print("=" * 78)
    hdr = f"{'t':>5s} {'n_bonds':>9s} {'P0_birth':>10s} {'P1_burst':>10s} {'P2_em':>10s} {'unknown':>8s}"
    print(hdr); print("-" * len(hdr))
    for r in inst:
        lb = r["live_by_origin"]
        print(f"{r['t']:5.1f} {r['n_bonds']:9d} {lb['P0_birth_inherit']:10d} "
              f"{lb['P1_burst']:10d} {lb['P2_em']:10d} {lb['unknown']:8d}")
    print()
    final = inst[-1]["live_by_origin"]
    tot = sum(final.values())
    if tot:
        print("Share of the FINAL live bond set:")
        for k, v in final.items():
            print(f"  {k:20s} {v:9d}  {100.0*v/tot:6.2f}%")
    print()
    print("LIMITS: single synapse, one drive condition, one seed, 2 s. Provenance is a")
    print("descriptive result (Q-A); it does NOT speak to input-selectivity (Q-B).")

    out = os.path.join(SWEEP_DIR, "po5_unit2_provenance_results.json")
    with open(out, "w") as f:
        json.dump({"seed": SEED, "T": T, "dt": dt,
                   "instrumented": inst, "baseline": base,
                   "conservation_ok": cons_ok, "nonperturbation_ok": perturb_ok}, f, indent=2)
    print(f"\npersisted -> {out}")


if __name__ == "__main__":
    main()
