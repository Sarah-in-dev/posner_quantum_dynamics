#!/usr/bin/env python3
"""
PO-5 UNIT 1 — is Pathway 2's 1/r^3 factor `g` inert in practice?

Pre-registered: docs/PREREG_PO5_UNIT1_G_INERTNESS.md (committed BEFORE this run).
Charter: quantum-system-canonical §8 Keystone #1, via
         coordination/requests/po5-selectivity/mo-rescope-001.md

`dimer_particles.py:453` clamps with np.maximum, so every pair closer than
coupling_length (5.0 nm) gets g == 1.0 exactly. This probe measures the intra-synapse
r_ij distribution and the g distribution it induces, and classifies g as
INERT_BY_SATURATION / INERT_BY_VANISHING / INERT_BY_FLATNESS / LIVE.

The classifier is demonstrated discriminating all four outcomes on synthetic input
BEFORE the model is constructed. If it cannot, the probe aborts.

`g` is geometry, not input. A LIVE verdict does NOT advance the keystone; it only
means the later pair-level test is not operating on a constant.
"""

import sys, os, json
import logging
import numpy as np

logging.disable(logging.INFO)

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)

# Registered thresholds — PREREG §3. Do not move these after the run.
SAT_THRESHOLD = 0.90      # f_sat >= this  -> INERT_BY_SATURATION
VANISH_THRESHOLD = 1e-3   # g_p90 < this   -> INERT_BY_VANISHING
FLAT_THRESHOLD = 2.0      # D < this       -> INERT_BY_FLATNESS


# ---------------------------------------------------------------------------
# The measured quantities and the verdict function (PREREG §2, §3)
# ---------------------------------------------------------------------------
def g_stats(r_ij, coupling_length):
    """Compute the registered quantities from a pair-separation array.

    Uses the model's own expression (dimer_particles.py:453) and the
    coupling_length read off the live object, not a hard-coded copy.
    """
    r = np.asarray(r_ij, dtype=float)
    g = (coupling_length / np.maximum(r, coupling_length)) ** 3
    f_sat = float(np.mean(r <= coupling_length))
    g_p10, g_p50, g_p90 = (float(x) for x in np.percentile(g, [10, 50, 90]))
    D = (g_p90 / g_p10) if g_p10 > 0 else float("inf")
    return {
        "n_pairs": int(r.size),
        "f_sat": f_sat,
        "r_p10": float(np.percentile(r, 10)),
        "r_p50": float(np.percentile(r, 50)),
        "r_p90": float(np.percentile(r, 90)),
        "r_max": float(np.max(r)),
        "g_p10": g_p10, "g_p50": g_p50, "g_p90": g_p90,
        "D": D,
    }


def classify_g(f_sat, g_p10, g_p90):
    """PREREG §3. Precedence: saturation, then vanishing, then flatness, then LIVE."""
    if f_sat >= SAT_THRESHOLD:
        return "INERT_BY_SATURATION"
    if g_p90 < VANISH_THRESHOLD:
        return "INERT_BY_VANISHING"
    D = (g_p90 / g_p10) if g_p10 > 0 else float("inf")
    if D < FLAT_THRESHOLD:
        return "INERT_BY_FLATNESS"
    return "LIVE"


# ---------------------------------------------------------------------------
# PREREG §4 — the classifier must be shown discriminating all four outcomes
# ---------------------------------------------------------------------------
def demonstrate_verdict(coupling_length=5.0):
    cases = [
        ("all r = 1 nm (inside the clamp)", np.full(500, 1.0), "INERT_BY_SATURATION"),
        ("all r = 5000 nm",                 np.full(500, 5000.0), "INERT_BY_VANISHING"),
        ("all r = 20 nm exactly",           np.full(500, 20.0), "INERT_BY_FLATNESS"),
        ("r log-spaced 5 -> 100 nm",        np.logspace(np.log10(5.0), np.log10(100.0), 500), "LIVE"),
    ]
    print("=" * 78)
    print("VERDICT DEMONSTRATION (PREREG §4) — must discriminate all four before use")
    print("=" * 78)
    ok = True
    for label, r, required in cases:
        s = g_stats(r, coupling_length)
        got = classify_g(s["f_sat"], s["g_p10"], s["g_p90"])
        status = "ok" if got == required else "MISMATCH"
        if got != required:
            ok = False
        print(f"  {label:34s} f_sat={s['f_sat']:.3f} g_p90={s['g_p90']:.3e} "
              f"D={s['D']:>9.3g}  -> {got:20s} [{status}]")
    print()
    if not ok:
        print("ABORT: the classifier does not discriminate its own outcomes. "
              "No verdict on real data is admissible. (PREREG §4)")
        sys.exit(1)
    print("All four outcomes reachable and correctly labelled. Classifier admissible.")
    print()
    return True


# ---------------------------------------------------------------------------
# Measurement on the live model
# ---------------------------------------------------------------------------
def intra_pair_separations(dp, entangled_only=True):
    """All-pair separations over one synapse's dimer set.

    Mirrors dimer_particles.py:451-452 (pos[iu] - pos[ju]); positions are nm
    (dimer_particles.py:30).
    """
    dimers = [d for d in dp.dimers if d.is_entangled] if entangled_only else list(dp.dimers)
    n = len(dimers)
    if n < 2:
        return np.array([]), n
    pos = np.asarray([d.position for d in dimers], dtype=float)
    iu, ju = np.triu_indices(n, k=1)
    diff = pos[iu] - pos[ju]
    r = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    return r, n


def bond_saturation(dp):
    """Realised bond-graph saturation over the entangled set."""
    ent_ids = {d.id for d in dp.dimers if d.is_entangled}
    n = len(ent_ids)
    if n < 2:
        return 0.0, 0, n
    n_bonds = sum(1 for b in dp.entanglement_bonds
                  if b.dimer_i in ent_ids and b.dimer_j in ent_ids)
    return n_bonds / (n * (n - 1) / 2), n_bonds, n


def main():
    demonstrate_verdict()

    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    params = Model6Parameters()
    params.em_coupling_enabled = True

    T_total = 5.0
    dt = 0.005
    sample_times = [0.5, 1.0, 2.5, 5.0]

    network = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    network.initialize(Model6QuantumSynapse, params)
    for s in network.synapses:
        s.set_microtubule_invasion(True)
    stimulus = {"voltage": -10e-3, "reward": False}

    dp = network.synapses[0].dimer_particles
    L = dp.coupling_length          # read off the live object, not hard-coded

    print("=" * 78)
    print("PO-5 UNIT 1 — g-inertness on the live model")
    print("=" * 78)
    print(f"  coupling_length = {L} nm   (dimer_particles.py:129, read from the object)")
    print(f"  positions in nm (dimer_particles.py:30); grid {dp.grid_shape} @ dx_nm={dp.dx_nm}")
    print(f"  domain = {dp.grid_shape[0]*dp.dx_nm:.0f} x {dp.grid_shape[1]*dp.dx_nm:.0f} x 20 nm")
    print(f"  drive: voltage={stimulus['voltage']*1e3:.0f}mV, T={T_total}s, dt={dt}")
    print()

    hdr = (f"{'t':>5s} {'n_ent':>6s} {'n_pairs':>9s} {'f_sat':>7s} "
           f"{'r_p10':>7s} {'r_p50':>7s} {'r_p90':>7s} {'r_max':>7s} "
           f"{'g_p10':>9s} {'g_p50':>9s} {'g_p90':>9s} {'D':>7s} {'sat_bonds':>9s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    results = []
    steps = int(round(T_total / dt))
    t = 0.0
    next_i = 0

    for _ in range(steps):
        network.step(dt, stimulus)
        t += dt
        if next_i < len(sample_times) and t >= sample_times[next_i] - dt / 2:
            r, n = intra_pair_separations(dp)
            if r.size == 0:
                print(f"{t:5.1f} {n:6d}  <2 entangled dimers — no pairs to measure>")
                next_i += 1
                continue
            s = g_stats(r, L)
            # PREREG §5 positive control on the geometry: non-degenerate positions
            degenerate = not (s["g_p90"] > s["g_p10"])
            verdict = classify_g(s["f_sat"], s["g_p10"], s["g_p90"])
            sat, n_bonds, _ = bond_saturation(dp)
            row = {"t": round(t, 3), "n_entangled": n, **s,
                   "sat_bonds": sat, "n_bonds": n_bonds,
                   "verdict": verdict, "degenerate_positions": degenerate}
            results.append(row)
            flag = "  [DEGENERATE POSITIONS — verdict not physics]" if degenerate else ""
            print(f"{t:5.1f} {n:6d} {s['n_pairs']:9d} {s['f_sat']:7.4f} "
                  f"{s['r_p10']:7.2f} {s['r_p50']:7.2f} {s['r_p90']:7.2f} {s['r_max']:7.2f} "
                  f"{s['g_p10']:9.3e} {s['g_p50']:9.3e} {s['g_p90']:9.3e} {s['D']:7.2f} "
                  f"{sat:9.4f}  {verdict}{flag}")
            next_i += 1

    print()
    if results:
        labels = {r["verdict"] for r in results}
        f_sats = [r["f_sat"] for r in results]
        print("=" * 78)
        print(f"VERDICT: {'/'.join(sorted(labels))}-under-stated-conditions")
        print(f"  f_sat across samples: {min(f_sats):.4f} .. {max(f_sats):.4f} "
              f"(registered limit: report whether it moves — PREREG §6)")
        print(f"  realised bond saturation: "
              f"{min(r['sat_bonds'] for r in results):.4f} .. "
              f"{max(r['sat_bonds'] for r in results):.4f}")
        print("=" * 78)
        print("LIMITS (PREREG §6): single synapse, one drive condition, one seed. "
              "g is GEOMETRY, not input — this does not advance §8's keystone.")

    out = os.path.join(PROJECT_ROOT, "results", "po5", "unit1_g_inertness.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"coupling_length_nm": L,
                   "thresholds": {"SAT": SAT_THRESHOLD, "VANISH": VANISH_THRESHOLD,
                                  "FLAT": FLAT_THRESHOLD},
                   "samples": results}, f, indent=2)
    print(f"\npersisted -> {out}")


if __name__ == "__main__":
    main()
