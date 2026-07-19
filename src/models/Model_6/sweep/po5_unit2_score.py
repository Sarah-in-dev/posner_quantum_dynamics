#!/usr/bin/env python3
"""
PO-5 UNIT 2 · Q-B — the OFFLINE SCORER.

Composed from PO-3's pattern (sweep/score_leta5.py): scoring is a SEPARATE step that reads
a persisted trace, so a scoring bug can never again destroy physics. Binding rule from MO
ruling 028: a heavy-slot run persists its scored intermediate; scoring is offline.

WHAT L·PO5-3 GOT WRONG, and what this fixes
-------------------------------------------
The first Q-B scorer indexed cells by EACH RUN'S OWN occupied set, so index i denoted a
different physical location in every run. A Frobenius distance between such matrices is
meaningless EVEN WHEN THE SHAPES MATCH -- the crash was the lucky failure mode.

Here every cell is keyed by ABSOLUTE lattice coordinates (floor(x/CELL_NM), floor(y/CELL_NM)),
so a cell is the same physical place in every run, and comparison is restricted to the cell
set occupied at >= MIN_OCC in ALL runs. Index i means one location, everywhere.

VALIDATION GATE (MO ruling 028)
-------------------------------
The scorer must separate a KNOWN PLANTED pair structure from a KNOWN FLAT one. A scorer that
runs without crashing but cannot tell a signal from its absence is NOT validated. If the
planted case does not score CONFIRMED and the flat case FALSIFIED, this module aborts and no
real data may be scored with it.
"""

import sys, os, json
import numpy as np

# ---- registered thresholds. Unchanged since PREREG §5 / A2.2. NOT movable. ----
RATIO_CONFIRM = 3.0
RATIO_FALSIFY = 1.5
MIN_OCC = 5
MIN_CELLS = 10
SUBSET_MIN_BONDS = 1000
CELL_NM = 6.0
SUBSETS = ["ALL", "P0_birth_inherit", "P1_burst", "P2_em"]


# ===========================================================================
# Core scoring
# ===========================================================================
def common_cells(runs):
    """Cells occupied at >= MIN_OCC in EVERY run. Absolute coords, so comparable."""
    sets = []
    for r in runs:
        sets.append({c for c, n in r["cells"].items() if n >= MIN_OCC})
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def build_matrix(run, cells, subset):
    """P_bond over the FIXED cell list. Index i is the same absolute cell in every run."""
    idx = {c: i for i, c in enumerate(cells)}
    K = len(cells)
    counts = np.zeros((K, K))
    for key, n in run["pairs"].get(subset, {}).items():
        a, b = key.split("|")
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        counts[ia, ib] += n
        if ia != ib:
            counts[ib, ia] += n
    occ = np.array([run["cells"][c] for c in cells], dtype=float)
    avail = np.outer(occ, occ).astype(float)
    np.fill_diagonal(avail, occ * (occ - 1) / 2.0)
    avail[avail <= 0] = np.nan
    return counts / avail


def separations(cells):
    pts = np.array([[float(c.split(",")[0].strip("()")) + 0.5,
                     float(c.split(",")[1].strip("()")) + 0.5] for c in cells]) * CELL_NM
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)


def residual(P, sep):
    """R = P - f_hat(|a-b|); f_hat binned, fitted PER RUN on that run's own data (PREREG §3).

    `g` is geometry not input (Unit 1: D=33.5), so distance dependence is regressed out
    before anything is attributed to input.
    """
    R = np.full_like(P, np.nan)
    finite = np.isfinite(P)
    if finite.sum() < 3:
        return R
    edges = np.unique(np.quantile(sep[finite], np.linspace(0, 1, 9)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = finite & (sep >= lo) & (sep <= hi)
        if m.sum() >= 2:
            R[m] = P[m] - np.nanmean(P[m])
    return R


def _flat(R):
    v = R[np.isfinite(R)]
    return v


def _dist(Ra, Rb):
    """Frobenius distance over entries finite in BOTH — same absolute cells by construction."""
    m = np.isfinite(Ra) & np.isfinite(Rb)
    if m.sum() == 0:
        return np.nan
    return float(np.linalg.norm(Ra[m] - Rb[m]))


def _mean_within(mats):
    ds = [_dist(mats[i], mats[j]) for i in range(len(mats)) for j in range(i + 1, len(mats))]
    ds = [d for d in ds if np.isfinite(d)]
    return float(np.mean(ds)) if ds else np.nan


def _mean_between(A, B):
    ds = [_dist(a, b) for a in A for b in B]
    ds = [d for d in ds if np.isfinite(d)]
    return float(np.mean(ds)) if ds else np.nan


def classify(ratio, n_cells, drive_ok, instrument_ok):
    if not instrument_ok:
        return "INSTRUMENT_INVALID"
    if not drive_ok or n_cells < MIN_CELLS or ratio is None or not np.isfinite(ratio):
        return "INCONCLUSIVE"
    if ratio >= RATIO_CONFIRM:
        return "CONFIRMED"
    if ratio <= RATIO_FALSIFY:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def score(runs, drive_ok=True, instrument_ok=True, verbose=True):
    cells = common_cells(runs)
    K = len(cells)
    sep = separations(cells) if K else None
    out = {"n_common_cells": K, "verdicts": {}, "detail": {}}
    if verbose:
        print(f"  common cells occupied >= MIN_OCC({MIN_OCC}) in ALL runs: {K} "
              f"(need >= {MIN_CELLS})")
    if K < MIN_CELLS:
        for s in SUBSETS:
            out["verdicts"][s] = "INCONCLUSIVE"
        out["note"] = (f"all-run intersection {K} < MIN_CELLS {MIN_CELLS}: the instrument "
                       f"cannot resolve pair structure in this geometry. Threshold NOT moved.")
        if verbose:
            print(f"  -> {out['note']}")
        return out

    for subset in SUBSETS:
        nb = np.mean([sum(r["pairs"].get(subset, {}).values()) for r in runs])
        if subset != "ALL" and nb < SUBSET_MIN_BONDS:
            out["verdicts"][subset] = "INSUFFICIENT"
            out["detail"][subset] = {"mean_bonds": float(nb)}
            if verbose:
                print(f"  {subset:20s} INSUFFICIENT (mean {nb:.0f} bonds < {SUBSET_MIN_BONDS})")
            continue
        R = {lbl: [residual(build_matrix(r, cells, subset), sep)
                   for r in runs if r["label"] == lbl] for lbl in ("A", "B", "NULL")}
        d_null = _mean_within(R["A"] + R["NULL"])
        d_in = _mean_between(R["A"], R["B"])
        ratio = (d_in / d_null) if (np.isfinite(d_in) and np.isfinite(d_null) and d_null > 0) else None
        v = classify(ratio, K, drive_ok, instrument_ok)
        out["verdicts"][subset] = v
        out["detail"][subset] = {"d_input": d_in, "d_null": d_null,
                                 "ratio": ratio, "mean_bonds": float(nb)}
        if verbose:
            rt = f"{ratio:.3f}" if ratio is not None else "n/a"
            tag = "PRIMARY" if subset == "ALL" else "secondary"
            print(f"  {subset:20s} d_input={d_in:.6f} d_null={d_null:.6f} "
                  f"ratio={rt} -> {v}  [{tag}]")
    return out


# ===========================================================================
# MO ruling 028: the scorer must SEPARATE planted structure from flat
# ===========================================================================
def _synth(planted, seed=0, K=16, n_runs=3, amp=2.0):
    """Build synthetic runs on a fixed lattice.

    flat    : every arm's bond probability drawn from one distribution -> no input effect
    planted : arm B has a KNOWN elevated bond probability on a specific CELL PAIR that
              arm A and NULL do not have -- a pair-level signal at a known location.
    """
    rng = np.random.default_rng(seed)
    cells = [f"({i},{j})" for i in range(4) for j in range(4)][:K]
    runs = []
    labels = ["A"] * n_runs + ["B"] * n_runs + ["NULL"] * n_runs
    for n, lbl in enumerate(labels):
        occ = {c: 40 for c in cells}
        pairs = {}
        for i, ca in enumerate(cells):
            for cb in cells[i:]:
                base = 0.30 + 0.02 * rng.normal()
                if planted and lbl == "B" and ca in cells[:2] and cb in cells[:2]:
                    base += amp            # the planted pair-level structure
                navail = occ[ca] * occ[cb] if ca != cb else occ[ca] * (occ[ca] - 1) / 2
                pairs[f"{ca}|{cb}"] = max(0.0, base) * navail
        runs.append({"label": lbl, "arm": lbl, "seed": 900 + n,
                     "cells": occ, "pairs": {"ALL": pairs}})
    return runs


def detection_floor():
    """Sweep planted amplitude to find where the scorer crosses RATIO_CONFIRM.

    This is NOT calibration -- no threshold moves. It characterises SENSITIVITY, so a
    later FALSIFIED can be reported as a bounded statement ("no effect larger than X")
    instead of a bare negative.
    """
    print("\nDETECTION FLOOR — planted amplitude vs scored ratio (thresholds fixed)")
    print(f"  {'amp':>6s} {'ratio':>8s}  verdict")
    floor = None
    for amp in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
        r = score(_synth(planted=True, seed=1, amp=amp), verbose=False)
        ratio = r["detail"]["ALL"]["ratio"]
        v = r["verdicts"]["ALL"]
        if floor is None and v == "CONFIRMED":
            floor = amp
        print(f"  {amp:6.2f} {ratio:8.3f}  {v}")
    print(f"  => smallest planted amplitude reaching CONFIRMED: "
          f"{floor if floor is not None else 'none in range'}")
    return floor


def validate_scorer():
    print("=" * 78)
    print("SCORER VALIDATION (MO ruling 028) — must SEPARATE planted structure from flat")
    print("=" * 78)
    print("\n[case 1] KNOWN FLAT — no arm differs; required verdict FALSIFIED")
    flat = score(_synth(planted=False, seed=1), verbose=True)
    v_flat = flat["verdicts"]["ALL"]
    print(f"  => {v_flat}")

    print("\n[case 2] KNOWN PLANTED pair structure in arm B; required verdict CONFIRMED")
    plant = score(_synth(planted=True, seed=1), verbose=True)
    v_plant = plant["verdicts"]["ALL"]
    print(f"  => {v_plant}")

    ok = (v_flat == "FALSIFIED") and (v_plant == "CONFIRMED")
    print("\n" + "-" * 78)
    print(f"  flat    -> {v_flat:12s} (required FALSIFIED)")
    print(f"  planted -> {v_plant:12s} (required CONFIRMED)")
    print(f"  ratio separation: flat {flat['detail']['ALL']['ratio']:.3f} "
          f"vs planted {plant['detail']['ALL']['ratio']:.3f}")
    if not ok:
        print("\nABORT: the scorer cannot separate a signal from its absence. NOT VALIDATED —")
        print("no real data may be scored with it. (MO ruling 028)")
        sys.exit(1)
    print("\nVALIDATED: the scorer distinguishes planted pair structure from flat.")
    detection_floor()
    print("=" * 78)
    return True


def main():
    validate_scorer()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            data = json.load(f)
        runs = data["runs"]
        print(f"\nScoring real trace: {path}  ({len(runs)} runs)")
        res = score(runs, drive_ok=data.get("drive_ok", True),
                    instrument_ok=data.get("instrument_ok", True))
        print(f"\nVERDICT (PRIMARY, whole realised bond set): {res['verdicts'].get('ALL')}")
        print("A2.2 precedence: the whole-set verdict stands regardless of any sub-set result.")
        out = path.replace(".json", "_scored.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"persisted -> {out}")
    else:
        print("\nNo trace supplied — validation only. Pass a results JSON to score it.")


if __name__ == "__main__":
    main()
