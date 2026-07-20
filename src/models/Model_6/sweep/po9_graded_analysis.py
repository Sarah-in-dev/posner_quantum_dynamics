#!/usr/bin/env python3
"""
PO-9 graded-overlap analysis — scores cross_w vs co-ignition against the pre-registered null.

PRE-REGISTERED VERDICT (PREREG_PO9 Amendment 2):
  STEP / presence detector (the NULL): cross_w ~ saturated for ANY co-ignition > ~0, then cliffs.
  GRADED / computation: cross_w rises smoothly with co-ignition duration, a resolved monotone trend
    with a width set by bond kinetics.

TESTS (both reported):
  (1) Spearman rho(cross_w, coignition_s) across ALL draws (offsets 10..30 + SYNC + STAGGER).
      Graded => rho strongly positive & significant.
  (2) Step-rejection: is cross_w at INTERMEDIATE co-ignition (the offset=20 cell, ~3-4 s) significantly
      BELOW the saturated value (offset=10 cell, ~11 s)? Welch t. If yes, the step null is rejected.
  (3) Per-offset mean +/- SD of cross_w and co-ignition, for the figure.

Readout delay fixed at 20 s (structure alive, before the ~57 s Werner-floor crossing). NO refitting.
Usage: python sweep/po9_graded_analysis.py
"""
import os, sys, glob, json
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))), "results", "po9_unitB")
DELAY = 20.0


def load(tag_glob):
    rows = []
    for f in glob.glob(os.path.join(D, tag_glob)):
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    return rows


def cw_at(r, delay=DELAY):
    return next(s["cross_w"] for s in r["snapshots"] if s["delay"] == delay)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/denom) if denom > 0 else 0.0


def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va/len(a) + vb/len(b))
    if se == 0: return float("nan"), float("nan")
    t = (a.mean() - b.mean())/se
    df = (va/len(a)+vb/len(b))**2 / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    return float(t), float(df)


def main():
    cells = {0: load("po9_unitB_sy214_w*.jsonl")}     # SYNC == offset 0, full overlap
    for o in (10, 15, 20, 25, 30):
        cells[o] = load(f"po9_unitB_grade_o{o}_w*.jsonl")
    cells[40] = load("po9_unitB_st214_w*.jsonl")       # STAGGER == offset 40, zero overlap

    print(f"cross_w vs co-ignition @ delay={DELAY:.0f}s  (dir: {D})")
    print(" offset | n | co-ign mean(s) | cross_w mean+/-sd | cross_w>0")
    allx, ally = [], []
    for o in sorted(cells):
        rows = [r for r in cells[o] if r.get("ignited", True)]
        if not rows:
            print(f"  {o:4d}  | 0 | (no draws yet)"); continue
        ci = [r.get("coignition_s", 25.0 if o == 0 else 0.0) for r in rows]
        cw = [cw_at(r) for r in rows]
        allx += list(ci); ally += list(cw)
        print(f"  {o:4d}  | {len(cw)} | {np.mean(ci):6.1f}         | {np.mean(cw):6.0f} +/- {np.std(cw):4.0f}   | {sum(1 for x in cw if x>0)}/{len(cw)}")

    if len(allx) >= 6:
        rho = spearman(allx, ally)
        print(f"\n(1) Spearman rho(co-ignition, cross_w) across {len(allx)} draws = {rho:+.3f}")
        print("    graded => strongly positive; step => ~flat above 0 then cliff (weaker monotone).")
        mid = [cw_at(r) for r in cells.get(20, [])]     # ~3-4 s co-ignition
        hi = [cw_at(r) for r in cells.get(10, [])]      # ~11 s co-ignition (near-saturated)
        if len(mid) >= 2 and len(hi) >= 2:
            t, df = welch_t(hi, mid)
            print(f"(2) step-rejection: cross_w(hi co-ign, o10) {np.mean(hi):.0f} vs (mid, o20) {np.mean(mid):.0f}; "
                  f"Welch t={t:.2f} df={df:.1f}")
            print("    t>~2 => intermediate is significantly BELOW saturated => STEP NULL REJECTED (graded).")
        print("\nVERDICT: GRADED (computation) if rho strongly + AND step rejected; else STEP (detector).")
    else:
        print("\n(insufficient draws for the verdict yet)")


if __name__ == "__main__":
    main()
