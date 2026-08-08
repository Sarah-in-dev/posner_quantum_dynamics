#!/usr/bin/env python3
"""
PO-10 covariance-route readout — does a DIRECTIONAL weight readout recover the input pairing WITHOUT the
abundance leak?

Context (PO10_ADVISOR_UPDATE_UNIT_C, Addendum 2): the entanglement partition reaches the weights, but
- the SIGN-AGREEMENT decoder (sign(Δw_a)·sign(Δw_b)) is binding-specific but MODEST (~0.75 = readout noise);
- the MAGNITUDE decoder (|Δw|) hits ceiling but LEAKS — 1.000 on `full` AND `scramble`/`lamshort`, because
  |Δw_cluster| ∝ √(committed-dimer count) = drive-timing ABUNDANCE, which needs no binding.
Open question: is there a readout that recovers domain co-membership from a directional (magnitude-weighted)
signal WITHOUT the abundance leak? The advisor's proposal: a COVARIANCE-ACROSS-TRIALS statistic — co-membered
clusters co-vary in Δw trial-to-trial (shared collapse coin); abundance lives in the per-cluster MEAN and is
centered out. This decides whether ~0.75 is the fundamental ceiling (sign-invariant correlation = the honest
output; "quantum constrains, classical computes") or REMOVABLE (a genuine directional computation).

THE COVARIANCE READOUT (this file): z-score each cluster's Δw across trials — z_a = (Δw_a − μ_a)/σ_a. This
removes the abundance MEAN (μ) and SCALE (σ), so it is abundance-free by construction. The per-trial product
z_a·z_b is the per-trial contribution to the CORRELATION between clusters; its SIGN co-fluctuation requires
the shared coin (binding), so it is chance under scramble/lamshort/bindoff. Decode pair1 vs pair2 from the
6 pairwise z-products (LOO-CV vs shuffle null), per arm — directly comparable to the 0.75 sign decoder.

Also computes the two reference decoders (sign-agreement; raw-magnitude) so the leak/ceiling are visible,
and a population correlation-structure test. Pure post-hoc on existing data — NO model runs.
"""
import sys, os, json, glob
import numpy as np

D = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/trusting-heyrovsky-1338e9/results/po10_unitC"
CLUS = ["A", "B", "C", "D"]
PAIRS = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
WITHIN = {"pair1": {("A", "B"), ("C", "D")}, "pair2": {("A", "C"), ("B", "D")}}
ARMS = ["full", "bindoff", "scramble", "lamshort"]


def load(arm):
    """Return dw [n,4] signed per-cluster Δw, y [n] (0=pair1,1=pair2), ignited-only."""
    dw, y = [], []
    for mode, lab in [("pair1", 0), ("pair2", 1)]:
        for f in sorted(glob.glob(f"{D}/po10_unitC_ucB_{arm}_{mode}*_w*.jsonl")):   # the keystone dataset (registered scorer)
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("ignited", False):
                    continue
                dc = r["dw_cluster"]
                dw.append([float(dc[c]) for c in CLUS]); y.append(lab)
    return np.array(dw, float), np.array(y, int)


def loo(X, y):
    """Leave-one-out nearest-centroid decode accuracy (matches the registered scorer)."""
    n = len(y)
    if n < 2 or len(set(y)) < 2:
        return float("nan")
    ok = 0
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        c0 = X[m][y[m] == 0].mean(0); c1 = X[m][y[m] == 1].mean(0)
        ok += (0 if np.linalg.norm(X[i]-c0) <= np.linalg.norm(X[i]-c1) else 1) == y[i]
    return ok / n


def null_p95(X, y, T=2000):
    rng = np.random.default_rng(0)
    a = [loo(X, rng.permutation(y)) for _ in range(T)]
    a = np.array([v for v in a if not np.isnan(v)])
    return float(a.mean()), float(np.percentile(a, 95))


def features(dw, y, kind):
    idx = {c: i for i, c in enumerate(CLUS)}
    if kind == "sign":                       # sign-agreement (binding-specific, ~0.75 reference)
        s = np.sign(dw)
        return np.array([[s[t, idx[a]]*s[t, idx[b]] for a, b in PAIRS] for t in range(len(dw))])
    if kind == "magnitude":                  # raw |Δw| per cluster (the abundance-leak reference)
        return np.abs(dw)
    if kind == "zproduct":                   # naive covariance route (per-cluster z) — LEAKS via common mode
        mu, sd = dw.mean(0), dw.std(0) + 1e-9
        z = (dw - mu) / sd
        return np.array([[z[t, idx[a]]*z[t, idx[b]] for a, b in PAIRS] for t in range(len(dw))])
    if kind == "zproduct_cm":                # THE FIX: remove the per-trial COMMON MODE (whole-trial abundance)
        dw_cm = dw - dw.mean(1, keepdims=True)   # subtract each trial's mean-across-clusters
        mu, sd = dw_cm.mean(0), dw_cm.std(0) + 1e-9
        z = (dw_cm - mu) / sd                    # then per-cluster z-score ⇒ common-mode + abundance both gone
        return np.array([[z[t, idx[a]]*z[t, idx[b]] for a, b in PAIRS] for t in range(len(dw))])
    raise ValueError(kind)


def corr_structure_test(dw, y, T=2000):
    """Population test: do co-membered clusters CORRELATE across trials, matching the pairing? Returns
    (S_true, p). S = mean(within-pair corr) − mean(cross-pair corr), summed over the two modes."""
    dw = dw - dw.mean(1, keepdims=True)      # remove the per-trial common mode (whole-trial abundance)
    idx = {c: i for i, c in enumerate(CLUS)}
    def S(labels):
        tot = 0.0
        for mode, lab in [("pair1", 0), ("pair2", 1)]:
            sub = dw[labels == lab]
            if len(sub) < 3:
                return np.nan
            C = np.corrcoef(sub.T)
            win = [C[idx[a], idx[b]] for a, b in PAIRS if (a, b) in WITHIN[mode]]
            cro = [C[idx[a], idx[b]] for a, b in PAIRS if (a, b) not in WITHIN[mode]]
            tot += np.mean(win) - np.mean(cro)
        return tot / 2.0
    s_true = S(y)
    rng = np.random.default_rng(1); ge = 0; ok = 0
    for _ in range(T):
        s = S(rng.permutation(y))
        if not np.isnan(s):
            ok += 1; ge += (s >= s_true)
    return s_true, (ge / ok if ok else float("nan"))


def main():
    kinds = [("sign", "sign(0.75)"), ("magnitude", "magn(leak)"), ("zproduct", "zprod(naive)"), ("zproduct_cm", "ZPROD_CM")]
    print(f"{'arm':10s}{'n':>4s} | " + "".join(f"{lab:>16s}" for _, lab in kinds) + "  | corr-struct S(p)")
    print("-" * 104)
    for arm in ARMS:
        dw, y = load(arm)
        if len(y) < 4:
            print(f"{arm:10s}{len(y):4d} | (insufficient)"); continue
        row = f"{arm:10s}{len(y):4d} | "
        for kind, _ in kinds:
            X = features(dw, y, kind)
            acc = loo(X, y); _, p95 = null_p95(X, y)
            row += f"{acc:.3f}/{p95:.3f}{'DEC' if acc > p95 else 'chn':>3s}".rjust(16)
        s, p = corr_structure_test(dw, y)
        row += f"  | S={s:+.3f} p={p:.3f}"
        print(row)
    print("\nlegend: decode/null-p95; DEC=>p95, chn=chance. ZPROD_CM = covariance route with the common-mode removed.")
    print("WIN if: ZPROD_CM DECODES on `full` (ideally > 0.75 sign) AND is CHANCE on bindoff/scramble/lamshort;")
    print("        corr-struct S>0,p<0.05 on full ONLY. (naive zprod leaks via whole-trial common-mode abundance.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
