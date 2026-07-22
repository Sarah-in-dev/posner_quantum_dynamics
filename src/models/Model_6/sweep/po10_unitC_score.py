"""
PO-10 Unit C scorer — decode trial type (pair1 vs pair2) from the SIGNED Δw readout, per arm.

The registered computation-level measure: can a linear classifier recover the input (trial type) from
the weight-change vector? Features are the SIGN-INVARIANT pairwise cluster agreement structure
(agree[a,b] = sign(Δw_a)·sign(Δw_b) ∈ {-1,0,+1}), so decoding is invariant to the arbitrary per-domain
sign flip. Reports leave-one-out CV accuracy vs a shuffled-label null, per arm.

Prediction (registered): arm 'full' near-ceiling; arms 'bindoff', 'scramble', 'lamshort' at chance.

Reads all results/po10_unitC/po10_unitC_uc_<arm>_<mode>_w*.jsonl. NO model dependence — pure post-hoc.
"""
import sys, os, json, glob
import numpy as np

D = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/trusting-heyrovsky-1338e9/results/po10_unitC"
PAIR_KEYS = ["AB", "AC", "AD", "BC", "BD", "CD"]
ARMS = ["full", "bindoff", "scramble", "lamshort"]


def load(arm):
    """Return X [n,6] agreement features, y [n] labels (0=pair1,1=pair2), plus ignited mask."""
    X, y, ig = [], [], []
    for mode, lab in [("pair1", 0), ("pair2", 1)]:
        for f in sorted(glob.glob(f"{D}/po10_unitC_ucB_{arm}_{mode}*_w*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                X.append([r["agree"].get(k, 0) for k in PAIR_KEYS])
                y.append(lab); ig.append(bool(r["ignited"]))
    return np.array(X, float), np.array(y, int), np.array(ig, bool)


def loo_accuracy(X, y):
    """Leave-one-out nearest-centroid on the agreement features (robust for small n)."""
    n = len(y)
    if n < 2 or len(set(y)) < 2:
        return float("nan")
    correct = 0
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        c0 = X[m][y[m] == 0].mean(0) if (y[m] == 0).any() else np.zeros(X.shape[1])
        c1 = X[m][y[m] == 1].mean(0) if (y[m] == 1).any() else np.zeros(X.shape[1])
        pred = 0 if np.linalg.norm(X[i] - c0) <= np.linalg.norm(X[i] - c1) else 1
        correct += (pred == y[i])
    return correct / n


def null_band(X, y, n_shuf=2000):
    accs = []
    rng = np.random.default_rng()
    for _ in range(n_shuf):
        accs.append(loo_accuracy(X, rng.permutation(y)))
    accs = np.array([a for a in accs if not np.isnan(a)])
    return float(accs.mean()), float(np.percentile(accs, 95))


def main():
    print(f"{'arm':10s} {'n':>4s} {'ign':>4s} {'decode':>8s} {'null_mu':>8s} {'null_p95':>9s}  verdict")
    for arm in ARMS:
        X, y, ig = load(arm)
        if len(y) == 0:
            print(f"{arm:10s}    -    -        -        -         -   (no data yet)")
            continue
        acc = loo_accuracy(X, y)
        nmu, np95 = null_band(X, y) if len(y) >= 4 else (float("nan"), float("nan"))
        verdict = ""
        if not np.isnan(np95):
            verdict = "DECODES (>null p95)" if acc > np95 else "chance"
        print(f"{arm:10s} {len(y):4d} {int(ig.sum()):4d} {acc:8.3f} {nmu:8.3f} {np95:9.3f}  {verdict}")
    # show the per-draw agreement patterns for eyeballing
    print("\nper-draw agreement patterns (AB AC AD BC BD CD):")
    for arm in ARMS:
        for mode in ("pair1", "pair2"):
            for f in sorted(glob.glob(f"{D}/po10_unitC_ucB_{arm}_{mode}*_w*.jsonl")):
                for line in open(f):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    vec = " ".join(f"{r['agree'].get(k,0):+d}" for k in PAIR_KEYS)
                    print(f"  {arm:9s} {mode}  [{vec}]  n_dom={r['n_domains']}  ignited={r['ignited']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
