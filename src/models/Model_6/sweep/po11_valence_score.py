"""
PO-11 valence scorer — the REGISTERED readout (within-condition partial correlation) plus the
magnitude reference (the PO-10 leak), reading Unit C / valence jsonl (per-draw dw_cluster, order, mode).
Readouts are the ones validated on synthetic data in po11_valence_synthetic_check.py.

Decisive real-data check (no physics): on the `full` arm the within-condition partial correlation
should RECOVER the pairing above a label-shuffle null; on the `scramble` arm it should be at chance
while the magnitude reference still separates (the leak). Reproduces P2 on real data.

Usage:  python po11_valence_score.py --glob '<abs>/results/po10_unitC/*ucB_full*'  --label full
"""
import json, glob, argparse
import numpy as np

CLUS = ['A', 'B', 'C', 'D']
PAIRINGS = {'pair1': [(0, 1), (2, 3)], 'pair2': [(0, 2), (1, 3)]}
CROSS = {'pair1': [(0, 2), (0, 3), (1, 2), (1, 3)],   # the four cross-pair edges (complement of within)
         'pair2': [(0, 1), (2, 3), (0, 3), (1, 2)]}


def load(globpat):
    draws = []
    for fp in glob.glob(globpat):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                dw = np.array([r['dw_cluster'][c] for c in CLUS], float)
                draws.append((r['mode'], r['order'], dw))
    return draws


def partial_corr(X):
    Xc = X - X.mean(0, keepdims=True)
    if np.any(Xc.std(0) < 1e-9):
        C = np.corrcoef(Xc, rowvar=False)
        return np.nan_to_num(C)
    C = np.corrcoef(Xc, rowvar=False) + 1e-6 * np.eye(4)
    P = np.linalg.inv(C)
    d = np.sqrt(np.diag(P))
    PC = -P / np.outer(d, d)
    np.fill_diagonal(PC, 1.0)
    return PC


def within_cond_pcorr(draws):
    mats = []
    for o in sorted(set(d[1] for d in draws)):
        X = np.array([d[2] for d in draws if d[1] == o])
        if len(X) >= 4:
            mats.append(partial_corr(X))
    return np.mean(mats, axis=0) if mats else np.eye(4)


def mag_agreement(draws):                                   # PO-10 magnitude leak, pooled
    X = np.abs(np.array([d[2] for d in draws]))
    hi = (X >= np.median(X, 0, keepdims=True)).astype(float)
    return np.array([[np.mean(hi[:, i] == hi[:, j]) for j in range(4)] for i in range(4)])


def separation(M, pairing):
    within = np.mean([M[i, j] for i, j in PAIRINGS[pairing]])
    cross = np.mean([M[i, j] for i, j in CROSS[pairing]])
    return within - cross


def decode(M):
    s = {p: np.mean([M[i, j] for i, j in PAIRINGS[p]]) for p in PAIRINGS}
    return 'pair1' if s['pair1'] >= s['pair2'] else 'pair2'


def score(draws, readout):
    """Real separation (true within-pair minus cross-pair) averaged over the two pairings,
    each read within-condition; and decode-correct flags."""
    seps, correct = [], []
    for p in PAIRINGS:
        sub = [d for d in draws if d[0] == p]
        if len(sub) < 8:
            continue
        M = readout(sub)
        seps.append(separation(M, p))
        correct.append(decode(M) == p)
    return float(np.mean(seps)) if seps else 0.0, correct


def null_p(draws, readout, real_sep, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    modes = np.array([d[0] for d in draws])
    ge = 0
    for _ in range(n):
        perm = rng.permutation(len(draws))
        shuffled = [(modes[perm[k]], draws[k][1], draws[k][2]) for k in range(len(draws))]
        s, _ = score(shuffled, readout)
        ge += (s >= real_sep)
    return (ge + 1) / (n + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    draws = load(a.glob)
    n_by = {}
    for m, o, _ in draws:
        n_by[(m, o)] = n_by.get((m, o), 0) + 1
    print(f"[{a.label}] {len(draws)} draws  cells={dict(sorted(n_by.items()))}")
    for name, ro in [("within-cond partial-corr (REGISTERED)", within_cond_pcorr),
                     ("magnitude (leak reference)", mag_agreement)]:
        sep, correct = score(draws, ro)
        p = null_p(draws, ro, sep, seed=1)
        flag = "RECOVERS" if (p < 0.05 and all(correct)) else ("chance" if p >= 0.05 else "partial")
        print(f"   {name:40s} sep={sep:+.3f}  decode={correct}  null-p={p:.3f}  -> {flag}")
