"""
PO-11 Valence — precondition P2: validate the leak-immune readout on SYNTHETIC Δw
BEFORE any physics is spent (the L·PO5-3 lesson: a scoring bug destroyed 58 min of physics).

Question P2 must settle: with a reward-FIXED sign (all committed domains → +), does a
covariance/partial-correlation readout actually dodge the PO-10 magnitude leak — or does
it leak like magnitude, because with a fixed sign |Δw| and signed Δw look alike?

Synthetic generator (mirrors the model's structure, no physics):
  dw[c] = commit[domain(c)] * (+1) * (abundance[c] + noise)
  - commit: one Bernoulli coin PER DOMAIN, SHARED by co-membered clusters (the partition signal).
  - abundance: per-cluster drive level, CONSTANT within a drive condition, CORRELATED with the
    true pairing, and DIFFERENT across the two counterbalanced conditions (the leak source —
    this is drive timing, exactly what made magnitude leak in PO-10).
  scramble: reassign domain membership at random (destroys the partition) but LEAVE abundance
    (drive) intact — so a leaky readout still decodes the true pairing via abundance; an immune
    one goes to chance.

Candidate readouts:
  R1  partial correlation, POOLED across both drive conditions        (prereg's naive form)
  R2  partial correlation, WITHIN each drive condition, then averaged (abundance constant within)
  R3  commit co-occurrence (binary Δw≠0), within condition, averaged  (reads pure commit sharing)
  Rmag magnitude agreement, pooled (the PO-10 leak reference)

PASS if a readout decodes `full` above chance AND is at chance on `scramble`.
Chance = 0.5 (two pairings).
"""
import numpy as np

PAIRINGS = {1: [(0, 1), (2, 3)], 2: [(0, 2), (1, 3)]}   # A,B,C,D = 0,1,2,3


def domains_of(pairing):
    dom = {}
    for d, (i, j) in enumerate(PAIRINGS[pairing]):
        dom[i] = d; dom[j] = d
    return dom


def gen_experiment(true_pairing, n_per_cond, rng, scramble=False, p_commit=0.7, noise=0.12):
    dom_true = domains_of(true_pairing)
    rows, cond_id = [], []
    for cond in range(2):                                    # two counterbalanced drive conditions
        ab = np.ones(4)
        for (i, j) in PAIRINGS[true_pairing]:               # abundance correlated w/ true pairing,
            lvl = 1.0 + (0.9 if cond == 0 else 0.3) * rng.random()   # and different across conditions
            ab[i] = lvl; ab[j] = lvl
        commit_dom = dict(dom_true)
        if scramble:                                        # destroy partition, keep abundance
            perm = rng.permutation(4)
            commit_dom = {perm[0]: 0, perm[1]: 0, perm[2]: 1, perm[3]: 1}
        for _ in range(n_per_cond):
            c_commit = rng.random(2) < p_commit             # one coin per domain (shared)
            row = np.array([c_commit[commit_dom[c]] * (ab[c] + noise * rng.standard_normal())
                            for c in range(4)])
            rows.append(row); cond_id.append(cond)
    return np.array(rows), np.array(cond_id)


def _partial_corr(X):
    Xc = X - X.mean(0, keepdims=True)
    if np.allclose(Xc.std(0), 0):
        return np.eye(4)
    C = np.corrcoef(Xc, rowvar=False) + 1e-6 * np.eye(4)
    P = np.linalg.inv(C)
    d = np.sqrt(np.diag(P))
    PC = -P / np.outer(d, d)
    np.fill_diagonal(PC, 1.0)
    return PC


def _decode(scoremat):
    s1 = np.mean([scoremat[i, j] for i, j in PAIRINGS[1]])
    s2 = np.mean([scoremat[i, j] for i, j in PAIRINGS[2]])
    return 1 if s1 >= s2 else 2


def r1_pooled_pcorr(X, cond):
    return _decode(_partial_corr(X))


def r2_within_pcorr(X, cond):
    mats = [_partial_corr(X[cond == c]) for c in np.unique(cond)]
    return _decode(np.mean(mats, axis=0))


def r3_commit_cooccur(X, cond):
    b = (np.abs(X) > 1e-9).astype(float)
    mats = []
    for c in np.unique(cond):
        bc = b[cond == c]
        M = np.array([[np.mean(bc[:, i] * bc[:, j]) for j in range(4)] for i in range(4)])
        mats.append(M)
    return _decode(np.mean(mats, axis=0))


def rmag_pooled(X, cond):
    mag = np.abs(X)
    hi = (mag >= np.median(mag, 0, keepdims=True)).astype(float)
    M = np.array([[np.mean(hi[:, i] == hi[:, j]) for j in range(4)] for i in range(4)])
    return _decode(M)


READOUTS = {"R1 pcorr-pooled": r1_pooled_pcorr, "R2 pcorr-within": r2_within_pcorr,
            "R3 commit-cooccur": r3_commit_cooccur, "Rmag magnitude(leak ref)": rmag_pooled}


def accuracy(readout, scramble, n_exp=600, n_per_cond=15, seed=0):
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(n_exp):
        tp = int(rng.integers(1, 3))
        X, cond = gen_experiment(tp, n_per_cond, rng, scramble=scramble)
        correct += (readout(X, cond) == tp)
    return correct / n_exp


if __name__ == "__main__":
    print("PO-11 P2 synthetic validation  (chance = 0.500; n_exp=600, n=15/cond)")
    print(f"{'readout':26s} {'full':>8s} {'scramble':>10s}   verdict")
    print("-" * 60)
    for name, fn in READOUTS.items():
        a_full = accuracy(fn, scramble=False, seed=1)
        a_scr = accuracy(fn, scramble=True, seed=2)
        immune = a_full > 0.75 and abs(a_scr - 0.5) < 0.08
        leaks = a_scr > 0.65
        verdict = "LEAK-IMMUNE" if immune else ("LEAKS" if leaks else "weak/unclear")
        print(f"{name:26s} {a_full:8.3f} {a_scr:10.3f}   {verdict}")
    print("-" * 60)
    print("PASS criterion: a readout with full>0.75 AND scramble≈0.5 exists.")
