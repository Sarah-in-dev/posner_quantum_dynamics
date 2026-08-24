"""
BENCHMARK 5 — REAL DATA. Public medical-appointment no-show dataset (110,527 Brazilian clinic appointments,
fully de-identified, Hoppen / Aquarela Analytics via Kaggle). Structurally the same problem PAUL solves:
predict whether a booked appointment is honoured, from categorical features, where the outcome is only known
later.

WHY THIS DATASET FIRST, and not PAUL's own data: it is public and carries no PHI, published baselines exist
so we cannot fool ourselves about what "good" looks like, and if the primitive finds nothing here it will
find nothing in production either -- learned without touching a patient record.

HONEST MAPPING CAVEAT, stated up front. The primitive binds units that are CO-ACTIVE IN TIME. Tabular rows
have no time dimension, so every feature of an appointment binds into one component and the learner becomes a
SET-SIMILARITY MEMORY: it stores observed feature-sets with their outcomes and answers new rows by weighted
overlap (the partial-reactivation mechanism). That is a legitimate learner and the comparison below is fair,
but it exercises the memory/retrieval half of the primitive, NOT the temporal-binding half. Do not read a
result here as validating temporal binding.

BASELINES: logistic regression (the additive model -- the real-world analogue of the scalar trace) and
gradient boosting (a strong tabular model that DOES capture interactions). Expectation set in advance:
gradient boosting will probably win on raw discrimination. The questions worth asking are whether the
primitive is COMPETITIVE while learning ONLINE IN A SINGLE PASS, and whether the conjunctions it surfaces
are real.

SPLIT: chronological (train on earlier appointments, test on later) -- the realistic setting, and harder
than a random split.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from graph_cgl import GraphCoherenceGatedLearner, GraphCGLParams

CSV = "/tmp/noshow.csv"


def build():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    sched = pd.to_datetime(df["ScheduledDay"], errors="coerce")
    appt = pd.to_datetime(df["AppointmentDay"], errors="coerce")
    lead = (appt.dt.tz_localize(None) - sched.dt.tz_localize(None)).dt.days.clip(lower=0)
    age = df["Age"].clip(lower=0, upper=100)

    feats = {}
    feats["male"] = (df["Gender"] == "M").astype(int)
    for lo, hi in [(0, 5), (5, 18), (18, 35), (35, 55), (55, 70), (70, 101)]:
        feats[f"age{lo}_{hi}"] = ((age >= lo) & (age < hi)).astype(int)
    for lo, hi in [(0, 1), (1, 3), (3, 8), (8, 15), (15, 31), (31, 200)]:
        feats[f"lead{lo}_{hi}"] = ((lead >= lo) & (lead < hi)).astype(int)
    for d in range(6):
        feats[f"dow{d}"] = (appt.dt.dayofweek == d).astype(int)
    for c in ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]:
        feats[c.lower()] = (df[c] > 0).astype(int)
    feats["handicap"] = (df["Handcap"] > 0).astype(int)
    top = df["Neighbourhood"].value_counts().head(10).index
    for n in top:
        feats["nb_" + str(n)[:12].replace(" ", "_")] = (df["Neighbourhood"] == n).astype(int)

    X = pd.DataFrame(feats).astype(np.int8)
    y = (df["No-show"].astype(str).str.strip().str.lower() == "yes").astype(int).values  # 1 = NO-SHOW
    order = np.argsort(appt.values)                     # chronological
    return X.values[order], y[order], list(X.columns)


class SetMemory:
    """The primitive applied to tabular rows: one component per row, answered by overlap (inverted index)."""
    def __init__(self, n_units, sharpness=4.0, cap=40000):
        self.p = GraphCGLParams(overlap_sharpness=sharpness)
        self.store = {}                                  # frozenset -> [sum_reward, count]
        self.index = {}                                  # unit -> set of keys
        self.cap = cap
    def observe(self, units, r):
        key = frozenset(units)
        if key not in self.store:
            if len(self.store) >= self.cap:
                return
            self.store[key] = [0.0, 0]
            for u in key:
                self.index.setdefault(u, set()).add(key)
        s = self.store[key]; s[0] += r; s[1] += 1
    def score(self, units):
        key = frozenset(units)
        if key in self.store:
            s = self.store[key]
            return s[0] / s[1]
        cand = set()
        for u in key:
            cand |= self.index.get(u, set())
        if not cand:
            return 0.0
        num = den = 0.0
        for k in cand:
            inter = len(key & k)
            sim = inter / len(key | k)
            w = sim ** self.p.overlap_sharpness
            s = self.store[k]
            num += w * (s[0] / s[1]); den += w
        return num / den if den > 0 else 0.0


def evaluate(scores, y, name):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    auc = roc_auc_score(y, scores)
    thr = np.quantile(scores, 1 - y.mean())              # match predicted positive rate to base rate
    bacc = balanced_accuracy_score(y, (scores >= thr).astype(int))
    print(f"  {name:>34}   AUC {auc:.4f}   balanced-acc {bacc:.4f}")
    return auc


if __name__ == "__main__":
    X, y, names = build()
    n_tr = int(len(y) * 0.7)
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    print("=" * 92)
    print("BENCHMARK 5 — REAL DATA: medical appointment no-shows (public, de-identified)")
    print(f"{len(y):,} appointments | {X.shape[1]} binary features | chronological split "
          f"{len(ytr):,} train / {len(yte):,} test | no-show rate {y.mean():.3f}")
    print("=" * 92)

    t0 = time.time()
    M = SetMemory(X.shape[1])
    for i in range(len(ytr)):                            # ONLINE, SINGLE PASS
        u = np.flatnonzero(Xtr[i])
        M.observe(u, +1.0 if ytr[i] == 1 else -1.0)
    s_cgl = np.array([M.score(np.flatnonzero(r)) for r in Xte])
    t_cgl = time.time() - t0

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    t1 = time.time()
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    s_lr = lr.predict_proba(Xte)[:, 1]; t_lr = time.time() - t1
    t2 = time.time()
    gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, ytr)
    s_gb = gb.predict_proba(Xte)[:, 1]; t_gb = time.time() - t2

    print()
    a_c = evaluate(s_cgl, yte, "coherence-gated (online, 1 pass)")
    a_l = evaluate(s_lr, yte, "logistic regression (additive)")
    a_g = evaluate(s_gb, yte, "gradient boosting (interactions)")
    print(f"\n  chance AUC = 0.5 | components stored = {len(M.store):,}")
    print(f"  fit time: cgl {t_cgl:.1f}s (single pass)  logreg {t_lr:.1f}s  gbm {t_gb:.1f}s")
