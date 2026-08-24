"""
BENCHMARK 6 — REAL DATA, WITH TIME. The test Benchmark 5 should have been.

Benchmark 5 flattened every appointment into one row with all features simultaneous. That threw away the
entire temporal dimension -- which is the primitive's actual mechanism -- and it also withheld real signal
from every model: 66% of appointments belong to repeat patients, and the no-show rate is 35.2% after a
previous no-show versus 18.7% after a previous show. Nobody got that information, so Benchmark 5 measured
three handicapped learners.

THIS benchmark uses each patient's appointment SEQUENCE. For the primitive, a past outcome lays a TRACE that
decays with REAL ELAPSED DAYS; traces still alive when the next appointment arrives BIND with that
appointment's features, so history-x-context conjunctions ("missed recently AND booked far ahead") become
learnable components. tau is therefore in days and is a real, interpretable quantity.

FAIRNESS: the baselines are given the SAME history, hand-engineered in the standard way (previous outcome,
days since last visit, number of prior visits, prior no-show rate). So this asks a sharp question: does the
primitive's NATIVE temporal machinery match features a human had to design by hand?

Chronological split. A loss is reported as a loss.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

TAU_DAYS = 45.0          # trace decay in DAYS (the eligibility horizon, now a real-world quantity)
FLOOR, READ = 0.25, 0.707
CSV = "/tmp/noshow.csv"


def load():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    sched = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
    appt = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)
    df["lead"] = (appt - sched).dt.days.clip(lower=0)
    df["appt"] = appt
    df["age"] = df["Age"].clip(0, 100)
    df["y"] = (df["No-show"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df.sort_values("appt").reset_index(drop=True)


def context_units(r):
    """Binary context units for one appointment (the 'current' side of a conjunction)."""
    u = []
    u.append("male" if r.Gender == "M" else "female")
    a = r.age
    u.append("age" + ("0_18" if a < 18 else "18_35" if a < 35 else "35_55" if a < 55 else "55p"))
    L = r.lead
    u.append("lead" + ("0" if L < 1 else "1_3" if L < 3 else "3_8" if L < 8 else "8_15" if L < 15 else "15p"))
    u.append("dow%d" % r.appt.dayofweek)
    if r.SMS_received > 0: u.append("sms")
    if r.Scholarship > 0: u.append("scholar")
    if r.Hipertension > 0: u.append("htn")
    if r.Diabetes > 0: u.append("dm")
    if r.Alcoholism > 0: u.append("alc")
    return u


class TemporalCGL:
    """Per-patient traces decaying in DAYS; live traces bind with current context into a component."""
    def __init__(self, tau=TAU_DAYS, sharpness=4.0):
        self.tau, self.sharp = tau, sharpness
        self.hist = {}                      # pid -> list of (day, unit)
        self.store, self.index = {}, {}

    def _live_history(self, pid, day):
        out = []
        for (d0, u) in self.hist.get(pid, []):
            t = FLOOR + (1 - FLOOR) * np.exp(-(day - d0) / self.tau)
            if t > READ:                    # still readable -> it participates in binding
                out.append(u)
        return out

    def units(self, r, day):
        return context_units(r) + self._live_history(r.PatientId, day)

    def score(self, units):
        key = frozenset(units)
        if key in self.store:
            s = self.store[key]; return s[0] / s[1]
        cand = set()
        for u in key:
            cand |= self.index.get(u, set())
        if not cand: return 0.0
        num = den = 0.0
        for k in cand:
            sim = len(key & k) / len(key | k)
            w = sim ** self.sharp
            s = self.store[k]; num += w * (s[0] / s[1]); den += w
        return num / den if den else 0.0

    def observe(self, r, day, units, y):
        key = frozenset(units)
        if key not in self.store:
            self.store[key] = [0.0, 0]
            for u in key: self.index.setdefault(u, set()).add(key)
        s = self.store[key]; s[0] += (1.0 if y else -1.0); s[1] += 1
        self.hist.setdefault(r.PatientId, []).append((day, "prior_noshow" if y else "prior_show"))


def engineered(df):
    """The same history, hand-built the standard way, for the baselines."""
    df = df.copy()
    g = df.groupby("PatientId")
    df["n_prior"] = g.cumcount()
    prev = g["y"].shift()
    df["prev_noshow"] = prev.fillna(-1)
    df["prior_rate"] = g["y"].apply(lambda s: s.shift().expanding().mean()).reset_index(level=0, drop=True).fillna(-1)
    last = g["appt"].shift()
    df["days_since"] = (df["appt"] - last).dt.days.fillna(-1)
    return df


if __name__ == "__main__":
    df = load()
    day0 = df["appt"].min()
    df["day"] = (df["appt"] - day0).dt.days
    dfe = engineered(df)
    n_tr = int(len(df) * 0.7)

    print("=" * 96)
    print("BENCHMARK 6 — REAL DATA WITH TIME: per-patient appointment sequences")
    print(f"{len(df):,} appointments | chronological split {n_tr:,}/{len(df)-n_tr:,} | "
          f"trace tau = {TAU_DAYS:.0f} days")
    print("=" * 96)

    t0 = time.time()
    M = TemporalCGL()
    scores = np.zeros(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        u = M.units(r, r.appt.toordinal())
        if i >= n_tr:
            scores[i] = M.score(u)
        M.observe(r, r.appt.toordinal(), u, r.y)      # ONLINE: learn from every appointment as it happens
    t_cgl = time.time() - t0
    yte = df["y"].values[n_tr:]
    s_cgl = scores[n_tr:]

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    base_cols = ["age", "lead", "SMS_received", "Scholarship", "Hipertension", "Diabetes", "Alcoholism"]
    hist_cols = ["n_prior", "prev_noshow", "prior_rate", "days_since"]
    Xb = dfe[base_cols].values.astype(float)
    Xh = dfe[base_cols + hist_cols].values.astype(float)
    y = dfe["y"].values

    def fit_eval(X, name):
        lr = LogisticRegression(max_iter=3000).fit(X[:n_tr], y[:n_tr])
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[:n_tr], y[:n_tr])
        sl = lr.predict_proba(X[n_tr:])[:, 1]; sg = gb.predict_proba(X[n_tr:])[:, 1]
        return (roc_auc_score(y[n_tr:], sl), roc_auc_score(y[n_tr:], sg), sl, sg)

    lr_nohist, gb_nohist, _, _ = fit_eval(Xb, "no history")
    lr_hist, gb_hist, s_lrh, s_gbh = fit_eval(Xh, "with history")

    def bacc(s):
        thr = np.quantile(s, 1 - yte.mean())
        return balanced_accuracy_score(yte, (s >= thr).astype(int))

    print(f"\n  {'model':>44} {'AUC':>8} {'bal-acc':>9}")
    print("  " + "-" * 64)
    print(f"  {'coherence-gated, native time (online 1 pass)':>44} {roc_auc_score(yte, s_cgl):>8.4f} {bacc(s_cgl):>9.4f}")
    print(f"  {'logistic regression + hand-built history':>44} {lr_hist:>8.4f} {bacc(s_lrh):>9.4f}")
    print(f"  {'gradient boosting + hand-built history':>44} {gb_hist:>8.4f} {bacc(s_gbh):>9.4f}")
    print(f"  {'logistic regression, NO history':>44} {lr_nohist:>8.4f}")
    print(f"  {'gradient boosting, NO history':>44} {gb_nohist:>8.4f}")
    print(f"\n  components stored: {len(M.store):,} | cgl fit {t_cgl:.1f}s (single online pass)")
    print(f"  interaction headroom (gbm - logreg, with history): {gb_hist - lr_hist:+.4f}")
