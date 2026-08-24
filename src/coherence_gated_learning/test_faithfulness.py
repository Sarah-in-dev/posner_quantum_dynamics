"""Is the abstract primitive FAITHFUL to the Model 6 results it was extracted from?

Reproduces, with no physics at all, the three findings the biophysics established. This is the acceptance
gate for the extraction: if the abstract learner cannot show these, it is not the same primitive.

  A. DELAY CURVE (Model 6 F3-f) — credit survives long delays and degrades as the trace passes its readable
     horizon. Model 6: readable until P_S crosses the Werner floor at ~107 s (T_singlet=216 s). With the same
     floor/threshold the abstract horizon is tau*ln((1-floor)/(thresh-floor)) -- identical by construction at
     tau=216, which is the point: the horizon is a decay constant, not a quantum property.
  B. REWARD NECESSITY (Model 6 F4-a no-reward arm: 0 commitments in 32 slots) — activity alone never commits.
  C. SPECIFICITY (Model 6 F4-a: precision 1.000, permutation p=0.0003) — ONE global scalar reward, broadcast
     identically, still commits only the units that were driven.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cgl import CoherenceGatedLearner, CGLParams

FLOOR, THRESH, TAU = 0.25, 0.707, 216.0
horizon = TAU * np.log((1 - FLOOR) / (THRESH - FLOOR))

def mk(seed, p_forms=1.0):
    return CoherenceGatedLearner(8, CGLParams(tau=TAU, trace_floor=FLOOR, readable_threshold=THRESH,
                                              p_trace_forms=p_forms, seed=seed))

print("=" * 88)
print("FAITHFULNESS OF THE ABSTRACT PRIMITIVE TO THE MODEL 6 BIOPHYSICS")
print(f"readable horizon = tau*ln((1-floor)/(thresh-floor)) = {horizon:.1f} time units "
      f"(Model 6 measured ~107 s)")
print("=" * 88)

t0 = time.time()

# --- A. delay curve -------------------------------------------------------------------------------
print("\nA. DELAY CURVE — commit rate vs delay between activity and reward")
print(f"   {'delay':>7} {'commit rate':>12}   (Model 6 F3-f: credits 5-30 s, degrades by 60 s)")
delays = [1, 2, 5, 10, 30, 60, 100, 150, 200]
rateA = {}
for d in delays:
    hits = []
    for seed in range(200):
        L = mk(seed)
        L.activate([True] + [False] * 7)
        L.decay(d)
        hits.append(bool(L.reward(1.0)[0]))
    rateA[d] = float(np.mean(hits))
    mark = "" if d < horizon else "   <- past readable horizon"
    print(f"   {d:>7} {rateA[d]:>12.2f}{mark}")

# --- B. reward necessity --------------------------------------------------------------------------
print("\nB. REWARD NECESSITY — identical activity, no reward delivered")
commits_noreward = 0
for seed in range(200):
    L = mk(seed)
    L.activate([True] * 4 + [False] * 4)
    L.decay(10)
    commits_noreward += int(L.reward(0.0).sum())
print(f"   commitments across 200 trials x 8 units with r=0 : {commits_noreward}   "
      f"(Model 6 F4-a: 0 across 32 slots)")

# --- C. specificity -------------------------------------------------------------------------------
print("\nC. SPECIFICITY — ONE global scalar reward, broadcast to every unit")
driven_hits, undriven_hits, nd, nu = 0, 0, 0, 0
rng = np.random.default_rng(0)
for seed in range(300):
    L = mk(seed)
    drv = np.zeros(8, dtype=bool)
    drv[rng.choice(8, size=3, replace=False)] = True
    L.activate(drv)
    L.decay(10)
    fired = L.reward(1.0)
    driven_hits += int((fired & drv).sum()); nd += int(drv.sum())
    undriven_hits += int((fired & ~drv).sum()); nu += int((~drv).sum())
prec = driven_hits / (driven_hits + undriven_hits) if (driven_hits + undriven_hits) else float("nan")
print(f"   driven   commit rate {driven_hits/nd:.3f}  (n={nd})")
print(f"   undriven commit rate {undriven_hits/nu:.3f}  (n={nu})")
print(f"   precision (committers that were driven) = {prec:.3f}   (Model 6 F4-a: 1.000)")

elapsed = time.time() - t0

# --- acceptance -----------------------------------------------------------------------------------
print("\n" + "=" * 88)
checks = [
    ("A  credit survives well past a short trace (delay 30 >= 0.9)",      rateA[30] >= 0.9),
    ("A  credit degrades past the readable horizon (delay 150 <= 0.05)",  rateA[150] <= 0.05),
    ("A  horizon matches Model 6's ~107 s to within 5%",                  abs(horizon - 107.0) / 107.0 < 0.05),
    ("B  reward is NECESSARY (zero commitments without reward)",          commits_noreward == 0),
    ("C  credit is SPECIFIC (precision == 1.000)",                        prec == 1.0),
    ("C  undriven units never commit on a global reward",                 undriven_hits == 0),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
allok = all(o for _, ok in [(n, o) for n, o in checks] for o in [ok])
print(f"\n  ALL: {'PASS' if all(o for _, o in checks) else 'FAIL'}")
print(f"  runtime for {200*len(delays) + 200 + 300} learning trials: {elapsed:.3f} s")
print("  (the Model 6 biophysics needed ~35 h for ONE 6-synapse trial)")
print("=" * 88)
