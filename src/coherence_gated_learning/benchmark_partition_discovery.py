#!/usr/bin/env python3
"""
THE INERT-COIN TEST — mechanism attribution on a drifting planted partition.

WHY THIS FILE EXISTS, AND WHY IT IS NOT A FOURTH BENCHMARK ATTEMPT
==================================================================
Three prior benchmarks LOST (RESULT_nonstationary_bandit.md, RESULT_contextual_bandit.md,
RESULT_network_primitive.md). The now-understood root cause is that all three SCORED THE WRONG
OBJECT: they scored action choices, while the canonical spec says the OUTPUT is the PARTITION of
the entanglement graph. All three also used pre-carved representations handed over by a dataset --
"a room that came with a map already drawn" -- so the discovery step had already been done for us.

And the correction F2-c (2026-08-02) falsified the "network-scale partition is the object read"
claim in the grounded model: collapse fires at median 88 ms, BEFORE the cross-synapse partition
forms; 40/40 controlled draws read partition_at_measure == []. What replaced it is a HYPOTHESIS,
not a locked claim: THE SUBSTRATE ACTS LOCALLY ON COMPONENT-LEVEL DISCORDANCE AS IT APPEARS.
That undemonstrated hypothesis is what this file tests.

So this is a MECHANISM-ATTRIBUTION experiment, not a quality claim. After three benchmark losses,
"which mechanisms carry measurable content" is the question that can still return information;
"win a benchmark" cannot. P4 below PRE-REGISTERS THE LOSS so it cannot be quietly dropped.

THE HYPOTHESES (three, each able to die separately)
==================================================
H1 (structural, near-certain, cheap)
    `network_cgl.py` as shipped CANNOT EXPRESS ENCOUNTER FREQUENCY. ARI(T) flat over a 16x
    encounter range. The cause is readable in the source, not inferred:
      - `activate()` contains `if not self.bonded[i,j]` -- an already-bonded pair is SKIPPED, so a
        re-encounter deposits nothing;
      - `self.bonded` is a BOOLEAN matrix -- there is no place to put a graded count;
      - capacity REFUSES (`self.n_frustrated += 1; continue`) rather than EVICTING, so the first
        four bonds a unit happens to form are the four it keeps forever.
    A structure whose edges are write-once and boolean has no channel for "how often".

H2 (the real test)
    In `cgl_primitive.py` GRAPH MODE, of the spec's mechanisms only DESTRUCTIVE LOCAL COMMITMENT
    (the anti-percolation device) and the DISCORDANCE TRIGGER carry measurable content. The
    PER-COMPONENT JOINT COIN -- mechanism 3, the spec's central claim, "the mechanism a scalar
    eligibility trace provably cannot express" -- IS ARITHMETICALLY INERT at the codebase's own
    validated `commit_gain = 3.0`, and acquires content only once `commit_gain` is lowered far
    enough that p_commit < 1.

    THE PROOF (this is algebra over the shipped source, not a fit):
      `reward()` sets  p_commit = clip(commit_gain * trace[members].mean() * abs(r), 0, 1).
      `components()` builds adjacency from  fidelity() > edge_threshold,  i.e.  t_i * t_j > 0.5.
      Since every trace satisfies t <= 1.0, an edge t_i*t_j > 0.5 forces BOTH t_i > 0.5 and
      t_j > 0.5. Hence every member of every multi-unit component has t >= 0.5, so
      mean(trace over members) >= 0.5. Singleton components are gated harder still, at
      t > sqrt(edge_threshold) = 0.7071 (mechanism 4). This arm gates every component at the same
      0.7071 bound. Therefore
            p_commit = clip(3.0 * (>=0.7071) * 1.0) = clip(>= 2.121) = 1.0   IDENTICALLY.
      The joint coin NEVER FLIPS. Joint and independent-coin arms must be byte-identical.
    That converts an unfalsifiable claim ("the coin is the novelty") into a measured function of
    one EXISTING knob. `commit_gain` is the independent variable of this experiment.

H3 (pre-registered loss)
    No configuration of the substrate beats plain recency-windowed co-occurrence counting on
    partition recovery. NO QUALITY CLAIM IS AVAILABLE AND NONE WILL BE MADE.

PRE-REGISTERED PREDICTIONS, WITH THE ARITHMETIC
===============================================
All target numbers below were measured at 8 seeds in exactly this regime before this file was
written; the 30-seed run must reproduce them. Margins are set from the measured gaps, not guessed.

P1  THE INERT COIN -- a predicted EXACT IDENTITY.
    At commit_gain = 3.0:  |ARI(TEST) - ARI(C7_independent_coins)| <= 0.005 in 30/30 seeds, and
    mean p_commit = 1.000 +/- 0.000.
    Arithmetic: min trace in any component >= 0.5 (edge gate t_i*t_j > 0.5 with t_j <= 1); the arm
    gates further at mean > 0.7071; clip(3.0 * 0.7071) = clip(2.121) = 1.0.
    Measured at 8 seeds: delta = +0.000, 0/8 seeds differ.
    NON-TRIVIAL because any implementation bug breaks an exact identity. This is simultaneously
    the experiment's correctness check.

P2  THE COIN ACQUIRES CONTENT ONLY BELOW SATURATION.
    ARI(TEST) - ARI(C7) >= +0.08 (steady) at commit_gain = 0.4, in >= 27/30 seeds.
    Arithmetic: measured +0.115 (0.200 vs 0.085), 8/8 seeds, SEs 0.006 and 0.001 -> ~19 pooled SE.
    Margin 0.08 sits at 70% of the observed effect: REACHABLE (observed 0.115) and NON-TRIVIAL
    (it is 0.000 one gain-step away; at gain 1.0 the gap is only +0.009).
    The prediction is a DOSE-RESPONSE CURVE:
        gap  = 0.000 / 0.009 / 0.093 / 0.115   at   mean p = 1.000 / 0.991 / 0.595 / 0.397
    Monotone in p; Spearman rho(gap, 1-p) = 1.0 required.

P3  THE DISCORDANCE TRIGGER HAS CONTENT.
    ARI(TEST) - ARI(C6_rate_matched_random) >= +0.03 (steady) at commit_gain = 3.0, >= 24/30 seeds.
    Arithmetic: measured +0.044 (0.246 vs 0.201), 8/8 seeds, pooled SE 0.005 -> ~9 SE. Margin 0.03
    is 68% of the observed effect. NON-TRIVIAL: the post-drift form of the same gap is only +0.028
    at 6/8 seeds, i.e. the weaker form of this prediction is already borderline and can fail.

P4  PRE-REGISTERED LOSS. The test arm does not beat plain counting.
    ARI(C1 best H) - ARI(TEST) >= +0.15 steady at EVERY commit_gain, in >= 27/30 seeds.
    Arithmetic: 0.492 - 0.246 = +0.246 steady; post-drift 0.535 - 0.207 = +0.328.
    THE MARGIN RUNS AGAINST THE TEST ARM, so no rescue is possible and no quality claim can be
    manufactured. This is the direct guard against the failure mode that dissolved the bandit win
    (plain recency-weighted greedy scored an IDENTICAL 82.9) and the mushroom win (a plain
    random-subspace running-mean table beat full CGL 2581 vs 17228).

P5  CEILING. C4 (weighted spectral, K=12 SUPPLIED) >= 0.90 steady in every above-threshold cell
    (measured 0.948). NOBODY MAY BEAT IT. C3 (Bethe Hessian, k NOT supplied) between C1 and C4.

P6  H1. |ARI(T=960) - ARI(T=120)| < 0.02 and ARI < 0.02 at every T for `network_cgl` as shipped.
    Measured: 0.001 / 0.000 / 0.000 / 0.000 / 0.001 / 0.002 across T = 120 -> 960.

P7  GLOBAL-LATE HAS NO REGIME. C5 <= 0.06 at every W_late; W<=25 returns >= 110 singletons;
    W=100 returns largest-component fraction >= 0.65.
    Measured 0.043 / 0.043 / 0.043 / 0.012 with max-frac 0.064 -> 0.726.

P8  BUG DETECTORS. In the STRUCTURE-FREE NULL cell (p_in = p_out = 0.0667) EVERY arm including C4
    scores ARI <= 0.03. At p_out = 0.35 (T*/T = 20.3, provably below threshold) every arm scores
    <= 0.15 (measured C4 = 0.100). ANY ARM SCORING > 0.15 IN EITHER CELL VOIDS THE ENTIRE BATCH.

WHAT WOULD FALSIFY THIS
=======================
1. P1 FAILS -- TEST and C7 differ at commit_gain = 3.0. Then the algebraic proof above is wrong or
   the implementation is. HALT; the whole framing is void.
2. P3 FAILS -- C6 within 0.02 of TEST at every gain. Then "acts on component-level discordance" is
   really "carves often", D(C) is decoration, and the arm is DenStream with stochastic eviction.
   THIS IS THE MOST PROBABLE KILL, put at ~35%: the post-drift form of the gap is already 6/8.
3. P2 FAILS -- no dose-response in p_commit. Then the joint coin is inert EVERYWHERE, not merely at
   the validated parameters, and the spec's central claim has no computational content in this
   substrate at any setting.
4. P4 FAILS IN THE ARM'S FAVOUR -- TEST beats C1. SUSPECT A BUG BEFORE CELEBRATING; check
   largest-component fraction and the C8 degenerate floors first.
5. C1d TIES OR BEATS the stochastic ledger (already measured 0.577 vs 0.068 at T=960). Then the
   k_entangle/eta rate law contributes nothing and the honest report is "this is a Misra-Gries
   sketch; the physics vocabulary is decoration."
6. C9 SCORES HIGHER BELOW THE WERNER BOUND. Reported as a problem for the physics claim.
   THE BOUND IS NOT MOVED. The 0.5 bound is PHYSICS, not a fitted knob.

WHAT THE NULL LOOKS LIKE CONCRETELY: a table where C6 ~ TEST at every gain, C7 ~ TEST at every
gain, TEST sits ~0.25 below C1 and ~0.70 below C4, C5 sits at 0.04, A0 is flat at 0.001. Written
up: "Of the seven mechanisms, graph mode exercises three; the joint coin is arithmetically pinned
at p = 1 and never fires; the trigger is rate, not discordance; what remains is destructive
eviction, which is the only anti-percolation device in the file and is worse than counting."
That is the fourth negative and it is a MECHANISTIC EXPLANATION, not another mismatched benchmark.

WHAT THIS REDUCES TO AMONG EXISTING METHODS -- stated as the headline, not a caveat
==================================================================================
The test arm is: ONLINE SINGLE-LINKAGE AGGLOMERATIVE CLUSTERING ON AN EXPONENTIALLY-DECAYED
CO-OCCURRENCE GRAPH, CUT AT A FIXED HEIGHT, WITH PER-CLUSTER DRIFT-TRIGGERED DESTRUCTIVE EVICTION.
Every modifier already has a name:
  1. `fidelity() > edge_threshold` + BFS IS single-linkage at a fixed threshold. It inherits
     single-linkage's chaining fragility, and that fragility is the dominant measured failure mode.
  2. The whole loop is DAMPED-WINDOW STREAMING MICRO-CLUSTERING (CluStream, Aggarwal 2003;
     DenStream, Cao et al. 2006). Trace decay IS the damped window; the Werner bound IS the density
     threshold; the destructive commit IS outlier eviction.
  3. The coverage-discordance trigger is PER-CLUSTER CONCEPT-DRIFT DETECTION (ADWIN, Bifet &
     Gavalda 2007; DDM, Gama et al. 2004) scoped to one cluster instead of a global error rate.
     A variation, not a new kind of thing.
  4. C1d is MISRA-GRIES / SPACE-SAVING heavy-hitter tracking with exponential decay.
IT IS NOT MCL. MCL alternates expansion (matrix power) with inflation (elementwise power + column
renormalisation) to convergence on a BATCH matrix. This design has no matrix power, no inflation,
no renormalisation, no iteration, and is one-pass streaming. MCL appears as control C2, where it
belongs, and the question is settled by measurement rather than assertion.
WHAT IS GENUINELY LEFT OVER, and it is exactly two things, both of which this file measures:
  (a) THE JOINT PER-COMPONENT COIN. Standard streaming clusterers emit deterministically. P1 shows
      it is arithmetically inert at the codebase's validated parameters; P2 tests whether it has
      content anywhere.
  (b) DESTRUCTIVE READOUT THAT RECYCLES UNITS. Committing consumes the traces and clears the bonds,
      freeing units to re-bind immediately. Closest analogue is cluster eviction; the difference is
      the units survive and are recycled rather than the cluster being deleted. This is the only
      anti-percolation device in `cgl_primitive.py`.

HOW THIS FILE TREATS THE SHIPPED CODE -- SAID LOUDLY
====================================================
*** `cgl_primitive.py` AND `network_cgl.py` ARE NOT MODIFIED. NOT ONE LINE. ***
`test_primitive_regression.py` must still print 7/7; this file runs it and reports the result.

*** WHAT IS RE-EXPRESSED HERE, AND WHY IT IS NOT A SILENT FORK ***
The test arm needs `reward()` WITH ITS TRIGGER LOCALIZED -- fire on ONE component, when that
component reads discordant, rather than sweeping all components on a scalar reward signal. The
primitive has no such entry point. Two options existed: add a method to `cgl_primitive.py`, or
express the localized commit in this file over the primitive's own state. THIS FILE TAKES THE
SECOND OPTION, so that not one line of the validated primitive changes. What is re-expressed is
`reward()`'s BODY, transcribed line-for-line so the coin semantics are IDENTICAL:
       p_commit = clip(commit_gain * trace[members].mean() * abs(r), 0, 1)     [r fixed to 1.0]
       one shared coin for the whole component
       on success: trace[members] = trace_floor          ("the readout consumes the trace")
It ADDS exactly one thing `reward()` does not do: `bonded[ix_(members,members)] = False`, the
destructive carve. That is mechanism (b) above and it is the thing under test, so it is stated
here rather than buried.
Everything else -- trace decay, two-timescale binding, the Werner-thresholded fidelity matrix, the
edge gate, the singleton bound -- is CALLED on the shipped object, never re-implemented:
`L.activate()`, `L.decay()`, `L.fidelity()`, `L.p.edge_threshold`, `L.trace`, `L.p.trace_floor`.
The component BFS is run over `L.fidelity() > L.p.edge_threshold` -- the SAME adjacency
`components()` builds -- but restricted to the components the episode TOUCHED, because a global
`components()` sweep per episode is both O(N^2) wasted work and semantically wrong for a LOCAL
rule. The edge threshold is NOT lowered; C9 exists to CONFIRM the Werner bound is load-bearing.

TWO STRUCTURAL FINDINGS THAT STAND REGARDLESS OF OUTCOME
=========================================================
  - `network_cgl.py` as shipped cannot express encounter frequency (H1/P6).
  - `cgl_primitive.py` has NO VIABLE LATE-GLOBAL REGIME: bonds are monotone, so accumulation
    percolates, and `clear_episode()` starves it to singletons. The "network-scale partition read
    late" object DOES NOT EXIST IN THIS CODE. That is a structural restatement of F2-c and is
    reported as such -- NOT dressed up as a win for the local arm (C5/P7).

RUN:  /Users/sarahdavidson/miniforge3/bin/python3 benchmark_partition_discovery.py
      (--quick for a 6-seed smoke; --seeds N to override)
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cgl_primitive import CGLParams, CoherenceGatedPrimitive          # noqa: E402
from network_cgl import NetworkCGLParams, NetworkCoherenceGatedPrimitive  # noqa: E402

# =====================================================================================
# GENERATOR — a drifting planted partition over UNDIFFERENTIATED units.
#
# WHY THIS GENERATOR AND NOT A DATASET. All three prior losses used pre-carved representations. Here
# units are BARE INTEGERS: no features, no geometry, no ordering, no reward signal. The only thing
# any arm ever sees is a boolean co-activation vector. Labels, K, and the drift times are NEVER shown
# to any arm (C4 is the deliberate exception -- it is the CEILING and is given K=12 on purpose).
# This is "a room with no map": the discovery step has not been done for us.
#
# p_in = 0.6 means a group is NEVER SEEN WHOLE (~6 of 10 members per episode) -- dropout. p_out adds
# contaminants. Both are required: without dropout the problem is trivial set-identity; without
# contamination single-linkage never has a chance to chain and the failure mode under test never
# appears.
# =====================================================================================
N_UNITS = 120
K_GROUPS = 12
GROUP_SIZE = 10
T_EPISODES = 1200
EPOCH = 300                    # drift every 300 episodes -> 4 epochs
RHO_DRIFT = 0.25               # 25% of units change group at each boundary (15 swap-pairs)
P_IN = 0.6
BURN_IN = 50
PROBE_STEP = 25
PROBES: List[int] = list(range(BURN_IN, T_EPISODES, PROBE_STEP))   # 46 probes
POST_DRIFT_WINDOW = 50         # probes <= 50 episodes after a drift are "post"
STEADY_MIN_AGE = 150           # probes >= 150 episodes since drift are "steady"

# The p_out sweep plus the STRUCTURE-FREE NULL. p_out=0.35 and the null are the BUG DETECTORS (P8).
CELLS: List[Tuple[str, float, bool]] = [
    ("p_out=0.02", 0.02, False),
    ("p_out=0.10", 0.10, False),
    ("p_out=0.20", 0.20, False),
    ("p_out=0.35", 0.35, False),
    ("NULL(no-structure)", 0.0667, True),
]
PRIMARY_CELL = "p_out=0.02"

# TEST-arm sweeps. commit_gain is THE INDEPENDENT VARIABLE (H2/P1/P2).
GAINS = (3.0, 1.0, 0.6, 0.4)
THETAS = (0.25, 0.35, 0.50)
L_MEMS = (150, 300, 600)
THETA_DEFAULT = 0.35
L_MEM_DEFAULT = 300
W_DISCORD = 30.0               # exponential window of the coverage-discordance counters, episodes
EDGE_THRESHOLDS_C9 = (0.35, 0.20)   # C9 physics check. The 0.5 bound itself is NEVER moved.
HS_C1 = (50, 100, 200, 300)
HALF_LIVES_C1B = (30.0, 100.0)
W_LATES_C5 = (1, 5, 25, 100)
MCL_INFLATIONS = (1.4, 2.0, 3.0)
H_FOR_SPECTRAL = 300           # the window handed to C2/C3/C4; C1 sweeps H and is reported at best

# H1 sweep (stationary, no drift)
H1_TS = (60, 120, 160, 240, 320, 480, 960)
ETA_H1 = 0.33
SPIN_CAPACITY = 4
DELTA_LEDGER = 0.35            # per-re-encounter increment for the graded sketch (C1d/A2)
K_RELEASE = 1.0 / 216.0 + 1.0 / 200.0
K_ENTANGLE = 0.5

NO_LABEL_BASE = 10_000         # unit u with no live ledger entry gets its own singleton label


def make_stream(seed: int, p_out: float, structure_free: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (acts[T,N] bool, truth[T,N] int). truth is the planted partition IN FORCE at t."""
    rng = np.random.default_rng(seed)
    # THE PLANTED GROUPS ARE SCATTERED ACROSS UNIT INDEX, NOT CONTIGUOUS BLOCKS.
    # WHY (found by the P8 bug detectors, and fixed rather than reported around): with the obvious
    # `np.repeat(np.arange(12), 10)` the truth is CONTIGUOUS IN UNIT INDEX. Any arm whose internal
    # ordering is index-dependent then scores above chance for a reason that has nothing to do with
    # the data. C1d is exactly such an arm -- it processes co-active pairs in ascending index order
    # with evict-lowest, so its surviving bonds are biased toward index-adjacent pairs. Measured in
    # the STRUCTURE-FREE NULL, where there is provably nothing to detect:
    #        truth index-contiguous : ARI = +0.0094 +- 0.0008
    #        units re-indexed       : ARI = +0.0006 +- 0.0005
    # Scattering the groups removes the confound for EVERY arm. Note the direction: this fix can
    # only LOWER apparent scores, never manufacture one.
    perm = rng.permutation(N_UNITS)
    lab = np.empty(N_UNITS, dtype=int)
    lab[perm] = np.repeat(np.arange(K_GROUPS), GROUP_SIZE)   # group sizes stay exactly GROUP_SIZE
    truths = [lab.copy()]
    for _ in range(1, T_EPISODES // EPOCH):
        lab = lab.copy()
        n_swaps = int(RHO_DRIFT * N_UNITS) // 2       # 15 swap-pairs; group sizes stay exactly 10
        for _ in range(n_swaps):
            while True:
                i, j = int(rng.integers(N_UNITS)), int(rng.integers(N_UNITS))
                if lab[i] != lab[j]:
                    break
            lab[i], lab[j] = lab[j], lab[i]
        truths.append(lab.copy())
    acts = np.zeros((T_EPISODES, N_UNITS), dtype=bool)
    truth = np.zeros((T_EPISODES, N_UNITS), dtype=int)
    for t in range(T_EPISODES):
        L = truths[min(t // EPOCH, len(truths) - 1)]
        truth[t] = L
        if structure_free:
            # No blocks at all. The planted labels are still recorded as the reference so ARI has
            # something to score against -- and it MUST come out ~0. This is bug detector #1.
            acts[t] = rng.random(N_UNITS) < p_out
        else:
            k = rng.integers(K_GROUPS)
            acts[t] = rng.random(N_UNITS) < np.where(L == k, P_IN, p_out)
    return acts, truth


def _drift_age(t: int) -> int:
    """Episodes since the most recent drift. Pre-first-drift returns a large number (= steady)."""
    return t % EPOCH if t >= EPOCH else 10 ** 9


# =====================================================================================
# METRICS
#
# ARI is PRIMARY because it is chance-corrected: the below-threshold and structure-free cells must
# read ~0, and that is the built-in bug detector (P8).
# The secondaries are MANDATORY because the arms fail in OPPOSITE directions and ARI hides that:
# the test arm OVER-SPLITS (max-component ~0.21, ~18 clusters) while C1 top-4 OVER-MERGES
# (max-component ~0.54, ~6 clusters). Both score "low ARI" and they are not the same failure.
# =====================================================================================
def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    """Meila 2007, in BITS. VI = H(A) + H(B) - 2 I(A;B). Lower is better; 0 = identical."""
    n = len(a)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(cont, (ai, bi), 1.0)
    p = cont / n
    pa, pb = p.sum(1), p.sum(0)
    ha = -np.sum(pa[pa > 0] * np.log2(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log2(pb[pb > 0]))
    nz = p > 0
    outer = np.outer(pa, pb)
    mi = float(np.sum(p[nz] * np.log2(p[nz] / outer[nz])))
    return float(ha + hb - 2.0 * mi)


@dataclass
class ArmScore:
    ari_steady: float
    ari_post: float
    vi_steady: float
    max_frac: float
    n_clusters: float
    commits_per_ep: float = float("nan")
    mean_p_commit: float = float("nan")
    extra: float = float("nan")      # arm-specific (n_frustrated, inferred k, ...)
    # --- substrate diagnostics, populated for the CGL arms only ---
    eligible_per_ep: float = float("nan")   # components/episode that reached the trigger
    mean_comp_size: float = float("nan")    # size of those components (percolation detector)
    min_edge_F: float = float("nan")        # mean over episodes of the SMALLEST live-edge fidelity
    frac_contested: float = float("nan")    # live edges in (0.20, 0.50] -- what C9 would newly admit


def _attach(s: "ArmScore", st: Dict[str, float]) -> "ArmScore":
    s.commits_per_ep = st["commits_per_ep"]
    s.mean_p_commit = st["mean_p_commit"]
    s.eligible_per_ep = st["eligible_per_ep"]
    s.mean_comp_size = st["mean_comp_size"]
    s.min_edge_F = st["min_edge_F"]
    s.frac_contested = st["frac_edges_contested"]
    return s


def score_partitions(parts: Sequence[Tuple[int, np.ndarray]], truth: np.ndarray) -> ArmScore:
    st_a, po_a, st_v, mf, nc = [], [], [], [], []
    for t, lab in parts:
        a = ARI(truth[t], lab)
        age = _drift_age(t)
        if t >= EPOCH and age <= POST_DRIFT_WINDOW:
            po_a.append(a)
        elif age >= STEADY_MIN_AGE or t < EPOCH:
            st_a.append(a)
            st_v.append(variation_of_information(truth[t], lab))
        _, cnt = np.unique(lab, return_counts=True)
        mf.append(cnt.max() / N_UNITS)
        nc.append(len(cnt))
    return ArmScore(
        ari_steady=float(np.mean(st_a)) if st_a else float("nan"),
        ari_post=float(np.mean(po_a)) if po_a else float("nan"),
        vi_steady=float(np.mean(st_v)) if st_v else float("nan"),
        max_frac=float(np.mean(mf)),
        n_clusters=float(np.mean(nc)),
    )


# =====================================================================================
# THE TEST ARM — LOCAL-EARLY, a thin wrapper over the UNMODIFIED CoherenceGatedPrimitive.
#
# `clear_episode()` is NEVER called. Withholding segmentation is the point: the substrate is not
# told where one encounter ends and the next begins. (C5, the counterfactual, IS given the episode
# boundaries -- a handicap in ITS favour, deliberately the opposite of a strawman.)
#
# `bind_window = 0.0` with dt = 1.0: `recent` inside activate() then contains only THIS episode's
# units, so binding is a clique on the ~8 co-active units. The shipped default of 2.0 spans three
# episodes and binds three DIFFERENT groups -- percolation by construction. THIS IS A PARAMETER
# CHOICE, STATED AS SUCH, NOT A DERIVATION.
#
# THE DISCORDANCE POLARITY IS THE LOAD-BEARING CORRECTION AND IT IS COVERAGE, NOT STRADDLE.
# A "does this event straddle the component's boundary" trigger fires LEAST when the component has
# over-merged -- exactly backwards -- and measured 0 commits in 1200 episodes. COVERAGE-discordance
#     D(C) = 1 - mean_{u in C}( cov[u] / tot[u] )
# fires HARDEST on an over-merged component (a 120-unit blob touched by 8 units reads D = 0.93; a
# correctly-sized group reads D ~ 0.4). It is the signal that "the standing component is larger
# than the arriving evidence supports" -- which is precisely component-level discordance.
# =====================================================================================
def run_local_early(
    acts: np.ndarray,
    seed: int,
    gain: float = 3.0,
    theta_d: float = THETA_DEFAULT,
    l_mems: Sequence[int] = (L_MEM_DEFAULT,),
    joint_coin: bool = True,              # C7 sets this False -> independent per-unit coins at same p
    trigger: str = "discord",             # C6 sets this "rate" -> Bernoulli at a matched rate
    match_rate: Optional[float] = None,   # the per-eligible-component rate C6 must match
    edge_threshold: Optional[float] = None,   # C9 only. The 0.5 bound is NOT moved to rescue results.
) -> Tuple[Dict[int, List[Tuple[int, np.ndarray]]], Dict[str, float]]:
    """Returns ({L_mem: [(probe_t, labels)]}, stats). One run emits every L_mem readout -- L_mem is
    a READOUT-ONLY knob, so sweeping it costs nothing extra and cannot change the dynamics."""
    p = CGLParams(seed=seed, bind_window=0.0, commit_gain=gain)
    if edge_threshold is not None:
        p.edge_threshold = edge_threshold
    L = CoherenceGatedPrimitive(N_UNITS, p)
    rng = np.random.default_rng(seed + 7)          # separate stream: the arm's coins, not the data's

    cov = np.zeros(N_UNITS)
    tot = np.zeros(N_UNITS)
    dec = math.exp(-1.0 / W_DISCORD)
    singleton_bound = math.sqrt(L.p.edge_threshold)

    ledger: Dict[int, Tuple[int, int]] = {}        # unit -> (episode, commit_id); last-writer-wins
    out: Dict[int, List[Tuple[int, np.ndarray]]] = {lm: [] for lm in l_mems}
    probe_i = 0
    n_commits = 0
    n_eligible = 0      # components that passed size + trace gates (the denominator C6 matches on)
    n_passed = 0        # ... and then passed the discordance gate (the numerator)
    p_used: List[float] = []
    # --- diagnostics that explain WHY a gate is or is not load-bearing (see C9) ---
    edge_f_min: List[float] = []       # smallest live-edge fidelity seen in an episode
    edge_f_contested = 0               # live edges landing in (0.20, 0.50] -- what C9 would admit
    edge_f_total = 0
    comp_sizes: List[int] = []

    for t in range(T_EPISODES):
        a = acts[t]
        idx = np.flatnonzero(a)
        if len(idx) == 0:
            L.decay(1.0)
            cov *= dec
            tot *= dec
        else:
            L.activate(a)          # SHIPPED two-timescale binding. Not re-implemented.
            L.decay(1.0)           # SHIPPED trace decay. Not re-implemented.

            # (1) Locate the TOUCHED components only. Same adjacency components() builds --
            #     fidelity() > edge_threshold, i.e. the Werner-gated F = t_i*t_j > 0.5 -- but a
            #     LOCAL rule has no business sweeping all N units every episode.
            Fw = L.fidelity()
            live = Fw[np.triu(L.bonded, 1)]
            if live.size:
                edge_f_min.append(float(live.min()))
                edge_f_contested += int(((live > 0.20) & (live <= 0.50)).sum())
                edge_f_total += int(live.size)
            F = Fw > L.p.edge_threshold
            seen = set()
            comps: List[List[int]] = []
            for s0 in idx:
                s0 = int(s0)
                if s0 in seen:
                    continue
                stack, comp = [s0], {s0}
                seen.add(s0)
                while stack:
                    u = stack.pop()
                    for v in np.flatnonzero(F[u]):
                        v = int(v)
                        if v not in seen:
                            seen.add(v)
                            comp.add(v)
                            stack.append(v)
                comps.append(sorted(comp))

            # (2) Coverage discordance, exponential window W_D = 30 episodes.
            cov *= dec
            tot *= dec
            for mem in comps:
                for u in mem:
                    tot[u] += 1.0
                    if a[u]:
                        cov[u] += 1.0

            # (3) LOCAL-EARLY COMMIT — reward()'s body, trigger localized, r fixed to 1.0.
            for mem in comps:
                if len(mem) < 2:
                    continue
                mean_tr = float(L.trace[mem].mean())
                if mean_tr <= singleton_bound:        # mechanism 4's bound, applied to every component
                    continue
                n_eligible += 1
                comp_sizes.append(len(mem))
                if trigger == "discord":
                    d = 1.0 - float(np.mean(cov[mem] / (tot[mem] + 1e-9)))
                    if d < theta_d:
                        continue
                else:                                  # C6: rate, not discordance
                    if rng.random() >= (match_rate if match_rate is not None else 0.0):
                        continue
                n_passed += 1

                # --- transcribed from reward(): ONE shared coin for the whole component ---
                p_commit = float(np.clip(L.p.commit_gain * mean_tr * 1.0, 0.0, 1.0))
                p_used.append(p_commit)
                if joint_coin:
                    if rng.random() >= p_commit:
                        continue
                    win = mem
                else:                                  # C7: independent per-unit coins at the same p
                    win = [m for m in mem if rng.random() < p_commit]
                    if len(win) < 2:
                        continue
                n_commits += 1
                for m in win:
                    ledger[m] = (t, n_commits)
                L.trace[win] = L.p.trace_floor          # "the readout consumes the trace"
                L.bonded[np.ix_(win, win)] = False      # DESTRUCTIVE CARVE -- the one addition

        # (4) Readout: last-writer-wins per unit within L_mem. Units with no live entry are singletons.
        if probe_i < len(PROBES) and t == PROBES[probe_i]:
            probe_i += 1
            for lm in l_mems:
                lab = np.arange(N_UNITS) + NO_LABEL_BASE
                for u, (tt, cid) in ledger.items():
                    if t - tt <= lm:
                        lab[u] = cid
                out[lm].append((t, lab.copy()))

    stats = {
        "commits_per_ep": n_commits / T_EPISODES,
        "mean_p_commit": float(np.mean(p_used)) if p_used else float("nan"),
        "trigger_rate": (n_passed / n_eligible) if n_eligible else 0.0,
        "n_eligible": float(n_eligible),
        "eligible_per_ep": n_eligible / T_EPISODES,
        "mean_comp_size": float(np.mean(comp_sizes)) if comp_sizes else float("nan"),
        "min_edge_F": float(np.mean(edge_f_min)) if edge_f_min else float("nan"),
        "frac_edges_contested": (edge_f_contested / edge_f_total) if edge_f_total else float("nan"),
    }
    return out, stats


# =====================================================================================
# C5 — GLOBAL-LATE counterfactual, kept as a MEASUREMENT and not as a contrast to beat.
# Identical substrate, no per-episode check, no discordance, no carve. It IS handed the episode
# boundaries the test arm is denied (clear_episode() every W_late). Its failure is the concrete
# content of F2-c: THERE IS NO LATE-GLOBAL REGIME IN THIS SUBSTRATE.
# =====================================================================================
def run_global_late(acts: np.ndarray, seed: int, w_late: int) -> List[Tuple[int, np.ndarray]]:
    L = CoherenceGatedPrimitive(N_UNITS, CGLParams(seed=seed, bind_window=0.0))
    out, probe_i = [], 0
    for t in range(T_EPISODES):
        if acts[t].any():
            L.activate(acts[t])
        L.decay(1.0)
        if probe_i < len(PROBES) and t == PROBES[probe_i]:
            probe_i += 1
            lab = np.arange(N_UNITS) + NO_LABEL_BASE
            for ci, c in enumerate(L.components()):    # the SHIPPED global readout
                for u in c:
                    lab[u] = ci
            out.append((t, lab.copy()))
        if (t + 1) % w_late == 0:
            L.clear_episode()
    return out


# =====================================================================================
# COUNTING / INCUMBENT CONTROLS.
#
# C1 is the direct analogue of the plain recency-weighted greedy that dissolved the 4x bandit win
# at an identical 82.9, and of the random-subspace running-mean table that beat full CGL 2581 vs
# 17228 on mushroom. It is ORACLE-TUNED over H and REPORTED AT ITS BEST -- steelmanned, while the
# test arm is not. THE CONTROL IS DESIGNED FIRST, THEN THE TEST.
# =====================================================================================
def window_counts_at_probes(acts: np.ndarray, h: int) -> Dict[int, np.ndarray]:
    """Incremental sliding window: W(t) = sum_{s=t-h+1..t} outer(a_s). Snapshotted at each probe."""
    W = np.zeros((N_UNITS, N_UNITS))
    idxs = [np.flatnonzero(acts[t]) for t in range(T_EPISODES)]
    probe_set = set(PROBES)
    snaps: Dict[int, np.ndarray] = {}
    for t in range(T_EPISODES):
        W[np.ix_(idxs[t], idxs[t])] += 1.0
        old = t - h
        if old >= 0:
            W[np.ix_(idxs[old], idxs[old])] -= 1.0
        if t in probe_set:
            S = W.copy()
            np.fill_diagonal(S, 0.0)
            snaps[t] = S
    return snaps


def decayed_counts_at_probes(acts: np.ndarray, half_life: float) -> Dict[int, np.ndarray]:
    W = np.zeros((N_UNITS, N_UNITS))
    d = math.exp(-math.log(2.0) / half_life)
    probe_set = set(PROBES)
    snaps: Dict[int, np.ndarray] = {}
    for t in range(T_EPISODES):
        W *= d
        i = np.flatnonzero(acts[t])
        W[np.ix_(i, i)] += 1.0
        if t in probe_set:
            S = W.copy()
            np.fill_diagonal(S, 0.0)
            snaps[t] = S
    return snaps


def _cc_labels(adj: np.ndarray) -> np.ndarray:
    lab = -np.ones(N_UNITS, dtype=int)
    c = 0
    for s in range(N_UNITS):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if lab[v] < 0:
                    lab[v] = c
                    stack.append(v)
        c += 1
    return lab


def top_k_cc(W: np.ndarray, cap: int = 4) -> np.ndarray:
    """C1/C1b: keep each node's top-`cap` co-occurrence partners, symmetrise, connected components.
    No traces, no coins, no gates. cap=4 mirrors the spin_capacity=4 the substrate is given, so the
    comparison is capacity-matched rather than handing the control an unbounded degree."""
    A = np.zeros((N_UNITS, N_UNITS), dtype=bool)
    for u in range(N_UNITS):
        for v in np.argsort(-W[u])[:cap]:
            if W[u, v] > 0:
                A[u, v] = A[v, u] = True
    return _cc_labels(A)


def run_ledger_sketch(acts: np.ndarray, seed: int, stochastic: bool, evict: bool,
                      probe: bool = True) -> Tuple[List[Tuple[int, np.ndarray]], int]:
    """C1d (deterministic) / A2 (stochastic) — the DETERMINISTIC BOUNDED SKETCH.

    Graded per-bond score, +DELTA on re-encounter, exponential decay at the model's OWN
    k_release = 1/216 + 1/200, capacity 4 with EVICT-LOWEST (vs `network_cgl`'s REFUSE), CC readout.
    `stochastic=False` REMOVES the admission gate 1 - exp(-k_entangle*eta*dt); everything else is
    identical, so the pair isolates exactly what the rate law contributes.

    HONEST NAMING: this is MISRA-GRIES / SPACE-SAVING heavy-hitter tracking with exponential decay,
    per node over co-occurrence pairs. Nothing about it is novel and it is not claimed to be.
    HONEST CAVEAT, SAID LOUDLY: a graded SCORE is not a probability, so there is no Werner analogue
    for it. The gate here is the CAPACITY plus a prune at 0.02; it is NOT a fidelity threshold and
    is not presented as one.
    """
    rng = np.random.default_rng(seed + 11)
    nb: List[Dict[int, float]] = [dict() for _ in range(N_UNITS)]
    p_admit = 1.0 - math.exp(-K_ENTANGLE * ETA_H1 * 1.0)
    dec = math.exp(-K_RELEASE)
    n_frustrated = 0
    probe_set = set(PROBES) if probe else set()
    out: List[Tuple[int, np.ndarray]] = []
    for t in range(len(acts)):
        idx = np.flatnonzero(acts[t])
        for ai in range(len(idx)):
            for bj in range(ai + 1, len(idx)):
                i, j = int(idx[ai]), int(idx[bj])
                if stochastic and rng.random() >= p_admit:
                    continue
                if j in nb[i]:
                    nb[i][j] = min(1.0, nb[i][j] + DELTA_LEDGER)   # RE-ENCOUNTER DEPOSITS SOMETHING
                    nb[j][i] = nb[i][j]
                    continue
                ok = True
                for (u, _v) in ((i, j), (j, i)):
                    if len(nb[u]) >= SPIN_CAPACITY:
                        if not evict:
                            ok = False
                            break
                        weakest = min(nb[u], key=nb[u].get)
                        if nb[u][weakest] >= 1.0:      # cannot evict a saturated bond
                            ok = False
                            break
                if not ok:
                    n_frustrated += 1
                    continue
                for (u, _v) in ((i, j), (j, i)):
                    if len(nb[u]) >= SPIN_CAPACITY:
                        weakest = min(nb[u], key=nb[u].get)
                        del nb[u][weakest]
                        nb[weakest].pop(u, None)
                nb[i][j] = DELTA_LEDGER
                nb[j][i] = DELTA_LEDGER
        for u in range(N_UNITS):
            d = nb[u]
            for v in list(d):
                d[v] *= dec
                if d[v] < 0.02:
                    del d[v]
        if t in probe_set:
            A = np.zeros((N_UNITS, N_UNITS), dtype=bool)
            for u in range(N_UNITS):
                for v in nb[u]:
                    A[u, v] = A[v, u] = True
            out.append((t, _cc_labels(A)))
    if not probe:
        A = np.zeros((N_UNITS, N_UNITS), dtype=bool)
        for u in range(N_UNITS):
            for v in nb[u]:
                A[u, v] = A[v, u] = True
        out.append((len(acts) - 1, _cc_labels(A)))
    return out, n_frustrated


def mcl(W: np.ndarray, inflation: float, expansion: int = 2, iters: int = 30,
        tol: float = 1e-6) -> np.ndarray:
    """C2 — Markov Clustering (van Dongen 2000). An adversarial completeness review warned that a
    SIBLING design reduces to MCL. This design does not (no matrix power, no inflation, no
    renormalisation, one-pass streaming) -- so MCL is run as a control and the question is settled
    by measurement instead of assertion."""
    M = W.astype(float).copy()
    if M.max() <= 0:
        return np.arange(N_UNITS)
    np.fill_diagonal(M, M.max(axis=1))                  # self-loops, standard MCL
    M /= np.maximum(M.sum(axis=0, keepdims=True), 1e-12)
    for _ in range(iters):
        prev = M
        M = M @ M                                       # expansion (expansion=2)
        M = M ** inflation                              # inflation
        M /= np.maximum(M.sum(axis=0, keepdims=True), 1e-12)
        M[M < 1e-8] = 0.0
        if np.abs(M - prev).max() < tol:
            break
    attractor = np.argmax(M, axis=0)                    # each column's attractor row
    A = np.zeros((N_UNITS, N_UNITS), dtype=bool)
    for j in range(N_UNITS):
        A[j, attractor[j]] = A[attractor[j], j] = True  # union node with its attractor
    return _cc_labels(A)


def bethe_hessian(W: np.ndarray) -> Tuple[np.ndarray, int]:
    """C3 — non-backtracking / Bethe Hessian spectral clustering (Krzakala et al., PNAS 110:20935,
    2013). THE FAIR INCUMBENT: provably detects down to the Kesten-Stigum limit in exactly the
    sparse regime a CGL win would claim as home turf. k IS INFERRED from the count of negative
    eigenvalues -- NOT SUPPLIED. Returns (labels, inferred_k).

    IMPLEMENTER'S CHOICE, STATED LOUDLY: BH is defined for sparse graphs, and the raw co-occurrence
    matrix here is dense. The binarisation is a CONFIGURATION-MODEL (Chung-Lu) null, which is
    ORACLE-FREE -- it uses only the observed per-unit activation marginals, never the labels:
        edge iff  W_ij > E_ij + 2*sqrt(E_ij),   E_ij = H * p_i * p_j
    No knob in it was tuned against the score.
    """
    H = W.sum() / max((W > 0).sum(), 1)     # unused scale ref; kept explicit for readability
    tot = W.sum()
    deg = W.sum(axis=1)
    if tot <= 0:
        return np.arange(N_UNITS), 0
    E = np.outer(deg, deg) / tot            # configuration-model expectation
    A = (W > E + 2.0 * np.sqrt(np.maximum(E, 1e-12))).astype(float)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    c = d.mean()
    if c <= 1.0:
        return np.arange(N_UNITS), 0
    r = math.sqrt(c)
    BH = (r * r - 1.0) * np.eye(N_UNITS) - r * A + np.diag(d)
    w, V = np.linalg.eigh(BH)
    k = int((w < 0).sum())
    if k < 2:
        return np.zeros(N_UNITS, dtype=int), k
    k_use = min(k, 40)                      # guard only; the INFERRED k is what gets reported
    X = V[:, :k_use]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return KMeans(n_clusters=k_use, n_init=4, random_state=0).fit_predict(X), k


def weighted_spectral(W: np.ndarray, k: int = K_GROUPS) -> np.ndarray:
    """C4 — THE CEILING. Weighted normalized spectral + KMeans with K = 12 SUPPLIED. It is given an
    advantage no CGL arm gets. NOBODY MAY BEAT IT; if an arm does, suspect a bug first (P5)."""
    d = W.sum(axis=1)
    D = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    Ln = D @ W @ D
    _, V = np.linalg.eigh(Ln)
    X = V[:, -k:]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(X)


# =====================================================================================
# C8 — DEGENERATE FLOORS. Without these, a percolated arm scoring 0.00 is INDISTINGUISHABLE FROM A
# METRIC BUG. All three must score ARI ~ 0.
# =====================================================================================
def floor_partitions(kind: str, sizes_ref: Optional[List[np.ndarray]] = None,
                     seed: int = 0) -> List[Tuple[int, np.ndarray]]:
    rng = np.random.default_rng(seed + 31)
    out = []
    for pi, t in enumerate(PROBES):
        if kind == "singletons":
            lab = np.arange(N_UNITS)
        elif kind == "one_blob":
            lab = np.zeros(N_UNITS, dtype=int)
        else:                       # random partition SIZE-MATCHED to a reference arm's own sizes
            sizes = sizes_ref[pi] if sizes_ref else np.array([N_UNITS])
            perm = rng.permutation(N_UNITS)
            lab = np.empty(N_UNITS, dtype=int)
            pos = 0
            for ci, s in enumerate(sizes):
                lab[perm[pos:pos + int(s)]] = ci
                pos += int(s)
            if pos < N_UNITS:
                lab[perm[pos:]] = np.arange(len(sizes), len(sizes) + (N_UNITS - pos))
        out.append((t, lab))
    return out


def sizes_of(parts: Sequence[Tuple[int, np.ndarray]]) -> List[np.ndarray]:
    return [np.unique(lab, return_counts=True)[1] for _, lab in parts]


# =====================================================================================
# DETECTABILITY — computed and VALIDATED before running.
#
# This is a DENSE WEIGHTED planted partition (every pair is observed T times), so the relevant
# transition is BBP, not Kesten-Stigum. If the regime sat below threshold, a "win" there would be a
# BUG, NOT A RESULT -- which is exactly why p_out=0.35 and the structure-free null are bug detectors.
# =====================================================================================
def detectability_table() -> List[Tuple[float, float, float, float, float, float]]:
    rows = []
    for p_out in (0.02, 0.10, 0.20, 0.35):
        Pw = (1.0 / K_GROUPS) * P_IN ** 2 + ((K_GROUPS - 1) / K_GROUPS) * p_out ** 2
        Pc = (2.0 / K_GROUPS) * P_IN * p_out + ((K_GROUPS - 2) / K_GROUPS) * p_out ** 2
        sig = (N_UNITS / K_GROUPS) * (Pw - Pc)
        var = ((GROUP_SIZE - 1) * Pw + (N_UNITS - GROUP_SIZE) * Pc) / (N_UNITS - 1)
        noi = 2.0 * math.sqrt(N_UNITS * var)
        t_star = (noi / sig) ** 2
        rows.append((p_out, Pw, Pc, sig, noi, t_star))
    return rows


# =====================================================================================
# THE MAIN BATTERY — one worker per (seed, cell). The SAME activation stream is replayed
# IDENTICALLY to every arm. That is REQUIRED, not optional: it is what makes the per-seed paired
# comparisons in P1-P4 valid.
# =====================================================================================
def run_cell(args) -> Tuple[str, int, Dict[str, ArmScore]]:
    seed, cell_name, p_out, structure_free, full_sweep = args
    acts, truth = make_stream(seed, p_out, structure_free)
    R: Dict[str, ArmScore] = {}

    # ---- TEST arm across the commit_gain sweep, plus its two ablations C6 and C7 ----
    for g in GAINS:
        parts, st = run_local_early(acts, seed, gain=g, l_mems=(L_MEM_DEFAULT,))
        R[f"TEST gain={g}"] = _attach(score_partitions(parts[L_MEM_DEFAULT], truth), st)

        parts7, _ = run_local_early(acts, seed, gain=g, l_mems=(L_MEM_DEFAULT,), joint_coin=False)
        R[f"C7 indep-coins gain={g}"] = score_partitions(parts7[L_MEM_DEFAULT], truth)

        # C6 rate is matched at the TRIGGER (fraction of ELIGIBLE components that fire), not at the
        # commit count -- a strictly tighter match than matching commits/episode.
        parts6, st6 = run_local_early(acts, seed, gain=g, l_mems=(L_MEM_DEFAULT,),
                                      trigger="rate", match_rate=st["trigger_rate"])
        s6 = _attach(score_partitions(parts6[L_MEM_DEFAULT], truth), st6)
        s6.extra = st6["trigger_rate"]
        R[f"C6 rate-matched gain={g}"] = s6

    if full_sweep:
        # theta_D and L_mem sweeps. L_mem is a READOUT-ONLY knob so one run emits all three.
        for th in THETAS:
            parts, st = run_local_early(acts, seed, gain=3.0, theta_d=th, l_mems=L_MEMS)
            for lm in L_MEMS:
                R[f"TEST theta={th} L_mem={lm}"] = _attach(score_partitions(parts[lm], truth), st)
        # C9 physics check. Run to CONFIRM the Werner bound is load-bearing, not to rescue anything.
        for et in EDGE_THRESHOLDS_C9:
            parts, st = run_local_early(acts, seed, gain=3.0, l_mems=(L_MEM_DEFAULT,),
                                        edge_threshold=et)
            R[f"C9 edge_thr={et}"] = _attach(score_partitions(parts[L_MEM_DEFAULT], truth), st)

    # ---- C5 GLOBAL-LATE ----
    for w in W_LATES_C5:
        R[f"C5 global-late W={w}"] = score_partitions(run_global_late(acts, seed, w), truth)

    # ---- counting / incumbent controls, all off the SAME streams ----
    snaps: Dict[int, Dict[int, np.ndarray]] = {h: window_counts_at_probes(acts, h) for h in HS_C1}
    for h in HS_C1:
        R[f"C1 top4 H={h}"] = score_partitions([(t, top_k_cc(snaps[h][t])) for t in PROBES], truth)
    for hl in HALF_LIVES_C1B:
        dsn = decayed_counts_at_probes(acts, hl)
        R[f"C1b decayed top4 hl={hl}"] = score_partitions(
            [(t, top_k_cc(dsn[t])) for t in PROBES], truth)

    ledger_parts, n_fr = run_ledger_sketch(acts, seed, stochastic=False, evict=True)
    s = score_partitions(ledger_parts, truth)
    s.extra = float(n_fr)
    R["C1d ledger determ+evict"] = s
    ledger_s, n_fr_s = run_ledger_sketch(acts, seed, stochastic=True, evict=True)
    s = score_partitions(ledger_s, truth)
    s.extra = float(n_fr_s)
    R["A2 ledger stoch+evict"] = s

    Wsn = snaps[H_FOR_SPECTRAL]
    for infl in MCL_INFLATIONS:
        R[f"C2 MCL r={infl}"] = score_partitions([(t, mcl(Wsn[t], infl)) for t in PROBES], truth)
    bh_parts, bh_ks = [], []
    for t in PROBES:
        lab, kk = bethe_hessian(Wsn[t])
        bh_parts.append((t, lab))
        bh_ks.append(kk)
    s = score_partitions(bh_parts, truth)
    s.extra = float(np.mean(bh_ks))
    R["C3 BetheHessian (k inferred)"] = s
    R["C4 spectral K=12 SUPPLIED"] = score_partitions(
        [(t, weighted_spectral(Wsn[t])) for t in PROBES], truth)

    # ---- C8 degenerate floors ----
    test_parts, _ = run_local_early(acts, seed, gain=3.0, l_mems=(L_MEM_DEFAULT,))
    c1_parts = [(t, top_k_cc(snaps[200][t])) for t in PROBES]
    R["C8 floor all-singletons"] = score_partitions(floor_partitions("singletons"), truth)
    R["C8 floor one-blob"] = score_partitions(floor_partitions("one_blob"), truth)
    R["C8 floor rand~TEST sizes"] = score_partitions(
        floor_partitions("rand", sizes_of(test_parts[L_MEM_DEFAULT]), seed), truth)
    R["C8 floor rand~C1 sizes"] = score_partitions(
        floor_partitions("rand", sizes_of(c1_parts), seed), truth)

    return cell_name, seed, R


# =====================================================================================
# THE H1 SWEEP — `network_cgl.py` EXACTLY AS WRITTEN, stationary, over a 16x encounter range.
#
# coupling_length = 1e9 and positions = 0 so w == 1. THIS IS HONEST AND NECESSARY: network_cgl's own
# docstring calls positions "a modelling choice, not something the biology dictates", and random
# positions would impose a geometry UNCORRELATED with the planted blocks -- a handicap that has
# nothing to do with the hypothesis. The random-position number is reported alongside as a
# documented side-condition, so the choice is visible rather than buried.
# =====================================================================================
def _stationary_truth(seed: int) -> np.ndarray:
    """Groups SCATTERED across unit index — same confound fix as make_stream(); see the comment
    there. C1d's evict-lowest is index-order dependent, so contiguous groups hand it free score."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_UNITS)
    lab = np.empty(N_UNITS, dtype=int)
    lab[perm] = np.repeat(np.arange(K_GROUPS), GROUP_SIZE)
    return lab


def _stationary_stream(seed: int, T: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = _stationary_truth(seed)
    A = np.zeros((T, N_UNITS), dtype=bool)
    for t in range(T):
        k = rng.integers(K_GROUPS)
        A[t] = rng.random(N_UNITS) < np.where(truth == k, P_IN, 0.02)
    return A


def run_a0(acts: np.ndarray, seed: int, random_positions: bool = False) -> Tuple[np.ndarray, int]:
    p = NetworkCGLParams(eta=ETA_H1, spin_capacity=SPIN_CAPACITY, bind_window=0.0,
                         coupling_length=1e9, seed=seed)
    pos = None if random_positions else np.zeros((N_UNITS, 2))
    if random_positions:
        p.coupling_length = 5.0
    L = NetworkCoherenceGatedPrimitive(N_UNITS, positions=pos, params=p)
    for t in range(len(acts)):
        L.activate(acts[t], dt=1.0)
        L.decay(1.0)
    # Readout: connected components of the SURVIVING BOND SET, topology unmasked by trace decay.
    saved = L.trace.copy()
    L.trace[:] = 1.0
    comps = L.components()
    L.trace[:] = saved
    lab = np.arange(N_UNITS) + NO_LABEL_BASE
    for ci, c in enumerate(comps):
        for u in c:
            lab[u] = ci
    return lab, L.n_frustrated


def run_h1(args) -> Tuple[int, int, Dict[str, float]]:
    seed, T = args
    truth = _stationary_truth(seed)
    A = _stationary_stream(seed, T)
    res: Dict[str, float] = {}
    lab, nfr = run_a0(A, seed)
    res["A0 network_cgl AS SHIPPED"] = ARI(truth, lab)
    res["A0 n_frustrated"] = float(nfr)
    res["A0 maxfrac"] = float(np.unique(lab, return_counts=True)[1].max() / N_UNITS)
    lab_rp, _ = run_a0(A, seed, random_positions=True)
    res["A0 (random positions, side-condition)"] = ARI(truth, lab_rp)
    for name, stoch in (("A2 ledger stoch+evict", True), ("C1d ledger determ+evict", False)):
        parts, _ = run_ledger_sketch(A, seed, stochastic=stoch, evict=True, probe=False)
        res[name] = ARI(truth, parts[-1][1])
    parts, _ = run_ledger_sketch(A, seed, stochastic=True, evict=False, probe=False)
    res["C1e ledger stoch, REFUSE (no evict)"] = ARI(truth, parts[-1][1])
    W = np.zeros((N_UNITS, N_UNITS))
    for t in range(T):
        i = np.flatnonzero(A[t])
        W[np.ix_(i, i)] += 1.0
    np.fill_diagonal(W, 0.0)
    res["C1 top4 counts"] = ARI(truth, top_k_cc(W))
    res["C4 spectral K=12 SUPPLIED"] = ARI(truth, weighted_spectral(W))
    return seed, T, res


# =====================================================================================
# REPORTING
# =====================================================================================
def agg(vals: List[float]) -> Tuple[float, float]:
    a = np.array([v for v in vals if not math.isnan(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0


def paired(res: Dict[str, List[ArmScore]], a: str, b: str, field: str) -> Tuple[float, int, int]:
    xa = [getattr(s, field) for s in res[a]]
    xb = [getattr(s, field) for s in res[b]]
    d = [x - y for x, y in zip(xa, xb) if not (math.isnan(x) or math.isnan(y))]
    return (float(np.mean(d)) if d else float("nan"), sum(1 for v in d if v > 0), len(d))


def main() -> None:
    n_seeds = 30
    if "--quick" in sys.argv:
        n_seeds = 6
    if "--seeds" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    seeds = list(range(n_seeds))
    t_start = time.time()

    print("=" * 110)
    print("THE INERT-COIN TEST — mechanism attribution on a drifting planted partition")
    print(f"run at {datetime.now(timezone.utc).isoformat()}   seeds={n_seeds}")
    print("=" * 110)

    # ---- 0. the regression suite MUST still be 7/7 ----
    print("\n[0] REGRESSION GATE — cgl_primitive.py must be untouched and still reproduce everything")
    try:
        out = subprocess.run([sys.executable, os.path.join(_HERE, "test_primitive_regression.py")],
                             capture_output=True, text=True, timeout=600)
        tail = [l for l in out.stdout.strip().splitlines() if "passed" in l or "FAIL" in l]
        print("   " + (" | ".join(tail) if tail else out.stdout.strip()[-200:]))
        regression_ok = "7/7 passed" in out.stdout
    except Exception as e:                                  # noqa: BLE001
        print(f"   COULD NOT RUN: {e}")
        regression_ok = False
    print(f"   REGRESSION SUITE: {'PASS 7/7' if regression_ok else '*** NOT 7/7 — HALT ***'}")

    # ---- 1. detectability, printed BEFORE any result ----
    print("\n[1] DETECTABILITY (BBP, dense weighted planted partition) — computed BEFORE running.")
    print(f"    {'p_out':>7s} {'P_w':>9s} {'P_c':>9s} {'sig/T':>9s} {'noise/sqT':>10s} {'T*':>10s} {'T*/1200':>9s}")
    for p_out, Pw, Pc, sig, noi, ts in detectability_table():
        print(f"    {p_out:7.2f} {Pw:9.5f} {Pc:9.5f} {sig:9.4f} {noi:10.4f} {ts:10.0f} {ts/T_EPISODES:9.2f}")
    print("    HONEST CAVEAT (must be printed, not buried): the closed form is CONSERVATIVE BY ~1 DECADE.")
    print("    It calls p_out=0.20 marginal (T*/T = 1.22) yet spectral scores ~1.000 there. The bound is")
    print("    VALIDATED ONLY AT p_out=0.35. Therefore p_out=0.20 is a SCORING cell, NOT a bug-detector")
    print("    cell; the bug detectors are p_out=0.35 and the structure-free null. Do not read a score at")
    print("    p_out=0.20 as an artefact.")

    # ---- 2. the main battery ----
    tasks = [(s, name, p_out, sf, name == PRIMARY_CELL)
             for name, p_out, sf in CELLS for s in seeds]
    print(f"\n[2] MAIN BATTERY — {len(tasks)} (seed, cell) runs over {len(CELLS)} cells ...")
    with Pool(processes=min(os.cpu_count() or 4, 14)) as pool:
        raw = pool.map(run_cell, tasks)
    by_cell: Dict[str, Dict[str, List[ArmScore]]] = {c[0]: {} for c in CELLS}
    for cell_name, _seed, R in raw:
        for arm, s in R.items():
            by_cell[cell_name].setdefault(arm, []).append(s)
    print(f"    done in {time.time() - t_start:.0f}s")

    # ---- 3. tables ----
    for cell_name, _p, _sf in CELLS:
        res = by_cell[cell_name]
        print("\n" + "=" * 110)
        print(f"CELL {cell_name}" + ("   [BUG DETECTOR — every arm must score <= 0.15 / <= 0.03]"
                                     if cell_name in ("p_out=0.35", "NULL(no-structure)") else ""))
        print("=" * 110)
        print(f"{'arm':32s} {'ARI steady':>16s} {'ARI post':>16s} {'VI(bits)':>9s} "
              f"{'maxfrac':>8s} {'nclust':>7s} {'cmt/ep':>7s} {'mean p':>7s} {'extra':>8s}")
        for arm in res:
            v = res[arm]
            ms, es = agg([s.ari_steady for s in v])
            mp, ep = agg([s.ari_post for s in v])
            mv, _ = agg([s.vi_steady for s in v])
            mf, _ = agg([s.max_frac for s in v])
            nc, _ = agg([s.n_clusters for s in v])
            cm, _ = agg([s.commits_per_ep for s in v])
            pc, _ = agg([s.mean_p_commit for s in v])
            ex, _ = agg([s.extra for s in v])
            print(f"{arm:32s} {ms:8.3f}±{es:6.3f} {mp:8.3f}±{ep:6.3f} {mv:9.3f} "
                  f"{mf:8.3f} {nc:7.1f} {cm:7.3f} {pc:7.3f} {ex:8.2f}")

    # ---- 4. H1 sweep ----
    print("\n" + "=" * 110)
    print("[4] H1 — `network_cgl.py` AS SHIPPED over a 16x encounter range (stationary, no drift)")
    print("=" * 110)
    h1_tasks = [(s, T) for T in H1_TS for s in seeds]
    with Pool(processes=min(os.cpu_count() or 4, 14)) as pool:
        h1_raw = pool.map(run_h1, h1_tasks)
    h1: Dict[str, Dict[int, List[float]]] = {}
    for _s, T, R in h1_raw:
        for k, v in R.items():
            h1.setdefault(k, {}).setdefault(T, []).append(v)
    print(f"{'arm':40s}" + "".join(f"{T:>13d}" for T in H1_TS))
    for k in h1:
        row = "".join(f"  {np.mean(h1[k][T]):8.3f}   " if k.endswith("frustrated")
                      else f"  {np.mean(h1[k][T]):.3f}±{np.std(h1[k][T], ddof=1)/math.sqrt(len(seeds)):.3f}"
                      for T in H1_TS)
        print(f"{k:40s}{row}")

    # ---- 5. pre-registered verdicts ----
    print("\n" + "=" * 110)
    print("[5] PRE-REGISTERED VERDICTS — every one of these was written down BEFORE the run")
    print("=" * 110)
    prim = by_cell[PRIMARY_CELL]
    verdicts: List[Tuple[str, bool, str]] = []

    d, w, n = paired(prim, "TEST gain=3.0", "C7 indep-coins gain=3.0", "ari_steady")
    identical = sum(1 for x, y in zip([s.ari_steady for s in prim["TEST gain=3.0"]],
                                      [s.ari_steady for s in prim["C7 indep-coins gain=3.0"]])
                    if abs(x - y) <= 0.005)
    mp, _ = agg([s.mean_p_commit for s in prim["TEST gain=3.0"]])
    ok1 = identical == n and abs(mp - 1.0) < 1e-9
    verdicts.append(("P1 inert coin (EXACT IDENTITY at gain=3.0)", ok1,
                     f"|TEST-C7|<=0.005 in {identical}/{n} seeds (need {n}/{n}); "
                     f"mean p_commit = {mp:.4f} (need 1.0000); mean delta = {d:+.4f}"))

    d2, w2, n2 = paired(prim, "TEST gain=0.4", "C7 indep-coins gain=0.4", "ari_steady")
    gaps = []
    for g in GAINS:
        dg, _, _ = paired(prim, f"TEST gain={g}", f"C7 indep-coins gain={g}", "ari_steady")
        pg, _ = agg([s.mean_p_commit for s in prim[f"TEST gain={g}"]])
        gaps.append((g, pg, dg))
    mono = all(gaps[i][2] <= gaps[i + 1][2] + 1e-9 for i in range(len(gaps) - 1))
    ok2 = (d2 >= 0.08) and (w2 >= math.ceil(0.9 * n2))
    verdicts.append(("P2 dose-response: coin has content only below saturation", ok2,
                     f"gap at gain=0.4 = {d2:+.3f} (need >=+0.08) in {w2}/{n2} seeds "
                     f"(need >={math.ceil(0.9*n2)}); curve " +
                     " ".join(f"[g={g} p={p:.3f} gap={x:+.3f}]" for g, p, x in gaps) +
                     f"; monotone-in-(1-p) = {mono}"))

    d3, w3, n3 = paired(prim, "TEST gain=3.0", "C6 rate-matched gain=3.0", "ari_steady")
    ok3 = (d3 >= 0.03) and (w3 >= math.ceil(0.8 * n3))
    verdicts.append(("P3 discordance trigger has content (vs rate-matched random)", ok3,
                     f"gap = {d3:+.3f} (need >=+0.03) in {w3}/{n3} seeds (need >={math.ceil(0.8*n3)})"))

    best_h = max(HS_C1, key=lambda h: agg([s.ari_steady for s in prim[f"C1 top4 H={h}"]])[0])
    p4_ok, p4_detail = True, []
    for g in GAINS:
        dg, wg, ng = paired(prim, f"C1 top4 H={best_h}", f"TEST gain={g}", "ari_steady")
        good = (dg >= 0.15) and (wg >= math.ceil(0.9 * ng))
        p4_ok &= good
        p4_detail.append(f"[gain={g}: C1-TEST={dg:+.3f} in {wg}/{ng}]")
    verdicts.append((f"P4 PRE-REGISTERED LOSS: C1(best H={best_h}) beats TEST by >=0.15", p4_ok,
                     " ".join(p4_detail)))

    c4_ok = True
    c4_detail = []
    for cname, _p, sf in CELLS:
        if cname in ("p_out=0.35", "NULL(no-structure)"):
            continue
        m, _ = agg([s.ari_steady for s in by_cell[cname]["C4 spectral K=12 SUPPLIED"]])
        c4_detail.append(f"[{cname}: {m:.3f}]")
        c4_ok &= m >= 0.90
    beat = []
    for cname, _p, sf in CELLS:
        c4m, _ = agg([s.ari_steady for s in by_cell[cname]["C4 spectral K=12 SUPPLIED"]])
        for arm, v in by_cell[cname].items():
            if arm.startswith("C4"):
                continue
            m, _ = agg([s.ari_steady for s in v])
            if not math.isnan(m) and m > c4m + 0.01:
                beat.append(f"{cname}/{arm} {m:.3f}>{c4m:.3f}")
    # P5 is reported as its TWO clauses so the reader can see which one moved. NEITHER THRESHOLD IS
    # CHANGED -- the criteria are exactly as pre-registered; only the presentation is split.
    verdicts.append(("P5a ceiling level: C4 >= 0.90 steady in every above-threshold cell", c4_ok,
                     " ".join(c4_detail)))
    verdicts.append(("P5b ceiling unbeaten: no arm exceeds C4 by >0.01 in any cell", not beat,
                     "nothing beat it" if not beat else f"BEATEN BY: {beat}"))

    a0 = h1["A0 network_cgl AS SHIPPED"]
    flat = abs(np.mean(a0[960]) - np.mean(a0[120])) < 0.02 and all(np.mean(a0[T]) < 0.02 for T in H1_TS)
    verdicts.append(("P6 H1: network_cgl cannot express encounter frequency", flat,
                     "ARI(T): " + " ".join(f"{T}:{np.mean(a0[T]):.3f}" for T in H1_TS)))

    p7_ok, p7_detail = True, []
    for w_late in W_LATES_C5:
        m, _ = agg([s.ari_steady for s in prim[f"C5 global-late W={w_late}"]])
        mf, _ = agg([s.max_frac for s in prim[f"C5 global-late W={w_late}"]])
        ncl, _ = agg([s.n_clusters for s in prim[f"C5 global-late W={w_late}"]])
        p7_detail.append(f"[W={w_late}: ARI={m:.3f} maxfrac={mf:.3f} nclust={ncl:.0f}]")
        p7_ok &= m <= 0.06
    verdicts.append(("P7 GLOBAL-LATE has no regime (structural restatement of F2-c)", p7_ok,
                     " ".join(p7_detail)))

    p8_viol = []
    for cname, lim in (("p_out=0.35", 0.15), ("NULL(no-structure)", 0.03)):
        for arm, v in by_cell[cname].items():
            m, _ = agg([s.ari_steady for s in v])
            mpo, _ = agg([s.ari_post for s in v])
            worst = max([x for x in (m, mpo) if not math.isnan(x)] or [0.0])
            if worst > lim:
                p8_viol.append(f"{cname}/{arm}={worst:.3f}>{lim}")
    verdicts.append(("P8 BUG DETECTORS (a violation VOIDS THE ENTIRE BATCH)", not p8_viol,
                     "clean" if not p8_viol else f"VIOLATIONS: {p8_viol}"))

    for name, ok, detail in verdicts:
        print(f"  [{'HELD' if ok else 'FAILED'}] {name}\n           {detail}")

    # ---- 6. the falsifiers that are not predictions ----
    print("\n" + "=" * 110)
    print("[6] THE OTHER FALSIFIERS")
    print("=" * 110)
    c1d, _ = agg([s.ari_steady for s in prim["C1d ledger determ+evict"]])
    a2, _ = agg([s.ari_steady for s in prim["A2 ledger stoch+evict"]])
    print(f"  Falsifier 5 — does removing the k_entangle/eta rate law change anything?")
    print(f"    C1d determ={c1d:.3f}  vs  A2 stoch={a2:.3f}  (drift battery, {PRIMARY_CELL})")
    print(f"    H1 sweep T=960: C1d={np.mean(h1['C1d ledger determ+evict'][960]):.3f} "
          f"A2={np.mean(h1['A2 ledger stoch+evict'][960]):.3f} "
          f"C1e(REFUSE, no evict)={np.mean(h1['C1e ledger stoch, REFUSE (no evict)'][960]):.3f}")
    print(f"    -> if C1d >= A2, the rate law contributes nothing and the honest report is")
    print(f"       'this is a Misra-Gries sketch; the physics vocabulary is decoration.'")
    tv = prim["TEST gain=3.0"]
    base, _ = agg([s.ari_steady for s in tv])
    print(f"\n  Falsifier 6 — C9 physics check. The Werner bound is NOT moved to rescue a result.")
    print(f"    edge_threshold=0.5 (WERNER, shipped): {base:.3f}")
    for et in EDGE_THRESHOLDS_C9:
        if f"C9 edge_thr={et}" in prim:
            m, _ = agg([s.ari_steady for s in prim[f"C9 edge_thr={et}"]])
            mf, _ = agg([s.max_frac for s in prim[f"C9 edge_thr={et}"]])
            flag = "  <-- SCORES HIGHER BELOW THE BOUND: A PROBLEM FOR THE PHYSICS CLAIM" if m > base + 0.01 else ""
            print(f"    edge_threshold={et}: {m:.3f}  (maxfrac {mf:.3f}){flag}")
    fmin, _ = agg([s.min_edge_F for s in tv])
    fcon, _ = agg([s.frac_contested for s in tv])
    print(f"    WHY: mean smallest LIVE-edge fidelity per episode = {fmin:.4f}; fraction of live edges")
    print(f"    landing in the contested band (0.20, 0.50] = {fcon:.4f}. With bind_window=0.0 a bond only")
    print(f"    exists between units co-active in the SAME episode, so both traces are ~1.0 when the edge")
    print(f"    is evaluated: F is BIMODAL (0 if unbonded, >{fmin:.2f} if bonded). Lowering the bound admits")
    print(f"    nothing because NOTHING SITS NEAR IT. The boolean `bonded` mask, not the Werner threshold,")
    print(f"    is the operative gate here. That is a null for C9's premise, not a rescue of the bound.")

    ep, _ = agg([s.eligible_per_ep for s in tv])
    cs, _ = agg([s.mean_comp_size for s in tv])
    print(f"\n  SUBSTRATE DIAGNOSTIC (TEST gain=3.0): {ep:.2f} eligible components/episode, mean size {cs:.1f}")
    print(f"    of 120 units, ~8 co-active. If eligible/episode ~ 1.0 the episode's active units all chain")
    print(f"    into ONE component -- single-linkage percolation -- and the destructive carve is the only")
    print(f"    thing breaking it up. That is mechanism (b), and it is doing the work.")
    print(f"\n  C2 MCL — settling the 'does this reduce to MCL' warning by measurement:")
    for infl in MCL_INFLATIONS:
        m, _ = agg([s.ari_steady for s in prim[f"C2 MCL r={infl}"]])
        print(f"    MCL inflation={infl}: {m:.3f}")
    bh, _ = agg([s.ari_steady for s in prim["C3 BetheHessian (k inferred)"]])
    bhk, _ = agg([s.extra for s in prim["C3 BetheHessian (k inferred)"]])
    print(f"  C3 Bethe Hessian (k NOT supplied): ARI {bh:.3f}, inferred k = {bhk:.1f} (true K = 12)")

    print("\n" + "=" * 110)
    print(f"REGRESSION SUITE: {'PASS 7/7 — cgl_primitive.py untouched' if regression_ok else 'NOT 7/7'}")
    print(f"total wall {time.time() - t_start:.0f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
