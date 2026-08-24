# Scaling and the boundary of the graph primitive (2026-08-24)

Three benchmarks now characterise the primitive honestly — two strengths and one hard limit.

## B3 — CAPACITY under overlapping conjunctive rules: the graph holds, the scalar degrades

K rules, each a feature PAIR drawn from a 12-feature pool so features are shared across rules by
construction (a feature appears in both rewarded and punished rules). Accuracy = predicting each rule's sign.

| K rules | graph | scalar baseline | components stored |
|---|---|---|---|
| 2 | 100.0% | 100.0% | 30 |
| 8 | 100.0% | 90.0% | 36 |
| 16 | 100.0% | 83.8% | 44 |
| 32 | 100.0% | 73.8% | 60 |
| 64 | **100.0%** | **68.0%** | 92 |

The graph is flat at 100% while the per-feature learner decays toward chance as overlap forces conflicts.
**Component storage grows LINEARLY (30 -> 92 across a 32x increase in K), not combinatorially** — the feared
blow-up did not occur, because only observed components are ever stored.

## B4 — GENERALIZATION to unseen combinations: the graph is at EXACTLY chance

Rule: +1 iff BOTH features are in a hidden subset S. This is linearly separable at the feature level, so it
is deliberately FAVOURABLE to the scalar baseline. Train on 60% of pairs, test on the held-out 40%.

**Reported as BALANCED accuracy** (mean of per-class recall). Raw accuracy is misleading here: only ~22% of
pairs are +1, so "always predict -1" already scores 78%, and a first pass at raw accuracy made the graph look
like it generalised at 81.5% when it was doing nothing of the kind.

| | seen pairs | HELD-OUT pairs |
|---|---|---|
| graph (component lookup) | **100.0%** | **50.0% — exactly chance** |
| scalar trace baseline | 71.8% | **61.9%** |
| chance | 50.0% | 50.0% |

**The graph memorises perfectly and generalises not at all.** A combination it has never observed has no
component entry, so it has nothing to say. The scalar learner is the mirror image: worse on what it has seen,
but genuinely above chance on what it has not.

## The honest characterisation

| | structure / conjunctions | capacity under interference | generalization to unseen |
|---|---|---|---|
| graph + per-component collapse | **solves what scalars provably cannot** | **flat to K=64** | **none (chance)** |
| scalar eligibility trace | blind (chance) | degrades to 68% | modest but real |

They are complementary, not ranked. The graph is a **structure memoriser** with excellent capacity and
interference resistance and zero compositional generalization.

## What is missing, mechanistically

The primitive has **no notion of similarity between components**. Components are atomic keys: `{a,b}` and
`{a,c}` are as unrelated as `{a,b}` and `{x,y}`, so nothing can transfer between them. Any generalization
would have to come from partial overlap or partial reactivation of components — which is plausibly what
`coherence-gated-learning` primitive #3 ("learned state modifies the input distribution") and the
graph-as-attractor framing were reaching for, and which is NOT implemented here.

That is the next real question for this workstream, and it is a design question, not a tuning one.

## Caveat carried forward

All four benchmarks use cleanly time-separated inputs. Overlapping/continuous input streams — where the
binding window has to do real work — remain untested, and are the other obvious way this could fail.

---

# UPDATE (2026-08-24): partial reactivation closes the generalization gap at no cost

The boundary above ("components are atomic keys, so nothing transfers") was addressed with the mechanism the
diagnosis pointed to: **an unfamiliar component is answered by the stored components it OVERLAPS**, weighted
by Jaccard similarity, with an EXACT match always taking priority. A partial input partially reactivates the
patterns it resembles — the attractor / pattern-completion behaviour the program's framing invokes.

| benchmark | before | after |
|---|---|---|
| B2 structured credit (conjunction) | 100.0% | **100.0%** (unchanged) |
| B3 capacity at K=64 overlapping rules | 100.0% | **100.0%** (unchanged) |
| B4 generalization to unseen, balanced | **50.0% (chance)** | **61.9%** |

**Full picture now:**

| | conjunctions | capacity (K=64) | unseen combinations |
|---|---|---|---|
| graph + per-component collapse + partial reactivation | **100%** | **100%** | **61.9%** |
| scalar eligibility trace | chance | 68% | 61.9% |

The graph now **matches** the scalar learner exactly where the scalar learner used to be better, while
remaining far ahead everywhere else. It no longer has to trade structure for generalization.

## Honest accounting of what this is

- Mechanically, partial reactivation is **similarity-weighted retrieval over stored components**. It is
  motivated by the attractor framing and by the measured failure it fixes — it is NOT something Model 6
  measured, and it should not be described as biologically validated.
- Generalization at 61.9% is **parity with the baseline, not a win**. The result is that the graph gets
  generalization without giving up structure — not that it generalises better.
- All four benchmarks still use cleanly time-separated inputs. Overlapping/continuous streams, where the
  binding window must do real work, remain the main untested failure mode.
