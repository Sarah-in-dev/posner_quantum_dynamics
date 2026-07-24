# PRE-REGISTRATION — PO-11 Valence: does reward-directed collapse land credit on the *partition*, leak-immune?

**Status: REGISTERED 2026-07-24.** Design locked; append-only from here. Scorability gated on the preconditions
below — **P2 PASSED (2026-07-24)**; P1/P3 pending the off-path harness. Provisional ID PO-11 (successor to
PO-10 Unit C). Python `/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python`. **NEVER seed a physics run.**

---

## The bridge it tests

Unit C (`L·PO10-2`) showed the input's pairing structure reaches Δw and is recoverable — the **which**
(which synapses share fate). It left the **to-what-end** open: the collapse assigns each domain a *random
±1* sign, so Δw is "informative, not yet consistently useful — a rule needs a direction." This probe asks
whether the **reward signal can direct the plasticity so credit lands on the co-membered groups** — reward
strengthens the *right* synapses — and whether that is real or an abundance confound.

Builds on: `PREREG_PO10_UNIT_C_WEIGHT_LEVEL_KEYSTONE.md` (the task, the arms, the leak); `L·PO10-2` POST-REVIEW
UPDATE 2 (the magnitude leak); the advisor Q2 (dopamine/CaMKII rectifies the collapse into a directional Δw).

---

## The task — reuse Unit C exactly (makes binding NECESSARY)

The identical matched-marginal 4-cluster design: clusters A,B,C,D in **one compartment**, all within λ_F.
Two pairings — **{AB,CD}** vs **{AC,BD}** — with marginals identical by construction (each cluster active the
same duration/intensity in both; total drive, per-synapse activity, calcium integral matched). The ONLY
difference is *which pairs coincide*. Any per-synapse/scalar mechanism is at chance; the information lives only
in the pairing. No new task — and it is exactly where the magnitude leak was characterized.

## The mechanism — reward-directed readout, built OFF-PATH (the model path is a dead stub)

The reward-directed valence is `Δw = learning_rate · reward · correlated_eligibility` — reward signs Δw, the
partition carries co-membership. That logic exists in the model as **`apply_reward_correlated()`**
(`multi_synapse_network.py:1611`) **but it is an orphaned, unfinished stub and must NOT be driven:** zero call
sites; its body says `# Apply to synapses (need to implement weight storage)`; it writes
`syn._committed_memory_level`, which the **D20 durable-state audit** (`RESEARCH_LOG_CALCIUM_DIMER.md`, 2026-07-18,
[GROUNDED]) found is *not* the loop-closing weight — "coupling weights never mutate; `apply_reward_correlated` …
would be near-no-ops if wired (they write the lever that does not set magnitude)." Driving it drives a no-op.

So the reward-directed Δw is built **entirely off-path in the harness**, replicating that logic from the
tracker's existing collapse structure — exactly as Unit C reimplemented its signed readout rather than calling
model methods. Reward sets the sign (all committed domains → +); the partition (the tracker's
`_find_all_clusters()` domains) sets co-membership; magnitude ∝ committed-dimer abundance, as in the model.

**Off-path / DO-NOT-MODIFY:** zero model `.py` edits; do not touch `apply_reward_correlated`,
`spine_plasticity_module.py`, `analytical_gap`, or the model files. Readout + scorer live in experiment scripts
composed from `sweep/po10_unitC_experiment.py`.

> **Scope note — the deeper blocker (Option B, logged not solved here).** D20 is broader than this one stub: the
> model's *actual* plasticity/durable-state layer is broken — spine volume (the only loop-closing weight) decays
> below baseline by ~t=3000 s, falsifying the self-maintenance primitive; AMPAR is dead; coupling weights never
> mutate. **This probe is a weight-READOUT result** (can reward land partition-specific credit, leak-immune),
> **NOT a claim the closed loop learns.** Fixing the durable-state layer so the maze can learn is a separate
> thread (Option B).

## The measure — and how it dodges the PO-10 leak

**The leak (PO-10, recorded):** single-trial `|Δw_cluster| ∝ √(committed-dimer count)` = **abundance**, set by
drive TIMING. A magnitude decoder reads the pairing *without any binding* — it survived `scramble` (1.000) and
`lamshort` (0.833). Magnitude reads a marginal (timing) channel, not the partition.

**The dodge — partial correlation across trials, computed WITHIN drive condition (P2-validated).** Abundance is
~constant *within a matched drive condition*, so there it is a fixed per-cluster offset that cannot co-vary; the
shared collapse coin varies trial-to-trial and co-membered clusters share it. Partial correlation — conditioning
each pair on the others to remove the common-mode global commit-rate — computed **within each counterbalanced
condition, then averaged**, isolates co-membership and is blind to abundance.

**Pooling matters, and P2 proved it** (synthetic, `sweep/po11_valence_synthetic_check.py`, 2026-07-24; chance 0.5):

| readout | `full` | `scramble` | |
|---|---|---|---|
| partial-corr, **pooled** across conditions | 1.000 | **0.597** | leaks |
| partial-corr, **within** condition — REGISTERED | 1.000 | **0.527** | leak-immune |
| commit co-occurrence, within condition — cross-check | 1.000 | 0.528 | leak-immune |
| magnitude, pooled (PO-10 leak reference) | 0.997 | 0.598 | leaks |

Pooling across the counterbalanced conditions reintroduces the abundance leak (scramble 0.60); computing partial
correlation *within* condition removes it (scramble ≈ chance). **Registered readout: within-condition partial
correlation**, with **commit co-occurrence** as the registered cross-check (both must clear the ladder). Decode
the pairing from the readout matrix, LOO-CV vs a shuffled-label null. Leak-immunity is **proven by the arms** —
`scramble`/`lamshort` → chance where the magnitude reference leaks.

## The ablation ladder (registered)

1. **Full** — reward on, binding on. The test.
2. **Binding off** (`collapse_independent`, `:998`) — no joint collapse → no shared coin → **chance**.
3. **Scramble** — domain membership shuffled, sizes preserved → shared-coin structure destroyed → **chance**.
   *This is the arm that proves leak-immunity:* magnitude survives it (1.000), partial correlation must not.
4. **λ-short** (λ_F=5, below Werner) — no cross-bonds → **chance**; ties the result to the PO10-1 physics.
5. **Reward off** (full binding, dopamine absent) — no commitment written → partition-specific signal
   **vanishes**. The negative baseline that earns the word *valence*: it shows **reward is the gate that
   writes the credit**, not merely that the partition commits. (Replaces the coded-in — hence circular —
   reward-vs-no-reward *directionality* contrast, which is not run.)
6. **Magnitude decoder (reference, registered to LEAK)** — the PO-10 single-trial magnitude readout, on the
   *same* runs. Registered prediction: it lights up on `scramble`/`lamshort`. The **contrast** — partial
   correlation clean where magnitude leaks, on identical data — is the result.

## Verdict rule (registered BEFORE running)

**POSITIVE** iff the partial-correlation decoder (i) decodes `full` above the shuffle-null p95, **and** (ii) is
at chance on **binding-off, scramble, λ-short, AND reward-off** — the full ladder, including the arm magnitude
fails. Report bootstrap 95% CIs (the PO-10 "detected vs robust" discipline).

**LEAK / NEGATIVE** if the partial-correlation decoder also lights up on `scramble`/`lamshort` — then
reward-directed credit is riding abundance, not binding, and the valence bridge is a confound.

**NULL** if `full` does not decode — the reward path does not preserve co-membership at all.

## Preconditions (pilots — INVALID + unscored if any fails)

- **P1 — off-path readout live.** The harness's off-path reward-directed readout produces nonzero, reward-signed
  Δw from the collapse structure under drive (it reads the tracker's committed domains, as Unit C's signed readout
  did). *(The model's `apply_reward_correlated` is NOT used — it is a dead stub.)*
- **P2 — scorer validated on SYNTHETIC. ✅ PASSED 2026-07-24** (`sweep/po11_valence_synthetic_check.py`):
  within-condition partial correlation decodes a known synthetic partition (`full` 1.000) and is at chance on a
  shuffled one (`scramble` 0.527), while the magnitude reference and the *pooled* partial correlation both leak
  (scramble ~0.60). This set the registered readout (within-condition, not pooled) before any physics is spent —
  the L·PO5-3 lesson (a scoring bug once destroyed 58 min of physics).
- **P3 — leak reproduced.** The magnitude decoder leaks on *these* runs (reproduce PO-10 on identical data), so
  the clean-vs-leak contrast is within-experiment, not across experiments.

## Honest limits (registered)

- This is the **to-what-end bridge at the weight level**, NOT behavioral learning. It does **not** run the water
  maze or close the loop. It asks only: can reward direct credit onto the partition, verifiably and leak-immune?
- **Directionality (reward → +) is built into `apply_reward_correlated` — not a discovery.** The load-bearing,
  non-circular claim is **partition-specificity under a leak-immune readout**, which is not built in.
- Still the **(A)** model — no non-classicality claim.
- **If POSITIVE:** reward lands credit on the right groups, leak-immune → the maze's "reward strengthens the
  goal-predictive features" is mechanistically supported, and step 1 of the maze path is cleared.
- **If LEAK/NULL:** the existing reward path strengthens by confound → the valence needs a different mechanism
  before any closed loop can learn. That is itself the finding that unblocks (re-scopes) the maze work.

## Harness plan (off-path, composed from Unit C)

- Drive + arms: reuse `sweep/po10_unitC_experiment.py` (4-cluster matched-marginal drive, counterbalanced
  fwd/rev, the arm switches). Add `--readout {collapse_signed, reward_correlated}` selecting the readout path,
  and `--reward {on, off}` for arm 5. Save per-trial per-synapse Δw (already emitted) for the offline scorer.
- Scorer: new `sweep/po11_valence_score.py` — partial-correlation-across-trials decoder + the magnitude
  reference decoder, LOO-CV vs shuffle null, per arm. Validated on synthetic (P2) before physics.
- **Compute:** Unit-C-class. Partial correlation is data-hungrier than raw covariance (it estimates a
  conditioned matrix), so budget **n ≥ ~20 per (pairing × order)**, above Unit C's 12. Smoke-test ONE draw
  before any batch; concurrency ≤ 4; results force-added under `results/po11_valence/`.

---

## AMENDMENT 1 + REAL-DATA VALIDATION — 2026-07-24 (registered)

**(1) The off-path readout's `reward` is the SIGN MODE, not a commit gate.** In Unit C's readout the
commitment is the per-domain P_S coin; `--fixed_sign` only sets the collapse *sign* (+1 = reward-directed
vs random ±1). There is no separate reward-gates-commitment knob to switch off. So **arm 5 is reframed**:
not "reward-off → chance," but the **random-sign reference** (`fixed_sign=False` = Unit C's un-rewarded
collapse) — predicted to ALSO recover the partition, establishing that the reward-directed (fixed) sign adds
*directionality* without costing recoverability. The ladder that must go to chance stays bindoff / scramble / λ-short.

**(2) The registered readout is validated on REAL physics data (no new physics), `po11_valence_score.py`:**

| arm (existing Unit C data, RANDOM sign) | within-cond partial-corr | magnitude (leak ref) |
|---|---|---|
| `full` (n=48)     | **RECOVERS** sep +0.965, null-p 0.000 | recovers +0.688, p 0.000 |
| `scramble` (n=24) | **chance** sep −0.215, null-p 0.994    | **LEAKS** +0.625, p 0.000 |
| `bindoff` (n=48)  | **chance** sep +0.090, null-p 0.423    | chance +0.010, p 0.273 |

Reproduces P2 on real data: within-condition partial correlation is **partition-specific and leak-immune**
(recovers `full`, chance on `scramble`/`bindoff`) where magnitude **leaks** (recovers `scramble`). **P3 (leak
reproduction) is discharged on existing data**; the scorer is validated on real physics output. This is the
RANDOM-sign data (the arm-5 reference — recovery via shared sign); the **fixed-sign arms are now in physics**
to close the valence-specific claim: that a directional (reward-fixed) sign is *still* recoverable via
within-condition partial correlation (recovery via shared *commit*, which may be a weaker signal — the reason
new physics is warranted rather than a re-score).
