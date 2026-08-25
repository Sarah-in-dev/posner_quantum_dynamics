# Session Handoff — 2026-08-25 · MOMO: the network blocker (biology) + three benchmark negatives (primitive)

**For the next MOMO. Two tracks run in parallel here: the BIOLOGY (Model 6) and the PRIMITIVE (the extracted
computational abstraction). This session produced a hard blocker on one and three clean negatives on the
other. Both are real results. Neither is a reason to restart from scratch.**

## 0. One line — and the GUARDRAIL that must survive

**Biology:** the F4 drive protocol produces **~40× too little metabolic power** to condense the backbone
(r=0.039 vs the r>1 required), so cross-synapse entanglement is absent from every F4 run. **This is a
CONFIGURATION failure, not a physical verdict** — see the CORRECTION box below.
**Primitive:** consolidated into one implementation (7/7 regression), then **lost three benchmarks in a row**,
each with the cause identified and each apparent win **retracted against a control**.

> **GUARDRAIL (Sarah, restated all session):** *ground in biology, maintain emergent behavior, and enable the
> system to produce.* GROUND, do not TUNE. **A null is a result.** Validate at the data/result level —
> "it ran / errors=0" is not validation.

> **THE CONTROL RULE (new, and it is the most important thing in this handoff):** before claiming ANY win,
> run the control that could explain it without your mechanism. This session, two apparent wins evaporated
> that way. Report the control's number next to yours, always.

### >>> CORRECTION (2026-08-25, after the handoff was first written) <<<

The first draft of this handoff claimed cross-synapse entanglement "has never once been reachable in this
codebase". **That is WRONG, and it was caught by recovering an orphaned commit (`4b4cc2d`) while checking
whether `git worktree prune` was safe.** The evidence:

- `sweep/po7_unit8_eta2_partition.py` header: **`NMDAR open 0.0000 -> 0.3806, r 0.3509 -> 1.6234,
  eta 0.0000 -> 0.2376`** — the script's own acceptance criterion is that r reaches ~1.62 and eta ~0.238.
  **Condensation is REACHABLE.**
- Sarah's recovered Q7-1 correction (4 runs, 2 synapses, 45 s, seed 7): `eta_max` **0.0487–0.1069**,
  `cross_bonds` **1179–2578**. Non-zero eta and thousands of cross bonds, in that configuration.

So F4-b's `r = 0.039` is **~40× below a configuration that demonstrably condenses**, and the blocker is the
**F4 DRIVE PROTOCOL**, not Fröhlich condensation per se. `eta_native = (r-1)/(r+1) if r >= 1 else 0.0`
(`po7_unit8_eta2_partition.py:106`), so r<1 clamps eta to exactly 0 — which is all F4 ever saw.

**This also changes the §4 recommendation: option (A) is no longer "re-derive P_c from scratch". It is the
much cheaper DIFF — what does the PO-7/eta_probe rig drive that the F4 harness does not?** Start at
`compute_metabolic_power(E_invasion, ca, p_active_max_W)` and compare the NMDAR open fraction (0.3806 in the
condensing rig) against what F4's `--drive-mv -20 --elig-s 25` actually produces. Also note Sarah's caveat,
preserved: whether the backbone condenses **is not reproducible at a fixed seed** (the range includes zero),
so any comparison needs N runs, not one.

## 1. Read-order (ground first — do not skip)

1. `src/models/Model_6/docs/README.md` — the documentation map (canonical vs stale).
2. `session-discipline` and `agent-grounding-protocol` skills, in full.
3. This handoff.
4. Biology: `RESEARCH_LOG_CALCIUM_DIMER.md` entry **F4-b — 2026-08-24** (the blocker) AND its
   **CORRECTION** appended 2026-08-25. Read both; the first framing was wrong.
   Then `coordination/leads/po7-construct-validity.md` — final section, the recovered Q7-1 correction.
5. Primitive: `src/coherence_gated_learning/` — `RESULT_nonstationary_bandit.md`,
   `RESULT_contextual_bandit.md`, `RESULT_network_primitive.md`, in that order.
6. Code: `cgl_primitive.py` (the consolidated primitive), `network_cgl.py` (the network version),
   `test_primitive_regression.py` (7/7 — run it first, it is the ground truth).

## 2. SETTLED — do NOT relitigate

- **The coherent P_S *is* the eligibility trace, read AT REWARD.** The fast binding-melt measurement is fine
  and does NOT need to be "held" across the delay. **Two separate threads have destroyed a session by getting
  this backwards.** Read `reward_gating.py` before touching anything here.
- **Per-component collapse is a CONSOLIDATION rule, not an action-selection rule.** It decides whether *this*
  memory commits. It degrades to near-uniform when candidate values are close (measured: P(best)=0.384 vs
  chance 0.333 with *perfect* value estimates supplied). Action selection among competing alternatives is a
  different circuit (striatal competition). Do not re-derive this the hard way.
- **The graph only computes something when grouping is determined by TIMING.** Verified: 22 features presented
  simultaneously → **one** component of size 22 (identical to a hand-built key); the same units separated in
  time → **three** components. Static tabular rows and context-free bandit arms are no-ops *by construction*.
- **eta = (r−1)/(r+1), so eta = 1.0 requires infinite metabolic power.** Do not run the primitive at eta=1.0
  and report the giant-component blob as a finding; it is an artifact of an unreachable parameter.

### RETRACTED this session — do not resurrect
- **"CGL beats SW-UCB 4×"** — plain recency-weighted greedy with *no CGL* scored the identical 82.9. On a
  context-free bandit every arm is a singleton, so `value()` is a leaky average and `argmax` is greedy.
- **"The spin ledger never fires"** — true only for a SINGLE presentation. Under repeated presentation it
  fires 27k–95k times and *dominates*. A regime-specific result was generalised; both regimes are real.

## 3. DONE this session (all committed and pushed to master)

**Biology**
- `ea26189` — **F4-b KILLED.** 8 shards × 24.4 h produced **zero completed runs**. Diagnostic stable across
  all 308 samples: `P_met=0.84 fW  P_c=21.51 fW  r=0.039  eta=0.0000  invaded=True`.
  The longer drive (`--elig-s 25 --drive-mv -20`) **did** fix F4-a's invasion bug (`invaded=True`), but
  condensation needs `r>1` and r is pinned at 0.039–0.078 — a **~25× power deficit**. `r` is a ratio of
  POWERS, not an accumulating quantity, so runtime cannot fix it. Evidence salvaged to
  `results/f4_specificity_ec2/sweep_f4b_2026-08-24.log`. **EC2 `i-0a8c5aa3d33db595f` STOPPED** (not
  terminated — disk intact; restart with `aws ec2 start-instances`).

**Primitive**
- `ddf8a18` — consolidated **all seven mechanisms into `cgl_primitive.py`** (they had been spread across five
  partial implementations, none holding more than four). `test_primitive_regression.py` re-runs every original
  protocol against it: **7/7**. Also separated `commit_gain` (readout, 3.0) from `select_gain` (selection, 1.6),
  a conflation that had been hidden by both being called "gain".
  Scope correction: `bind_window` is load-bearing *only* with partial reactivation off — with it on, accuracy
  is identical but the store blows up **21 → 547 keys (26×)**. Binding buys COMPACTNESS; partial reactivation
  buys ROBUSTNESS to bad binding.
- `7dd404e` — **Benchmark 9 (non-stationary bandit): NEGATIVE.** CGL 1163/1321 vs SW-UCB 333, Disc.Thompson 433
  — and the baselines were handed the true breakpoint count. Two grounded fixes (adaptive baseline; consolidation
  sharpening) both **failed**. Apparent WTA win **retracted** (see §2).
- `1098e72` — **Benchmark 10 (mushroom): NEGATIVE, and CGL's mechanisms actively HURT.** Ablation:
  conjunctions-only running mean **2581** < leaky/LTD 4534 < full CGL 17228. The 2–4× win over linear belongs to
  "use feature conjunctions", not to us. Stationary data rewards neither active depression (nothing to revise)
  nor partial reactivation (exact evidence nearly always exists).
- `52f4caf`, `5a496f6` — **`network_cgl.py`**: spatial coupling `exp(-d/5.0µm)`, bounded degree (4 ³¹P nuclei,
  refusal = frustration), global gate. **Finding: eta is a continuous GRANULARITY knob** setting the ORDER of
  representable conjunctions (~0.1 pairs, 0.33 4-way, 0.6 6-way) — a concrete computational role for Fröhlich
  condensation, from the model's own rate law. `eta=0` reproduces F4-b exactly (22 singletons).
  **But on task it FAILED:** 47052 regret vs 2581 hand-picked, against an always-abstain baseline of 51800.

## 4. THE FRONTIER — three options, pick deliberately (do NOT default to #2)

**(A) BIOLOGY — re-derive `P_c` and `P_met`. This is the highest-value move and it is a DESK CALCULATION,
not compute.** A 25× gap is one of: a parameterisation error, a genuine physical verdict on Fröhlich
condensation at physiological power, or a power source missing from the model. **Until this is settled, no
network run can produce network behaviour**, so every multi-synapse experiment is blocked. Start at
`spine_plasticity` / the Fröhlich block, and at `P_c = 21.51 fW` — where does that number come from, and what
would have to be true for `P_agg` to reach it?

**(B) PRIMITIVE — persistent units with FIXED RECEPTIVE FIELDS.** Measured cause of the network failure:
**67.1% of keys are seen exactly once** (mean reuse 2.56×), because units are feature-VALUES (117) rather than
feature-SLOTS (22), so binding yields value-conjunctions that never recur. A synapse in the biology is a
PERSISTENT structure with a FIXED input source, not a transient value-group. Letting the graph decide which
*slots* group would fix recurrence directly. **This is grounded and probably right — but it would be the
FOURTH consecutive fix-and-retry, and that pattern needs a human decision, not momentum.**

**(C) STOP EXPANDING AND WRITE UP.** What is genuinely established is more than it currently feels like:
conjunctive credit from a single delayed scalar where per-feature trace learners are *provably* at chance;
capacity to K=64; adaptation to unsignalled change with no exploration parameter; and now eta-as-granularity.

## 5. Discipline (LOCKED)

- Surgical edits, one validated step at a time. **Never `datetime.utcnow()`** — use `datetime.now(timezone.utc)`.
- Long batches: self-daemonize (double-fork + `os.setsid`; macOS has no `setsid` binary), verify ppid=1, then poll.
- **Run `test_primitive_regression.py` before and after any change to `cgl_primitive.py`.** 7/7 or stop.
- Sarah's env is broken for some packages — use `/Users/sarahdavidson/miniforge3/bin/python3`, do not modify her venv.
- Ask decisions in plain chat text. Never the AskUserQuestion pop-up.

## 6. FAILURE MODES from this session — do NOT repeat

1. **Two benchmarks were chosen that were structurally incapable of testing the mechanism.** Check FIRST
   whether the task has temporal structure; if everything arrives at once, the graph is a no-op and the
   benchmark cannot say anything.
2. **Baselines were given exploration devices and CGL none** → it deadlocked into always-abstaining. The tell
   was two different representations returning *byte-identical* regret. Identical numbers = bug, not result.
3. **A guess was treated as evidence.** Partial reactivation reports "evidence" for any overlapping component,
   which silently neutralised the optimism that prevents deadlock. Gate optimism on the EXACT key.
4. **A regime-specific finding was generalised** ("frustration never fires"). Check both regimes before claiming.
5. **An apparent win was nearly reported before the attribution control ran.** It was caught, but only just.
   Run the control FIRST.
6. **Four consecutive fix-and-retry cycles.** Each fix was grounded and each found a real cause one level
   down — but momentum is not evidence of convergence. Surface the pattern to Sarah rather than continuing.
