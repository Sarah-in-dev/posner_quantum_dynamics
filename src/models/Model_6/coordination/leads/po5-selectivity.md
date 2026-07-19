# Lead: po5-selectivity (PO-5 · §8 keystone — pair-level selectivity) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** does which dimers bond depend on INPUT at pair
resolution? Pre-registered, null that cannot show the effect, positive control demonstrated to
fire, verdict able to return FALSIFIED.

**Status:** LIVE. Re-scoped by Sarah 2026-07-18 20:14Z — see `requests/po5-selectivity/mo-rescope-001.md`.
**Current unit:** UNIT 2 Q-B — **offline rebuild COMPLETE and VALIDATED (A2.7, zero compute).**
Slot **RE-REQUESTED** (`requests/model6-mo/po5-selectivity-008.md`), sequenced **behind PO-2**.
**🔓 SLOT STILL RELEASED — PO-5 holds no compute.**
**Last heartbeat:** 2026-07-19 00:44Z

**A2.7 — what ruling 028 required, delivered:** scorer separates **known FLAT → FALSIFIED (1.028)**
from **known PLANTED → CONFIRMED (9.128)**. Fixed **global** lattice (absolute cell coords,
all-run intersection) retires L·PO5-3 flaw 1 at the root. Scoring split into
`sweep/po5_unit2_score.py`, composed from `score_leta5.py`; probe persists cells+pairs **after every
run**; end-to-end verified probe → JSON → scorer → verdict. Scorer **refuses** real data unless
planted-vs-flat separates first.

**DISCLOSED — the moment tuning could have entered:** the first planted control (amp 0.60) scored
**2.905**, just under `RATIO_CONFIRM = 3.0`. **The threshold was NOT lowered.** A positive control on
the decision boundary tests nothing, so the *control* was strengthened to amp 2.0; every registered
threshold stands.

**DETECTION FLOOR = 0.80** (amp→ratio: 0.2→1.37, 0.4→2.08, 0.6→2.91, **0.8→3.77**, 2.0→9.13).
**This is the gain from the failed run:** a future FALSIFIED is no longer a bare negative but the
bounded claim *"no input-driven pair-level effect ≥ ~0.8 in `P_bond` on a 2-cell block, 6 nm cells,
under these conditions"* — **fixed before the data is seen**, so it cannot be back-fitted.

**Pre-flight change so L·PO5-3 flaw 2 cannot recur:** it asserts the **all-run intersection** across
a multi-seed sample, not one seed. If that is below `MIN_CELLS`, **PO-5 STOPS and reports that the
instrument cannot resolve pair structure in this geometry.** No threshold moves.

**WHAT THE SLOT BOUGHT, stated plainly: no verdict on §8.** All 9 runs completed; instrument
conservation PASS; A2.3 `_remove_dimer` tripwire PASS (zero calls); positive control `max_glu > 0`
PASS (min 1.000); drive matching PASS (A=2.7540 vs B=2.7460, 0.3%). **Then the scoring step raised
`ValueError: operands could not be broadcast together with shapes (169,) (36,)`.**

**THREE FLAWS, mine, and the third is the expensive one:**
1. **The statistic is not comparable across runs.** Cells were indexed by *each run's own* occupied
   set (`remap` = sorted occupied cells), so index *i* denotes a **different physical location** in
   every run. Frobenius distance between such matrices is meaningless even when the shapes happen to
   match. **This is a design flaw, not a coding slip.**
2. **Occupied-cell count varies 6–14 across seeds**; only 3 of 9 runs cleared `MIN_CELLS = 10`. The
   A2.6 pre-flight sampled **one** seed (13 cells) and was unrepresentative of the arm.
3. **I did not persist the scored intermediate**, so a scoring bug destroyed 58 minutes of physics.
   **PO-3 had already solved this** — `sweep/score_leta5.py` scores offline from a persisted trace.
   That pattern was in front of me and I did not compose from it. A no-reinvention miss.

**Next, and it needs ZERO compute:** rebuild the statistic on a **fixed global lattice** (absolute
cell coordinates, so a cell means the same place in every run), compare on the cell set occupied in
**all** runs, persist the matrices, and split scoring into a separate offline scorer. **Validate it
on synthetic data before requesting the slot again.**

**Not moving `MIN_OCC` or `MIN_CELLS`.** If the all-run intersection is below `MIN_CELLS`, the
honest answer is that the instrument cannot resolve pair structure in this geometry — which is a
finding about the measurement and is reported as one, per the registered hard stop.

**MY DEFECT, corrected:** this file read *"Q-B unrun, gated on the compute slot"* while Q-B was
running, so the MO correctly read PO-5 as idle for ~50 min off a stale heartbeat. **The MO offered to
take that as its own failure; it is mine** — the backbone said blocked, so blocked is what it read.
Heartbeat now updates at every STATE CHANGE, not every milestone.

**Q-B launch history — three starts, two stopped by PO-5 before any scoring:**
- **L1 killed @~11 min, 3/9 runs, unscored.** `cells = 4` vs `MIN_CELLS = 10` ⇒ verdict could only
  return INCONCLUSIVE whatever the physics did. Cause: `CELL_NM = 40 nm` against Unit 1's measured
  cloud `r_max = 36.45 nm` — **the bin was bigger than the object.** A2.4.
- **L2 aborted by its own pre-flight in 57 s.** `cells = 9` vs 10 at 8 nm. The A2.5 gate did its job,
  and exposed that it was gating a 1 s condition while scoring at 5 s. A2.6.
- **L3 RUNNING:** `CELL_NM = 6 nm`, pre-flight now gates the **scored** condition.

**Integrity constraint on the two cell-size moves:** selection rule fixed in advance (above
`r_p10 = 3.71`, below `r_p50 = 9.78`, clears `MIN_CELLS`), all bounds from Unit 1's prior geometry;
**no verdict computed at any cell size, ever** — nothing existed to select toward; no verdict
threshold moved. **Registered hard stop: if 6 nm also fails pre-flight, PO-5 STOPS and reports**
rather than stepping the value down a third time.

**Timing for MO sequencing:** ~290–370 s/run × 9 + pre-flight ⇒ **~50–60 min**, inside the ~90 min
estimate and well inside the 2× stop-trigger. A2.3 `_remove_dimer` tripwire armed, zero calls.

**UNIT 2 Q-A RESULT — `L·PO5-2` / DECISION RECORD `PO5-2`.** Pre-registered
`docs/PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md`; probe `sweep/po5_unit2_provenance.py`; **zero edits to
`dimer_particles.py`** (instance wrapping, since four POs share this tree and PO-1 is editing that
file). **The instrument gate FAILED FIRST on real data** — orphans 0→909→4851, cause traced to
`_remove_all_bonds_for_dimer` (`:245`) bypassing `_remove_bond`; AMENDMENT A2.1; post-fix both gates
pass, including **bit-for-bit** instrumented-vs-uninstrumented identity.

**MEASURED @2.0 s:** P0 birth-inheritance (`:218-228`) **82.86%** · P1 burst **0.00%** (22 bonds) ·
P2 EM **17.14%**. **83% of bonds never evaluate `em_rate`** — so the kickoff's `g`/`coh` decomposition
describes the *minority* mechanism, and Unit 1's `D = 33.5` applies to the 17%. P1 is shadowed by
construction (`p1` needs `~has_bond`; the birth loop already bonded those pairs).

**NOT CLAIMED:** that this defeats §8. Birth timing is downstream of input, so a deterministic birth
rule is not automatically input-blind — pair-level vs gate-level is Q-B, and Q-B is unrun. Deliberately
not repeating the inference ruling-010 caught in `L·PO5-1`.

**Routed to MO** (`requests/model6-mo/po5-selectivity-003.md`): does Q-B's target change now that the
bond set is 83% birth loop (recommendation: keep whole-set target, additionally report verdict split
by provenance) · and a latent defect — `_remove_dimer` (`:252-261`) never pops `_bond_lookup`,
currently **dead code**, reported not fixed (death path, not my surface).

**RULINGS ABSORBED:** `mo-ruling-001` (Pathway 1 in scope; the two 5.0 `coupling_length`s — checked,
my probe reads the nm one off the live object; `P_product` framing corrected) · `mo-ruling-010`
(Q3 = YES, the trivial-partition finding stays in my acceptance; Q2 closed) · `po1-6a-002`
(PO-1 edited `dimer_particles.py:288-289`, behaviour-identical, my regions untouched — accepted).

**SELF-CORRECTION, made this cycle:** `L·PO5-1` CORRECTION 1. My claim that the single connected
component meant the pair-resolution *"does not reach the topology"* was an INFERENCE, not a
measurement, and it read the intra layer against a network-layer standard.
`quantum-system-canonical:139` [LOCKED] makes single-synapse one-giant-component **correct physics.**
**All measured numbers survive; only the inference is withdrawn.** Caught by the MO, not by me — and
notably I had quoted §5's neighbouring lines in my own brief and reasoned past them.

**RAISED, not resolved:** `requests/model6-mo/po5-selectivity-002.md` §2 — §8 wants pair-level
selectivity, §7 #1 says single-synapse-scale, but §5 LOCKS one-component-per-synapse as correct and
puts the meaningful partition cross-synapse. **Those cannot all be operative as written.** Three
readings offered; recommending (a) the unbonded margin + (c) Pathway 1 birth structure, which keeps
Sarah's re-scope intact. **Not blocking — Unit 2 proceeds under (a)+(c).**

**UNIT 1 RESULT — `g` is LIVE, not inert.** Pre-registered `docs/PREREG_PO5_UNIT1_G_INERTNESS.md`
(`cc80fcc`, before the probe existed); probe `src/models/Model_6/sweep/po5_unit1_g_inertness.py`
(`1dbef17`); classifier demonstrated ABORTing before it was allowed to score. **`f_sat = 0.176`**
(registered saturation bar was ≥0.90), `r_p50 = 9.75 nm`, **`D = g_p90/g_p10 = 33.5`**, stable
across four samples. Log row **PO5-1**.

**Both priors were refuted, including this PO's own.** The board predicted inert-by-saturation
(`board.md:919-922`); PO-5's brief predicted inert-by-vanishing off the 400 nm birth domain. Dimers
cluster at templates (`dimer_particles.py:189-196`), so the brief was wrong by ~15× in `r`.
`model6-entanglement-partition-werner:60`'s *"~7 nm"* is the prose that was right — **no correction
owed to that skill**, and the tension the brief flagged resolves in its favour.

**The finding that matters, and it relocates the keystone's failure mode:** the graph `g` builds is
**0.75–0.83 saturated, one connected component, `largest_frac = 1.000`**. A rate varying 33× across
pairs yields a near-complete graph with a trivial partition. Pair-resolution in the RATE that does
not survive into the TOPOLOGY buys §8 nothing. **UNVERIFIED and not claimed:** which pathway causes
the saturation — Unit 2.

**Open, non-blocking:** `queue/po5-selectivity.md` Q2 (three MO-owned artifacts carry the refuted
inertness framing) and Q3 (does the trivial partition sit inside PO-5's acceptance? — proceeding on
"yes").
**Blocked on:** nothing. **The old η/partition gate is RETIRED — PO-5 needs no backbone.**

**Scope change:** `MO_MODEL6.md` §3 scoped PO-5 to selectivity through the partition, gated on η
reaching threshold. §8 never asked for that (it mentions η nowhere) and its owning section says
the keystone is **single-synapse-scale, needs no backbone.** The `P_product` fallback is also
retired — it is the gate-level case §8 rules insufficient.

**First unit:** the `g`-inertness check. `coupling_length = 5.0 nm` and `g` saturates at 1.0 below
that, so measure intra-synapse `r_ij` — if most pairs are under 5 nm the 1/r³ term is inert in
practice and Pathway 2 is flat-rate by a different route.

**Live, MO-verified:** the 1/r³ IS implemented (`dimer_particles.py:451-455`), so
`quantum-computation-and-attribution` §7 #1's "no J_ij" claim is STALE — MO owes that skill a fix.

**Carried:** `mo-f3-001.md` (read the MO CORRECTION, not the superseded top) · F-4 — L·ETA-4's
NMDAR half is vacuous, do not build on it.

---
---

# 🔻 CLOSING HEARTBEAT — PO-5, 2026-07-19 00:50Z. Seat retired. Read this before touching §8.

## 1. STATUS — where Q-B actually stands

**§8 Keystone #1 is UNVERIFIED. No verdict exists, in either direction.** Nothing in any PO-5
artifact licenses a claim that the keystone holds or fails.

| unit | state |
|---|---|
| **Unit 1** — `g`-inertness | **COMPLETE.** `g` is **LIVE** (`f_sat = 0.176`, `D = 33.5`). `L·PO5-1`. One inference withdrawn (CORRECTION 1). |
| **Unit 2 Q-A** — bond provenance | **COMPLETE.** P0 birth **82.86%** / P1 **0.00%** / P2 EM **17.14%**. Both instrument gates pass. `L·PO5-2`. |
| **Unit 2 Q-B** — the keystone | **ATTEMPTED, NO VERDICT.** Ran 58.2 min, 9/9 runs, every gate passed, scorer was built wrong. `L·PO5-3`. |
| **Q-B rebuild** | **COMPLETE + VALIDATED, never run on physics.** A2.7. |

**PO-5 holds no compute. The slot was released 23:25Z and never re-taken.**

## 2. TO RUN Q-B — the exact recipe

```
# 1. run the matrix (~60 min, exclusive heavy slot). ALWAYS cd first — see trap T1.
cd /Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6
nohup /Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python -u \
  src/models/Model_6/sweep/po5_unit2_qb_selectivity.py > <logfile> 2>&1 &

# 2. score OFFLINE — the probe no longer scores, by MO ruling 028
/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python -u \
  src/models/Model_6/sweep/po5_unit2_score.py \
  src/models/Model_6/sweep/po5_unit2_qb_results.json
```

- **The scorer self-validates before it will touch real data** (planted → CONFIRMED 9.128, flat →
  FALSIFIED 1.028). If that gate fails it aborts. **Do not bypass it.**
- **The probe persists cells+pairs after EVERY run.** A scoring bug can no longer cost physics —
  that rule cost 58 minutes to learn.
- **Fixed global lattice**: cells keyed by **absolute** coords. Never index by a run's own occupied
  set (that is what killed the first scorer).
- **Cost:** ~290–460 s/run × 9 + a pre-flight ⇒ **~60 min**.

### ⛔ `MIN_CELLS = 10` MUST NOT BE RELAXED
Also `MIN_OCC = 5`, `RATIO_CONFIRM = 3.0`, `RATIO_FALSIFY = 1.5`. **Expect pressure to lower
`MIN_CELLS`** — individual runs gave only **6–14** occupied cells, so the **all-run intersection may
well land below 10.** If it does, **the correct output is "the instrument cannot resolve pair
structure in this geometry at 6 nm cells"** — a finding about the measurement, reported as one. That
is a registered hard stop (A2.6) and MO-binding (ruling 028). Lowering it to obtain a verdict would
manufacture one.

## 3. WHAT EXISTS NOWHERE ON DISK — the irreplaceable part

### 3a. ⚠️ TWO THINGS IN MY OWN ARTIFACTS ARE WRONG OR WILL VANISH — fix these first

**(i) The multi-seed pre-flight I promised is NOT IMPLEMENTED.**
`requests/model6-mo/po5-selectivity-008.md` says the pre-flight will assert the **all-run
intersection across multiple seeds**. **It does not.** `po5_unit2_qb_selectivity.py:323` still reads
`pf = run_arm("A", 999, T, dt, samples, log)` — **one seed.** That single-seed pre-flight is exactly
what produced L·PO5-3's flaw 2 (certified an arm at 13 cells whose runs actually ranged 6–14). **A
successor reading 008 would believe this is done. It is not. Implement it before spending the slot.**

**(ii) Every raw log I cite is UNTRACKED and will be lost.** `results/` is **gitignored**
(`.gitignore:30 *.log`, `:63 *.json`). `git ls-files results/po5/` returns **zero**. The four failure
logs my log entries cite as evidence — `unit2_qb_KILLED_cell40.log`,
`unit2_qb_PREFLIGHT_FAIL_cell8.log`, `unit2_qb_SCORER_CRASH_cell6.log`,
`unit2_provenance_FAILING_v1.log` — **exist only in this worktree and are not committed.** The
committed JSONs under `sweep/` survive only because they were **force-added** (`git add -f`). If
those logs matter to anyone, force-add them now; otherwise the L·PO5-3 narrative loses its evidence.

### 3b. §8 OVERCLAIM RISKS — the ones I was pressed toward and refused

**THE BIG ONE. "83% of bonds come from a deterministic rule with no input term, therefore the graph
is input-blind, therefore §8 fails." THIS IS WRONG. DO NOT WRITE IT.** It is the most seductive
sentence available in my results and I declined it twice under direct pressure; MO ruling 019
explicitly barred the MO from shortcutting it. **Why it is wrong:** the birth loop
(`dimer_particles.py:218-228`) is deterministic *in its pairing rule* — given which dimers exist,
where, and when, it bonds every template-bound pair born within 100 ms. **But which dimers are born,
where, and when is downstream of calcium, which is downstream of input.** The determinism is in the
pairing, **not in the population** — and the population is precisely where input can enter. Whether
that carries **pair-level** or merely **gate-level** information is the unrun question. *Nothing is
known about it.*

**`comps = 1` IS NOT A FAILURE.** `quantum-system-canonical:139` [LOCKED]: *"A single-synapse 'one
giant component' is correct physics, not a bug."* **I made this exact error** and was corrected by
ruling 010; the withdrawal is `L·PO5-1` CORRECTION 1. The intra graph being one saturated blob is
the predicted result at single-synapse scale, not evidence against the keystone.

**`D = 33.5` sounds stronger than it is.** Unit 1's 33× dynamic range in `g` is real — and applies to
the **17%** of bonds that go through `em_rate`. **83% never evaluate it.** Quoting `D = 33.5` as
"the bond rate is strongly pair-resolved" without that qualifier misrepresents the system.

**The saturation decline 0.944 → 0.606 over 30 s is MEASURED; its CAUSE IS NOT.** It is *consistent
with* a birth-blob eroding into a distance-shaped graph. **That mechanism was never measured** and I
labelled it explicitly not-claimed. Do not upgrade it.

**The detection floor 0.80 is in SYNTHETIC units, not physical ones.** It is an amplitude added to a
0.30 baseline on a 2-cell block of a 16-cell synthetic lattice. It bounds **the scorer**, calibrated
on synthetic geometry. **It is not a physical bond-probability bound and real-data sensitivity may
differ.** State it as an instrument property or not at all.

### 3c. TRAPS THAT COST ME TIME

- **T1 — the shell cwd resets between calls.** A command chain beginning without `cd` runs in the
  **wrong worktree** (`xenodochial-rubin-cad5db`, ~20 commits stale). It cost me a launch: `cp`
  failed, `&&` short-circuited, and **nothing ran and nothing committed while appearing to succeed.**
  **Prefix every command with the explicit `cd`.**
- **T2 — two `sweep/` trees, and my code spans both.** The probe lives in
  `src/models/Model_6/sweep/` but imports `presynaptic_release` from the **repo-root** `./sweep/` via
  `sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))`. Move either file and it breaks. The
  corroborating `observe_pathway2_selectivity.py` is also repo-root.
- **T3 — the `unknown` provenance bucket is a silent blind spot.** It was 0 in Q-A only because the
  instrument wraps **all three** bond-creation sites. **Add a fourth creation site and `unknown`
  absorbs it while conservation still passes.** Re-audit `_create_bond` call sites before trusting
  any future provenance split.
- **T4 — instance-level wrapping is lost on re-init.** The instrumentation patches the *instance*.
  Deep-copy or re-initialise the network and the wrappers silently vanish, leaving provenance empty
  and conservation trivially "passing" on an empty map.
- **T5 — `_remove_dimer` is confirmed unreachable BY MEASUREMENT, not just by grep.** The A2.3
  tripwire logged **zero calls across a full 9-run protocol.** That is stronger evidence than the
  call-site grep and is worth keeping when PO-7 addresses the defect (`:252-261` discards from
  `entanglement_bonds` without popping `_bond_lookup`).
- **T6 — paired seeds are deliberate.** Arms A and B share seeds 101–103 (both `np.random` **and**
  the `PresynapticRelease` object); the NULL uses 201–203 for both. That pairing is by design — do
  not "fix" it into independent seeding without re-deriving what the null then measures.

### 3d. THE ONE JUDGEMENT I WOULD PASS ON

The design question I was working when the seat closed, unresolved and worth more than the re-run:
**given that 83% of bonds come from birth-pairing, is a drive-matched INPUT-A/B *timing* contrast
actually the sharpest test of pair-level dependence?** A same-total-drive/different-timing contrast
shifts **birth cohorts**, which is the P0 mechanism's own input channel — that may discriminate §8's
pair-level from gate-level far more directly than the current spatial-residual statistic. **I would
settle that before spending another hour of slot.**

**Final word: the honest state of §8 after a full day on it is UNVERIFIED, with a validated
instrument that has never been pointed at physics. That is a smaller result than anyone wanted, and
it is the true one.**
