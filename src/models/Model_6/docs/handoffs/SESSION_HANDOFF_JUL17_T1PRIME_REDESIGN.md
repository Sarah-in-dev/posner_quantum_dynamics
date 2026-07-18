# SESSION HANDOFF — 2026-07-17 — T1′ redesigned, wide-ladder run LAUNCHED

**For a new thread.** Everything below is measured against code/data this session unless
tagged otherwise. Branch `claude/nervous-hertz-7ccff6` is **PUSHED** through `64711d4`;
three later commits (research log, replication, ERR-1 correction) are **local-only**.
Upstream `origin/master` is untouched at `029f52b`.

**STATUS: T1′ IS COMPLETE AND CONCLUSIVE — 4/4 seeds, far-pairs-first, p≈3×10⁻⁶.**
Nothing is running. See §7 for the result table. The authoritative record is the research
log `docs/RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` (entries T1'-1…T1'-5, dt-1, ERR-1) —
read that before this handoff; this doc is the session narrative, the log is the provenance.

Companion: `SESSION_HANDOFF_JUL16_TOPOLOGY_DT_FIX.md` (the failed run + the dt fix). This
handoff supersedes that one's §3 REDESIGN plan — **that plan was wrong; see §2 here.**

---

## 1. WHERE T1′ STANDS — **DONE. CONFIRMED. CONCLUSIVE.**

T1′ asks: as coherence decays, does the entanglement partition fragment FAR-PAIRS-FIRST
(largest gap first)? It is the discriminating experiment — a classical scalar eligibility
trace decays uniformly and carries no spatial structure, so it CANNOT produce a
spacing-ordered cascade.

**ANSWERED: yes.** Four independent seeds, all four breaking in the exact pre-registered
order 3.35 → 2.90 → 2.45 → 2.00 µm. p = (1/24)⁴ ≈ 3.0×10⁻⁶ against the classical null.
Full table and the confound argument in §7 and in research-log entry T1'-5.

**Altitude (do not inflate):** this is the **(A)** reading — the *model's* partition
carries spatial structure a classical scalar trace cannot. It is a discrimination win for
the theory of expected operation. It is **NOT** a quantumness claim; the attribution gap
(`quantum-computation-and-attribution` §5) is untouched. Sarah's locked frame: *"we will
never be able to model quantum in a silicon system, but we can model how the system expects
to work."*

---

## 2. THE JUL-16 REDESIGN PLAN WAS WRONG — MEASURED, NOT ARGUED

The Jul-16 handoff §3 said: put the gaps JUST UNDER d*(0)=3.45um (3.4/3.3/3.2/3.1) so the
cascade "lands early, while the population is >90% alive." **Both halves of that are
false, and I built to it before catching it — the grounding brief did not flag it.**

- **d*(0) MEASURED = 3.4521um median** (n≈2200, `measure_dstar0.py`), population spread
  **3.387–3.466um** (min–max). The prompt's 3.45 was right; now it is measured. The
  spread is the new fact: d* is a distribution, not a number.
- **Tight gaps do NOT break early.** An edge survives while ANY bonded pair clears F>0.5,
  so the governing radius is `d*_eff = 5·ln(max_pair(P_S²)/0.5)` — the extreme TAIL, not
  the median. The tail decays **~3.6× slower** than the median. In the 8 s sanity run the
  median radius fell below ALL four gaps (3.04um at t=7.5s) while ALL four edges were
  still alive. Zero breaks in 8 s. The null killed the plan.
- **Tight gaps also destroy the order signal.** Each synapse's tail is set by ITS OWN
  luckiest dimer, whose `T_eff` is frozen at creation (`dimer_particles.py:47-50`,
  `j_couplings_intra` drawn once, never reassigned). Between-synapse scatter in d*_eff is
  **~0.29um**. Rungs finer than that are decided by luck.

**Measured order-test POWER** (`order_power_probe.py`, 10 seeds via the validated d*_eff
replay, order correct = break times increase as gaps narrow):

| geometry | live gaps (um) | power | note |
|---|---|---|---|
| tight  | 3.35 3.25 3.15 3.05 | **6/10** | 0.10um rungs < 0.29um scatter |
| medium | 3.35 3.10 2.85 2.60 | **5/10** | still under-resolved |
| wide   | 3.35 2.90 2.45 2.00 | **10/10** | 0.45um rungs clear the scatter |

Chance for 4 rungs = 1/24 = 0.042. **Wider is better** (counterintuitive: it costs more
wall-clock because breaks land later, but the order becomes resolvable). The old
2.0–3.0um gaps the Jul-16 handoff called "the root error" were directionally closer to
right; their real flaw was the 35 s window, not the spacing.

---

## 3. THE PROBE (`sweep/coherence_fragmentation_probe.py`) — WHAT CHANGED

- **Geometry → WIDE** `[3.35, 4.5, 2.90, 4.5, 2.45, 4.5, 2.00]` (4 live gaps, 3 dark
  4.5um controls that can never bond). Chosen for the 10/10 power above.
- **The mis-derived time predictions are DELETED.** The old `PRED`/`PRED_FLOOR` dicts and
  the "predicted" verdict column are gone. `PRED_ORDER` is the whole prediction. Times are
  recorded, never scored (Jul-16 handoff §3: they are not analytically predictable here).
- **Guards intact:** `CONSECUTIVE_ABSENT=3` (flicker guard), `MIN_BREAKS=3` (verdict
  returns INCONCLUSIVE below this — it CAN fail).
- **`--seeds` and `--dt`** CLI. A t=0 geometry check reports any live gap that fails to
  form or any dark control that bonds (previously invisible). `d*_EFF` (from max P_S) is
  logged next to `d*_med` — the instrument gap that hid the tail-governs fact for 3
  sessions.

---

## 4. dt CONVERGENCE — the production step is fine for T1′, and the operating point survives

Prompted by the dt=0.01 speedup idea (rejected). Measured (`dt_convergence_drive.py`,
`dt_convergence_operating_point.py`):

- **P_S and the Werner edge set are dt-CONVERGED**: `d*_med=3.45`, `edges=5` at every dt
  from 1e-4 to 5e-3. So the ORDER test (which reads P_S + edges) is honest at the
  production dt=1e-3. Free re-confirmation of the static probe's edge count.
- **The dimer COUNT is NOT converged in the drive transient** — ~+38% at dt=1e-3 vs 1e-4,
  climbing monotonically. This is explicit-Euler error on stiff formation, NOT a new bug
  (the `fd83460` dissolution fix stands). dt=0.01 OVERFLOWS (count → 10k, 2 overflow
  warnings) — do not use it.
- **BUT the saturated OPERATING POINT is converged to ~5%**: sustained-drive plateau
  156.7um (dt=5e-4) vs 163.9um (dt=1e-3), +4.6%. At saturation formation balances the
  correctly-dt-scaled dissolution, so the transient inflation cancels. **The ~155um
  operating point is NOT a dt artifact.** (I initially over-extrapolated the 38% to the
  operating point; direct measurement corrected it to ~5%.)

**Consequence for the open D8/D14/D16/D17 re-read:** it is NOT reopened by dt — the
operating point stands on dt grounds. It remains open for the physics reasons in the
Jul-16 handoff (the ~3× shift, 47→155um). Physics call, Sarah's.

---

## 5. INSTRUMENTS (now in `sweep/`, ported from scratchpad, run from that dir)

- `measure_dstar0.py` — measures d*(0) and the P_S distribution in the rig. ~21 s.
- `dstar_eff_replay.py` — **the crown jewel.** Replays `step_coherence` in vectorised
  numpy (no network needed — P_S dynamics don't touch the network), 200 s of sim in ~5 s
  of compute. Emits d*_eff(t) as an UPPER BOUND on break times (fixed population ⇒ higher
  max ⇒ later breaks). VALIDATED: median P_S matches the real 8 s run to <0.0005. Its
  P_S→d*_eff mapping is UNVALIDATED by construction (no honest cascade datum exists — the
  only candidate, "gap 3.0 broke at 34 s", is the retracted FLICKER), so it is a bound for
  SIZING, never a prediction to score.
- `order_power_probe.py` — measures order-test power across geometries (§2 table). ~14 min.
- `dt_convergence_drive.py`, `dt_convergence_operating_point.py`, `dt_independence_tail.py`
  — the dt controls behind §4.

---

## 6. MISTAKES THIS SESSION (recorded so they are not re-inherited)

Same pattern the program keeps hitting: **inferring instead of measuring, treating a
tail-governed quantity as if the bulk governed it.**

1. **Built the Jul-16 redesign without catching that its premise was wrong.** The tail, not
   the median, governs edge survival; the grounding brief quoted the handoff's plan and did
   not test it. Caught by the 8 s sanity null.
2. **Validated `dstar_eff_replay` against the FLICKER** ("gap 3.0 @ 34 s") — the exact
   fabricated datum the Jul-16 session was retracted over. Quoted it in my own brief, then
   used it as ground truth. Removed; the replay now emits a bound, not a validated time.
3. **Stated the bound direction backwards** (called replay break-times a lower bound; they
   are upper bounds — fixed population ⇒ higher max ⇒ LATER breaks). Corrected.
4. **Over-extrapolated the 38% drive-transient count inflation to the operating point.**
   Direct measurement showed the saturated operating point is converged to ~5% (§4).
5. **Gave a wrong mechanism** for why the medium geometry underperformed ("later breaks
   noisier") — the wide geometry breaks latest AND has perfect power. The real driver is
   rung-width vs the ~0.29um scatter. Corrected off the data.

The guards and the "score ORDER only" discipline held throughout — they are why none of
these reached a false CONFIRMED.

---

## 7. THE COMPLETED RUNS — 4/4, all in the pre-registered order

Wide ladder `[3.35, 4.5, 2.90, 4.5, 2.45, 4.5, 2.00]µm`, 90 s silence, dt=0.001, under
`caffeinate` + `nohup`. Seed 0 ran alone (~2.7 h); seeds 1–3 ran in PARALLEL on separate
cores (~2.7 h wall for all three — parallelism cost ~nothing: each held ~355 s/sim-s, the
same rate seed 0 ran at alone, ~490 MB each).

| seed | gap 3.35 | gap 2.90 | gap 2.45 | gap 2.00 | verdict |
|---|---|---|---|---|---|
| 0 | 14.5 s | 32.5 s | 61.5 s | 78.0 s | CONFIRMED |
| 1 | 14.0 s | 37.0 s | 55.0 s | 82.5 s | CONFIRMED |
| 2 | 14.0 s | 42.0 s | 54.5 s | 71.0 s | CONFIRMED |
| 3 | 11.0 s | 32.5 s | 64.5 s | 82.0 s | CONFIRMED |

**Order invariant; times not** — gap 2.90 broke anywhere from 32.5 to 42.0 s across seeds.
That is the score-order-never-times decision vindicated in data, not argument.

**The population confound, and why replication closes it.** Breaks 3–4 land in a cratered
population (~2200 → <100 dimers), so their TIMES are confounded by dimer-loss. Their ORDER
is not: dissolution is spatially UNIFORM, so it drives edges toward dying *together*, not
in gap-spaced order. It cannot fake a consistent gap-ordered cascade across four
independent stochastic realizations; only the coherence/distance mechanism does.

**Guards worked live:** flickers were caught and NOT scored (seed 0 flickered on gap 3.35
at t=14.0 before its real break at 14.5). The verdict function that CAN return INCONCLUSIVE
returned CONFIRMED honestly — the opposite of the Jul-16 failure.

Logs are session-scoped scratchpad (`T1prime_wide_seed{0,1,2,3}_90s.log`) — **copy them if
you want them**. The durable record is research-log entry T1'-5.

---

## 8. OPEN ITEMS

0. **T1′ is CLOSED** — 4/4 confirmed, documented (research log T1'-1…T1'-5). Do not re-run
   it, re-tune it, or "improve" the geometry. If anyone wants more seeds the machinery
   takes `--seeds N M ...`, but the result does not need them.
2. **D8/D14/D16/D17 re-read** against the ~3× operating-point shift — physics call, NOT
   reopened by dt (§4). Sarah's.
3. **Physics ECS account.** The AWS credentials on this machine reach ONLY an unrelated
   (non-research) production account — zero posner/model6/quantum ECS clusters,
   task-defs, or ECR images, even inactive. The physics ECS (if it exists) is a SEPARATE
   account whose credentials are not on this machine. Deferred by Sarah. Do NOT stand up
   research compute on that unrelated production account. (Account/IAM specifics kept out
   of this doc deliberately — public repo.)
4. **Skills owing updates:** `model6-codebase-operations` still carries an EC2 section
   (flagged for reconcile) — but note the physics ECS above; large runs have been LOCAL,
   and the ECS story is a separate-account question, not "EC2 retired". Resolve once the
   physics account is located.
