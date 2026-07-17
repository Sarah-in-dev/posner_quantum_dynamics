# SESSION HANDOFF — 2026-07-16 — entanglement topology, the dt bug, T1′

**For a new management thread.** Everything below is verified against code/data this
session unless tagged otherwise. Six commits landed, all **LOCAL/UNPUSHED** (upstream
`origin/master` is still at `029f52b`). A run is **LIVE** — see §1 first.

---

## 1. THE T1′ RUN FINISHED AND **FAILED**. IT PRINTED A FALSE POSITIVE.

Nothing is running now. The run completed in ~2 h (7137 s).

**Its log says `VERDICT: far-pairs-first fragmentation CONFIRMED`. That is FALSE.
Do not believe it. Do not cite it.** Three compounding defects, all mine:

1. **Only ONE edge "broke"** (gap 3.0 @ t=34.0 s). The verdict's monotonicity check is
   `all(rows[i] >= rows[i+1] ...)`, which over a single row is **VACUOUSLY TRUE**. One
   point cannot test an ordering. The script confidently confirmed nothing.
2. **That one "break" was a FLICKER, not a break.** The log's own next row:
   ```
   34.0  ... edges 4   gap3.0@34.0s   <- "broke"
   34.5  ... edges 5                  <- came BACK
   ```
   Bonds keep forming under clamped η and the `step_coherence` noise walk lets P_S
   wander back over threshold, so edges flicker near d*. The probe's `alive` dict marked
   the FIRST absence as a permanent break and never un-marked it ⇒ **fabricated data**.
3. **The window was sized off a mis-derived time** (see §3), so 35 s could only ever
   catch ~1 of 4 breaks.

**FIXED IN THE PROBE (2026-07-17), so this cannot silently recur:**
- `CONSECUTIVE_ABSENT = 3` — an edge must be absent 3 consecutive samples to count as
  broken; returns are logged as `FLICKERS ... NOT breaks`.
- `MIN_BREAKS = 3` — the verdict now returns **INCONCLUSIVE** on fewer than 3 clean
  breaks instead of passing vacuously.

**NET SCIENTIFIC RESULT OF T1′ SO FAR: nothing. The order prediction is UNTESTED.**
The one real datum is weak and directional only: the widest gap (3.0) was the first to
lose its edge, at t≈34 s. That is consistent with far-pairs-first, and is not evidence
for it.

**Log (scratchpad, session-scoped — copy it if you want it):**
`/private/tmp/claude-501/-Users-sarahdavidson-posner-quantum-dynamics/53e45e43-3406-40d3-89a1-954119590b6a/scratchpad/T1prime.log`

---

## 2. WHAT LANDED (6 commits, unpushed)

| commit | what |
|---|---|
| `862481d` | vectorise the CROSS-synapse bond loop (23–43×, physics unchanged); fix both SOC docs' broken script paths; fix the signed/unsigned coboundary docstring |
| `c905575` | the Werner partition IS a coherence-set distance graph (CONFIRMED, pre-registered); RETIRE the SOC power-law sweep |
| `b2a546c` | correct my own wrong "dead `k_dissolution`" diagnosis + wrong cost claim |
| `fd83460` | **the real bug**: dimer dissolution was missing its `dt` |
| `40e9986` | vectorise the INTRA-synapse `step_entanglement` — **bit-identical**, 13× |
| `78b08ce` | T1′ pre-registration re-confirmed post-fix; 35 s window; floor demoted |

### The two results worth carrying forward

**(a) The Werner partition is a distance rule.** `F = P_S_i·P_S_j·w`, `w = exp(-d/5µm)`,
cut at `F>0.5` ⇒ **bond iff `d < d* = 5·ln(P_product/0.5)`**. At P_S≈0.998, d*=3.45 µm.
Two theorems fall out of the Werner bound, neither tuned:
- `w ≤ 1 ⇒ F ≤ P_S²`, so `F>0.5` needs **P_S > 1/√2 ≈ 0.7071** — a hard coherence floor.
- The radius **shrinks** as coherence decays ⇒ the partition must fragment far-pairs-first.

CONFIRMED pre-registered (`sweep/coherence_radius_probe.py`, ~43 s): 7/7 gaps called,
exact edge list, `betti0_cross=3`, `component_sizes=[3,3,2]`. **Retrodicts the Stage 3
chain/ring validation with no free parameters.**

**(b) The dissolution dt bug.** `ca_triphosphate_complex.update_dimerization` receives
`dt` and every FORMATION term carries it (L353/358/374), but DISSOLUTION did not
(L420-421), with the result added straight in (L427). `k_diss` is s⁻¹, so dissolution
ran **per-STEP not per-SECOND — 1000× too fast at dt=1e-3**. D8's isolated probe
(`sweep/phosphate_conservation_probe.py:103`) does it right; the live module was the
outlier. Caught by a dt-independence control: same 4 s of sim → dimers = **56/619/1501**
at dt=1e-3/1e-2/5e-2, while P_S and the Werner edge set were dt-INDEPENDENT.

**Verified consequences (Sarah approved landing the fix):**
- D14 ("SOC loop closed / self-limiting") and D17 ("BOUNDED, no runaway") **SURVIVE**.
  Sustained drive still saturates: 40.5→…→152.6 µM, increments +33→+7→+2.4.
- **BUT the operating point moves ~3×**: ~155 µM post-fix vs A3/D8's 47 µM, D16's 49 µM.
  D16's *"forms 49 µM (matches A3's 47)"* was **two errors cancelling** — 1000×
  over-dissolution against ~500× coherence protection (`(1-singlet_excess)` at
  P_S≈0.998). Remaining 155-vs-47 gap is legitimate: the probe models no coherence
  protection, the live path does.
- **OPEN, not done: D8/D14/D16/D17 need re-reading against the 3× shift.** A
  physics-conclusions call. Nobody has done it.

---

## 3. ⚠️ THE T1′ PRE-REGISTRATION IS MIS-DERIVED — READ BEFORE SCORING THE RUN

**The error (mine).** I derived the break times from the **median** P_S. But a
synapse-quotient edge survives while **ANY** bonded dimer pair clears F>0.5 — it is a
**max over hundreds of pairs**, i.e. the **extreme upper tail** of P_S, not the median.

`T_eff = 216/(spread_factor · template_factor)`, `spread_factor ≥ 1`,
`template_factor = 0.7` for template-bound (97.7% are) ⇒ **analytic ceiling
T_eff ≤ 308.6 s** (median measured 151.8, p95 209.9). The longest-lived dimers sit near
that ceiling and hold their edge open.

| gap (µm) | P_S_crit | median T=152 (**what the script says**) | ceiling T=309 (**upper bound**) |
|---|---|---|---|
| 3.0 | 0.9545 | 9.5 s | **19.3 s** |
| 2.8 | 0.9356 | 13.6 s | **27.7 s** |
| 2.5 | 0.9079 | 19.9 s | **40.4 s** |
| 2.0 | 0.8637 | 30.5 s | **61.9 s** |

**Observed at t=18.5 s: all 5 edges still alive** — median-based said gap 3.0 died at
9.5 s; ceiling-based says 19.3 s. The data is consistent with the ceiling, i.e. with the
corrected reading. The edges are behaving correctly; the prediction was computed off the
wrong statistic.

**What survives:** the **ORDER** prediction — widest gap breaks first — is the
discriminating claim (a classical scalar eligibility trace decays uniformly and carries
no spatial structure, so it cannot produce a spacing-ordered cascade). Order is robust
to which quantile governs. **Score the order. Do NOT score the times.**

**What it costs:** the 35 s window (sized off the median times) catches only **2 of 4**
breaks (~19.3, ~27.7 s). All four needs ~65 s ≈ 6.5 h.

**AND THE CEILING WAS WRONG TOO — three failed derivations, ending in the data.**
Predicted gap-3.0 break: median-based **9.5 s** → p95-based **~13 s** → ceiling-based
**19.3 s** → **OBSERVED 34.0 s**. Cause of the third failure: I ignored the
**multiplicative noise** in `step_coherence` —
`noise = 1.0 + 0.01*sqrt(dt)*randn()`, applied to `P_excess` every step. Over 34,000
steps that compounds into a **±5.8% random walk on P_excess**
(`0.01*sqrt(dt)*sqrt(n_steps) = 0.058`), which lets outlier dimers sit **ABOVE** the
analytic ceiling I had called a hard bound: deterministic P_S(34 s) ≤ 0.9218
(P²=0.8496, cannot clear gap 3.0's 0.9111), but +1σ of the noise walk gives P_S=0.9621
(P²=0.9256, clears). The physics is fine; my model of it was incomplete three times.

**⇒ THE BREAK TIMES ARE NOT ANALYTICALLY PREDICTABLE HERE.** They are an extreme-value
statistic over a noisy random walk across hundreds of pairs. **Stop trying to
pre-register times. Pre-register the ORDER only** — it is the discriminating claim, it
is robust to which tail governs, and it is what a classical scalar trace cannot produce.

### THE REDESIGN — this is the actual next job

The 2.0–3.0 µm gaps were the root error: they force the cascade out to t≈34–110 s, where
**both** confounds bite (population collapsing, noise walk dominating). Fix the geometry
so the whole cascade lands **early**, while the population is >90% alive:

- **Put the gaps just under d\*(0) = 3.45 µm** — e.g. **3.4 / 3.3 / 3.2 / 3.1 µm**
  (P_S_crit ≈ 0.988 / 0.978 / 0.968 / 0.958). All four then cross threshold in the first
  seconds, in order, while the population is barely touched. Verify d*(0) against the rig
  first (`coherence_radius_probe.py` measures P_S; d* = 5·ln(P_product/0.5)).
- **Keep the 4.5 µm dark controls** — P_S_crit=1.109 > 1, they can never bond, and they
  independently retrodicted the static probe.
- **Keep the flicker guard and MIN_BREAKS** (§1) — they are why this can be trusted now.
- **Replicate.** One run of a stochastic cascade is an anecdote. Several seeds, and score
  the order per seed.
- **Sanity-check the window empirically before committing hours**: run 5 s, confirm ≥2
  breaks have landed, then size the full run.

**The confound, already handled:** dissolution is coherence-PROTECTED
(`k_diss = k_classical·(1-singlet_excess)·template_enhancement`, and
`singlet_excess(t) = exp(-t/T)` exactly), so it ACCELERATES as coherence decays;
`template_enhancement` measured max 50. Analytic survival
`ln(N/N0) = -k_classical·te·[t + T(e^{-t/T}-1)]` ⇒ 93% at 9.5 s, 49% at 30.5 s, **1.9%
at 75 s**. So the **floor prediction (partition dies at P_S=1/√2, t≈75 s) is CONFOUNDED**
— cannot be separated from "dimers ran out" — and is DEMOTED: report if reached, never
score. `n_dimers` is logged as the control: coherence ⇒ SHARP synchronised loss of a
whole gap-class; dimer loss ⇒ SMOOTH erosion tracking the population curve.

---

## 4. THE OTHER STANDING RESULT: T1 (SOC power law) IS RETIRED

Killed on structural grounds for ~0 compute — **not** on a negative result:
- **1D forbids it.** The honest dendrite is 1D (`_generate_positions` `'linear'` =
  "synapses along a straight dendrite"; coupling runs along the MT backbone). In 1D
  connectivity is decided entirely by CONSECUTIVE gaps, so clump size is *exactly*
  geometric `P(k)=p^(k-1)(1-p)` — exponential tail. Verified: geometric fit R²=0.97–0.996
  at every density, beating the power-law fit every time. Regular spacing is worse —
  binary, no distribution at all. 2D near percolation DOES give a power law, but it is
  **pure geometry**: no drive, no η, nothing quantum.
- **D18 forbids it.** Nucleation is all-or-none bistable with a forbidden gap and a
  QUANTAL drive-independent ON amplitude (~135) — a characteristic **scale**, where SOC
  requires scale-**freedom**.
- **Two "SOC" claims were conflated.** *SOC-chemistry* (phosphate depletion ⇒ S→1) is
  ESTABLISHED (D8/D14) and stands. *SOC-topology* (power-law clump sizes) is what T1
  chased; it inherited the name.

---

## 5. OPEN ITEMS

1. **Score the live T1′ run** — order only, per §3.
2. **Re-read D8/D14/D16/D17** against the 3× operating-point shift (§2b). Physics call.
3. **Next perf bottleneck** is `get_network_metrics` / `find_entangled_clusters` (~57% of
   what remains after `40e9986`). `net.step` is now 350 ms at ~2.4k dimers (was 644 ms).
4. **The appendix is uncommitted** in `murmur-platform/.claude/skills/entanglement-topology-measurement/SKILL.md`
   (Appendix A + A7b). Left deliberately — that repo has ~300 uncommitted files; not
   this session's to sweep.
5. **Skills owing updates** (drift found, verified):
   - `quantum-computation-and-attribution:81` says *"the phosphate-runaway means the
     reset feedback does not yet exist"* — **stale**; D8/D14 closed it 2026-06-28.
   - `model6-codebase-operations` still carries an EC2 section; `model6-architecture:18`
     says *"There is no EC2 in this program — earlier EC2 references are retired. Large
     runs are local or deferred."* The ops skill flags the conflict itself. **EC2 is
     retired — large runs are LOCAL** (precedent: June 7, 40 synapses, single-core ~10 h).
   - `entanglement-topology-measurement` was "DESIGN PHASE" / Fiedler-flux / PH-vineyards
     headlined; corrected in the uncommitted appendix (§5.4).

---

## 6. MISTAKES THIS SESSION — recorded so they are not re-inherited

The pattern in all four: **inferring instead of measuring, then measuring and being wrong.**

1. **"Dead `k_dissolution` / population path broken / §6.5 construct-validity gap."**
   WRONG. Dissolution was implemented in the OTHER module; the particle-system slaving is
   DECLARED DESIGN (`dimer_particles.py:163-165`); `k_dissolution` (L127) is a vestigial
   duplicate. Cause: **grepped `RESEARCH_LOG_CALCIUM_DIMER.md` instead of reading D1–D18.**
   Sarah caught it ("I thought we had already done this research on the dimers?").
2. **"T2: ~12 h → ~1–3.5 h post-vectorisation."** WRONG — extrapolated from a
   microbenchmark of the one function I'd fixed instead of profiling the whole step.
   Amdahl. Real: 14.3 h; the INTRA loop was 79%.
3. **"The chain/ring rig sits on a knife-edge (2.5 µm vs 2.41 µm cut)."** WRONG — used an
   assumed P_S=0.90. Real P_S is 0.998 ⇒ d*=3.45 µm, comfortable margins both ways.
4. **The T1′ break times (§3) — THREE times.** Pre-registered against the median (9.5 s),
   then the p95 (~13 s), then an "analytic ceiling" (19.3 s). Observed: 34.0 s. Each
   correction was itself an inference, not a measurement. The third failed because I
   ignored the multiplicative noise in `step_coherence`, which lets dimers exceed the
   ceiling I called a hard bound. The lesson is not "use a better statistic" — it is that
   the times are an extreme-value statistic over a noisy walk and are not analytically
   predictable. **Pre-register the ORDER; never the times.**
5. **`collective_field_kT` is 0** (would have licensed an early-exit in the intra loop).
   WRONG — it runs ~21.7. **Measured before acting**; an early-exit would have silently
   deleted Pathway 2.
6. **The T1′ verdict function passed vacuously.** `all(...)` over one break is True, so
   the probe printed "far-pairs-first fragmentation CONFIRMED" off a single FLICKER. I
   wrote a test that could not fail. Guarded now (`MIN_BREAKS=3`, `CONSECUTIVE_ABSENT=3`),
   but the lesson is the one `session-discipline` names: **shortcuts produce
   defensible-looking wrong numbers, not working software.** A verdict that cannot return
   INCONCLUSIVE is not a verdict.
7. **First intra vectorisation drew for all pairs** → diverged on gated pairs (42 vs 37).
   Caught because an occupancy guard flagged the statistical test as underpowered (0.6%,
   ~3 bonds); re-running at real occupancy exposed it. Fixed to draw only for active
   pairs in triu order ⇒ **bit-identical**.

---

## 7. OPENING MESSAGE FOR THE NEW THREAD

> Continue the Model 6 entanglement-topology work. Read
> `src/models/Model_6/docs/handoffs/SESSION_HANDOFF_JUL16_TOPOLOGY_DT_FIX.md` FIRST —
> it names the live run, the six unpushed commits, and (critically, §3) a mis-derived
> pre-registration you must not score naively. Then read
> `src/models/Model_6/docs/SOC_topology_NEXT_THREAD_kickoff.md` and
> `SOC_topology_experiment_handoff.md`. Per this repo's CLAUDE.md this is a
> human-bridged thread: run the GROUND sequence and return a `### GROUNDING BRIEF` as
> your first message, then STOP for my confirmation before building. First job: check
> whether the T1′ run (pid 13499, log in §1) has finished and score its ORDER prediction
> only — the times in the script docstring are wrong, see §3 for the corrected ceiling
> derivation.
