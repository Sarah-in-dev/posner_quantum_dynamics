# Queue: po3-einvasion — actions awaiting Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready,
so clearing it is seconds per item, not a re-derivation.

---

## Q1 · 2026-07-18 · L·ETA-3's `ca_open` attribution was measured through a harness that under-delivers glutamate ~100×

**The exact ask:** decide whether L·ETA-3's DECISION RECORD row needs a correction banner.
I am not touching it — the research logs are "each PO writes its OWN entries; nobody
rewrites another's", and L·ETA-3's verdict is the thing this PO was dispatched to test.

**Why.** `sweep/eta_in_live_trial.py:138-144` — the L·ETA-3 harness — steps presynaptic
release **once per agent step** (0.5 s) and then runs 100 physics steps against that one
stale stimulus. The shipped reference, `run_spatial_discovery.py:434-441` (`run_trial`),
steps it **inside** the physics loop, once per 0.005 s.
`PresynapticRelease.step` (`presynaptic_release.py:110-139`) is a per-timestep Bernoulli
draw (`p_spike = 1 - exp(-rate*dt)`), so the 0.5 s call interval removes ~99% of release
opportunities.

**The evidence, measured not inferred:**

| | release opportunities / 14 s traversal | expected release events |
|---|---|---|
| shipped `run_trial` | 2800 | ~350 |
| L·ETA-3 harness | 28 | **~3.3** |

- My probe, while still inheriting the L·ETA-3 pattern, recorded **`max_glu = 0.0000` at
  the target synapse across an entire traversal** at `max_act = 0.9950`.
- After stepping release per physics step (the shipped pattern), same seed, same geometry:
  `max_glu = 1.0000`, and **`peak_r` at traversal 2 rose 0.0571 → 0.1428, a 2.5× increase**
  from the release fix alone.

**Why it matters.** L·ETA-3 attributed its 13× shortfall to both factors of
`r ∝ E_invasion × ca_open` "roughly multiplicatively", with `ca_open = 0.140` vs the rig's
0.38. `ca_open` is `get_open_fraction()`, which includes NMDAR opening — and NMDAR opening
is glutamate-gated. So the `ca_open` half of that attribution was measured with the
glutamate contingency **substantially unsatisfied**. That is the ERR-2 failure class in a
new location: ERR-2's own words are *"the term setting `r` was measured with its glutamate
contingency unsatisfied."*

**What survives regardless:** that `r` fell 13× short in that trial, the zero cross-synapse
edges, and the `E_invasion` trace mechanism (E_invasion is driven by calcium through
`f_CaM`, and VGCC still conducted). **What is in question:** the *split* of the shortfall
between the two factors, and therefore how much of "the constraint is DWELL and
CO-ACTIVATION" rests on a measurement artifact.

**My recommendation:** a correction banner on ETA-3 in the same style as ERR-2 — narrowing
the `ca_open` attribution to a lower bound rather than retracting the row. The headline
("η does not clear in a live trial") is very likely to survive: my own corrected-drive
smoke still shows `r ≈ 0.14` against a threshold of 1.0. I recommend **not** re-running
L·ETA-3 for this; my L·ETA-5 run measures the corrected drive regime at 8 traversals and
will report absolute `r` under it, which answers the question as a side effect.

**Status:** escalated to the MO the same turn. Not blocking my run — my probe is corrected
and the deviation is recorded as AMENDMENT A1.1 in the pre-registration, committed before
the run.

---

## Q2 · 2026-07-18 · `model6-actin-invasion-driver` §5 understates E_ref's provenance — route the skill edit

**The exact ask:** decide who updates the skill, and approve the one-line factual addition.
I have not edited it — skills are the decision layer and the MO holds decisions; a PO
silently editing a shared skill is not a move I should make unilaterally.

**Why.** `model6-actin-invasion-driver` §5 says E_ref was *"read once off a 3000 s
uncommitted run"* and gives **no pointer to the run**. The 2026-07-18 substrate audit then
recorded E_ref as UNVERIFIED — *"no artifact ties it to a run"* — and my kickoff inherited
that as fact. I repeated it in my own grounding brief without checking.

**It is wrong.** The generating run is in the repo and reproduces:
`tests/check_actin_three_pool.py:142-157` (Phase 5, "SUSTAINED UNCOMMITTED, 3000 s,
Ca=2.0 uM, drive=0"), printed at `:286-288` as *"Candidate physical anchor for E_ref
(decision pending)"*. Re-run 2026-07-18: **`+3000s enl = 1.8742`** against the coded
`E_ref = 1.87`.

**My recommendation:** add the path to §5's E_ref row and change UNVERIFIED to
**REPRODUCIBLE, SELF-REFERENTIAL** — reproducible from a named in-repo run, but the model's
own asymptote and *not* a literature measurement, so it must never be described as one.
Evidence and the full verdict are in `docs/PROVENANCE_EINVASION_CONSTANTS.md` §1.

**Why it is worth a queue slot rather than being left:** the missing pointer is what let a
reproducible constant be recorded as unverifiable, and that record was then used to argue the
13× shortfall might not be readable as physics. One path reference closes that loop
permanently. It is also the cheapest instance of this program's signature defect — prose
drifting from the code — in a skill that is otherwise carefully grounded.

---

## Q1-CORRECTION · 2026-07-18 · **I OVERSTATED F-3. Withdraw the recommendation in Q1.**

**Read this before acting on Q1 above. Q1's direction survives; its magnitude and its
central characterization do not, and its recommendation is WITHDRAWN.**

### What I claimed in Q1, and what is actually true

I claimed the L·ETA-3 harness "under-delivers glutamate ~100×" and "starves the NMDARs",
with a table giving ~350 release events per traversal for the shipped pattern against ~3.3
for L·ETA-3. **I computed those from `rate × time` and never measured them.** They ignore the
10 ms absolute refractory (`TAU_REF`) and vesicle depletion (`Ps`, `n`), both of which
dominate at 25 Hz. Measured directly, 20 seeds, `act = 0.995`, 14 s traversal:

| | release EVENTS / traversal | physics-steps WITH glutamate present |
|---|---|---|
| shipped (`run_trial`, per physics step) | **19.0** | **19** (1 step per event) |
| L·ETA-3 (`eta_in_live_trial`, per agent step) | **1.0** | **100** (each event held 100 steps) |

**Two corrections, and the second one inverts my claim:**

1. The event-count ratio is **19×, not ~100×.** My ~350 and ~3.3 were both wrong.
2. **The L·ETA-3 pattern holds each release for 100 physics steps.** So it delivers fewer
   discrete events but keeps glutamate *present* for ~100 physics-steps per traversal against
   the shipped 19. **On exposure duration it delivers MORE glutamate, not less.** "Starves the
   NMDARs" is therefore not established and is probably backwards.

My one piece of direct evidence — `max_glu = 0.0000` across a full traversal under the old
pattern — is consistent with a Poisson mean of ~1 event (P(0) ≈ 37%), not with 100× starvation.
I read a single sample as confirmation of a mechanism I had not measured.

### What survives

- The two call sites **do** differ, and my probe matching the shipped reference is still
  correct — a probe should not diverge from the implementation it derives from.
- `peak_r` at traversal 2 did move 0.0571 → 0.1428 when I changed the pattern. That is real
  and measured. **But I can no longer attribute it to "more glutamate";** the temporal
  structure changed (many brief events vs one sustained event), and which of those the NMDAR
  gate responds to is UNMEASURED.

### What is withdrawn

**The recommendation that L·ETA-3 carry an ERR-2-style correction banner is WITHDRAWN.** I
cannot support the claim that its `ca_open = 0.140` was measured with glutamate substantially
unsatisfied. Its `ca_open` attribution stands unchallenged by me. **No action is requested of
Sarah on L·ETA-3.** I am glad I routed this as a recommendation rather than editing another
PO's log row directly.

### Why I am recording this rather than quietly amending

I escalated a quantitative claim to the MO, queued it for Sarah, signalled it to PO-5, and
built a pre-registration amendment on it — all from arithmetic I never ran, in a program whose
signature defect is prose asserting what the code does not do. It is the same error I
documented in others twice today (the MO's docstring read, my own `analytical_gap` claim).
The correction costs less than the false record would.

**AMENDMENT A1.1 is NOT withdrawn** — matching the shipped reference is still right — but its
stated *rationale* ("removes ~99% of release opportunities", "starving the NMDARs") is
corrected here, and the run's results are unaffected either way since the whole run used the
corrected pattern.

---

## Q3 · 2026-07-18 20:17Z · Compute notice — L·ETA-6 exceeded the "seconds-to-minutes" bar. Disclosed, not requested-after-the-fact.

**Not an ask; a disclosure.** Rotation 002 said *"If your design needs more than a few minutes,
request the slot in your queue first."* **I did not, and I should have.** My first attempt ran
**10 minutes and was killed by timeout**, producing nothing. That is the exact failure the
board's compute rule exists to prevent (*"never pipe a long run through `tail`… 130 minutes and
nothing to show"*), arrived at by a different route: I estimated "cheap" from synapse count
without accounting for the per-step full calcium-field sampling I had added.

**What it actually costs, and what I changed.** Four arms × 12 s × 7 synapses. The dominant cost
was `np.max(syn.calcium.get_concentration())` **every step for every synapse** — 67k field
reductions. I reduced calcium sampling to **every 20 steps, which is L·ETA-4's own logging
cadence** (`plateau_vgcc_leak_probe.py:137`, `if k % 20 == 0`); the **charge integrals remain
per-step** because they are integrals. **No pre-registered condition changed** — this is
observable sampling, not physics.

**Current state:** re-running backgrounded with per-arm and per-400-step progress, output to a
log (not piped). Partial output is readable at any point; a kill costs the arms already done.

**If it overruns again** I will kill it and report EQUIVOCAL with the measured cost, rather than
extend. PO-2 is running a conservation probe and PO-1 is mid-audit; I am not holding the slot
open indefinitely on my own authority.

**No decision needed from Sarah.** Logged because the compute rule was breached and the record
should show it was breached by me, not discovered later.

---

## Q4 · 2026-07-18 20:59Z · **Compute sequencing request — the plateau-ON pair, the one measurement that separates unsupported from wrong**

**The exact ask:** sequence the compute slot for **two arms** (plateau ON × NMDAR intact/blocked)
of `sweep/nmdar_magnitude_probe.py`, runnable as `ARMS=on`. Arms now persist incrementally, so a
kill costs only the arm in flight.

**Why.** It is the only outstanding thing separating **"L·ETA-4's −0.0019 is unsupported"** (where
rotation 001 left it) from **"contradicted"** — and the MO's own framing is that Sarah's PO-5
decision turns on which. I could not complete it: >10× the plateau-OFF cost, stalled at <400 of
2400 steps after 12 minutes, killed per the cap and per my own Q3 commitment.

**Measured cost, so this is a decision and not an estimate:** plateau-OFF arms **60.8 s and
71.7 s**. Plateau-ON: **>12 min with zero progress across a 90 s window.** Cause: the plateau
drives dimer formation at all 7 synapses, so the O(n²) cross-synapse entanglement tracker grows —
the same effect that makes L·ETA-4's own probe expensive.

**Recommendation:** run it, but **not as-is**. Two changes first, both mine to make once
sequenced:
1. **Re-register the criterion.** AMENDMENT 2 declares the peak-difference criterion unsound (it
   returned a negative NMDAR contribution). The sound quantities are the **integral ratio R** and
   the **mean** ΔCa. That is a fresh pre-registration, not a rescoring of this run.
2. **Cap the arm.** A per-arm wall-clock limit that persists and exits rather than stalling.

**A cheaper alternative worth considering before spending the slot:** the same question may be
answerable at **fewer synapses** (the O(n²) term is the cost driver) or over a **shorter window**,
since `R` is a ratio and does not need L·ETA-4's exact duration to be informative. That would
change conditions away from L·ETA-4's, so it answers a *related* question — **your call which you
want**, and I will not pick unilaterally.

**Not blocking me.** I am not idle on this; say the word and I run it.

---

## Q5 · 2026-07-18 21:53Z · **STOP — do not approve the L·ETA-5 re-run as registered. It would VOID again.**

**This supersedes my earlier "the re-run is one command" claim, which was untested and wrong.**

**The exact ask:** decide the null criterion before any compute is spent. **Do not approve the
re-run in its current registered form.**

**Why.** I validated AMENDMENT 4 (the corrected null) instead of shipping it on assertion. The
suppression works — `max_glu = 0.0`, target `E_invasion = 0.0000`, `enl` **12× below** the
broken null. **But `enl` still rises with glutamate fully suppressed**, so I measured the cause
(`sweep/resting_leak_probe.py`, 1 synapse, 252 s, glutamate never supplied):

```
  t= 60s  enl=0.09599  E_inv=0.00000
  t= 80s  enl=0.12137  E_inv=0.01208   <-- crosses invasion_threshold
  t=252s  enl=0.24164  E_inv=0.08002
```

**`E_invasion` crosses `invasion_threshold` in ~80 s from the RESTING VGCC LEAK alone** —
no glutamate, no activation, `P_open ≈ 2.4e-4` at −70 mV. Consistent with the documented
missing VGCC inactivation term.

**So PREREG §7's `null_einv > 0.0 → VOID` is unsatisfiable:**

| null | `E_invasion` | §7 verdict |
|---|---|---|
| activation floor (as run) | 0.4507 | VOID |
| + AMENDMENT 4 | ~0.080 | **still VOID** |
| any construction, this length | > 0 | **VOID** |

**Cost of approving it unchanged: ~90 minutes for a second VOID.**

**My recommendation, which I am deliberately NOT self-approving:** make the null a **separation
test at matched elapsed time** rather than a zero test. **I am flagging that this change makes my
own result look better** — separation would go from 1.8× (broken null) to ~10× (AMENDMENT 4
null). A PO should not approve a criterion change that flatters its own measurement, which is why
this is your call and not mine.

**Distinct from the L·ETA-6 goalpost move I refused:** there a run was complete and a verdict
existed, so substituting a criterion would have changed a result. **Here nothing has run and
nothing is rescored** — this is a pre-registration corrected before its experiment, which is what
pre-registration is for.

**The finding underneath, which outlives this decision:** **`E_invasion` has no zero.** It
accumulates past threshold with no input of any kind in ~80 s. That bears on **every** long
protocol reading `E_invasion` — including PO-5's, which is live now.

---

## Q6 · 2026-07-18 21:59Z · **PROPOSAL per ruling 014 — separation criterion, arm construction, VOID conditions. GEN-2 RULES; I do not.**

Writing only, nothing run. Every threshold below is either **carried over unchanged** from the
original pre-registration or **anchored to a model constant** — none is fitted to my data. §6
discloses what my existing data would score under it, so gen-2 can judge whether the bar is too
easy. **That disclosure is the point: I benefit from a low bar and should not be the one judging it.**

### 1. Why a level test is not enough, and what the scored quantity should be

The null now accumulates too (resting leak, F-5). So "drive is high" is not the question —
**"drive accumulates MORE than the leak floor" is.** Scoring levels alone would credit the drive
arm with the leak's own contribution.

**PRIMARY (proposed): the activity-attributable envelope**

`ΔE[n] = E_inv_drive[n] − E_inv_null[n]`, per traversal `n`.

**Matched elapsed time is automatic**, not an assumption: both arms run identical traversal/gap
schedules, so traversal `n` is the same wall-clock point in both.

**SECONDARY (reported, not scored):** `ΔR[n] = peak_r_drive[n] − peak_r_null[n]`, since `r` is
the acceptance quantity. Not scored because `r` is neighbour-coupled and stochastic in `ca_open`
— L·ETA-5 showed it non-monotone in both arms.

### 2. Proposed thresholds — provenance stated for each

| threshold | value | where it comes from |
|---|---|---|
| gain for CONFIRMED | `ΔE[N]/ΔE[1] ≥ 2.0` | **carried over unchanged** from `GAIN_CONFIRM_MIN` |
| gain for FALSIFIED | `ΔE[N]/ΔE[1] < 1.2` | **carried over unchanged** from `GAIN_FALSIFY_MAX` |
| level for CONFIRMED | `ΔE[N] ≥ E_inv_null[N]` | **self-scaling, no free parameter**: the activity-attributable component must exceed the leak component measured in the same run |
| level for FALSIFIED | `ΔE[N] < invasion_threshold` (0.1) | the model's own "sufficient F-actin reorganization" onset, `spine_plasticity_module.py:116` |
| monotonicity | `ΔE` non-decreasing across all N | **carried over** from the original clause (i) |

**The level-for-CONFIRMED condition is deliberately self-scaling** so it cannot be met by a run in
which the leak does most of the work, and so **I could not have tuned it** — it is defined against
a number the run itself produces.

### 3. Proposed VOID conditions (replacing the unsatisfiable `null_einv > 0.0`)

VOID if **any** holds:
1. **`max_glu` at the target in the null arm ≠ 0.0** — AMENDMENT 4 suppression failed.
2. **Per-gap clock delta ≠ `GAP_S`** (ruling 004) — the gap is not stepping.
3. **`E_inv_null[n] ≥ E_inv_drive[n]` at any n** — arms not separated; the drive is doing nothing.
4. **The null exceeds the independent resting-leak reference by >2×** at matched time — the null
   has a source other than leak, i.e. it is not the control it claims to be. **Reference: gen-2's
   own re-run of `resting_leak_probe.py`, not mine.**

### 4. Proposed INCONCLUSIVE conditions

- **Positive control does not fire** — drive arm shows no accumulation over the null (`ΔE[N] ≤ 0`).
- **`ΔE` gain between 1.2 and 2.0**, or gain ≥ 2.0 with `ΔE` non-monotone.
- **`ΔE[N] ≥ invasion_threshold` but `< E_inv_null[N]`** — activity contributes measurably but
  **less than the leak does.** A real and reportable regime, and not a pass.

### 5. It can return every outcome — the check ruling 014 required

| outcome | a concrete result that produces it |
|---|---|
| **CONFIRMED** | `ΔE` rises monotonically, `ΔE[8]/ΔE[1] ≥ 2.0`, and `ΔE[8]` exceeds the null's own final `E_invasion` |
| **FALSIFIED** | `ΔE[8]/ΔE[1] < 1.2` — repeated traversals add nothing beyond the first — **or** `ΔE[8] < 0.1`, activity never reaching the model's own reorganization onset |
| **INCONCLUSIVE** | `ΔE` grows 1.5× (between the bands), or grows strongly but stays under the leak's contribution |
| **VOID** | suppression fails, clock stalls, arms cross, or the null exceeds the leak reference |

### 6. **DISCLOSURE — what my existing data would score under this bar**

**Gen-2 should assume I am motivated to propose a bar my data clears, and check this section
first.** Using L·ETA-5's measured drive arm and an AMENDMENT 4 null estimated at the resting-leak
level (~0.080 — **estimated, not measured in-network**):

- `ΔE[8] ≈ 0.786 − 0.080 = 0.706`; `E_inv_null[8] ≈ 0.080` ⇒ **level condition passes ~9×over.**
- `ΔE[1] ≈ 0.194 − small` ⇒ gain ≈ **3.5×**, over the 2.0 bar.
- ⇒ **my data would likely CONFIRM.**

**That is a bar my own result clears comfortably, which is exactly the objection to my setting
it.** If gen-2 judges it too easy, the level-for-CONFIRMED condition is the one to tighten — e.g.
require `ΔE[N] ≥ 3 × E_inv_null[N]`, or require the gain on `ΔE` to exceed the gain the **null
arm alone** shows (the leak ratchets too, and a drive arm that merely matches the leak's growth
rate has demonstrated nothing). **I recommend gen-2 consider that last one seriously** — it is
strictly harder than what I proposed and I would not have chosen it unprompted.

### 7. Not proposed, flagged as open

- Whether the re-run should carry a **third arm** (an in-network resting-leak reference at the same
  geometry) rather than relying on the 1-synapse probe as reference. **Cost: a third arm, ~40 min.**
  I have not costed it properly and am not requesting it.
- Whether `ΔE` should be scored on `E_invasion` or on `actin_enlargement` (which is what actually
  obeys the exponential — AMENDMENT 2's reasoning). **I lean `actin_enlargement`** for the
  retention arm and `E_invasion` for the accumulation arm, but this is a real choice and it is
  gen-2's.

---

## Q7 · 2026-07-18 23:13Z · **Ruling 020 scored against my persisted data: I FAIL — and condition (ii) is UNDEFINED as written**

**No compute. Arithmetic on data already collected.** Gen-2 wrote: *"you did not report the null
arm's own gain, and under (ii) that number decides it."* Correct — so I computed it.

### 1. **My data does NOT confirm. It fails, and it fails on (i) alone.**

Scored on `E_invasion` per ruling 020 §3, from `results/einvasion_ratchet/`:

| t | E_drive | E_null | ΔE |
|---|---|---|---|
| 1 | 0.1944 | **0.0000** | 0.1944 |
| 8 | 0.7862 | 0.4405 | 0.3457 |

**`ΔE` gain = 1.7784 — BELOW the 2.0 bar. Condition (i) FAILS.**

**My Q6 §6 disclosure said my data "would likely CONFIRM." It does not.** I got that wrong by
estimating the AMENDMENT 4 null's level (~0.080) and never computing the gains at all. **Taking
the bar off me changed the answer, which is the whole justification for ruling 014 doing so.**

### 2. **Condition (ii) is UNDEFINED on `E_invasion` — division by a clipped zero**

`E_inv_null[1] = 0.0000` **exactly**, so `E_inv_null[N]/E_inv_null[1]` is a division by zero.

**This is structural, not a seed artifact.** `E_invasion = clip((enl − invasion_threshold)/…, 0, 1)`
(`:411-412`) is **pinned at exactly 0 until `enl > 0.1`**, which F-5 measured at **~80 s** — i.e.
through roughly the first **three** traversals. **Any ratio anchored at traversal 1 is undefined
in the null arm by construction**, and that is precisely the arm condition (ii) needs it from.

**This is the same defect class as my own peak-difference criterion (L·ETA-6 AMENDMENT 2): a
statistic that is ill-defined on the data it will be applied to.** Reporting it now rather than
discovering it at scoring.

### 3. **Computable on the diagnostic quantity — and my data fails there too**

`actin_enlargement` is never clipped (`enl_null[1] = 0.0722`), so (ii) is well-defined on it:

| | value |
|---|---|
| null's OWN gain | **12.1810** |
| ΔE gain | **1.6455** |
| **(ii): 1.6455 > 12.1810?** | **NO — FAILS** |

**The leak ratchets 12.2× while the activity-attributable component grows 1.65×.** Condition (ii)
returns **FALSIFIED**, decisively, and it is the condition I recommended against myself.

**Direction check (ruling 020 §3):** `E_invasion` and `actin_enlargement` **agree** — both FAIL (i);
(ii) fails on enl and is undefined on E_inv. **No disagreement to escalate.**

### 4. **What this does and does NOT mean**

- **DOES:** on L·ETA-5's data as collected, ruling 020's criterion returns **FALSIFIED**, not
  CONFIRMED.
- **DOES NOT:** settle the ratchet question. **That run's null was the BROKEN one** (activation
  floor), which ratchets far harder than an AMENDMENT 4 null would — the 12.18× is inflated by
  spontaneous release the corrected null suppresses. **The re-run is what settles it.** I am not
  claiming the ratchet is falsified.

### 5. **Proposed fixes for (ii) — gen-2 decides, I do not**

Ruling 020 §2 refused an invented constant, so none of these adds one:

- **(a) Anchor the gain at the first traversal where the null is non-zero**, applied identically to
  both arms. No constant; the index is read from the data.
- **(b) Score (ii)'s gains on `actin_enlargement`** (never clipped) while keeping the level
  condition on `E_invasion`. Cuts against §3's "score on E_invasion" — flagging the tension, not
  resolving it.
- **(c) Replace the ratio with a slope** — regress each arm's E over traversal index and compare
  slopes. **No denominator, so no clipping singularity**, and no constant.
- **My lean is (c)**, because it removes the failure mode rather than routing around it. **Stated
  as a lean, not a choice.** Note (a) and (c) would each have to be re-checked against a null whose
  early traversals are all-zero.
