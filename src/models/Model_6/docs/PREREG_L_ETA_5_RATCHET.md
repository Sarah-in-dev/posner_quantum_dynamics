# PRE-REGISTRATION — L·ETA-5: does `E_invasion` RATCHET across traversals?

**Registered 2026-07-18 by PO-3, BEFORE the run.** Committed as its own commit; the probe
and its results land in later commits. Per `MO_MODEL6.md` §2.4: the discriminating quantity
is fixed here, the thresholds are numeric and fixed here, there is a null arm that cannot
show the effect, and the verdict function can return CONFIRMED, FALSIFIED **or**
INCONCLUSIVE.

---

## 1. The question

L·ETA-2 (characterization rig, sustained `act=1.0`) measured `r = 1.6234` against a
threshold of 1.0. L·ETA-3 (live spatial-discovery trial, one traversal) measured
`r = 0.0768` — 13× short — and read off the trace that `E_invasion` "is exactly 0 for the
first ~34 s then climbs 0.011->0.052->0.075 and is STILL RISING at trial end."

`E_invasion` is an actin integrator. A navigating agent gives each feature a brief
transient. **The question is whether the integrator retains enough between traversals that
repeated passes accumulate** — i.e. whether L·ETA-3's shortfall is an artifact of measuring
one traversal.

The prediction comes from the **grounded** constant, not the inherited one:
`tau_extrude = 180.0` (`spine_plasticity_module.py:109`, Honkura 2008, in-band) against a
fixed 20 s inter-traversal gap.

## 2. The mechanism, read off the code (not asserted)

`spine_plasticity_module.py:386-390`:

```
formation = a.k_polymerization_max * f_CaM * (self.actin_monomer / a.S0) * room
k_extrude = 1.0 / a.tau_extrude
extrusion = k_extrude * (1.0 - conf) * self.actin_enlargement
retention = a.k_stabilization_max * conf * self.actin_enlargement
```

During a silent gap `f_CaM → 0` (no calcium), so `formation → 0` and, **while `conf ≈ 0`**,
`actin_enlargement` decays as a pure exponential at `1/tau_extrude`.

**Two consequences that shape the pre-registration:**

**(A) The retention fraction must be scored on `actin_enlargement`, NOT on `E_invasion`.**
`spine_plasticity_module.py:411-412` maps enlargement to `E_invasion` through an **affine**
transform with a subtracted offset:

```
denom = max(1e-6, a.E_ref - a.invasion_threshold)
self.E_invasion = clip((self.actin_enlargement - a.invasion_threshold) / denom, 0.0, 1.0)
```

Because `invasion_threshold = 0.1` is subtracted before normalizing, an exponential decay in
`actin_enlargement` is **not** an exponential decay in `E_invasion`. Scoring the retention
fraction on `E_invasion` would compare the measurement against a prediction the code does
not make. The exponential lives in the state variable; the pre-registered retention
threshold below is therefore on `actin_enlargement`.

**(B) `conf > 0` legitimately shuts extrusion off.** If commitment (`structural_drive`)
latches the confinement latch during the run, `extrusion` is gated to zero and retention
goes to ~100% **as real physics**, not as an artifact. This is physically distinct from the
frozen-clock artifact but numerically identical, so `confinement` is logged per traversal
and the verdict function branches on it explicitly (§6). Not doing this would let a genuine
confinement latch and a stopped clock print the same verdict.

## 3. The frozen-clock hazard this probe deliberately avoids

`sweep/run_spatial_discovery.py:55-78` `analytical_gap()` advances P_S decoherence,
dissolution, bond cleanup and stochastic disentanglement through a silent gap. **It does not
advance spine plasticity.** Actin appears neither in its computed list nor in its
"NOT computed (negligible during silence)" list — it is silently frozen.

Calling it would produce **100% inter-traversal retention** and a clean-looking ratchet that
was an artifact of a stopped clock, which would then read as confirmation of `tau_extrude`.

**Therefore this probe steps real physics through the gap and never calls `analytical_gap`.**
(MO ruling, 2026-07-18: approved as a protocol choice inside PO-3's scope. `analytical_gap`
is PO-4's surface and is not edited. Routed to PO-4 as `requests/po4-analytical-gap/mo-f2-001.md`.)

**This is a stated limit of the measurement, declared in advance:** the silence model here
differs from the shipped experiment's. This probe's gap is *more* physically complete for
actin and *more expensive*; results are not directly comparable to a shipped
`run_spatial_discovery` multi-trial run until F-2 is resolved.

## 4. Protocol

**Geometry and drive — inherited unchanged from the L·ETA-3 harness.** `N_FEATURES = 12`,
`PHYSICS_DT = 0.005`, `AGENT_DT = 0.5`, `SEED = 7`, same `make_network`, same
`activations_to_stimuli` **including the −40 mV synaptic cap at
`run_spatial_discovery.py:368`** (LOCKED, `MO_MODEL6.md` §7 — not touched, not raised).
Coupling row-sums are reported, since feature count sets them.

**A "traversal" is a scripted straight-line pass through one feature's centre.** The random
walk is replaced by a deterministic path so that "traversal N" is a defined object and the
retention fraction is well-posed (MO ruling: fixed gap primary; emergent revisit interval
descriptive only, reported in §7 but not scored).

The agent crosses the target feature's centre at the shipped speed (`0.2` units/s) along a
straight line. With `feature_sigma = 0.5` (`spatial_environment.py:19`) and the 0.05
activation floor, activation exceeds the floor for `|d| < 1.224` units, i.e. **2.45 units of
path = 12.2 s of dwell per traversal** — a brief transient, matching what L·ETA-3 described
a navigating agent as delivering.

- `N_TRAVERSALS = 8`
- `GAP_S = 20.0` (fixed, between traversals; real physics stepped throughout)
- Target feature index: the one whose traversal path is clear of other feature centres,
  chosen from geometry **before** the run and recorded in the trace.

**Runtime assertion (the ERR-2 / `model6-input-engine` drift guard):** the probe asserts
`glutamate > 0` reaches the target synapse during traversal 1 and **records the measured
value in the trace**. `model6-input-engine` still documents the glutamate gap as OPEN (it is
dated June 14 and has drifted); L·ETA-2 records it as closed. Neither is trusted — it is
asserted at runtime. If the assertion fails the run aborts and the verdict is INCONCLUSIVE
(NMDARs silent ⇒ the ERR-2 retraction class).

## 5. The discriminating quantities (fixed before the run)

Per traversal `n ∈ [1..8]`, at the target synapse:

| symbol | definition |
|---|---|
| `peak_r[n]` | max `r` during traversal `n` (recomputed by `r_eta_per_synapse`, unmodified) |
| `enl_end[n]` | `actin_enlargement` at the last physics step of traversal `n` |
| `enl_start[n]` | `actin_enlargement` at the first physics step of traversal `n` |
| `rho[n]` | **retention fraction** = `enl_start[n+1] / enl_end[n]`, for `n ∈ [1..7]` |
| `conf[n]` | `confinement` at the end of traversal `n` |
| `E_inv_start[n]` | `E_invasion` at traversal start (reported; not the scored quantity — see §2A) |

**The prediction, from the grounded constant alone:**

`rho_pred = exp(-GAP_S / tau_extrude) = exp(-20/180) = 0.8948`

## 6. The verdict function — thresholds fixed here, in advance

Evaluated in this order. `rho_mean` = mean of `rho[1..7]`.

**GATE 0 — positive control (must fire, else INCONCLUSIVE).**
The measurement channel must be demonstrated live before any verdict is read:
`max(peak_r[1..8]) > peak_r` at rest **and** `max(E_inv_start[2..8]) > 0` **and** the
glutamate assertion passed. If the positive control does not fire, the probe reports
**INCONCLUSIVE — POSITIVE CONTROL DID NOT FIRE** and stops. *(This is the L·ETA-4 scar
guarded directly: a test whose positive control never fires tests nothing.)*

**GATE 1 — frozen-clock / confinement discrimination.**
- If `rho_mean >= 0.99` **and** `max(conf) < 0.05` → **INCONCLUSIVE — GAP NOT STEPPING.**
  Retention indistinguishable from 100% with the extrusion gate open is a red flag that the
  probe's own gap is not advancing actin, not a strong positive. *(MO ruling, adopted.)*
- If `rho_mean >= 0.99` **and** `max(conf) >= 0.05` → **CONFINED-RATCHET**, reported
  separately: extrusion is legitimately gated off by the confinement latch. This is real
  physics but it is **not** the `tau_extrude` retention hypothesis, and it is reported as a
  distinct outcome, not folded into CONFIRMED.

**GATE 2 — the ratchet verdict** (only reached when `rho_mean < 0.99`):
- **CONFIRMED** iff **both**: (i) `peak_r` is monotonically non-decreasing across all 8
  traversals with `peak_r[8] / peak_r[1] >= 2.0`; **and** (ii) `rho_mean ∈ [0.80, 0.95]` —
  i.e. consistent with the 0.8948 prediction, and excluding both the ~1.0 artifact band and
  a weak-retention regime that would not accumulate.
- **FALSIFIED** iff `peak_r[8] / peak_r[1] < 1.2` **or** `rho_mean < 0.5`. The integrator
  does not retain across a behavioural-timescale gap; repeated traversals do not accumulate.
- **PARTIAL / INCONCLUSIVE** otherwise (e.g. accumulation present but `rho_mean` outside the
  predicted band, or non-monotone `peak_r`) — reported with the numbers, no verdict claimed.

**Reported regardless of verdict, never as a verdict:** whether `peak_r` reaches 1.0 within
8 traversals, and the linear/geometric extrapolation of how many traversals it would take.
Extrapolation is descriptive. It is not evidence and does not enter the verdict.

## 7. The null arm — cannot show the effect, by construction

Identical in every respect (same seed, same network construction, same traversal count, same
gaps, same total wall-clock physics) **except** the target feature's activation is held
below the 0.05 floor for the whole run, so `voltage` stays at −70 mV rest and `f_CaM ≈ 0`.

**Pre-registered null expectation:** `actin_enlargement` stays at its resting value,
`E_invasion` stays 0.0000, `peak_r` is flat across traversals, and `rho` is undefined or
~1.0 at the resting floor. **If the null arm shows a ratchet, the probe is measuring
something other than activity-driven actin and the whole measurement is void** — reported as
INCONCLUSIVE — NULL ARM RATCHETED.

Also recorded (descriptive only, not scored): the emergent revisit interval a free-running
agent would produce at this geometry, for comparison against the fixed 20 s.

## 8. Compute

ONE backgrounded run (board cap). `python -u`, never piped through `tail`. Per-traversal
progress to stdout. `r`-vs-traversal and the per-traversal state row **persisted
incrementally after each traversal**, so a mid-flight kill leaves analysable partial data.
Both arms write to `results/einvasion_ratchet/`.

Estimated simulated time: `8 × (12.2 + 20.0) ≈ 258 s` per arm at `dt = 0.005` over 12
synapses — roughly 6.4× L·ETA-3's 40 s trial. If the run exceeds the cap it is killed and
the partial per-traversal data is analysed as-is; the cap is raised at most once, with a
stated reason in `queue/po3-einvasion.md`.

## 9. What this measurement does NOT do

- It does not touch any constant. `k_polymerization_max`, `E_ref`, `tau_extrude`,
  `invasion_threshold` and the −40 mV cap are all read, never written. (`MO_MODEL6.md` §7:
  *"Emergent physics only. No constant tuned to a downstream target."*)
- It does not extend the protocol to reach threshold. If 8 traversals do not ratchet, that
  is the result.
- **The negative branch is Sarah's call.** PO-3 measures, writes it up, and STOPS — no
  remedy proposed, no constant moved, no protocol extended to rescue it (`board.md`,
  "Decided, do not re-open").

---

# AMENDMENT 1 — 2026-07-18, BEFORE THE RUN, from the smoke test

Two changes, both made **before any scored run**, both disclosed here rather than applied
silently. **No verdict threshold in §6 is changed.** The §5 discriminating quantities, the
§6 thresholds, the §7 null arm and the GATE order are exactly as originally registered.

## A1.1 — Presynaptic release is stepped per PHYSICS step, not per agent step

**This reverses the §4 promise to inherit the L·ETA-3 drive "unchanged", and that is
deliberate: the inherited harness has a defect.**

`sweep/eta_in_live_trial.py:138-144` steps presynaptic release **once per agent step**
(`AGENT_DT = 0.5 s`) and then runs 100 physics steps against that one stale stimulus:

```
for i in range(len(network.synapses)):
    g = network.presynaptic_release[i].step(acts[i], PHYSICS_DT)
    if g:
        stimuli[i]['glutamate'] = g
for _ in range(phys_per):
    step_network_per_synapse(network, PHYSICS_DT, stimuli)
```

The shipped reference implementation, `run_spatial_discovery.py:434-441` (`run_trial`),
steps it **inside** the physics loop, once per `physics_dt`:

```
for _ in range(physics_steps_per_agent_step):
    for i, syn in enumerate(network.synapses):
        glu_event = network.presynaptic_release[i].step(activations[i], physics_dt)
        if active_mask[i]:
            stimuli[i]['glutamate'] = glu_event
```

`PresynapticRelease.step` (`presynaptic_release.py:110-139`) is a per-timestep Bernoulli
draw — `p_spike = 1 - exp(-rate*dt)` — so calling it at 0.5 s intervals instead of 0.005 s
removes ~99% of the release opportunities. At `PEAK_RATE_MEDIAN = 25 Hz` over a 14 s
traversal at saturating activation:

| | release opportunities | expected release events |
|---|---|---|
| shipped `run_trial` | 2800 | ~350 |
| L·ETA-3 harness | 28 | **~3.3** |

**Measured, not inferred:** the smoke run of this probe, inheriting the L·ETA-3 pattern,
recorded `max_glu = 0.0000` at the target synapse across an entire traversal at
`max_act = 0.9950`.

**Why this is not a constant moved to rescue a result.** Nothing tuned, no parameter
touched: this changes a *harness call site* to match the shipped reference implementation
it was derived from. Leaving it would run the ratchet measurement with the NMDAR half of
the channel population starved — the ERR-2 failure class exactly (*"the term setting `r`
was measured with its glutamate contingency unsatisfied"*), which is the specific scar this
PO was dispatched with instructions not to repeat.

**Consequence for comparability, stated plainly:** this probe's absolute `E_invasion`,
`ca_open` and `r` are therefore **NOT** directly comparable to L·ETA-3's numbers. Its
pre-registered quantities (a retention *ratio* and an `r` *ratio*) remain well-posed.
The implication for L·ETA-3's own `ca_open` attribution is escalated to the MO, not
adjudicated here.

## A1.2 — A descriptive late-gap retention diagnostic is added (nothing rescored)

The smoke run measured `rho = 1.0135` — retention **above 1.0** across a 3 s gap. Mechanism,
read off the code: calcium decays over ~seconds, so `f_CaM` and therefore `formation`
(`:386`) remain non-zero into the early gap. The gap is not pure exponential decay; it is a
brief continued-rise phase followed by decay at `1/tau_extrude`.

The pre-registered `rho` (first-step-of-next-traversal / last-step-of-this-traversal) is
**kept exactly as registered and remains the scored quantity.** In addition, and as
**descriptive output only**, the probe records `actin_enlargement` sampled through the gap so
the late-gap decay constant can be read separately from the calcium-tail phase.

**This is why the band was pre-registered rather than a point estimate**, and it is
recorded here that the calcium tail biases the scored `rho` *upward*, i.e. toward the
artifact band — so a `rho` at or above the band is expected to be at least partly this
effect and must not be read as strong confirmation. The thresholds stand as written; this
note exists so the bias direction is on record before the numbers are seen.

---

# AMENDMENT 2 — 2026-07-18, BEFORE SCORING — the retention prediction was MIS-DERIVED

**Origin: PO-4's finding, ruled on by the MO (`requests/po3-einvasion/mo-ruling-002.md`),
on PO-3's own surface. PO-3 re-derived it independently and confirms it. This corrects an
error in §2(B) and §5 of this document.**

## What was wrong

§2(B) above says:

> **(B) `conf > 0` legitimately shuts extrusion off.** ... `extrusion` is gated to zero and
> retention goes to ~100% **as real physics**, not as an artifact.

**That is wrong.** It read `:389` and stopped:

```
extrusion = k_extrude * (1.0 - conf) * self.actin_enlargement      # CLEARING: unconfined -> shaft
retention = a.k_stabilization_max * conf * self.actin_enlargement  # confined -> stable
```

`conf` gates extrusion **off** at `:389` and simultaneously gates **retention on** at `:390`
— and retention is *also* a drain on `actin_enlargement` (`d_enlarge = formation - extrusion
- retention`, `:395`). Since `E_invasion` reads `actin_enlargement` **alone** (`:412`), a
committed spine's `E_invasion` decays **faster**, not slower.

This is the same defect class this PO was dispatched to hunt — prose asserting a mechanism
the code does not implement — committed in PO-3's own pre-registration, in the very section
titled "the mechanism, read off the code (not asserted)".

## The correct derivation (re-derived by PO-3, matching PO-4 and the MO)

Total drain on `actin_enlargement` during a silent gap (`formation → 0` as `f_CaM → 0`):

```
drain(conf) = k_stabilization_max·conf + (1 − conf)/tau_extrude
rho_pred(conf, gap) = exp(−drain(conf)·gap)
```

With `k_stabilization_max = 0.02` (`:99`), `tau_extrude = 180.0` (`:109`),
`k_conf = 0.02` (`:113`), `k_unconf = 0.0005` (`:114`), `conf_ss = k_conf/(k_conf+k_unconf)`:

```
conf_ss      = 0.97561
uncommitted  drain=5.5556e-03  tau=180.0s  ret@20s=0.8948
committed    drain=1.9648e-02  tau= 50.9s  ret@20s=0.6751
speedup      = 3.54x
```

And because `k_unconf = 0.0005 s⁻¹` is slow, **confinement persists**: a spine that has ever
committed does not return to the 180 s branch within these gaps.

**No constant is changed.** This selects which of two formulas *already in the code* the
prediction uses. `tau_extrude`, `k_conf`, `k_unconf` and `k_stabilization_max` are untouched
and remain LOCKED against tuning.

## What is now pre-registered, replacing §5's single number

**The prediction is computed PER GAP from the MEASURED confinement**, using the code's own
formula above — strictly better than two discrete numbers, because it is exact at any `conf`:

| symbol | definition |
|---|---|
| `conf_gap[n]` | `confinement` at the START of gap `n` (measured, logged) |
| `rho_pred[n]` | `exp(−(k_stab·conf_gap[n] + (1−conf_gap[n])/tau_extrude)·GAP_S)` |
| `rho_ratio[n]` | **`rho[n] / rho_pred[n]`** — the scored quantity |

**The band is carried over, not re-invented.** The original §6 band on raw `rho` was
`[0.80, 0.95]` around a prediction of `0.8948`. Re-expressed as a ratio that is
`[0.80/0.8948, 0.95/0.8948] = [0.894, 1.062]`, pre-registered as **`rho_ratio ∈ [0.89, 1.07]`**.
Widening or narrowing it would be moving a goalpost; it is the same band in ratio form.

**GATE 1 is corrected and becomes STRONGER.** The old GATE 1 had a `CONFINED-RATCHET` branch
treating `rho ≈ 1.0` with high `conf` as legitimate physics. That branch rested on the error
above and is **deleted**. Under the correct physics a committed spine drains *faster*, so
`rho ≈ 1.0` is an artifact signature **regardless of `conf`**:

- If `rho_mean_raw >= 0.99` → **INCONCLUSIVE — GAP NOT STEPPING**, unconditionally.

**GATE 2 is unchanged except that clause (ii) now reads `rho_ratio` in `[0.89, 1.07]`**
instead of raw `rho` in `[0.80, 0.95]`. Clause (i) — monotone `peak_r`, gain ≥ 2.0 — and the
FALSIFIED thresholds are untouched. **RULING 2 adopted: the `peak_r` ratio arm was already
sound and is not rewritten.**

## Why this had to land before scoring, in this program specifically

`master` HEAD carries `64346a0` (flagging a mis-derived T1′ pre-registration) immediately
followed by `683b82f` (T1′ false positive). **A mis-derived pre-registration immediately
preceded a false positive here before.** Concretely: had the traversals committed and been
scored against 0.8948, a spine retaining the physically-correct 0.6751 would have read as
**ratchet FALSIFIED** — routing a negative result about the network story to Sarah off a
wrong number, on the branch `MO_MODEL6.md` §3 makes hers.

**Note on the run in flight:** `make_network` sets `network.disable_auto_commitment = True`
(`run_spatial_discovery.py:394`), and traversals 1–2 measured `conf = 0.0000`, so the
uncommitted branch is expected to apply throughout and the original 0.8948 may turn out to
be numerically right *by luck*. The correction stands regardless: `conf` is measured and
logged per gap, and the arm actually taken is recorded rather than assumed.

## Consequence for the in-flight run

The running process will print a verdict computed with the **superseded** GATE 1/GATE 2
logic. **That printed verdict is void and is not the result.** The scored verdict is
recomputed offline from the incrementally-persisted per-traversal JSON using the corrected
function. This is why the pre-registration required incremental persistence.

---

# CORRECTION 1 — §3's "frozen clock" claim was WRONG (PO-4, confirmed by PO-3)

§3 above states that `analytical_gap` "does not advance spine plasticity" and that actin is
"silently frozen". **PO-4 corrected this and PO-3 verified it directly in the code.** The
function's tail runs, after jumping `network.time` by the full gap:

```
    # Advance network time
    network.time += gap_duration_s

    # One full step to sync all internal state (calcium baseline, etc.)
    network.step(0.001, {"voltage": -70e-3, "reward": False})
```

So actin advances **1 ms per gap**, not zero. Inter-traversal retention under
`analytical_gap` would be `exp(-0.001/180) = 0.9999944`, not exactly 1.0.

**How the error was made, recorded rather than quietly fixed:** PO-3 read the docstring's two
lists ("computed" / "NOT computed"), observed actin in neither, read the loop body, and
concluded "frozen" from absence — without reading the function tail. That is prose checked
against prose, reported under a `[code SHOWN]` tag, in a program whose signature defect is
prose contradicting code. The original claim is left in place above per the log convention
(supersede, never rewrite).

**What survives:** the decision not to call `analytical_gap` is unchanged and, per PO-4, more
strongly motivated — 0.9999944 reads as an even cleaner ratchet than a frozen clock would,
while being no more real. The hazard was correctly identified; its magnitude was misstated.

---

# AMENDMENT 3 / POST-HOC CHECK — MO ruling 004: the clock-delta ASSERTION, and it corrects A1.2

**Run after scoring, so this is a post-hoc diagnostic, not a scored quantity.** Ruling 004
required a direct observation rather than a threshold: `D19` (`RESEARCH_LOG_CALCIUM_DIMER`)
names a false-ratchet generator — *"only *active* synapses are stepped, so silent ones never
run their decay term"* — and GATE 1 tests for it only by the *symptom* `rho_mean >= 0.99`.

**Probe:** `sweep/gap_clock_assert.py`. One traversal, then a 20 s parked gap sampling the
target's own `spine_plasticity.time` plus calcium, `f_CaM`, `formation` and `extrusion`.

## Result 1 — the gap IS stepping. D19 ruled out for this probe.

```
CLOCK DELTA over the gap : 20.0000 s   (expected 20.0 s)
ASSERTION spine clock advanced by the FULL gap: PASS
```

`step_network_per_synapse` steps **every** synapse unconditionally
(`run_spatial_discovery.py:321-322`, no active mask), unlike the shipped `run_trial:434-441`
which steps only active ones. So L·ETA-5 does not carry D19's defect.

**Consequence for my own gate, recorded against interest:** `rho_mean` was `0.9915`, i.e.
`>= 0.99`. **Had the null arm passed, GATE 1 would have returned `INCONCLUSIVE — GAP NOT
STEPPING` — a FALSE DIAGNOSIS**, since the clock demonstrably advanced in full. GATE 1 is a
symptom test standing in for a mechanism, and ruling 004 was right that an observed clock
delta is the proof. **Any re-run should assert the clock and not rely on the retention
threshold to detect non-stepping.**

## Result 2 — A1.2's stated MECHANISM was wrong; the bias is impulsive, not a tail

AMENDMENT A1.2 attributed `rho > 1` to a decaying calcium tail: *"calcium decays over
~seconds, so `f_CaM` and therefore `formation` remain non-zero into the early gap."* The
measurement does not support that:

| t_gap (s) | ca (µM) | `f_CaM` | formation | extrusion | net |
|---|---|---|---|---|---|
| 8 | 0.1130 | 0.00016 | 0.000009 | 0.002420 | −0.002411 |
| **14** | **3.1266** | **0.98964** | **0.045805** | 0.002619 | **+0.043186** |
| 16 | 0.1151 | 0.00018 | 0.000009 | 0.002669 | −0.002660 |

For most of the gap `f_CaM ≈ 1.6e-4` and the net is **negative** — clean extrusion at
`tau_extrude`, exactly as predicted. The excess retention comes from **discrete calcium
spikes** (0.11 → 3.13 µM) that saturate `f_CaM` to ~0.99 and produce formation bursts ~5000×
baseline, briefly swamping extrusion. Those spikes are **spontaneous release events**, not a
decaying transient from the traversal.

## Why this matters more than a corrected footnote

**It unifies the two failures into ONE mechanism.** The spontaneous release floor that voided
the null arm (`INCONCLUSIVE_NULL_RATCHETED`) is the *same* process inflating `rho` above the
predicted `0.8948` in the drive arm. There are not two problems with this measurement; there
is one, and it is that **`PresynapticRelease` never goes silent**, so no gap in this design is
a true decay window.

**Therefore the re-run fix is single, not double.** The earlier write-up called for two
independent changes (a null that suppresses spontaneous release, *and* a longer gap to clear a
calcium tail). **A longer gap would NOT help** — the spikes are Poisson in time, so a longer
gap collects proportionally more of them. **Suppressing spontaneous release during gaps is the
one change that fixes both arms.** Still a protocol change to a pre-registered design, and
still not made unilaterally.

---

# AMENDMENT 4 — the CORRECTED NULL, registered before any re-run (MO rotation 001)

**REGISTERED, NOT RUN.** The re-run is gated on Sarah (`MO_MODEL6.md` §3 hard stop). This
exists so that if she approves, the re-run is one command rather than a design cycle.

## The defect being fixed

L·ETA-5's null arm was VOID. It suppressed *activation* (`acts[target] = 0.0`) but not
*release*: `PresynapticRelease.step` uses `rate = baseline_rate + a*peak_rate`
(`presynaptic_release.py:124`) with `BASELINE_RATE_HZ = 0.5` (`:65`), so a synapse at
`act = 0.0` still releases at ~0.2 Hz at full amplitude. The null reached
**`E_invasion` = 0.4507** and **out-gained the drive arm (7.46× vs 5.65×)**.

AMENDMENT 3 showed the same floor is what inflates `rho` above the predicted 0.8948 in the
**drive** arm — discrete calcium spikes to 3.13 µM saturating `f_CaM` — so **one fix addresses
both arms.**

## The change (null arm ONLY)

`SUPPRESS_SPONTANEOUS = True`. In the null arm the target's cleft event is discarded, so the
control cannot receive glutamate by any path:

```
if (not drive_target) and i == target and SUPPRESS_SPONTANEOUS:
    g = 0.0
```

The release object is **still stepped**, so its RRP and facilitation state advance identically
and the two arms stay comparable; only the cleft output is suppressed. **The drive arm is
bit-identical to the scored L·ETA-5 run** — same seeds, same call order, same RNG consumption.

## What is NOT changed, and why

- **The gap stays at 20 s.** AMENDMENT 3 established the spikes are Poisson in time, so a
  longer gap collects proportionally more of them. Lengthening the gap does not help and would
  cost CPU. This supersedes the two-change recommendation in the first L·ETA-5 write-up.
- **No verdict threshold moves.** GATE 0/1/2, `RATIO_BAND = (0.89, 1.07)`, `GAIN_CONFIRM_MIN`,
  the FALSIFIED bounds and the per-gap `rho_predicted(conf)` are all exactly as registered.
- **No constant is touched.**

## Added to the re-run, from MO ruling 004

The clock assertion from `sweep/gap_clock_assert.py` becomes a per-gap **logged quantity**, not
a separate probe: log `spine_plasticity.time` at gap start and end and assert the delta equals
`GAP_S`. **An observed clock delta is proof; GATE 1's retention threshold is only a symptom** —
and on the L·ETA-5 data GATE 1 would have returned a FALSE `GAP NOT STEPPING` (`rho_mean` =
0.9915 ≥ 0.99) while the clock in fact advanced in full.

## Pre-registered null expectation, restated for the corrected arm

With release suppressed, the target receives **no glutamate at all**. NMDAR opening is
glutamate-gated only (`analytical_calcium_system.py:129`), so NMDAR open fraction → 0; at
−70 mV the VGCC Boltzmann term is ~2.4e-4. Expected: `actin_enlargement` stays at its resting
value, **`E_invasion` stays 0.0000**, `peak_r` flat at the `P_BASAL/P_c` floor (~0.039).

**If the corrected null still ratchets, the effect is not activity-driven at all and the
finding becomes a substantive negative result about the driver — which is Sarah's branch, not
mine to call.**

## Stated limit carried forward

This suppresses a **modeled physiological process** in the control arm. Spontaneous release is
real biology, not an artifact. The corrected null therefore answers *"does traversal-driven
activity ratchet `E_invasion` above what tonic release alone produces"* — it does **not** claim
tonic release should be absent from the model. **The L·ETA-5 finding that tonic release alone
carries `E_invasion` past `invasion_threshold` stands on its own and is not undone by fixing
the control.**
