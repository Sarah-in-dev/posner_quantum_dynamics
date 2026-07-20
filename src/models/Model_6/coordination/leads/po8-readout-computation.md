# PO-8 — readout-time computation (lead file, PO-8-owned)

**Objective:** measure the correlated-domain partition AT READOUT (dopamine-timed) and establish
whether INPUT determines which synapses share a correlated domain at collapse. Either answer is a
result; a clean decomposition-null is reportable.

**Worktree:** `.claude/worktrees/po5-keystone`, branch `claude/nervous-hertz-7ccff6`.
Python: `/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python`.

---

## Heartbeat — 2026-07-20 16:4xZ · ⚠ TIMING CORRECTION — the trace TARGET is 60-100s, not 2-40s

**I inverted the trace timescale and Sarah corrected it.** Our research is explicit:
`coherence-gated-learning:63` — "**Classical eligibility traces (τ≈2s) die after seconds.** Our
coherence window (~100-200s)... cause-effect separation is 30-90+ seconds"; `quantum-biology-primer:21`
— "across **60-100 second windows**, when all known classical neural mechanisms degrade on much
shorter timescales". **The 2s (BTSP) / 40s (CaMKII) are the CLASSICAL baseline being BEATEN, not the
requirement.** The trace TARGET is 60-100+s (up to 200s) — that IS the quantum differentiator; a 40s
trace would carry no advantage (CaMKII already does 40s). The advisor's Q4 (cross dies 74s, intra
107s, from T2=216s, lambda=5) is EXACTLY the target and is correct.

**Consequences, corrected:**
- My external-review conclusion "200s over-provisions, drop to 40-60s" is WRONG (misread the classical
  baseline as the requirement). Corrected in-place in `PO8_EXTERNAL_LIT_REVIEW_2026-07-20.md`.
- **Unit A's PURPOSE, now clear:** the unmoored cross rate kills bridges at ~18s — INSIDE the classical
  regime, useless. The derived rate makes bond death COHERENCE-LIMITED so the trace reaches 74-107s —
  the quantum window. That is the enabling fix.
- My continuous-drive Unit A test (bonds ~6-30s) measured the WRONG regime (turnover, not aging). The
  `analytical_gap` measurement is the right one; ACCEPTANCE BAR = does the correlated domain survive
  to ~74-107s with the derived rate (ON) and die there, vs collapse early (OFF).
- lambda: the advisor's call, NOT mine to flip on a subagent lit-take. Recorded as tension, not a
  recommendation.

## Heartbeat — 2026-07-20 16:xxZ · THE INVESTIGATION: what we have vs what we're missing for the trace

Investigation (internal code map + external lit review running), synthesising the advisor's guidance
rather than probing blindly. The advisor already gave the mechanism (trace falls out of T2 =
coherence death at the Werner floor), the knob (lambda), and the capability (signed partition).

### WHAT WE HAVE (internal, code-confirmed)
- **The trace mechanism exists and is validated: `analytical_gap`** (`run_theta_burst_45s.py:45`).
  During a readout delay it ages the population by the INTRINSIC dissolution
  `k=K_CLASSICAL(0.005/s)*(1-singlet_excess)*template_enh` (tau~200s, Turhan 2024, GROUNDED), decays
  P_S toward 0.25 (T_eff), removes bonds at P_S<0.5 (coherence death = the Werner-floor crossing),
  and stochastic disentangle `0.01*(1-PSi PSj)`. THIS is "the trace falls out of T2."
- **The canonical READOUT protocol already exists** (`:615-681`): theta-burst traversals (WRITE) ->
  `analytical_gap(DOPAMINE_DELAY)` (AGE) -> `net.step(reward=True)` (READ, fires the measurement).
- Domain metric (Unit-18), Werner bound, ignition on the net.step path — all in hand.
- **Unit A** (`physical_release_rate`, derived 9.63e-3/s, gate-clean) — corrects the unmoored cross
  rate so cross-bond death is coherence-limited during the gap.

### WHAT WE'RE MISSING / THE REAL OBSTRUCTIONS (measured or flagged)
- **My earlier tests used the WRONG regime.** Continuous drive = steady-state turnover (~30s), and my
  improvised "quiet" collapsed the calcium peak (step_population np.max, L·PO5-11) culling in ~30s.
  BOTH bypassed `analytical_gap`. Under the intrinsic-dissolution gap the population ages at ~200s,
  not ~30s — so the trace CAN reach the coherence window. **Correct protocol = analytical_gap.**
- **lambda = 5um vs 214um** (advisor Q3): sets whether the trace is two-timescale (cross dies 74s,
  intra 107s) or one-timescale (both ~106s). The advisor argues 5um is a rate-vs-fidelity category
  error vs the locked L_coh=214um. Being MEASURED as an arm here, not asked.
- **External lit review RUNNING** (subagent): does the literature support tau_dimer~200s and
  T2~216s, and does the required behavioural (BTSP) window match coherence or chemistry? This is the
  "what are we missing" question the model's own numbers can't answer.

### THE MEASUREMENT (running): `sweep/po8_unitB_trace_readout.py`
Canonical protocol (imports `run_burst_traversal` + `analytical_gap`, does NOT edit them). WRITE 3
theta-burst traversals -> snapshot correlated-domain partition across a swept gap [0..120s] ->
2x2 arms {rate off/on} x {lambda 5/214}, free draws, NO seeding. Control first: confirm the
theta-burst drive IGNITES (peak eta>0) and writes a live domain before any delay. Then: does the
domain survive the gap, and does it die at the predicted Werner crossings (74s / 107s / 106s)? That
curve IS the eligibility trace, and off-vs-on / lambda-5-vs-214 says what enables and shapes it.

## Heartbeat — 2026-07-20 15:xxZ · L·PO8-1 RETRACTED; Unit A (physical release rate) BUILT + gate-clean; acceptance running

**L·PO8-1 WITHDRAWN by me (do not cite).** Root cause: I never read
`docs/PO7_TECHNICAL_BRIEF_2026-07-20.md`, which the kickoff names as required. The ~7 s "trace" was a
protocol artifact: I invented a drive-then-quiet protocol that collapses `peak_conc` with tau~42 s,
and `step_population` targets `np.max` (a PEAK, `L·PO5-11`), so killing the drive culls the population
far faster than real dimer chemistry. It also contradicted the brief's tau_dimer=200 s and validated
commit-rate decay over 0-120 s — checks I skipped. Retraction committed; DECISION RECORD row flipped.

**System verified intact after all my probes:** network regression digest `515772101786800`
reproduces; **zero system .py modified by any probe** (all `sweep/po8_*` are new files).

**Unit A BUILT (the eligibility-trace lever).** `multi_synapse_network.py`: opt-in
`physical_release_rate` (default False). ON replaces the unmoored cross rate
`K_DISENTANGLE_BASE*(1-eta*P_product)` (~0.056/s, tau~18 s — bridges die ~4x before their ~74 s
Werner crossing) with the DERIVED driven-NESS rate `k=1/T2+1/tau_dimer=9.63e-3/s` (tau=103.8 s),
constant in F (advisor Q4: trace falls out of T2; fidelity selection already emerges from
k_cross prop F). **OFF-path network gate bit-identical: offpath digest 515772101786800 reproduces
with flag off.** Committed.

**Acceptance measurement RUNNING** (`sweep/po8_unitA_bond_lifetime.py`, standard rig, continuous
`net.step` drive, NO seeding, 100 s, off-vs-on, 2 free draws/arm): cross/intra BOND lifetime AND
DIMER lifetime (the hard cap — a bond dies with its dimer regardless of release rate), with
right-censoring reported honestly. Predicts: ON should push cross-bond lifetime from ~18 s toward the
~74 s Werner crossing IF dimer lifetime allows; the dimer-lifetime number tells us whether a SECOND
fix (population persistence) is needed for the trace to reach the behavioural timescale. Acceptance is
the measured lifetime distribution, not "it ran".

## Heartbeat — 2026-07-20 14:5xZ · PROCESS CORRECTION + Unit B1 (graph persistence) running

**Process correction (Sarah, via the README).** I was using chat as the coordination backbone —
asking Sarah to ground me and to adjudicate questions that are MEASUREMENTS. The README is explicit:
durable shared state, not messaging; do all non-gated work; escalate only what is gated, in
`queue/` with ask+why+recommendation; acceptance is a MEASUREMENT. Corrected: state lives here from
now on; the open "does the graph survive a quiescent delay" question is being MEASURED (Unit B1),
not asked.

**Grounding gap closed (was real).** I had read the research-log entries L-PO7-1..5 but NOT
`docs/handoffs/PO7_HANDOFF_2026-07-19_EVENING.md` nor
`coordination/requests/po7-provenance-network/notes.md`, both of which the kickoff points at. Now
read in full. What they change:
- The graph is **write-once** — bonds do not dissolve at the coded rate; the graph is not meant to
  fade on its own. PO-7 always measured under SUSTAINED drive at end-of-run; **nobody has run a
  drive-then-quiet readout**, so that protocol is untrodden ground (not prior art I ignored).
- Unit 11: with the shared ledger and NON-provenance intra bonds, cross bonds are structurally
  STARVED (intra locks all 4 nuclei first) — and §8's open question is whether that is physics or
  UPDATE ORDER. Under `provenance_bonding=True` intra is sparse (~3.87 of 4 nuclei free,
  L-PO7-4 §1), so my rig sits in the NON-starved regime and forms ~1900 cross bonds. Config-
  dependent; worth stating whenever cross-bond counts are quoted.
- notes.md **Q6**: ON-vs-OFF drive confounds identity with DENSITY (inactive synapses make no
  dimers, so edges can only form among active ones — `Q_act=+0.0000`, Newman's degenerate value).
  The prescribed fix is **HIGH-vs-LOW drive with ALL synapses above dimer-forming threshold**.
  My Unit B prereg must adopt this; "same number of synapses active" is NOT density matching.
- notes.md Q1: under `pattern="linear"` a contiguous split IS the spatial half — interleaved ARM 2
  is the discriminator (already in my prereg skeleton).
- Handoff §9: three unseeded RNGs make the 7-synapse rig non-reproducible run-to-run, so any single
  run is ONE DRAW from a distribution. (Correct for physics — free draws — but it means n=1 tables
  must never be reported as values.)

**⚠ My B0 quiet-protocol result was NOT a finding — mechanism identified, and it is DESIGNED
BEHAVIOUR.** I reported a table showing the graph collapsing ~253->7 domain size within 20 s of
drive cessation. Cause, SHOWN in code: `dimer_particles.step()` separates
"2. Population: birth/death to track concentration (FAST chemistry)" from
"3. Coherence: T2 decay (SLOW quantum)", and `step_population` sets
`target_count = peak(dimer_concentration)*az_volume*N_A`, culling dimers when count>target and
**deleting their bonds** (`_remove_all_bonds_for_dimer`). Zeroing the drive collapses calcium ->
concentration -> target -> the population is culled and the graph goes with it. **"Write-once"
protects a bond from DISSOLVING; it does not protect a bond whose dimer is CULLED.** So the graph's
lifetime may be capped by the FAST chemistry, not by T_singlet=216 s or the bond-release rate.
That is a real and important possibility — but it needed measuring, not asserting, and n=1.

**Unit B1 launched (3 free draws, no seeding, net.step drive):**
`sweep/po8_unit_b1_graph_persistence.py` — drive 20 s, then 60 s quiet, sampling every 1 s:
peak dimer_concentration, target_count, n_dimers, culled deaths/s, births, n_intra, n_cross,
mean P_S, mean correlated-domain size. It separates the three candidate causes of graph loss:
(a) CULLING (deaths track a collapsing target), (b) BOND DEATH (bonds fall while dimers persist),
(c) COHERENCE (mean P_S falls toward the 0.5 floor). Whichever dominates **sets the true
eligibility-trace lifetime** — which is the number Unit B's readout delay must be chosen against.
Instrumentation is read-only (step_population wrapped, established probe pattern); no physics
modified.

## Heartbeat — 2026-07-20 13:2xZ · advisor R6 answered; B0 probing; a self-correction; third decision (lambda) queued

**Since 12:51Z:**
- Answered the advisor's R6 check-in (relayed by Sarah) with a grounded physics read: domains stay
  synapse-scale then collapse abruptly ~107 s (arithmetic independently reproduced); the abrupt
  collapse is INTRA cores dissolving as a unit (common P_S(t)), not branching; the trace is a SCOPE
  trace (multi-synapse binding decays before per-synapse). Adopted the (A)-safe framing (drop
  "program in superposition"; monogamy is the load-bearing quantum constraint).
- Queued **Q3 (lambda = 5 um vs 214 um)** — the advisor's biggest item, GROUNDED: `coupling_length_um
  = 5.0` is used as the entanglement fidelity weight but our LOCKED feasibility calc #1
  (`model6-network-layer-feasibility-may30:73-78`) puts the condensate coherence length at 214 um.
  It flips Unit B from two-timescale to one-timescale. Not my call (LOCKED-adjacent); escalated.
- Wrote `docs/PREREG_PO8_UNIT_B_READOUT_KEYSTONE.md` (SKELETON) — fixes the design invariants
  (density-matched sync-vs-stagger, SYNAPSE-level scoring to avoid the dimer-level trap inherited
  from the superseded Unit-2 prereg, the decomposition-null falsifier), with lambda/API/framing slots
  marked GATED.
- Built `sweep/po8_unit_b0_readout_time_sweep.py` (readout-time domain sweep, reuses Unit-18 metric
  verbatim, no seeding, run/analyze CLI, per-draw fsync).

**⚠ SELF-CORRECTION (SHOWN by data, `sweep/po8_smoke_timing.py` + `po8_probe_ignition_lambda.py`):**
I earlier told Sarah the readout-time domain-size curve is lambda-INDEPENDENT ("intra F=P_S^2 carries
no spatial weight"). That is wrong about DOMAIN SIZE. Data: pre-ignition (eta=0, no cross bonds) the
intra-only provenance graph is SPARSE and domains are only ~7-14 dimers; the synapse-scale domains
(Unit-18's 468) appear only AFTER ignition (~10-20 s) via CROSS bridges. So domain size is
cross-bridge dominated => lambda-DEPENDENT (maxF measured 0.784 at lambda=5, consistent with the
0.815 ceiling). Consequence: the collapse TIMING is lambda-dependent (cross dies ~74 s at lambda=5 vs
~107 s at lambda=214), so B0 must run at BOTH lambda values. Correcting this makes the lambda ruling
MORE central, and note lambda is a plain constructor kwarg (`multi_synapse_network.py:1034`), NOT an
owned-file edit and NOT a locked default I am changing — so I can run both arms as an experiment
variable with zero code changes and hand Sarah the evidence.

**Facts SHOWN (net.step drive, no seed):** ignition onset ~9-10 s (peak_eta 0->0.05-0.08, n_cross
0->28-38, crossF_med ~0.75); cost ~7-8 wall-s per sim-s at V~1600 and rising with dimer count. A
25 s two-lambda probe is running to confirm dimer-count plateau (feasibility of a 120 s run) and the
lambda=214 maxF (~0.99 expected) before I launch the full B0 sweep as parallel background workers.

**Next:** confirm V plateaus -> launch B0 (both lambda, >=5 draws lambda=5 primary, T_MAX~115 s,
4 workers) -> analyze collapse shape/timing. Unit A and scored Unit B remain gated on Q1/Q2/Q3.

## Heartbeat — 2026-07-20 12:51Z · GROUNDED, brief returned, two decisions queued

**Status:** grounding complete; no code written yet (correct order). Unit A and Unit B both have
premise corrections that need a ruling before building — queued to
`coordination/queue/po8-readout-computation.md`.

### Grounding done
Read in full: kickoff; skills `agent-grounding-protocol`, `session-discipline`,
`experiment-design-patterns`, `model6-entanglement-partition-werner`, `quantum-system-canonical` §5,
`model6-research-findings-may29`; `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` L·PO7-1…5;
`PO7_ADVISOR_REVIEW_2026-07-20.md`; `coordination/README.md` + `board.md`.
Code SHOWN: `multi_synapse_network.py` (`step`, `_update_entanglement`, `_step_network_provenance`,
`_apply_measurement_collapse`, stimulus path), `dimer_particles.py` (`step_entanglement`, provenance
branch, disentanglement block), `sweep/po7_unit11_shared_ledger.py`,
`sweep/po7_unit18_correlation_domains.py`.

### Three corrections to inherited premises (reported, not worked around)

1. **Unit A's rate premise is wrong-path and sign-inverted.** Three release paths exist:
   - cross-synapse `multi_synapse_network.py:488` — `k_diss = 0.1*(1 - eta_factor*P_product)` gives
     **0.051–0.094 /s (tau 11–20 s)** at observed eta 0.2–0.49, P_S 0.55–1.0: ~10x FASTER than the
     physical 9.63e-3/s, not 96x slower.
   - intra non-provenance `dimer_particles.py:665` — ~1e-4/s at coh≈0.99. This is the cited number.
   - intra with `provenance_bonding=True` (**the physical config**) — `dimer_particles.py:580-586`
     early-returns before the disentanglement block. **No rate at all**; bonds die only by coherence
     death at the Werner floor (~107 s step function).
   Corroborated by the locked framing at `model6-research-findings-may29:91`:
   "The current `K_DISENTANGLE_BASE = 0.1` (cross half-life ~9s) is unmoored from physics."

2. **Dopamine-triggered decoherence is largely unimplemented.** `_apply_measurement_collapse`
   (`multi_synapse_network.py:1776-1845`) writes no `P_S`, never touches
   `intra_synapse_bonds_cache`, and removes only cross bonds between CaMKII-**discordant** synapses
   at p=0.8. With `disable_auto_commitment=True` nothing commits => zero bonds removed.
   `collapse_factor = 0.3` is assigned and never read. **Consequence for Unit B:** what fragments
   domains at readout is P_S(t) decay under T_singlet=216 s, which happens with or without dopamine.
   Dopamine sets the CLOCK, not the decoherence. Unit B stays well-posed; the framing must say so.

3. **Unit B's input contrast is not currently expressible.** `net.step` broadcasts one stimulus dict
   to all synapses (`:1257-1260`, "In future, could have synapse-specific stimuli"). Synchronous-vs-
   staggered neighbour-group activation at matched density REQUIRES per-synapse stimuli. Per-synapse
   `s.step()` is the forbidden path (eta==0, L-PO7-4 section 7). So Unit B needs a backwards-
   compatible per-synapse stimulus argument on `net.step`.

### Instruments confirmed
- Unit-18 correlated-domain metric (`sweep/po7_unit18_correlation_domains.py:84-147, 242-279`) is
  complete and reusable verbatim: p=(4F-1)/3, w_e=-ln p_e, bounded Dijkstra D_MAX=8.0,
  S(u)=sum exp(-d), effective domains = sum 1/S(u). No F-threshold anywhere. Good.
- Regression gate: `PO7_U11_MODE=offpath PO7_U11_TAG=<tag> python sweep/po7_unit11_shared_ledger.py`,
  compare printed digest to `515772101786800`. NOTE: it is an eyeball diff — there is **no assert**.
  I will add one in my own harness. It seeds (`np.random.seed(4242)`); legitimate there and only there.

### Discipline held
No seeding in any physics measurement. Drive via `net.step` only. Explicit-path commits only.
Werner bound 0.5 untouched. Cross-synapse provenance not reopened.

### Next (once the two rulings land)
Unit A build + off-path regression proof; then pre-register Unit B (with the stated falsifier and a
demonstrated verdict-function failure) before any scoring.

### Known limits
- `.claude/skills` is not symlinked into this worktree; skills read from the repo-root tree.
- Pre-registration, the guard/falsifier rule, "demonstrate the verdict function failing", and
  minimum-draws are NOT written in `experiment-design-patterns` — they are program practice and
  kickoff text, not skill text. Following them as instructed; flagging the gap.
