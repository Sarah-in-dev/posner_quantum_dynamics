# MO — Model 6 Master Orchestrator

> **⚠ STATUS: 2026-07-18 SNAPSHOT — stale as a status board.** The six-PO board (§3: pump / phosphate /
> E_invasion) is superseded and §8 "HOT SET" reads "Nothing dispatched yet." The **current** read-order,
> frontier, and status live in **`docs/README.md`**. Read §2 (adaptations) and §7 (LOCKED) here — those still
> hold — but do NOT treat §3/§8 as the live board.

*Opened 2026-07-18, immediately after the substrate audit (`SUBSTRATE_AUDIT_JUL18.md`).
Method adapted from `talon-orchestrator` + `consumer-acceptance-gate`; the adaptations are
stated explicitly in §2 rather than assumed, because this repo differs from TALON in ways
that matter.*

---

## 1. What the MO is, and is not

- **It is** the planning layer. It plans, writes PO kickoffs, integrates returns, sequences
  work against dependency edges, and holds thin coordination state.
- **It holds ONLY:** the board below, the dependency edges, the surface-ownership map, and
  which POs are hot. Nothing else.
- **It does NOT do the work,** and does **NOT ingest a PO's substance** — findings, traces,
  code dumps, parameter tables. That is what killed the May TALON orchestrator
  conversations three times. If the MO finds itself holding a PO's numbers, it stops and
  points at the research log instead.

**Where state lives (four homes, kept separate):**

| what | where |
|---|---|
| Substance — findings, measurements, traces, provenance | `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`, `RESEARCH_LOG_CALCIUM_DIMER.md`, `SUBSTRATE_AUDIT_JUL18.md`, persisted traces under `results/` |
| Decisions, LOCKED items, rationale | the `model6-*` / `quantum-*` skills |
| Live coordination — hot POs, edges, what's blocked | **this file**, kept thin |
| Method | `talon-orchestrator`, `agent-grounding-protocol`, `consumer-acceptance-gate`, `session-discipline` |

---

## 2. ADAPTATIONS from the TALON method — read before applying it

The TALON orchestrator manual assumes a deploy pipeline this repo does not have. Four
deliberate departures:

1. **There is no deploy queue and no AWS.** The serialized funnel in TALON is deploys; here
   there is none. Runs are local (`model6-codebase-operations`; the physics ECS is a
   separate account and deferred). **The real funnels are (a) CPU — a single probe ran
   130+ minutes CPU on 2026-07-18 — and (b) Sarah's review bandwidth.** Long runs must be
   backgrounded with progress instrumentation, not piped through `tail`.

2. **"Claude Code never touches git" does NOT apply here.** In this repo Claude Code
   commits directly, with Sarah reviewing diffs. That is the established practice of this
   session and is not to be re-litigated back to the TALON rule.

3. **The CONSUMER is a MEASUREMENT, not a UI.** `consumer-acceptance-gate` rule 1 —
   done = demonstrated at the consumer — translates here as: **done = a measurement at the
   data level demonstrates the claim, with its limits stated.** Producer-green in this repo
   looks like "the probe ran", "the code is committed", "errors=0", "the verdict printed
   CONFIRMED". Every one of those has failed here: `683b82f` printed CONFIRMED off a
   flickered edge; the L·ETA-4 probe printed "selectivity holds" when its own positive
   control never fired. **A verdict that cannot distinguish its outcomes is not a result.**

4. **Pre-registration replaces the deploy gate.** Where TALON gates on deploy sequencing,
   this program gates on: pre-register the discriminating quantity BEFORE running; score a
   structural invariant, never times; keep a null that cannot show the effect; and a verdict
   function that can return FALSIFIED and INCONCLUSIVE.

---

## 3. THE BOARD

Six POs. Each is ONE acceptance-scoped objective, days not weeks, and rotates at its
acceptance boundary.

### PO-1 — B2: retire the per-synapse pump site
**Coordinates:** substrate correctness → the pump → the two sites must be one mode.
**Why now:** it is the single highest-value job on the board because it kills four defects
at once, and Sarah has already decided it (2026-07-18): *do not fix the 2π error, retire
the code that contains it.*
**Kills:** (1) the factor-of-2π error (`vibrational_cascade_module.py:315`, `:755-756`) —
dissolves as a side effect of deletion rather than as a separate fix; (2) the calibration
fiction `kT_ref = 22.1` / `r_at_E_ref = 100.0e9`, which has **no derivation to do** because
`r_c` is an artificial reference scale that →0 in large-D; (3) the 40 GHz ω₀ at `:85` — the
mode-conflation bug; per the pin **both pumps are the same 8 MHz collective mode, two
lattice segments, not a fork**; (4) φ/χ — set φ from the pin (`φ = ω₀/Q ≲ 0.8 MHz`), drop
the Zhang citation (its 6 GHz belongs to the retired mode family and adopting it re-imports
the conflation), keep χ with an explicit note that it sets slope above threshold only and
is not load-bearing.
**Also fix:** the docstring at `model6_parameters.py:759` claiming `r_c` is computed from
D, φ, χ — false at the backbone today and false at both sites after B2.
**Acceptance:** the per-synapse site calls `bose_einstein_occupation`
(`model6_parameters.py:46`, verified correct across eight call sites) on the `n_ex = n̄_s`
form; no hand-rolled `hbar`; `kT_ref` and `r_at_E_ref` gone from the live path; a
measurement shows the per-synapse and backbone pumps agree on the same mode; T1′ static
probe still 7/7.

### PO-2 — the phosphate loop
**Coordinates:** chemistry → SOC engine → mass conservation.
**Why now:** Sarah's re-rank moved this above the missing Q sweep dimension, correctly.
**Two items, different consequences — do not merge them:**
- `phosphate_total` goes stale (`atp_system.py:428` recomputes only inside
  `add_phosphate_from_atp`; dimer consumption at `model6_core.py:450-452` decrements
  `phosphate_structural` without it) ⇒ **J-coupling (`atp_system.py:485`) reads a phosphate
  field that ignores dimer consumption.** Contained correctness bug.
- **ATP↔Pi is not mass-conserving** — hydrolysis credits Pi (`atp_system.py:130`), recovery
  regenerates ATP (`:163`, `:169-170`) without debiting any pool. **This is the one that
  breaks Step E.** The finite pool is not finite around the loop ⇒ the reset feedback does
  not close ⇒ **the SOC engine does not exist.**
**Acceptance:** a measurement shows total phosphate conserved around a full
hydrolysis→consumption→dissolution→recovery cycle to a stated tolerance, and J-coupling
tracks dimer consumption. **This gates PO-6's sweep** — running it first would test a
system that structurally cannot self-organize.

### PO-3 — E_invasion: provenance and the ratchet
**Coordinates:** plasticity → the invasion driver → is the 13× live-trial shortfall real?
**Why now:** this is the physics call under L·ETA-3's "protocol, not physics" verdict, and
that verdict is currently resting on an ungrounded constant.
**State (from the 2026-07-18 provenance audit):** the DISCHARGE side is grounded
(`tau_extrude = 180 s`, Honkura 2008, 2-15 min band); the CHARGE side is **not**
(`k_polymerization_max = 0.1` is INHERITED — Bosch 2014 is cited for a *fold-change*, never
for a rate constant, and the arithmetic link appears nowhere). `E_ref = 1.87` is the model's
own asymptote frozen as a constant, **UNVERIFIED** — no artifact ties it to a run.
**The cheap decisive test first:** does `E_invasion` RATCHET across traversals?
`tau_extrude = 180 s` against a 20 s inter-traversal gap predicts **89% retention**, so it
should ratchet strongly — and that prediction comes from the GROUNDED constant, not the
inherited one. Run more traversals, watch `r` climb.
**Acceptance:** a measurement of `r` across N traversals showing ratchet or no-ratchet, plus
a provenance verdict on `k_polymerization_max`. If it ratchets, L·ETA-2 and L·ETA-3
reconcile with no constant touched. If not, the "condensate cannot track behavioural
timescales" reading is a substantive negative result about the network story and is taken
as one.

### PO-4 — the analytical gap, biologically grounded
**Coordinates:** experiment protocol → the silence interval → which subsystems advance.
**Why now:** the spec already exists (2026-07-18 literature pass) and is unusually complete;
the gap currently advances the plasticity clock **1 ms per 30 s**, which freezes CaMKII,
DDSC and spine dynamics across every silence in every multi-trial run.
**Known-contradicted today — ~~RESOLVED 2026-07-18~~, struck by PO-4's rotation 002.** ~~the gap
dissolves dimers at `K_CLASSICAL = 0.05` while the chemistry skill retired that to `0.005` and
`dimer_particles.py:127` uses `0.001` — a 50× spread on the number deciding how much dimer
survives silence.~~ **The gap now runs the GROUNDED `0.005`** — `sweep/run_theta_burst_45s.py:147`,
**read in the file by MO gen-2, not relayed**, and matching `quantum-system-canonical:99`
*"k_classical = 0.005 s⁻¹ … [GROUNDED — Turhan 2024]"*. One live site, because PO-4's
consolidation had already removed the second copy. **The delta was measured, not assumed:** at the
same driven state (2034 dimers pre-gap), dimers lost fell **141 → 15 at a 20 s gap (9.40×)** and
**539 → 66 at 45 s (8.17×)**, inside a bracket pre-registered *before* the change. **Quote the loss
column, not survival** — survival moved only 6.7% at 20 s, which would make a 10× physics change
read as noise. *(PO-4-reported measurements; MO re-run still owed — see the verification ledger.)* **Also:** Jain 2024's DDSC
window is 30-40 s post-induction, i.e. exactly the gap that is skipped, so delayed
commitment cannot resolve in any gap-based experiment as written.
**Honest exclusions to KEEP:** EM field / new dimer formation (no drive), dopamine (clears
~2 s), ATP level (τ≈5 s). Those are "settles fast then clamp", not shortcuts — relabel them
so they are not tarred with the dishonest ones.
**Acceptance:** each subsystem either advances with a cited timescale or is excluded with a
stated reason; a measurement shows committed vs uncommitted spine volume SEPARATING across
an honest gap — the full model has never been allowed to show it.

> **CORRECTION 2026-07-18 (MO's own defect, raised by PO-4 as Q4-2).** This bar previously read
> *"the isolated-module numbers say **1.291 vs 2.389** at +300 s"*. **Those numbers have no
> artifact behind them.** `grep -rn '1.291\|2.389'` over the repo returns hits in coordination
> prose only — no code, no results file, no log entry produces them. **The MO put unsourced
> numbers into a PO's definition of done**, which is this program's characteristic defect class
> (prose asserting a quantity the code does not demonstrate) sitting in the acceptance bar
> itself. PO-4's reproduction (5 reps, dt=0.005, 300 s, thermal noise on) gives **3.7031 ± 0.0649
> committed vs 3.0432 ± 0.0572 uncommitted** — disagreeing in magnitude **and in ordering**.
> The quoted pair is struck; **PO-4 pre-registers against its own reproduction.** Superseded by
> PO-4's measurement when it lands.
>
> **LANDED 2026-07-18 (MO gen-2).** PO-4 reports its measured separation pair as **1.1639 /
> 1.9403** — *"same ordering, different magnitudes, and I did not tune toward them."* **That last
> clause is the acceptance**, not the numbers: the bar was never a target to hit, and a PO reporting
> a miss against a struck number rather than closing the gap to it is the behaviour this program
> optimises for.
>
> **TAGGED AS RELAY, NOT EVIDENCE.** These are PO-4's self-reported figures. **MO gen-2 has not
> re-run them**, and gen-1's defect #8 was propagating exactly such a number into a durable artifact
> and to Sarah, where it proved to be 19× not 100× with the mechanism inverted. **They are recorded
> here as PO-4's claim pending the MO's own run — do not cite them as MO-verified.** Note also they
> are not obviously the same quantity as the 3.7031 / 3.0432 volume pair above; **the MO has not
> established that they are commensurable and is not asserting it.**
>
> **Note the ordering, because it is the substantive part:** volume is HIGHER in the committed
> arm while `E_invasion` is **26× LOWER** (0.0313 vs 0.8222) — commitment redirects enlargement
> into `actin_stable`, and `E_invasion` reads the transient pool alone
> (`spine_plasticity_module.py:412`). Since `r ∝ E_invasion × ca_open`, **a synapse that commits
> loses pump drive.** Escalated to Sarah as a physics call; bears on §8 and on PO-5.

### PO-5 — selectivity, re-asked where it can survive

> **RE-SCOPED BY SARAH — 2026-07-18 20:14Z. The text below is SUPERSEDED; kept per the log
> convention.** PO-5 now tests **§8's keystone as actually written**: *does which dimers bond
> depend on INPUT, at pair resolution?* — `requests/po5-selectivity/mo-rescope-001.md`.
>
> **Why.** §8 is `quantum-system-canonical` §8 Keystone #1, and it **mentions η nowhere**. Its
> owning section (`quantum-computation-and-attribution` §7 #1) states the keystone is
> **"Single-synapse-scale — needs no backbone."** So the η/partition framing below, and the
> "blocked on PO-3" edge, were never what §8 required. **PO-5 is UNBLOCKED.**
>
> **The `P_product` hypothesis below is also retired.** `P_product` is the dimer population
> "which forms only where NMDAR calcium arrived" — i.e. **which regions are eligible**, §8's
> **gate-level**, which §8 says *"collapses to 'scalar as computation.'"* It answers the wrong
> question. Its supporting evidence is independently vacuous (F-4: L·ETA-4's silent synapses were
> not silent).

**Coordinates:** the keystone → input-selectivity → but not through η.
**Why now:** L·ETA-4 measured the plateau raising the condensation drive branch-wide —
silent-synapse `E_invasion` **identical to the driven one to four decimals**, `r` not
separable. **§8's premise fails as written:** η cannot carry input-selectivity once a
plateau is present.
**The surviving hypothesis:** selectivity lives in `P_product` — the dimer population, which
forms only where NMDAR calcium arrived, and the NMDAR AND-gate is measured intact
(silent-synapse NMDAR gain from plateau: **−0.0019**, i.e. zero; consistent with Jain 2024's
7±8% no-glutamate control).
**Acceptance:** a pre-registered test of whether the partition discriminates input patterns
through `P_product` at fixed geometry and fixed plateau, with a null that cannot show the
effect and a verdict that can return FALSIFIED. **Blocked on PO-3** (η must reach threshold
in a live regime before a partition exists to be selective) — see the edges.

### PO-6 — debt retirement and the sweep harness
**Coordinates:** hygiene → sweepability → the drive × damping sweep.
**Scope:** six orphan modules (`eligibility_trace`, `singlet_dynamics`, `calcium_system`,
`implicit_diffusion`, the debug subsystem, and now `em_coupling_module` — imported at
`model6_core.py:84`, never instantiated, keeping ~15 uncited dead constants alive);
~151 dead parameter fields (up from ~120 at may29). Then the sweep harness: add a **Q**
dimension (absent — zero hits in `quantum_dimensions.py`), a drive-amplitude dimension keyed
to the plateau, and an η/`r` readout (`sweep_runner` has no condensation observable).
**Note:** `eligibility_trace.py` carries the P31/P32 isotope parameterisation — check the
isotope kill-switch control wants it before deleting.
**Acceptance:** the sweep runs over Q × drive and reports η/`r`. **Blocked on PO-2.**

---

## 4. DEPENDENCY EDGES

- **PO-2 → PO-6.** The sweep tests self-organization; if the phosphate loop is not
  mass-conserving the SOC engine does not exist and the sweep measures nothing. **Hard
  block, and it outranks the missing Q dimension.**
- **PO-3 → PO-5. RETIRED 2026-07-18 (Sarah).** This edge assumed PO-5 tests selectivity through
  the partition. §8 does not ask for that and its keystone is single-synapse-scale, needing no
  backbone — so PO-5 does not wait on η reaching threshold. The *observation* (η = 0 ⇒ zero
  cross-synapse edges) stands and is still true; it simply does not gate PO-5.
- **PO-1 ∥ everything.** B2 touches `vibrational_cascade_module.py` and the backbone
  parameter block; no other PO owns those. Runs in parallel.
- **PO-4 ∥ PO-1, PO-2.** Different files; but PO-4's `K_CLASSICAL` decision touches
  chemistry rates that PO-2 reasons about — **route the dissolution-rate choice through the
  MO**, do not let two POs pick it independently.
- **Shared-file hazard:** `model6_core.py` is touched by PO-1 (pump call), PO-2 (phosphate),
  PO-4 (gap). Only one PO holds uncommitted edits to it at a time. Commit each slice at
  session boundaries.

## 5. SURFACE OWNERSHIP

| surface | owner |
|---|---|
| `vibrational_cascade_module.py`, backbone params (`model6_parameters.py:759-805`) | PO-1 |
| `atp_system.py`, phosphate path in `model6_core.py` | PO-2 |
| `spine_plasticity_module.py` actin/E_invasion block | PO-3 |
| `analytical_gap` in the drivers, `run_theta_burst_45s.py` | PO-4 |
| `multi_synapse_network.py` partition path, the T1′ probe family | PO-5 |
| the orphan modules, `quantum_dimensions.py`, `sweep_runner.py` | PO-6 |
| the research logs | ALL — each PO writes its own entries; nobody rewrites another's |

## 6. WRITING A PO KICKOFF

Per `talon-orchestrator`: a detailed, thoughtful prompt written fresh — **not a template**.
It leads with the arc (what this PO accomplishes and why now), then coordinates, then:

- the reads named in full, `agent-grounding-protocol` first
- the GROUND specifics — what history, what code to SHOW, what to VERIFY, and the **prior
  art to reuse with `file:line`**
- **`### GROUNDING BRIEF` demanded as the first returned message**, with line-located
  verbatim quotes; paraphrase is bounced
- Owns / Boundaries / Success / Return
- **the acceptance stated as a MEASUREMENT** (§2.3), and the pre-registration requirement
  (§2.4) where anything is scored
- what NOT to touch, and who owns it

## 7. LOCKED — surface, do not relitigate

- **Emergent physics only.** No constant tuned to a downstream target. If the physics does
  not give the result, the log records the gap.
- **Score the ORDER, never the times** (the T1′ scar).
- **Werner 0.5 is a theorem**, not a tunable cutoff; do not apply it to intra bonds.
- **One synapse = one nanodomain**; do not manufacture sub-spine clusters.
- **The −40 mV synaptic cap stays.** Raising it to make η ignite destroys the
  plateau/synaptic separation the BTSP grounding rests on (L·ETA-3).
- **T1′ is CLOSED** — 4/4, p≈3×10⁻⁶. Do not re-run, re-tune, or "improve" the geometry.
  Its probe family is deliberately left NMDAR-shut; wiring it re-validates a closed result
  and is Sarah's call.
- **`f_coherent` stays parked** — degenerate by design; η reaches the partition via
  `k_cross`, not `f_coherent`.

## 8. HOT SET — 2026-07-18

Nothing dispatched yet. Recommended first two, by Sarah's re-rank:
**PO-1** (decided, unblocked, kills four defects) and **PO-3** (cheap ratchet test, and it
unblocks PO-5). PO-2 next, because it gates PO-6.

**Standing MO action, not a PO:** the interim disclosure on `kT_ref` / `r_at_E_ref` — state
in the research log **today** that the per-synapse threshold result is calibrated and not
evidential, ahead of B2 landing. Sarah's instruction: right as interim disclosure, wrong as
an endpoint.
