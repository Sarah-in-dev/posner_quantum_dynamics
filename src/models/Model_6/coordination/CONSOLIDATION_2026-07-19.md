# MODEL 6 — CONSOLIDATION, 2026-07-19

**Written by MODEL6-MASTER gen-2 at the program reset. This is the seating document: a new PO is
briefed from THIS FILE plus the artifacts it points at — never from a conversation.**

**Basis:** 271 commits on `claude/nervous-hertz-7ccff6` (2026-07-18), seven seats, all four surviving
seats' **closing heartbeats captured before retirement** (`79ab369`, `9ee2492`, `90fef4c`, `3048da2`).

> **HOW TO USE THIS FILE.** §1–§5 are the durable state and the operating discipline — they do not
> age quickly. §6–§7 are inventory and will drift. **Every number below is tagged with who measured
> it and whether the MO re-ran it.** A tag of MO-VERIFIED means gen-2 executed it; REPORTED means a
> PO measured it and gen-2 did not. **Do not promote a REPORTED number to a claim without re-running
> it.**

---

# §1 — THE PHYSICS: what is settled, what moved, what is still open

## 1.1 The result of the day — η is a GATE, not a selectivity channel

**`quantum-system-canonical` §4.3 was FALSIFIED and struck.** The retired claim: *"eta is the
selectivity channel … which synapses condense is input-dependent."*

Three independent legs, none the same mechanism:
- **L·ETA-4** — with a dendritic plateau the condensation drive is **branch-global**; silent-synapse
  `E_invasion` identical to the driven synapse's to four decimals (0.2115).
- **L·ETA-5** — `E_invasion` accumulates on **tonic spontaneous release alone**; the undriven arm
  out-gained the driven one (7.46× vs 5.65×).
- **Commitment depletes it** — `E_invasion` **26.2× lower** in a committed spine, independently
  reproduced.

**What survives:** the identity `eta = 0 ⇒ k_cross = 0` is arithmetic and stands. **What fails is
that eta discriminates inputs.**

## 1.2 `E_invasion` has no zero — [MO-VERIFIED]

**It crosses `invasion_threshold` on the RESTING VGCC LEAK ALONE**, glutamate never supplied, in
**well under 100 s**. Gen-2 ran PO-3's probe independently (crossing in the 40–60 s bin vs PO-3's
~80 s; PO-3's "~80 s" is inside run-to-run variation — **use the robust form**). PO-7 later
re-measured across both source trees: **stronger still.**

> **DESIGN RULE, earned three times: a null defined as an ABSENCE is unsatisfiable in this system.**
> Three void controls (L·ETA-4's NMDAR half, L·ETA-5, and the registered L·ETA-5 re-run) all failed
> for this one reason. **Score separation at matched elapsed time; never against zero.**

## 1.3 §2.2 upgraded to DERIVED — the eligibility-window correspondence

With `P_S` floor 0.25, the Werner floor 1/√2, and the live `T_singlet = 216 s`, **`P_S` crosses the
Werner bound at 107.0 s** — inside the ontology's ~100–200 s band. At the retired declared 500 s it
is 247.6 s, **outside**. **216 s is the value that makes the correspondence work.**

## 1.4 The §8 keystone — STILL NO VERDICT. This is the program's central open claim.

Two seats and one 58-minute exclusive slot did not produce one. **What IS established:**
- **Pair structure exists in the RATE** — `D = g_p90/g_p10 = 33.5`, `f_sat = 0.176` [REPORTED, PO-5;
  positive control demonstrated ABORTing before scoring].
- **It does not survive into the TOPOLOGY** — the intra-synapse graph is ~78% complete, one
  component, `largest_frac = 1.000`.
- **83% of bonds come from the deterministic birth loop** (`dimer_particles.py:218-228`) and never
  evaluate `em_rate`. **The MO's own `g`/`coh` framing described the minority pathway.**

**NOT ESTABLISHED, and it must not be inferred:** that any of this defeats §8. **Birth timing is
downstream of input, so a deterministic birth rule is not automatically input-blind.** PO-5 refused
this inference explicitly and gen-2 made the refusal binding. **A successor that shortcuts it will be
wrong for a reason already written down.**

**Q-B is READY TO RUN.** PO-5 rebuilt the scorer offline and validated it (A2.7, zero compute) —
fixed global lattice, absolute cell coordinates, all-run intersection, persisted matrices, separate
offline scorer. **`MIN_CELLS` must NOT be relaxed to obtain a verdict** — if the intersection is
below it, *the instrument cannot resolve pair structure in this geometry* and that is the finding.

## 1.5 The phosphate loop — conservation MET, self-limiting UNEXERCISED

- **Mass conservation MET** — `max |dP|/P = 9.339e-15` against ε = 1e-12.
- **The ATP debit is STRUCTURAL-FIRST**, on literature: F₁F₀-ATP synthase phosphorylates ADP using
  **free inorganic Pi** via PiC/SLC25A3; `phosphate_structural` is the model's free pool. **PO-2
  overturned its OWN pre-registration to land this.**
- **PO2-10 overturns PO2-9 — the one-way valve is NOT closed, it is slowed ~144×** [REPORTED, PO-2,
  from a run killed mid-flight whose **trace survived and was scored offline**].
- **Self-limiting remains UNEXERCISED.** The Pi limit never binds in any run to date.

## 1.6 Fixed-seed nondeterminism — REAL, SCOPE UNMAPPED. Highest-value open item.

**Measured** [PO-7]: `cross_bonds` **1179 vs 1848** at a fixed seed in separate processes (1.57×);
`eta_max` across four driven runs **0.0, 0.0709, 0.0940, 0.1069** — *whether the backbone condenses
at all* was not reproducible.

**Cause shown in code — three `np.random.default_rng()` with no argument**, seeded from OS entropy,
ignoring any caller's seed: `camkii_module.py:199` · `spine_plasticity_module.py:274` ·
`multi_synapse_network.py:1188`.

**BUT the boundary is NOT "driven vs resting"** [MO-VERIFIED]: gen-2 ran PO-4's 2-synapse, 30-drive-step
probe **twice at identical HEAD and got bit-identical results**; PO-4 reports its probe stable across
four processes. **So the nondeterminism is intermittent or configuration-specific — which is worse
than uniform, not better.** The real boundary is *which modules a run reaches*, and it is unmapped.

> **CONSEQUENCE, binding until resolved: do NOT treat any cross-session numeric difference in this
> codebase as evidence of a code change.** Gen-2 did exactly that and was wrong (§5, #19a).

---

# §2 — WHAT LANDED IN THE CODE

| change | status |
|---|---|
| `T_singlet_dimer` 500 → **216 s**, de-duplicated to one source | **MO-VERIFIED** |
| `K_CLASSICAL` 0.05 → **0.005** in the gap (grounded, Turhan 2024) | **MO-VERIFIED** |
| The gap's `k_diss` now carries `template_enhancement` — a **LOCKED** detailed-balance symmetry that was broken | **MO-VERIFIED** (3 independent measurements) |
| ATP debit → **structural-first** (literature) | REPORTED, PO-2 |
| Two `hasattr` input-guards now **raise** instead of silently skipping | REPORTED, PO-1 |
| `q2_t2_p31` re-declared as a **sensitivity** sweep, arms annotated in/out of band | REPORTED, PO-1 |
| `model6-architecture` **F4 corrected** (claimed a file didn't exist; it does — and there are TWO steppers, in two different `sweep/` dirs) | **MO-VERIFIED** |

**The detailed-balance finding is the most consequential:** 97.4% of dimers are `template_bound`,
formation carried the ~33× catalyst, gap dissolution did not. **`quantum-system-canonical:100`
[LOCKED] requires the catalyst apply symmetrically.**

---

# §3 — OPEN ITEMS AND THEIR OWNERS

## 3.1 ⚠ UNOWNED — these have NO seat and will be lost if not assigned

1. **`_remove_dimer` (`dimer_particles.py:252-261`) never pops `_bond_lookup`.** Dead code today; a
   live corruption the moment the death path is exercised. **Gen-2 announced routing it to PO-7 and
   never wrote the file — see §5 #19b. PO-7 verified it never received it.** *(Its tripwire PASSED
   with zero calls in PO-5's Q-B run — it is latent, not firing.)*
2. **The 2034 → 1915 dimer-count discrepancy.** Three candidate causes, none confirmed: PO-2's A2.5
   (PO-2 refutes on timing; PO-4 reports zero `.py` changes in the interval), PO-7's nondeterminism
   (but gen-2's reruns were bit-identical), or unknown. **Gen-2's attribution to A2.5 is WITHDRAWN as
   unproven.**
3. **The `tier5_rnn` third copy of the gap** — determined to be orphan; deletion never routed.
4. **The 7 remaining INERT sweep dimensions** — DELETE verdicts recorded in `quantum_dimensions.py`,
   batch held behind the isotope question, never executed.
5. **J-coupling's residual gap** — Fisher locates *protection* in cluster incorporation; the model has
   no dimer/cluster term, so it computes the **birth-pathway proxy, not the protection mechanism.**
   Explicitly **not** the claim that J should read ambient phosphate (it should not).

## 3.2 Human decisions (Sarah)

1. **Seed the three RNGs?** Physics call — DDSC is stochastic *by design* (Jain 2024, §4.4). **MO
   recommendation: scope it to `multi_synapse_network` first, leave DDSC pending a physics answer.**
2. **L·ETA-5 re-run** — criterion is SET (harder than PO-3 proposed; under it PO-3's own data no
   longer obviously confirms). **MO recommendation: three arms (~130 min) or not at all.**
3. **`P_met` drive change** — gen-1 endorsed CONFIRM; **gen-2 has not re-examined it. Inherited.**
4. **Flat-η re-read scope** — bounded: only consumer is `condensation_boost → k_agg_enhanced`; never
   reaches the partition.
5. **Unpark constants centralisation** for *physical* constants only (four defects in one day).

---

# §4 — STANDING RULES EARNED (the operating discipline; carry these forward)

1. **`git commit -m "..." -- <explicit paths>`. Never `git add` then `git commit`.** All agents share
   one index; a bare commit carries whatever anyone else staged. **Verify `git show --stat HEAD`.**
   Never `checkout`/`restore`/`reset`. *(Root-caused after three sweeps.)*
2. **A run costing a heavy slot MUST persist its scored intermediate; scoring is a SEPARATE offline
   step.** *Compute buys the trace; the verdict is derived from it.* **Validated within hours** —
   PO-5 lost 58 min of good physics to a scoring bug, and PO-2's killed-mid-flight run survived
   because the trace was on disk.
3. **A null defined as an ABSENCE is unsatisfiable here.** Score separation at matched elapsed time.
4. **A reproduction recipe states the commit hash it was measured at.** Seeds are pinned; the model
   underneath them is not.
5. **Re-verify a routed row against the code before dispatch.** A dated record is a claim about when
   it was written.
6. **Rulings on constants name the ACT and the evidence standard, never the identifier.** Forbidding a
   field forbids grounding along with tuning.
7. **When a criterion must be harder, prefer a STRUCTURALLY harder condition over a bigger arbitrary
   multiplier.**
8. **Distrust a lead file's HEADER; read its last entry.** PO-2's header described its first twenty
   minutes for eight hours. **Append-only logs stay current; headers do not, and nothing forces them.**
9. **Weight field-valued statistics by where the population actually is.** A grid mean said 1.015; the
   concentration-weighted mean was ~33. Both are "the mean". **0.03% of the grid held 97.4% of the
   particles.**
10. **Acceptance is a MEASUREMENT, and the demonstration that a check CAN FAIL is what gets verified
    — not the passing run.**

---

# §5 — THE MO DEFECT LEDGER — 19. **The most useful artifact here. Do not repeat these.**

Gen-1's 1–16 are in `leads/model6-mo.md`. **The through-line for 1–15: trusting a document where the
discipline says go to the code.** #16 was a new shape: *a correctly-verified premise reasoned to a
wrong conclusion.*

**Gen-2's own:**

- **#17 — a check applied ASYMMETRICALLY to a symmetric condition.** Flagged one sweep arm as outside
  the ontology band; **three of five were.** Gen-2 arrived carrying the 500 s case (above the band)
  and never asked what the low end did. *Caught by PO-1.* **A spot check wearing a check's confidence.**
- **#18 — a ruling that prohibited a FIELD, not an ACT.** Forbade changing the 90/10 split; §7 forbids
  *tuning to an outcome*. PO-2 had already changed it **under Sarah's direct authorisation**, on
  literature. **Withdrawn.** *Caught by PO-2.*
- **#19a — attributed a discrepancy to the nearest recent change without establishing repeat-run
  variance.** The 2034→1915 shift was pinned on PO-2's A2.5. **Withdrawn as unproven.** *A difference
  between two runs is not evidence of a difference between two versions.*
- **#19b — ANNOUNCED A ROUTING AND NEVER PERFORMED IT.** Ruling 019 states *"Routed by gen-2 to
  PO-7"* for the `_remove_dimer` defect. **No such file was ever written; PO-7's lane contains only
  rulings 018 and 023.** *Caught by PO-7, which verified before writing rather than accepting the
  instruction.* **This is the confabulated-write failure `agent-grounding-protocol` names — a claim
  to have updated something the file disproves. An inherited to-do its supposed owner never received
  is how phantom work outlives a program.**

**POs caught 17 of 19 defects across both MO generations.** *That ratio is the strongest evidence the
board's discipline works — and the reason a PO must never be discouraged from correcting the MO.*

---

# §6 — ARTIFACT MAP (where the evidence lives)

- **Coordination backbone:** `coordination/board.md` (MO-owned registry + every cycle entry) ·
  `leads/<po>.md` (each seat's log; **closing heartbeats at the end**) · `queue/<po>.md` ·
  `requests/<to>/<from>-<id>.md` (28 rulings) · `signals/`
- **Durable program board:** `docs/MO_MODEL6.md` (read **§2 ADAPTATIONS** first)
- **Pre-registrations:** `PREREG_L_ETA_5_RATCHET.md` · `PREREG_L_ETA_6_NMDAR_MAGNITUDE.md` ·
  `PREREG_PO2_PHOSPHATE.md` · `PREREG_PO4_GAP.md` · `PREREG_PO5_UNIT1_G_INERTNESS.md` ·
  `PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md`
- **Research logs:** `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` · `RESEARCH_LOG_CALCIUM_DIMER.md` ·
  `PROVENANCE_EINVASION_CONSTANTS.md` · `K_CLASSICAL_BLAST_RADIUS.md` (**Part II supersedes Part I —
  the two corrections move in OPPOSITE directions**)
- **Reusable instruments:** `dimension_consumer_audit.py` (read-tracing, calibrated against
  known-LIVE/known-INERT) · `gap_phase_coverage_check.py` (mechanical docstring-vs-code check that
  FAILS) · `score_leta5.py` (**offline scoring from a persisted trace — the pattern rule #2 requires**)
  · `resting_leak_probe.py` · `gap_template_symmetry_probe.py`
- **Prior handoffs:** `HANDOFF_SARAH_2026-07-19_AM.md` · `leads/model6-mo.md` (gen-1's, **state
  section stale — its limits and traps sections still hold**)

**⚠ TWO `sweep/` DIRECTORIES EXIST** — repo root and `src/models/Model_6/`. **A check run in the wrong
tree finds a file absent and concludes it was consolidated.** That produced a wrong skill entry that
survived six weeks. **Always use repo-root-relative paths.**

---

# §7 — SEAT HISTORY (retired 2026-07-19)

| seat | outcome |
|---|---|
| **PO-1 / PO-6a** | Acceptance **MO-VERIFIED at measurement level**, both positive controls fired. The sweep audit: 9/19 dimensions inert. **Closing heartbeat carries 9 off-disk items — the best wrap artifact of the day.** |
| **PO-2** | Conservation MET; self-limiting UNEXERCISED. Overturned its **own** pre-registration on literature, then its **own** PO2-9 on a heavy-slot run. |
| **PO-3** | L·ETA-5 scored **VOID rather than published**; F-5 (`E_invasion` has no zero). **Archived before its closing heartbeat — that loss is permanent and bounded.** |
| **PO-4** | Both bars MET and mechanically enforced. Found the **broken LOCKED detailed-balance symmetry**. **Four self-catches, two at direct cost to its own credit.** |
| **PO-5** | The keystone. Unit 1 and Q-A landed; Q-B ran 58 min and returned **no verdict**; scorer rebuilt and validated offline. **Refused the inference that would have overclaimed §8.** |
| **PO-7** | Seated late. Retired the stepper hazard, then found **fixed-seed nondeterminism** and caught the MO's phantom routing. |

---

# §8 — WHAT A NEW BOARD INHERITS

**Ready to run immediately, no design work:**
- **The §8 keystone Q-B** — validated scorer, fixed lattice, persisted intermediates. *The single
  highest-value run available.*
- **The nondeterminism scope map** — which modules reach the unseeded generators. Cheap, and it gates
  the trustworthiness of every driven number in the program.

**Needs a decision before work:** the RNG seeding (§3.2 #1) and the L·ETA-5 re-run (§3.2 #2).

**Needs an owner or it dies:** everything in §3.1 — especially `_remove_dimer`, which no seat has
ever actually held.

*End of consolidation.*
