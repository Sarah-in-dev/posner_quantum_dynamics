---
name: research-log-calcium-dimer
description: >
  Append-only research log + decision record for the Model 6 calcium → dimer-formation
  revalidation (the supersaturation-gate / grounded-calcium / conserved-phosphate coupled
  correction). The PRIMARY provenance record the paper draws from: every load-bearing
  claim carries a source and an epistemic-status tag, every decision carries its reason
  and date. The dimer-formation-chemistry skill and quantum-system-canonical distil the
  LOCKED decisions; this log carries the granular "why" and the literature evidence behind
  them. Read when continuing the calcium→dimer work, writing it up, or reconstructing why a
  constant or threshold was chosen. APPEND newest entry at the top of the LOG; never rewrite
  history — supersede with a dated note.
---

# Research Log — Calcium → Dimer Formation Revalidation

## Purpose & how to use this

This is the **decision-provenance record** for the calcium→dimer revalidation. It exists so
that when we write this up, every number and every modeling choice can be traced to its
source and the reason it was chosen — and so a future session can see *why* a decision was
made, not just what it was.

- **Append, don't rewrite.** New work goes in a dated entry at the top of *The Log*. If a
  later finding overturns an earlier one, add a new entry that **supersedes** it with a
  pointer — leave the original in place (the paper needs the trail, including the wrong turns).
- **Two companion layers:** `model6-dimer-formation-chemistry` (skill) and
  `quantum-system-canonical` (ontology) carry the **distilled LOCKED decisions**; this log
  carries the **granular evidence and rationale**. When a decision locks, summarize it in the
  skill and point back here for the "why."
- **Epistemic-status tags** (same legend as `quantum-system-canonical` §0):
  `[PROVEN]` literature-established · `[GROUNDED]` tied to a named measurement ·
  `[MODELED]` defensible choice not forced by physics · `[INFERRED]` follows from the model ·
  `[CONTESTED]` an unsettled bet · `[LOCKED]` settled, not relitigated without new physics.
- **Discipline (LOCKED):** emergent physics only — no constant tuned to a downstream target.
  If the physics doesn't give the result, the log records the gap; it is not a license to
  slide a knob. Sources are cited inline; the full reference list is at the bottom.

---

## DECISION RECORD (running summary — newest first)

| # | Date | Decision / finding | Status | Entry |
|---|------|--------------------|--------|-------|
| SWEEP-4 | 2026-07-18 | **108 of 220 declared parameter fields are UNREAD — and the two constants the coherence-window arithmetic rests on are declared in a place the model ignores.** PO-6a Unit 2 second half, `sweep/dead_parameter_audit.py` (`7c48696`). Static AST only; never imports or runs the model, so it was safe alongside PO-5's exclusive heavy slot. **Method, and the error direction is the whole point:** counts `ast.Attribute` reads in **Load** context (a Store is a write, not consumption) plus names passed as string arguments to `getattr`/`setattr`/`hasattr`. A name-keyed attribute scan **over-reports liveness**, so it can call a dead field live but **cannot call a live field dead** — zero on both channels is a sound DEAD verdict, while a nonzero count is *not* a claim of real use. Same principle as the read-tracer: choose the instrument whose error direction cannot manufacture the finding you are looking for. **The audit's own control caught a bug in the audit.** Version 1 counted *every* string literal and reported the known-dead `kT_per_modulation_unit` as LIVE, because `quantum_dimensions.py` declares `variable="kT_per_modulation_unit"` as dimension **metadata**. A name in a data table is not a read; counting it that way would have silently suppressed real dead fields — the exact class this audit exists to find. Channel narrowed to dynamic-accessor arguments; both controls pass. **RESULT: 220 declared, 112 live, 108 DEAD.** Worst: `PNCParameters` **8 of 8**, `PosnerParameters` 16/18, `MultiSynapseParameters` 12/14, `QuantumParameters` 17/27. The substrate audit reported ~151; **this is not a correction of that number** — different method, and 108 is a **lower bound** because the scan over-reports live. **THE FINDING THAT MATTERS MORE THAN THE COUNT — the `T_singlet_dimer` defect (ruling 006) repeats on the two constants the Werner arithmetic actually uses:** (1) **`QuantumParameters.singlet_thermal = 0.25`** (`:412`) is read **only** by `singlet_dynamics.py:129`, an **orphan nothing imports**, while the live code hardcodes `0.25` in **three** places — `dimer_particles.py:283`, `quantum_coherence.py:101`, `multi_synapse_network.py:435`. (2) **`QuantumParameters.singlet_entanglement_threshold = 0.5`** (`:411`) is **DEAD outright**; the live Werner bound is a separate class constant `WERNER_ENTANGLEMENT_BOUND` at `multi_synapse_network.py:94`. **These are exactly the two numbers used to select 216 s over 500 s** (`P_S` thermal floor 0.25 and the `1/√2` pair bound), and that PO-1 re-derived for the ruling-017 bracket annotation as `t_cross = 0.49516·T`. They are load-bearing, duplicated, and declared where nothing reads them — so a future change to either would have to find three literals and a class constant, and the declared parameter would still be ignored. **ROUTED, not fixed:** no verdict has been given, and the literals sit in PO-5's files. Adds an **ORPHAN-ONLY** pass — fields live on the main tally *only* because an orphan reads them; exactly one today (`singlet_thermal`), which belongs in the orphan deletion batch rather than a separate one. **Nothing deleted; deletions remain held behind the isotope gate.** | [GROUNDED, AST-verified; controls passed] | commit `7c48696` |
| SWEEP-3 | 2026-07-18 | **THE MODEL'S DIMER COHERENCE TIME IS NOW A PARAMETER, SET TO 216 s — and the audit that found it upgraded a canonical claim.** MO ruling 006; commit `3632fce`. **The direction of the fix was ruled one-way and matters:** the PARAMETER was moved to the physics (`quantum.T_singlet_dimer` 500.0 → **216.0**), never the hardcoded 216 adjusted to the declared 500. **WHY 216, from the model's own constants** (MO's arithmetic, independent of the fact that 216 was what happened to be running): with the `P_S` thermal floor **0.25** (`dimer_particles.py:26,62`) and the Werner separability floor **1/√2**, `P_S` crosses the Werner floor at **107.0 s** for `T_singlet = 216` and **247.6 s** for `T_singlet = 500`. `quantum-system-canonical` §1/§2.2 place the coherence window at **~100–200 s** and call its mapping to the BTSP window *"the load-bearing correspondence"*. **107 s is inside that band; 247.6 s is not** — so 216 s is not merely the live value, it is the value that makes the ontology's central correspondence hold, and the declared parameter would have broken it. The 500 s cited *"~100-1000s from Agarwal"*, a range so wide it constrains nothing. **Consequence for the ontology:** `quantum-system-canonical` §2.2 now tags that correspondence **DERIVED from the model's own constants** rather than INFERRED — i.e. a sweep-harness audit ended up strengthening the epistemic status of a claim in the program's canonical layer. **DE-DUPLICATED:** 216.0/0.4 were literals in **two** files (`dimer_particles.py:288`, `quantum_coherence.py:107`) — two copies of one physical constant, the same defect class as the two `coupling_length` constants and as `phosphate_total` naming two objects. Both now read `QuantumParameters`; note the sites hold *different* param objects (`dimer_particles` → `Model6Parameters`, needs `.quantum`; `quantum_coherence` → `QuantumParameters` directly). New field `T_singlet_dimer_P32 = 0.4` makes the isotope control a parameter too. **VERIFIED BIT-IDENTICAL:** post-change fingerprint `(695, 0.998892874976)` equals the pre-change baseline computed from git at the same seed/steps — the de-duplication changed nothing, which was the requirement. **THE DIMENSION IS NOW GENUINELY SWEEPABLE** (PO-6a Unit 3's objective): read-trace `0 → 48` reads; driving `T_singlet_dimer` 50/216/500 moves `mean_P_S` 0.998512/0.998893/0.998949 — monotonic, correct sign (longer singlet lifetime ⇒ higher `P_S`). Audit INERT count **9 → 8**, and the effect case is retained as a **regression guard** so a revert to local literals would be caught. **Honest limit, not oversold:** the effect is small at short horizons *by construction* — `T_singlet` is a ~100 s time constant and the probe runs 40 ms — so what is demonstrated is that the parameter reaches the physics with the right sign, not the magnitude a real sweep would see. **216 s is NOT to be retuned:** Agarwal-grounded and load-bearing for §2.2; changing it to improve a downstream result is the emergent-physics violation (`MO_MODEL6` §7). **Bearing on the orphan hold:** `singlet_dynamics.py` was the only reader of the wrong 500 s value and now reads a corrected field; the isotope hold still stands until the P31/P32 kill-switch question resolves. | [GROUNDED, measured; bit-identity verified] | commit `3632fce`; MO ruling 006 |
| GAP-4 | 2026-07-18 | **`K_CLASSICAL` 0.05 → 0.005 IN THE GAP — the retired rate was live, and the delta is measured.** The gap dissolved dimers at an **uncited `0.05`** while `quantum-system-canonical` §3 carries **`0.005 s⁻¹` [GROUNDED — Turhan 2024]** (cluster lifetime τ≈200 s) and `model6-dimer-formation-chemistry` §1 item 4 records the retirement. `sweep/phosphate_conservation_probe.py:69` already ran `0.005` — a third site corroborating. PO-4's consolidation (GAP-1) is what reduced this to **one** site. **MEASURED at the same driven state (2034 dimers pre-gap), pre-registered bracket (`PREREG_PO4_GAP.md` AMENDMENT E) before the change:** at 20 s, dimers lost **141 → 15 (9.40×)**; at 45 s, **539 → 66 (8.17×)**; survival 0.9307→0.9926 and 0.7350→0.9676, inside the registered bracket at both rates and both gaps. **Stated carefully because survival UNDERSTATES it by an order of magnitude — the loss column is the honest one** (survival moved 6.7% at 20 s while loss fell 9.4×). Not exactly 10× because `k_diss = K·(1−singlet_excess)` and fresh dimers enter a gap at `se ≈ 0.997`, so coherence protection suppresses dissolution at gap entry and `K` only dominates as `P_S` decays toward thermal — the shortfall is that suppression, not an error. **NOT DAMPED** per MO rotation 002: **every multi-trial dissolution number produced before 2026-07-18 inherits `0.05` and must be re-read.** **The GAP-2 headline is INVARIANT to the change — verified by re-running, not assumed: ΔV = +0.7764 at `0.05` vs +0.7727 at `0.005`, a 0.0037 difference inside thermal noise**, as expected for an actin/volume observable that does not read the dissolution rate. | [GROUNDED, MEASURED] | `docs/PREREG_PO4_GAP.md` AMENDMENT E; probe `sweep/gap_dissolution_probe.py` |
| SUPER-1 | 2026-07-18 | **SUPERSEDES D21(5) AND `SUBSTRATE_AUDIT_JUL18` ITEM 16 — the `coupling_weights` omission was ALREADY FIXED when both were written up as open.** D21(5) states *"No cross-synapse bonds form during trials at all — `run_trial` omits `coupling_weights` (`run_spatial_discovery.py:446-449`) and `_update_entanglement` early-returns without them"*, and audit item 16 records the same as *"a gap in that fix"*. **`git log -S` places the fix at `15abd39` ("items 7, 3, 5"), whose ranked item 7 was exactly "Pass `coupling_weights` in `step_with_coordination` and `run_place_field_learning`"** — landed before the MO session opened. **Measured 2026-07-18 (PO-4, `sweep/coupling_weights_reach_probe.py`, independently re-run by the MO): `coupling_weights` reaches `_update_entanglement` at BOTH driver sites — site1 6/6 calls, site2 37/37, `all_arrived = true`.** The probe was demonstrated capable of detecting the historical omission before it was allowed to pass. **Bonds formed: still 0 — but the sole remaining blocker is `eta = 0` at every synapse** (`k_cross ∝ √(η_i·η_j) = 0`), i.e. **physics, not wiring**, converging with L·ETA-1, L·ETA-3 and the 2026-07-18 falsification of `quantum-system-canonical` §4.3. **Process note:** the MO routed D21(5) to a PO as a live defect without checking for a superseding commit. **A dated decision-record row describes the moment it was written; it is not a live status.** Recorded as MO defect #14 so the next reader does not route it again. **LIMITS of the measurement:** 3 synapses, 2.0 s budget — it measures the CALL PATH, not long-run topology. | [GROUNDED, measured; MO-verified] | PO-4 Q4-5; `15abd39` |
| SWEEP-2 | 2026-07-18 | **THE MODEL'S DIMER COHERENCE TIME IS A HARDCODED 216 s LITERAL, AND THE DECLARED PARAMETER SAYS 500 s.** PO-6a Unit 3 (`427b47c`). Investigating the two critical inert dimensions produced **three different diagnoses, not one wiring bug** — and the instruction "do not invent a consumer" was load-bearing, because for one of them the consumer exists and for the other it does not. **(A) `q2_t2_p31` — CONSUMER HARDCODED, VALUES DISAGREE.** The physics *does* use a dimer singlet lifetime, so this is NOT "T2 does not matter". But it is a literal: `T_singlet_P31 = 216.0` / `T_singlet_P32 = 0.4` at **`dimer_particles.py:288-289`**, **duplicated** at `quantum_coherence.py:107-108`. The parameter the sweep writes, `quantum.T_singlet_dimer = 500.0`, is read **only** by `singlet_dynamics.py:122` — an orphan never imported. **So the declared dimer coherence time (500 s) is not the one the model runs (216 s); anything quoting 500 s is quoting a number the model does not use**, and the sweep has been varying the dead one. The dimension's own label *"current default 500s"* is wrong about the live model. The isotope weighting around the literals (`fraction_P31`) is live and correct — only the lifetimes are hardcoded. **(B) `q2_j_coupling_hz` — NO CONSUMER, and scale-mismatched to its own name.** `quantum.J_intrinsic_dimer = 15.0 Hz` has **one write and ZERO reads anywhere in the tree**. J-coupling itself is live by a different route: the ATP-derived field (`atp_system.py:296-339`, from `atp.J_PO_free`/`J_PP_atp`) plus per-dimer `j_couplings_intra` drawn from a **hardcoded `N(0.15, 0.15)`** at `dimer_particles.py:49` — mean 0.15, i.e. ~100× below this parameter's 15.0 Hz. Re-targeting the dimension requires deciding **which J is meant**; that is a physics call. **(C) `q2_k_agg_baseline` — the silent `hasattr` no-op is NOT mechanically fixable.** `k_base = 18918.67 M⁻¹s⁻¹`, while the dimension's values are `[0.001, 0.005, 0.01, 0.05]` — which match `k_classical = 0.005` **exactly**. The values were written for a **dissolution** rate and duplicate `q2_k_classical`; re-pointing at `k_base` would inject values ~10⁶ too small. **NONE of the three was "fixed" — all three ROUTED**, because each fix is either another PO's live file or a physics decision. **What DID land, on PO-6a's own surface:** `INERT_DIMENSIONS`, a **machine-readable** registry in `quantum_dimensions.py` carrying the measured reason for each of the nine (a comment can be skimmed past; a registry can be asserted against), plus `assert_no_inert()`; `sweep_runner` now **warns before the run** and stamps `inert`/`inert_reason` into the **results JSON**, so a saved results file outlives the console warning and cannot be misread later. It **warns rather than dropping** inert dimensions — silently removing them would hide the defect instead of surfacing it. **THE `hasattr` DEFECT CLASS** (MO asked for the class, not the instance): 46 `hasattr`-guarded assignment blocks with no `else`, but **most are legitimate** — optional-subsystem reads and the lazy-init idiom. The defect is the narrow subset guarding **application of an external input**, where a False guard silently discards it: `sweep_runner.py:92` and `exp_sensitivity_analysis.py:176-179` (which gropes through three candidate attribute names and emits no spec if none match). **Class: guard-around-input-application — these must fail loudly, not skip.** Same mechanism as `Model6Parameters` having no `cascade` attribute (B2-1). | [GROUNDED, measured; routed not fixed] | commit `427b47c`; `requests/po5-selectivity/po1-6a-001.md` |
| PO2-6 | 2026-07-18 | **THE DEBIT RULE IS SETTLED BY LITERATURE, AND IT OVERTURNS MY OWN PRE-REGISTERED CHOICE — structural-first, not metabolic-first. Plus a one-way-valve problem it exposes in the 90/10 split.** Sarah's instruction was *"go back to the literature and physics and it will give us the best answer"*; it did, and it went against me. **The physics:** F1F0-ATP synthase phosphorylates ADP using **inorganic phosphate** imported by the mitochondrial phosphate carrier **PiC/SLC25A3** (proton-coupled symport; its loss abolishes oxidative ATP synthesis and causes mitochondrial cardiomyopathy). **The substrate is FREE inorganic Pi; protein-bound phosphate is not a substrate.** Neuronal cytosolic free Pi is low-millimolar and *rises* by millimolar amounts within seconds of stimulation (Rosen et al., PNAS 2026) — activity **liberates** free Pi rather than sequestering it. **Mapped to the model:** `phosphate_structural` IS the free inorganic pool (*"Only 'free' inorganic pool forms Posners"*, `add_phosphate_from_atp`); `phosphate_metabolic` is the protein-bound one. **So the grounded debit is STRUCTURAL, and ATP resynthesis therefore COMPETES WITH POSNER FORMATION FOR THE SAME FREE Pi POOL** — a depletion feedback arising from stoichiometry rather than installed, which is what §7 LOCKED requires of the SOC engine. **MEASURED, three arms, identical seed/config:** `metabolic_first` dP +9.74e-15 rel, struct depleted **0.143%**, maxS 1.8341, dimer 3.7248e-03 · `proportional` +1.04e-14, **0.259%**, 1.8341, 3.8334e-03 · **`structural_first` +9.54e-15, 0.264%, 1.8341, 3.9747e-03**. **Conservation invariant across all three** (all ~1e-14 vs ε=1e-12) ⇒ **acceptance item 1 untouched by the reversal**; `max S` identical across all three, independently re-confirming PO2-4's calcium-control finding. **Stated against my own interest:** structural-first depletes ~1.8× faster, the direction that would flatter the SOC story, and it **does not rescue it** — 0.264%/5 s is still nowhere near limiting and the Pi limit still never binds. **PO2-3/PO2-4's conclusion stands unchanged: conservation MET, self-limiting UNEXERCISED.** Registered as **PREREG AMENDMENT A2.4**, disclosed as a reversal of A2.3 rather than edited over; **proportional is revealed as a near-equivalent of structural-first** (structural is ~500× metabolic, so it absorbs almost any proportional debit), making **metabolic-first — the choice I registered — the outlier and the least physical of the three.** **NEW PROBLEM THIS EXPOSES, and it is not mine to fix:** hydrolysis credits 90% of released Pi to `metabolic` while grounded synthesis draws 100% from `structural`, so **`metabolic` becomes a one-way sink** — measured accumulating to 3.02e-2 while structural drains 0.264% in 5 s, extrapolating to **full structural depletion in ~32 min of simulated time.** A long SOC run would therefore shut formation off entirely via an accounting asymmetry rather than via physics. **The 90/10 split itself now needs review** — and note Rosen 2026 argues activity *raises* free cytosolic Pi, which is the opposite of routing 90% of it to a protein-bound sink. **Escalated, not touched: the split is a stated modelling choice and changing it is a physics call.** | [GROUNDED, literature + measured; reverses my own registration] | A2.4; `atp_system.py` `consume_for_atp_synthesis` |
| PO2-9 | 2026-07-18 | **PO2-6's ONE-WAY VALVE IS CLOSED AT THE GROUNDED VALUE — measured as a TREND, not extrapolated, and without spending the heavy slot.** Discharges MO ruling 021 item 2, whose bar was *"an extrapolation is not a measurement — run it long enough to see the pool actually bind, **or state that you did not**."* **Both halves answered.** **THE MEASUREMENT:** linear regression on the persisted 20 s free-pool trajectory (8 points/arm, matched seed), asking whether a drain **trend** exists at all — a question that does not require running to depletion. `frac=0.02` (pre-A2.5): slope **−4.851849e-03 pool-units/s** (−0.04852 %/s), **R² = 0.999167**, **t = −84.85**, **monotonic TRUE** (all 7 successive differences negative), time-to-zero **34.4 min** — *which reproduces PO2-6's earlier ~32 min extrapolation to within 7%, so the extrapolation was sound and is now superseded by the fit.* `frac=1.0` (grounded): slope **+8.048627e-05/s** — **positive**, i.e. very slightly rising — **R² = 0.083528**, **t = +0.74**, **monotonic FALSE**. **VERDICT: the depletion does NOT survive grounding.** Magnitude falls **60×**, the sign flips, R² collapses 0.999 → 0.084, and **t = 0.74 is not significant** (n=8, df=6; ~2.45 needed at p<0.05) — the correct statement is *no drain distinguishable from noise over this window*, not *the slope is zero*. **Loss of monotonicity is the specific discriminator**, because a one-way valve predicts monotonicity and nothing else here does. **WHAT I DID NOT DO, per the ruling's own alternative:** I did **not** run the pool to actual binding. At `0.02` that is 34.4 min simulated = **~413,000 steps at dt=0.005, ~100× the 20 s run, order 10–17 h single-core** — not worth an exclusive slot to confirm a drain already fitted at R²=0.999; and at the grounded value **there is nothing to run to.** **LIMIT, stated:** this bounds a 20 s window only; a slow nonlinearity on longer horizons would not appear here and **I am not claiming the pool can never bind.** **Compute cost of this row: zero** — arithmetic on data already persisted from the A2.5 run. | [GROUNDED, measured; trend statistics on persisted data] | ruling 021; `requests/model6-mo/po2-003-ruling021-complete.md` |
| PO2-8 | 2026-07-18 | **STEP E's SECOND CLAUSE DISCHARGED — the "2% ATP replenish" is grounded to 100%, and the one-way valve is closed. Sarah's authority to let literature decide.** The may30 pin's Step E reads *"return Pi from particle dissolution, **ground the 2% ATP replenish**"*; the 2% is `metabolic_to_structural_fraction`, and it was the unaddressed half of PO-2's charter. **TWO CODE-LEVEL CORRECTIONS FIRST.** (a) **The live value was 0.02, not the 0.10 everything documented** — `model6_parameters.py:199` sets `0.02` while `atp_system.py:419`'s `getattr(..., 0.10)` default is **never reached**, so `add_phosphate_from_atp`'s docstring (*"MOST goes to metabolic pool (90%) … SMALL FRACTION enters structural pool (10%)"*) described a split **the model does not use**. Third instance of prose contradicting code on this surface. (b) **The fraction does NOT set the pool size** — `phosphate_structural` is initialised to `params.phosphate_total` = 1 mM regardless (`atp_system.py:352`); the fraction routes only *newly hydrolysed* Pi (~0.1% of the pool per 5 s). **I had registered a prediction that assumed otherwise and struck it BEFORE running**, which is the only reason it did not become a false result. **THE GROUNDING.** ATP hydrolysis (ATP + H₂O → ADP + Pi) releases **free inorganic** phosphate; protein-bound phosphate is made by **kinase phosphotransfer**, a different reaction `update_hydrolysis` does not compute. Free cytosolic Pi is **0.29 mM at rest → 2.3 mM near maximal demand** (myocyte), and neuronal cytosolic Pi rises by **millimolar amounts within seconds** of stimulation (Rosen et al., PNAS 2026) — **activity LIBERATES free Pi.** Grounded routing ≈ **100% free**. **THE OLD JUSTIFICATION TESTED AND FALSIFIED:** the cited basis was *"Cells actively prevent Ca-PO₄ precipitation"*; against the model's own gate, resting Ca (100 nM) gives **S = 0.0060 with the ENTIRE pool free — 170× below threshold.** **Calcium prevents precipitation; the split was not doing that job.** What it was doing is `quantum-system-canonical` §6's named hazard — *"a calibration that silently does a missing mechanism's job is a hidden drift."* **MEASURED, 20 s, matched seed, 0.02 vs 1.0:** free pool **9.9736 → 9.9008 MONOTONIC DRAIN** (−0.99%/20 s, ~34 min to full depletion) vs **oscillating 10.0003–10.0058 about baseline, BALANCED**; `phosphate_metabolic` accumulating vs **exactly 0**; dimer 5.2380e-03 vs 5.3587e-03 (**+2.3%, small — as the CORRECTED prediction said, and opposite in sign to the struck one**); conservation **+3.2e-14 vs −2.0e-16**, both under ε, the grounded arm conserving *better*. **So PO2-6's one-way valve is CLOSED by grounding, not by patching.** Probe re-verified after the change: **exit 0, CONSERVED, all three controls still firing, 0 clamp activations.** **REGISTERED RISK NOT TRIGGERED at 20 s:** no runaway (`experiment-design-patterns:122`); **explicitly NOT cleared for long runs**, and A2.5 registered in advance that a runaway would be reported as a finding, never damped by re-tuning the fraction (§7 LOCKED). **LIMIT:** real cells do buffer phosphate (acidocalcisomes, protein binding), but that is a separate **dynamic** process, not a routing split applied at the instant of release; if buffering is wanted it belongs as its own rate. Consequence: **`phosphate_metabolic` is now vestigial (stays 0)** unless such a rate is added — flagged, not deleted. | [GROUNDED, literature + measured; prediction corrected pre-run] | A2.5; `model6_parameters.py` `metabolic_to_structural_fraction`; Rosen 2026 |
| PO2-7 | 2026-07-18 | **J-COUPLING/PHOSPHATE: CLOSED on documented physics, and the model's formulation is RIGHT — only its docstring was wrong.** Sarah: *"we know very well how j coupling and phosphate work. it is well documented."* Correct, and it resolves the question the MO had escalated and then withdrawn — but for a reason worth stating explicitly rather than resting on the withdrawal. **J-coupling is the INDIRECT SPIN-SPIN COUPLING transmitted through chemical bonds — it is a property of the MOLECULE, not of the solution.** In ATP the α/β/γ phosphates are mutually J-coupled through the P–O–P bridges (~20 Hz, Cohn & Hughes 1962 — **the citation the model already carries** at `J_PP_atp`); a free orthophosphate ion has no intramolecular ³¹P partner and therefore no such coupling (the model's `J_PO_free` = 0.2 Hz residual). **Ambient free-phosphate CONCENTRATION cannot set an intramolecular coupling constant.** Fisher 2015's pathway is consistent: entanglement is created at **pyrophosphate hydrolysis** (PPi → 2Pi) and is then protected by **incorporation into the Ca-phosphate cluster**, not by the ambient pool — *"in bulk water the survival time … is too short due to proton attack; if each entangled phosphate enters a Posner molecule, the entanglement can be preserved."* **Therefore `calculate_j_coupling` reading the ATP-bound fraction and NOT the ambient phosphate field is correct physics, not an omission.** The `phosphate` parameter it accepted and never read was a **docstring error only** — removed, with the reason recorded at the definition (`450a8cf`). **This closes queue Q3 on documented physics rather than on an MO withdrawal**, which matters because the MO was operating as a deliberate antagonist. **Residual gap, flagged not fixed:** Fisher locates the *protection* in cluster incorporation, and the model's J-coupling has no dimer/cluster term at all — so the quantity it computes is the birth-pathway proxy, not the protection mechanism. That is a real modelling gap, larger than PO-2's surface, and it is **not** the same claim as "J should read ambient phosphate" (it should not). | [GROUNDED, literature; code SHOWN] | Cohn & Hughes 1962 via `J_PP_atp`; Fisher 2015; commit `450a8cf` |
| PO2-5 | 2026-07-18 | **THE ATP-SYNTHESIS DEBIT RULE IS A MECHANISM CHOICE WITH A CHEMICAL CONSEQUENCE — recorded alongside the conservation result, per MO ruling 003.** Carried here rather than left in coordination, because the substance home is the log. **Pre-registered choice: METABOLIC-FIRST**, stated as a modelling decision with its reason — `add_phosphate_from_atp` sends 90% of hydrolysis-released Pi to the metabolic pool, described in its own docstring as *"protein binding, rapid cycling"*, which is physically the pool mitochondrial resynthesis draws from; the structural pool is the Posner-available one (`atp_system.py:453-457`, *"Metabolic pool doesn't participate in Posner chemistry"*). It is also the **conservative** choice for this PO's own claim: sparing the chemically active pool makes the SOC depletion feedback **weaker** and the claim **harder**. **CONSERVATION IS INVARIANT to the choice** — `max |dP|/P = 1.157e-14` across 6 runs, both modes, against ε=1e-12 — and structurally so, since the ledger sums both pools and the total debited is mode-independent. **So acceptance item 1 does not depend on this.** **What DOES depend on it, measured, same seed and configuration:** `metabolic_first structural = 9.985709043` vs `proportional structural = 9.974125559` — **0.116% difference on the chemically active pool.** Deterministic bookkeeping, not a statistical claim: the structural pool is ~500× the metabolic, so a proportional debit takes ~99.8% of itself from the chemically active pool while metabolic-first spares it. **NOT CLAIMED:** standing dimer ran ~2.3% higher under proportional with the sign positive in 3/3 seeds — **that is not an effect** (sign test p = 0.25, the smallest that design can return; between-seed spread 9.4e-05 vs effect 8.7e-05), and D18 established this regime is near-critical and bistable, exactly where n=3 misleads. Resolving it needs ~20 seeds and its own compute slot. **This row corrects a superseded MO ruling:** ruling 002 §1 dissolved the question on the grounds that the debit *"cannot affect the chemistry, only the ledger"*; **ruling 003 WITHDREW that conclusion** (MO defect #16 — *"verifying a premise is not verifying a conclusion"*) after this measurement. **The choice is ESCALATED TO SARAH**, MO's non-binding read being metabolic-first; a ruling the other way is a one-line change in `consume_for_atp_synthesis` and leaves item 1 untouched. | [GROUNDED, measured; downstream explicitly NOT claimed] | ruling 003; commits `11aec6f`, `450a8cf` |
| PO2-4 | 2026-07-18 | **SHARPENS PO2-3 — the SOC gate IS engaged, but it is CALCIUM-controlled, not phosphate-limited, at this drive. PO2-3 measured the coarser proxy.** Prompted by `quantum-system-canonical` §88, which I read rather than taking the MO's paraphrase: *"**PO₄³⁻ is the genuinely limiting species** (scarce, nM), re-supplied by acid-base speciation from the ~0.8 mM HPO₄²⁻ pool. Conserving a finite phosphate budget (~1 mM total = free + dimer-bound) is what lets the formation–dissolution cycle **self-limit (SOC)**."* **PO2-3 measured depletion of the total structural pool. That is not the SOC-relevant quantity** — the gate is `S = (Ca³·[PO₄³⁻]²/Ksp)^0.2` (`ca_triphosphate_complex.py:398-400`), so what matters is whether depletion moves **S**. **MEASURED, live, same 5 s configuration:** `max S = 1.8341`; **S>1 somewhere on 516/1000 steps (51.6%)**; but only **0.66% of the grid at peak, 0.065% mean** — nucleation fires in tiny nanodomain hotspots, which is the physically correct picture, not a failure. **So the gate is NOT dead and formation is NOT starved** — a reading PO2-3's pool-fraction framing could have been mistaken for. **THE DECISIVE ARITHMETIC:** `S ∝ P^0.4` and `S ∝ Ca^0.6`. The measured 0.1429% pool depletion moves S by **0.057% — negligible**, while calcium swings orders of magnitude per burst. **To halve S by phosphate alone the pool must fall 82.3%, which at the measured 0.0286%/s takes ~48 min of simulated time.** **Therefore: what turns the gate off in this regime is the calcium transient ending, not phosphate running out.** The depletion feedback the may30 pin calls *"the engine of the whole loop"* is built and correct (`11aec6f`) and is **not the controlling term at this drive** — consistent with the Pi-availability limit never binding. **This does not falsify SOC; it locates what a real test needs.** **Direct input to PO-6:** a drive×damping sweep must reach a regime where phosphate depletion competes with calcium control of `S`, or it will measure calcium dynamics and report them as self-organization. On this measurement the shipped configuration is not in that regime, and **run length and drive amplitude are therefore first-class sweep dimensions, not defaults.** | [GROUNDED, measured; arithmetic SHOWN] | ruling 002 §4; canonical §88; `ca_triphosphate_complex.py:398-400` |
| PO2-3 | 2026-07-18 | **IS SOC SELF-LIMITING NOW TESTABLE? Numerically YES by 11 orders; but the tested regime is ~350× too weak to reach it — and that is PO-6's most useful input.** Answering MO ruling 002 §4 (*"state whether conservation now holds well enough for that self-limiting behaviour to be testable — not merely that a ledger balances"*), which is the right question and is not the same question as item 1. **Numerically:** conservation error is `9.745e-15` relative against a structural-pool depletion signal of `1.429e-03` relative — **signal/noise 1.47e11**, so depletion is measurable far above the ledger's floor. Self-limiting is no longer masked by a loop that creates mass. **Physically, it was nowhere near reached:** the structural pool ran `10.0000 → 9.985709` over 5 s = **0.1429% depleted, 0.0286%/s**, extrapolating to **~29 min of simulated time for 50% depletion at this drive**. Against the program's ~60 s trials that is ~1.7% — **the pool never becomes limiting, and the Pi-availability limit added in `11aec6f` never bound once** (ATP recovered bit-identical before/after the fix, `1.098435256113e-02`). **So: conservation MET; self-limiting UNEXERCISED, not falsified.** The may30 pin makes the finite pool *"the engine of the whole loop"*; this row says the engine is now correctly built and has not yet been run at a load where it would turn over. **Consequence for the PO-2 → PO-6 edge (the MO's call, not this PO's):** the sweep would no longer measure a loop that creates mass — the hard block's stated reason is discharged — but it would test a loop that **can** self-limit with no evidence that it **does**. PO-6 should treat drive amplitude and run length as first-class sweep dimensions rather than assuming the shipped trial length reaches the depletion regime; on this measurement it does not, by ~350×. | [GROUNDED, measured; arithmetic SHOWN] | commit `11aec6f`; ruling 002 §4 |
| PO2-1 | 2026-07-18 | **THE PHOSPHATE LOOP NOW CONSERVES — and D14's "SOC loop already closed in live code" was a measurement of the half that never leaked.** Commits: prereg `a9f0767`, probe committed FAILING `305e096`, defect 1 `837a511`, defect 2 `11aec6f`. **THE LEAK, measured exactly:** `dP = +1.098435e-02` against `hydrolysis.total_recovered = 1.098435e-02` — the pre-registered prediction `dP == total_recovered` reproduced to **twelve significant figures**, residual 3.4e-13. Phosphate was created from nothing every step, in the exact amount of ATP regenerated: `update_recovery` (`atp_system.py:163`) credited ATP and debited ADP (`:169-171`) while debiting **no Pi**, though ATP = ADP + Pi. Relative drift 3.138e-04 = **~3e8 × the registered ε of 1e-12**. **AFTER: `dP = +3.410605e-13` (9.745e-15 relative), VERDICT CONSERVED — leak removed by 3.221e10.** **THE FIX IS STOICHIOMETRY, NOT A CORRECTION TERM** (§7 LOCKED, and the temptation this PO's kickoff named explicitly): synthesis is limited by available Pi and that Pi is debited (`consume_for_atp_synthesis`, metabolic-first per queue Q2); a second stoichiometric limit was also required because ADP previously capped only its own decrement while ATP was credited the full delta, so ATP could be synthesised with no ADP to make it from. No constant introduced, fitted, or moved. **THE CHEAT-CHECK — conservation was NOT bought by damping the leaky term:** ATP recovered is **bit-identical before and after** (`1.098435256113e-02` both), so ATP dynamics are unchanged and only the bookkeeping moved. **AND THAT IS THE LIMIT:** because recovery is unchanged, the Pi-availability limit **never bound** (P≈35 vs 1.1e-2 recovered) — so **conservation is demonstrated; DEPLETION FEEDBACK IS NOT.** Conservation is *necessary, not sufficient* for SOC; whether the loop self-organises is PO-6's drive×damping sweep and this PO does not pre-empt it. **RECONCILIATION WITH D8/D14, registered as a prediction BEFORE running and confirmed:** control C2 suppresses ATP recovery and the same ledger returns `dP = 3.8e-13` (1.1e-14 relative) — **strip the ATP arm and the loop conserves to machine precision.** `grep -n "ATP\|hydrolys\|recovery" sweep/phosphate_conservation_probe.py` → **ZERO hits**: the A3 probe has no ATP arm at all, and D14's own words are *"phosphate feedback **mimicking** model6_core"*. **D8 and D14 are correct measurements whose SCOPE, not arithmetic, was over-read** — half a loop recorded as a property of the whole one. D14's "no B3 edit needed" is **superseded**. **CONTROLS:** C1 injected a known 1.0 leak and the ledger detected 1.000000 (unfixed: 1.000127) — the detector is **not blind**; C2 as above; C3 clamp detector 0 activations in every arm, so the `np.maximum` floor at `model6_core.py:451`/`:757` contributes nothing to this drift. **TOLERANCE ε = 1e-12 relative**, justified against float64 accumulation *and* empirically validated by C2's observed 1.1e-14 noise floor (~90× headroom), with the leak 3e8 above it — **any ε in [1e-13, 1e-5] returns the same verdict**, registered in advance so it cannot be read as chosen to reach the outcome. **AMENDMENT A2.2, self-disclosed:** the C1 gate was defective (missing `abs()`, and a `predicted_dP` assuming recovery leaks); on fixed code it slipped under by sign accident, and *with* `abs()` would have returned a **FALSE** INVALID. Replaced with a leak-state-independent criterion `\|(dP_C1 − dP_main) − injected\|/P ≤ 1e-4`; **both** runs re-scored under it (unfixed 3.6e-06 PASS, fixed ~1e-14 PASS), corrected before either result was relied on. | [GROUNDED, measured; controls fired; shown FAILING first] | commits `a9f0767`, `305e096`, `837a511`, `11aec6f`; `docs/PREREG_PO2_PHOSPHATE.md` |
| PO2-2 | 2026-07-18 | **`calculate_j_coupling` NEVER READS ITS `phosphate` ARGUMENT — so PO-2's acceptance item 2 cannot be met, and the dispatch's stated mechanism is false.** AST-verified over `atp_system.py:263-306` (not read off the body, and emphatically not off the docstring): `signature args ['self','atp','phosphate','activity']`; `atp` READ, `activity` READ, **`phosphate` *** NEVER READ ***`. The body is `frac_atp_bound = atp/(atp+K_bind)` times an activity multiplier. Its own docstring at `:277` declares *"phosphate: Total phosphate field (M)"* — **prose asserting a dependency the body does not have**, this program's signature defect class. **CONSEQUENCE:** `MO_MODEL6.md` §3 PO-2 and PO-2's kickoff both state the defect as *"J-coupling reads a phosphate field that ignores dimer consumption."* **It reads NO phosphate field.** Fixing the stale `phosphate_total` therefore **cannot** make J-coupling track consumption, and the measurement confirms it: `corr(J, cumulative PO₄ consumed) = −0.069` with `J std = 9.4e-02` **both before and after** the fix — the field varies, just never with phosphate. That non-change was **registered as a prediction before fixing**. **AND IT DOWNGRADES DEFECT 1'S SEVERITY:** repo-wide grep shows the instance field `phosphate_total`'s **only** consumer is that dead argument, so the stale value was **INERT — a real correctness bug with no live consumer.** Fixed anyway (`837a511`) as a **derived property**, so staleness is structurally impossible rather than patched at the two known sites (`model6_core.py:450-452` **and `:756-757`** — the dispatch named one; assignment now raises `AttributeError`). Verified non-tautologically: a direct structural decrement of 1e-5 over 10 000 points moves the total by exactly −1.0e-01. **NOT claimed as fixing a live defect.** **Acceptance item 2 is reported NOT MET rather than substituting a weaker demonstration that would pass.** Wiring `phosphate` into the J-coupling physics would change the quantum-protection mechanism itself (Fisher 2015 `J_PP_atp` 20 Hz vs `J_PO_free` 0.2 Hz) — new physics, **escalated to Sarah, not decided** (`coordination/queue/po2-phosphate.md` Q3). **Second instance of declared-but-unread quantum coupling in this program**, after D21(1)'s inert `quantum_field_kT` *("accepted at three call sites, read in none")* — the pattern is worth a sweep beyond this surface. | [GROUNDED, AST-verified + measured] | commit `837a511`; queue Q3 |
| SWEEP-1 | 2026-07-18 | **THE SWEEP HARNESS IS LYING — 9 of 19 read-traceable dimensions are INERT, including TWO marked `importance="critical"`.** PO-6a Unit 1, `sweep/dimension_consumer_audit.py` (`9b4819f`, `dbe9548`). **Measured, not grepped:** the probe patches `__getattribute__` on the parameter dataclasses and on `ThetaBurstScenario`, drives the real model (construct -> network steps -> backbone update, plus a full small scenario run), and records which attributes are **actually read while physics runs**. `reads == 0` is definitive INERT — nothing looked at the value. Read-tracing is the right instrument here precisely because it is **asymmetric: it can only ever over-report LIVE**, so a "never read" verdict is sound. **Three controls, all passing** — (A) tracer sees reads, (B) tracer sees *absence* of reads, (C) calibration against B2 ground truth: `omega_0` must be LIVE (21 reads), `D_modes` must be INERT (0 reads). Without C this is an assertion; with C it is an instrument. **THE INERT NINE:** `q1_d_modes` (:61), `q1_phi_dissipation` (:63), `q1_chi_redistribution` (:65), `q1_kT_per_modulation` (:67), **`q2_t2_p31` (:71) [CRITICAL]**, `q2_j_coupling_hz` (:73), `q2_k_agg_baseline` (:92-93), **`stim_ca_amplitude` (:145) [CRITICAL]**, `stim_burst_duration_ms` (:145). **Every one returns a FLAT RESPONSE, and a flat response over a swept parameter reads as "this parameter does not matter" — a physical null. It is not. It is a wiring gap wearing the costume of a result.** This is the program's characteristic defect (prose asserting mechanisms the code does not implement) promoted into the measurement apparatus itself, where it is far more dangerous: a lying docstring is one wrong sentence; a lying sweep manufactures wrong findings at scale and every one of them looks like data. **THREE DISTINCT MECHANISMS — they need different fixes:** **(1) NO CONSUMER** (the six params-level). `q2_t2_p31` is the most serious: declared `importance="critical"`, *"controls eligibility trace window"* — the ~100-200 s window the whole thesis rests on — and its **only** reader in the tree is `singlet_dynamics.py:122`, an **ORPHAN module never imported or instantiated**. The LIVE coherence path instead reads `T2_single_P31` (`quantum_coherence.py:59`), which **no dimension sweeps**. The sweep varies a dead field while the live field is never varied. Confirmed NOT a short-run artifact: at 150 steps with **1075 dimers present**, reads are still zero. `J_intrinsic_dimer` is starker — **one write (`sweep_runner.py:73`), zero reads anywhere in the codebase**: a parameter that exists only to be swept. **(2) SILENT GUARD** — `q2_k_agg_baseline` is guarded by `hasattr(dimerization,'k_agg')`, which is **False** (the real attribute is `k_base`), so the write never executes; the code carries its own unactioned TODO at `:91` (*"k_agg may be on dimerization or a separate attribute — adjust if needed"*). **(3) HARDCODED OVERRIDE** — `_run_epoch`'s own docstring states *"Calcium enters via voltage-gated channel physics, **not direct injection**"* while `stim_ca_amplitude`'s dimension text claims *"peak calcium per burst **(direct injection)**"*; and `burst_duration_ms` is overridden by a hardcoded `spikes_per_burst x spike_period = 40 ms`, so the swept 20/50/100/200 ms values are never consulted. **PRIOR KNOWLEDGE NEVER READ AS A DEFECT:** `model6-architecture:48` already records *"T2_p31 sweeps `T_singlet_dimer`, not `T2_single_P31`"* — filed among neutral "attribute path corrections". The fact was on the books; nobody read it as *the sweep's most critical Q2 dimension is wired to a dead field*. **CONSEQUENCE:** no sweep result over any of these nine can be interpreted, and any past reading of a flat response as physics must be re-read. Network dimensions are applied structurally and are excluded from the denominator rather than silently counted as passing. **Limits:** INERT holds for the stated driving conditions (a consumer on a branch this run never reached would look identical, so each is INERT-under-stated-conditions); REACHED means consumption, not demonstrated effect. | [GROUNDED, measured; controls verified] | `sweep/dimension_consumer_audit.py`; `9b4819f`, `dbe9548` |
| GAP-1 | 2026-07-18 | **THE ANALYTICAL GAP NOW ADVANCES THE PLASTICITY CLOCK — and the defect was a 1 ms tick, not a freeze.** `analytical_gap` jumped `network.time` by the full gap then ran one `network.step(0.001, ...)`, so actin / `E_invasion` / CaMKII / DDSC advanced **1 ms per gap regardless of gap length**; its docstring listed 5 advanced and 6 excluded items and left all four in **neither** column. **Supersedes the "frozen / 100% retention" reading** in PO-3's F-2 and the MO's addendum: measured retention was **0.999994**, not 1.0 — *worse*, because it reads as an even cleaner ratchet. This row also **corrects D20's own framing**: D20 had the 1 ms fact and the right discriminator, but the consequence — that every multi-trial result ran through a stopped plasticity clock — was not drawn. **FIXED:** consolidated to ONE definition (the `run_spatial_discovery.py` copy was byte-identical bar a Unicode arrow; three consumers now import it), plasticity/CaMKII/DDSC integrate at `dt_sub`, commitment still evaluated only on the real path (`model6_core.py:671-685`) so `model6-commitment-pathway` LOCKED holds, and `run_place_field_learning.py`'s manual gap-advance workaround removed (it would have **double-advanced**, 40 s per 20 s gap, and took the whole 20 s as one Euler step). **Jain 2024's 30–40 s DDSC window is now reachable**; it previously fell entirely inside the skipped interval, so delayed commitment could not resolve in any gap-based experiment. **VERIFIED:** clock delta `spine_plasticity.time` +20.0010 s vs `network.time` +20.0010 s (was 0.0010); retention matches the closed form at **8 out-of-sample points** across a 6× range of gap duration, both confinement arms, max error 0.0039 against a 0.02 tolerance; first-order Euler convergence confirmed (ratio 2.04–2.08). **`K_CLASSICAL = 0.05` — the RETIRED rate — is live inside this function and was deliberately NOT touched (MO-held).** | [GROUNDED, code SHOWN, MEASURED] | `docs/PREREG_PO4_GAP.md`; probes `sweep/gap_retention_probe.py`, `gap_dt_convergence.py` |
| GAP-2 | 2026-07-18 | **COMMITTED vs UNCOMMITTED SPINE VOLUME SEPARATE ACROSS AN HONEST GAP — the board's PO-4 acceptance, met, and the frozen clock is shown to have made it impossible.** At +300 s from an identical start (V=1.0000): **committed 1.9403 ± 0.0187 vs uncommitted 1.1639 ± 0.0228, ΔV = +0.7764**, against a pre-registered 4σ floor of 0.26. Seed-only null separates by −0.0146 (**53× smaller**); both positive controls fired (`_camkii_committed`, confinement 0.976); neither arm at the 3.80 actin-limited ceiling. **Frozen-clock control, measured against pre-fix code, not asserted: ΔV = +0.000299 — a factor of 2595.** Mechanism registered in advance so it could fail: commitment does not add actin, it **redirects** it (`spine_plasticity_module.py:389-390`) — unconfined enlargement extrudes to the shaft and is lost, confined enlargement is retained into `actin_stable` (τ=1000 s) — i.e. **"commitment buys durability, not amplitude"**, independently D19's result from a different probe. **`MO_MODEL6.md:140`'s "1.291 vs 2.389" is UNSOURCED** (grep → coordination prose only; MO ruling 003 confirmed this as the MO's defect and instructed against reconciling to it). This measurement recovers the **same ordering and regime** but not those magnitudes, and was scored on sign + floor, never against them. **LIMITS:** controlled initial condition — the live drive path is not exercised and **does not reach this regime** (a 12-cycle traversal leaves `E_invasion` at 0.0000, 10× below `invasion_threshold`, at ~127× slower than realtime); two synapses, one network. | [GROUNDED, MEASURED, limits stated] | `docs/PREREG_PO4_GAP.md` AMENDMENT D; probe `sweep/gap_separation_probe.py` |
| GAP-3 | 2026-07-18 | **`E_invasion` IS A TRANSIENT-POOL READOUT THAT COMMITMENT DEPLETES — and the 89% retention prediction is CONFINEMENT-CONDITIONAL, not a constant.** `E_invasion` is computed from `actin_enlargement` alone (`spine_plasticity_module.py:412`), and `:389-390` gates extrusion on `(1−conf)` while routing the remainder to stabilization. So a **committed** spine drains `E_invasion` at τ_eff ≈ **50.9 s** against the uncommitted **180 s** — **3.5× faster** — and `exp(−gap/tau_extrude) ⇒ 0.8948` is the **uncommitted branch only**. Measured in the isolated module at +300 s: `E_invasion` **0.0313 committed vs 0.8222 uncommitted — a 26× inversion**, while spine volume moves the *opposite* way (3.7031 vs 3.0432). Routed to PO-3 **pre-run** (`requests/po3-einvasion/po4-conf-001.md`) and **adopted as an MO ruling** on PO-3's surface; PO-4 did not edit that module. **The MO escalated the larger reading to Sarah as a physics call** (since `r ∝ E_invasion × ca_open`, the condensation pump is driven by the *uncommitted* transient pool, so a synapse that commits loses pump drive) — **outside PO-4's bar and not pursued here.** | [GROUNDED, code SHOWN, MEASURED] | `coordination/requests/po3-einvasion/po4-conf-001.md` |
| B2-4 | 2026-07-18 | **D, φ and χ do NO physics after B2 — at EITHER pump site — and the large-D validity of η is recorded as an UNVERIFIED limit.** Closes the pin's second B2 obligation ("D and φ: verify in B2 … confirm their remaining role/consistency rather than assume"), which the MO's acceptance bar had omitted (MO ruling 005 §2; MO records it as its own defect, not PO-1's). **Verified on executable code** with comments and docstrings stripped (`ast`/`tokenize`): per-synapse `D_modes` = declaration + 2 prints; `phi_dissipation` = declaration + `__post_init__` derivation + 2 prints; `chi_redistribution` = same. Backbone: **all three are declarations only, zero reads in `multi_synapse_network.py`.** The live chain consumes ω₀, Q and T alone — `P_c = n̄_s·ℏ·(2π·ω₀)²/Q → r = P_met/P_c → η = (r−1)/(r+1)`. They are retained rather than deleted (Zhang still describes above-threshold dynamics if a consumer is ever wired) but are now explicitly marked inert at both sites; **changing any of them moves no computed quantity.** **STATED LIMIT (not resolved):** `η = (r−1)/(r+1)` is the **large-D limit** of Wang/Wang 2022; the pin calls for D ≳ 200, per-synapse runs 20, backbone 50. Because D does not enter the formula this does not change the number computed — it bears on whether the large-D *form* is the right one to use at all. Finite-D corrections have not been derived here, so **adequacy at D = 20/50 is UNVERIFIED and now qualifies every η these sites report.** D was NOT raised to sit inside the limit: that would be moving a constant to reach an outcome (`MO_MODEL6` §7). Closing it needs the finite-D expansion from the source paper — a physics unit, not a code change. **TWO FALSE STATEMENTS CORRECTED, both authored by PO-1 in `c280e85`:** the claim that χ was "kept because the steady-state solution needs a nonlinear term" (B2 deleted the quadratic — there is no nonlinear term), and that D_modes/χ "survive as above-threshold slope parameters" (nothing consumes them). The characteristic defect of this program, caught twice inside PO-1's own diff. **HAZARD routed to PO-6:** `sweep_runner.py` **writes** `params.dendritic_backbone.D_modes` from the `q1_d_modes` dimension (`quantum_dimensions.py`), but nothing reads it — **a swept dimension with no consumer**, which will return a flat response readable as a physical null. | [GROUNDED, code SHOWN; limit stated] | commit `1f75582`; MO ruling 005 §2 |
| B2-1 | 2026-07-18 | **SUPERSEDES DISC-1 — the per-synapse pump site is RETIRED, and the two pumps now run one mode and one convention.** Landed `c280e85` (+ probe `fa12009`). Sarah's call, executed not relitigated: *do not fix the 2π error, retire the code that contains it.* **Gone from the live path** (grep-proven on executable code with comments, docstrings AND string literals stripped — 0 references): `kT_ref=22.1`, `r_at_E_ref=100e9`, `pump_exponent`, `E_ref_pump`, `_critical_threshold` (Zhang Eq. 4 `r_c`), both hand-rolled `hbar=1.0546e-34` copies, and `omega_max=160e9` (declared, never read). **Replaced by the backbone's own physics**, not a new invention: `P_c = n̄_s·ℏ·(2π·ω₀)²/Q`, `r = P_met/P_c`, `η = (r−1)/(r+1)` (Wang/Wang 2022, reference-free `n_ex = n̄_s`, β=1) — identical to `_update_backbone_field`, the one intended difference being NO aggregation per-synapse. ω₀ 40 GHz→8 MHz; φ 10 GHz→ω₀/Q=0.8 MHz; φ and χ now DERIVED in `__post_init__` so they cannot drift out of step with ω₀/Q; Zhang's BSA values deliberately NOT adopted (they belong to the retired mode family — adopting them re-imports the conflation wearing a citation). **THE MEASUREMENT** (`sweep/pump_mode_agreement_probe.py`, committed FAILING first at `fa12009`, then passing): A1 mode ratio 5000→1.000000; A2 convention ratio 6.301→1.000000; **both positive controls (C1 mode-conflation, C2 2π) still FIRE**, so the verdict can still distinguish its outcomes — the L·ETA-4 failure mode is excluded by construction, and the probe returns INVALID rather than PASS if a control goes silent. A2's reference is recomputed from CODATA, never by calling `bose_einstein_occupation`, so it cannot compare the shipped function to itself. **Independent corroborations, computed not copied:** `P_c = 21.514 fW` (pin: 21.5 fW); rest `r = 0.039` (pin: "rest 0.04, SUBcritical"); per-synapse `P_c` ≡ backbone `P_c` to machine precision; `n̄(8 MHz)=8.0742e5` (pin: 8.07e5). **LIMITS:** this proves the two sites AGREE; it does NOT prove 8 MHz is the right mode — that stays the May-30 bet (Q≳10, Pokorný slip-layer vs Foster/Baish). If the bet is wrong both sites are now wrong together, and the probe would still say PASS. **DISC-1's embargoed claim is not rehabilitated — it is dissolved:** "the per-synapse pump exceeds its Fröhlich threshold under MT invasion" was an arithmetic identity, and under the replacement there is no per-synapse threshold-crossing result to cite at all; crossing is a drive question with no reference scale to exceed. **Widens DISC-1 by one level:** `Model6Parameters` has NO `cascade` attribute (one comment hit only, `model6_parameters.py:784`), so `VibrationalCascadeModule.__init__` always fell through to `TubulinCascadeParameters()` defaults — **the ENTIRE per-synapse pump parameter set was unreachable by `sweep_runner`, not just `kT_ref`.** No sweep ever run in this program could have varied it. Still true after B2; wiring `params.cascade` is open, unowned. | [GROUNDED, measured; controls verified] | `docs/SUBSTRATE_AUDIT_JUL18.md` §C1; commits `fa12009`, `c280e85` |
| B2-2 | 2026-07-18 | **The retired per-synapse η was a constant dressed as a variable — and the `k_agg` delta is reported, not damped.** Measured old-vs-new at matched physical conditions (both modules loaded side by side; old from git). **OLD:** η = 0.2160 at 12.8 kT → 0.2211 at 28.6 kT — i.e. **essentially INVARIANT (±1%) across the entire drive range**, pinned by the Zhang steady-state source term. `above_threshold` flipped at exactly 22.1 kT because 22.1 was `kT_ref` itself. So the per-synapse condensate did no drive-dependent work: any downstream result attributing variation to it was reading a constant. **NEW:** η tracks drive properly — 0 (r<1) → 0.0952 (r=1.21) → 0.4086 (r=2.38) → 0.4775 (r=2.83). Forward enhancement max 1.78→4.40 (~2.5×). Per MO ruling this is reported, NOT damped: damping it would be tuning to protect a downstream result. Consequence to watch: higher `k_agg` ⇒ more dimers ⇒ the O(n²) entanglement tracker gets slower, and any standing result that depended on the flat η should be re-read. **Escalated to the MO, not decided here.** | [GROUNDED, measured] | this session |
| B2-3 | 2026-07-18 | **SKILL DRIFT — "a single synapse is subcritical by design" is CONTINGENT, not structural.** `model6-architecture` §Key Constraints states it as a design property ("basal not aggregated… Do not 'fix' a subcritical single synapse"), citing the June-7 figure *"P_met maxes ~17 fW < P_c = 21.5 fW, r peaks 0.803"*. That figure reproduces **exactly** at `E_invasion=0.495, ca_open=0.55` — which is what the actin envelope had climbed to by 30 s, i.e. **the drive ACHIEVED in a short run, not a ceiling.** At sustained full invasion the same arithmetic gives `P_met = 51.24 fW`, `r = 2.382`, comfortably above threshold with no aggregation at all. What the backbone's aggregation actually buys is **crossing at LOWER per-spine drive**, not crossing at all. The skill sentence should be corrected; "subcritical by design" must not be read as "cannot cross". Flagged, not edited — skills are not PO-1's surface. | [GROUNDED, arithmetic SHOWN] | this session |
| DISC-1 | 2026-07-18 | **INTERIM DISCLOSURE — the per-synapse threshold result is CALIBRATED and NOT EVIDENTIAL.** Issued ahead of B2 rather than waiting for it, per Sarah's instruction ("right as interim disclosure, wrong as an endpoint"). `kT_ref = 22.1` is a **function-body literal** at `vibrational_cascade_module.py:246` — not a dataclass field, so invisible to `TubulinCascadeParameters` and to `sweep_runner`, i.e. **structurally unsweepable**. Together with `r_at_E_ref = 100.0e9` (`:115`, whose own comment reads *"Calibrated so that full MT invasion (22 kT field) produces r > r_c"*) and `r_c = (φ/(D+1))(1+φ/χ) ≈ 9.57e10` (`:212-214`), the MT+ condition gives **r/r_c ≈ 1.045 — an ARITHMETIC IDENTITY between two numbers chosen to produce it**, not a measurement. **There is no derivation to do:** `r_c` here is the classical critical pump, which →0 in large-D — an artificial reference scale, as the May-30 session established. Asking to justify `kT_ref` is asking to justify scaffolding; there is nothing underneath it. **Any claim of the form "the per-synapse pump exceeds its Fröhlich threshold under MT invasion" must NOT be cited as evidence** until B2 replaces this site with the `n_ex = n̄_s` treatment. The BACKBONE pump is unaffected and is clean (`P_c` from Bose-Einstein, `η = (r−1)/(r+1)`, correct `h·f` convention) — the two must not be tarred together. Superseded when B2 lands (MO PO-1). | **[SUPERSEDED 2026-07-18 by B2-1 — B2 has landed (`c280e85`); the embargoed claim is dissolved rather than released, and DISC-1's scope was one level too narrow. Row left intact per the append-only rule.]** [GROUNDED, code SHOWN] | `docs/SUBSTRATE_AUDIT_JUL18.md` §C1 |
| AUDIT-1 | 2026-07-18 | **SUBSTRATE AUDIT — full adversarial code audit, `docs/SUBSTRATE_AUDIT_JUL18.md`.** Four parallel read-only agents, `file:line` required for every claim, UNVERIFIED where code could not confirm. **Five headline findings:** (1) **factor-of-2pi error** on the per-synapse pump — `vibrational_cascade_module.py:315` uses `hbar*f` on a LINEAR frequency, n_bar inflated **6.28x**; the backbone pump is CORRECT (`h*f`, `model6_parameters.py:46`). (2) **The calibration fiction survives and is unsweepable** — `kT_ref = 22.1` is a function-body literal (`:246`), invisible to the params dataclass and to sweep_runner; with `r_at_E_ref = 100e9` it makes **r/r_c ~ 1.045 at MT+ an arithmetic identity, not a result**. (3) **Three docstrings assert mechanisms absent from the code** — a Hill function (`multi_synapse_network.py:1332-1334` vs `:1381-1392`), a 30% collapse (`:1423-1425`, `collapse_factor` never read), and "No fitted parameters!" (`:1238-1242`) beside two fitted parameters. (4) **Cited sources contradict their values** — phi/chi cite Zhang 2019 which gives 6 GHz / 0.07 GHz; code uses 10 GHz / 0.05 GHz. (5) **The two pump sites run different threshold physics** — backbone `n_ex = n_bar_s`, per-synapse still Zhang Eq. 4. **WHAT SURVIVES:** the entanglement/partition layer — Werner 0.5 is a THEOREM not a cutoff, eta is exactly `(r-1)/(r+1)` with no fitted curve, commitment is a real CaMKII integrator with a genuine DDSC delay. **Debt REGRESSED:** ~151 dead parameter fields (was ~120), six orphan modules, none removed. Also found: `phosphate_total` goes stale so J-coupling reads a field ignoring dimer consumption; ATP<->Pi is not mass-conserving; `step_with_coordination` and `run_place_field_learning` still form ZERO cross-synapse bonds (a gap in the same-day fix). | [GROUNDED, code SHOWN] | `docs/SUBSTRATE_AUDIT_JUL18.md` |
| D19 | 2026-07-18 | **D17's CROSS-TRIAL READING IS RETRACTED — the quantum measurement fires once per EXPERIMENT, not once per trial.** Observed (`sweep/loop_audit_2026_07_18/probe_latch2.py`, instrumented 3-trial run): `perform_quantum_measurement` called **1× in trial 0, 0× in trials 1–2**; `_network_measurement_performed` (`multi_synapse_network.py:1335-1337`) is a one-shot latch that `run_spatial_discovery.py:417-419` never resets. Spine volume nevertheless accumulates, by two non-quantum paths: **(A)** `_measurement_gate_opened` is written True at one site (`:1383`) and **never cleared anywhere, including `reset()`**, so later trials re-commit on stale-flag AND a classical CaMKII calcium integral — observed with `pqm_calls=0`; **(B)** commitment is not needed at all — a never-gated, never-committed synapse grew 1.065→1.430, because actin formation is calcium-only and `structural_drive` enters solely through `confinement`. **Commitment buys durability, not amplitude** (measured: drive 0→1 at fixed Ca *lowers* enlargement 1.447→1.099). Ratchet compounder: only *active* synapses are stepped, so silent ones never run their decay term. **SURVIVES:** all single-trial chemistry (emergent/bounded/rise-and-fall/localized) and D18. **RETRACTED:** "plasticity accumulates across traversals" as evidence of quantum-gated learning, and "formation→spine-growth→behavior closes". **Also invalid by construction:** the coordinated-vs-independent control — `_evaluate_independent_gate` shares the same latch (`:1476/:1478`), so whichever fires first locks out the other. | [GROUNDED, observed; reduced config] | E2·A–C |
| D20 | 2026-07-18 | **Durable-state audit — one working channel, and it does not self-maintain.** Spine volume is the only channel closing the loop (read back as the agent's weight, `run_spatial_discovery.py:372-375`); ceiling **3.80**, actin-limited at `spine_plasticity_module.py:332-333` (not the 3.9 clip at `:381` — the two ceilings differ). **It decays to below baseline by t=3000 s (τ≈1000–2000 s), falsifying `coherence-gated-learning` primitive 4** ("self-reinforcing maintenance that resists decay"). `analytical_gap` advances the plasticity clock by **1 ms per 30 s gap** (observed `network.time`=46.5 vs `spine_plasticity.time`=16.5–31.5), so inter-trial decay is skipped rather than resisted. **AMPAR is dead twice over** — the 1800 s onset needs 159–809 trials at the measured duty cycle (0.189 best / 0.037 mean) against a shipped `n_trials=5`, AND its whole chain is gated on `spine_calcium_feedback`, defaulted False. **Template feedback (`spine_volume>1.25 → n_templates`) DOES fire early (~8 s)** and is the most reachable second channel: mean rate effect only ×1.010–1.015, but it ~**doubles the template-bound fraction**, raising `T_eff` — a coherence-window lever, not a rate lever; net standing count unsigned because dissolution carries the same factor. Coupling weights never mutate; `apply_reward_correlated`/`sample_correlated_eligibilities` orphaned and would be near-no-ops if wired (they write the lever that does not set magnitude); `eligibility_trace.py` fully orphaned and superseded. | [GROUNDED, measured] | E2·D |
| D21 | 2026-07-18 | **CONSTRUCT-VALIDITY GAPS — declared ≠ implemented, five of them, blocking any load-bearing behavioural result** (`quantum-computation-and-attribution` §6.5). (1) **`quantum_field_kT` is INERT in spine plasticity** — measured bit-identical volume for kT ∈ {0,1,5,20,100}; accepted at three call sites, read in none; five `ActinParameters` barrier fields declared and never referenced; the module docstring describes a quantum barrier-modulation mechanism **that does not exist in the code**. (2) The same docstring names `molecular_memory` as the driver of storage; calcium is. (3) `spatial-discovery-experiment` skill claims AMPAR persists as structural state; **AMPAR never changes in any run**. (4) primitive 4 falsified (D20). (5) **No cross-synapse bonds form during trials at all** — `run_trial` omits `coupling_weights` (`run_spatial_discovery.py:446-449`) and `_update_entanglement` early-returns without them (`:276-279`); combined with `L·ETA-1` (eta=0 ⇒ k_cross=0) there are **two independent reasons the eligibility structure is absent during a learning trial**. | [GROUNDED, measured + code SHOWN] | E2·E |
| D18 | 2026-06-28 | **Near-critical variability CHARACTERIZED — dimer nucleation is an ALL-OR-NONE bistable switch** (D17's "criticality" confirmed mechanistically). New single-synapse probe (`sweep/criticality_variability_probe.py`), N≈120/condition, reseeded global RNG isolates the stochastic gate from the agent/structural/start-position confounds in the 5-trial run. Across the subthreshold drive band the peak-dimer distribution is **bimodal with a FORBIDDEN GAP**: replicates land at silent/fizzle (≤~8 dimers) or **full (~125–150)**, NEVER between — **0 of 480** in 11–120. ON-amplitude is **quantal** (~135, drive- AND duration-independent → a real attractor = the supersaturation runaway); drive/duration/input-noise tune only **P(catch)**, a sharp sigmoid. Critical point ≈ **−43 mV / ~570 µM** (just under the 716 µM gate; matches D11/D12 616 µM→S≈0.91). Susceptibility (Fano = var/mean) peaks ~120–148 at the midpoint. Mechanism: stochastic channel-opening **COINCIDENCE** (`analytical_calcium_system.py:132`) intermittently crosses S>1; once caught it runs to the attractor. Explains the D17 trial spread (9/1/31/23/22 = many such switches integrated over a 60 s traversal). **Controls PASS:** gap survives 0.25–2.0 s durations (not a window artifact) AND presynaptic-release stochasticity (not a constant-glutamate artifact). **Altitude:** a CLASSICAL stochastic nucleation criticality on the **(A)** floor — says nothing about **(B)**. Probe + scratch data uncommitted. | [GROUNDED probe + controls] | E1·L |
| D17 | 2026-06-28 | **FULL INTEGRATION VALIDATION — the grounded stack works end-to-end in the live network.** `run_spatial_discovery`, 5 trials, 20 synapses, B1+B2a+D16 live. Runs clean. Dimer formation **EMERGENT, BOUNDED, rise-and-fall** (end-of-trial totals 9/1/31/23/22; peak transient 318; no runaway → resolves the parked unbounded-accumulation problem). **LOCALIZED** (1/1/5/4/3 of 20 synapses). **Plasticity ACCUMULATES across traversals** (max spine vol 1.63→1.90→2.59→2.59→2.67) and the agent **FOUND the goal in trial 3** (t=62 s) — formation→spine-growth→behavior closes. Stochastic/near-critical (unseeded channel gating → trial-to-trial variability = predicted criticality). First run that is emergent + bounded + rise/fall. (Earlier expectation of ~0 dimers was WRONG — B2a's grounded amplitude lets stochastic coincident openings cross the gate at subthreshold V.) **Live edits: B1 committed; B2a+D16 uncommitted.** | [GROUNDED integration] | E1·K |
| D16 | 2026-06-28 | **Species blocker (D13) RESOLVED — Option B outcome.** Dropped the invalid bulk-Ca/P sigmoid in `update_dimerization`; `dimer_fraction = 1` (formation → dimer). Grounding chain: Ca/P invalid (skill §3) + aggregation-rate ungroundable so a split would be tuning (D15) + **coherence selects the dimer downstream** (dimer ~100s s, trimer sub-second; Agarwal). MODELED choice, flagged in code. Validated: integration loop now forms 49 µM dimer (matches A3's 47 µM), S pins at 1.0, P_struct stabilizes 0.81 mM → grounded Ca→gate→dimers→SOC, correct species. **Live edit, uncommitted.** (Dead `calculate_dimer_fraction` recompute at ~L425 now orphaned — harmless, clean later. Canonical-skill §2.2/§3 owes the "formation species-selection ungroundable; coherence is the selector" keystone.) | [GROUNDED; MODELED choice] | E1·K |
| D15 | 2026-06-28 | **Option-B research: species determinant is AGGREGATION EXTENT (kinetic), and it does NOT cleanly favor the dimer.** Lit (Garcia/Mancardi 2019; Posner & Betts; CaP nucleation MD): growth ion-complex → dimer (2 units, early/metastable) → Ca₉ Posner (3 units) → ACP ("glass of Posner clusters"); higher supersaturation + time + dehydration drive aggregation FORWARD → at high nanodomain Ca the principled determinant favors the TRIMER, opposite the model's goal. **Agarwal is SILENT on which forms** (pure coherence argument) — "dimer is the qubit" ≠ "dimer is what forms". Routes to dimer-dominance are all unestablished hypotheses: (a) kinetic trapping in the transient nanodomain (aggregation timescale not pinned in lit); (b) coherence-protected persistence (model already dissolves trimers 10×); (c) template size-stabilization. ⇒ **Whether dimers DOMINATE formation in vivo is an OPEN keystone the program had assumed.** Decision pending: A (assert dimer as Agarwal-grounded modeling choice, flagged) vs B-kinetic (model aggregation kinetics + nanodomain transient, let species emerge — risks showing trimer-dominance). | [GROUNDED lit; CONTESTED keystone] | E1·K |
| D14 | 2026-06-28 | **SOC loop already closed in live code (no B3 edit needed).** Integration test (live gated `update_dimerization` + phosphate feedback mimicking model6_core): S pins at 1.000, `phosphate_structural` stabilizes 1.0→0.81 mM. B1+B2a+existing consumption plumbing self-limit. | [GROUNDED probe] | E1·K |
| D13 | 2026-06-28 | **BLOCKER — grounded Ca routes formation to the INERT TRIMER.** Same test: 0.19 mM P consumed but only 2 nM dimer (≈31 µM trimer). At grounded Ca 823 µM / HPO₄ 0.585 mM → Ca/P=1.4 > sigmoid center 0.5 → `calculate_dimer_fraction` → 99.99% trimer. The latent Ca/P-sigmoid issue (dimer-chemistry skill §3) ACTIVATED by B2a. Bulk Ca/P can't physically select species (both Ca₆/Ca₉ are Ca/P=1.5 products). Grounded model yields NO qubits until species selection is fixed (skill: drive by aggregation/templating/supersaturation, not bulk Ca/P). **Decision needed.** | [GROUNDED probe; CONTESTED mechanism] | E1·K |
| D12 | 2026-06-28 | **B2a DONE — calcium amplitude grounded (live edit).** `analytical_calcium_system.py`: calibrated 0.5 µM/channel → Naraghi-Neher closed form `i/(z·F·4π·D_ca·r)`; **corrected `D_eff`→free `D_ca` in the 1/r prefactor** (real physics fix, not calibration); λ pump 190 nm → buffer 117 nm; read floor `dx`→5.5 nm (D11). Validated: λ=117 nm, single channel 97.5 µM @5.5 nm, 7-ch cluster 616 µM (~170–280× the old). **Emergent:** 616 µM → S≈0.91 (just sub-threshold) → dense open-channel clusters clear the gate, sparse don't; with NMDAR/VGCC gating, formation needs coincidence + clustering, nothing tuned. **Uncommitted.** Next: B2b PO₄³⁻ plumb (D7), B3 conservation. | [GROUNDED probe] | E1·J |
| D11 | 2026-06-28 | **Read distance grounded ≈ 5.5 nm (measured), NOT 1–2 nm.** FRET channel-mouth-to-tethered-sensor ≈ 55 Å (CaV2.2, ncomms1777); nanodomain coupling 5–50 nm; nanocolumns tens of nm. This is within Naraghi-Neher LBA validity (the formula is built for "[Ca] at the mouth", 5–50 nm) → no r→0 divergence; the sub-nm push is biologically wrong. Biology pins r near the model's 4 nm grid floor. Consequence: single channel ~100 µM (sub-threshold) → nucleation requires the CLUSTER SUM over OPEN channels → emergent clustering + coincidence (glu+depol) requirement, nothing tuned. **B2 floors the nearest-channel r at ~5.5 nm (5–20 nm uncertainty), reads the cluster-field, no sub-nm push.** | [GROUNDED lit] | E1·I |
| D10 | 2026-06-28 | **B1 DONE — gate wired (live edit, option b).** Supersaturation gate inserted in `update_dimerization` (`ca_triphosphate_complex.py:387-402`), PO₄³⁻ derived from the HPO₄²⁻ arg at rest pH 7.35, gates formation only. Grep-verified; data-validated: 0.5 µM→0 dimer (dead control), boundary matches A2 (137 µM off, 823 µM on). **Uncommitted live edit.** B2/B3 replace the rest-pH derive with the real PO₄³⁻ plumb (D7). | [GROUNDED probe] | E1·H |
| D9 | 2026-06-28 | **Pin-1 resolved: "4 nm read" is the grid floor `dx`, confirmed convenience not biology.** `n_channels_per_site=50`, placed `center+randint(-2,3)` → a 5×5 voxel (±8 nm) random cluster; `template_positions = channel_positions[:3]` (3 scaffold voxels). Real calcium at a site = 1/r sum over ~50 channels spread ±8 nm (A1 `cluster_field_physics`), not "3–6 co-located @4 nm". B2 read-distance must be grounded **sub-grid** (~1–2 nm molecular scaffold-channel) or read the cluster-field at the template voxel — never accept `dx`. | [GROUNDED code] | E1·H |
| D8 | 2026-06-28 | **A3 validates the B3 premise.** Finite phosphate + the gate → **exact conservation** (2e-17 M) and **SOC**: at sustained 823 µM the system self-organizes to **S=1** (47 µM dimer = 4.7% of P), self-limited by phosphate depletion; on Ca removal dissolution returns P (τ≈200 s). A2+A3 together prove the coupling: grounded Ca is what makes conservation load-bearing (at 0.5 µM nothing formed, so P was untouched). Caveats: clean S=1 pin uses a "can't-form-past-saturation" cap; SOC operating point couples to the Ksp band (at Meyer-Eanes 1e-25, 823 µM is sub-threshold) and to pH (D5). | [GROUNDED probe] | E1·G |
| D7 | 2026-06-28 | **Gate-wiring correctness:** the supersaturation gate is thermodynamic and must read the **trivalent PO₄³⁻** (`atp_system.PO4`, matching the `[Ca]³[PO₄³⁻]²` Ksp). The model's existing kinetic chemistry uses **HPO₄²⁻** (`get_posner_forming_species`, McDonogh 2024). Consistent across layers, but B-wiring must NOT read `get_posner_forming_species()` for the gate. | [GROUNDED code] | E1·F |
| D6 | 2026-06-28 | **pH sign is a B-phase decision, not a current bug.** Formation rate is ∝[Ca]² (via [PNC]); it does NOT read pH-driven [PO₄³⁻] as a rate term today — pH only sets the dimer/trimer split. So D5's pH-sign issue has ~zero effect on output until the gate is wired. The model's `pH_dynamics.py` is acidification-only (Krishtal 1987 / Chesler 2003 — extracellular/metabolic), no NHE alkalinization. | [GROUNDED code] | E1·F |
| D5 | 2026-06-28 | Intracellular spine pH during activity is **alkalinizing**, not the model's `pH_active=6.8` acidification → the gate likely **opens** (not shuts) during a burst. Model `pH_active` looks wrong-signed / wrong-compartment. | [GROUNDED lit; CONTESTED in model] | E1 |
| D4 | 2026-06-28 | The nucleation **threshold is a BAND, not a line** — published ACP pKsp spans ~24–28 → threshold ~150 µM–3.3 mM at rest. Validate the gate qualitatively (off at rest / on in nanodomain), not against a knife-edge. | [GROUNDED] | E1 |
| D3 | 2026-06-28 | Canonical ACP Ksp ≈ **1×10⁻²⁵** (Meyer & Eanes, Ca₃(PO₄)₂ unit); model uses 1×10⁻²⁶. Gate should eventually use ion **activities**, not concentrations (γ at I≈0.15 M raises threshold further). | [GROUNDED] | E1 |
| D2 | 2026-06-28 | At the model's [PO₄³⁻], **nucleation requires a channel CLUSTER**: bare 1-channel never nucleates at a physical radius; ≥6 co-located channels needed at the 4 nm read radius; the model's 3-channel template is sub-threshold there. Read distance + 3.9 kT template entry are now load-bearing for B2. | [GROUNDED probe] | E1 |
| D1 | 2026-06-28 | A2 confirms the **calibration/gate cancellation**: at 0.5 µM, S≈0.013 ≪1 (thermodynamically dead). The low-Ca calibration was silently doing the gate's job. | [GROUNDED probe] | E1 |
| C0 | 2026-06-28 | Committed the pre-revalidation baseline as 3 clean commits (chemistry reformulation `a992ee7`, input-engine `95990fd`, calcium probe/PDE `0ef0e0e`) before any Phase-B wiring, so the B-diff lands isolated. | [LOCKED] | E1 |

---

## THE LOG (newest first)

### E2 — 2026-07-18 · Forward-learning loop audit — D17's cross-trial reading RETRACTED

**Session shape.** Three parallel diagnostic agents, read-only against the code, each
instrumenting and running the live system. No repo source modified during the audit. Probes
persisted to `sweep/loop_audit_2026_07_18/` **because the T1′ scar applies here**: D17 and the
April-7 place-field result survive only as prose transcribed into handoffs and are not
independently re-derivable. These are.

**Trigger.** The question was operational — "is the multi-synapse system ready to carry a
forward-learning experiment with no backprop?" The audit was scoped to answer that. It
instead falsified the cross-trial half of D17.

#### A. The measurement gate fires ONCE PER EXPERIMENT, not once per trial  `[GROUNDED, observed]`

`sweep/loop_audit_2026_07_18/probe_latch2.py`, instrumented 3-trial run (reduced config: 4
synapses, 14 s budget, deterministic agent — every trial reached the goal and delivered
dopamine):

| trial | gate called with reward | `perform_quantum_measurement` CALLS | latch early-returns |
|---|---|---|---|
| 0 | 1 | **1** | 0 |
| 1 | 1 | **0** | **1** |
| 2 | 1 | **0** | **1** |

`_network_measurement_performed` (`multi_synapse_network.py:1335-1337`) is a one-shot latch
reset only in `__init__`, `reset()`, and two probes. `run_spatial_discovery.py:417-419` clears
`_camkii_committed` and `network_committed` per trial and **not** the latch.

**The classical control shares the same flag.** `_evaluate_independent_gate` reads and writes
the identical latch at `:1476/:1478`. Whichever gate fires first locks out the other, so the
coordinated-vs-independent comparison — the control for whether the correlated partition
matters at all — **is invalid by construction**. `[GROUNDED code SHOWN; the comparison itself
was not re-run]`

#### B. Spine volume accumulates anyway, by two paths, neither of them quantum  `[GROUNDED, observed]`

Observed end-of-trial spine volume in the same run:

| syn | gate_opened | T0 | T1 | T2 |
|---|---|---|---|---|
| 0 | True | 1.243 | 1.488 | 1.838 |
| 1 | True | 1.246 | 1.685 | 2.124 |
| 2 | True | 1.184 | 1.554 | 1.815 |
| **3** | **False — never gated, never committed** | **1.065** | **1.230** | **1.430** |

- **Path A — stale flag.** `_measurement_gate_opened` is written `True` at exactly one site
  (`multi_synapse_network.py:1383`) and **never cleared anywhere, including by `reset()`**.
  With `_camkii_committed` cleared each trial, later trials re-commit on `(stale flag) AND
  (CaMKII calcium integral > 0.5)` — observed with `pqm_calls = 0`. Two synapses committed
  for the FIRST time in trial 2 with no measurement having occurred.
- **Path B — commitment is not required at all.** Synapse 3 grew comparably while never
  gated and never committed. Actin formation (`spine_plasticity_module.py:340`) is a function
  of **calcium only**; `structural_drive` enters solely through `confinement` (`:325-326`),
  which trades extrusion for retention. **Commitment buys durability, not amplitude.**
  Measured directly (`probe_spine_volume.py`): driving `committed_memory_level` 0→1 at fixed
  Ca=5 µM *lowers* enlargement 1.447→1.099. Magnitude is the Hill-4 calcium term at `:320`
  (Ca=0.5→V=1.22; Ca=1.0→V=2.34).
- **Compounding both:** `run_spatial_discovery.py:434-437` steps only *active* synapses, so a
  silent synapse never runs its extrusion/decay term. Volume **ratchets** rather than
  accumulating against decay.

#### C. What of D17 survives  `[the retraction]`

**SURVIVES** — everything upstream of the gate, i.e. the single-trial chemistry this log
exists to record: dimer formation emergent, bounded, rise-and-fall (9/1/31/23/22, peak
transient 318, no runaway), localized (1/1/5/4/3 of 20). D18's bistable-nucleation result is
untouched (single-synapse probe, no network gate).

**RETRACTED** — the cross-trial reading. *"Plasticity ACCUMULATES across traversals
(1.63→1.90→2.59→2.59→2.67)"* is real as a number and **false as evidence of quantum-gated
learning**: trials 2–5 contained no quantum measurement at all. *"formation→spine-growth→
behavior closes"* does not survive either — the behavioural read is `get_synaptic_strengths()`
= spine volume, contaminated by Paths A and B above.

**Caveat, stated so it is not over-read:** the instrumented run was a reduced config with a
deterministic agent. The latch behaviour is structural and config-independent — that part is
certain. The *relative weight* of Path A vs Path B at the real 20-synapse/60 s config was NOT
observed and must not be quoted as if it were. `[INFERRED]` that D17's specific run followed
this exact chain; `[GROUNDED]` that it contained no measurement after trial 0.

#### D. Durable-state channels — what could carry learning at all  `[GROUNDED, measured]`

`probe_spine_volume.py`, `probe_templates_and_drive.py`, `probe_duty.py`:

- **Spine volume** — the only working channel. Ceiling **3.80** (actin-limited at
  `spine_plasticity_module.py:332-333`, *not* the `max_enlargement_ratio=3.9` clip at `:381`;
  the two ceilings do not coincide). **It decays**: τ≈1000–2000 s, falling *below* baseline
  1.0 by t=3000 s. This **falsifies primitive 4** of `coherence-gated-learning`
  ("self-reinforcing maintenance regime that resists decay") — committed decays *slower* than
  uncommitted, but does not self-maintain.
- **`analytical_gap` does not advance the plasticity clock.** It adds `gap_duration_s` to
  `network.time` directly (`run_spatial_discovery.py:296`) then runs one 1 ms
  `network.step`. Observed at end of run: `network.time = 46.518` vs
  `spine_plasticity.time` = 25.5/31.5/25.5/16.5. **A 30 s gap contributes 1 ms.** Volume
  survives gaps because the gap is *skipped*, not because it resists decay.
- **AMPAR is dead twice over.** Clock: `ampar_onset_delay = 1800.0` with an unconditional
  early return (`:420-425`), and measured duty cycle (10 seeds, shipped config) is 0.189 best
  / 0.037 mean ⇒ **159–809 trials** to reach onset against a shipped `n_trials=5`. Flag:
  the whole AMPAR→voltage chain (`model6_core.py:309-316`) is gated on
  `spine_calcium_feedback`, which `make_network` defaults **False**. No experiment has come
  within 1.5–2 orders of magnitude.
- **Template feedback DOES fire and is the most reachable second channel.** `spine_volume >
  1.25 → n_templates 5` (`model6_core.py:632-644`) is unconditional in the EM path and V
  reaches 1.25 by ~8 s at Ca=5 µM. Measured effect is small on rates (mean template
  enhancement ×1.010 at n=5, ×1.015 at n=6) **but roughly doubles the template-bound
  fraction** at n=6 — which raises `T_eff` via `template_factor=0.7`
  (`dimer_particles.py:311`). That is a *coherence-window* lever, not a rate lever. Caveat:
  dissolution carries the same `template_enhancement` factor
  (`ca_triphosphate_complex.py:418`), so the net standing-count effect is unsigned by
  inspection and needs measuring in a live run.
- **Coupling weights are never a learning channel** — computed once from geometry
  (`multi_synapse_network.py:807-817`, called once from `__init__`), never mutated anywhere.
- **`apply_reward_correlated` (`:1187`) and `sample_correlated_eligibilities` (`:1147`) are
  orphaned, and would be near-no-ops if wired** — they write to `_committed_memory_level`,
  which §B measured does not set enlargement magnitude, and which is *assigned* (not
  accumulated) at the next commitment (`model6_core.py:616`). This looks like why they were
  abandoned rather than an oversight.
- **`src/models/Model_6/eligibility_trace.py` is fully orphaned** (zero importers), superseded
  by `model6_core.py` PHASE 9. Its P31/P32 isotope parameterisation is the only part without
  an in-tree equivalent — check the isotope kill-switch control before deleting.

#### E. Construct-validity gaps found (declared ≠ implemented)  `[GROUNDED]`

Per `quantum-computation-and-attribution` §6.5 — these must be closed before any behavioural
result is load-bearing:

1. **`quantum_field_kT` is inert in spine plasticity.** Measured **bit-identical** volume to 8
   decimals for kT ∈ {0, 1, 5, 20, 100}. It is accepted at `spine_plasticity_module.py:273,
   311, 400` and read in no body. Five `ActinParameters` fields (`barrier_polymerization_kT`,
   `barrier_depolymerization_kT`, `barrier_stabilization_kT`, `quantum_coupling_efficiency`,
   `barrier_electrostatic_fraction_*`, `:95-107`) are declared and never referenced. **The
   module docstring describes a quantum barrier-modulation mechanism that does not exist in
   the code.**
2. **The module docstring lists `molecular_memory` as the input that drives storage.** It does
   not drive magnitude; calcium does (§B).
3. **`spatial-discovery-experiment` skill** states "structural state (spine volume, AMPAR)
   persists". Spine volume persists; **AMPAR never changes at all** in any run performed.
4. **`coherence-gated-learning` primitive 4** is falsified as written (§D).
5. **No cross-synapse bonds form during trials.** `run_trial` calls the tracker without
   `coupling_weights` (`run_spatial_discovery.py:446-449`) and `_update_entanglement` returns
   early when they are `None` (`:276-279`). The topology is built only on the single reward
   step. Combined with `L·ETA-1` (eta=0 ⇒ `k_cross`=0), there are **two independent reasons
   the eligibility structure does not exist during a learning trial.**

#### F. Net verdict on the trigger question  `[the answer]`

**The system is not ready, and the gap is not the one that was visible from the outside.**
What ran end-to-end was calcium-driven spine growth with a quantum gate that fired once and
wrote to a lever that does not control magnitude. All three links of the architecture are
independently broken: input (eta=0 ⇒ no partition), readout (one-shot latch, shared with its
own control), and write (commitment sets durability, not amplitude).

**Sorted, because the two classes need different treatment:**
- **Bugs** — fix without physics judgment: both never-reset latches (including `reset()`
  itself), the shared coordinated/independent flag, `coupling_weights` not passed in
  `run_trial`, silent-synapse decay never running, results never persisted
  (`run_spatial_discovery.py:24` imports `json` and never uses it).
- **Model-design decisions — Sarah's, NOT the thread's:** whether commitment should drive
  enlargement amplitude rather than only durability; whether `analytical_gap` should advance
  the plasticity clock; whether `quantum_field_kT` in spine plasticity is to be implemented or
  its declaration deleted; and the `L·ETA-1` fork on the pump. **None of these is a tuning
  knob and none was touched.**

### E1 — 2026-06-28 · A2 supersaturation gate probe + first external literature pass

**Session shape.** Claude Code (this session) had direct repo + web access; Sarah approved
all git/execution. Work order: reconcile the uncommitted working tree against the handoff →
commit the baseline → build/run the A2 gate probe → interpret → external literature pass.

#### A. Baseline reconciliation & commit  `[LOCKED]`
The working tree on `master` held two finished-but-uncommitted bundles that the handoff
treated as not-started; reconciled against code (code = what IS):
- `ca_triphosphate_complex.py` = the **Option-B detailed-balance reformulation** already
  documented in `model6-dimer-formation-chemistry` §1 (remove 1 µM gate + 8 µM MM overlay;
  `k_base` 8e5→1.9e4 = productive_fraction×Smoluchowski; `k_classical` 0.05→0.005; symmetric
  template). Pre-revalidation baseline.
- `analytical_calcium_system.py` + `model6_parameters.py` + `run_spatial_discovery.py` =
  **input-engine glutamate/NMDAR-VGCC + presynaptic-release wiring** (separate effort) plus
  the voltage Edit 1 (peak −10 mV→−40 mV subthreshold ceiling).
- `calcium_system.py` = a buffer **sign fix** (`b: +Ca_tot→−Ca_tot`) in the now-retired PDE.
Confirmed **the supersaturation gate / grounded calcium / conserved phosphate are genuinely
NOT in the code** — the handoff's "Phase B not started" holds.
Committed as `a992ee7` (chemistry), `95990fd` (input-engine), `0ef0e0e` (calcium probe + PDE).
Pushed `ccd89d7..0ef0e0e`. `.claude/` symlink, `CLAUDE.md`, and the `.npz` result binary
deliberately excluded.

#### B. A2 probe — method  `sweep/supersaturation_gate_probe.py`  `[GROUNDED]`
Isolated pure-algebra probe; no live code touched. Reuses (replicated verbatim, self-test
asserted): the Naraghi-Neher closed-form calcium `ca_physics` from
`nanodomain_closedform_probe.py:115-123`, and the triprotic phosphate speciation from
`atp_system.py:382-401`. Gate: `Ksp = [Ca]³[PO₄³⁻]² = 1e-26 M⁵`; `S = (IAP/Ksp)^(1/5)`;
nucleation allowed iff `S>1`.

**Grounded inputs read from code (not estimates):**
- λ = 165 nm (the A1 self-test value, `B_FREE=B_total=300 µM`, `k_on=2.7e7`); handoff's 117 nm
  uses 600 µM binding sites — differs <1% at the 4 nm read radius, so the near-mouth peak is
  robust. Self-test: 137.3 µM @ 4 nm, single 0.3 pA channel — **PASS**.
- pKa1/2/3 = 2.1/7.2/12.4; structural phosphate = 1 mM; `pH_rest` = 7.35.
- **[PO₄³⁻] @ rest = α₃·1mM = 5.2 nM** (α₃ = 5.22×10⁻⁶). *Handoff §3b estimated ~10 nM —
  code is ~2× lower.* (D-correction)

**Results (model's [PO₄³⁻]=5.2 nM, rest pH):**

| scenario | [Ca] | S | nucleates |
|---|---|---|---|
| rest (100 nM) | 0.10 µM | 0.005 | no |
| old calibration | 0.50 µM | 0.013 | no |
| bare 1 channel @4 nm | 137 µM | 0.37 | no |
| 3-channel template @4 nm | 412 µM | 0.72 | no |
| 6-channel template @4 nm | 823 µM | 1.09 | **yes** |

Threshold (S=1): **716 µM @ pH 7.35**, 643 µM @ pH 7.4. r×n sweep: only (n=6, r=4 nm)
crosses; everything ≥8 nm is sub-threshold. A single channel crosses only at r≲0.7 nm
(sub-physical). *Handoff §3b's "S≈10⁻³⁰ at 0.5 µM" was wrong by ~28 orders — with the
5th-root definition S≈0.013; conclusion (undersaturated) unchanged.* (D-correction)

**Findings D1, D2** (see decision record). The §8 selectivity question — *does the gate +
multi-channel geometry give selectivity on its own?* — **answered YES, and sharper than
expected**: nucleation requires a channel cluster. This promotes two handoff-"convenience"
items to load-bearing for B2: the template→channel **read distance** and **how the 3.9 kT
(=ln 50) template heterogeneous catalysis enters** (as a rate factor it acts only *above*
threshold, so it can't rescue the sub-threshold 3-channel case — only an effective
critical-S shift would, which needs the deferred ACP interfacial-energy pin).

#### C. External literature pass — five pins  `[GROUNDED / CONTESTED as tagged]`

**Pin 2 — ACP Ksp → the threshold is a BAND (D3, D4).**
Canonical **ACP Ksp ≈ 1×10⁻²⁵** for a TCP-like Ca₃(PO₄)₂ unit (Meyer & Eanes) — the exact
`[Ca]³[PO₄]²` form. Model uses 1×10⁻²⁶ (Fetuin-A paper). Swapping in 1e-25 raises the rest
threshold **716 µM → ~1.5 mM** (then even a 6-channel template is sub-threshold).
**Published pKsp(ACP) spans ~24–28** (phase/hydration/activity convention) → threshold
**~150 µM (pKsp 28) – 3.3 mM (pKsp 24)**. *The Ksp uncertainty alone spans the entire
nucleation outcome.* Also: probe uses **concentrations**; literature Ksp is **ion-activity**
based — at I≈0.15 M (γ_Ca≈0.4, γ_PO4≈0.1) the effective IAP drops, raising the threshold
further. → Treat the gate as band-validated `S>1`, checked qualitatively, with the Ksp
uncertainty stated, not as a sharp 716 µM line. `[GROUNDED]`

**Pin 4 — intracellular pH sign → likely a model error; flips the burst finding (D5).**
Literature: NMDA activation drives a **biphasic** intracellular pH change — brief initial
acidification then a **dominant alkalinization of the dendritic spine** (NHE5 recruitment),
hundreds of ms; the **synaptic cleft acidifies** while the **intracellular spine alkalinizes**.
The model's `pH_active=6.8` (acidification) looks like the **cleft** value applied to the
**intracellular** formation site. Correcting to intracellular alkalinization → more PO₄³⁻
during activity → threshold drops → **the gate OPENS during the burst**. This *reverses* the
A2-derived "gate shuts at burst peak" concern, which was an artifact of the model's pH sign.
**Highest-impact, best-supported correction the lit pass surfaced.** `[GROUNDED lit direction;
CONTESTED in model — magnitude/compartment still to pin]`

**Pin 3 — free PO₄³⁻: regime right, availability is a lever.**
pKa3 = 12.4 confirmed; free PO₄³⁻ at cytosolic pH is **nM or below** — model's ~5 nM is
correctly in-regime. Lever: total free cytosolic Pi cited **1–10 mM** (model uses 1 mM
"structural"); upper end → ~50 nM PO₄³⁻ → threshold ~3× lower. ATP hydrolysis transiently
raises local Pi during activity (pathway already in the model). `[GROUNDED]`

**Pin 1 — read distance: 4 nm is generous, not conservative.**
Typical nanodomain Ca-sensor coupling is **20–50 nm**; the model reads at **4 nm** (very near
mouth). Relative to a diffusible sensor that *over*-estimates calcium. Sub-nanodomain reading
is justified only by the Tao-2010 **scaffold-physically-on-the-channel-cluster** claim — a
specific structural commitment to be confirmed by reconning `model6_core` geometry, not the
generic sensor distance. (Posner cluster ≈ 0.95 nm, so ~1 nm is its own size scale.)
`[GROUNDED lit; MODELED in code — needs geometry recon]`

**Gate concept itself is well-supported.** The "active niche" framing — transient
supersaturation becoming thermodynamically accessible in synaptic nanodomains during intense
Ca²⁺ influx — is the literature picture (Fisher line; Meyer-Eanes: ACP releases
"supersaturating levels"). The gate is the *right object*; only its absolute threshold is
uncertain. `[GROUNDED]`

#### D. New references for the program
- **PNAS 2025 — quantum effect in Li-doped ACP formation** (10.1073/pnas.2423211122): a real
  experiment on Li-doped ACP with a claimed quantum effect — bears directly on the lithium
  attribution bet and the in-vitro witness (`quantum-computation-and-attribution` §5–6). A
  *discriminating-measurement* candidate, not just substrate physics. **Follow up: fetch for
  the protocol + exact result.**
- **arXiv:2108.08822 — "The Dynamical Ensemble of the Posner Molecule is not Symmetric"**:
  independent support for the Agarwal Posner-asymmetry / dimer-not-trimer correction.

#### E. Net implications & open questions for B2
"Does a 3-channel template nucleate?" **cannot be settled by Ksp** — it's inside the band.
What determines formation, in priority order:
1. **Fix the pH sign** (D5) — likely flips suppression→enhancement during activity. *Next step.*
2. **Phosphate availability** (Pin 3) — 1 vs up-to-10 mM + the ATP transient.
3. **Ground the read distance** (Pin 1) on the scaffold geometry — recon `model6_core`.
Then: gate as band-validated `S>1`, threshold uncertainty stated. B1 (wire gate, Ca still
0.5 µM) remains a clean dead negative control.

**Deferred / not yet done:** A3 phosphate-conservation probe; B-phase wiring; ACP interfacial
energy pin (for CNT / the 3.9 kT-as-effective-S-shift); fetch Meyer-Eanes + the PNAS Li paper
for exact numbers; confirm `model6_core` channels-per-template & scaffold distance.

#### F. Pin 4 recon — pH path & the phosphate-species layer  `[GROUNDED code SHOWN]`
SHOWN: `pH_dynamics.py`, `atp_system.py`, `ca_triphosphate_complex.py`.
- **The model's pH is acidification-only.** `pHSources.calculate_h_production` produces H⁺ from
  ATP hydrolysis + lactate/glycolysis + Ca-buffering → pH 7.35→6.8, then `pHRecovery` relaxes
  back toward baseline (no overshoot). Cites Krishtal 1987 (activity-induced acidification) +
  Chesler 2003 — a real but **extracellular/metabolic** picture; **no NHE-driven intracellular
  alkalinization** is modeled. Confirms the model is acidification-signed, opposite to the
  intracellular-spine literature (D5).
- **The pH sign currently has ~zero effect on formation (D6).** `update_speciation(pH)` is driven
  by the dynamic pH, so `[PO₄³⁻]`/`[HPO₄²⁻]` do move with activity — but the formation *rate* is
  ∝[Ca]² (via the [PNC] clamp); phosphate enters only through the CaHPO₄ ion-pair and the
  dimer/trimer Ca/P split, **not as a rate-limiting term**. So pH sign matters only once the gate
  is wired (B-phase). Not a now-fix.
- **Species layer (D7):** `CaHPO4DimerSystem.step` documents its `po4_conc` arg as
  *"HPO₄²⁻ concentration (M) at pH 7.3"* (`ca_triphosphate_complex.py:560`), and
  `get_posner_forming_species()` returns **HPO₄²⁻** (`atp_system.py:430`, McDonogh 2024). The
  thermodynamic gate, by contrast, needs the **trivalent PO₄³⁻** (`atp_system.PO4`, line 401) to
  match its `[Ca]³[PO₄³⁻]²` Ksp. Both are correct *at their own layer* (kinetic pathway vs
  thermodynamic solubility); the B-wiring must read `phosphate.PO4` for the gate, never
  `get_posner_forming_species()`.

**Reframed next-step priority:** D5's pH-sign fix is no longer the immediate highest-impact change
(it's inert until the gate is wired). The remaining isolated-probe work is **A3 (phosphate
conservation)**; the pH-sign + species-layer items are now logged **B-phase wiring constraints**.

#### G. A3 — phosphate conservation + SOC probe  `sweep/phosphate_conservation_probe.py`  `[GROUNDED]`
Grounded the live conservation path first (SHOWN): `ca_triphosphate_complex.py:414-415`
(`_po4_consumed = 4·d_dimer + 6·d_trimer`, signed net so dissolution returns P) and
`model6_core.py:388-392/671-675` (subtract `po4_consumed` from `phosphate_structural`, `max(...,0)`
floor). Plumbing partly exists; the missing piece is the **feedback** — nothing reads `[PO₄³⁻]` to
gate formation, so depletion can't self-limit via S. A3 builds that full loop in isolation.

Probe: finite P_total=1 mM; formation `k_base·[PNC]²·(S>1)`, dissolution `k_classical·dimer`;
4 PO₄/dimer; formation capped so it cannot drive S below 1 in a step (the thermodynamic
statement — the one modeling choice). Drive Ca=823 µM (A2 6-ch) for 0–150 s, then rest to 500 s.

Result — **all three A3 validations pass**:
1. **Conservation exact:** max|P_free + 4·dimer − P_total| = 2×10⁻¹⁷ M (machine precision).
2. **PO₄³⁻ buffered by HPO₄²⁻:** both = α·P_free; scarce ~5 nM PO₄³⁻ on the 0.5 mM HPO₄²⁻
   reservoir (α₃/α₂ = 8.9×10⁻⁶, pH-fixed).
3. **SOC self-limiting:** dimer 18→47 µM in ~1 s, S 1.087→**1.000 and pinned** through the drive
   (47 µM = 4.7% of P); on Ca removal S→0.0045, dimer 47→8.2 µM and P_free recovers 0.812→0.967 mM
   over τ≈200 s.

**Significance (D8):** A2+A3 prove the coupled-fix logic — grounded calcium is what makes phosphate
conservation load-bearing (at the old 0.5 µM nothing formed, so P was untouched and conservation
was inert). The SOC attractor at S=1 emerges from physics (mass-action + Ksp gate + conservation),
not tuning. **Phase A is now complete (A1 calcium, A2 gate, A3 conservation+SOC).** Remaining work
is all Phase B (wiring) + the logged B-constraints (pH sign/compartment D5/D6, PO₄³⁻-not-HPO₄²⁻ D7,
Ksp band D4, read distance Pin 1). Caveat: the SOC operating point is contingent on the Ksp band —
at Meyer-Eanes 1e-25, 823 µM Ca is sub-threshold (S≈0.68), so which calcium triggers SOC moves with
the Ksp/read-distance uncertainty.

#### H. Pin-1 geometry recon (D9) + B1 gate wiring (D10)  `[GROUNDED code SHOWN]`
**Pin-1 (read-distance) — SHOWN `model6_core.py:222-258`, `model6_parameters.py:128`:**
`n_channels_per_site = 50`; channels placed `center + randint(-2,3)` in x,y → a **5×5 voxel
(±8 nm) random cluster** (`dx`=4 nm grid). `template_positions = channel_positions[:3]` → 3 scaffold
voxels, each at a channel location. So the **"4 nm read distance" is the grid floor `dx`** — the
minimum resolvable separation — **confirmed grid convenience, not biology** (handoff §10 suspicion
right). Real calcium at a dimer site = 1/r-weighted sum over ~50 channels spread ±8 nm (A1
`cluster_field_physics`), richer than A2's idealized "3/6 co-located @4 nm". **B2 decision:** read
the cluster-field at the template voxel and ground the near-mouth distance **sub-grid** (~1–2 nm,
the molecular scaffold-channel distance), not accept `dx`. Template enhancement field: 50× decaying
1.5 nm from the scaffold surface (`ca_triphosphate_complex.py:509-514`).

**B1 gate wiring (D10) — option (b), single hunk:** inserted at `ca_triphosphate_complex.py:387-402`,
multiplying `dimer_formation`/`trimer_formation` by `gate = (S>1)`; `S` from `ca_conc` and PO₄³⁻
derived in-place as `po4_conc(HPO₄²⁻) × 10^(7.35−12.4)` (rest pH). Dissolution untouched. Chosen over
the multi-file plumb (a) because B1 is only the dead-control — the proper live-PO₄³⁻ plumb (D7) lands
at B2 where the calcium/speciation path is already being reworked. **Validation (chemistry-level,
400×0.05 s, template=1):** rest 0.1 µM → 0 dimer; 0.5 µM → **0 dimer (dead control PASS)**; 137 µM →
0 dimer; 823 µM → 0.084 µM (forms). Boundary matches A2 exactly. Live edit, **uncommitted**.

**Phase B status:** B1 ✅ (dead control). Next: **B2** — ground the calcium amplitude (cluster-field
+ sub-grid read distance D9; how the 3.9 kT template catalysis enters §4) and replace the B1 rest-pH
derive with the real PO₄³⁻ plumb (D7) + the pH-sign/compartment decision (D5/D6). Then **B3** —
conservation feedback → live SOC.

#### I. Read-distance grounding literature pass (D11)  `[GROUNDED lit]`
Question (from the "what keeps us biologically realistic" decision): the channel-to-scaffold read
distance and the Naraghi-Neher validity floor — so B2's amplitude is grounded, not tuned.
- **Structural distance is measured: ≈ 5.5 nm.** Tethered genetically-encoded sensor →
  channel-cytoplasmic-mouth distance ≈ 55 Å for CaV2.2 (Nature Comms `ncomms1777`). General
  nanodomain Ca-channel↔sensor coupling = a few tens of nm (Eggermann/Jonas, Nat Rev Neurosci);
  trans-synaptic nanocolumns align scaffolds within tens of nm (Tang & Blanpied 2016, Nature).
  → A channel-tethered scaffold sits ~5.5–20 nm from the mouth, **not sub-nm**.
- **Validity:** Naraghi & Neher 1997 (J Neurosci 17:6961) is explicitly the calculation of "[Ca²⁺]
  at the mouth", LBA valid in the 5–50 nm nanodomain regime. At ~5.5 nm we are inside validity →
  **no r→0 divergence**; the near-mouth-breakdown / sub-nm push is biologically wrong.
- **Resolution (D11):** biology pins r near the model's 4 nm grid floor (≈5.5 nm). Single channel
  ~100 µM @5.5 nm → sub-threshold; **nucleation requires the cluster sum over OPEN channels** →
  emergent clustering + coincidence (glutamate+depolarization) requirement, nothing tuned. The read
  distance is therefore NOT the dramatic free lever feared at D9 — it is tightly constrained, and to
  a no-tuning value. **B2: cluster-field sum, nearest-channel r floored ≈ 5.5 nm (5–20 nm
  uncertainty), closed-form amplitude, λ≈117 nm, then run and let the gate decide.**

#### J. B2a — calcium amplitude grounded (D12)  `[GROUNDED, live edit]`
`analytical_calcium_system.py` `AnalyticalNanodomainCalculator`, 5 hunks (3 behavioral + 2
docstring), grep-verified, calibrated `ca_per_channel=0.5e-6` removed from both read methods.
- **Amplitude:** flat 0.5 µM·(i/0.3pA)·exp(−r/λ) → `i/(z·F·4π·D_ca·r)·exp(−r/λ)·1e-3` (the 1/r).
- **Prefactor diffusion fix:** live code used **buffered `D_eff`≈3.6** in the prefactor; corrected
  to **free `D_ca`=220** (Naraghi-Neher: buffer is NOT equilibrated in the nanodomain → free D in
  the prefactor, buffering only in λ). Genuine correctness fix.
- **λ:** pump-set `√(D_eff/k_pump)`≈190 nm → buffer-set `√(D_ca/(k_on·κ_s·Kd))`≈117 nm
  (k_on=2.7e7 Nägerl; [B]=κ_s·Kd=600 µM).
- **Read floor:** `dx`(4 nm) → 5.5 nm (D11).
Validation (`/tmp/b2_validate.py`, data-level): λ=117 nm; single 0.3 pA channel = 97.5 µM @5.5 nm,
65.6 @8, 30.6 @16, 14.5 @30; 7-channel cluster (±2 voxels) = 616 µM. **Emergent check:** 616 µM →
S≈0.91 (just below the 716 µM gate) → sparse clusters don't nucleate, dense do; with NMDAR/VGCC
gating → coincidence + clustering required. No tuned value (D_ca, k_on, κ_s·Kd, r=5.5 nm all
sourced). **Remaining Phase B:** B2b — multi-file PO₄³⁻ plumb (D7) so the gate reads live pH-driven
PO₄³⁻; B3 — conservation feedback → live SOC; then a full `run_spatial_discovery` integration run.

#### K. Species blocker → resolution → full integration (D13–D17)
- **D13 blocker:** B2a-grounded calcium (Ca/P>0.5) flipped the bulk-Ca/P sigmoid to the INERT trimer
  (integration loop: 0.19 mM P consumed but only 2 nM dimer ≈ 31 µM trimer).
- **D14:** the SOC loop is already closed by B1+B2a+existing consumption plumbing (S pins 1.0,
  P_struct stabilizes) — **no B3 edit needed.**
- **D15 (Option-B research):** species selection = aggregation extent (kinetic); aggregation rate is
  ungroundable (~ns at high SS → trimer; up to hours via Ca/Pi control → dimer), so a formation split
  would be tuning. Agarwal is SILENT on which forms.
- **D16 (resolution):** drop the invalid Ca/P sigmoid → `dimer_fraction=1`; the operative species is
  selected DOWNSTREAM by coherence (dimer persists ~100s s, trimer decoheres sub-second). Loop now
  forms 49 µM dimer (= A3's 47 µM), S=1, P_struct 0.81 mM.
- **D17 (full integration):** `run_spatial_discovery` 5 trials × 20 synapses — EMERGENT, BOUNDED
  (end totals 1–31, peak 318, no runaway), LOCALIZED (1–5/20 synapses), plasticity ACCUMULATES across
  traversals (spine 1.63→2.67), agent FOUND goal in trial 3. First emergent + bounded + rise/fall run.
  Validates the **(A)** coherence-gated floor; the **(B)** genuine-quantum claim is untouched.

**Session close (2026-06-28):** the calcium→dimer coupled revalidation is LANDED and integration-
validated. Committed `2ab02d8` ("dimer working system" = B2a calcium + D16 species + this log) on top
of B1 (`49c7453`). Open: B2b PO₄³⁻ plumb + pH-sign (D5/D6/D7); skill updates; orphaned
`calculate_dimer_fraction` (~L425); characterize the near-critical variability. See the session handoff.

#### L. Near-critical variability — the all-or-none switch (D18) · 2026-06-28 (cont. after close)

Follow-on to D17's parenthetical "stochastic/near-critical." D17's trial-to-trial spread
(9/1/31/23/22 dimers) conflates three sources — genuine per-traversal nucleation, evolving
structural/learned state, and re-randomized start positions — so it cannot be read as a clean
criticality signal. A criticality claim is a **distribution** claim and needs many independent
samples under controlled drive.

- **Instrument** (`sweep/criticality_variability_probe.py`, `[GROUNDED probe]`): ONE synapse
  (config identical to the network's — EM on, P31, feedback OFF, MT-invaded), held at a FIXED
  subthreshold drive for a fixed episode, dimer count recorded; repeated N≈120× with the global
  `np.random` reseeded per replicate. This isolates the gate's stochasticity (channel-gating CTMC
  `analytical_calcium_system.py:132` + dimerization noise `ca_triphosphate_complex.py:405,410`)
  from the agent/structural/start-position confounds. Glutamate held constant (sustained agonist)
  so the control parameter is drive voltage alone; presynaptic-release stochasticity is excluded by
  design (added back in the control). Order-parameter proxy = peak nanodomain [Ca] vs the S>1 gate.

- **Result — bistable switch with a forbidden gap** (N=120/drive, peak-dimer count):

  | act | V (mV) | peak [Ca] µM | P(any nucleation) | P(full ≥60) | Fano (var/mean) |
  |----:|-------:|-------------:|------------------:|------------:|----------------:|
  | 0.70–0.80 | −49…−46 | 423–466 | 0.00 | 0.00 | 0 (hard zero) |
  | 0.85 | −44.5 | 515 | 0.19 | — | 2.4 |
  | 0.88 | −43.6 | 562 | 0.28 | 0.07 | 122 |
  | **0.90** | **−43.0** | **568** | **0.30** | **0.11** | **119** |
  | 0.92 | −42.4 | 600 | 0.40 | 0.13 | **148** ← peak |
  | 0.95 | −41.5 | 632 | 0.76 | 0.35 | 96 |
  | 1.00 | −40.0 | 715 | 1.00 | 0.75 | 65 |

  Across the four high-N drives, **0 of 480 replicates** landed in 11–120 dimers. Replicates are
  either *silent/fizzle* (0, or a transient ≤~8 that dissolves) or *full* (~125–151). The empty
  middle is the unstable separatrix of a bistable system — all-or-none, not a fat tail (Sarle
  bimodality coeff 0.77–0.98, all > 0.555). The ON-state is **quantal** (~135 dimers, ~drive-
  independent): the supersaturation runaway has a fixed attractor; drive tunes only `P(catch)`.
  Critical point ≈ −43 mV / ~570 µM, just under the 716 µM gate (consistent with D11/D12).

- **Mechanism:** subthreshold drive makes channel openings rare and uncoupled; the nanodomain only
  crosses S>1 when openings happen to **coincide and cluster**. Below threshold this ~never happens
  (hard zero); near threshold it happens stochastically (the switch); once S>1 is crossed, the gate
  opens and supersaturation runs to the attractor. This is the place-cell all-or-none / BTSP one-
  shot form (`experiment-design-patterns`) at the mechanistic level, and it **explains the D17
  spread**: a 60 s multi-feature traversal integrates many such independent switches across synapses.

- **Controls (both PASS):**
  - *Duration sensitivity* (act 0.90, durations 0.25/0.5/1.0/2.0 s, N=100): the gap holds at every
    duration (fizzle max 3→6, full 125→232); duration does **not** scale the ON-amplitude — it only
    raises `P(catch)` (0.03→0.06→0.09→0.19). Confirms the attractor is real and the 1 s window is not
    an artifact.
  - *Presynaptic-on* (`--presynaptic`, the real stochastic cleft-glutamate layer added back, N=100):
    the gap survives (act 0.90: fizzle max 10, full 126–148, Fano 118); the critical point barely
    moves. A second independent noise source does not smear the bistability. Confirms it is not an
    artifact of the constant-glutamate idealization.

- **Altitude (LOCKED):** this is a **classical** stochastic nucleation criticality on the **(A)**
  coherence-gated floor (channel coincidence → supersaturation gate). It is NOT a quantum effect and
  says nothing about the **(B)** genuine-quantum claim. Do not describe it in (B)'s language.

- **Experimental prediction (falsifiable; for a future (B)/lab build).** If dimer (Posner-precursor)
  formation at a spine is the gated supersaturation switch this model implies, then under graded,
  controlled Ca influx (e.g. graded uncaging / graded depolarization at a single spine):
  1. the **formation order parameter is bimodal** — events are all-or-none, not graded — with a
     **quantal "ON" amount** that is ~independent of the drive level;
  2. **P(ON) is a sharp sigmoid** of Ca drive (Hill-steep), centered just below the ACP S=1 gate;
  3. **trial-to-trial variance (Fano) peaks at the midpoint** of that sigmoid (critical fluctuations);
  4. the signatures are **invariant to stimulus duration** (duration shifts P(ON), not the ON amount)
     and to upstream input noise.
  **Kill conditions:** a graded (unimodal, drive-proportional) formation amount, or a variance that
  does not peak at the P(ON)=0.5 drive, falsifies the gated-switch picture. **Controls to run in a
  (B) build:** P32 isotope (must NOT change the *classical* switch — separates (A) from (B)); Ksp-band
  sweep (D4) — the critical drive should track the band, not a single line.

- **Provenance:** probe `sweep/criticality_variability_probe.py` (uncommitted); analysis + JSON in
  session scratch (uncommitted). `experiment-design-patterns` owes a pointer to this prediction +
  its controls. No model code was edited for this characterization (the `--presynaptic` flag is the
  only probe-side addition).

---

## REFERENCES (with what each grounds)

- **Meyer & Eanes** — thermodynamic analysis of ACP→crystalline transformation; ACP Ksp ≈ 1e-25,
  TCP-like unit. (Pin 2 / D3-D4) https://link.springer.com/article/10.1007/BF02010752
- **Calcium Phosphates: Structure, Composition, Solubility, and Stability** (Springer) — pKsp
  spread across CaP phases. (Pin 2) https://link.springer.com/chapter/10.1007/978-1-4615-5517-9_1
- **NMDA induces a biphasic change in intracellular pH** (ScienceDirect, hippocampal slices) —
  acidification→alkalinization. (Pin 4 / D5) https://www.sciencedirect.com/science/article/abs/pii/S0006899397002783
- **Activity-dependent NHE5 / dendritic-spine pH** (PMC3128527) — intracellular spine
  alkalinization vs cleft acidification. (Pin 4 / D5) https://pmc.ncbi.nlm.nih.gov/articles/PMC3128527/
- **Distribution of phosphate in body fluid compartments** (Deranged Physiology) — free PO₄³⁻
  negligible at cytosolic pH; pKa 2.2/7.2/12.4; free Pi 1–10 mM. (Pin 3) https://derangedphysiology.com/main/cicm-primary-exam/body-fluids-and-electrolytes/Chapter-122/distribution-phosphate-body-fluid-compartments
- **Nano-organization of synaptic calcium signaling** (Portland Press) — 20–50 nm nanodomain
  Ca-sensor coupling. (Pin 1) https://portlandpress.com/biochemsoctrans/article/52/3/1459/234448/
- **Fisher 2015** (arXiv:1508.05929) — Posner/³¹P proposal; supersaturation "active niche".
  https://arxiv.org/pdf/1508.05929
- **PNAS 2025** — quantum effect in Li-doped ACP formation. https://www.pnas.org/doi/10.1073/pnas.2423211122
- **arXiv:2108.08822** — Posner dynamical ensemble not symmetric (supports Agarwal). https://arxiv.org/pdf/2108.08822
- **Tethered GECI / CaV2.2 nanodomain** (Nature Comms `ncomms1777`) — channel-mouth-to-sensor ≈ 5.5 nm. (Pin 1 / D11) https://www.nature.com/articles/ncomms1777
- **Eggermann, Bucurenciu, Goswami, Jonas** (Nat Rev Neurosci) — nanodomain Ca-channel↔sensor coupling, tens of nm. (Pin 1 / D11) https://www.nature.com/articles/nrn3125
- **Tang & Blanpied 2016** (Nature) — trans-synaptic nanocolumn alignment, tens of nm. (Pin 1 / D11) https://www.nature.com/articles/nature19058
- **Naraghi & Neher 1997** (J Neurosci 17:6961) — linearized buffer approximation; [Ca²⁺] at the channel mouth; 5–50 nm nanodomain validity. (Pin 1 / D11; the closed form A1/A2/A3 use) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6868209/
- **Garcia/Mancardi et al. 2019** (CaP PNC simulation) — ion-complex → dimeric prenucleation species; dimerization favorable to Ca/HPO₄ 1:2; "repetition of dimerization unlikely the only mechanism". (D15) https://pmc.ncbi.nlm.nih.gov/articles/PMC7011744/
- **Detection of Posner's clusters during CaP nucleation, MD** (J Mater Chem B, C7TB01199G) — Posner-like clusters assemble in ~0.5–1 ns at high supersaturation; no clean dimer-intermediate sequence. (D15) https://pubs.rsc.org/en/content/articlehtml/2017/tb/c7tb01199g
- **Agarwal, Kattnig, Aiello, Banerjee 2023** (J Phys Chem Lett 14:2518; arXiv:2210.14812) — Ca₆ dimer holds entanglement ~100s s, Ca₉ Posner trimer decoheres sub-second; SILENT on which forms. (D16) https://arxiv.org/abs/2210.14812

### In-repo provenance
- A2 probe: `sweep/supersaturation_gate_probe.py`. A3: `sweep/phosphate_conservation_probe.py`. A1: `sweep/nanodomain_closedform_probe.py`.
- Speciation source: `src/models/Model_6/atp_system.py:364-401`. Params: `model6_parameters.py:193,209-211,824`.
- Baseline commits: `a992ee7` (chemistry), `95990fd` (input-engine), `0ef0e0e` (calcium probe/PDE).
- Phase B commits: `f0aaffd` (A2+A3 probes), `49c7453` (B1 gate), `e9bb2c3` (log D1–D11), **`2ab02d8`** ("dimer working system" = B2a calcium grounding + D16 species fix + log D12–D17).
