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
