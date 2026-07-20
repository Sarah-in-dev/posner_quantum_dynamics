---
name: research-log-entanglement-topology
description: >
  Append-only research log + decision record for the Model 6 ENTANGLEMENT-TOPOLOGY
  sub-program (the entanglement partition as the computational primitive: the Werner
  distance-graph, the coherence-set radius, the far-pairs-first fragmentation test T1',
  and the retired SOC-topology power-law T1). The PRIMARY provenance record the paper
  draws from for the topology claims: every load-bearing claim carries a source and an
  epistemic-status tag, every decision carries its reason and date. The
  entanglement-topology-measurement and model6-entanglement-partition-werner skills distil
  the LOCKED decisions; this log carries the granular "why" and the evidence behind them.
  Sibling to RESEARCH_LOG_CALCIUM_DIMER.md (the chemistry sub-program). Read when
  continuing the topology work, writing it up, or reconstructing why a geometry, threshold,
  or verdict rule was chosen. APPEND newest entry at the top of the LOG; never rewrite
  history — supersede with a dated note.
---

# Research Log — Entanglement Topology (the computational primitive)

## Purpose & how to use this

This is the **decision-provenance record** for the entanglement-topology sub-program — the
half of Model 6 that asks *what the computation IS* (the partition of the entanglement
graph) as opposed to *what forms the dimers* (the calcium→dimer chemistry, logged
separately in `RESEARCH_LOG_CALCIUM_DIMER.md`). It exists so that when we write this up,
every number and every modeling choice traces to its source and the reason it was chosen,
and a future session can see *why* a decision was made, not just what it was.

- **Append, don't rewrite.** New work goes in a dated entry at the top of *The Log*. If a
  later finding overturns an earlier one, add a new entry that **supersedes** it with a
  pointer — leave the original in place (the paper needs the trail, including the wrong
  turns; this sub-program has several instructive ones).
- **Companion layers:** `entanglement-topology-measurement` (Appendix A is the current
  authority) and `model6-entanglement-partition-werner` carry the **distilled LOCKED
  decisions**; this log carries the **granular evidence and rationale**. When a decision
  locks, summarize it in the skill and point back here for the "why."
- **Epistemic-status tags** (same legend as the chemistry log):
  `[PROVEN]` literature/algebra-established · `[GROUNDED]` tied to a named measurement ·
  `[MODELED]` defensible choice not forced by physics · `[INFERRED]` follows from the model ·
  `[CONTESTED]` an unsettled bet · `[LOCKED]` settled, not relitigated without new physics.
- **Discipline (LOCKED):** emergent physics only — no constant tuned to a downstream target.
  The Werner bound 0.5 is a separability THEOREM; d*, T_eff, k_classical are not knobs. If
  the physics doesn't give the result, the log records the gap; it is not a license to slide
  a knob. **Score the ORDER, never the times** (T1' §3 below is the scar behind this).

## The epistemic frame (Sarah, LOCKED 2026-07-17)

> "We will never be able to model quantum in a silicon system, but we can model how the
> system expects to work."

This log's results are **not** claims about nature. A simulation has no privileged access
to whether the biology is quantum (`quantum-computation-and-attribution` §5, the attribution
gap: no experiment measures the quantum state *in the living computational system* AND
attributes the computation to it — attribution always routes through theory). The model is
the **theory of how the system expects to operate**, made realistic and — the load-bearing
part — made **DISCRIMINATING**: built so it predicts *differently* from its classical rival
on a measurement we can actually make (§6.3). T1' is exactly such a discrimination test.
A confirmed far-pairs-first cascade does not prove the biology is quantum; it shows the
*model's* memory carries spatial structure a classical scalar trace cannot, which is what
makes the model falsifiable and worth believing by convergence rather than by fit.

---

## DECISION RECORD (running summary — newest first)

| # | Date | Decision / finding | Status | Entry |
|---|------|--------------------|--------|-------|
| PO7-3 | 2026-07-19 | **THE MISSING REPRESENTATION WAS THE SPIN. Spin-resolved bonding built — and THE BLOB BREAKS: `largest_frac` 1.000 → 0.112, 1 component → 184, on a graph that is for the first time PHYSICALLY ADMISSIBLE.** The model bonded dimers as featureless nodes. A Ca₆(PO₄)₄ carries **four ³¹P spin-½ nuclei** (`quantum-system-canonical:43`), a singlet-strength bond consumes **one at each end**, and monogamy forbids a spin mediating two bonds. Nothing represented that. **Build** (opt-in `spin_resolved`, OFF ⇒ bit-identical `1034/369740/0.991922159684`): every dimer owns 4 slots; a bond must claim a FREE slot at both ends or it does not form; provenance bonds must claim their **NAMED inherited slot** (the which-spin tag), so two inheritances competing for one slot cannot both be satisfied; **degree ≤ 4 is DERIVED, never capped.** **Measured (fingerprint rig, 1 synapse, 200 steps, seed 31337):** edges **369,740 → 2,031** (0.55% retained); mean degree **715.16 → 3.93**; max degree **902 → 4**; dimers over bound **1034/1034 → 0**; components **1 → 184**; **`largest_frac` 1.000 → 0.112**; **491,566 bonds refused**. Refused bonds are **not lost edges — they are FRUSTRATION** (pairs individually satisfiable, jointly not), the H¹ obstruction the Unit-5 "sheaf" structurally could not express. **Also this cycle — ignition reproduced and characterised (Unit 8):** L·ETA-2's rig re-run with glutamate wired (the original `eta_probe.py` drives voltage-only — the ERR-2 defect); `E_invasion` **0.3508 vs 0.3518 (0.28%)**, peak r **2.53**, **7/7 synapses condensed**. **The pump is NOT dead** — the r≈0.077 figure repeated from the PO-7 kickoff was never measured and is superseded. **Pre-ignition the partition is 7 components / `largest_frac` 0.22** (one per synapse, exactly §5), collapsing to **`largest_frac` 1.000 on just 98 cross-bonds** — the percolation §3's Werner fix was meant to prevent, at F>0.5. **Ignition occurs at row-sums 4.14–5.05, well below L·ETA-1's stated requirement of >6.89** — that table was computed in the NMDAR-silent regime and its "unreachable at ≥2 µm at any N" claim needs re-deriving before it is cited again. | [GROUNDED, measured — build + positive] | `L·PO7-3` below |
| PO7-2 | 2026-07-19 | **⚠ CORRECTION — `L·PO7-1`'s CROSS-SYNAPSE PREMISE IS UNPHYSICAL AND IS WITHDRAWN. Phosphate provenance is a LOCAL mechanism; it is not a cross-synapse channel.** Sarah's correction, verified against `model6-entanglement-partition-werner` §2: the model's cross-synapse mechanism is `k_cross = K_ENTANGLE_EM_BASE(0.5)·√(η_i·η_j)·w_spatial·P_product`, hard-gated on both spines `mt_invaded`, with `w_spatial = exp(−d/5 µm)` the **CONDENSATE coupling length** — cross-synapse entanglement is mediated by the Fröhlich backbone, **not** by a phosphate traversing between synapses. Direct entanglement is only ever local. **Grounding failure:** PO-7 read §1 (*where* the partition lives) and treated it as licensing a mechanism only §2 describes; §2 was never read. The kickoff's premise ("a dimer in ANY synapse claims events in absolute network coordinates") was inherited without checking it against the mechanism. **WITHDRAWN:** the network-shared event pool as a physical claim; `L·PO7-1`'s framing of the 2 µm structural zero as a "landmine" (**it is CORRECT PHYSICS — phosphate should not reach**; PO-7 inverted it); the Unit-2 keystone verdict (tested a mechanism that should not exist); the Unit-7 claim-radius derivation (moot — within one 400 nm nanodomain the 500 nm reach was never limiting); Unit 6 (killed mid-run); the R4 advisor packet's core framing. **The tell PO-7 missed:** cross edges required forcing spacing to 0.2 µm, and PO-7 recorded that this "excludes the upper half of the physiological range" as a *limitation* rather than as evidence the mechanism was wrong. **WHAT SURVIVES — all local, all unaffected:** (1) **monogamy violation** — mean degree 715 vs the hard 4-spin bound, max 902, 100% of dimers over, **99.44% of edges physically inadmissible**, max admissible E=2068 vs actual 369740; (2) **provenance is monogamy-CLEAN** — max 1 bond per spin, 0/434 mediators over bound, where the clique rule violates 179×; (3) the **WHICH-SPIN slot tag** (Fisher names the phosphate slot, not just the partner) — more useful now, since local is the only scale that matters; (4) **coincidence window ≤50 ms** ⇒ `provenance_net_age_s = 2.0` is NON-OPERATIVE and needs no justification; (5) two real inherited defects fixed (frozen fidelity; dropped coherence death); (6) the Unit-5 sheaf is a **direct sum of 3 graph Laplacians** (cross-block edges 0/369740, identity verified), so it is not irreducible sheaf structure. **THE REAL FINDING THIS EXPOSES:** the charter sought cross-synapse edges *without* η because the pump is dead (r≈0.077, η=0). If entanglement is only local, **there is no η-free route** — so **η being dead is a HARD BLOCKER on cross-synapse entanglement, not an obstacle to engineer around.** The provenance network was getting the desired answer by changing the physics. The dead pump is the target. | [CORRECTION — supersedes `L·PO7-1`'s cross-synapse claims] | `L·PO7-2` below |
| PO7-1 | 2026-07-19 | **NETWORK-SHARED PROVENANCE BUILT AND VALIDATED (η-free cross-synapse edges are REAL) — but the multi-synapse §8 keystone is a PRE-REGISTERED NEGATIVE: decomposition null.** **Build:** lifted provenance events from per-synapse to a NETWORK pool (`multi_synapse_network._step_network_provenance`), opt-in, **off-path bit-identical `1034/369740/0.991922159684`** (re-gated after scoring). Repaired two defects in the inherited WIP: **frozen fidelity** (F stored at claim time, never refreshed, while `_find_all_clusters` tested it against the live Werner bound) and **dropped coherence death** (pruned on `is_entangled`, not `P_S<=0.5` — the U16 write-once shape again). **Unit 1 — the layer WORKS:** cross-synapse provenance edges form, η-free (`eta_cross = 0` in all 20 scored runs — the dead pump contributes nothing; every cross edge is Fisher-inherited). Registered geometric prediction held exactly: zero cross edges above 0.9 µm, nonzero below (grid span 400 nm vs reach 500 nm). **⚠ At the COMMITTED DEFAULT `spacing_um=2.0` cross edges are STRUCTURALLY IMPOSSIBLE** — the layer as inherited could never have produced one, and a run at defaults would have recorded a physical-looking null for a purely geometric reason. **Unit 1b — a hard CEILING:** cross edges appear once (t=0.25 s, 2 edges, 1/15 synapse pairs) then never again through t=2.0 s while `prov_total` grows 153→481 (+328 bonds, ZERO new cross edges). **Unit 2 (pre-registered, 5 seeds, 2 arms) — NEGATIVE, decomposition null:** `mean n_multi` = **0.60** (contiguous) / **0.50** (interleaved), both < 1 ⇒ the partition splits cleanly per-synapse. **Input-LOCATED, not input-COMPUTING.** Seed dominates the condition contrast (seed 7 high-yield in every arm; seeds flip direction between A and B) — the L·PO5-13 `d=0.02` shape by a different route. **⚠ THREE DESIGN DEFECTS FOUND IN MY OWN PREREG, all recorded not absorbed:** (1) dimer-level Q would have been ≈1.0 BY CONSTRUCTION (all ~370k intra-clique edges sit inside one activation label) — caught BEFORE scoring, moved to the synapse-level graph; (2) **"density matched by construction" is FALSE** — inactive synapses at −70 mV make no dimers, so cross edges can only form among active ones trivially; `Q_act = +0.0000` in **all 20 runs** (Newman's degenerate value when every edge lies in one community) and `d=1.61/1.33` is an artifact of the null going negative, NOT a signal — **a PASS on Q would have been a THIRD false positive**; (3) ARM 2 interleaving changes co-active spacing 0.2→0.4 µm, so criterion 3 conflates layout with distance. **What this does NOT close:** whether a density-matched, adequately-powered design would move the partition. §8 remains OPEN, not answered negative. | [GROUNDED, measured — mechanism YES, keystone NO] | `L·PO7-1` below |
| PO5-13 | 2026-07-19 | **PROVENANCE BONDING BUILT (first non-blob graph) — but the computation test is a FALSE POSITIVE the probe declared and PO-5 OVERRODE.** Build: Fisher's actual mechanism in `dimer_particles.py`, opt-in, **provenance-off bit-identical**; events at Ca-elevated cells (2 slots), dimer claims ≤2 nearest, bond iff shared; EM pathway skipped (LOCC). **Mechanism works:** 459,889-bond blob (off) → **~500 bonds, largest_frac 0.05** (on) — genuinely sparse/pairwise, the first non-blob in the whole investigation. **⚠ BUT the probe's verdict "carries input-dependent partition" is WRONG:** it scored `Q(input)=0.15 >> Q(shuffled)=0`, but the "input label" was spatial half — so Q detects **spatial locality (GEOMETRY, §8-insufficient)**, not input. **The decisive A-vs-B input contrast is d=0.02 — flat null.** Same criterion-mis-registration as Units 8/9/13. **Honest verdict: NOT keystone-supported.** The provenance assortativity is spatial, and single-synapse pulsed-vs-sustained doesn't vary the spatial Ca pattern (the peak-saturated weak-contrast wall again). Establishes: mechanism faithful + sparse + carries spatial structure. Does NOT establish either way whether input that genuinely varies spatial Ca would move the partition — and sharpens the tension that the pair-level channel may need the multi-synapse scale §7#1 said it doesn't. No constant tuned; d=0.02 holds regardless of event_rate. | [GROUNDED, measured — mechanism yes, keystone no] | `L·PO5-13` below |
| PO5-12 | 2026-07-19 | **DEEP RESEARCH: provenance-bonding is FAITHFUL to Fisher, computational capability UNPROVEN — de-risk before build.** External (adversarially-verified primary sources) + internal (code map). **GREEN:** Fisher's inheritance is strictly pairwise/provenance — two Posners entangle iff they share a common-origin phosphate pair; *"dimers bond iff they share a hydrolysis event"* is literally his channel, and the clique + distance-kernel rules are both wrong. **RED (fix both, 3-0):** (1) ⚠ the research said "fix stoichiometry to Ca₉(PO₄)₆" — WRONG, contradicts LOCKED `quantum-system-canonical:43` (Agarwal: the Ca₆(PO₄)₄ DIMER is the qubit, the Ca₉ trimer is inert). Correct object is the dimer, 4 spins = 2 singlet pairs ⇒ ≤2 events/node. Physics was already documented; the workflow mis-corrected it — PO-5 process failure; (2) genuine inter-Posner entanglement does NOT survive (Player & Hore: seconds; no inversion symmetry) ⇒ lean on the CLASSICAL common-cause reading. **THE GATE:** the object is exactly the s=1 Random Intersection Graph (threshold c=n·Σp_w², clique union, richer-than-density below m<n³) — but **NO source shows it carries INPUT-DEPENDENT partition structure**, which is the exact §8 property unproven for the replacement too. **Caveats:** "bond" = entanglement edge not chemical binding; a QDS binding-induced channel is orthogonal and invisible to a provenance graph. **Internal:** ~50–75 lines / 3 files, gated & bit-identical-off; ATP layer already computes the burst mask (`atp_system.py:111`) and discards it (`:119`) — capture not invent; one physics call = the birth↔event spatial-join rule. **DECISION: de-risk with an abstract RIG sim (minutes, no physics) BEFORE the build** — test input-dependent vs input-constant event assignment for partition-beyond-density; the research names this as the gate. | [external lit + internal map] | `L·PO5-12` below |
| PO5-11 | 2026-07-19 | **TOPOLOGY IS A FUNCTION OF DENSITY ALONE — §8's "scalar as computation", on a contrast that varies.** **Unit 13:** the fidelity cut (only lever that fragments) is **COSMETIC** — advisor's spatial covariate shows birth geometry identical between drive conditions (r_med d=0.00) and the partition follows (max d=0.54). Per-seed rows: SUSTAINED and PULSED are **BIT-IDENTICAL** in V/positions/topology — drive PATTERN never reaches the population (`peak_conc = np.max`, a peak not an integral, `:174`; write-once bonds never revisit). **Unit 14:** amplitude DOES vary V (per-seed [1101,1192]); fitting topology~V, every cut measure has residual scatter ≤ seed noise (comps_cut 1.26, lf_cut 1.23, H0_cut 1.00) ⇒ **the partition carries exactly what the scalar V carries.** **Correction owed:** Units 9/10/13 used the pattern contrast that Unit 13 shows doesn't reach the population, so their nulls were under-founded (Unit 9's sign-reversal now reads as a non-contrast); keystone conclusion unchanged, Unit 14 establishes it properly. **Convergence with L·PO5-10:** no input dimension reaches the topology as pair structure — the population is peak-saturated and write-once, the bond rules are input-blind proxies, because the model has no representation of the entangling origin. **Next step is the provenance build, not another test.** External + internal deep research commissioned. | [GROUNDED, measured — negative] | `L·PO5-11` below |
| PO5-10 | 2026-07-19 | **THE EM PATHWAY IS NOT AN ENTANGLING MECHANISM; the clique is NOT what percolates; the fidelity cut is the only lever.** **(1) STRUCTURAL/LOCC:** `em_rate` is a classical scalar scaling a *local* formation rate — a local operation, which cannot create entanglement (theorem). Entangling needs a direct `H_ij` or a common quantised mode; `em_rate` has neither. The code's own docstring already concedes there is no microscopic Hamiltonian and a ~10¹⁵ Hz vs ~Hz gap. **Dichotomy:** bonds are genuine entanglement ⇒ the pathway cannot make them; or they are classical correlations ⇒ the partition claim loses quantum grounding. **Asymmetry:** P0 (83%) has a real entangling origin (shared pyrophosphate) implemented as an unphysical clique; **P2 (17%) has no mechanism at all and is what percolates.** **(2) Unit 11:** degree cap k=4 removes **65% of edges**, `largest_frac` stays **1.0000**; k=1 also 1.0000; J-compat null (−0.8%); all-three −74% → still 0.9821. **Density is not the problem**, and this refutes the round-1 prediction that fixing the clique would fragment. **Per-event matching is unimplementable** — dimers are born from a concentration field, no per-phosphate provenance; the clique is a proxy for a **missing representation**, which is the diagnosis: Fisher's mechanism is inheritance, and every available proxy is input-blind by construction. **(3) Unit 12:** intra bonds have NO fidelity threshold (bare existence); storing `F=P_S_i·P_S_j·g(r)` and sweeping gives largest_frac 1.0000→**0.8314** and sheaf H⁰ **3→15** — the only lever that moves. **⚠ Two conceded defects: the bimodality is MANUFACTURED by the uncited 5 nm plateau, and `F` is a CATEGORY ERROR built from a rate term — fidelity must come from state, `F(t)=¼+(F₀−¼)e^(−t/T₂)`. No threshold nominated.** **(4) The ℝ⁶ latent space is populated with INPUT-INDEPENDENT NOISE** — `j_couplings_intra` is an independent per-dimer `normal(0.15,0.15,6)` draw (std 0.1478, median ‖ΔJ‖₂ 0.4864), so bonding on it cannot carry input information whatever the metric. The latent-dimension *principle* survives; this instance dies. **(5) OPEN:** dipolar coupling is a genuine parameter-free option (19.58 Hz @1nm … 0.02 Hz @10nm verified) but the claimed 1–2 nm range needs a ~100 ms competing timescale nobody can name — against T₂=216 s it reaches 10 nm, *more* permissive than today. **Do not build on it yet.** | [GROUNDED, measured + structural] | `L·PO5-10` below |
| PO5-9 | 2026-07-19 | **THE TRANSIENT HYPOTHESIS FAILS — and it falsifies a `PO5-7` claim written the same day.** Probe `sweep/po5_unit8_transient.py`, pre-registered, **no overrides** — only when we look. **P1 FALSIFIED: the field is `min = max = 22.095` at EVERY sample from t=0.01 s on.** Unit 7's `range [0.000, 22.095]` was one or two samples before the first physics step, **not a gradual rise**. **`PO5-7`'s "it starts at zero and climbs every run / transits the whole fragmented→blob range" is WITHDRAWN** — the field reaches operating value in ~2 timesteps and stays. Error shape: **reading a min/max range as a trajectory without checking the time series** — the same mistake as reading a mean without its spread, one level up. **P2 fired on the registered criterion (`components>1 AND largest_frac<0.99`) at t=0.01–0.02, but the criterion was TOO WEAK and PO-5 does not bank the pass:** those states are 10–14 components with `largest_frac` 0.962–0.983 — a giant component holding 96–98% plus crumbs, not a fragmented state. **Substantively this is P3: saturation precedes any structure.** A discriminating bar would have been `largest_frac < 0.5`, as Unit 7 used. **This CLOSES the most hopeful remaining possibility** — there is no early critical window; the graph is a blob from the first measurable instant at the native field. | [GROUNDED, measured — negative] | `L·PO5-9` below |
| PO5-8 | 2026-07-19 | **THE 100 ms BIRTH WINDOW IS NOT DERIVED — flagged sweepable 2026-05-29, never swept, and it decides the architecture.** Documentary, no compute. `model6-research-findings-may29:66`: *"birth_window value: 100ms hardcoded. Within Fisher's 1s budget. Conservative but defensible. **Tunable parameter — candidate for TALON sweep, not arbitrary calibration.**"* So its only grounding is an **upper** bound (~1 s, Fisher). **Unit 7's structure-preserving regime (2–10 ms) is inside the same permitted band**, so sweeping it is sanctioned, not tuning — and it was nominated for a sweep seven weeks ago and never done. **A parameter recorded as "conservative but defensible" turns out to decide whether §8's keystone can work at all** (P0 threshold ~2–10 ms vs native 100 ms = 10–50× above). **Mechanism gap, also already known and STILL UNCORRECTED:** `:64`/`:141` record that the docstring *"Phosphates from same pyrophosphate hydrolysis are born entangled"* **OVERCLAIMS — actual gate is spatial proximity to template field**; verified 2026-07-19 the docstring at `dimer_particles.py:234-236` is unchanged. Fisher entangles TWO phosphates from ONE molecule (a pair); the code entangles ALL template-bound dimers in a 100 ms window (a clique) — and the clique reading is what generates the 60–90-dimer groups that percolate the graph. **No value change proposed.** | [GROUNDED, documentary] | `L·PO5-8` below |
| PO5-7 | 2026-07-19 | **BOTH MECHANISMS PERCOLATE INDEPENDENTLY — `PO5-5` CORRECTED TWICE.** Pre-registered as a self-correction before the run; `birth_window` promoted to an attribute, **bit-identical verified**. **P1 CONFIRMED: `largest_frac ≥ 0.8725` at EVERY bus incl. 0 (zero P2 bonds), χ peaks at bus=0 and decays monotonically ⇒ the bus does NOT form the giant component. `PO5-5`'s "the BUS is a real percolation control parameter" is WRONG.** **The follow-up framing ("it only absorbs stragglers") is ALSO wrong:** at native bus with the birth window shrunk 50× — which fragments P0 to `largest_frac = 0.4145` alone — the field still gives `largest_frac = 1.0000`. **The field spans the graph BY ITSELF regardless of birth structure.** ⇒ **the system is DOUBLY supercritical; fragmenting requires reducing BOTH levers, which retires every single-lever fix including PO-5's own SOC/regulation proposal.** P0 threshold measured ~2–10 ms vs native 100 ms. **And the number PO-5 had been averaging away:** native field = mean 21.984, **std 1.558, range [0.000, 22.095]** — it **starts at zero and climbs every run**, so the system transits the whole fragmented→blob range each time and parks above it, while every measurement in this program sampled only the endpoint. | [GROUNDED, measured] | `L·PO5-7` below |
| PO5-6 | 2026-07-19 | **J-MISMATCH DISSOLUTION: NOT SUPPORTED — and the dissolution channel is INERT.** Pre-registered before the Tier-3 code change; opt-in flag, **flag-off verified bit-identical**. `(B−A)/spread = 0.00` at all three bus values; REAL vs OFF differ by 29–49 bonds out of ~300k (≈0.015%), identical components and sheaf H0. **Computed cause:** `k_disentangle = 0.01*(1-coh)/(1+protection) ≈ 9.95e-05 /s` with `coh≈0.98`; P(dissolve) over 30 s is 0.003 OFF vs 0.009 ON. **The graph is effectively WRITE-ONCE**, so nothing multiplying dissolution can matter — and this retro-explains `PO5-1`'s 0.944→0.606 saturation decline as **dimer death, not bond dissolution**. **Consequence: structural gating must act on FORMATION, not decay; PO-5 aimed at the wrong term.** **⚠ The scrambled control is CONFOUNDED — `np.random.permutation` draws from the global RNG and shifts the downstream stream, so arm C is a different realisation, not a matched control; its `(B−C)/spread` flips sign (+1.07, −0.71). Do not cite arm C.** The verdict rests on REAL vs OFF, a clean null. Per prereg: the mechanism is wrong, the scale is NOT re-registered, and the recommendation on file is to revert. | [GROUNDED, measured — negative] | `L·PO5-6` below |
| PO5-5 | 2026-07-19 | **[CORRECTED TWICE BY PO5-7 — the bus is NOT the percolation control parameter and does NOT merely absorb stragglers; see that row. The measured sweep numbers stand; the framing does not.]**  **THE BUS IS A REAL PERCOLATION CONTROL PARAMETER — components 60→1 — AND THE SYSTEM NATIVELY SITS PAST THE TRANSITION.** Probe `sweep/po5_unit4_bus_percolation.py`, predictions registered pre-run; raw JSON **and log committed**. Architecture verified in code: `model6_core.py:555` tryptophan → `collective_field_kT`; `dimer_particles.py:454` `em_rate = k_base*(collective_field_kT/reference_kT)*coh*g` — a single **global** scalar gain on every pair. **Positive control PASSED** (bus=0 ⇒ 0 P2 bonds ⇒ components_all == components_P0only). **Measured:** bus 0/0.1/0.25/0.5/1/2/5/10/20 → components **60/44/39/29/20/14/7/6/1**, monotone. **NATIVE bus = 21.98 kT** (vs `reference_kT`=20, `FIELD_THRESHOLD_KT`=20) → **1 component**. **(1)** A transition EXISTS — the architecture has an operating point where topology can hold information. **(2)** The system is parked **past** it: the informative band is bus ≈0.1–5 (7–44 components) and the model runs ~4×+ above it — **wrong phase, not broken mechanism.** **(3)** λ₂ is **exactly 0** in every fragmented state and only informative once connected (0.9106, 0.9807) ⇒ component-count and λ₂ are exactly complementary, and **the live path reads only `dim ker L₀` while sitting in the connected phase — reading the one channel provably empty at its own operating point.** λ₂ used as DIAGNOSTIC only, per A5. **NOT shown:** the condensate is not reconnected — this overrides the bus directly and says nothing about whether `backbone_eta*E_invasion` (`model6_core.py:543`, both factors 0.0000 in live trials) can reach it. **No §8 verdict.** | [GROUNDED, measured] | `L·PO5-5` below |
| PO5-4b | 2026-07-19 | **PO5-4 PROMOTED: the indifference-graph mechanism CONFIRMED 18/18 across 3 seeds × 2 arms** (`sweep/po5_unit3_birth_cohorts_results.json`, committed). Registered prediction `components(P0) = 1 + count(birth gaps > 0.1 s)` held at **every** sample. **AND ONE SINGLE-SEED CLAIM IN PO5-4 IS CORRECTED:** PO5-4 reported PULSED produced *no more* >100 ms gaps than SUSTAINED and flagged it as weak (2 samples). With 3 seeds the opposite holds — **PULSED max gap 3.5650 s / 6 gaps / 6 cohorts vs SUSTAINED 2.6700 s / 5 gaps / 4 cohorts.** **Drive pattern DOES modulate burst structure**, so formation can be gated finely enough to split the P0 graph — the channel §8 needs is alive, where the single-seed run suggested it might not be. The weak-signal flag on PO5-4 did its job. | [GROUNDED, measured] | `L·PO5-4` above |
| PO5-4 | 2026-07-19 | **THE P0 GRAPH IS AN INDIFFERENCE GRAPH ON BIRTH TIME — PREDICTED==MEASURED 5/5 — AND P2 IS WHAT ERASES THE STRUCTURE.** Pre-registered `docs/PREREG_PO5_UNIT3_BIRTH_COHORTS.md` (`becc8e3`, before the run); probe `sweep/po5_unit3_birth_cohorts.py`. **REPORTED not MO-VERIFIED — the run's log lived in gitignored `results/` and the worktree was removed by consolidation mid-run; the probe IS committed, so re-running promotes it.** **Registered prediction: components(P0) = 1 + count(birth gaps > 0.1 s). Held exactly at 5/5 samples** (SUSTAINED t=1/3/5 → 2/4/6; PULSED t=1/3 → 2/5), `max_glu = 1.000`. Mechanism confirmed from `dimer_particles.py:218-228` + `:210`: bond iff both template-bound and `|Δbirth| < 0.1 s`, and a whole `step_population` batch shares one birth_time ⇒ **unit-interval graph on the birth-time axis**, components = maximal runs with no >100 ms gap, no fitted quantity. **PO-5's OWN PROSE REASONING WAS FALSIFIED:** it predicted continuous births ⇒ one component; births are **bursty** — 12–17 distinct birth times in 5 s, gaps to **2.665 s**, ~60–90 dimers per event ⇒ **SIX P0 components under sustained drive.** **THE INVERSION:** the FULL graph measures `comps = 1, largest_frac = 1.000` (`L·PO5-1`) while P0-only has 6, and P0 is 82.86% of bonds vs P2's 17.14% (`L·PO5-2`) ⇒ **the temporal cohort structure EXISTS and the 17% spatially-mediated P2 bonds BRIDGE the cohorts into one blob.** The intra partition is trivial not because formation is structureless but because **a spatially promiscuous minority pathway erases a temporally structured majority.** **Weak, 2 samples:** PULSED produced no more >100 ms gaps than SUSTAINED (0.755 vs 0.770 s), hinting birth timing follows supersaturation dynamics rather than instantaneous drive — **NOT established.** **NO §8 VERDICT:** whether input modulates cohort structure, and whether it survives P2 bridging, is unrun; "P2 erases the structure" is NOT "§8 fails". **Do NOT weaken P2 to preserve cohorts** — that is tuning to an outcome (§7 LOCKED). | [GROUNDED, measured — REPORTED] | `L·PO5-4` below |
| PO5-3 | 2026-07-18 | **Q-B RAN 58.2 MIN ON THE EXCLUSIVE SLOT, ALL 9 RUNS, EVERY GATE PASSED — AND RETURNED NO VERDICT.** Pre-registered (A2.2–A2.6, all before the run); probe `sweep/po5_unit2_qb_selectivity.py`. **Gates all PASS:** instrument conservation; A2.3 `_remove_dimer` tripwire **zero calls** (confirming that defect is unreachable under a full protocol); positive control `max_glu > 0` every run (min 1.000); drive matching A=2.7540 vs B=2.7460 (**0.3%**, registered ≤5%). **Then scoring crashed** — `ValueError: shapes (169,) vs (36,)`. **`ratio` was never computed; NOTHING is claimed about §8 in either direction and the keystone stands exactly as unverified as before.** **Three flaws, all PO-5's:** (1) **the statistic was never comparable across runs** — cells indexed by *each run's own* occupied set, so index *i* meant a different physical location per run; Frobenius distance between them is meaningless **even when shapes coincide**, so had the counts matched this would have produced *a confident number that was silently garbage* — **the crash is the lucky outcome**; (2) the A2.6 pre-flight sampled **one** seed (13 cells) and certified an arm whose occupancy actually ranged **6–14**, with only 3 of 9 runs clearing `MIN_CELLS = 10` — the same single-sample-as-confirmation error PO-3 named when withdrawing F-3; (3) **the scored intermediate was not persisted, so a SCORING bug destroyed 58 minutes of PHYSICS** — and `sweep/score_leta5.py` had already solved this, was named as prior art in PO-5's own grounding brief, and was not composed from. **Fix, needing zero compute:** fixed **global** lattice (absolute cell coords), comparison on the all-run intersection, matrices persisted, scoring split into an offline scorer composed from `score_leta5.py`, validated on synthetic data before physics is spent. **NO registered threshold is being moved** — if the all-run intersection falls below `MIN_CELLS`, the honest verdict is *"the instrument cannot resolve pair structure in this geometry"*, per the registered hard stop. **Logged rather than swept because L·ETA-5 set the precedent:** a properly-conducted run that does not answer its question is a result about the instrument, and the next PO building a cross-run spatial statistic here needs to know per-run occupancy indexing does not survive the comparison step. | [GROUNDED, measured — null about the INSTRUMENT] | `L·PO5-3` below |
| PO5-2 | 2026-07-18 | **83% OF THE BOND SET COMES FROM A THIRD, DETERMINISTIC MECHANISM THAT NEITHER PATHWAY DECOMPOSITION NAMES.** Pre-registered `docs/PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md` §2; probe `sweep/po5_unit2_provenance.py`; **provenance recovered with ZERO edits to `dimer_particles.py`** (instance-level wrapping, four POs share this tree). Classification is EXACT not statistical — `:439` `p1 = both_ent & same_burst & both_tmpl & ~has_bond`, `:450` `p2 = both_ent & ~p1`, phases separated by wrapping `step_population` vs `step_entanglement`. **The instrument gate FAILED FIRST on real data** (orphans 0→909→4851); cause traced to `_remove_all_bonds_for_dimer` (`:245`) popping `_bond_lookup` without routing through `_remove_bond`; AMENDMENT A2.1, instrument fixed, physics untouched; failing run preserved. **Post-fix both gates PASS:** conservation exact (missing=0, orphan=0 vs 474256 live bonds) and instrumented-vs-uninstrumented **bit-for-bit identical** on n_dimers/n_entangled/n_bonds/mean P_S. **MEASURED at t=2.0s:** P0 birth-inheritance (`:218-228`) **392952 = 82.86%**, P1 burst **22 = 0.00%**, P2 EM **81282 = 17.14%**. **Two structural findings:** (1) the dominant site is the birth loop at `:218-228`, which is **deterministic — no rate, no RNG draw, no distance term** — bonding every template-bound dimer born within 100 ms unconditionally, i.e. a near-complete blob by construction; **83% of bonds never evaluate `em_rate` at all**, so Unit 1's `D = 33.5` spread in `g` applies only to the 17% minority, and the kickoff's `em_rate` decomposition (`mo-rescope-001.md:49-53`, `quantum-computation-and-attribution` §7 #1) describes the minority mechanism. (2) **P1 is shadowed by construction** — `p1` requires `~has_bond` and the birth loop has already bonded every same-burst template-bound pair, so P1 is near-dead code (22 bonds). **EXPLICITLY NOT CLAIMED:** that this defeats §8. Birth timing and template binding are downstream of input, so a deterministic birth rule is **not automatically input-blind**; whether it carries **pair-level** vs §8's **gate-level** information is Q-B, and **Q-B is unrun**. No keystone verdict is stated or implied — the inference `L·PO5-1` CORRECTION 1 withdrew is not repeated. **Latent defect routed, not fixed:** `_remove_dimer` (`:252-261`) discards from `entanglement_bonds` but never pops `_bond_lookup`; currently **dead code** (no call sites), so nothing is broken today. | [GROUNDED, measured] | `L·PO5-2` below |
| PO5-1 | 2026-07-18 | **[PARTLY SUPERSEDED — see CORRECTION 1 in `L·PO5-1`: the measurements stand, the "trivial partition" INFERENCE is withdrawn as wrong-layer, `quantum-system-canonical:139` LOCKS single-synapse one-giant-component as correct physics.]** **`g` IS LIVE — the 1/r³ is NOT inert, and BOTH standing predictions about it were wrong. But the graph it builds is a ~78%-complete SINGLE COMPONENT, so the pair-resolution in the RATE does not reach the TOPOLOGY.** Pre-registered `docs/PREREG_PO5_UNIT1_G_INERTNESS.md` (committed `cc80fcc` before the run); probe `src/models/Model_6/sweep/po5_unit1_g_inertness.py` (`1dbef17`); classifier demonstrated ABORTing on a deliberately broken threshold before it was allowed to score. Single synapse, -10 mV, 5 s, dt=0.005, 4 sample times. **Measured:** `f_sat = 0.176` (only 17.6% of pairs inside the 5 nm clamp, vs the ≥0.90 registered saturation bar), `r_p10/p50/p90 = 3.70/9.75/16.11 nm`, `r_max = 36.45`, `g_p10..p90 = 2.99e-2 .. 1.00`, **dynamic range `D = 33.5`**, stable to 3 decimals across all four samples. **Verdict `LIVE`-under-stated-conditions.** **Both priors refuted:** the board/kickoff (`board.md:919-922`, `mo-rescope-001.md:55-59`) predicted `g ≈ 1` **inert by saturation** — no, only 17.6% clamp; PO-5's own grounding brief predicted `g ≈ 3.7e-5` **inert by vanishing** off the 400 nm birth domain — no, dimers cluster at templates and sit ~10 nm apart, so the brief's a-priori was wrong by ~15× in `r` and is recorded as such. `model6-entanglement-partition-werner:60`'s *"intra edges at ~7 nm"* is the prose that was RIGHT (`r_p10 = 3.70`, `r_p50 = 9.75`). **THE CONSEQUENCE, which is the finding:** realised intra bond saturation is **0.75–0.83** and the corroborating probe (`sweep/observe_pathway2_selectivity.py`) reads **`comps = 1`, `largest_frac = 1.000`** at t=5 s and t=10 s, with bonded-pair median separation **9.5 nm vs all-pair 10.3 nm** — i.e. the bonded set is barely distinguishable from the all-pairs set. **A rate that varies 33× across pairs is producing a near-complete graph with a trivial partition.** Since the computation IS the partition (`model6-entanglement-partition-werner`, LOCKED), pair-resolution in `em_rate` that does not survive into the component structure buys the keystone nothing. **NOT YET ATTRIBUTED — UNVERIFIED:** whether the saturation is Pathway 1 (birth entanglement, `dimer_particles.py:218-228`, which bonded 94.4% of pairs at the very first sample) or Pathway 2. That separation is PO-5 Unit 2 and no claim is made on it here. **`g` is GEOMETRY, not input — this unit does NOT advance §8's keystone**, it establishes that the later pair-level test is not operating on a constant. | [GROUNDED, measured] | `L·PO5-1` below |
| AUDIT-1 | 2026-07-18 | **SUBSTRATE AUDIT — full adversarial code audit, `docs/SUBSTRATE_AUDIT_JUL18.md`.** Four parallel read-only agents, `file:line` required for every claim, UNVERIFIED where code could not confirm. **Five headline findings:** (1) **factor-of-2pi error** on the per-synapse pump — `vibrational_cascade_module.py:315` uses `hbar*f` on a LINEAR frequency, n_bar inflated **6.28x**; the backbone pump is CORRECT (`h*f`, `model6_parameters.py:46`). (2) **The calibration fiction survives and is unsweepable** — `kT_ref = 22.1` is a function-body literal (`:246`), invisible to the params dataclass and to sweep_runner; with `r_at_E_ref = 100e9` it makes **r/r_c ~ 1.045 at MT+ an arithmetic identity, not a result**. (3) **Three docstrings assert mechanisms absent from the code** — a Hill function (`multi_synapse_network.py:1332-1334` vs `:1381-1392`), a 30% collapse (`:1423-1425`, `collapse_factor` never read), and "No fitted parameters!" (`:1238-1242`) beside two fitted parameters. (4) **Cited sources contradict their values** — phi/chi cite Zhang 2019 which gives 6 GHz / 0.07 GHz; code uses 10 GHz / 0.05 GHz. (5) **The two pump sites run different threshold physics** — backbone `n_ex = n_bar_s`, per-synapse still Zhang Eq. 4. **WHAT SURVIVES:** the entanglement/partition layer — Werner 0.5 is a THEOREM not a cutoff, eta is exactly `(r-1)/(r+1)` with no fitted curve, commitment is a real CaMKII integrator with a genuine DDSC delay. **Debt REGRESSED:** ~151 dead parameter fields (was ~120), six orphan modules, none removed. Also found: `phosphate_total` goes stale so J-coupling reads a field ignoring dimer consumption; ATP<->Pi is not mass-conserving; `step_with_coordination` and `run_place_field_learning` still form ZERO cross-synapse bonds (a gap in the same-day fix). | [GROUNDED, code SHOWN] | `docs/SUBSTRATE_AUDIT_JUL18.md` |
| ETA-6 | 2026-07-18 | **EQUIVOCAL — the magnitude question is NOT answered, for two independent reasons.** Pre-registered (`docs/PREREG_L_ETA_6_NMDAR_MAGNITUDE.md`, thresholds anchored to Jain 2024's own no-glutamate control, 7/56.3 = 0.124, fixed BEFORE measuring). Audit of L·ETA-4's conditions; its probe was not modified and its verdict not re-derived. **(1) The scored condition (plateau ON) was NOT MEASURED** — those arms cost >10x the plateau-OFF arms (>12 min without reaching step 400/2400, zero progress in a 90 s window) and were killed per the compute cap. **(2) One registered criterion is UNSOUND:** §3 differenced a calcium PEAK across two INDEPENDENT stochastic arms — an extreme-value statistic with no sign guarantee — and it duly returned **dCa peak = -14.65 uM**, i.e. blocking NMDAR 'raised' calcium. Defect in the criterion, not the model. **MEASURED and SOUND, but plateau OFF and therefore NOT the scored condition:** NMDAR charge ratio **R = 0.0147** (silent/driven), below the negligible threshold 0.05; and dCa **mean** **+0.51 uM**, positive as physics requires and ~half of `K_calcium_poly` = 1.0 uM, so **not obviously small** — reported with status, not scored. The mean was NOT substituted for the peak and rescored: swapping a criterion after seeing the registered one give an uncomfortable answer is the goalpost move the discipline prevents. **So: L·ETA-4's -0.0019 is still UNSUPPORTED (L·ETA-5 rotation-001) and NOT YET CONTRADICTED — unsupported and wrong remain unseparated,** which is precisely what Sarah's PO-5 decision was waiting on. **Also verified in code this cycle:** `k_cross = K_ENTANGLE_EM_BASE * eta_factor * w_spatial * P_product` (`multi_synapse_network.py:340-341`) — `P_product` is a **multiplicative co-factor with eta, not an alternative**, so eta = 0 zeroes `k_cross` whatever `P_product` does. That makes L·ETA-5's zero-cross-edges result structurally necessary rather than incidental. **Cost for the MO to sequence: the plateau-ON pair, >10x the ~65 s/arm plateau-OFF cost, cause = O(n^2) entanglement growth once the plateau drives dimer formation across all 7 synapses.** | [GROUNDED, partial] | L·ETA-6 |
| ETA-5 | 2026-07-18 | **THE RATCHET TEST IS VOID — ITS NULL ARM IS THE RESULT: `E_invasion` accumulates WITHOUT ACTIVATION.** Pre-registered (`docs/PREREG_L_ETA_5_RATCHET.md`), 8 traversals x 14 s, 20 s gaps, real physics through every gap (`analytical_gap` deliberately NOT called). **Scored verdict `INCONCLUSIVE — NULL ARM RATCHETED`: the dispatch question is NOT ANSWERED.** The null held the target below the 0.05 activation floor but did NOT suppress presynaptic release (`BASELINE_RATE_HZ = 0.5`, `presynaptic_release.py:124`) — so the 'silent' synapse still received glutamate. **null max `E_invasion` = 0.4507** (registered: must stay 0.0000) and **null `peak_r` gain = 7.46x** — *larger than the driven arm's 5.65x*. **The finding that survives is about the driver itself:** `E_invasion` climbs past `invasion_threshold` on tonic spontaneous release alone, growing even during silent gaps (`rho` up to 2.26), and the driven/undriven separation COLLAPSES with traversal count (6.15x -> 1.70x). This is a stronger, plateau-free version of L·ETA-4: selectivity in the `E_invasion -> r -> eta` channel is weak on this driver's own dynamics. **Driven `r` DID cross threshold** (1.0721 at t3, 1.4050 at t8 — first live-regime crossing after L·ETA-3) but **cross-synapse edges = 0 in BOTH arms**, since `k_cross` ~ sqrt(eta_i*eta_j) and only one feature was driven: **eta != 0 shown, a PARTITION was NOT.** PO-3 -> PO-5 at most PARTIALLY cleared. Not the negative branch (FALSIFIED needed gain < 1.2; measured 5.65x). Even absent the null failure the drive arm would not have CONFIRMED: `peak_r` non-monotone (t3 1.0721 -> t4 0.9911) and `ratio_mean` 1.1080 outside the registered [0.89, 1.07], the calcium-tail overshoot flagged in AMENDMENT A1.2 BEFORE the run. **Three PO-3 errors corrected in-cycle:** the 'frozen gap' claim (wrong — 1 ms/gap, caught by PO-4), the committed-branch retention derivation (wrong — committed spines drain 3.54x FASTER, caught by PO-4, fixed before scoring), and F-3's '~100x NMDAR starvation' (**overstated and inverted** — measured 19x on events, but the old pattern HOLDS each release 100 steps so it delivers MORE exposure; **the L·ETA-3 correction-banner recommendation is WITHDRAWN**). Re-run needs a null suppressing spontaneous release and a gap clearing the calcium tail — protocol changes, not made unilaterally. | [GROUNDED, measured] | L·ETA-5 |
| ETA-4 | 2026-07-18 | **THE PLATEAU MAKES THE CONDENSATION DRIVE BRANCH-GLOBAL — §8's premise FAILS as written.** Probe `sweep/plateau_vgcc_leak_probe.py`, 7 synapses @1um, ONLY synapse 3 driven, rest silent, with/without plateau. **Selectivity survives in NMDAR exactly as Jain 2024 requires** (silent-synapse NMDAR gain from plateau **-0.0019**, i.e. zero — no glutamate, no current, however depolarized). **But it is destroyed in the VGCC->E_invasion->r channel:** silent-synapse VGCC open fraction **0.0017 -> 0.4783 (+0.492)**, and that propagates — **E_invasion silent 0.0000 -> 0.2115, IDENTICAL to the driven synapse's 0.2115 to four decimals**. `r` silent **0.754-0.822** vs driven **0.812**: the driven synapse is **NOT SEPARABLE** from the silent ones. When r crosses 1 they cross TOGETHER. **VERDICT-LOGIC CORRECTION (important):** the probe's first auto-verdict printed "eta stays SELECTIVE" because eta==0 at silent synapses — but eta==0 at the DRIVEN synapse too (r=0.812<1), so the test was VACUOUS, the 683b82f failure class exactly. Logic corrected to return INCONCLUSIVE-on-eta and to read the DRIVE channel instead. **Consequence:** eta cannot carry input-selectivity once a plateau is present; selectivity must live in `P_product` (the dimer population, which forms only where NMDAR calcium arrived). That is a coherent story but NOT the one §8 is written against — §8 assumes drive patterns the partition THROUGH eta. | [GROUNDED, measured] | L·ETA-4 |
| ETA-3 | 2026-07-18 | **eta does NOT clear in a LIVE trial — the pump is capable, the PROTOCOL is not.** L·ETA-2's flagged precondition, measured and negative (`sweep/eta_in_live_trial.py`). Real spatial-discovery trial, input engine fully wired: **max r = 0.0768 vs threshold 1.0 (13x short), eta = 0.0000, zero synapses condensed, ZERO cross-synapse edges all trial.** Attribution — both factors of `r ∝ E_invasion x ca_open` fall short roughly multiplicatively: **E_invasion 0.0868 (rig 0.35, 4x short)** and **ca_open 0.140 (rig 0.38, 2.7x short)**. Mechanism read off the trace: E_invasion is exactly 0 for the first ~34 s then climbs 0.011->0.052->0.075 and is STILL RISING at trial end — it is the actin integrator and needs sustained activity (~45-60 s per model6-actin-invasion-driver), but a navigating agent gives each feature only a brief transient; and only 1-4 of 12 features are active at once with max_act mostly <0.35, so the -70+act*30 mV map leaves B(V) negligible. **The constraint is DWELL and CO-ACTIVATION, not physics.** Levers (Sarah's call): slower agent, denser/clustered features so neighbours co-drive the aggregation, longer trials, or the plateau carrying the depolarization. **DO NOT raise the synaptic voltage** — the -40 mV cap deliberately keeps the plateau out of the synaptic knob. §8 stays blocked IN PRACTICE, for a better-understood reason than ETA-1 gave. | [GROUNDED, measured] | L·ETA-3 |
| ETA-2 | 2026-07-18 | **THE PUMP IGNITES — L·ETA-1 SUPERSEDED.** Input engine finished (glutamate into every learning driver; plateau_potential wired), eta re-measured on the same rig: NMDAR open fraction **0.0000 -> 0.3806**, **r 0.3509 -> 1.6234** (4.63x, threshold is 1.0), **eta 0.0000 -> 0.2376**. Condensation fires. The gain is entirely `ca_open`, not `E_invasion` (which moved 1.02x): the NMDAR half of the 25/25 population is no longer structurally shut, and 0.3806 is almost exactly the 0.33 equilibrium `model6-input-engine` predicts from alpha/beta at saturating glutamate. **Nothing tuned.** Bonus check: the naturally-driven eta lands within 9% of the **0.26 the topology probes CLAMP to** — the imposed stand-in matches a driven value, a retroactive check on T1's operating point. **WIRE 3: DDSC fires for the first time ever** — triggered False without plateau, True with it; the >=20 kT field gate is not binding. **LIMITS:** characterization rig at act=1.0 sustained, 7 synapses @1um — NOT a live-trial measurement, and whether eta clears threshold under spatial-discovery's Gaussian activations is UNMEASURED. Plateau duration is MODELED/PROVISIONAL (0.3 s, ungrounded). T1' probes deliberately NOT wired (would invalidate a closed result — open decision). | [GROUNDED, measured] | L·ETA-2 |
| ERR-2 | 2026-07-18 | **CORRECTION — ETA-1 overstates its conclusion; its measurements were taken with NMDARs SILENT.** `eta_probe.py:68` drives voltage-only (`{"voltage": v, "reward": False}`); `analytical_calcium_system` defaults `glutamate=0.0`, so only 25 of 50 channels conducted. `model6-input-engine` documents this as a KNOWN OPEN INTEGRATION GAP and predicted the symptom verbatim — *"will show reduced calcium and lower E_invasion ... NOT a regression. Do not read it as one"* — and states that `get_open_fraction()` IS the `ca_open_fraction` feeding `compute_metabolic_power`. So the term setting `r` was measured with its glutamate contingency unsatisfied. **SURVIVES:** the arithmetic, the `d*`/row-sum geometry, the duty-cycle analysis, and the identity that eta=0 ⇒ no cross-synapse bonds. **DOES NOT SURVIVE:** "the pump does not ignite" as a claim about the model — measured `r` is a **LOWER BOUND**, and ETA-1's three-way fork is premature because it presumes a complete input path. **Grounding failure:** `model6-input-engine` was not read before dispatching the measurement; it owns the input path and carries a "READ THIS before any full-model run" section naming this exact misreading. Supersede with L·ETA-2 after the integration is finished and eta re-measured. | [GROUNDED, code SHOWN] | L·ETA-1 (banner) |
| ETA-1 | 2026-07-18 | **[SEE ERR-2 — CONCLUSION CORRECTED]** ~~THE PUMP DOES NOT IGNITE — the cross-synapse partition exists only under an IMPOSED η, never a driven one.** Measured across 4 conditions (`sweep/loop_audit_2026_07_18/eta_probe.py`): `eta = 0` under rest, under the spatial-discovery −40 mV drive, and under the −10 mV theta burst **with the clamp removed**. `r = p_met_agg/P_c` floors at **0.0390** and peaks at **0.1409** after 30 s of sustained drive; extrapolating `E_invasion` to its ceiling gives `r ≈ 0.32`, still **3× below** the `r ≥ 1` condensation threshold. Cause: `ca_open` is the binding constraint and measures **0.063** duty-averaged over the real theta protocol, not the `≈1` assumed at `soc_pump_threshold_stage1.py:88` — a ~10× overstatement, and why that falsifier never fired. **Consequence:** `k_cross ∝ sqrt(eta_i·eta_j)`, so at eta=0 **zero cross-synapse bonds form** — the clamp in every topology probe is what creates the partition, not merely what holds it steady. **T1′ is untouched** (it clamps by declaration, scores order only, and the Werner algebra is independent) but its LICENSE is narrowed: "the topology IS the eligibility trace **in a running network**" is unsupported — in every end-to-end run the topology is empty. **NOT FIXED BY TUNING:** `P_c`(ω₀,Q) and `p_active_max_W` were left alone deliberately; moving them is the §6.1 emergent-physics failure mode. Open fork: wrong drive protocol vs condensation being intrinsically a population phenomenon (~10–20 synapses at ≤1 µm; unreachable at ≥2 µm at any N). **Blocks the §8 input-selectivity phase**, whose "vary only the drive" constraint is currently unsatisfiable. | [GROUNDED, 4 conditions] | L·ETA-1 |
| T1'-6 | 2026-07-18 | **CHANNEL SEPARATION — the population confound is measured INERT; §6's conclusion stands, §6's argument is replaced.** Adversarial review charged that dimer loss feeds the same extreme-value statistic that sets `d*_eff`, so it produces far-pairs-first too and replication separates nothing. Pre-registered 4 arms (+2 counterfactual) BEFORE running (`abdb549`). Result: **arm A ≡ arm B bit-identically** (`max|A−B| = 0.000e+00` on every `d*_eff`/`max_pair` sample, 4/4 seeds) and **arm C produces ZERO breaks** while the population falls 2223→~50, retaining **100.0000%** of `max_pair(P_S²)`. Cause, from code: `dimer_particles.py:230-241` removes **lowest-`P_S`-first** (`coherence` is an affine increasing map of `P_S`, :57-62), so attrition is rank-selective and the argmax is the LAST thing removed. The charge additionally fails **under its own assumption**: arm C_rand (uniform random removal) also gives zero breaks — `Δd*_eff = −0.002 µm` against a 1.35 µm span — because `P_S(0)` is packed at its ceiling (median 0.9986, max 1.0), leaving no room below the max. Null arm D clean. **Where it does bite:** arm A_rand (random removal + decay) confirms 4/4 with systematically earlier breaks — a model that removed randomly *would* have a live population channel. **SECONDARY, damaging:** order-recovery power measured **37/40 ≈ 92%** across noise draws (seed 0 = 7/10), not the **10/10** §4 cites — different estimators (this run applies the probe's `CONSECUTIVE_ABSENT=3` guard; the power probe used unguarded first-crossing). Headline unaffected: `p≈3.0×10⁻⁶` is against the classical null, which power does not move, and 4/4 at 92% power has probability 0.72. | [GROUNDED, 4 seeds × 6 arms] | L·T1'-6 |
| ERR-1 | 2026-07-17 | **CORRECTION — three P_S_crit annotations were arithmetically wrong when first committed.** The wide ladder's critical-coherence values were logged as 2.90µm→0.9327, 2.45→0.8801, 2.00→0.8305. Correct values (`P_S_crit = sqrt(0.5·exp(gap/λ))`, λ=5µm) are **0.9450 / 0.9034 / 0.8637**; 3.35µm→0.9885 was right. Verified against the Jul-16 ladder's values, which were correct (3.0→0.9545, 2.8→0.9356, 2.5→0.9079, 2.0→0.8637 all reproduce exactly). **The T1' result is UNAFFECTED**: these are descriptive annotations only — the probe never reads them, it computes d* at runtime from measured P_S and compares gaps directly, so the 4/4 cascade used the real geometry and the real Werner threshold. Caught when Sarah asked for a from-scratch explanation of the mathematics and the derivation was re-done by hand. Fixed in `sweep/coherence_fragmentation_probe.py` and in T1'-3/L·T1'-3 below. | [GROUNDED, corrected] | — |
| T1'-5 | 2026-07-17 | **T1' REPLICATION COMPLETE — far-pairs-first CONFIRMED across 4/4 independent seeds; CONCLUSIVE.** Seeds 0,1,2,3 (wide ladder, 90 s, dt=1e-3; seeds 1–3 run in parallel). ALL FOUR broke in the exact pre-registered order **3.35 > 2.90 > 2.45 > 2.00**. Under the classical null (no spatial structure ⇒ break order is a uniformly random permutation of 4 ⇒ 1/24 per seed), 4/4 in order ⇒ **p = (1/24)⁴ ≈ 3.0×10⁻⁶**. Break TIMES scatter seed-to-seed (gap 2.90 broke at 32.5/37.0/42.0/32.5 s; gap 2.45 at 61.5/55.0/54.5/64.5 s) while the ORDER is invariant — vindicating the score-order-not-times decision (L·T1'-2). Replication also DEFEATS the population-collapse confound on the late breaks: uniform dissolution lowers every pair's radius together (edges would die ~simultaneously), so it CANNOT manufacture a consistent gap-spaced order across independent seeds — only the coherence/distance mechanism does. Altitude unchanged: **(A)** — the model's partition carries spatial structure a classical scalar eligibility trace cannot; NOT a claim about quantumness (attribution gap stands; see the epistemic frame). Completes L·T1'-4. | [GROUNDED, 4 seeds] | L·T1'-5 |
| T1'-4 | 2026-07-17 | **T1' DYNAMIC — far-pairs-first fragmentation CONFIRMED, seed 0 (single seed).** → *completed by T1'-5 (4/4 replication); kept as the trail — the moment the result was one seed and only suggestive.* Wide ladder `[3.35,4.5,2.90,4.5,2.45,4.5,2.00]µm`, 90 s silence, dt=1e-3. All 4 live edges broke in the exact pre-registered gap order — 3.35µm@14.5s, 2.90@32.5s, 2.45@61.5s, 2.00@78.0s — verdict `far-pairs-first order CONFIRMED over 4 breaks`. This is the DISCRIMINATING result: a classical scalar eligibility trace decays uniformly and cannot produce a spacing-ordered cascade. **Single seed ⇒ p≈1/24≈0.042 vs the classical null — suggestive, NOT conclusive** (a 2nd independent seed in order → p≈0.0017). Breaks 1–2 landed at HEALTHY population (1843, 1043 dimers) — clean, coherence-driven; breaks 3–4 at CRATERED population (259, 98) — TIMING confounded by dimer-loss, ORDER preserved (uniform dissolution lowers every pair's radius equally). Guards worked live: 2 flickers (gap3.35, gap2.45) correctly rejected, not scored. Altitude: **(A)** — the partition carries SPATIAL structure; says nothing about **(B)**/quantumness. Ran 2.7 h not the projected ~8 h (O(n²) tracker cost collapsed with the population). `sweep/coherence_fragmentation_probe.py`; log session-scoped. | [GROUNDED, single seed] | L·T1'-4 |
| T1'-3 | 2026-07-17 | **The trustworthy rebuild — geometry chosen for POWER, not for "early breaks".** The Jul-16 redesign premise ("gaps just under d*(0) so the cascade lands early") was FALSIFIED, measured not argued: edge survival is governed by the P_S TAIL (max over bonded pairs), which decays ~3.6× slower than the median — an 8 s null showed the median radius below all 4 gaps while all 4 edges lived. Worse, each synapse's tail is set by its own frozen-at-creation `T_eff`, giving ~0.29µm between-synapse d*_eff scatter, so rungs finer than that are decided by luck. **Measured order-test power (10 seeds via the validated d*_eff replay): tight 0.10µm rungs 6/10, medium 0.25µm 5/10, WIDE 0.45µm 10/10.** Geometry set to the wide ladder — 0.45µm rungs clear the scatter. Cost: cascade lands later (~90 s), accepted for a resolvable order. | [GROUNDED, 10-seed replay] | L·T1'-3 |
| T1'-2 | 2026-07-17 | **d*(0) MEASURED = 3.4521µm (median), NOT assumed.** Rig = the confirmed static-probe rig, t=0.08 s, n≈2200. P_S median 0.9987, min 0.9922, max 1.0 ⇒ d* min 3.387 / median 3.452 / max 3.466 µm. The SPREAD is the load-bearing fact (two prior wrong claims came from assuming P_S). d*_eff replay (`sweep/dstar_eff_replay.py`) VALIDATED against the real rig (median P_S matches to <0.0005); its P_S→d*_eff mapping is UNVALIDATED by construction (no honest cascade datum — the only candidate, "gap 3.0 broke @34 s", is the retracted Jul-16 FLICKER) ⇒ it emits an UPPER BOUND on break times, never a prediction to score. | [GROUNDED] | L·T1'-2 |
| dt-1 | 2026-07-17 | **dt convergence — the order test is honest at production dt, and the operating point survives.** P_S and the Werner edge set are dt-CONVERGED (`d*_med=3.45`, `edges=5` at every dt from 1e-4 to 5e-3), so the ORDER test reads converged quantities at dt=1e-3. The dimer COUNT is NOT converged in the drive transient (~+38% at dt=1e-3 vs 1e-4; dt=1e-2 OVERFLOWS — do not use) — explicit-Euler error on stiff formation, NOT a new bug. BUT the saturated OPERATING POINT is converged to ~5% (plateau 156.7µM @dt=5e-4 vs 163.9 @dt=1e-3): at saturation formation balances the correctly-dt-scaled dissolution, so the transient inflation cancels. **The ~155µM operating point is NOT a dt artifact.** (Corrects an in-session over-extrapolation of the 38% to the operating point.) Does NOT reopen D8/D14/D16/D17 on dt grounds. | [GROUNDED] | L·dt-1 |
| T1'-1 | 2026-07-16 | **T1' STATIC HALF — the Werner cut IS a coherence-set distance rule.** Algebra from live code: `F = P_S_i·P_S_j·w`, `w=exp(-d/5µm)`, edge iff `F>0.5` ⇒ **bond iff `d < d* = 5·ln(P_product/0.5)`**. Two theorems fall out, neither tuned: (1) hard coherence floor `P_S>1/√2≈0.7071` (since `w≤1 ⇒ F≤P_S²`); (2) radius SHRINKS as coherence decays ⇒ partition must fragment far-pairs-first. CONFIRMED pre-registered (`sweep/coherence_radius_probe.py`, ~43 s): 8-synapse ladder called 7/7 gaps, exact edge list, betti0_cross=3, sizes=[3,3,2], betti1=0. **Retrodicts the Stage 3 chain/ring validation with NO free parameters.** This GENERATES the T1' dynamic prediction. | [PROVEN algebra + GROUNDED probe] | L·T1'-1 |
| T1-RET | 2026-07-16 | **T1 (SOC-topology power-law) RETIRED — structural, ~0 compute, NOT a negative result.** (1) 1D forbids it: the honest dendrite is 1D (`_generate_positions 'linear'`); in 1D clump size is exactly geometric `P(k)=p^(k-1)(1-p)` — exponential tail (geometric fit R²=0.97–0.996 at every density, beats power-law every time). (2) D18 forbids it: nucleation is all-or-none bistable with a quantal, drive-independent ON amplitude (~135) — a characteristic SCALE, where SOC needs scale-FREEDOM. (3) Two "SOC" claims were conflated: SOC-*chemistry* (phosphate depletion→S→1, D8/D14) STANDS; SOC-*topology* (power-law clumps) inherited the name and is what's retired. | [GROUNDED structural] | L·T1-RET |

---

## THE LOG (newest first)

### L·PO7-3 — the missing representation was the SPIN; spin-resolved bonding breaks the blob · 2026-07-19 `[GROUNDED, measured — build + positive]`

**Advisor R4 named the object; this builds it.** Prior units in this cycle characterised the
*existing* graph; this one replaces the rule that produced it.

#### 1. What was missing

The model bonded dimers as **featureless nodes**. But a Ca₆(PO₄)₄ dimer carries **four ³¹P
spin-½ nuclei** (`quantum-system-canonical:43`, LOCKED); a singlet-strength bond consumes
**one spin at each endpoint**; and **monogamy of entanglement** forbids a spin that is
maximally entangled with one partner from also entangling another. None of that was
represented, so the graph reached mean degree **715** against a hard bound of **4** — with
**99.44% of edges physically inadmissible** (`L·PO7-2` §2, measured).

The advisor's framing, adopted: the six J-couplings are **not** the stalk (that was Unit 5's
category error) — they are the intra-dimer Hamiltonian *acting on* a stalk whose real content
is the spin state. Restriction to a bond is a **partial trace onto the mediating spin**, which
mixes coordinates — which is what a genuine sheaf requires and what a coordinate projection
can never give.

#### 2. The build

`dimer_particles.py`, opt-in `spin_resolved`, **OFF ⇒ bit-identical
`1034 / 369740 / 0.991922159684`** (gated before and after):

- every dimer owns **4 spin slots**;
- `_create_bond` must claim a **free** slot at **both** endpoints, or the bond does not form;
- provenance-inherited bonds must claim their **NAMED** slot (Unit 4's which-spin tag) — the
  inherited nucleus sits in a specific slot, so two inheritances competing for one slot
  **cannot both be satisfied**;
- **max degree 4 is DERIVED from the molecule, not imposed as a cap.**

#### 3. The measurement — the blob breaks

Standing fingerprint rig: 1 synapse, 200 steps, seed 31337, MT invaded.

| | spin_resolved OFF | ON |
|---|---|---|
| edges | 369,740 | **2,031** (0.55% retained) |
| mean degree | 715.16 | **3.93** |
| max degree | 902 | **4** |
| dimers over bound | 1034 (100%) | **0** |
| components | 1 | **184** |
| **largest_frac** | **1.000** | **0.112** |
| frustrated bonds | — | **491,566** |

**The first physically admissible entanglement graph in this investigation, and it carries
non-trivial partition structure** where the inadmissible graph was one component containing
every dimer.

**The 491,566 refused bonds are the physics, not waste.** They are **frustration**: pairs each
individually satisfiable, jointly not. That is precisely the **H¹ obstruction** the Unit-5
construction could not express — `L·PO7-2` §6 verified that construction decomposes into **3
ordinary graph Laplacians** (cross-block edges **0/369,740**), because coordinate-projection
restrictions are diagonal.

#### 4. Unit 8 — ignition reproduced, and the pump is NOT dead

L·ETA-2's rig re-run. **The original `eta_probe.py` drives `{"voltage": v, "reward": False}` —
voltage only, the ERR-2 defect**; glutamate was supplied via `PresynapticRelease` as L·ETA-2's
corrected rig requires.

- `E_invasion` **0.3508 vs L·ETA-2's 0.3518 — 0.28%**; peak r **2.53**; **7/7 condensed**.
- **The `r≈0.077` "dead pump" figure repeated from the PO-7 kickoff was never measured.** A
  direct measurement at 1 s returns r = 0.0390 — which is **L·ETA-1's documented rest floor**,
  not a ceiling. **Superseded: the pump ignites.**
- **Condensation strobes rather than latches** — `r` crosses threshold sample to sample on
  stochastic `ca_open`, `n_cond` flicking 0 → 3 → 7 within 0.15 s.
- **Ignition at row-sums 4.14–5.05**, well under L·ETA-1's stated **>6.89**. That table was
  computed with NMDARs silent (`ca_open` ≈ 0.05); with glutamate wired `ca_open` reaches
  0.34–0.70. **Its "unreachable at ≥2 µm spacing at any N" claim must be re-derived before
  being cited again.**

#### 5. The percolation number that motivates §3

Pre-ignition the partition is **7 components, `largest_frac` 0.22** — one per synapse, exactly
as §5 predicts. Then:

```
t=9.05   comps=7   largest_frac=0.218   cross_bonds=0
t=9.55   comps=1   largest_frac=1.000   cross_bonds=98
```

**98 cross-bonds — already past the Werner F>0.5 filter — collapse 100% of dimers into one
component.** Each synapse is a near-complete clique (the 715-degree violation), so any single
cross-bond fuses two dense balls wholesale. **That is why the spin bound is the lever:** it
shatters the cliques so a handful of cross-bonds can no longer weld giant components together.

#### 6. Method notes (both corrections were mine, recorded not buried)

- **A false alarm I raised and withdrew:** bond counts looked monotonic across dissolution
  events (the named red flag in `experiment-design-patterns`). Measuring properly,
  **124 of 399 samples show decreases** — build-destroy-rebuild, which that same skill lists as
  *correct*. The alarm came from reading 2-second snapshots that skipped the drops.
- **An ambiguity in my own instrument:** `n_components == 1` is **not** evidence of a blob —
  `_find_all_clusters` omits unbonded dimers, so 1 can mean "everything merged" *or* "only one
  small bonded cluster exists". The early `comps=1` at t=0.1 s was the latter. `largest_frac`
  was added and is the measure the blob claim rests on.
- **A gate I had to re-specify:** the first Unit-8 gate demanded peak `r` match L·ETA-2's
  1.6234, but that rig used `pattern="clustered"` (`randn × spacing × 0.5`) — **random geometry
  whose row-sums were never recorded**, so peak `r` was not reproducible by construction. The
  gate moved to `E_invasion` (deterministic; matched to 0.28%) with geometry made `linear`.
  Recorded because changing a gate after a failure is exactly what the discipline polices.

#### 7. What is NOT established

- Spin resolution currently governs **INTRA-synapse bonds only** (`dimer_particles._create_bond`).
  **Cross-synapse bonds are created in `multi_synapse_network._update_entanglement` and are not
  spin-accounted.** A dimer's four nuclei must be shared between its intra AND cross bonds; until
  the ledger spans both, cross-bonds consume no spins and can still weld cliques for free.
- The 184-component result is **one synapse, one seed**. Not yet a multi-synapse partition, and
  therefore **not yet a §8 keystone claim**.
- Nothing here revives cross-synapse *provenance*; `L·PO7-2`'s withdrawal stands. This is about
  the bonding rule, which is local.



### L·PO7-2 — ⚠ CORRECTION: the cross-synapse provenance premise is unphysical and is withdrawn; what survives is local · 2026-07-19 `[CORRECTION — supersedes L·PO7-1's cross-synapse claims]`

**Raised by Sarah; verified against the mechanism skill before accepting.** Appended rather than
edited, per this log's append-only rule.

#### The physics PO-7 got wrong

`model6-entanglement-partition-werner` §2 states the cross-synapse mechanism outright:

```
k_cross = K_ENTANGLE_EM_BASE(0.5) * sqrt(eta_i * eta_j) * w_spatial * P_product
```

hard-gated on both spines `mt_invaded`, with **`w_spatial = exp(-distance / 5.0 µm)` — the
CONDENSATE coupling length.** Cross-synapse entanglement is mediated by the Fröhlich backbone /
tryptophan network. **Phosphate provenance is LOCAL**: dimers born in one nanodomain from one
hydrolysis event share an origin. It is not, and was never, a cross-synapse channel.

#### How the error happened (the grounding failure, named)

PO-7 read §1 of that skill — *where* the partition lives — and treated it as licensing a
mechanism that only §2 describes. §2 sits one screen below §1 and was never read. The kickoff's
premise ("hydrolysis events carry absolute network coordinates; a dimer in ANY synapse can claim
an event within reach") was inherited and validated *as implemented* without ever being checked
against the mechanism it was supposed to instantiate. Every downstream unit inherited it.

**The tell that was recorded and misread:** cross edges only formed when spacing was forced to
0.2 µm, and PO-7 wrote in `notes.md` Q2 that the required spacing "excludes the upper half of the
physiological spine-spacing range" — filing it as a *limitation of the result* rather than as
evidence the *mechanism* was wrong. A mechanism that only works at unphysical geometry is not a
mechanism with a caveat.

#### Withdrawn

- The network-shared event pool **as a physical claim** (the code remains, flag-gated and
  bit-identical-off; it is not deleted, but it does not model a real channel).
- `L·PO7-1`'s framing of the 2 µm structural zero as a **"landmine"**. **It is correct physics** —
  phosphate should not reach an adjacent synapse. PO-7 reported the model being right as a bug.
- The **Unit-2 keystone verdict** — it scored a mechanism that should not exist. Its decomposition
  null is uninformative, not a finding.
- The **Unit-7 claim-radius derivation** — moot. Within a single 400 nm nanodomain a 500 nm reach
  was never the limiting factor, so the reaction-diffusion length answers a question that does not
  arise. (The arithmetic is not wrong; the question is.)
- **Unit 6** (synchrony vs stagger) — killed mid-run; it was a cross-synapse design.
- The **R4 advisor packet's core framing**. It must not go out as written.

#### What survives — all local, none of it touched by the correction

1. **Monogamy is violated catastrophically.** Four ³¹P spins per Ca₆(PO₄)₄ bound a dimer to ≤4
   singlet-strength bonds. Measured at the standing fingerprint point: mean degree **715.2**
   (179×), max **902** (226×), **1034/1034 dimers over bound**, **99.44% of edges physically
   inadmissible**, max admissible E = **2068** against an actual **369,740**.
2. **Provenance bonding is monogamy-CLEAN** — max 1 bond per spin, **0 of 434** mediators over
   bound. It satisfies the bound with no cap and no tuning, where the clique rule breaks it by two
   orders of magnitude. This is independent physical evidence for provenance **as the local
   bonding rule**, and it does not depend on any input-computing claim.
3. **The WHICH-SPIN slot tag** — Fisher's inheritance names the phosphate slot, not just the
   partner. Now recorded (`_prov_slot_of`, `_prov_bond_spins`), side-band, ON-path regression
   verified. More useful after this correction, not less: local is the only scale that matters.
4. **The coincidence window is ≤50 ms** (p90 creation→second claim; 84.3% of events consumed
   within one tracker step) against a nominal 2.0 s age ⇒ **`provenance_net_age_s` is
   NON-OPERATIVE** and needs no physical justification.
5. **Two real inherited defects fixed** — frozen fidelity (F stored at claim time while
   `_find_all_clusters` tested it against the live Werner bound) and dropped coherence death
   (pruned on `is_entangled`, not `P_S<=0.5`). Both are defects on any reading of the physics.
6. **The Unit-5 sheaf is a direct sum of graph Laplacians** — cross-block edges **0 / 369,740**,
   decomposition identity verified numerically. Not irreducible sheaf structure. (Correction to
   the advisor's own count: the channel rule fuses channels in PAIRS, giving **3** blocks
   `{0,1},{2,3},{4,5}`, not 6; at this operating point each block has exactly 1 component, so
   `H0_engaged = 3` = the number of spatial axes.)

#### ⚠ The real finding this exposes

The charter sought cross-synapse edges **without η**, reasoning that the pump is dead
(r≈0.077, η=0) and a route around it was needed. **If direct entanglement is only ever local,
there is no η-free route.** Therefore:

> **η being dead is a HARD BLOCKER on cross-synapse entanglement — not an obstacle to engineer
> around.**

The network-provenance layer was a way of obtaining the desired answer by changing the physics.
That is a more consequential result than anything in `L·PO7-1`, and it relocates the target: the
dead pump (`r≈0.077`) is the thing to fix, and §8's cross-synapse keystone is blocked behind it.



### L·PO7-1 — network-shared provenance BUILT and VALIDATED (η-free cross-synapse edges are real), multi-synapse §8 keystone is a pre-registered NEGATIVE (decomposition null), and three design defects found in my own pre-registration · 2026-07-19 `[GROUNDED, measured — mechanism YES, keystone NO]`

**Pre-registered:** `docs/PREREG_PO7_UNIT2_MULTISYNAPSE_KEYSTONE.md` (guard statement included, three
independent achievable routes to a negative named before running). **Charter:**
`coordination/requests/po7-provenance-network/kickoff.md`. **Open questions + design corrections:**
`coordination/requests/po7-provenance-network/notes.md`.

#### 1. THE BUILD — the handoff's open item is closed with data

Inherited PO-7 WIP (`514d637`) verified by reading `_step_network_provenance` in full rather than
trusting it. `global_id = (syn_idx, dimer.id)` is a TUPLE (`:183`), so the `a[0] != b[0]`
cross-synapse diagnostic is sound. **Two defects repaired** (both inside the opt-in path):

1. **Frozen fidelity** — `_prov_bonds` stored `F = P_S_i·P_S_j` at claim time and never refreshed,
   while `_find_all_clusters` tests it against the live Werner bound. A decohered pair kept its
   birth fidelity. Now recomputed each step.
2. **Dropped coherence death** — the prune tested `is_entangled` only, contradicting the method's own
   docstring. Same dropped-channel shape as the U16 write-once bug (`07fd02a`). Now prunes on
   `P_S <= 0.5`.

**Off-path bit-identity `1034 / 369740 / 0.991922159684` — PASS before the build and re-gated
after scoring.**

#### 2. UNIT 1 — the mechanism is REAL, and η-free

Cross-synapse provenance edges form. **`eta_cross = 0` in all 20 scored runs**: the η-mediated
channel contributes nothing (dead pump, r≈0.077), so **every** cross edge observed is
Fisher-inherited from a shared hydrolysis event. That is the η-free claim the layer was built to
demonstrate, shown at the data level.

| spacing | mean cross edges | mean overlap | seeds with cross |
|---|---|---|---|
| 0.2 µm | 2.5 | 0.046 | 2/2 |
| 0.4 µm | 0.5 | 0.007 | 1/2 |
| 0.6 / 0.8 / 1.2 / 2.0 µm | 0.0 | 0.000 | 0/2 |

The prediction registered in the probe header BEFORE the run held exactly: zero above 0.9 µm,
nonzero below — from grid span 400 nm (`dimer_particles.py:109`) vs reach 500 nm (`:124`), giving a
minimum cross-synapse distance of `spacing_nm − 400`.

**⚠ The committed DEFAULT `spacing_um = 2.0` sits in the structural-zero regime.** The layer as
inherited could never have produced a single cross edge, and anyone running it at defaults would
have recorded a physical-looking null for a purely geometric reason.

*Unplanned consistency check:* 0.6/0.8/1.2 µm give byte-identical intra counts per seed — once
spacing exceeds reach the synapses are fully independent, confirming spacing touches ONLY the
cross-synapse channel.

#### 3. UNIT 1b — a hard CEILING on the cross channel

Cross edges appear **once** (t=0.25 s: 2 edges, 1 of 15 synapse pairs, `n_multi=1`) and then never
again through t=2.0 s, while `prov_total` grows **153 → 481 (+328 bonds, ZERO new cross edges)**,
including two ~85-dimer birth cohorts. Adequate power is therefore **not** reachable by running
longer. *(Corrected mid-run: I first read a lull as "the graph is frozen" and quoted an accrual rate
from what was a single formation event — both overstated, both retracted in-thread.)*

#### 4. UNIT 2 — THE PRE-REGISTERED VERDICT: NEGATIVE, decomposition null

6 synapses, 0.2 µm (selected by the pre-registered overlap rule, not by yield), 5 seeds, 2 arms.

| arm | mean Q_act | mean Q_shuf | d | **mean n_multi** |
|---|---|---|---|---|
| arm1 contiguous {0,1,2} vs {3,4,5} | −0.0000 | −0.1608 | +1.609 | **0.60** |
| arm2 interleaved {0,2,4} vs {1,3,5} | +0.0000 | −0.1525 | +1.332 | **0.50** |

> **NEGATIVE: DECOMPOSITION NULL — partition splits cleanly per-synapse (mean n_multi < 1) —
> input-LOCATED but not input-COMPUTING.**

Seed dominates the contrast: seed 7 is high-yield in every arm; seeds 4242 and 90210 flip direction
between COND-A and COND-B. Which synapses are active does **not** systematically determine whether a
cross-synapse component forms. Same shape as L·PO5-13's `d=0.02`, reached by a different route.

#### 5. ⚠ THREE DESIGN DEFECTS IN MY OWN PRE-REGISTRATION — recorded, not absorbed

1. **Dimer-level Q would have been ≈1.0 BY CONSTRUCTION.** `intra_synapse_bonds_cache` is the dense
   clique (E=369,740 intra edges in the single-synapse fingerprint alone) and every intra edge lies
   inside one synapse hence one activation label. **Caught BEFORE scoring**; Q moved to the
   synapse-level graph that `quantum-system-canonical` §5 names.
2. **"Density matched by construction" is FALSE.** Equal *counts* of active synapses ≠ matched
   density across nodes: inactive synapses at −70 mV make almost no dimers, so cross edges can form
   only among active ones **trivially**. Evidence: `Q_act = +0.0000` in **all 20 runs** — Newman's
   degenerate value when every edge sits in one community — so `d = 1.61 / 1.33` is an artifact of
   the shuffled null going negative, **not a signal**. **A PASS on the Q criteria would have been a
   THIRD false positive in this series.** The verdict function's all-three-criteria requirement is
   what prevented it. This is L·PO5-11's "topology is a function of density alone" reappearing
   inside a metric built to dodge L·PO5-13.
3. **ARM 2 conflates layout with distance.** Interleaving moves co-active synapses from 0.2 µm to
   0.4 µm apart — the edge of the band — so criterion 3 is not a clean spatial-confound test. ARM 2
   still breaks the space/identity alignment it was designed for.

#### 6. What this establishes, and what it does NOT close

**Establishes:** the network-shared provenance layer is built, validated, and produces genuine
η-free cross-synapse entanglement edges and components — the first time the cross-synapse channel
has been shown to work without the dead pump. **Does NOT establish:** that provenance cannot carry
an input-dependent partition. The verdict is a **decomposition null under a density-confounded
label and a ceiling-limited edge count.** §8 at multi-synapse scale remains **OPEN, not answered
negative.**

**No constant was tuned to reach a verdict.** Spacing was fixed by the pre-registered overlap rule
before scoring; the Werner bound stayed at 0.5; both verdict functions were demonstrated FAILING on
synthetic negatives before being allowed to pass.

#### 7. THE DESIGN FIX for the next worker (the real deliverable)

Drive **all six synapses above dimer-forming threshold** and vary the input as **HIGH vs LOW drive**
rather than **ON vs OFF**, keeping an interleaved layout with **co-active spacing held constant
across arms**. That is confounded by neither space nor density. **It must be combined with a
cross-edge yield increase** — via `provenance_net_reach_nm`, `provenance_net_event_rate`, or the
born-with claiming rule (`:504`, which permanently denies provenance to dimers born before any
event exists) — or the statistic still has too few edges to read. **Fix the confound and the power
together, or the next test is unreadable too.**

**Stated limits:** the claim radius is a **2D projection** (`abs_xy` drops z while `pattern="linear"`
jitters z by ±0.2 µm), biasing cross-edge formation upward. Cross edges require spacing < 0.9 µm,
excluding the upper half of the physiological spine-spacing range. This is an **(A)-reading** result
and is not evidence for (B) (`quantum-system-canonical` §5.1, LOCKED).



### L·PO5-13 — provenance bonding BUILT and works (first non-blob graph), but the computation test is a FALSE POSITIVE the probe declared and PO-5 overrode · 2026-07-19 `[GROUNDED, measured — mechanism yes, keystone NO]`

**Pre-registered:** `docs/PREREG_PO5_UNIT16_PROVENANCE_BUILD.md`. **Build:** provenance-based bonding
in `dimer_particles.py`, opt-in, **provenance-off bit-identical** (`1034 / 369740 / 0.991922159684`).
Fisher's actual mechanism (per `L·PO5-12`): events at calcium-elevated cells (2 slots each), a
newborn dimer claims its ≤2 nearest events, two dimers bond iff they share an event. Provenance also
skips the EM pathway (LOCC non-entangling, percolates alone — `L·PO5-10`).

#### THE MECHANISM WORKS — the first non-blob graph in the investigation

| | bonds | components | largest_frac |
|---|---|---|---|
| provenance OFF (clique+EM) | 459,889 | 1 | 1.000 |
| **provenance ON** | ~500 | ~700 | **0.05** |

A genuinely sparse, pairwise, fragmented graph. Every prior mechanism produced a blob; this one does
not. That part is real and it is new.

#### ⚠ THE COMPUTATION TEST IS A FALSE POSITIVE — the probe said SUCCESS, PO-5 says NO

`po5_unit16_computation_test.py`'s verdict function printed *"PROVENANCE CARRIES INPUT-DEPENDENT
PARTITION BEYOND DENSITY."* **It is wrong, and PO-5 overrides it.** The probe scored on
`Q(input) = 0.15 >> Q(shuffled) = 0` — but the "input label" was **which spatial half a dimer sits
in**, so Q=0.15 detects **spatial locality** (dimers bond to neighbours because they claim the
nearest events). **That is GEOMETRY, which §8 explicitly rules insufficient** (`g` is geometry, not
input). The decisive quantity is the actual input contrast:

**COND-A vs COND-B (sustained vs pulsed), 5 seeds: component effect size d = 0.02 — a flat null.**
Changing the input does NOT change the partition.

So: the metric passed without answering the question — the same criterion-mis-registration error as
Units 8, 9, 13. **The honest verdict is NOT keystone-supported.**

#### Why the null, and what it does and does NOT close

The provenance assortativity is **spatial** (nearest-event claiming), and at single-synapse scale the
pulsed-vs-sustained contrast does not produce meaningfully different *spatial* calcium patterns — the
same peak-saturated, weak-contrast wall as Units 9/10/13/14. So this run tested provenance with an
input that barely varies the spatial pattern. **It establishes:** the mechanism is faithful, sparse,
and carries spatial structure. **It does NOT establish, either way:** whether input that genuinely
varies the spatial calcium pattern would move the partition. Unit 15 proved the RIG CAN carry input
structure *if event-sharing is assortative-by-input*; the build produced assortativity-by-SPACE, and
this test's input does not drive the space.

#### What a real test needs (registered for next time, not run)

An input contrast that produces **demonstrably different spatial calcium patterns** — which
single-synapse, one-nanodomain geometry may be unable to provide (§5 LOCKED: one synapse = one
component). The honest possibility is that **the pair-level input channel requires the multi-synapse
scale** the whole program was told the keystone does NOT need (§7 #1 "single-synapse-scale"). That
tension (`po5-selectivity-002.md` §2) is now sharpened by a measurement, not just argued.

**No constant tuned.** event_rate was swept for an unsaturated regime, not to a target; the verdict
stands on d=0.02 regardless of rate.



### L·PO5-12 — DEEP RESEARCH (external + internal): provenance-bonding is FAITHFUL to Fisher, but its computational capability is UNPROVEN and must be de-risked before the build · 2026-07-19 `[external lit review, adversarially verified; internal code map]`

**External:** deep-research harness, 5 angles, 3-vote adversarial verification, primary sources
(Fisher 2015 Annals of Physics / arXiv 1508.05929; Swift/Fisher/Radzihovsky 2018; Player & Hore 2018
J.R.Soc.Interface; Agarwal 2021/2023; RIG literature Karoński–Scheinerman–Singer-Cohen, Deijfen–Kets).
**Internal:** codebase change-surface map (agent, read-only).

#### GREEN — provenance-bonding is Fisher's ACTUAL mechanism (high confidence, 3-0)

Fisher's inheritance is **strictly PAIRWISE and provenance-based**: enzymatic hydrolysis of ONE
pyrophosphate (PPi → Pi + Pi) leaves the two daughter ³¹P spins in a two-qubit singlet; two Posners
are entangled **iff they each carry one member of a shared, common-origin pair.** Verbatim Fisher:
*"if two such Posner molecules share an entangled phosphate pair, their spins will be entangled."*
**So "dimers carry their hydrolysis events, bond iff they share one" is literally Fisher's channel —
and the current clique (born-in-a-window, all-to-all) and distance-kernel rules are both wrong.**

#### RED FLAGS — two premises to fix, both high-confidence 3-0

1. **Stoichiometry — ⚠ PO-5 CORRECTION, the research had this BACKWARDS.** The deep-research agent
   flagged the model's Ca₆(PO₄)₄ as a "wrong premise" to fix toward Fisher's Ca₉(PO₄)₆. **That is
   wrong and it contradicts a LOCKED decision.** `quantum-system-canonical:43` [PROVEN — Agarwal 2023]
   [LOCKED]: *"the Ca₆(PO₄)₄ dimer is the computational qubit. The Ca₉(PO₄)₆ Posner trimer lacks a
   rotational symmetry axis and decoheres sub-second — computationally inert."* **The program
   DELIBERATELY uses the 4-phosphate dimer precisely because the 6-phosphate trimer is inert; the
   research question's Ca₉(PO₄)₆ framing is the thing the program already rejected.** The correct
   object is the **dimer: 4 ³¹P spins = 2 singlet pairs from pyrophosphate** (`dimer_particles.py:23`),
   so provenance is **≤2 events per node** (multi-event, but 2 not 3), built on the LOCKED dimer.
   **This physics was already documented; the workflow rediscovered it and then mis-corrected it
   against a locked skill. Logged as a PO-5 process failure.**
2. **Genuine entanglement almost certainly does NOT survive.** Player & Hore: inter-Posner singlet
   lifetime is **seconds, not the required long times**; two independently tumbling Posners lack the
   inversion symmetry that shields a single molecule, so intermolecular singlets cannot be protected.
   **⇒ lean on the CLASSICAL common-cause reading, not the quantum one.** Provenance-bonding is
   legitimate under both (shared origin is a textbook common cause), so this does not kill the build —
   it reframes its justification.

#### THE GATE — the one thing the research does NOT establish (high confidence, and it is decisive)

The proposed object is exactly the **s=1 general Random Intersection Graph** (nodes = dimers, shared
event = edge). Known math: sharp giant-component threshold `c = n·Σ_w p_w²`; edge set is a union of
event-induced cliques; **ER-equivalent only when features are very large (m ≥ n³), and BELOW that it
provably carries clique/triangle structure beyond mere density.** So it is *richer than density* —
necessary. **But NO source shows the RIG carries INPUT-DEPENDENT partition structure** (partitions
that change with stimulus, not just with density). *"Faithfulness to Fisher is established;
computational capability distinct from density is NOT."* **This is the exact §8 property PO-5 has
failed to find five times, now shown to be unproven in the literature for the proposed replacement
too.**

#### Two caveats that shape the build

- **"Bond" must mean the ENTANGLEMENT EDGE, not literal chemical binding.** Fisher separates
  inheritance (provenance) from binding (readout). Conflating them misrepresents him.
- **A competing channel exists and a pure provenance graph is blind to it:** Quantum Dynamical
  Selection predicts binding-*induced* pseudospin entanglement between two Posners, NOT conditioned on
  shared provenance. Orthogonal to this model; noted, not built.

#### Internal change surface (agent-mapped, read-only)

**Smaller than PO-5 estimated: ~50–75 lines across three files, all gated by a `provenance_bonding`
flag, bit-identical when off.** The ATP layer **already computes the per-event burst mask**
(`atp_system.py:111` `burst_events = np.random.rand(...) < burst_probability`) and **discards it at
:119** — so provenance raw material exists transiently and need only be captured, not invented. Bulk
of work: capture events (id/time/location), thread across `atp_system → model6_core → dimer_particles`
(all currently pass only fields). **One genuine physics/design call surfaced:** births are placed by
sampling a probability field, so matching a birth to its event(s) needs a spatial-join rule (nearest?
in-cell? within-radius?) — *that rule IS the "which phosphates ended up in which dimer" physics.*

#### DECISION POSTURE

Build is faithful and the code path is cheap — **but the research says its computational payload is
unproven, and that is the precise thing five prior units failed to find.** **Recommendation: de-risk
FIRST with an abstract RIG simulation** (pure graph, no physics, minutes) — assign events to nodes
under input-varying vs input-constant regimes and test whether the partition moves beyond density. If
yes → the 50–75-line build is worth it. If no → the whole provenance direction is dead and days are
saved. **This is not more drift: it is the validation the research explicitly names as the gate.**

#### ⚠ PROCESS FAILURE, recorded (Sarah, 2026-07-19)

The physics half of this research (Fisher pairwise; dimer-not-trimer; sub-second inter-Posner
decoherence) was **already documented** in `quantum-system-canonical`, `model6-research-findings-may29`
and the code docstrings, and was in front of PO-5 the whole session. The 100-agent workflow largely
**rediscovered documented material**, and its stoichiometry "correction" **contradicted a LOCKED
decision** (above). **The only genuinely new contribution was the random-intersection-graph math**
(absent from the repo — grep confirms), which is the one part that bears on the go/no-go. The correct
move would have been to READ the skills for the physics and commission research ONLY on the RIG
question. Recorded so the cost is not repeated.



### L·PO5-11 — the topology is a FUNCTION OF DENSITY ALONE; and the input→population channel is measured shut · 2026-07-19 `[GROUNDED, measured — negative]`

**Units 13–14.** Together these establish §8's failure with a contrast that actually varies, and
localise WHY one layer upstream of everything measured before.

#### Unit 13 — the fidelity cut is COSMETIC (spatial covariate, 5 seeds)

The only lever that fragments the graph (Unit 12) does not carry input. Advisor's covariate — spatial
statistics of births, not density — showed **birth geometry is identical between drive conditions**
(`r_med` d=0.00, `r_p10` d=0.00), and the partition after the cut follows (max d=0.54, bar 2.0).
**VERDICT: COSMETIC** — the cut fragments the blob but the structure carries no input information.
PO-5 had pre-named this as the single most likely way it was wrong; it landed.

**And the per-seed rows exposed the deeper cause:** SUSTAINED and PULSED gave **BIT-IDENTICAL** V,
positions and topology on seeds 72/73 — the drive PATTERN never reached the dimer population.
`target_count` derives from `peak_conc = np.max(dimer_concentration)` (`:174-175`) — a PEAK, not an
integral — so both protocols hit the same peak in their first burst; with write-once bonds the
population never revisits it.

#### Unit 14 — topology is a function of V alone

Amplitude DOES vary the population (per-seed V range **[1101, 1192]**; P1 passed — note the
per-amplitude *means* are a narrow 1125–1137, and PO-5 briefly misread those means as the full range
mid-run, then corrected against the per-seed spread). Fitting topology ~ V and comparing fit-residual
scatter to within-seed scatter:

| measure | slope | resid SD | seed SD | ratio | verdict |
|---|---|---|---|---|---|
| comps_cut | −0.017 | 2.71 | 2.14 | **1.26** | DENSITY ALONE |
| lf_cut | +0.00002 | 0.003 | 0.002 | **1.23** | DENSITY ALONE |
| H0_cut | −0.031 | 4.41 | 4.40 | **1.00** | DENSITY ALONE |
| comps_raw | +0.001 | 0.22 | 0.10 | 2.28 | inconclusive |

**Every cut-topology measure has residual scatter no larger than seed noise: the partition carries
exactly what the scalar V carries.** That is LITERALLY §8's *"scalar as computation"* — now
established with a contrast that varies, unlike Units 9/10/13. P3 pattern arm reproduced Unit 13's
null (SUSTAINED comps 9.7 vs PULSED 9.0 at matched amplitude).

#### The correction PO-5 owes on Units 9/10/13

Those three tested input-dependence with the SUSTAINED-vs-PULSED **pattern** contrast, which Unit 13
now shows **does not reach the dimer population** (bit-identical V). So their nulls were under-founded
— they tested input-dependence with an input that barely varied. Unit 9's effect reversing sign on
fresh seeds reads, in hindsight, as exactly what a non-contrast produces. **The keystone conclusion
is unchanged (not supported); the correction is that three of its tests were weaker than reported,
and Unit 14 is the one that establishes it on a contrast that works.**

#### The finding this converges on, with the advisor's diagnosis

**No input dimension reaches the topology as pair structure.** Pattern doesn't reach the population at
all; amplitude reaches it only through V, and topology is a function of V alone. This is the
population-side arrival at the advisor's diagnosis (`L·PO5-10`): the model substitutes input-blind
proxies at every layer because it has **no representation of the entangling origin**. `peak_conc`
saturation + write-once dynamics = an input-insensitive population; the birth window and distance
kernel = input-blind bond rules on top of it. **The next step is not another test — it is to build
the provenance layer** (dimer↔hydrolysis-event↔dimer), the one edge rule with a live input→topology
path. External + internal deep research commissioned to ground that build before committing to it.



### L·PO5-10 — the EM pathway is NOT AN ENTANGLING MECHANISM; the clique is not what percolates; and the fidelity cut is the only lever · 2026-07-19 `[GROUNDED, measured + structural]`

**Units 11–12**, plus two structural findings from the round-2 advisor exchange.

#### 1. STRUCTURAL — the pathway carrying the percolation cannot create entanglement

`dimer_particles.py`'s EM-pathway docstring already concedes: *"The microscopic Hamiltonian for
EM-mediated nuclear spin coupling in biological systems **remains an open question**. The
UV-frequency tryptophan field (~10¹⁵ Hz) **cannot couple directly to nuclear spin dynamics (~Hz)**
via standard dispersive mechanisms. Our **phenomenological treatment** captures…"*

**The frequency gap is a symptom; the LOCC argument is the disease.** A field amplitude Φ that
modulates a *formation rate at each dimer independently* is a **local** operation. Local operations
cannot create entanglement — a theorem, not a modelling concern. Creating entanglement between two
nuclear spins requires either **(a)** a direct interaction term `H_ij`, or **(b)** a common quantised
mode with virtual-excitation exchange. **`em_rate` has neither** — it is a classical scalar scaling a
rate.

**The dichotomy, and the program cannot take both branches:**
- if the bonds are **genuine entanglement** → the EM pathway cannot create them;
- if they are **classical correlations** → the mechanism is fine but the entanglement-partition
  claim loses its quantum grounding.

**The asymmetry worth carrying:** **P0 (83%) has a legitimate entangling origin** — shared
pyrophosphate is a real common origin — but is implemented as an unphysical clique. **P2 (17%) has
no entangling mechanism at all** — and P2 is what percolates (Units 7, 11). *The mechanism with
physical grounding is implemented wrongly; the one carrying the result has no grounding.*

#### 2. MEASURED — removing the clique changes nothing (Unit 11, native field, 2 seeds)

| arm | edges | components | largest_frac |
|---|---|---|---|
| baseline | 407,332 | 1.0 | 1.0000 |
| degree cap k=4 (Fisher-consistent) | 140,697 (**−65%**) | 1.0 | **1.0000** |
| degree cap k=1 | 138,722 (−66%) | 1.0 | **1.0000** |
| J-compatibility on formation | 404,109 (−0.8%) | 1.0 | 1.0000 |
| coupling_length 5→1 nm | 372,170 | 20.0 | 0.9821 |
| all three combined | 105,536 (**−74%**) | 20.0 | 0.9821 |

**Two-thirds of edges delete with zero effect on connectivity. Density is not the problem.** This
also **refutes the round-1 prediction** that fixing the clique would fragment the graph: the physics
of that critique was right (coincidence is not entangling, entanglement is not transitive) but the
*diagnosis* was wrong — Unit 7 had already shown P2 spans alone.

**Per-event matching is not implementable.** Dimers are born from a **concentration field**
(`:210-213`) — no pyrophosphate objects, no per-phosphate provenance. **The 100 ms clique is a proxy
for a representation the model does not have.** Filed as the diagnosis, not the obstacle: Fisher's
mechanism is *inheritance*, and a model with no origin representation must substitute proxies —
every available proxy (birth window, distance kernel) being input-blind by construction is why no
threshold sweep rescues §8.

#### 3. MEASURED — the fidelity cut is the only lever that moves (Unit 12)

Intra bonds have **no fidelity threshold**: connectivity is *bare existence*, so a bond made at 15 nm
counts identically to one made at 3 nm. Storing `F = P_S_i·P_S_j·g(r_ij)` and sweeping (**no value
nominated**; the skill's *"do NOT apply the 0.5 bound to intra"* is respected — measured `F ≈ 0.36`
at 7 nm, so the caution is quantitatively right):

| threshold | edges kept | components | largest_frac | sheaf H⁰ |
|---|---|---|---|---|
| 0.00 | 100% | 1.0 | 1.0000 | 3.0 |
| 0.20 | 32% | 5.0 | 0.9957 | 10.5 |
| 0.30 | 26% | 7.0 | 0.9943 | 14.5 |
| 0.50 | 21% | 11.0 | **0.8314** | **15.0** |

`F` distribution is **bimodal** (p10 = 0.019, med = 0.090, p90 = 0.980), split at the `g` plateau.
**Sheaf H⁰ rises monotonically 3 → 15** — readout resolution and graph structure moving together for
the first time.

**⚠ TWO KNOWN DEFECTS IN THIS `F`, both conceded:** (i) the bimodality is **manufactured** — the
split sits exactly at the 5 nm plateau, which is a numerical regularisation of a 1/r³ divergence, not
physics, and `coupling_length` is uncited; (ii) **`F` is a category error** — it is built from a
*rate* term, reproducing the very dead-store pattern it was meant to fix. Fidelity must come from
**state**: `F(t) = ¼ + (F₀ − ¼)·exp(−t/T₂)`. **No threshold may be nominated on the current `F`.**

#### 4. MEASURED — the ℝ⁶ latent space is populated with INPUT-INDEPENDENT NOISE

`j_couplings_intra = np.random.normal(0.15, 0.15, size=6)`, drawn **independently per dimer at
construction** (`:48-50`) with no dependence on position, birth time, calcium or input. Measured
between-dimer spread: per-coordinate std **0.1478 Hz**, median `‖J_i − J_j‖₂` **0.4864 Hz**.

**So the latent dimension is not unpopulated — it is populated with noise that has no causal path
from input.** Bonding on ℝ⁶ compatibility therefore *cannot* carry input information, whatever the
metric. (PO-5's `min_k|ΔJ|` rule was separately too permissive — a union of six slabs, not a metric —
but fixing it to `‖·‖₂` would bond on better-measured noise.) **The latent-dimension principle
survives; this instance of it does not.** For it to bite, the ℝ⁶ would have to be *derived* from
something input-coupled.

#### 5. OPEN — the dipolar-coupling proposal needs a timescale before anyone builds on it

Direct ³¹P–³¹P nuclear dipolar coupling is a genuine option (a): a real Hamiltonian, genuinely
entangling, **no free parameter** (coupling fixed by γ_P). PO-5 verified the magnitudes:

```
r =  1 nm : 19.58 Hz   t_ent ≈    51 ms
r =  2 nm :  2.45 Hz   t_ent ≈   409 ms
r =  5 nm :  0.16 Hz   t_ent ≈   6.4 s
r = 10 nm :  0.02 Hz   t_ent ≈  51.1 s
```

**But the claimed ~1–2 nm range does not follow.** `t_ent` must be compared to something; against
T₂ = 216 s or dimer lifetime ~1000 s (`k_dissolution = 0.001/s`), entanglement still forms at 10 nm
and fails only past ~15–20 nm — **more permissive than the current 5 nm plateau**, against a measured
median separation of 9.75 nm. **The proposal is sound; the range claim requires a ~100 ms competing
timescale that PO-5 cannot identify** (the model has no dimer diffusion). **Flagged as open — do not
build on it until that timescale is named.**



### L·PO5-9 — the transient hypothesis FAILS, and it falsifies a claim PO-5 wrote into this log minutes earlier · 2026-07-19 `[GROUNDED, measured — negative]`

**Pre-registered** in `sweep/po5_unit8_transient.py`'s header before the run. **No overrides — the
system running natively; the only change is WHEN we look.** 3 seeds × 11 sample times.

#### P1 FALSIFIED — there is no rise

```
field across ALL samples, all seeds: min = 22.095   max = 22.095
```

**The field is already saturated at the first sample (t = 0.01 s) and never moves.** Unit 7's
`range [0.000, 22.095]` came from one or two samples at the very start of the run, before the first
physics step took effect — **not from a gradual sweep.**

**This directly falsifies `L·PO5-7`'s claim**, written earlier the same day, that the field *"STARTS
AT ZERO and climbs to ~22 every run"* and that the system *"transits the entire fragmented→blob range
on every run."* **It does not.** It reaches its operating value within ~2 timesteps and stays. That
claim is **WITHDRAWN**; `L·PO5-7` is annotated. The error was reading a min/max range as a
trajectory without checking the time series — the same shape as reading a mean without its spread,
one level up.

#### P2 "confirmed" on the registered criterion — and the criterion was too weak

The registered test was `components > 1 AND largest_frac < 0.99`. It fires at t = 0.01–0.02. But:

| t | components | largest_frac |
|---|---|---|
| 0.010 | 10–14 | 0.962 – 0.983 |
| 0.020 | 6–11 | 0.980 – 0.993 |
| 0.040 | 4–6 | 0.994 – 0.996 |

**That is a giant component holding 96–98% of dimers plus a handful of crumbs — not a fragmented,
informative state.** The giant component is present at the earliest observable moment. **Read
substantively the answer is P3, not P2:** saturation precedes any structure, and the transient
hypothesis fails.

**The registered criterion was too lenient and PO-5 says so rather than banking the pass.**
`largest_frac < 0.99` admits "blob plus crumbs". A criterion that could have discriminated would
have required something like `largest_frac < 0.5` — the same bar Unit 7 used to call P0 fragmented.

#### What this closes

The most hopeful remaining possibility — that the computation happens in an early critical window and
the endpoint destroys it — **is closed.** There is no such window. The graph is a blob from the first
measurable instant, at the native field, under this drive.

#### Limits
Single synapse, 3 seeds, one drive condition, 1 s. Says nothing about whether input modulates
whatever little structure exists in the first 20 ms.



### L·PO5-8 — the 100 ms birth window is NOT derived, was flagged sweepable on 2026-05-29, was never swept, and it turns out to decide the architecture · 2026-07-19 `[GROUNDED, documentary]`

**No compute. A provenance check, prompted by Unit 7 measuring that this parameter sets whether
the topology can carry information at all.**

**What the program already knew, `model6-research-findings-may29`:**

> `:66` — *"**birth_window value:** 100ms hardcoded. Within Fisher's 1s budget. Conservative but
> defensible. **Tunable parameter — candidate for TALON sweep, not arbitrary calibration.**"*
> `:199` — *"birth_window 100ms in correct regime (Fisher 2015 <1s budget)"*

**So the 100 ms is bounded, not measured.** Its only grounding is an *upper* bound — Fisher's ~1 s
coherence budget. Anything below 1 s is equally "defensible" on that basis, and **Unit 7's
structure-preserving regime (2–10 ms) sits inside the same permitted band.** Sweeping it is
therefore **sanctioned, not tuning** — it was explicitly nominated for a sweep on 2026-05-29 and
**never swept in the seven weeks since.**

**The consequence, stated plainly:** a parameter recorded as *"conservative but defensible"* is the
one that decides whether §8's keystone can work. Unit 7 measured the P0 percolation threshold at
~2–10 ms against a native 100 ms — **10–50× above it.** "Defensible within a bound" was doing far
more load-bearing work than that phrasing implies.

#### The mechanism gap, also already known and still uncorrected

> `:64` — *"The docstring at `dimer_particles.py:218-219` ('Phosphates from same pyrophosphate
> hydrolysis are born entangled') **OVERCLAIMS what the code does.** Comment should be updated to
> reflect spatial-temporal proxy."*
> `:141` — *"**Actual gate is spatial proximity to template field** (scaffolding protein density)."*

**Verified 2026-07-19: the docstring is UNCHANGED** (`dimer_particles.py:234-236` still reads
*"Phosphates from same pyrophosphate hydrolysis are born entangled"*). Flagged 2026-05-29, still
present.

This is the gap PO-5 rediscovered independently in `L·PO5-2`: **Fisher's mechanism entangles TWO
phosphates from ONE pyrophosphate — a pair. The code entangles ALL template-bound dimers born in a
100 ms window — a clique.** The program identified the overclaim seven weeks ago; what was not
noticed is that the all-to-all reading is what generates the 60–90-dimer cliques that percolate the
graph.

**Not proposing a value change.** The finding is that the parameter is unmeasured inside a wide
permitted band and is load-bearing. What it needs is grounding, or an honest sweep with the response
reported — not a setting chosen because it yields the desired topology.

---

### L·PO5-7 — BOTH mechanisms percolate independently; `L·PO5-5` is corrected TWICE · 2026-07-19 `[GROUNDED, measured]`

**Pre-registered:** `docs/PREREG_PO5_UNIT7_CRITICAL_POINT.md`, registered **as a self-correction**
before the run. **Probe:** `sweep/po5_unit7_critical_point.py`. **Raw committed.**
`birth_window` promoted from two local literals to an attribute, **verified bit-identical**
(`1034 / 369740 / 0.991922159684`).

#### P1 CONFIRMED — the bus does NOT form the giant component

`largest_frac >= 0.8725` at **every** bus value including **0**, where there are **zero** P2 bonds.
χ (mean finite-cluster size) peaks at bus = 0 and decays monotonically (23.63 → 1.00 → 0.00), so the
threshold in the bus direction is **at or below the lowest accessible value**.

**`L·PO5-5`'s headline — *"the BUS is a real percolation control parameter"* — is WRONG and is
corrected here.**

#### And the follow-up framing was ALSO wrong

PO-5 then said the bus *"only absorbs stragglers."* Measured at native bus with the birth window
shrunk 50× (which fragments P0 to `largest_frac = 0.4145` on its own):

| birth window | largest_frac @ bus=0 | largest_frac @ NATIVE bus |
|---|---|---|
| 2 ms | **0.4145** (fragmented) | **1.0000** |
| 10 ms | 0.5079 | 1.0000 |
| 50 ms | 0.8663 | 1.0000 |
| 100 ms (native) | 0.8945 | 1.0000 |

**At native strength the field spans the whole graph BY ITSELF, regardless of birth structure.** It
is not a straggler-absorber; it is an independently sufficient percolating structure.

**So the system is DOUBLY supercritical: P0 alone percolates, and P2 alone percolates. Fragmenting
the graph requires reducing BOTH. Neither lever alone can work** — which retires every single-lever
fix previously proposed, including PO-5's own SOC/regulation suggestion.

**P0's threshold is ~2–10 ms** (largest_frac crosses 0.5 between 2 and 10 ms) against a native
100 ms.

#### ⚠ WITHDRAWN by `L·PO5-9` (same day) — the section below is WRONG

**PO-5 read Unit 7's `range [0.000, 22.095]` as a trajectory. It is not.** Unit 8 sampled the field
directly across the run: **min = max = 22.095 at every sample from t = 0.01 s onward.** The field
reaches its operating value within ~2 timesteps. **There is no transit, no sweep, and no dynamic
crossing.** The text below is preserved per the log convention; do not build on it.

#### The number PO-5 had been averaging away [WITHDRAWN — see above]

```
NATIVE field: mean 21.984   std 1.558   range [0.000, 22.095]
```

**It STARTS AT ZERO and climbs to ~22 every run.** PO-5 reported only the mean in `L·PO5-5`. The
system therefore **transits the entire fragmented→blob range on every run and then parks above it**,
and every measurement in this program — including all of PO-5's — sampled only the endpoint.
Followed up in Unit 8.

---

### L·PO5-6 — J-mismatch dissolution: NOT SUPPORTED. The dissolution channel is inert. · 2026-07-19 `[GROUNDED, measured — negative]`

**Pre-registered:** `docs/PREREG_PO5_UNIT6_J_MISMATCH.md`, committed **before** the code change.
**Tier-3 change**, opt-in, flag-off **verified bit-identical** to pre-change code.

**Verdict at all three bus values: `NOT SUPPORTED (no effect)`.** `(B−A)/spread = 0.00` everywhere;
REAL vs OFF differ by 29–49 bonds out of ~291,000–324,000 (≈0.015%), with identical component counts
and identical sheaf H0.

**Why, computed rather than guessed:**

```
k_disentangle (OFF) = 0.01*(1-coh)/(1+protection) = 9.95e-05 /s
typical J-mismatch multiplier                     = 3.0x
P(a bond dissolves) over  1 s : OFF 0.00010  ON 0.00030
                    over 30 s : OFF 0.00298  ON 0.00891
```

**With `coh ≈ 0.98` the `(1-coh)` factor makes the base rate ~1e-4/s. The graph is effectively
WRITE-ONCE.** Tripling a rate that small cannot matter on any timescale this model runs. This also
retro-explains `L·PO5-1`'s saturation decline (0.944 → 0.606 over 30 s): that was **dimers dying and
taking bonds with them**, not bonds dissolving.

**Consequence:** any structural gating must act on **formation** (`em_rate`), not decay. PO-5 aimed
at the wrong term.

**⚠ The scrambled control arm is CONFOUNDED and its numbers must not be cited.**
`np.random.permutation(delta)` draws from the **global** NumPy RNG, shifting the downstream random
stream, so arm C is a different random realisation rather than a matched control. Its
`(B−C)/spread` flips sign across bus values (+1.07, −0.71) — noise. **The verdict does not depend on
it**: REAL vs OFF is a clean within-arm null. Needs a dedicated RNG before arm C means anything.

**Per the pre-registration, the honest conclusion is that PO-5's proposed mechanism is wrong.**
`j_mismatch_scale` is NOT re-registered at a friendlier value. Recommendation on file: revert the
edit rather than keep a dead mechanism in source.



### L·PO5-5 — the BUS is a real percolation control parameter (60 -> 1 components), and the system natively sits PAST the transition · 2026-07-19 `[GROUNDED, measured]`

**Probe:** `sweep/po5_unit4_bus_percolation.py` (predictions registered in its header before the
run). **Raw:** `sweep/po5_unit4_bus_percolation_results.json`, `po5_unit4_run.log` — **committed
this time**, not left in gitignored `results/`.

#### The architecture, verified in code — not an analogy

`model6_core.py:555` `self._collective_field_kT = trp_state['output']['collective_field_kT']`, and
`dimer_particles.py:454` `em_rate = k_base * (collective_field_kT / reference_kT) * coh * g`.
**The tryptophan module is the network; `collective_field_kT` is a single global scalar gain on
every pair's bond rate; dimers are the bits.**

#### Measured (single synapse, 1 s, seed 7777, override applied at `dp.step`)

| bus (kT) | P2 bonds | components | largest_frac | λ₂ |
|---|---|---|---|---|
| 0.00 | 0 | **60** | 0.8866 | 0 |
| 0.10 | 319 | 44 | 0.9636 | 0 |
| 0.25 | 744 | 39 | 0.9679 | 0 |
| 0.50 | 1447 | 29 | 0.9763 | 0 |
| 1.00 | 2823 | 20 | 0.9839 | 0 |
| 2.00 | 5586 | 14 | 0.9890 | 0 |
| 5.00 | 13447 | 7 | 0.9949 | 0 |
| 10.00 | 24828 | 6 | 0.9958 | 0 |
| 20.00 | 44459 | **1** | 1.0000 | 0.9106 |
| **NATIVE 21.98** | — | **1** | 1.0000 | 0.9807 |

**Positive control PASSED:** bus = 0 gives exactly **zero** P2 bonds and `components_all ==
components_P0only`, so the override demonstrably acts on the intended term.

#### Three findings

**1. A TRANSITION EXISTS.** Components fall monotonically 60 → 1 as the bus rises. The architecture
**does** have an operating point at which the topology can hold information. This was the open
question and the answer is yes.

**2. THE SYSTEM IS PARKED PAST IT.** The tryptophan module natively produces **21.98 kT** (against
`reference_kT = 20.0` and `FIELD_THRESHOLD_KT = 20.0`), which yields **one component**. The
informative band is roughly bus **0.1–5**, giving 7–44 distinguishable components. **The model runs
at least ~4× above the top of that band.** The mechanism is not incapable; it is **in the wrong
phase**.

**3. THE READOUT IS BLIND AT THE NATIVE OPERATING POINT — now measured, not argued.** λ₂ is
**exactly 0** in every fragmented state (λ₂ > 0 iff connected) and only becomes informative once
connected (0.9106, 0.9807). So component-count and λ₂ are exactly complementary: in the fragmented
phase components carry everything and λ₂ nothing; in the connected phase components are pinned at 1
and **all** remaining structure is in λ₂. **The live path reads only component count
(`dim ker L₀`), and the system sits in the connected phase — it reads the one channel provably empty
at its own operating point.** Per A5, λ₂ is used here strictly as a **diagnostic**, never proposed
as the readout.

#### What this does NOT show

**The condensate is NOT reconnected.** This probe overrides the bus directly. It says nothing about
whether `backbone_eta * E_invasion` (`model6_core.py:543`) can reach it — and both factors measure
0.0000 in every live trial (η in this run's own backbone diagnostic; `E_invasion` per PO-4). **No §8
verdict** — this is about whether an operating point exists, not about whether input reaches it.

#### Limits
Single synapse, 1 s, one seed per bus value. `components_P0only = 60` counts non-template-bound
dimers as singletons (Unit 3 counted only template-bound), so P0 component counts are not directly
comparable between the two units; within this sweep the comparison is consistent.



### L·PO5-4 — the P0 graph IS an indifference graph on birth time (5/5 predicted==measured), births are BURSTY not continuous, and P2 is what erases the structure · 2026-07-19 `[GROUNDED, measured — REPORTED, log did not survive]`

**Pre-registered:** `docs/PREREG_PO5_UNIT3_BIRTH_COHORTS.md`, committed `becc8e3` **before** the run.
**Probe:** `src/models/Model_6/sweep/po5_unit3_birth_cohorts.py`, same commit.
**Reopened on Sarah's direction after the PO-5 seat closed.**

⚠️ **PROVENANCE — read this before citing.** The run executed, but its log lived in `results/`,
which is gitignored, and **the `nervous-hertz-7ccff6` worktree was removed by the consolidation while
the run was still in flight.** The numbers below were read directly from the probe's live output and
are **REPORTED, not MO-VERIFIED**, per `CONSOLIDATION_2026-07-19.md`'s rule. **The probe and its
pre-registration are committed, so this is fully reproducible by re-running it** — that is the
correct way to promote these to MO-VERIFIED. This is exactly the loss PO-5's closing heartbeat
warned about (§3a-ii), landing on PO-5's own final run within hours.

#### The registered prediction, and it held

> **PREDICTED components of the P0-only bond graph = `1 + count(birth-time gaps > 0.1 s)`**

| arm | t | template-bound | distinct births | max gap | gaps >0.1s | predicted | measured | |
|---|---|---|---|---|---|---|---|---|
| SUSTAINED | 1.0 | 1022 | 12 | 0.4700 s | 1 | 2 | 2 | **MATCH** |
| SUSTAINED | 3.0 | 1044 | 14 | 0.7700 s | 3 | 4 | 4 | **MATCH** |
| SUSTAINED | 5.0 | 1046 | 16 | 2.6650 s | 5 | 6 | 6 | **MATCH** |
| PULSED | 1.0 | 1013 | 12 | 0.7550 s | 1 | 2 | 2 | **MATCH** |
| PULSED | 3.0 | 1087 | 17 | 0.7550 s | 4 | 5 | 5 | **MATCH** |

**5 of 5 samples matched exactly. `max_glu = 1.000` (positive control fired).** The PULSED arm's
final sample was not observed before the worktree was removed.

**The mechanism is confirmed:** the P0 bond graph is a **unit-interval (indifference) graph on the
birth-time axis**, exactly as derived from `dimer_particles.py:218-228` + `:210`. Its components are
the maximal runs of births with no gap > 100 ms — predicted from birth times alone, with no fitted
quantity, and reproduced exactly at every sample.

#### PO-5's own reasoning was WRONG, and the measurement corrected it

PO-5 predicted in prose that *"under sustained drive, births occur at most 5 ms timesteps, so gaps
are ≪ 100 ms ⇒ one component."* **False.** Births are **bursty**: only **12–17 distinct birth times
in 5 s**, with gaps up to **2.665 s**. **The P0 graph has SIX components under sustained drive, not
one.** Each birth event creates ~60–90 dimers at once, so the graph is a handful of large cliques,
not a continuum.

#### The consequence, and it inverts the earlier picture

The **full** graph was measured at **`comps = 1`, `largest_frac = 1.000`** (`L·PO5-1`, corroborating
probe). The **P0-only** graph has **6**. Since P0 is 82.86% of bonds and P2 is 17.14% (`L·PO5-2`):

> **The temporal cohort structure EXISTS — and the 17% spatially-mediated P2 bonds are what BRIDGE
> the cohorts into a single blob.**

So the intra partition is not trivial because formation is structureless. It is trivial because a
**spatially promiscuous minority pathway erases the temporally structured majority.** The pair-level
information is created and then washed out.

#### Also observed, weakly

PULSED did **not** produce more >100 ms birth gaps than SUSTAINED (max gap 0.755 s vs 0.770 s at
t = 3.0). **Two samples, one seed — weak.** It suggests birth timing is governed by the
concentration/supersaturation dynamics rather than by instantaneous drive, which would mean input
cannot gate formation timing directly. **NOT established; flagged for measurement.**

#### What is NOT claimed

**No §8 verdict.** This tests a mechanism. Whether input can modulate the cohort structure — and
whether that survives P2 bridging into the final partition — is unrun. **Do not read "P2 erases the
structure" as "§8 fails":** it locates where the structure goes, not whether input put any there.

**Do NOT respond by weakening P2 to preserve cohorts.** That would be tuning a constant to reach an
outcome (`MO_MODEL6.md` §7 LOCKED). The bridging is the physics as written; if it erases the
partition, that is the finding.

#### Limits

Single synapse, 5 s, one seed per arm, 3 samples (SUSTAINED) + 2 (PULSED). REPORTED, not
MO-VERIFIED. Re-run `po5_unit3_birth_cohorts.py` to promote.



### L·PO5-3 — Q-B ran, every gate passed, and it returned NO VERDICT. The statistic was never comparable across runs. · 2026-07-18 `[GROUNDED, measured — null result about the INSTRUMENT]`

**Pre-registered:** `docs/PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md` incl. AMENDMENTS A2.2–A2.6, all
committed before the run. **Probe:** `sweep/po5_unit2_qb_selectivity.py`.
**Raw:** `results/po5/unit2_qb_SCORER_CRASH_cell6.log`, `sweep/po5_unit2_qb_results.json` (per-run
metadata only — see flaw 3). **Ran on the exclusive heavy slot granted by MO ruling 019.**

**SCORED VERDICT: NONE. `ratio` was never computed. Nothing is claimed about §8 in either
direction.** The §8 keystone remains exactly as unverified as before this run.

#### What ran, and what passed

All 9 runs completed in **3492 s (58.2 min)** — inside the ~90 min estimate. **Every registered gate
passed:**

| gate | result |
|---|---|
| instrument conservation (provenance ≡ `_bond_lookup`) | **PASS** |
| A2.3 `_remove_dimer` tripwire | **PASS** — zero calls, confirming the defect is unreachable |
| positive control `max_glu > 0` every run | **PASS** (min 1.000) |
| drive matching A vs B (registered ≤5%) | **PASS** — A = 2.7540, B = 2.7460 → **0.3%** |

**The physics ran and the guards worked.** Then scoring raised
`ValueError: operands could not be broadcast together with shapes (169,) (36,)`.

#### Three flaws, and the crash is the least of them

**1. The statistic was never comparable across runs — a DESIGN flaw, not a coding slip.** Cells were
indexed by *each run's own* occupied set (`remap` = that run's sorted occupied cells), so index *i*
denotes a **different physical location in every run**. A Frobenius distance between two such
matrices is meaningless **even when the shapes coincide** — meaning that had cell counts happened to
match, this would have produced a *confident number that was silently garbage* instead of a crash.
**The crash is the lucky outcome.**

**2. The pre-flight was unrepresentative.** Occupied cells ranged **6–14 across seeds** (13, 6, 6,
14, 6, 7, 8, 12, 9); only **3 of 9** runs cleared `MIN_CELLS = 10`. A2.6's pre-flight sampled **one**
seed, read 13, and certified the arm. Same error PO-3 named in withdrawing F-3: *"I read a single
sample as confirmation of a mechanism I had not measured."*

**3. The scored intermediate was not persisted, so a SCORING bug destroyed 58 minutes of PHYSICS.**
The `P_bond` matrices lived in memory and were excluded from `persist()`. **`sweep/score_leta5.py`
already solved this** — PO-3 scores offline from a persisted trace, and L·ETA-5 records that the
in-run verdict was void while the offline scorer was authoritative. **That prior art was named in
PO-5's own grounding brief and not composed from.** This is the reinvention failure, and it is what
turned a small bug into an hour of the program's only heavy slot.

#### Why this is logged as a result rather than swept

L·ETA-5 set the precedent: a properly-conducted measurement that does not answer its question is a
result **about the instrument**. This one is weaker than L·ETA-5's — that run's null arm carried real
physics, this one produced no comparable quantity at all — but the record belongs here, because the
next PO to build a cross-run spatial statistic in this codebase needs to know that **per-run
occupancy indexing does not survive the comparison step.**

#### What changes before any re-run — no compute required

Fixed **global** lattice (absolute cell coordinates, so a cell is the same place in every run);
comparison restricted to the cell set occupied in **all** runs; matrices **persisted**; scoring split
into a **separate offline scorer**, composed from `score_leta5.py`; scorer **validated on synthetic
data with known answers** before physics is spent.

**Registered thresholds are NOT being moved.** `MIN_OCC = 5`, `MIN_CELLS = 10`, `RATIO_CONFIRM = 3.0`,
`RATIO_FALSIFY = 1.5` and the A2.2 precedence all stand. **If the all-run intersection falls below
`MIN_CELLS`, the honest verdict is that the instrument cannot resolve pair structure in this
geometry** — reported as a finding about the measurement, per the registered hard stop.

#### What survives and is reusable

The instrumentation layer is validated and unaffected: provenance conservation exact, the A2.3
tripwire confirmed the `_remove_dimer` defect unreachable under a full 9-run protocol, drive matching
holds to 0.3%, and the release-inside-the-physics-loop pattern reproduced `max_glu = 1.000` in every
arm — the positive control `mo-f3-001` requires.

### L·PO5-2 — **83% of the bond set is formed by a THIRD mechanism neither pathway decomposition names, and it is deterministic** · 2026-07-18 `[GROUNDED, measured]`

**Pre-registered:** `docs/PREREG_PO5_UNIT2_PAIR_SELECTIVITY.md` §2 (committed before the probe ran).
**Probe:** `src/models/Model_6/sweep/po5_unit2_provenance.py`.
**Raw:** `src/models/Model_6/sweep/po5_unit2_provenance_results.json`.
**Scope:** this is **Q-A (provenance), descriptive.** It is **not** an input-selectivity result and
is not reported as one. Q-B is unrun.

#### The instrument gate FAILED first, on real data, and the failure was diagnostic

The first instrumented run **failed its own registered conservation check** — orphaned provenance
entries growing 0 → 909 → 4851 across samples. **Cause traced, not guessed:**
`_remove_all_bonds_for_dimer` (`dimer_particles.py:245`, called from the death path at `:239`) pops
`_bond_lookup` **directly** and never routes through `_remove_bond`, so the wrapper never saw those
removals. Registered as **AMENDMENT A2.1**, instrument fixed, physics untouched. Failing run
preserved at `results/po5/unit2_provenance_FAILING_v1.log`.

Post-fix, **both registered gates pass**: conservation exact at all three samples (`missing = 0,
orphan = 0` against 415643 / 436683 / 474256 live bonds), and instrumented vs uninstrumented runs at
the same seed agree **bit-for-bit** on `n_dimers`, `n_entangled`, `n_bonds` and `mean P_S`.

#### The measurement — live bonds by originating mechanism (single synapse, −10 mV, 2 s, seed 20260718)

| t | n_bonds | P0 birth-inherit | P1 burst | P2 EM |
|---|---|---|---|---|
| 0.5 | 415643 | 396659 | 1 | 18983 |
| 1.0 | 436683 | 396003 | 4 | 40676 |
| 2.0 | 474256 | **392952 (82.86%)** | **22 (0.00%)** | **81282 (17.14%)** |

Classification is **exact, not statistical**: `dimer_particles.py:439` sets
`p1 = both_ent & same_burst & both_tmpl & ~has_bond` and `:450` sets `p2 = both_ent & ~p1`, so within
`step_entanglement` a newly formed bond took Pathway 1 iff `same_burst & both_tmpl`. Phase is
separated by wrapping `step_population` vs `step_entanglement`. No RNG replay, no guessed branch.

#### Two structural findings

**1. The dominant mechanism is a third site that neither pathway decomposition names.**
`dimer_particles.py:218-228`, inside the birth loop:

```python
if template_bound:
    birth_window = 0.1
    for other in self.dimers[:-1]:
        if other.template_bound and other.is_entangled:
            if abs(other.birth_time - dimer.birth_time) < birth_window:
                self._create_bond(dimer.id, other.id, strength=...)
```

**It is deterministic — there is no rate, no RNG draw, no distance term, and no coherence gate
beyond `is_entangled`.** Every template-bound dimer born within 100 ms of another is bonded to it,
unconditionally. That is a near-complete blob **by construction**, and it produces **83%** of the
live bond set.

**This matters for the keystone's framing.** The kickoff, `mo-rescope-001.md:49-53` and
`quantum-computation-and-attribution` §7 #1 all decompose `em_rate` into `g` / `collective_field_kT`
/ `coh` and locate the keystone in `coh`. **83% of bonds never evaluate `em_rate` at all.** Unit 1
measured `g`'s dynamic range at `D = 33.5` — that 33× spread is being applied to the 17% minority.

**2. Pathway 1 is almost entirely shadowed — 22 bonds, 0.00%.** The cause is structural and follows
from the code: `p1` requires `~has_bond`, and the birth loop has *already* bonded every same-burst
template-bound pair. **P0 pre-empts P1 by construction.** So the `p1` branch at `:437-444`, which the
documentation treats as one of two pathways, is very nearly dead code under these conditions.

#### What is NOT claimed

**Whether this defeats §8's keystone is NOT measured here, and the temptation to say so is exactly
the inference `L·PO5-1` CORRECTION 1 already withdrew once.** Birth timing and template binding are
themselves downstream of input (calcium → dimer concentration → births), so a deterministic
birth-pairing rule is **not automatically input-blind**. Whether it carries **pair-level** input
information — as against §8's *"gate-level … which regions/timings are eligible"* — is precisely
Q-B, and Q-B has not been run. **No verdict on the keystone is stated or implied.**

Note also `mo-ruling-001` §3: birth-pairing is *"arguably a more natural home for input-dependent
pair structure than Pathway 2's EM-mediated route."* This measurement says that home holds 83% of
the residents; it does not say what they encode.

#### A latent defect found in passing — reported, not fixed

`_remove_dimer` (`dimer_particles.py:252-261`) discards bonds from `self.entanglement_bonds` but
**never pops `self._bond_lookup`**, so the two containers would diverge if it ran. **It is currently
dead code** — `grep -n "_remove_dimer"` returns only its definition, no call sites — so nothing is
broken today. Routed to the MO rather than fixed: it is a death-path function, not the Pathway 2
formation path PO-5 owns.

#### Limits

Single synapse, one drive condition, one seed, 2 s. The 83/17 split is a property of **these**
driving conditions; a regime with more spread-out births would shift it. Provenance shares are
reported for the **live** bond set at each sample, not cumulative creations (both are in the JSON).

### L·PO5-1 — `g` is LIVE, both priors were wrong, and the pair-resolution does not reach the topology · 2026-07-18 `[GROUNDED, measured]`

**Pre-registered:** `docs/PREREG_PO5_UNIT1_G_INERTNESS.md`, committed `cc80fcc` **before** the probe
existed. **Probe:** `src/models/Model_6/sweep/po5_unit1_g_inertness.py` (`1dbef17`).
**Corroborating trace:** `sweep/observe_pathway2_selectivity.py` (pre-existing; PO-5 reused it rather
than rebuilding, and it is the source of the component counts below).
**Raw (committed):** `src/models/Model_6/sweep/po5_unit1_g_inertness_results.json` and
`..._run.log`. Persisted beside the probe, not under `results/`, which is gitignored — an artifact
written there would not survive as provenance.

#### Why this unit ran first

`dimer_particles.py:453` clamps with `np.maximum`, so every pair closer than
`coupling_length = 5.0` nm (`:129`) receives `g = 1.0` exactly. The 1/r³ can therefore be present in
code and still carry no pair information — by saturation (`g ≈ 1` everywhere) or by vanishing
(`g ≈ 0` everywhere). Which one holds changes the meaning of every later selectivity result.

#### The verdict function was demonstrated failing before it was allowed to pass

`demonstrate_verdict()` runs before the model is constructed and requires all four labels on
synthetic input with known answers. Shown ABORTing (exit 1, three MISMATCHes) with `SAT_THRESHOLD`
deliberately set to 0.0, then passing 4/4 at the registered thresholds. Per `MO_MODEL6.md` §2.3 and
the `683b82f` scar: a verdict that cannot distinguish its outcomes is not a result.

#### Measured — single synapse, −10 mV, 5 s, dt = 0.005

| t | n_ent | n_pairs | f_sat | r_p10 | r_p50 | r_p90 | r_max | g_p10 | g_p90 | D | sat_bonds | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1082 | 584821 | 0.1762 | 3.70 | 9.75 | 16.11 | 36.45 | 2.99e-2 | 1.000 | 33.46 | 0.7476 | LIVE |
| 1.0 | 1080 | 582660 | 0.1761 | 3.70 | 9.74 | 16.11 | 36.45 | 2.99e-2 | 1.000 | 33.44 | 0.7783 | LIVE |
| 2.5 | 1074 | 576201 | 0.1761 | 3.70 | 9.74 | 16.10 | 36.45 | 3.00e-2 | 1.000 | 33.38 | 0.8315 | LIVE |
| 5.0 | 1110 | 615495 | 0.1745 | 3.71 | 9.78 | 16.18 | 36.45 | 2.95e-2 | 1.000 | 33.89 | 0.8144 | LIVE |

**`VERDICT: LIVE-under-stated-conditions.`** `f_sat` moves by 0.0017 across the four samples, so the
quantity is not transient-sensitive at this dt (PREREG §6 asked whether it moves; it does not).

#### Both standing predictions about `g` were wrong, and one of them was PO-5's own

- **The board's:** *"if intra-synapse `r_ij` mostly sits under 5 nm, the 1/r³ is present in code but
  **inert in practice**"* (`board.md:919-922`; same in `mo-rescope-001.md:55-59`). **Refuted** —
  only **17.6%** of pairs sit inside the clamp, against a registered saturation bar of 90%.
- **PO-5's own grounding brief**, which predicted the opposite failure — `g ≈ (5/150)³ ≈ 3.7e-5`,
  inert by *vanishing*, reasoning from the 400 × 400 × 20 nm birth domain
  (`dimer_particles.py:199-203`). **Also refuted, and by more.** Dimer births are
  concentration- and template-weighted (`:189-196`), so dimers cluster far tighter than the domain
  implies: `r_p50 = 9.75` nm, not ~150. **The brief's a-priori was wrong by roughly 15× in `r`**;
  recording it because it was stated as a located prediction before the measurement and the trail is
  worth more than the tidier version.
- **The prose that was right:** `model6-entanglement-partition-werner:60` — *"their near-field 1/r³
  coupling is at the nm scale … would wrongly cut working intra edges at ~7 nm."* Measured
  `r_p10 = 3.70`, `r_p50 = 9.75`. The brief flagged this skill line as in tension with the geometry;
  **the tension resolves in the skill's favour** and no correction is owed to it.

#### The consequence, which is the actual finding

`g` varies **33×** across the pair set — genuinely pair-resolved, not a constant. And it does not
matter yet, because of what the graph looks like:

- realised intra bond saturation **0.75 – 0.83** (this probe, four samples);
- **`comps = 1`, `largest_frac = 1.000`** at t = 5 s and t = 10 s (corroborating probe) — every
  entangled dimer in **one** connected component;
- bonded-pair median separation **9.5 nm** vs all-pair median **10.3 nm**; p90 **15.2** vs **16.5**.
  The bonded set is only marginally closer than the pair set at large.

**A rate that varies 33× across pairs is producing a near-complete graph with a trivial partition.**
`model6-entanglement-partition-werner` (LOCKED) puts the computation in the **partition** — one
commit per connected component. Pair-resolution that lives in `em_rate` but does not survive into the
component structure buys §8's keystone nothing. This does **not** settle the keystone (that needs
*input*-dependence, and `g` is geometry), but it relocates where the keystone can fail: not at the
rate, at the realised graph.

It also updates `entanglement-topology-measurement:287-289`'s *"Still at ~77% saturation … F2
carryover"* (May 14) — **still true today, measured 0.75–0.83**, two months and a 1/r³ landing later.

#### UNVERIFIED — stated as such, not filled in

**Which pathway produces the saturation is not attributed.** The corroborating probe's **first**
sample, at t = 0.0, already reads `sat = 0.9440` over 493 dimers, and Pathway 1 birth entanglement
(`dimer_particles.py:218-228`) bonds every template-bound dimer pair born within a 100 ms window with
no distance term at all — so a Pathway-1 origin is *plausible* and is exactly the kind of plausible
attribution this program's defect class punishes. **It is not claimed.** Separating Pathway 1 from
Pathway 2 in the realised bond set is PO-5 Unit 2.

#### ADDENDUM — the corroborating trace completed to 30 s, and the graph is NOT static

`sweep/observe_pathway2_selectivity.py`, full run (single synapse, −10 mV, 30 s):

| t | n_ent | bonds | sat | bonded_med | all_med | comps | largest_frac |
|---|---|---|---|---|---|---|---|
| 0.0 | 493 | 114481 | **0.9440** | 10.0 | 10.2 | 15 | 0.972 |
| 5.0 | 1132 | 480369 | 0.7504 | 9.5 | 10.2 | 1 | 1.000 |
| 10.0 | 1110 | 480562 | 0.7808 | 9.5 | 10.3 | 1 | 1.000 |
| 15.0 | 1124 | 456378 | 0.7231 | 9.4 | 10.4 | 1 | 1.000 |
| 20.0 | 1086 | 414162 | 0.7030 | 9.3 | 10.5 | 3 | 0.998 |
| 25.0 | 1123 | 394135 | 0.6256 | 9.1 | 10.7 | 2 | 0.999 |
| 30.0 | 1104 | 368747 | **0.6056** | 9.0 | 10.9 | 1 | 1.000 |

**Two monotone trends, measured:** saturation **falls 0.944 → 0.606**, and the bonded/all-pair
median separation gap **widens from 0.2 nm to 1.9 nm** (bonded drifts down 10.0 → 9.0 while all-pair
drifts up 10.2 → 10.9). **The bond set becomes progressively more distance-shaped over 30 s.**

`comps` stays at 1–3 with `largest_frac ≥ 0.998` throughout — consistent with
`quantum-system-canonical:139`'s LOCKED expectation for a single synapse, and **not** read here as a
finding either way (see CORRECTION 1).

**Directional, and explicitly NOT an attribution.** A near-complete graph at t = 0 eroding toward a
distance-shaped one is *consistent with* Pathway 1 (birth-pairing, no distance term,
`dimer_particles.py:218-228`) laying down the initial blob and Pathway 2's 1/r³ plus dissolution
shaping it thereafter. **That mechanism is NOT measured and is not claimed** — the same trap Unit 1
already refused once. It is what Unit 2 tests.

#### ⚠ CORRECTION 1 — PO-5's own, 2026-07-18, after MO ruling 010. **The measurements stand; one INFERENCE above is wrong-layer and is withdrawn.**

**What is withdrawn:** the framing that the single connected component means *"the pair-resolution in
the RATE does not reach the TOPOLOGY"* and that the partition is *"trivial."* That is a
**conclusion, not a measurement**, and it reads the intra-synapse layer against a standard that
belongs to the network layer.

**`quantum-system-canonical:139` [LOCKED], verbatim:**

> *"It lives at the NETWORK scale (synapses = nodes, cross-synapse bonds = edges), **forced by
> physics**: one synapse is one nanodomain = one dense dimer cloud = one component. **A
> single-synapse "one giant component" is correct physics, not a bug.** The meaningful,
> input-dependent partition is over *which synapses* condense and cross-bond into which
> components."*

So `comps = 1` at a single synapse is **the predicted result**, not evidence against anything. PO-5
measured it, then inferred a consequence the ontology had already ruled on — the shape gen-1
recorded as **defect #16: premise verified correctly, conclusion drawn wrongly.** Caught by the MO,
not by PO-5.

**What SURVIVES unchanged, because it was measured rather than inferred:** `f_sat = 0.176`,
`D = 33.5`, `r_p10/p50/p90 = 3.70/9.75/16.11 nm`, bond saturation **0.75–0.83**, `comps = 1`,
`largest_frac = 1.000`, bonded median 9.5 nm vs all-pair 10.3 nm, and the `LIVE` verdict. Every
number in the table is untouched. The refutation of both `g` priors is untouched.

**What replaces the withdrawn inference — a question, not a claim.** §8 asks for **pair-level**
selectivity and `quantum-computation-and-attribution` §7 #1 calls the keystone
**"single-synapse-scale — needs no backbone."** §5 LOCKS the meaningful input-dependent partition as
**cross-synapse**, with one-component-per-synapse as correct physics. **Those two are in tension and
PO-5 does not resolve it unilaterally** — it is raised to the MO in
`requests/model6-mo/po5-selectivity-002.md`. Stated plainly: if a near-complete intra graph is
correct physics, then "which dimers bond" *within* a synapse is near-flat **by construction**, and
whatever §8's pair-level selectivity means at single-synapse scale, it cannot mean the intra
component structure.

#### Limits

Single synapse, one drive condition (−10 mV sustained), one seed, 5 s. `f_sat` is a property of these
driving conditions — a different calcium/template regime places dimers differently, so the verdict is
`LIVE`-**under-stated-conditions** per PO-1's formulation (`board.md:804-806`). `dt-1` records that
dimer *count* is not converged in the drive transient; `f_sat` is a ratio over pairs and was measured
stable across four samples, but no dt sweep was run and none is claimed.

### L·ETA-5 — the ratchet test is VOID; its NULL arm is the result: `E_invasion` accumulates without activation · 2026-07-18  `[GROUNDED, measured]`

**Pre-registered:** `docs/PREREG_L_ETA_5_RATCHET.md` (`2084960`, before the run; AMENDMENT 1
before the run, AMENDMENT 2 + CORRECTION 1 before scoring). **Probe:**
`sweep/einvasion_ratchet_probe.py`. **Scored offline:** `sweep/score_leta5.py` — the in-run
verdict used superseded gate logic and is void. **Run pinned to `1b43b89`** in a clean
checkout, because PO-1's in-flight `vibrational_cascade_module.py` edit blocked model
construction in the shared worktree.

**SCORED VERDICT: `INCONCLUSIVE — NULL ARM RATCHETED`. The measurement is VOID on its own
registered terms.** The dispatch question — does `E_invasion` ratchet across traversals — is
**NOT ANSWERED** by this run.

#### What was measured (8 traversals, 14 s each, 20 s gaps, real physics through every gap)

| t | drive `enl` | null `enl` | ratio | drive `E_inv` | null `E_inv` | drive `r` | null `r` |
|---|---|---|---|---|---|---|---|
| 1 | 0.4440 | 0.0722 | 6.15 | 0.1944 | 0.0000 | 0.2488 | 0.1157 |
| 2 | 0.7757 | 0.2803 | 2.77 | 0.3817 | 0.1019 | 0.6575 | 0.2809 |
| 3 | 1.0590 | 0.4651 | 2.28 | 0.5418 | 0.2063 | **1.0721** | 0.5498 |
| 4 | 1.2754 | 0.7066 | 1.80 | 0.6640 | 0.3427 | 0.9911 | 0.4914 |
| 5 | 1.3755 | 0.7733 | 1.78 | 0.7206 | 0.3804 | 1.0344 | 0.5889 |
| 6 | 1.4516 | 0.8419 | 1.72 | 0.7636 | 0.4191 | 1.3643 | 0.9777 |
| 7 | 1.4677 | 0.8977 | 1.63 | 0.7727 | 0.4507 | 1.3729 | 0.6786 |
| 8 | 1.4915 | 0.8796 | 1.70 | 0.7862 | 0.4405 | **1.4050** | 0.8631 |

`conf = 0.0000` at every gap in both arms, so the uncommitted retention branch
(`rho_pred = 0.8948`) is the correct one throughout — the AMENDMENT 2 correction did not bite
on this run, but it was required before scoring and would have inverted the verdict had the
traversals committed.

#### Why it is VOID

PREREG §7 registered a null arm that **cannot show the effect**: the target held below the
0.05 activation floor, expected to hold `E_invasion = 0.0000` and a flat `peak_r`. Both
criteria failed:

- **null max `E_invasion` = 0.4507** (registered: must stay 0.0000)
- **null `peak_r` gain = 7.46×** (registered: void at ≥ 1.2×) — *larger than the drive arm's
  5.65×*

**The null arm ratcheted harder than the driven arm.** `E_invasion` in a synapse that was
never activated climbed past `invasion_threshold` by traversal 2 and reached 0.45; its `r`
reached 0.978, within 2% of the condensation threshold.

**The design error is mine.** The null suppressed *activation* (`acts[target] = 0`) but not
*presynaptic release*: `PresynapticRelease` fires at `BASELINE_RATE_HZ = 0.5` independently of
activation (`presynaptic_release.py:124`, `rate = baseline_rate + a*peak_rate`). A synapse
receiving glutamate is not a null.

#### The finding that survives, and it is about the driver itself

**In this model, `E_invasion` accumulates past `invasion_threshold` on tonic spontaneous
release alone, with no activation at any point.** This is not a probe artifact — spontaneous
release is a modeled physiological process, and the accumulation is the actin integrator doing
what it is written to do. Read off the trace:

- The undriven synapse's `actin_enlargement` grew **during silent gaps** (`rho` up to 2.26 at
  gap 1) — formation exceeded extrusion with the agent parked where max activation measured
  0.0000.
- Working back through `formation = k_poly·f_CaM·(S/S0)·room` (`:386`), sustaining the
  observed +0.091 over a 20 s gap needs `f_CaM ≈ 0.045`, i.e. calcium held near **0.47 µM**
  through a gap designed to be silent, against a ~0.1 µM baseline. *(Inferred from the rate
  equation, NOT independently measured — labelled as such.)*
- **The driven/undriven separation collapses with traversal count:** 6.15× → 2.77× → 2.28× →
  1.80× → 1.78× → 1.72× → 1.63× → 1.70×. The driven arm saturates into the room-limited
  regime while the undriven arm keeps accumulating.

**Consequence for input-selectivity (PO-5).** L·ETA-4 found `E_invasion` at a silent synapse
*identical to the driven one* under a plateau (0.2115 vs 0.2115) and concluded selectivity
cannot ride the η channel. This is a **stronger and more general** version of that result: **no
plateau is required.** Tonic release alone carries `E_invasion` to within a factor ~1.7 of the
driven value. Selectivity in the `E_invasion → r → η` channel is not merely destroyed by the
plateau; it is weak on this driver's own dynamics at these timescales.

#### What did NOT happen, and must not be read into this

- **`r` crossing 1.0 does NOT unblock PO-5.** The driven arm crossed threshold (1.0721 at t3,
  1.4050 at t8) — the first live-regime crossing after L·ETA-3's 13× shortfall — but
  **cross-synapse edges = 0 in BOTH arms, all run.** `k_cross ∝ √(η_i·η_j)` and this design
  drives exactly one feature, so every cross term vanishes by construction. **η ≠ 0 was
  demonstrated; a PARTITION was not.** PO-3 → PO-5 is at most partially cleared.
- **This is not the pre-registered negative branch.** FALSIFIED required gain < 1.2 or
  ratio_mean < 0.5; measured gain was 5.65×. There is no "condensate cannot track behavioural
  timescales" negative result here to route to Sarah — the run is void, not negative.
- **Absolute values are not comparable to L·ETA-3's** (different release call pattern, AMENDMENT
  A1.1) and **absolute `r` is not comparable across B2** (`P_c` depends on backbone `omega_0`/`Q`,
  which PO-1 is changing). The pre-registered quantities were ratios for this reason.

#### Two of the registered gates would ALSO have failed, recorded so the design is judged whole

Had the null passed, the drive arm still would not have returned CONFIRMED:

- `peak_r` **non-monotone** (t3 1.0721 → t4 0.9911), failing clause (i). `r ∝ E_inv × ca_open`
  and `ca_open` is stochastic; the scatter is visible in both arms (null t6 0.978 → t7 0.679).
- `ratio_mean = 1.1080`, **outside** the registered band `[0.89, 1.07]`, failing clause (ii).
  Raw `rho_mean = 0.9915` would additionally have tripped GATE 1's `≥ 0.99` artifact gate.

The retention overshoot is the calcium tail flagged in AMENDMENT A1.2 **before the run**: a
20 s gap after a 14 s traversal is not a clean decay window because formation has not stopped
when the gap begins. `rho` converges downward toward the predicted 0.8948 across the run
(1.068 → 1.052 → 1.037 → 0.952 → 0.958 → 0.941 → 0.933) exactly as that mechanism implies.
**A cleaner test needs a gap long enough for calcium to clear before retention is scored.**

#### Errors made and corrected in this cycle (all recorded, none found by PO-3 first except the last)

1. **"`analytical_gap` freezes actin" — WRONG.** Its tail runs `network.step(0.001, ...)`;
   actin advances 1 ms per gap (retention 0.9999944, not 1.0). Concluded from absence in the
   docstring's two lists without reading the function tail, and reported tagged `[code SHOWN]`.
   Caught by PO-4. (CORRECTION 1.)
2. **The committed-branch retention prediction — MIS-DERIVED.** `conf` gates extrusion off at
   `:389` *and* retention on at `:390`; both drain `actin_enlargement`, which is what
   `E_invasion` reads (`:412`). A committed spine drains **3.54× faster**, not slower. Caught
   by PO-4, ruled by the MO, fixed before scoring. (AMENDMENT 2.)
3. **F-3 "the L·ETA-3 harness starves the NMDARs ~100×" — OVERSTATED, and it inverts.**
   Measured: 19.0 vs 1.0 release events per traversal (**19×**, not ~100×), and the L·ETA-3
   pattern *holds* each release for 100 physics steps, so on exposure duration it delivers
   **more** glutamate, not less. The ~350/~3.3 figures came from `rate × time`, ignoring the
   10 ms refractory and vesicle depletion. **The recommendation that L·ETA-3 carry a correction
   banner is WITHDRAWN; its `ca_open` attribution stands unchallenged.**

All three were mechanism claims asserted from arithmetic rather than measurement — the defect
class this sub-program is named for.


#### ADDENDUM (post-hoc, MO ruling 004) — the gap IS stepping, and both failures are ONE mechanism

`sweep/gap_clock_assert.py`, run after scoring. **Clock delta over a 20 s gap = 20.0000 s,
assertion PASS** — the target's own `spine_plasticity.time` advanced in full while inactive, so
**D19's "only active synapses are stepped" false-ratchet mechanism is RULED OUT for this probe**
(`step_network_per_synapse` carries no active mask, unlike the shipped `run_trial:434-441`).

**Recorded against interest:** `rho_mean` was 0.9915, so had the null arm passed, **GATE 1 would
have returned `INCONCLUSIVE — GAP NOT STEPPING`, a FALSE DIAGNOSIS.** A retention threshold is a
symptom standing in for a mechanism; ruling 004's clock assertion is the proof. Any re-run should
assert the clock directly.

**And A1.2's stated mechanism was wrong.** `rho > 1` is *not* a decaying calcium tail. Measured
through the gap, `f_CaM ≈ 1.6e-4` and the net is **negative** (clean extrusion at `tau_extrude`)
except at discrete moments when calcium spikes 0.11 → **3.13 µM**, saturating `f_CaM` to 0.99 and
driving formation bursts ~5000× baseline (`+0.0432` vs a steady `−0.0024`). Those are **spontaneous
release events**.

**This unifies the two failures.** The spontaneous-release floor that voided the null arm is the
SAME process inflating `rho` in the drive arm. **Consequence for any re-run: a longer gap would NOT
help** — the spikes are Poisson, so a longer gap collects proportionally more. **Suppressing
spontaneous release during gaps is the single change that fixes both arms.**

#### Status

`E_invasion` ratchet: **UNRESOLVED.** A re-run needs (a) a null that suppresses spontaneous
release, not just activation, and (b) a gap long enough to clear the calcium tail. **Both are
protocol changes to a pre-registered design and are not made unilaterally — PO-3 measured,
recorded, and stopped.**

---

### L·ETA-3 — eta does NOT clear in a live trial: the pump is capable, the PROTOCOL is not · 2026-07-18  `[GROUNDED, measured]`

**This is the precondition L·ETA-2 flagged, and it comes back negative.** The pump
ignites on a characterization rig (L·ETA-2) and does **not** ignite in an actual
spatial-discovery trial. Instrument: `sweep/eta_in_live_trial.py`; trace persisted to
`results/eta_live_trial/`.

Live trial, 12 features/synapses, 40 s budget, agent_dt=0.5, physics_dt=5e-3, seed 7,
input engine fully wired (glutamate + plateau):

| quantity | live trial | L·ETA-2 rig | short by |
|---|---|---|---|
| `E_invasion` (max) | **0.0868** | 0.35 | 4x |
| `ca_open` (max) | **0.140** | ~0.38 | 2.7x |
| **`r` (max)** | **0.0768** | 1.6234 | **13x** |
| eta (max) | **0.0000** | 0.2376 | — |
| synapses condensed | **0** | — | — |
| cross-synapse edges | **0** all trial | — | — |

**Both factors of `r = E_invasion x ca_open x rowsum` fall short, roughly
multiplicatively** (4 x 2.7 ~ 11, against the observed 13x). Neither alone explains it,
so this is not a single broken term.

**The mechanism, read off the trace rather than inferred.** `E_invasion` sits at exactly
0.0000 for the first ~34 s of the trial, then climbs 0.0111 -> 0.0517 -> 0.0747 as the
agent reaches a high-activation region — **and is still rising when the trial ends**.
`E_invasion` is the actin invasion driver and needs SUSTAINED activity to accumulate
(`model6-actin-invasion-driver`: reaches ~0.74 over ~45-60 s of sustained drive). A
navigating agent gives each feature a brief transient, so the integrator never fills.
Meanwhile only **1-4 of 12** features are active at once and `max_act` is mostly < 0.35,
so `ca_open` stays near its floor: activation maps to `-70mV + act*30mV`, and at act=0.3
that is -61 mV, where the Jahr-Stevens Mg-unblock `B(V)` is negligible.

**So the constraint is DWELL and CO-ACTIVATION, not the pump.** The physics is capable;
the experiment does not drive it into the regime where it works. That is a protocol
question and it is Sarah's call. The obvious levers, none of them taken here:
slower agent / longer dwell per feature; denser or clustered features so neighbours
co-drive (recall `p_met_agg` AGGREGATES over neighbours); longer trials so `E_invasion`
can fill; or the plateau carrying depolarization the subthreshold synaptic knob
deliberately does not.

**DO NOT resolve this by raising the synaptic voltage.** The -40 mV cap is a deliberate,
documented choice (`run_spatial_discovery.py:360-365`: "Was -10 mV, which illegally
merged the plateau into the synaptic knob"). Raising it would manufacture condensation by
collapsing exactly the plateau/synaptic separation the BTSP grounding rests on.

**Consequence for §8.** The selectivity phase stays **blocked in practice**, for a
different and better-understood reason than L·ETA-1 gave. L·ETA-1 said the pump cannot
ignite (wrong — that was an unfinished input path). L·ETA-3 says the pump ignites but
this experiment never asks it to. "Vary only the DRIVE" remains unsatisfiable until the
drive regime reaches threshold, because at eta=0 there is no partition to be selective —
confirmed here by **zero cross-synapse edges across the entire trial**.

**Limits.** One seed, one geometry, 12 features, 40 s. `E_invasion` was still rising at
the end, so a longer trial would land higher — this measures the SHIPPED regime, not a
ceiling. Whether any protocol in the reachable space clears threshold is UNMEASURED.


### L·ETA-2 — The pump IGNITES once the input engine is finished; L·ETA-1 SUPERSEDED · 2026-07-18  `[GROUNDED, measured]`

**Supersedes L·ETA-1's conclusion** (already corrected by ERR-2). The input-engine
integration was finished (wires 1 and 3) and eta was re-measured on the same rig.

| metric | voltage only | + glutamate | change |
|---|---|---|---|
| NMDAR open fraction | 0.0000 | **0.3806** | — |
| VGCC open fraction | 0.1339 | 0.1359 | 1.01× |
| E_invasion (max) | 0.3462 | 0.3518 | 1.02× |
| **r = p_met_agg / P_c (max)** | **0.3509** | **1.6234** | **4.63×** |
| **eta (max)** | **0.0000** | **0.2376** | — |

Rig: 7 synapses @1 µm, mt_invaded, −40 mV **subthreshold** synaptic drive (the drivers'
own band — the plateau is NOT merged into this knob), act=1.0 sustained 20 s, dt=5e-3,
511 stochastic release events. `P_c = 21.51 fW` unchanged.

**The mechanism is entirely `ca_open`, not `E_invasion`.** `r ∝ E_inv · ca_open · rowsum`;
`E_invasion` moved 1.02× while the NMDAR population went 0.0000 → 0.3806 — almost exactly
the 0.33 equilibrium `model6-input-engine` predicts at saturating glutamate from
`alpha·g_bind/(alpha·g_bind+beta)`. **Nothing was tuned.** The only change is that the
NMDAR half of the 25/25 channel population is no longer structurally shut, which is what
that skill said would happen and what ERR-2 predicted would invalidate L·ETA-1's number.

**The naturally-driven eta (0.2376) lands within 9% of the 0.26 the topology probes
CLAMP to.** That is a retroactive check on T1′'s operating point, not just on this
measurement: the clamp was an imposed stand-in with no evidence it matched a driven
value, and it does. `[GROUNDED]` — but note this rig is co-driven at act=1.0; it is not
a claim about what eta reaches in a live spatial-discovery trial (see limits).

**WIRE 3 — DDSC fires for the first time.** `plateau_potential` was read at
`model6_core.py:647` and set by no driver. Measured: DDSC triggered **False** without
plateau, **True** with it, on the same rig. The ≥20 kT `collective_field_kT` gate is NOT
the binding constraint at this operating point. The model's whole delayed-commitment
mechanism (Jain 2024 DDSC) had never executed in any learning run and now does.

**LIMITS — do not over-read this:**
- Measured on a **characterization rig at act=1.0 sustained**, not in a live experiment.
  Spatial discovery drives Gaussian feature activations that are mostly < 1.0 and rarely
  co-active across 7 neighbours. **Whether eta clears threshold in an actual trial is a
  SEPARATE, UNMEASURED question.**
- 7 synapses @1 µm. L·ETA-1's geometry table (still valid — it was arithmetic, not
  affected by ERR-2) says `d*` saturates with N and is unreachable at ≥2 µm spacing at
  any N. This result does not change that; it changes the `ca_open` that enters it.
- The plateau's DURATION is a `[MODELED, PROVISIONAL]` constant
  (`PLATEAU_DURATION_S = 0.3 s`). The literature pass pinned the DDSC *response* (Jain:
  peak 30–40 s post-induction) but NOT the plateau's own width. Do not cite it.
- **The T1′ probe family was deliberately NOT wired.** Those rigs pass voltage only, so
  they still run NMDAR-shut. Wiring them changes their calcium, dimer population, and
  therefore `d*(0)=3.4521` and every break time — a re-validation of a closed result, not
  a fix. **Open decision, Sarah's.** The *order* would likely survive; the numbers would not.

**What this does to the §8 selectivity phase:** the BLOCKED banner on
`SESSION_HANDOFF_JUL17_T1PRIME_REDESIGN.md` §8 comes OFF in principle — drive can move
eta, so "vary only the DRIVE" is satisfiable. But it is not yet demonstrated in a live
trial, which is the precondition that actually matters. Re-measure in-experiment before
designing the selectivity test.

### L·ETA-1 — The pump does not ignite: the cross-synapse partition exists only under an imposed η · 2026-07-18  `[GROUNDED, 4 measured conditions + computed thresholds]`

> **CORRECTED SAME DAY — ERR-2. The headline below overstates its conclusion; read this
> first.** Every measurement in this entry was taken with **NMDARs structurally silent**.
> `eta_probe.py:68` drives `net.step(dt, {"voltage": v, "reward": False})` with **no
> glutamate**, and `analytical_calcium_system` defaults `glutamate = 0.0`, so only the 25
> VGCC of the 50-channel population conducted. `model6-input-engine` (June 14) documents
> this as a KNOWN OPEN INTEGRATION GAP and predicted this exact symptom in advance:
> *"Any full-model run between now and that wiring will show reduced calcium and lower
> `E_invasion` — this is the expected consequence of the half-VGCC live path, **not a
> regression**. Do not read it as one."* It further states that
> `channels.get_open_fraction()` **is** the `ca_open_fraction` consumed by
> `compute_metabolic_power` — i.e. the term that sets `r` is glutamate-contingent, and was
> measured with its glutamate contingency unsatisfied.
>
> **What survives:** the arithmetic (`P_c = 21.51 fW`, the `d*` / row-sum geometry table,
> the duty-cycle analysis), and the structural consequence that at eta=0 no cross-synapse
> bonds form — that is an identity in `k_cross`, independent of why eta is 0. The T1′
> caveat also survives: the topology probes clamp, and they too pass voltage only.
>
> **What does NOT survive:** the conclusion *"the pump does not ignite"* as a statement
> about the model. It ignites or not under an input path that is **documented-incomplete in
> exactly the term that drives it**. The measured `r` values are a **LOWER BOUND**, not the
> model's capability. The three-way fork below (protocol / population / genuine negative) is
> **premature** — it presumes the input path was complete, and it was not.
>
> **Also now suspected, not confirmed:** the "unresolved ~4× discrepancy" noted below
> between isolated and in-network `ca_open` is NOT explained by glutamate — `ca_probe.py:13`
> calls `update_gating(0.001, v)` without it either, so NMDARs were closed in both. That
> discrepancy remains genuinely open.
>
> **Grounding failure recorded, per house discipline:** `model6-input-engine` was not read
> before the eta measurement was dispatched. It is the skill that owns the input path and it
> carries a "READ THIS before any full-model run" section naming this precise
> misinterpretation. The topology and learning skills were read; the input one was not.
> Superseded by **L·ETA-2** once the integration is finished and eta is re-measured.

**This entry does not overturn T1′. It states the regime T1′ lives in, which was never
written down, and which materially qualifies "the topology IS the eligibility trace."**

**The measurement.** `sweep/loop_audit_2026_07_18/eta_probe.py`. Under every drive the
learning drivers actually apply, `eta` is pinned at **0**:

| condition | ca_open peak | E_invasion peak | r | eta > 0 |
|---|---|---|---|---|
| −70 mV rest, N=6 | 0.020 | 0.0000 | 0.0390 | NO |
| −40 mV sustained 3 s, N=6 (spatial-discovery drive) | 0.120 | 0.0005 | 0.0391 | NO |
| theta burst −10 mV, N=7, **clamp removed** | 0.440 | 0.0000 | 0.0390 | NO |
| −40 mV sustained **30 s**, N=7 | 0.060 | **0.3596** | **0.1409** | NO |

`eta = (r−1)/(r+1) if r ≥ 1 else 0` (`multi_synapse_network.py:1130`), with
`r = p_met_agg / P_c`, `P_c = n̄ℏω₀²/Q = 21.51 fW` (ω₀=8e6 Hz, Q=10,
`model6_parameters.py:801-802`) and `P_BASAL_W = 0.84 fW`. So `r` has a hard floor of
**0.0390** at zero drive. Run F is the decisive one: `E_invasion` does climb (0 → 0.36 over
30 s) and `r` tracks it, but linear extrapolation to the ceiling `E_invasion = 1.0` gives
**r ≈ 0.32 — still 3× below threshold.** `[MODELED]` on the extrapolation; the run was
stopped at 30 s on CPU budget. The verdict does not turn on it (3× is not an extrapolation
artifact), but the number is not a measurement.

**Why: `ca_open` is the binding constraint, and a prior probe overstated it ~10×.**
In-network `ca_open` at −40 mV measures 0.04–0.06. `sweep/soc_pump_threshold_stage1.py:88`
sets `reachable = 0.74 * 1.0` with the comment `ca_open(burst)~1`. Measured
(`ca_probe.py`): analytic steady state is 0.726 at −10 mV and 0.179 at −40 mV, and
**duty-averaged over the actual theta protocol** (4 spikes × 2 ms per 125 ms = 6.4% duty,
`coherence_fragmentation_probe.py:196-198`) it is **0.063**. That is why the stage-1
falsifier did not fire. `[GROUNDED]`

**CONSEQUENCE FOR THE PARTITION — the part that matters.** `k_cross ∝ sqrt(eta_i·eta_j)`
(`multi_synapse_network.py:309, 321`). **At eta = 0, zero cross-synapse bonds form.** So the
clamp every topology probe applies is not a convenience and not merely a control that "holds
the pump fixed" — *without it there is no cross-synapse entanglement at all, hence no
partition to measure.* T1′, the static radius probe, and the two-cluster Werner validation
all live in a regime the model does not reach from its own drive.

**What this does and does not do to T1′** (state both, do not let the caveat inflate):
- **Does NOT touch it.** T1′ clamps eta at 0.26 as a *declared* control, says so in the
  write-up (§4 "Drive protocol"), scores order only, and the Werner/distance algebra it tests
  is independent of how eta got there. The 4/4 result stands exactly as recorded in T1'-5.
- **DOES qualify what it licenses.** T1′ shows the partition *of this model, under an imposed
  pump*, carries spatial structure a classical scalar trace cannot. It does not show the
  model reaches that partition on its own. Any claim of the form "the entanglement topology
  IS the eligibility trace **in a running network**" is currently **unsupported** — in every
  end-to-end run the topology is empty.

**Geometry dependence** `[computed]`, `dep_probe.py`. Critical drive
`d* = (P_c − P_BASAL)/(p_active_max · rowsum)`; `d*` **saturates** with N because coupling is
`exp(−d/λ)`, λ=5 µm. To cross r=1 at −40 mV with `E_inv` saturated (d≈0.05) needs
rowsum > 6.89: **~10–20 synapses at 1 µm, ~10 at 0.5 µm; unreachable at ≥2 µm spacing at any
N** (rowsum floors at ~5.1).

**THE DECISION THIS FORCES (Sarah's, explicitly NOT the thread's).** The shortfall is 3×, and
`P_c` is set by `omega_0` and `Q` while `p_active_max_W` is a free parameter. Moving any of
them would make eta ignite. **That is the named emergent-physics failure mode** ("what value
makes the result come out right", `quantum-computation-and-attribution` §6.1) and this entry
records that it was NOT done. The real fork is: (a) the drive protocol is wrong — the
subthreshold −40 mV was a deliberate choice (`run_spatial_discovery.py:360-365`, "was −10 mV,
which illegally merged the plateau into the synaptic knob"); or (b) the pump genuinely does
not ignite at few-synapse scale and requires the ~10–20 co-driven synapses at ≤1 µm the table
above implies — in which case condensation is intrinsically a *population* phenomenon and the
experiment must be designed at that scale. **Open.**

**Bearing on the designated next phase (input-selectivity).** The phase directive
(`SESSION_HANDOFF_JUL17_T1PRIME_REDESIGN.md` §8) inherits from T1′ the constraint "fixed
geometry across conditions; vary only the DRIVE." That constraint is currently
**unsatisfiable**: drive does not move eta, eta is the only input-dependent channel into the
partition, and at eta=0 there is no partition. The selectivity experiment cannot be designed
until the fork above is resolved. This is a prerequisite, not a scheduling detail.

### L·T1'-6 — Channel separation: the population channel does not order, and cannot · 2026-07-18  `[GROUNDED, 4 seeds × 6 arms + 40-draw stability]`

**Outcome under the pre-registered reading (L·T1'-6-PRE, committed before the run at
`abdb549`): "B orders, C does not ⇒ coherence-driven."** §6's *conclusion* survives. §6's
*argument* does not, and is replaced below by the property that actually does the work.

`sweep/population_channel_arms.py`; traces persisted to `results/T1prime6_arms/`.

| arm | P_S | population | rule | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|---|---|
| A | decays | decays | model | FALSIFIED | CONFIRMED | CONFIRMED | CONFIRMED |
| B | decays | held | — | FALSIFIED | CONFIRMED | CONFIRMED | CONFIRMED |
| C | frozen | decays | model | **INCONCLUSIVE (0 breaks)** | 0 breaks | 0 breaks | 0 breaks |
| D | frozen | held | — | 0 breaks | 0 breaks | 0 breaks | 0 breaks |
| A_rand | decays | decays | random | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |
| C_rand | frozen | decays | random | **0 breaks** | 0 breaks | 0 breaks | 0 breaks |

**1. The population channel is EXACTLY inert under the model's own removal rule.**
Arm A and arm B are **bit-identical**: `max|A−B| = 0.000e+00` across every sample of every
`d*_eff` and `max_pair(P_S²)` column, in all four seeds. Arm C retains **100.0000%** of
`max_pair(P_S²)` while the population falls 2223 → ~50. This is the measured consequence of
`dimer_particles.py:230-241` removing lowest-`P_S`-first: the argmax is the last thing
removed, so attrition cannot touch the extreme-value statistic that governs edge survival.
Not "small" — zero.

**2. The criticism is immaterial even under ITS OWN assumption.** Arm C_rand imposes uniform
random removal — the charge's implicit rule — and still produces **zero breaks**:
`max_pair(P_S²)` retained 99.92–99.96%, `Δd*_eff = −0.002 µm` against the **1.35 µm** span
the cascade must traverse (0.15%). The reason is measured, not argued: `P_S(0)` is packed
against its ceiling (median 0.9986, max 1.0000), so the max over ~50 draws ≈ the max over
~2200. The extreme-value intuition in step 3 of the charge needs a distribution with room
below the maximum; this one has none at t=0. **So the charge fails twice over** — once on
the model's actual removal rule, once on the shape of the `P_S` distribution.

**3. The null is clean.** Arm D produced no breaks in any seed. The rig is not manufacturing
orderings.

**4. Where population loss DOES bite — reported because it is real.** Arm A_rand (random
removal *plus* coherence decay) confirmed 4/4 with systematically **earlier** breaks
(e.g. seed 3: 16.0/33.5/52.5/65.5 s vs arm B's 16.0/37.5/66.5/83.5 s). Once `P_S` spreads
out under decay, the tail is carried by a few long-`T_eff` dimers, and random removal can
kill them. So a model that removed dimers randomly *would* have a live population channel —
it would still order correctly, but the times would be confounded. This model does not
remove randomly. The distinction belongs in §6 rather than being allowed to read as "the
criticism was wrong".

**5. SECONDARY, and it damages a different claim: the ladder's order-recovery power is
~92%, not 10/10.** Across 40 independent noise draws (arm B, 200 s horizon, 4 seeds × 10
draws): **37 CONFIRMED / 3 FALSIFIED / 0 INCONCLUSIVE**. Seed 0 alone is **7/10** — its
last two rungs (2.45, 2.00) break within ~1.5 s of each other and invert under some draws,
which is why arm A/B above show seed 0 FALSIFIED. `RESULTS §4` cites **10/10** for this
geometry from `order_power_probe.py`. The two numbers are different **estimators**, not a
contradiction: the power probe uses unguarded first-crossing detection and a different noise
stream (`seed+10000`), while this run applies the probe's own `CONSECUTIVE_ABSENT=3` guard,
which is stricter and converts marginal orderings into violations. **The 92% is the better
number** and §4 should carry it. `[MODELED]` — it is an estimate over noise draws, not a
property of the published run.

**Does this damage the headline? No, and the arithmetic should be stated rather than
asserted.** The p-value is computed against the classical null (uniform permutation, 1/24
per seed); power affects the experiment's *sensitivity*, not the null, so `p ≈ 3.0×10⁻⁶` is
untouched. At 92% per-seed power, observing 4/4 has probability `0.92⁴ ≈ 0.72` — entirely
unremarkable. The result is not weakened; the *power claim behind the geometry choice* was
overstated.

**Replay-vs-rig honesty.** Absolute break times here run LATER than the published rig
(seed 0: 16.0/41.5/76.5 s vs 14.5/32.5/61.5 s), exactly the documented upper-bound direction
of the replay (`dstar_eff_replay.py:24-41` — fixed population ⇒ higher max ⇒ later breaks).
No replay reproduces a published realisation: the rig interleaves its noise draws with
network stepping. **This instrument is valid for ORDER questions and invalid for times**,
which is why times are not scored in any arm.

**Carried limitation** `[MODELED]`, pre-declared: N(t) is log-linear through five transcribed
seed-0 anchors, applied to all four seeds, because the raw logs are gone. Findings 1–3 are
insensitive to it — arm C's erosion is *exactly zero* under any monotone trajectory, since
the rule preserves the argmax regardless of how fast the population falls. Finding 4 is
trajectory-sensitive and is reported as directional, not quantitative.

**Open item this does NOT close:** §6 defends only breaks 1–2 as unconfounded, which is two
breaks — below the experiment's own `MIN_BREAKS=3` bar. This run makes that defence
unnecessary (the confound is measured inert, so all four breaks are unconfounded by
population loss), but §6 must still stop claiming a two-break subset as its control.

### L·T1'-6-PRE — PRE-REGISTRATION: population vs coherence channel separation · 2026-07-18  `[PRE-REGISTERED — written and committed BEFORE the run]`

**This entry is committed before the arms are run. Nothing below has been observed.**

**The charge.** Adversarial review of `RESULTS_T1prime_far_pairs_first.md` §6. §6 defends
the ordering with *"Dissolution is spatially uniform — it lowers every pair's effective
radius equally... It cannot generate a consistent spacing-ordered cascade."* The refutation,
assembled from §4's own physics: (1) edge survival is governed by
`d*_eff = λ·ln(max_pair(P_S²)/0.5)`, an EXTREME-VALUE statistic; (2) bonded pairs scale
~N², and N falls 2200→98; (3) the max of fewer draws is smaller, so `d*_eff` contracts from
population loss ALONE; (4) any contracting radius crosses gaps in width order (§2).
⇒ dissolution is uniform in RATE but not order-neutral in EFFECT, and replication across
seeds does not separate the channels.

**Grounding finding that motivates the arms** `[GROUNDED, code]`: step (3) assumes RANDOM
removal. `dimer_particles.py:230-241` removes `sorted(self.dimers, key=lambda d: d.coherence)`
from index 0 — i.e. LOWEST-coherence first — and `coherence` is a strictly increasing affine
map of `P_S` (`dimer_particles.py:57-62`). Attrition is therefore **rank-selective**: the
survivor set is the top-n by current `P_S` and the argmax is the last thing removed. This is
also the only live attrition path (`step_population` is called every step at
`dimer_particles.py:602`; `_remove_dimer` at :252 has no callers). If that reading is right,
the population channel should be *unable* to erode `max_pair(P_S²)` at all. **The arms are
run anyway, because an argument from code is not a measurement.**

**The arms** (`sweep/population_channel_arms.py`, replay — not fresh 2.7 h/seed sims), seeds
0–3, wide ladder, 90 s, guards `CONSECUTIVE_ABSENT=3` / `MIN_BREAKS=3`, verdict can return
CONFIRMED / FALSIFIED / INCONCLUSIVE:

| arm | P_S decays | population decays | attrition rule | purpose |
|---|---|---|---|---|
| A | yes | yes | model (top-n) | reproduce the published result |
| B | yes | **held** | — | coherence-only channel |
| C | **frozen** | yes | model (top-n) | population-only channel |
| D | **frozen** | **held** | — | null — ordering here means the rig is broken |
| A_rand | yes | yes | **random** | the criticism's world, full |
| C_rand | **frozen** | yes | **random** | the criticism's world, isolated |

A_rand/C_rand are outside the pre-registered four; they exist so "is the N-dependence
material?" gets a NUMBER under the criticism's own assumption rather than a dismissal.

**PRE-REGISTERED READING OF OUTCOMES** (committed before looking):
- **B orders, C does not** ⇒ coherence-driven. §6's conclusion stands; its *argument* is
  rewritten on the rank-selectivity ground, which is the property that actually does the work.
- **C orders, B does not** ⇒ the result is population-driven. §6 is REFUTED and the
  coherence claim in §1/§5 must be retracted or restated. This entry commits us to reporting
  that outcome if it occurs.
- **BOTH order** ⇒ degenerate channels; T1′ cannot separate them. §6 must say so plainly and
  the claim narrows to "spatially structured", not "coherence-driven".
- **D orders** ⇒ rig artifact. Everything downstream is suspect; stop and report.
- **C_rand orders while C does not** ⇒ the criticism is valid *in general* and refuted *for
  this model specifically*, by the rank-selective removal rule. That distinction gets stated
  explicitly rather than being allowed to read as "the criticism was wrong".

**LIMITATION, pre-declared** `[MODELED]`: the population trajectory N(t) is NOT replayed
from physics — the raw run logs were session-scoped scratchpad and are GONE. N(t) is
log-linear interpolation through five transcribed seed-0 anchors from L·T1'-4 (2200 at t=0;
1843/1043/259/98 at 14.5/32.5/61.5/78.0 s), applied to **all four seeds** because seeds 1–3
trajectories were never persisted. So the C arms test whether the population channel orders
**at all**, not seed-specific timing. Times are not scored in any arm. From this run on,
traces are persisted to a tracked path (`src/models/Model_6/results/T1prime6_arms/`).

**Known live tension to address regardless of outcome:** §6 defends only breaks 1–2 as
unconfounded, which is TWO breaks — below the experiment's own `MIN_BREAKS=3` sufficiency
bar. The unconfounded subset does not score under the probe's own rule. This must be stated
in §6 explicitly, not left implicit.

### L·T1'-5 — T1' replication complete: 4/4 seeds far-pairs-first, CONCLUSIVE · 2026-07-17  `[GROUNDED, 4 seeds]`

Seeds 1–3 run in parallel (3 cores, ~2.7 h wall; parallelism cost ~nothing — each ran at
the same ~355 s/sim-s as seed 0 alone). All four seeds broke in the exact pre-registered
order. **The order is invariant; the times are not** — which is the whole methodology,
borne out:

| seed | gap 3.35 | gap 2.90 | gap 2.45 | gap 2.00 | order |
|---|---|---|---|---|---|
| 0 | 14.5 s | 32.5 s | 61.5 s | 78.0 s | ✓ |
| 1 | 14.0 s | 37.0 s | 55.0 s | 82.5 s | ✓ |
| 2 | 14.0 s | 42.0 s | 54.5 s | 71.0 s | ✓ |
| 3 | 11.0 s | 32.5 s | 64.5 s | 82.0 s | ✓ |

**Statistical standing.** Pre-registered discriminating claim: as coherence decays the
partition fragments in gap order, widest first. Classical null: a scalar eligibility trace
decays uniformly, carries no spatial structure, so the break order is uninformative — a
uniformly random permutation of the 4 rungs, P(exact order) = 1/24 per seed. Observed 4/4
⇒ **p = (1/24)⁴ ≈ 3.0×10⁻⁶**. Conclusive by any reasonable threshold.

**Why replication is more than 4× the confidence.** Two objections a single seed could not
answer, both closed by the replication:
1. *Luck.* At the measured 10/10 geometry power (L·T1'-3), one in-order seed is already
   unlikely by chance; four independent ones is decisive.
2. *The population-collapse confound.* Breaks 3–4 land in a cratered population, so their
   TIMES are confounded by dimer-loss. But dimer-loss is spatially UNIFORM — it lowers
   every pair's radius together, which drives edges toward dying *simultaneously*, not in
   gap-spaced order. A consistent gap-ordered cascade across four independent stochastic
   realizations is the signature of the coherence/distance mechanism specifically; uniform
   dissolution cannot fake it repeatably. So the ORDER result is clean even though the late
   TIMES are confounded (and the times were never scored anyway).

**What it establishes / does not.** Establishes: the model's entanglement partition
fragments with SPATIAL structure — far pairs decouple first — the discriminating behavior a
classical scalar trace cannot produce; this is now a conclusive result for the *model*.
Does NOT establish quantumness: this is the **(A)** reading (one shared coin per component,
classical common-cause), and per the epistemic frame the result is a discrimination win for
the theory of expected operation, not a claim about nature (the attribution gap is
untouched). Next candidate work: L·T1'-4's caveats are retired; open directions are the
small-N non-classicality witness (the **(B)** question, separate build) and folding this
into the coherence-gated-learning discrimination story.

### L·T1'-4 — T1' dynamic: far-pairs-first CONFIRMED, seed 0 · 2026-07-17  `[GROUNDED, single seed]`

*→ Completed by L·T1'-5 (4/4 replication, conclusive). Retained as the trail: the point at
which the result was a single seed (p≈0.042) and honestly only suggestive — we replicated
rather than declaring victory on one run.*

**The claim under test.** As coherence decays, d* shrinks, so the cross-synapse partition
must fragment in GAP ORDER — widest gap first. This is the discriminating claim for the
whole "topology-is-the-computation" thesis: a classical scalar eligibility trace decays
uniformly and carries no spatial structure, so it *cannot* produce a spacing-ordered
cascade (`coherence-gated-learning` primitive #1; `entanglement-topology-measurement` A7b).

**Result (seed 0).** Wide ladder, 90 s silence, dt=1e-3. All four live edges broke in the
exact pre-registered order:

| gap (µm) | broke at | population at break | regime |
|---|---|---|---|
| 3.35 | 14.5 s | 1843 dimers | healthy — clean |
| 2.90 | 32.5 s | 1043 dimers | healthy — clean |
| 2.45 | 61.5 s |  259 dimers | cratered — timing confounded |
| 2.00 | 78.0 s |   98 dimers | cratered — timing confounded |

Verdict function (guarded, CAN return INCONCLUSIVE): `far-pairs-first order CONFIRMED over
4 breaks`. Two flickers (gap3.35, gap2.45) were caught and NOT scored — the guard that the
Jul-16 false positive lacked.

**What it establishes, and what it does not.**
- Establishes: the *model's* partition fragments with spatial structure (far pairs first) —
  the thing a classical scalar trace cannot do. First non-vacuous T1' result.
- Does NOT establish: (a) conclusiveness — ONE seed, p≈1/24≈0.042 vs the classical null;
  needs a 2nd independent seed (→ p≈0.0017) to be defensible. (b) quantumness — this is the
  **(A)** reading (one shared coin per component, classical common-cause), unrelated to the
  attribution gap. Per the epistemic frame: a discrimination win for the model, not a claim
  about nature.

**The confound, handled.** Breaks 3–4 landed in a cratered population (259, 98 dimers), so
their break TIMES are confounded by dimer-loss. Their ORDER is not: dissolution is spatially
uniform, lowering every pair's radius equally, so closer pairs outlast farther ones
regardless of mechanism. Only a total collapse to zero would scramble the order. `n_dimers`
is logged as the control; **times are recorded, never scored** (see L·T1'-2 for why times
are not analytically predictable here).

**Cost note.** Ran 2.7 h, not the projected ~8 h. The entanglement tracker is O(n²) in
dimer count; the population collapsed 2223→92 over the run, so cost/step fell ~500× by the
end. The ~8 h projection extrapolated the high-population front-regime cost flat — a milder
instance of the "cost from a microbenchmark, not a profile" scar. Seed 1 will be ~2.7 h too.

**Next:** run seed 1 (independent) for p≈0.0017; then decide on 2–3 more seeds for a solid
per-seed distribution. Score the ORDER per seed.

### L·T1'-3 — The trustworthy rebuild: geometry for POWER · 2026-07-17  `[GROUNDED, 10-seed replay]`

The Jul-16 handoff §3 said put the four live gaps just under d*(0)=3.45µm (3.4/3.3/3.2/3.1)
so the cascade "lands early, while the population is >90% alive." **Both halves false —
measured, and I built to it before catching it (the grounding brief did not flag it).**

1. **Tight gaps do NOT break early.** An edge survives while ANY bonded pair clears F>0.5,
   so the governing radius is `d*_eff = 5·ln(max_pair(P_S²)/0.5)` — the extreme TAIL, not
   the median. The tail decays ~3.6× slower than the median. An 8 s sanity run: median
   radius fell below ALL four gaps (3.04µm @7.5s) while ALL four edges were still alive.
   Zero breaks in 8 s. The null killed the plan.
2. **Tight gaps destroy the order signal.** Each synapse's tail is set by its own luckiest
   dimer, whose `T_eff` is frozen at creation (`dimer_particles.py:47-50`). Between-synapse
   scatter in d*_eff ≈ 0.29µm. Rungs finer than that are decided by luck.

**Measured order-test power** (`sweep/order_power_probe.py`, 10 seeds via the validated
d*_eff replay):

| geometry | live gaps (µm) | power | vs chance (1/24) |
|---|---|---|---|
| tight  | 3.35 3.25 3.15 3.05 | 6/10 | 0.10µm rungs < 0.29µm scatter |
| medium | 3.35 3.10 2.85 2.60 | 5/10 | still under-resolved |
| WIDE   | 3.35 2.90 2.45 2.00 | **10/10** | 0.45µm rungs clear the scatter |

Counterintuitively WIDER is better (it costs more wall-clock — breaks land later — but the
order becomes resolvable). The Jul-16 2.0–3.0µm gaps were directionally closer to right;
their real flaw was the 35 s window, not the spacing. Power is what lets the verdict
FALSIFY: at 6/10 a WRONG result means nothing; at 10/10 it is real evidence against.

### L·T1'-2 — d*(0) measured; the d*_eff replay instrument · 2026-07-17  `[GROUNDED]`

**d*(0) MEASURED** (`sweep/measure_dstar0.py`), not assumed — two prior wrong claims came
from assuming P_S (the "knife-edge" off an assumed 0.90 when it is 0.998; §6.3 of the
Jul-16 handoff). Rig = confirmed static-probe rig, t=0.08 s, n≈2200: P_S median 0.9987
(min 0.9922, max 1.0) ⇒ d* min 3.387 / median 3.4521 / max 3.4657 µm. The *distribution*
(not a single number) is the fact that drives the geometry choice in L·T1'-3.

**The d*_eff replay** (`sweep/dstar_eff_replay.py`) — the reusable sizing instrument.
P_S dynamics are intra-synapse only (`step_coherence` reads local J-field + template
binding; `T_eff` fixed per dimer for life; the network never feeds back into P_S), so the
tail can be replayed in vectorised numpy with NO network — 200 s of sim in ~5 s of compute.
**VALIDATED:** median P_S matches the real 8 s rig to <0.0005. Its P_S→d*_eff mapping is
**UNVALIDATED by construction** — there is no honest cascade datum to check it against (the
only candidate, "gap 3.0 broke @34 s" from Jul-16, is the retracted FLICKER = fabricated
data). Because the replay holds the population fixed (higher max ⇒ later breaks), it emits
an **UPPER BOUND on break times**, correct for SIZING a window, never a prediction to score.

**Why the times are never scored:** they are an extreme-value statistic over a noisy random
walk (`step_coherence` multiplicative noise, ~±5.8% on P_excess over 34k steps) across
hundreds of pairs. Mis-derived THREE times on Jul-16 (median 9.5s → p95 ~13s → "ceiling"
19.3s vs observed 34.0s). **Pre-register and score the ORDER only.**

### L·dt-1 — dt convergence · 2026-07-17  `[GROUNDED]`

See the DECISION RECORD row. Instruments: `sweep/dt_convergence_drive.py`,
`sweep/dt_convergence_operating_point.py`, `sweep/dt_independence_tail.py`. Bottom line:
ORDER test valid at dt=1e-3 (P_S + edges converged); operating point ~155µM is not a dt
artifact (converged to ~5% at saturation); dt=1e-2 overflows the count — do not use.

### L·T1'-1 — T1' static half: the Werner cut IS a distance rule · 2026-07-16  `[PROVEN algebra + GROUNDED probe]`

See the DECISION RECORD row. `bond iff d < d* = 5·ln(P_product/0.5)`, confirmed 7/7 on a
pre-registered ladder (`sweep/coherence_radius_probe.py`), retrodicts the Stage 3 chain/ring
validation with no free parameters. This is what GENERATES the T1' dynamic order prediction.
Detail and the two theorems (hard floor P_S>1/√2; radius shrinks) in
`entanglement-topology-measurement` Appendix A7b.

### L·T1-RET — T1 (SOC-topology power-law) retired · 2026-07-16  `[GROUNDED structural]`

See the DECISION RECORD row. Killed on structure (1D geometric clump-size; D18 quantal
scale), not on a negative run. SOC-*chemistry* (D8/D14) is untouched and stands.

---

## Cross-references

- **Skills:** `entanglement-topology-measurement` (Appendix A = current authority),
  `model6-entanglement-partition-werner` (the LOCKED partition + Werner bound),
  `quantum-computation-and-attribution` (the A/B fork, the attribution gap, the
  discrimination discipline this log's epistemic frame rests on),
  `coherence-gated-learning` (why the topology-as-trace claim is discriminating).
- **Sibling log:** `RESEARCH_LOG_CALCIUM_DIMER.md` (the chemistry sub-program, D1–D18).
- **Handoffs:** `docs/handoffs/SESSION_HANDOFF_JUL17_T1PRIME_REDESIGN.md` (this session's
  thread baton), `SESSION_HANDOFF_JUL16_TOPOLOGY_DT_FIX.md` (the failed run + dt fix).
- **Code:** `sweep/coherence_fragmentation_probe.py` (T1' dynamic),
  `sweep/coherence_radius_probe.py` (T1' static, CONFIRMED 7/7),
  `sweep/{measure_dstar0,dstar_eff_replay,order_power_probe,dt_*}.py` (instruments).
