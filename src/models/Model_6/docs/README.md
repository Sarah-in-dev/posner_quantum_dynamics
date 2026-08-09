# Model 6 — documentation map & read-order (START HERE)

*Purpose: a new agent (or a returning human) should know, from this one file, exactly what to read, in what
order, and what is canonical vs stale vs already-answered. This file is a **map, not a knowledge base** — it
points; the substance lives in the skills, the research logs, and the code.*

*Last reconciled: 2026-08-09, against a full read of both research logs, the canonical spine, the model code,
and the test surface.*

---

## 0. The one-paragraph north star

Model 6 is an in-silico theory of how the brain might solve **temporal credit assignment** — linking an action
to a consequence tens of seconds to minutes later — via **coherence-gated computation** in a calcium-phosphate
(Ca₆(PO₄)₄ **dimer**, not trimer) nuclear-spin substrate that stays coherent ~100 s. The goal is to
**characterize the computational primitive and show it solves real problems**, *not* to certify quantum-ness.

**The model is (A), and that is [LOCKED]:** a *coherence-gated **classical** correlated-partition computer* — the
graph holds magnitudes (P_S, scalar weights) and the dynamics are classical stochastic updates **parameterized by
quantum-derived quantities**. The honest operative verdict is **"quantum CONSTRAINS, classical COMPUTES."**
Never describe the (A) model in (B) (genuine-quantum) language — that is a named, locked anti-pattern.
(Source: `quantum-system-canonical` §1, §5.1; epistemic frame LOCKED by Sarah 2026-07-17, `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`.)

---

## 1. Read-order for any new task (the grounding path)

Follow `CLAUDE.md`'s discipline, in this order. Do not skip; the failure mode is acting on a stale picture.

1. **`session-discipline`** and **`agent-grounding-protocol`** (skills) — how to work here; run the GROUND sequence.
2. **`quantum-system-canonical`** (skill) — the ontology: what the system IS, how it couples, what the computation
   is, and the A/B discipline. Read FIRST for any "what is this" question. **Status: v0.1 DRAFT** (canonical intent,
   not yet locked; §2.3/2.4/3 track decided-but-not-landed physics).
3. **The task-relevant `model6-*` / `quantum-*` skills** — the LOCKED decisions + mechanism detail (see §4 map).
4. **The research logs, newest-first** — the substance/results (see §2). This is where the live state lives; skills drift.
5. **The code** — SHOW the file you intend to touch; code is behavior-truth. Reconciliation rule: **code + data =
   what IS; skill + recent log = what was DECIDED.** When a file disagrees with prose, the file wins.

---

## 2. Where the substance lives (the "four homes")

| what | where |
|---|---|
| **Substance** — findings, measurements, provenance | the two research logs below + persisted `results/` |
| **Decisions, LOCKED items, rationale** | the `model6-*` / `quantum-*` skills |
| **Method / process** | `session-discipline`, `agent-grounding-protocol`, `MO_MODEL6.md` (§2/§7), `consumer-acceptance-gate` |
| **Live coordination (which POs are hot)** | `MO_MODEL6.md` — **but it is a 2026-07-18 snapshot; see §5** |

**The two research logs (append-only; newest entry at top; never rewrite — supersede with a dated note):**
- **`RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`** — *what the computation IS* (the entanglement partition). Sub-program
  PO5→PO11 + COV. The live frontier of the "does the partition compute?" question.
- **`RESEARCH_LOG_CALCIUM_DIMER.md`** — *what forms the dimers* + the reward/readout half (the F-series F1→F3).

---

## 3. Status — what is SETTLED vs the LIVE FRONTIER (so nobody re-runs a closed question)

### Settled / LOCKED (do not relitigate without new physics)
- **The model is (A); "quantum constrains, classical computes."** Never (B) language. (`quantum-system-canonical` §5.1)
- **The computation IS the partition** of the entanglement graph; readout = joint per-component collapse
  (one shared coin ≈ mean P_S). (`model6-entanglement-partition-werner`)
- **Werner math:** cross-bond fidelity `F = P_S_i·P_S_j·w_spatial`, edge iff **F > 0.5** (separability theorem —
  not a tunable knob; do not lower it; do not apply it to intra bonds).
- **Ca₆(PO₄)₄ dimer is the qubit; the Ca₉(PO₄)₆ trimer is inert.** (Agarwal 2023)
- **One synapse = one nanodomain = one component** (single-synapse "one giant component" is correct physics).
- **Emergent physics only** — no constant tuned to a downstream target. Score a structural invariant, never times.
- **The measurement trigger is the spin-selective binding-melt, NOT reward** (F2, 2026-08-02). Reward is a
  **separate** three-factor learning signal.
- **T1′ (far-pairs-first) is CLOSED** — 4/4 seeds, p≈3×10⁻⁶. Do not re-run/re-tune.

### The live frontier — the directed quantum readout cascade
- **The cascade IS the mechanism (F-series).** The coherent singlet ³¹P tag is read out by Fisher's
  **spin-selective Posner binding-melt** (QDS — only the coherent/singlet channel binds and melts) →
  **Ca²⁺ shower → glutamate → plasticity**. Dopamine (reward) **gates/times** the melt via
  D1/D2→cAMP/PKA→local Ca²⁺/pH and **signs** it (burst→+1 LTP / dip→−1 LTD). It is *directed* (state-selective +
  signed), it is a genuine *quantum readout* (it reads the ³¹P spin-coherence tag), and it is (A)-consistent: the
  quantum constrains WHICH channel reads out; the classical cascade amplifies and drives plasticity. Grounded in
  Fisher 2015 + QDS (Fisher & Radzihovsky 2018) + Agarwal 2022. Physics: `RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08.md`.
- **Status: PROVEN at the single-synapse level.** F3 passes — delayed-credit 3/3 (the ~100 s coherent tag credits
  where the classical ~2 s trace is dead), and the isotope arm moves temporal credit (⁶Li long / ⁷Li short).
  `reward_gating.py` `da_sign` drives the update at `model6_core.py:693–743`, gated behind the Werner floor.
- **The open, on-program next step:** carry the reward-signed cascade from the controlled single-synapse probe
  toward the multi-synapse / living loop (Phase C), and supply PO10-2's unresolved weight **sign from the reward
  signal** (`da_sign`) — the sign comes from reward, classically. F1→F2→F3 is the cascade; extending it is the work.
- **Contested substrate premises** (carried, not settled): microtubule Q (~10, AMRIS-class), λ_F fidelity length
  (unmeasured), dimer coherence lifetime (Agarwal ~100–1000 s vs Fisher ~a day vs Player&Hore skeptical), lithium
  attribution (Posner vs radical-pair vs classical), the small-N (B) non-classicality witness (not built).

---

## 4. Owning-skill map (where mechanism detail lives)

| topic | skill |
|---|---|
| Ontology / A-vs-B / attribution discipline | `quantum-system-canonical`, `quantum-computation-and-attribution` |
| The partition / Werner fidelity / clusters | `model6-entanglement-partition-werner`, `entanglement-topology-measurement` |
| Dimer chemistry / P_S / species correction | `model6-dimer-formation-chemistry` |
| Calcium cascade / input engine | `model6-input-engine` |
| Commitment / measurement pathway | `model6-commitment-pathway` |
| Coherence-gated learning / eligibility | `coherence-gated-learning` |
| Isotope lever / reward readout physics | (F-series) `RESEARCH_LOG_CALCIUM_DIMER.md` + `RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08.md` |
| Codebase ops / how to run | `model6-codebase-operations`, `experiment-design-patterns` |
| Cross-domain bridge (to the PAUL/TALON work) | `cross-domain-integration`, `transition-framework` |

---

## 5. Doc-status notes (what is canonical vs stale vs a handoff)

- **`quantum-system-canonical`** = the canonical ontology, but **a v0.1 DRAFT** (not locked; some sections track
  decided-but-not-landed physics).
- **`MO_MODEL6.md`** = the method/board, but a **2026-07-18 SNAPSHOT.** Its six POs (pump, phosphate, E_invasion)
  are superseded; its "HOT SET" says "Nothing dispatched yet." Read it for §2 (adaptations) and §7 (LOCKED) — those
  still hold — but **not as the current status board.** The live board is the frontier in §3 above + the logs.
- **`handoffs/SESSION_HANDOFF_*` and the `PREREG_PO_*` "charters"** = **session-to-session notes written by prior
  Claude sessions in first-person voice** (e.g. the 2026-08-08 F-series handoff + the "PO charter" it dispatched).
  They record what one session decided/dispatched; they are **not** human-authored canonical direction. Treat them
  as history, verify against the logs + skills before acting on them.
- **`SYNTHESIS_TEMPORAL_CREDIT_F1_F2_F3_2026-08-08.md`** = the F-series synthesis; self-flagged as partly
  OVERCLAIMED by the later reground — read it through that caveat.
- **`PREREG_*` files** = pre-registrations, mixed vintage; **many are already resolved in the logs but not marked**
  (a status-header pass is the next cleanup step).

---

## 6. Testing — how validation actually works here (there is no single runner)

Three disconnected layers (no `pytest`, no unified entry point — a known gap):
1. **Module self-tests** — most core `.py` files carry an `if __name__ == '__main__':` block printing
   `=== ACCEPTANCE CHECKS ===` with `[PASS]/[FAIL]` (e.g. `reward_gating.py`, `posner_binding.py`,
   `nuclear_relaxation.py`). Run the file to self-validate that module.
2. **Pre-registered keystones** — `sweep/*_test.py` (e.g. `po7_unit2_keystone_test.py`): decode-vs-shuffle-null
   (LOO nearest-centroid vs a 2000-perm null p95, or Newman modularity vs permuted labels, Cohen's d bands). Bound
   to a `PREREG_*.md`; a negative is a result.
3. **Full-system experiments** — `Full_System_Experiments/run_all_tiers.py` (physics tiers, not the above).

**Caveats:** the integration test `test_cross_neuron_integration.py` is interactive (`input()`); some post-hoc
scorers hardcode absolute result paths; verdicts aren't surfaced through exit codes. "Run the tests" has no single
answer today — this is on the cleanup list.

---

## 7. The mathematics in one place (pointers, not a restatement)

- **P_S singlet decay:** `P_S(t) = 0.25 + (P_S₀−0.25)·e^(−t/T)`, `T_singlet_P31 = 216 s` (`dimer_particles.py`);
  crosses the Werner floor 1/√2 at ~107 s (the effective coherence window).
- **Isotope lever (F1):** T2 derived from ⁷Li–³¹P scalar relaxation (`nuclear_relaxation.py`): ⁷Li≈14 s, ⁶Li≈216 s.
- **Partition (F=P_S²·w, F>0.5) + union-find clusters:** `multi_synapse_network.py` `_update_entanglement`,
  `_find_all_clusters`; derived cutoff `d* = λ·ln(P_S_i·P_S_j / 0.5)`.
- **Readout / commit coin (unsigned count):** `multi_synapse_network.py` `perform_quantum_measurement`. NB: the
  live model has **no signed Δw** — any signed weight readout in a `sweep/` script is an experiment-side construct,
  and the *sign* is meant to come from the reward signal (`reward_gating.da_sign`), not from the partition.
- **Reward-gated update (single-synapse, wired):** `model6_core.py:693–743` uses `reward_gating.quantum_credit`.

---

*If anything in this map disagrees with a research-log entry or the code, the log/code wins and this map owes an
update (say so and fix it). This file is a construct-validity instrument: it is only useful while it stays glued to
the logs and the code.*
