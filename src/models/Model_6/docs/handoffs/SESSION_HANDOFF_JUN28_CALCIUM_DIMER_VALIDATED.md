# Session Handoff — Calcium → Dimer Revalidation LANDED & Integration-Validated (2026-06-28)

**Thread topic:** Bottom-up emergent-physics revalidation of Model 6's calcium → dimer-formation
stack. Repo: `/Users/sarahdavidson/posner_quantum_dynamics` (GitHub: Sarah-in-dev).
**Workflow:** claude.ai reasons + writes prompts; Claude Code reads/edits the repo; Sarah runs all
git/execution and approves every change. Claude Code never commits.

---

## 0. How to start the next session (read order — do not skim)
1. `session-discipline`, `working-process`, `agent-grounding-protocol` — the discipline layer.
2. `model6-dimer-formation-chemistry` (UPDATED this session — §0b landed-summary, §3 sigmoid removed),
   `quantum-system-canonical` (UPDATED — banner + §8), then **`research-log-calcium-dimer`** (the full
   decision trail C0/D1–D17 — this is the primary provenance record; read it).
3. `model6-architecture`, `model6-codebase-operations`, `model6-input-engine`.
4. Re-read the most recent conversation (`conversation_search`/`recent_chats`) — ground truth.
5. Then this handoff. **Ground before acting:** the code descriptions here are the session's working
   understanding, not a fresh read — recon before any new claim.

---

## 1. Where we are — the milestone
The calcium→dimer coupled revalidation is **LANDED in code and integration-validated.** For the first
time the system runs **emergent, bounded, rise-and-fall**, localized, with plasticity accumulating
across traversals and the agent solving the task — **no tuned constant anywhere in the chain.**

The coupled fix (all grounded, all landed):
- **Grounded calcium** (`analytical_calcium_system.py`): calibrated 0.5 µM/channel snapshot → Naraghi-
  Neher closed form `i/(z·F·4π·D_ca·r)·exp(−r/λ)`; **free D_ca** in the prefactor (was wrongly buffered
  D_eff); buffer **λ≈117 nm** (was pump-set 190); read floor **≈5.5 nm** (FRET-grounded, D11).
- **ACP supersaturation gate** (`ca_triphosphate_complex.py` `update_dimerization`): formation only when
  `S=([Ca]³[PO₄³⁻]²/Ksp)^(1/5) > 1`, pKsp≈26. Gates formation only (dissolution untouched).
- **Finite phosphate / SOC:** self-limiting **emerged** from the gate + the existing consumption
  plumbing — no separate B3 edit needed (D14).
- **Species = dimer** (D16): the invalid bulk-Ca/P sigmoid was removed (`dimer_fraction=1`); selection
  is downstream by **coherence** (dimer persists, trimer decoheres sub-second — Agarwal).

**Integration (`run_spatial_discovery`, 5 trials × 20 synapses):** end-of-trial dimers 9/1/31/23/22;
peak transient 318; localized to 1–5/20 synapses; max spine vol 1.63→2.67 across laps; agent FOUND the
goal in trial 3. Stochastic/near-critical (unseeded channel gating → trial-to-trial variability).

**Claim altitude (LOCKED discipline):** this validates the **(A)** coherence-gated classical floor. It
says **nothing** about the **(B)** genuine-quantum claim — that is the separate small-N non-classicality
witness, not built. Do not describe (A)'s result in (B)'s language.

---

## 2. Git state
Clean tree (only untracked `CLAUDE.md`). Commit chain on `master`, pushed:
- `0ef0e0e` A1 probe + retire PDE · `95990fd` input-engine · `f0aaffd` A2+A3 probes ·
  `49c7453` **B1 gate** · `e9bb2c3` research log D1–D11 · **`2ab02d8` "dimer working system"**
  (B2a calcium + D16 species + log D12–D17).
- **Uncommitted after the push (this session's wrap):** research-log section K + provenance update
  (posner); the two skill updates `model6-dimer-formation-chemistry` + `quantum-system-canonical`
  (murmur repo, where `.claude/skills` physically lives). Commit when ready.

---

## 3. What's next (prioritized; none are blocking — the core is done)
1. **B2b — live pH-driven PO₄³⁻ plumb + the pH-sign decision.** The gate currently derives PO₄³⁻ from
   the HPO₄²⁻ arg at **rest pH 7.35** (which already tracks the structural pool, so SOC works). The
   refinement is the multi-file plumb of trivalent `atp.phosphate.PO4` (D7: read `phosphate.PO4`, NOT
   `get_posner_forming_species()` which returns HPO₄²⁻). **Gated on a pH-sign literature pass:** the
   model's `pH_dynamics.py` is acidification-only (7.35→6.8), but the intracellular spine literature
   says it **alkalinizes** during activity (NHE5; D5). Ground the alkalinization magnitude/timecourse
   first, then plumb + correct the sign together. Not load-bearing now.
2. **Characterize the near-critical variability** (D17): trial-to-trial 1–31 dimers is the criticality
   signature `experimental-validation` predicts (place-cell all-or-none). Worth a dedicated analysis —
   it may be a real experimental prediction.
3. **Validate against the Ksp BAND, not 716 µM** (D4): published pKsp(ACP) spans 24–28 → threshold
   150 µM–3.3 mM. Re-run scenarios across the band; state the uncertainty, never tune to a line.
4. **Cleanup:** the orphaned `calculate_dimer_fraction` recompute (~`ca_triphosphate_complex.py:425`),
   now dead.
5. **Skill lock for `quantum-system-canonical`:** resolve the `[verify vs <skill>]` flags (§2.1, §4.2,
   §4.3 backbone, §4.4 commitment); add owning-skill back-pointers; correct `quantum-biology-primer`'s
   pre-Agarwal dimer-precursor framing (~line 39).

---

## 4. Open keystones / contested bets (carried so they are not lost)
- **#1b — does FORMATION favor the dimer?** [CONTESTED, de-fanged] Naive aggregation thermodynamics
  favors the *trimer* at high nanodomain Ca; the aggregation rate is **ungroundable** (ns→hours).
  Resolved *for the model* by coherence (only the dimer persists as a qubit), but formation-dominance
  itself is unproven. (D15)
- **#1 — pair-level selectivity** (canonical §8 keystone #1): "topology is the computation" needs
  which-dimers-bond to depend on input, not just which regions are eligible. Verify before resuming
  graph-as-computation claims.
- **pH sign / compartment** (D5/D6): intracellular spine alkalinizes vs the model's acidification.
- **The (B) genuine-quantum claim:** the small-N non-classicality witness — separate microscopic build,
  not started.

---

## 5. Hard constraints / do-not-undo
- **Emergent physics only.** Every rate is a measurement; nothing tuned to a target. The species fix is
  coherence-grounded (Agarwal), NOT a chosen fraction — do not re-introduce a bulk-Ca/P dimer/trimer
  sigmoid (D13/D16).
- The **read distance ≈5.5 nm** is FRET-grounded biology (D11), not the 4 nm grid floor; do not push it
  sub-nm to chase formation (the closed form diverges and it's outside Naraghi-Neher validity).
- **Free D_ca** (not D_eff) in the closed-form prefactor; buffering only in λ.
- The dimer is the qubit (Agarwal); the trimer is inert. SOC at S=1 is emergent — do not add a
  conservation hack.
- Multi-hunk applies silently fail — **grep-verify after every apply.** Grep to a file (INFO floods).
- `run_spatial_discovery` with a real time budget is slow (~50–150 s wall/trial); run backgrounded,
  suppress INFO logging, redirect to a file. Don't double-background (`nohup … &` inside a bg tool call).
  ~0 dimers in a single short traversal is *correct* — formation accumulates over laps.

---

## 6. Key files
- **Live (edited this session):** `analytical_calcium_system.py` (grounded calcium, B2a);
  `ca_triphosphate_complex.py` (`update_dimerization` gate B1 + species fix D16).
- **Probes:** `sweep/supersaturation_gate_probe.py` (A2), `sweep/phosphate_conservation_probe.py` (A3),
  `sweep/nanodomain_closedform_probe.py` (A1). Scratch validations were in `/tmp` (not committed).
- **Runner:** `sweep/run_spatial_discovery.py` (the integration; `__main__` runs 5 trials).
- **Record:** `src/models/Model_6/docs/RESEARCH_LOG_CALCIUM_DIMER.md` (C0/D1–D17 + references).
- **Geometry:** `model6_core.py:222–258` (50 channels ±8 nm, templates = `channel_positions[:3]`);
  `atp_system.py:364–401` (speciation), `:430` (`get_posner_forming_species` = HPO₄²⁻).

End of handoff. Next action: pick from §3 (B2b+pH-sign needs its own literature pass first; or the
near-critical-variability characterization is a clean, self-contained next investigation).
