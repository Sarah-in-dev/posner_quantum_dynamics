# Session Handoff — 2026-08-09 · reward-signed readout: correction done, bistable switch built, Part 2 is the frontier

> **⚠ STATUS: prior-session handoff (Claude-authored). This is a POINTER, not an authority.** Do NOT act on this
> file the way this session was misdirected by the *previous* handoff (see FAILURE MODES §6). **START at
> `docs/README.md`** (the map), then the two grounding docs named below. Verify every claim here against the
> research log + code before acting. If this handoff disagrees with the log/code, the log/code wins.

## 0. One line
The reward-signed readout was made **biologically correct** (dopamine reinforces CaMKII via DARPP-32/PP1, it does
NOT bypass it), a **bistable CaMKII switch** was built and validated (hysteresis), and the remaining work — **Part 2:
making the full system DA-decisive** — was **fully mapped to four coupled conditions** but not closed. Part 2 is the
task. It is a deliberate multi-coupled calibration, not a quick sweep.

## 1. Read-order (ground first — do not skip)
1. `docs/README.md` — the documentation map: north-star, read-order, canonical vs stale, current frontier.
2. `session-discipline`, `agent-grounding-protocol` (skills) — how to work here; return a GROUNDING BRIEF first.
3. `docs/RESEARCH_DOPAMINE_CAMKII_REINFORCEMENT_2026-08-09.md` — **the primary Part-2 spec.** Read in full: the
   biology (Yagishita/Nakano/Zhabotinsky/Mayadevi), the correction, the F3-e blocker, the resolution experiment,
   **Part 1 (built)**, and **the Part-2 CONSTRAINT MAP + the exact next experiment.**
4. `RESEARCH_LOG_CALCIUM_DIMER.md` entry **F3-e** (newest) — the through-CaMKII correction + the calcium-domination
   finding. `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` — the topology sub-program (for context on what's SETTLED).
5. The code: `camkii_module.py` (bistable mode), `darpp32_pp1_module.py`, `model6_core.py:~690–745` (the reward
   commit path), `sweep/f3e_calcium_pp1_regime_probe.py`, `sweep/f3_delayed_credit_probe.py`.

## 2. SETTLED — do NOT relitigate (this is where the previous handoff went wrong)
- **The model is (A): a coherence-gated CLASSICAL correlated-partition computer** — quantum-derived quantities
  parameterize a classical computation. `quantum-system-canonical` §5.1 [LOCKED]. Never (B) language.
- **The directed, signed readout is the reward-gated binding-melt CASCADE** (spin-selective binding-melt → Ca²⁺ →
  glutamate → plasticity; dopamine gates/signs it). It is DIRECTED and it is real. **The direction comes from the
  REWARD signal (classically), NOT from mining the entanglement partition.**
- **"Is there a directed *quantum* readout out of the partition?" is a SETTLED NEGATIVE** (PO5-11, PO7-1, PO10-2,
  COV-1). A covariance-matched design chasing it (COV-2) was **off-program and reverted this session** — it
  relitigated the locked (A). **Do not go there.** (`PREREG_PO_COVARIANCE_MATCHED_DESIGN` is marked SUPERSEDED.)
- **Commitment routes THROUGH CaMKII (DDSC), never bypasses it** (`model6-commitment-pathway` LOCKED). The corrected
  F3 path honors this; dopamine REINFORCES CaMKII (via PP1), it does not commit around it.

## 3. DONE this session (committed local on master, NOT pushed — Sarah decides pushing)
- **Documentation overhaul**: `docs/README.md` map wired into `CLAUDE.md`; status headers on stale/misleading docs;
  prereg/handoff status index; misfiled docs foldered; the off-program COV commits reverted.
- **The correction (`c57c470`, `ff0ad62`, `5eaa9ff`, `61a4bd7`)**: dopamine→DARPP-32/PP1→CaMKII reinforcement;
  `pp1_factor` wired into CaMKII `k_dephos`; the F3 credit-bypass replaced by through-CaMKII commitment. Discrimination
  signatures re-validated (temporal-gap p=0.031, isotope p=0.012). The commit-rate is now stochastic (DDSC) — the
  deterministic pre-registered thresholds no longer pass, and that is honest, not a regression (see F3-e).
- **Part 1 (`7090069`)**: opt-in **bistable CaMKII switch** (Zhabotinsky/Lisman autocatalytic autophosphorylation vs
  saturating PP1; `bistable=True`, default off = bit-identical). VALIDATED hysteresis: a transient drive latches an
  autonomous UP state (pT286 0.72 held 30 s after drive-off); rest stays DOWN.

## 4. STEP 2 — the task (the frontier)
**Goal:** make the full reward-signed readout DA-DECISIVE — a dopamine burst commits (LTP), a dip does not — so the
reward signal actually decides commitment, at a physiological operating point. **Four coupled conditions must ALL
hold** (grounded this session; details in the research doc §Part-2 constraint map):
1. **CaMKII must integrate PSD-distance calcium, NOT the nanodomain peak.** `model6_core.py:602,833` feeds CaMKII
   `calcium_uM = np.max(ca_conc)·1e6` (channel-mouth peak). CaMKII is at the PSD; it should see the diffuse spine
   calcium (~1 µM, near its threshold). This is the calcium-domination root (canonical §2.3, in-flight).
2. **Drive near the switch threshold** (not saturating) — with `bistable=True`.
3. **Dopamine must FOLLOW calcium, not overlap it (Nakano):** at CaMKII-driving calcium, PP2B/calcineurin strips
   DARPP-32-Thr34 and keeps PP1 active, overwhelming dopamine's inhibition. Dopamine only gets a vote after calcium
   (and PP2B) decays.
4. **The bistable switch must hold near-threshold through the reward delay** — the coherent P_S tag's role.

**FIRST EXPERIMENT (deliberate, defined):** the **Nakano-timed protocol** — feed the (bistable) CaMKII a
PSD-distance near-threshold calcium during a brief eligibility window, hold, then deliver a **DELAYED** dopamine
burst (after PP2B decays) vs a dip; test whether dopamine now decides (burst→UP, dip→DOWN). PASS ⇒ wire
`bistable=True` + PSD-calcium into the reward-gated path and re-validate F3 end-to-end. FAIL ⇒ a deeper structural
finding (report it, do not tune around it).

## 5. Discipline (LOCKED)
- **Emergent physics only — GROUND, do not TUNE.** Every constant is cited or flagged `[MODELED]`; nothing is set to
  make a downstream result come out. A null is a result.
- **Validate at the data level.** The CONSUMER is a MEASUREMENT that distinguishes its outcomes with limits stated.
- **Thread-cap every run** (`OMP_NUM_THREADS=1 …`); background/daemonize long runs; per-draw checkpoint.
- Commit locally to master by EXPLICIT path; do NOT push. Append findings to the research log/doc; supersede, never
  rewrite.

## 6. FAILURE MODES from today — do NOT repeat
- **Do NOT act on a handoff without grounding.** This session executed the previous handoff's COV charter as gospel,
  spent hours on an off-program covariance experiment that relitigated the locked (A), and had to revert it. Ground
  against the README + log + code FIRST; a handoff is a pointer, not an authority.
- **Do NOT chase directedness from the quantum partition.** Direction comes from the reward signal. (§2.)
- **Do NOT parameter-hunt / sweep to force DA-decisiveness.** Part 2 revealed a NEW grounded coupled constraint at
  each sweep (PP1 strength → persistence tradeoff → drive/threshold → PP2B/dopamine timing). When a sweep isn't
  decisive, that usually means another grounded condition is unmet — STOP, find and ground it, then design one
  deliberate experiment. Do not tune constants until it "passes."
- **Do NOT bypass CaMKII** to make commitment reward-timed (that was the original bug this session fixed).
