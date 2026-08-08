# PRE-REGISTRATION / PO CHARTER — the covariance-matched design (does a DIRECTED readout beat the abundance leak?)

**Written 2026-08-08, BEFORE the model change. Dispatched by the F-series session (see
`handoffs/SESSION_HANDOFF_2026-08-08_F-SERIES_AND_THE_REGROUND.md`).** This PO continues the *real* problem —
whether the entanglement partition is a genuine DIRECTED computation or "quantum constrains, classical computes."

## The arc (why now)
PO10 Unit C: the partition reaches the weights, but the binding-specific readout is SIGN-INVARIANT (decode ~0.75,
readout noise); any DIRECTIONAL (magnitude-weighted) readout LEAKS. **COV-1 (this session)** pinned the leak: a
covariance-across-trials readout, even with the whole-trial common mode removed, still decodes `scramble` 0.958
because **co-DRIVEN clusters have correlated committed-dimer COUNTS trial-to-trial** — a magnitude channel that
survives membership-scrambling and needs no binding. The matched-marginal design closed the MEAN abundance, never
the trial-to-trial CO-ACTIVATION COVARIANCE. **The dopamine sign-resolution job (F3) needs a leak-free directional
channel; this design is the prerequisite for it to be testable at all.**

## The hypothesis and the sharp risk
If a design MATCHES the co-activation covariance — paired (co-membered) and unpaired cluster pairs have EQUAL
trial-to-trial committed-dimer-count covariance by construction — then a directional covariance readout would be
binding-specific (it could only decode via the shared COLLAPSE COIN, i.e. binding, not counts). **The sharp risk,
which this PO must confront honestly:** binding (sharing a domain) arises FROM co-activation (co-active → cross-bonds
→ shared domain), and co-activation ALSO drives correlated counts — so binding and count-covariance may be
INSEPARABLE in this architecture. If so, that is itself a fundamental result: the partition cannot be read
directionally without the abundance leak, because *binding is count-correlation*. Either outcome is a real answer.

## GROUND first (agent-grounding-protocol)
- Read this doc's read-order sources (the handoff §Read-order). SHOW the code: `sweep/po10_unitC_experiment.py`
  (the 4-cluster pairing task + the arm logic full/bindoff/scramble/lamshort), `sweep/po10_covariance_readout.py`
  (the decoder + the leak diagnosis), `sweep/po10_unitC_score.py` (the registered sign-agreement scorer).
- **Reproduce first:** run `po10_unitC_score.py` → `full`=0.750, controls chance; run `po10_covariance_readout.py`
  → the COV-1 leak (scramble 0.958). Data is `ucB_<arm>_<mode>fwd/rev` in the OLD worktree
  `.claude/worktrees/trusting-heyrovsky-1338e9/results/po10_unitC` (path trap — see the handoff). Do NOT trust any
  new decoder until the registered baseline reproduces.
- Return a `### GROUNDING BRIEF` (line-quoted, source-tagged) as the first message if human-bridged; if autonomous,
  record it in the research log and PROCEED.

## The build (design a covariance-matched protocol; emergent-only)
Modify the Unit C task (opt-in flag, model bit-identical when off) so that, per pre-registered target, the
committed-dimer-count covariance is EQUAL between co-membered and non-co-membered cluster pairs. Candidate levers
(the PO chooses + justifies, does NOT tune to a result): partial/probabilistic co-activation, timing jitter that
preserves within-window coincidence (binding) while decorrelating counts, or a count-normalizing readout that is
provably orthogonal to the count channel. **The design must DECOUPLE "which pairs share a domain" from "which pairs
have correlated counts," or establish that they cannot be decoupled.**

## Pre-registered acceptance (the CONSUMER is a measurement; a null that can FALSIFY)
1. **The matching worked (gate, measured before scoring):** a pilot shows paired vs unpaired cluster pairs have
   EQUAL trial-to-trial committed-dimer-count covariance (to a stated tolerance) under the new design. If the
   matching cannot be achieved, that is the "binding = count-correlation is inseparable" finding — report it.
2. **The directional readout, scored on the control ladder** (`po10_covariance_readout.py`, common-mode removed):
   - **PASS (ceiling REMOVABLE):** the covariance readout DECODES `full` (> its own shuffle null, ideally > 0.75)
     AND is CHANCE on `scramble` AND `bindoff` AND `lamshort`. Then there IS a directed computation.
   - **NULL (ceiling is HONEST):** the covariance readout still decodes `scramble` (leak persists) OR loses `full`
     once counts are matched (the directed signal WAS the count channel). Then ~0.75 sign-invariant is the honest
     output — "quantum constrains, classical computes" — a real result, reported not engineered.
3. Sign-agreement (~0.75) and raw-magnitude (leak) must be re-reported as references on the new data.

## Discipline (LOCKED)
Emergent physics only — no constant tuned to the outcome. Pre-register the discriminating quantity + the null +
the FALSIFIED path BEFORE running. Thread-cap + background/daemonize long runs (see handoff §Discipline). Write
findings to `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` (append newest-first; supersede, never rewrite). Return a
measurement that distinguishes its outcomes, with limits stated — not "the probe ran."

## Artifacts to produce
- Grounding brief (research log). The covariance-matched harness (opt-in, bit-identical-off) + a pilot proving the
  count-covariance match (or its impossibility). Results under `results/`. A `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`
  entry (COV-2) with the verdict: removable vs honest-ceiling vs inseparable. Do NOT push; commit locally.
