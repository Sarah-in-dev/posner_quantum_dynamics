# PO-8 — readout-time computation (lead file, PO-8-owned)

**Objective:** measure the correlated-domain partition AT READOUT (dopamine-timed) and establish
whether INPUT determines which synapses share a correlated domain at collapse. Either answer is a
result; a clean decomposition-null is reportable.

**Worktree:** `.claude/worktrees/po5-keystone`, branch `claude/nervous-hertz-7ccff6`.
Python: `/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python`.

---

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
