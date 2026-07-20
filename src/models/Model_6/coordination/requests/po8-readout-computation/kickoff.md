# PO-8 KICKOFF — the readout-time computation: measure the partition at collapse, and turn it into an engine
**Dispatched 2026-07-20, on Sarah's instruction. Follows the PO-7 measurement-is-computation reframe (`L·PO7-5`).**

## Your one objective
Measure the **correlated-domain partition at READOUT** — the graph as it stands when a dopamine
event triggers decoherence — and establish whether **input determines which synapses share a
correlated domain at collapse.** That partition is the computational output. If it carries input
structure, we have a working computation engine; wire the readout so the rest of the system can
consume it. **Either answer is a result.**

This is **not** "build more graph." PO-7 established the graph. Your job is the *reading* of it,
at the right time, with the right measure.

---

## GROUND FIRST — return a `### GROUNDING BRIEF` before any code (line-quoted, tagged by source)
Read in full, in this order:
1. `.claude/skills/agent-grounding-protocol`, `session-discipline`, `experiment-design-patterns`,
   `model6-entanglement-partition-werner`, `quantum-system-canonical` §5.
2. **The research log `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`, entries `L·PO7-1` … `L·PO7-5`** —
   the whole PO-7 arc. `L·PO7-5` is the reframe and is the one you must internalise; `L·PO7-2` is
   the withdrawal you must not re-tread (below).
3. `docs/PO7_TECHNICAL_BRIEF_2026-07-20.md` (the physics in equations) and
   `docs/PO7_ADVISOR_REVIEW_2026-07-20.md` (the open questions we put to the reviewer).

**⚠ BEFORE CONSTRUCTING ANY PROTOCOL: grep the research logs (`RESEARCH_LOG_CALCIUM_DIMER.md`,
`RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`) and `sweep/` for prior art on the exact thing you are
about to build.** Four times in one day a worker on this program (PO-7 included) went and measured
something that was already logged. A grep for `gap` surfaces `analytical_gap`,
`gap_retention_probe.py`, `L·GAP-1` and `L·GAP-4` immediately. This is the single highest-yield
habit here.

Your grounding brief must include a **line-quoted read of `_step_network_provenance` and the
cross-bond formation block in `multi_synapse_network._update_entanglement`** — you inherit both and
must verify, not trust, them.

---

## THE ABSOLUTE RULE: NO SEEDING. The stochasticity IS the physics.
This system is stochastic by construction (vesicle release, channel gating, dimer birth). Its
output is a **distribution**, measured over **free-running draws** — never a single seeded
trajectory. **Do not `np.random.seed()`. Do not pass an integer seed to any constructor.**
`PresynapticRelease(None)` = OS entropy = a free draw. If you ever want to fix a seed to make a
run reproducible or to make something ignite, **STOP** — that is the exact error that cost PO-7
hours (it recurs; see `L·PO7-4` §7). The lone exception: the bit-identity regression gate is a
*software* check for a disabled opt-in flag, never a physics measurement.

**Drive via `net.step(dt, stimulus)`, NEVER per-synapse `s.step()`.** `_update_backbone_field()` —
the only setter of `_backbone_eta`, which gates the entire cross-synapse channel — runs *only*
inside `net.step()` (`multi_synapse_network.py:1286`). A probe that steps synapses individually
leaves η≡0 and nothing ignites. This trap already produced one false "ignition is a coin-flip"
finding (`L·PO7-4` §7). It is the single most likely way to waste a day.

---

## WHAT IS ESTABLISHED (do NOT re-derive; build on it)
All in `L·PO7-1…5`, measured:
- **Monogamy:** 4 ³¹P spins/dimer ⇒ ≤4 bonds. `spin_resolved` flag enforces it (opt-in,
  regression-gated). The 715-degree blob was 97% manufactured by the clique + EM pathway, both
  unphysical; `provenance_bonding=True` drops them.
- **The computation is the MEASUREMENT.** Graph = program written at birth, executed by dopamine.
  Fidelity (reading) and input-selection (writing) are separable tests.
- **Joint collapse is graded:** Werner correlation `p=(4F-1)/3`, multiplies along paths. The
  computational unit is the **correlated domain**, NOT the connected component. Use the correlation
  metric d(u,v)=Σ(-ln p_e), effective correlation e^{-d}. **No F-threshold — the exponential does
  the work.** (Do NOT nominate an F-cut; that was the four-round dead end.)
- **Ceiling is geometry:** F_max = exp(-spacing/λ); spine spacing sets inter-synaptic correlation
  length. 1µm→p≤0.76, 2µm→0.56, 3µm→0.40.
- **Domains at WRITE-TIME are synapse-scale** (~1.35 synapses; intra F=1 lossless, cross F≈0.7),
  giving ~45 effective domains vs 1 giant component — structure where connectivity is pinned.

## ⚠ WHAT IS WITHDRAWN — do not resurrect (see `L·PO7-2`)
Phosphate provenance is **LOCAL**. It is **not** a cross-synapse channel — cross-synapse
entanglement is condensate-mediated (`k_cross=0.5·√(η_iη_j)·w_spatial·P_product`,
`model6-entanglement-partition-werner` §2). The "network-shared event pool" code is retained but is
**not a physical claim**. Do not build on it as cross-synapse physics.

---

## YOUR UNITS, IN ORDER

### Unit A — implement the corrected bond-release rate (physics change; opt-in + regression-gated)
The graph is effectively write-once: bonds dissolve at ~1e-4/s but the physical rate is
**k = 1/T₂ + 1/τ_dimer = 1/216 + 1/200 = 9.63e-3/s** (τ≈104 s), **96× faster** (`L·PO7-4` §2,
independently checked against the model's own 107 s Werner-floor crossing). Make it opt-in
(`physical_release_rate` flag or similar), off = unchanged, and use the **network-path regression
gate** (`sweep/po7_unit11_shared_ledger.py MODE=offpath`, fingerprint `515772101786800`) — NOT
`po7_bitident_check.py`, which does not exercise `_update_entanglement` (`L·PO7-3` §7). Optionally
make it **fidelity-dependent** per the reviewer: a bond born at F₀ lives t_bond=T₂·ln(4F₀-1), so
weak bonds die first and the graph self-cleans — verify that behaviour at the data level.

### Unit B — the READOUT experiment (the keystone, done right)

**PROTOCOL: drive (write) → `analytical_gap(net, delay_s, dt_sub=1.0)` → dopamine (read).**
Import as the existing consumer does (`sweep/gap_retention_probe.py`):
`from run_theta_burst_45s import analytical_gap`.

**⚠ Do NOT hand-roll a quiet period by dropping voltage/glutamate and stepping normally.**
`step_population` slaves the dimer count to **instantaneous** calcium
(`target_count = peak_conc × az_volume_L × N_A`, `dimer_particles.py:258`), so a collapsed calcium
field **CULLS the population and deletes the graph with it** (`_remove_all_bonds_for_dimer`, `:336`).
That is deletion, not decay — a trap PO-8 hit on its first attempt (`requests/po8-readout-computation/
po7-001-use-analytical-gap.md`).

**The graph DOES persist through a proper gap — already measured, do not re-measure.** PO-4's
`L·GAP-4` (`docs/RESEARCH_LOG_CALCIUM_DIMER.md`; `docs/PREREG_PO4_GAP.md` AMENDMENT E) at the
corrected `K_CLASSICAL = 0.005 s⁻¹`: **survival 0.9926 at 20 s, 0.9676 at 45 s.** Read that, plus
`sweep/gap_retention_probe.py`, before designing. The decay that does occur is the **self-cleaning**
the reframe predicts (`t_bond = T₂·ln(4F₀−1)`, weak bonds die ~4× faster), so the graph **at readout
is ENRICHED in high-fidelity bonds** relative to write time — a feature, not a loss.

Free-running ensemble (≥12 draws, no seed). Drive to a **dopamine event at a realistic delay**
(tens of s; sweep the delay as an independent variable — it sets how much P_S decay has occurred).
At the dopamine step, measure the **correlated-domain partition** (the d(u,v) metric, effective
domains, domain sizes). Two input conditions, **same drive strength, different identity** — the
cleanest is **synchronous vs staggered activation of neighbour groups** (matched density by
construction; avoids the on/off density confound that sank PO-7 Unit 2 — see `notes.md` Q6 in
`coordination/requests/po7-provenance-network/`). **Pre-register before scoring**, with the guard:
state what domain-partition value would make you conclude input does NOT structure the readout.
The decomposition null (partition splits per-synapse with no input dependence) is a real,
reportable negative.

### Unit C — only if B is positive: wire the readout for consumption
The output is an **agreement pattern** (which synapses collapsed correlated). Expose it in the form
CaMKII / downstream plasticity can consume. Scope this with Sarah before building.

---

## BOUNDARIES / DISCIPLINE
- Explicit-path `git commit -- <paths>` only, **never `git add -A`**; `git show --stat HEAD` after
  each. Force-add JSON (`git add -f`; `results/` is gitignored). Persist traces as-you-go (a
  scoring crash once destroyed 58 min of physics — `L·PO5-3`).
- Every physics change: opt-in, off-path regression-gated (network path, not the synapse-only
  fingerprint). Demonstrate any verdict function FAILING before it passes. ≥5 free draws for any
  scored claim (the 3-seed scars).
- Do NOT **MODIFY**: `spine_plasticity_module.py`, `atp_system.py` phosphate path, `analytical_gap`,
  `sweep_runner.py`. **⚠ `analytical_gap` you MUST USE — see Unit B. "Do not touch" means do not
  EDIT it; it is the documented mechanism for advancing through a silent delay, and using it is
  required, not forbidden.** Do NOT move the Werner bound (0.5, LOCKED). Do NOT re-open
  cross-synapse provenance.
- **Use subagents** for the ensembles and the release-rate build — they parallelise and keep the
  main thread grounded. Cap concurrency at 4.
- Heartbeat to `coordination/leads/po8-readout-computation.md` each cycle with a `date -u` stamp.

## RETURN
The grounding brief; Unit A committed with the network-path regression proof; the pre-registered
Unit B with its guard and the readout-time domain-partition result (positive or a clean
decomposition null); a `RESEARCH_LOG` entry `L·PO8-1` + DECISION RECORD row. If B is positive, the
Unit C plan. **The bar: we should come out of this knowing whether the computation engine reads
input — and if it does, how to plug it in.**
