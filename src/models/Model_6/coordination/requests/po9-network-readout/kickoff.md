# PO-9 KICKOFF — the dimer network across synapses and across time, read out at dopamine
**Dispatched 2026-07-20 on Sarah's instruction. Supersedes the PO-8 dispatch, which failed.**

> **Why PO-8 failed, so you don't repeat it.** Its kickoff was a task list written for someone who
> already understood the system. It never taught the model, and it listed the one function needed
> for the core experiment (`analytical_gap`) inside a "do not touch" list — so PO-8 avoided it,
> hand-rolled a silent period by zeroing the drive, watched the model delete its own dimer
> population, and concluded the physics was broken. **The answer was already on disk.** This
> document therefore teaches the system first and gates you on comprehension before you write code.

---

# PART 0 — YOUR FIRST DELIVERABLE IS A COMPREHENSION GATE, NOT CODE

Before any code, return a `### GROUNDING BRIEF` that **answers the eight questions in Part 6**.
Each answer needs a **line-quoted citation** (`file:line` or research-log entry ID). If you cannot
answer one from the documents, say so explicitly — that is a passing answer; guessing is not.

**The single highest-yield habit on this program: the answer is usually already on disk.** Four
separate times in one day a worker measured something already logged. **Before you construct any
protocol, grep `docs/RESEARCH_LOG_*.md` and `sweep/` for the thing you're about to build.** A grep
for `gap` surfaces `analytical_gap`, `gap_retention_probe.py`, `L·GAP-1`, `L·GAP-4` instantly.

**Read in this order:**
1. Skills: `agent-grounding-protocol`, `session-discipline`, `experiment-design-patterns`,
   `model6-entanglement-partition-werner`, `quantum-system-canonical` §5.
2. `docs/RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entries **`L·PO7-1` … `L·PO7-5`** (the foundation).
3. `docs/RESEARCH_LOG_CALCIUM_DIMER.md` entries **`L·GAP-1`, `L·GAP-4`** (how silence is modelled).
4. `docs/PO7_TECHNICAL_BRIEF_2026-07-20.md` (the physics in equations).
5. `coordination/README.md` (how POs coordinate — you run this loop).

---

# PART 1 — WHAT THIS SYSTEM MODELS (the physics, in plain terms)

**The biological question.** A brain links an action to a reward that arrives *tens of seconds
later*. Every classical synaptic mechanism has decayed by then. The hypothesis: a molecular
quantum coherence holds the causal link open across a **100–200 s window**.

**The molecule.** The carrier is the **Ca₆(PO₄)₄ dimer** — a calcium-phosphate cluster carrying
**four ³¹P nuclear spins**. (Not the Ca₉(PO₄)₆ Posner *trimer*; that lacks a symmetry axis and
decoheres sub-second. LOCKED, `quantum-system-canonical:43`.) Dimers form stochastically in a
dendritic spine wherever calcium is elevated.

**How dimers become entangled — two channels, and they are NOT the same:**

- **LOCAL (within one synapse) — Fisher inheritance.** ATP hydrolysis releases two phosphates in a
  correlated singlet state. Two dimers that capture the two daughters of *one* hydrolysis event are
  born entangled. Strictly pairwise, provenance-based. **Phosphates do NOT travel between
  synapses** — this channel is local, full stop.
- **CROSS-SYNAPSE — condensate-mediated.** A microtubule backbone enters a Fröhlich condensate when
  metabolic power crosses threshold `P_c = 21.51 fW`. That condensate mediates entanglement between
  dimers in *different* spines. This is the only cross-synapse channel. It is field-mediated, not
  diffusive.

**The gate on the cross-synapse channel.**
`r = P_agg/P_c`, and `η = (r−1)/(r+1)` if `r ≥ 1`, else **exactly 0**. Bond rate
`k_cross = K·√(η_i·η_j)·W_ij·P_S^a·P_S^b`. Because η enters as a **geometric mean**, a single
un-condensed endpoint zeroes the channel. When people say "the pump ignites," they mean r crossed 1.

**Bond strength.** Each bond is a **Werner state** of fidelity
`F = P_S^a · P_S^b · W_ij`, where `W_ij = exp(−d_ij/λ)`, `λ = 5 µm` is the condensate coupling
length. A bond counts only if `F > ½` (Werner separability bound — **physics, LOCKED, never tune
it**).

**Why that matters for you:** since `P_S ≤ 1`, **F is capped by the spatial weight.** At 1 µm
spacing `exp(−1/5) = 0.8187`, and measured max across 28,673 edges is **0.8151**. So **synapse
spacing sets the maximum cross-synapse bond strength** — not metabolism.

---

# PART 2 — WHAT THE COMPUTATION IS (the reframe — internalise this)

**The entanglement graph is NOT the output. It is the PROGRAM.**

- **Written** at dimer birth — provenance fixes which dimers share an origin.
- **Held** in the coherence window while the animal does other things.
- **Executed** when **dopamine** arrives and triggers decoherence. The correlated collapse *is* the
  readout.

Three consequences you must hold onto:

1. **Writing and reading are separate tests.** "Does input determine the graph" (writing) is a
   different question from "how faithfully does collapse express it" (reading). A limitation in one
   does not falsify the other.
2. **The output is an AGREEMENT PATTERN** — *which dimers collapsed to correlated outcomes* — not a
   number. That is what downstream CaMKII can consume.
3. **Joint collapse is GRADED, not thresholded.** For a Werner bond, the correlation coefficient is
   **p = (4F−1)/3** (F=0.5→p=0.33; F=0.67→0.56; F=0.815→0.75). **Correlation multiplies along
   paths.** So connectivity and correlation are *different lengths*: the graph can be one connected
   blob while containing many separate **correlated domains**.

**Therefore the computational unit is the CORRELATED DOMAIN, not the connected component.** Measure
it with the correlation metric: edge weight `w_e = −ln(p_e)`, distance `d(u,v) = Σ w_e` along the
best path, effective correlation `e^{−d}`. **There is NO fidelity threshold to nominate** — the
exponential does the work. (Four earlier rounds were wasted hunting an F-cut. Do not resume that.)

---

# PART 3 — THE FOUNDATION (what PO-7 established; build on it, do NOT re-derive)

| finding | number | where |
|---|---|---|
| Monogamy: 4 ³¹P spins ⇒ **≤4 bonds/dimer**, derived not capped | graph had **mean degree 715**, 99.44% of edges physically impossible | `L·PO7-3` |
| Two unphysical mechanisms (100 ms "clique" + phenomenological EM) manufactured the blob | **97%** of intra bonds (3.80 of 3.93/dimer) | `L·PO7-4` |
| `provenance_bonding=True` drops both; `spin_resolved=True` enforces monogamy | with both on, degree 0.13, monogamy never binds | `L·PO7-4` |
| Condensation ignites reliably | **16/16** free draws; peak η 0.33–0.49 | `L·PO7-4` |
| Fidelity ceiling is **geometry** | `F_max = exp(−spacing/λ)`; 0.8187 predicted vs 0.8151 observed | `L·PO7-5` |
| Correlated domains exist where connectivity is pinned | **45 effective domains** vs 1 giant component | `L·PO7-5` |
| Intra bonds are **lossless** at write time (P_S≈1 ⇒ F≈1.0000); cross bonds F≈0.70 | domain ≈ 1.35 synapses at 20 s | `L·PO7-5` |
| Bond release rate in code is **96× too slow** — derived `k = 1/T₂ + 1/τ_dimer = 1/216 + 1/200 = 9.63e-3/s` (τ≈104 s) | code runs ~1e-4/s | `L·PO7-4` |
| **Dimers SURVIVE a silent gap** | survival **0.9926 @ 20 s**, **0.9676 @ 45 s** | `L·GAP-4` |

**The open question you inherit:** those domain measurements are all at **write time** (20 s, P_S
still ≈1, so intra bonds are lossless and a whole synapse collapses as one unit). **Nobody has
measured the graph at READOUT** — after a delay, when P_S has decayed and weak bonds have died.
That is your job.

---

# PART 4 — THE MECHANICS YOU MUST USE (enablers, not prohibitions)

**1. `analytical_gap` — how silence is modelled. YOU MUST USE THIS.**
```python
from run_theta_burst_45s import analytical_gap
analytical_gap(net, gap_s, dt_sub=1.0)
```
It advances the network through a quiet interval, integrating: P_S decoherence toward 0.25;
dissolution `k_diss`; **removal of lowest-coherence particles**; bond cleanup below P_S<0.5;
stochastic disentanglement; actin/E_invasion (τ_extrude 180 s); spine volume (τ 5 s); CaMKII/DDSC
(Jain's 30–40 s window). Working consumer: `sweep/gap_retention_probe.py`.

> **⚠ DO NOT hand-roll a quiet period** by dropping voltage/glutamate and stepping normally.
> `step_population` slaves the dimer count to **instantaneous** calcium
> (`target_count = peak_conc × az_volume_L × N_A`, `dimer_particles.py:258`), so a collapsed
> calcium field **culls the population and deletes bonds with it** (`:336`). That is *deletion*,
> not decay. **This is exactly how PO-8 failed.**
> `analytical_gap` may be **modified by nobody** — but **using** it is required.

**2. Drive with `net.step(dt, stimulus)` — NEVER per-synapse `s.step()`.**
`_update_backbone_field()` — the **only** setter of `_backbone_eta`, which gates the entire
cross-synapse channel — runs **only inside `net.step()`** (`multi_synapse_network.py:1286`).
Per-synapse stepping silently leaves η≡0 and nothing ignites. This already produced one false
"ignition is a coin-flip" finding.

**3. NEVER SEED. The stochasticity IS the physics.**
Vesicle release, channel gating, dimer birth are random; the output is a **distribution over
free-running draws**. Do not call `np.random.seed()`; do not pass an integer seed to a constructor.
`PresynapticRelease(None)` = OS entropy = a free draw. **If you find yourself wanting to fix a seed
so a run reproduces or so something ignites — STOP.** That is selecting the outcome. (The sole
exception: a bit-identity *regression* check that a disabled opt-in flag changed no code path —
never a physics claim.)

**4. Regression-gate physics changes on the NETWORK path.**
Use `sweep/po7_unit11_shared_ledger.py MODE=offpath` (fingerprint `515772101786800`).
**Not** `po7_bitident_check.py` — that drives one synapse and never exercises
`_update_entanglement`, so it would pass even if you broke cross-synapse bonding entirely.

---

# PART 5 — YOUR OBJECTIVE AND UNITS

## Objective
**Model the dimer network across synapses AND across time, and reconcile it as a readout at
dopamine.** Concretely: does the correlated-domain partition *at the moment dopamine arrives*
carry information about *which input was presented* — and does it still carry it after a
behaviourally realistic delay?

### Unit A — the corrected bond-release rate
Implement `k_release = 1/T₂ + 1/τ_dimer ≈ 9.63e-3/s` (opt-in flag, off = unchanged, network-path
regression-gated). Optionally fidelity-dependent: a bond born at F₀ lives `t_bond = T₂·ln(4F₀−1)`
(F₀=0.815 → 176 s; 0.67 → 112 s; 0.55 → 39 s), so **weak bonds die ~4× faster and the graph
self-cleans**. Validate at the data level: show the fidelity distribution *shifting upward* over a
gap. That self-cleaning is the eligibility trace doing its job.

### Unit B — the readout experiment (the keystone)
**Protocol: drive (write) → `analytical_gap(delay)` → dopamine (read).** Sweep the delay
(e.g. 5 / 20 / 45 / 90 s) — it is the independent variable that sets how much P_S decay has
occurred. Free-running ensemble, **≥12 draws, no seed**.

**Two input conditions, matched density by construction:** synchronous vs staggered activation of
neighbour groups, **all synapses driven** in both. **Do NOT use on/off** — inactive synapses make
no dimers, so on/off confounds input identity with material density. (That confound sank an earlier
attempt; see `coordination/requests/po7-provenance-network/notes.md` Q6.)

**Score:** the correlated-domain partition at the dopamine step — effective domain sizes, domain
count, and whether domain membership tracks the input condition. **Pre-register before scoring**,
including the guard: *state what value would make you conclude input does NOT structure the
readout.* **Demonstrate your verdict function failing before it passes.**

**The decomposition null is a real, reportable negative:** if domains split cleanly per-synapse with
no input dependence, the mechanism is input-*located* but not input-*computing*. Report it as a
result, not a failure.

### Unit C — only if B is positive
Expose the agreement pattern (which synapses collapsed correlated) in a form downstream plasticity
(CaMKII) can consume. **Scope with Sarah before building.**

---

# PART 6 — THE COMPREHENSION GATE (answer all eight in your grounding brief)

1. What molecule carries the computation, how many ³¹P spins does it have, and what does that imply
   about the maximum number of entanglement bonds per dimer?
2. Name the **two** entangling channels. Which one can act between different synapses, and which
   one physically cannot — and why?
3. Write `p = ?` in terms of F, and explain in one sentence why the computational unit is a
   *correlated domain* rather than a *connected component*.
4. What sets the maximum possible fidelity of a cross-synapse bond? Give the formula and the number
   at 1 µm spacing.
5. What happens to the dimer population if you simulate silence by setting voltage to rest and
   glutamate to zero, and what should you use instead? Cite the line numbers.
6. What is `_backbone_eta`, what is the *only* function that sets it, and what breaks if you drive
   synapses individually?
7. Why must you never fix a random seed for a physics measurement here — and what is the one
   legitimate exception?
8. In one sentence each: what did PO-7 establish about (a) monogamy, (b) the clique/EM pathways,
   (c) the fidelity ceiling, (d) whether the graph survives a quiet gap?

**Then, and only then**, state your Unit A plan.

---

# PART 7 — DISCIPLINE

- **Validate at the DATA level.** "It ran" / "committed" / "errors=0" is not acceptance. Acceptance
  is a measurement.
- **Explicit-path commits only** (`git commit -- <paths>`), **never `git add -A`**. Run
  `git show --stat HEAD` after each and confirm nothing unintended was swept. Force-add JSON
  (`git add -f`; `results/` is gitignored). **Persist traces as you go** — a scoring bug once
  destroyed 58 minutes of physics.
- **≥12 free draws** for any scored claim; report **distributions**, never a single run.
- **Do NOT MODIFY:** `spine_plasticity_module.py`, `atp_system.py` phosphate path, `analytical_gap`,
  `sweep_runner.py`. **Do NOT move the Werner bound (0.5).** **Do NOT re-open cross-synapse
  provenance** — phosphate provenance is local, settled in `L·PO7-2`.
- **Use subagents** for ensembles and builds (cap concurrency at 4).
- Run the PO loop in `coordination/README.md`: heartbeat to
  `coordination/leads/po9-network-readout.md` each cycle with a `date -u` stamp; append anything
  needing Sarah to `coordination/queue/po9-network-readout.md` with the ask, why, and your
  recommendation; **check `coordination/requests/po9-network-readout/` every cycle** for messages.
- **If you get stuck or something contradicts this document, SAY SO and ask.** Do not invent a
  workaround. PO-8's failure was avoidable; it ran a protocol on a wrong mental model instead of
  asking.

## RETURN
The grounding brief with all eight gate answers; Unit A committed with its network-path regression
proof and data-level validation of self-cleaning; the pre-registered Unit B with its guard and the
readout-time domain-partition result (positive **or** a clean decomposition null, with stated
limits); a `RESEARCH_LOG` entry `L·PO9-1` + DECISION RECORD row; the Unit C plan if B is positive.

**The bar: we come out of this knowing whether the dimer network, evolved across synapses and
across a realistic delay, still carries which input it saw — at the moment dopamine reads it.**
