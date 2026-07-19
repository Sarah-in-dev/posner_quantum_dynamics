# PRE-REGISTRATION — PO-7 Unit 2: the multi-synapse §8 keystone test

**Registered 2026-07-19, BEFORE the scored run. PO-7.**
Prior art: `PREREG_PO5_UNIT16_PROVENANCE_BUILD.md`, `sweep/po5_unit16_computation_test.py`,
`sweep/po5_unit15_rig_derisk.py`. Scars this design is built to avoid: `L·PO5-13` (spatial-half
false positive), `L·PO5-11` (topology = density alone), Units 8/9/13/16b (criterion
mis-registration).

## The question

§8 is structurally void at single-synapse scale — one nanodomain = one component
(`quantum-system-canonical:139`, LOCKED). The keystone lives cross-synapse, where the input has
a real degree of freedom: **which synapses are active.** Does the entanglement-graph partition
depend on *which* synapses are active, beyond what density alone explains?

## Preconditions (must hold or the run is INVALID and is not scored)

1. `po7_bitident_check.py` prints `BIT-IDENTICAL: PASS` (`1034 / 369740 / 0.991922159684`)
   with all physics edits in place and the flag OFF. **Re-run immediately before scoring.**
2. Unit 1 has SHOWN cross-synapse provenance edges forming at the chosen spacing.
3. `coupling_weights` is passed to `tracker.step` (else the η-graph is silently empty).

## Design

- 6 synapses, `pattern="linear"`, `provenance_network = True`.
- **Spacing is fixed BEFORE scoring** by this rule, applied to Unit 1's output: choose the
  spacing whose mean event-pool overlap is **closest to the middle of its observed range** —
  the partial-overlap regime (0 = trivial, full = trivial). **Explicitly NOT the spacing that
  maximises the outcome statistic.** Once chosen it is frozen for all arms and seeds.
- Active synapse: `voltage = -10 mV` + glutamate. Inactive: `-70 mV` resting, no glutamate.
- **Density matched by construction:** every condition activates exactly 3 of 6 synapses.
- **5 seeds** per cell (`31337, 4242, 90210, 7, 123456`).

### Two arms — and why ARM 2 is the whole point

| arm | COND-A active | COND-B active | activation identity vs space |
|---|---|---|---|
| **1 — contiguous** | `{0,1,2}` | `{3,4,5}` | **CONFOUNDED** (= spatial half) |
| **2 — interleaved** | `{0,2,4}` | `{1,3,5}` | **ORTHOGONAL** |

Under `pattern="linear"` (`multi_synapse_network.py:961-963`) synapses 0,1,2 are the left half
of the dendrite. So the kickoff's literal `{1,2,3}` vs `{4,5,6}` split **is the spatial half** —
running it alone would re-commit the exact L·PO5-13 error:

> the "input label" was **which spatial half a dimer sits in**, so Q=0.15 detects **spatial
> locality** … **That is GEOMETRY, which §8 explicitly rules insufficient**

**ARM 2 is the discriminator. ARM 1 alone cannot distinguish "the partition tracks which
synapses are active" from "the partition tracks where things are."**

## Statistics (fixed now)

### Amendment, registered BEFORE any scored run — the graph is SYNAPSE-level, not dimer-level

While implementing, I caught a mis-registration in my own first draft and am recording it rather
than quietly fixing it. The draft computed `Q` on the **dimer-level** graph. But
`intra_synapse_bonds_cache` is the dense per-synapse clique — the single-synapse fingerprint alone
carries **E = 369 740** intra edges — and **every intra edge lies inside one synapse, hence inside
one activation label.** Dimer-level `Q` against activation identity would therefore sit near 1.0
**by construction, for any input whatsoever**: a statistic that can only pass. That is exactly the
failure this document's guard exists to reject, and it would have manufactured a second false
positive on top of L·PO5-13.

**The graph is therefore the 6-node SYNAPSE graph** — the object
`quantum-system-canonical` §5 and `model6-entanglement-partition-werner` §1 both name:
> "synapses = nodes, cross-synapse bonds = edges"

Nodes are synapses; edge weight `W_ij` counts **cross-synapse** bonds (provenance + η, each above
the Werner bound). Intra-synapse clique edges are excluded — they carry no cross-synapse
information and their only effect is to saturate the statistic.

- `Q_act` — Newman modularity of the **synapse-level cross-bond graph** against the 2-group
  **activation-identity** label (each synapse labelled active / inactive in that run).
- `Q_shuf` — same, with synapse→label assignment permuted. The null.
- `n_multi` — number of connected components spanning **≥2 synapses**. The decomposition statistic.
- Effect size: Cohen's `d` across seeds.

## THE VERDICT FUNCTION

**PASS ("provenance carries input-dependent cross-synapse partition") requires ALL of:**
1. `mean(n_multi) ≥ 1` in **both** arms — cross-synapse components actually exist;
2. `d(Q_act vs Q_shuf) ≥ 0.8` in **both** arms;
3. ARM 2 not collapsed relative to ARM 1: `d(Q_act^arm1 vs Q_act^arm2) < 0.8`.

**Anything else is a NEGATIVE, reported as such.**

## THE PRE-REGISTRATION GUARD — what would make me conclude the OPPOSITE

Stated before the run, per the standing advisor-R3 guard:

| observed value | verdict | achievable? |
|---|---|---|
| `mean(n_multi) < 1` | **NEGATIVE — decomposition null.** Partition splits cleanly per-synapse ⇒ mechanism is input-**LOCATED**, not input-**COMPUTING**. A real result. | **Yes — and it is the most likely outcome.** Unit 1's first row gave 4 cross edges against 382 intra (~1%); a graph that sparse plausibly yields zero spanning components. |
| `d(Q_act vs Q_shuf) < 0.3` in either arm | **NEGATIVE — flat null**, the L·PO5-13 shape (that run measured `d = 0.02`). | **Yes** — this is precisely what the single-synapse test returned. |
| ARM 1 high but ARM 2 `d < 0.3` | **NEGATIVE — the signal is SPATIAL, not activation identity.** The L·PO5-13 false positive, caught. | **Yes** — nearest-event claiming is inherently spatially assortative, so this is a live outcome. |
| all three PASS criteria met | **POSITIVE** | **Yes** — Unit 15's RIG de-risk reached `Q = 0.65` against `Q_shuf ≈ 0` when event-sharing is assortative-by-input, so this range is reachable in principle. |

**Both verdicts have achievable statistic values, so the statistic is not mis-registered.** The
failure mode this guard exists to catch — a metric that can only ever pass — is excluded: three
independent, individually-sufficient routes to a negative are named above, and the most likely
of them (the decomposition null) is the one I expect.

## Committed in advance

- **No constant will be tuned to reach a verdict.** Spacing is selected by the overlap rule
  above, before scoring, and frozen.
- **The Werner bound stays at 0.5** (`quantum-system-canonical:140`: "do NOT lower it to rescue
  a result").
- The verdict function is demonstrated **failing** on synthetic inputs before it is allowed to
  pass on real ones.
- A negative is written up with the same weight as a positive. Per the kickoff: "**Either
  answer is a result.**"

## Stated limits (known before running, reported with the verdict either way)

- The provenance claim radius is a **2D projection** — `abs_xy` drops z while `pattern="linear"`
  jitters z by ±0.2 µm, biasing cross-edge formation **upward**. A positive would need
  re-checking in 3D. (`notes.md` Q3.)
- Cross-synapse edges require `spacing < 0.9 µm` at `reach_nm = 500`; the upper half of the
  physiological spine-spacing range cannot produce them at all. Any positive is conditional on
  the close-spacing regime.
- This is an **(A)-reading** result — classical common-cause correlated partition. It is not
  evidence for (B) and will not be described in (B)'s language
  (`quantum-system-canonical` §5.1, LOCKED).
