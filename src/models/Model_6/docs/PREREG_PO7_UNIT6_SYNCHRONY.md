# PRE-REGISTRATION — PO-7 Unit 6: synchrony vs stagger at matched density

**Registered 2026-07-19, BEFORE the scored run. PO-7. Advisor R4 step 2.**

## Why this design and not HIGH-vs-LOW

The advisor's exit from the density confound: *"Same synapses, same total drive, different
temporal phase."* Every synapse is driven identically in both conditions — same burst duration,
same duty cycle, same total calcium integral. **The only difference is whether the
elevated-calcium windows of neighbouring synapses overlap in time.** Density is matched by
construction, not by covariate adjustment — which is exactly what Unit 2's design failed to do
(`notes.md` Q6: inactive synapses make no dimers, so activation and density were the same label).

It is also the biologically correct variable: coincidence detection is what the architecture
claims to do, and synchrony is the input dimension that claim is about.

## What makes it measurable (Unit 5)

The coincidence window is **≤ 50 ms** (p90 of creation→second claim; median below the 50 ms
sampling resolution), against a nominal 2.0 s event age. 84.3% of events are fully consumed
within one tracker step. So a stagger of ~200 ms — 4× the measured window — separates the
conditions, and the experiment is affordable at 6 synapses.

## Design

- 6 synapses, `pattern="linear"`, spacing **0.2 µm** (the only spacing with reliable cross
  edges, Unit 1), `provenance_network = True`.
- **Group A = {0,2,4}, Group B = {1,3,5}.** Adjacent synapses — the only pairs close enough to
  bond — are always in *different* groups.
- Burst 0.1 s, period 0.4 s, 3 periods (T_SIM = 1.2 s).
  - **COND-SYNC:** A and B both driven in `[0, 0.1)` of each period.
  - **COND-STAGGER:** A driven in `[0, 0.1)`, B driven in `[0.2, 0.3)`.
- **Every synapse is driven for exactly 0.1 s per 0.4 s period in BOTH conditions.**
- 5 seeds: `31337, 4242, 90210, 7, 123456`.

## Preconditions (run is INVALID if these fail)

1. `po7_bitident_check.py` prints PASS immediately before scoring.
2. **THE DENSITY CHECK — the thing Unit 2 lacked.** Mean dimer count per synapse must agree
   between conditions within **10%**. If it does not, density is *not* matched and the result is
   confounded exactly as Unit 2 was. This is checked and reported before any verdict is read.

## Statistics

- `n_cross` — cross-synapse provenance edges (primary).
- `n_multi` — components spanning ≥2 synapses.
- `n_dimers` — total dimers, per condition (the density check).
- Effect size: Cohen's `d` across 5 seeds, SYNC vs STAGGER.

## THE VERDICT FUNCTION

- **POSITIVE (coincidence detection):** `d(n_cross: SYNC vs STAGGER) ≥ 0.8` **with SYNC > STAGGER**
  **and** the density check passing.
- **NEGATIVE (no coincidence dependence):** `|d| < 0.3`.
- **ANOMALY:** STAGGER > SYNC with `d ≤ −0.8` — an anti-coincidence effect, which the mechanism
  gives no account of and which would mean something is wrong. Reported as anomaly, not as a
  result.

## THE PRE-REGISTRATION GUARD — what would make me conclude the OPPOSITE

| observed | verdict | achievable? |
|---|---|---|
| `\|d\| < 0.3` — cross edges equal in both | **NEGATIVE. The mechanism is NOT a coincidence detector**, and the Unit-1b ceiling is a power limit rather than a temporal window. This kills the advisor's reinterpretation of §2.2. | **Yes.** Unit 1 got cross edges under *continuous* drive with no synchrony structure at all, so yield may simply not care about phase. |
| `d ≥ 0.8`, SYNC > STAGGER | **POSITIVE — the partition depends on temporal input structure at matched density.** The first input-dependence result in this program not attributable to density or geometry. | **Yes.** Unit 5 shows sharing requires co-elevation within ≤50 ms, which stagger removes. |
| `d ≤ −0.8` | **ANOMALY** — unexplained by the mechanism; would trigger a re-read, not a claim. | Yes, and it would falsify my understanding of the claiming rule. |
| density check fails (>10% dimer-count gap) | **INVALID — not scored.** Same confound as Unit 2. | **Yes** — bursty drive could plausibly produce different totals. |

**Both verdicts have achievable values, and there is an explicit route to INVALID that is not a
verdict at all.** The statistic is not mis-registered.

## Known leak, and its direction (stated before running)

Events live 2.0 s nominally, and 15% are never claimed. Those survivors persist across a 200 ms
stagger and *could* be claimed by the later group. **This leak makes SYNC and STAGGER more
similar, not less** — it can only shrink the measured effect. So a POSITIVE survives the leak,
while a NEGATIVE is partly attributable to it. That asymmetry is reported with the verdict
either way, and it is why a null here will be stated as "no effect detected under a leak that
biases toward null", not as "the mechanism is not a coincidence detector".

## Committed in advance

- No constant tuned to reach a verdict. Burst/period/stagger are set from Unit 5's **measured**
  50 ms window (4× margin), not swept for effect.
- The Werner bound stays 0.5. `provenance_net_age_s` and `reach_nm` are untouched.
- The verdict function is demonstrated **failing** on synthetic negatives before it may pass.
- A negative is written up with the same weight as a positive.
