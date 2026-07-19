# PRE-REGISTRATION — PO-5 UNIT 7 · where is the percolation threshold actually?

**Registered 2026-07-19 BEFORE the run.** Supersedes the design promised in Unit 4's follow-up.

## Why the planned sweep was the wrong sweep

Unit 4 reported `largest_frac` and PO-5 under-read it. **At bus = 0, with ZERO P2 bonds,
`largest_frac = 0.8866`.** The giant component is already present from birth-pairing alone. Across
the whole bus range `largest_frac` only moves 0.887 -> 1.000.

**So the bus never forms the giant component; it absorbs residual stragglers.** `L·PO5-5`'s
headline — *"the BUS is a real percolation control parameter"* — is **overstated**, and this unit is
registered to test that as a self-correction rather than to defend it.

The structure that percolates is **P0 birth-pairing**, whose control parameter is the birth window
(`dimer_particles.py:222,389`, `birth_window = 0.1` s), not the bus.

## Refactor required, and it must be behaviour-identical

`birth_window` is a LOCAL literal at two sites, so it cannot be overridden from outside. It is
promoted to `self.birth_window = 0.1` and both sites read it. **Same value, so behaviour is
unchanged — verified bit-for-bit against pre-refactor code before any arm is scored.** If it is not
bit-identical, the unit is INVALID.

## Predictions

- **P1 (the self-correction).** `largest_frac >= 0.85` at EVERY bus value including 0.
  ⇒ the bus does not control giant-component formation, and `L·PO5-5`'s framing is corrected.
- **P2.** Reducing `birth_window` fragments the graph: there exists a window at which
  `largest_frac < 0.5`. That is the real transition.
- **P3.** The susceptibility chi = sum(s^2)/sum(s) over FINITE clusters (largest excluded) **peaks**
  at that window, not at any bus value. chi peaking is the textbook percolation locator; a component
  count is a step function and is noisiest exactly where it matters.

**If P2 fails** — if no accessible birth window fragments the graph — then the model has NO
subcritical regime at all, the topology cannot carry information at any setting, and that is the
finding. It is reported, not engineered around.

## Design

Multiple seeds per point, because fluctuations DIVERGE at a critical point and Unit 4's single seed
was weakest exactly where it needed to be strongest.

| arm | bus | birth_window | seeds |
|---|---|---|---|
| BUS | 0, 1, 10, NATIVE | 0.1 (native) | 2 |
| BW@0 | 0 (P0 isolated) | 0.002, 0.01, 0.05, 0.1 | 2 |
| BW@NATIVE | NATIVE | 0.002, 0.01, 0.05, 0.1 | 2 |

Also recorded: the NATIVE bus **distribution** (mean/std/min/max), not just its mean. If it
fluctuates across the threshold within a run, the system is crossing it dynamically — a third
picture, and closer to the SOC story than either static answer.

## Limits
Single synapse, 1 s, 2 seeds/point. Locates a threshold in THIS regime; does not establish that the
biological system sits anywhere near it.
