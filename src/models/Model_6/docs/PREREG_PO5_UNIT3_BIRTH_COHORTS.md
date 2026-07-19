# PRE-REGISTRATION — PO-5 UNIT 3 · is the P0 bond graph an indifference graph on birth time?

**Registered:** 2026-07-19, BEFORE the run. **Reopened on Sarah's direction** after the PO-5 seat
closed. **Probe:** `src/models/Model_6/sweep/po5_unit3_birth_cohorts.py`.

## The structural claim, derived from code

`dimer_particles.py:218-228` forms a bond iff **both** dimers are `template_bound`, the older is
`is_entangled`, and `|Δbirth_time| < 0.1 s`. All dimers created in one `step_population` call share
`birth_time = self.time` (`:210`).

**Therefore the P0 graph is a unit-interval (indifference) graph on the birth-time axis**, and its
connected components are exactly the **maximal runs of birth times with no gap > 100 ms.**

This is a claim about mechanism, and it is falsifiable.

## THE PRE-REGISTERED PREDICTION — stated before any number is seen

Let `B` = sorted distinct birth times of **alive, template-bound** dimers at the sample, and let
`gaps` = consecutive differences.

> **PREDICTED components of the P0-only bond graph = `1 + count(gaps > 0.1 s)`.**

Scored as:

| verdict | condition |
|---|---|
| **CONFIRMED** | predicted == measured, at every sample, in both arms |
| **FALSIFIED** | predicted != measured anywhere |
| **INCONCLUSIVE** | fewer than 2 template-bound dimers, or no P0 bonds at all |

**One-sided caveat, registered in advance:** the disentanglement branch (`:460-466`) can dissolve an
existing bond, which can only **fragment** further. So strictly `measured >= predicted`. With
`k_disentangle = 0.01·(1-coh)/(1+protection)` and coh ≈ 0.996 over 5 s, the expected number of
dissolutions is negligible, so **equality is the prediction** and any excess is reported with its
size rather than explained away.

## The second question — can input create >100 ms formation gaps?

If sustained drive yields **zero** gaps > 100 ms, then `comps = 1` (L·PO5-1, L·PO5-3) is fully
explained by the birth mechanism alone, and §8's keystone acquires a concrete physical requirement:
**the input must be able to create >100 ms gaps in dimer formation.** Two arms:

| arm | drive |
|---|---|
| **SUSTAINED** | activation 0.95 throughout |
| **PULSED** | activation 0.95 for 0.2 s, then 0.0 for 0.4 s, repeating |

**The PULSED arm is NOT claimed to be silent.** `BASELINE_RATE_HZ = 0.5` means zeroed activation
still delivers spontaneous release, and dimer birth is driven by calcium, which does not stop.
Whether births actually pause is **the measurement**, not an assumption — this board has three scars
from controls assumed silent that were not, and this arm is deliberately labelled *reduced drive*.

**Registered outcomes:** if PULSED also shows zero gaps > 100 ms, that is a **substantive negative**
— it would mean the formation process cannot be gated finely enough to produce more than one
component at this timescale, and §8's keystone would fail through the birth channel for a stated
physical reason. Reported as a finding, not a protocol problem.

## Reported regardless

Gap distribution (max, count > 100 ms, quantiles), births per step, fraction of births
template-bound, alive template-bound dimer count, P0 component count, and — as a check on the
`is_entangled` precondition — the minimum `P_S` observed.

## Limits

Single synapse, 5 s, one seed per arm. This tests a **mechanism**, not input-selectivity; it does not
produce a §8 verdict, and none is claimed from it.
