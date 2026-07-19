# PRE-REGISTRATION — PO-5 UNIT 16 · provenance-based bonding + the computation test

**Registered 2026-07-19 BEFORE the code change. Sarah authorised the build.** Tier-3 change to birth
physics, opt-in, flag-off bit-identical.

## What is built (one file, `dimer_particles.py`, gated by `self.provenance_bonding = False`)

Fisher's ACTUAL mechanism (per `L·PO5-12`, and faithful to LOCKED `quantum-system-canonical:43` — the
Ca₆(PO₄)₄ **dimer** with 2 singlet pairs, so K=2 events/dimer):

1. **Hydrolysis events.** Sourced where calcium is elevated — the same rule `atp_system.py:90`
   already uses (`active_sites = calcium > threshold`), so event locations are **input-correlated by
   construction** (calcium is input-driven; verified this session). Each event has **2 phosphate
   slots** (2 entangled daughters). Events age out after a window (phosphates consumed within
   seconds).
2. **Provenance at birth.** A newborn dimer claims up to K=2 **nearest** recent events that still have
   a free slot (deterministic by distance — no RNG, preserving the off-path stream). It records their
   ids in `event_ids`.
3. **Shared-event bonding.** Two dimers bond **iff they share an event** — i.e. they hold the two
   daughters of one hydrolysis. Because each event has exactly 2 slots, **an event yields at most one
   edge** (the pair that claimed it). The graph is a union of pairwise, input-located edges — the
   sparse RIG Unit 15 showed CAN carry input-dependent partition beyond density.

When `provenance_bonding` is False, the birth loop runs the existing clique rule unchanged.

## Bit-identity requirement
Flag OFF must reproduce the pre-change baseline bit-for-bit (`1034 / 369740 / 0.991922159684`). ALL
new work (event generation, the RNG it consumes, assignment, the new bond rule) sits inside
`if self.provenance_bonding:`. Verified before any result is scored; else INVALID.

## The computation test (Unit 16b)

The provenance channel is NOT through density — it is through WHERE calcium is elevated → which
dimers share events. So input can move the partition at FIXED population. Test:

- **Two spatial input conditions** (calcium elevated in different sub-regions), provenance ON.
- **Metric: Newman modularity Q of the input labeling on the bond graph** (density-corrected by
  construction — the Unit 15 metric), plus component partition.
- **P1 (channel opens):** provenance-ON graph is SPARSE and fragmented (not the clique blob);
  largest_frac < 0.9. If it is still a blob, the mechanism did not change the topology and the test
  is moot.
- **P2 (the keystone):** the partition tracks the input condition beyond density —
  Q(input) >> Q(shuffled), and the two conditions give partitions more different than seed noise
  (effect size d ≥ 2 on component structure), at matched population.
- **P3 (honest null):** if Q(input) ≈ Q(shuffled) and conditions don't separate, provenance did NOT
  rescue §8 even with a faithful mechanism — reported as the finding. **≥5 seeds** (the 3-seed scars).

## Limits
Single synapse, 1 s. The event-generation rate and aging window are modelling choices, reported and
swept, NOT tuned to a target. A positive result establishes the channel EXISTS in the model; it does
not by itself certify the rate constants are physical.
