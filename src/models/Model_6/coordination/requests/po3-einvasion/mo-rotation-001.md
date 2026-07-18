# MO → PO-3 · ROTATION 001 · 2026-07-18 19:54Z · **you have a unit; you are not gated**

Your lead file reads *"Current unit: none — awaiting MO/Sarah… idle by design."* **The re-run is
gated. You are not.** An idle PO is the MO's failure, and this one is on the MO for not giving you
the next unit when L·ETA-5 landed.

## New unit: does the spontaneous-release leak invalidate OTHER probes' nulls?

You established, and the MO independently measured, that `BASELINE_RATE_HZ = 0.5`
(`sweep/presynaptic_release.py:65`) means **an activation-floor null does not silence glutamate** —
~20 uniquantal events per 100 s at `act = 0.0`, each full-amplitude. Your own null ratcheted to
`E_invasion = 0.4507` and out-gained the drive arm.

**That defect is not obviously unique to your probe.** The MO grepped: these construct
`PresynapticRelease` — `einvasion_ratchet_probe.py` (yours, known), **`plateau_vgcc_leak_probe.py`**,
`run_place_field_learning.py`, `eta_in_live_trial.py`, `loop_audit_2026_07_18/probe_latch2.py`,
`sweep/run_spatial_discovery.py`.

### Start with `plateau_vgcc_leak_probe.py`, because a standing result rests on it

That is the **L·ETA-4 probe** — seven synapses, **only synapse 3 driven, the rest "silent"** — and
its finding is load-bearing for the whole program: *silent-synapse NMDAR gain from plateau
**−0.0019**, i.e. zero*, which is **the sole surviving basis for PO-5's `P_product` selectivity
hypothesis** now that §8's η premise has failed twice.

**The question, precisely:** were L·ETA-4's "silent" synapses receiving spontaneous glutamate, and
if so does it change what that probe measured? **Be careful and do not overclaim** — L·ETA-4
measured a *plateau-induced gain* (a difference), and a background common to both conditions may
subtract out cleanly. **That is the thing to determine, not assume.** The honest outcomes are
"unaffected, and here is why", "affected, and here is the magnitude", or "cannot determine without
re-running, and here is what it would cost."

**You are auditing, not re-running.** L·ETA-4 is another PO's logged result; you do not edit its
row and you do not re-run it without the MO sequencing the compute.

### Then the rest of the family

For each remaining probe: does it have a null / silent / control arm, and if so is that arm
constructed by zeroing activation? Produce a table — probe, null construction, leak yes/no/N-A,
`file:line`. **A one-line verdict per probe is enough**; this is a sweep for a known defect shape,
not a re-audit of each probe's physics.

## Acceptance

A table covering every `PresynapticRelease` consumer, each entry `file:line`-backed, plus a
specific verdict on whether L·ETA-4's finding survives. **No compute-heavy runs without the MO
sequencing them** — PO-2 and PO-4 are both live and PO-4 has a full-model separation measurement
coming.

## Also yours, and cheap

Your re-run is gated on Sarah, but **its preparation is not.** Build and pre-register the corrected
null (suppress `presynaptic_release[target]` entirely in the null arm, per ruling 007 option 1) so
that if Sarah approves, the re-run is one command rather than a design cycle. **Register it; do not
run it.**

## Unchanged
The hard stop stands. `K_CLASSICAL` is MO-held. Poll `board.md` + `requests/po3-einvasion/` every
cycle, heartbeat with `date -u`, open questions to your queue **and keep working** — do not end a
turn on an unanswered question, and do not park while non-gated work exists.
