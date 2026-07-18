# Queue: po3-einvasion — actions awaiting Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready,
so clearing it is seconds per item, not a re-derivation.

---

## Q1 · 2026-07-18 · L·ETA-3's `ca_open` attribution was measured through a harness that under-delivers glutamate ~100×

**The exact ask:** decide whether L·ETA-3's DECISION RECORD row needs a correction banner.
I am not touching it — the research logs are "each PO writes its OWN entries; nobody
rewrites another's", and L·ETA-3's verdict is the thing this PO was dispatched to test.

**Why.** `sweep/eta_in_live_trial.py:138-144` — the L·ETA-3 harness — steps presynaptic
release **once per agent step** (0.5 s) and then runs 100 physics steps against that one
stale stimulus. The shipped reference, `run_spatial_discovery.py:434-441` (`run_trial`),
steps it **inside** the physics loop, once per 0.005 s.
`PresynapticRelease.step` (`presynaptic_release.py:110-139`) is a per-timestep Bernoulli
draw (`p_spike = 1 - exp(-rate*dt)`), so the 0.5 s call interval removes ~99% of release
opportunities.

**The evidence, measured not inferred:**

| | release opportunities / 14 s traversal | expected release events |
|---|---|---|
| shipped `run_trial` | 2800 | ~350 |
| L·ETA-3 harness | 28 | **~3.3** |

- My probe, while still inheriting the L·ETA-3 pattern, recorded **`max_glu = 0.0000` at
  the target synapse across an entire traversal** at `max_act = 0.9950`.
- After stepping release per physics step (the shipped pattern), same seed, same geometry:
  `max_glu = 1.0000`, and **`peak_r` at traversal 2 rose 0.0571 → 0.1428, a 2.5× increase**
  from the release fix alone.

**Why it matters.** L·ETA-3 attributed its 13× shortfall to both factors of
`r ∝ E_invasion × ca_open` "roughly multiplicatively", with `ca_open = 0.140` vs the rig's
0.38. `ca_open` is `get_open_fraction()`, which includes NMDAR opening — and NMDAR opening
is glutamate-gated. So the `ca_open` half of that attribution was measured with the
glutamate contingency **substantially unsatisfied**. That is the ERR-2 failure class in a
new location: ERR-2's own words are *"the term setting `r` was measured with its glutamate
contingency unsatisfied."*

**What survives regardless:** that `r` fell 13× short in that trial, the zero cross-synapse
edges, and the `E_invasion` trace mechanism (E_invasion is driven by calcium through
`f_CaM`, and VGCC still conducted). **What is in question:** the *split* of the shortfall
between the two factors, and therefore how much of "the constraint is DWELL and
CO-ACTIVATION" rests on a measurement artifact.

**My recommendation:** a correction banner on ETA-3 in the same style as ERR-2 — narrowing
the `ca_open` attribution to a lower bound rather than retracting the row. The headline
("η does not clear in a live trial") is very likely to survive: my own corrected-drive
smoke still shows `r ≈ 0.14` against a threshold of 1.0. I recommend **not** re-running
L·ETA-3 for this; my L·ETA-5 run measures the corrected drive regime at 8 traversals and
will report absolute `r` under it, which answers the question as a side effect.

**Status:** escalated to the MO the same turn. Not blocking my run — my probe is corrected
and the deviation is recorded as AMENDMENT A1.1 in the pre-registration, committed before
the run.
