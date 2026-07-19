# PO-7 — network-shared provenance events + multi-synapse §8 keystone

## Heartbeat log

### 2026-07-19 20:21 UTC — cycle 1
**Grounded and building.** Returned the GROUNDING BRIEF (skills: agent-grounding-protocol,
session-discipline, experiment-design-patterns, model6-entanglement-partition-werner,
quantum-system-canonical §5/:43/:139; log entries L·PO5-10..13; PREREG U16).

**Verified what I inherited** — read `_step_network_provenance` in full. Confirmed
`global_id = (syn_idx, dimer.id)` is a TUPLE (`multi_synapse_network.py:183`), so the
`a[0] != b[0]` cross-synapse diagnostic is genuinely comparing synapse indices.

**Found before building (the reason the layer was never going to validate as-committed):**
grid = 100×100 × 4nm = 400nm span, reach = 500nm, so min cross-synapse dimer↔event distance
for adjacent synapses is `spacing_nm − 400`. At the DEFAULT `spacing_um=2.0` that is 1600nm
vs a 500nm reach ⇒ **cross-synapse edges are structurally impossible**. A run at defaults
would have reported a physical-looking null for a purely geometric reason — the same
mis-registration class that nulled Units 8/9/13/16b.

**Two defects fixed in the inherited layer** (flag-guarded, off-path untouched):
1. **Frozen fidelity** — `_prov_bonds` stored F at claim time and never refreshed, while
   `_find_all_clusters` tests it against the live Werner bound. Now recomputed each step.
2. **Dropped coherence death** — prune tested `is_entangled` only, contradicting the method's
   own docstring; same dropped-channel shape as the U16 write-once bug (07fd02a). Now prunes
   on `P_S <= 0.5`.

**Bit-identity gate: PASS** — `1034 / 369740 / 0.991922159684` with the edit in place.

**Running:** Unit 1 cross-edge validation, spacing sweep 0.2–2.0 µm × 5 seeds, with the
geometric prediction registered in the probe header BEFORE the run.

**Open questions** → `notes.md` (recommendations recorded; not blocking).

### 2026-07-19 21:05 UTC — cycle 2

**Unit 1 (geometry locator, 9/12 rows in) — the layer DOES form η-free cross-synapse edges,
but only in a narrow, very sparse band:**

| spacing | cross edges (per seed) | overlap |
|---|---|---|
| 0.2 µm | 2, 3 | 0.036, 0.056 |
| 0.4 µm | 1, 0 | 0.014, 0.000 |
| 0.6 µm | 0, 0 | 0.000 |
| 0.8 µm | 0, 0 | 0.000 |
| 1.2 / 2.0 µm | pending (falsification rows) | — |

Monotone decline with spacing, hard zero by 0.6 µm — consistent with the registered geometric
prediction (reach 500 nm vs 400 nm grid span). **Pre-registered spacing rule selects 0.2 µm**
(mean overlap 0.046 vs midpoint 0.028 of the observed 0–0.056 range), independently of yield.

**Unit 1b (power check, unscored) — first cross-synapse COMPONENT demonstrated, then a CEILING:**
- `t=0.25 s`: first cross edges, `n_multi=1` — **a connected component spanning two synapses.**
  This is the first data-level proof the layer does what it was built to do, η-free.
- `t=0.25 → 0.60 s`: **seven consecutive flat steps.** `cum_cross` pinned at 2,
  `syn_pairs_ever` at 1/15, while `prov_total` grew 153→267. The cross channel stalls while the
  intra channel keeps running.
- **Causal read (candidate, pending full trace):** `dimers` saturates (~6590, birth ≈ death) and
  provenance is **born-with** — only newly-seen dimers claim events (`:504`). Once the population
  stops growing, new claims decay, and cross-synapse claims (a ~1% tail) stop almost entirely.
  The ceiling follows from the born-with rule × population saturation, not from geometry alone.

**Consequence for the scored test (flagged, notes.md Q0):** with 2 cross edges across 6 synapses,
Newman Q on the synapse graph is noise-dominated. A "decomposition null" verdict from that would
**overclaim a negative.** Distinction being held: *"the mechanism cannot support the test"* ≠
*"the mechanism fails the test."* Only the former would be supported; §8 would remain OPEN.

**Bit-identity re-verified this cycle: PASS.** Commits `c3ba6aa`, `b03b90a` — explicit paths,
`git show --stat` clean both times.

### 2026-07-19 21:45 UTC — cycle 3 · COMPLETE

**VERDICT (pre-registered): NEGATIVE — DECOMPOSITION NULL.** `mean n_multi` = 0.60 (contiguous) /
0.50 (interleaved), both < 1 ⇒ the partition splits cleanly per-synapse. **Input-LOCATED, not
input-COMPUTING.** Research log entry `L·PO7-1` + DECISION RECORD row written.

**But the build objective SUCCEEDED:** η-free cross-synapse provenance edges and components are
real and demonstrated (`eta_cross = 0` in all 20 runs — every cross edge is Fisher-inherited).

**Bit-identity re-gated AFTER scoring: PASS** (`1034 / 369740 / 0.991922159684`).

**Three design defects found in my own prereg** (dimer-level Q degenerate ≈1.0 — caught before
scoring; "density matched by construction" false — `Q_act = +0.0000` in all 20 runs, so d=1.61/1.33
is an artifact and a Q-PASS would have been a THIRD false positive; ARM 2 conflates layout with
distance). All recorded in `notes.md` Q6 and `L·PO7-1` §5, none quietly absorbed.

**§8 remains OPEN, not answered negative.** The fix for the next worker is in `L·PO7-1` §7:
HIGH-vs-LOW drive on all six synapses (density matched) + constant co-active spacing + a
cross-edge yield increase. Confound and power must be fixed together.

### 2026-07-19 — CORRECTION (Sarah): the cross-synapse premise is unphysical

Phosphate provenance is **local**. Cross-synapse entanglement is mediated by the condensate
backbone (`model6-entanglement-partition-werner` §2: `k_cross = 0.5·√(η_i·η_j)·w_spatial·P_product`,
`w_spatial = exp(−d/5 µm)` = the CONDENSATE coupling length). PO-7 read §1 and never read §2.

**Withdrawn:** the network event pool as a physical claim; the "2 µm landmine" (it is CORRECT
physics); the Unit-2 verdict; the Unit-7 claim-radius derivation; Unit 6 (killed); the R4 packet.

**Survives, all local:** monogamy violated 179× (99.44% of edges inadmissible); provenance
monogamy-CLEAN (0/434 over bound); the which-spin slot tag; coincidence window ≤50 ms ⇒
`provenance_net_age_s` non-operative; two real inherited defects fixed; the Unit-5 sheaf is a
direct sum of 3 graph Laplacians.

**The finding that replaces the objective:** there is no η-free cross-synapse route, so **η being
dead is a HARD BLOCKER, not something to engineer around.** Full record: `L·PO7-2`.
