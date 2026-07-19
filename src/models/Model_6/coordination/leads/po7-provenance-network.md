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
