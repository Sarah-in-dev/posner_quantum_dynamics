# Queue: model6-mo — decisions awaiting Sarah (MO-owned, decision-ready)

Each entry: **the exact ask · why · the MO's recommendation · the evidence.** Written so clearing
it is seconds, not a re-derivation (`talon-operator-surface`: actionability-as-gate).

---

## MO-1 · 2026-07-18 · **`K_CLASSICAL` — the spread is 10×, not 50×, and it is confined to the gap**

**The ask:** approve changing the dissolution rate inside `analytical_gap` from `0.05` to
`0.005`, to match the live chemistry path and its citation. One line, after PO-4's consolidation.

### The board overstates this, and the MO is correcting its own framing

`board.md` and `MO_MODEL6.md` §3 describe *"a 50× spread across three sites: 0.05 / 0.005 /
0.001."* **Verified from code — reachability, not just presence — that is not the situation:**

| site | value | status | evidence |
|---|---|---|---|
| `dimer_particles.py:127` `k_dissolution` | 0.001 | **DEAD** | only two hits in the tree: the docstring `:93` and the assignment `:127`. **Never read anywhere.** Not a competing value — dead code. |
| `ca_triphosphate_complex.py:160` `k_classical` | **0.005** | **LIVE — the real chemistry path** | read at `:418`: `k_diss = self.k_classical * (1.0 - singlet_excess) * template_enhancement`. Also the **sweepable** one (`sweep_runner.py:89` → dimension `q2_k_classical`, `quantum_dimensions.py:102-103`). |
| `analytical_gap` `K_CLASSICAL` | **0.05** | **LIVE, gap-local literal** | `sweep/run_spatial_discovery.py:80` and `src/models/Model_6/sweep/run_theta_burst_45s.py:69` — a function-body literal, not a parameter. |

**So the live disagreement is a single 10× one, between the grounded chemistry rate and a
function-body literal inside the gap.** The 0.001 site contributes nothing because nothing reads
it. Calling it 50×/three-sites made the decision sound like a physics re-derivation; it is a
one-line reconciliation.

### Which value is right is already settled, not open

`model6-dimer-formation-chemistry:64` — *"`k_classical` (reverse rate): `0.05 → 0.005 s⁻¹` —
cluster lifetime τ≈200 s … (was uncited `0.05`)"*. The retirement already happened; the gap
simply never got the memo. `sweep/phosphate_conservation_probe.py:69` already uses `0.005` and
cites Turhan 2024.

### Why it matters, and it compounds with the other gap defect

**The gap has been dissolving dimers ~10× too fast in every multi-trial run.** Combined with the
stopped plasticity clock (D20: 1 ms per 30 s), `analytical_gap` is wrong in **two independent
directions at once**: chemistry decays far too fast, plasticity does not decay at all. Any
multi-trial result that spans a gap inherits both.

### Recommendation

**APPROVE `0.05 → 0.005`,** applied *after* PO-4's consolidation lands — at which point there is
**one** site to change instead of two, making it a genuine one-line diff with no chance of the
audit-item-16 partial-fix shape recurring.

**Not bundled into PO-4's consolidation.** PO-4 is explicitly forbidden from touching it, so that
the consolidation diff stays purely structural and its acceptance stays attributable. This lands
as a separate MO-owned commit with its own before/after dimer-count measurement.

**If vetoed:** the gap keeps the retired rate and every gap-spanning result must carry that as a
stated limit. That is a defensible position — it preserves comparability with prior runs — but it
must then be *written down*, not left implicit.

### Cost

One line, plus a measurement showing the dimer-count delta across a gap. No physics re-derivation:
both values already exist in the tree and the authority already chose between them.
