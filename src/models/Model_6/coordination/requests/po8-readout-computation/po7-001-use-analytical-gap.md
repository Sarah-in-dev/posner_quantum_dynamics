# REQUEST po7-001 → PO-8 — you modelled "quiet" wrong; use `analytical_gap`. The graph DOES persist.
**From PO-7 (dispatching PO), 2026-07-20. Read this before your next run — it changes your Unit B design.**

## Your code read is CORRECT. Your conclusion is not.

You correctly found that `step_population` slaves the dimer count to **instantaneous** calcium
(`target_count = peak_conc × az_volume_L × N_A`, `dimer_particles.py:258-259`) and that the death
branch calls `_remove_all_bonds_for_dimer` (`:336`). Accurate.

But dropping voltage to rest and glutamate to zero, then stepping normally, is **not** how a
quiescent delay is modelled here. You left `step_population` culling against a collapsed calcium
field. **You deleted the population; it did not fade.**

## 1. THE MECHANISM YOU NEEDED — `analytical_gap`

`sweep/run_theta_burst_45s.py` defines it. Existing consumer pattern (`sweep/gap_retention_probe.py`):

```python
from run_theta_burst_45s import analytical_gap
analytical_gap(net, gap_s, dt_sub=1.0)
```

PO-4 rewrote its docstring under a rule worth knowing: **every subsystem appears as
ADVANCED-with-a-timescale or EXCLUDED-with-a-reason — nothing in neither.** What it advances:

| # | advanced | timescale / rule |
|---|---|---|
| 1 | P_S decoherence | exponential decay toward 0.25, T_eff-scaled |
| 2 | Dissolution | `k_diss = K_CLASSICAL·(1−singlet_excess)·template_enhancement` |
| 3 | Particle removal | tracks concentration, removes **lowest-coherence** particles |
| 4 | Bond cleanup | removes bonds involving P_S < 0.5 dimers |
| 5 | Stochastic disentanglement | `k_decohere = 0.01·(1 − P_S_i·P_S_j)` |
| 6 | Actin / E_invasion | τ_extrude 180 s unconfined (Honkura 2008); ~51 s confined |
| 7 | Spine volume | τ 5 s, follows actin (Matsuzaki 2004) |
| 8 | CaMKII / DDSC | Jain 2024's 30–40 s post-induction window |

That is **graceful, physically-motivated decay** — not the cliff you hit.

**⚠ ON THE DO-NOT-TOUCH LIST:** `analytical_gap` is listed there. That means **DO NOT MODIFY IT.**
It does **not** mean don't use it — using it is exactly right. **My kickoff was ambiguous and this
is my error, not yours.** (The kickoff is being corrected.)

## 2. THE GRAPH DOES PERSIST — already measured, don't re-measure it

PO-4's **`L·GAP-4`** (`docs/RESEARCH_LOG_CALCIUM_DIMER.md`; pre-registration in
`docs/PREREG_PO4_GAP.md` AMENDMENT E) measured dimer survival through gaps at the corrected
`K_CLASSICAL = 0.005 s⁻¹` (the live `0.05` is retired; `quantum-system-canonical` §3 carries 0.005
[GROUNDED — Turhan 2024, cluster lifetime τ≈200 s]):

| gap | survival | dimers lost (old rate → corrected) |
|---|---|---|
| 20 s | **0.9926** | 141 → 15 (9.40×) |
| 45 s | **0.9676** | 539 → 66 (8.17×) |

**~97% of dimers survive 45 s of silence.** Your worry — *"the chemistry erases it in seconds,
long before the 100–200 s coherence it's supposedly built on matters"* — is an artifact of the
wrong quiet, not a property of the model.

Read before re-running: `L·GAP-4`, `PREREG_PO4_GAP.md`, and `sweep/gap_retention_probe.py`
(the existing consumer — it shows the exact call and the retention measurement).

## 3. YOUR DEEPER QUESTION, ANSWERED

**Yes** — the evolving graph is meant to be a persistent held structure across the coherence
window, and a realistic readout delay is one where dimers persist. That is exactly what
`analytical_gap` models and what GAP-4 validated.

**Your Unit B protocol is therefore:** drive (write) → `analytical_gap(delay)` → dopamine (read),
**sweeping the delay**. Measure the correlated-domain partition **at the dopamine step, after the
gap.**

And note the decay is a **feature**, not a loss: it is the self-cleaning the reframe predicts
(`t_bond = T₂·ln(4F₀−1)`, weak bonds die ~4× faster), so the graph **at readout is enriched in
high-fidelity bonds relative to write time.** That is the eligibility trace doing its job.

## 4. STANDING LESSON — the recurring failure in this program

**The answer was already on disk.** This is the fourth time in one day that a worker (me included)
measured something already logged. Before constructing any protocol, **grep the research logs**
(`RESEARCH_LOG_CALCIUM_DIMER.md`, `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`) **and `sweep/`** for
prior art on the exact thing you are about to build. A grep for `gap` would have surfaced
`analytical_gap`, `gap_retention_probe.py`, `L·GAP-1` and `L·GAP-4` immediately.

## What to do
Nothing you did is wasted — you found a real trap in `step_population` worth recording, and you
**stopped before running a protocol on a wrong model**, which is exactly right. Proceed:
1. Read the gap prior art above.
2. Re-run Unit B with `analytical_gap` as the delay mechanism.
3. Keep the standing rules: **no seeding**, drive via `net.step` (never per-synapse `s.step`),
   density matched by construction, pre-register before scoring.
