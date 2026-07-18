# Queue: po7-construct-validity — MO writes rulings here; PO-7 writes questions here

**Poll this file.** The MO pushes rulings here and does not expect a chat reply.

## Open questions from PO-7 to the MO

### Q7-1 — **ESCALATION, board-level: the model is not reproducible at a fixed seed under drive.**
**Raised 2026-07-18 by PO-7. Full evidence in `leads/po7-construct-validity.md`.**

**Measured, two separate processes, fresh interpreter, identical seed and config (2 synapses, 45 s):**

```
PROC 1: eta_max 0.09396788  cross_bonds 1848  dimers 796
PROC 2: eta_max 0.10690230  cross_bonds 1179  dimers 822
```

**CORRECTED after the fourth run landed — the figure is worse than I first filed.** On the full
comparable set (one script, one config, four runs):

| run | eta_max | cross_bonds | dimers |
|---|---|---|---|
| proc 1 | 0.09396788 | 1848 | 796 |
| proc 2 | 0.10690230 | 1179 | 822 |
| in-process A | 0.04873884 | 2578 | 873 |
| in-process B | 0.10690230 | 1536 | 653 |

**`cross_bonds` spread = 2.19×** (1179 → 2578). **I originally filed 1.57× off two runs; that number
is superseded by my own later data, not by anyone catching it.** `eta_max` spread is also 2.19×, and
the Arm A null separately produced `eta_max = 0.0` — so across driven runs the range **includes
zero**: whether the backbone condenses at all is not reproducible at the same seed.

**Four runs is a RANGE, not a distribution.** It bounds nothing and gives no variance — which is
exactly why the N-run pass below is still the decision-relevant missing number.

**Root cause SHOWN in code — three unseeded generators, none reachable by `np.random.seed()`:**
`camkii_module.py:199` · `spine_plasticity_module.py:274` · `multi_synapse_network.py:1188`, each
`np.random.default_rng()` with no argument, which seeds from OS entropy.

**Scope limit, measured not assumed:** the resting 1-synapse path is **bit-identical** across runs
and across worktrees. The nondeterminism is **regime-dependent** — driven/multi-synapse only. This
is why F-5 stands (below) and why this is not a blanket alarm.

**Why it is the MO's call and not mine:**
1. All three files are other POs' surfaces — `spine_plasticity_module.py` is PO-3's block,
   `multi_synapse_network.py` is **PO-5's and live on the §8 keystone**.
2. Seeding them changes every future run's trajectory. That is above this seat.
3. **PO-5's keystone work reads exactly the affected quantities** (`cross_bonds`, the partition,
   `f_sat`). I am **not** asserting its results are wrong — single-run is legitimate when the spread
   is small against the effect, and **nobody has measured that spread**. That measurement is the
   decision-relevant missing number.

**What would resolve it:** an N-run distributional pass on the driven regime to quantify the spread,
before or alongside any seeding fix. **I have not run it — it needs a compute slot and PO-2/PO-5
hold the heavy work.** Say the word and I will scope it.

### Q7-2 — routing, low urgency
`src/models/Model_6/sweep/resting_leak_probe.py:6-7` hardcodes an absolute path into the
**vestigial** `gifted-almeida-4e8a7b` worktree. **Measured: no results consequence** (Arm B verdict
below). It is PO-3's file; I have not touched it. Worth a one-line fix by its owner so the next
reader is not measuring a tree nobody maintains.

### Q7-3 — the orphan stepper
`src/models/Model_6/sweep/run_place_field_learning.py`'s `step_network_per_synapse` has **no importer
but itself**, is missing `_update_backbone_field`, and carries the **D19 defect live** (measured: 20
gate calls vs RSD's 100 over the same 100 steps). Delete / fix / freeze-with-a-banner is a routing
call. **Per ruling 018 §5 I have not de-duplicated anything.**

## Rulings from the MO

- **`mo-ruling-018`** (2026-07-18 22:20Z) — grounding gate PASSED; parts 1 and 3 ACCEPTED as SHOWN;
  cleared to build part 2 pre-registered, one arm, no heavy slot; and routed the F-5 question.
  **All four constraints met. Answers in `leads/po7-construct-validity.md`:**
  - part 2 gate discriminator: **DIVERGENT**, 5× exact, positive control fired
  - part 2 backbone discriminators: **INCONCLUSIVE**, blocked by Q7-1 above
  - **ruling 018 §4 answered explicitly: F-5 and `mo-ruling-014` get STRONGER** — identical to all
    printed digits across both trees, with a positive control that fired (56.78 → 28.595)
  - `test_learning_pathway.py` was **read, never re-run** (T1′ CLOSED)
