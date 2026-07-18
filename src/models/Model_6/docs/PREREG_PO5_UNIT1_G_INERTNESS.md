# PRE-REGISTRATION — PO-5 UNIT 1 · is the 1/r³ factor `g` inert in practice?

**Registered:** 2026-07-18, BEFORE the scored run. **PO:** po5-selectivity.
**Probe:** `src/models/Model_6/sweep/po5_unit1_g_inertness.py`.
**Charter:** `quantum-system-canonical` §8 Keystone #1 via `requests/po5-selectivity/mo-rescope-001.md`.

## 1. The question, and why it is asked first

`dimer_particles.py:453-454` folds a pair-resolved distance factor into Pathway 2's rate:

```python
g = (self.coupling_length / np.maximum(r_ij, self.coupling_length)) ** 3
em_rate = k_entangle_em_base * (collective_field_kT / reference_kT) * coh * g
```

`self.coupling_length = 5.0` nm (`:129`). `np.maximum` **clamps**: every pair closer than 5 nm gets
`g = 1.0` exactly. So the 1/r³ can be present in code and still carry no pair information, by either
of two routes — saturation (`g ≈ 1` everywhere) or vanishing (`g ≈ 0` everywhere). **Which of those
holds, or neither, changes the meaning of every later selectivity result**, so it is measured first.

`g` is **geometry, not input** and does not satisfy §8 on its own. This unit does not test the
keystone; it establishes what the keystone's later test is operating on.

## 2. What is measured

At each sample time, over the **intra-synapse** dimer set of a single synapse:

- `r_ij` for all `C(n,2)` pairs — the same `pos[iu] - pos[ju]` reduction the model itself uses
  (`dimer_particles.py:451-452`), read, not re-derived.
- `g_ij` computed with the model's own expression and the model's own `coupling_length`, read off
  the live object rather than hard-coded.

**Reported quantities (all pre-registered before the run):**

| symbol | definition |
|---|---|
| `f_sat` | fraction of pairs with `r_ij <= coupling_length` (i.e. `g` clamped to exactly 1.0) |
| `g_p10`, `g_p50`, `g_p90` | percentiles of the `g` distribution over all pairs |
| `D` | dynamic range `g_p90 / g_p10` — how much `g` actually varies across the bulk of the pair set |
| `sat_bonds` | realised bond-graph saturation, `n_bonds / C(n,2)` — reported alongside because a
near-complete graph is pair-flat regardless of what `g` does |

## 3. The verdict function — registered with its thresholds, and it can return FALSIFIED-equivalents

`classify_g(f_sat, g_p10, g_p90)` returns exactly one of four labels. Thresholds fixed here, before
the run, and not to be moved afterwards:

1. **`INERT_BY_SATURATION`** — `f_sat >= 0.90`. Nine in ten pairs sit inside the clamp, so `g` is
   effectively the constant 1.0 and Pathway 2 is flat-rate by a different route. *This is the
   outcome the kickoff and `board.md:919-922` anticipated.*
2. **`INERT_BY_VANISHING`** — `g_p90 < 1e-3`. `g` is so small across the pair set that
   `em_rate * dt` is numerically negligible and Pathway 2 forms essentially nothing. *This is the
   outcome PO-5's grounding brief anticipated, from the 400 nm birth domain.*
3. **`INERT_BY_FLATNESS`** — `D < 2.0`. `g` is neither pinned at 1 nor at 0, but varies by less than
   a factor of two across p10–p90, so it cannot meaningfully rank pairs.
4. **`LIVE`** — none of the above. `g` varies materially across pairs; the 1/r³ is doing work.
   Reported with `D` so the magnitude is stated, not just the label.

**Precedence is as listed** (saturation checked first, then vanishing, then flatness), so the label
is unambiguous when more than one condition could hold.

## 4. Demonstrating the verdict can fail before it is allowed to pass

Per `MO_MODEL6.md` §2.3 and the `683b82f` scar — *"a verdict that cannot distinguish its outcomes is
not a result"* — the probe runs `demonstrate_verdict()` **first**, on synthetic pair sets with known
answers, and prints the result **before** any model is constructed:

| synthetic input | required label |
|---|---|
| all separations 1 nm (inside the clamp) | `INERT_BY_SATURATION` |
| all separations 5000 nm | `INERT_BY_VANISHING` |
| all separations 20.0 nm exactly (no spread) | `INERT_BY_FLATNESS` |
| separations log-spaced 5 → 100 nm | `LIVE` |

**If any of the four does not produce its required label, the probe aborts and reports nothing
else.** The classifier must be shown discriminating all four of its outcomes before its verdict on
real data is admissible.

## 5. Null / control

This unit measures a **static geometric distribution**, not a driven effect, so an activation null
is not the applicable control and is deliberately not built. (Note also that an activation-floor
null is forbidden to this PO outright — `BASELINE_RATE_HZ = 0.5`, `sweep/presynaptic_release.py:65`.)

The control that *is* applicable, and is registered: **the classifier's own four-way demonstration
in §4**, plus a **positive control on the geometry itself** — the probe asserts `n_pairs > 0` and
that the measured `r_ij` set is non-degenerate (`g_p90 > g_p10`, i.e. the positions are not all
coincident). A degenerate position set would make `D = 1.0` and print `INERT_BY_FLATNESS` for a
reason that has nothing to do with physics; that failure mode is checked, not assumed away.

## 6. Stated limits, registered in advance

- **Single synapse, one drive condition, one seed** unless stated otherwise in the result. `f_sat` is
  a property of *these* driving conditions; a different calcium/template regime places dimers
  differently. The verdict is reported as **`<label>`-under-stated-conditions**, adopting PO-1's
  formulation (`board.md:804-806`).
- **This unit says nothing about input-selectivity.** `g` is geometry. A `LIVE` verdict here does
  **not** advance the keystone; it only means the later pair-level test is not operating on a
  constant.
- `dt-1` in the topology log records that **dimer COUNT is not converged in the drive transient**
  (~+38% at dt=1e-3 vs 1e-4). `f_sat` is a ratio over pairs and is expected to be far less
  count-sensitive than a count itself, but that expectation is **not** inherited from dt-1 — the
  probe samples at multiple times and reports whether `f_sat` moves.
