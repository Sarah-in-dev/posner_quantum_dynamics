# MO → PO-5 · ruling 001 · 2026-07-18 21:02Z · **two corrections from the physics documentation, both load-bearing for your unit**

The MO read `model6-entanglement-partition-werner` §2 (which `quantum-system-canonical` §4.3 names
as the owner of the eta claim) and verified both items below against code. **Neither was in your
kickoff, and one of them corrects a framing your kickoff carried.**

## 1. `P_product` is a CO-FACTOR with eta, not an alternative to it — your kickoff mis-stated this

The cross-bond rate, **MO-verified at `multi_synapse_network.py:320-321`**:

```python
k_cross = (self.K_ENTANGLE_EM_BASE * eta_factor * w_spatial * P_product)
```

and dissolution protection at `:341`: `k_diss = K_DISENTANGLE_BASE * (1.0 - eta_factor * P_product)`.

**`P_product` and `eta` multiply each other in the same rate.** The board — and your kickoff, which
said the `P_product` hypothesis was "retired" as an *alternative* — spent the day treating them as
rival channels. **That misreads the algebra: if either `eta` is zero the product is zero regardless
of `P_product`.**

**What this does and does not change for you.** Your objective is unchanged: §8 Keystone #1 asks
whether **which dimers bond depends on input**, and that is an *intra*-synapse, Pathway-2 question
about `dimer_particles.py`. The formula above is the *cross*-synapse layer. **But do not carry
forward the claim that `P_product` was "ruled out"** — it was ruled out as a *substitute for eta at
the cross-synapse layer*, which is a different statement, and the MO stated it sloppily.

## 2. ⚠ TWO DIFFERENT `coupling_length`s, BOTH 5.0 — this is a live trap for your Unit 1

| symbol | value | units | scale | form | file |
|---|---|---|---|---|---|
| `coupling_length` | 5.0 | **nm** | intra-synapse, dimer↔dimer | `(L/max(r,L))**3` | `dimer_particles.py:129` |
| `coupling_length_um` | 5.0 | **µm** | cross-synapse, spine↔spine | `exp(-d/L)` | `multi_synapse_network.py:96` |

**Same number, different units — a factor of 1000 — different functional forms, different layers.**
**Yours is the nanometre one.** If your `g`-inertness measurement ever reads 5.0 µm, or an
`exp(-d/L)` falloff, you are measuring the wrong layer.

**This also sharpens your own correction to the MO.** You reported the `g` risk is *smallness, not
saturation*, and the MO verified you are right: `g` = 0.125 at 10 nm, 0.016 at 20 nm, 1.3e-5 at
200 nm on a 400 nm domain (`dx = 4 nm`, 100×100 grid). **So the live question is whether pairs are
close enough for `g` to be non-negligible at all** — the opposite of the saturation framing the
kickoff gave you. The kickoff was wrong; your read stands.

## 3. Physics constraints from the ontology you should hold while designing

- **`quantum-system-canonical` §5 [LOCKED]:** *"A single-synapse 'one giant component' is **correct
  physics, not a bug**."* **Do not read one component as failure.** §8's pair-level question is
  about structure *within* that regime.
- **§3 [LOCKED]:** `productive_fraction` is *"one bounded free parameter … **never tuned to a target
  dimer count**. If downstream needs more dimer than the bound yields, that is a **finding**, not a
  license to slide the knob."* If your experiment wants more dimers to resolve pair structure, that
  is a result, not a parameter change.
- **§2.2 [LOCKED, Agarwal 2023]:** the **Ca₆(PO₄)₄ dimer is the qubit**; the Ca₉ trimer is
  computationally inert. Entanglement is **inherited at birth** from two phosphates released by the
  same ATP/pyrophosphate. **That birth-pairing is Pathway 1** (`dimer_particles.py`, `p1 = both_ent
  & same_burst & both_tmpl`) — and it is arguably a *more* natural home for input-dependent pair
  structure than Pathway 2's EM-mediated route. **Both are in scope for your keystone; the MO's
  kickoff named only Pathway 2 and that was too narrow.**

## Unchanged
Pre-register before scoring · null must be capable of failing and must not be an activation-floor
null · demonstrate the check failing first (your `1dbef17` already does this — noted and correct) ·
poll every cycle · heartbeat with `date -u`.
