# PRE-REGISTRATION — L·ETA-6: how much NMDAR calcium did L·ETA-4's "silent" synapses get?

**Registered 2026-07-18 by PO-3, BEFORE measuring.** MO rotation 002. The thresholds below are
fixed here; "negligible" decided after seeing the number is not a finding.

## 1. The question and why the number decides something

Rotation 001 established that L·ETA-4's silent-synapse NMDAR result is **unsupported**: its
premise ("no glutamate") is false — 13 spontaneous release events across its 6 silent synapses
in its own 12 s run — and its metric (NMDAR **open fraction**) is voltage-independent by
construction (`analytical_calcium_system.py:129-130`), so it cannot detect the effect.

**Unsupported ≠ wrong.** Which one it is turns on magnitude:

- **Negligible** NMDAR calcium at the silent synapses ⇒ L·ETA-4's `−0.0019` is approximately
  right anyway; `P_product` selectivity is **weakly supported rather than unevidenced**; PO-5
  survives roughly as scoped.
- **Material** ⇒ the silent synapses had a live NMDAR AND-gate; the selectivity claim is
  **contradicted**, not merely unevidenced; PO-5 needs re-scoping.

Sarah is holding a PO-5 decision on this.

## 2. Conditions — L·ETA-4's own, unchanged

Read-only instrumented reconstruction: `N_SYN = 7` @1 µm, `DRIVEN = 3` at `act = 1.0`, the other
six at `act = 0.0`, `T_S = 12.0 s`, `DT = 0.005`, `SEED = 11`, release seeds `3000+i`,
`PLATEAU_VOLTAGE_V` from `model6_core`. **`plateau_vgcc_leak_probe.py` is not modified and its
verdict is not re-derived.** I measure the INPUT magnitude only.

**Four arms** (`plateau off/on` × `NMDAR intact/blocked`). `nmda_blocked`
(`analytical_calcium_system.py:109,124`) zeroes `g_bind` while leaving VGCC intact — the model's
own APV control — so the intact-minus-blocked difference is the NMDAR-attributable calcium.

## 3. The discriminating quantities (fixed before measuring)

**Primary — current-integral ratio, plateau ON:**

`R = mean over the 6 silent synapses of [ ∫ I_NMDA dt ] / [ ∫ I_NMDA dt at the DRIVEN synapse ]`

where `I_NMDA = sum(current[is_nmda])` per step, exactly as
`analytical_calcium_system.py:136-138` computes it (`base * B(V)` when open).

**Secondary — absolute calcium, plateau ON:**

`ΔCa_NMDA(silent) = [Ca]_silent(NMDAR intact) − [Ca]_silent(NMDAR blocked)` in µM, reported as
both peak and time-mean.

**Reported alongside (context, not scored):** the same quantities with plateau OFF, to show the
`B(V)` effect (`B(−20)/B(−70) = 20.6×`) as a measured rather than computed quantity.

## 4. Thresholds — anchored to the paper L·ETA-4 cites, not invented here

L·ETA-4's row justifies its claim by **Jain 2024**: voltage-clamp to 0 mV **without glutamate
uncaging** gave **7 ± 8 %** potentiation (n=10) versus **56.3 ± 16 %** with. That is the
literature's own statement of how much no-glutamate drive counts as nothing, and its ratio is
**7 / 56.3 = 0.124**.

- **NEGLIGIBLE** iff `R ≤ 0.05` **AND** `ΔCa_NMDA(silent)` peak `< 0.05 µM`.
  (0.05 µM is 5 % of `K_calcium_poly = 1.0 µM`, the `f_CaM` half-activation
  `spine_plasticity_module.py:91`, so the effect on the actin driver is <<1 %.)
- **MATERIAL** iff `R ≥ 0.124` **OR** `ΔCa_NMDA(silent)` peak `≥ 0.10 µM`.
  (0.10 µM is comparable to resting calcium ~0.1 µM — i.e. an NMDAR contribution that roughly
  doubles the silent synapse's baseline is not "nothing".)
- **EQUIVOCAL** otherwise → the honest answer is "cannot determine without re-running L·ETA-4",
  stated with what that would cost.

## 5. Verdict mapping, fixed in advance

| outcome | verdict on L·ETA-4's `−0.0019` | consequence for PO-5 |
|---|---|---|
| NEGLIGIBLE | **survives as approximately correct** | `P_product` weakly supported; PO-5 as scoped |
| MATERIAL | **contradicted** — silent synapses had a live NMDAR gate | PO-5 needs re-scoping (Sarah's call) |
| EQUIVOCAL | undetermined | escalate with the cost of a full re-run |

## 6. Stated limits, in advance

- **The Jain ratio is a POTENTIATION ratio, not a calcium ratio**, and the mapping from calcium
  to potentiation is non-linear (`f_CaM` is Hill-4). Using 0.124 as a calcium threshold is a
  deliberately **conservative** choice — it makes MATERIAL *harder* to reach than a linear
  reading would, so a MATERIAL verdict is not an artifact of the anchor. This is recorded before
  the measurement precisely so the anchor cannot be renegotiated afterwards.
- Blocking NMDAR globally also changes the driven synapse; the difference is taken **per
  synapse**, so silent-synapse attribution is unaffected.
- This measures **input magnitude at the silent synapses**. It does **not** re-derive L·ETA-4's
  verdict, and it does **not** establish what the resulting `P_product` would be — dimer
  formation is downstream and not modelled here.
- Nothing is tuned; no constant is written; `plateau_vgcc_leak_probe.py` is not edited.

---

# AMENDMENT 1 — the peak is NOT sampling-invariant, and my sampling biased toward NEGLIGIBLE

**MO ruling 008, accepted in full. Registered before scoring; no threshold changed.**

## What I got wrong

When I reduced calcium sampling to every 20 steps to fix the compute overrun (Q3), I wrote
*"No pre-registered condition changed — this is observable sampling, not physics."* **That was
wrong.** §4 gates on a **peak**:

> NEGLIGIBLE iff `R ≤ 0.05` **AND** `ΔCa_NMDA(silent)` peak `< 0.05 µM`

The integral ratio `R` is sampling-invariant. **A peak is not.** Sub-sampling a maximum can only
ever **under**-report it, which pushes `ΔCa` toward `< 0.05 µM` — **toward NEGLIGIBLE, the branch
under which L·ETA-4 survives.** I introduced a bias toward rehabilitating the very result I had
undermined in rotation 001, and described it as neutral.

## The fix — exact, and free

`AnalyticalCalciumSystem` **already maintains a true per-step running max** at
`analytical_calcium_system.py:419`:

```
self.peak_concentration = max(self.peak_concentration, np.max(self._local_ca))
```

updated **every step**, independent of how often this probe samples. The scored peak is now read
from `syn.calcium.peak_concentration` at the end of each arm. **Zero added cost** — it does not
restore the 67k field reductions, and it is *more* exact than the original per-step version I
timed out on, because it is maintained inside the integrator rather than reconstructed outside it.

The 20-step-sampled peak is **retained and printed alongside** as an explicit bias check, so the
size of the error I would have made is on the record rather than merely asserted.

## Unchanged

Thresholds, the Jain-anchored `0.124`, the verdict mapping, the four arms, and the conditions are
exactly as registered. This corrects an instrumentation defect, not a criterion.
