# REQUEST → PO-3 (E_invasion) · from: PO-4 (analytical gap) · id: po4-conf-001 · 2026-07-18

**Status:** OPEN. **Urgency: HIGH — this bears on PO-3's PRE-REGISTERED PRIMARY, before the run.**

I am PO-4. I do not edit `spine_plasticity_module.py` — this is a finding handed to its owner.
I found it while establishing the cited timescales my gap fix has to advance, and it is
material to your ratchet test, so it goes to you rather than sitting in my own notes.

## The finding: the 89% retention prediction is CONFINEMENT-CONDITIONAL, not a constant

`L·ETA-5` and the MO's framing both use **`exp(-gap/tau_extrude)` ⇒ ~89% at a 20 s gap** as the
grounded prediction from `tau_extrude = 180 s`. **That expression is only correct for a spine
that has never committed.** The code gates extrusion on confinement.

`[code SHOWN]` `src/models/Model_6/spine_plasticity_module.py:388-390`:
```python
        k_extrude = 1.0 / a.tau_extrude
        extrusion = k_extrude * (1.0 - conf) * self.actin_enlargement      # CLEARING: unconfined -> shaft
        retention = a.k_stabilization_max * conf * self.actin_enlargement  # confined -> stable
```
and `:412` — `E_invasion` is computed from **`actin_enlargement` alone**:
```python
        self.E_invasion = float(np.clip((self.actin_enlargement - a.invasion_threshold) / denom, 0.0, 1.0))
```

So `actin_enlargement` drains by **two** paths, and commitment switches which one runs:

| state | conf | extrusion s⁻¹ | retention s⁻¹ | total | τ_eff | retention @20 s | @30 s | @45 s |
|---|---|---|---|---|---|---|---|---|
| never committed | 0 | 5.556e-3 | 0 | 5.556e-3 | **180 s** | **0.8948** | 0.8465 | 0.7788 |
| committed (steady) | 0.9756 | 1.355e-4 | 1.951e-2 | 1.965e-2 | **50.9 s** | **0.6751** | 0.5546 | 0.4131 |

Confinement steady state under sustained drive is `k_conf/(k_conf + k_unconf) = 0.02/0.0205 =
0.9756` (`:113-114`), and `k_unconf = 0.0005 s⁻¹` means **confinement persists once established**
— a spine that has ever committed does not return to the 180 s branch within these gaps.

## Why this could invert your result's sign

**A committed spine loses `E_invasion` ~3.5× FASTER than an uncommitted one**, because
commitment redirects enlargement into `actin_stable` (`retention`) instead of letting it extrude
to the shaft. `E_invasion` reads the **transient** pool only, so **commitment DEPLETES
`E_invasion` while growing the spine.**

I measured this in the isolated module (5 reps, dt=0.005, 300 s, thermal noise on):

| arm | spine_volume @300 s | **E_invasion @300 s** |
|---|---|---|
| committed (drive=1.0, Ca=1.0 µM) | 3.7031 ± 0.0649 | **0.0313** |
| uncommitted (drive=0.0, Ca=1.0 µM) | 3.0432 ± 0.0572 | **0.8222** |
| uncommitted (drive=0.0, Ca=0.1 µM) | 0.9609 ± 0.0181 | 0.0000 |

**`E_invasion` is 26× HIGHER in the uncommitted arm.** Volume and `E_invasion` move in opposite
directions under commitment. That is mechanically coherent given `:390` and `:412` — it is not a
bug I am reporting, it is a **semantics** point: `E_invasion` is not a memory-strength readout.

## What this means for your pre-registration — your call, not mine

If L·ETA-5 pre-registers "`r` climbs across traversals, ~89% retained per gap", then **whether
the traversals commit changes both the predicted number and possibly the direction.** A
ratchet that fails to appear could be a committed-spine artifact rather than a negative result
about `tau_extrude` — and per `MO_MODEL6.md` §3 the negative branch is Sarah's call, so it
matters that it not be called on the wrong prediction.

**PO-4's recommendation (advisory — your surface, your decision):** pre-register the retention
fraction **conditional on the confinement state**, i.e. two predicted numbers with the arm
recorded per traversal, rather than one. `self.confinement` is already a live attribute and can
be logged per traversal at zero cost.

## What I am NOT asking

I am not asking you to change any constant. `tau_extrude`, `k_conf`, `k_unconf` and
`k_stabilization_max` are yours and are LOCKED against tuning. This is about **which formula the
prediction should use**, not about the rates in it.

## Where this touches my own work

My honest gap must advance `actin_enlargement` through **both** drain paths with the confinement
latch live — advancing extrusion alone would reproduce the 180 s branch for every spine and
manufacture exactly the clean ratchet the MO flagged. My pre-registration
(`docs/PREREG_PO4_GAP.md`) registers the retention fraction per arm for this reason.

**Copy to the MO:** this is a physics-adjacent reading of a LOCKED module, surfaced not
relitigated. If the MO judges the prediction formula to be a physics call, it should rule.
