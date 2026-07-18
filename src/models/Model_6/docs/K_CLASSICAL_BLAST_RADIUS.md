# Blast radius of the retired `K_CLASSICAL = 0.05`

**PO-4 · rotation 003 (MO gen-2, `mo-ruling-011.md` §2) · 2026-07-18.**
Read-and-enumerate. **No re-runs bought.** Nothing in the NEEDS column was executed.

---

## 0. The finding that narrows everything: there were TWO dissolution paths, and only one was wrong

This is the fact the whole enumeration turns on, and it was not known when the rotation was written.

| path | formula | rate used |
|---|---|---|
| **within-trial** `ca_triphosphate_complex.py:418` | `k_diss = k_classical · (1 − singlet_excess) · template_enhancement` | **`0.005`** (`:160`) — the GROUNDED rate, already correct |
| **the gap** `run_theta_burst_45s.py:225` (inline, pre-fix) | `k_diss = K_CLASSICAL · (1 − singlet_excess)` | **`0.05`** — the retired rate |

**The gap never called `update_dimerization`** (verified: one reference in the file, not in the
dissolution stage). So the retired rate was **confined to dissolution during a silent gap**.

**Consequence for this enumeration: any number measured WITHIN a trial is unaffected.** Only
quantities that depend on what survives a gap can have inherited it. That is a far smaller set
than "every multi-trial dissolution number", which was my own sentence and was **too broad** — the
correction is recorded against my own framing.

---

## 1. What routes through the gap at all

`analytical_gap` has three production consumers (post-consolidation, one definition):

| driver | gap usage |
|---|---|
| `sweep/run_spatial_discovery.py:312` | 30 s inter-trial |
| `src/models/Model_6/sweep/run_place_field_learning.py:343` | 20 s inter-traversal |
| `src/models/Model_6/sweep/run_theta_burst_45s.py` | 5 s validation, inter-traversal, dopamine delay, silence |
| `sweep/loop_audit_2026_07_18/probe_latch2.py:112` | audit probe, via `RSD.analytical_gap` |

**Anything not driven by these did not inherit the retired rate.**

---

## 2. THE TABLE

**YES** = conclusion survives the corrected rate · **NO** = does not · **NEEDS RE-MEASUREMENT** =
cannot be determined without running it, and per §2's constraint I did not estimate.

### 2a. Artifacts that DID route through the gap

| artifact | the number it reports | conclusion survives? |
|---|---|---|
| **D17** (calcium log) — full integration validation, `run_spatial_discovery`, 5 trials × 30 s gaps | end-of-trial dimer totals **9/1/31/23/22**, peak transient **318**, localization 1/1/5/4/3 of 20, "**BOUNDED, no runaway**" | **NEEDS RE-MEASUREMENT** — see §3 |
| **D19** (calcium log) — `probe_latch2.py`, instrumented 3-trial run | `pqm_calls` **1 / 0 / 0**; `_measurement_gate_opened` never cleared; uncommitted synapse grew **1.065 → 1.430**; drive 0→1 *lowers* enlargement **1.447 → 1.099** | **YES** — every conclusion is latch-structural or actin/calcium. None reads dimer survival. The one-shot latch prevents re-measurement *regardless* of how many dimers survive a gap. |
| **D20** (calcium log) — durable-state audit | volume ceiling **3.80**; decay τ ≈ 1000–2000 s; the **1 ms/gap** clock; AMPAR 1800 s onset dead; template feedback fires ~8 s | **YES** — all actin / volume / clock / duty-cycle. No conclusion reads the dissolution rate. (Its *framing* was already corrected by GAP-1, on a different axis.) |
| **D21** (calcium log) — construct-validity gaps | `quantum_field_kT` inert (bit-identical volume, kT ∈ {0,1,5,20,100}); `molecular_memory` vs calcium; AMPAR never changes; primitive 4 falsified; **zero cross-synapse bonds** | **YES** — the kT sweep is a direct isolated-module sweep with no gap; the bond result is `coupling_weights` + `η = 0`, not dissolution. |
| **GAP-1** (mine) — clock delta + retention | clock **+20.0010 vs +20.0010 s**; 8/8 out-of-sample retention points | **YES** — `R` is an `E_invasion` (actin) ratio, not a dimer count. Independent of `K`. |
| **GAP-2** (mine) — the board's headline separation | **ΔV = +0.7764** | **YES — verified by re-run, not argued**: +0.7764 at `0.05` vs **+0.7727** at `0.005`, 0.0037 apart, inside thermal noise. |
| **GAP-3 / GAP-4** (mine) | the before/after delta itself | **N/A** — these *are* the measurement of the change. |

### 2b. Artifacts that did NOT route through the gap — checked, not assumed

| artifact | why it is clean |
|---|---|
| **T1′-1 … T1′-5** — far-pairs-first, **4/4 seeds, p ≈ 3.0×10⁻⁶** | 90 s silence **stepped at `dt = 1e-3`**, full physics — `coherence_fragmentation_probe.py` is **not** in the `analytical_gap` caller list. Its dissolution ran the already-correct `0.005`. **The program's most load-bearing topology result does not inherit the defect.** |
| **ETA-5** — the ratchet test (VOID, null-arm result) | its own row states *"real physics through every gap (`analytical_gap` deliberately NOT called)"* — PO-3 sidestepping the gap on the MO's ruling. That decision **paid off exactly here.** |
| **ETA-1 / ERR-2** | `eta_probe.py`, voltage-only, no gap call. |
| **D18** — criticality / all-or-none bistability, 0-of-480 forbidden gap | `criticality_variability_probe.py` contains **zero** `analytical_gap` references. |
| **D8, D10, D13–D16** — chemistry/SOC integration | in-file integration tests against `update_dimerization`, which already used `0.005`. |
| **PO2-1 … PO2-5** — phosphate conservation | `sweep/phosphate_conservation_probe.py:69` declares its **own** `K_CLASSICAL = 0.005`. Never touched the gap's constant. |
| **B2-1 … B2-4, DISC-1** — pump / per-synapse retirement | static two-site pump comparisons; `pump_mode_agreement_probe.py` has zero gap references (gen-1 verified this independently at ruling 003). |

**No standing artifact is scored NO.** Nothing in the record is *overturned* by the correction.

---

## 3. The single NEEDS RE-MEASUREMENT, and precisely why I stopped

**D17's boundedness claim.** The row concludes dimer formation is *"EMERGENT, BOUNDED, rise-and-fall
… no runaway → **resolves the parked unbounded-accumulation problem**."*

- Trial 1 is **unaffected** — nothing precedes it.
- **Trials 2–5 begin from a post-gap state.** With ~9× less dimer loss per 30 s gap, each later
  trial starts from a larger residual population.
- The within-trial bound is **formation-side** (phosphate limitation; D8/D14: `S` pins at 1.0,
  `phosphate_structural` stabilises), which argues the bound holds. **That is an argument, not a
  measurement**, and §2's constraint is explicit: *"if you cannot determine whether a conclusion
  survives without re-running it, say NEEDS RE-MEASUREMENT and stop. Do not estimate."*
  Gen-1's defect #16 was reasoning a conclusion from a correctly-verified premise instead of
  measuring it. **So: NEEDS RE-MEASUREMENT.**

**What buying it would cost, for the MO's decision:** `run_spatial_discovery`, 5 trials × 20
synapses. D17's own note records the earlier 40-feature smoke test at **5.4 hours**; the substrate
audit records a single probe at **130+ min CPU**. **This is a heavy-slot job, and I hold no slot.**
**I am not requesting one** — the boundedness claim is not blocking any live PO, and D17's
cross-trial reading was *already retracted* by D19 on independent grounds, which lowers what a
re-run would buy.

---

## 4. Second finding, surfaced by this enumeration and NOT in its scope

**The two dissolution paths are still not equivalent after the `K` fix.** The gap omits
`template_enhancement`, which the within-trial path multiplies in:

```
within-trial:  k_diss = k_classical · (1 − se) · template_enhancement
the gap:       k_diss = K_CLASSICAL · (1 − se)
```

**MEASURED — and my first measurement of this was MISLEADING. Corrected below.**

I initially reported: *"`template_enhancement` is `1.0` everywhere except 3 template voxels where
it is `50.0` — 0.03% of the grid, mean 1.015 … the omission is real but spatially confined."*
**That framing is wrong, and wrong in the direction that makes the defect look harmless.**

`0.03% of the grid` is the wrong denominator. **Dimers are not uniformly distributed — they are
BORN at template sites**, because formation is itself template-catalysed
(`k_eff = k_base · template_enhancement`, `ca_triphosphate_complex.py:346`). Measured on a driven
2-synapse network (30 steps):

| measure | value |
|---|---|
| grid-mean `template_enhancement` | **1.015** ← the misleading number |
| **concentration-weighted `template_enhancement`** | **32.5 – 34.4** ← the physically relevant one |
| fraction of dimer *concentration* on templated voxels | **64 – 68%** |
| fraction of dimer *particles* flagged `template_bound` | **97.4%** |

`dimer_concentration` is a `(100,100)` field and `template_enhancement` multiplies it elementwise,
so the relevant factor is the concentration-weighted mean: **≈ 33×, not 1.015×.**

### The consequence, and it partially reframes rotation 002

Effective dissolution coefficients at the same state:

| path | effective `k_diss` coefficient | vs within-trial |
|---|---|---|
| within-trial | `0.005 × te ≈ 0.165` | — |
| gap **before** my fix | `0.05` | **3.3× too slow** |
| gap **after** my fix | `0.005` | **33× too slow** |

**My `K` correction widened the gap-vs-within-trial mismatch from ~3.3× to ~33×.**

The old uncited `0.05` was *accidentally closer* to the within-trial effective rate — for the
wrong reason: a wrong bare constant partially compensating for a missing catalytic term. Removing
the compensation without supplying the term leaves the two paths further apart than before.

**This does not mean the `K` change was wrong.** `0.005` is the grounded **bare** rate
(`quantum-system-canonical` §3, Turhan 2024) and the gap's comment now says so correctly. It means
the gap's *formula* is incomplete, and fixing the constant **exposed** that rather than causing it.
**NOT DAMPED, NOT REVERTED** — reported, per rotation 002's standing instruction that a moved
result is an escalation.

**Escalated as Q4-10 (physics call, not PO-4's).** The chemistry skill's detailed-balance argument
— a catalyst must act on **both** directions, the reasoning that retired the one-sided template
application as a *"thermodynamic inconsistency"* — applies here in the same shape.

**The caveat I cannot resolve and am not estimating:** formation is OFF during silence, so
including the template term would mean ~33× faster gap dissolution with no compensating formation.
Whether that is correct physics (the catalytic surface is still present in silence) or whether the
gap needs a different treatment is exactly the call being routed.

---

## 5. Correction to my own framing, recorded

My sentence *"every multi-trial dissolution number produced before today inherits `0.05`"* became
this rotation's dispatch. **It was too broad.** The retired rate was confined to the gap's inline
path; the within-trial path was already correct. The accurate statement is:

> **Every number that depends on what survives a silent gap, in a run driven by
> `run_spatial_discovery` / `run_place_field_learning` / `run_theta_burst_45s` /
> `probe_latch2`, inherits `0.05`.**

Which, enumerated above, is **one** conclusion at risk — not the program.


---

## 6. Q4-7 — is `tier5_rnn` live? (ruling-011 §3: **determine, do not execute**)

**Measured, and the proposed test does not settle it — so I did not delete.**

Gen-2's test was *"a module nothing imports is dead."* **Something does import it:**

- `src/models/Model_6/Full_System_Experiments/run_tier3.py:66` —
  `from src.models.Model_6.Full_System_Experiments.tier5_rnn import exp_network_communication`,
  registered as a live experiment module at `:152`.
- `run_tier3.py` is imported by nothing, but it is a **documented CLI entry point**
  (`python run_tier3.py --list`), so "nothing imports it" is expected and proves nothing.

**But that import path is BROKEN, measured:**

```
from ...tier5_rnn import exp_network_communication
  -> ModuleNotFoundError: No module named 'run_credit_assignment'
```

Cause: `tier5_rnn/__init__.py:44` does `from run_credit_assignment import (...)` — an
**unqualified sibling import**, which only resolves if `cwd` is inside `tier5_rnn`. So the
package cannot be imported the way its own importer imports it.

**A second, independent blocker — and it is environmental, not the code's fault:**

```
import matplotlib.pyplot
  -> ImportError: cannot import name 'Group' from 'pyparsing'
```

`matplotlib` is **broken venv-wide**, and `exp_network_communication.py:42` imports it. So the
module cannot load even with `cwd` fixed.

**Staleness:** the entire `Full_System_Experiments/` tree was last modified **2026-04-08** — it
predates the option-c cutover (May 13–14), the Step-B backbone rebuild (June 2), the `E_invasion`
keystone (June 7), and today's consolidation.

### Finding

**`tier5_rnn` is UNREACHABLE in this environment, so its `coupling_weights` omission at
`exp_network_communication.py:200` cannot currently execute.** But *unreachable* is not *dead*:
it has a real importer and a documented entry point, and one of the two blockers is an
environment fault that a `pyparsing`/`matplotlib` repair would clear. **I did not delete, per the
ruling.** Delete / revive / quarantine is the MO's routing call.

**Incidental, checked because it could have been serious:** **zero** live sweep probes import
`matplotlib` (`src/models/Model_6/sweep/*.py`, `sweep/*.py`). The broken venv install does **not**
bite any live PO work. Worth one line in the MO's ledger, not an alarm.


---

# PART II — TWO corrections per artifact (MO gen-2, compute-sequencing note)

**An artifact can survive one correction and not the other, because THE TWO MOVE IN OPPOSITE
DIRECTIONS.** This is the part Part I could not see.

| state | effective gap `k_diss` coefficient | vs the conditions every standing artifact was computed under |
|---|---|---|
| **as all standing artifacts were run** | `0.05` (bare, no catalyst) | — |
| after the **`K`** correction alone | `0.005` | **10× SLOWER** |
| after **`K` + template symmetry** | `0.005 × 33 ≈ 0.165` | **3.3× FASTER** |

**The template correction dominates and reverses the sign of the change.** Under `K` alone the gap
clears *less* than any artifact assumed; under both it clears *more*. **A conclusion at risk from
one is often protected by the other.**

## The re-judged table

| artifact | survives `K` alone? | survives **both**? | why the two differ |
|---|---|---|---|
| **D17** — *"BOUNDED, no runaway"* | **NEEDS RE-MEASUREMENT** | **YES** | Under `K` alone, 10× less inter-trial clearing raises carryover and puts a *no-runaway* claim at risk. Under both, clearing is **3.3× faster than D17's own conditions** — and **faster clearing cannot manufacture unbounded accumulation.** This is a direction-of-inequality argument about an upper-bound claim, **not an estimate of a number**: the claim survives *a fortiori*. Its specific totals (9/1/31/23/22) still change, but that is digit-level and out of scope. |
| **D19** — latch structure | YES | **YES** | Latch counts and actin growth read neither dissolution rate. |
| **D20** — durable state | YES | **YES** | Actin / volume / clock / duty-cycle only. |
| **D21** — construct validity | YES | **YES** | kT sweep is gap-free; the bond result is `coupling_weights` + `η`. |
| **GAP-1** — clock + retention | YES | **YES** | `R` is an `E_invasion` (actin) ratio. |
| **GAP-2** — the ΔV separation | YES *(re-run verified)* | **YES — and now with a mechanism** | See below. |

### Why GAP-2 is insulated — checked, because it was the one genuinely exposed

Volume is actin-driven, and actin is driven by CaMKII/DDSC, which are driven by **calcium**. The
commitment pathway is explicitly *"dimers dissolve → **calcium return to spine** → CaMKII"*
(`model6-commitment-pathway`). So a 33× increase in gap dissolution **could** have fed CaMKII and
moved ΔV.

**It cannot, and the reason is a defect:** `apply_return` is called at `model6_core.py:484` and
`:782` — **the within-trial path only. The gap never calls it.** Gap dissolution destroys dimers
and **silently discards their calcium.** So ΔV is insulated from any change in gap dissolution
rate, at both corrections.

## NEW FINDING — the gap breaks mass conservation, and the template fix makes it worse

**The gap dissolves dimers without returning their calcium or phosphate.** My own advance/exclude
table lists *"Calcium dynamics — at baseline within ~2 s; clamped at baseline"*, which is a
defensible exclusion for **relaxation** — but it also silently discards the **source term** from
dissolution, which is a different quantity. **The stated reason does not cover what the code
actually drops.** That is a fault in my own table, found by asking whether GAP-2 was exposed.

**It scales with the correction:** at `0.05` the gap discarded the calcium of ~7 particles per 20 s;
at `0.005 × 33` it will discard ~3.3× more. **Fixing the template symmetry makes an unfixed
conservation break bigger.**

**Routed, not fixed — this is PO-2's surface** (mass conservation is that PO's whole objective, and
`atp_system.py` / the phosphate path are explicitly not mine). Whether the gap should return calcium
and phosphate on dissolution is a physics call adjacent to PO-2's finite-pool work.

## Consequence for the sequencing

**If the template fix lands, D17's NEEDS RE-MEASUREMENT closes as YES without buying a run** — the
direction argument above resolves it. **If it does NOT land, D17 stays NEEDS.** So the Q4-9
recommendation (*do not buy the re-measurement*) is strengthened, not weakened: **waiting is
strictly better than spending PO-5's slot.**
