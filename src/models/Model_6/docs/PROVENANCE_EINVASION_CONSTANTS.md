# PROVENANCE VERDICT — the two `E_invasion` constants

**PO-3, 2026-07-18.** The second half of PO-3's acceptance. Every claim below carries its
`file:line` or its measurement; anything not verified is labelled UNVERIFIED rather than
filled in with something plausible.

**Scope:** `k_polymerization_max` (the CHARGE side) and `E_ref` (the load-bearing
denominator). `tau_extrude` is included for contrast because it is the one that was already
grounded and is the constant the ratchet prediction rests on.

**No constant was written.** The sensitivity figures below come from in-memory parameter
variants in a throwaway audit script; `spine_plasticity_module.py` is unmodified.

---

## Summary table

| constant | value | `file:line` | VERDICT |
|---|---|---|---|
| `tau_extrude` | 180.0 s | `:109` | **GROUNDED** — Honkura 2008, in-band, and independently reproduced in-code (below) |
| `E_ref` | 1.87 | `:115` | **REPRODUCIBLE, SELF-REFERENTIAL** — not UNVERIFIED as the audit assumed |
| `k_polymerization_max` | 0.1 s⁻¹ | `:90` | **INHERITED, and 3.57× its own citation** — but largely inert at the ceiling, decisive early |

---

## 1. `E_ref = 1.87` — REPRODUCIBLE, SELF-REFERENTIAL (upgraded from UNVERIFIED)

The 2026-07-18 substrate audit called this UNVERIFIED — *"the model's own asymptote frozen
as a constant, with no artifact tying it to a run."* **The first half is right and the
second half is wrong: the generating run is in the repo.**

`tests/check_actin_three_pool.py:142-157` is exactly the run the comment at `:115` names —
*"asymptotic enlargement under sustained maximal Ca (3000 s uncommitted run)"*:

```
# ---- Phase 5: SUSTAINED UNCOMMITTED (3000 s, Ca=2.0 uM, drive=0, fresh instance) --
mod5 = SpinePlasticityModule(params)   # same config as phases 1-4, fresh state
...
enl_ceiling_uncommitted = mod5.actin_enlargement
```

and `:286-288` prints it, labelled *"Candidate physical anchor for E_ref (decision
pending)."*

**Re-run 2026-07-18 by PO-3. It reproduces:**

```
+ 1200s  enl=1.8423   + 1800s  enl=1.8595   + 2400s  enl=1.8690   + 3000s  enl=1.8742
```

`1.8742` against the coded `1.87`. **Reproduced to the precision the constant is stated at.**

**Verdict: REPRODUCIBLE, SELF-REFERENTIAL.** It is not a literature measurement and must
never be described as one — it is the model's own uncommitted asymptote, used as the
normalization that defines "`E_invasion = 1` means saturated". That is a legitimate
*definition* of a scale, but it is not independent evidence, and a paper claim of the form
"`E_invasion` is grounded in measurement" would be false at this constant.

**Consequence for the 13× shortfall — this is the load-bearing part.** Because `E_ref` is
internally consistent and reproducible, **the live-trial shortfall cannot be explained by
`E_ref` being wrong.** The MO flagged `E_ref` as the denominator setting the whole scale of
`E_invasion` and therefore of `r`, and asked whether an unverified constant in that position
was why the shortfall could not be read as physics. Answer: **it is not.** The denominator
is reproducible. That reading is closed.

**The doc drift to fix (not fixed here — it is prose, and prose about my own surface):**
`model6-actin-invasion-driver` §5 says E_ref was *"read once off a 3000 s uncommitted run"*
and gives no pointer; the run is `tests/check_actin_three_pool.py` Phase 5 and should be
cited by path so the next reader does not have to rediscover it.

---

## 2. `k_polymerization_max = 0.1 s⁻¹` — INHERITED, and 3.57× its own citation

### 2.1 The citation does not support the value

`spine_plasticity_module.py:88-90`:

```
    # Polymerization (Arp2/3 mediated, calcium/CaMKII dependent)
    # Bosch et al. 2014: maximal polymerization ~2x baseline in first 2 min
    k_polymerization_max: float = 0.1    # s⁻¹, rate constant for new actin
```

A **fold-change over a window** is cited; a **rate constant in s⁻¹** is coded. No arithmetic
connects them anywhere in the repo (`grep` over all `.py`/`.md`: the symbol appears at its
definition `:90`, its single use `:386`, and in coordination/doc prose — nowhere is it
derived).

The file's own header already flags the source as unverified — `:29`:

```
    - Bosch et al. 2014 (Neuron): actin dynamics and phases. Attributed, not verified.
```

### 2.2 It is INHERITED in the literal git sense

`git log -S "k_polymerization_max: float = 0.1"` returns exactly one commit: **`703d394`,
2025-12-08, "keep building the model."** It has **never been touched since** — in
particular it survived the grounded three-pool rewrite (`2b59fc2`, 2026-06-07) untouched,
which is the commit that grounded `tau_extrude`, `S0`, `k_exchange`, `k_conf` and `k_unconf`
against named papers. **The charge side was not re-derived when the discharge side was.**

**Verdict: INHERITED.** Confirmed, with the mechanism of inheritance identified.

### 2.3 What the value would have to be — measured

Audit computation (in-memory variants; the model's own `SpinePlasticityModule` integrated to
2 min, `structural_drive = 0`, so this is the model answering for itself):

```
  k (s^-1)   Ca=2uM fold   Ca=15uM fold
     0.010         1.446          1.469
     0.020         1.769          1.802
     0.030         2.006          2.043
     0.050         2.323          2.360
     0.100         2.693          2.719   <== CODED
     0.200         2.907          2.918
```

**The value reproducing Bosch's cited "~2× baseline in first 2 min" at the live nanodomain
calcium is `k ≈ 0.0280 s⁻¹`. The coded value is 3.57× larger than its own citation implies.**

### 2.4 Two opposite consequences — and the second one is the one that matters

**At the ceiling, the constant is nearly inert.** Doubling `k` from 0.1 → 0.2 moves the
2-min fold only 2.693 → 2.907 (+8%). By `k = 0.1` the system is already **room-limited**:
`room = max(0, 1 - F/F_max)` at `:383` is what bounds the pool, not the rate. So for
sustained/asymptotic behaviour — including `E_ref` itself — this constant is not doing the
work its name suggests, and its 3.57× error is largely absorbed by the ceiling.

**In the LIVE regime it is decisive, and it is decisive in the unhelpful direction.** The
live trial operates far from the ceiling (`E_invasion = 0.0868`), where `room ≈ 1` and
formation is approximately linear in `k`:

```
  t (s)   E_inv k=0.028   E_inv k=0.100    ratio
     14          0.0637          0.2490     3.91
     34          0.2015          0.5439     2.70
     60          0.3358          0.7615     2.27
    120          0.5182          0.9240     1.78
```

**Therefore: correcting `k_polymerization_max` toward its own citation would make the live
shortfall roughly 3.9× WORSE at traversal timescales, not better.**

This forecloses the tempting reading — that because the charge side is ungrounded, grounding
it might be what closes the 13× gap. It does the opposite. **The ungrounded constant is
already ~3.6× more generous than its citation supports**, and the measured shortfall exists
*despite* that generosity. Any future decision to ground this constant properly must be
taken knowing it moves the model away from ignition, and it is a physics call
(`MO_MODEL6.md` §7: no constant tuned to a downstream target) — **not PO-3's, and not
proposed here.**

---

## 3. `tau_extrude = 180 s` — GROUNDED, and independently confirmed in-code

Already grounded (Honkura 2008, 2–15 min band, `>6 min` retention threshold). Recorded here
because the ratchet prediction rests on it and the confirmation is worth having on file.

`tests/check_actin_three_pool.py` re-run, 2026-07-18:

```
[PASS] DECAY emergent tau: ratio@180s=0.3681 (~0.37 ideal), implied tau=180.1s = 3.00 min
```

The decay constant is recovered from the integrated module at **180.1 s** against the coded
180.0 s. The extrusion clock does what the constant says, in code, at the isolated-module
level. This is also a useful positive control for the L·ETA-5 gap question: an isolated
module *does* decay at `tau_extrude` when stepped, which is the behaviour
`analytical_gap` skips (finding F-2).

---

## 4. What this audit does NOT establish

- It does **not** verify Bosch et al. 2014 itself. The paper was not read; `:29` already
  labels it "Attributed, not verified", and this audit takes the citation *as written in the
  comment* and tests the arithmetic against it. **If the comment misstates Bosch, the 3.57×
  figure is against a misquotation, not against the paper.** Resolving that needs the paper.
- It does **not** establish that `k_polymerization_max` should be 0.028. That is what the
  *cited fold-change* implies under this model's own dynamics; it is not a measured rate
  constant, and a fold-change plus a model is not a rate.
- `dynamic_pool_baseline = 0.85` / `stable_pool_baseline = 0.15` remain **UNGROUNDED as
  fixed fractions**, already annotated at `:79-86`. Not re-litigated here.
