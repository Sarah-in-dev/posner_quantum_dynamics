# PRE-REGISTRATION — PO-2 · the phosphate loop · L·PO4-1

**Registered:** 2026-07-18, BEFORE the measurement is run and BEFORE any physics change.
**Author:** PO-2 (the phosphate loop). **Method:** `MO_MODEL6.md` §2.4, `experiment-design-patterns`.
**Precedent followed:** `fa12009` (PO-1 committed its acceptance probe *failing* on current code
before touching physics) and PO-1's B2 probe design (positive controls that must FIRE, or the
probe returns INVALID rather than PASS).

---

## 0. The trap this registration exists to defeat

**"Total phosphate is conserved" is trivially true if you choose the wrong ledger.** Count only
`phosphate_structural + phosphate_metabolic` and the ATP arm is invisible; count only the
formation/dissolution half and you reproduce D8's *"exact conservation (2e-17 M)"* — a real
measurement of a loop that was never wired to the leak.

**That is not a hypothetical. It already happened in this program.** DECISION RECORD **D14**
records *"SOC loop already closed in live code (no B3 edit needed)"* off a probe whose phosphate
feedback was *"mimicking model6_core"*. `grep -n "ATP|hydrolys|recovery"
sweep/phosphate_conservation_probe.py` returns **zero hits** — that probe has no ATP arm at all.
**So a conserving result is the DEFAULT outcome of a badly-scoped ledger**, and a probe that
returns CONSERVED proves nothing unless its ledger provably spans the ATP arm.

The discriminator is therefore **not** "is it conserved" but **"does the measured drift equal a
quantity predicted in advance from the defect's mechanism."**

---

## 1. The conserved quantity (fixed now, not after)

The cycling-phosphate ledger, summed over the whole grid. One terminal phosphate per ATP:

```
P_total(t) = Σ[ATP]                     # terminal Pi carried on ATP
           + Σ phosphate_released        # hydrolysed, in flight, not yet binned
           + Σ phosphate_metabolic       # the 90% rapid-cycling pool
           + Σ phosphate_structural      # the 10% Posner-available pool
           + 4·Σ[dimer] + 6·Σ[trimer]    # locked into clusters (ca_triphosphate_complex.py:437-438)
```

**Per-transition audit, derived from the code, registered as the mechanism under test:**

| transition | site | ledger effect |
|---|---|---|
| hydrolysis `ATP -= d; phosphate_released += d` | `atp_system.py:128-130` | **conserved** |
| binning `released → metabolic + structural` | `atp_system.py:419-428` | **conserved** |
| formation/dissolution `structural ∓ 4·d_dimer` | `model6_core.py:450-452`, `:756-757` | **conserved** (signed; `ca_triphosphate_complex.py:430-438`) |
| **recovery `ATP += d; ADP -= d`** | **`atp_system.py:163, 169-171`** | **`P_total` INCREASES by `d` — nothing debited** |

---

## 2. The registered prediction — a NUMBER, not a direction

The defect is not merely "it leaks". The leak is **exactly** the ATP regenerated, which the code
already accumulates for free at `atp_system.py:174` (`self.total_recovered += np.sum(delta_atp)`).

> **PRIMARY REGISTERED PREDICTION:**
> ```
> ΔP  ≡  P_total(t_end) − P_total(t_0)  ==  hydrolysis.total_recovered
> ```
> to within the tolerance in §3 — **on current, unfixed code.**

This is what makes the measurement able to fail. Three distinguishable outcomes, registered now:

- `ΔP ≈ total_recovered` → the defect is **exactly** as diagnosed. Fix is stoichiometric.
- `ΔP ≈ 0` → **there is no leak** and my dispatch's central claim is wrong. I report that and
  the SOC engine's status is re-opened, not defended.
- `ΔP ∉ {0, total_recovered}` → there is a **second, unidentified** phosphate source or sink.
  I do not fix anything until it is identified; a partial fix on a misdiagnosed ledger is the
  `683b82f` shape.

**After the fix, the registered pass condition is `|ΔP|/P_total(t_0) ≤ ε` with `total_recovered
> 0`** — i.e. recovery must actually have run. A conservation pass with `total_recovered == 0`
is the D14 failure and is scored **INVALID, not PASS.**

---

## 3. Tolerance — stated, with its justification

**ε = 1×10⁻¹² relative drift**, i.e. `|ΔP| / P_total(t_0) ≤ 1e-12`.

**Why this number.** The ledger is a float64 sum over `grid_shape` points accumulated across
`N` steps. Machine epsilon is 2.22e-16; worst-case accumulated round-off for pairwise/naive
summation over `N·G` additions grows no faster than `~sqrt(N·G)·eps` for uncorrelated error.
At `N ≈ 10⁴` steps and `G ≈ 10³` grid points that bound is `~7e-14` — so **1e-12 leaves a
factor ~14 of headroom over numerical noise while remaining far below any physical effect.**

**Why the exact value does not decide the verdict — which is the real defense.** The predicted
leak is of order the ATP turnover, i.e. **relative drift ~1e-1 to 1e0**. That is **eleven to
twelve orders of magnitude above ε.** Any tolerance between 1e-14 and 1e-3 returns the same
verdict. **I state ε because §2.3 requires it, not because the result is sensitive to it** — and
I register that insensitivity now so that a later reader cannot suspect the tolerance was chosen
to reach the outcome. If the measured drift ever lands *near* ε, the verdict is INCONCLUSIVE and
I report it as such rather than rounding it to a pass.

---

## 4. Controls — both must fire, or the run is INVALID

**C1 — the detector can fire (positive control).** Inject a known synthetic phosphate creation
`Δ_inject` directly into `phosphate_structural` mid-run. The ledger must report drift equal to
`Δ_inject` (plus the recovery leak) to within ε. **If C1 does not fire, the ledger is blind and
the probe returns INVALID** — a conservation checker that has never been seen to detect a leak
is indistinguishable from one that cannot.

**C2 — the detector does not cry wolf (negative control), AND it explains D8/D14.** Run the
identical ledger with ATP recovery suppressed (`atp_recovery_tau → ∞`, no physics edited).
Registered prediction: **`ΔP` collapses to ≤ ε** while formation, dissolution and consumption all
still run.

C2 is doing double duty and that is deliberate: it is the control **and** it is the reconciliation
with the decision record. If C2 conserves and the full run does not, then **D8 and D14 were correct
measurements of the half of the loop that does not leak** — and their scope, not their arithmetic,
is what was over-read. That is a testable claim about a prior finding, registered before running.

**C3 — the clamp detector.** `model6_core.py:451`/`:757` clamp `phosphate_structural` at 0 via
`np.maximum`, which *creates* phosphate when it binds. The probe counts clamp activations. Any
activation is reported as a **distinct** failure mode (`CLAMP_FIRED`), never folded into the
headline drift — see `queue/po2-phosphate.md` Q1, where I recommend instrumenting rather than
removing it.

---

## 5. Defect 1 (the stale `phosphate_total`) — registered separately

**Kept in its own commit and its own verdict**, per the dispatch: contained correctness bug vs
structural failure.

**Registered prediction:** across a run in which dimers form and consume phosphate,
`phosphate_total` (`atp_system.py:428`) and `phosphate_structural + phosphate_metabolic` **diverge**,
because the total is recomputed only inside `add_phosphate_from_atp` while consumption at
`model6_core.py:450-452` and `:756-757` decrements the structural pool without it.

**The consumer-level demonstration (this is the acceptance, not the divergence itself):** J-coupling
reads `phosphate_total` at `atp_system.py:485`. Registered: on current code, the J-coupling field is
**invariant** to dimer consumption; after the fix it **tracks** it. Reported as the correlation
between J-coupling and cumulative phosphate consumed, before and after — a data-level demonstration,
not "the field changed".

**Both consumption sites are in scope.** A fix applied to one is a partial fix of the exact shape
audit item 16 recorded for `analytical_gap`. **Preferred remedy: make `phosphate_total` a derived
property** so it cannot go stale at any site, present or future, rather than adding a third
recompute call.

---

## 6. What I will NOT do (§7 LOCKED, and the temptation named in my dispatch)

- **No compensating term.** The easy way to make this balance is to add a correction that absorbs
  the residual. That is tuning a constant to reach an outcome. The pool must balance because
  ATP = ADP + Pi, or the log records the gap.
- **No touching `K_CLASSICAL`.** MO-held. Reported, not moved.
- **No re-deriving `productive_fraction`** or any chemistry rate to improve the balance.
- **No widening ε after seeing the result.** ε is fixed at §3 as of this commit.
- **If the fix does not close the loop, I report a non-conserving loop as the finding** and PO-6
  stays blocked. A negative result here is a real result: it says the SOC engine does not exist.

---

## 6b. AMENDMENT A2.3 — the debit rule, registered as a stated modelling decision

*(Added 2026-07-18 21:30Z per MO ruling 003, which withdrew ruling 002 §1's conclusion that this
choice was pure bookkeeping. Registered here because ruling 002's "pick for verifiability" framing
would have left it unregistered, and it is not a bookkeeping choice.)*

**REGISTERED CHOICE: metabolic-first.** ATP synthesis debits `phosphate_metabolic` first and draws
on `phosphate_structural` only for the remainder.

**Reason, stated as a modelling decision rather than derived:** `add_phosphate_from_atp` sends 90%
of hydrolysis-released Pi to the metabolic pool, whose own docstring calls it *"protein binding,
rapid cycling"* — physically the pool mitochondrial resynthesis draws from. Recovery taking back
what hydrolysis just released, before drawing on the ~1 mM structural reserve, is the more
defensible mechanism. It is **also the conservative choice for this pre-registration's own claim**:
it spares the chemically active pool, making the SOC depletion feedback weaker and the claim harder.

**Registered as measurable, not assumed inert:** the alternative (proportional) is implemented as a
switchable arm and both are reported. Conservation is predicted **invariant** to the choice (the
ledger sums both pools; the total debited is mode-independent), while the **structural** pool — the
one speciation reads — is predicted to **differ**, because the structural pool is ~500× the
metabolic and absorbs ~99.8% of a proportional debit. Both predictions confirmed: `max |dP|/P =
1.157e-14` both modes; structural differs 0.116%.

**Escalated to Sarah** as a mechanism choice with a chemical consequence. A ruling either way is a
one-line change in `consume_for_atp_synthesis` and leaves §2's conservation result untouched.

## 7. Limits of this measurement, stated in advance

- It measures **mass conservation of phosphate**, nothing else. Conservation is **necessary, not
  sufficient**, for SOC. A conserved loop can still fail to self-organize — that is PO-6's
  drive×damping sweep to determine, and this PO does not pre-empt its verdict.
- It does not ground the **"2% ATP replenish"** clause of Step E (`may30` pin `:51`). Making the
  loop conserve and grounding the replenish rate are different jobs; I report the second as
  outstanding rather than quietly satisfying it.
- The ledger asserts **one terminal phosphate per ATP**. ADP's remaining phosphates are not
  tracked because the code never cycles them; if that assumption is wrong the ledger is wrong,
  so it is registered here as an assumption rather than buried in the probe.
- Grid/step configuration is fixed before the run and reported with the result.
