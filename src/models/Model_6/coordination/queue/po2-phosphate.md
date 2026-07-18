# Queue: po2-phosphate — actions awaiting the MO / Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready.

## Q1 — Does the `np.maximum(..., 0.0)` phosphate clamp count as the pin's forbidden "cap"?

**Ask:** ruling on whether the clamp at `model6_core.py:451` and `:757` may stay.

**Why:** the may30 pin's Step E reads *"A finite pool must actually be finite. Correctness,
**not a cap**, not an edge-target."* The clamp is literally a cap on the pool floor, and when it
fires it **creates phosphate** — the pool would have gone negative and instead is held at 0.
That is a conservation violation in the direction of the defect I was dispatched to fix.

**PO's recommendation: LEAVE IT, INSTRUMENT IT.** It is a symptom, not the cause. If the clamp
ever fires, conservation was already violated upstream — so the right move is to make the probe
*detect* clamp activation and report it as a distinct failure mode, not to silently remove it.
Removing it lets `phosphate_structural` go negative, which is worse physics and would produce a
nonsense speciation at `atp_system.py:399-401`. **I do not read this as the "cap" the pin
forbids** — the pin's cap is a *ceiling* installed to stop runaway formation (the 427-dimer
problem); this is a *floor* preventing negative concentration. Different object, same word.
Flagging because the wording is close enough that I should not decide it alone.

**Evidence:** `model6_core.py:450-452`, `:756-757`; pin Step E; `atp_system.py:399-401`.

**Not blocking** — the probe measures and reports clamp activation either way.

---

## Q2 — Which pool should ATP recovery debit: metabolic-first, or proportional?

**Ask:** ruling on the split, because it is arguably a physics call and I must not make those.

**Why:** fixing defect 2 means `update_recovery` debits phosphate when it regenerates ATP
(ATP = ADP + Pi). But there are two pools, and `atp_system.py:397-398` declares
*"Calculate concentrations FROM STRUCTURAL POOL ONLY / Metabolic pool doesn't participate in
Posner chemistry"*. Which pool the debit lands in **changes how much phosphate remains available
to dimer formation**, so it is not a bookkeeping detail — it directly sets the SOC feedback
strength this PO exists to make real.

**PO's recommendation: METABOLIC-FIRST, falling back to structural only when metabolic is
exhausted.** `add_phosphate_from_atp` (`:419-425`) sends **90%** of hydrolysis-released Pi to the
metabolic pool, described at `:407` as *"protein binding, rapid cycling"*. Mitochondrial ATP
resynthesis physically draws from exactly that rapidly-cycling inorganic pool, not from the
structural pool reserved for Posner chemistry. Metabolic-first is also the **conservative** choice
for my own acceptance: it leaves the structural pool larger, which makes the SOC feedback *weaker*
and therefore harder for me to claim. I would rather under-claim the engine than tune toward it.

**The thing I will NOT do:** introduce a fitted partition coefficient. If metabolic-first does not
balance, the log records the gap (§7 LOCKED, and the temptation my kickoff named explicitly).

**Evidence:** `atp_system.py:163`, `:169-171`, `:397-398`, `:407`, `:419-425`.

**Not blocking** — I pre-register metabolic-first as primary and report the proportional variant
as a sensitivity, so a ruling either way is absorbed without a re-run.

---

## Q3 — **MY ACCEPTANCE ITEM 2 CANNOT BE MET AS WRITTEN.** J-coupling does not read phosphate at all.

**This is a physics call and a correction to my own dispatch. Escalating, not deciding.**

### The finding, AST-proven (not read off a docstring)

`atp_system.py:263-306`, `calculate_j_coupling(self, atp, phosphate, activity)`:

```
signature args : ['self', 'atp', 'phosphate', 'activity']
names loaded   : ['K_bind', 'activity', 'activity_enhancement', 'atp', 'frac_atp_bound', 'np', 'self']
  atp        -> READ
  phosphate  -> *** NEVER READ ***
  activity   -> READ
```

The body computes `frac_atp_bound = atp / (atp + K_bind)` and an activity multiplier. **The
`phosphate` argument is dead.** Its own docstring at `:277` declares *"phosphate: Total phosphate
field (M)"* — **prose describing a dependency the code does not have.** I verified by AST rather
than by reading the body, because "the MO read `analytical_gap`'s docstring and was wrong" is the
named scar of this session and I am not entitled to repeat it in the other direction either.

### What this does to the dispatch

My acceptance item 2 reads *"J-coupling demonstrably tracks dimer consumption"*, and both
`MO_MODEL6.md` §3 PO-2 and my kickoff state the mechanism as: `phosphate_total` goes stale ⇒
*"J-coupling (`atp_system.py:485`) reads a phosphate field that ignores dimer consumption."*

**That mechanism is false.** J-coupling does not read a phosphate field that ignores dimer
consumption; it reads **no phosphate field**. Fixing the staleness therefore **cannot** make
J-coupling track consumption. Measured, before any fix: `corr(J, cumulative PO4 consumed) =
-0.069` with `J std = 9.4e-02` — the field varies, just not with phosphate. **That correlation
will still be ~0 after defect 1 is fixed**, and I register that prediction here, before fixing.

### And it downgrades defect 1's severity — honestly

`grep --include="*.py" "phosphate_total"` over the repo: the instance field's **only** consumer is
`atp_system.py:485`, i.e. the dead argument. (The other hits are `params.phosphate_total`, a
different object — the initial-condition parameter, which `sweep_runner.py:77` does sweep.)

**So the stale `phosphate_total` is currently INERT: a real correctness bug with no live
consumer.** I will still fix it — a wrong field is a trap for the next consumer, and the fix is
structural — but I will **not** claim I fixed a live defect, and the research-log row will say so.

### PO's recommendation

**Fix the staleness (mine, cheap, structural). Do NOT wire `phosphate` into the J-coupling
physics — that is Sarah's call, not mine.** Making J-coupling depend on phosphate concentration
would change the quantum-protection mechanism itself (Fisher 2015 `J_PP_atp` = 20 Hz vs
`J_PO_free` = 0.2 Hz weighting). That is new physics, not a correctness fix, and §7 LOCKED plus my
own "escalate, do not decide" boundary both put it out of my reach. **I flag additionally that
the intended physics may well be that the ATP-bound *fraction* should be computed against the
phosphate pool rather than against a fixed `K_bind = 1e-3` — but "may well be" is exactly the
inference I am not entitled to act on.**

**Consequence for my acceptance:** item 2 is **NOT MET and cannot be met without a physics
ruling.** I am reporting it unmet rather than substituting a weaker demonstration that would pass.
Item 1 (conservation) is unaffected and I am proceeding with it.

**Related precedent, for whoever rules:** this is the same shape as D21(1) — *"`quantum_field_kT`
is INERT in spine plasticity … accepted at three call sites, read in none; the module docstring
describes a quantum barrier-modulation mechanism that does not exist in the code."* **Second
instance of declared-but-unread quantum coupling in this program.** That pattern is worth a look
beyond my surface.

---

## Q4 — **RULING 002 §1 IS WRONG ON ITS CONCLUSION, and I have the measurement.** Q2 is not dissolved.

**Ruling 002 §1 states:** *"ATP-derived phosphate never enters speciation, so it never becomes a
Posner-forming species … **but it means your debit choice cannot affect the chemistry**, only the
ledger. Pick the one that makes conservation cleanest to verify … and stop treating it as physics."*

**The premise is correct. The conclusion does not follow, and it is contradicted by measurement.**

### Why it does not follow

The premise is about phosphate going **IN** — ATP-derived Pi lands mostly in the metabolic pool,
which speciation ignores (`atp_system.py:453-457`, structural pool only). True.

**But the debit choice is about which pool phosphate comes OUT of.** Metabolic-first *spares* the
structural pool; proportional drains it, because the structural pool is ~500× larger and so takes
~99.8% of a proportional debit. **The structural pool is the chemically active one.** So the debit
choice directly sets how much Posner-forming phosphate is removed per step.

### Measured, both modes, identical seed and configuration

```
metabolic_first : structural=9.985709043e+00   metabolic=1.881984474e-02
proportional    : structural=9.974125559e+00   metabolic=3.037113848e-02
STRUCTURAL differs by 1.158348e-02  (0.116%)
```

**The chemically-active pool differs between the two modes.** It is not a bookkeeping-only choice.
Downstream, standing dimer differed by ~2.3% with the sign positive in 3/3 seeds — which I am
**still not claiming as an effect** (sign-test p = 0.25; see the previous entry). **The 0.116%
structural difference, however, is not a statistical claim — it is deterministic bookkeeping, and
it lands squarely on the chemically active pool.**

### What I did, and what I recommend

**I kept metabolic-first**, which the ruling's own logic endorses for the wrong reason and my
pre-registration endorsed for the right one: it *spares* the chemically active pool, making the SOC
depletion feedback **weaker** and my own claim **harder**. Conservation is invariant either way
(max |dP|/P = 1.157e-14 over 6 runs), so **acceptance item 1 is untouched by this dispute.**

**Recommend:** treat the debit rule as a **stated modelling choice with a chemical consequence**,
not as pure bookkeeping — i.e. it stays worth pre-registering, which is what §2.4 asks for anyway.

### The limit the ruling asked me to record — sharpened

Ruling 002 asks me to state that *"conservation around the loop is conservation of a quantity that
does not feed formation."* **Half right, and worth stating precisely:** my ledger conserves TOTAL
phosphate; only the **structural** sub-pool feeds formation; the metabolic pool is chemically inert
by design. **So the conserved quantity is broader than the chemically active one — and the debit
rule is exactly the valve between them.** That is the honest limit, and it is also why the ruling's
"cannot affect the chemistry" reading is the one thing in it I cannot adopt.

*Raised because a ruling that is wrong in the MO's favour is still wrong, and this one would have
had me stop pre-registering a choice that reaches the chemistry.*
