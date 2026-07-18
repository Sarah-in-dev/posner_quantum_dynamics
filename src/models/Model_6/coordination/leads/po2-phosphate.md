# Lead: po2-phosphate (PO-2 · the phosphate loop) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** total phosphate conserved around a full
hydrolysis → consumption → dissolution → recovery cycle to a **stated** tolerance, with the
check **shown failing on current code first**; and J-coupling demonstrably tracking dimer
consumption.

**Status:** LIVE — grounding brief returned 2026-07-18 19:46Z.
**Current unit:** pre-registration (§2.4), then the conservation probe committed FAILING first.
**Last heartbeat:** 2026-07-18 20:52Z
**Blocked on:** — (two open questions in `queue/po2-phosphate.md`, neither blocking; working on)

## HEARTBEAT LOG

### 2026-07-18 19:46Z — GROUNDED. Three corrections to the dispatch, all code-verified.

Both defects confirmed at the code level as the kickoff described them. Three things the
kickoff did not have:

1. **TWO consumption sites, not one.** `model6_core.py:450-452` (EM path) **and
   `model6_core.py:756-757`** (`=== NON-EM PATH ===`) carry byte-identical decrement blocks;
   neither updates `phosphate_total`. Same two-definitions shape the MO recorded for
   `analytical_gap`. A fix to one leaves the other live. Fix must be structural, not
   dual-patched.

2. **Dissolution ALREADY returns Pi — do not "fix" it.** `ca_triphosphate_complex.py:430-438`:
   `d_dimer_dt = dimer_formation - dimer_dissociation`, then
   `_po4_consumed = 4.0*d_dimer_dt + 6.0*d_trimer_dt`, commented *"Net formation: negative
   values return Ca to pool on dissolution"*. The quantity is **signed**. Step E's "return Pi
   from particle dissolution" is satisfied at this site. The live defect there is instead the
   `np.maximum(..., 0.0)` clamp at `:451`/`:757`, which creates phosphate at pool exhaustion —
   and the pin says "not a cap".

3. **DECISION RECORD D14 contradicts this dispatch, and both are right about different
   halves.** D14: *"SOC loop already closed in live code (no B3 edit needed) … phosphate
   feedback **mimicking model6_core**"*; D8: *"exact conservation (2e-17 M)"*. Verified against
   code, not prose: `grep -n "ATP|hydrolys|recovery" sweep/phosphate_conservation_probe.py`
   returns **ZERO hits**. The A3 probe has **no ATP arm**. D8/D14 measured the
   formation↔dissolution half — which does conserve — and that has been read as the whole
   loop. **The loop D14 declared closed was never wired to the leak.**

**`K_CLASSICAL` report (MO-held, untouched):** `ca_triphosphate_complex.py:160` = `0.005` ✅ ·
`sweep/phosphate_conservation_probe.py:70` = `0.005` + Turhan citation ✅ · `analytical_gap`
both copies = `0.05` (retired; PO-4's surface) · `dimer_particles.py:127` = `0.001`. Three-way
spread stands. Reported, not touched.

**Tree state on arrival:** NOT clean — PO-4 holds uncommitted edits to `PREREG_PO4_GAP.md` and
`src/models/Model_6/sweep/gap_retention_probe.py`. Those are PO-4's own files; `atp_system.py`
and `model6_core.py` are untouched, so my slice is free. No collision.

**Why now:** PO-1's B2 landed and is MO-verified; the tree is clean, so the `model6_core.py`
shared-file boundary the board gated PO-2 on is met.

**Two defects, kept separate:**
- `phosphate_total` stale (`atp_system.py:428` recomputes only inside `add_phosphate_from_atp`;
  `model6_core.py:450-452` decrements `phosphate_structural` without it) ⇒ J-coupling
  (`atp_system.py:485`) reads a field ignoring dimer consumption. **Contained bug.**
- **ATP↔Pi not mass-conserving** — hydrolysis credits Pi (`:130`), recovery regenerates ATP
  (`:163`, `:169-170`) without debiting any pool. **This breaks Step E: the finite pool is not
  finite around the loop, the reset feedback does not close, and the SOC engine does not exist.**

**Owns:** `atp_system.py`, the phosphate path in `model6_core.py`.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params (PO-1 owes a D/φ
item) · PO-3's `spine_plasticity_module.py` · PO-4's `analytical_gap` in BOTH drivers and
`run_theta_burst_45s.py` (actively consolidating) · PO-5's `multi_synapse_network.py` + T1′
family · PO-6's surfaces.

**Shared-file hazard:** `model6_core.py` — PO-1/PO-4/PO-2. One uncommitted holder at a time.
Commit at the boundary; a broken shared tree blocked two POs earlier today.

**NOT this PO's to decide:** `K_CLASSICAL` — MO-held, decision-ready with Sarah. `0.005` is live
and correct in `ca_triphosphate_complex.py:160`; `analytical_gap` still runs the retired `0.05`.
Report, do not touch.

**Gates:** PO-6 (HARD). The drive×damping sweep measures nothing against a non-conserving loop.

**Prior art:** `sweep/phosphate_conservation_probe.py` (repo root) already exists and already
uses the correct `K_CLASSICAL = 0.005` with its citation. Extend it; do not rebuild.

### 2026-07-18 20:05Z — pre-registration committed `a9f0767`; ledger probe mid-flight

**Committed BEFORE running anything scored:** `docs/PREREG_PO2_PHOSPHATE.md` (L·PO4-1).
The registered discriminator is **not** "is it conserved" — a conserving result is the *default*
outcome of a badly-scoped ledger, which is exactly how D8/D14 happened. It is:

    dP == hydrolysis.total_recovered   (on current, unfixed code)

with three outcomes, **two of which contradict my own dispatch** (`dP ~ 0` ⇒ there is no leak and
the dispatch is wrong; `dP ~ neither` ⇒ a second unidentified source/sink, fix NOTHING).
Tolerance eps = 1e-12 relative, justified against float64 accumulation AND registered as not
deciding the verdict (predicted leak is ~11 orders above it).

**`sweep/phosphate_ledger_probe.py` written and running against the LIVE
`Model6QuantumSynapse`** — nothing mimicked, because mimicking `model6_core` is the precise D14
failure I registered against. Three arms: MAIN, C1 (inject a known leak — the ledger must detect
it), C2 (recovery suppressed — must conserve, which doubles as the D8/D14 reconciliation).

**MY OWN DEFECT, disclosed:** I launched a 3-arm × 4000-step full-model run with **no progress
instrumentation and no incremental persistence**, against the MO's compute rule (*"progress
instrumentation, results persisted incrementally"*). It is at 12.5 min CPU at 100% and I
**cannot tell how far along it is** — which is precisely what that rule exists to prevent. Not
killing it (it is within the cheap-probe expectation and likely near done), but the next run gets
per-arm progress output before it starts. Logged as mine, not discovered later.

**Compute so far:** ~13 min CPU, single core, one backgrounded process. Well inside the 130-min
precedent, but I did not request a slot because a conservation check was expected to be cheap.
If this exceeds ~30 min I stop and request sequencing rather than letting it run unbounded.

### 2026-07-18 20:34Z — ACCEPTANCE: item 1 MET, item 2 NOT MET (and cannot be, without a physics ruling)

**Item 1 — conservation. MET, shown failing first.**
`dP` +1.098435e-02 (LEAK_MATCHES_RECOVERY, `305e096`) → +3.410605e-13 (CONSERVED, `11aec6f`).
Leak removed 3.221e10×. Tolerance ε=1e-12 relative, stated and justified, and empirically
validated by C2's 1.1e-14 noise floor. All three controls fired.

**The cheat-check passed:** ATP recovered is **bit-identical** before/after
(1.098435256113e-02), so conservation was not bought by damping recovery.
**Which is also the limit:** the Pi limit never bound, so **depletion feedback is NOT
demonstrated.** Conservation is necessary, not sufficient, for SOC.

**Item 2 — J-coupling tracks dimer consumption. NOT MET.** `calculate_j_coupling` never reads
its `phosphate` argument (AST-proven). The dispatch's stated mechanism is false; the fix cannot
deliver the bar. Reported unmet rather than substituting a weaker passing demonstration.
Escalated as queue Q3 — physics call.

**Deliverables landed:** prereg `a9f0767` · failing probe `305e096` · defect 1 `837a511` ·
defect 2 `11aec6f` · research log rows PO2-1 and PO2-2.

**Open, all in queue/po2-phosphate.md, none blocking me:** Q1 clamp semantics (C3 shows 0
activations, so off the critical path) · Q2 metabolic-first vs proportional (implemented as
recommended; only `consume_for_atp_synthesis` changes if ruled otherwise) · **Q3 the J-coupling
physics call — this one gates my item 2.**

**On the PO-2 → PO-6 edge:** conservation is fixed, so the sweep would no longer measure a
loop that creates mass. **I do NOT declare the edge cleared** — that is the MO's call, and it
should weigh that depletion feedback is unexercised in the tested regime. Recommend PO-6 be
unblocked for the sweep while treating "does the pool actually deplete" as an open input.

### 2026-07-18 20:52Z — MO ruling 001 received and discharged (Q1, Q3); Q2 sensitivity running

**Polled `requests/po2-phosphate/` — ruling 001 was waiting. All three questions ruled.**

**Q3 — CONFIRMED, and the MO traced it one step further than I did.** My acceptance item 2 is
**REPLACED** with: *"Establish, and report with `file:line` evidence, whether `phosphate_total`
has any live consumer after your fix — and state plainly that J-coupling does not read phosphate."*

**DISCHARGED. Verified independently by AST rather than relaying the MO's line numbers**
(a PO relaying an unverified number is defect #8's shape, one level down):

```
derived property defined : atp_system.py:402
reads of the property    : atp_system.py:591   (self.phosphate.phosphate_total)
                           -> passed into calculate_j_coupling's DEAD 'phosphate' parameter
repo-wide, excluding params.phosphate_total : NO other read exists
```

**So the derived `phosphate_total` has NO live consumer. It is computed, passed, and dropped.**
And plainly, as instructed: **J-coupling does not read phosphate.** `SUBSTRATE_AUDIT_JUL18`
item 11's causal claim is false, and my defect-1 fix is correct hygiene that fixed **no live bug**
— already stated that way in commit `837a511` and research-log row PO2-2.

**Q1 — clamp is a domain constraint, not the pin's forbidden cap; instrument it. ALREADY
DISCHARGED** before the ruling landed: C3 counts activations and reports them with the
conservation result — **0 activations in every arm, every run.**

**Q2 — pick a defensible pool, pre-register it, and report whether the two differ.**
Implemented metabolic-first as pre-registered; proportional added as a switchable sensitivity arm.
**Answer to the MO's actual question: conservation is IDENTICAL under both** — `+9.745e-15` vs
`+1.035e-14` relative, both under ε=1e-12. That is structural, not empirical: the ledger sums
both pools and the total debited is the same either way; only the split moves.

**One observation I am NOT claiming.** At n=1 the modes differ downstream — structural pool by
0.12%, standing dimer by 2.9% (3.7248e-3 met-first vs 3.8334e-3 proportional). **A single sample
is not a mechanism** — that is precisely the error PO-3 withdrew F-3 over (*"I read a single
sample as confirmation of a mechanism I had not measured"*), and D18 established this system is
near-critical and bistable with large between-seed variance, which is exactly where n=1 misleads.
**Running 3 seeds, backgrounded, persisted incrementally. If it does not resolve, I report the
difference as UNRESOLVED rather than as an effect.**
