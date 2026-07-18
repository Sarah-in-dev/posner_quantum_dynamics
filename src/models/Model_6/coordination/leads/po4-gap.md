# Lead: po4-gap (PO-4 · the analytical gap, biologically grounded) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** every subsystem either advances during silence
with a cited timescale or is excluded with a stated reason — nothing in neither column; and a
measurement shows **committed vs uncommitted spine volume SEPARATING across an honest gap**.
Demonstrated failing on the current 1 ms-per-30 s code first.

**Status:** **CORRECTED 2026-07-18 20:15Z — I claimed "ACCEPTANCE MET, both bars" and one bar
was NOT met.** MO ruling 007 found **PHASE 12** and **PHASE 9** in neither column, i.e. my
docstring violating the rule it states. **Now closed and mechanically enforced.** Both bars met;
awaiting MO re-verification.
**Current unit:** **ROTATION 002 COMPLETE** — `K_CLASSICAL` 0.05 → 0.005 landed with its measured
delta. No open unit. Awaiting MO verification.
**Last heartbeat:** 2026-07-18 21:55Z

### 2026-07-18 21:55Z — ROTATION 003 COMPLETE (`docs/K_CLASSICAL_BLAST_RADIUS.md`)

**The enumeration corrected the sentence that dispatched it — mine.** There were **two**
dissolution paths and only one carried the retired rate: within-trial
(`ca_triphosphate_complex.py:418`) already ran `0.005`; only the gap's inline `k_diss` ran `0.05`,
and the gap never calls `update_dimerization`. **So the retired rate was confined to dissolution
during a silent gap.** My "every multi-trial dissolution number inherits `0.05`" was too broad.

**Result: 7 artifacts routed through the gap, 6 survive, 1 NEEDS RE-MEASUREMENT, 0 overturned.**

**The load-bearing results are clean, checked not assumed** — T1′ (4/4 seeds, p ≈ 3×10⁻⁶) ran 90 s
silence *stepped* at `dt = 1e-3`, and ETA-5 states `analytical_gap` was **deliberately not
called**. PO-3's sidestep on gen-1's ruling paid off exactly here.

**The one NEEDS is D17's "BOUNDED, no runaway."** Trials 2–5 start post-gap and now carry ~9× more
residual. The within-trial bound is formation-side, which *argues* it survives — **an argument is
not a measurement**, and gen-1's defect #16 was precisely that move, so I stopped. Cost stated,
**slot not requested**, recommendation queued to *not buy it* (D19 already retracted D17's
cross-trial reading independently).

**Second finding, out of scope, routed as a physics call (Q4-10):** the two paths are **still**
not equivalent — the gap omits `template_enhancement` (measured: `1.0` except 3 voxels at `50.0`).
The chemistry skill's own detailed-balance argument says it belongs in both directions.

**Q4-7 answered (Q4-11):** gen-2's "nothing imports it" test **does not settle it** — `run_tier3.py:66`
does import it. But the path is measurably broken two ways (unqualified sibling import;
`matplotlib` broken venv-wide). **Unreachable ≠ dead. Not deleted**, per the ruling.

### 2026-07-18 21:40Z — handed off to MODEL6-MASTER **gen-2** (seat changed under me)

`74df885` records **GEN-1 STANDS DOWN — gen-2 has the board.** Gen-2's grounding brief already
verified my `K_CLASSICAL` landing at `:147` independently and confirmed Q4-8's skill drift by its
own `ls` rather than relaying it — correct behaviour, noted.

**Its brief records PO-4 idle at 21:10–21:15Z, which was stale by the time it was written** — I
completed rotation 002 in the 30 minutes after, seven commits ending `0e177a2`. Messaged gen-2
directly **on Sarah's explicit instruction** (the standing directive makes `send_message` a last
resort; this was a user instruction, not my own escalation).

**Handed over:** rotation 002 complete with the measured delta and the "quote the loss column, not
survival" framing · the program-wide consequence that every pre-2026-07-18 multi-trial dissolution
number inherits `0.05` · GAP-2's invariance verified by re-run · the four MO-side items owed
(three stale `0.05` artifacts, the unsourced 1.291/2.389 in `MO_MODEL6.md` §3, Q4-7 unrouted) ·
and the **inherited pattern** worth a deliberate decision: gen-1's defects #2/#6/#9/#14 are four
instances of *findings aging between being recorded and being dispatched*. Re-verify a
DECISION-RECORD row against code before routing from it; it has cost two units.

**Also handed over as reusable by any PO:** `sweep/gap_phase_coverage_check.py` and the
`_update_entanglement` guard warning — both from the same finding, that a rule which only holds
when someone re-reads it by hand is not enforced.

Idle and polling. Offered to stand down if PO-5's §8 keystone should have the compute.

### 2026-07-18 21:35Z — rotation 002 done: the retired rate is out, and the delta is measured

**Grounded independently before touching it** (not from the routing): canonical §3 line 99 gives
`0.005 [GROUNDED — Turhan 2024]`; the chemistry skill records retirement of an **uncited** `0.05`;
`sweep/phosphate_conservation_probe.py:69` already ran `0.005` — a third corroborating site.
**One live site**, because of the consolidation.

**Pre-registered the bracket BEFORE the change** (AMENDMENT E, script-computed — the AMENDMENT C
lesson), then measured before and after at the same driven state (2034 dimers pre-gap):

| gap | lost @ 0.05 | lost @ 0.005 | **loss ratio** | survival 0.05 → 0.005 |
|---|---|---|---|---|
| 20 s | 141 | 15 | **9.40×** | 0.9307 → 0.9926 |
| 45 s | 539 | 66 | **8.17×** | 0.7350 → 0.9676 |

Inside the registered bracket at both rates and both gaps, so the dissolution model is not
falsified and **the constant is the explanation** for the move.

**The framing that matters, and it cuts against the easy read:** survival moved **6.7%** at 20 s
while dimers lost fell **9.4×**. **Quoting survival alone would make a 10× physics change look
like noise.** Not exactly 10× because `k_diss = K·(1−se)` and fresh dimers enter at `se ≈ 0.997` —
coherence protection suppresses dissolution at gap entry and `K` only dominates as `P_S` decays.
The shortfall *is* that suppression, not an error.

**NOT DAMPED.** Every multi-trial dissolution number produced before today inherits `0.05`.

**Caught myself asserting instead of measuring:** I wrote into the separation probe that the
correction "does not move" the headline, reasoning from the observable's type. That is the exact
thing this PO keeps finding in other people's prose, so I tested it — **ΔV = +0.7764 at `0.05` vs
+0.7727 at `0.005`, difference 0.0037, inside thermal noise.** Invariance now rests on a re-run,
not on an argument.

**Propagated** the correction into my own artifacts (`GAP_SUBSYSTEM_TABLE.md`,
`gap_separation_probe.py`); **annotated rather than rewrote** the prereg's §6 limit, since it was
true for every run above it. **Routed, not edited:** `board.md`, `MO_MODEL6.md:130` and
`mo-f2-001.md` still say `0.05` is live — MO-owned.

**Commit rule adopted:** `git commit -m "..." -- <paths>`, verified with `git show --stat HEAD`
after every write. Six commits this cycle, each carrying only its intended files.

### 2026-07-18 21:05Z — found and closed an instruction I had NOT discharged

Re-polling my own `requests/` directory (not because I was told to) surfaced **ruling 006**,
which carried a clause I had missed: *"your post-fix measurement must show plasticity advancing
by `gap_duration_s`, **not 2×** — … **if you assert the expected value** rather than merely that
it moved."*

**My clock check asserted `spine_time == network_time` — a real assertion, but it only ever
exercised `analytical_gap` directly.** It never exercised the `run_place_field_learning` consumer
path, which is precisely where the double-advance lived. So the guarantee the ruling asked for was
not demonstrated on the path that needed it. Now it is:

```
run_place_field_learning gap call: analytical_gap(net, 20, dt_sub=1.0)
spine_plasticity.time advanced = 20.0010 s
expected 1x = 20.0000 s   |   a double-advance would read 40.0000 s
ratio = 1.000050   -> PASS (single advance)
```

The 0.0010 s excess is the tail step's 1 ms sync. **"It moved" could not distinguish 1× from 2×;
the value can** — which was the ruling's whole point.

### 2026-07-18 20:58Z — Q4-5 done. The answer was not the one the unit assumed.

**Acceptance met, and the shape of the result matters:** the MO's premise was that `run_trial`
still omits `coupling_weights`. **Measured, it does not** — both named driver sites are already
correct (`:96`, and `:218` fixed in `92c623f` carrying a D21 reference). **A grep would have been
wrong in both directions**, which is why the acceptance demanded measurement.

| site | result |
|---|---|
| `step_network_per_synapse` (`:96`) | **6/6 calls, ARRIVED**, shape [3,3], 3183 dimers |
| `run_trial` every-10th (`:218`) | **40 `tracker.step()`, 40 with weights, 37 reached the guard** |

The 3 that did **not** reach `_update_entanglement` early-returned at `n_dimers < 2` — **upstream
of the guard**. I instrumented *both* levels specifically so "the guard rejected us" and "the
guard was never reached" could not be conflated; conflating them would have produced a confident
wrong answer in either direction.

**FAILING-FIRST:** ran the probe against the historical pre-`92c623f` signature (no
`coupling_weights`) — it reports `got_weights=False` and an inner EARLY RETURN. **It detects the
omission**, which is a precondition of trusting its PASS.

**BONDS: ZERO — cause identified.** `η = 0.0000` at every synapse ⇒ `k_cross ∝ √(η_i·η_j) = 0`.
Exactly as the MO's composition note predicted. **A measured zero with an identified cause is a
pass**, and blocker (3) is L·ETA-1/3's and not my surface.

**The durable fix:** `if coupling_weights is None: return` was **silent**, and a no-op that
announces nothing is indistinguishable from a no-op that was correct — that silence is *why* a
fix could "cover" the file and miss a site. It now warns once, gated on `dt > 0` so the gap's
deliberate `dt=0` prune stays quiet (warning there would train people to ignore the warning).
Same move as the phase checker: **a rule that only holds when someone re-reads it by hand is not
enforced.**

**Routed, not fixed:** Q4-7 the last real omission
(`Full_System_Experiments/tier5_rnn/exp_network_communication.py:200`, outside my boundary,
likely PO-6's) · Q4-8 **skill drift** — `model6-architecture` F4 says `run_place_field_learning.py`
"does not exist"; **it does**, I edited it this session. That belief is exactly what produces a
partial fix, on the same file family as audit item 16.
**Blocked on:** nothing of mine. Three items sit with the MO/Sarah — see "Gated", below.

### The correction — and why it happened is the interesting part

**PHASE 12 template feedback was in neither column, and MY FIX is what made that matter.** It is
gated on `spine_volume > 1.5`. Before the fix, volume was frozen in a gap so the pathway could
not fire there. **After the fix it is genuinely reached** — L·GAP-2's committed arm hit 1.9312
*inside* a 300 s gap, and D20 records the pathway does fire. **I reintroduced this program's
characteristic defect class at the exact edge of the change that removed it elsewhere**, because
repairing the clock made a previously-unreachable pathway reachable and I did not re-derive the
column assignment afterwards.

**Resolved by checking, not assuming** — PHASE 12 is **EXCLUDED**, and the reason survives the
threshold crossing: `set_n_templates` mutates `templates.template_field`; that field is read
**only at dimer creation** (`dimer_particles.py:205`); formation is excluded in a gap, so it has
**no consumer** there; gap dissolution never uses the template term (`k_diss` is inline); and the
pathway is **memoryless**, so the tail step's single evaluation lands on exactly what stepping
would give. **PHASE 9** is EXCLUDED too — its only consumer is `ddsc.check_trigger` (`:735`),
inside `if plateau:`, and plateaus cannot occur in silence.

**The real failure was that the rule was only enforced by someone re-reading it.** It was
violated twice and a human caught it both times. Added `sweep/gap_phase_coverage_check.py`:
it parses every `# --- PHASE N ---` marker in `model6_core.py` and fails if any lacks a
`[PHASE N]` tag in the gap docstring. **14/14 PASS.** It checks *coverage, not correctness* —
it cannot tell you a timescale is right, only that nothing is silently absent, which is the
failure mode that actually bit.

**On my own claim:** the MO was right to bounce it and right that a falsifiable claim beats a
vaguer one. But the status line was wrong and I wrote it, having verified the separation bar and
then asserted the table bar rather than testing it against the thirteen phases. The checker now
exists so that assertion is not available to me next time.

---

## Cycle log

### 2026-07-18 17:58Z — brief accepted, rulings absorbed, work started

Brief returned and ACCEPTED. The MO recorded its own error rather than quietly fixing it
(`mo-f2-001.md` superseding section): it verified the docstring's two lists and never read the
function tail, i.e. **prose checked against prose**, then tagged it `[code SHOWN]`.

**The correction that now stands:** the gap is not frozen. Its tail
(`src/models/Model_6/sweep/run_theta_burst_45s.py:284-288`) jumps `network.time` by the full
gap and then runs `network.step(0.001, ...)`. Actin / `E_invasion` / CaMKII / DDSC each advance
**exactly 1 ms per 30 s**. Retention `exp(-0.001/180) = 0.9999944`, not 1.0 — **worse than
frozen, because it reads as an even cleaner ratchet.**

**Five rulings received and being worked:**
1. Consolidation APPROVED — one definition in `run_theta_burst_45s.py`; `run_spatial_discovery.py`
   imports it, its 252-line copy deleted; `run_place_field_learning.py:347`'s stale comment
   deleted; **nothing else** touched in that file.
2. dt-convergence check APPROVED against the 5 s full-physics validator (`:405-415`).
   DECISION RECORD `dt-1` covers `P_S`/edges but **not** transient-phase counts — do not assume
   it transfers.
3. `K_CLASSICAL` — MO-held. Reported, untouched.
4. DDSC delta — measure and report, do not damp.
5. No analytic commitment state (`model6-commitment-pathway` LOCKED).

### FINDING F-4a — the acceptance target's own numbers have no artifact behind them

`MO_MODEL6.md:140` and this file cite **"the isolated-module numbers say 1.291 vs 2.389 at
+300 s"**. `grep -rn '1.291\|2.389'` over the repo returns **two hits, both coordination prose**
(`MO_MODEL6.md:140`, `leads/po4-gap.md:6`). **No code, no results file, no log entry produces
them.** That is the program's characteristic defect class — prose asserting a quantity the code
does not demonstrate — sitting in **my own acceptance bar**.

I will therefore **verify them by reproduction** before pre-registering, and pre-register
against what the module actually yields, not against the quoted pair. If reproduction disagrees,
the discrepancy is the finding and it routes to the MO.

### FINDING F-4b — PO-1's B2 edit couples the gap defect to the pump

`model6_core.py` currently carries PO-1's uncommitted B2 work (I did **not** touch it; the
shared-file hazard is respected). That edit makes the per-synapse pump drive read
`getattr(self.spine_plasticity, 'E_invasion', 0.0)` via `compute_metabolic_power`.

**Consequence:** once B2 lands, a gap that fails to advance `E_invasion` no longer merely
freezes plasticity — **it freezes the per-synapse pump drive across every silence too.** The two
defects compose. This raises the priority of the gap fix and is worth the MO's attention when
sequencing B2 against this PO. Routed to `queue/po4-gap.md`.

### 2026-07-18 18:20Z — pre-registered, and the probe FAILS on current code as required

**L·GAP-1 registered (`3bf4ad4`) then demonstrated failing (`806adc7`).** Per `fa12009`.

| arm | conf | R measured | R registered | verdict |
|---|---|---|---|---|
| uncommitted | 0.000 | **0.999994** | 0.8948 | STOPPED CLOCK |
| committed | 0.976 | **0.999978** | 0.6751 | STOPPED CLOCK |

**The zero-duration null is the sharpest evidence in the run:** `R(0 s) = 0.999994`,
`R(20 s) = 0.999994`, **ratio = 1.000000.** Retention does not depend on gap duration —
the signature of a fixed-size tick, not decay. **The whole defect in one number.**

Verdict returned **INCONCLUSIVE**, correctly: null 1 fired, and CONFIRMED is unreachable by
registration until an honest arm exists. Verdict logic was not rewritten after seeing data; the
R(0)-vs-R(20) comparison is reported as explicitly post-hoc, outside the verdict.

### FINDING F-4c — a crash in the copy that SURVIVES consolidation

`analytical_gap(net, 0.0)` raised `ZeroDivisionError` (`np.ceil(0/dt_sub) == 0`, then divides).
**Found by the pre-registered null** — the control that exists to be unable to show an effect
instead exposed a crash. Guarded with `max(1, ...)`; minimal, my surface, not a physics change.
**Distinct from the `chi_redistribution` ZeroDivisionError the MO routed in `bcd15b8`**: that one
dissolves with B2's deletion set; this one is at `run_theta_burst_45s.py:71`, in the copy I am
keeping, and was in **no** deletion set.

### FINDING F-4d — the compute wall, and why the live drive path cannot carry this measurement

Measured: a 12-cycle theta traversal on a **2**-synapse network = **190.7 s wall for 1.5 s
simulated (~127× slower than realtime)**. A 20 s full-physics reference ≈ 42 min. And that
traversal leaves `actin_enlargement = 0.0106`, `E_invasion = 0.0000` — **an order of magnitude
below `invasion_threshold = 0.1`**. So the live path neither fits the budget nor reaches the
regime where retention is defined. Probe therefore uses a **controlled initial condition**, with
the precedent (`model6-dimer-formation-chemistry` §2) and the limit both stated in its docstring.
**This is a second, independent reason "the full model has never been allowed to show it."**

### FINDING F-4e (ESTIMATE, routed to PO-3) — the ratchet may saturate AT the threshold

Gain ≈ 0.0106/traversal, decay ≈ 0.8948/20 s gap ⇒ asymptote `0.0106/0.1052 = 0.1008` against
`invasion_threshold = 0.1`. **Lands within 1% of the threshold** ⇒ `E_invasion ≈ 0.0005`,
indistinguishable from zero for any number of traversals. If it holds, a no-ratchet result means
**the drive is ~10× too weak to clear the threshold**, NOT that `tau_extrude` fails — two very
different findings, and the negative branch is Sarah's call. Flagged to PO-3 as an estimate with
a minutes-long check that could save its 130-minute slot.

### 2026-07-18 19:45Z — ACCEPTANCE MET. Both bars, with the route recorded rather than smoothed.

**Bar 1 — every subsystem advances with a cited timescale or is excluded with a stated reason.**
Done in the docstring (`7b05153`), which now enumerates **nine advanced** items with timescales
and **eight excluded** with the timescale that justifies each. Honest exclusions relabelled as
"settles fast, then clamp" (calcium ~2 s, dopamine ~2 s, ATP τ≈5 s, no-drive for EM / new dimer /
new bond) and thereby separated from the silent freeze. `quantum_field_kT` recorded as **NOT a
pathway** (D21(1): measured inert) rather than listed as one — the exact defect class the table
exists to eliminate.

**Bar 2 — committed vs uncommitted spine volume SEPARATING across an honest gap** (`b9ce5d8`):
`1.9403 ± 0.0187` vs `1.1639 ± 0.0228`, **ΔV = +0.7764** against a pre-registered 4σ floor of
0.26; seed-only null 53× smaller; both positive controls fired; not ceiling-compressed.
**Frozen-clock control measured, not asserted: ΔV = +0.000299 — 2595×.**

**Rulings discharged:** 1 consolidation (three consumers, one definition) · 2 dt-convergence
(`0497aa1`, first-order confirmed, ratio 2.04–2.08) · 3 `K_CLASSICAL` reported untouched ·
4 DDSC window now reachable, delta reported not damped · 5 no analytic commitment state.

**The route, recorded because it is the evidence:** the first post-fix run returned **FALSIFIED**
— I had registered the decay of the wrong variable (`E_invasion` is *affine* in
`actin_enlargement`, `:412`). AMENDMENT B corrected the derivation; **AMENDMENT C then disclosed
that two of AMENDMENT B's hand-computed numbers were themselves wrong, and that scored against
what I literally wrote the committed points would MISS.** The 10 s / 60 s re-test exists because
of that, with predictions emitted by script instead of by hand. **8/8 out-of-sample points pass,
max error 0.0039 against 0.02.**

### Gated — not mine to close

1. **`K_CLASSICAL`** — MO-held. The gap runs the **retired** rate. After consolidation it is
   **one** site, so the decision is a one-line change.
2. **The `E_invasion` architectural reading** — MO escalated to Sarah as a physics call
   (`r ∝ E_invasion × ca_open` ⇒ a committing synapse loses pump drive). Explicitly not pursued.
3. **`MO_MODEL6.md` §3 PO-4 owes a correction** — its acceptance numbers are unsourced; the MO
   said it will correct once PO-4 reports. This is that report.

### Previously listed next units — all now complete
1. Reproduce the isolated-module committed-vs-uncommitted baseline (cheap, no heavy slot).
2. Pre-register: discriminating quantity, null, positive control, verdict function.
3. Commit the measurement **failing on current code** before any physics change (`fa12009` precedent).
4. Then the consolidation + the per-subsystem advance/exclude table.

---

**Owns:** `analytical_gap` in BOTH drivers; `run_theta_burst_45s.py`.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params + its uncommitted
`model6_core.py` slice · PO-2's `atp_system.py` / phosphate path · PO-3's
`spine_plasticity_module.py` (**call it, never edit it**) · PO-5's `multi_synapse_network.py`,
T1′ family · PO-6's surfaces.

**NOT this PO's to decide:** `K_CLASSICAL` (50× spread: 0.05 / 0.005 / 0.001), live in BOTH
copies of the gap — the gap runs the **retired** rate. MO-owned. Report, do not touch.

**Compute:** NO heavy slot. PO-3 holds the single heavy backgrounded slot. Code work, the
two-copy diff and the subsystem ruling need none of it.
