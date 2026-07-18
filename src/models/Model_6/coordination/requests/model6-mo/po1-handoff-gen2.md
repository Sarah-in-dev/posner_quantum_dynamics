# PO-1 / PO-6a → MO gen-2 · state handoff · 2026-07-18

**You inherit nothing, and you already demonstrated you verify rather than inherit — so this
is written as pointers you can check, not claims you should take.** Every item below carries
the command that proves or refutes it. If any disagrees with the code, the code wins and I
want to know.

**Who I am:** started as PO-1 (B2, the per-synapse pump), rotated to **PO-6a** (sweep-harness
validity) at gen-1's rotation-001. My lead file is `leads/po1-b2.md` — it still carries the
PO-1 name because rotating the filename would have broken your predecessor's pointers.

---

## 1. What has landed — verify with `git log --oneline`

| commit | what |
|---|---|
| `fa12009` | B2 acceptance probe, **committed FAILING first** |
| `c280e85` | B2 — per-synapse pump retired; both sites now one 8 MHz mode |
| `1f75582` | ruling 005 §2 — D/φ/χ do no physics at either pump site |
| `9b4819f` `dbe9548` | dimension-consumer audit; 9 of 19 dimensions INERT |
| `427b47c` | INERT registry + `sweep_runner` warning + results-JSON stamping |
| `de8e0df` | effect-test stage (and the blind-observable it caught) |
| `3632fce` | ruling 006 — `T_singlet_dimer` 500 → **216 s**, de-duplicated |
| `adbf37a` `56c0960` | SWEEP-3 log row, PO-5 disclosure, heartbeat |

**Re-run the acceptance measurements yourself — both are seconds, not minutes:**
```
python src/models/Model_6/sweep/pump_mode_agreement_probe.py    # B2: expect PASS, C1+C2 FIRED
python src/models/Model_6/sweep/coherence_radius_probe.py       # T1' floor: expect 7/7 CONFIRMED
python src/models/Model_6/sweep/dimension_consumer_audit.py     # expect 8 INERT, controls A/B/C PASS
```
The audit is the slow one (~3 min; it drives the real model four times for effect tests).

**Current state, read from the registry rather than from me:**
```
python -c "import sys;sys.path.insert(0,'.');sys.path.insert(0,'src/models/Model_6');
from sweep.quantum_dimensions import ALL_DIMENSIONS,LIVE_DIMENSIONS,INERT_DIMENSIONS as I;
print(len(ALL_DIMENSIONS),len(LIVE_DIMENSIONS),len(I));print(sorted(I))"
```
→ `22 14 8` and the eight names.

---

## 2. Open with Sarah/you — `queue/po1-b2.md`, Q1–Q9

The ones that still need a decision, shortest first:

- **Q9 · `q2_k_agg_baseline` cannot be fixed mechanically.** `hasattr(dimerization,'k_agg')`
  is False so the write never fires — but the real attribute `k_base = 18918.67` while the
  dimension's values are `[0.001…0.05]`, matching `k_classical = 0.005` exactly. The values
  were written for a **dissolution** rate. **"Fixing the guard" would inject values ~10⁶ off
  and produce a curve that looks like physics.** Delete the dimension or re-declare its
  values; do not just fix the guard.
- **Q7 · the remaining 8 INERT.** Four are the Q1 backbone ones B2 already proved do no
  physics anywhere — those are arguably deletions, not fixes. `stim_ca_amplitude` and
  `stim_burst_duration_ms` assert mechanisms the code explicitly disclaims (see §4).
- **Q6 · η's large-D validity.** Gen-1 escalated to Sarah and told me not to pursue it.
  **It is still open and still qualifies every η the program reports** — see §3.
- **Q1/Q2 · B2's drive change and the flat-η re-read.** Gen-1 recorded that it endorsed
  CONFIRM on the drive change and that it remains Sarah's veto. **I do not know whether Sarah
  ever ruled.** Worth checking rather than assuming closure.

---

## 3. Stated limits that must survive this handoff

**These are the items most likely to be lost, because they qualify results that otherwise
read as clean.** Each is recorded in code and log, but a successor who quotes the result
without the limit will overstate it:

1. **η's large-D validity is UNVERIFIED.** `η = (r−1)/(r+1)` is the large-D limit
   (Wang/Wang 2022); the pin wants D ≳ 200, the sites run 20 and 50. D does not enter the
   formula, so this changes no number — it bears on whether the large-D *form* applies at
   all. Finite-D corrections were never derived. **This qualifies every η both pump sites
   report.** Do not close it by raising D — that is tuning a constant to reach an outcome.
2. **B2's mode agreement proves the two sites AGREE, not that 8 MHz is right.** If the
   May-30 bet (Q ≳ 10, Pokorný slip-layer vs Foster/Baish) is wrong, both sites are now wrong
   *together* and the probe still reports PASS.
3. **INERT is definitive only under the stated driving conditions.** A consumer on a branch
   the probe never reaches would look identical to no consumer. Each verdict is
   INERT-under-stated-conditions.
4. **REACHED ≠ effective.** A read can be a log line. Only the four dimensions in the audit's
   `EFFECT_CASES` have a demonstrated downstream effect.
5. **`q2_t2_p31`'s demonstrated effect is small by construction** — `T_singlet` is a ~100 s
   time constant, the probe runs 40 ms. It shows the parameter reaches the physics with the
   right sign, **not** the magnitude a real sweep would see.

---

## 4. Traps — things that will cost you time if you meet them cold

- **`git commit -m "..." -- <paths>` cannot see an untracked file.** New files still need
  `git add <file>` first; the pathspec constrains everything else. My `adbf37a` hit this.
- **The two de-duplicated coherence sites hold DIFFERENT param objects.**
  `dimer_particles` has `Model6Parameters` (needs `.quantum`); `quantum_coherence` has
  `QuantumParameters` directly. Getting it wrong raises rather than silently misbehaves, but
  it will cost a minute.
- **`model6-architecture` says six orphan modules. That list is wrong.** Only
  `eligibility_trace` and `singlet_dynamics` have no importer. **`calcium_system` is imported
  at `analytical_calcium_system.py:535`, inside a live `use_analytical=False` fallback** —
  deleting it on the board's say-so breaks a live module. `implicit_diffusion` is its
  dependency. AST-level proof, not grep, is what separates these.
- **`stim_ca_amplitude` is not a wiring bug you can fix.** The scenario's own docstring
  (`theta_burst_scenario.py:_run_epoch`) says *"Calcium enters via voltage-gated channel
  physics, not direct injection"* while the dimension claims *"(direct injection)"*. Wiring it
  means adding an injection mechanism — that is physics, not plumbing.
- **`216 s` is load-bearing and must not be retuned.** It is what makes
  `quantum-system-canonical` §2.2's correspondence hold (Werner crossing at 107 s, inside the
  100–200 s band). Changing it to improve a downstream result is the emergent-physics
  violation.

---

## 5. Held, not forgotten

**Orphan deletions (`eligibility_trace`, `singlet_dynamics`) are HELD** behind the isotope
kill-switch question, per rotation-002 and ruling 006. Both files still exist — verify with
`ls src/models/Model_6/{eligibility_trace,singlet_dynamics}.py`.

My finding on the constraint: the isotope kill-switch does **not** depend on
`eligibility_trace` — the live control is the continuous `environment.fraction_P31`
(`model6_parameters.py:896` → `dimer_particles.py:292,301`, `model6_core.py:297`), which is
strictly more general than `eligibility_trace`'s two-value enum. **The hold is yours to lift
or keep; I have not acted on it.** Before deletion, `eligibility_trace`'s isotope T2 figures
(P31 ~68 s, P32 ~0.3 s — which differ from the live 216/0.4) are worth logging as provenance.

---

## 6. My own errors, so you do not inherit a flattering picture

- **Two false statements I wrote and then caught in my own diff** (`1f75582`): that χ was
  "kept because the steady-state solution needs a nonlinear term" (B2 had deleted the
  quadratic) and that D/χ "survive as slope parameters" (nothing consumes them). The
  program's characteristic defect, produced by me, twice.
- **A blind observable that nearly produced a false null** (`de8e0df`): my first effect test
  used one global fingerprint and reported "NO EFFECT" for two *live* Q1 dimensions. Caught
  only because a 10× change gave byte-identical output. Observables are now per-dimension and
  a non-moving one reports UNDEMONSTRATED, never INERT.
- **`d95e826` contains PO-2's `phosphate_ledger_probe.py`.** Gen-1 diagnosed this as the
  shared git index rather than my error, and changed the commit rule. Recording it because
  PO-2's "committed failing first" provenance runs through a commit of mine with an unrelated
  message — disclosed at `requests/po2-phosphate/po1-6a-002.md`.

---

## 7. What I would pick up next, if you want a recommendation

**Nothing on my surface is blocked, but everything left needs a decision rather than code**
(Q7/Q9). The one genuinely unblocked mechanical unit is the **~151 dead parameter fields**
from Unit 2, which I have not started. If you would rather I take something else, say so —
I have no attachment to the sweep surface.

**Open request to PO-2 still outstanding:** `requests/po2-phosphate/po1-6a-001.md` asks for
one line at `model6_core.py:84` (an `em_coupling_module` import that is never instantiated)
whenever PO-2 releases the file. Low priority, no urgency attached.

— PO-1 / PO-6a
