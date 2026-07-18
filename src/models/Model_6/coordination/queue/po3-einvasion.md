# Queue: po3-einvasion — actions awaiting Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready,
so clearing it is seconds per item, not a re-derivation.

---

## Q1 · 2026-07-18 · L·ETA-3's `ca_open` attribution was measured through a harness that under-delivers glutamate ~100×

**The exact ask:** decide whether L·ETA-3's DECISION RECORD row needs a correction banner.
I am not touching it — the research logs are "each PO writes its OWN entries; nobody
rewrites another's", and L·ETA-3's verdict is the thing this PO was dispatched to test.

**Why.** `sweep/eta_in_live_trial.py:138-144` — the L·ETA-3 harness — steps presynaptic
release **once per agent step** (0.5 s) and then runs 100 physics steps against that one
stale stimulus. The shipped reference, `run_spatial_discovery.py:434-441` (`run_trial`),
steps it **inside** the physics loop, once per 0.005 s.
`PresynapticRelease.step` (`presynaptic_release.py:110-139`) is a per-timestep Bernoulli
draw (`p_spike = 1 - exp(-rate*dt)`), so the 0.5 s call interval removes ~99% of release
opportunities.

**The evidence, measured not inferred:**

| | release opportunities / 14 s traversal | expected release events |
|---|---|---|
| shipped `run_trial` | 2800 | ~350 |
| L·ETA-3 harness | 28 | **~3.3** |

- My probe, while still inheriting the L·ETA-3 pattern, recorded **`max_glu = 0.0000` at
  the target synapse across an entire traversal** at `max_act = 0.9950`.
- After stepping release per physics step (the shipped pattern), same seed, same geometry:
  `max_glu = 1.0000`, and **`peak_r` at traversal 2 rose 0.0571 → 0.1428, a 2.5× increase**
  from the release fix alone.

**Why it matters.** L·ETA-3 attributed its 13× shortfall to both factors of
`r ∝ E_invasion × ca_open` "roughly multiplicatively", with `ca_open = 0.140` vs the rig's
0.38. `ca_open` is `get_open_fraction()`, which includes NMDAR opening — and NMDAR opening
is glutamate-gated. So the `ca_open` half of that attribution was measured with the
glutamate contingency **substantially unsatisfied**. That is the ERR-2 failure class in a
new location: ERR-2's own words are *"the term setting `r` was measured with its glutamate
contingency unsatisfied."*

**What survives regardless:** that `r` fell 13× short in that trial, the zero cross-synapse
edges, and the `E_invasion` trace mechanism (E_invasion is driven by calcium through
`f_CaM`, and VGCC still conducted). **What is in question:** the *split* of the shortfall
between the two factors, and therefore how much of "the constraint is DWELL and
CO-ACTIVATION" rests on a measurement artifact.

**My recommendation:** a correction banner on ETA-3 in the same style as ERR-2 — narrowing
the `ca_open` attribution to a lower bound rather than retracting the row. The headline
("η does not clear in a live trial") is very likely to survive: my own corrected-drive
smoke still shows `r ≈ 0.14` against a threshold of 1.0. I recommend **not** re-running
L·ETA-3 for this; my L·ETA-5 run measures the corrected drive regime at 8 traversals and
will report absolute `r` under it, which answers the question as a side effect.

**Status:** escalated to the MO the same turn. Not blocking my run — my probe is corrected
and the deviation is recorded as AMENDMENT A1.1 in the pre-registration, committed before
the run.

---

## Q2 · 2026-07-18 · `model6-actin-invasion-driver` §5 understates E_ref's provenance — route the skill edit

**The exact ask:** decide who updates the skill, and approve the one-line factual addition.
I have not edited it — skills are the decision layer and the MO holds decisions; a PO
silently editing a shared skill is not a move I should make unilaterally.

**Why.** `model6-actin-invasion-driver` §5 says E_ref was *"read once off a 3000 s
uncommitted run"* and gives **no pointer to the run**. The 2026-07-18 substrate audit then
recorded E_ref as UNVERIFIED — *"no artifact ties it to a run"* — and my kickoff inherited
that as fact. I repeated it in my own grounding brief without checking.

**It is wrong.** The generating run is in the repo and reproduces:
`tests/check_actin_three_pool.py:142-157` (Phase 5, "SUSTAINED UNCOMMITTED, 3000 s,
Ca=2.0 uM, drive=0"), printed at `:286-288` as *"Candidate physical anchor for E_ref
(decision pending)"*. Re-run 2026-07-18: **`+3000s enl = 1.8742`** against the coded
`E_ref = 1.87`.

**My recommendation:** add the path to §5's E_ref row and change UNVERIFIED to
**REPRODUCIBLE, SELF-REFERENTIAL** — reproducible from a named in-repo run, but the model's
own asymptote and *not* a literature measurement, so it must never be described as one.
Evidence and the full verdict are in `docs/PROVENANCE_EINVASION_CONSTANTS.md` §1.

**Why it is worth a queue slot rather than being left:** the missing pointer is what let a
reproducible constant be recorded as unverifiable, and that record was then used to argue the
13× shortfall might not be readable as physics. One path reference closes that loop
permanently. It is also the cheapest instance of this program's signature defect — prose
drifting from the code — in a skill that is otherwise carefully grounded.

---

## Q1-CORRECTION · 2026-07-18 · **I OVERSTATED F-3. Withdraw the recommendation in Q1.**

**Read this before acting on Q1 above. Q1's direction survives; its magnitude and its
central characterization do not, and its recommendation is WITHDRAWN.**

### What I claimed in Q1, and what is actually true

I claimed the L·ETA-3 harness "under-delivers glutamate ~100×" and "starves the NMDARs",
with a table giving ~350 release events per traversal for the shipped pattern against ~3.3
for L·ETA-3. **I computed those from `rate × time` and never measured them.** They ignore the
10 ms absolute refractory (`TAU_REF`) and vesicle depletion (`Ps`, `n`), both of which
dominate at 25 Hz. Measured directly, 20 seeds, `act = 0.995`, 14 s traversal:

| | release EVENTS / traversal | physics-steps WITH glutamate present |
|---|---|---|
| shipped (`run_trial`, per physics step) | **19.0** | **19** (1 step per event) |
| L·ETA-3 (`eta_in_live_trial`, per agent step) | **1.0** | **100** (each event held 100 steps) |

**Two corrections, and the second one inverts my claim:**

1. The event-count ratio is **19×, not ~100×.** My ~350 and ~3.3 were both wrong.
2. **The L·ETA-3 pattern holds each release for 100 physics steps.** So it delivers fewer
   discrete events but keeps glutamate *present* for ~100 physics-steps per traversal against
   the shipped 19. **On exposure duration it delivers MORE glutamate, not less.** "Starves the
   NMDARs" is therefore not established and is probably backwards.

My one piece of direct evidence — `max_glu = 0.0000` across a full traversal under the old
pattern — is consistent with a Poisson mean of ~1 event (P(0) ≈ 37%), not with 100× starvation.
I read a single sample as confirmation of a mechanism I had not measured.

### What survives

- The two call sites **do** differ, and my probe matching the shipped reference is still
  correct — a probe should not diverge from the implementation it derives from.
- `peak_r` at traversal 2 did move 0.0571 → 0.1428 when I changed the pattern. That is real
  and measured. **But I can no longer attribute it to "more glutamate";** the temporal
  structure changed (many brief events vs one sustained event), and which of those the NMDAR
  gate responds to is UNMEASURED.

### What is withdrawn

**The recommendation that L·ETA-3 carry an ERR-2-style correction banner is WITHDRAWN.** I
cannot support the claim that its `ca_open = 0.140` was measured with glutamate substantially
unsatisfied. Its `ca_open` attribution stands unchallenged by me. **No action is requested of
Sarah on L·ETA-3.** I am glad I routed this as a recommendation rather than editing another
PO's log row directly.

### Why I am recording this rather than quietly amending

I escalated a quantitative claim to the MO, queued it for Sarah, signalled it to PO-5, and
built a pre-registration amendment on it — all from arithmetic I never ran, in a program whose
signature defect is prose asserting what the code does not do. It is the same error I
documented in others twice today (the MO's docstring read, my own `analytical_gap` claim).
The correction costs less than the false record would.

**AMENDMENT A1.1 is NOT withdrawn** — matching the shipped reference is still right — but its
stated *rationale* ("removes ~99% of release opportunities", "starving the NMDARs") is
corrected here, and the run's results are unaffected either way since the whole run used the
corrected pattern.
