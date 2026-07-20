# PO-7 handoff — evening of 2026-07-19

**Read this first in the morning.** Branch `claude/nervous-hertz-7ccff6`, worktree
`.claude/worktrees/po5-keystone`.

> ## ⚠ LATE ADDENDUM (2026-07-20) — THREE THINGS BELOW ARE SUPERSEDED. Read this box first.
>
> The body of this handoff (§1–§12) was written before a reviewer round and a framing correction.
> The findings still stand, but three of its framings are now wrong. The authoritative current
> statement is `docs/PO7_TECHNICAL_BRIEF_2026-07-20.md` and research-log entries after `L·PO7-3`.
>
> 1. **"Frustration = the H¹ obstruction" (appears in §1, §5→R5, and `L·PO7-3`) is RETRACTED.**
>    Resource contention among binary spin slots is a **matching deficiency** (Hall's condition;
>    under CKW, LP-infeasibility over a capacity polytope) — **not cohomology.** The genuine ℂ¹⁶
>    sheaf answers a *different* question (spin-state consistency).
>
> 2. **"The rig is not reproducible, being fixed" (§9) is the WRONG FRAME.** The system is
>    stochastic by construction; its output is a *distribution*, not a value, and whether the
>    condensate ignites at all varies across free draws. That is the physics, not a defect.
>    Seeding it to get a clean or an igniting run would be **selecting the outcome** — we do not.
>    A seeding capability was added (commit `57ccd75`) for narrow *software-regression* use only
>    and is left dormant. **P(ignition) is now a first-class finding**; a free-running ensemble
>    (no seeds) is measuring it.
>
> 3. **The starvation / update-order question (§8, §12) is largely RESOLVED, not open.** The
>    monogamy bound (4) cannot prevent percolation because the graph percolates at **mean degree
>    ≈ 1** (Erdős–Rényi). The measured "starvation" was a **run-length artifact**: the physical
>    bond-release rate is `k = 1/T₂ + 1/τ_dimer = 1/216 + 1/200 = 9.63e-3/s` (τ ≈ 104 s), **96×
>    faster** than the model's ~1e-4/s, and our 20 s runs are ~5× too short for slots to recycle.
>    The decisive remaining question is **whether the percolating bridges sit near the Werner
>    floor** (F≈0.5, negligible entanglement ⇒ connectivity is the wrong invariant and the blob is
>    an artifact) — being measured now.
>
> Units 13–16 (order-sensitivity, stoichiometry, freed-budget, bridge-fidelity) and the reviewer
> exchange all postdate the body below. The body is kept intact as the record of how the picture
> was reached.

---

## 1. THE HEADLINE

**The missing physics was the SPIN REPRESENTATION, and supplying it breaks the blob.**

Dimers were bonding as featureless nodes. A Ca₆(PO₄)₄ carries **four ³¹P spin-½ nuclei**; a
singlet-strength bond consumes **one at each end**; monogamy forbids a spin mediating two bonds.
Nothing in the model represented that, so the graph carried **mean degree 715 against a hard
bound of 4** — **99.44% of edges physically inadmissible.**

Spin-resolved bonding (opt-in, OFF ⇒ bit-identical), single synapse, fingerprint rig:

| | OFF | ON |
|---|---|---|
| edges | 369,740 | **2,031** (0.55%) |
| mean degree | 715.16 | **3.93** |
| max degree | 902 | **4** |
| dimers over bound | 1034 (100%) | **0** |
| components | 1 | **184** |
| **largest_frac** | **1.000** | **0.112** |
| frustrated bonds | — | **491,566** |

First physically admissible entanglement graph in the investigation, and it carries non-trivial
partition structure. The 491,566 refusals are **frustration** — pairs individually satisfiable,
jointly not — i.e. the **H¹ obstruction** the advisor said is unobtainable any other way, and
which the Unit-5 "sheaf" structurally could not express (it decomposes into 3 ordinary graph
Laplacians; cross-block edges 0/369,740, verified).

## 2. AT MULTI-SYNAPSE SCALE — a real window, then collapse

7 synapses @1 µm, 20 s, spin resolution ON for **intra** bonds only:

```
t=1–9    xbond=0     comps 109→310  largest_frac 0.007–0.013  nmulti=0
t=10.05  xbond=128   comps=210      largest_frac=0.209        nmulti=30
t=11.05  xbond=121   comps=229      largest_frac=0.185        nmulti=31
t=12.05  xbond=879   comps=19       largest_frac=0.959        nmulti=1
t=20.00  xbond=9256  comps=1        largest_frac=1.000        nmulti=1
```

**49 samples show `largest_frac` 0.092–0.296 with up to 33 components spanning ≥2 synapses** —
the first cross-synapse structure here that is neither empty nor a blob. Then it collapses,
because **cross-synapse bonds are not spin-accounted** and accumulate without limit.

**⇒ The shared ledger is required, established by measurement.** A dimer's four nuclei must be
spent across intra AND cross bonds together. **A subagent was building this when the session
ended — check its status and its commit before doing anything else.**

## 3. TWO STALE FIGURES NOW SUPERSEDED

- **"The pump is dead, r ≈ 0.077."** Never measured; it propagated from the PO-7 kickoff. Direct
  measurement gives r = 0.0390 = **L·ETA-1's rest floor**, not a ceiling. Re-running L·ETA-2's
  rig with glutamate wired: `E_invasion` **0.3508 vs 0.3518 (0.28%)**, peak r **2.53**, **7/7
  synapses condensed. The pump ignites.** (The archived `eta_probe.py` still drives voltage-only
  — the ERR-2 defect — so re-running it reproduces the silent-NMDAR artifact.)
- **L·ETA-1's "rowsum > 6.89; unreachable at ≥2 µm at any N."** Ignition observed at row-sums
  **4.14–5.05**. That table was computed with NMDARs silent (`ca_open` ≈ 0.05); with glutamate
  wired `ca_open` reaches 0.34–0.70. **Re-derive before citing.**

Also measured: condensation **strobes rather than latches** (`n_cond` flicks 0 → 3 → 7 within
0.15 s on stochastic calcium), and **98 cross-bonds — already past Werner F>0.5 — take
`largest_frac` from 0.22 to 1.000**, because each synapse was a near-complete clique.

## 4. ⚠ WHAT WAS WITHDRAWN THIS SESSION (my error)

**The cross-synapse provenance premise is unphysical.** Phosphate does not travel between
synapses; cross-synapse entanglement is mediated by the condensate backbone
(`k_cross = 0.5·√(η_i·η_j)·w_spatial·P_product`, `w_spatial = exp(−d/5 µm)` = the **condensate**
coupling length — `model6-entanglement-partition-werner` §2).

Grounding failure: I read §1 of that skill (*where* the partition lives) and treated it as
licensing a mechanism only §2 describes. §2 was never read. The kickoff's premise was inherited
and validated *as implemented* without checking it against the mechanism.

The tell I recorded and misread: cross edges only formed at a forced 0.2 µm spacing, and I filed
"excludes the upper half of the physiological range" as a limitation of the *result* rather than
evidence the *mechanism* was wrong.

**Withdrawn:** the network event pool as a physical claim; the "2 µm landmine" (it is CORRECT
physics); the Unit-2 keystone verdict; the Unit-7 claim-radius derivation; Unit 6 (killed);
the R4 advisor packet (marked DO NOT SEND). Full record: `L·PO7-2`.

**Nothing in §1–§3 above depends on it** — the spin work is local and unaffected.

## 5. STATE ON DISK

**Log:** `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` — `L·PO7-1` (build, later partly withdrawn),
`L·PO7-2` (the correction), `L·PO7-3` (the spin build). Append-only; nothing was rewritten.

**Advisor:** `PO7_ADVISOR_PACKET_R5_2026-07-19.md` — ready to send. R4 is marked DO NOT SEND.
R5 asks two questions: (a) is enforced *occupancy* a faithful H¹, or only a lower bound that
ignores partial entanglement — i.e. does the ℂ¹⁶ stalk move up the ordering? (b) with cross-bonds
in the same ledger, is the cross-synapse keystone **structurally starved** once a dimer spends
its four nuclei locally?

**Probes** (all under `sweep/`, results force-added):
`po7_unit1_cross_edge_validation.py`, `po7_unit1b_power_check.py`, `po7_unit3_monogamy_and_sheaf_check.py`,
`po7_unit4_slot_tag_check.py`, `po7_unit5_event_lifetime.py`, `po7_unit8_eta2_partition.py`,
`po7_unit9_spin_resolved.py`. Bit-identity gate: `po7_bitident_check.py`
(`1034 / 369740 / 0.991922159684`) — run before and after every physics edit.

**Open questions with recommendations:** `coordination/requests/po7-provenance-network/notes.md`.

## 6. SUGGESTED ORDER IN THE MORNING

1. **Check the subagent's shared-ledger commit** — bit-identity gate, `git show --stat`, and its
   measured numbers. Its key comparison: without the ledger this rig hits `largest_frac` 0.959 by
   t=12 s.
2. **If the ledger holds the partition open**, that is the first physically admissible
   multi-synapse partition — and the §8 keystone becomes askable for the first time on a graph
   that could exist. Pre-register before scoring.
3. **If it does not**, the honest reading is that a dimer spending four nuclei locally has none
   left for the network scale — the cross-synapse keystone is **structurally starved**, which is
   R5's second question and a real result.
4. **Send R5** either way; the advisor's steer on the ℂ¹⁶ stalk decides whether frustration
   counting is enough or the full spin state is needed.

**Do not** re-open cross-synapse provenance, move the Werner bound (0.5, LOCKED), or touch
`spine_plasticity_module.py`, `atp_system.py` phosphate path, `analytical_gap`, `sweep_runner.py`.

## 7. HONEST NOTE ON THIS SESSION

Several hours went into the cross-synapse provenance premise before Sarah corrected it, and I
twice reported findings I then had to withdraw (a false bond-tracking red flag; a blob claim made
on an ambiguous statistic before `largest_frac` was instrumented). Both are recorded in `L·PO7-3`
§6 rather than quietly fixed. The recurring failure was reaching for new measurement instead of
reading prior art already on disk — the co-activation rig I proposed building had been run and
logged as L·ETA-2.

---

# ADDENDUM — subagent result + two corrections that change how to read §1–§3

## 8. UNIT 11: the shared ledger is built, and cross-synapse bonds are STRUCTURALLY STARVED

Commit `6595c1b`. Cross-synapse bonds now claim from the same per-dimer 4×³¹P ledger the intra
bonds spend — six removal paths wired (not the two I scoped), provenance bonds claim their
**named** slot. 7 synapses @1 µm, 12 s, flag ON:

| | value |
|---|---|
| max **total** degree (intra + cross) | **4** — `over_bound = 0` at all 240 samples |
| mean total degree | **3.69** — ~92% of nuclei spent on INTRA bonds |
| cross bonds ever formed | **1** (zero ever above the Werner bound) |
| cross frustration | **710** |
| components / largest_frac / n_multi | 373 / **0.0119** / **0** |

**Monogamy now holds network-wide and the blob is gone** (Unit 10's 0.959 → 0.0119). But the
intra layer wins the competition for nuclei and **no cross-synapse structure survives at all** —
`n_multi = 0` for the entire run, against Unit 10's peak of 33. Nothing tuned; Werner bound
untouched.

**The physical picture this completes:**
1. There is **no η-free route** to cross-synapse entanglement (`L·PO7-2`).
2. **η does ignite** — 7/7 synapses condense (Unit 8).
3. **But once monogamy is enforced, dimers have no spare nuclei for cross bonds.**

⇒ **The cross-synapse partition may be structurally unavailable in this model.** That is a real
result and it bears directly on §8.

### ⚠ THE KEY OPEN QUESTION (do not skip this in the morning)

**Is the starvation physical, or an artifact of ORDERING?** Intra bonds form early and lock up
all four slots before η ignites at t≈10 s. If cross-bond formation had temporal priority, or if
the competition were resolved by fidelity rather than arrival order, the outcome could differ.
Nothing in the physics says intra bonds should win — it is simply what runs first. **This needs
deciding before "structurally starved" is treated as a finding about the model rather than about
the update order.**

## 9. ⚠ THE 7-SYNAPSE RIG IS NOT REPRODUCIBLE — this qualifies §1–§3

The subagent's OFF-flag gate failed, and the cause was **not** its edit: two runs of *identical*
code diverge at the same sample (`n_dimers` 1520 vs 1580 at t=8.70), with `cross_synapse_bonds`
empty throughout the identical window, so no edited line was live.

**This was already recorded this morning** in `coordination/HANDOFF_SARAH_2026-07-19_AM.md:18-24`
— three unseeded `np.random.default_rng()` calls (`camkii_module.py:199`,
`spine_plasticity_module.py:274`, `multi_synapse_network.py:1188`) that seed from OS entropy and
ignore the caller's seed. It is listed there as **blocking PO-7**. I did not read that handoff.

**Consequence — read §1–§3 with this attached:** Unit 8's and Unit 10's numbers are **single
draws from a distribution, not reproducible values**. That includes the ignition figures
(r = 2.53, 7/7 condensed), the `E_invasion` 0.28% match, Unit 10's 33-component window, and the
98-cross-bond percolation number. The morning handoff quotes them as measurements; they are
measurements of one draw. The AM handoff records `eta_max` varying **0.0, 0.0709, 0.0940, 0.1069**
across four driven runs — i.e. *whether the backbone condenses at all* was already known to vary.

**Unit 9's single-synapse result is NOT affected** (the fingerprint path is deterministic and
gated), so the headline in §1 — the blob breaking, 715 → 3.93 degree, largest_frac 1.000 → 0.112
— stands as reported.

## 10. ⚠ MY BIT-IDENTITY GATE DID NOT EXERCISE THE NETWORK CODE

`po7_bitident_check.py` drives `net.synapses[0].step(...)` — the synapse, not the network — so
`tracker.step` / `_update_entanglement` / `_step_network_provenance` never execute during it.
Every "BIT-IDENTICAL: PASS" I reported for `multi_synapse_network.py` changes (frozen fidelity,
coherence death, the provenance edits) was **necessary but not sufficient**; the gate would have
passed with cross-bond formation completely broken. My commit messages imply stronger
verification than I had.

**Fixed:** the subagent built a deterministic gate that *does* exercise the edit — `MODE=offpath`
in `sweep/po7_unit11_shared_ledger.py` drives `_update_entanglement` directly over synthetic
dimers with turnover, a zero-η synapse and an mt_invaded window, covering all four in-function
paths. Fingerprint `515772101786800`, pre-change == post-change, verified by stashing the edit.
**Use that gate for network-layer changes from now on.**

## 11. KNOWN LIMITATION IN UNIT 11

If a pair were ever both a `_prov_bond` and a `cross_synapse_bond`, they share one
`_cross_bond_spins` entry and the first removal releases spins the other still needs. Not
exercised here (`provenance_network=False`), but real if both are ever enabled together.

## 12. REVISED ORDER FOR THE MORNING

1. **Settle §8's ordering question** — is intra-first arrival why cross bonds starve? Cheapest
   test: reverse the order, or resolve claims by fidelity, and see whether cross structure
   survives. This decides whether "structurally starved" is physics or bookkeeping.
2. **Fix the three unseeded RNGs** — until then no multi-synapse number is reproducible and every
   comparison is between draws. It was already flagged as blocking; it now demonstrably blocks.
3. **Re-run Units 8/10 across ≥5 seeds** once seeded, to convert single draws into distributions.
4. **Send R5**, updated with §8's result — its second question ("is the cross-synapse keystone
   structurally starved?") now has a measured, if provisional, answer: yes, pending §8's ordering
   check.
