# PO-7 handoff — evening of 2026-07-19

**Read this first in the morning.** Branch `claude/nervous-hertz-7ccff6`, worktree
`.claude/worktrees/po5-keystone`. Working tree clean; 17 commits from `78c60cc`.

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
