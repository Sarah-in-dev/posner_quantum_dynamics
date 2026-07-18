# AUDIT — does the spontaneous-release floor invalidate other probes' nulls?

**PO-3, 2026-07-18, MO rotation 001.** Audit only: no probe was re-run, no other PO's log row
edited. Every claim carries `file:line` or a measurement.

**The defect shape.** `PresynapticRelease.step` (`sweep/presynaptic_release.py:124`) computes
`rate = self.baseline_rate + a * self.peak_rate` with `BASELINE_RATE_HZ = 0.5` (`:65`). So a
synapse held at `act = 0.0` **still releases glutamate** (~0.2 Hz measured, full amplitude).
**Zeroing activation does not silence a synapse.** Any control arm built by zeroing activation
is not a null.

---

## 1. `plateau_vgcc_leak_probe.py` — the L·ETA-4 probe. **AFFECTED — the NMDAR half does not survive as evidence.**

**This is the one that matters**: L·ETA-4's silent-synapse NMDAR result is the sole surviving
basis for PO-5's `P_product` selectivity hypothesis.

### The leak is present, measured with the probe's own seeds

`run()` builds `rel = [PresynapticRelease(seed=3000+i) ...]` (`:124`) and sets
`acts = np.zeros(N_SYN); acts[DRIVEN] = 1.0` (`:125`) — the six non-driven synapses sit at
`act = 0.0`. Replaying those exact seeds for the probe's own `T_S = 12.0 s` at `DT = 0.005`:

```
synapse 0: 5   synapse 1: 2   synapse 2: 0
synapse 4: 1   synapse 5: 3   synapse 6: 2      -> 13 release events across 6 "silent" synapses
```

At `tau_nmda = 0.166 s` per event that is **~2.16 synapse-seconds of NMDAR occupancy at
synapses the probe calls silent.** L·ETA-4's row states *"no glutamate, so no NMDAR opening"*
and *"no glutamate, no current, however depolarized"* — **the premise is false.**

### But the deeper problem: the METRIC cannot detect the effect it is cited for

`split_open()` (`:108-118`) returns `np.mean(st[m])` where `st = ch.state` — the channel
**open-state** array. So `nmda` is NMDAR **open fraction**, not current and not calcium.

NMDAR gating in `analytical_calcium_system.py:129-130`:

```
alpha_eff = np.where(self.is_nmda, self.alpha * g_bind,        self.alpha * P_open_v)
beta_eff  = np.where(self.is_nmda, self.beta, self.beta * (1.0 - P_open_v))
```

**NMDAR opening and closing depend on glutamate only. Neither term contains voltage.** `B(V)`
(`:115`) scales the **current**, never the gating — this is the documented design
(`model6-input-engine`: *"Glutamate gates opening; depolarization gates conduction"*).

**Therefore `sil_nm_gain` — the plateau-induced change in silent-synapse NMDAR open fraction —
is ~0 BY CONSTRUCTION.** A plateau cannot move it whether or not calcium flowed. The measured
`−0.0019` is the residual of two RNG streams, not a physical result. **The test is structurally
incapable of detecting the quantity it is cited as establishing.**

This is the **same vacuity class already corrected once in this very probe** — its first
auto-verdict printed "eta stays SELECTIVE" because `eta == 0` at silent synapses when `eta == 0`
at the driven one too. That was fixed at the verdict layer; **an equivalent vacuity remains one
layer down, in the metric.**

### Direction of the error — it runs AGAINST the hypothesis it was used to support

The plateau raises Mg-unblock ~21×:

```
B(-70 mV) = 0.00364      B(-40 mV) = 0.02292      B(-20 mV) = 0.07498      B(0 mV) = 0.21881
B(-20)/B(-70) = 20.6x
```

So during those ~2.16 synapse-seconds of occupancy, silent-synapse NMDAR calcium current is
~21× what it would be at rest. **The plateau plausibly DOES drive NMDAR calcium at silent
synapses** — the opposite of what L·ETA-4's row concludes.

### Verdict, stated to the MO's three options

**AFFECTED.** The NMDAR half of L·ETA-4 does not survive as evidence: its premise (no glutamate)
is false and its metric (open fraction) is voltage-independent by construction.
**The magnitude of the real leak is UNDETERMINED** — establishing it needs a re-run with a
current- or calcium-based metric at the silent synapses, which is a re-run of another PO's
result and is **the MO's to sequence, not mine.**

### What SURVIVES, so this is not read as a retraction

- **The VGCC half stands.** `sil_vg_gain = +0.492` (open fraction 0.0017 → 0.4783). VGCC gating
  **is** voltage-dependent (`alpha * P_open_v`, `:129`), so open fraction is the right metric
  there. That measurement is sound.
- **L·ETA-4's CONCLUSION stands on the VGCC evidence:** `E_invasion` silent 0.2115 vs driven
  0.2115, `r` not separable, η cannot carry input-selectivity under a plateau. **§8's premise
  still fails.** Nothing here rehabilitates it.
- What fails is specifically the **positive basis offered for `P_product` selectivity** — the
  claim that the NMDAR channel stays clean. **PO-5's foundation is weaker than the log states.**

---

## 2. The rest of the family

| probe | null / control arm | built by zeroing activation? | leak | verdict |
|---|---|---|---|---|
| `sweep/einvasion_ratchet_probe.py` (mine, L·ETA-5) | yes — null arm | **yes** (`:152-153`, `acts[target] = 0.0`) | **YES** | **VOID, already reported.** Null reached `E_invasion = 0.4507`, out-gained the drive arm 7.46× vs 5.65× |
| `sweep/plateau_vgcc_leak_probe.py` (L·ETA-4) | yes — 6 "silent" synapses | **yes** (`:125`) | **YES** (13 events / 12 s) | **AFFECTED** — see §1. NMDAR half vacuous; VGCC half and the conclusion stand |
| `sweep/run_place_field_learning.py` | no null arm; non-active synapses at `act = 0.0` (`:252`) | n/a — not a control | **YES** | **NO VERDICT INVALIDATED** (no null to invalidate), but inactive synapses accumulate drive. Relevant if per-synapse gradients are read as activity-specific |
| `sweep/eta_in_live_trial.py` (L·ETA-3) | **no null arm** — all synapses Gaussian-driven | n/a | yes, at below-floor synapses | **N/A.** L·ETA-3 reports a shortfall, not a contrast, so there is no control to void. Its `ca_open` attribution stands unchallenged by me (see queue Q1-CORRECTION) |
| `sweep/loop_audit_2026_07_18/probe_latch2.py` | no null arm — latch/counter audit | n/a | **gaps are clean** | **UNAFFECTED.** It calls `r.advance_silent(GAP)` (`:114`), whose docstring states spontaneous release during the gap *"is neglected"* — the one consumer that suppresses the floor |
| `sweep/run_spatial_discovery.py` (`run_trial`) | no null arm | n/a | **suppressed, by a different bug** | **UNAFFECTED by the leak — carries D19 instead.** Release is stepped for all, but `if active_mask[i]:` (`:203`) means **only active synapses are stepped at all**, so inactive ones neither decay nor accumulate |

### The contrast in the last two rows is the useful finding

**`run_spatial_discovery` and my probe fail in opposite directions.** The shipped runner gates
stepping on `active_mask` (D19: inactive synapses never run their decay term) — which
incidentally suppresses the spontaneous-release leak but freezes decay. My probe steps every
synapse unconditionally (`step_network_per_synapse`, `:321-322`, no mask) — which correctly
runs decay (clock assertion PASS, AMENDMENT 3) but exposes the leak.

**Neither gives a clean silent synapse.** A correct probe needs *both*: step every synapse
**and** suppress spontaneous release in the control arm. **No probe in this family currently
does both.**

---

## Limits of this audit

- No probe was re-run. §1's leak counts come from replaying `PresynapticRelease` with the
  probe's own seeds and constants — the release layer only, not the full model.
- The **magnitude** of NMDAR calcium at L·ETA-4's silent synapses is **not** established here.
  Only that it is non-zero-by-premise and invisible to the metric used.
- `run_place_field_learning.py` and `probe_latch2.py` were read for null construction only, per
  the MO's "one-line verdict per probe", not audited for physics.
