# Forward-learning loop audit — probe scripts, 2026-07-18

These are the probes behind research-log entries **E2** (`RESEARCH_LOG_CALCIUM_DIMER.md`) and
**L·ETA-1** (`RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`). They are kept because the T1′ scar
applies here too: the original spatial-discovery and place-field results survive only as
prose transcribed into handoffs, and are not independently re-derivable. These are.

They are **diagnostic probes, not experiments** — they read and instrument the live code, and
none of them modifies repo source. Run from this directory with the project venv:

    /Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python <probe>.py

| probe | question it answers | key measured result |
|---|---|---|
| `eta_probe.py` | does `eta` clear `r ≥ 1` under the drivers' drive? | NO — `r` floors at 0.039, peaks 0.141 at 30 s |
| `dep_probe.py` | how does `r` depend on N and spacing? | `d*` saturates; unreachable at ≥2 µm spacing at any N |
| `ca_probe.py` | what is `ca_open` really, vs the `≈1` assumed in `soc_pump_threshold_stage1.py:88`? | 0.726 @ −10 mV steady, **0.063** duty-averaged over the theta protocol |
| `probe_latch2.py` | how often does `perform_quantum_measurement` fire across trials? | once in trial 0, zero after; spine volume still accumulates |
| `probe_spine_volume.py` | what drives spine volume, and does it self-maintain? | calcium drives magnitude; commitment only sets durability; decays below baseline by 3000 s |
| `probe_templates_and_drive.py` | does the `n_templates` feedback fire, and how large is it? | fires early (V>1.25 by ~8 s); mean rate +1.0–1.5%; template-bound fraction ~2× |
| `probe_duty.py` | what duty cycle do synapses actually see? | best 0.189, mean 0.037 ⇒ AMPAR onset needs ~159–809 trials |

## Caveats carried from the runs

- `probe_latch2.py` uses a **reduced** config (4 synapses, 3 trials, 14 s budget, deterministic
  agent walking to the goal). The latch behaviour is structural and config-independent; the
  *relative weight* of the two accumulation paths at the real 20-synapse config is NOT
  observed and should not be quoted as if it were.
- `eta_probe.py`'s run F was stopped at t=30 s (the 60 s version exceeded 15 min CPU). The
  `E_invasion → 1.0` value of `r ≈ 0.32` is **extrapolated**, not measured. It is 3× below
  threshold, so the verdict does not turn on the extrapolation, but the number is not a
  measurement.
- An unresolved ~4× discrepancy exists between isolated-channel `ca_open` at −40 mV (0.233)
  and in-network (0.04–0.06). The in-network value is the one entering `r`, and it is the
  *lower* one, so it does not soften the verdict — but the cause is unexplained.
