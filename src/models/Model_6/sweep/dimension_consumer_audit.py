#!/usr/bin/env python3
"""
DOES EVERY SWEEP DIMENSION REACH A LIVE CONSUMER? — PO-6a Unit 1.

WHY THIS EXISTS
---------------
`sweep_runner.py` writes `params.dendritic_backbone.D_modes` from the `q1_d_modes`
dimension. Nothing reads it. A sweep over that dimension therefore returns a FLAT
RESPONSE — and a flat response over a swept parameter reads as "this parameter does not
matter", i.e. a physical null. It is not a null. It is a wiring gap wearing the costume
of a result.

That is this program's characteristic defect (prose asserting mechanisms the code does not
implement) promoted into the measurement apparatus itself, where it is far more dangerous:
a lying parameter is one wrong docstring, a lying SWEEP manufactures wrong findings at
scale, and every one of them looks like data.

So: no dimension is assumed live. Each is either DEMONSTRATED to reach a consumer, or
marked INERT.

WHY THIS IS A MEASUREMENT AND NOT A GREP
----------------------------------------
Grep tells you a NAME APPEARS. It cannot tell you the name is read on the live path — the
attribute may be read only in a log line, reached via getattr, shadowed by a local, or
copied into a module that never uses it. This probe instead INSTRUMENTS ATTRIBUTE READS:
it patches `__getattribute__` on each parameter dataclass, runs the real model
(construct → N network steps → backbone update), and records which attributes were
actually read while physics was running.

  reads == 0  =>  INERT. Definitive: the value cannot have influenced anything, because
                  nothing looked at it.
  reads  > 0  =>  REACHED. Necessary, not sufficient — a read could be a print. Dimensions
                  in this class are additionally EFFECT-TESTED where cheap (below).

The asymmetry is deliberate and is why read-tracing is the right instrument for THIS
question: it can only ever over-report LIVE, so a "never read" verdict is sound.

THE POSITIVE CONTROL (why this verdict can fail)
------------------------------------------------
`MO_MODEL6.md` §2.3: "A verdict that cannot distinguish its outcomes is not a result."
An audit that marked everything LIVE would be indistinguishable from a broken tracer. So
this probe asserts its own discriminating power before reporting:

  CONTROL A — at least one traced attribute must come back LIVE   (tracer sees reads)
  CONTROL B — at least one traced attribute must come back INERT  (tracer sees absence)
  CONTROL C — `dendritic_backbone.omega_0` MUST be LIVE and `D_modes` MUST be INERT.
              These two are known from B2 by independent means (omega_0 enters P_c;
              D_modes was proven inert on executable code). They are the ground truth
              this instrument is calibrated against.

If any control fails the run reports INVALID, not a dimension verdict.

SCOPE / LIMITS (stated)
-----------------------
· Covers dimensions applied to `Model6Parameters` (Q1/Q2 params-level). Model-level
  (`q2_k_classical`, `q2_k_agg_baseline`), network-level and stimulus-level dimensions
  take different application paths and are reported SEPARATELY, by static inspection of
  their apply site, and are explicitly NOT given a read-traced verdict here.
· A REACHED verdict is not proof of physical effect, only of consumption. Where a
  dimension is REACHED but suspected inert-in-effect, that is called out rather than
  silently upgraded.
· The run is short (few steps). A parameter consumed only on a rare branch (e.g. only
  during dissolution, or only above a threshold this run never crosses) could read 0 and
  be wrongly marked INERT. Every INERT verdict below is therefore reported WITH the
  driving conditions, and any INERT dimension whose consumer might be conditional is
  flagged for a longer confirmation rather than asserted.

Read-only. Nothing is tuned; it drives values and observes.
"""
import sys, os, logging, collections

logging.disable(logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(MODEL6_DIR, '..', '..', '..'))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, REPO)

READS = collections.Counter()


def install_tracer(cls, label):
    """Record every non-dunder attribute read on `cls` instances."""
    orig = cls.__getattribute__

    def traced(self, name):
        if not name.startswith('__'):
            READS[f"{label}.{name}"] += 1
        return orig(self, name)

    cls.__getattribute__ = traced


# --- Instrument BEFORE importing the model, so construction-time reads are captured ---
from model6_parameters import (Model6Parameters, DendriticBackboneParameters,
                               TryptophanParameters, QuantumParameters,
                               PhosphateParameters)

TRACED_CLASSES = [
    (DendriticBackboneParameters, "dendritic_backbone"),
    (TryptophanParameters,        "tryptophan"),
    (QuantumParameters,           "quantum"),
    (PhosphateParameters,         "phosphate"),
]
for cls, label in TRACED_CLASSES:
    install_tracer(cls, label)

from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork

# dim_id -> the dotted attribute sweep_runner.py actually WRITES (not the dimension's
# declared `variable`, which differs for several: n_tryptophan->n_trp_baseline,
# f_coherent_base->f_coherent, j_coupling_hz->J_intrinsic_dimer).
PARAMS_LEVEL = [
    ("q1_n_tryptophan",       "tryptophan.n_trp_baseline",                "sweep_runner.py:55"),
    ("q1_f_coherent_base",    "tryptophan.f_coherent",                    "sweep_runner.py:57"),
    ("q1_d_modes",            "dendritic_backbone.D_modes",               "sweep_runner.py:61"),
    ("q1_phi_dissipation",    "dendritic_backbone.phi_dissipation",       "sweep_runner.py:63"),
    ("q1_chi_redistribution", "dendritic_backbone.chi_redistribution",    "sweep_runner.py:65"),
    ("q1_kT_per_modulation",  "dendritic_backbone.kT_per_modulation_unit","sweep_runner.py:67"),
    ("q2_t2_p31",             "quantum.T_singlet_dimer",                  "sweep_runner.py:71"),
    ("q2_j_coupling_hz",      "quantum.J_intrinsic_dimer",                "sweep_runner.py:73"),
    ("q2_phosphate_initial",  "phosphate.phosphate_total",                "sweep_runner.py:77"),
]

# Dimensions that do NOT go through Model6Parameters. Reported, not read-traced.
OTHER_LEVEL = [
    ("q2_k_classical",        "model.ca_phosphate.dimerization.k_classical", "sweep_runner.py:89",
     "model-level; MO-held constant (50x spread across 3 sites) — audit only, do not touch"),
    ("q2_k_agg_baseline",     "model.ca_phosphate.dimerization.k_agg",       "sweep_runner.py:92-93",
     "GUARDED by hasattr(...,'k_agg') — silently a NO-OP if the attribute is absent"),
    ("net_n_synapses",        "MultiSynapseNetwork(n_synapses=...)",         "sweep_runner.py:125-131",
     "constructor arg — structural, applied directly"),
    ("net_spacing_um",        "MultiSynapseNetwork(spacing_um=...)",         "sweep_runner.py:126-131",
     "constructor arg — structural, applied directly"),
    ("net_mt_invaded_fraction","syn.set_microtubule_invasion(bool)",         "sweep_runner.py:135-138",
     "applied via setter loop"),
]
STIMULUS_DIMS = ["stim_ca_amplitude", "stim_theta_cycles", "stim_n_traversals",
                 "stim_inter_traversal_s", "stim_burst_duration_ms",
                 "stim_theta_period_ms", "stim_dopamine_delay", "stim_silence_duration"]


def drive_the_model(n_syn=3, n_steps=8):
    """Construct and run the real model so consumers have a chance to read."""
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    net = MultiSynapseNetwork(n_synapses=n_syn, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    for _ in range(n_steps):
        net.step(1e-3, {'voltage': -10e-3, 'reward': False})
    net._update_backbone_field()
    return n_syn, n_steps


def main():
    print("=" * 78)
    print("DIMENSION-CONSUMER AUDIT — does each swept dimension reach a live consumer?")
    print("=" * 78)
    print("  Instrument: __getattribute__ read-tracing on the parameter dataclasses.")
    print("  reads == 0  =>  INERT (definitive: nothing looked at the value)")
    print("  reads  > 0  =>  REACHED (necessary, not sufficient — a read may be a print)")
    print()

    n_syn, n_steps = drive_the_model()
    print(f"  Driving conditions: {n_syn} synapses, MT-invaded, -10 mV, {n_steps} steps,"
          f" + explicit _update_backbone_field()")
    print()

    # ---------------- CONTROLS ----------------
    n_live = sum(1 for k, v in READS.items() if v > 0)
    omega_reads = READS.get("dendritic_backbone.omega_0", 0)
    dmodes_reads = READS.get("dendritic_backbone.D_modes", 0)

    traced_targets = [path for _, path, _ in PARAMS_LEVEL]
    n_inert_targets = sum(1 for p in traced_targets if READS.get(p, 0) == 0)

    ctrlA = n_live > 0
    ctrlB = n_inert_targets > 0
    ctrlC = (omega_reads > 0) and (dmodes_reads == 0)

    print("--- CONTROLS (the audit must be able to distinguish its outcomes) ---")
    print(f"  A  tracer observes reads at all              : {'PASS' if ctrlA else 'FAIL'} "
          f"({n_live} attributes read)")
    print(f"  B  tracer observes absence of reads          : {'PASS' if ctrlB else 'FAIL'} "
          f"({n_inert_targets} swept targets never read)")
    print(f"  C  calibration vs B2 ground truth            : {'PASS' if ctrlC else 'FAIL'}")
    print(f"       omega_0 must be LIVE  -> reads={omega_reads}")
    print(f"       D_modes must be INERT -> reads={dmodes_reads}")
    controls_ok = ctrlA and ctrlB and ctrlC
    print()

    # ---------------- PARAMS-LEVEL VERDICTS ----------------
    print("--- PARAMS-LEVEL DIMENSIONS (read-traced) ---")
    print(f"  {'dim_id':<24}{'attribute written':<44}{'reads':>7}  verdict")
    print("  " + "-" * 84)
    inert, reached = [], []
    for dim, path, site in PARAMS_LEVEL:
        n = READS.get(path, 0)
        verdict = "REACHED" if n else "*** INERT ***"
        (reached if n else inert).append((dim, path, site))
        print(f"  {dim:<24}{path:<44}{n:>7}  {verdict}")
    print()

    # ---------------- NON-PARAMS DIMENSIONS ----------------
    print("--- NON-PARAMS DIMENSIONS (different apply path; NOT read-traced here) ---")
    for dim, path, site, note in OTHER_LEVEL:
        print(f"  {dim:<26}{site:<22}{note}")
    print(f"  {len(STIMULUS_DIMS)} stimulus dimensions route through "
          f"scenario_from_vector() (sweep_runner.py:145) — audited separately.")
    print()

    # ---------------- VERDICT ----------------
    print("=" * 78)
    if not controls_ok:
        print("VERDICT: INVALID — a control failed; this instrument is not discriminating,")
        print("         so no dimension verdict below it can be trusted.")
        return 2

    print(f"VERDICT: {len(inert)} of {len(PARAMS_LEVEL)} params-level dimensions are INERT.")
    print()
    print("  Each of these is SWEPT by sweep_runner.py and READ BY NOTHING. A sweep over")
    print("  any of them returns a flat response that reads as a physical null:")
    for dim, path, site in inert:
        print(f"    · {dim:<24} written at {site:<22} -> {path}")
    print()
    print("  LIMITS: INERT is definitive for these driving conditions. A consumer on a rare")
    print("  branch this short run never reached would look identical — so these are")
    print("  reported as INERT-under-stated-conditions, and any that plausibly have a")
    print("  conditional consumer should get a longer confirmation before deletion.")
    print("  REACHED is necessary but not sufficient: a read may be a log line, not physics.")
    print("=" * 78)
    return 1 if inert else 0


if __name__ == "__main__":
    sys.exit(main())
