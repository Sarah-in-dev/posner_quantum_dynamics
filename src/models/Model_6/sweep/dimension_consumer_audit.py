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

# Model-level dimensions: applied to the model instance, not to Model6Parameters.
# Verdict is by direct attribute probe of the apply target.
MODEL_LEVEL = [
    ("q2_k_classical",    "ca_phosphate.dimerization.k_classical", "sweep_runner.py:89",
     "unguarded write; MO-held constant (50x spread across 3 sites) — audit only, never touch"),
    ("q2_k_agg_baseline", "ca_phosphate.dimerization.k_agg",       "sweep_runner.py:92-93",
     "GUARDED by hasattr(...,'k_agg'); silently a NO-OP if the attribute is absent"),
]

# Network dimensions are structural constructor/setter arguments, applied directly and
# unconditionally at sweep_runner.py:125-138. They are not attribute writes onto a params
# object, so read-tracing does not apply; they are recorded as APPLIED-STRUCTURALLY.
NETWORK_LEVEL = [
    ("net_n_synapses",         "MultiSynapseNetwork(n_synapses=...)", "sweep_runner.py:125-131"),
    ("net_spacing_um",         "MultiSynapseNetwork(spacing_um=...)", "sweep_runner.py:126-131"),
    ("net_mt_invaded_fraction","syn.set_microtubule_invasion(bool)",  "sweep_runner.py:135-138"),
]

# Stimulus dimensions -> ThetaBurstScenario constructor args (sweep_runner.py:145).
# dim_id -> scenario attribute.
STIMULUS_LEVEL = {
    "stim_ca_amplitude":      "ca_amplitude",
    "stim_theta_cycles":      "theta_cycles_per_traversal",
    "stim_n_traversals":      "n_traversals",
    "stim_inter_traversal_s": "inter_traversal_interval_s",
    "stim_burst_duration_ms": "burst_duration_ms",
    "stim_theta_period_ms":   "theta_period_ms",
    "stim_dopamine_delay":    "dopamine_delay_s",
    "stim_silence_duration":  "silence_duration_s",
}


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


# --- STAGE 2: EFFECT TEST -------------------------------------------------------------
#
# Read-tracing answers "was it consumed". It does NOT answer "did it change anything" — a
# read can be a log line. This stage drives each REACHED dimension at two values under a
# fixed seed and checks whether a downstream quantity moves.
#
# THE OBSERVABLE IS DECLARED PER DIMENSION, AND THAT IS NOT A CONVENIENCE.
# The first version of this test used ONE global fingerprint (dimer count, mean P_S,
# k_enhancement, peak calcium) for every dimension, and returned "NO EFFECT" for
# q1_n_tryptophan and q1_f_coherent_base. That was WRONG. Both are live: they move
# collective_field_kT (18.6 -> 23.0 and 14.0 -> 22.1 respectively) — a channel the global
# fingerprint simply did not span. A null from an instrument that cannot see the channel is
# not a null; it is a blind spot, and it would have condemned two working dimensions.
#
# So each dimension names the observable its effect should appear in, and BLINDNESS IS
# TREATED AS A TEST FAILURE, not as evidence of inertness: if the declared observable does
# not move, the result is reported as UNDEMONSTRATED (the honest verdict) rather than as
# INERT (a claim this stage is not entitled to make).
EFFECT_CASES = [
    # (dim_id, dotted params path, low value, high value, observable)
    ("q1_n_tryptophan",     "tryptophan.n_trp_baseline", 50,     500,   "collective_field_kT"),
    ("q1_f_coherent_base",  "tryptophan.f_coherent",     0.04,   0.10,  "collective_field_kT"),
    ("q2_phosphate_initial","phosphate.phosphate_total", 0.0001, 0.010, "n_dimers"),
]


def _observables(syn):
    import numpy as _np
    return {
        "n_dimers": len(syn.dimer_particles.dimers),
        "collective_field_kT": round(float(getattr(syn, '_collective_field_kT', 0.0)), 6),
        "em_field_trp": round(float(getattr(syn, '_em_field_trp', 0.0)), 3),
        "k_enhancement": round(float(getattr(syn, '_k_enhancement', 0.0)), 9),
    }


def _run_with(path, value, seed=1234, steps=40):
    import numpy as _np
    _np.random.seed(seed)
    p = Model6Parameters()
    p.em_coupling_enabled = True
    if path is not None:
        obj = p
        parts = path.split('.')
        for q in parts[:-1]:
            obj = getattr(obj, q)
        setattr(obj, parts[-1], value)
    syn = Model6QuantumSynapse(p)
    syn.set_microtubule_invasion(True)
    for _ in range(steps):
        syn.step(1e-3, {'voltage': -10e-3, 'reward': False})
    return _observables(syn)


def run_effect_tests():
    """Returns (determinism_ok, [(dim, observable, lo, hi, moved)])."""
    a = _run_with(None, None)
    b = _run_with(None, None)
    determinism_ok = (a == b)   # without this, any difference below is meaningless noise
    rows = []
    for dim, path, lo, hi, obs in EFFECT_CASES:
        r_lo = _run_with(path, lo)
        r_hi = _run_with(path, hi)
        rows.append((dim, obs, r_lo[obs], r_hi[obs], r_lo[obs] != r_hi[obs]))
    return determinism_ok, rows


def trace_stimulus_dimensions():
    """Read-trace ThetaBurstScenario attributes through a real (small) scenario run.

    The scenario is deliberately shrunk — 2 traversals so an inter-traversal gap exists,
    short bursts, short silence — but every branch of run()/_run_epoch is exercised:
    bursts, gap, dopamine, final silence, snapshots.
    """
    from sweep.theta_burst_scenario import ThetaBurstScenario
    reads = collections.Counter()
    orig = ThetaBurstScenario.__getattribute__

    def traced(self, name):
        if not name.startswith('_'):
            reads[name] += 1
        return orig(self, name)

    ThetaBurstScenario.__getattribute__ = traced
    try:
        p = Model6Parameters()
        p.em_coupling_enabled = True
        p.multi_synapse_enabled = True
        net = MultiSynapseNetwork(n_synapses=2, pattern="clustered", spacing_um=1.0)
        net.initialize(Model6QuantumSynapse, p)
        for s in net.synapses:
            s.set_microtubule_invasion(True)
        sc = ThetaBurstScenario(
            ca_amplitude=1e-5, theta_cycles_per_traversal=2, n_traversals=2,
            inter_traversal_interval_s=1.0, burst_duration_ms=20.0,
            theta_period_ms=100.0, dopamine_delay_s=0.3, silence_duration_s=1.0,
        )
        sc.run(net, dt=1e-3)
    finally:
        ThetaBurstScenario.__getattribute__ = orig
    return reads


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

    # ---------------- MODEL-LEVEL ----------------
    print("--- MODEL-LEVEL DIMENSIONS (apply target probed directly) ---")
    from model6_parameters import Model6Parameters as _MP
    from model6_core import Model6QuantumSynapse as _Syn
    probe_syn = _Syn(_MP())
    for dim, path, site, note in MODEL_LEVEL:
        obj = probe_syn
        ok = True
        for part in path.split('.'):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        verdict = "REACHED" if ok else "*** INERT (silent no-op) ***"
        if not ok:
            inert.append((dim, path, site))
        else:
            reached.append((dim, path, site))
        print(f"  {dim:<24}{path:<42}{verdict}")
        print(f"  {'':<24}{site} — {note}")
    print()

    # ---------------- NETWORK-LEVEL ----------------
    print("--- NETWORK DIMENSIONS (structural constructor/setter args) ---")
    for dim, path, site in NETWORK_LEVEL:
        print(f"  {dim:<26}{path:<40}APPLIED-STRUCTURALLY  {site}")
    print("  Applied directly and unconditionally; not attribute writes, so read-tracing")
    print("  does not apply. Effect on results not asserted here.")
    print()

    # ---------------- STIMULUS-LEVEL ----------------
    print("--- STIMULUS DIMENSIONS (read-traced through a real scenario run) ---")
    stim_reads = trace_stimulus_dimensions()
    print(f"  {'dim_id':<26}{'scenario attribute':<30}{'reads':>7}  verdict")
    print("  " + "-" * 76)
    for dim, attr in STIMULUS_LEVEL.items():
        n = stim_reads.get(attr, 0)
        verdict = "REACHED" if n else "*** INERT ***"
        site = "sweep_runner.py:145"
        (reached if n else inert).append((dim, f"ThetaBurstScenario.{attr}", site))
        print(f"  {dim:<26}{attr:<30}{n:>7}  {verdict}")
    print()

    # ---------------- STAGE 2: EFFECT TESTS ----------------
    print("--- EFFECT TESTS (does a driven value MOVE a downstream quantity?) ---")
    det_ok, rows = run_effect_tests()
    print(f"  determinism control (same params twice -> same observables): "
          f"{'PASS' if det_ok else 'FAIL — differences below are noise, not effect'}")
    print(f"  {'dimension':<24}{'observable':<24}{'low':>16}{'high':>16}  moved?")
    print("  " + "-" * 88)
    undemonstrated = []
    for dim, obs, lo, hi, moved in rows:
        if not moved:
            undemonstrated.append(dim)
        print(f"  {dim:<24}{obs:<24}{str(lo):>16}{str(hi):>16}  "
              f"{'YES' if moved else 'UNDEMONSTRATED'}")
    if undemonstrated:
        print(f"  NOTE: UNDEMONSTRATED means this stage could not show an effect in the")
        print(f"        declared observable. It does NOT mean inert — it may mean the")
        print(f"        observable is blind to the channel. See EFFECT_CASES.")
    print()

    # ---------------- VERDICT ----------------
    print("=" * 78)
    if not controls_ok:
        print("VERDICT: INVALID — a control failed; this instrument is not discriminating,")
        print("         so no dimension verdict below it can be trusted.")
        return 2

    n_audited = len(PARAMS_LEVEL) + len(MODEL_LEVEL) + len(STIMULUS_LEVEL)
    print(f"VERDICT: {len(inert)} of {n_audited} read-traceable dimensions are INERT.")
    print(f"         ({len(NETWORK_LEVEL)} network dimensions are applied structurally and")
    print(f"          are excluded from this denominator — see their section above.)")
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
