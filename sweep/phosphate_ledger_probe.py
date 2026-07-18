"""
L·PO4-1 — PHOSPHATE MASS-CONSERVATION LEDGER, AROUND THE LIVE LOOP
══════════════════════════════════════════════════════════════════════════════════════

PO-2's acceptance measurement. Pre-registered in `src/models/Model_6/docs/PREREG_PO2_PHOSPHATE.md`
BEFORE this file was written. Committed FAILING on current code first, per `fa12009`.

WHAT MAKES THIS DIFFERENT FROM sweep/phosphate_conservation_probe.py (A3, the prior art)
────────────────────────────────────────────────────────────────────────────────────────
A3 is an ISOLATED reimplementation of the chemistry. `grep -n "ATP|hydrolys|recovery"` over it
returns ZERO hits: it has NO ATP ARM. It therefore measured the half of the loop that does not
leak, found "exact conservation (2e-17 M)" (DECISION RECORD D8), and DECISION RECORD D14 read
that as "SOC loop already closed in live code (no B3 edit needed)" off a probe whose phosphate
feedback was, in D14's own words, "mimicking model6_core".

THIS probe instantiates the LIVE `Model6QuantumSynapse` and steps it. Nothing is mimicked.
That is the entire methodological point: a conserving result is the DEFAULT outcome of a
badly-scoped ledger, so the ledger must provably span the ATP arm or the verdict is worthless.

A3's speciation/gate/K_CLASSICAL grounding is reused by reference, not re-derived here.

THE LEDGER (pre-registration §1) — one terminal phosphate per ATP
────────────────────────────────────────────────────────────────────────────────────────
    P_total = Σ[ATP] + Σ phosphate_released + Σ phosphate_metabolic
              + Σ phosphate_structural + 4·Σ[dimer] + 6·Σ[trimer]

Every transition in the loop is elementwise and local (ATP diffusion is commented out at
`atp_system.py:490`), so a plain grid sum is a valid conserved quantity. If diffusion is ever
re-enabled this assumption needs re-checking — registered as an assumption, not buried.

    hydrolysis   atp_system.py:128-130      ATP -= d ; phosphate_released += d   CONSERVED
    binning      atp_system.py:419-428      released -> metabolic + structural   CONSERVED
    formation    model6_core.py:450-452     structural -= 4*d_dimer (SIGNED, so   CONSERVED
                 model6_core.py:756-757     dissolution returns it)
    recovery     atp_system.py:163,169-171  ATP += d ; ADP -= d ; **no Pi debit** LEAKS

REGISTERED PREDICTION (pre-registration §2), on current unfixed code:

    dP  ==  hydrolysis.total_recovered        (to within eps)

Three registered outcomes, TWO OF WHICH CONTRADICT THE DISPATCH:
    dP ~ total_recovered  -> defect exactly as diagnosed; fix is stoichiometric
    dP ~ 0                -> THERE IS NO LEAK; the dispatch's central claim is wrong
    dP ~ neither          -> a second unidentified source/sink; fix NOTHING until identified

TOLERANCE (pre-registration §3): eps = 1e-12 relative. Justified against float64 accumulation
(~7e-14 bound at N~1e4 steps, G~1e3 points; machine eps 2.22e-16). Registered as NOT deciding
the verdict: the predicted leak is ~11 orders above it, so any eps in [1e-14, 1e-3] returns the
same answer. eps is fixed and is NOT widened after seeing the result.

DISCIPLINE: emergent physics only (MO_MODEL6.md §7 LOCKED). No compensating term is permitted
to make this balance. If it will not balance, the probe reports the gap.

Run:  /Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python -u sweep/phosphate_ledger_probe.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models", "Model_6"))

from model6_core import Model6QuantumSynapse          # noqa: E402
from model6_parameters import Model6Parameters        # noqa: E402

# ── registered constants — fixed at pre-registration, NOT tunable here ────────────────
EPS_REL       = 1e-12      # pre-registration §3
PO4_PER_DIMER = 4.0        # Ca6(PO4)4   — ca_triphosphate_complex.py:437-438
PO4_PER_TRIMER = 6.0       # Ca9(PO4)6   — ca_triphosphate_complex.py:437-438

DT       = 0.005           # s, the program's standard physics step
N_STEPS  = 1000            # 5 s
# AMENDMENT A2.1 (registered 2026-07-18 20:10Z, BEFORE any scored result was seen):
# N_STEPS 4000 -> 1000. The first launch was killed for violating the MO compute rule
# ("progress instrumentation, results persisted incrementally", never piped through `tail`) —
# it ran 13 min CPU with zero visibility. This does NOT touch the discriminator: the registered
# prediction is dP == total_recovered, an identity that holds per-step, so it is testable over
# any run long enough to exercise hydrolysis + formation + dissolution + recovery. 5 s at
# dt=0.005 covers all four (burst period 400 steps -> 2.5 burst/rest cycles). Shortening a run
# cannot make a leak look smaller RELATIVE to the recovery it is predicted to equal.
V_DEPOL  = -10e-3          # V, burst depolarisation (experiment-design-patterns:73)
V_REST   = -70e-3          # V


def ledger(syn):
    """P_total over the live objects. Pre-registration §1. Reads state, mutates nothing."""
    h = syn.atp.hydrolysis
    p = syn.atp.phosphate
    d = syn.ca_phosphate.dimerization
    return (
        float(np.sum(h.atp))
        + float(np.sum(h.phosphate_released))
        + float(np.sum(p.phosphate_metabolic))
        + float(np.sum(p.phosphate_structural))
        + PO4_PER_DIMER * float(np.sum(d.dimer_concentration))
        + PO4_PER_TRIMER * float(np.sum(d.trimer_concentration))
    )


def run(label, n_steps=N_STEPS, suppress_recovery=False, inject_at=None, inject_amount=0.0,
        seed=1000):
    """Step the LIVE model, tracking the ledger. Returns a result dict."""
    np.random.seed(seed)
    syn = Model6QuantumSynapse(Model6Parameters())

    if suppress_recovery:
        # C2: config change only — no physics edited. recovery_rate -> ~0.
        syn.atp.hydrolysis.params.atp_recovery_tau = 1e30

    p0 = ledger(syn)
    clamp_hits = 0
    injected = 0.0
    jc_trace, consumed_trace = [], []
    cum_consumed = 0.0

    print(f"    [{label}] starting {n_steps} steps ...", flush=True)
    for i in range(n_steps):
        if i and i % 100 == 0:
            # MO compute rule: progress instrumentation, persisted incrementally.
            cur = ledger(syn)
            print(f"    [{label}] step {i}/{n_steps}  P={cur:.9e}  "
                  f"dP={cur - p0:+.3e}  recovered={syn.atp.hydrolysis.total_recovered:.3e}",
                  flush=True)
        # C3: the np.maximum clamp at model6_core.py:451/:757 CREATES phosphate when it binds.
        # Detect it by checking, before the step, whether the pending decrement would go
        # negative anywhere. Counted as a distinct failure mode, never folded into dP.
        pend = syn.ca_phosphate.get_phosphate_consumed()
        if pend is not None and np.any(
            syn.atp.phosphate.phosphate_structural - pend < 0.0
        ):
            clamp_hits += 1

        stim = {"voltage": V_DEPOL if (i % 400) < 200 else V_REST}
        syn.step(DT, stim)

        if inject_at is not None and i == inject_at:
            # C1: positive control — a KNOWN leak the ledger must detect exactly.
            syn.atp.phosphate.phosphate_structural += inject_amount
            injected = inject_amount * syn.atp.phosphate.phosphate_structural.size

        # defect-1 instrumentation: does J-coupling track dimer consumption?
        cons = syn.ca_phosphate.get_phosphate_consumed()
        if cons is not None:
            cum_consumed += float(np.sum(cons))
        jc_trace.append(float(np.mean(syn.atp.j_coupling.j_coupling)))
        consumed_trace.append(cum_consumed)

    p1 = ledger(syn)
    recovered = float(syn.atp.hydrolysis.total_recovered)
    dP = p1 - p0

    # defect 1: the stale total vs the live sum of its two parts
    p_obj = syn.atp.phosphate
    stale_gap = float(
        np.sum(np.abs(p_obj.phosphate_total
                      - (p_obj.phosphate_structural + p_obj.phosphate_metabolic)))
    )

    # does J-coupling track consumption? (Pearson r over the run)
    jc = np.asarray(jc_trace)
    cc = np.asarray(consumed_trace)
    if jc.std() > 0 and cc.std() > 0:
        jc_r = float(np.corrcoef(jc, cc)[0, 1])
    else:
        jc_r = float("nan")   # a flat J-coupling is exactly the defect-1 signature

    return {
        "label": label,
        "P_initial": p0,
        "P_final": p1,
        "dP": dP,
        "dP_relative": dP / p0 if p0 else float("nan"),
        "total_recovered": recovered,
        "injected": injected,
        # NOTE: this prediction is the UNFIXED-code expectation (recovery leaks). On fixed
        # code the leak term is absent and the residual equals -recovered by construction.
        # The verdict function does NOT use it as a gate — see AMENDMENT A2.2.
        "predicted_dP_if_unfixed": recovered + injected,
        "residual_vs_prediction": dP - (recovered + injected),
        "clamp_activations": clamp_hits,
        "conserved_at_eps": abs(dP / p0) <= EPS_REL if p0 else False,
        "stale_total_gap": stale_gap,
        "jcoupling_vs_consumed_r": jc_r,
        "jcoupling_std": float(jc.std()),
        "cumulative_po4_consumed": cum_consumed,
    }


def verdict(main, c1, c2):
    """
    Registered verdict function. MUST be able to return every one of these.

    AMENDMENT A2.2 (2026-07-18, disclosed — the C1 gate was DEFECTIVE as first written).
    It read `c1["residual_vs_prediction"] / c1["P_initial"] > EPS_REL * 1e6` — with no
    `abs()`, and against a `predicted_dP` of `recovered + injected` that silently assumes
    recovery LEAKS. Two consequences:

      * On the unfixed code the residual was +3.9e-13 and the gate correctly stayed quiet,
        so the committed FAILING verdict (LEAK_MATCHES_RECOVERY, 305e096) is UNAFFECTED.
      * On the FIXED code recovery no longer leaks, so `predicted_dP` over-predicts by exactly
        the recovery: residual -1.111e-02, i.e. -3.175e-04 relative. Being negative it slipped
        under a gate missing `abs()`. **With `abs()` it would have returned INVALID_C1_BLIND —
        a FALSE invalid**, because C1 in substance worked perfectly: it reported dP = 1.000000
        against an injection of exactly 1.0.

    So the first CONSERVED reading was reached partly by a sign accident, and I am not entitled
    to keep a pass I got that way. Replaced with a criterion that does not depend on whether the
    code under test leaks: **the ledger must detect the injection as a DIFFERENCE against the
    matched main arm.**

        |(dP_C1 - dP_main) - injected| / P_initial  <=  C1_TOL

    Re-scored both ways: unfixed -> 3.6e-06 (passes), fixed -> ~1e-14 (passes). The verdicts
    stand under the corrected gate; the correction was made before relying on either.
    """
    C1_TOL = 1e-4  # loose vs eps by design: this gate asks "did the detector SEE it", not
                   # "is it conserved". The injection is 1.0 against P~35, so a detector that
                   # missed it would be off by ~3e-2 relative — 300x this tolerance.
    c1_detected = (c1["dP"] - main["dP"]) - c1["injected"]
    if abs(c1_detected) / c1["P_initial"] > C1_TOL:
        return ("INVALID_C1_BLIND",
                f"positive control injected {c1['injected']:.3e} but the ledger detected "
                f"{c1['dP'] - main['dP']:.3e} against the matched main arm")
    if not c2["conserved_at_eps"]:
        return ("INVALID_C2_CRIES_WOLF",
                "recovery-suppressed arm did not conserve: the ledger itself is wrong, not the code")
    if main["total_recovered"] <= 0.0:
        return ("INVALID_RECOVERY_NEVER_RAN",
                "no ATP recovery occurred; this is the D14 scope failure and is NOT a pass")

    if main["conserved_at_eps"]:
        return "CONSERVED", "no leak detected at the registered tolerance"

    rel_resid = abs(main["residual_vs_prediction"]) / main["P_initial"]
    if rel_resid <= EPS_REL * 1e3:
        return ("LEAK_MATCHES_RECOVERY",
                "drift equals ATP regenerated: defect exactly as diagnosed, fix is stoichiometric")
    return ("LEAK_UNEXPLAINED",
            "drift is neither zero nor the ATP regenerated: a second source/sink exists. FIX NOTHING YET")


def main():
    print(__doc__.split("Run:")[0])
    print("=" * 88)
    print(f"REGISTERED TOLERANCE eps = {EPS_REL:.0e} relative   (pre-registration §3, fixed)")
    print("=" * 88)

    print("\n[MAIN] live Model6QuantumSynapse, full loop, recovery ON ...")
    main_r = run("main", seed=1000)

    print("[C1]   positive control — inject a KNOWN leak, ledger must report it ...")
    c1_r = run("C1_inject", inject_at=N_STEPS // 2, inject_amount=1e-4, seed=1000)

    print("[C2]   negative control — recovery suppressed, must conserve (reproduces D8/D14) ...")
    c2_r = run("C2_no_recovery", suppress_recovery=True, seed=1000)

    v, why = verdict(main_r, c1_r, c2_r)

    for r in (main_r, c1_r, c2_r):
        print("\n" + "-" * 88)
        print(f"  {r['label']}")
        print(f"    P_initial              : {r['P_initial']:.12e}")
        print(f"    P_final                : {r['P_final']:.12e}")
        print(f"    dP                     : {r['dP']:+.6e}   ({r['dP_relative']:+.3e} relative)")
        print(f"    ATP recovered          : {r['total_recovered']:.6e}")
        print(f"    injected (C1 only)     : {r['injected']:.6e}")
        print(f"    predicted dP if unfixed: {r['predicted_dP_if_unfixed']:.6e}")
        print(f"    residual vs prediction : {r['residual_vs_prediction']:+.6e}")
        print(f"    conserved at eps       : {r['conserved_at_eps']}")
        print(f"    clamp activations (C3) : {r['clamp_activations']}")
        print(f"    [defect 1] stale total gap        : {r['stale_total_gap']:.6e}")
        print(f"    [defect 1] J-coupling std          : {r['jcoupling_std']:.6e}")
        print(f"    [defect 1] corr(J, cum. consumed)  : {r['jcoupling_vs_consumed_r']}")

    print("\n" + "=" * 88)
    print(f"  VERDICT : {v}")
    print(f"  because : {why}")
    print("=" * 88)

    out = os.path.join(os.path.dirname(__file__), "phosphate_ledger_probe_results.json")
    with open(out, "w") as f:
        json.dump({"verdict": v, "why": why, "eps_rel": EPS_REL,
                   "main": main_r, "C1": c1_r, "C2": c2_r}, f, indent=2)
    print(f"\nresults -> {out}")

    # Exit non-zero on anything that is not a clean conserved pass, so "it ran" cannot be
    # mistaken for "it passed" (MO_MODEL6.md §2.3).
    return 0 if v == "CONSERVED" else 1


if __name__ == "__main__":
    sys.exit(main())
