"""
PO-2 · heavy-slot RUNNER — phosphate depletion bound at the grounded fraction.

PRE-REGISTERED as PREREG_PO2_PHOSPHATE.md AMENDMENT A2.6, before this ran.

THIS FILE COMPUTES NO VERDICT. It buys the trace; the verdict is derived offline by
`sweep/score_phosphate_depletion.py`. That split is MO gen-2's standing rule —
"compute buys the trace, the verdict is derived from the trace" — written after PO-5 lost
58 minutes of good physics to a scoring bug in a run whose intermediate was not persisted.
Shape composed from `src/models/Model_6/sweep/score_leta5.py`.

WHAT THIS MEASURES, AND WHAT IT CANNOT
──────────────────────────────────────────────────────────────────────────────────────
Ruling 021 asked for a run that "sees the pool actually bind, or state that you did not."
MEASURED CALIBRATION: 5.1 model-steps/s. Binding at the retired frac=0.02 needs 34.4 min
simulated = 413,000 steps = 22.3 h. NOT purchasable with one slot, and it would confirm a
configuration no longer in the code. At the GROUNDED value there is nothing to bind to —
PO2-9 measured no drain (slope +8.05e-05/s, t=+0.74, non-monotonic).

So the slot buys the thing that IS open, named by PO2-9 as its own limit: does "no drain"
survive a longer horizon? The deliverable is an UPPER BOUND on the grounded drain rate.

Run:  venv/bin/python -u sweep/phosphate_depletion_bound_probe.py [minutes]
"""

import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "models", "Model_6"))
sys.path.insert(0, HERE)

from model6_core import Model6QuantumSynapse          # noqa: E402
from model6_parameters import Model6Parameters        # noqa: E402
from phosphate_ledger_probe import ledger, DT, V_DEPOL, V_REST   # noqa: E402

SEED         = 1000
SAMPLE_EVERY = 200          # steps between persisted samples (~1 s simulated)
BUDGET_MIN   = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
OUTDIR       = os.path.join(HERE, "..", "src", "models", "Model_6", "results",
                            "phosphate_depletion")
OUTFILE      = os.path.join(OUTDIR, f"depletion_grounded_seed{SEED}.json")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(SEED)

    params = Model6Parameters()
    frac = params.phosphate.metabolic_to_structural_fraction
    syn = Model6QuantumSynapse(params)

    p0 = ledger(syn)
    P_struct_0 = float(np.sum(syn.atp.phosphate.phosphate_structural))
    t_start = time.time()
    deadline = t_start + BUDGET_MIN * 60.0

    print(f"grounded metabolic_to_structural_fraction = {frac}", flush=True)
    print(f"budget {BUDGET_MIN:.0f} min | sampling every {SAMPLE_EVERY} steps "
          f"({SAMPLE_EVERY*DT:.1f} s simulated) | seed {SEED}", flush=True)
    print(f"P_total(0) = {p0:.12e}   structural(0) = {P_struct_0:.9f}", flush=True)

    samples = []
    i = 0
    while time.time() < deadline:
        syn.step(DT, {"voltage": V_DEPOL if (i % 400) < 200 else V_REST})
        i += 1

        if i % SAMPLE_EVERY == 0:
            ph = syn.atp.phosphate
            P_now = ledger(syn)
            samples.append({
                "step": i,
                "t_sim": i * DT,
                "structural": float(np.sum(ph.phosphate_structural)),
                "metabolic": float(np.sum(ph.phosphate_metabolic)),
                "dimer": float(np.sum(syn.ca_phosphate.dimerization.dimer_concentration)),
                "trimer": float(np.sum(syn.ca_phosphate.dimerization.trimer_concentration)),
                "P_total": P_now,
                "cons_rel": (P_now - p0) / p0,
                "atp_recovered": float(syn.atp.hydrolysis.total_recovered),
            })
            # persist EVERY sample: the trace must survive the process (gen-2's rule)
            with open(OUTFILE, "w") as fh:
                json.dump({
                    "seed": SEED,
                    "fraction": frac,
                    "dt": DT,
                    "sample_every": SAMPLE_EVERY,
                    "P_total_initial": p0,
                    "structural_initial": P_struct_0,
                    "budget_min": BUDGET_MIN,
                    "wall_elapsed_s": time.time() - t_start,
                    "n_samples": len(samples),
                    "steps_done": i,
                    "t_sim_done": i * DT,
                    "samples": samples,
                }, fh, indent=1)

            s = samples[-1]
            print(f"  step {i:>7}  t={s['t_sim']:7.2f}s  struct={s['structural']:.9f}  "
                  f"({(s['structural']-P_struct_0)/P_struct_0*100:+.5f}%)  "
                  f"dimer={s['dimer']:.4e}  cons_rel={s['cons_rel']:+.2e}", flush=True)

    print(f"\nDONE — {i} steps, {i*DT:.1f} s simulated, "
          f"{(time.time()-t_start)/60:.1f} min wall", flush=True)
    print(f"trace -> {OUTFILE}", flush=True)
    print("NO VERDICT COMPUTED HERE. Score with: "
          "venv/bin/python sweep/score_phosphate_depletion.py", flush=True)


if __name__ == "__main__":
    main()
