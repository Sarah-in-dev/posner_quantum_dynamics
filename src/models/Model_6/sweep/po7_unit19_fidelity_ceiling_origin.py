#!/usr/bin/env python3
"""
PO-7 UNIT 19 — the origin of the cross-synapse fidelity ceiling. Analytic + data, no seeding.

CLAIM (tested here): the F ≈ 0.815 ceiling on cross-synapse bonds is the SPATIAL WEIGHT for
nearest-neighbour synapses, NOT the metabolic/η budget an earlier guess attributed it to.

The cross-bond Werner fidelity in the model is
    F_ab = P_S^a · P_S^b · W_ij ,   W_ij = exp(-d_ij / λ),  λ = 5 µm   [from _update_entanglement]
Since P_S ≤ 1, F is bounded above by W_ij, and W_ij is largest for the CLOSEST synapse pair. For
a linear rig at 1 µm spacing the nearest-neighbour weight is exp(-1/5) = 0.8187 — so no
cross-synapse bond can exceed ~0.819 however strongly the backbone condenses. η enters the
FORMATION RATE (k ∝ √(η_i η_j)·W·P_S²), not the fidelity, so η sets HOW MANY bonds form, not how
strong they can be.

Consequence for the correlated-domain picture: per-bridge correlation p=(4F-1)/3 is capped by
SYNAPSE SPACING. exp(-spacing/λ) → F_max → p_max → the inter-synaptic correlation length. Spacing
is therefore the biological knob on domain size:
    1 µm → F≤0.819, p≤0.75 ;  2 µm → F≤0.670, p≤0.56 ;  3 µm → F≤0.549, p≤0.40.

This probe (a) prints the analytic ceilings, (b) reads the committed 16-run free ensemble
(results/po7_unit17_results.json) and confirms the observed max cross-F matches exp(-1/5).
No simulation is run here; nothing is seeded because nothing stochastic happens.
"""
import os, sys, json
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SWEEP_DIR))))
LAMBDA_UM = 5.0


def p_corr(F):
    return (4.0 * F - 1.0) / 3.0


def main():
    print("=" * 84)
    print("PO-7 UNIT 19 — the fidelity ceiling is the spatial weight, not η")
    print("=" * 84)
    print(f"  F_max(spacing) = exp(-spacing/λ),  λ = {LAMBDA_UM} µm  (P_S≤1 makes W the ceiling)\n")
    print(f"  {'spacing µm':>10} {'W=F_max':>9} {'p_max=(4F-1)/3':>15} {'ξ=-1/ln p':>11}")
    rows = []
    for d in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        W = float(np.exp(-d / LAMBDA_UM))
        p = p_corr(W)
        xi = (-1.0 / np.log(p)) if 0 < p < 1 else float('inf')
        rows.append({'spacing_um': d, 'F_max': W, 'p_max': p, 'xi_bonds': xi})
        print(f"  {d:>10.1f} {W:>9.4f} {p:>15.4f} {xi:>11.2f}")

    # results/ lives at the worktree root: sweep -> Model_6 -> models -> src -> <root>
    root = SWEEP_DIR
    for _ in range(4):
        root = os.path.dirname(root)
    ens = os.path.join(root, "results", "po7_unit17_results.json")
    print("\n  --- check against the committed 16-run free ensemble ---")
    if os.path.exists(ens):
        d = json.load(open(ens))
        allF = [f for r in d['runs'] for f in r['cross_F']]
        obs = max(allF)
        pred = float(np.exp(-1.0 / LAMBDA_UM))
        print(f"  observed max cross-F (16 runs, {len(allF)} edges): {obs:.4f}")
        print(f"  predicted nearest-neighbour ceiling exp(-1/5):     {pred:.4f}")
        print(f"  gap (attributable to P_S<1):                       {pred - obs:.4f}")
        verdict = "CONFIRMED — the ceiling is geometry (spatial weight), not η." \
            if abs(pred - obs) < 0.02 else "MISMATCH — re-examine."
        print(f"  => {verdict}")
    else:
        print(f"  (ensemble file not found at {ens}; analytic table above stands on its own)")

    out = {'lambda_um': LAMBDA_UM, 'ceilings_by_spacing': rows,
           'note': 'F_max = exp(-spacing/lambda); eta sets bond COUNT not fidelity'}
    with open(os.path.join(SWEEP_DIR, 'po7_unit19_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote po7_unit19_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
