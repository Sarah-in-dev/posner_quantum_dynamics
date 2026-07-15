#!/usr/bin/env python3
"""
SOC experiment — STAGE 1 (cheap, analytic): does the Frohlich pump cross
threshold with honest drive in the two-cluster geometry?

The pump order parameter eta>0 iff r = p_met_agg / P_c >= 1, and
    p_met_agg[i] = P_BASAL_W + p_active_max * sum_j W[i,j] * (E_inv_j * ca_open_j)
depends ONLY on the drive d_j = E_inv_j*ca_open_j, p_active_max, and the coupling
matrix W[i,j]=exp(-dist_ij/lambda) (diag=1). None of this needs the O(n^2) dimer
sim. So we find the CRITICAL uniform drive d* where r crosses 1, cheaply, before
spending the ~8h full sim (Stage 2) to read betti1_cross in a regime we know is
reachable.

Honest-reachability bar: E_invasion tops out ~0.74 by 45s, ca_open ~1 during
bursts, so the sustained (time-averaged) drive d is realistically <~0.5. If d* is
above that, the pre-registered falsifier fires: the backbone-network story is
unsupported at honest values (do NOT crank Q/D/drive to rescue it).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model6_parameters import (
    Model6Parameters, P_BASAL_W, bose_einstein_occupation, hbar,
)

LAMBDA_UM = 5.0   # coupling_length_um default (NetworkEntanglementTracker + network)


def two_cluster_positions(per_cluster=4, sep_um=15.0, within_um=1.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((per_cluster, 3)); a[:, 0] = rng.normal(0.0,  within_um, per_cluster)
    b = np.zeros((per_cluster, 3)); b[:, 0] = rng.normal(sep_um, within_um, per_cluster)
    a[:, 1:] = rng.normal(0, 0.2, (per_cluster, 2))
    b[:, 1:] = rng.normal(0, 0.2, (per_cluster, 2))
    return np.vstack([a, b])


def coupling_matrix(positions, lam=LAMBDA_UM):
    n = len(positions)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j]); D[i, j] = D[j, i] = d
    W = np.exp(-D / lam); np.fill_diagonal(W, 1.0)
    return W, D


def main():
    bp = Model6Parameters().dendritic_backbone
    omega_ang = 2.0 * np.pi * bp.omega_0
    n_bar = bose_einstein_occupation(bp.omega_0)
    P_c = n_bar * hbar * omega_ang**2 / bp.Q
    p_active_max = bp.p_active_max_W

    print(f"P_c = {P_c*1e15:.2f} fW   P_BASAL = {P_BASAL_W*1e15:.2f} fW   "
          f"p_active_max = {p_active_max*1e15:.0f} fW   Q={bp.Q:.0f}  "
          f"omega0/2pi={bp.omega_0/1e6:.0f}MHz")

    pos = two_cluster_positions()
    W, D = coupling_matrix(pos)
    rowsum_cluster = W[0].sum()           # a within-cluster synapse row-sum
    print(f"\nTwo-cluster geometry (4+4, sep=15um, lambda={LAMBDA_UM}um):")
    print(f"  within-cluster W ~ {np.exp(-1.0/LAMBDA_UM):.3f}   "
          f"cross-cluster W ~ {np.exp(-15.0/LAMBDA_UM):.3f}")
    print(f"  row-sum for a clustered synapse = {rowsum_cluster:.3f}  "
          f"(single-synapse row-sum = 1.000)")

    def r_for_drive(d, w_rowsum):
        p_agg = P_BASAL_W + p_active_max * d * w_rowsum
        return p_agg / P_c

    # Critical uniform drive d* where r crosses 1, cluster vs isolated synapse.
    def d_star(w_rowsum):
        # r=1 => P_BASAL + p_active_max*d*rowsum = P_c
        return (P_c - P_BASAL_W) / (p_active_max * w_rowsum)

    d_star_cluster = d_star(rowsum_cluster)
    d_star_single = d_star(1.0)
    print(f"\nCRITICAL DRIVE d* = E_inv*ca_open at which the pump ignites (r=1):")
    print(f"  isolated synapse : d* = {d_star_single:.3f}")
    print(f"  clustered synapse: d* = {d_star_cluster:.3f}   "
          f"(collective coupling lowers it {d_star_single/d_star_cluster:.1f}x)")

    print(f"\nr and eta across uniform drive d (all synapses at d):")
    print(f"  {'d':>5} {'r_cluster':>10} {'eta_cluster':>12} {'r_single':>9}")
    for d in [0.0, 0.05, 0.094, 0.15, 0.25, 0.5, 0.74, 1.0]:
        rc = r_for_drive(d, rowsum_cluster); rs = r_for_drive(d, 1.0)
        eta = (rc - 1) / (rc + 1) if rc >= 1 else 0.0
        print(f"  {d:5.3f} {rc:10.3f} {eta:12.4f} {rs:9.3f}")

    reachable = 0.74 * 1.0   # E_inv_max * ca_open(burst)
    verdict = ("REACHABLE — pump ignites in-cluster below honest max drive"
               if d_star_cluster < reachable else
               "FALSIFIER FIRES — d* exceeds honest max; backbone story unsupported")
    print(f"\nHonest max sustained drive ~ E_inv_max(0.74)*ca_open ~ {reachable:.2f}")
    print(f"VERDICT: {verdict}")
    print(f"  => Stage 2 (full sim + betti1_cross) should drive the two clusters "
          f"to d >~ {max(d_star_cluster*1.5, 0.15):.2f} and hold.")


if __name__ == "__main__":
    main()
