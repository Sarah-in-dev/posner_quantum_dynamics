#!/usr/bin/env python3
"""
CA1 Theta Burst — 6 traversals, 45s gaps — ANALYTICAL GAP + VALIDATION
========================================================================
Uses analytical fast-decay during inter-traversal gaps:
  - P_S decoherence decay (per-dimer T_eff)
  - Coherence-dependent dissolution (k_classical = 0.05 s⁻¹)
  - Particle removal for dissolved dimers
  - Bond cleanup for decoherent dimers (P_S < 0.5)
  - Stochastic disentanglement (no EM protection at rest)

Includes a 5s validation run comparing full-physics vs analytical.
"""

import sys
import os
import time
import logging
import numpy as np
import copy

# Suppress noise
logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

MODEL6_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL6_DIR)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork


# =============================================================================
# ANALYTICAL GAP
# =============================================================================

def analytical_gap(network, gap_duration_s, dt_sub=1.0, diagnostics=False):
    """
    Advance the network through a silent gap analytically.

    Physics during silence (V = -70 mV, no reward):
      1. P_S decoherence: per-dimer exponential decay toward 0.25
      2. Dissolution: concentration decays at k_diss = k_classical*(1 - singlet_excess)
      3. Particle removal: track concentration, remove lowest-coherence particles
      4. Bond cleanup: remove bonds involving P_S < 0.5 dimers
      5. Stochastic disentanglement: k_decohere = 0.01*(1 - P_Si*P_Sj)

    NOT computed (negligible during silence):
      - Calcium dynamics (already at baseline within ~2s)
      - ATP hydrolysis / phosphate production
      - EM field / tryptophan superradiance
      - New dimer formation (no calcium)
      - New bond formation (no EM field)
      - Network entanglement tracker O(n^2) recalc

    Sub-interval dt_sub controls accuracy of the coupling between
    P_S decay and dissolution rate. 1s is fine for 45s gaps.

    If diagnostics=True, returns a dict with per-stage counts.
    """
    P_THERMAL = 0.25
    K_CLASSICAL = 0.05  # s^-1, bare dissolution rate

    n_subs = int(np.ceil(gap_duration_s / dt_sub))
    actual_dt = gap_duration_s / n_subs

    # Diagnostics counters
    diag = None
    if diagnostics:
        pre_dimers = sum(len(s.dimer_particles.dimers) for s in network.synapses)
        pre_per_syn_bonds = sum(len(s.dimer_particles.entanglement_bonds)
                                for s in network.synapses)
        pre_net_bonds = len(network.entanglement_tracker.entanglement_bonds)
        # P_S distribution of all dimers pre-gap
        pre_ps = []
        for syn in network.synapses:
            for d in syn.dimer_particles.dimers:
                pre_ps.append(d.singlet_probability)
        # Count per-synapse bonds per dimer (connectivity)
        pre_bond_counts = {}
        for syn in network.synapses:
            for b in syn.dimer_particles.entanglement_bonds:
                pre_bond_counts[b.dimer_i] = pre_bond_counts.get(b.dimer_i, 0) + 1
                pre_bond_counts[b.dimer_j] = pre_bond_counts.get(b.dimer_j, 0) + 1

        diag = {
            "pre_dimers": pre_dimers,
            "pre_per_syn_bonds": pre_per_syn_bonds,
            "pre_net_bonds": pre_net_bonds,
            "pre_ps_min": float(np.min(pre_ps)) if pre_ps else 0,
            "pre_ps_max": float(np.max(pre_ps)) if pre_ps else 0,
            "pre_ps_mean": float(np.mean(pre_ps)) if pre_ps else 0,
            "total_dimers_dissolved": 0,
            "dissolved_bond_counts": [],  # per-syn bond count of each dissolved dimer
            "dissolved_ps_values": [],    # P_S of each dissolved dimer at removal
            "bonds_removed_step3": 0,     # per-syn bonds removed with dissolved dimers
            "bonds_removed_step4": 0,     # per-syn bonds removed for P_S < 0.5
            "bonds_removed_step5": 0,     # per-syn bonds removed stochastically
        }

    for sub_idx in range(n_subs):

        # ------------------------------------------------------------------
        # 1. Decay singlet probability per dimer (matches step_coherence)
        # ------------------------------------------------------------------
        for syn in network.synapses:
            frac_P31 = getattr(syn.params.environment, 'fraction_P31', 1.0)
            T_base = frac_P31 * 216.0 + (1.0 - frac_P31) * 0.4

            for dimer in syn.dimer_particles.dimers:
                j_spread = np.std(dimer.j_couplings_intra)
                j_mean = np.abs(np.mean(dimer.j_couplings_intra))
                spread_factor = 1.0 + 2.0 * j_spread / (j_mean + 0.1)
                template_factor = 0.7 if dimer.template_bound else 1.0
                T_eff = max(T_base / (spread_factor * template_factor), 0.1)

                decay = np.exp(-actual_dt / T_eff)
                noise = 1.0 + 0.01 * np.sqrt(actual_dt) * np.random.randn()
                P_excess = dimer.singlet_probability - P_THERMAL
                dimer.singlet_probability = float(np.clip(
                    P_THERMAL + P_excess * decay * noise, P_THERMAL, 1.0
                ))

        # ------------------------------------------------------------------
        # 2. Dissolution of concentration field
        # ------------------------------------------------------------------
        for syn in network.synapses:
            ps_vals = [d.singlet_probability for d in syn.dimer_particles.dimers]
            mean_ps = np.mean(ps_vals) if ps_vals else P_THERMAL

            # Update the chemistry module's knowledge of P_S
            syn.ca_phosphate.dimerization.set_mean_singlet_probability(mean_ps)

            singlet_excess = max(0.0, (mean_ps - P_THERMAL) / 0.75)
            k_diss = K_CLASSICAL * (1.0 - singlet_excess)

            # Exponential decay of concentration
            decay = np.exp(-k_diss * actual_dt)
            syn.ca_phosphate.dimerization.dimer_concentration *= decay
            syn.ca_phosphate.dimerization.trimer_concentration *= np.exp(
                -k_diss * 10.0 * actual_dt
            )
            # Clamp
            syn.ca_phosphate.dimerization.dimer_concentration = np.maximum(
                syn.ca_phosphate.dimerization.dimer_concentration, 0
            )
            syn.ca_phosphate.dimerization.trimer_concentration = np.maximum(
                syn.ca_phosphate.dimerization.trimer_concentration, 0
            )

        # ------------------------------------------------------------------
        # 3. Remove particles that exceed new concentration target
        # ------------------------------------------------------------------
        for syn in network.synapses:
            az_volume_L = 1e-17  # 0.01 um^3
            N_A = 6.022e23
            peak_conc = float(np.max(
                syn.ca_phosphate.dimerization.dimer_concentration
            ))
            target_count = int(round(peak_conc * az_volume_L * N_A))
            current_count = len(syn.dimer_particles.dimers)

            if current_count > target_count:
                n_remove = current_count - target_count
                sorted_dimers = sorted(
                    syn.dimer_particles.dimers,
                    key=lambda d: d.singlet_probability
                )
                for i in range(min(n_remove, len(sorted_dimers))):
                    d = sorted_dimers[i]
                    if diag is not None:
                        diag["total_dimers_dissolved"] += 1
                        diag["dissolved_ps_values"].append(d.singlet_probability)
                        # Count per-syn bonds this dimer participates in
                        n_bonds_this = sum(
                            1 for b in syn.dimer_particles.entanglement_bonds
                            if b.dimer_i == d.id or b.dimer_j == d.id
                        )
                        diag["dissolved_bond_counts"].append(n_bonds_this)
                        bonds_before = len(syn.dimer_particles.entanglement_bonds)
                    syn.dimer_particles.dimers.remove(d)
                    syn.dimer_particles._remove_all_bonds_for_dimer(d.id)
                    if diag is not None:
                        bonds_after = len(syn.dimer_particles.entanglement_bonds)
                        diag["bonds_removed_step3"] += (bonds_before - bonds_after)
                        bonds_before = bonds_after  # for next iteration

        # ------------------------------------------------------------------
        # 4. Remove bonds for decoherent dimers (P_S < 0.5)
        # ------------------------------------------------------------------
        if diag is not None:
            bonds_before_step4 = sum(len(s.dimer_particles.entanglement_bonds)
                                     for s in network.synapses)
        for syn in network.synapses:
            for dimer in syn.dimer_particles.dimers:
                if dimer.singlet_probability < 0.5:
                    syn.dimer_particles._remove_all_bonds_for_dimer(dimer.id)
        if diag is not None:
            bonds_after_step4 = sum(len(s.dimer_particles.entanglement_bonds)
                                    for s in network.synapses)
            diag["bonds_removed_step4"] += (bonds_before_step4 - bonds_after_step4)

        # ------------------------------------------------------------------
        # 5. Stochastic disentanglement (no EM protection at rest)
        # ------------------------------------------------------------------
        if diag is not None:
            bonds_before_step5 = sum(len(s.dimer_particles.entanglement_bonds)
                                     for s in network.synapses)
        for syn in network.synapses:
            # Build dimer lookup for this synapse
            dimer_map = {d.id: d for d in syn.dimer_particles.dimers}
            to_remove = []
            for bond in list(syn.dimer_particles.entanglement_bonds):
                di = dimer_map.get(bond.dimer_i)
                dj = dimer_map.get(bond.dimer_j)
                if di is None or dj is None:
                    to_remove.append((bond.dimer_i, bond.dimer_j))
                    continue
                coh_factor = di.singlet_probability * dj.singlet_probability
                k_decohere = 0.01 * (1.0 - coh_factor)
                p_disentangle = 1.0 - np.exp(-k_decohere * actual_dt)
                if np.random.random() < p_disentangle:
                    to_remove.append((bond.dimer_i, bond.dimer_j))

            for id_i, id_j in to_remove:
                syn.dimer_particles._remove_bond(id_i, id_j)
        if diag is not None:
            bonds_after_step5 = sum(len(s.dimer_particles.entanglement_bonds)
                                    for s in network.synapses)
            diag["bonds_removed_step5"] += (bonds_before_step5 - bonds_after_step5)

    # Refresh network entanglement tracker after dissolutions
    if diag is not None:
        post_dimers = sum(len(s.dimer_particles.dimers) for s in network.synapses)
        post_per_syn_bonds = sum(len(s.dimer_particles.entanglement_bonds)
                                 for s in network.synapses)
        net_bonds_before_refresh = len(network.entanglement_tracker.entanglement_bonds)
        diag["post_dimers"] = post_dimers
        diag["post_per_syn_bonds"] = post_per_syn_bonds
        diag["net_bonds_before_refresh"] = net_bonds_before_refresh

    # Rebuild all_dimers from current (post-dissolution) per-synapse dimer lists
    # With stable IDs (syn_idx, dimer.id), the bond set correctly identifies
    # which dimers still exist — dissolved dimers' IDs are simply absent.
    network.entanglement_tracker.collect_dimers(network.synapses, network.positions)

    if diag is not None:
        new_ids = {d['global_id'] for d in network.entanglement_tracker.all_dimers}
        surviving = {b for b in network.entanglement_tracker.entanglement_bonds
                     if b[0] in new_ids and b[1] in new_ids}
        diag["net_bonds_surviving_id_check"] = len(surviving)
        diag["net_bonds_pruned_by_dissolution"] = net_bonds_before_refresh - len(surviving)

    # Prune stale bonds + check coherence/coupling thresholds (dt=0 → no new bonds)
    network.entanglement_tracker._update_entanglement(0)

    if diag is not None:
        diag["net_bonds_after_update_ent"] = len(network.entanglement_tracker.entanglement_bonds)

    # Rebuild cluster metrics and update the cache that get_experimental_metrics() reads
    largest_cluster = network.entanglement_tracker._find_largest_cluster()
    entangled_ids = set()
    for bond in network.entanglement_tracker.entanglement_bonds:
        entangled_ids.add(bond[0])
        entangled_ids.add(bond[1])
    n = len(network.entanglement_tracker.all_dimers)
    network._network_entanglement = {
        'n_total_dimers': n,
        'n_entangled_network': len(largest_cluster),
        'n_bonds': len(network.entanglement_tracker.entanglement_bonds),
        'f_entangled': len(entangled_ids) / n if n > 0 else 0.0,
    }

    if diag is not None:
        diag["net_bonds_final_cache"] = network._network_entanglement['n_bonds']

    # Advance network time
    network.time += gap_duration_s

    # One full step to sync all internal state (calcium baseline, etc.)
    network.step(0.001, {"voltage": -70e-3, "reward": False})

    if diag is not None:
        diag["net_bonds_after_final_step"] = (
            network._network_entanglement.get('n_bonds', '?'))
        return diag


# =============================================================================
# HELPERS
# =============================================================================

def snapshot(network):
    """Capture topology metrics."""
    net = network.get_experimental_metrics()
    ps_mean = float(np.mean([
        s.get_mean_singlet_probability() for s in network.synapses
    ]))
    return {
        "total_dimers":         net.get("total_dimers", 0.0),
        "n_entangled_network":  net.get("n_entangled_network", 0),
        "total_bonds":          net.get("total_bonds", 0),
        "cross_synapse_bonds":  net.get("cross_synapse_bonds", 0),
        "mean_coherence":       net.get("mean_coherence", 0.0),
        "n_clusters":           net.get("n_clusters", 0),
        "ps_mean":              ps_mean,
        "committed":            net.get("network_committed", False),
    }


def run_burst_traversal(network, dt, n_theta_cycles=12, theta_period_s=0.125):
    """Run one traversal: n_theta_cycles bursts of 4 spikes at 100 Hz."""
    spike_period = 0.010
    depol_duration = 0.002
    spikes_per_burst = 4
    burst_active = spikes_per_burst * spike_period  # 40ms

    for cycle in range(n_theta_cycles):
        burst_steps = int(burst_active / dt)
        rest_steps = int((theta_period_s - burst_active) / dt)

        for s in range(burst_steps):
            t_in_spike = (s * dt) % spike_period
            v = -10e-3 if t_in_spike < depol_duration else -70e-3
            network.step(dt, {"voltage": v, "reward": False})

        for _ in range(rest_steps):
            network.step(dt, {"voltage": -70e-3, "reward": False})


def run_silence(network, dt, duration_s):
    """Run full-physics silence."""
    n_steps = int(duration_s / dt)
    for _ in range(n_steps):
        network.step(dt, {"voltage": -70e-3, "reward": False})


def print_snap(label, snap):
    print(f"  {label:>10}  dim={snap['total_dimers']:5.0f}  "
          f"ent={snap['n_entangled_network']:4d}  "
          f"bonds={snap['total_bonds']:5d}  "
          f"cross={snap['cross_synapse_bonds']:4d}  "
          f"coh={snap['mean_coherence']:.4f}  "
          f"P_S={snap['ps_mean']:.4f}  "
          f"clust={snap['n_clusters']:3d}", flush=True)


def make_network():
    """Create fresh 5-synapse clustered network."""
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0

    network = MultiSynapseNetwork(
        n_synapses=5, pattern="clustered", spacing_um=1.0,
    )
    network.initialize(Model6QuantumSynapse, params)
    for syn in network.synapses:
        syn.set_microtubule_invasion(True)
    network.disable_auto_commitment = True
    return network


# =============================================================================
# VALIDATION: 5s gap, full-physics vs analytical
# =============================================================================

def run_validation():
    """Compare 5s gap: full-physics vs analytical."""
    print("=" * 90, flush=True)
    print("VALIDATION: 5s gap — full-physics vs analytical", flush=True)
    print("=" * 90, flush=True)

    # Seed for reproducibility within each arm
    np.random.seed(42)
    net_full = make_network()
    np.random.seed(42)
    net_anal = make_network()

    # Identical stimulation (same seed)
    np.random.seed(123)
    run_burst_traversal(net_full, 0.001, 12, 0.125)
    snap_pre_full = snapshot(net_full)

    np.random.seed(123)
    run_burst_traversal(net_anal, 0.001, 12, 0.125)
    snap_pre_anal = snapshot(net_anal)

    print("After identical traversal:", flush=True)
    print_snap("full", snap_pre_full)
    print_snap("analytical", snap_pre_anal)

    # --- 5s gap: full physics ---
    np.random.seed(999)
    t0 = time.time()
    run_silence(net_full, 0.001, 5.0)
    t_full = time.time() - t0
    snap_post_full = snapshot(net_full)

    # --- 5s gap: analytical ---
    np.random.seed(999)
    t0 = time.time()
    analytical_gap(net_anal, 5.0, dt_sub=1.0)
    t_anal = time.time() - t0
    snap_post_anal = snapshot(net_anal)

    print(f"\nAfter 5s gap:", flush=True)
    print_snap("full", snap_post_full)
    print_snap("analytical", snap_post_anal)

    print(f"\n  Wall time — full: {t_full:.1f}s  analytical: {t_anal:.2f}s  "
          f"speedup: {t_full/max(t_anal,0.001):.0f}×", flush=True)

    # Compare key metrics
    print(f"\n  Delta dimers:  {abs(snap_post_full['total_dimers'] - snap_post_anal['total_dimers']):.0f}", flush=True)
    print(f"  Delta P_S:     {abs(snap_post_full['ps_mean'] - snap_post_anal['ps_mean']):.4f}", flush=True)
    print(f"  Delta bonds:   {abs(snap_post_full['total_bonds'] - snap_post_anal['total_bonds'])}", flush=True)
    print(flush=True)

    return True  # proceed regardless


# =============================================================================
# SINGLE SCENARIO (returns results dict)
# =============================================================================

def run_single(inter_traversal_s, verbose=False):
    """
    Run one 6-traversal scenario at the given inter-traversal interval.

    Returns dict with summary metrics and full snapshots list.
    If verbose=True, prints per-traversal detail and diagnostics.
    """
    N_TRAVERSALS = 6
    THETA_CYCLES = 12
    THETA_PERIOD_S = 0.125
    DOPAMINE_DELAY_S = 1.0
    SILENCE_S = 60.0
    DT = 0.001

    network = make_network()

    snapshots = []
    gap_survival_fracs = []  # post-gap dimers / pre-gap dimers for each gap

    for trav in range(N_TRAVERSALS):
        # --- Traversal (full physics) ---
        t0 = time.time()
        run_burst_traversal(network, DT, THETA_CYCLES, THETA_PERIOD_S)
        t_trav = time.time() - t0

        snap = snapshot(network)
        snap["traversal"] = trav
        snapshots.append(snap)

        if verbose:
            print(f"  Trav {trav}: {t_trav:6.1f}s wall | ", end="", flush=True)
            print(f"dim={snap['total_dimers']:.0f}  ent={snap['n_entangled_network']}  "
                  f"bonds={snap['total_bonds']}  cross={snap['cross_synapse_bonds']}  "
                  f"coh={snap['mean_coherence']:.4f}  clust={snap['n_clusters']}",
                  flush=True)

        # --- Gap (analytical) ---
        if trav < N_TRAVERSALS - 1:
            pre_gap_dimers = snap['total_dimers']
            t0 = time.time()
            gap_diag = analytical_gap(network, inter_traversal_s, dt_sub=1.0,
                                       diagnostics=verbose)
            t_gap = time.time() - t0

            snap_post = snapshot(network)
            snap_post["traversal"] = f"gap {trav}\u2192{trav+1}"
            snapshots.append(snap_post)

            survival = snap_post['total_dimers'] / max(pre_gap_dimers, 1)
            gap_survival_fracs.append(survival)

            if verbose:
                print(f"    Gap {trav}\u2192{trav+1}: {t_gap:5.1f}s wall | ", end="", flush=True)
                print(f"dim={snap_post['total_dimers']:.0f}  "
                      f"ent={snap_post['n_entangled_network']}  "
                      f"bonds={snap_post['total_bonds']}  "
                      f"P_S={snap_post['ps_mean']:.4f}", flush=True)

                if gap_diag:
                    print(f"      DIAG: {gap_diag['pre_dimers']}\u2192{gap_diag['post_dimers']} dimers "
                          f"({gap_diag['total_dimers_dissolved']} dissolved)", flush=True)
                    print(f"      Per-syn bonds: {gap_diag['pre_per_syn_bonds']}\u2192"
                          f"{gap_diag['post_per_syn_bonds']}  "
                          f"(step3={gap_diag['bonds_removed_step3']}, "
                          f"step4={gap_diag['bonds_removed_step4']}, "
                          f"step5={gap_diag['bonds_removed_step5']})", flush=True)
                    print(f"      Net tracker bonds: {gap_diag['net_bonds_before_refresh']} "
                          f"\u2192 prune dissolved \u2192 {gap_diag['net_bonds_surviving_id_check']} "
                          f"({gap_diag['net_bonds_pruned_by_dissolution']} pruned) "
                          f"\u2192 _update_ent \u2192 {gap_diag['net_bonds_after_update_ent']} "
                          f"\u2192 final_step \u2192 {gap_diag['net_bonds_after_final_step']}",
                          flush=True)
                    bc = gap_diag['dissolved_bond_counts']
                    ps = gap_diag['dissolved_ps_values']
                    if bc:
                        print(f"      Dissolved dimers: P_S range [{min(ps):.3f}, {max(ps):.3f}] "
                              f"mean={np.mean(ps):.3f}  "
                              f"per-syn bonds [min={min(bc)}, max={max(bc)}, "
                              f"mean={np.mean(bc):.1f}]", flush=True)
                    if gap_diag['net_bonds_before_refresh'] > 0:
                        loss_pct = 100.0 * gap_diag['net_bonds_pruned_by_dissolution'] / gap_diag['net_bonds_before_refresh']
                        dimer_loss_pct = 100.0 * gap_diag['total_dimers_dissolved'] / max(gap_diag['pre_dimers'], 1)
                        print(f"      Loss ratio: {dimer_loss_pct:.0f}% dimers \u2192 "
                              f"{loss_pct:.0f}% net bonds", flush=True)

    # --- Dopamine ---
    committed_before_da = network.network_committed
    analytical_gap(network, DOPAMINE_DELAY_S, dt_sub=0.1)
    network.step(DT, {"voltage": -70e-3, "reward": True})
    committed_at_da = network.network_committed

    # Capture measurement diagnostics (set during the reward step)
    meas = getattr(network.entanglement_tracker, '_last_measurement', {})
    n_clusters_measured = meas.get('n_clusters_measured', 0)
    n_clusters_singlet = meas.get('n_clusters_singlet', 0)
    per_syn_counts = [int(getattr(syn, '_committed_dimer_count', 0))
                      for syn in network.synapses]
    per_syn_drive = [float(getattr(syn, '_committed_memory_level', 0.0))
                     for syn in network.synapses]
    mean_drive = float(np.mean(per_syn_drive))

    snap_da = snapshot(network)
    snap_da["traversal"] = "post-DA"
    snapshots.append(snap_da)

    if verbose:
        print(f"  Post-DA:   dim={snap_da['total_dimers']:.0f}  "
              f"ent={snap_da['n_entangled_network']}  "
              f"bonds={snap_da['total_bonds']}  "
              f"committed={snap_da['committed']}", flush=True)
        print(f"    Measurement: {n_clusters_measured} clusters, "
              f"{n_clusters_singlet} singlet", flush=True)
        print(f"    Per-syn counts: {per_syn_counts}  "
              f"drive: [{', '.join(f'{d:.3f}' for d in per_syn_drive)}]  "
              f"mean={mean_drive:.3f}", flush=True)

    # --- Final silence ---
    analytical_gap(network, SILENCE_S, dt_sub=1.0)
    committed_after_silence = network.network_committed

    snap_final = snapshot(network)
    snap_final["traversal"] = "final"
    snapshots.append(snap_final)

    # Determine commitment timing
    if committed_at_da and not committed_before_da:
        commit_when = "at DA"
    elif committed_after_silence and not committed_at_da:
        commit_when = "in silence"
    elif committed_before_da:
        commit_when = "before DA"
    else:
        commit_when = "none"

    if verbose:
        if commit_when != "none":
            print(f"  ** Commitment fired {commit_when} **", flush=True)
        else:
            print(f"  ** No commitment **", flush=True)

    # Find trav 5 snapshot
    trav5_snap = next(s for s in snapshots if s["traversal"] == N_TRAVERSALS - 1)

    return {
        "inter_traversal_s": inter_traversal_s,
        "gap_survival_mean": float(np.mean(gap_survival_fracs)) if gap_survival_fracs else 0,
        "trav5_bonds": trav5_snap["total_bonds"],
        "trav5_entangled": trav5_snap["n_entangled_network"],
        "trav5_dimers": trav5_snap["total_dimers"],
        "postda_bonds": snap_da["total_bonds"],
        "postda_entangled": snap_da["n_entangled_network"],
        "committed": committed_after_silence or committed_at_da,
        "commit_when": commit_when,
        # Measurement diagnostics
        "n_clusters_measured": n_clusters_measured,
        "n_clusters_singlet": n_clusters_singlet,
        "per_syn_counts": per_syn_counts,
        "per_syn_drive": per_syn_drive,
        "mean_drive": mean_drive,
        "snapshots": snapshots,
    }


def print_full_table(result):
    """Print the full traversal+gap table for a result."""
    snapshots = result["snapshots"]
    print(f"\n  Full table for {result['inter_traversal_s']}s intervals:", flush=True)
    print(f"  {'-' * 95}", flush=True)
    print(f"  {'Phase':>12}  {'dimers':>7}  {'entangled':>9}  {'bonds':>6}  "
          f"{'cross':>6}  {'coh':>7}  {'P_S':>7}  {'clust':>5}", flush=True)
    print(f"  {'-' * 95}", flush=True)
    for snap in snapshots:
        trav = snap["traversal"]
        label = str(trav) if not isinstance(trav, str) else trav
        print(f"  {label:>12}  {snap['total_dimers']:>7.0f}  "
              f"{snap['n_entangled_network']:>9}  "
              f"{snap['total_bonds']:>6}  "
              f"{snap['cross_synapse_bonds']:>6}  "
              f"{snap['mean_coherence']:>7.4f}  "
              f"{snap['ps_mean']:>7.4f}  "
              f"{snap['n_clusters']:>5}", flush=True)
    print(f"  {'-' * 95}", flush=True)


# =============================================================================
# INTERVAL SWEEP
# =============================================================================

def run_interval_sweep():
    intervals = [10, 15, 30, 45, 60, 90]
    verbose_intervals = {60, 90}

    print("=" * 100, flush=True)
    print("INTER-TRAVERSAL INTERVAL SWEEP", flush=True)
    print("=" * 100, flush=True)
    print("6 traversals, 12 theta cycles, P31, 5 synapses, DA after last traversal", flush=True)
    print(f"Intervals: {intervals}s", flush=True)
    print(flush=True)

    results = []
    wall_start = time.time()

    for interval in intervals:
        t0 = time.time()
        verbose = interval in verbose_intervals
        if verbose:
            print(f"\n--- {interval}s interval (verbose) ---", flush=True)

        result = run_single(interval, verbose=verbose)
        wall = time.time() - t0
        result["wall_s"] = wall
        results.append(result)

        if not verbose:
            print(f"  {interval:3d}s interval: {wall:5.1f}s wall", flush=True)

    # === Summary table ===
    print(flush=True)
    print("=" * 120, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 120, flush=True)
    print(f"  {'gap_s':>5}  {'surv%':>6}  {'t5_bnd':>6}  {'DA_ent':>6}  "
          f"{'clust':>5}  {'sing':>4}  {'commit':>6}  {'mean_drv':>8}  "
          f"{'per_syn_counts':>30}  {'per_syn_drive':>35}",
          flush=True)
    print(f"  {'-' * 115}", flush=True)

    for r in results:
        counts_str = str(r['per_syn_counts'])
        drive_str = '[' + ', '.join(f'{d:.3f}' for d in r['per_syn_drive']) + ']'
        print(f"  {r['inter_traversal_s']:>5.0f}  "
              f"{r['gap_survival_mean']*100:>5.1f}%  "
              f"{r['trav5_bonds']:>6}  "
              f"{r['postda_entangled']:>6}  "
              f"{r['n_clusters_measured']:>5}  "
              f"{r['n_clusters_singlet']:>4}  "
              f"{'Yes' if r['committed'] else 'No':>6}  "
              f"{r['mean_drive']:>8.4f}  "
              f"{counts_str:>30}  "
              f"{drive_str:>35}",
              flush=True)
    print(f"  {'-' * 115}", flush=True)

    # Full tables for selected intervals
    for r in results:
        if r["inter_traversal_s"] in verbose_intervals:
            print_full_table(r)

    wall_total = time.time() - wall_start
    print(f"\nTotal wall clock: {wall_total:.0f}s ({wall_total/60:.1f} min)", flush=True)
    print("=" * 100, flush=True)


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    run_interval_sweep()
