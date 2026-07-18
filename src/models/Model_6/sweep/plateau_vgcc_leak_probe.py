#!/usr/bin/env python3
"""
DOES THE PLATEAU MAKE eta GLOBAL? — the VGCC-leak test, and it decides §8's premise.

THE CONCERN (from the 2026-07-18 plateau literature pass)
---------------------------------------------------------
Jain 2024 establishes that a plateau ALONE does not potentiate: voltage-clamp to 0 mV
without glutamate uncaging gave 7 +/- 8% potentiation (n=10), vs 56.3 +/- 16% with
uncaging. Model 6 reproduces that correctly in the NMDAR pathway, because NMDAR opening
is glutamate-gated (`g_bind`) and only its CONDUCTION is voltage-dependent — no
glutamate, no NMDAR current, however depolarized the branch.

BUT Model 6's VGCC population (`is_nmda == False`) is gated by VOLTAGE ONLY. A
dendrite-wide plateau at -20 mV sits exactly at the VGCC Boltzmann midpoint
(`analytical_calcium_system.py:80`, V_half = -0.020 V), so it opens VGCCs at P_open ~ 0.5
at EVERY synapse on the branch — including ones receiving no input at all. With
`nmda_fraction = 0.5`, half the channel population is affected.

WHY IT MATTERS MORE THAN IT LOOKS
---------------------------------
VGCC calcium drives E_invasion, E_invasion drives r, and r drives eta. If a plateau
lifts eta at EVERY synapse on the branch, then **eta encodes "there was a plateau", not
"which synapses were driven"** — and eta is the only input-dependent channel into the
cross-synapse partition (k_cross ∝ sqrt(eta_i·eta_j)·w_spatial·P_product; w_spatial is
fixed geometry). The §8 input-selectivity phase assumes drive patterns the partition
THROUGH eta. If eta is plateau-global, that assumption fails and selectivity would have
to live in P_product (the dimer population, which only forms where NMDAR calcium
arrived) — a coherent story, but a DIFFERENT one from the one §8 is written against.

THE DESIGN
----------
One synapse driven (act=1.0), all others SILENT, at fixed geometry. Two conditions:
    NO PLATEAU   — baseline; any eta at silent synapses is the known, by-design
                   AGGREGATION spillover (p_met_agg = P_BASAL + coupling_weights @
                   p_active couples neighbours deliberately).
    PLATEAU      — same drive, plateau on.
The DIFFERENCE isolates the plateau's contribution, so aggregation spillover is not
mistaken for the leak.

READS
-----
per synapse: NMDAR vs VGCC open fraction (the mechanism), E_invasion, r, eta.
  * silent synapses gain VGCC but not NMDAR  => the leak is real and is VGCC-borne
  * silent synapses reach eta > 0 under plateau => eta IS plateau-global => §8 premise
    fails as written
  * silent synapses stay at eta = 0          => selectivity survives; premise holds

Nothing is tuned. This only reads.
"""
import sys, os, json
import numpy as np
import logging

logging.disable(logging.INFO)
for name in ['model6_core', 'multi_synapse_network', 'dimer_particles',
             'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
             'quantum_coherence', 'pH_dynamics', 'dopamine_system',
             'em_tryptophan_module', 'em_coupling_module', 'local_dimer_tubulin_coupling',
             'camkii_module', 'spine_plasticity_module', 'photon_emission_module',
             'photon_receiver_module', 'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(name).setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(MODEL6_DIR, '..', '..', '..'))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(REPO, 'sweep'))

from model6_parameters import (Model6Parameters, P_BASAL_W, compute_metabolic_power,
                               bose_einstein_occupation)
from model6_core import Model6QuantumSynapse, PLATEAU_VOLTAGE_V
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease

hbar = 1.0545718e-34
N_SYN = 7
DRIVEN = 3                # centre synapse — worst case for spillover, fairest test
T_S = 12.0
DT = 0.005
SEED = 11


def build():
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net, p


def r_eta(net, p):
    bp = p.dendritic_backbone
    P_c = bose_einstein_occupation(bp.omega_0) * hbar * (2*np.pi*bp.omega_0)**2 / bp.Q
    p_met = np.array([
        compute_metabolic_power(getattr(s.spine_plasticity, 'E_invasion', 0.0),
                                s.calcium.channels.get_open_fraction(),
                                bp.p_active_max_W) for s in net.synapses])
    agg = P_BASAL_W + net.coupling_weights @ (p_met - P_BASAL_W)
    rs = agg / P_c
    return rs, np.array([(r-1)/(r+1) if r >= 1 else 0.0 for r in rs])


def split_open(net):
    nm, vg = [], []
    for s in net.synapses:
        ch = s.calcium.channels
        m = getattr(ch, 'is_nmda', None)
        st = ch.state
        if m is None:
            nm.append(float('nan')); vg.append(float(np.mean(st)))
        else:
            nm.append(float(np.mean(st[m])) if m.any() else 0.0)
            vg.append(float(np.mean(st[~m])) if (~m).any() else 0.0)
    return np.array(nm), np.array(vg)


def run(plateau):
    np.random.seed(SEED)
    net, p = build()
    rel = [PresynapticRelease(seed=3000+i) for i in range(N_SYN)]
    acts = np.zeros(N_SYN); acts[DRIVEN] = 1.0
    nm_a, vg_a = np.zeros(N_SYN), np.zeros(N_SYN)
    n_log = 0

    for k in range(int(T_S/DT)):
        for i, syn in enumerate(net.synapses):
            v = -70e-3 + acts[i]*30e-3
            stim = {'voltage': v, 'reward': False}
            g = rel[i].step(acts[i], DT)
            if g:
                stim['glutamate'] = g
            if plateau:
                stim['plateau_potential'] = True
            syn.step(DT, stim)
        net._update_backbone_field()
        if k % 20 == 0:
            a, b = split_open(net)
            nm_a += a; vg_a += b; n_log += 1

    rs, etas = r_eta(net, p)
    e_inv = np.array([getattr(s.spine_plasticity, 'E_invasion', 0.0) for s in net.synapses])
    return dict(nmda=nm_a/n_log, vgcc=vg_a/n_log, e_inv=e_inv, r=rs, eta=etas)


def main():
    print("=" * 104)
    print("PLATEAU VGCC-LEAK PROBE — does a branch-wide plateau make eta GLOBAL?")
    print("=" * 104)
    print(f"  {N_SYN} synapses @1um, ONLY synapse {DRIVEN} driven (act=1.0), "
          f"rest SILENT. {T_S}s, dt={DT}, seed={SEED}")
    print(f"  plateau voltage = {PLATEAU_VOLTAGE_V*1e3:.0f} mV "
          f"(VGCC V_half = -20 mV -> P_open ~ 0.5 by construction)")
    print()

    off = run(plateau=False)
    on = run(plateau=True)

    print(f"{'syn':>4} {'state':>7} | {'NMDAR off':>10} {'NMDAR on':>9} | "
          f"{'VGCC off':>9} {'VGCC on':>8} | {'E_inv off':>10} {'E_inv on':>9} | "
          f"{'r off':>7} {'r on':>7} | {'eta on':>7}")
    print("-" * 104)
    for i in range(N_SYN):
        tag = "DRIVEN" if i == DRIVEN else "silent"
        print(f"{i:>4} {tag:>7} | {off['nmda'][i]:10.4f} {on['nmda'][i]:9.4f} | "
              f"{off['vgcc'][i]:9.4f} {on['vgcc'][i]:8.4f} | "
              f"{off['e_inv'][i]:10.4f} {on['e_inv'][i]:9.4f} | "
              f"{off['r'][i]:7.3f} {on['r'][i]:7.3f} | {on['eta'][i]:7.4f}")

    silent = [i for i in range(N_SYN) if i != DRIVEN]
    sil_eta_on = float(np.max(on['eta'][silent]))
    sil_eta_off = float(np.max(off['eta'][silent]))
    sil_nm_gain = float(np.mean(on['nmda'][silent] - off['nmda'][silent]))
    sil_vg_gain = float(np.mean(on['vgcc'][silent] - off['vgcc'][silent]))

    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    print(f"  silent-synapse NMDAR gain from plateau : {sil_nm_gain:+.4f}"
          f"   (expect ~0 — no glutamate, so no NMDAR opening)")
    print(f"  silent-synapse VGCC  gain from plateau : {sil_vg_gain:+.4f}"
          f"   (this IS the leak, if nonzero)")
    print(f"  max eta at a SILENT synapse, no plateau: {sil_eta_off:.4f}")
    print(f"  max eta at a SILENT synapse, plateau   : {sil_eta_on:.4f}")
    print(f"  eta at the DRIVEN synapse, plateau     : {on['eta'][DRIVEN]:.4f}")
    print()
    if sil_eta_on > 0:
        print("  => eta IS PLATEAU-GLOBAL. Silent synapses condense purely because the")
        print("     branch was depolarized. eta then encodes 'a plateau happened', NOT")
        print("     'which synapses were driven'. §8's premise — that drive patterns the")
        print("     partition THROUGH eta — FAILS AS WRITTEN. Selectivity would have to")
        print("     live in P_product (dimers form only where NMDAR calcium arrived).")
    else:
        print("  => eta stays SELECTIVE under a plateau: silent synapses do not condense.")
        print("     §8's premise holds. The VGCC leak exists at the channel level but")
        print("     does not propagate to condensation.")

    out = os.path.join(MODEL6_DIR, 'results', 'plateau_leak')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'leak_seed{SEED}.json'), 'w') as fh:
        json.dump({k: {kk: list(map(float, vv)) for kk, vv in d.items()}
                   for k, d in (('no_plateau', off), ('plateau', on))}, fh, indent=1)
    print(f"\n  trace -> {out}/leak_seed{SEED}.json")


if __name__ == "__main__":
    main()
