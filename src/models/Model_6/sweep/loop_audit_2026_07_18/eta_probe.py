#!/usr/bin/env python3
"""DIAGNOSTIC: can eta clear threshold under the drives the learning drivers apply?
Read-only w.r.t. the repo. Recomputes r/eta with the SAME formula as
multi_synapse_network._update_backbone_field (lines 1108-1131) without editing it.
"""
import sys, os, types
import numpy as np

M6 = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6/src/models/Model_6"
sys.path.insert(0, M6)
sys.path.insert(0, os.path.join(M6, "sweep"))

from model6_parameters import (
    Model6Parameters, P_BASAL_W, bose_einstein_occupation, hbar,
    compute_metabolic_power,
)
from multi_synapse_network import MultiSynapseNetwork
from model6_core import Model6QuantumSynapse


def pc(bp):
    return bose_einstein_occupation(bp.omega_0) * hbar * (2*np.pi*bp.omega_0)**2 / bp.Q


def build(n, spacing, pattern="clustered", invaded=True):
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=n, pattern=pattern, spacing_um=spacing)
    net.initialize(Model6QuantumSynapse, params)
    if invaded:
        for s in net.synapses:
            s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def probe_r(net, P_c):
    bp = net.params.dendritic_backbone
    e = np.array([getattr(s.spine_plasticity, 'E_invasion', 0.0) for s in net.synapses])
    ca = np.array([s.calcium.channels.get_open_fraction() for s in net.synapses])
    pm = np.array([compute_metabolic_power(e[i], ca[i], bp.p_active_max_W)
                   for i in range(len(net.synapses))])
    agg = P_BASAL_W + net.coupling_weights @ (pm - P_BASAL_W)
    r = agg / P_c
    eta_native = np.array([(x-1)/(x+1) if x >= 1.0 else 0.0 for x in r])
    eta_set = np.array([getattr(s, '_backbone_eta', np.nan) for s in net.synapses])
    return e, ca, r, eta_native, eta_set


def run(label, n, spacing, seconds, dt, volt_fn, seed=0):
    np.random.seed(seed)
    net = build(n, spacing)
    P_c = pc(net.params.dendritic_backbone)
    rows = net.coupling_weights.sum(axis=1)
    print(f"\n===== {label} =====")
    print(f"  N={n} spacing={spacing}um  P_c={P_c*1e15:.2f}fW  "
          f"W row-sums: min={rows.min():.3f} max={rows.max():.3f}")
    print(f"  {'t(s)':>7} {'V(mV)':>7} {'ca_open_mx':>10} {'E_inv_mx':>9} "
          f"{'r_max':>8} {'eta_nat_mx':>10} {'eta_set_mx':>10}")
    n_steps = int(seconds/dt)
    log_every = max(1, n_steps//10)
    peak = dict(r=0.0, eta=0.0, e=0.0, ca=0.0)
    for k in range(n_steps):
        t = k*dt
        v = volt_fn(t)
        net.step(dt, {"voltage": v, "reward": False})
        e, ca, r, en, es = probe_r(net, P_c)
        peak['r'] = max(peak['r'], r.max()); peak['eta'] = max(peak['eta'], en.max())
        peak['e'] = max(peak['e'], e.max()); peak['ca'] = max(peak['ca'], ca.max())
        if k % log_every == 0 or k == n_steps-1:
            print(f"  {t:7.3f} {v*1e3:7.1f} {ca.max():10.4f} {e.max():9.4f} "
                  f"{r.max():8.4f} {en.max():10.4f} {np.nanmax(es):10.4f}")
    print(f"  PEAK over run: ca_open={peak['ca']:.4f} E_inv={peak['e']:.4f} "
          f"r={peak['r']:.4f} eta_native={peak['eta']:.4f}   "
          f"ETA EVER > 0? {'YES' if peak['eta'] > 0 else 'NO'}")
    return peak


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    dt = 0.001

    if which in ("all", "sd"):
        # spatial-discovery drive: -70mV -> -40mV, act=1 (max drive), sustained
        run("A. SPATIAL-DISCOVERY max drive (-40mV sustained, act=1.0)",
            n=6, spacing=1.0, seconds=3.0, dt=dt, volt_fn=lambda t: -40e-3)
        run("B. REST (-70mV)", n=6, spacing=1.0, seconds=1.0, dt=dt,
            volt_fn=lambda t: -70e-3)

    if which in ("all", "theta"):
        # theta-burst as in coherence_fragmentation_probe run_one, clamp NOT applied
        sp, dd, spikes, theta = 0.010, 0.002, 4, 0.125
        def tb(t):
            ph = t % theta
            return -10e-3 if (ph < spikes*sp and (ph % sp) < dd) else -70e-3
        run("C. THETA-BURST (-10mV bursts, native eta, clamp REMOVED) 0.08s = probe's DRIVE_S",
            n=7, spacing=1.0, seconds=0.08, dt=dt, volt_fn=tb)
        run("D. THETA-BURST sustained 5s (far beyond probe's 0.08s)",
            n=7, spacing=1.0, seconds=5.0, dt=dt, volt_fn=tb)

    if which in ("all", "hold"):
        run("E. TONIC -10mV held 5s (upper bound: max ca_open, no duty-cycle dilution)",
            n=7, spacing=1.0, seconds=5.0, dt=dt, volt_fn=lambda t: -10e-3)
