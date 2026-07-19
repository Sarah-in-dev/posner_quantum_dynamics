#!/usr/bin/env python3
"""
PO-7 bit-identity fingerprint check.

Replicates the standing single-synapse fingerprint used by the PO-5 probes
(seed 31337, 200 steps, driven synapse, MT invaded):
    baseline = (V=1034, E=369740, mean_singlet=0.991922159684)

Run with no args to print the fingerprint of the CURRENT code state. Any physics
change that is opt-in and flag-OFF must reproduce this triple bit-for-bit, else INVALID.
"""
import sys, os, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

BASELINE = (1034, 369740, 0.991922159684)
T_SIM, DT, SEED = 1.0, 0.005, 31337

def fingerprint(seed=SEED):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    dp = net.synapses[0].dimer_particles
    rel = PresynapticRelease(seed=seed)
    for _ in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = set(d.id for d in ent)
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in ids and b in ids]
    mean_singlet = float(np.mean([d.singlet_probability for d in ent])) if ent else 0.0
    return len(ent), len(edges), mean_singlet

def main():
    V, E, ms = fingerprint()
    print(f"fingerprint: V={V} E={E} mean_singlet={ms:.12f}")
    print(f"baseline:    V={BASELINE[0]} E={BASELINE[1]} mean_singlet={BASELINE[2]:.12f}")
    ok = (V == BASELINE[0] and E == BASELINE[1]
          and abs(ms - BASELINE[2]) < 1e-11)
    print("BIT-IDENTICAL: PASS" if ok else "BIT-IDENTICAL: FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
