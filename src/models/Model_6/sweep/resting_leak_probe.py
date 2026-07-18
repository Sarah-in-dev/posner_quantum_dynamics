"""Does a FULLY SILENT synapse (no glutamate at all) still cross invasion_threshold over the
full L·ETA-5 protocol duration? Measured, not extrapolated. 1 synapse, 272 s, no release."""
import sys, os, logging
import numpy as np
logging.disable(logging.INFO)
GA='/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/gifted-almeida-4e8a7b'
sys.path.insert(0, GA+'/src/models/Model_6'); sys.path.insert(0, GA+'/sweep')
from run_spatial_discovery import make_network, step_network_per_synapse

DT = 0.005
TOTAL = 8*14.0 + 7*20.0     # the L·ETA-5 protocol duration = 252 s
net = make_network(n_synapses=1, seed=7)
sp = net.synapses[0].spine_plasticity
thr = sp.params.actin.invasion_threshold
print(f"1 synapse, {TOTAL:.0f}s at rest (-70 mV), glutamate NEVER supplied.")
print(f"invasion_threshold = {thr}")
print(f"{'t(s)':>6} {'enl':>10} {'E_inv':>9} {'crossed?':>9}")
stim = [{'voltage': -70e-3, 'reward': False, 'glutamate': 0.0}]
crossed_at = None
for k in range(int(TOTAL/DT)):
    step_network_per_synapse(net, DT, stim)
    if k % int(20.0/DT) == 0:
        t = k*DT
        c = sp.actin_enlargement > thr
        if c and crossed_at is None: crossed_at = t
        print(f"{t:6.0f} {sp.actin_enlargement:10.5f} {sp.E_invasion:9.5f} {str(c):>9}", flush=True)
print()
print(f"FINAL enl={sp.actin_enlargement:.5f}  E_invasion={sp.E_invasion:.5f}")
if crossed_at is not None:
    print(f"=> AMENDMENT 4 IS NECESSARY BUT NOT SUFFICIENT: a fully silent synapse crosses")
    print(f"   invasion_threshold at t~{crossed_at:.0f}s from the RESTING VGCC LEAK alone.")
else:
    print(f"=> AMENDMENT 4 IS SUFFICIENT: fully silent stays below threshold for the whole")
    print(f"   protocol. E_invasion = {sp.E_invasion:.5f}.")
