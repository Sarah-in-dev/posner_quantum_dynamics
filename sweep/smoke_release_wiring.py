"""Smoke test for the PresynapticRelease wiring in run_spatial_discovery.
Part 1: run the REAL run_trial loop briefly -> the wired loop runs without error,
        and release modules are being stepped (RRP depletes).
Part 2: deterministically drive one synapse at act=1.0 with glutamate present ->
        confirms release -> g_engaged -> NMDAR -> calcium -> E_invasion is live,
        regardless of whether the agent wandered into a field in Part 1.
Fast: small N, short drive, entanglement tracker not exercised in Part 2."""

import numpy as np
from run_spatial_discovery import make_network, run_trial
from spatial_environment import SpatialEnvironment, Agent

SEED = 42
N = 5
DT = 0.005

print("=== Part 1: real run_trial loop (short) ===")
env = SpatialEnvironment(n_features=N, seed=SEED)
net = make_network(n_synapses=env.n_features, seed=SEED)
agent = Agent()
agent.reset(env.size, np.random.default_rng(SEED))
run_trial(net, env, agent, 0, trial_time_budget=3.0)
print("run_trial completed OK (no exception).")
print("per-synapse release state after trial (n < N_max => that module stepped/released):")
for i, syn in enumerate(net.synapses):
    r = net.presynaptic_release[i]
    print(f"  syn {i}: N_max={r.N_max:.0f}  n={r.n:.2f}  pv0={r.pv0:.4f}  pv={r.pv:.4f}  Ps0={r.Ps0:.3f}  peak={r.peak_rate:.1f}Hz")

print("\n=== Part 2: deterministic single-synapse drive (syn 0 at act=1.0, 3 s) ===")
net2 = make_network(n_synapses=N, seed=SEED)
syn0 = net2.synapses[0]
rel0 = net2.presynaptic_release[0]
n_steps = int(3.0 / DT)
releases = 0
g_max = 0.0
for k in range(n_steps):
    for i, syn in enumerate(net2.synapses):
        act = 1.0 if i == 0 else 0.0
        glu = net2.presynaptic_release[i].step(act, DT)
        if act > 0.05:
            releases += int(glu > 0)
            syn.step(DT, {'voltage': -10e-3, 'reward': False, 'glutamate': glu})
    ch = getattr(getattr(syn0, 'calcium', None), 'channels', None)
    g_now = getattr(ch, 'g_engaged', None) if ch is not None else None
    if g_now is not None:
        g_max = max(g_max, g_now)
    if (k + 1) % int(1.0 / DT) == 0:
        e = getattr(syn0.spine_plasticity, 'E_invasion', None)
        e_str = f"{e:.4f}" if e is not None else "n/a"
        g_str = f"{g_now:.3f}" if g_now is not None else "n/a"
        print(f"  t={(k+1)*DT:.1f}s  releases={releases}  g_engaged(now)={g_str}  g_max={g_max:.3f}  E_invasion={e_str}  rel0.n={rel0.n:.2f}")
print("done.")
