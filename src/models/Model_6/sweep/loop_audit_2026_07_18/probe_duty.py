import sys, os, numpy as np
ROOT = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6"
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from spatial_environment import SpatialEnvironment, Agent

print("REALISTIC ACTIVATION DUTY CYCLE (10 seeds, run_spatial_discovery config)")
print("n_features=20, trial_time_budget=60 s, agent_dt=0.5 -> 120 agent steps/trial")
allmax, allmean = [], []
for seed in range(10):
    rng = np.random.default_rng(seed)
    env = SpatialEnvironment(n_features=20, seed=seed)
    agent = Agent(rng=rng); agent.reset(env.size, rng)
    n_steps = 120
    cnt = np.zeros(env.n_features)
    for _ in range(n_steps):
        acts = env.get_activations(agent.position)
        cnt += (acts > 0.05)
        agent.step(0.5, env, np.zeros(env.n_features))
    duty = cnt / n_steps
    allmax.append(duty.max()); allmean.append(duty.mean())
    print(f"  seed {seed}: max_duty={duty.max():.4f} mean_duty={duty.mean():.4f} "
          f"n_ever_active={int((duty>0).sum())}/{env.n_features}")
mx, mn = float(np.mean(allmax)), float(np.mean(allmean))
print(f"\n  across seeds: mean(max_duty)={mx:.4f}  mean(mean_duty)={mn:.4f}")
for label, d in [("best synapse", mx), ("mean synapse", mn)]:
    st = d * 60.0
    print(f"  {label}: spine time/trial = {st:.3f} s -> trials to reach "
          f"ampar_onset_delay=1800 s: {1800.0/max(1e-9, st):,.0f}")
