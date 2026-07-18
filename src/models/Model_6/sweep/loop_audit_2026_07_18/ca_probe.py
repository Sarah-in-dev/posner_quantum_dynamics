import sys, numpy as np
M6="/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6/src/models/Model_6"
sys.path.insert(0,M6)
from model6_parameters import Model6Parameters
from calcium_system import CalciumChannels, CalciumParameters
p=Model6Parameters()
print("spine_calcium_feedback =", getattr(p,'spine_calcium_feedback','ABSENT'))
cp=p.calcium if hasattr(p,'calcium') else CalciumParameters()
pos=[(i%7,i//7) for i in range(50)]
for v in [-70e-3,-50e-3,-40e-3,-30e-3,-10e-3]:
    np.random.seed(1); ch=CalciumChannels(pos,cp); acc=[]
    for k in range(3000):
        ch.update_gating(0.001,v); acc.append(ch.get_open_fraction())
    print(f"  V={v*1e3:6.1f}mV  MEASURED mean ca_open={np.mean(acc):.4f}  max={np.max(acc):.4f}")
