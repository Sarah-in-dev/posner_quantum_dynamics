#!/usr/bin/env python3
"""
PO-7 Unit 1 part 2 — do the two `step_network_per_synapse` copies differ in BEHAVIOUR,
and does the resting_leak_probe tree-skew change what F-5 measured?

Pre-registered in coordination/leads/po7-construct-validity.md at commit 3b6bc7b, BEFORE
this file existed. Read the prereg before reading the verdict.

ARM A  two stepper copies, one tree, identical network + stimuli.
       Discriminators: backbone eta, cross-synapse bond count, gate-call count.
ARM B  one stepper, two trees (pinned vs the vestigial tree resting_leak_probe.py
       hardcodes). Discriminators: actin_enlargement, E_invasion, crossing time.

Every arm carries a NULL (must show zero divergence) and a POSITIVE CONTROL (must show
divergence). If a positive control does not fire, the arm reports ABORT and no verdict.
Read-only: this probe edits no model code and no other owner's file.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)

REPO = '/Users/sarahdavidson/posner_quantum_dynamics'
PINNED = os.path.join(REPO, '.claude/worktrees/nervous-hertz-7ccff6')
VESTIGIAL = os.path.join(REPO, '.claude/worktrees/gifted-almeida-4e8a7b')

SEED = 7
DT = 0.005


# =============================================================================
# VERDICT VOCABULARY — fixed before any run. Exactly one is returned per arm.
# =============================================================================
DIVERGENT = "DIVERGENT"
NO_MATERIAL_DIVERGENCE = "NO MATERIAL DIVERGENCE"
INCONCLUSIVE = "INCONCLUSIVE"
ABORT = "ABORT"


def _load(tree, modname, path_rel):
    """Import a module from a specific worktree, isolated under a unique key."""
    import importlib.util
    m6 = os.path.join(tree, 'src/models/Model_6')
    for p in (m6, os.path.join(tree, 'sweep'), tree):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(modname, os.path.join(tree, path_rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# ARM A — the two stepper copies
# =============================================================================

def _stimuli(t, n_syn):
    """Identical stimulus sequence for every arm. Includes a reward falling edge at
    t=0.30 s, which is what exercises the D19 one-shot-latch difference."""
    reward = (0.20 <= t < 0.30)
    return [{'voltage': -40e-3, 'reward': reward, 'glutamate': 1.0} for _ in range(n_syn)]


ARM_A_DURATION = 45.0   # s. Chosen from the model's own constants BEFORE the corrected run:
# P_met = P_BASAL + E_inv*ca_open*p_active_max; P_c = 21.51 fW, p_active_max = 60 fW,
# P_BASAL ~ 0.84 fW  =>  crossing needs E_inv*ca_open >~ 0.345, i.e. E_inv >~ 0.63 at
# ca_open ~ 0.55. model6-architecture:119 records E_invasion ~0.495 at 30 s heading to
# ~0.74 by ~45 s. The first run used 0.5 s, where eta is identically 0 in every arm and
# the backbone discriminators are degenerate -- that run ABORTED on its own control.


def _run_arm_a(step_fn, n_syn=2, duration=ARM_A_DURATION,
               skip_backbone=False, gate_on_reward_only=False):
    """Build an identical network, step it with `step_fn`, return the discriminators."""
    np.random.seed(SEED)
    RSD = sys.modules['rsd_pinned']
    net = RSD.make_network(n_synapses=n_syn, seed=SEED)

    # Instrument the gate WITHOUT editing model code: wrap the bound method.
    counter = {'gate_calls': 0}
    real_gate = net._evaluate_coordinated_gate

    def counting_gate(payload, *a, **kw):
        # POSITIVE CONTROL for the gate_calls discriminator: swallow non-reward calls,
        # which is precisely the RPFL copy's behaviour. Regime-independent, so unlike the
        # backbone control it fires at any duration.
        if gate_on_reward_only and not payload.get('reward', False):
            return None
        counter['gate_calls'] += 1
        return real_gate(payload, *a, **kw)
    net._evaluate_coordinated_gate = counting_gate

    if skip_backbone:
        # POSITIVE CONTROL for arm A: make the backbone update a no-op, which is
        # exactly the difference the RPFL copy has. The instrument must see this.
        net._update_backbone_field = lambda *a, **kw: None

    for k in range(int(duration / DT)):
        step_fn(net, DT, _stimuli(k * DT, n_syn))

    etas = [float(getattr(s, '_backbone_eta', 0.0)) for s in net.synapses]
    xbonds = len(getattr(net.entanglement_tracker, 'cross_synapse_bonds', {}))
    return {'eta_max': max(etas), 'etas': etas,
            'cross_bonds': xbonds, 'gate_calls': counter['gate_calls']}


def _verdict_a(x, y):
    """Returns DIVERGENT if ANY discriminator differs. Boring answer is reachable."""
    diffs = []
    if abs(x['eta_max'] - y['eta_max']) > 1e-12:
        diffs.append(f"eta_max {x['eta_max']:.6g} vs {y['eta_max']:.6g}")
    if x['cross_bonds'] != y['cross_bonds']:
        diffs.append(f"cross_bonds {x['cross_bonds']} vs {y['cross_bonds']}")
    if x['gate_calls'] != y['gate_calls']:
        diffs.append(f"gate_calls {x['gate_calls']} vs {y['gate_calls']}")
    return (DIVERGENT if diffs else NO_MATERIAL_DIVERGENCE), diffs


def arm_a():
    print("=" * 78)
    print("ARM A — two stepper copies, one tree, identical network and stimuli")
    print("=" * 78)

    rsd = sys.modules['rsd_pinned']
    rpfl = _load(PINNED, 'rpfl_pinned', 'src/models/Model_6/sweep/run_place_field_learning.py')

    # ---- NULL: same stepper twice. MUST be zero divergence. -------------------
    n1 = _run_arm_a(rsd.step_network_per_synapse)
    n2 = _run_arm_a(rsd.step_network_per_synapse)
    null_v, null_d = _verdict_a(n1, n2)
    print(f"NULL   (RSD vs RSD)          -> {null_v}  {null_d}")
    if null_v != NO_MATERIAL_DIVERGENCE:
        print(f"\n  ARM A = {INCONCLUSIVE}: the harness diverges against itself, so it is")
        print("  measuring stochastic noise, not stepper divergence. No verdict reported.")
        return INCONCLUSIVE, {'null': null_d}

    # ---- REGIME PRECONDITION -------------------------------------------------
    # The backbone discriminators (eta, cross_bonds) can only distinguish outcomes
    # where the backbone update actually produces eta > 0. If the reference arm is at
    # eta == 0, comparing it to another zero is NOT a null result -- it is no result.
    regime_ok = n1['eta_max'] > 0.0
    print(f"REGIME (RSD eta_max={n1['eta_max']:.6g}) -> "
          f"{'backbone discriminators LIVE' if regime_ok else 'backbone discriminators DEGENERATE'}")

    # ---- POSITIVE CONTROLS: one per discriminator. Each must FIRE. -----------
    pc_gate = _run_arm_a(rsd.step_network_per_synapse, gate_on_reward_only=True)
    gate_fired = pc_gate['gate_calls'] != n1['gate_calls']
    print(f"POSCTL-1 gate_calls   ({n1['gate_calls']} vs {pc_gate['gate_calls']}) -> "
          f"{'FIRED' if gate_fired else 'DID NOT FIRE'}")
    if not gate_fired:
        print(f"\n  ARM A = {ABORT}: the gate control did not fire.")
        return ABORT, {'poscontrol_gate': 'did not fire'}

    backbone_fired = None
    if regime_ok:
        pc_bb = _run_arm_a(rsd.step_network_per_synapse, skip_backbone=True)
        bb_v, bb_d = _verdict_a(n1, pc_bb)
        backbone_fired = (bb_v == DIVERGENT)
        print(f"POSCTL-2 backbone     -> {'FIRED' if backbone_fired else 'DID NOT FIRE'}  {bb_d}")
        if not backbone_fired:
            print(f"\n  ARM A = {ABORT}: the backbone control did not fire in a LIVE regime.")
            return ABORT, {'poscontrol_backbone': 'did not fire in live regime'}

    # ---- THE MEASUREMENT -----------------------------------------------------
    a_rsd = n1
    a_rpfl = _run_arm_a(rpfl.step_network_per_synapse)
    v, d = _verdict_a(a_rsd, a_rpfl)
    print(f"\n  RSD  : {a_rsd}")
    print(f"  RPFL : {a_rpfl}")
    print(f"\n  ARM A VERDICT = {v}")
    for line in d:
        print(f"      - {line}")
    if not regime_ok:
        print("\n  SCOPE LIMIT, stated rather than buried: eta stayed 0 in the reference arm,")
        print("  so the backbone discriminators are NEEDS MEASUREMENT, not verified-equal.")
        print("  Only the gate_calls discriminator is evidential in this run.")
    return v, {'rsd': a_rsd, 'rpfl': a_rpfl, 'diffs': d,
               'regime_ok': regime_ok, 'backbone_control_fired': backbone_fired,
               'backbone_discriminators': 'EVIDENTIAL' if regime_ok else 'NEEDS MEASUREMENT'}


# =============================================================================
# ARM B — the tree skew behind F-5
# =============================================================================

def _run_arm_b(tree, key, duration=120.0, threshold_scale=1.0):
    """Replicate resting_leak_probe.py's config exactly: 1 synapse, seed 7,
    -70 mV, glutamate never supplied. We do NOT edit that probe; we replicate it."""
    np.random.seed(SEED)
    RSD = _load(tree, key, 'sweep/run_spatial_discovery.py')
    net = RSD.make_network(n_synapses=1, seed=SEED)
    sp = net.synapses[0].spine_plasticity
    thr = sp.params.actin.invasion_threshold * threshold_scale

    stim = [{'voltage': -70e-3, 'reward': False, 'glutamate': 0.0}]
    crossed_at, traj = None, []
    for k in range(int(duration / DT)):
        RSD.step_network_per_synapse(net, DT, stim)
        if k % int(20.0 / DT) == 0:
            traj.append((k * DT, float(sp.actin_enlargement), float(sp.E_invasion)))
        if crossed_at is None and sp.actin_enlargement > thr:
            crossed_at = k * DT
    return {'threshold': float(thr), 'crossed_at': crossed_at,
            'final_enl': float(sp.actin_enlargement),
            'final_einv': float(sp.E_invasion), 'traj': traj}


def _verdict_b(x, y):
    diffs = []
    if x['crossed_at'] != y['crossed_at']:
        diffs.append(f"crossing {x['crossed_at']} vs {y['crossed_at']}")
    if abs(x['final_enl'] - y['final_enl']) > 1e-9:
        diffs.append(f"final_enl {x['final_enl']:.9g} vs {y['final_enl']:.9g}")
    if abs(x['final_einv'] - y['final_einv']) > 1e-9:
        diffs.append(f"final_E_inv {x['final_einv']:.9g} vs {y['final_einv']:.9g}")
    return (DIVERGENT if diffs else NO_MATERIAL_DIVERGENCE), diffs


def arm_b():
    print()
    print("=" * 78)
    print("ARM B — one stepper, two trees (the skew resting_leak_probe.py hardcodes)")
    print("=" * 78)

    pin = _run_arm_b(PINNED, 'rsd_b_pinned')
    print(f"  pinned    : crossed_at={pin['crossed_at']}  enl={pin['final_enl']:.6f}  E_inv={pin['final_einv']:.6f}")

    # ---- NULL: same tree twice at the same seed. MUST be identical. ----------
    pin2 = _run_arm_b(PINNED, 'rsd_b_pinned_null')
    null_v, null_d = _verdict_b(pin, pin2)
    print(f"  NULL   (pinned vs pinned)     -> {null_v}  {null_d}")
    if null_v != NO_MATERIAL_DIVERGENCE:
        print(f"\n  ARM B = {INCONCLUSIVE}: the model is nondeterministic at a fixed seed,")
        print("  so a cross-tree comparison cannot attribute a difference to the trees.")
        return INCONCLUSIVE, {'null': null_d}

    # ---- POSITIVE CONTROL: perturb the threshold; comparator must fire. ------
    pc = _run_arm_b(PINNED, 'rsd_b_pc', threshold_scale=0.5)
    pc_v, pc_d = _verdict_b(pin, pc)
    print(f"  POSCTL (threshold x0.5)       -> {pc_v}  {pc_d}")
    if pc_v != DIVERGENT:
        print(f"\n  ARM B = {ABORT}: positive control did not fire. No verdict reported.")
        return ABORT, {'poscontrol': 'did not fire'}

    # ---- THE MEASUREMENT ----------------------------------------------------
    ves = _run_arm_b(VESTIGIAL, 'rsd_b_vestigial')
    print(f"  vestigial : crossed_at={ves['crossed_at']}  enl={ves['final_enl']:.6f}  E_inv={ves['final_einv']:.6f}")
    v, d = _verdict_b(pin, ves)
    print(f"\n  ARM B VERDICT = {v}")
    for line in d:
        print(f"      - {line}")
    print()
    if v == NO_MATERIAL_DIVERGENCE:
        print("  => F-5 and mo-ruling-014 get STRONGER: the finding is robust to a")
        print("     dependency skew that could have invalidated it. The hardcoded")
        print("     cross-worktree path remains a hygiene defect with no results consequence.")
    else:
        print("  => F-5 NEEDS RE-READING, and so does mo-ruling-014, which was built on it.")
    return v, {'pinned': pin, 'vestigial': ves, 'diffs': d}


def arm_a_gate():
    """ARM A restricted to the ONE discriminator that survives the nondeterminism found
    on 2026-07-18: `gate_calls` is a pure count of call sites and consumes no RNG, so it
    is exact where eta/cross_bonds are distributional. Short run: the D19 difference is
    structural and needs only one reward falling edge, not a condensation regime."""
    print("=" * 78)
    print("ARM A (gate) — the RNG-independent discriminator, 0.5 s")
    print("=" * 78)
    rsd = sys.modules['rsd_pinned']
    rpfl = _load(PINNED, 'rpfl_pinned', 'src/models/Model_6/sweep/run_place_field_learning.py')
    D = 0.5

    base = _run_arm_a(rsd.step_network_per_synapse, duration=D)
    null = _run_arm_a(rsd.step_network_per_synapse, duration=D)
    print(f"NULL   RSD vs RSD          : {base['gate_calls']} vs {null['gate_calls']}")
    if base['gate_calls'] != null['gate_calls']:
        print(f"ARM A (gate) = {INCONCLUSIVE}: gate count not reproducible against itself.")
        return INCONCLUSIVE, {}

    pc = _run_arm_a(rsd.step_network_per_synapse, duration=D, gate_on_reward_only=True)
    print(f"POSCTL reward-only gate    : {base['gate_calls']} vs {pc['gate_calls']} -> "
          f"{'FIRED' if pc['gate_calls'] != base['gate_calls'] else 'DID NOT FIRE'}")
    if pc['gate_calls'] == base['gate_calls']:
        print(f"ARM A (gate) = {ABORT}: positive control did not fire.")
        return ABORT, {}

    rp = _run_arm_a(rpfl.step_network_per_synapse, duration=D)
    print(f"\n  RSD  gate_calls = {base['gate_calls']}")
    print(f"  RPFL gate_calls = {rp['gate_calls']}")
    v = DIVERGENT if base['gate_calls'] != rp['gate_calls'] else NO_MATERIAL_DIVERGENCE
    print(f"\n  ARM A (gate) VERDICT = {v}")
    return v, {'rsd_gate_calls': base['gate_calls'], 'rpfl_gate_calls': rp['gate_calls'],
               'poscontrol_gate_calls': pc['gate_calls']}


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if only == 'gate':
        _load(PINNED, 'rsd_pinned', 'sweep/run_spatial_discovery.py')
        v, d = arm_a_gate()
        print(json.dumps(d))
        sys.exit(0)
    _load(PINNED, 'rsd_pinned', 'sweep/run_spatial_discovery.py')
    va, da = arm_a() if only in ('both', 'a') else ('(not run)', {})
    vb, db = arm_b() if only in ('both', 'b') else ('(not run)', {})
    print()
    print("=" * 78)
    print(f"ARM A (two stepper copies) = {va}")
    print(f"ARM B (tree skew / F-5)    = {vb}")
    print("=" * 78)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'po7_stepper_divergence_results.json')
    with open(out, 'w') as f:
        json.dump({'arm_a': {'verdict': va, 'data': da},
                   'arm_b': {'verdict': vb, 'data': db}}, f, indent=2, default=str)
    print(f"results -> {out}")
