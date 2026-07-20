#!/usr/bin/env python3
"""
PO-7 UNIT 11 — THE SHARED PER-DIMER SPIN LEDGER (intra AND cross bonds spend the same 4 nuclei).

WHAT WAS MISSING
----------------
Unit 9 gave every dimer 4 x 31P spin slots and made INTRA bonds claim them
(dimer_particles._create_bond). But CROSS-synapse bonds are not created there — they are
created in multi_synapse_network._update_entanglement (the vectorised `form` mask) and in
_step_network_provenance (_prov_bonds). Those consumed NO spins, so a dimer could hold 4
intra bonds AND unlimited cross bonds. Monogamy was enforced per-synapse and violated
network-wide.

Unit 10 measured the consequence on this exact rig [commit 5ad2f97]:
    t=10.05  xbond=128   comps=210  largest_frac=0.209  nmulti=30
    t=11.05  xbond=121   comps=229  largest_frac=0.185  nmulti=31
    t=12.05  xbond=879   comps=19   largest_frac=0.959  nmulti=1     <-- collapse
    t=20.00  xbond=9256  comps=1    largest_frac=1.000  nmulti=1
i.e. shattering the intra cliques alone only DELAYS percolation, because cross bonds are free.

WHAT THIS PROBE MEASURES
  max TOTAL degree per dimer (intra + cross combined) -- must be <= 4 when ON;
  cross-bond frustration (bonds refused for want of a free spin at either endpoint);
  components, largest_frac, n_multi (components spanning >= 2 synapses).

THE REAL GATE (po7_bitident_check.py is NOT sufficient here)
  po7_bitident_check.py builds n_synapses=1 and drives net.synapses[0].step(...) directly,
  so tracker.step / _update_entanglement NEVER EXECUTE. It would pass even if cross-bond
  formation were broken outright. So MODE=off on this 7-synapse rig is the real regression
  gate for multi_synapse_network.py: with the flag OFF the cross-bond count, component count
  and largest_frac must be identical to the pre-change code, measured not assumed.

COMPOSED FROM (not rebuilt):
  po7_unit8_eta2_partition.py -- build() rig, the L.ETA-2 settings, JSONL-as-we-go persistence.
  po7_unit9_spin_resolved.py  -- the OFF/ON framing, degree/frustration accounting, JSON dump.

RIG (matches po7_unit8_eta2_partition.py exactly):
  7 synapses, pattern="linear", spacing_um=1.0, dt=5e-3, voltage -40e-3 sustained,
  glutamate via PresynapticRelease(seed=0), em_coupling_enabled, multi_synapse_enabled,
  fraction_P31=1.0, set_microtubule_invasion(True), disable_auto_commitment=True.
  net.step calls tracker.step every 10 steps with coupling_weights (multi_synapse_network.py
  :1159-1163) -- omitting coupling_weights forms zero cross bonds and the code warns.

Nothing is tuned. The Werner bound stays LOCKED at 0.5.
"""
import sys, os, json, logging, time
from collections import Counter
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

N_SYN, SPACING = 7, 1.0
DT = 5e-3
T_SIM = float(os.environ.get("PO7_U11_SECONDS", "12.0"))
VOLT = -40e-3
SEED = 0
PATTERN = "linear"
TRACKER_EVERY = 10
SPINS = 4

# "off" = flag OFF (the regression gate). "on" = shared ledger active.
MODE = os.environ.get("PO7_U11_MODE", "on").lower()
TAG = os.environ.get("PO7_U11_TAG", MODE)


def offpath_fingerprint():
    """MODE=offpath — the REAL flag-OFF regression gate for the code this unit edits.

    WHY THIS EXISTS. Two gates were tried first and both are unusable here:
      * po7_bitident_check.py drives net.synapses[0].step(...) on a 1-synapse rig, so
        tracker.step / _update_entanglement NEVER EXECUTE. It cannot see this change at all.
      * The full 7-synapse rig is NOT REPRODUCIBLE run-to-run under drive. Measured this
        session: two runs of IDENTICAL code diverge at t=8.70 (n_dimers 1520 vs 1580). This is
        a known, documented, OPEN issue -- coordination/HANDOFF_SARAH_2026-07-19_AM.md:20-22
        names three unseeded `np.random.default_rng()` calls (camkii_module.py:199,
        spine_plasticity_module.py:274, multi_synapse_network.py:1188) that seed from OS
        entropy and ignore the caller's seed, and records the same signature (cross_bonds
        1179 vs 1848 at one seed). Seeding them is an open PHYSICS decision, deliberately not
        taken, and spine_plasticity_module.py is out of scope for this unit.

    So this gate drives _update_entanglement DIRECTLY over synthetic dimers. Only np.random
    is consumed inside it, so it IS deterministic, and it exercises every path this unit
    edited: the vanished-dimer prune, the mt_invaded gate, the `form`/`diss` masks and the
    refresh write. Compare the printed fingerprint before and after the change; with the flag
    OFF it must be identical.
    """
    from multi_synapse_network import NetworkEntanglementTracker

    N_SYN_F, PER_SYN, STEPS, DT_F = 5, 12, 60, 0.02
    tr = NetworkEntanglementTracker()
    W = np.zeros((N_SYN_F, N_SYN_F))
    for i in range(N_SYN_F):
        for j in range(N_SYN_F):
            W[i, j] = np.exp(-abs(i - j) * 1.0 / 5.0)

    np.random.seed(4242)
    checks = []
    for step in range(STEPS):
        # Dimer turnover: population churns so the vanished-dimer prune actually fires.
        alive = []
        for s in range(N_SYN_F):
            for d in range(PER_SYN):
                if (step + s + d) % 7 == 0:
                    continue                      # this dimer is absent this step
                alive.append({
                    'global_id': (s, d), 'synapse_idx': s,
                    'P_S': 0.70 + 0.29 * ((step * 7 + s * 3 + d) % 10) / 10.0,
                    'eta': 0.0 if (s == 4 and step % 5 == 0) else 0.25,
                    'mt_invaded': not (s == 3 and 20 <= step < 30),   # exercises the gate
                })
        tr.all_dimers = alive
        tr._update_entanglement(DT_F, coupling_weights=W)
        if (step + 1) % 10 == 0:
            keys = sorted(tr.cross_synapse_bonds)
            fsum = sum(tr.cross_synapse_bonds[k] for k in keys)
            checks.append((step + 1, len(keys), round(fsum, 10),
                           hash(tuple(keys)) % (10 ** 12)))
    return checks


def build(spin_resolved):
    """Rig verbatim from po7_unit8_eta2_partition.py:67-87."""
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern=PATTERN, spacing_um=SPACING)
    net.initialize(Model6QuantumSynapse, params)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    if spin_resolved:
        for s in net.synapses:
            s.dimer_particles.spin_resolved = True
    return net


def sample(net):
    """One measurement of the partition + the ledger.

    Degree counts EVERY existing bond in both containers, not only edges above the Werner
    bound: the spin ledger governs bond EXISTENCE (a formed bond has consumed its two
    nuclei regardless of whether its fidelity clears 0.5 for connectivity purposes). So
    max_total_degree is the honest test of monogamy; the Werner-filtered counts below are
    the partition.
    """
    tr = net.entanglement_tracker
    ids = {d['global_id'] for d in tr.all_dimers}
    nd = len(ids)

    deg = Counter()
    n_intra = 0
    for (a, b) in tr.intra_synapse_bonds_cache:
        if a in ids and b in ids:
            deg[a] += 1; deg[b] += 1; n_intra += 1
    n_cross_all = 0
    for (a, b) in tr.cross_synapse_bonds:
        if a in ids and b in ids:
            deg[a] += 1; deg[b] += 1; n_cross_all += 1
    if tr.provenance_network:
        for (a, b) in tr._prov_bonds:
            if a in ids and b in ids:
                deg[a] += 1; deg[b] += 1

    degs = np.array([deg.get(i, 0) for i in ids], float) if ids else np.zeros(1)

    # Werner-thresholded cross edges = what actually counts for the partition.
    xb = sum(1 for (a, b), f in tr.cross_synapse_bonds.items()
             if f > tr.WERNER_ENTANGLEMENT_BOUND and a[0] != b[0])

    comps = tr._find_all_clusters()
    nmulti = sum(1 for c in comps if len({g[0] for g in c}) >= 2)
    sizes = sorted((len(c) for c in comps), reverse=True)
    largest = sizes[0] if sizes else 0

    intra_frustrated = sum(int(getattr(s.dimer_particles, '_spin_frustrated', 0))
                           for s in net.synapses)
    cross_frustrated = int(getattr(tr, '_cross_spin_frustrated', 0))

    return {
        'n_dimers': nd,
        'n_intra_bonds': n_intra,
        'n_cross_bonds_all': n_cross_all,
        'n_cross_bonds_werner': xb,
        'max_total_degree': float(degs.max()),
        'mean_total_degree': float(degs.mean()),
        'over_bound': int((degs > SPINS).sum()),
        'n_components': len(comps),
        'largest_comp': largest,
        'largest_frac': (largest / nd) if nd else 0.0,
        'n_multi': nmulti,
        'intra_frustrated': intra_frustrated,
        'cross_frustrated': cross_frustrated,
    }


def main():
    from presynaptic_release import PresynapticRelease

    if MODE == "offpath":
        print("=" * 96)
        print("PO-7 UNIT 11 — FLAG-OFF REGRESSION GATE (drives _update_entanglement directly)")
        print("=" * 96)
        rows = offpath_fingerprint()
        for step, n, fsum, kh in rows:
            print(f"  step {step:4d}   bonds={n:5d}   sum_F={fsum:16.10f}   keyhash={kh:012d}")
        digest = hash(tuple(rows)) % (10 ** 15)
        print(f"\n  OFFPATH FINGERPRINT: {digest:015d}")
        print("  (must be identical before and after the change, with the flag OFF)")
        with open(os.path.join(SWEEP_DIR, f'po7_unit11_offpath_{TAG}.json'), 'w') as f:
            json.dump({'rows': rows, 'digest': digest}, f, indent=2)
        return 0

    spin_on = (MODE == "on")
    trace_path = os.path.join(SWEEP_DIR, f'po7_unit11_trace_{TAG}.jsonl')
    open(trace_path, 'w').close()

    print("=" * 104)
    print(f"PO-7 UNIT 11 — shared per-dimer spin ledger (intra + cross)   MODE={MODE.upper()}  tag={TAG}")
    print("=" * 104)
    print(f"  N={N_SYN} spacing={SPACING}um pattern={PATTERN}  drive {VOLT*1e3:.0f}mV + glutamate"
          f"  T={T_SIM}s dt={DT} seed={SEED}")
    print(f"  spin_resolved={spin_on}  (OFF => this is the regression gate for "
          f"multi_synapse_network.py)\n")
    print(f"  {'t(s)':>7} {'dimers':>7} {'intra':>7} {'xall':>7} {'xwern':>7} {'maxdeg':>7} "
          f"{'over':>6} {'comps':>6} {'lgfrac':>7} {'nmulti':>6} {'ifrust':>9} {'xfrust':>9} {'s/step':>7}")

    np.random.seed(SEED)
    net = build(spin_on)
    rel = PresynapticRelease(seed=SEED)
    n_steps = int(round(T_SIM / DT))
    t0 = time.time()
    samples = []
    peak = {'max_total_degree': 0.0, 'over_bound': 0, 'largest_frac': 0.0,
            'n_multi': 0, 'n_cross_bonds_werner': 0}

    for k in range(n_steps):
        g = rel.step(0.95, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        if (k + 1) % TRACKER_EVERY:
            continue
        rec = sample(net)
        rec['t'] = round((k + 1) * DT, 4)
        samples.append(rec)
        for key in peak:
            peak[key] = max(peak[key], rec[key])
        with open(trace_path, 'a') as f:          # persist AS WE GO (L.PO5-3 scar)
            f.write(json.dumps(rec) + "\n")
        if (k + 1) % (TRACKER_EVERY * 20) == 0 or rec['t'] <= 0.05:
            print(f"  {rec['t']:7.3f} {rec['n_dimers']:7d} {rec['n_intra_bonds']:7d} "
                  f"{rec['n_cross_bonds_all']:7d} {rec['n_cross_bonds_werner']:7d} "
                  f"{rec['max_total_degree']:7.0f} {rec['over_bound']:6d} "
                  f"{rec['n_components']:6d} {rec['largest_frac']:7.3f} {rec['n_multi']:6d} "
                  f"{rec['intra_frustrated']:9d} {rec['cross_frustrated']:9d} "
                  f"{(time.time()-t0)/(k+1):7.3f}")
            sys.stdout.flush()

    final = samples[-1] if samples else {}
    print("\n" + "=" * 104)
    print("RESULT")
    print("=" * 104)
    print(f"  final t={final.get('t')}  dimers={final.get('n_dimers')}")
    print(f"    intra bonds            {final.get('n_intra_bonds')}")
    print(f"    cross bonds (all)      {final.get('n_cross_bonds_all')}")
    print(f"    cross bonds (F>0.5)    {final.get('n_cross_bonds_werner')}")
    print(f"    max TOTAL degree       {final.get('max_total_degree')}   (bound {SPINS})")
    print(f"    dimers over bound      {final.get('over_bound')}")
    print(f"    components             {final.get('n_components')}")
    print(f"    largest_frac           {final.get('largest_frac'):.4f}")
    print(f"    n_multi (>=2 synapses) {final.get('n_multi')}")
    print(f"    intra frustration      {final.get('intra_frustrated')}")
    print(f"    cross frustration      {final.get('cross_frustrated')}")
    print(f"\n  PEAK over run: max_total_degree={peak['max_total_degree']:.0f} "
          f"over_bound={peak['over_bound']} largest_frac={peak['largest_frac']:.4f} "
          f"n_multi={peak['n_multi']} xwerner={peak['n_cross_bonds_werner']}")

    if spin_on:
        ok = peak['max_total_degree'] <= SPINS and peak['over_bound'] == 0
        print(f"\n  MONOGAMY NETWORK-WIDE: {'SATISFIED' if ok else 'VIOLATED'} "
              f"(peak max total degree {peak['max_total_degree']:.0f} vs bound {SPINS})")
        print(f"  Unit 10 reference on this rig (intra-only ledger): "
              f"largest_frac 0.959 at t=12.05, n_multi=1")
        if final.get('n_cross_bonds_all', 0) == 0:
            print("  => STRUCTURALLY STARVED: no cross bonds survive the shared ledger at all.")
            print("     Dimers spend all four nuclei on intra bonds; nothing is left for the")
            print("     network scale. A real result, reported as-is — nothing was adjusted.")

    out = {'mode': MODE, 'tag': TAG,
           'config': {'n_syn': N_SYN, 'spacing_um': SPACING, 'dt': DT, 't_sim': T_SIM,
                      'volt_mV': VOLT * 1e3, 'seed': SEED, 'pattern': PATTERN,
                      'spin_resolved': spin_on},
           'final': final, 'peak': peak, 'n_samples': len(samples),
           'trace': os.path.basename(trace_path)}
    out_path = os.path.join(SWEEP_DIR, f'po7_unit11_results_{TAG}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.basename(out_path)} (+ trace {os.path.basename(trace_path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
