#!/usr/bin/env python3
"""
L·ETA-5 — DOES `E_invasion` RATCHET ACROSS TRAVERSALS?

PRE-REGISTERED: docs/PREREG_L_ETA_5_RATCHET.md (commit 2084960, BEFORE this file).
Thresholds, the null arm, and the verdict branches are fixed there; this file implements
them and must not diverge. Read it first.

THE QUESTION
------------
L·ETA-3 measured r = 0.0768 in a live trial (13x short) and read off the trace that
E_invasion "is exactly 0 for the first ~34 s then climbs 0.011->0.052->0.075 and is STILL
RISING at trial end". E_invasion is an actin integrator; a navigating agent gives each
feature only a brief transient. Does the integrator RETAIN enough between traversals that
repeated passes accumulate?

The prediction comes from the GROUNDED constant (tau_extrude = 180 s, Honkura 2008,
spine_plasticity_module.py:109) and not from the inherited k_polymerization_max:

    rho_pred = exp(-GAP_S / tau_extrude) = exp(-20/180) = 0.8948

WHY THIS DOES NOT CALL analytical_gap
-------------------------------------
run_spatial_discovery.py:55-78 `analytical_gap()` does NOT advance spine plasticity —
actin appears neither in its computed list nor in its "NOT computed" list. Calling it
would freeze actin across every gap and report 100% retention: a clean-looking ratchet
that is a STOPPED CLOCK, reading as confirmation of tau_extrude. This probe steps real
physics through the gap instead. That is a stated limit (the silence model differs from
the shipped experiment's) and is declared in the pre-registration, §3.

WHAT IS SCORED (and what deliberately is not)
---------------------------------------------
The retention fraction is scored on `actin_enlargement`, NOT on E_invasion. :411-412
subtracts invasion_threshold before normalizing, so the exponential decay in the state
variable is NOT an exponential in E_invasion. E_invasion is reported, never scored for
retention.

HONEST BY CONSTRUCTION
----------------------
The positive control must fire before any verdict is read (the L·ETA-4 vacuous-verdict
scar). The null arm cannot show the effect by construction, and a ratchet there VOIDS the
measurement. rho >= 0.99 with the extrusion gate open is reported as a red flag that this
probe's own gap is not stepping — not as a strong positive. Nothing here is tuned; it
only reads. No constant is written.
"""
import sys, os, json, math, time
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

from run_spatial_discovery import (make_network, activations_to_stimuli,
                                   step_network_per_synapse)
from spatial_environment import SpatialEnvironment

# Reuse the L·ETA-3 read-only instruments verbatim — same r, same attribution factors.
from eta_in_live_trial import r_eta_per_synapse, cross_topology

# ---------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS (docs/PREREG_L_ETA_5_RATCHET.md §4-§6). Do not retune.
# ---------------------------------------------------------------------------
N_FEATURES     = 12        # inherited from L·ETA-3 (sets the coupling row-sums)
SEED           = 7         # inherited from L·ETA-3
PHYSICS_DT     = 0.005
AGENT_DT       = 0.5
AGENT_SPEED    = 0.2       # units/s, shipped Agent default (spatial_environment.py:106)
N_TRAVERSALS   = 8
GAP_S          = 20.0
HALF_PATH      = 1.4       # units either side of centre -> 2.8 u -> 14.0 s per traversal
ACT_FLOOR      = 0.05      # the shipped activation floor
TAU_EXTRUDE    = 180.0     # READ from spine_plasticity_module.py:109 (asserted at runtime)

K_STAB         = 0.02      # k_stabilization_max, spine_plasticity_module.py:99
RHO_PRED       = math.exp(-GAP_S / TAU_EXTRUDE)   # 0.8948 — UNCOMMITTED branch ONLY


def rho_predicted(conf, gap=GAP_S):
    """PREREG AMENDMENT 2. The gap drains actin_enlargement by BOTH paths at :389-390:
    extrusion (1-conf)/tau_extrude AND retention k_stab*conf. E_invasion reads
    actin_enlargement alone (:412), so a COMMITTED spine drains ~3.54x FASTER, not slower.
    The original single number 0.8948 was the uncommitted branch mis-stated as general."""
    drain = K_STAB * conf + (1.0 - conf) / TAU_EXTRUDE
    return math.exp(-drain * gap)

# Verdict thresholds — FIXED IN THE PRE-REGISTRATION, §6.
RHO_ARTIFACT_MIN   = 0.99   # >= this with conf open => gap not stepping
CONF_LATCH_MIN     = 0.05   # >= this => confinement legitimately gates extrusion off
RHO_BAND           = (0.80, 0.95)   # SUPERSEDED by RATIO_BAND (AMENDMENT 2)
RATIO_BAND         = (0.89, 1.07)   # = RHO_BAND / 0.8948, carried over not re-invented
GAIN_CONFIRM_MIN   = 2.0    # peak_r[N]/peak_r[1]
GAIN_FALSIFY_MAX   = 1.2
RHO_FALSIFY_MAX    = 0.5

# --- PREREG AMENDMENT 4 (MO rotation 001, ruling 007 option 1) -----------------------
# The L·ETA-5 null was VOID because zeroing activation does NOT silence a synapse:
# PresynapticRelease.step uses rate = baseline_rate + a*peak_rate with
# BASELINE_RATE_HZ = 0.5 (presynaptic_release.py:65,124), so act=0.0 still releases
# (~0.2 Hz, full amplitude). The null reached E_invasion = 0.4507 and OUT-GAINED the
# drive arm (7.46x vs 5.65x).
#
# SUPPRESS_SPONTANEOUS suppresses release at the TARGET in the null arm entirely, so the
# control cannot receive glutamate by any path. It changes the NULL arm only; the drive
# arm is bit-identical to the scored L·ETA-5 run.
#
# Registered, NOT RUN. The re-run is gated on Sarah (MO_MODEL6.md §3 hard stop).
SUPPRESS_SPONTANEOUS = True   # null arm: target releases NOTHING


def pick_target_and_heading(env):
    """Choose the feature with the clearest straight-line traversal, deterministically.

    Target = the feature whose nearest neighbour is furthest away. Heading = whichever of
    36 candidate directions maximizes the minimum distance from the traversal path to any
    OTHER feature centre, so the pass drives the target and not its neighbours.
    """
    pos = env.feature_positions
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    target = int(np.argmax(d.min(axis=1)))
    c = pos[target]
    others = np.delete(pos, target, axis=0)

    best_head, best_clear = 0.0, -np.inf
    for th in np.linspace(0.0, np.pi, 36, endpoint=False):
        u = np.array([np.cos(th), np.sin(th)])
        ts = np.linspace(-HALF_PATH, HALF_PATH, 60)
        path = c[None, :] + ts[:, None] * u[None, :]
        clear = np.min(np.linalg.norm(path[:, None, :] - others[None, :, :], axis=-1))
        if clear > best_clear:
            best_clear, best_head = clear, float(th)
    return target, best_head, float(best_clear)


def pick_park_position(env):
    """The point on a grid maximizing distance to every feature — the gap position.

    The agent is parked here during gaps so all activations sit below the floor and the
    gap is genuinely silent. The achieved max activation is measured and recorded.
    """
    g = np.linspace(0.2, env.size - 0.2, 60)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dmin = np.min(np.linalg.norm(pts[:, None, :] - env.feature_positions[None, :, :],
                                 axis=-1), axis=1)
    best = pts[int(np.argmax(dmin))]
    return best, float(np.max(env.get_activations(best)))


def step_physics(network, env, position, n_agent_steps, drive_target, target,
                 collect=None):
    """Advance the model for n_agent_steps at a FIXED agent position.

    drive_target=False zeroes the target's activation (the null arm): it holds the
    synapse below the 0.05 floor so voltage stays at -70 mV rest and f_CaM ~ 0.
    """
    phys_per = int(AGENT_DT / PHYSICS_DT)
    peak_r = 0.0
    max_glu = 0.0
    max_act = 0.0
    n_release = 0
    for _ in range(n_agent_steps):
        acts = env.get_activations(position)
        if not drive_target:
            acts = acts.copy()
            acts[target] = 0.0
        stimuli = activations_to_stimuli(acts)
        # PREREG AMENDMENT A1.1: release is stepped PER PHYSICS STEP, matching the shipped
        # run_spatial_discovery.run_trial:434-441. The L·ETA-3 harness stepped it once per
        # AGENT step, removing ~99% of the Bernoulli release opportunities and starving the
        # NMDARs — the ERR-2 class. Do not "simplify" this back out of the inner loop.
        for _ in range(phys_per):
            for i in range(len(network.synapses)):
                g = network.presynaptic_release[i].step(acts[i], PHYSICS_DT)
                # AMENDMENT 4: in the null arm the target must be TRULY silent. Stepping
                # the release object still advances its RRP/facilitation state (so the
                # arms stay comparable), but the cleft event is discarded.
                if (not drive_target) and i == target and SUPPRESS_SPONTANEOUS:
                    g = 0.0
                stimuli[i]['glutamate'] = g
                if i == target and g:
                    max_glu = max(max_glu, float(g))
                    n_release += 1
            step_network_per_synapse(network, PHYSICS_DT, stimuli)
        rs, etas, P_c, e_inv, ca_op = r_eta_per_synapse(network)
        peak_r = max(peak_r, float(rs[target]))
        max_act = max(max_act, float(acts[target]))
        if collect is not None:
            collect.append(dict(r_target=float(rs[target]),
                                eta_target=float(etas[target]),
                                E_inv=float(e_inv[target]),
                                ca_open=float(ca_op[target])))
    return peak_r, max_glu, max_act, n_release


def spine_state(network, target):
    sp = network.synapses[target].spine_plasticity
    return (float(sp.actin_enlargement), float(sp.E_invasion),
            float(sp.confinement), float(sp.actin_stable))


def run_arm(arm, outdir):
    """One arm. arm='drive' or 'null'."""
    drive = (arm == 'drive')
    np.random.seed(SEED)
    env = SpatialEnvironment(n_features=N_FEATURES, seed=SEED)
    network = make_network(n_synapses=N_FEATURES, seed=SEED)

    # Assert the constant this whole prediction rests on is what we read on disk.
    tau_live = float(network.synapses[0].spine_plasticity.params.actin.tau_extrude)
    assert abs(tau_live - TAU_EXTRUDE) < 1e-9, (
        f"tau_extrude changed under the pre-registration: {tau_live} != {TAU_EXTRUDE}")

    target, heading, clearance = pick_target_and_heading(env)
    park, park_act = pick_park_position(env)
    u = np.array([np.cos(heading), np.sin(heading)])
    rowsum = network.coupling_weights.sum(axis=1)

    path_len = 2.0 * HALF_PATH
    trav_s = path_len / AGENT_SPEED
    n_trav_steps = int(round(trav_s / AGENT_DT))
    n_gap_steps = int(round(GAP_S / AGENT_DT))

    print(f"[{arm}] target feature {target}, heading {heading:.3f} rad, "
          f"path clearance from other features {clearance:.2f} u")
    print(f"[{arm}] traversal {trav_s:.1f}s ({n_trav_steps} agent steps), "
          f"gap {GAP_S:.1f}s ({n_gap_steps} steps), {N_TRAVERSALS} traversals")
    print(f"[{arm}] gap park position {park.round(2)}, max activation there "
          f"{park_act:.4f} (floor {ACT_FLOOR})")
    print(f"[{arm}] coupling row-sums: min {rowsum.min():.2f} median "
          f"{np.median(rowsum):.2f} max {rowsum.max():.2f}")
    print(f"[{arm}] tau_extrude asserted live = {tau_live} s -> rho_pred = {RHO_PRED:.4f}")
    print()
    hdr = (f"{'trav':>5} {'enl_start':>10} {'enl_end':>9} {'rho':>7} {'peak_r':>9} "
           f"{'E_inv_end':>10} {'conf':>7} {'max_act':>8} {'max_glu':>8} {'secs':>7}")
    print(hdr)
    print("-" * len(hdr))

    traversals, prev_enl_end = [], None
    glu_assert_value, glu_assert_count = None, None
    gap_samples = []
    t0 = time.time()

    for n in range(1, N_TRAVERSALS + 1):
        enl_start, einv_start, conf_start, stable_start = spine_state(network, target)
        rho = (enl_start / prev_enl_end) if (prev_enl_end and prev_enl_end > 0) else None

        # --- traversal: agent walks the straight path through the feature centre ---
        peak_r, max_glu, max_act, n_rel = 0.0, 0.0, 0.0, 0
        for k in range(n_trav_steps):
            s = -HALF_PATH + (k + 0.5) * (path_len / n_trav_steps)
            position = env.feature_positions[target] + s * u
            pr, mg, ma, nr = step_physics(network, env, position, 1, drive, target)
            peak_r = max(peak_r, pr); max_glu = max(max_glu, mg)
            max_act = max(max_act, ma); n_rel += nr

        enl_end, einv_end, conf_end, stable_end = spine_state(network, target)

        # ERR-2 GUARD: glutamate must actually reach the synapse in the driven arm.
        if drive and n == 1:
            glu_assert_value = max_glu
            glu_assert_count = n_rel

        # --- gap: REAL physics, at a parked position. analytical_gap is NOT called. ---
        # A1.2 descriptive: sample enlargement through the gap so the late-gap decay
        # constant can be read separately from the early calcium-tail phase.
        gap_samples = []
        if n < N_TRAVERSALS:
            for _ in range(n_gap_steps):
                step_physics(network, env, park, 1, drive, target)
                gap_samples.append(spine_state(network, target)[0])

        row = dict(traversal=n, enl_start=enl_start, enl_end=enl_end,
                   rho_into_this=rho, peak_r=peak_r,
                   E_inv_start=einv_start, E_inv_end=einv_end,
                   conf_start=conf_start, conf_end=conf_end,
                   stable_end=stable_end, max_act=max_act, max_glu=max_glu,
                   n_release_events=n_rel, gap_enl_samples=gap_samples,
                   elapsed_s=time.time() - t0)
        traversals.append(row)

        print(f"{n:5d} {enl_start:10.5f} {enl_end:9.5f} "
              f"{(f'{rho:7.4f}' if rho is not None else '      -')} {peak_r:9.5f} "
              f"{einv_end:10.5f} {conf_end:7.4f} {max_act:8.4f} {max_glu:8.4f} "
              f"{time.time()-t0:7.1f}")
        sys.stdout.flush()

        prev_enl_end = enl_end

        # INCREMENTAL PERSIST — a mid-flight kill costs nothing (board compute cap).
        payload = dict(arm=arm, n_features=N_FEATURES, seed=SEED,
                       target=target, heading=heading, path_clearance=clearance,
                       park_max_activation=park_act,
                       gap_s=GAP_S, traversal_s=trav_s, tau_extrude=tau_live,
                       rho_pred=RHO_PRED, physics_dt=PHYSICS_DT, agent_dt=AGENT_DT,
                       rowsum_min=float(rowsum.min()), rowsum_max=float(rowsum.max()),
                       glutamate_assert_value=glu_assert_value,
                       glutamate_assert_count=glu_assert_count,
                       n_traversals_planned=N_TRAVERSALS,
                       n_traversals_done=n, traversals=traversals)
        with open(os.path.join(outdir, f'ratchet_{arm}_seed{SEED}.json'), 'w') as fh:
            json.dump(payload, fh, indent=1)

    x_edges, betti0 = cross_topology(network)
    payload['cross_edges_final'] = x_edges
    payload['betti0_final'] = betti0
    with open(os.path.join(outdir, f'ratchet_{arm}_seed{SEED}.json'), 'w') as fh:
        json.dump(payload, fh, indent=1)
    print(f"[{arm}] done in {time.time()-t0:.1f}s; cross-synapse edges at end: {x_edges}")
    print()
    return payload


def verdict(drive_payload, null_payload):
    """The PRE-REGISTERED verdict function (PREREG §6). Can return CONFIRMED, FALSIFIED,
    PARTIAL, or three distinct INCONCLUSIVE outcomes. Evaluated in the registered order."""
    d = drive_payload['traversals']
    peak_r = [t['peak_r'] for t in d]
    rhos = [t['rho_into_this'] for t in d if t['rho_into_this'] is not None]
    confs = [t['conf_end'] for t in d]
    einv_starts = [t['E_inv_start'] for t in d[1:]]
    rho_mean = float(np.mean(rhos)) if rhos else float('nan')
    # AMENDMENT 2: predict PER GAP from the MEASURED confinement at gap start.
    conf_at_gap = [t['conf_end'] for t in d[:-1]]
    preds = [rho_predicted(c) for c in conf_at_gap]
    ratios = [r / p for r, p in zip(rhos, preds) if p > 0]
    ratio_mean = float(np.mean(ratios)) if ratios else float('nan')

    print("=" * 100)
    print("VERDICT — thresholds fixed in docs/PREREG_L_ETA_5_RATCHET.md §6, before the run")
    print("=" * 100)
    print(f"  peak_r by traversal      : {[round(x, 5) for x in peak_r]}")
    print(f"  retention rho by gap     : {[round(x, 4) for x in rhos]}")
    print(f"  rho_mean (raw)           : {rho_mean:.4f}")
    print(f"  conf at each gap start   : {[round(c, 4) for c in conf_at_gap]}")
    print(f"  rho_pred per gap         : {[round(p, 4) for p in preds]}")
    print(f"  rho_ratio (SCORED)       : {[round(x, 4) for x in ratios]}")
    print(f"  ratio_mean               : {ratio_mean:.4f}   (band {RATIO_BAND}, ideal 1.0)")
    print(f"  [uncommitted branch would predict {RHO_PRED:.4f}; committed "
          f"{rho_predicted(0.97561):.4f}]")
    print(f"  max confinement          : {max(confs):.4f}")
    print(f"  gain peak_r[N]/peak_r[1] : "
          f"{(peak_r[-1]/peak_r[0]) if peak_r[0] > 0 else float('nan'):.4f}")
    print(f"  glutamate at target, t1  : {drive_payload.get('glutamate_assert_value')}"
          f"  ({drive_payload.get('glutamate_assert_count')} release events)")
    print()

    # ---- NULL ARM CHECK (PREREG §7) — a ratchet here VOIDS the measurement. ----
    nd = null_payload['traversals']
    null_peak = [t['peak_r'] for t in nd]
    null_einv = max(t['E_inv_end'] for t in nd)
    null_gain = (null_peak[-1] / null_peak[0]) if null_peak[0] > 0 else 1.0
    print(f"  NULL arm: max E_invasion {null_einv:.6f}, peak_r gain {null_gain:.4f}")
    if null_einv > 0.0 or null_gain >= GAIN_FALSIFY_MAX:
        print("  => INCONCLUSIVE — NULL ARM RATCHETED. The probe is measuring something")
        print("     other than activity-driven actin. The measurement is VOID.")
        return "INCONCLUSIVE_NULL_RATCHETED"
    print("  => null arm flat, as pre-registered (cannot show the effect by construction)")
    print()

    # ---- GATE 0: positive control must fire (the L·ETA-4 vacuous-verdict scar). ----
    glu = drive_payload.get('glutamate_assert_value')
    pc_glu = glu is not None and glu > 0.0
    pc_einv = max(einv_starts) > 0.0 if einv_starts else False
    pc_r = max(peak_r) > min(null_peak)
    print(f"  GATE 0 positive control: glutamate>0 {pc_glu}, "
          f"E_inv_start>0 after t1 {pc_einv}, r above null {pc_r}")
    if not (pc_glu and pc_einv and pc_r):
        print("  => INCONCLUSIVE — POSITIVE CONTROL DID NOT FIRE. The driven arm did not")
        print("     demonstrably drive, so nothing about retention can be read. (If")
        print("     glutamate is 0 this is the ERR-2 class: NMDARs silent.)")
        return "INCONCLUSIVE_POSITIVE_CONTROL"
    print("  => positive control FIRES")
    print()

    # ---- GATE 1: frozen-clock detection (AMENDMENT 2: unconditional on conf). ----
    # The old CONFINED-RATCHET branch treated rho~1.0 at high conf as legitimate physics.
    # That rested on the §2(B) error: a committed spine drains FASTER (retention at :390),
    # so rho~1.0 is an artifact signature at ANY confinement. Branch deleted.
    if rho_mean >= RHO_ARTIFACT_MIN:
        print(f"  => INCONCLUSIVE — GAP NOT STEPPING. rho_mean {rho_mean:.4f} >= "
              f"{RHO_ARTIFACT_MIN} is indistinguishable from a stopped clock, and under the")
        print("     corrected physics a committed spine drains FASTER, so no confinement")
        print("     state makes ~100% retention legitimate. Red flag, not a positive.")
        return "INCONCLUSIVE_GAP_NOT_STEPPING"

    # ---- GATE 2: the ratchet verdict. ----
    gain = (peak_r[-1] / peak_r[0]) if peak_r[0] > 0 else 0.0
    monotone = all(peak_r[i + 1] >= peak_r[i] for i in range(len(peak_r) - 1))
    in_band = RATIO_BAND[0] <= ratio_mean <= RATIO_BAND[1]

    if monotone and gain >= GAIN_CONFIRM_MIN and in_band:
        print(f"  => RATCHET CONFIRMED. peak_r monotone across {len(peak_r)} traversals,")
        print(f"     gain {gain:.2f}x >= {GAIN_CONFIRM_MIN}, and rho_ratio {ratio_mean:.4f} sits in")
        print(f"     the pre-registered band {RATIO_BAND} against the code's OWN per-gap drain")
        print("     formula. L·ETA-2 and L·ETA-3 reconcile with NO constant touched.")
        result = "CONFIRMED"
    elif gain < GAIN_FALSIFY_MAX or ratio_mean < RHO_FALSIFY_MAX:
        print(f"  => RATCHET FALSIFIED. gain {gain:.4f}, rho_mean {rho_mean:.4f}. The")
        print("     integrator does not retain across a behavioural-timescale gap;")
        print("     repeated traversals do not accumulate. This is a SUBSTANTIVE NEGATIVE")
        print("     RESULT about the network story and is taken as one.")
        result = "FALSIFIED"
    else:
        print(f"  => PARTIAL / INCONCLUSIVE. gain {gain:.4f} (monotone={monotone}),")
        print(f"     ratio_mean {ratio_mean:.4f} vs band {RATIO_BAND}. Numbers reported; no")
        print("     verdict claimed.")
        result = "PARTIAL"

    # Descriptive only — explicitly NOT evidence, per PREREG §6.
    print()
    print("  DESCRIPTIVE (not evidence, does not enter the verdict):")
    print(f"    peak_r reached 1.0 within {len(peak_r)} traversals: {max(peak_r) >= 1.0}")
    if len(peak_r) > 1 and peak_r[0] > 0 and peak_r[-1] > peak_r[0]:
        g = (peak_r[-1] / peak_r[0]) ** (1.0 / (len(peak_r) - 1))
        if g > 1.0 and max(peak_r) < 1.0:
            need = math.log(1.0 / peak_r[-1]) / math.log(g)
            print(f"    geometric extrapolation to r=1.0: ~{need:.0f} further traversals")
            print("    (extrapolation is descriptive; the model is not linear in traversals)")
    return result


def main():
    print("=" * 100)
    print("L·ETA-5 — DOES E_invasion RATCHET ACROSS TRAVERSALS?")
    print("pre-registered: docs/PREREG_L_ETA_5_RATCHET.md   |   no constant is written")
    print("=" * 100)
    print()
    outdir = os.path.join(MODEL6_DIR, 'results', 'einvasion_ratchet')
    os.makedirs(outdir, exist_ok=True)

    drive_payload = run_arm('drive', outdir)
    null_payload = run_arm('null', outdir)

    result = verdict(drive_payload, null_payload)
    with open(os.path.join(outdir, f'verdict_seed{SEED}.json'), 'w') as fh:
        json.dump(dict(verdict=result, rho_pred=RHO_PRED,
                       prereg='docs/PREREG_L_ETA_5_RATCHET.md'), fh, indent=1)
    print()
    print(f"  traces -> {outdir}/")


if __name__ == "__main__":
    main()
