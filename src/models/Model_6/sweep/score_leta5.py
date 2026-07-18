#!/usr/bin/env python3
"""OFFLINE SCORER for L·ETA-5 — the authoritative verdict.

The in-run verdict printed by einvasion_ratchet_probe.py's own main() was computed with the
SUPERSEDED GATE 1/GATE 2 logic (before prereg AMENDMENT 2 corrected the mis-derived retention
prediction). That printed verdict is VOID. This scorer re-reads the incrementally-persisted
per-traversal JSON and applies the CORRECTED, currently-registered verdict function.

This is exactly what the pre-registration's incremental-persistence requirement was for: the
data survives independently of the process that produced it, so a corrected verdict can be
applied to it without re-running.

Usage:  python sweep/score_leta5.py
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from einvasion_ratchet_probe import verdict, SEED

OUT = os.path.join(os.path.dirname(HERE), 'results', 'einvasion_ratchet')


def load(arm):
    p = os.path.join(OUT, f'ratchet_{arm}_seed{SEED}.json')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    drive, null = load('drive'), load('null')
    if drive is None:
        print("no drive-arm trace yet"); return
    nd = drive['n_traversals_done']; npl = drive['n_traversals_planned']
    print(f"drive arm: {nd}/{npl} traversals persisted")
    if null is None:
        print(f"null arm: not started/persisted yet — CANNOT SCORE.")
        print("The null arm is a pre-registered requirement (PREREG §7): without it a")
        print("ratchet in the drive arm cannot be distinguished from the probe measuring")
        print("something other than activity-driven actin. Partial drive data below, no verdict.")
        for t in drive['traversals']:
            print(f"  trav {t['traversal']}: peak_r={t['peak_r']:.5f} "
                  f"enl {t['enl_start']:.5f}->{t['enl_end']:.5f} "
                  f"rho={t['rho_into_this']} conf={t['conf_end']:.4f}")
        return
    print(f"null arm : {null['n_traversals_done']}/{null['n_traversals_planned']} persisted")
    if nd < npl or null['n_traversals_done'] < npl:
        print("\n!! PARTIAL DATA — scoring anyway, but the verdict is over fewer traversals")
        print("   than pre-registered. State this in the write-up; do not report it as the")
        print("   registered 8-traversal result.\n")
    print()
    result = verdict(drive, null)
    print()
    print(f"SCORED VERDICT (corrected function, prereg AMENDMENT 2): {result}")
    with open(os.path.join(OUT, f'verdict_scored_seed{SEED}.json'), 'w') as fh:
        json.dump(dict(verdict=result,
                       n_traversals_drive=nd,
                       n_traversals_null=null['n_traversals_done'],
                       scored_offline=True,
                       supersedes_in_run_verdict=True,
                       prereg='docs/PREREG_L_ETA_5_RATCHET.md (AMENDMENT 2)'), fh, indent=1)


if __name__ == "__main__":
    main()
