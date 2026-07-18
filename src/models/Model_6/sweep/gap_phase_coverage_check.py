#!/usr/bin/env python3
"""
Phase-coverage check for analytical_gap's docstring.  PO-4.
===========================================================
The gap's docstring states a rule: EVERY subsystem is either ADVANCED with a
timescale or EXCLUDED with a reason, and nothing is in neither.

That rule was violated twice, and BOTH times a human found it:
  - originally, actin / E_invasion / CaMKII / DDSC were in neither column
    (the defect PO-4 was dispatched to fix);
  - then PHASE 12 (template feedback) and PHASE 9 (eligibility) were in neither,
    found by the MO at ruling 007 -- reintroduced at the edge of the fix that
    removed it elsewhere, because fixing the clock made PHASE 12 reachable.

A rule that only holds when someone re-reads it by hand is not enforced. This
makes it mechanically checkable: every `# --- PHASE N: ... ---` marker in
model6_core.py must be accounted for by a [PHASE N] tag in the gap's docstring.

Exit code 0 = every phase accounted for. Non-zero = at least one in neither
column, with the offenders named. Cheap; no network construction.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.dirname(HERE)
CORE = os.path.join(M6, 'model6_core.py')
GAP = os.path.join(HERE, 'run_theta_burst_45s.py')

# PHASE 9 is used twice in model6_core (:599 eligibility, :617 the three-factor
# gate) -- a numbering collision in the file's own comments, routed to the MO in
# queue/po4-gap.md Q4-6 and NOT fixed here (not PO-4's surface). The docstring
# distinguishes them as [PHASE 9] and [PHASE 9b]; this check accepts either for
# the second occurrence.
DUPLICATE_PHASE_ALIASES = {9: ['[PHASE 9]', '[PHASE 9b]']}


def core_phases(text):
    """Every '# --- PHASE N: NAME ---' marker, in order, with duplicates kept."""
    return [(int(n), name.strip())
            for n, name in re.findall(r'#\s*---\s*PHASE\s+(\d+):\s*(.+?)\s*---', text)]


def gap_docstring(text):
    m = re.search(r'def analytical_gap\(.*?\):\s*"""(.*?)"""', text, re.S)
    if not m:
        print("FAIL: could not locate analytical_gap's docstring", file=sys.stderr)
        sys.exit(2)
    return m.group(1)


def main():
    phases = core_phases(open(CORE).read())
    doc = gap_docstring(open(GAP).read())

    print("=" * 78)
    print("PHASE COVERAGE — analytical_gap docstring vs model6_core.py step phases")
    print("=" * 78)
    print(f"  {len(phases)} phase markers found in model6_core.py\n")

    seen, missing = {}, []
    for num, name in phases:
        seen[num] = seen.get(num, 0) + 1
        tags = DUPLICATE_PHASE_ALIASES.get(num, [f'[PHASE {num}]'])
        # For a duplicated number, the k-th occurrence needs the k-th alias.
        tag = tags[min(seen[num] - 1, len(tags) - 1)]
        ok = tag in doc
        print(f"  {'OK ' if ok else 'MISSING'}  {tag:<14} {name[:52]}")
        if not ok:
            missing.append((tag, name))

    print()
    if missing:
        print("=" * 78)
        print(f"FAIL — {len(missing)} phase(s) in NEITHER column:")
        for tag, name in missing:
            print(f"  - {tag} {name}")
        print("\nThe docstring's own rule says nothing may be in neither. Put each in")
        print("ADVANCED with a timescale or EXCLUDED with a reason.")
        print("=" * 78)
        return 1

    print("=" * 78)
    print("PASS — every phase is accounted for in one column or the other.")
    print("Note: this checks COVERAGE, not correctness. It cannot tell you a")
    print("timescale is right or an exclusion is honest -- only that nothing is")
    print("silently absent, which is the failure mode that bit twice.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
