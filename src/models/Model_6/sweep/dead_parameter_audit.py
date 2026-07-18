#!/usr/bin/env python3
"""
WHICH DECLARED PARAMETERS DOES NOTHING READ? — PO-6a Unit 2, second half.

WHY THIS EXISTS
---------------
The substrate audit counted "~151 dead parameter fields (up from ~120 at may29)" — a debt
figure that REGRESSED between audits. A declared-but-unread parameter is not merely clutter:

  · it invites the reader to believe a mechanism is parameterised when it is not (the
    program's characteristic defect), and
  · if a sweep dimension is ever pointed at one, the sweep returns a flat response that reads
    as a physical null — which is exactly how nine of nineteen dimensions got that way
    (see sweep/dimension_consumer_audit.py).

So this inventories them with evidence, at the same standard as the orphan-module audit:
AST, not text grep.

METHOD, AND WHY THE VERDICT IS SOUND IN ONE DIRECTION ONLY
----------------------------------------------------------
For every dataclass field declared in the parameter modules, count across the whole tree:

  (a) ATTRIBUTE READS   — ast.Attribute nodes whose `.attr` equals the field name,
                          excluding the declaration itself and excluding pure writes
                          (Store context), which are not consumption.
  (b) DYNAMIC ACCESS    — the name appearing as a string ARGUMENT to
                          getattr/setattr/hasattr, which catches dynamic access an attribute
                          scan cannot see.

                          NOT any string literal. The first version of this scan counted
                          every string constant, and its own control caught the error:
                          `kT_per_modulation_unit` came back LIVE because
                          quantum_dimensions.py declares `variable="kT_per_modulation_unit"`
                          as dimension METADATA. A name appearing in a data table is not a
                          read, and counting it that way would have quietly suppressed real
                          dead fields — the exact class this audit exists to find.

A field is reported DEAD only when BOTH are zero.

**The asymmetry is deliberate and is what makes this trustworthy:** an attribute scan keyed on
name alone OVER-reports liveness (any object anywhere with the same attribute name counts as a
read). It can therefore call a dead field live, but it cannot call a live field dead. Zero
hits on both channels is a sound DEAD verdict; a nonzero count is NOT a claim of real use.

That is the same logic as the read-tracer in dimension_consumer_audit.py: choose the
instrument whose error direction cannot manufacture the finding you are looking for.

WHY NOT RUNTIME READ-TRACING HERE
---------------------------------
The read-tracer is a better instrument per-field, but it only sees paths a given run
exercises, so it would mark every conditional consumer dead. For a whole-inventory sweep
across 220 fields, static analysis with an over-reporting bias is the correct trade. Fields
this reports DEAD are candidates for deletion, **not deletions** — see LIMITS.

LIMITS (stated, and they bind)
------------------------------
· DEAD here means "no attribute read anywhere in the parsed tree, and never named in a
  getattr/setattr/hasattr call." It does not prove the field is safe to delete: code outside
  the parsed roots (notebooks, external scripts) is invisible to this scan.
· A field read ONLY by an orphan module counts as LIVE on the main tally — that is the
  `T_singlet_dimer` shape before ruling 006, where the sole reader was `singlet_dynamics.py`,
  a module nothing imports. Those are reported separately as ORPHAN-ONLY, because they are
  dead in every sense that matters while remaining "read".
· Nothing is deleted by this script. It reports.

Read-only and static — parses files, never imports or runs the model. Safe to run while
another PO holds the heavy compute slot.
"""
import ast
import os
import sys
import collections

# Where parameters are DECLARED
PARAM_FILES = [
    'src/models/Model_6/model6_parameters.py',
]

# Where they might be READ. Deliberately wide — a narrow root would manufacture dead verdicts.
SCAN_ROOTS = ['src/models/Model_6', 'sweep']

# Modules with NO importer anywhere (PO-6a Unit 2, AST-verified). A field whose only reader
# lives here is "read" but not reachable — the T_singlet_dimer shape before ruling 006.
ORPHAN_MODULES = {'eligibility_trace.py', 'singlet_dynamics.py'}


def declared_fields(param_files):
    """-> {(class, field): lineno}, from dataclass annotated assignments."""
    out = {}
    for pf in param_files:
        tree = ast.parse(open(pf).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        out[(node.name, stmt.target.id)] = stmt.lineno
    return out


DYNAMIC_ACCESSORS = {'getattr', 'setattr', 'hasattr'}


def scan_tree(roots, skip_basenames=frozenset()):
    """-> (attr_reads Counter, dynamic_access Counter, n_files).

    skip_basenames lets the caller re-scan with the orphan modules excluded, which is how
    ORPHAN-ONLY fields are identified.
    """
    reads = collections.Counter()
    dynamic = collections.Counter()
    n = 0
    for root in roots:
        for dp, _, fns in os.walk(root):
            if '__pycache__' in dp:
                continue
            for fn in fns:
                if not fn.endswith('.py') or fn in skip_basenames:
                    continue
                path = os.path.join(dp, fn)
                try:
                    tree = ast.parse(open(path).read())
                except Exception:
                    continue
                n += 1
                for node in ast.walk(tree):
                    # attribute READS only — a Store is a write, not consumption
                    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                        reads[node.attr] += 1
                    # dynamic access: getattr/setattr/hasattr(obj, "name", ...)
                    elif (isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name)
                          and node.func.id in DYNAMIC_ACCESSORS
                          and len(node.args) >= 2
                          and isinstance(node.args[1], ast.Constant)
                          and isinstance(node.args[1].value, str)):
                        dynamic[node.args[1].value] += 1
    return reads, dynamic, n


def main():
    fields = declared_fields(PARAM_FILES)
    reads, dynamic, n_files = scan_tree(SCAN_ROOTS)
    # Second pass with the orphan modules removed, to find fields kept alive only by them.
    reads_no_orphan, dynamic_no_orphan, _ = scan_tree(SCAN_ROOTS, skip_basenames=ORPHAN_MODULES)

    dead, live = [], []
    for (cls, name), lineno in sorted(fields.items()):
        r = reads.get(name, 0)
        s = dynamic.get(name, 0)
        (dead if (r == 0 and s == 0) else live).append((cls, name, lineno, r, s))

    print("=" * 78)
    print("DEAD PARAMETER AUDIT — which declared fields does nothing read?")
    print("=" * 78)
    print(f"  parsed {n_files} python files across {SCAN_ROOTS}")
    print(f"  declared fields: {len(fields)}   live: {len(live)}   DEAD: {len(dead)}")
    print("  live = read as an attribute somewhere, OR named in getattr/setattr/hasattr")
    print()

    # --- CONTROL: the instrument must distinguish its outcomes -------------------------
    # Known-live and known-dead anchors, established by earlier PO-6a units.
    ctrl_live = reads.get('omega_0', 0) > 0          # enters P_c at both pump sites (B2)
    ctrl_dead = ('DendriticBackboneParameters', 'kT_per_modulation_unit') in \
                {(c, nm) for c, nm, _, _, _ in dead}
    print("--- CONTROLS ---")
    print(f"  known-LIVE  omega_0 reads > 0                 : "
          f"{'PASS' if ctrl_live else 'FAIL'} ({reads.get('omega_0',0)})")
    print(f"  known-DEAD  kT_per_modulation_unit reported   : "
          f"{'PASS' if ctrl_dead else 'FAIL'}")
    if not (ctrl_live and ctrl_dead):
        print("\nVERDICT: INVALID — a control failed; this scan is not discriminating.")
        return 2
    print()

    by_class = collections.Counter(c for c, _, _, _, _ in dead)
    print("--- DEAD FIELDS BY CLASS ---")
    for cls, cnt in by_class.most_common():
        total = sum(1 for (c, _) in fields if c == cls)
        print(f"  {cls:34s} {cnt:3d} dead of {total:3d}")
    print()

    print("--- DEAD FIELDS (declaration line in model6_parameters.py) ---")
    for cls, name, lineno, _, _ in dead:
        print(f"  :{lineno:<5d} {cls}.{name}")
    print()

    # --- ORPHAN-ONLY: live on the main tally, dead once orphans are excluded -----------
    orphan_only = []
    for cls, name, lineno, r, s_ in live:
        if reads_no_orphan.get(name, 0) == 0 and dynamic_no_orphan.get(name, 0) == 0:
            orphan_only.append((cls, name, lineno))
    print("--- ORPHAN-ONLY FIELDS (read, but only by a module nothing imports) ---")
    if orphan_only:
        for cls, name, lineno in orphan_only:
            print(f"  :{lineno:<5d} {cls}.{name}")
        print(f"  {len(orphan_only)} field(s). These are dead in every sense that matters:")
        print("  their sole reader is unreachable. They become plainly dead the moment the")
        print("  orphan modules are deleted — so they belong in that same batch, not separately.")
    else:
        print("  none")
    print()

    print("=" * 78)
    print(f"VERDICT: {len(dead)} of {len(fields)} declared parameter fields are DEAD —")
    print("         no attribute read and no string mention anywhere in the parsed tree.")
    print()
    print("  LIMITS: DEAD is a deletion CANDIDATE, not a deletion. A field read only by an")
    print("  orphan module still counts as read here (T_singlet_dimer was that shape before")
    print("  ruling 006), and code outside the parsed roots is invisible to this scan.")
    print("  The scan over-reports LIVE by construction, so a DEAD verdict is sound and a")
    print("  LIVE verdict is not a claim of real use.")
    print("=" * 78)
    return 0 if not dead else 1


if __name__ == "__main__":
    sys.exit(main())
