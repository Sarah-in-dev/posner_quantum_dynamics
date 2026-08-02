#!/bin/bash
# Parallel daemonized launcher for the F2 FULL control-ladder run (consolidation / washout question).
# Each cell = ONE thread-capped, daemonized process writing its OWN resume-append jsonl, so a crash/
# reboot resumes (re-run this script; append mode continues each cell from its existing draws).
#
# 10 cells (5 conds × 2 modes) → 10 cores. Safe on a 14-core Mac (4 free). The thread caps make each
# worker exactly 1 core — the 2026-07-30 reboot was UNCAPPED workers fanning BLAS across all cores.
#
# Usage:  bash sweep/f2_launch_full.sh [DRAWS=6] [OUTDIR=results/f2_full]
# Watch:  for f in results/f2_full/*.jsonl; do echo "$(wc -l <"$f") $(basename "$f")"; done
set -euo pipefail
cd "$(dirname "$0")/.."
DRAWS="${1:-6}"
OUT="${2:-results/f2_full}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# guard: refuse to overload — need <= (cores-2) free after the launch
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
BUSY=$(ps aux | grep -c '[f]2_control_ladder' || true)
WANT=10
if (( BUSY + WANT > CORES - 2 )); then
  echo "REFUSING: ${BUSY} f2 workers already running + ${WANT} new > ${CORES}-2 cores. Wait or reduce." >&2
  exit 1
fi

launch() {  # cond iso mode
  local cond="$1" iso="$2" mode="$3"
  local args=(--cond "$cond" --mode "$mode" --n "$DRAWS" --out "$OUT")
  [ "$iso" != "-" ] && args+=(--iso "$iso")
  nohup python3 sweep/f2_control_ladder.py "${args[@]}" --daemonize "$OUT/${cond}_${iso}_${mode}.log" >/dev/null 2>&1 &
}

for mode in pair1 pair2; do
  launch C1  -   "$mode"   # undoped, reward-absent  — does consolidation form WITHOUT reward?
  launch C3  -   "$mode"   # shuffle control         — selectivity must vanish
  launch C2  Li6 "$mode"   # isotope, coherent       — downstream consolidation contrast?
  launch C2  Li7 "$mode"   # isotope, fast-decohere
  launch C0  -   "$mode"   # undoped, reward-present — STEP2 baseline (~6.9/8 commit)
done
sleep 2
echo "launched $(ps aux | grep -c '[f]2_control_ladder') f2 cells, draws=${DRAWS} -> ${OUT}"
echo "score when done:  for c in C0 C1 C2_Li6 C2_Li7 C3; do python3 sweep/po11_valence_score.py --glob \"${OUT}/\${c}*_*.jsonl\" --label \$c; done"
