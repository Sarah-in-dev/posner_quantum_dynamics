#!/bin/bash
# PHASED, self-orchestrating launcher for the F2 FULL control-ladder run.
# Run it daemonized so it survives teardown and manages both phases itself:
#   nohup setsid bash sweep/f2_launch_full.sh 6 results/f2_full >/dev/null 2>&1 &
#
# Safety design (answers "will this overwhelm the machine?"):
#   - Every worker is thread-capped (OMP/OPENBLAS/MKL/VECLIB/NUMEXPR = 1) → exactly 1 core each
#     (empirically 98-100% CPU/proc). The 2026-07-30 reboot was UNCAPPED workers fanning BLAS
#     across all cores; that cause is eliminated.
#   - PHASED so peak concurrency is bounded and the long overnight tail is light:
#       Phase 1: 8 reward-absent cells (~4 min/draw) in parallel → 8 cores, ~24 min at DRAWS=6.
#       Phase 2: 2 reward-present C0 cells (~48 min/draw, plateau-heavy) → only 2 cores, ~5 h.
#     Never more than 8 cores / 8 processes at once; the 5-hour tail is just 2 processes.
#   - Resume-append per cell: re-running continues each cell from its existing draws.
set -uo pipefail
cd "$(dirname "$0")/.."
DRAWS="${1:-6}"
OUT="${2:-results/f2_full}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="$OUT/orchestrator.log"; : >"$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" >>"$LOG"; }

CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)

launch(){ # cond iso mode  → nohup child (NOT self-daemonized; this orchestrator is the parent)
  local cond="$1" iso="$2" mode="$3"
  local args=(--cond "$cond" --mode "$mode" --n "$DRAWS" --out "$OUT")
  [ "$iso" != "-" ] && args+=(--iso "$iso")
  nohup python3 sweep/f2_control_ladder.py "${args[@]}" >>"$OUT/${cond}_${iso}_${mode}.log" 2>&1 &
}

wait_phase(){ # wait until every listed jsonl has >= DRAWS lines
  local -n files=$1
  while true; do
    local done=1
    for f in "${files[@]}"; do
      local c; c=$(wc -l <"$OUT/$f" 2>/dev/null || echo 0)
      (( c < DRAWS )) && { done=0; break; }
    done
    (( done )) && return 0
    sleep 30
  done
}

# guard
BUSY=$(ps aux | grep -c '[f]2_control_ladder' || true)
if (( BUSY > 0 )); then say "REFUSING: $BUSY f2 workers already running."; exit 1; fi

# --- Phase 1: 8 reward-absent cells (8 cores, ~24 min) ---
say "PHASE 1 (8 reward-absent cells, DRAWS=$DRAWS) on ${CORES}-core host"
P1=()
for mode in pair1 pair2; do
  launch C1  -   "$mode"; P1+=("C1_-_${mode}.jsonl")
  launch C3  -   "$mode"; P1+=("C3_-_${mode}.jsonl")
  launch C2  Li6 "$mode"; P1+=("C2_Li6_${mode}.jsonl")
  launch C2  Li7 "$mode"; P1+=("C2_Li7_${mode}.jsonl")
done
say "phase 1 launched: $(ps aux | grep -c '[f]2_control_ladder') workers"
wait_phase P1
say "PHASE 1 DONE"

# --- Phase 2: 2 reward-present C0 cells only (2 cores, ~5 h) ---
say "PHASE 2 (2 reward-present C0 cells) — long tail, only 2 cores"
P2=()
for mode in pair1 pair2; do launch C0 - "$mode"; P2+=("C0_-_${mode}.jsonl"); done
wait_phase P2
say "PHASE 2 DONE — F2 FULL LADDER COMPLETE"
