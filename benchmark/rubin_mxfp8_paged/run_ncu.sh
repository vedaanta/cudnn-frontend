#!/bin/bash
# ncu per config: one `--set full` report per (config, mask, S), profiling the 4th FROST launch (3 warm-ups) of
# bench.py --ncu-run, then exports the raw metrics page and the SASS source page (per-instruction executed counts
# + stall samples) as CSV next to the report, so ncu_table.py / unit_sol.py need no ncu binary off-board.
# Holds the shared GPU timing lock.  No sudo on 0614: ncu runs as the user (RmProfilingAdminOnly=0).
#
#   run_ncu.sh <wrapper.sh> <outdir> <d> "<configs>" "<masks>" [seqlens...]
#   e.g. run_ncu.sh board_py_0614.sh /tmp/vagarwalla/mxb/ncu 128 "corr=default corr=always" "none causal" 16384
# env: HEADS="32 8" BATCH=1 SCHED=natural SCHED_CAUSAL=$SCHED NCU_SET=full NCU_METRICS="" EXTRA="" SKIP_EXISTING=1
#      NCU_BIN=/tmp/vagarwalla/ncu/ncu LOCK=/tmp/vagarwalla/ladder/gpu_timing.lock
set -u
WRAP=$1; OUTDIR=$2; D=$3; CONFIGS=${4:-"corr=default"}; MASKS=${5:-"none causal"}; shift 5 || shift $#
SEQS=${@:-16384}
HEADS=${HEADS:-"32 8"}; BATCH=${BATCH:-1}; SCHED=${SCHED:-natural}; SCHED_CAUSAL=${SCHED_CAUSAL:-$SCHED}
EXTRA=${EXTRA:-}; LOCK=${LOCK:-/tmp/vagarwalla/ladder/gpu_timing.lock}; NCU_BIN=${NCU_BIN:-/tmp/vagarwalla/ncu/ncu}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $OUTDIR $OUTDIR/.ncu_home "$(dirname "$LOCK")"
exec 9>"$LOCK"
echo "=== waiting for GPU timing lock $LOCK ($(date +%H:%M:%S))"
flock 9
echo "=== lock acquired ($(date +%H:%M:%S))"
for M in $MASKS; do
  SP=$SCHED; [ "$M" = causal ] && SP=$SCHED_CAUSAL
  for C in $CONFIGS; do
    NAME=$(bash $HERE/$WRAP $HERE/bench.py --d $D --config "$C" --sched $SP --print-name 2>/dev/null | tail -1)
    [ -z "$NAME" ] && { echo "!! cannot resolve config name for '$C'"; continue; }
    for S in $SEQS; do
      N=d${D}_${NAME}_${M}_${S}; [ "$SP" != natural ] && N=${N}_${SP}
      if [ "${SKIP_EXISTING:-1}" = 1 ] && ls $OUTDIR/$N.ncu-rep* >/dev/null 2>&1; then echo "skip $N (exists)"; continue; fi
      echo "=== ncu $N ($(date +%H:%M:%S)) $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader) ==="
      NCU=1 NCU_BIN=$NCU_BIN NCU_K=cudnn_kernel__kernel NCU_SKIP=3 NCU_COUNT=1 NCU_OUT=$OUTDIR/$N NCU_SET=${NCU_SET:-full} NCU_METRICS="${NCU_METRICS:-}" \
        bash $HERE/$WRAP $HERE/bench.py --d $D --config "$C" --seqlens $S --mask $M --heads $HEADS --batch $BATCH --sched $SP --ncu-run --warmup 3 --validate none $EXTRA 2>&1 \
        | grep -E "NCU_LAUNCH_DONE|==ERROR==|==WARNING==|Traceback|Error|error" | cut -c1-300 | head -8
      REP=$(ls $OUTDIR/$N.ncu-rep* 2>/dev/null | head -1)
      if [ -n "$REP" ]; then
        # ncu --import needs a WRITABLE HOME (else an empty CSV); the raw page = every metric of the set
        HOME=$OUTDIR/.ncu_home $NCU_BIN --import "$REP" --csv --page raw > $OUTDIR/$N.raw.csv 2>$OUTDIR/$N.raw.err
        HOME=$OUTDIR/.ncu_home $NCU_BIN --import "$REP" --csv --page source --print-source sass > $OUTDIR/$N.sass.csv 2>$OUTDIR/$N.sass.err
        echo "    report $(du -h "$REP" | cut -f1) raw.csv $(wc -c < $OUTDIR/$N.raw.csv) B sass.csv $(wc -l < $OUTDIR/$N.sass.csv) lines"
      else
        echo "    !! no report written for $N"
      fi
    done
  done
done
echo NCU_SWEEP_DONE
