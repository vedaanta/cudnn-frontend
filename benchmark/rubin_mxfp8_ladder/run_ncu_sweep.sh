#!/bin/bash
# ncu sweep driver: one --set full report per (rung, mask, S), profiling the 4th FROST launch
# (3 warmups) of the harness in --ncu-run mode.
# Usage: run_ncu_sweep.sh <wrapper.sh> <outdir> "<rungs>" "<masks>" [seqlens...]
WRAP=$1; OUTDIR=$2; RUNGS=${3:-"A B C D"}; MASKS=${4:-"causal"}; shift 4
SEQS=${@:-8192 16384 32768}
SCHED=${SCHED:-natural}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $OUTDIR
for M in $MASKS; do
  for R in $RUNGS; do
    for S in $SEQS; do
      N=${R}_${M}_${S}; [ "$SCHED" != natural ] && N=${N}_${SCHED}
      [ -f $OUTDIR/$N.ncu-repz ] && { echo "skip $N (exists)"; continue; }
      echo "=== ncu $N ($(date +%H:%M:%S)) ==="
      NCU=1 NCU_K=cudnn_kernel__kernel NCU_SKIP=3 NCU_COUNT=1 NCU_OUT=$OUTDIR/$N \
        bash $HERE/$WRAP $HERE/bench_ladder.py --rung $R --seqlens $S --mask $M --sched $SCHED --ncu-run --warmup 3 2>&1 \
        | grep -E "NCU_LAUNCH_DONE|==ERROR==|==WARNING==|Traceback|Error" | head -5
    done
  done
done
echo NCU_SWEEP_DONE
