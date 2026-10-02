#!/bin/bash
# Timing sweep driver.  Usage: run_sweep.sh <wrapper.sh> <out.jsonl> "<rungs>" "<masks>" [seqlens...]
#   e.g. run_sweep.sh board_py.sh /tmp/vagarwalla/ladder/results_board.jsonl "A B C D" "causal none" 8192 16384 32768
WRAP=$1; OUT=$2; RUNGS=${3:-"A B C D"}; MASKS=${4:-"causal none"}; shift 4
SEQS=${@:-8192 16384 32768}
SCHED=${SCHED:-natural}
HERE=$(cd "$(dirname "$0")" && pwd)
for M in $MASKS; do
  for R in $RUNGS; do
    echo "=== rung $R mask $M seqs $SEQS ($(date +%H:%M:%S)) ==="
    bash $HERE/$WRAP $HERE/bench_ladder.py --rung $R --seqlens $SEQS --mask $M --validate first --out $OUT --sched $SCHED --tag "$(hostname)" 2>&1 | grep -vE "RuntimeWarning|AttrBuilder.insert|experimental API"
    echo "=== exit $? ==="
  done
done
echo SWEEP_DONE
