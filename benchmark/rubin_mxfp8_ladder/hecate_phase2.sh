#!/bin/bash
# hecate phase 2 (runs inside the holder job's container after the A-D ncu sweep):
#   1. rung E timing (causal + none)          2. rung E ncu (causal + none)
#   3. A-E causal timing with the LPT scheduler  4. A-E causal ncu with LPT
set -u
L=/home/vagarwalla/ladder
H=$L/fe/benchmark/rubin_mxfp8_ladder
cd $H
echo "### phase2 start $(date +%H:%M:%S) on $(hostname)"
echo "### 1. E timing"
bash run_sweep.sh hecate_py.sh $L/results/results_hecate.jsonl "E" "causal none" 8192 16384 32768
echo "### 2. E ncu"
bash run_ncu_sweep.sh hecate_py.sh $L/ncu "E" "causal none" 8192 16384 32768
echo "### 3. A-E causal LPT timing"
SCHED=lpt bash run_sweep.sh hecate_py.sh $L/results/results_hecate.jsonl "A B C D E" "causal" 8192 16384 32768
echo "### 4. A-E causal LPT ncu"
SCHED=lpt bash run_ncu_sweep.sh hecate_py.sh $L/ncu "A B C D E" "causal" 8192 16384 32768
echo "### phase2 done $(date +%H:%M:%S)"
echo PHASE2_DONE
