#!/bin/bash
# First on-board run of MR 28758's gqa.py through the shim (Rubin SMEM 320 KiB).
B=/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/rubin-ladder-bench/benchmark/rubin_mxfp8_ladder
R=/home/scratch.vagarwalla_gpu/.claude/jobs/00fe3b3d/tmp/j/dkg
export GQA_SMEM_BYTES=327680 CUTE_DSL_CACHE_DIR=/tmp/vagarwalla/ladder/dsl_cache_gqa PYTHONPATH_TAIL=/home/scratch.vagarwalla_gpu/.claude/jobs/00fe3b3d/tmp/j/pycupti
cd $B && timeout ${TMO:-420} bash board_py.sh $R/run_gqa.py "$@" 2>&1 | grep -v "^\s*$" | tail -${TAIL:-40}
echo "rc=${PIPESTATUS[0]}"
