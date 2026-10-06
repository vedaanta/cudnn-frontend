#!/bin/bash
# Same-board comparison: MR 28758 gqa.py (per-tensor fp8, fp16 softmax, temporal, Q in TMEM) vs our F (mxfp8 ladder).
set -u
B=/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/rubin-ladder-bench/benchmark/rubin_mxfp8_ladder
R=/home/scratch.vagarwalla_gpu/.claude/jobs/00fe3b3d/tmp/j/dkg
L=/tmp/vagarwalla/ladder; OUT=$L/results_cmp_gqa.jsonl; rm -f $OUT; LOG=$R/compare_gqa.txt; : > $LOG
export GQA_SMEM_BYTES=327680 CUTE_DSL_CACHE_DIR=$L/dsl_cache_gqa PYTHONPATH_TAIL=/home/scratch.vagarwalla_gpu/.claude/jobs/00fe3b3d/tmp/j/pycupti
cd $B
gqa() { local TAG=$1; shift
  local clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1)
  local out=$(timeout 900 bash board_py.sh $R/run_gqa.py "$@" --skip_ref_check --warmups ${WARM:-20} --iterations ${ITERS:-50} 2>&1 | grep -v "RuntimeWarning\|AttrBuilder")
  local us=$(echo "$out" | grep -oE "^[0-9.]+ us" | tail -1); local tf=$(echo "$out" | grep -oE "^[0-9.]+ TFLOPS/s" | tail -1)
  local err=$(echo "$out" | grep -E "Error|error:|Traceback" | head -1 | cut -c1-100)
  echo "GQA $TAG | clk_before $clk | $us | $tf | $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1) $err" | tee -a $LOG; }
ours() { local TAG=$1 M=$2 EXTRA=$3
  timeout 420 bash board_py.sh bench_ladder.py --rung F --seqlens ${SEQ:-32768} --mask $M --validate none --reps 20 --rounds 3 --out $OUT --tag "$TAG" $EXTRA 2>&1 | grep -E "Traceback|Error" | head -2
  python3 - <<PY | tee -a $LOG
import json
r=[json.loads(l) for l in open("$OUT")][-1]
print(f"OURS $TAG | S {r['S']} {r['mask']} {r.get('sched')} | {r['time_us_graph']} us | {r['tflops']} TFLOPS/s | clocks_after {r['clocks_after']}")
PY
}
echo "=== compare $(date +%H:%M:%S) $(nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv,noheader)" | tee -a $LOG
for i in 1 2; do
  for S in 16384 32768; do
    SEQ=$S ours f_none_$S none ""
    gqa d128_h32_2_${S}_none --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none --mma_dtype Float8E4M3FN --out_dtype BFloat16
    SEQ=$S ours f_causal_$S causal "--sched lpt"
    gqa d128_h32_2_${S}_causal --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right 0 --mma_dtype Float8E4M3FN --out_dtype BFloat16
  done
done
# the MR's own shape (d192/128, h128/128, B=1) for a board-to-board sanity check against the MR table
for S in 16384 32768; do
  gqa d192_h128_${S}_none --d 192,128 --h_q 128 --h_k 128 --s_q $S --s_k $S --window_left none --window_right none --mma_dtype Float8E4M3FN --out_dtype BFloat16
  gqa d192_h128_${S}_causal --d 192,128 --h_q 128 --h_k 128 --s_q $S --s_k $S --window_left none --window_right 0 --mma_dtype Float8E4M3FN --out_dtype BFloat16
done
# knobs: fp32 softmax path (no fp16), no mma rowsum, non-persistent -- what each of Richard's levers is worth on this board
S=32768
gqa d128_${S}_none_nonpersist --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none --mma_dtype Float8E4M3FN --out_dtype BFloat16 --disable_persistence
gqa d128_${S}_none_normarowsum --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none --mma_dtype Float8E4M3FN --out_dtype BFloat16 --disable_mma_rowsum
gqa d128_${S}_none_bf16 --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none --mma_dtype BFloat16 --out_dtype BFloat16
echo CMP_DONE | tee -a $LOG
