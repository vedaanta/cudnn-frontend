#!/bin/bash
# Follow-up cells (after the GPU comes back): MHA vs GQA packing, the MR's d192/128 shape, real ifence, knob cells, ncu.
set -u
B=/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/rubin-ladder-bench/benchmark/rubin_mxfp8_ladder
R=/home/scratch.vagarwalla_gpu/.claude/jobs/00fe3b3d/tmp/j/dkg
L=/tmp/vagarwalla/ladder; OUT=$L/results_cmp2_gqa.jsonl; rm -f $OUT; LOG=$R/compare2_gqa.txt; : > $LOG
export GQA_SMEM_BYTES=327680 CUTE_DSL_CACHE_DIR=$L/dsl_cache_gqa PYTHONPATH_TAIL=/home/scratch.vagarwalla_gpu/.claude/jobs/00fe3b3d/tmp/j/pycupti
cd $B
gqa() { local TAG=$1; shift
  local clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1)
  local out=$(timeout 900 bash board_py.sh $R/run_gqa.py "$@" --skip_ref_check --warmups ${WARM:-20} --iterations ${ITERS:-50} 2>&1 | grep -v "RuntimeWarning\|AttrBuilder")
  local us=$(echo "$out" | grep -oE "^[0-9.]+ us" | tail -1); local tf=$(echo "$out" | grep -oE "^[0-9.]+ TFLOPS/s" | tail -1)
  local err=$(echo "$out" | grep -E "Error|error:|Traceback" | head -1 | cut -c1-100)
  echo "GQA $TAG | clk_before $clk | $us | $tf | $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1) $err" | tee -a $LOG; }
ours() { local TAG=$1 M=$2 EXTRA=$3
  timeout 420 bash board_py.sh bench_ladder.py --rung F --seqlens ${SEQ:-32768} --mask $M --validate ${VAL:-none} --reps 20 --rounds 3 --out $OUT --tag "$TAG" $EXTRA 2>&1 | grep -E "Traceback|Error" | head -2
  python3 - <<PY | tee -a $LOG
import json
rows=[json.loads(l) for l in open("$OUT")]; r=rows[-1]
ok = r["tag"]=="$TAG"
print(f"OURS $TAG | S {r['S']} {r['mask']} {r.get('sched')} | {r['time_us_graph']} us | {r['tflops']} TFLOPS/s | clocks_after {r['clocks_after']}" if ok else "OURS $TAG | FAILED (no row)")
PY
}
echo "=== compare2 $(date +%H:%M:%S) $(nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv,noheader)" | tee -a $LOG
SEQ=4096 VAL=all ours f_val4k none ""
D="--mma_dtype Float8E4M3FN --out_dtype BFloat16"
S=32768
gqa d128_mha32_${S}_none --d 128 --h_q 32 --h_k 32 --s_q $S --s_k $S --window_left none --window_right none $D
gqa d128_gqa32_2_${S}_none --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none $D
GQA_IFENCE=1 gqa d128_gqa32_2_${S}_none_IFENCE --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none $D
SEQ=$S ours f_none_$S none ""
for S in 16384 32768; do
  gqa d192_h128_${S}_none --d 192,128 --h_q 128 --h_k 128 --s_q $S --s_k $S --window_left none --window_right none $D
  gqa d192_h128_${S}_causal --d 192,128 --h_q 128 --h_k 128 --s_q $S --s_k $S --window_left none --window_right 0 $D
done
S=32768
gqa d128_gqa32_2_${S}_nonpersist --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none $D --disable_persistence
gqa d128_gqa32_2_${S}_normarowsum --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none $D --disable_mma_rowsum
gqa d128_gqa32_2_${S}_bf16 --d 128 --h_q 32 --h_k 2 --s_q $S --s_k $S --window_left none --window_right none --mma_dtype BFloat16 --out_dtype BFloat16
gqa d128_mha32_${S}_causal --d 128 --h_q 32 --h_k 32 --s_q $S --s_k $S --window_left none --window_right 0 $D
SEQ=$S ours f_causal_$S causal "--sched lpt"
echo CMP2_DONE | tee -a $LOG
