#!/bin/bash
# Timing sweep with the board cooldown protocol: for every mask, ONE bench.py coordinator per cell interleaves all
# configs round-robin (REPS graph replays per burst, COOLDOWN_MS idle after each burst, ROUNDS rounds), so clock
# drift (2.4 -> 1.8 GHz under sustained load on 0614) cannot bias one config.  Holds the shared GPU timing lock.
#
#   run_sweep.sh <wrapper.sh> <out.jsonl> <d> "<configs>" "<masks>" [seqlens...]
#   e.g. run_sweep.sh board_py_0614.sh /tmp/vagarwalla/mxb/results_d128.jsonl 128 \
#          "corr=default corr=always corr=never" "none causal" 8192 16384 32768
# env: ROUNDS=15 REPS=4 COOLDOWN_MS=150 VALIDATE=first HEADS="32 8" BATCH=1 SCHED=natural SCHED_CAUSAL=$SCHED
#      EXTRA="--kv-ramp --rescale-stats" TAG="" LOCK=/tmp/vagarwalla/ladder/gpu_timing.lock
set -u
WRAP=$1; OUT=$2; D=$3; CONFIGS=${4:-"corr=default"}; MASKS=${5:-"none causal"}; shift 5 || shift $#
SEQS=${@:-8192 16384 32768}
ROUNDS=${ROUNDS:-15}; REPS=${REPS:-4}; COOLDOWN_MS=${COOLDOWN_MS:-150}; VALIDATE=${VALIDATE:-first}
HEADS=${HEADS:-"32 8"}; BATCH=${BATCH:-1}; SCHED=${SCHED:-natural}; SCHED_CAUSAL=${SCHED_CAUSAL:-$SCHED}
EXTRA=${EXTRA:-}; TAG=${TAG:-$(hostname)}; LOCK=${LOCK:-/tmp/vagarwalla/ladder/gpu_timing.lock}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$LOCK")" "$(dirname "$OUT")"
exec 9>"$LOCK"
echo "=== waiting for GPU timing lock $LOCK ($(date +%H:%M:%S))"
flock 9
echo "=== lock acquired ($(date +%H:%M:%S)); clocks: $(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader)"
for M in $MASKS; do
  S=$SCHED; [ "$M" = causal ] && S=$SCHED_CAUSAL
  echo "=== d$D mask $M sched $S seqs [$SEQS] configs [$CONFIGS] ($(date +%H:%M:%S)) ==="
  bash $HERE/$WRAP $HERE/bench.py --d $D --mask $M --seqlens $SEQS --configs $CONFIGS --heads $HEADS --batch $BATCH --sched $S \
      --rounds $ROUNDS --reps $REPS --cooldown-ms $COOLDOWN_MS --validate $VALIDATE --out $OUT --tag "$TAG" $EXTRA 2>&1 \
      | grep -vE "RuntimeWarning|AttrBuilder.insert|experimental API"
  echo "=== exit ${PIPESTATUS[0]} ($(date +%H:%M:%S)) ==="
done
echo SWEEP_DONE
