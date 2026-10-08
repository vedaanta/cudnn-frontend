#!/bin/bash
# K1 timing sanity on board 0614 (under the GPU timing lock): 32k none B1 H32/8, stack on (paged64, CORRFAST,
# HOIST, HALF + prefolded), BENCH_CORR default / always / never + the product dense kernel as the reference arm;
# ROUNDS rounds x 4 replays, arms alternated (one process per burst: the process start is the idle gap).
set -u
K=/tmp/vagarwalla/k1; OUT=$K/out; P="bash $K/k1_py.sh"; J=$OUT/timing_32k.jsonl
ROUNDS=${ROUNDS:-5}
cd /tmp/vagarwalla
rm -f $J
COMMON="--half 1 --prefolded 1 --seqlens 32768 --mask none --heads 32 8 --validate 0 --warmup 2 --time 1 --rounds 1 --reps 4 --out $J"
exec 9>/tmp/vagarwalla/ladder/gpu_timing.lock
flock 9
echo "lock acquired $(date +%T)"
for r in $(seq 1 $ROUNDS); do
  for corr in default always never; do
    echo "-- $(date +%T) r$r bench paged64 corr=$corr"
    $P --kernel bench --paged 64 --corr $corr --corrfast 1 --hoist 1 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*, "clocks_per_burst": \[[^]]*\]'
    sleep 0.2
  done
  echo "-- $(date +%T) r$r product dense"
  $P --kernel product --paged 0 --corr default --corrfast 0 --hoist 0 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*, "clocks_per_burst": \[[^]]*\]'
  sleep 0.2
done
echo TIMING_DONE
