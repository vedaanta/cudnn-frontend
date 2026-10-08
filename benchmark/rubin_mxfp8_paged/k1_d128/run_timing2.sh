#!/bin/bash
# K1 timing sanity, part 2 (under the GPU timing lock): separate the paging cost from the CORRFAST + HOIST gain at
# 32k none B1 H32/8, HALF + prefolded, corr default: product dense | bench dense cf0 h0 | bench dense cf1 h1 |
# bench paged64 cf0 h0 | bench paged64 cf1 h1; ROUNDS rounds x 4 replays, arms alternated.
set -u
K=/tmp/vagarwalla/k1; OUT=$K/out; P="bash $K/k1_py.sh"; J=$OUT/timing2_32k.jsonl
ROUNDS=${ROUNDS:-3}
cd /tmp/vagarwalla
rm -f $J
COMMON="--half 1 --prefolded 1 --seqlens 32768 --mask none --heads 32 8 --validate 0 --warmup 2 --time 1 --rounds 1 --reps 4 --out $J"
exec 9>/tmp/vagarwalla/ladder/gpu_timing.lock
flock 9
echo "lock acquired $(date +%T)"
for r in $(seq 1 $ROUNDS); do
  echo "-- $(date +%T) r$r"
  $P --kernel product --paged 0 --corr default --corrfast 0 --hoist 0 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*' | sed 's/^/product dense        /'
  sleep 0.2
  $P --kernel bench --paged 0 --corr default --corrfast 0 --hoist 0 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*' | sed 's/^/bench dense cf0 h0   /'
  sleep 0.2
  $P --kernel bench --paged 0 --corr default --corrfast 1 --hoist 1 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*' | sed 's/^/bench dense cf1 h1   /'
  sleep 0.2
  $P --kernel bench --paged 64 --corr default --corrfast 0 --hoist 0 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*' | sed 's/^/bench paged64 cf0 h0 /'
  sleep 0.2
  $P --kernel bench --paged 64 --corr default --corrfast 1 --hoist 1 $COMMON --tag r$r 2>&1 | grep -o '"time_us_median": [0-9.]*' | sed 's/^/bench paged64 cf1 h1 /'
  sleep 0.2
done
echo TIMING2_DONE
