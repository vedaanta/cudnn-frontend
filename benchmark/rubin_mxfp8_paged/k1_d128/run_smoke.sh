#!/bin/bash
# Smoke on the board: (1) product dense HALF+prefolded reference run with O dump, (2) bench dense all-defaults vs the
# product O (bitwise expected), (3) bench paged64 + CORRFAST + HOIST, corr default.  S=2048, B1 H4/2, none.
set -u
K=/tmp/vagarwalla/k1; OUT=$K/out; P="bash $K/k1_py.sh"
cd /tmp/vagarwalla
echo "== product dense"
$P --kernel product --paged 0 --corr default --corrfast 0 --hoist 0 --half 1 --prefolded 1 --seqlens 2048 --mask none --heads 4 2 --dump-o $OUT/o_prod_2k_none.pt --out $OUT/smoke.jsonl --tag smoke 2>&1 | tail -4
echo "== bench dense, all levers default (expect bitwise == product)"
$P --kernel bench --paged 0 --corr default --corrfast 0 --hoist 0 --half 1 --prefolded 1 --seqlens 2048 --mask none --heads 4 2 --compare-o $OUT/o_prod_2k_none.pt --out $OUT/smoke.jsonl --tag smoke 2>&1 | tail -4
echo "== bench paged64 + corrfast + hoist, corr default"
$P --kernel bench --paged 64 --corr default --corrfast 1 --hoist 1 --half 1 --prefolded 1 --seqlens 2048 --mask none --heads 4 2 --compare-o $OUT/o_prod_2k_none.pt --out $OUT/smoke.jsonl --tag smoke 2>&1 | tail -4
echo SMOKE_DONE
