#!/bin/bash
# K1 validation matrix on board 0614 (functional; no GPU lock needed).  Appends JSON lines to out/val.jsonl.
set -u
K=/tmp/vagarwalla/k1; OUT=$K/out; P="bash $K/k1_py.sh"; J=$OUT/val.jsonl
cd /tmp/vagarwalla
rm -f $J
STACK="--half 1 --prefolded 1"
run() { echo "== $(date +%T) $*"; $P "$@" --out $J 2>&1 | grep -v "RuntimeWarning\|AttrBuilder\|experimental API" | tail -2 | cut -c1-400; }
# A. product dense dumps + bench dense (all levers default) compare: bitwise expected
for m in none causal; do for S in 2048 4096; do
  run --kernel product --paged 0 --corr default --corrfast 0 --hoist 0 $STACK --seqlens $S --mask $m --heads 4 2 --dump-o $OUT/o_prod_${S}_${m}.pt --tag A_product
  run --kernel bench --paged 0 --corr default --corrfast 0 --hoist 0 $STACK --seqlens $S --mask $m --heads 4 2 --compare-o $OUT/o_prod_${S}_${m}.pt --tag A_bench_dense
done; done
# B/C/D. paged64 + CORRFAST + HOIST stack: corr default / always / never (rescale emulation on each)
for corr in default always never; do for m in none causal; do for S in 2048 4096; do
  run --kernel bench --paged 64 --corr $corr --corrfast 1 --hoist 1 $STACK --seqlens $S --mask $m --heads 4 2 --rescale-stats 1 --compare-o $OUT/o_prod_${S}_${m}.pt --tag B_paged_$corr
done; done; done
for corr in default always; do for m in none causal; do for S in 2048 4096; do
  run --kernel bench --paged 64 --corr $corr --corrfast 1 --hoist 1 $STACK --seqlens $S --mask $m --heads 1 1 --tag B_paged_${corr}_h11
done; done; done
# E. K/V ramp (per-32-token E8M0 scales differ 2x): dense bench dump, paged compare (bitwise expected)
for m in none causal; do
  run --kernel bench --paged 0 --corr default --corrfast 1 --hoist 1 $STACK --seqlens 2048 --mask $m --heads 4 2 --kv-ramp 1 --dump-o $OUT/o_bench_dense_ramp_${m}.pt --tag E_ramp_dense
  run --kernel bench --paged 64 --corr default --corrfast 1 --hoist 1 $STACK --seqlens 2048 --mask $m --heads 4 2 --kv-ramp 1 --compare-o $OUT/o_bench_dense_ramp_${m}.pt --tag E_ramp_paged
done
# F. K tile growth 1.3^t (forces rescales): product vs default / always / never (never is WRONG here by design)
for m in none causal; do
  run --kernel product --paged 0 --corr default --corrfast 0 --hoist 0 $STACK --seqlens 2048 --mask $m --heads 4 2 --kv-growth 1.3 --dump-o $OUT/o_prod_growth_${m}.pt --rescale-stats 1 --tag F_growth_product
  for corr in default always never; do
    run --kernel bench --paged 64 --corr $corr --corrfast 1 --hoist 1 $STACK --seqlens 2048 --mask $m --heads 4 2 --kv-growth 1.3 --compare-o $OUT/o_prod_growth_${m}.pt --rescale-stats 1 --tag F_growth_$corr
  done
done
# G. prefolded through the env override; H. lever cross combos
run --kernel bench --paged 64 --corr default --corrfast 1 --hoist 1 --half 1 --prefolded 1 --prefolded-via env --seqlens 2048 --mask none --heads 4 2 --compare-o $OUT/o_prod_2048_none.pt --tag G_env_prefolded
run --kernel bench --paged 64 --corr always --corrfast 0 --hoist 0 $STACK --seqlens 2048 --mask none --heads 4 2 --compare-o $OUT/o_prod_2048_none.pt --tag H_always_cf0_h0
run --kernel bench --paged 0 --corr never --corrfast 1 --hoist 1 $STACK --seqlens 2048 --mask causal --heads 4 2 --compare-o $OUT/o_prod_2048_causal.pt --tag H_never_dense
run --kernel bench --paged 64 --corr default --corrfast 1 --hoist 1 --half 0 --prefolded 0 --seqlens 2048 --mask none --heads 4 2 --tag H_float_unfolded
echo VAL_DONE
