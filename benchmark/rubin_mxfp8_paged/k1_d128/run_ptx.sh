#!/bin/bash
# sm_107a PTX renderings: product vs bench (dense, all levers default) must agree; the lever arms show their deltas.
set -u
K=/tmp/vagarwalla/k1; D=$K/out/ptx; rm -rf $D; mkdir -p $D
export LV_WT=/tmp/vagarwalla/wt_k1
PY=/tmp/vagarwalla/lv_py.sh
cd /tmp/vagarwalla
BENCH_CORR=default BENCH_CORRFAST=0 BENCH_HOIST=0 $PY $K/ptx_md5.py $D/product product 2>&1 | grep "^ARM\|FAIL\|Error"
BENCH_CORR=default BENCH_CORRFAST=0 BENCH_HOIST=0 $PY $K/ptx_md5.py $D/bench_default bench 2>&1 | grep "^ARM\|FAIL\|Error"
BENCH_CORR=default BENCH_CORRFAST=1 BENCH_HOIST=1 $PY $K/ptx_md5.py $D/bench_cf_h bench 2>&1 | grep "^ARM\|FAIL\|Error"
BENCH_CORR=always BENCH_CORRFAST=1 BENCH_HOIST=1 $PY $K/ptx_md5.py $D/bench_always bench 2>&1 | grep "^ARM\|FAIL\|Error"
BENCH_CORR=never BENCH_CORRFAST=1 BENCH_HOIST=1 $PY $K/ptx_md5.py $D/bench_never bench 2>&1 | grep "^ARM\|FAIL\|Error"
BENCH_CORR=default BENCH_CORRFAST=1 BENCH_HOIST=1 $PY $K/ptx_md5.py $D/bench_paged bench 1 2>&1 | grep "^ARM\|FAIL\|Error"
echo PTX_DONE
