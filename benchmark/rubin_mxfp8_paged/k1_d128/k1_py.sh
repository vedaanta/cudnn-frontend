#!/bin/bash
# Board-side python wrapper for the K1 tree: LV_WT=/tmp/vagarwalla/wt_k1 through the levers wrapper, with the K1
# DSL cache root (the driver suffixes CUTE_DSL_CACHE_DIR and the wrapper's XDG_CACHE_HOME per lever set).
# Usage: bash /tmp/vagarwalla/k1/k1_py.sh <driver args>
export LV_WT=/tmp/vagarwalla/wt_k1
export CUTE_DSL_CACHE_DIR=/tmp/vagarwalla/ladder/dsl_cache_k1
mkdir -p $CUTE_DSL_CACHE_DIR /tmp/vagarwalla/k1/out
exec /tmp/vagarwalla/lv_py.sh /tmp/vagarwalla/k1/bench_k1.py "$@"
