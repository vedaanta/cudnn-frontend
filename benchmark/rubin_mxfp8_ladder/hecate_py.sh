#!/bin/bash
# Env wrapper for hecate batch-xdr nodes (Vera aarch64 + Rubin GR100), run INSIDE the
# gitlab-master.nvidia.com#dl/dgx/pytorch:rubin-py3-devel container:
#   hecate_py.sh <python args...>
#   NCU=1 NCU_OUT=/home/vagarwalla/ladder/ncu/<name> NCU_SKIP=3 NCU_K=regex hecate_py.sh <python args...>
L=/home/vagarwalla/ladder
FE=${FE:-$L/fe}
PY=/home/vagarwalla/vitbench/venv/bin/python
T=${LADDER_TMP:-/tmp/vagarwalla_ladder}
mkdir -p $T/cache $T/dsl_cache $L/ncu $L/results
export PYTHONPATH=$FE/python:$FE/test/python
export LD_LIBRARY_PATH=/home/vagarwalla/fp8bench_arm_lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/home/vagarwalla/fp8bench_arm_lib/libcudnn.so.9
export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
export XDG_CACHE_HOME=$T/cache
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-$T/dsl_cache}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LADDER_F16EXP LADDER_NOSCALE LADDER_PAGED64
cd $FE/benchmark/rubin_mxfp8_ladder
if [ "${NCU:-0}" = "1" ]; then
  KARG=(); [ -n "${NCU_K:-}" ] && KARG=(-k "regex:$NCU_K")
  NCU_BIN=${NCU_BIN:-$L/ncu/ncu}
  exec $NCU_BIN --target-processes all ${NCU_SET:---set full} --launch-skip ${NCU_SKIP:-3} --launch-count ${NCU_COUNT:-1} \
      "${KARG[@]}" -f -o ${NCU_OUT:-$L/ncu/ladder} $PY "$@"
fi
exec $PY "$@"
