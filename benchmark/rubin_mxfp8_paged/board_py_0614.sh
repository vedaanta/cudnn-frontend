#!/bin/bash
# Python env wrapper for this bench on board w2u1g-lc-0614 (x86_64 cc 10.7; no NFS mounts, no sudo; everything
# staged under /tmp/vagarwalla):  board_py_0614.sh <python args...>
#   MXB_WT   tree to run from (default /tmp/vagarwalla/wt_b = rsync of this worktree's python/ test/ benchmark/)
#   MXB_ROOT scratch root (default /tmp/vagarwalla/mxb): caches, logs, ncu reports; bench.py suffixes the two cache
#            dirs per config (kernel file + every BENCH_* value) -- a shared cache serves the WRONG kernel.
#   NCU=1 NCU_OUT=<report path, no ext> [NCU_K=cudnn_kernel__kernel NCU_SKIP=3 NCU_COUNT=1 NCU_SET=full NCU_METRICS=...]
#            wraps python in the locally staged Nsight Compute (/tmp/vagarwalla/ncu, internal 2026.4); the board has
#            RmProfilingAdminOnly=0 so no sudo is needed.
#   APIC=1   leave LD_PRELOAD alone (the APIC capturer owns it); libcudnn is then found through LD_LIBRARY_PATH.
# env_0614.sh exports PY (uv python3.10), SP (venv_sp site-packages: torch cu130 + internal CuTe DSL 0.3.0) and
# CUDNN_LIB (cudnn 9.26); its WT (the levers bench tree) is deliberately NOT used here.
source /tmp/vagarwalla/env_0614.sh
WT=${MXB_WT:-/tmp/vagarwalla/wt_b}
R=${MXB_ROOT:-/tmp/vagarwalla/mxb}
mkdir -p $R/cache $R/dsl_cache $R/logs $R/ncu $R/tmp $R/ncu_home
# The internal DSL installs via a .pth (sys.path += nvidia_cutlass_dsl/dsl_packages); PYTHONPATH skips .pth files.
export PYTHONPATH=$WT/python:$WT/test/python:$SP:$SP/nvidia_cutlass_dsl/dsl_packages
export LD_LIBRARY_PATH=$CUDNN_LIB:${LD_LIBRARY_PATH:-}
[ "${APIC:-0}" = 1 ] || export LD_PRELOAD=$CUDNN_LIB/libcudnn.so.9
export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$R/cache}
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-$R/dsl_cache}
export BENCH_LOG_DIR=$R/logs
export TMPDIR=$R/tmp
if [ "${NCU:-0}" = 1 ]; then
  NCU_BIN=${NCU_BIN:-/tmp/vagarwalla/ncu/ncu}
  KARG=(); [ -n "${NCU_K:-cudnn_kernel__kernel}" ] && KARG=(-k "regex:${NCU_K:-cudnn_kernel__kernel}")
  MARG=(); [ -n "${NCU_METRICS:-}" ] && MARG=(--metrics "$NCU_METRICS")
  # ncu needs a writable HOME for its section deployment; the 5 GB NFS home fills up -> keep it local.
  exec env HOME=$R/ncu_home $NCU_BIN --target-processes all --set ${NCU_SET:-full} "${MARG[@]}" --import-source no \
      --launch-skip ${NCU_SKIP:-3} --launch-count ${NCU_COUNT:-1} "${KARG[@]}" -f -o ${NCU_OUT:-$R/ncu/mxb} $PY "$@"
fi
exec $PY "$@"
