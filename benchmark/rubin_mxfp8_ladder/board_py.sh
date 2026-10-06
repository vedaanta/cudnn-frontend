#!/bin/bash
# Env wrapper for board w2u1g-lc-0030 (x86 Rubin GR100 cc 10.7).
#   board_py.sh <python args...>          uv python3.10 + venv_frost site-packages (torch cu130, internal CuTe DSL)
#   NCU=1 NCU_OUT=/tmp/... NCU_SKIP=3 NCU_K=regex board_py.sh <python args...>   wrap in sudo ncu
WT=${WT:-/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/rubin-ladder-bench}
PY=$(ls -d /home/vagarwalla/.local/share/uv/python/cpython-3.10.*-linux-x86_64-gnu/bin/python3.10 | tail -1)
SP=/home/scratch.vagarwalla_libs_1/venv_frost/lib/python3.10/site-packages
T=/tmp/vagarwalla/ladder
mkdir -p $T/cache $T/ncu $T/dsl_cache $T/dsl_cache_root $T/cache_root
# The internal DSL installs via a .pth (sys.path += nvidia_cutlass_dsl/dsl_packages); PYTHONPATH skips .pth files.
export PYTHONPATH=$WT/python:$WT/test/python:$SP:$SP/nvidia_cutlass_dsl/dsl_packages
export LD_PRELOAD=/home/scratch.vagarwalla_gpu/cudnn_9.26.0.51/lib/libcudnn.so.9
export LD_LIBRARY_PATH=/home/scratch.vagarwalla_gpu/cudnn_9.26.0.51/lib:${LD_LIBRARY_PATH:-}
export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
export XDG_CACHE_HOME=$T/cache
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-$T/dsl_cache}
export LADDER_F16EXP LADDER_NOSCALE LADDER_PAGED64 LADDER_CORRFAST LADDER_SDOUBLE LADDER_PSMEM
cd $WT/benchmark/rubin_mxfp8_ladder
if [ "${NCU:-0}" = "1" ]; then
  KARG=(); [ -n "${NCU_K:-}" ] && KARG=(-k "regex:$NCU_K")
  # root is squashed on NFS and cannot read $HOME: run a local copy of the uv python from /tmp.
  PYROOT=/tmp/vagarwalla/py310
  [ -x $PYROOT/bin/python3.10 ] || cp -r $(dirname $(dirname $PY)) $PYROOT
  PY=$PYROOT/bin/python3.10
  mkdir -p $T/root_home
  # forward EVERY LADDER_* knob through sudo (the first version forwarded six and silently profiled plain F for the rest)
  LADDER_ENV=$(env | grep -E '^LADDER_[A-Z0-9_]+=' | tr '\n' ' ')
  NCU_BIN=/home/scratch.svc_compute_arch/release/nsightCompute/internal/x86_64/latest/ncu
  exec sudo -n env HOME=$T/root_home PYTHONPATH=$PYTHONPATH LD_PRELOAD=$LD_PRELOAD LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
      CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 XDG_CACHE_HOME=$T/cache_root CUTE_DSL_CACHE_DIR=$T/dsl_cache_root \
      $LADDER_ENV \
      $NCU_BIN --target-processes all ${NCU_SET:---set full} --launch-skip ${NCU_SKIP:-3} --launch-count ${NCU_COUNT:-1} \
      "${KARG[@]}" -f -o ${NCU_OUT:-$T/ncu/ladder} $PY "$@"
fi
exec $PY "$@"
