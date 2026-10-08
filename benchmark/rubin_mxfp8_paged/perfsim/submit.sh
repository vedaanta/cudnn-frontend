#!/bin/bash
# Submit captured traces to PerfSim gr100 (SSAF route).  RUN ON computelab-sc-01-frontend-NN (ssh to a numbered
# frontend directly; the load-balanced alias puts pkill/pgrep on different hosts).  Needs ~/.ssaf_token + the SSAF RunAs
# install on the computelab home (see project_perfsim_rubin_dsv3: SSAF UNBLOCKED).  NOT run by the harness agent.
#   submit.sh <suffix> "<case glob or list>"      e.g. submit.sh r1 "gr100_mxb_d128_*"
# Flow = latest; config_gr100.yaml = perfsim_ladder's (SSAF_args pmMode PM_KERNEL, gr100 TMEM 288 KB knob, the enum-typed
# uninitialized*Check knobs dropped).  One flow per trace, in the background; ~75-115 min per PIC at B1 H1 S4k.
set -u
SUF=${1:-r1}; PAT=${2:-"gr100_mxb_*"}
export PATH=/home/scratch.svc_compute_arch/release/flow.perfsim/${FLOW_REL:-latest}:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${PERFSIM_MXB_HOST:-/home/scratch.vagarwalla_gpu/perfsim_mxb}
CFG=${CONFIG:-$HERE/config_gr100.yaml}
OUT=$WORK/perfsim_output
mkdir -p $OUT
cd $WORK
for CASEDIR in $(ls -d $WORK/trace_output/$PAT 2>/dev/null); do
  CASE=$(basename $CASEDIR)
  TRACE=$(find $CASEDIR -name cuda.tgz | sort | tail -1)   # last captured instance = warmed execution
  if [ -z "$TRACE" ]; then echo "no trace for $CASE"; continue; fi
  if [ -d $OUT/${CASE}_$SUF ]; then echo "exists: $OUT/${CASE}_$SUF (pick another suffix)"; continue; fi
  echo "$CASE: $TRACE"
  nohup flow.perfsim run -chip gr100 -trace "$TRACE" -config $CFG \
    -dir $OUT/${CASE}_$SUF -enableMorph -pic > $OUT/${CASE}_$SUF.flow.log 2>&1 &
  echo "  pid $! -> $OUT/${CASE}_$SUF"
  sleep 2
done
