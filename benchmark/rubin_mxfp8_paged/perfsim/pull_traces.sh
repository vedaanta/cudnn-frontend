#!/bin/bash
# Copy the LAST captured k-instance of every case (or one case) from the board to the scratch host, keeping the
# CUDA_APIC_TRACES/PID*/p*_k*/ path flow.perfsim expects:  pull_traces.sh [case glob]   (run on the scratch host)
set -u
H=${BOARD:-w2u1g-lc-0614}; SSH="/usr/bin/ssh -o BatchMode=yes"
SRC=${PERFSIM_MXB:-/tmp/vagarwalla/perfsim_mxb}/trace_output
DST=${DST:-/home/scratch.vagarwalla_gpu/perfsim_mxb/trace_output}
PAT=${1:-gr100_mxb_*}
mkdir -p $DST
for CASE in $($SSH $H "ls -d $SRC/$PAT 2>/dev/null | xargs -n1 basename"); do
  LAST=$($SSH $H "find $SRC/$CASE -name cuda.tgz | sort | tail -1")
  [ -z "$LAST" ] && { echo "$CASE: no cuda.tgz"; continue; }
  REL=${LAST#$SRC/}
  mkdir -p "$DST/$(dirname "$REL")"
  rsync -a -e "$SSH" "$H:$(dirname "$LAST")/" "$DST/$(dirname "$REL")/" && echo "$CASE: $DST/$REL ($(du -h "$DST/$REL" | cut -f1))"
done
