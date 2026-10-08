#!/bin/bash
# APIC capture of one bench config on board w2u1g-lc-0614 (run ON the board, under the shared GPU lock):
#   capture.sh <d> <config string> [case suffix]
#   e.g. capture.sh 128 "corr=default"            -> gr100_mxb_d128_corrdefault_p0_half_pf_prod_b1h1s4k_none
#        capture.sh 128 "corr=always,paged=64"     (bench kernel; every BENCH_* lever is set by bench.py from the config)
# Tool: the cuda_apic release copied to /tmp/vagarwalla/cuda_apic/<rel>/ (the board mounts no NFS; its shebang
# /home/utils/perl-5.8.8 does not exist here -> invoked through the system perl 5.38, core modules only).
# Output: /tmp/vagarwalla/perfsim_mxb/trace_output/<case>/CUDA_APIC_TRACES/PID*/p*_k<NNNN>_cudnn_kernel__kernel_*/cuda.tgz,
# one per FROST launch (k-index = launch order); submit the LAST one.  pull_traces.sh copies them to the scratch host.
set -uo pipefail
D=$1; CFG=$2; SUF=${3:-}
APIC=${APIC_TOOL:-/tmp/vagarwalla/cuda_apic/0.1.2026091607291789568946/cuda_apic_capture.pl}
LOCK=${LOCK:-/tmp/vagarwalla/ladder/gpu_timing.lock}
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$(bash $HERE/../board_py_0614.sh $HERE/../bench.py --d $D --config "$CFG" --print-name 2>/dev/null | tail -1)
[ -z "$NAME" ] && { echo "cannot resolve config name for '$CFG'"; exit 2; }
CASE=gr100_mxb_d${D}_${NAME}_b1h1s4k_none${SUF}
WORK=${PERFSIM_MXB:-/tmp/vagarwalla/perfsim_mxb}
export TMPDIR=$WORK/tmp
mkdir -p $TMPDIR $WORK/trace_output "$(dirname "$LOCK")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec 9>"$LOCK"
echo "=== waiting for GPU lock ($(date +%H:%M:%S))"; flock 9
cd $WORK/trace_output && rm -rf $CASE && mkdir -p $CASE
echo "=== capture $CASE ($(date +%H:%M:%S)) ==="
perl $APIC \
  --out_dir=$CASE \
  --app_binary=/bin/bash \
  --app_cmd_args="$HERE/app_capture.sh $D $CFG" \
  --knob DumpControl="(func=cudnn_kernel*)" \
  --capture_mode=discrete \
  --no_use_local_storage \
  --minimize_nvbit=0 \
  --dry_run_timeout 0 \
  2>&1 | tail -12
echo "exit: ${PIPESTATUS[0]} ($(date +%H:%M:%S))"
find $WORK/trace_output/$CASE -name cuda.tgz | sort | while read -r f; do echo "$(du -h "$f" | cut -f1) $f"; done
echo "LAST: $(find $WORK/trace_output/$CASE -name cuda.tgz | sort | tail -1)"
