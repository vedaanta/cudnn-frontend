#!/bin/bash
# App wrapper the APIC capturer launches on board 0614: bench.py --ncu-run at the PerfSim shape (B=1, H_q=H_kv=1,
# S=4096, no mask; 16 CTAs on 16 SMs), 3 warm-ups + 1 more launch.  APIC captures EVERY FROST launch (DumpControl
# func=cudnn_kernel*); the LAST k-instance is the warmed one to submit (a cold first execute sims ~40 % no-instruction
# stalls -- see project_perfsim_rubin_dsv3 "cold-capture artifact").
#   app_capture.sh <d> <config string> [S] [mask] [H_q H_kv]
# NO LD_PRELOAD here: the APIC injection owns LD_PRELOAD; APIC=1 makes board_py_0614.sh put libcudnn on LD_LIBRARY_PATH only.
set -u
D=$1; CFG=$2; S=${3:-4096}; M=${4:-none}; HQ=${5:-1}; HKV=${6:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
export APIC=1
export MXB_ROOT=${MXB_ROOT:-/tmp/vagarwalla/mxb}
exec bash $HERE/board_py_0614.sh $HERE/bench.py --d $D --config "$CFG" --seqlens $S --mask $M --heads $HQ $HKV --batch 1 --ncu-run --warmup 3 --validate none
