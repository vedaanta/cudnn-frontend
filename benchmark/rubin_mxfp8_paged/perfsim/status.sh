#!/bin/bash
# Status of the PerfSim runs (anywhere with the scratch mount):  status.sh [perfsim_output dir]
OUT=${1:-${PERFSIM_MXB_HOST:-/home/scratch.vagarwalla_gpu/perfsim_mxb}/perfsim_output}
for d in $OUT/*/; do
  [ -d "$d" ] || continue
  n=$(basename $d)
  pfm=$(find $d/perfsim/pic_analysis -name full.pfm 2>/dev/null | head -1)
  t3d=$(find $d/perfsim/release/trace3d_gen -name test.tgz 2>/dev/null | head -1)
  apdf=$(find $d/perfsim/perfsim -name perfsim_apdf_0.xml.gz 2>/dev/null | head -1)
  log=$OUT/$n.flow.log
  last=$(grep -vE "^\s*$" $log 2>/dev/null | tail -1 | cut -c1-100)
  st="running"
  # a flow whose orchestrator died (e.g. the computelab frontends' 512 MiB per-user memory cgroup OOM-killing it) leaves
  # the directory behind: its remote LSF jobs may still write into it, but nothing will drive the later stages
  [ -f "$log" ] || st="NO FLOW LOG (orchestrator gone; resubmit under another suffix)"
  pgrep -f "[f]low.perfsim run.*$n" > /dev/null 2>&1 || [ ! -f "$log" ] || st="orchestrator not on this host"
  [ -n "$t3d" ] && st="trace3d OK"
  [ -n "$apdf" ] && st="SMART done"
  [ -n "$pfm" ] && st="DONE full.pfm"
  grep -qiE "error|fail|abort" $log 2>/dev/null && [ -z "$pfm" ] && st="$st (log has error lines; real text in perfsim/run.A/*/ace/run.log or results/000001/*.log)"
  printf "%-60s %-24s %s\n" "$n" "$st" "$last"
done
