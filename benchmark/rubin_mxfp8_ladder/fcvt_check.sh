#!/bin/bash
# LADDER_FCVT validation + timing on board 0030 (needs the GPU back).
set -u
WT=${WT:-/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/rubin-ladder-bench}
B=$WT/benchmark/rubin_mxfp8_ladder
PYBIN=${PY:-python3}  # venv python for the torch-based O diff
L=/tmp/vagarwalla/ladder; OUT=$L/results_fcvt.jsonl; rm -f $OUT; mkdir -p $L/dump
cd $B
run() { local R=$1 TAG=$2 M=$3 EXTRA=$4 ENVS=$5
  timeout ${TMO:-420} env $ENVS bash board_py.sh bench_ladder.py --rung $R --seqlens ${SEQ:-32768} --mask $M --validate ${VAL:-none} --reps 20 --rounds 3 --out $OUT --tag "$TAG" $EXTRA 2>&1 | grep -E "Traceback|Error|error\[|error:|usage:|unrecognized|rel_err|\"kmod\"" | sed -E 's/.*"rel_err": ([0-9.e-]+).*/  rel_err \1/; s/.*"kmod": (\{[^}]*\}).*/  kmod \1/' | head -8
  local rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && echo "  !! $TAG $M exit $rc"; }
NS2="LADDER_NOSCALE=1 LADDER_SUB2=1"
NSF="LADDER_NOSCALE=1 LADDER_FCVT=1"
echo "=== validation $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader)"
# A/B at 2k: NOSCALE+SUB2 (today, RN) vs NOSCALE+FCVT (fused, RZ) -- tolerance vs the fp32 reference; |dO| between them reported
SEQ=2048 VAL=all run F ns2_val none "--dump-o $L/dump/o_ns2_none.pt" "$NS2"
SEQ=2048 VAL=all run F nsf_val none "--dump-o $L/dump/o_nsf_none.pt" "$NSF"
SEQ=2048 VAL=all run F ns2_val causal "--dump-o $L/dump/o_ns2_causal.pt" "$NS2"
SEQ=2048 VAL=all run F nsf_val causal "--dump-o $L/dump/o_nsf_causal.pt" "$NSF"
SEQ=4096 VAL=all run F ns2_val4k none "" "$NS2"
SEQ=4096 VAL=all run F nsf_val4k none "" "$NSF"
SEQ=4096 VAL=all run F ns2_val4k causal "" "$NS2"
SEQ=4096 VAL=all run F nsf_val4k causal "" "$NSF"
SEQ=4096 VAL=all run F fcvt_val none "" "LADDER_FCVT=1"
SEQ=4096 VAL=all run F fcvt_val causal "" "LADDER_FCVT=1"
SEQ=2048 VAL=all run F fcvt_ramp none "" "LADDER_FCVT=1 LADDER_KV_RAMP=1"
SEQ=2048 VAL=all run F nsf_ramp causal "" "$NSF LADDER_KV_RAMP=1"
SEQ=2048 VAL=all run F pref_fcvt_val none "" "LADDER_PREF=1 LADDER_FCVT=1"
PYTHONPATH=${SP:-/home/scratch.vagarwalla_libs_1/venv_frost/lib/python3.10/site-packages} $PYBIN - <<PY
import torch
for m in ("none", "causal"):
    a = torch.load("$L/dump/o_ns2_%s.pt" % m); b = torch.load("$L/dump/o_nsf_%s.pt" % m)
    d = (a.float() - b.float()).abs()
    print(f"DIFF SUB2 vs FCVT {m}: identical={torch.equal(a, b)} max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} (RZ fused convert: small nonzero expected)")
PY
echo "=== timing 32k $(date +%H:%M:%S)"
for i in 1 2; do
  for M in "none|" "causal|--sched lpt"; do
    MASK=${M%%|*}; EX=${M#*|}
    run F f$i $MASK "$EX" ""
    run F ns2_$i $MASK "$EX" "$NS2"
    run F nsf_$i $MASK "$EX" "$NSF"
    run F fcvt$i $MASK "$EX" "LADDER_FCVT=1"
    run F stack$i $MASK "$EX" "$NSF LADDER_HOIST=1 LADDER_CORR_NORESCALE=2"
    run F preff$i $MASK "$EX" "LADDER_PREF=1 LADDER_FCVT=1 LADDER_NOSCALE=1"
  done
done
python3 - <<PY
import json, collections, re, statistics
rows = collections.defaultdict(list)
for l in open("$OUT"):
    r = json.loads(l)
    if r["S"] in (2048, 4096):
        v = r.get("validation"); km = r.get("kmod") or {}
        print("VAL", r["tag"], r["mask"], "rel %.4f max_abs %.3e finite %s" % (v["rel_err"], v["max_abs_err"], v["finite"]) if v else None, "| kmod", {x: y for x, y in km.items() if x in ("FCVT", "NOSCALE", "PREF", "HOIST")}); continue
    rows[(re.sub(r"\d+$", "", r["tag"]).rstrip("_"), r["mask"])].append((r["time_us_graph"], r["clocks_after"].split(",")[0].strip()))
F = {m: statistics.median([x[0] for x in rows[("f", m)]]) for m in ("none", "causal")}
for k, v in sorted(rows.items()):
    med = statistics.median([x[0] for x in v]); print(f"{k[0]:7s} {k[1]:6s} median {med:8.1f} vs F {100*(med/F[k[1]]-1):+5.1f}%  {v}")
PY
echo FCVT_DONE
