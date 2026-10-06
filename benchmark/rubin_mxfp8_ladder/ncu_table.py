#!/usr/bin/env python3
"""Extract the ladder's ncu metrics from a directory of <rung>_<mask>_<S>.ncu-repz reports.

Usage: ncu_table.py --ncu <ncu binary> --dir <reports dir> [--timing results.jsonl] --out <prefix>
Writes <prefix>.csv, <prefix>.json and prints a markdown table.  Import is done with the ncu CLI
(``--import --csv --page raw``), so any host ncu of a compatible version can read reports made on
another machine.
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import subprocess

B, HQ, HKV, D = 1, 32, 2, 128

# column -> (metric name, scale)
METRICS = {
    "dur_us": ("gpu__time_duration.sum", 1.0),
    "clk_ghz": ("sm__cycles_elapsed.avg.per_second", 1.0),
    "cyc_elapsed": ("sm__cycles_elapsed.avg", 1.0),
    "cyc_active": ("sm__cycles_active.avg", 1.0),
    "sm_thru_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0),
    "tensor_act_pct": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 1.0),
    "tensor_elapsed_pct": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 1.0),
    "tc_pipe_pct": ("sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active", 1.0),
    "xu_mufu_pct": ("sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", 1.0),
    "fma_pct": ("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", 1.0),
    "fmaheavy_pct": ("sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active", 1.0),
    "alu_pct": ("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active", 1.0),
    "tmem_inst_pct": ("sm__inst_executed_pipe_tmem.avg.pct_of_peak_sustained_active", 1.0),
    "tma_pipe_pct": ("sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active", 1.0),
    "issue_pct": ("smsp__issue_active.avg.pct_of_peak_sustained_active", 1.0),
    "ipc_active": ("sm__inst_executed.avg.per_cycle_active", 1.0),
    "inst_total": ("sm__inst_executed.sum", 1.0),
    "warp_lat_cyc": ("smsp__average_warp_latency_per_inst_issued.ratio", 1.0),
    "stall_long_sb": ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", 1.0),
    "stall_short_sb": ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", 1.0),
    "stall_wait": ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", 1.0),
    "stall_barrier": ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", 1.0),
    "stall_math_throttle": ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", 1.0),
    "warps_active_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", 1.0),
    "dram_gbs": ("dram__bytes.sum.per_second", 1.0),
    "dram_pct": ("dram__throughput.avg.pct_of_peak_sustained_elapsed", 1.0),
    "l2_pct": ("lts__throughput.avg.pct_of_peak_sustained_elapsed", 1.0),
    "l2_hit_pct": ("lts__t_sector_hit_rate.pct", 1.0),
    "l1_pct": ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", 1.0),
    "tma_ld_mb": ("l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum", 1.0),
    "fp8_mma_ops": ("sm__ops_path_tensor_op_utcqmma_src_fp8_dst_fp32_sparsity_off.sum", 1.0),
    "regs": ("launch__registers_per_thread", 1.0),
    "smem_kb": ("launch__shared_mem_per_block", 1.0),
    "grid": ("launch__grid_size", 1.0),
    "waves": ("launch__waves_per_multiprocessor", 1.0),
}
UNIT_SCALE = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3, "byte": 1e-6, "Mbyte/s": 1e-3, "Gbyte/s": 1.0, "Tbyte/s": 1e3, "Mhz": 1e-3, "Ghz": 1.0, "Khz": 1e-6}


def _num(v):
    v = v.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


def read_report(ncu, path):
    # ncu needs a writable HOME to deploy its section files, else it emits an empty CSV.
    env = dict(os.environ)
    env["HOME"] = os.environ.get("HOME_NCU") or os.path.join(os.path.dirname(os.path.abspath(path)), ".ncu_home")
    os.makedirs(env["HOME"], exist_ok=True)
    out = subprocess.run([ncu, "--import", path, "--csv", "--page", "raw"], capture_output=True, text=True, env=env)
    rows = list(csv.reader(io.StringIO(out.stdout)))
    if len(rows) < 3 or "Kernel Name" not in rows[0]:
        raise RuntimeError(f"ncu import failed for {path}: {out.stderr[:300]}")
    hdr, units, vals = rows[0], rows[1], rows[2]
    idx = {h: i for i, h in enumerate(hdr)}
    rec = {"kernel": vals[idx["Kernel Name"]][:60]}
    for col, (m, _) in METRICS.items():
        if m not in idx:
            rec[col] = None
            continue
        v = _num(vals[idx[m]])
        u = units[idx[m]]
        if v is None:
            rec[col] = None
            continue
        if col == "dur_us":
            v *= UNIT_SCALE.get(u, 1.0)
        elif col == "clk_ghz":
            v *= UNIT_SCALE.get(u, 1.0)
        elif col == "dram_gbs":
            v *= {"Mbyte/s": 1e-3, "Gbyte/s": 1.0, "Tbyte/s": 1e3, "Kbyte/s": 1e-6}.get(u, 1.0)
        elif col == "tma_ld_mb":
            v *= {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}.get(u, 1.0)
        elif col == "smem_kb":
            v *= {"byte/block": 1e-3, "Kbyte/block": 1.0, "Mbyte/block": 1e3}.get(u, 1.0)
        rec[col] = v
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--timing", default=None, help="results.jsonl from bench_ladder.py (graph-replay times)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    timing = {}
    if args.timing and os.path.exists(args.timing):
        for line in open(args.timing):
            r = json.loads(line)
            timing[(r["rung"], r["mask"], r["S"], r.get("sched", "natural"))] = r
    recs = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.ncu-rep*"))):
        name = os.path.basename(path).split(".ncu-rep")[0]
        m = re.match(r"([A-G])_(causal|none)_(\d+)(?:_(\w+))?$", name)
        if not m:
            continue
        rung, mask, S, sched = m.group(1), m.group(2), int(m.group(3)), m.group(4) or "natural"
        rec = dict(rung=rung, mask=mask, S=S, sched=sched)
        rec.update(read_report(args.ncu, path))
        causal = mask == "causal"
        flops = 4.0 * B * HQ * S * S * D * (0.5 * (1 + 1.0 / S) if causal else 1.0)
        if rec.get("dur_us"):
            rec["tflops_ncu"] = flops / (rec["dur_us"] * 1e-6) / 1e12
        if rec.get("cyc_elapsed") and rec.get("cyc_active"):
            rec["sm_active_pct"] = 100.0 * rec["cyc_active"] / rec["cyc_elapsed"]
        t = timing.get((rung, mask, S, sched))
        if t:
            rec["graph_us"] = t.get("time_us_graph")
            rec["tflops_graph"] = t.get("tflops")
        recs.append(rec)
    recs.sort(key=lambda r: (r["mask"], r["sched"], r["S"], r["rung"]))
    cols = ["rung", "mask", "sched", "S", "dur_us", "graph_us", "tflops_ncu", "clk_ghz", "sm_active_pct", "tensor_act_pct", "tensor_elapsed_pct", "xu_mufu_pct", "fma_pct", "fmaheavy_pct", "alu_pct", "issue_pct", "ipc_active", "warp_lat_cyc", "stall_long_sb", "stall_wait", "stall_short_sb", "stall_barrier", "dram_gbs", "l2_pct", "l2_hit_pct", "tma_ld_mb", "regs", "smem_kb", "waves", "inst_total", "fp8_mma_ops"]
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    with open(args.out + ".json", "w") as f:
        json.dump(recs, f, indent=1)
    # markdown
    show = ["rung", "mask", "sched", "S", "dur_us", "tflops_ncu", "clk_ghz", "sm_active_pct", "tensor_act_pct", "xu_mufu_pct", "fma_pct", "alu_pct", "issue_pct", "warp_lat_cyc", "stall_long_sb", "dram_gbs", "l2_hit_pct", "regs", "smem_kb"]
    print("| " + " | ".join(show) + " |")
    print("|" + "---|" * len(show))
    for r in recs:
        cells = []
        for c in show:
            v = r.get(c)
            cells.append("" if v is None else (f"{v:.0f}" if c in ("S", "regs", "smem_kb", "dur_us") and v >= 100 else f"{v:.2f}" if isinstance(v, float) else str(v)))
        print("| " + " | ".join(cells) + " |")
    print(f"\n{len(recs)} reports -> {args.out}.csv / .json")


if __name__ == "__main__":
    main()
