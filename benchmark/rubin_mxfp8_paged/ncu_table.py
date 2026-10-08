#!/usr/bin/env python3
"""Extract the per-unit ncu metrics of this bench from a directory of reports written by run_ncu.sh.

  ncu_table.py --dir <reports dir> [--ncu <ncu binary>] [--timing results.jsonl] --out <prefix>

Report names: d<d>_<config name>_<mask>_<S>[_<sched>].ncu-rep*; run_ncu.sh also leaves <name>.raw.csv (the raw metrics
page) and <name>.sass.csv (the SASS source page) next to each report -- those are read first, so no ncu binary is
needed off-board; with --ncu the missing CSVs are produced by `ncu --import` (needs a writable HOME).
Writes <prefix>.csv / <prefix>.json and prints a markdown table.  Per-unit utilisation columns are % of each pipe's
peak over SM-active cycles; the table also carries the raw executed-instruction counts per pipe and the SASS-derived
opcode-class counts (sass_tools.ncu_source) that unit_sol.py turns into per-step issue / pipe SOLs.
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_tools  # noqa: E402

# column -> metric name (values are read with their unit and scaled to the column's unit)
METRICS = {
    "dur_us": "gpu__time_duration.sum",
    "clk_ghz": "sm__cycles_elapsed.avg.per_second",
    "cyc_elapsed": "sm__cycles_elapsed.avg",
    "cyc_active": "sm__cycles_active.avg",
    "sm_count": "device__attribute_multiprocessor_count",
    "sm_thru_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_act_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "tensor_elapsed_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_cycles": "sm__pipe_tensor_cycles_active.sum",
    "xu_mufu_pct": "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "fma_pct": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "fmaheavy_pct": "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active",
    "fmalite_pct": "sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active",
    "fma_inst_pct": "sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active",
    "alu_pct": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "alu_inst_pct": "sm__inst_executed_pipe_alu.sum.pct_of_peak_sustained_active",
    "tmem_inst_pct": "sm__inst_executed_pipe_tmem.avg.pct_of_peak_sustained_active",
    "lsu_inst_pct": "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "uniform_inst_pct": "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active",
    "adu_pct": "sm__pipe_adu_cycles_active.avg.pct_of_peak_sustained_active",
    "tma_pipe_pct": "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active",
    "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "ipc_active": "sm__inst_executed.avg.per_cycle_active",
    "inst_total": "smsp__inst_executed.sum",
    "inst_issued": "smsp__inst_issued.sum",
    "branch_inst": "smsp__inst_executed_op_branch.sum",
    "warp_lat_cyc": "smsp__average_warp_latency_per_inst_issued.ratio",
    "warps_active_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "stall_long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_short_sb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "stall_wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "stall_barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "stall_math_throttle": "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "stall_mio_throttle": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "stall_membar": "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "stall_sleeping": "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "stall_not_selected": "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "stall_no_instruction": "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "stall_dispatch_stall": "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "stall_lg_throttle": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "stall_branch_resolving": "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "stall_misc": "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    "stall_selected": "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
    "stall_drain": "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "stall_imc_miss": "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
    "smem_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "smem_ld_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "smem_st_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "smem_bank_conflicts_ld": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_bank_conflicts_st": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_data_pipe_pct": "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
    "tma_ld_mb": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sectors": "lts__t_sectors.sum",  # x 32 B
    "l2_sectors_pct": "lts__t_sectors.sum.pct_of_peak_sustained_elapsed",
    "l2_read_requests": "lts__t_requests_op_read.sum",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "dram_gbs": "dram__bytes.sum.per_second",
    "dram_peak_kb_per_cycle": "dram__bytes.sum.peak_sustained",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "fp8_mma_ops": "sm__ops_path_tensor_op_utcqmma_src_fp8_dst_fp32_sparsity_off.sum",  # FLOPs (2 per MAC) incl. the ones-MMA row-sum
    "local_ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_st_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "regs": "launch__registers_per_thread",
    "smem_kb": "launch__shared_mem_per_block",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "waves": "launch__waves_per_multiprocessor",
}
TO_US = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "s": 1e6}
TO_GHZ = {"hz": 1e-9, "Khz": 1e-6, "Mhz": 1e-3, "Ghz": 1.0, "cycle/nsecond": 1.0, "cycle/usecond": 1e-3, "cycle/second": 1e-9}
TO_GBS = {"byte/s": 1e-9, "Kbyte/s": 1e-6, "Mbyte/s": 1e-3, "Gbyte/s": 1.0, "Tbyte/s": 1e3, "byte/second": 1e-9, "Kbyte/second": 1e-6, "Mbyte/second": 1e-3, "Gbyte/second": 1.0, "Tbyte/second": 1e3}
TO_MB = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}
TO_KB_BLOCK = {"byte/block": 1e-3, "Kbyte/block": 1.0, "Mbyte/block": 1e3}
NAME_RX = re.compile(r"^d(\d+)_(.+)_(causal|none)_(\d+)(?:_(lpt|lpt_l2))?$")


def _num(v):
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def ncu_import(ncu, report, page, extra=()):
    env = dict(os.environ)
    env["HOME"] = os.path.join(os.path.dirname(os.path.abspath(report)), ".ncu_home")
    os.makedirs(env["HOME"], exist_ok=True)
    out = subprocess.run([ncu, "--import", report, "--csv", "--page", page, *extra], capture_output=True, text=True, env=env)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"ncu import failed for {report}: {out.stderr[:300]}")
    return out.stdout


def read_raw(text):
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3 or "Kernel Name" not in rows[0]:
        raise RuntimeError("raw page: unexpected layout")
    hdr, units, vals = rows[0], rows[1], rows[2]
    idx = {h: i for i, h in enumerate(hdr)}
    rec = {"kernel": vals[idx["Kernel Name"]][:80]}
    for col, m in METRICS.items():
        if m not in idx:
            rec[col] = None
            continue
        v = _num(vals[idx[m]])
        u = units[idx[m]]
        if v is None:
            rec[col] = None
            continue
        if col == "dur_us":
            v *= TO_US.get(u, 1.0)
        elif col == "clk_ghz":
            v *= TO_GHZ.get(u, 1.0)
        elif col in ("dram_gbs", "l2_gbs"):
            v *= TO_GBS.get(u, 1.0)
        elif col in ("tma_ld_mb", "l2_bytes_mb", "dram_mb"):
            v *= TO_MB.get(u, 1.0)
        elif col == "smem_kb":
            v *= TO_KB_BLOCK.get(u, 1.0)
        rec[col] = v
    return rec


def read_report(path, ncu):
    base = path.split(".ncu-rep")[0]
    raw_csv, sass_csv = base + ".raw.csv", base + ".sass.csv"
    if os.path.exists(raw_csv) and os.path.getsize(raw_csv) > 0:
        rec = read_raw(open(raw_csv, errors="ignore").read())
    elif ncu:
        text = ncu_import(ncu, path, "raw")
        open(raw_csv, "w").write(text)
        rec = read_raw(text)
    else:
        raise RuntimeError(f"{raw_csv} missing and no --ncu given")
    if not (os.path.exists(sass_csv) and os.path.getsize(sass_csv) > 0) and ncu:
        try:
            open(sass_csv, "w").write(ncu_import(ncu, path, "source", ("--print-source", "sass")))
        except RuntimeError as e:
            print(f"  (no SASS page for {os.path.basename(path)}: {e})", file=sys.stderr)
    if os.path.exists(sass_csv) and os.path.getsize(sass_csv) > 0:
        try:
            h = sass_tools.ncu_source(sass_csv, None)
            rec["sass_exec_total"] = h["executed_total"]
            rec["sass_by_class"] = h["by_class"]
            rec["sass_top"] = dict(list(h["opcodes"].items())[:40])
        except SystemExit as e:
            print(f"  (SASS page unusable for {os.path.basename(path)}: {e})", file=sys.stderr)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ncu", default=None, help="ncu binary for reports without the exported CSV pages")
    ap.add_argument("--timing", default=None, help="results.jsonl from bench.py (graph-replay times, joined on (d, name, mask, S, sched))")
    ap.add_argument("--heads", type=int, nargs=2, default=[32, 8])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    timing = {}
    if args.timing and os.path.exists(args.timing):
        for line in open(args.timing):
            if line.strip():
                r = json.loads(line)
                timing[(r["d"], r["name"], r["mask"], r["S"], r.get("sched", "natural"))] = r
    recs = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.ncu-rep*"))):
        name = os.path.basename(path).split(".ncu-rep")[0]
        m = NAME_RX.match(name)
        if not m:
            print(f"skip {name}: not d<d>_<config>_<mask>_<S>", file=sys.stderr)
            continue
        d, cfg, mask, S, sched = int(m.group(1)), m.group(2), m.group(3), int(m.group(4)), m.group(5) or "natural"
        rec = dict(d=d, name=cfg, mask=mask, S=S, sched=sched, report=name)
        try:
            rec.update(read_report(path, args.ncu))
        except RuntimeError as e:
            print(f"skip {name}: {e}", file=sys.stderr)
            continue
        causal = mask == "causal"
        flops = 4.0 * args.batch * args.heads[0] * S * S * d * (0.5 * (1 + 1.0 / S) if causal else 1.0)
        if rec.get("dur_us"):
            rec["tflops_ncu"] = flops / (rec["dur_us"] * 1e-6) / 1e12
            if rec.get("l2_sectors"):
                rec["l2_bytes_mb"] = rec["l2_sectors"] * 32 / 1e6
                rec["l2_gbs"] = rec["l2_sectors"] * 32 / (rec["dur_us"] * 1e-6) / 1e9
                if rec.get("l2_sectors_pct"):
                    rec["l2_peak_tbs_ncu"] = rec["l2_gbs"] / (rec["l2_sectors_pct"] / 100.0) / 1e3
            if rec.get("dram_gbs"):
                rec["dram_mb"] = rec["dram_gbs"] * 1e9 * rec["dur_us"] * 1e-6 / 1e6
        if rec.get("cyc_elapsed") and rec.get("cyc_active"):
            rec["sm_active_pct"] = 100.0 * rec["cyc_active"] / rec["cyc_elapsed"]
        t = timing.get((d, cfg, mask, S, sched))
        if t:
            rec["graph_us"] = t.get("time_us_graph")
            rec["tflops_graph"] = t.get("tflops")
            rec["cta_mma"] = t.get("cta_mma")
        recs.append(rec)
    recs.sort(key=lambda r: (r["d"], r["mask"], r["sched"], r["S"], r["name"]))
    cols = [c for c in ("d", "name", "mask", "sched", "S", "report", "kernel", "dur_us", "graph_us", "tflops_ncu", "tflops_graph", "clk_ghz", "sm_count", "sm_active_pct", *METRICS.keys(), "sass_exec_total") if c not in ("dur_us", "clk_ghz", "sm_count") or True]
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    with open(args.out + ".json", "w") as f:
        json.dump(recs, f, indent=1)
    show = ["d", "name", "mask", "S", "dur_us", "tflops_ncu", "clk_ghz", "sm_active_pct", "tensor_act_pct", "xu_mufu_pct", "fma_pct", "alu_pct", "tmem_inst_pct", "issue_pct", "warp_lat_cyc", "stall_long_sb", "l1_pct", "l2_pct", "dram_pct", "regs", "smem_kb"]
    print("| " + " | ".join(show) + " |")
    print("|" + "---|" * len(show))
    for r in recs:
        cells = []
        for c in show:
            v = r.get(c)
            cells.append("" if v is None else (f"{v:.0f}" if c in ("S", "regs", "smem_kb", "dur_us") and isinstance(v, float) and v >= 100 else f"{v:.2f}" if isinstance(v, float) else str(v)))
        print("| " + " | ".join(cells) + " |")
    print(f"\n{len(recs)} reports -> {args.out}.csv / .json")


if __name__ == "__main__":
    main()
