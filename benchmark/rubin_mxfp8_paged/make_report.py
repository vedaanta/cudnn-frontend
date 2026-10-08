#!/usr/bin/env python3
"""Merge bench timing (results.jsonl), ncu metrics (ncu_table.py JSON), the unit-SOL model (unit_sol.py JSON) and the
PerfSim summary (perfsim/summarize.py JSON, optional) into results/REPORT.md.

  make_report.py --timing results/results.jsonl [--ncu results/ncu_table.json] [--unit-sol results/unit_sol.json]
                 [--perfsim results/perfsim_RESULTS.json] --out results/REPORT.md
Several --timing files may be given (later files win on identical (d, name, mask, sched, S) keys).
"""

import argparse
import json
import os
from collections import OrderedDict

SEQS = [8192, 16384, 32768]


def load_timing(paths):
    d = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[(r["d"], r["name"], r["mask"], r.get("sched", "natural"), r["S"])] = r
    return d


def f(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "–"


def pct(a, b):
    try:
        return f"{100.0 * (float(a) / float(b) - 1.0):+.1f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "–"


def config_label(r):
    c = r.get("config", {})
    km = r.get("kmod", {})
    parts = [
        f"corr={c.get('corr')}",
        f"paged={c.get('paged')}",
        "f16-softmax" if km.get("SOFTMAX_F16", c.get("half")) else "f32-softmax",
        "prefolded" + ("+fused" if km.get("_FUSED_SHIFT_CVT") else "") if km.get("SCALE_PREFOLDED", c.get("prefolded")) else "scale in-kernel",
        f"{c.get('kernel')} kernel",
    ]
    if c.get("kernel") == "bench":
        parts.append(f"corrfast={c.get('corrfast')} hoist={c.get('hoist')}" + (f" rowsum_mma={c.get('rowsum_mma')}" if c.get("rowsum_mma") else ""))
    if r.get("cta_mma"):
        parts.append(f"cga{r['cta_mma']}")
    return ", ".join(parts)


def timing_table(T, d, mask, sched, U):
    names = OrderedDict()
    for (dd, name, m, s, S), r in T.items():
        if dd == d and m == mask and s == sched:
            names.setdefault(name, r)
    if not names:
        return ""
    seqs = sorted({k[4] for k in T if k[0] == d and k[2] == mask and k[3] == sched})
    out = [
        f"### d={d}, {mask} mask, scheduler {sched}",
        "",
        "| config | " + " | ".join(f"{S // 1024}k µs (TF/s) | MMA util | Δ first" for S in seqs) + " |",
        "|---|" + "---|---|---|" * len(seqs),
    ]
    first = next(iter(names))
    for name, r0 in names.items():
        cells = [f"`{name}`<br>{config_label(r0)}"]
        for S in seqs:
            t = T.get((d, name, mask, sched, S))
            b = T.get((d, first, mask, sched, S))
            if t is None:
                cells += ["–", "–", "–"]
                continue
            us = t.get("time_us_graph")
            u = U.get((d, name, mask, sched, S))
            cells.append(f"{f(us)} ({f(t.get('tflops'), 0)})" + (" *timing-only*" if t.get("timing_only") else ""))
            cells.append(f"{f(u['mma_util_pct'])} %" if u else "–")
            cells.append(pct(us, b.get("time_us_graph")) if b and name != first else "–")
        out.append("| " + " | ".join(cells) + " |")
    notes = []
    for name in names:
        for S in seqs:
            t = T.get((d, name, mask, sched, S))
            if t and t.get("validation"):
                v = t["validation"]
                notes.append(f"{name}@{S // 1024}k rel {100 * v['rel_err']:.2f}% rms {100 * v.get('rms_rel', 0):.2f}%{'' if v['finite'] else ' NON-FINITE'}")
            if t and t.get("rescale_stats"):
                rs = t["rescale_stats"]
                notes.append(f"{name}@{S // 1024}k rescales {rs['warp_steps_rescaled']}/{rs['warp_steps_live']} warp-steps ({rs['pct']:.2f}%)")
    clk = sorted({(t.get("clock_mhz_median"), t.get("power_w_median")) for t in names.values()})
    out.append("")
    out.append(
        f"Median SM clock / power during timing: {', '.join(f'{c} MHz / {p} W' for c, p in clk if c)}. "
        + ("Validation vs fp32 reference (max abs err / ref amax; rms rel): " + "; ".join(notes) + "." if notes else "")
    )
    return "\n".join(out) + "\n"


UNIT_ORDER = ["tensor", "mufu", "issue", "fma", "alu", "smem", "tmem", "l2", "dram"]


def unit_sol_table(U, d, mask, sched, S):
    rows = [u for (dd, name, m, s, SS), u in U.items() if dd == d and m == mask and s == sched and SS == S]
    if not rows:
        return ""
    out = [
        f"### d={d}, {mask}, S={S // 1024}k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)",
        "",
        "| config | step clk | " + " | ".join(UNIT_ORDER) + " | binding |",
        "|---|---|" + "---|" * len(UNIT_ORDER) + "---|",
    ]
    for u in rows:
        um = {r["unit"]: r for r in u["units"]}
        xc = u.get("ncu", {})
        cells = [f"`{u['name']}`", f"{u['step_clk']:.0f}"]
        for unit in UNIT_ORDER:
            r = um.get(unit)
            x = xc.get(unit)
            cells.append(("–" if not r else f"{r['util_pct']:.0f}% [{r['sol_clk']:.0f}]") + (f" ({x:.0f})" if x is not None else ""))
        cells.append(f"{u['binding_unit']} {u['binding_pct']:.0f}%")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


NCU_COLS = [
    ("dur_us", "ncu µs", 0),
    ("tflops_ncu", "TF/s", 0),
    ("clk_ghz", "clk GHz", 2),
    ("sm_active_pct", "SM active %", 1),
    ("tensor_act_pct", "tensor pipe %", 1),
    ("fp8_mma_pct", "UTCQMMA fp8 % (realtime)", 1),
    ("xu_mufu_pct", "MUFU (XU) %", 1),
    ("fma_pct", "FMA %", 1),
    ("alu_pct", "ALU %", 1),
    ("tmem_inst_pct", "TMEM instr %", 1),
    ("issue_pct", "issue %", 1),
    ("ipc_active", "IPC", 2),
    ("warp_lat_cyc", "warp lat cyc", 1),
    ("stall_long_sb", "stall long-sb", 2),
    ("stall_wait", "stall wait", 2),
    ("stall_short_sb", "stall short-sb", 2),
    ("stall_barrier", "stall barrier", 2),
    ("stall_math_throttle", "stall math-throttle", 2),
    ("stall_mio_throttle", "stall mio-throttle", 2),
    ("l1_pct", "L1/SMEM %", 1),
    ("l2_pct", "L2 %", 1),
    ("l2_hit_pct", "L2 hit %", 1),
    ("dram_pct", "DRAM %", 1),
    ("dram_gbs", "DRAM GB/s", 0),
    ("regs", "regs", 0),
    ("smem_kb", "SMEM KB", 0),
    ("waves", "waves", 2),
]


def ncu_table(N, d, mask, sched):
    rows = [n for n in N if n["d"] == d and n["mask"] == mask and n.get("sched", "natural") == sched]
    if not rows:
        return ""
    out = [
        f"### d={d}, {mask} mask, scheduler {sched}",
        "",
        "| S | config | " + " | ".join(c[1] for c in NCU_COLS) + " |",
        "|---|---|" + "---|" * len(NCU_COLS),
    ]
    for n in sorted(rows, key=lambda r: (r["S"], r["name"])):
        out.append(f"| {n['S'] // 1024}k | `{n['name']}` | " + " | ".join(f(n.get(c[0]), c[2]) for c in NCU_COLS) + " |")
    return "\n".join(out) + "\n"


def perfsim_section(P):
    if not P:
        return [
            "## PerfSim (GR100, B=1 H=1 S=4096 no mask, SSAF route)",
            "",
            "_pending: traces captured with perfsim/capture.sh, submitted with perfsim/submit.sh; run perfsim/summarize.py --out results/perfsim_RESULTS.md and pass the JSON to --perfsim._",
            "",
        ]
    L = [
        "## PerfSim (GR100, B=1 H=1 S=4096 no mask, SSAF route)",
        "",
        "| run | sim cycles | sim µs | top-3 SOL | mainloop MMA util | issue % | XU % | LST % | TMEM rd/wr % | RF rd ports % (mainloop) | RF wr ports % (mainloop) | rd-dispatch stall % | TMEM→RF wb stall % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in P:
        u, rf = r.get("util", {}), r.get("rf", {})
        L.append(
            f"| {r['label']} | {r['dur_cyc']} | {r['dur_us']:.2f} | {', '.join(f'{n} {p:.1f}%' for n, p in r['sol'])} | {f(r.get('mma_sol_pct'))}% | {f(u.get('issue %'))} | {f(u.get('XU (MUFU) %'))} | {f(u.get('SMEM wavefronts (LST) %'))} | {f(u.get('TMEM rd %'))}/{f(u.get('TMEM wr %'))} | {f(rf.get('rf_read_port_util_mainloop_pct'))} | {f(rf.get('rf_write_port_util_mainloop_pct'))} | {f(rf.get('rf_read_dispatch_stall_pct'), 2)} | {f(rf.get('tmem_rf_wb_stall_pct'), 2)} |"
        )
    L += ["", "Full per-unit SOL and register-file tables: `results/perfsim_RESULTS.md` (perfsim/summarize.py).", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", nargs="+", required=True)
    ap.add_argument("--ncu", nargs="*", default=[], help="ncu_table.py JSON file(s), e.g. one per head dim")
    ap.add_argument("--unit-sol", nargs="*", default=[], help="unit_sol.py JSON file(s), e.g. one per head dim")
    ap.add_argument("--perfsim", nargs="*", default=[], help="perfsim/summarize.py JSON file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Rubin GR100 (cc 10.7) MXFP8 Q/K/V GQA prefill: dH=128 / dH=256, paged KV page 64, correction ON / OFF")
    args = ap.parse_args()

    def load_lists(paths):
        out = []
        for p in paths:
            if p and os.path.exists(p):
                out += json.load(open(p))
        return out

    T = load_timing(args.timing)
    N = load_lists(args.ncu)
    U = {}
    for u in load_lists(args.unit_sol):
        U[(u["d"], u["name"], u["mask"], u.get("sched", "natural"), u["S"])] = u
    P = load_lists(args.perfsim)
    hosts = sorted({r.get("host", "?") for r in T.values()})
    heads = sorted({(r.get("B", 1), r.get("h_q"), r.get("h_kv")) for r in T.values()})
    md = [
        f"# {args.title}",
        "",
        f"Timing host(s): {', '.join(hosts)} (cc 10.7, {next((r.get('sm_count') for r in T.values() if r.get('sm_count')), '?')} SMs). Shapes B/H_q/H_kv = {', '.join(f'{b}/{hq}/{hkv}' for b, hq, hkv in heads)}; Q/K/V MXFP8 e4m3 + E8M0/32, O bf16, no Stats / Amax_O. "
        "Time = median over interleaved rounds of 4-replay CUDA-graph bursts with 150 ms gaps (board cooldown protocol); TF/s = 4·B·H_q·S²·d (×(S+1)/2S causal) / time. "
        "MMA util = tensor SOL clocks per 128x128 step (BMM1 + BMM2 [+ ones-MMA row-sum], 16384 MAC/clk/SM) / measured clocks per step (unit_sol.py). "
        "corr=never rows are *timing-only* unless the rescale emulation found zero rescales. ",
        "",
    ]
    md.append("## Timing\n")
    ds = sorted({k[0] for k in T})
    for d in ds:
        for sched in ("natural", "lpt", "lpt_l2"):
            for mask in ("none", "causal"):
                md.append(timing_table(T, d, mask, sched, U))
    if U:
        md.append("## Per-unit utilisation vs SOL (analytic model + ncu cross-check)\n")
        md.append(
            "Units: tensor (UTCQMMA MACs), mufu (exp2 lanes), issue (warp-instr/4), fma / alu (pipe warp-instr / 2), smem (TMA writes + UMMA operand reads at 256 B/clk, the operand feed of a 16384 MAC/clk tensor core; the LSU path alone is 128), tmem (tcgen05.ld/st at an assumed 512 B/clk), l2 (K/V(+SF) bytes per step per SM at the ncu-derived L2 peak), dram (whole-kernel bytes). "
            "Cell = util% [SOL clk/step] (ncu pipe %). Constants and sources: `unit_sol.py` header.\n"
        )
        for d in ds:
            for sched in ("natural", "lpt", "lpt_l2"):
                for mask in ("none", "causal"):
                    for S in sorted({k[4] for k in U if k[0] == d and k[2] == mask and k[3] == sched}):
                        md.append(unit_sol_table(U, d, mask, sched, S))
    if N:
        md.append("## ncu (one launch, `--set full`, 4th launch after 3 warm-ups; pipe % of peak sustained over SM-active cycles)\n")
        for d in ds:
            for sched in ("natural", "lpt", "lpt_l2"):
                for mask in ("none", "causal"):
                    md.append(ncu_table(N, d, mask, sched))
    md += perfsim_section(P)
    md += [
        "## Files",
        "",
        "- `results/*.jsonl` -- bench.py records (one per config x S x mask: all burst samples, clocks/power per round, kernel CFG/kmod flags, validation, rescale emulation)",
        "- `results/ncu_table.{csv,json}` -- ncu_table.py over run_ncu.sh reports (+ `.raw.csv` / `.sass.csv` exports per report)",
        "- `results/unit_sol.{json,md}` -- unit_sol.py",
        "- `results/perfsim_RESULTS.{md,json}` -- perfsim/summarize.py",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
