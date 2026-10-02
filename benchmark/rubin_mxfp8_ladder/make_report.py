#!/usr/bin/env python3
"""Merge ladder timing (results.jsonl) + ncu metrics (ncu_table.py CSV) into results/REPORT.md.

Usage: make_report.py --timing results_hecate.jsonl --ncu ncu_hecate.csv --out REPORT.md
       [--extra-timing label=path ...] (e.g. board cross-check)
"""
import argparse
import csv
import json
from collections import OrderedDict

RUNG_LABEL = OrderedDict(
    A="fp8 per-tensor, fp32 softmax",
    B="mxfp8, fp32 softmax",
    C="B + f16 exp (MUFU.EX2.F16x2)",
    D="C + 1/ln2 folded outside",
    E="D + paged KV, page 64",
)
SEQS = [8192, 16384, 32768]


def load_timing(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d[(r["rung"], r["mask"], r.get("sched", "natural"), r["S"])] = r  # last write wins
    return d


def load_ncu(path):
    d = {}
    if not path:
        return d
    for r in csv.DictReader(open(path)):
        d[(r["rung"], r["mask"], r.get("sched") or "natural", int(r["S"]))] = r
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


def timing_table(T, mask, sched, title):
    rungs = [r for r in RUNG_LABEL if any((r, mask, sched, s) in T for s in SEQS)]
    if not rungs:
        return ""
    out = [f"### {title}", "", "| rung | config | " + " | ".join(f"{s//1024}k µs (TFLOPS) | Δ prev | Δ A" for s in SEQS) + " |", "|---|---|" + "---|---|---|" * len(SEQS)]
    prev = None
    for r in rungs:
        cells = [r, RUNG_LABEL[r]]
        for s in SEQS:
            t = T.get((r, mask, sched, s))
            a = T.get(("A", mask, sched, s))
            p = T.get((prev, mask, sched, s)) if prev else None
            if t is None:
                cells += ["–", "–", "–"]
                continue
            us = t.get("time_us_graph") or t.get("time_us_eager")
            cells.append(f"{f(us)} ({f(t.get('tflops'), 0)})")
            cells.append(pct(us, (p.get("time_us_graph") or p.get("time_us_eager")) if p else None) if p else "–")
            cells.append(pct(us, (a.get("time_us_graph") or a.get("time_us_eager")) if a else None) if a and r != "A" else "–")
        out.append("| " + " | ".join(cells) + " |")
        prev = r
    # validation line
    vals = []
    for r in rungs:
        for s in SEQS:
            t = T.get((r, mask, sched, s))
            if t and t.get("validation"):
                v = t["validation"]
                vals.append(f"{r}@{s//1024}k rel {100*v['rel_err']:.2f}%{'' if v['finite'] else ' NON-FINITE'}")
    if vals:
        out += ["", "Validation vs fp32 reference (max abs err / ref amax): " + "; ".join(vals) + "."]
    return "\n".join(out) + "\n"


NCU_COLS = [
    ("dur_us", "ncu µs", 0),
    ("tflops_ncu", "TFLOPS", 0),
    ("clk_ghz", "clk GHz", 2),
    ("sm_active_pct", "SM active %", 1),
    ("tensor_act_pct", "tensor pipe % (active)", 1),
    ("xu_mufu_pct", "MUFU (XU) %", 1),
    ("fma_pct", "FMA %", 1),
    ("alu_pct", "ALU %", 1),
    ("issue_pct", "issue %", 1),
    ("ipc_active", "IPC", 2),
    ("warp_lat_cyc", "warp lat cyc", 1),
    ("stall_long_sb", "stall long-sb", 2),
    ("stall_wait", "stall wait", 2),
    ("dram_gbs", "DRAM GB/s", 0),
    ("l2_hit_pct", "L2 hit %", 1),
    ("tma_ld_mb", "TMA ld MB", 0),
    ("smem_kb", "SMEM KB", 0),
    ("regs", "regs", 0),
    ("waves", "waves", 2),
]


def ncu_table(N, mask, sched, title):
    keys = [k for k in N if k[1] == mask and k[2] == sched]
    if not keys:
        return ""
    out = [f"### {title}", "", "| S | rung | " + " | ".join(c[1] for c in NCU_COLS) + " |", "|---|---|" + "---|" * len(NCU_COLS)]
    for s in SEQS:
        for r in RUNG_LABEL:
            n = N.get((r, mask, sched, s))
            if not n:
                continue
            out.append(f"| {s//1024}k | {r} | " + " | ".join(f(n.get(c[0]), c[2]) for c in NCU_COLS) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", required=True)
    ap.add_argument("--ncu", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-timing", nargs="*", default=[], help="label=path for cross-check tables")
    ap.add_argument("--title", default="Rubin GR100 MXFP8 prefill ladder")
    args = ap.parse_args()
    T = load_timing(args.timing)
    N = load_ncu(args.ncu)
    hosts = sorted({r.get("host", "?") for r in T.values()})
    md = [f"# {args.title}", "", f"Timing host(s): {', '.join(hosts)}.  B=1, H_q=32, H_kv=2, d=128, bf16 O, no Stats / Amax_O. "
          "Time = median CUDA-graph replay of one launch; TFLOPS = 4·S²·d·H_q (×½ causal) / time. "
          "Rungs C–E run the bench one-off kernel `sm107/prefill_d128_mxfp8_ladder.py`; E pages only the K/V payload (page 64, HND, random table), SF stay dense.", ""]
    md.append("## Timing\n")
    for sched in ("natural", "lpt", "lpt_l2"):
        for mask in ("causal", "none"):
            md.append(timing_table(T, mask, sched, f"{mask} mask, scheduler {sched}"))
    if N:
        md.append("## ncu (one launch, `--set full`, 4th launch after 3 warmups; pipe % are of peak sustained over SM-active cycles)\n")
        for sched in ("natural", "lpt", "lpt_l2"):
            for mask in ("causal", "none"):
                md.append(ncu_table(N, mask, sched, f"{mask} mask, scheduler {sched}"))
    for item in args.extra_timing:
        label, path = item.split("=", 1)
        X = load_timing(path)
        md.append(f"## Cross-check timing: {label}\n")
        for sched in ("natural", "lpt", "lpt_l2"):
            for mask in ("causal", "none"):
                md.append(timing_table(X, mask, sched, f"{label}: {mask} mask, scheduler {sched}"))
    open(args.out, "w").write("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
