#!/usr/bin/env python3
"""Summarize GR100 PerfSim runs (flow.perfsim -chip gr100 -enableMorph -pic) into a markdown + JSON table:
kernel cycles, top-3 SOL units, mainloop MMA util, the per-unit SOL table (TPC/MMA, SM issue, XU, LST, TMEM rd/wr, L2,
FB) and the REGISTER-FILE bandwidth utilisation from the raw SMART counters (read ports, write ports, uniform RF,
register-read dispatch stalls, TMEM->RF write-back stalls).

  summarize.py --root <perfsim_output> --pattern 'gr100_mxb_*_r1' --out RESULTS.md
  summarize.py --root /home/scratch.vagarwalla_gpu/perfsim_ladder/perfsim_output --pattern 'gr100_ladder_[CF]_b1h1s4k_none_r1'
Optional --silicon <bench results.jsonl> adds the board's graph-replay us of the same (d, config) at B1 H1 S4096.

Per run dir <root>/<case>_<suf>/perfsim/:
  pic_analysis/run.A.dir.0/*/pic-analysis/pi/grid_info.csv        kernel start/end/duration cycles, CTAs / SMs
  .../pi/sumry_file.csv                                             nvclk
  .../pi/pi_nvpdm/pic_report_summary.xml                            top-3 SOL units + bottlenecks
  .../pi/web/nvpdm_math_sol_table_0.html                            per-SM MMA SOL (instrumented cluster)
  .../pi/pi_nvpdm/pic_report/{SM,TPC,LST,LTC,FB,GPCMMU}.json.gz      per-unit SOL sub-metrics (busiest instance)
  perfsim/run.A.dir.0/*/perfsim/*/PERFSIM/batch_*/results/000001/perfsim_apdf_0.xml.gz   raw SMART counters

Register-file normalisation (per SM, instrumented unit SM0_0_0; clocks = the SM's elapsedClocks, or the mainloop
window first-MMA-issue..last-MMA-retire for the "mainloop" columns):
  * READ ports: `register_reads_bank{0,1}_q` (hardware-counted LRF operand reads; `simulator_only_register_reads_bank*`
    is the simulator's own count, within a few % -- both are reported).  Each SMSP has 2 LRF banks delivering one
    32-lane register per clock each -> peak = 4 SMSP x 2 banks = 8 reads/clk/SM; util = reads / (8 x clocks).
  * WRITE ports: `register_writes_bank{b}_hw{h}_q` -- h = the two half-warp write ports of a bank; a 32-lane write is
    counted once on EACH half (hw0 == hw1 in every run), so writes per bank = register_writes_bank{b}_hw0_q and the peak
    is one 32-lane write per bank per clock: util = (bank0 + bank1 writes) / (8 x clocks).  Equivalently
    sum over b,h / (16 half-warp ports x clocks) -- the same number.  `register_writes_coupled_q` (fixed-latency math
    pipes) + `register_writes_decoupled_q` (variable-latency MIO / MUFU / TMEM / LSU write-backs) = the total; the
    coupled share tells how much of the write traffic is the FFMA2 / F2FP / ALU stream.
  * Operand-collector reuse: `register_reads_avoided_collector_reuse_f_pipe_q` reads served by the reuse cache (not
    on the ports); `register_reads_coupled_f{3,4,5}` = reads by 3/4/5-operand coupled instructions.
  * UNIFORM RF: counters named uniform_*register* / urf_* when the SMART build emits them (else n/a; the F run only has
    `uniform_pipe_active_q*` and `inst_issued_uniform_pipe_q`, reported as uniform-pipe active % and UR instr/clk).
  * Dispatch stalls: `cant_dispatch_register_read_q` (bank_0/bank_1/f_pipe/m_pipe breakdown) and
    `cant_dispatch_uniform_register_read_q` as % of SMSP-clocks (sum / 4 / clocks).
  * TMEM -> RF: `tmem_rf_wb_stall_q{0..3}` (+ `_for_mufu_bank*`) summed, as % of SMSP-clocks; TMEM traffic
    `tmem_reads_q` / `tmem_writes_q` split by op (ldtm, utcmma_c accumulator reads, sttm, utcmma writes, utccp SF).
"""

import argparse
import csv
import glob
import gzip
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

PI_URL = "https://perf-inspector/server/?infoWin=On&panels=%5B%5D&path={web}/full"
# per-unit SOL columns: (label, unit json, SOL metric name)
UTIL = [
    ("issue %", "SM", "issueUtilization"),
    ("MMA pipe % (whole kernel)", "SM", "mmaPipeActivePerSM"),
    ("XU (MUFU) %", "SM", "xuPipeActivePerSM"),
    ("FMA-heavy %", "SM", "fmaheavy_pipe_IPCperSM"),
    ("FMA-lite %", "SM", "fmalite_pipe_IPCperSM"),
    ("ALU %", "SM", "alu_pipe_IPCperSM"),
    ("ADU %", "SM", "adu_pipe_IPCperSM"),
    ("uniform pipe %", "SM", "uniform_pipe_IPCperSM"),
    ("LSU %", "SM", "lsu_pipe_IPCperSM"),
    ("TMEM rd %", "SM", "tmemReadsPerClk"),
    ("TMEM wr %", "SM", "tmemWritesPerClk"),
    ("SMEM wavefronts (LST) %", "LST", "wavefrontsPerClk"),
    ("LST TC-MMA wavefronts %", "LST", "l1dataTcmmaWavefrontsPerClk"),
    ("ICC instr fetch (TPC) %", "TPC", "ICCRequestsPerTPC"),
    ("L2 data-bank accesses %", "LTC", "dataBankAccessesPerCycle"),
    ("L2 xbar read bytes %", "LTC", "xbarReadBytesPerCycle"),
    ("L2 tag lookups %", "LTC", "tagLookupsLinesPerCycle"),
    ("DRAM (FB) bytes %", "FB", "bytesPerNVClock"),
    ("GPCMMU ltp requests %", "GPCMMU", "ltpRequestPerClock"),
]
RF_RX = re.compile(
    r"^(simulator_only_register_reads_bank[01]_q|register_reads_bank[01]_q|register_writes_bank[01]_hw[01]_q|register_writes_coupled_q|register_writes_decoupled_q|"
    r"register_writes_(math|mio|mufu)_bank[01]_q|register_reads_(mio|mufu|agu|other_mio)_bank[01]_q|register_reads_coupled_f[345]_q|register_reads_decoupled_q|"
    r"register_reads_avoided_collector_reuse_f_pipe_q|register_reads_avoided_duplicate_f_pipe_q|cant_dispatch_register_read_q|cant_dispatch_register_read_(bank_[01]|f_pipe|m_pipe)_q|"
    r"cant_dispatch_uniform_register_(read|write)_q|tmem_rf_wb_stall_q[0-3]|tmem_rf_wb_stall_for_mufu_bank[01]_q|tmem_rf_rd_stall_q[0-3]|"
    r"tmem_reads_q|tmem_reads_op_[a-z_]+_q|tmem_writes_q|tmem_writes_op_[a-z0-9_]+_q|uniform_pipe_active_q[0-3]|inst_issued_q|inst_issued_[a-z0-9]+_pipe_q|"
    r"rf_arb_retries_per_grant_|active_rf_spass_q|clocksActive|elapsedClocks|[a-z_]*uniform[a-z_]*reg[a-z_]*|urf_[a-z_]+)$"
)


def parse_tables(path):
    t = open(path, errors="ignore").read()
    out = []
    for tb in re.findall(r"<table[^>]*>(.*?)</table>", t, re.S):
        rows = []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S):
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
            if cells:
                rows.append(cells)
        out.append(rows)
    return out


def unit_busiest(pi, unit, metric):
    p = f"{pi}/pi_nvpdm/pic_report/{unit}.json.gz"
    if not os.path.exists(p):
        return None
    j = json.load(gzip.open(p))
    for e in j.get("SOL", []):
        if e["name"] == metric:
            vals = [float(i["percent"]) for i in e.get("instancevalues", []) if i.get("percent") not in (None, "")]
            return max(vals, default=None)
    return None


def apdf_stats(apdf, unit="SM0_0_0"):
    """All register-file / TMEM / issue counters of one SM unit from the raw SMART stats."""
    st, stack = {}, []
    for ev, el in ET.iterparse(gzip.open(apdf), events=("start", "end")):
        tag = el.tag.split("}")[-1]
        if ev == "start" and tag == "unit":
            stack.append(el.get("name"))
        elif ev == "end":
            if tag == "unit":
                stack.pop()
                el.clear()
            elif tag == "stat" and stack and stack[-1] == unit:
                n = el.get("name") or ""
                if RF_RX.match(n):
                    try:
                        st[n] = float(el.text or 0)
                    except ValueError:
                        pass
    return st


def rf_metrics(st, elapsed, mainloop=None):
    g = lambda *names: sum(st.get(n, 0.0) for n in names)  # noqa: E731
    reads_hw = g("register_reads_bank0_q", "register_reads_bank1_q")
    reads_sim = g("simulator_only_register_reads_bank0_q", "simulator_only_register_reads_bank1_q")
    reads = reads_hw or reads_sim
    writes = g("register_writes_bank0_hw0_q", "register_writes_bank1_hw0_q")  # hw0 == hw1: every 32-lane write hits both half-warp ports
    writes_hw_all = g(*[f"register_writes_bank{b}_hw{h}_q" for b in (0, 1) for h in (0, 1)])
    out = dict(
        rf_reads=reads,
        rf_reads_hw=reads_hw,
        rf_reads_sim=reads_sim,
        rf_reads_bank_split=(st.get("register_reads_bank0_q", 0.0), st.get("register_reads_bank1_q", 0.0)),
        rf_reads_avoided_reuse=st.get("register_reads_avoided_collector_reuse_f_pipe_q", 0.0),
        rf_reads_coupled_f345=(
            st.get("register_reads_coupled_f3_q", 0.0),
            st.get("register_reads_coupled_f4_q", 0.0),
            st.get("register_reads_coupled_f5_q", 0.0),
        ),
        rf_reads_decoupled=st.get("register_reads_decoupled_q", 0.0),
        rf_reads_by_consumer=dict(
            mio=g("register_reads_mio_bank0_q", "register_reads_mio_bank1_q"),
            mufu=g("register_reads_mufu_bank0_q", "register_reads_mufu_bank1_q"),
            agu=g("register_reads_agu_bank0_q", "register_reads_agu_bank1_q"),
            other_mio=g("register_reads_other_mio_bank0_q", "register_reads_other_mio_bank1_q"),
        ),
        rf_writes=writes,
        rf_writes_halfwarp_ports_total=writes_hw_all,
        rf_writes_bank_split=(st.get("register_writes_bank0_hw0_q", 0.0), st.get("register_writes_bank1_hw0_q", 0.0)),
        rf_writes_coupled=st.get("register_writes_coupled_q", 0.0),
        rf_writes_decoupled=st.get("register_writes_decoupled_q", 0.0),
        rf_writes_by_source=dict(
            math=g("register_writes_math_bank0_q", "register_writes_math_bank1_q"),
            mio=g("register_writes_mio_bank0_q", "register_writes_mio_bank1_q"),
            mufu=g("register_writes_mufu_bank0_q", "register_writes_mufu_bank1_q"),
        ),
        rf_read_port_util_elapsed_pct=100.0 * reads / 8.0 / elapsed,
        rf_write_port_util_elapsed_pct=100.0 * writes / 8.0 / elapsed,
        rf_reads_per_clk_sm=reads / elapsed,
        rf_writes_per_clk_sm=writes / elapsed,
        rf_read_dispatch_stall_pct=100.0 * st.get("cant_dispatch_register_read_q", 0.0) / 4.0 / elapsed,
        rf_read_dispatch_stall_split=dict(
            bank0=st.get("cant_dispatch_register_read_bank_0_q", 0.0),
            bank1=st.get("cant_dispatch_register_read_bank_1_q", 0.0),
            f_pipe=st.get("cant_dispatch_register_read_f_pipe_q", 0.0),
            m_pipe=st.get("cant_dispatch_register_read_m_pipe_q", 0.0),
        ),
        urf_read_dispatch_stall_pct=100.0 * st.get("cant_dispatch_uniform_register_read_q", 0.0) / 4.0 / elapsed,
        rf_arb_retries=st.get("rf_arb_retries_per_grant_", 0.0),
        tmem_rf_wb_stall_pct=100.0 * g(*[f"tmem_rf_wb_stall_q{i}" for i in range(4)]) / 4.0 / elapsed,
        tmem_rf_wb_stall_for_mufu=g("tmem_rf_wb_stall_for_mufu_bank0_q", "tmem_rf_wb_stall_for_mufu_bank1_q"),
        tmem_reads=st.get("tmem_reads_q", 0.0),
        tmem_writes=st.get("tmem_writes_q", 0.0),
        tmem_reads_by_op={k[len("tmem_reads_op_") : -2]: v for k, v in st.items() if k.startswith("tmem_reads_op_")},
        tmem_writes_by_op={k[len("tmem_writes_op_") : -2]: v for k, v in st.items() if k.startswith("tmem_writes_op_")},
        uniform_pipe_active_pct=100.0 * g(*[f"uniform_pipe_active_q{i}" for i in range(4)]) / 4.0 / elapsed,
        uniform_inst_per_clk=st.get("inst_issued_uniform_pipe_q", 0.0) / elapsed,
        inst_issued=st.get("inst_issued_q", 0.0),
        inst_issued_per_clk=st.get("inst_issued_q", 0.0) / elapsed,
        inst_issued_by_pipe={
            k[len("inst_issued_") : -len("_pipe_q")]: v for k, v in st.items() if k.startswith("inst_issued_") and k.endswith("_pipe_q") and v
        },
        apdf_elapsed=elapsed,
    )
    urf = {k: v for k, v in st.items() if ("uniform" in k and "reg" in k and "cant_dispatch" not in k) or k.startswith("urf_")}
    out["uniform_rf_counters"] = urf  # `uniform_register_reads_q` / `uniform_register_writes_q` (per-SM sums over the 4 SMSPs)
    if urf:
        ur = sum(v for k, v in urf.items() if "read" in k)
        uw = sum(v for k, v in urf.items() if "write" in k)
        out["urf_reads_per_clk"] = ur / elapsed
        out["urf_writes_per_clk"] = uw / elapsed
        # one uniform-register read and one write per SMSP per clock (the uniform datapath is scalar per SMSP)
        out["urf_read_port_util_pct"] = 100.0 * ur / 4.0 / elapsed
        out["urf_write_port_util_pct"] = 100.0 * uw / 4.0 / elapsed
        if mainloop:
            out["urf_read_port_util_mainloop_pct"] = 100.0 * ur / 4.0 / mainloop
    if mainloop:
        out["rf_read_port_util_mainloop_pct"] = 100.0 * reads / 8.0 / mainloop
        out["rf_write_port_util_mainloop_pct"] = 100.0 * writes / 8.0 / mainloop
    return out


def summarize_run(run_dir, label, with_rf=True):
    hits = glob.glob(f"{run_dir}/perfsim/pic_analysis/run.A.dir.0/*/pic-analysis/pi")
    # the pi directory appears before PIC analysis has written its tables: treat a run as finished only once the
    # grid info, the summary csv and the full.pfm webview are all there
    hits = [h for h in hits if all(os.path.exists(f"{h}/{f}") for f in ("grid_info.csv", "sumry_file.csv", "web/full.pfm"))]
    if not hits:
        return None
    pi = hits[0]
    g = list(csv.DictReader(open(f"{pi}/grid_info.csv")))[0]
    s = list(csv.DictReader(open(f"{pi}/sumry_file.csv")))[0]
    x = open(f"{pi}/pi_nvpdm/pic_report_summary.xml", errors="ignore").read()
    a = dict(re.findall(r"(\S+?)='([^']*)'", re.search(r"<instance\s+([^>]*)>", x).group(1)))
    tbs = parse_tables(f"{pi}/web/nvpdm_math_sol_table_0.html")
    gpu_sol = {r[0]: float(r[1]) for r in tbs[0][1:]} if tbs else {}
    sm_rows = [r for r in tbs[1][1:] if len(r) >= 6 and r[1] not in ("0", "0.0")] if len(tbs) > 1 else []
    sm0 = sm_rows[0] if sm_rows else None
    clk = float(s["nvclk"])
    rec = dict(
        run=os.path.basename(run_dir.rstrip("/")),
        label=label,
        web=f"{pi}/web",
        pi_url=PI_URL.format(web=f"{pi}/web"),
        dur_cyc=int(g["duration_cycle"]),
        dur_us=int(g["duration_cycle"]) / clk,
        clk_mhz=clk,
        ctas=int(g["cta_num"]),
        sms=int(g["sm_num"]),
        start=int(g["start_cycle"]),
        end=int(g["end_cycle"]),
        sol=[(a.get(f"sol_{i}_id"), float(a.get(f"sol_{i}_pct", 0) or 0)) for i in range(3)],
        bneck=[(a.get(f"btl_{i}_id"), float(a.get(f"btl_{i}_pct", 0) or 0)) for i in range(3) if a.get(f"btl_{i}_id")],
        gpu_sol_full=gpu_sol.get("GPU SOL Ratio (%) - Full Chip"),
        gpu_sol_mma_sms=gpu_sol.get("GPU SOL Ratio (%) - Excl Non-MMA SMs"),
        util={lab: unit_busiest(pi, unit, m) for lab, unit, m in UTIL},
    )
    if sm0:
        rec.update(mma_sol_pct=float(sm0[1]), mma_sol_cyc=int(sm0[2]), mma_actual_cyc=int(sm0[3]), first_mma=int(sm0[4]), last_mma=int(sm0[5]))
        rec["pre_mma_cyc"] = rec["first_mma"] - rec["start"]
        rec["post_mma_cyc"] = rec["end"] - rec["last_mma"]
    if with_rf:
        apdfs = glob.glob(f"{run_dir}/perfsim/perfsim/**/results/000001/perfsim_apdf_0.xml.gz", recursive=True)
        if apdfs:
            st = apdf_stats(apdfs[0])
            elapsed = st.get("elapsedClocks") or float(rec["dur_cyc"])
            rec["rf"] = rf_metrics(st, elapsed, rec.get("mma_actual_cyc"))
    return rec


def f(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "–"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/home/scratch.vagarwalla_gpu/perfsim_mxb/perfsim_output")
    ap.add_argument("--pattern", default="gr100_mxb_*", help="glob of run dirs under --root")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit label=run_dir pairs (overrides --root/--pattern)")
    ap.add_argument("--silicon", default=None, help="bench.py results.jsonl with B1 H1 S4096 rows (label = config name)")
    ap.add_argument("--no-rf", action="store_true", help="skip the (slow, ~1 min per run) APDF register-file parse")
    ap.add_argument("--out", default=None, help="RESULTS.md path (also writes .json); default: print only")
    ap.add_argument("--title", default="Rubin GR100 PerfSim: MXFP8 prefill (B=1, H_q=H_kv=1, S=4096, no mask)")
    args = ap.parse_args()
    runs = []
    if args.runs:
        for item in args.runs:
            label, _, path = item.partition("=")
            runs.append((label, path or label))
    else:
        for d in sorted(glob.glob(os.path.join(args.root, args.pattern))):
            if os.path.isdir(d):
                runs.append((os.path.basename(d), d))
    recs = []
    for label, d in runs:
        r = summarize_run(d, label, with_rf=not args.no_rf)
        if r:
            recs.append(r)
        else:
            print(f"(no PIC yet: {d})", file=sys.stderr)
    if not recs:
        raise SystemExit("no finished runs")
    sil = {}
    if args.silicon and os.path.exists(args.silicon):
        for line in open(args.silicon):
            if line.strip():
                r = json.loads(line)
                if r.get("B", 1) == 1 and r.get("h_q") == 1 and r.get("S") == 4096:
                    sil[f"d{r['d']}_{r['name']}"] = r
    L = [f"# {args.title}", ""]
    L.append(
        f"Simulated at nvclk {recs[0]['clk_mhz']:.0f} MHz, {recs[0]['ctas']} CTAs on {recs[0]['sms']} SMs. Durations are the kernel's grid start-to-end in SM clocks; "
        "MMA numbers are the instrumented cluster's (SM0_0_0/1). Trace = 4th launch (3 warm-ups) captured with cuda_apic on w2u1g-lc-0614; ACE via SSAF, SMART + PIC via flow.perfsim latest."
    )
    L += ["", "## Timing and top-level SOL", ""]
    base = recs[0]
    hdr = "| run | sim cycles | sim µs | Δ first | SOL top-3 | mainloop MMA util | MMA cycles (SOL→actual) | pre-MMA | post-MMA | bottlenecks | GPU SOL full chip | silicon µs |"
    L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in recs:
        key = next((k for k in sil if k in r["run"]), None)
        s_us = f"{sil[key]['time_us_graph']:.1f}" if key else "–"
        L.append(
            f"| {r['label']} | {r['dur_cyc']} | {r['dur_us']:.2f} | {100 * (r['dur_cyc'] / base['dur_cyc'] - 1):+.1f}% | {', '.join(f'{n} {p:.1f}%' for n, p in r['sol'])} | {f(r.get('mma_sol_pct'))}% | "
            f"{r.get('mma_sol_cyc', '–')}→{r.get('mma_actual_cyc', '–')} | {r.get('pre_mma_cyc', '–')} | {r.get('post_mma_cyc', '–')} | {', '.join(f'{n} {p:.1f}%' for n, p in r['bneck'])} | {f(r.get('gpu_sol_full'), 2)} | {s_us} |"
        )
    L += ["", "## Per-unit SOL table (busiest / instrumented instance; % of each unit's peak over its elapsed clocks)", ""]
    cols = [lab for lab, _, _ in UTIL]
    L += ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in recs:
        L.append(f"| {r['label']} | " + " | ".join(f(r["util"].get(lab)) for lab in cols) + " |")
    if any("rf" in r for r in recs):
        L += ["", "## Register-file bandwidth (instrumented SM, raw SMART counters)", ""]
        cols = [
            "RF rd ports % (elapsed)",
            "RF rd ports % (mainloop)",
            "RF wr ports % (elapsed)",
            "RF wr ports % (mainloop)",
            "LRF reads/clk",
            "LRF writes/clk",
            "writes coupled / decoupled",
            "reads avoided by reuse cache",
            "rd dispatch stall % (bank0/bank1/f-pipe)",
            "URF rd port % (elapsed / mainloop)",
            "URF wr port %",
            "URF reads / writes",
            "URF rd dispatch stall %",
            "uniform pipe active %",
            "UR instr/clk",
            "TMEM→RF wb stall %",
            "TMEM reads (ldtm / utcmma_c / sf)",
            "TMEM writes (sttm / utcmma / utccp)",
            "issued instr/clk/SM",
            "issued by pipe",
        ]
        L += ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for r in recs:
            rf = r.get("rf")
            if not rf:
                L.append(f"| {r['label']} | " + " | ".join("–" for _ in cols) + " |")
                continue
            tr, tw = rf["tmem_reads_by_op"], rf["tmem_writes_by_op"]
            ds = rf["rf_read_dispatch_stall_split"]
            uc = rf.get("uniform_rf_counters") or {}
            if uc:
                urf_rd = f"{f(rf.get('urf_read_port_util_pct'))} / {f(rf.get('urf_read_port_util_mainloop_pct'))}"
                urf_wr = f(rf.get("urf_write_port_util_pct"))
                urf_n = f"{sum(v for k, v in uc.items() if 'read' in k):.0f} / {sum(v for k, v in uc.items() if 'write' in k):.0f}"
            else:
                urf_rd = urf_wr = urf_n = "n/a (no URF counters)"
            L.append(
                f"| {r['label']} | {f(rf['rf_read_port_util_elapsed_pct'])} | {f(rf.get('rf_read_port_util_mainloop_pct'))} | {f(rf['rf_write_port_util_elapsed_pct'])} | {f(rf.get('rf_write_port_util_mainloop_pct'))} | "
                f"{f(rf['rf_reads_per_clk_sm'], 2)} | {f(rf['rf_writes_per_clk_sm'], 2)} | {rf['rf_writes_coupled']:.0f} / {rf['rf_writes_decoupled']:.0f} | {rf['rf_reads_avoided_reuse']:.0f} | "
                f"{f(rf['rf_read_dispatch_stall_pct'], 2)} ({ds['bank0']:.0f}/{ds['bank1']:.0f}/{ds['f_pipe']:.0f}) | {urf_rd} | {urf_wr} | {urf_n} | {f(rf['urf_read_dispatch_stall_pct'], 2)} | {f(rf['uniform_pipe_active_pct'])} | {f(rf['uniform_inst_per_clk'], 3)} | "
                f"{f(rf['tmem_rf_wb_stall_pct'], 2)} | {rf['tmem_reads']:.0f} ({tr.get('ldtm', 0):.0f} / {tr.get('utcmma_c', 0):.0f} / {tr.get('utcmma_a_sp_sf', 0):.0f}) | {rf['tmem_writes']:.0f} ({tw.get('sttm', 0):.0f} / {tw.get('utcmma', 0):.0f} / {tw.get('utccp', 0):.0f}) | "
                f"{f(rf['inst_issued_per_clk'], 2)} | {', '.join(f'{k} {v:.0f}' for k, v in sorted(rf['inst_issued_by_pipe'].items(), key=lambda kv: -kv[1])[:6])} |"
            )
        L += [
            "",
            "Normalisation: read ports = LRF operand reads (`register_reads_bank{0,1}_q`, hardware count; the simulator-only count differs by a few %) / (4 SMSP x 2 banks x 1 read/clk) / clocks; "
            "write ports = `register_writes_bank{b}_hw0_q` summed over banks / (8/clk) / clocks -- hw0 and hw1 are the two half-warp ports and a 32-lane write is counted once on each (hw0 == hw1 in every run), so this equals the sum over all four hw counters / 16; "
            "mainloop = first-MMA-issue..last-MMA-retire window of the math SOL table (same numerator). Coupled writes come from fixed-latency math pipes, decoupled from MIO/MUFU/TMEM/LSU write-backs. "
            "Dispatch stalls = cycles an SMSP could not dispatch because of a register-read port/bank conflict, per SMSP-clock. TMEM→RF wb stall = tcgen05.ld write-back stalled against the RF write ports, per SMSP-clock. "
            "Uniform RF: `uniform_register_reads_q` / `uniform_register_writes_q` (per-SM sums) / (4 SMSP x 1 uniform read or write per clock) / clocks -- the uniform datapath is one scalar register access per SMSP per clock; "
            "`uniform_pipe_active_q*` and `inst_issued_uniform_pipe_q` give the UR pipe's activity for comparison.",
        ]
    L += ["", "## PIC-Smart webviews (perf-inspector)", ""]
    for r in recs:
        L.append(f"- {r['label']}: {r['pi_url']}")
    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)
        json.dump(recs, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
