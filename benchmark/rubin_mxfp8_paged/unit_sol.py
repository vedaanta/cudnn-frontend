#!/usr/bin/env python3
"""Analytic per-unit speed-of-light model for the cc 10.7 MXFP8 prefill kernels, per 128-query x 128-key STEP per SM.

  unit_sol.py --timing results.jsonl [--ncu ncu_table.json] [--d 128] [--mask none] [--S 32768] [--out unit_sol.json]

For every timing record (one config x S x mask) the model lists, per unit, the MINIMUM clocks a 128x128 step needs
if that unit ran at its documented peak, the measured clocks per step (kernel us x SM clock / steps per SM), and
util = SOL / measured.  ncu pipe percentages (ncu_table.py JSON) are shown next to the model as the cross-check; where
ncu is present the issue / FMA / ALU / XU instruction counts per step come from the EXECUTED SASS (ncu source page,
sass_tools.ncu_source) instead of the static estimate.

Units and constants (per SM, per clock) -- every number carries its source, see CONSTANTS below:
  tensor  dense FP8 UTCQMMA MACs/clk/SM             16384  derived: PerfSim gr100 math-SOL table (17408 SOL cycles for
                                                    32 kv steps x [2 x (128x128x128 BMM1 + 128x128x128 BMM2 + 128x16x128
                                                    row-sum)] at d128 TILES_Q=2) and the silicon cross-check (6.9 PF/s on
                                                    208 SMs @2.4 GHz = 6.9k MAC/clk/SM achieved at a 45 % ncu tensor-active).
                                                    NOTE: the plan quoted Blackwell's 8192; GR100 doubled the per-SM rate
                                                    (--tc-mac-per-clk overrides).
  xu      MUFU lanes/clk/SM                         32     ncu-consistent: fp32 EX2 (16384/step) ran at 770 clk/step with 62-68 %
                                                    MUFU on the ladder's fp32 rungs -> >= 21 lanes/clk; f16x2 at 34-38 %.  The
                                                    CUDA C Programming Guide table lists 16 for exp2 on cc 8.x-12.x
                                                    (--xu-lanes-per-clk 16 to use it).  MUFU.EX2.F16x2 = 2 results per lane.
  issue   warp-instructions/clk/SM                  4      1 per SMSP scheduler per clock (SM architecture).
  fma     warp-instr/clk/SM on fmaheavy+fmalite     2      16 lanes per pipe per SMSP: a 32-thread warp-instr occupies a pipe 2 clk;
                                                    FFMA2 / HFMA2-class instructions produce 2 results per lane at the same issue cost.
  alu     warp-instr/clk/SM (int ALU + F2FP cvt)     2      16 lanes/clk/SMSP (Ampere+ ALU datapath; conversions ride it on sm_10x).
  smem    bytes/clk/SM                               128    the plan's figure (32 banks x 4 B); counts TMA writes AND UMMA operand reads.
  tmem    bytes/clk/SM (tcgen05.ld/st side)          512    ASSUMED: 128 lanes x 32 bit per clock; the tensor core's own accumulator
                                                    traffic is not modelled (it is internal to the MMA pipe).
  l2      bytes/clk/SM                               ncu-derived when available (lts__t_bytes / lts__throughput pct), else 64 (ASSUMED).
  dram    bytes/s                                    ncu-derived when available (dram__bytes / dram__throughput pct), else 8e12 (ASSUMED).
Step definition: ONE 128-row Q sub-tile x ONE 128-key K/V tile on one SM.  d128 (TILES_Q=2) runs two sub-tiles per K/V tile,
so per-K/V-tile costs (TMA writes, L2 fetch, loader instructions) are divided by TILES_Q; at CTA_MMA=2 the pair shares each
K/V tile (multicast) so per-SM TMA/L2 bytes are halved and the UMMA reads only this CTA's half of K (B operand, N split)
and of V (N = d split).  steps_total = B x H_q x n x n (dense) or n(n+1)/2 (causal), n = S/128.
"""

import argparse
import json
import math
import os
import sys

CONSTANTS = {
    "tc_mac_per_clk": (16384, "PerfSim gr100 math-SOL (17408 cyc = 32 kv steps x 544) + silicon 6.9 PF/s @ 45 % tensor-active; Blackwell = 8192"),
    "xu_lanes_per_clk": (32, "ncu-consistent (fp32 EX2 at 62-68 % MUFU, 770 clk/step); CUDA guide table says 16 for exp2 on cc 10.x"),
    "issue_per_clk": (4, "4 SMSP schedulers x 1 warp-instruction/clk"),
    "fma_winst_per_clk": (2, "fmaheavy + fmalite, 16 lanes each per SMSP -> 2 clk per warp-instruction per pipe"),
    "alu_winst_per_clk": (2, "16 lanes/clk/SMSP"),
    "smem_bytes_per_clk": (128, "32 banks x 4 B (plan.md); TMA writes + UMMA operand reads"),
    "tmem_bytes_per_clk": (512, "ASSUMED 128 lanes x 32 bit per clock for tcgen05.ld/st"),
    "l2_bytes_per_clk_sm": (64, "ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present"),
    "dram_bytes_per_s": (8.0e12, "ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present"),
}
# Static instruction estimate per 128x128 step per SM (warp-instructions), used only when no ncu SASS page is present:
# softmax 4 warps x 297 (ladder F SASS: 64 FFMA2 + 128 F2FP + 65 MUFU.EX2.F16x2 + ~40 overhead per warp per 128-token step)
# + correction 4 warps x ~25 (vote + arrive, no rescale) + MMA warp ~40 + TMA/scheduler ~30.
STATIC_INST = dict(
    f16=dict(total=1188 + 100 + 40 + 30, fma=4 * 64, alu=4 * 128 + 60, xu=4 * 65),
    f32=dict(total=4 * 420 + 100 + 40 + 30, fma=4 * 128, alu=4 * 64 + 60, xu=4 * 129),
)


def steps_total(B, HQ, S_q, S_kv, causal):
    nq, nk = S_q // 128, S_kv // 128
    if causal:
        return B * HQ * (nq * (nq + 1) // 2) if nq == nk else B * HQ * sum(min(nk, i + 1 + (nk - nq)) for i in range(nq))
    return B * HQ * nq * nk


def model_step(d, cta_mma, tiles_q, rowsum_mma, f16, corr_always, paged, inst, C):
    """Per-unit SOL clocks for one 128x128 step on one SM.  Returns dict unit -> (clk, detail)."""
    tc = C["tc_mac_per_clk"]
    macs_bmm1 = 128 * 128 * d
    macs_bmm2 = 128 * d * 128
    macs_rowsum = 128 * 16 * 128 if rowsum_mma else 0
    tensor_clk = (macs_bmm1 + macs_bmm2 + macs_rowsum) / tc
    # MUFU: one exp per P element (f16x2 packs 2); +1 alpha per row (negligible)
    exps = 128 * 128
    xu_clk = (exps / (2 if f16 else 1)) / C["xu_lanes_per_clk"]
    # SMEM.  UMMA operand reads: BMM1 A = Q tile (SS, re-read every step) 128 x d B; B = this CTA's K half (N/CTA_MMA keys x d);
    # BMM2 B = this CTA's V half (128 keys x d/CTA_MMA cols); ones tile for the row-sum MMA (16 x 128 B).  TMA writes per K/V tile:
    # K + V (128 x d each) / CTA_MMA (multicast halves) + SF (128 x d/32 x 2), amortised over TILES_Q sub-tiles.
    q_bytes = 128 * d
    k_read = (128 // cta_mma) * d
    v_read = 128 * (d // cta_mma)
    ones = 16 * 128 if rowsum_mma else 0
    smem_reads = q_bytes + k_read + v_read + ones
    kv_tile_bytes = 2 * 128 * d + 2 * 128 * (d // 32)
    smem_writes = kv_tile_bytes / cta_mma / tiles_q
    smem_clk = (smem_reads + smem_writes) / C["smem_bytes_per_clk"]
    # TMEM, softmax side: S read (128 x 128 fp32), P write (128 x 128 fp8), alpha/stats ~1 KB; corr=always adds the O rescale
    # (tcgen05.ld + st of 128 x d fp32).
    tmem_bytes = 128 * 128 * 4 + 128 * 128 * 1 + 1024 + (2 * 128 * d * 4 if corr_always else 0)
    tmem_clk = tmem_bytes / C["tmem_bytes_per_clk"]
    # L2: K/V tile + SF fetched once per cluster (multicast) per K/V tile, amortised over the sub-tiles
    l2_bytes = kv_tile_bytes / cta_mma / tiles_q
    l2_clk = l2_bytes / C["l2_bytes_per_clk_sm"]
    issue_clk = inst["total"] / C["issue_per_clk"]
    fma_clk = inst["fma"] / C["fma_winst_per_clk"]
    alu_clk = inst["alu"] / C["alu_winst_per_clk"]
    cvt_clk = inst.get("cvt", 0.0) / C["alu_winst_per_clk"]
    xu_inst_clk = inst["xu"] * 32 / C["xu_lanes_per_clk"]
    out = {
        "tensor": (
            tensor_clk,
            f"BMM1 {macs_bmm1 // 1024}k + BMM2 {macs_bmm2 // 1024}k" + (f" + rowsum {macs_rowsum // 1024}k" if rowsum_mma else "") + f" MAC / {tc}",
        ),
        "mufu": (xu_clk, f"{exps} exp{' as f16x2 pairs' if f16 else ' fp32'} / {C['xu_lanes_per_clk']} lanes"),
        "issue": (
            issue_clk,
            f"{inst['total']:.0f} warp-instr / {C['issue_per_clk']} ({inst['src']})"
            + (f"; of which sync/branch {inst['ctrl']:.0f}" if inst.get("ctrl") else ""),
        ),
        "fma": (fma_clk, f"{inst['fma']:.0f} FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / {C['fma_winst_per_clk']} ({inst['src']})"),
        "alu": (alu_clk, f"{inst['alu']:.0f} integer/logic ALU warp-instr / {C['alu_winst_per_clk']} ({inst['src']})"),
        "cvt": (
            cvt_clk,
            f"{inst.get('cvt', 0):.0f} F2FP/I2F convert+pack warp-instr / {C['alu_winst_per_clk']} ({inst['src']}; ncu pipe attribution of F2FP differs)",
        ),
        "xu_inst": (xu_inst_clk, f"{inst['xu']:.0f} MUFU warp-instr x 32 lanes / {C['xu_lanes_per_clk']} ({inst['src']})"),
    }
    out.update(
        {
            "smem": (
                smem_clk,
                f"reads {smem_reads // 1024} KiB (Q {q_bytes // 1024} + K/{cta_mma} {k_read // 1024} + V/{cta_mma} {v_read // 1024}{' + ones 2' if rowsum_mma else ''}) + TMA writes {smem_writes / 1024:.1f} KiB / {C['smem_bytes_per_clk']} B",
            ),
            "tmem": (tmem_clk, f"{tmem_bytes // 1024} KiB tcgen05.ld/st / {C['tmem_bytes_per_clk']} B (assumed)"),
            "l2": (l2_clk, f"{l2_bytes / 1024:.1f} KiB K/V(+SF) per step per SM / {C['l2_bytes_per_clk_sm']:.0f} B/clk/SM"),
        }
    )
    return out


def inst_from_ncu(n, steps_all):
    """Executed warp-instructions per 128x128 step per SM from the ncu row (SASS classes when present, pipes otherwise)."""
    if not n:
        return None
    cls = n.get("sass_by_class")
    if cls and n.get("sass_exec_total"):
        tot = n["sass_exec_total"]
        return dict(
            total=tot / steps_all,
            fma=cls.get("fma", 0) / steps_all,
            alu=cls.get("alu_int", 0) / steps_all,
            cvt=cls.get("alu_cvt", 0) / steps_all,
            xu=cls.get("xu", 0) / steps_all,
            ctrl=cls.get("ctrl_sync", 0) / steps_all,
            tmem=cls.get("tmem", 0) / steps_all,
            lsu=cls.get("lsu_mem", 0) / steps_all,
            src="ncu SASS executed",
            by_class={k: v / steps_all for k, v in cls.items()},
        )
    if n.get("inst_total"):
        return dict(
            total=n["inst_total"] / steps_all,
            fma=(n.get("fma_inst") or 0) / steps_all,
            alu=(n.get("alu_inst") or 0) / steps_all,
            cvt=0.0,
            xu=(n.get("xu_inst") or 0) / steps_all,
            src="ncu pipe counters",
        )
    return None


def peaks_from_ncu(n, C):
    """Replace the ASSUMED L2 / DRAM peaks by ncu-derived ones (bytes per second / pct of peak)."""
    out = dict(C)
    notes = {}
    if n:
        sm = n.get("sm_count") or 208
        clk = (n.get("clk_ghz") or 2.4) * 1e9
        if n.get("l2_peak_tbs_ncu"):
            peak_bs = n["l2_peak_tbs_ncu"] * 1e12  # lts__t_sectors x 32 B / its pct of peak (sector throughput, all slices)
            out["l2_bytes_per_clk_sm"] = peak_bs / clk / sm
            notes["l2"] = f"ncu-derived L2 sector peak {peak_bs / 1e12:.1f} TB/s = {out['l2_bytes_per_clk_sm']:.0f} B/clk/SM (lts__t_sectors / pct_of_peak)"
        if n.get("dram_gbs") and n.get("dram_pct"):
            out["dram_bytes_per_s"] = n["dram_gbs"] * 1e9 / (n["dram_pct"] / 100.0)
            notes["dram"] = f"ncu-derived DRAM peak {out['dram_bytes_per_s'] / 1e12:.1f} TB/s"
        if n.get("inst_total") and n.get("sass_exec_total"):
            notes["issue"] = (
                f"ncu smsp__inst_executed.sum {n['inst_total'] / 1e6:.0f} M vs SASS-page executed {n['sass_exec_total'] / 1e6:.0f} M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)"
            )
    return out, notes


def analyze(rec, n, C0, args):
    d, S, causal = rec["d"], rec["S"], rec["mask"] == "causal"
    B, HQ, HKV = rec.get("B", 1), rec.get("h_q", 32), rec.get("h_kv", 8)
    cfg = rec.get("cfg") or {}
    kmod = rec.get("kmod") or {}
    cta_mma = rec.get("cta_mma") or cfg.get("CTA_MMA") or (2 if d == 128 else 1)
    tiles_q = cfg.get("TILES_Q") or (2 if d == 128 else 1)
    f16 = bool(kmod.get("SOFTMAX_F16", rec.get("config", {}).get("half", 1)))
    rowsum_mma = (d == 128) or bool(int(kmod.get("BENCH_ROWSUM_MMA") or rec.get("config", {}).get("rowsum_mma", 0)))
    corr_always = rec.get("config", {}).get("corr") == "always"
    paged = int(rec.get("config", {}).get("paged", 0))
    sm_count = rec.get("sm_count") or (n or {}).get("sm_count") or args.sm_count
    steps_all = steps_total(B, HQ, S, S, causal)
    steps_sm = steps_all / sm_count
    us = rec.get("time_us_graph")
    mhz = rec.get("clock_mhz_median") or ((n or {}).get("clk_ghz") or 0) * 1e3 or args.clock_mhz
    if not us or not mhz:
        return None
    step_clk = us * 1e-6 * mhz * 1e6 / steps_sm
    C, notes = peaks_from_ncu(n, C0)
    inst = inst_from_ncu(n, steps_all) or dict(STATIC_INST["f16" if f16 else "f32"], src="static estimate")
    if inst["src"] == "static estimate":
        inst = dict(inst)
        inst["cvt"] = 4 * 128 if f16 else 4 * 64  # F2FP converts + fp8 packs per softmax warp per step
        inst["alu"] -= inst["cvt"]
        if corr_always:
            extra = (d // 2 + d // 16) * 4  # per-warp O rescale (FMUL2 over d regs + tcgen05.ld/st) x 4 correction warps
            inst["total"] += extra
            inst["fma"] += d // 2 * 4
    sol = model_step(d, cta_mma, tiles_q, rowsum_mma, f16, corr_always, paged, inst, C)
    # DRAM: whole kernel -- Q + O + K + V (+ SF), each read / written once
    dram_bytes = B * HQ * S * d * (1 + 2) + 2 * B * HKV * S * d * (1 + 1 / 32)
    dram_us = dram_bytes / C["dram_bytes_per_s"] * 1e6
    # ncu pipe percentages are "of peak sustained over SM-ACTIVE cycles": compare them with util / (SM-active fraction)
    active_frac = ((n or {}).get("sm_active_pct") or 100.0) / 100.0
    rows = []
    for unit, (clk, detail) in sol.items():
        rows.append(dict(unit=unit, sol_clk=clk, util_pct=100.0 * clk / step_clk, util_active_pct=100.0 * clk / step_clk / active_frac, detail=detail))
    rows.append(
        dict(
            unit="dram",
            sol_clk=dram_us * 1e-6 * mhz * 1e6 / steps_sm,
            util_pct=100.0 * dram_us / us,
            util_active_pct=100.0 * dram_us / us / active_frac,
            detail=f"{dram_bytes / 1e6:.0f} MB whole kernel / {C['dram_bytes_per_s'] / 1e12:.1f} TB/s",
        )
    )
    binding = max(rows, key=lambda r: r["sol_clk"])
    xcheck = {}
    if n:
        xcheck = dict(
            tensor=n.get("tensor_act_pct"),
            mufu=n.get("xu_mufu_pct"),
            xu_inst=n.get("xu_mufu_pct"),
            issue=n.get("issue_pct"),
            fma=n.get("fma_pct"),
            alu=n.get("alu_pct"),
            tmem=n.get("tmem_inst_pct"),
            smem=n.get("l1_pct"),
            l2=n.get("l2_pct"),
            dram=n.get("dram_pct"),
            ncu_us=n.get("dur_us"),
            ncu_clk_ghz=n.get("clk_ghz"),
            sm_active_pct=n.get("sm_active_pct"),
        )
        if n.get("smem_wavefronts") and steps_all:
            xcheck["smem_wavefronts_per_step"] = n["smem_wavefronts"] / steps_all
    return dict(
        name=rec["name"],
        d=d,
        S=S,
        mask=rec["mask"],
        sched=rec.get("sched"),
        cta_mma=cta_mma,
        tiles_q=tiles_q,
        f16=f16,
        rowsum_mma=rowsum_mma,
        corr=rec.get("config", {}).get("corr"),
        paged=paged,
        time_us=us,
        clock_mhz=mhz,
        sm_count=sm_count,
        steps_total=steps_all,
        steps_per_sm=steps_sm,
        step_clk=step_clk,
        mma_util_pct=100.0 * sol["tensor"][0] / step_clk,
        binding_unit=binding["unit"],
        binding_pct=binding["util_pct"],
        inst_per_step=inst,
        units=rows,
        ncu=xcheck,
        peak_notes=notes,
        constants={k: C[k] for k in C},
    )


def md_table(a):
    L = [
        f"### {a['name']} d{a['d']} S={a['S']} {a['mask']} (cta_mma {a['cta_mma']}, TILES_Q {a['tiles_q']}, {'f16' if a['f16'] else 'f32'} exp, rowsum-MMA {int(a['rowsum_mma'])}, corr {a['corr']}, paged {a['paged']})",
        f"measured {a['time_us']:.1f} us @ {a['clock_mhz']:.0f} MHz, {a['steps_per_sm']:.0f} steps/SM -> **{a['step_clk']:.0f} clk per 128x128 step**; MMA util {a['mma_util_pct']:.1f} %; binding unit by SOL: {a['binding_unit']} ({a['binding_pct']:.0f} %)"
        + (
            f"; ncu: {a['ncu']['ncu_us']:.1f} us @ {a['ncu']['ncu_clk_ghz']:.2f} GHz, SM-active {a['ncu']['sm_active_pct']:.1f} %"
            if a.get("ncu", {}).get("ncu_us")
            else ""
        ),
        "",
        "| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |",
        "|---|---|---|---|---|---|",
    ]
    xc = a.get("ncu", {})
    for r in a["units"]:
        x = xc.get(r["unit"], None)
        L.append(
            f"| {r['unit']} | {r['sol_clk']:.0f} | {r['util_pct']:.1f} % | {r['util_active_pct']:.1f} % | {'' if x is None else f'{x:.1f}'} | {r['detail']} |"
        )
    if a["peak_notes"]:
        L.append("")
        L.append("; ".join(a["peak_notes"].values()))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", required=True, help="bench.py results.jsonl")
    ap.add_argument("--ncu", default=None, help="ncu_table.py JSON (per (d, name, mask, S, sched))")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--S", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sm-count", type=int, default=208)
    ap.add_argument("--clock-mhz", type=float, default=2400.0, help="fallback SM clock when the record has none")
    ap.add_argument("--tc-mac-per-clk", type=float, default=None)
    ap.add_argument("--xu-lanes-per-clk", type=float, default=None)
    ap.add_argument("--smem-bytes-per-clk", type=float, default=None)
    ap.add_argument("--out", default=None, help="write the per-record analysis as JSON")
    ap.add_argument("--md", default=None, help="write the markdown tables here (also printed)")
    args = ap.parse_args()
    C = {k: v[0] for k, v in CONSTANTS.items()}
    for k in ("tc_mac_per_clk", "xu_lanes_per_clk", "smem_bytes_per_clk"):
        v = getattr(args, k)
        if v is not None:
            C[k] = v
    ncu = {}
    if args.ncu and os.path.exists(args.ncu):
        for n in json.load(open(args.ncu)):
            ncu[(n["d"], n["name"], n["mask"], n["S"], n.get("sched", "natural"))] = n
    recs = {}
    for line in open(args.timing):
        if line.strip():
            r = json.loads(line)
            recs[(r["d"], r["name"], r["mask"], r["S"], r.get("sched", "natural"))] = r  # last write wins
    # The L2 / DRAM peaks are board properties, not per-record ones: records without their own ncu row (the 8k / 32k
    # timing cells) take the peaks derived from any ncu row of the same head dim instead of the ASSUMED fallbacks, so the
    # l2 / dram columns are comparable across S.
    peaks_by_d = {}
    for n in ncu.values():
        if n.get("l2_peak_tbs_ncu") and n["d"] not in peaks_by_d:
            peaks_by_d[n["d"]] = peaks_from_ncu(n, C)[0]
    out = []
    md = ["## Per-unit SOL model (per 128x128 step per SM)", "", "Constants: " + "; ".join(f"{k} = {C[k]:g} ({CONSTANTS[k][1]})" for k in CONSTANTS), ""]
    if peaks_by_d:
        md += [
            "Records without an ncu row use the ncu-derived L2 / DRAM peaks of the same head dim: "
            + "; ".join(
                f"d{d}: L2 {p['l2_bytes_per_clk_sm']:.0f} B/clk/SM, DRAM {p['dram_bytes_per_s'] / 1e12:.1f} TB/s" for d, p in sorted(peaks_by_d.items())
            ),
            "",
        ]
    for key in sorted(recs):
        r = recs[key]
        if (args.d and r["d"] != args.d) or (args.mask and r["mask"] != args.mask) or (args.S and r["S"] != args.S) or (args.name and r["name"] != args.name):
            continue
        a = analyze(r, ncu.get(key), peaks_by_d.get(r["d"], C) if ncu.get(key) is None else C, args)
        if a:
            out.append(a)
            md.append(md_table(a))
    text = "\n".join(md)
    print(text)
    if args.md:
        open(args.md, "w").write(text)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"wrote {args.out} ({len(out)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
