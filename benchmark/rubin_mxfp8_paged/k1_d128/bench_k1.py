#!/usr/bin/env python3
"""K1 driver: validate / time sm107/prefill_d128_mxfp8_bench.py (cc 10.7, d128 MXFP8 bench one-off) through the
standalone adapter.  ONE lever set per process: the kernel reads BENCH_* at import, so this driver sets them from
its CLI before importing cudnn and suffixes CUTE_DSL_CACHE_DIR / XDG_CACHE_HOME per lever set.

  --kernel bench|product    product = sm107/prefill_d128_mxfp8.py (dense only, corr default, CORRFAST/HOIST ignored)
  --paged 0|64              64: K/V page pools of 64-row pages behind a RANDOM block table through the bench loader
  --corr default|always|never  --corrfast 0|1  --hoist 0|1  --half 0|1  --prefolded 0|1  --prefolded-via params|env
  --kv-ramp 1               K/V x 2^((t//32) % 4 - 2): adjacent E8M0 blocks differ 2x (wrong SF mapping shows)
  --kv-growth g             K tile t (128 rows) x g**t: the running max grows every tile -> rescales happen
  --validate 1              O vs fp32 reference on the dequantized inputs; rel = max|O - ref| / max|ref|
  --rescale-stats 1         emulate the kernel's rescale rule on the inputs (warp-steps that rescale; 0 => `never` exact)
  --dump-o f.pt | --compare-o f.pt   O A/B between two runs (bitwise flag + max |dO|)
  --time 1 --reps 4 --rounds N       CUDA-graph replay bursts, median us, clocks/power around each burst

Paged plumbing (see the kernel header): the adapter's native MXFP8 binder admits pools only at whole-F8_128x4-atom
page sizes, so the plan is declared paged_page_size=128 over NHD pools (a 64-row-page pool viewed as 128-row pages),
the block tables are the (B, S/64) tables of 64-row page ids (the kernel addresses half-pages), and the dense
tile-indexed SF_K / SF_V bytes ride in the prefix of (n_pages128, H_kv, 1, 512)-shaped buffers.  Two adapter gates
are bypassed per instance: the "Rubin paged KV requires half ..." decline (via _not_implemented_error_if) and the
prefolded-on-paged routing decline (softmax_scale_prefolded toggled off around check_support(), then restored and
scale_softmax pinned to 1/log2 e as the prefolded contract does).
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--kernel", choices=["bench", "product"], default="bench")
ap.add_argument("--paged", type=int, choices=[0, 64], default=64)
ap.add_argument("--corr", choices=["default", "always", "never"], default="default")
ap.add_argument("--corrfast", type=int, default=1)
ap.add_argument("--hoist", type=int, default=1)
ap.add_argument("--half", type=int, default=1)
ap.add_argument("--prefolded", type=int, default=1)
ap.add_argument("--prefolded-via", choices=["params", "env"], default="params")
ap.add_argument("--seqlens", type=int, nargs="+", default=[2048])
ap.add_argument("--mask", choices=["none", "causal"], default="none")
ap.add_argument("--heads", type=int, nargs=2, default=[4, 2], metavar=("H_Q", "H_KV"))
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--sched", choices=["natural", "lpt", "lpt_l2"], default="natural")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--kv-ramp", type=int, default=0)
ap.add_argument("--kv-growth", type=float, default=1.0)
ap.add_argument("--validate", type=int, default=1)
ap.add_argument("--rescale-stats", type=int, default=0)
ap.add_argument("--dump-o", default=None)
ap.add_argument("--compare-o", default=None)
ap.add_argument("--time", type=int, default=0)
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--out", default=None)
ap.add_argument("--tag", default="")
args = ap.parse_args()

if args.kernel == "product":
    if args.paged or args.corr != "default" or args.corrfast or args.hoist:
        sys.exit("[k1] --kernel product serves dense / corr default only (set --paged 0 --corr default --corrfast 0 --hoist 0)")
    if args.prefolded_via == "env":
        sys.exit("[k1] --prefolded-via env is a bench-kernel lever")

# --- env levers BEFORE any cudnn import (the kernel module reads them at import) ---
os.environ["BENCH_PAGED64"] = "1" if (args.kernel == "bench" and args.paged) else "0"
os.environ["BENCH_CORR"] = args.corr
os.environ["BENCH_CORRFAST"] = str(int(bool(args.corrfast)))
os.environ["BENCH_HOIST"] = str(int(bool(args.hoist)))
os.environ["BENCH_HALF"] = "0"  # HALF always travels through TemplateParams (no adapter gate against it)
os.environ["BENCH_PREFOLDED"] = "1" if (args.kernel == "bench" and args.prefolded and args.prefolded_via == "env") else "0"
SUFFIX = f"_k1_{args.kernel}_p{args.paged}_corr{args.corr}_cf{int(bool(args.corrfast))}_h{int(bool(args.hoist))}_half{args.half}_pf{args.prefolded}{args.prefolded_via[0]}"
for var in ("CUTE_DSL_CACHE_DIR", "XDG_CACHE_HOME"):
    base = os.environ.get(var)
    if base:
        os.environ[var] = base.rstrip("/") + SUFFIX
        os.makedirs(os.environ[var], exist_ok=True)

import torch  # noqa: E402

B, HQ, HKV = args.batch, args.heads[0], args.heads[1]
D = 128
LOG2E = math.log2(math.e)
PAGE = 64
POOL_PAGE = 128  # the page size declared to the adapter (whole SF atoms per pool page)
SF_TILE_BYTES = 512  # F8_128x4 atom set for a 128-row x 128-d tile (TILE_N * 128 / 32)
dev = "cuda"
ONEOFF_FILE = "sm107/prefill_d128_mxfp8_bench.py"


def gpu_clocks():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw", "--format=csv,noheader"], text=True)
        return out.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def bshd(x):
    """(B,H,S,D) view over BSHD storage (the kernels' layout; avoids the adapter's gather copy)."""
    return x.permute(0, 2, 1, 3).contiguous().transpose(1, 2)


def ref_attention(q, k, v, scale, causal, blk=1024):
    """fp32 reference, GQA-aware, blocked over Q rows. q (B,HQ,S,D), k/v (B,HKV,S,D) fp32."""
    Bn, Hq, S, _ = q.shape
    G = Hq // k.shape[1]
    out = torch.empty(Bn, Hq, S, v.shape[-1], device=q.device, dtype=torch.float32)
    for b in range(Bn):
        for h in range(Hq):
            kh = h // G
            kt = k[b, kh].transpose(0, 1).contiguous()
            for r0 in range(0, S, blk):
                r1 = min(S, r0 + blk)
                s = torch.matmul(q[b, h, r0:r1], kt) * scale
                if causal:
                    i = torch.arange(r0, r1, device=q.device).view(-1, 1)
                    j = torch.arange(S, device=q.device).view(1, -1)
                    s = s.masked_fill(j > i, float("-inf"))
                p = torch.softmax(s, dim=-1)
                out[b, h, r0:r1] = torch.matmul(p, v[b, kh])
    return out


def rescale_stats(q, k, scale, causal, thresh, tile=128):
    """Emulate the kernel's online-max rule on the dequantized logits (log2 units): a correction WARP (32 rows)
    rescales on a step when any of its rows moves the running max on a step after its first live step.
    Returns (rescaling warp-steps, live warp-steps) for the product threshold rule and for the `always` ratchet."""
    Bn, Hq, S, _ = q.shape
    G = Hq // k.shape[1]
    scale_log2 = scale * LOG2E
    res = {}
    for name, th in (("default", thresh), ("always", 0.0)):
        n_resc = n_live = 0
        for b in range(Bn):
            for h in range(Hq):
                kh = k[b, h // G]
                m = torch.full((S,), float("-inf"), device=q.device)
                first = torch.ones((S,), dtype=torch.bool, device=q.device)
                for t in range(S // tile):
                    s_blk = (q[b, h] @ kh[t * tile : (t + 1) * tile].T) * scale_log2
                    if causal:
                        rows = torch.arange(S, device=q.device)[:, None]
                        cols = (t * tile + torch.arange(tile, device=q.device))[None, :]
                        s_blk = s_blk.masked_fill(cols > rows, float("-inf"))
                    cur = s_blk.amax(dim=1)
                    live = torch.isfinite(cur)
                    upd = live & (first | ((cur - m) > th))
                    m = torch.where(upd, cur, m)
                    resc_rows = upd & ~first & live
                    first = first & ~live
                    n_live += int(live.view(-1, 32).any(dim=1).sum())
                    n_resc += int(resc_rows.view(-1, 32).any(dim=1).sum())
        res[name] = (n_resc, n_live)
    return res


def quantize_mx(x, b, h, s, d, *, columnwise):
    """MXFP8 quantize (B,H,S,D) fp32 -> (fp8 data (B,H,S,D), F8_128x4-swizzled SF bytes, per-elem dequant)."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(x, b, h, s, d, 32, torch.float8_e4m3fn, with_ref=True)
    if columnwise:
        return data_s, swz_s.contiguous(), dq_s.reshape(b, h, s, d)
    return data_d, swz_d.contiguous(), dq_d.reshape(b, h, s, d)


def build_case(S, causal):
    torch.manual_seed(args.seed)
    scale = 1.0 / math.sqrt(D)
    qf = torch.randn(B, HQ, S, D, device=dev) * 0.5
    kf = torch.randn(B, HKV, S, D, device=dev) * 0.5
    vf = torch.randn(B, HKV, S, D, device=dev) * 0.5
    if args.kv_ramp:
        t = torch.arange(S, device=dev)
        ramp = torch.pow(2.0, ((t // 32) % 4 - 2).float()).view(1, 1, S, 1)
        kf = kf * ramp
        vf = vf * ramp
    if args.kv_growth != 1.0:
        t = torch.arange(S, device=dev)
        kf = kf * torch.pow(torch.tensor(args.kv_growth, device=dev), (t // 128).float()).view(1, 1, S, 1)
    prefolded = bool(args.prefolded)
    q_in = qf * (scale * LOG2E) if prefolded else qf
    ref_scale = math.log(2.0) if prefolded else scale
    o = torch.empty(B, S, HQ, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    q8, sf_q, dqq = quantize_mx(q_in, B, HQ, S, D, columnwise=False)
    k8, sf_k, dqk = quantize_mx(kf, B, HKV, S, D, columnwise=False)
    v8, sf_v, dqv = quantize_mx(vf, B, HKV, S, D, columnwise=True)
    qb, kb, vb = bshd(q8), bshd(k8), bshd(v8)
    case = dict(
        S=S,
        causal=causal,
        scale=scale,
        o=o,
        q=qb,
        k=kb,
        v=vb,
        exec_extra=dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v),
        adapter=dict(),
        ref_inputs=(qb.float() * dqq, kb.float() * dqk, vb.float() * dqv, ref_scale),
    )
    if args.paged:
        assert S % POOL_PAGE == 0
        n_pages = B * (S // PAGE) + 8  # even; >= 8 dead pages
        bt = torch.randperm(n_pages, device=dev)[: B * (S // PAGE)].to(torch.int32).view(B, S // PAGE).contiguous()
        pools = []
        for x8 in (k8, v8):
            pool = torch.zeros(n_pages, PAGE, HKV, D, device=dev, dtype=torch.float8_e4m3fn)  # NHD: [page, row, head, d]
            xd = x8.view(B, HKV, S // PAGE, PAGE, D).permute(0, 2, 3, 1, 4).reshape(B * (S // PAGE), PAGE, HKV, D)
            pool[bt.view(-1).long()] = xd
            # the adapter's container: (n_pages128, H_kv, 128, D) with NHD strides == the same bytes as 128-row pages
            pools.append(pool.view(n_pages // 2, POOL_PAGE, HKV, D).permute(0, 2, 1, 3))
        sf_pools = []
        for sf in (sf_k, sf_v):
            buf = torch.zeros(n_pages // 2, HKV, 1, SF_TILE_BYTES, device=dev, dtype=sf.dtype)
            assert sf.numel() == B * HKV * (S // 128) * SF_TILE_BYTES <= buf.numel(), (sf.shape, buf.shape)
            buf.view(-1)[: sf.numel()] = sf.reshape(-1)
            sf_pools.append(buf)
        case.update(
            k=pools[0],
            v=pools[1],
            exec_extra=dict(sf_q=sf_q, sf_k=sf_pools[0], sf_v=sf_pools[1]),
            adapter=dict(seq_kv_lens_present=True, paged_page_size=POOL_PAGE, paged_max_seq_len_kv=S),
            paged=dict(bt=bt, seq_kv_lens=torch.full((B,), S, dtype=torch.int32, device=dev), n_pages=n_pages),
        )
    return case


def make_api(case):
    import cudnn
    from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if args.kernel == "bench":
        api_dsl._SM107_MXFP8_KERNEL_FILES[(128, 128)] = ONEOFF_FILE
    prefolded_params = bool(args.prefolded) and args.prefolded_via == "params"
    kw = dict(
        sample_q=case["q"],
        sample_k=case["k"],
        sample_v=case["v"],
        sample_o=case["o"],
        is_causal=case["causal"],
        # prefolded contract: leave the scale unset; env override: pin scale_softmax_log2 to 1.0 (the kernel ignores it)
        scale_softmax=None if prefolded_params else (1.0 / LOG2E if args.prefolded else case["scale"]),
        dtype_o=case["o"].dtype,
        has_amax_o=False,
        split_kv=1,
        pertensor_fp8=False,
        sched_policy={"natural": SCHED_NATURAL, "lpt": SCHED_LPT, "lpt_l2": SCHED_LPT_L2}[args.sched],
        softmax_precision=cudnn.data_type.HALF if args.half else None,
        softmax_scale_prefolded=prefolded_params,
    )
    kw.update(case["adapter"])
    api = SdpaFwdDslSm100(**kw)
    bypassed = []
    if args.paged:
        orig = api._not_implemented_error_if

        def _lenient(cond, msg):
            if cond and msg and ("Rubin paged KV requires" in msg or "SM107 sibling" in msg or "multiple of 128" in msg or "F8_128x4 SF atoms" in msg):
                bypassed.append(msg)
                return
            return orig(cond, msg)

        api._not_implemented_error_if = _lenient
    if args.paged and prefolded_params:
        # the adapter routes prefolded+paged to the SM100 paged bodies and declines (a direct raise); this bench
        # kernel IS the paged body, so run the remaining checks with the flag off and restore it for compile()
        api.softmax_scale_prefolded = False
        ok = api.check_support()
        api.softmax_scale_prefolded = True
        api.scale_softmax = 1.0 / LOG2E
        bypassed.append("prefolded-on-paged routing decline (flag toggled around check_support)")
    else:
        ok = api.check_support()
    assert ok, "check_support declined"
    api.compile()
    return api, bypassed


def run_execute(api, case, ws):
    kw = dict(workspace=ws) if ws is not None else {}
    kw.update(case["exec_extra"])
    if case.get("paged"):
        kw.update(seq_kv_lens=case["paged"]["seq_kv_lens"], block_table=case["paged"]["bt"], block_table_v=case["paged"]["bt"])
    api.execute(case["q"], case["k"], case["v"], case["o"], **kw)


def time_graph(api, case, ws):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run_execute(api, case, ws)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            run_execute(api, case, ws)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    times, clocks = [], []
    for _ in range(args.rounds):
        clocks.append(gpu_clocks())
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3 / args.reps)
        if args.rounds > 1:
            time.sleep(0.15)
    return times, clocks


def validate(case):
    q, k, v, scale = case["ref_inputs"]
    ref = ref_attention(q, k, v, scale, case["causal"])
    out = case["o"].float()
    err = (out - ref).abs()
    return dict(
        max_abs_err=err.max().item(),
        mean_abs_err=err.mean().item(),
        ref_amax=ref.abs().max().item(),
        rel_err=(err.max() / ref.abs().max().clamp_min(1e-12)).item(),
        finite=bool(torch.isfinite(out).all().item()),
        n_bad_1e2=int((err > 1e-2 * ref.abs().max()).sum().item()),
    )


def main():
    cc = torch.cuda.get_device_capability()
    assert cc == (10, 7), f"cc 10.7 only; got {cc}"
    causal = args.mask == "causal"
    print(
        f"[k1] {args.kernel} paged={args.paged} corr={args.corr} corrfast={args.corrfast} hoist={args.hoist} half={args.half} prefolded={args.prefolded}({args.prefolded_via}) "
        f"B{B} H{HQ}/{HKV} mask={args.mask} | {torch.cuda.get_device_name()} | {gpu_clocks()} | cache {os.environ.get('CUTE_DSL_CACHE_DIR')}",
        flush=True,
    )
    for S in args.seqlens:
        case = build_case(S, causal)
        t0 = time.time()
        api, bypassed = make_api(case)
        ws_bytes = api.scratch_workspace_bytes()
        ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=dev) if ws_bytes else None
        compile_s = time.time() - t0
        km = getattr(api, "_k_mod", None)
        rec = dict(
            kernel=args.kernel,
            paged=args.paged,
            corr=args.corr,
            corrfast=args.corrfast,
            hoist=args.hoist,
            half=args.half,
            prefolded=args.prefolded,
            prefolded_via=args.prefolded_via,
            S=S,
            mask=args.mask,
            B=B,
            h_q=HQ,
            h_kv=HKV,
            d=D,
            sched=args.sched,
            kv_ramp=args.kv_ramp,
            kv_growth=args.kv_growth,
            seed=args.seed,
            kernel_template=getattr(api, "kernel_template", "?"),
            kernel_file=api_kernel_file(),
            kmod={
                k: getattr(km, k, None)
                for k in (
                    "PAGED_KV",
                    "PAGE_SIZE",
                    "POOL_PAGE_ROWS",
                    "HALF_PAGES",
                    "BENCH_CORR",
                    "BENCH_CORRFAST",
                    "BENCH_HOIST",
                    "SOFTMAX_F16",
                    "SCALE_PREFOLDED",
                    "_FUSED_SHIFT_CVT",
                    "FROST_SOURCE_DIGEST",
                )
            },
            cfg={
                k: getattr(getattr(km, "CFG", None), k, None)
                for k in ("CTA_MMA", "STAGES_KV", "TILE_N", "TILES_Q", "RESCALE_THRESHOLD", "MASK_FLAGS", "SEQ_KV_LENS_PRESENT")
            },
            bypassed=bypassed,
            compile_s=round(compile_s, 1),
            host=os.uname().nodename,
            tag=args.tag,
            cache_dir=os.environ.get("CUTE_DSL_CACHE_DIR"),
        )
        for _ in range(args.warmup):
            run_execute(api, case, ws)
        torch.cuda.synchronize()
        if args.validate:
            run_execute(api, case, ws)
            torch.cuda.synchronize()
            rec["validation"] = validate(case)
        if args.rescale_stats:
            q, k, _v, scale = case["ref_inputs"]
            thresh = float(getattr(getattr(km, "CFG", None), "RESCALE_THRESHOLD", 4.0))
            rec["rescale_stats"] = {n: dict(rescaling_warp_steps=r, live_warp_steps=l) for n, (r, l) in rescale_stats(q, k, scale, causal, thresh).items()}
        if args.dump_o:
            torch.save(case["o"].detach().clone().cpu(), args.dump_o)
        if args.compare_o:
            other = torch.load(args.compare_o).to(dev)
            d_o = (case["o"].float() - other.float()).abs()
            rec["compare_o"] = dict(
                path=args.compare_o,
                bitwise=bool(torch.equal(case["o"].view(torch.int16), other.view(torch.int16))),
                max_abs_diff=d_o.max().item(),
                rel_diff=(d_o.max() / other.float().abs().max().clamp_min(1e-12)).item(),
            )
        if args.time:
            times, clocks = time_graph(api, case, ws)
            rec["time_us"] = [round(t, 2) for t in times]
            rec["time_us_median"] = round(statistics.median(times), 2)
            rec["clocks_per_burst"] = clocks
            flops = 4.0 * B * HQ * S * S * D * (0.5 * (1 + 1.0 / S) if causal else 1.0)
            rec["tflops"] = round(flops / (rec["time_us_median"] * 1e-6) / 1e12, 1)
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        del api, case, ws
        torch.cuda.empty_cache()


def api_kernel_file():
    from cudnn.sdpa.fwd import api_dsl

    return api_dsl._SM107_MXFP8_KERNEL_FILES[(128, 128)]


if __name__ == "__main__":
    main()
