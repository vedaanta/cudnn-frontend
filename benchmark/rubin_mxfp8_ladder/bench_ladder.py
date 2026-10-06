#!/usr/bin/env python3
"""Rubin (cc 10.7) MXFP8 prefill ladder — FROST sm107 SDPA kernels via the direct adapter.

One rung per process (the one-off kernel reads its levers from env at import):

  A  per-tensor FP8 (e4m3), fp32 softmax           sdpa_fwd_prefill_sm107_fp8   (product kernel)
  B  MXFP8 (e4m3 + E8M0/32), fp32 softmax          sdpa_fwd_prefill_sm107_mxfp8 (product kernel)
  C  B + f16 exponent (MUFU.EX2.F16x2 + f16x2->fp8 P cast)      one-off kernel, LADDER_F16EXP=1
  D  C + softmax scale folded outside (Q pre-scaled by s*log2e) one-off kernel, LADDER_NOSCALE=1
  E  D + paged KV cache, page_size 64 (HND pools, random table)  one-off kernel, LADDER_PAGED64=1
  F  C + correction fast path (alpha via SMEM)                   one-off kernel, LADDER_CORRFAST=1
  G  F + S half-buffer double-buffering (BMM1 overlaps softmax)  one-off kernel, LADDER_SDOUBLE=1
  H  G + P in SMEM, early BMM1 (S half-buffer freed on read)      one-off kernel, LADDER_PSMEM=1

Shape: B=1, H_q=32, H_kv=2 (GQA 16:1), d=128, S_q=S_kv in {8k,16k,32k}, O in bf16,
no Stats (inference), no Amax_O.  Timing = CUDA-graph replay of one execute (falls back to
an eager loop); kernel-only time is also taken by ncu in --ncu-run mode.

Usage (inside the env wrapper):
  python bench_ladder.py --rung B --seqlens 8192 16384 32768 --mask causal --out results.jsonl
  python bench_ladder.py --rung B --seqlens 8192 --mask causal --ncu-run     # 3 warmups + 1 launch
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

RUNGS = {
    "A": dict(label="fp8 per-tensor, fp32 softmax", family="fp8", env={}),
    "B": dict(label="mxfp8, fp32 softmax", family="mxfp8", env={}),
    "C": dict(label="mxfp8 + f16 exp", family="mxfp8", env={"LADDER_F16EXP": "1"}),
    "D": dict(label="mxfp8 + f16 exp + scale outside", family="mxfp8", env={"LADDER_F16EXP": "1", "LADDER_NOSCALE": "1"}),
    "E": dict(label="mxfp8 + f16 exp + scale outside + paged64", family="mxfp8", env={"LADDER_F16EXP": "1", "LADDER_NOSCALE": "1", "LADDER_PAGED64": "1"}, paged=64),
    "F": dict(label="C + correction fast path (alpha via SMEM)", family="mxfp8", env={"LADDER_F16EXP": "1", "LADDER_CORRFAST": "1"}),
    "G": dict(label="F + S half-buffer double-buffering (BMM1 overlaps softmax)", family="mxfp8", env={"LADDER_F16EXP": "1", "LADDER_CORRFAST": "1", "LADDER_SDOUBLE": "1"}),
    "H": dict(label="G + P in SMEM, early BMM1 (S half-buffer freed on read)", family="mxfp8", env={"LADDER_F16EXP": "1", "LADDER_CORRFAST": "1", "LADDER_SDOUBLE": "1", "LADDER_PSMEM": "1"}),
}
ONEOFF_FILE = "sm107/prefill_d128_mxfp8_ladder.py"
B, HQ, HKV, D = 1, 32, 2, 128
LOG2E = math.log2(math.e)


def _env_from_rung(rung):
    for k, v in RUNGS[rung]["env"].items():
        os.environ[k] = v
    for k in ("LADDER_F16EXP", "LADDER_NOSCALE", "LADDER_PAGED64", "LADDER_CORRFAST", "LADDER_SDOUBLE", "LADDER_PSMEM"):
        os.environ.setdefault(k, "0")
    # One JIT / compiled-plan cache per rung: rungs C/D/E share ONE kernel file and differ only by
    # import-time env flags, which the DSL and FE plan caches do not key on.
    for var in ("CUTE_DSL_CACHE_DIR", "XDG_CACHE_HOME"):
        base = os.environ.get(var)
        if base:
            sub = "".join(f"_{k[10:].lower()}{os.environ[k]}" for k in ("LADDER_SD_PINGPONG", "LADDER_SD_PREFETCH", "LADDER_SD_LATE_ARRIVE", "LADDER_SD_SWMAX", "LADDER_CORR_EARLY", "LADDER_CORR_NORESCALE", "LADDER_SUB2", "LADDER_PIPE4", "LADDER_STFENCE", "LADDER_HOIST", "LADDER_PREF", "LADDER_SMREGS") if os.environ.get(k))
            if os.environ.get("LADDER_NOSCALE") == "1" and "LADDER_NOSCALE" not in RUNGS[rung]["env"]:
                sub += "_noscale1"
            os.environ[var] = f"{base.rstrip('/')}_rung{rung}{sub}"
            os.makedirs(os.environ[var], exist_ok=True)


def gpu_clocks():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv,noheader"], text=True
        )
        return out.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def bshd(x):
    """(B,H,S,D) view over BSHD storage (what the kernels want; avoids the adapter's gather copy)."""
    return x.permute(0, 2, 1, 3).contiguous().transpose(1, 2)


def ref_attention(q, k, v, scale, causal, blk=2048):
    """fp32 reference, GQA-aware, blocked over Q rows. q (B,HQ,S,D), k/v (B,HKV,S,D) fp32."""
    import torch

    Bn, Hq, S, Dd = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
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


def quantize_pertensor(x, fmax=448.0):
    import torch

    dq = (x.abs().amax().clamp_min(1e-8) / fmax).item()
    return (x / dq).to(torch.float8_e4m3fn), dq


def quantize_mx(x, b, h, s, d, *, columnwise):
    """MXFP8 quantize (B,H,S,D) fp32 -> (fp8 data (B,H,S,D), F8_128x4-swizzled SF, per-elem dequant)."""
    import torch
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(x, b, h, s, d, 32, torch.float8_e4m3fn, with_ref=True)
    if columnwise:
        return data_s, swz_s, dq_s.reshape(b, h, s, d)
    return data_d, swz_d, dq_d.reshape(b, h, s, d)


def build_case(rung, S, causal, seed=1234):
    """Returns dict with adapter kwargs, execute kwargs, and the fp32 reference inputs."""
    import torch

    dev = "cuda"
    torch.manual_seed(seed)
    spec = RUNGS[rung]
    scale = 1.0 / math.sqrt(D)
    qf = torch.randn(B, HQ, S, D, device=dev) * 0.5
    kf = torch.randn(B, HKV, S, D, device=dev) * 0.5
    vf = torch.randn(B, HKV, S, D, device=dev) * 0.5
    if os.environ.get("LADDER_KV_RAMP", "0") == "1":
        # Validation stress for the MXFP8 scale-factor plumbing: scale K and V per 32-token block by
        # 2^((t//32) % 4 - 2) so adjacent E8M0 blocks differ by 2x and the two 64-token halves of every
        # 128-token step differ by 4x.  Random inputs alone give near-identical scales and would hide a
        # wrong SF row/column mapping.  The reference sees the same (dequantized) inputs.
        t = torch.arange(S, device=dev)
        ramp = torch.pow(2.0, ((t // 32) % 4 - 2).float()).view(1, 1, S, 1)
        kf = kf * ramp
        vf = vf * ramp
    noscale = os.environ.get("LADDER_NOSCALE", "0") == "1"
    # Rung D/E: the caller folds attn_scale * log2(e) into Q; the kernel then runs exp2(S - m)
    # directly.  Reference: softmax_e(ln2 * Qpre K^T) == softmax_e(scale * Q K^T).
    q_in = qf * (scale * LOG2E) if noscale else qf
    ref_scale = math.log(2.0) if noscale else scale
    o = torch.empty(B, S, HQ, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    case = dict(rung=rung, S=S, causal=causal, scale=scale, o=o)
    if spec["family"] == "fp8":
        q8, dq = quantize_pertensor(q_in)
        k8, dk = quantize_pertensor(kf)
        v8, dv = quantize_pertensor(vf)
        qb, kb, vb = bshd(q8), bshd(k8), bshd(v8)
        t = lambda x: torch.tensor([x], dtype=torch.float32, device=dev)  # noqa: E731
        case.update(
            q=qb,
            k=kb,
            v=vb,
            adapter=dict(pertensor_fp8=True),
            exec_extra=dict(descale_q=t(dq), descale_k=t(dk), descale_v=t(dv), scale_o=t(1.0)),
            ref_inputs=(qb.float() * dq, kb.float() * dk, vb.float() * dv, ref_scale),
        )
    else:
        q8, sf_q, dqq = quantize_mx(q_in, B, HQ, S, D, columnwise=False)
        k8, sf_k, dqk = quantize_mx(kf, B, HKV, S, D, columnwise=False)
        v8, sf_v, dqv = quantize_mx(vf, B, HKV, S, D, columnwise=True)
        qb, kb, vb = bshd(q8), bshd(k8), bshd(v8)
        case.update(
            q=qb,
            k=kb,
            v=vb,
            adapter=dict(pertensor_fp8=False),
            exec_extra=dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v),
            ref_inputs=(qb.float() * dqq, kb.float() * dqk, vb.float() * dqv, ref_scale),
        )
        if spec.get("paged"):
            P = spec["paged"]
            # HND pools [num_pages, H_kv, P, D] filled from the dense K/V through a random table,
            # plus 7 dead pages.  SF stays DENSE (tile-indexed) in this one-off: a page-64 SF pool
            # cannot hold whole F8_128x4 atoms; the K/V payload paging is what the measurement is about.
            max_pages = S // P
            num_pages = B * max_pages + 7
            bt = torch.randperm(num_pages, device=dev)[: B * max_pages].to(torch.int32).view(B, max_pages).contiguous()
            k_pool = torch.zeros(num_pages, HKV, P, D, device=dev, dtype=torch.float8_e4m3fn)
            v_pool = torch.zeros(num_pages, HKV, P, D, device=dev, dtype=torch.float8_e4m3fn)
            kd = k8.view(B, HKV, max_pages, P, D).permute(0, 2, 1, 3, 4)  # (B, pages, H, P, D)
            vd = v8.view(B, HKV, max_pages, P, D).permute(0, 2, 1, 3, 4)
            k_pool[bt.view(-1).long()] = kd.reshape(B * max_pages, HKV, P, D)
            v_pool[bt.view(-1).long()] = vd.reshape(B * max_pages, HKV, P, D)
            # The adapter takes the [num_pages, H_kv, P, D] containers (HND/NHD from their strides).
            case.update(
                k=k_pool,
                v=v_pool,
                adapter=dict(pertensor_fp8=False, seq_kv_lens_present=True, paged_page_size=P, paged_max_seq_len_kv=S),
                paged=dict(bt=bt, seq_kv_lens=torch.full((B,), S, dtype=torch.int32, device=dev)),
            )
    return case


def make_api(case, sched="natural"):
    import cudnn
    from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    spec = RUNGS[case["rung"]]
    if spec["env"]:
        api_dsl._SM107_MXFP8_KERNEL_FILES[(128, 128)] = ONEOFF_FILE
    kw = dict(
        sample_q=case["q"],
        sample_k=case["k"],
        sample_v=case["v"],
        sample_o=case["o"],
        is_causal=case["causal"],
        scale_softmax=case["scale"],
        dtype_o=case["o"].dtype,
        has_amax_o=False,
        split_kv=1,
        sched_policy={"natural": SCHED_NATURAL, "lpt": SCHED_LPT, "lpt_l2": SCHED_LPT_L2}[sched],
    )
    kw.update(case["adapter"])
    if spec["family"] == "fp8":
        kw["softmax_precision"] = cudnn.data_type.FLOAT
    api = SdpaFwdDslSm100(**kw)
    if spec.get("paged"):
        # Bench-only bypass of two Rubin/MXFP8 paged gates (the one-off kernel implements the loader).
        orig = api._not_implemented_error_if

        def _lenient(cond, msg):
            if cond and msg and ("SM107 sibling" in msg or "multiple of 128" in msg or "F8_128x4 SF atoms" in msg):
                return
            return orig(cond, msg)

        api._not_implemented_error_if = _lenient
        # The prepared binder sizes K/V scale factors per page (page_size // 128 atoms per page); this
        # one-off keeps them DENSE, so present the plan as unpaged to that byte-count check only.
        from cudnn.sdpa.fwd import prepared as _prepared

        if not getattr(_prepared, "_ladder_patched", False):
            _orig_bind = _prepared._bind_mxfp8_scales

            class _DenseSFView:
                paged = False

                def __init__(self, spec):
                    self._spec = spec

                def __getattr__(self, name):
                    return getattr(self._spec, name)

            _prepared._bind_mxfp8_scales = lambda spec, facts: _orig_bind(_DenseSFView(spec), facts)
            _prepared._ladder_patched = True
    api.check_support()
    api.compile()
    return api


def run_execute(api, case, ws):
    kw = dict(workspace=ws) if ws is not None else {}
    kw.update(case["exec_extra"])
    if case.get("paged"):
        kw.update(seq_kv_lens=case["paged"]["seq_kv_lens"], block_table=case["paged"]["bt"], block_table_v=case["paged"]["bt"])
    api.execute(case["q"], case["k"], case["v"], case["o"], **kw)


def time_graph(api, case, ws, reps, rounds):
    import torch

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run_execute(api, case, ws)  # warm on the side stream
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            run_execute(api, case, ws)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3 / reps)
    return times


def time_eager(api, case, ws, reps, rounds):
    import torch

    times = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            run_execute(api, case, ws)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3 / reps)
    return times


def validate(case):
    import torch

    # Always against the dense dequantized inputs: the paged pools are built from them, so a
    # wrong table / pool layout shows up here as a mismatch.
    q, k, v, scale = case["ref_inputs"]
    ref = ref_attention(q, k, v, scale, case["causal"])
    out = case["o"].float()
    err = (out - ref).abs()
    finite = torch.isfinite(out).all().item()
    return dict(
        max_abs_err=err.max().item(),
        mean_abs_err=err.mean().item(),
        ref_amax=ref.abs().max().item(),
        rel_err=(err.max() / ref.abs().max().clamp_min(1e-12)).item(),
        finite=finite,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dump-o", default=None, help="save the validated O (first S) as a .pt for bitwise A/B between rungs")
    ap.add_argument("--seqlens", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--mask", choices=["causal", "none"], default="causal")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--validate", choices=["all", "first", "none"], default="first")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ncu-run", action="store_true", help="warmup launches then exactly ONE execute, then exit (for ncu)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sched", choices=["natural", "lpt", "lpt_l2"], default="natural", help="persistent-scheduler policy knob")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, nargs=2, default=[32, 2], metavar=("H_Q", "H_KV"))
    args = ap.parse_args()
    _env_from_rung(args.rung)
    global B, HQ, HKV
    B, HQ, HKV = args.batch, args.heads[0], args.heads[1]

    import torch

    causal = args.mask == "causal"
    cc = torch.cuda.get_device_capability()
    assert cc == (10, 7), f"this ladder targets Rubin cc 10.7; got {cc}"
    print(f"[ladder] rung {args.rung}: {RUNGS[args.rung]['label']} | env {RUNGS[args.rung]['env']} | {torch.cuda.get_device_name()} | clocks {gpu_clocks()}", flush=True)
    for S in args.seqlens:
        case = build_case(args.rung, S, causal)
        t0 = time.time()
        api = make_api(case, args.sched)
        ws_bytes = api.scratch_workspace_bytes()
        ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device="cuda") if ws_bytes else None
        compile_s = time.time() - t0
        kname = getattr(api, "kernel_template", "?")
        for _ in range(args.warmup):
            run_execute(api, case, ws)
        torch.cuda.synchronize()
        if args.ncu_run:
            run_execute(api, case, ws)
            torch.cuda.synchronize()
            print(f"[ladder] NCU_LAUNCH_DONE rung={args.rung} S={S} kernel={kname}", flush=True)
            continue
        rec = dict(
            rung=args.rung,
            label=RUNGS[args.rung]["label"],
            env=RUNGS[args.rung]["env"],
            S=S,
            mask=args.mask,
            sched=args.sched,
            B=B,
            h_q=HQ,
            h_kv=HKV,
            d=D,
            kernel=kname,
            cfg={k: getattr(getattr(api, "_k_mod", None).CFG, k, None) for k in ("SCHEDULER_POLICY", "CTA_MMA", "STAGES_KV", "TILE_M", "TILE_N", "TILES_Q", "RESCALE_THRESHOLD", "MASK_FLAGS", "PAGED_KV", "PAGE_SIZE")} if getattr(getattr(api, "_k_mod", None), "CFG", None) is not None else None,
            kmod=dict(PAGED_KV=getattr(getattr(api, "_k_mod", None), "PAGED_KV", None), PAGE_SIZE=getattr(getattr(api, "_k_mod", None), "PAGE_SIZE", None), F16EXP=getattr(getattr(api, "_k_mod", None), "LADDER_F16EXP", None), NOSCALE=getattr(getattr(api, "_k_mod", None), "LADDER_NOSCALE", None), CORRFAST=getattr(getattr(api, "_k_mod", None), "LADDER_CORRFAST", None), SDOUBLE=getattr(getattr(api, "_k_mod", None), "LADDER_SDOUBLE", None), SD_PINGPONG=getattr(getattr(api, "_k_mod", None), "LADDER_SD_PINGPONG", None), SD_PREFETCH=getattr(getattr(api, "_k_mod", None), "LADDER_SD_PREFETCH", None), SD_LATE_ARRIVE=getattr(getattr(api, "_k_mod", None), "LADDER_SD_LATE_ARRIVE", None), PSMEM=getattr(getattr(api, "_k_mod", None), "LADDER_PSMEM", None), HOIST=getattr(getattr(api, "_k_mod", None), "LADDER_HOIST", None), PREF=getattr(getattr(api, "_k_mod", None), "LADDER_PREF", None), SMREGS=getattr(getattr(api, "_k_mod", None), "LADDER_SMREGS", None)),
            compile_s=round(compile_s, 1),
            clocks_before=gpu_clocks(),
            tag=args.tag,
            host=os.uname().nodename,
        )
        try:
            tg = time_graph(api, case, ws, args.reps, args.rounds)
            rec["time_us_graph"] = round(statistics.median(tg), 2)
            rec["time_us_graph_min"] = round(min(tg), 2)
        except Exception as e:  # noqa: BLE001
            rec["graph_error"] = f"{type(e).__name__}: {e}"[:300]
        te = time_eager(api, case, ws, args.reps, max(2, args.rounds // 2))
        rec["time_us_eager"] = round(statistics.median(te), 2)
        t_us = rec.get("time_us_graph", rec["time_us_eager"])
        flops = 4.0 * B * HQ * S * S * D * (0.5 * (1 + 1.0 / S) if causal else 1.0)
        rec["tflops"] = round(flops / (t_us * 1e-6) / 1e12, 1)
        rec["clocks_after"] = gpu_clocks()
        if args.validate == "all" or (args.validate == "first" and S == args.seqlens[0]):
            run_execute(api, case, ws)
            torch.cuda.synchronize()
            rec["validation"] = validate(case)
            if args.dump_o:
                torch.save(case["o"].detach().clone().cpu(), args.dump_o)
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        del api, case, ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
