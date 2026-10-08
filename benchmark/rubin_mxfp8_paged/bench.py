#!/usr/bin/env python3
"""cc 10.7 MXFP8 Q/K/V GQA prefill benchmark -- FROST SDPA forward kernels through the direct adapter.

Shape (plan): B=1, H_q=32, H_kv=8 (GQA 4:1), d in {128, 256}, S_q=S_kv in {8k, 16k, 32k}, Q/K/V MXFP8
(e4m3 + E8M0 block scales per 32), O bf16, no Stats, no Amax_O (inference), masks none / causal.

Trick stack (all on by default): f16 softmax (softmax_precision=HALF), 1/ln2 folded outside the kernel
(softmax_scale_prefolded=True: Q is pre-multiplied by attn_scale*log2(e) before quantization, the kernel
runs exp2(S - m) with no per-score multiply; HALF + prefolded = fused FHADD2 shift+convert arm), and the
bench-only levers of the `prefill_d{128,256}_mxfp8_bench.py` kernels (env at import):
    BENCH_PAGED64=0|1   paged K/V cache, page size 64 (HND pools + random block table)
    BENCH_CORR=default|always|never   correction: product RESCALE_THRESHOLD / rescale every step / never
    BENCH_CORRFAST=0|1  correction fast path (alpha via SMEM slab)   BENCH_HOIST=0|1  TMEM-base hoist
    BENCH_ROWSUM_MMA=0|1  d256 stretch: ones-MMA row sum
The product kernels (`prefill_d{128,256}_mxfp8.py`) serve corr=default, paged=0 only.

One CONFIG = one worker process (the levers are read at kernel import and the DSL / compiled-plan caches
do not key on them), each with its own CUTE_DSL_CACHE_DIR + XDG_CACHE_HOME suffix derived from the kernel
file and every BENCH_* value.  The coordinator interleaves the configs of a cell round-robin: `--reps`
graph replays per burst, `--cooldown-ms` idle after every burst, `--rounds` rounds -- the board throttles
2.4 -> 1.8 GHz under sustained load, so sequential per-config timing measures the thermal state.

Usage (inside the board env wrapper, see board_py_0614.sh):
  bench.py --d 128 --seqlens 8192 16384 --mask none --configs corr=default corr=always corr=never \
           --rounds 15 --reps 4 --cooldown-ms 150 --validate first --out results.jsonl
  bench.py --d 256 --seqlens 16384 --mask causal --config corr=default,paged=0,kernel=product --ncu-run
Config string keys (comma separated key=val): corr, paged (0|64), half (0|1), prefolded (0|1),
kernel (product|bench|auto), corrfast (0|1|auto), hoist (0|1|auto), rowsum_mma (0|1), cga (1|2|auto),
sched (natural|lpt|lpt_l2), sf (dense|paged: scale-factor layout the paged bench kernel expects), name.
"""
import argparse
import json
import math
import os
import select
import statistics
import subprocess
import sys
import time

LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
PRODUCT_KERNEL_FILES = {128: "sm107/prefill_d128_mxfp8.py", 256: "sm107/prefill_d256_mxfp8.py"}
BENCH_KERNEL_FILES = {128: "sm107/prefill_d128_mxfp8_bench.py", 256: "sm107/prefill_d256_mxfp8_bench.py"}
LEVER_ENV = ("BENCH_PAGED64", "BENCH_CORR", "BENCH_CORRFAST", "BENCH_HOIST", "BENCH_ROWSUM_MMA", "BENCH_KV_RAMP")
CFG_FIELDS = ("TILE_M", "TILE_N", "TILE_K", "TILE_O", "TILES_Q", "CTA_MMA", "STAGES_KV", "RESCALE_THRESHOLD", "SCHEDULER_POLICY", "MASK_FLAGS", "PAGED_KV", "PAGE_SIZE", "SOFTMAX_WARPGROUPS", "CORRECTION_WARPS", "SOFTMAX_REGS", "CORRECTION_REGS", "TOTAL_WARPS")
KMOD_FLAGS = ("SOFTMAX_F16", "SCALE_PREFOLDED", "_FUSED_SHIFT_CVT", "PAGED_KV", "PAGE_SIZE", "BENCH_PAGED64", "BENCH_CORR", "BENCH_CORRFAST", "BENCH_HOIST", "BENCH_ROWSUM_MMA", "LADDER_F16EXP", "LADDER_NOSCALE", "LADDER_FCVT")
BYPASS_MSGS = ("SM107 sibling", "multiple of 128", "F8_128x4 SF atoms", "not wired in the paged-KV", "paged KV", "page size")


# ----------------------------------------------------------------------------------------------------
# config strings
# ----------------------------------------------------------------------------------------------------
def parse_config(s, d, defaults):
    """'corr=always,paged=64' -> dict with every key resolved (auto -> concrete where possible)."""
    c = dict(corr="default", paged=0, half=1, prefolded=1, kernel="auto", corrfast="auto", hoist="auto", rowsum_mma=0, cga="auto", sched=defaults.get("sched", "natural"), sf="auto", name=None)
    for kv in filter(None, (s or "").split(",")):
        k, _, v = kv.partition("=")
        k = k.strip().lower()
        if k not in c:
            raise SystemExit(f"unknown config key {k!r} in {s!r}")
        c[k] = v.strip()
    for k in ("paged", "half", "prefolded", "rowsum_mma"):
        c[k] = int(c[k])
    if c["paged"] not in (0, 64):
        raise SystemExit("paged must be 0 or 64")
    if c["corr"] not in ("default", "always", "never"):
        raise SystemExit("corr must be default|always|never")
    needs_bench = c["corr"] != "default" or c["paged"] != 0 or c["rowsum_mma"] or c["corrfast"] in ("1",) or c["hoist"] in ("1",)
    if c["kernel"] == "auto":
        c["kernel"] = "bench" if needs_bench else "product"
    if c["kernel"] == "product" and needs_bench:
        raise SystemExit(f"config {s!r} needs the bench kernel (corr/paged/rowsum_mma/corrfast/hoist levers)")
    for k in ("corrfast", "hoist"):
        if c[k] == "auto":
            c[k] = 1 if c["kernel"] == "bench" else 0
        c[k] = int(c[k])
    if c["cga"] != "auto":
        c["cga"] = int(c["cga"])
    if c["sf"] == "auto":
        # the d128 bench loader keeps the K/V scale factors dense (tile-indexed); the d256 one pages them
        c["sf"] = "dense" if d == 128 else "paged"
    if not c["name"]:
        parts = [f"corr{c['corr']}", f"p{c['paged']}", "half" if c["half"] else "f32", "pf" if c["prefolded"] else "scale", c["kernel"][:4]]
        if c["kernel"] == "bench":
            parts.append(f"cf{c['corrfast']}h{c['hoist']}" + (f"rs{c['rowsum_mma']}" if c["rowsum_mma"] else ""))
        if c["cga"] != "auto":
            parts.append(f"cga{c['cga']}")
        c["name"] = "_".join(parts)
    return c


def lever_env(c, kv_ramp):
    return {
        "BENCH_PAGED64": "1" if c["paged"] == 64 else "0",
        "BENCH_CORR": c["corr"],
        "BENCH_CORRFAST": str(c["corrfast"]),
        "BENCH_HOIST": str(c["hoist"]),
        "BENCH_ROWSUM_MMA": str(c["rowsum_mma"]),
        "BENCH_KV_RAMP": "1" if kv_ramp else "0",
    }


def cache_suffix(c, d, env):
    kfile = os.path.splitext(os.path.basename((BENCH_KERNEL_FILES if c["kernel"] == "bench" else PRODUCT_KERNEL_FILES)[d]))[0]
    levers = "_".join(f"{k[6:].lower()}{env[k]}" for k in LEVER_ENV if k != "BENCH_KV_RAMP")
    return f"_{kfile}_{levers}_half{c['half']}pf{c['prefolded']}"


def apply_env(c, d, kv_ramp):
    """Set the lever env + per-config cache dirs in THIS process (worker / --ncu-run), before importing cudnn."""
    env = lever_env(c, kv_ramp)
    os.environ.update(env)
    suf = cache_suffix(c, d, env)
    for var in ("CUTE_DSL_CACHE_DIR", "XDG_CACHE_HOME"):
        base = os.environ.get(var)
        if base:
            os.environ[var] = base.rstrip("/") + suf
            os.makedirs(os.environ[var], exist_ok=True)
    return env


# ----------------------------------------------------------------------------------------------------
# board state
# ----------------------------------------------------------------------------------------------------
def gpu_clocks():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv,noheader"], text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def clocks_mhz(s):
    try:
        return float(s.split(",")[0].split()[0])
    except Exception:  # noqa: BLE001
        return None


def power_w(s):
    try:
        return float(s.split(",")[3].split()[0])
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------
def bshd(x):
    """(B,H,S,D) view over BSHD storage (what the kernels address natively)."""
    return x.permute(0, 2, 1, 3).contiguous().transpose(1, 2)


def quantize_mx(x, b, h, s, d, *, columnwise):
    """MXFP8 quantize (B,H,S,D) fp32 -> (fp8 data (B,H,S,D), F8_128x4-swizzled SF, per-element dequant (B,H,S,D))."""
    import torch
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, torch.float8_e4m3fn, with_ref=True)
    if columnwise:
        return data_s, swz_s.contiguous(), dq_s.reshape(b, h, s, d)
    return data_d, swz_d.contiguous(), dq_d.reshape(b, h, s, d)


def build_case(c, d, S, causal, B, HQ, HKV, kv_ramp, seed=1234, keep_ref=True):
    import torch

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    scale = 1.0 / math.sqrt(d)
    qf = torch.randn(B, HQ, S, d, device=dev, generator=g) * 0.5
    kf = torch.randn(B, HKV, S, d, device=dev, generator=g) * 0.5
    vf = torch.randn(B, HKV, S, d, device=dev, generator=g) * 0.5
    if kv_ramp:
        # Scale K and V per 32-token block by 2^((t//32) % 4 - 2): adjacent E8M0 blocks differ 2x and the two
        # 64-token halves of a 128-token step 4x, so a wrong SF row/column/page mapping shows in the error.
        t = torch.arange(S, device=dev)
        ramp = torch.pow(2.0, ((t // 32) % 4 - 2).float()).view(1, 1, S, 1)
        kf = kf * ramp
        vf = vf * ramp
    prefolded = bool(c["prefolded"])
    q_in = qf * (scale * LOG2E) if prefolded else qf
    ref_scale = LN2 if prefolded else scale
    q8, sf_q, dqq = quantize_mx(q_in, B, HQ, S, d, columnwise=False)
    k8, sf_k, dqk = quantize_mx(kf, B, HKV, S, d, columnwise=False)
    v8, sf_v, dqv = quantize_mx(vf, B, HKV, S, d, columnwise=True)
    del qf, kf, vf, q_in
    qb, kb, vb = bshd(q8), bshd(k8), bshd(v8)
    o = torch.empty(B, S, HQ, d, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    case = dict(d=d, S=S, causal=causal, scale=scale, ref_scale=ref_scale, o=o, q=qb, k=kb, v=vb, B=B, HQ=HQ, HKV=HKV, adapter=dict(pertensor_fp8=False), exec_extra=dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v))
    if keep_ref:
        case["ref_inputs"] = (qb, dqq, kb, dqk, vb, dqv)  # dequantize lazily: data.float() * dq
    if c["paged"]:
        P = c["paged"]
        max_pages = S // P
        num_pages = B * max_pages + 7  # 7 dead pages
        bt = torch.randperm(num_pages, device=dev, generator=g)[: B * max_pages].to(torch.int32).view(B, max_pages).contiguous()
        k_pool = torch.zeros(num_pages, HKV, P, d, device=dev, dtype=torch.float8_e4m3fn)  # HND pools
        v_pool = torch.zeros(num_pages, HKV, P, d, device=dev, dtype=torch.float8_e4m3fn)
        kd = k8.view(B, HKV, max_pages, P, d).permute(0, 2, 1, 3, 4)  # (B, pages, H, P, D)
        vd = v8.view(B, HKV, max_pages, P, d).permute(0, 2, 1, 3, 4)
        k_pool[bt.view(-1).long()] = kd.reshape(B * max_pages, HKV, P, d)
        v_pool[bt.view(-1).long()] = vd.reshape(B * max_pages, HKV, P, d)
        case.update(k=k_pool, v=v_pool, adapter=dict(pertensor_fp8=False, seq_kv_lens_present=True, paged_page_size=P, paged_max_seq_len_kv=S), paged=dict(bt=bt, seq_kv_lens=torch.full((B,), S, dtype=torch.int32, device=dev)))
        if c["sf"] == "paged":
            case["exec_extra"].update(build_paged_sf(sf_k, sf_v, bt, B, HKV, S, d, P))
    return case


def build_paged_sf(sf_k, sf_v, bt, B, HKV, S, d, P):
    """Scale factors for a bench kernel that reads K/V SF per page (d256 loader).

    TODO(K2 report): the per-page SF pool layout is the kernel's contract; until the d256 bench kernel lands
    this returns the dense (tile-indexed) tensors unchanged, which is also what the d128 loader expects.
    """
    return dict(sf_k=sf_k, sf_v=sf_v)


# ----------------------------------------------------------------------------------------------------
# adapter
# ----------------------------------------------------------------------------------------------------
def make_api(case, c):
    import cudnn
    from cudnn.frost.tile_dsl.scheduler import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    d = case["d"]
    if c["kernel"] == "bench":
        api_dsl._SM107_MXFP8_KERNEL_FILES[(d, d)] = BENCH_KERNEL_FILES[d]
    kw = dict(
        sample_q=case["q"],
        sample_k=case["k"],
        sample_v=case["v"],
        sample_o=case["o"],
        is_causal=case["causal"],
        scale_softmax=None if c["prefolded"] else case["scale"],
        dtype_o=case["o"].dtype,
        has_amax_o=False,
        split_kv=1,
        sched_policy={"natural": SCHED_NATURAL, "lpt": SCHED_LPT, "lpt_l2": SCHED_LPT_L2}[c["sched"]],
        softmax_precision=cudnn.data_type.HALF if c["half"] else None,
        softmax_scale_prefolded=bool(c["prefolded"]),
    )
    if c["cga"] != "auto":
        kw["cga"] = c["cga"]
    kw.update(case["adapter"])
    api = SdpaFwdDslSm100(**kw)
    if c["paged"]:
        # Bench-only bypass of the cc 10.7 MXFP8 paged gates (the bench kernel implements the page-64 loader).
        orig = api._not_implemented_error_if

        def _lenient(cond, msg):
            if cond and msg and any(s in msg for s in BYPASS_MSGS):
                return
            return orig(cond, msg)

        api._not_implemented_error_if = _lenient
        if c["sf"] == "dense":
            # The prepared binder sizes K/V scale factors per page; the d128 loader keeps them DENSE, so present
            # the plan as unpaged to that byte-count check only.
            from cudnn.sdpa.fwd import prepared as _prepared

            if not getattr(_prepared, "_bench_dense_sf_patched", False):
                _orig_bind = _prepared._bind_mxfp8_scales

                class _DenseSFView:
                    paged = False

                    def __init__(self, spec):
                        self._spec = spec

                    def __getattr__(self, name):
                        return getattr(self._spec, name)

                _prepared._bind_mxfp8_scales = lambda spec, facts: _orig_bind(_DenseSFView(spec), facts)
                _prepared._bench_dense_sf_patched = True
    try:
        api.check_support()
    except NotImplementedError as e:
        if not (c["paged"] and any(s in str(e) for s in BYPASS_MSGS)):
            raise
        print(f"[bench] bypassing paged gate: {e}", file=sys.stderr, flush=True)
    api.compile()
    return api


def run_execute(api, case, ws):
    kw = dict(workspace=ws) if ws is not None else {}
    kw.update(case["exec_extra"])
    if case.get("paged"):
        kw.update(seq_kv_lens=case["paged"]["seq_kv_lens"], block_table=case["paged"]["bt"], block_table_v=case["paged"]["bt"])
    api.execute(case["q"], case["k"], case["v"], case["o"], **kw)


def capture_graph(api, case, ws):
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
    return g


def burst(g, reps):
    import torch

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / reps


def api_meta(api, c):
    km = getattr(api, "_k_mod", None)
    cfg = getattr(km, "CFG", None)
    return dict(
        kernel=getattr(api, "kernel_template", "?"),
        kernel_file=os.path.basename(getattr(km, "__file__", "?") or "?"),
        cfg={k: getattr(cfg, k, None) for k in CFG_FIELDS} if cfg is not None else None,
        kmod={k: getattr(km, k, None) for k in KMOD_FLAGS if hasattr(km, k)},
        cta_mma=getattr(cfg, "CTA_MMA", None) if cfg is not None else None,
    )


# ----------------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------------
def ref_attention(q, k, v, scale, causal, blk=1024):
    """fp32 reference, GQA by head repeat, blocked over query rows (no S x S materialization per head)."""
    import torch

    b, hq, s, _ = q.shape
    rep = hq // k.shape[1]
    out = torch.empty(b, hq, s, v.shape[-1], device=q.device, dtype=torch.float32)
    for h in range(hq):
        kh = k[:, h // rep]
        vh = v[:, h // rep]
        kt = kh.transpose(-1, -2).contiguous()
        for r0 in range(0, s, blk):
            r1 = min(s, r0 + blk)
            sc = torch.matmul(q[:, h, r0:r1], kt) * scale
            if causal:
                rows = torch.arange(r0, r1, device=q.device)[:, None]
                cols = torch.arange(s, device=q.device)[None, :]
                sc = sc.masked_fill(cols > rows, float("-inf"))
            out[:, h, r0:r1] = torch.matmul(torch.softmax(sc, dim=-1), vh)
    return out


def validate(case):
    import torch

    qb, dqq, kb, dqk, vb, dqv = case["ref_inputs"]
    q = qb.float() * dqq
    k = kb.float() * dqk
    v = vb.float() * dqv
    ref = ref_attention(q, k, v, case["ref_scale"], case["causal"])
    out = case["o"].float()
    err = (out - ref).abs()
    res = dict(
        max_abs_err=err.max().item(),
        mean_abs_err=err.mean().item(),
        ref_amax=ref.abs().max().item(),
        rel_err=(err.max() / ref.abs().max().clamp_min(1e-12)).item(),
        rms_rel=(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item(),
        finite=bool(torch.isfinite(out).all().item()),
    )
    del q, k, v, ref, out, err
    return res


def rescale_stats(case, threshold, tile=128):
    """Emulate the kernel's lazy-threshold rescale rule on the actual logits (log2 units): a correction warp
    (32 consecutive rows of a 128-row sub-tile) rescales on a step iff ANY of its rows raises its running max
    by more than `threshold` on a step other than its first live step.  corr=never is exact iff this is 0."""
    import torch

    qb, dqq, kb, dqk, _, _ = case["ref_inputs"]
    q = qb.float() * dqq
    k = kb.float() * dqk
    scale_log2 = case["ref_scale"] * LOG2E
    B, H, S, _ = q.shape
    grp = H // k.shape[1]
    causal = case["causal"]
    n_tiles = S // tile
    resc = live_total = 0
    for b in range(B):
        for h in range(H):
            qh = q[b, h]
            kh = k[b, h // grp]
            m = torch.full((S,), float("-inf"), device=q.device)
            first = torch.ones((S,), dtype=torch.bool, device=q.device)
            rows = torch.arange(S, device=q.device)[:, None]
            for t in range(n_tiles):
                s_blk = (qh @ kh[t * tile : (t + 1) * tile].T) * scale_log2
                if causal:
                    cols = (t * tile + torch.arange(tile, device=q.device))[None, :]
                    s_blk = s_blk.masked_fill(cols > rows, float("-inf"))
                cur = s_blk.amax(dim=1)
                live = torch.isfinite(cur)
                upd = live & (first | ((cur - m) > threshold))
                m = torch.where(upd, cur, m)
                resc_rows = upd & ~first & live
                first = first & ~live
                live_total += int(live.view(-1, 32).any(dim=1).sum())
                resc += int(resc_rows.view(-1, 32).any(dim=1).sum())
    return dict(threshold=threshold, warp_steps_rescaled=resc, warp_steps_live=live_total, pct=100.0 * resc / max(live_total, 1))


# ----------------------------------------------------------------------------------------------------
# worker process (one config)
# ----------------------------------------------------------------------------------------------------
def worker_main(args):
    c = parse_config(args.config, args.d, dict(sched=args.sched))
    env = apply_env(c, args.d, args.kv_ramp)
    import torch

    cc = torch.cuda.get_device_capability()
    assert cc == (10, 7), f"this bench targets cc 10.7; got {cc}"
    props = torch.cuda.get_device_properties(0)
    state = {}

    def say(tag, payload):
        print(f"{tag} {json.dumps(payload)}", flush=True)

    say("HELLO", dict(config=c, env=env, device=torch.cuda.get_device_name(), sm_count=props.multi_processor_count, cache_dirs={k: os.environ.get(k) for k in ("CUTE_DSL_CACHE_DIR", "XDG_CACHE_HOME")}))
    for line in sys.stdin:
        cmd, _, arg = line.strip().partition(" ")
        try:
            if cmd == "BUILD":
                spec = json.loads(arg)
                state.clear()
                torch.cuda.empty_cache()
                t0 = time.time()
                case = build_case(c, args.d, spec["S"], spec["causal"], args.batch, args.heads[0], args.heads[1], args.kv_ramp, seed=args.seed, keep_ref=spec.get("keep_ref", True))
                t1 = time.time()
                api = make_api(case, c)
                ws_bytes = api.scratch_workspace_bytes()
                ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device="cuda") if ws_bytes else None
                compile_s = time.time() - t1
                for _ in range(spec.get("warmup", 3)):
                    run_execute(api, case, ws)
                torch.cuda.synchronize()
                g = capture_graph(api, case, ws)
                state.update(case=case, api=api, ws=ws, g=g)
                meta = api_meta(api, c)
                meta.update(compile_s=round(compile_s, 1), inputs_s=round(t1 - t0, 1), workspace_bytes=ws_bytes, sm_count=props.multi_processor_count)
                say("READY", meta)
            elif cmd == "BURST":
                say("T", dict(us=burst(state["g"], int(arg))))
            elif cmd == "VALIDATE":
                run_execute(state["api"], state["case"], state["ws"])
                torch.cuda.synchronize()
                say("V", validate(state["case"]))
            elif cmd == "RESCALE":
                thr = float(arg) if arg else float(state["api"]._k_mod.CFG.RESCALE_THRESHOLD)
                say("R", rescale_stats(state["case"], thr))
            elif cmd == "DUMP_O":
                torch.save(state["case"]["o"].detach().clone().cpu(), arg)
                say("D", dict(path=arg))
            elif cmd == "QUIT":
                say("BYE", {})
                return 0
            else:
                say("ERR", dict(error=f"unknown command {cmd}"))
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            say("ERR", dict(error=f"{type(e).__name__}: {e}"[:600]))
    return 0


# ----------------------------------------------------------------------------------------------------
# coordinator
# ----------------------------------------------------------------------------------------------------
class Worker:
    def __init__(self, name, cmd, env, log_path):
        self.name = name
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        self.hello = self.expect("HELLO", timeout=600)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def expect(self, tag, timeout):
        """Read lines until one starts with `tag` (or ERR); raise on timeout / worker death."""
        deadline = time.time() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker {self.name} exited with {self.proc.returncode} (see its log)")
            r, _, _ = select.select([self.proc.stdout], [], [], min(5.0, max(0.0, deadline - time.time())))
            if not r:
                if time.time() > deadline:
                    raise RuntimeError(f"worker {self.name}: timeout waiting for {tag}")
                continue
            line = self.proc.stdout.readline()
            if not line:
                continue
            t, _, payload = line.rstrip("\n").partition(" ")
            if t == tag:
                return json.loads(payload) if payload else {}
            if t == "ERR":
                raise RuntimeError(f"worker {self.name}: {json.loads(payload).get('error')}")
            print(f"[{self.name}] {line.rstrip()}", flush=True)

    def close(self):
        try:
            self.send("QUIT")
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self.log.close()


def coordinator_main(args):
    configs = [parse_config(s, args.d, dict(sched=args.sched)) for s in (args.configs or [args.config])]
    names = [c["name"] for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate config names: {names}")
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[bench] d={args.d} heads {args.heads} B={args.batch} mask={args.mask} seqlens={args.seqlens} configs={names} | rounds={args.rounds} x reps={args.reps}, cooldown {args.cooldown_ms} ms | clocks {gpu_clocks()}", flush=True)
    workers = []
    for c in configs:
        env = dict(os.environ)
        env.update(lever_env(c, args.kv_ramp))
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--config", config_to_string(c), "--d", str(args.d), "--batch", str(args.batch), "--heads", str(args.heads[0]), str(args.heads[1]), "--sched", args.sched, "--seed", str(args.seed)]
        if args.kv_ramp:
            cmd.append("--kv-ramp")
        workers.append(Worker(c["name"], cmd, env, os.path.join(args.log_dir, f"worker_{c['name']}.log")))
        print(f"[bench] worker {c['name']}: {workers[-1].hello.get('cache_dirs')}", flush=True)
    causal = args.mask == "causal"
    try:
        for S in args.seqlens:
            keep_ref = args.validate == "all" or (args.validate == "first" and S == args.seqlens[0]) or args.rescale_stats
            metas = {}
            for w in workers:
                w.send("BUILD " + json.dumps(dict(S=S, causal=causal, keep_ref=keep_ref, warmup=args.warmup)))
            for w in workers:
                metas[w.name] = w.expect("READY", timeout=args.build_timeout)
                print(f"[bench] {w.name} S={S}: {metas[w.name]['kernel_file']} cta_mma={metas[w.name]['cta_mma']} compile {metas[w.name]['compile_s']} s kmod={metas[w.name]['kmod']}", flush=True)
            samples = {w.name: [] for w in workers}
            clocks = [gpu_clocks()]
            t_start = time.time()
            for r in range(args.rounds):
                for w in workers:
                    samples[w.name].append(w.expect_after("BURST %d" % args.reps, "T", 600)["us"] if hasattr(w, "expect_after") else _burst(w, args.reps))
                    if args.cooldown_ms:
                        time.sleep(args.cooldown_ms / 1000.0)
                if r % max(1, args.rounds // 5) == 0 or r == args.rounds - 1:
                    clocks.append(gpu_clocks())
            t_timing = time.time() - t_start
            for w, c in zip(workers, configs):
                ts = samples[w.name]
                med = statistics.median(ts)
                flops = 4.0 * args.batch * args.heads[0] * S * S * args.d * (0.5 * (1 + 1.0 / S) if causal else 1.0)
                rec = dict(
                    name=c["name"],
                    config=c,
                    env=lever_env(c, args.kv_ramp),
                    d=args.d,
                    S=S,
                    mask=args.mask,
                    sched=c["sched"],
                    B=args.batch,
                    h_q=args.heads[0],
                    h_kv=args.heads[1],
                    kv_ramp=bool(args.kv_ramp),
                    rounds=args.rounds,
                    reps=args.reps,
                    cooldown_ms=args.cooldown_ms,
                    time_us_graph=round(med, 2),
                    time_us_min=round(min(ts), 2),
                    time_us_mean=round(statistics.fmean(ts), 2),
                    spread_pct=round(100.0 * (max(ts) - min(ts)) / med, 1),
                    time_us_rounds=[round(t, 1) for t in ts],
                    tflops=round(flops / (med * 1e-6) / 1e12, 1),
                    tflops_min_time=round(flops / (min(ts) * 1e-6) / 1e12, 1),
                    clocks=clocks,
                    clock_mhz_median=statistics.median([x for x in map(clocks_mhz, clocks) if x is not None] or [0]),
                    power_w_median=statistics.median([x for x in map(power_w, clocks) if x is not None] or [0]),
                    timing_wall_s=round(t_timing, 1),
                    tag=args.tag,
                    host=os.uname().nodename,
                )
                rec.update(metas[w.name])
                if c["corr"] == "never":
                    rec["timing_only"] = True  # exact only when no step rescales -- see rescale_stats below
                if keep_ref and args.validate != "none":
                    w.send("VALIDATE")
                    rec["validation"] = w.expect("V", timeout=3600)
                if keep_ref and (args.rescale_stats or c["corr"] == "never"):
                    thr = metas[w.name]["cfg"]["RESCALE_THRESHOLD"] if metas[w.name].get("cfg") else 4.0
                    w.send(f"RESCALE {thr}")
                    rec["rescale_stats"] = w.expect("R", timeout=3600)
                    if c["corr"] == "never":
                        rec["timing_only"] = rec["rescale_stats"]["warp_steps_rescaled"] != 0
                if args.dump_o:
                    path = f"{args.dump_o}_{c['name']}_S{S}_{args.mask}.pt"
                    w.send(f"DUMP_O {path}")
                    w.expect("D", timeout=600)
                    rec["dump_o"] = path
                print(json.dumps(rec), flush=True)
                if args.out:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            summary = " | ".join(f"{n} {statistics.median(samples[n]):.1f} us" for n in names)
            print(f"[bench] S={S} {args.mask}: {summary} | clocks {clocks[-1]}", flush=True)
    finally:
        for w in workers:
            w.close()
    return 0


def _burst(w, reps):
    w.send(f"BURST {reps}")
    return w.expect("T", timeout=600)["us"]


def config_to_string(c):
    return ",".join(f"{k}={v}" for k, v in c.items() if v is not None)


# ----------------------------------------------------------------------------------------------------
# single-launch mode for ncu / APIC (in-process, one config)
# ----------------------------------------------------------------------------------------------------
def ncu_run_main(args):
    c = parse_config(args.configs[0] if args.configs else args.config, args.d, dict(sched=args.sched))
    apply_env(c, args.d, args.kv_ramp)
    import torch

    assert torch.cuda.get_device_capability() == (10, 7)
    causal = args.mask == "causal"
    for S in args.seqlens:
        case = build_case(c, args.d, S, causal, args.batch, args.heads[0], args.heads[1], args.kv_ramp, seed=args.seed, keep_ref=args.validate != "none")
        api = make_api(case, c)
        ws_bytes = api.scratch_workspace_bytes()
        ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device="cuda") if ws_bytes else None
        for _ in range(args.warmup):
            run_execute(api, case, ws)
        torch.cuda.synchronize()
        run_execute(api, case, ws)  # THE profiled / captured launch (launch index = warmup)
        torch.cuda.synchronize()
        meta = api_meta(api, c)
        if args.validate != "none":
            meta["validation"] = validate(case)
        print(f"[bench] NCU_LAUNCH_DONE d={args.d} S={S} mask={args.mask} config={c['name']} " + json.dumps(meta), flush=True)
        del api, case, ws
        torch.cuda.empty_cache()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, choices=[128, 256], required=True)
    ap.add_argument("--corr", choices=["default", "always", "never"], default=None, help="shorthand for the single config (see --config)")
    ap.add_argument("--paged", type=int, choices=[0, 64], default=None)
    ap.add_argument("--kernel", choices=["product", "bench", "auto"], default=None)
    ap.add_argument("--config", default="", help="one config string key=val,... (default: trick stack on the product kernel: corr=default,paged=0,half=1,prefolded=1)")
    ap.add_argument("--configs", nargs="*", default=None, help="several config strings, interleaved round-robin in one cell")
    ap.add_argument("--mask", choices=["none", "causal"], default="none")
    ap.add_argument("--seqlens", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--heads", type=int, nargs=2, default=[32, 8], metavar=("H_Q", "H_KV"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sched", choices=["natural", "lpt", "lpt_l2"], default="natural")
    ap.add_argument("--reps", type=int, default=4, help="graph replays per burst")
    ap.add_argument("--rounds", type=int, default=15, help="round-robin rounds over the configs")
    ap.add_argument("--cooldown-ms", type=int, default=150, help="idle gap after every burst")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--validate", choices=["all", "first", "none"], default="first")
    ap.add_argument("--rescale-stats", action="store_true", help="emulate the rescale rule on the inputs for every config (always on for corr=never)")
    ap.add_argument("--kv-ramp", action="store_true", help="K/V x 2^((t//32)%%4-2): scale-factor mapping stress (validation only)")
    ap.add_argument("--dump-o", default=None, help="path prefix: save each config's O (.pt) for bitwise A/B")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--log-dir", default=os.environ.get("BENCH_LOG_DIR", "/tmp/vagarwalla/mxb/logs"))
    ap.add_argument("--build-timeout", type=int, default=3600)
    ap.add_argument("--ncu-run", action="store_true", help="in-process: warmups then exactly ONE execute (for ncu / APIC), no coordinator")
    ap.add_argument("--tag", default="")
    ap.add_argument("--print-name", action="store_true", help="print the canonical name of each config and exit (file naming in the shell drivers)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.corr or args.paged is not None or args.kernel:
        extra = [f"corr={args.corr}" if args.corr else "", f"paged={args.paged}" if args.paged is not None else "", f"kernel={args.kernel}" if args.kernel else ""]
        args.config = ",".join(filter(None, [args.config, *extra]))
    if args.print_name:
        for s in args.configs or [args.config]:
            print(parse_config(s, args.d, dict(sched=args.sched))["name"])
        return 0
    if args.worker:
        return worker_main(args)
    if args.ncu_run:
        return ncu_run_main(args)
    return coordinator_main(args)


if __name__ == "__main__":
    sys.exit(main())
