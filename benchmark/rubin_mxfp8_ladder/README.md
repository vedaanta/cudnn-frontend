# Rubin MXFP8 prefill ladder (FROST sm107 SDPA, bench-only)

Measures a ladder of prefill-attention configurations on Rubin GR100 (cc 10.7) with the FROST
CuTe-DSL SDPA kernels, one lever at a time:

| rung | configuration | kernel |
|---|---|---|
| A | per-tensor FP8 (e4m3) in, fp32 softmax | `sm107/prefill_d128_fp8.py` (product) |
| B | MXFP8 in (e4m3 + E8M0 per 32), fp32 softmax | `sm107/prefill_d128_mxfp8.py` (product) |
| C | B + f16 exponent (`MUFU.EX2.F16x2`, f16x2 -> fp8 P cast) | `sm107/prefill_d128_mxfp8_ladder.py`, `LADDER_F16EXP=1` |
| D | C + softmax scale folded outside the kernel (caller pre-scales Q by `attn_scale*log2e`; kernel runs `exp2(S - m)`) | same, `LADDER_NOSCALE=1` |
| E | D + paged KV cache, page_size 64 (HND pools, random block table) | same, `LADDER_PAGED64=1` |

Shape: B=1, H_q=32, H_kv=2 (GQA 16:1), d=128, S_q=S_kv in {8k, 16k, 32k}, bf16 O, no Stats,
no Amax_O (inference), causal and no-mask, persistent scheduler NATURAL unless `--sched` is set.

`prefill_d128_mxfp8_ladder.py` is a verbatim copy of the Rubin MXFP8 kernel with three
import-time env levers. It is a **bench one-off, not a product kernel**: the page-64 loader pages
only the K/V payload through the block tables (one TMA box per page; V as two 64-row boxes per
128-row tile) and keeps the block-scale SF tensors dense / tile-indexed, because a 64-token page
cannot hold a whole F8_128x4 SF atom. `bench_ladder.py` bypasses the two Rubin / page-size
adapter gates and the prepared binder's pool-sized SF byte-count check for rung E only.

## Files

- `bench_ladder.py` — one rung per process; builds inputs (torch-only MXFP8 quantizer from
  `test/python/sdpa/mxfp8_quant.py`), drives `SdpaFwdDslSm100` directly, validates against an fp32
  blocked reference (rel. error of the output amax; ~1.5 % is fp8 P-quantization noise), times by
  CUDA-graph replay (and eager), writes JSON lines. `--ncu-run` does 3 warmups + exactly one launch.
- `run_sweep.sh` / `run_ncu_sweep.sh` — timing and ncu (`--set full`, 4th FROST launch,
  `-k regex:cudnn_kernel__kernel`) sweep drivers; `SCHED=lpt|lpt_l2` env selects the scheduler.
- `ncu_table.py` — imports `<rung>_<mask>_<S>[_<sched>].ncu-repz` reports via the ncu CLI and
  writes CSV/JSON + a markdown table (tensor / MUFU / FMA pipe utilisation, issue, stalls, DRAM, L2 ...).
- `hecate_py.sh` (Vera-Rubin node, in the `rubin-py3-devel` container) and `board_py.sh`
  (x86 Rubin board, `sudo ncu`) — environment wrappers.
- `results/` — measurements (see `results/README.md`).

Each rung gets its own CuTe-DSL / FE plan cache directory (`*_rung<X>`): rungs C/D/E share one
kernel file and differ only by env flags the caches do not key on (a shared cache silently served
rung D with rung C's cubin: 21 % error instead of 1.6 %).
