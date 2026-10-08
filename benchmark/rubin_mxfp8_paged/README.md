# Rubin (cc 10.7) MXFP8 Q/K/V GQA prefill benchmark -- dH=128 / dH=256, paged KV page 64, correction ON / OFF

Bench harness for the FROST CuTe-DSL SDPA forward kernels on GR100 (cc 10.7): graph-replay timing with the board
cooldown protocol, ncu per-unit metrics, an analytic per-unit SOL model, and the PerfSim (SSAF route) capture /
submit / summary scripts including register-file bandwidth utilisation.

Shape (plan): B=1, H_q=32, H_kv=8, d in {128, 256}, S_q=S_kv in {8k, 16k, 32k}, Q/K/V MXFP8 (e4m3 + E8M0 block
scales per 32), O bf16, no Stats / Amax_O, masks none + causal. Trick stack on by default: f16 softmax
(`softmax_precision=HALF`), 1/ln2 folded outside (`softmax_scale_prefolded=True`, Q pre-multiplied by
`attn_scale*log2(e)` before quantization -> fused FHADD2 shift+convert arm), ones-MMA row-sum (d128 product),
and the bench-kernel levers (paged page 64, correction fast path + TMEM-base hoist, correction always / never).

## Files

| file | what |
|---|---|
| `bench.py` | the harness. One CONFIG = one worker process (the bench kernels read `BENCH_*` at import and the DSL / compiled-plan caches do not key on them; each worker gets `CUTE_DSL_CACHE_DIR` + `XDG_CACHE_HOME` suffixed by kernel file + every lever). The coordinator interleaves the configs of a cell round-robin: `--reps` graph replays per burst, `--cooldown-ms` idle after each burst, `--rounds` rounds; records clocks / power per round (nvidia-smi), all burst samples, kernel CFG + module flags, validation vs an fp32 blocked reference, the rescale emulation (`corr=never` is exact only at zero rescales -> else `timing_only`). `--ncu-run` = in-process warm-ups + ONE launch (ncu / APIC). `--kv-ramp` scales K/V per 32-token block for scale-factor mapping validation. |
| `board_py_0614.sh` | env wrapper for board w2u1g-lc-0614 (uv py3.10 + venv_sp + cudnn 9.26 under /tmp/vagarwalla; `NCU=1` wraps python in the locally staged Nsight Compute; `APIC=1` drops LD_PRELOAD for the capturer). |
| `run_sweep.sh` | timing sweep: per mask one coordinator with all configs interleaved; holds `/tmp/vagarwalla/ladder/gpu_timing.lock`. |
| `run_ncu.sh` | one `--set full` report per (config, mask, S) at the 4th launch + the raw-metrics and SASS-source CSV exports next to it (so no ncu is needed off-board). |
| `ncu_table.py` | per-unit metrics table (tensor / UTCQMMA fp8 realtime / MUFU / FMA / ALU / TMEM / issue / stalls / SMEM wavefronts / L1 / L2 / DRAM / launch) from the exported CSVs (or `--ncu` to import), joined with the timing JSONL; executed-SASS opcode histogram per pipe class via `sass_tools.ncu_source`. |
| `unit_sol.py` | analytic per-unit minimum clocks per 128x128 step per SM (tensor, MUFU, issue, FMA, ALU, SMEM, TMEM, L2, DRAM) with every constant sourced in its header; util = SOL / measured step clocks (kernel us x SM clock / steps per SM); ncu cross-check column; instruction counts per step from the executed SASS when an ncu row exists. |
| `sass_tools.py` | carve the fatbin out of a compiled-plan `kernel.o` (`$XDG_CACHE_HOME/cudnn_frontend/compiled_plans/v2/*/*/kernel.o`), `cuobjdump -sass` with the board toolkit (needs `nvdisasm_internal` on PATH for sm_107a), static / executed opcode histograms. |
| `make_report.py` | `results/REPORT.md`: timing + MMA util tables, per-unit util-vs-SOL tables, ncu table, PerfSim section (placeholder until the PICs exist). |
| `perfsim/capture.sh` + `app_capture.sh` | APIC capture on 0614 of one config at B=1 H=1 S=4096 none (3 warm-ups + 1; submit the LAST k-instance). `perfsim/pull_traces.sh` copies the last instance per case to `/home/scratch.vagarwalla_gpu/perfsim_mxb/trace_output/`. |
| `perfsim/submit.sh`, `perfsim/status.sh`, `perfsim/config_gr100.yaml` | `flow.perfsim run -chip gr100 -enableMorph -pic` per trace from a computelab frontend (SSAF route; the config is perfsim_ladder's: TMEM 288 KB knob, enum knobs dropped, `SSAF_args.pmMode PM_KERNEL`). |
| `perfsim/summarize.py` | RESULTS.md / JSON from the PIC + raw SMART counters: kernel cycles, top-3 SOL, mainloop MMA util, per-unit SOL table (SM issue / MMA / XU / FMA / ALU / TMEM rd+wr / LST / TPC ICC / L2 / FB), register-file READ and WRITE port utilisation (normalisation documented in the header), coupled vs decoupled writes, reuse-cache hits, register-read dispatch stalls, uniform-RF (pipe active + UR instr/clk; this SMART build has no URF port counters), TMEM->RF write-back stalls, TMEM traffic by op. |

## Running on the board

```
# stage: rsync <worktree>/{python,test,benchmark} to 0614:/tmp/vagarwalla/wt_b/ (python/cudnn/*.so included)
B=/tmp/vagarwalla/wt_b/benchmark/rubin_mxfp8_paged
# timing (product kernels, trick stack): d128 + d256, both masks, 8k/16k/32k, 15 rounds x 4 replays, 150 ms gaps
bash $B/run_sweep.sh board_py_0614.sh /tmp/vagarwalla/mxb/results_d128.jsonl 128 "corr=default" "none causal" 8192 16384 32768
# with the bench kernels: three correction modes + paged 64, interleaved in every cell
bash $B/run_sweep.sh board_py_0614.sh /tmp/vagarwalla/mxb/results_d128.jsonl 128 \
   "corr=default,paged=64 corr=always,paged=64 corr=never,paged=64" "none causal" 8192 16384 32768
# ncu at 16k (and the SASS/raw CSV exports)
bash $B/run_ncu.sh board_py_0614.sh /tmp/vagarwalla/mxb/ncu 128 "corr=default,paged=64 corr=always,paged=64" "none causal" 16384
# APIC capture for PerfSim (one per primary config), then pull + submit from the scratch host / computelab
bash $B/perfsim/capture.sh 128 "corr=always,paged=64"
# tables
python ncu_table.py --dir <ncu dir> --timing results.jsonl --out results/ncu_table
python unit_sol.py --timing results.jsonl --ncu results/ncu_table.json --out results/unit_sol.json --md results/unit_sol.md
python make_report.py --timing results.jsonl --ncu results/ncu_table.json --unit-sol results/unit_sol.json --out results/REPORT.md
```

Config string keys (`key=val,...`): `corr` default|always|never, `paged` 0|64, `half` 0|1, `prefolded` 0|1,
`kernel` product|bench|auto (auto = bench iff a lever needs it), `corrfast` / `hoist` 0|1|auto (on with the bench
kernel), `rowsum_mma` 0|1 (d256 stretch), `cga` 1|2|auto (adapter default: d128 cga2, d256 MXFP8 cga1), `sched`,
`sf` dense|paged (scale-factor layout the paged loader expects; auto = dense for d128, paged for d256), `name`.

## Switching to the bench kernels

`kernel=bench` (or any non-default lever) monkeypatches `api_dsl._SM107_MXFP8_KERNEL_FILES[(d, d)]` to
`sm107/prefill_d{d}_mxfp8_bench.py` before the adapter is built, exports `BENCH_PAGED64 / BENCH_CORR /
BENCH_CORRFAST / BENCH_HOIST / BENCH_ROWSUM_MMA / BENCH_KV_RAMP` into the worker's environment, bypasses the
adapter's cc 10.7 MXFP8 paged gates (`BYPASS_MSGS`, incl. the prefolded-paged NotImplementedError) and, for
`sf=dense`, presents the plan as unpaged to the prepared binder's per-page scale-factor byte-count check
(`_DenseSFView`). HALF + prefolded reach the kernel through TemplateParams (`softmax_f16`, `softmax_scale_prefolded`).
Paged inputs: HND pools `(num_pages, H_kv, 64, d)` filled through a random block table (+7 dead pages),
`seq_kv_lens` = S; `build_paged_sf()` is the hook for a per-page scale-factor pool layout (d256 kernel contract).

## Gotchas

- Interleave configs; never time sequentially (2.4 -> 1.8 GHz throttling on 0614); check `clocks` / `power_w_median` in the records.
- A new lever MUST be in `LEVER_ENV` (cache suffix) or configs silently share a compiled plan and run the wrong kernel; `kmod` in the record proves which flags the module saw.
- `ncu --import` needs a writable HOME (run_ncu.sh sets one); the board has no sudo and needs none (RmProfilingAdminOnly=0).
- The APIC tool's shebang (`/home/utils/perl-5.8.8`) does not exist on the board: capture.sh runs it through the system perl (core modules only).
- The board reports 216 SMs (not 208); `steps per SM` uses the recorded `sm_count`.
