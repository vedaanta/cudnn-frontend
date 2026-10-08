# Rubin GR100 (cc 10.7) MXFP8 Q/K/V GQA prefill: dH=128 / dH=256, paged KV page 64, correction ON / OFF

Timing host(s): w2u1g-lc-0614 (cc 10.7, 216 SMs). Shapes B/H_q/H_kv = 1/32/8; Q/K/V MXFP8 e4m3 + E8M0/32, O bf16, no Stats / Amax_O. Time = median over interleaved rounds of 4-replay CUDA-graph bursts with 150 ms gaps (board cooldown protocol); TF/s = 4·B·H_q·S²·d (×(S+1)/2S causal) / time. MMA util = tensor SOL clocks per 128x128 step (BMM1 + BMM2 [+ ones-MMA row-sum], 16384 MAC/clk/SM) / measured clocks per step (unit_sol.py). corr=never rows are *timing-only* unless the rescale emulation found zero rescales. 

## Timing

### d=128, none mask, scheduler natural

| config | 8k µs (TF/s) | MMA util | Δ first | 32k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga2 | 172.2 (6387) | 40.6 % | – | 2417.8 (7276) | 46.2 % | – |
| `corrdefault_p0_f32_scale_prod`<br>corr=default, paged=0, f32-softmax, scale in-kernel, product kernel, cga2 | 217.0 (5067) | 32.2 % | +26.0% | – | – | – |
| `corrdefault_p64_half_pf_benc_cf1h1`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | – | – | – | 2515.9 (6992) | 44.4 % | +4.1% |
| `corralways_p64_half_pf_benc_cf1h1`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | – | – | – | 3594.7 (4894) | 31.1 % | +48.7% |
| `corrnever_p64_half_pf_benc_cf1h1`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | – | – | – | 2364.6 (7440) *timing-only* | 47.2 % | -2.2% |

Median SM clock / power during timing: 2364.0 MHz / 481.47 W, 2364.0 MHz / 600.89 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 3.64% rms 2.66%; corrdefault_p0_f32_scale_prod@8k rel 3.52% rms 2.66%.

### d=128, causal mask, scheduler natural

| config | 8k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga2 | 128.6 (4276) | 27.4 % | – |
| `corrdefault_p0_f32_scale_prod`<br>corr=default, paged=0, f32-softmax, scale in-kernel, product kernel, cga2 | 158.6 (3467) | 22.2 % | +23.4% |

Median SM clock / power during timing: 2382.0 MHz / 464.305 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 1.80% rms 2.31%; corrdefault_p0_f32_scale_prod@8k rel 1.57% rms 2.31%.





### d=256, none mask, scheduler natural

| config | 8k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga1 | 357.6 (6150) | 36.5 % | – |
| `corrdefault_p0_f32_scale_prod`<br>corr=default, paged=0, f32-softmax, scale in-kernel, product kernel, cga1 | 371.9 (5913) | 35.1 % | +4.0% |

Median SM clock / power during timing: 2382.0 MHz / 472.78499999999997 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 3.59% rms 2.62%; corrdefault_p0_f32_scale_prod@8k rel 3.43% rms 2.62%.

### d=256, causal mask, scheduler natural

| config | 8k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga1 | 202.4 (5434) | 33.0 % | – |
| `corrdefault_p0_f32_scale_prod`<br>corr=default, paged=0, f32-softmax, scale in-kernel, product kernel, cga1 | 214.5 (5126) | 31.1 % | +6.0% |

Median SM clock / power during timing: 2364.0 MHz / 472.65999999999997 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 1.45% rms 2.35%; corrdefault_p0_f32_scale_prod@8k rel 1.61% rms 2.35%.





## Per-unit utilisation vs SOL (analytic model + ncu cross-check)

Units: tensor (UTCQMMA MACs), mufu (exp2 lanes), issue (warp-instr/4), fma / alu (pipe warp-instr / 2), smem (TMA writes + UMMA operand reads at 128 B/clk), tmem (tcgen05.ld/st at an assumed 512 B/clk), l2 (K/V(+SF) bytes per step per SM at the ncu-derived L2 peak), dram (whole-kernel bytes). Cell = util% [SOL clk/step] (ncu pipe %). Constants and sources: `unit_sol.py` header.

### d=128, none, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_f32_scale_prod` | 845 | 32% [272] | 61% [512] | 55% [462] | 30% [256] | 4% [30] | 40% [338] | 19% [162] | 16% [132] | 7% [57] | xu_inst 61% |
| `corrdefault_p0_half_pf_prod` | 671 | 41% [272] (45) | 38% [256] (43) | 55% [367] (46) | 23% [154] (22) | 20% [131] (14) | 50% [338] (41) | 24% [162] (2) | 11% [72] (13) | 9% [57] | issue 55% |

### d=128, none, S=32k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 875 | 31% [272] | 29% [256] | 47% [412] | 29% [256] | 3% [30] | 39% [338] | 48% [418] | 15% [132] | 2% [14] | tmem 48% |
| `corrdefault_p0_half_pf_prod` | 589 | 46% [272] | 43% [256] | 58% [340] | 22% [128] | 5% [30] | 57% [338] | 28% [162] | 22% [132] | 2% [14] | issue 58% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 613 | 44% [272] | 42% [256] | 55% [340] | 21% [128] | 5% [30] | 55% [338] | 26% [162] | 22% [132] | 2% [14] | issue 55% |
| `corrnever_p64_half_pf_benc_cf1h1` | 576 | 47% [272] | 44% [256] | 59% [340] | 22% [128] | 5% [30] | 59% [338] | 28% [162] | 23% [132] | 2% [14] | issue 59% |

### d=128, causal, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_f32_scale_prod` | 1226 | 22% [272] | 42% [512] | 38% [462] | 21% [256] | 2% [30] | 28% [338] | 13% [162] | 11% [132] | 9% [114] | xu_inst 42% |
| `corrdefault_p0_half_pf_prod` | 994 | 27% [272] | 26% [256] | 34% [340] | 13% [128] | 3% [30] | 34% [338] | 16% [162] | 13% [132] | 11% [114] | issue 34% |

### d=256, none, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_f32_scale_prod` | 1460 | 35% [512] | 35% [512] | 32% [462] | 18% [256] | 2% [30] | 89% [1296] | 11% [162] | 72% [1056] | 8% [116] | smem 89% |
| `corrdefault_p0_half_pf_prod` | 1404 | 36% [512] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 92% [1296] | 12% [162] | 75% [1056] | 8% [116] | smem 92% |

### d=256, causal, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_f32_scale_prod` | 1646 | 31% [512] | 31% [512] | 28% [462] | 16% [256] | 2% [30] | 79% [1296] | 10% [162] | 64% [1056] | 14% [226] | smem 79% |
| `corrdefault_p0_half_pf_prod` | 1553 | 33% [512] | 16% [256] | 22% [340] | 8% [128] | 2% [30] | 83% [1296] | 10% [162] | 68% [1056] | 15% [226] | smem 83% |

## ncu (one launch, `--set full`, 4th launch after 3 warm-ups; pipe % of peak sustained over SM-active cycles)

### d=128, none mask, scheduler natural

| S | config | ncu µs | TF/s | clk GHz | SM active % | tensor pipe % | UTCQMMA fp8 % (realtime) | MUFU (XU) % | FMA % | ALU % | TMEM instr % | issue % | IPC | warp lat cyc | stall long-sb | stall wait | stall short-sb | stall barrier | stall math-throttle | stall mio-throttle | L1/SMEM % | L2 % | L2 hit % | DRAM % | DRAM GB/s | regs | SMEM KB | waves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8k | `corrdefault_p0_half_pf_prod` | 174 | 6317 | 2.27 | 92.2 | 45.3 | – | 43.3 | 22.1 | 14.2 | 1.7 | 45.8 | 1.84 | 8.5 | 4.96 | 0.85 | 0.80 | 0.12 | 0.02 | 0.01 | 41.2 | 12.7 | 83.3 | – | 398 | 128 | 174 | 4.74 |
| 8k | `corrdefault_p64_half_pf_benc_cf1h1` | 182 | 6029 | 2.26 | 92.4 | 43.4 | – | 41.5 | 21.5 | 14.6 | 1.1 | 44.6 | 1.78 | 8.8 | 5.18 | 0.91 | 0.64 | 0.39 | 0.03 | 0.01 | 39.5 | 12.5 | 80.0 | – | 382 | 128 | 175 | 4.74 |












## PerfSim (GR100, B=1 H=1 S=4096 no mask, SSAF route)

_pending: traces captured with perfsim/capture.sh, submitted with perfsim/submit.sh; run perfsim/summarize.py --out results/perfsim_RESULTS.md and pass the JSON to --perfsim._

## Files

- `results/*.jsonl` -- bench.py records (one per config x S x mask: all burst samples, clocks/power per round, kernel CFG/kmod flags, validation, rescale emulation)
- `results/ncu_table.{csv,json}` -- ncu_table.py over run_ncu.sh reports (+ `.raw.csv` / `.sass.csv` exports per report)
- `results/unit_sol.{json,md}` -- unit_sol.py
- `results/perfsim_RESULTS.{md,json}` -- perfsim/summarize.py
