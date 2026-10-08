# Rubin GR100 (cc 10.7) MXFP8 Q/K/V GQA prefill: dH=128 / dH=256, paged KV page 64, correction ON / OFF

Timing host(s): w2u1g-lc-0614 (cc 10.7, 216 SMs). Shapes B/H_q/H_kv = 1/32/8; Q/K/V MXFP8 e4m3 + E8M0/32, O bf16, no Stats / Amax_O. Time = median over interleaved rounds of 4-replay CUDA-graph bursts with 150 ms gaps (board cooldown protocol); TF/s = 4·B·H_q·S²·d (×(S+1)/2S causal) / time. MMA util = tensor SOL clocks per 128x128 step (BMM1 + BMM2 [+ ones-MMA row-sum], 16384 MAC/clk/SM) / measured clocks per step (unit_sol.py). corr=never rows are *timing-only* unless the rescale emulation found zero rescales. 

## Timing

### d=128, none mask, scheduler natural

| config | 8k µs (TF/s) | MMA util | Δ first | 16k µs (TF/s) | MMA util | Δ first | 32k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga2 | 172.7 (6366) | 40.4 % | – | 641.3 (6858) | 43.5 % | – | 2417.3 (7278) | 46.2 % | – |
| `corrdefault_p64_half_pf_benc_cf1h1`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | 181.4 (6062) | 38.5 % | +5.0% | 673.1 (6534) | 41.5 % | +4.9% | 2516.2 (6992) | 44.4 % | +4.1% |
| `corralways_p64_half_pf_benc_cf1h1`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | 249.0 (4416) | 28.0 % | +44.2% | 951.4 (4623) | 29.4 % | +48.3% | 3594.4 (4894) | 31.1 % | +48.7% |
| `corrnever_p64_half_pf_benc_cf1h1`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | 169.3 (6494) | 41.2 % | -2.0% | 625.4 (7032) *timing-only* | 44.7 % | -2.5% | 2364.6 (7440) *timing-only* | 47.2 % | -2.2% |

Median SM clock / power during timing: 2364.0 MHz / 470.29 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 3.64% rms 2.66%; corrdefault_p64_half_pf_benc_cf1h1@8k rel 3.64% rms 2.66%; corralways_p64_half_pf_benc_cf1h1@8k rel 3.33% rms 2.62%; corrnever_p64_half_pf_benc_cf1h1@8k rel 3.64% rms 2.66%; corrnever_p64_half_pf_benc_cf1h1@8k rescales 0/524288 warp-steps (0.00%).



### d=128, causal mask, scheduler lpt

| config | 8k µs (TF/s) | MMA util | Δ first | 16k µs (TF/s) | MMA util | Δ first | 32k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga2 | 104.3 (5272) | 34.0 % | – | 340.9 (6451) | 41.3 % | – | 1255.8 (7005) | 44.7 % | – |
| `corrdefault_p64_half_pf_benc_cf1h1`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | 103.8 (5294) | 34.1 % | -0.4% | 332.8 (6607) | 42.3 % | -2.4% | 1219.6 (7212) | 46.0 % | -2.9% |
| `corralways_p64_half_pf_benc_cf1h1`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | 137.0 (4014) | 25.9 % | +31.3% | 464.5 (4734) | 30.3 % | +36.3% | 1741.8 (5050) | 32.2 % | +38.7% |
| `corrnever_p64_half_pf_benc_cf1h1`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=1 hoist=1, cga2 | 103.4 (5318) | 34.3 % | -0.9% | 332.7 (6610) *timing-only* | 42.3 % | -2.4% | 1221.1 (7204) *timing-only* | 45.9 % | -2.8% |

Median SM clock / power during timing: 2364.0 MHz / 465.95 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 1.80% rms 2.31%; corrdefault_p64_half_pf_benc_cf1h1@8k rel 1.80% rms 2.31%; corralways_p64_half_pf_benc_cf1h1@8k rel 1.80% rms 2.30%; corrnever_p64_half_pf_benc_cf1h1@8k rel 1.80% rms 2.31%; corrnever_p64_half_pf_benc_cf1h1@8k rescales 0/266240 warp-steps (0.00%).



## Per-unit utilisation vs SOL (analytic model + ncu cross-check)

Units: tensor (UTCQMMA MACs), mufu (exp2 lanes), issue (warp-instr/4), fma / alu (pipe warp-instr / 2), smem (TMA writes + UMMA operand reads at 128 B/clk), tmem (tcgen05.ld/st at an assumed 512 B/clk), l2 (K/V(+SF) bytes per step per SM at the ncu-derived L2 peak), dram (whole-kernel bytes). Cell = util% [SOL clk/step] (ncu pipe %). Constants and sources: `unit_sol.py` header.

### d=128, none, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 970 | 28% [272] | 26% [256] | 42% [412] | 26% [256] | 3% [30] | 35% [338] | 43% [418] | 7% [72] | 6% [57] | tmem 43% |
| `corrdefault_p0_half_pf_prod` | 673 | 40% [272] | 38% [256] | 50% [340] | 19% [128] | 4% [30] | 50% [338] | 24% [162] | 11% [72] | 9% [57] | issue 50% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 707 | 38% [272] | 36% [256] | 48% [340] | 18% [128] | 4% [30] | 48% [338] | 23% [162] | 10% [72] | 8% [57] | issue 48% |
| `corrnever_p64_half_pf_benc_cf1h1` | 660 | 41% [272] | 39% [256] | 51% [340] | 19% [128] | 5% [30] | 51% [338] | 25% [162] | 11% [72] | 9% [57] | issue 51% |

### d=128, none, S=16k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 927 | 29% [272] (32) | 28% [256] (30) | 49% [453] (41) | 31% [286] (32) | 11% [102] (10) | 36% [338] (29) | 45% [418] (3) | 8% [72] (10) | 3% [29] | issue 49% |
| `corrdefault_p0_half_pf_prod` | 625 | 44% [272] (47) | 41% [256] (45) | 58% [362] (47) | 24% [151] (23) | 21% [129] (14) | 54% [338] (44) | 26% [162] (2) | 12% [72] (15) | 5% [29] | issue 58% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 656 | 41% [272] (45) | 39% [256] (43) | 55% [363] (46) | 22% [147] (22) | 19% [127] (15) | 52% [338] (42) | 25% [162] (1) | 11% [72] (14) | 4% [29] | issue 55% |
| `corrnever_p64_half_pf_benc_cf1h1` | 609 | 45% [272] (49) | 42% [256] (47) | 58% [352] (47) | 24% [145] (24) | 20% [119] (16) | 55% [338] (45) | 27% [162] (1) | 12% [72] (16) | 5% [29] | issue 58% |

### d=128, none, S=32k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 875 | 31% [272] | 29% [256] | 47% [412] | 29% [256] | 3% [30] | 39% [338] | 48% [418] | 8% [72] | 2% [14] | tmem 48% |
| `corrdefault_p0_half_pf_prod` | 589 | 46% [272] | 43% [256] | 58% [340] | 22% [128] | 5% [30] | 57% [338] | 28% [162] | 12% [72] | 2% [14] | issue 58% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 613 | 44% [272] | 42% [256] | 55% [340] | 21% [128] | 5% [30] | 55% [338] | 26% [162] | 12% [72] | 2% [14] | issue 55% |
| `corrnever_p64_half_pf_benc_cf1h1` | 576 | 47% [272] | 44% [256] | 59% [340] | 22% [128] | 5% [30] | 59% [338] | 28% [162] | 13% [72] | 2% [14] | issue 59% |

### d=128, causal, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 1051 | 26% [272] | 24% [256] | 39% [412] | 24% [256] | 3% [30] | 32% [338] | 40% [418] | 7% [72] | 11% [113] | tmem 40% |
| `corrdefault_p0_half_pf_prod` | 800 | 34% [272] | 32% [256] | 42% [340] | 16% [128] | 4% [30] | 42% [338] | 20% [162] | 9% [72] | 14% [113] | issue 42% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 797 | 34% [272] | 32% [256] | 43% [340] | 16% [128] | 4% [30] | 42% [338] | 20% [162] | 9% [72] | 14% [113] | issue 43% |
| `corrnever_p64_half_pf_benc_cf1h1` | 793 | 34% [272] | 32% [256] | 43% [340] | 16% [128] | 4% [30] | 43% [338] | 20% [162] | 9% [72] | 14% [113] | issue 43% |

### d=128, causal, S=16k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 898 | 30% [272] (32) | 29% [256] (31) | 52% [465] (45) | 35% [318] (33) | 12% [110] (12) | 38% [338] (31) | 47% [418] (3) | 8% [72] (12) | 6% [57] | issue 52% |
| `corrdefault_p0_half_pf_prod` | 659 | 41% [272] (44) | 39% [256] (43) | 61% [402] (48) | 28% [181] (22) | 22% [147] (17) | 51% [338] (43) | 25% [162] (2) | 11% [72] (16) | 9% [57] | issue 61% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 643 | 42% [272] (45) | 40% [256] (44) | 60% [384] (49) | 28% [177] (23) | 18% [114] (18) | 53% [338] (44) | 25% [162] (1) | 11% [72] (17) | 9% [57] | issue 60% |
| `corrnever_p64_half_pf_benc_cf1h1` | 643 | 42% [272] (45) | 40% [256] (44) | 61% [390] (48) | 27% [175] (23) | 22% [139] (18) | 53% [338] (44) | 25% [162] (1) | 11% [72] (17) | 9% [57] | issue 61% |

### d=128, causal, S=32k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf1h1` | 845 | 32% [272] | 30% [256] | 49% [412] | 30% [256] | 4% [30] | 40% [338] | 49% [418] | 9% [72] | 3% [29] | tmem 49% |
| `corrdefault_p0_half_pf_prod` | 609 | 45% [272] | 42% [256] | 56% [340] | 21% [128] | 5% [30] | 55% [338] | 27% [162] | 12% [72] | 5% [29] | issue 56% |
| `corrdefault_p64_half_pf_benc_cf1h1` | 592 | 46% [272] | 43% [256] | 57% [340] | 22% [128] | 5% [30] | 57% [338] | 27% [162] | 12% [72] | 5% [29] | issue 57% |
| `corrnever_p64_half_pf_benc_cf1h1` | 592 | 46% [272] | 43% [256] | 57% [340] | 22% [128] | 5% [30] | 57% [338] | 27% [162] | 12% [72] | 5% [29] | issue 57% |

## ncu (one launch, `--set full`, 4th launch after 3 warm-ups; pipe % of peak sustained over SM-active cycles)

### d=128, none mask, scheduler natural

| S | config | ncu µs | TF/s | clk GHz | SM active % | tensor pipe % | UTCQMMA fp8 % (realtime) | MUFU (XU) % | FMA % | ALU % | TMEM instr % | issue % | IPC | warp lat cyc | stall long-sb | stall wait | stall short-sb | stall barrier | stall math-throttle | stall mio-throttle | L1/SMEM % | L2 % | L2 hit % | DRAM % | DRAM GB/s | regs | SMEM KB | waves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16k | `corralways_p64_half_pf_benc_cf1h1` | 952 | 4619 | 2.32 | 94.6 | 31.6 | – | 30.2 | 32.1 | 10.1 | 2.9 | 41.4 | 1.66 | 9.6 | 5.23 | 0.93 | 1.09 | 0.83 | 0.11 | 0.02 | 29.4 | 10.4 | 88.4 | – | 201 | 128 | 175 | 9.48 |
| 16k | `corrdefault_p0_half_pf_prod` | 642 | 6853 | 2.30 | 94.9 | 47.1 | – | 45.0 | 22.7 | 14.4 | 1.7 | 47.1 | 1.89 | 8.4 | 4.94 | 0.84 | 0.80 | 0.08 | 0.02 | 0.01 | 44.0 | 14.6 | 91.3 | – | 298 | 128 | 174 | 9.48 |
| 16k | `corrdefault_p64_half_pf_benc_cf1h1` | 674 | 6529 | 2.29 | 94.6 | 45.2 | – | 43.2 | 22.2 | 14.8 | 1.2 | 45.6 | 1.82 | 8.7 | 5.13 | 0.90 | 0.64 | 0.37 | 0.03 | 0.01 | 42.1 | 14.3 | 89.0 | – | 284 | 128 | 175 | 9.48 |
| 16k | `corrnever_p64_half_pf_benc_cf1h1` | 627 | 7011 | 2.29 | 94.4 | 48.7 | – | 46.5 | 23.9 | 15.5 | 1.1 | 47.3 | 1.89 | 8.4 | 5.35 | 0.81 | 0.58 | 0.13 | 0.03 | 0.01 | 45.3 | 16.0 | 89.2 | – | 306 | 128 | 175 | 9.48 |



### d=128, causal mask, scheduler lpt

| S | config | ncu µs | TF/s | clk GHz | SM active % | tensor pipe % | UTCQMMA fp8 % (realtime) | MUFU (XU) % | FMA % | ALU % | TMEM instr % | issue % | IPC | warp lat cyc | stall long-sb | stall wait | stall short-sb | stall barrier | stall math-throttle | stall mio-throttle | L1/SMEM % | L2 % | L2 hit % | DRAM % | DRAM GB/s | regs | SMEM KB | waves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16k | `corralways_p64_half_pf_benc_cf1h1` | 468 | 4697 | 2.30 | 98.2 | 32.1 | – | 30.8 | 32.8 | 12.5 | 2.9 | 44.6 | 1.78 | 8.9 | 5.06 | 0.91 | 1.10 | 0.21 | 0.09 | 0.01 | 31.1 | 12.1 | 84.2 | – | 408 | 128 | 175 | 9.48 |
| 16k | `corrdefault_p0_half_pf_prod` | 344 | 6401 | 2.28 | 98.0 | 44.4 | – | 42.5 | 21.8 | 16.7 | 1.7 | 47.5 | 1.90 | 8.4 | 4.80 | 0.89 | 0.69 | 0.25 | 0.02 | 0.01 | 42.9 | 16.3 | 86.0 | – | 551 | 128 | 174 | 9.48 |
| 16k | `corrdefault_p64_half_pf_benc_cf1h1` | 337 | 6531 | 2.27 | 97.9 | 45.5 | – | 43.6 | 22.5 | 18.0 | 1.2 | 49.4 | 1.97 | 8.1 | 4.63 | 0.86 | 0.57 | 0.23 | 0.04 | 0.01 | 43.9 | 16.8 | 85.7 | – | 568 | 128 | 175 | 9.48 |
| 16k | `corrnever_p64_half_pf_benc_cf1h1` | 337 | 6528 | 2.27 | 97.8 | 45.5 | – | 43.6 | 22.6 | 17.6 | 1.0 | 47.7 | 1.91 | 8.3 | 5.15 | 0.78 | 0.53 | 0.19 | 0.04 | 0.01 | 43.8 | 16.8 | 83.8 | – | 569 | 128 | 175 | 9.48 |



## PerfSim (GR100, B=1 H=1 S=4096 no mask, SSAF route)

_pending: traces captured with perfsim/capture.sh, submitted with perfsim/submit.sh; run perfsim/summarize.py --out results/perfsim_RESULTS.md and pass the JSON to --perfsim._

## Files

- `results/*.jsonl` -- bench.py records (one per config x S x mask: all burst samples, clocks/power per round, kernel CFG/kmod flags, validation, rescale emulation)
- `results/ncu_table.{csv,json}` -- ncu_table.py over run_ncu.sh reports (+ `.raw.csv` / `.sass.csv` exports per report)
- `results/unit_sol.{json,md}` -- unit_sol.py
- `results/perfsim_RESULTS.{md,json}` -- perfsim/summarize.py
