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



### d=256, none mask, scheduler natural

| config | 8k µs (TF/s) | MMA util | Δ first | 16k µs (TF/s) | MMA util | Δ first | 32k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga1 | 356.7 (6165) | 36.8 % | – | 1230.0 (7152) | 42.7 % | – | 4813.8 (7309) | 43.7 % | – |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2`<br>corr=default, paged=0, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 326.7 (6732) | 40.2 % | -8.4% | 1112.0 (7910) | 47.3 % | -9.6% | 4224.2 (8329) | 49.8 % | -12.2% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 361.9 (6077) | 36.3 % | +1.4% | 1200.8 (7325) | 43.8 % | -2.4% | 4716.3 (7460) | 44.6 % | -2.0% |
| `corralways_p64_half_pf_benc_cf0h0_cga2`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 473.5 (4644) | 27.8 % | +32.7% | 1717.0 (5123) | 30.6 % | +39.6% | 6681.0 (5266) | 31.5 % | +38.8% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 358.0 (6142) | 36.7 % | +0.4% | 1188.5 (7401) *timing-only* | 44.2 % | -3.4% | 4663.6 (7544) *timing-only* | 45.1 % | -3.1% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0 rowsum_mma=1, cga2 | 363.2 (6054) | 37.3 % | +1.8% | 1205.5 (7297) | 45.0 % | -2.0% | 4683.0 (7513) | 46.3 % | -2.7% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0 rowsum_mma=1, cga2 | 478.2 (4599) | 28.3 % | +34.1% | 1742.9 (5047) | 31.1 % | +41.7% | 6806.2 (5170) | 31.9 % | +41.4% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0 rowsum_mma=1, cga2 | 361.9 (6077) | 37.5 % | +1.4% | 1198.6 (7339) *timing-only* | 45.2 % | -2.5% | 4744.0 (7417) *timing-only* | 45.7 % | -1.4% |

Median SM clock / power during timing: 2364.0 MHz / 479.98 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 3.59% rms 2.62%; corrdefault_p0_half_pf_benc_cf0h0_cga2@8k rel 3.59% rms 2.62%; corrdefault_p64_half_pf_benc_cf0h0_cga2@8k rel 3.59% rms 2.62%; corralways_p64_half_pf_benc_cf0h0_cga2@8k rel 3.58% rms 2.58%; corrnever_p64_half_pf_benc_cf0h0_cga2@8k rel 3.59% rms 2.62%; corrnever_p64_half_pf_benc_cf0h0_cga2@8k rescales 0/524288 warp-steps (0.00%); corrdefault_p64_half_pf_benc_cf0h0rs1_cga2@8k rel 3.59% rms 2.62%; corralways_p64_half_pf_benc_cf0h0rs1_cga2@8k rel 3.58% rms 2.58%; corrnever_p64_half_pf_benc_cf0h0rs1_cga2@8k rel 3.59% rms 2.62%; corrnever_p64_half_pf_benc_cf0h0rs1_cga2@8k rescales 0/524288 warp-steps (0.00%).



### d=256, causal mask, scheduler lpt

| config | 8k µs (TF/s) | MMA util | Δ first | 16k µs (TF/s) | MMA util | Δ first | 32k µs (TF/s) | MMA util | Δ first |
|---|---|---|---|---|---|---|---|---|---|
| `corrdefault_p0_half_pf_prod`<br>corr=default, paged=0, f16-softmax, prefolded+fused, product kernel, cga1 | 188.2 (5844) | 35.5 % | – | 728.1 (6041) | 36.4 % | – | 2958.4 (5947) | 35.7 % | – |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2`<br>corr=default, paged=0, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 178.5 (6161) | 37.4 % | -5.1% | 678.6 (6481) | 39.0 % | -6.8% | 2745.1 (6409) | 38.5 % | -7.2% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 196.3 (5601) | 34.0 % | +4.3% | 768.2 (5725) | 34.5 % | +5.5% | 3068.4 (5734) | 34.4 % | +3.7% |
| `corralways_p64_half_pf_benc_cf0h0_cga2`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 254.8 (4316) | 26.2 % | +35.4% | 902.1 (4876) | 29.4 % | +23.9% | 3405.8 (5166) | 31.0 % | +15.1% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0, cga2 | 195.9 (5614) | 34.1 % | +4.1% | 787.4 (5586) *timing-only* | 33.6 % | +8.1% | 3141.5 (5600) *timing-only* | 33.6 % | +6.2% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2`<br>corr=default, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0 rowsum_mma=1, cga2 | 189.1 (5814) | 36.4 % | +0.5% | 758.1 (5802) | 36.0 % | +4.1% | 3039.8 (5787) | 35.8 % | +2.8% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2`<br>corr=always, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0 rowsum_mma=1, cga2 | 255.8 (4299) | 26.9 % | +35.9% | 908.5 (4841) | 30.1 % | +24.8% | 3443.2 (5110) | 31.6 % | +16.4% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2`<br>corr=never, paged=64, f16-softmax, prefolded+fused, bench kernel, corrfast=0 hoist=0 rowsum_mma=1, cga2 | 192.4 (5714) | 35.8 % | +2.3% | 776.9 (5662) *timing-only* | 35.2 % | +6.7% | 3089.1 (5695) *timing-only* | 35.2 % | +4.4% |

Median SM clock / power during timing: 2364.0 MHz / 472.91 W. Validation vs fp32 reference (max abs err / ref amax; rms rel): corrdefault_p0_half_pf_prod@8k rel 1.45% rms 2.35%; corrdefault_p0_half_pf_benc_cf0h0_cga2@8k rel 1.45% rms 2.35%; corrdefault_p64_half_pf_benc_cf0h0_cga2@8k rel 1.45% rms 2.35%; corralways_p64_half_pf_benc_cf0h0_cga2@8k rel 1.45% rms 2.34%; corrnever_p64_half_pf_benc_cf0h0_cga2@8k rel 1.45% rms 2.35%; corrnever_p64_half_pf_benc_cf0h0_cga2@8k rescales 0/266240 warp-steps (0.00%); corrdefault_p64_half_pf_benc_cf0h0rs1_cga2@8k rel 1.45% rms 2.31%; corralways_p64_half_pf_benc_cf0h0rs1_cga2@8k rel 1.45% rms 2.30%; corrnever_p64_half_pf_benc_cf0h0rs1_cga2@8k rel 1.45% rms 2.31%; corrnever_p64_half_pf_benc_cf0h0rs1_cga2@8k rescales 0/266240 warp-steps (0.00%).



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

### d=256, none, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf0h0_cga2` | 1845 | 28% [512] | 14% [256] | 26% [484] | 21% [384] | 2% [30] | 42% [776] | 37% [674] | 16% [287] | 6% [115] | smem 42% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1863 | 28% [528] | 14% [256] | 26% [484] | 21% [384] | 2% [30] | 43% [792] | 36% [674] | 15% [287] | 6% [115] | smem 43% |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1273 | 40% [512] | 20% [256] | 27% [340] | 10% [128] | 2% [30] | 61% [776] | 13% [162] | 23% [287] | 9% [115] | smem 61% |
| `corrdefault_p0_half_pf_prod` | 1390 | 37% [512] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 93% [1296] | 12% [162] | 41% [574] | 8% [115] | smem 93% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1410 | 36% [512] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 55% [776] | 11% [162] | 20% [287] | 8% [115] | smem 55% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1415 | 37% [528] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 56% [792] | 11% [162] | 20% [287] | 8% [115] | smem 56% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1395 | 37% [512] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 56% [776] | 12% [162] | 21% [287] | 8% [115] | smem 56% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1410 | 37% [528] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 56% [792] | 11% [162] | 20% [287] | 8% [115] | smem 56% |

### d=256, none, S=16k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf0h0_cga2` | 1672 | 31% [512] (32) | 15% [256] (16) | 49% [816] (37) | 35% [578] (34) | 19% [314] (8) | 46% [776] (32) | 40% [674] (3) | 17% [288] (23) | 3% [57] | issue 49% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1697 | 31% [528] (32) | 15% [256] (16) | 38% [649] (32) | 25% [424] (25) | 16% [271] (8) | 47% [792] (32) | 40% [674] (3) | 17% [289] (23) | 3% [57] | smem 47% |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1083 | 47% [512] (53) | 24% [256] (27) | 63% [687] (43) | 30% [326] (29) | 26% [281] (10) | 72% [776] (54) | 15% [162] (1) | 25% [275] (26) | 5% [57] | smem 72% |
| `corrdefault_p0_half_pf_prod` | 1198 | 43% [512] (47) | 21% [256] (24) | 63% [755] (39) | 27% [326] (26) | 26% [312] (8) | 108% [1296] (71) | 14% [162] (1) | 47% [563] (39) | 5% [57] | smem 108% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1170 | 44% [512] (48) | 22% [256] (24) | 61% [712] (42) | 28% [326] (27) | 29% [336] (13) | 66% [776] (48) | 14% [162] (1) | 24% [281] (28) | 5% [57] | smem 66% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1174 | 45% [528] (49) | 22% [256] (24) | 47% [554] (33) | 13% [156] (13) | 26% [303] (12) | 67% [792] (49) | 14% [162] (1) | 24% [281] (26) | 5% [57] | smem 67% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1157 | 44% [512] (49) | 22% [256] (25) | 62% [716] (40) | 28% [324] (27) | 27% [312] (12) | 67% [776] (49) | 14% [162] (1) | 24% [280] (28) | 5% [57] | smem 67% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1167 | 45% [528] (50) | 22% [256] (24) | 46% [535] (31) | 13% [154] (13) | 23% [269] (11) | 68% [792] (49) | 14% [162] (1) | 24% [281] (25) | 5% [57] | smem 68% |

### d=256, none, S=32k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf0h0_cga2` | 1627 | 31% [512] | 16% [256] | 30% [484] | 24% [384] | 2% [30] | 48% [776] | 41% [674] | 18% [287] | 2% [29] | smem 48% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1657 | 32% [528] | 15% [256] | 29% [484] | 23% [384] | 2% [30] | 48% [792] | 41% [674] | 17% [287] | 2% [29] | smem 48% |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1029 | 50% [512] | 25% [256] | 33% [340] | 12% [128] | 3% [30] | 75% [776] | 16% [162] | 28% [287] | 3% [29] | smem 75% |
| `corrdefault_p0_half_pf_prod` | 1172 | 44% [512] | 22% [256] | 29% [340] | 11% [128] | 3% [30] | 111% [1296] | 14% [162] | 49% [574] | 2% [29] | smem 111% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1148 | 45% [512] | 22% [256] | 30% [340] | 11% [128] | 3% [30] | 68% [776] | 14% [162] | 25% [287] | 3% [29] | smem 68% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1140 | 46% [528] | 22% [256] | 30% [340] | 11% [128] | 3% [30] | 69% [792] | 14% [162] | 25% [287] | 3% [29] | smem 69% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1136 | 45% [512] | 23% [256] | 30% [340] | 11% [128] | 3% [30] | 68% [776] | 14% [162] | 25% [287] | 3% [29] | smem 68% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1155 | 46% [528] | 22% [256] | 29% [340] | 11% [128] | 3% [30] | 69% [792] | 14% [162] | 25% [287] | 2% [29] | smem 69% |

### d=256, causal, S=8k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf0h0_cga2` | 1954 | 26% [512] | 13% [256] | 25% [484] | 20% [384] | 2% [30] | 40% [776] | 34% [674] | 15% [287] | 12% [226] | smem 40% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1962 | 27% [528] | 13% [256] | 25% [484] | 20% [384] | 2% [30] | 40% [792] | 34% [674] | 15% [287] | 12% [226] | smem 40% |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1369 | 37% [512] | 19% [256] | 25% [340] | 9% [128] | 2% [30] | 57% [776] | 12% [162] | 21% [287] | 17% [226] | smem 57% |
| `corrdefault_p0_half_pf_prod` | 1444 | 35% [512] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 90% [1296] | 11% [162] | 40% [574] | 16% [226] | smem 90% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1506 | 34% [512] | 17% [256] | 23% [340] | 8% [128] | 2% [30] | 52% [776] | 11% [162] | 19% [287] | 15% [226] | smem 52% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1451 | 36% [528] | 18% [256] | 23% [340] | 9% [128] | 2% [30] | 55% [792] | 11% [162] | 20% [287] | 16% [226] | smem 55% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1503 | 34% [512] | 17% [256] | 23% [340] | 9% [128] | 2% [30] | 52% [776] | 11% [162] | 19% [287] | 15% [226] | smem 52% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1476 | 36% [528] | 17% [256] | 23% [340] | 9% [128] | 2% [30] | 54% [792] | 11% [162] | 19% [287] | 15% [226] | smem 54% |

### d=256, causal, S=16k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf0h0_cga2` | 1744 | 29% [512] (31) | 15% [256] (16) | 47% [817] (36) | 34% [599] (33) | 18% [308] (7) | 45% [776] (32) | 39% [674] (3) | 16% [287] (24) | 7% [114] | issue 47% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1756 | 30% [528] (31) | 15% [256] (16) | 38% [666] (31) | 25% [441] (24) | 15% [271] (6) | 45% [792] (32) | 38% [674] (3) | 16% [287] (24) | 6% [114] | smem 45% |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1312 | 39% [512] (41) | 20% [256] (21) | 55% [715] (35) | 26% [347] (23) | 22% [294] (8) | 59% [776] (42) | 12% [162] (1) | 21% [279] (17) | 9% [114] | smem 59% |
| `corrdefault_p0_half_pf_prod` | 1407 | 36% [512] (38) | 18% [256] (19) | 55% [778] (32) | 24% [338] (21) | 23% [325] (8) | 92% [1296] (58) | 12% [162] (1) | 40% [565] (26) | 8% [114] | smem 92% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1485 | 34% [512] (37) | 17% [256] (19) | 49% [731] (33) | 23% [347] (20) | 23% [337] (8) | 52% [776] (38) | 11% [162] (1) | 19% [280] (14) | 8% [114] | smem 52% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1465 | 36% [528] (39) | 17% [256] (19) | 38% [558] (27) | 12% [173] (10) | 20% [296] (8) | 54% [792] (39) | 11% [162] (1) | 19% [281] (14) | 8% [114] | smem 54% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1522 | 34% [512] (36) | 17% [256] (18) | 48% [735] (31) | 23% [345] (20) | 21% [326] (10) | 51% [776] (37) | 11% [162] (1) | 18% [282] (14) | 7% [114] | smem 51% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1501 | 35% [528] (38) | 17% [256] (19) | 38% [575] (25) | 12% [173] (10) | 19% [291] (9) | 53% [792] (38) | 11% [162] (1) | 19% [282] (13) | 8% [114] | smem 53% |

### d=256, causal, S=32k: per-unit util vs SOL (SOL = min clocks per 128x128 step per SM at the unit's peak; util = SOL / measured step clocks; ncu % in parentheses)

| config | step clk | tensor | mufu | issue | fma | alu | smem | tmem | l2 | dram | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `corralways_p64_half_pf_benc_cf0h0_cga2` | 1652 | 31% [512] | 15% [256] | 29% [484] | 23% [384] | 2% [30] | 47% [776] | 41% [674] | 17% [287] | 3% [57] | smem 47% |
| `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1670 | 32% [528] | 15% [256] | 29% [484] | 23% [384] | 2% [30] | 47% [792] | 40% [674] | 17% [287] | 3% [57] | smem 47% |
| `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1332 | 38% [512] | 19% [256] | 25% [340] | 10% [128] | 2% [30] | 58% [776] | 12% [162] | 22% [287] | 4% [57] | smem 58% |
| `corrdefault_p0_half_pf_prod` | 1435 | 36% [512] | 18% [256] | 24% [340] | 9% [128] | 2% [30] | 90% [1296] | 11% [162] | 40% [574] | 4% [57] | smem 90% |
| `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1488 | 34% [512] | 17% [256] | 23% [340] | 9% [128] | 2% [30] | 52% [776] | 11% [162] | 19% [287] | 4% [57] | smem 52% |
| `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1475 | 36% [528] | 17% [256] | 23% [340] | 9% [128] | 2% [30] | 54% [792] | 11% [162] | 19% [287] | 4% [57] | smem 54% |
| `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1524 | 34% [512] | 17% [256] | 22% [340] | 8% [128] | 2% [30] | 51% [776] | 11% [162] | 19% [287] | 4% [57] | smem 51% |
| `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1498 | 35% [528] | 17% [256] | 23% [340] | 9% [128] | 2% [30] | 53% [792] | 11% [162] | 19% [287] | 4% [57] | smem 53% |

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



### d=256, none mask, scheduler natural

| S | config | ncu µs | TF/s | clk GHz | SM active % | tensor pipe % | UTCQMMA fp8 % (realtime) | MUFU (XU) % | FMA % | ALU % | TMEM instr % | issue % | IPC | warp lat cyc | stall long-sb | stall wait | stall short-sb | stall barrier | stall math-throttle | stall mio-throttle | L1/SMEM % | L2 % | L2 hit % | DRAM % | DRAM GB/s | regs | SMEM KB | waves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16k | `corralways_p64_half_pf_benc_cf0h0_cga2` | 1722 | 5107 | 2.30 | 98.5 | 31.8 | – | 16.1 | 33.7 | 8.4 | 2.6 | 37.2 | 1.49 | 8.0 | 4.52 | 0.75 | 1.08 | 0.00 | 0.13 | 0.01 | 32.3 | 22.7 | 93.3 | – | 251 | 168 | 139 | 18.96 |
| 16k | `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 1750 | 5026 | 2.31 | 98.5 | 32.2 | – | 15.9 | 25.3 | 7.9 | 2.7 | 31.6 | 1.27 | 9.4 | 5.69 | 0.85 | 1.28 | 0.00 | 0.10 | 0.01 | 32.2 | 23.1 | 92.7 | – | 248 | 168 | 140 | 18.96 |
| 16k | `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 1110 | 7922 | 2.15 | 98.2 | 53.0 | – | 26.9 | 28.9 | 9.6 | 1.0 | 42.8 | 1.71 | 7.0 | 4.40 | 0.68 | 0.36 | 0.00 | 0.01 | 0.01 | 53.7 | 26.3 | 92.8 | – | 390 | 168 | 139 | 18.96 |
| 16k | `corrdefault_p0_half_pf_prod` | 1221 | 7206 | 2.20 | 97.7 | 47.5 | – | 24.1 | 25.9 | 8.5 | 0.9 | 38.6 | 1.55 | 7.8 | 4.87 | 0.62 | 0.75 | 0.00 | 0.01 | 0.03 | 71.1 | 39.2 | 90.2 | – | 355 | 168 | 204 | 18.96 |
| 16k | `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 1199 | 7334 | 2.22 | 97.1 | 48.1 | – | 24.4 | 26.8 | 12.7 | 1.0 | 41.5 | 1.66 | 7.3 | 4.58 | 0.71 | 0.36 | 0.00 | 0.05 | 0.02 | 48.2 | 28.2 | 96.2 | – | 361 | 168 | 139 | 18.96 |
| 16k | `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 1202 | 7317 | 2.23 | 98.1 | 48.8 | – | 24.0 | 12.9 | 12.0 | 0.9 | 32.9 | 1.31 | 9.1 | 6.32 | 0.77 | 0.47 | 0.01 | 0.05 | 0.01 | 48.6 | 26.4 | 90.8 | – | 360 | 168 | 140 | 18.96 |
| 16k | `corrnever_p64_half_pf_benc_cf0h0_cga2` | 1191 | 7388 | 2.22 | 97.0 | 48.6 | – | 24.7 | 26.9 | 11.7 | 0.8 | 39.6 | 1.58 | 7.6 | 5.06 | 0.62 | 0.36 | 0.00 | 0.04 | 0.01 | 48.6 | 28.1 | 87.9 | – | 363 | 168 | 139 | 18.96 |
| 16k | `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 1200 | 7332 | 2.23 | 96.6 | 49.6 | – | 24.4 | 13.0 | 11.1 | 0.8 | 30.9 | 1.24 | 9.7 | 6.90 | 0.66 | 0.47 | 0.00 | 0.04 | 0.02 | 48.7 | 25.2 | 101.8 | – | 361 | 168 | 140 | 18.96 |



### d=256, causal mask, scheduler lpt

| S | config | ncu µs | TF/s | clk GHz | SM active % | tensor pipe % | UTCQMMA fp8 % (realtime) | MUFU (XU) % | FMA % | ALU % | TMEM instr % | issue % | IPC | warp lat cyc | stall long-sb | stall wait | stall short-sb | stall barrier | stall math-throttle | stall mio-throttle | L1/SMEM % | L2 % | L2 hit % | DRAM % | DRAM GB/s | regs | SMEM KB | waves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16k | `corralways_p64_half_pf_benc_cf0h0_cga2` | 902 | 4874 | 2.29 | 99.4 | 30.8 | – | 15.7 | 32.5 | 6.5 | 2.5 | 36.5 | 1.46 | 8.2 | 4.79 | 0.78 | 1.00 | 0.01 | 0.11 | 0.01 | 31.6 | 23.7 | 87.8 | – | 805 | 168 | 139 | 18.96 |
| 16k | `corralways_p64_half_pf_benc_cf0h0rs1_cga2` | 911 | 4829 | 2.29 | 99.4 | 31.4 | – | 15.5 | 24.4 | 6.2 | 2.6 | 31.3 | 1.25 | 9.6 | 5.87 | 0.89 | 1.19 | 0.01 | 0.09 | 0.02 | 31.7 | 24.1 | 85.7 | – | 791 | 168 | 140 | 18.96 |
| 16k | `corrdefault_p0_half_pf_benc_cf0h0_cga2` | 701 | 6276 | 2.21 | 98.6 | 41.4 | – | 21.0 | 22.8 | 8.4 | 0.8 | 34.6 | 1.38 | 8.7 | 6.12 | 0.69 | 0.35 | 0.01 | 0.01 | 0.01 | 42.1 | 16.6 | 75.8 | – | 1020 | 168 | 139 | 18.96 |
| 16k | `corrdefault_p0_half_pf_prod` | 744 | 5910 | 2.24 | 98.6 | 38.3 | – | 19.4 | 20.9 | 7.6 | 0.8 | 32.0 | 1.28 | 9.4 | 6.56 | 0.66 | 0.74 | 0.00 | 0.01 | 0.03 | 57.8 | 26.1 | 84.3 | – | 959 | 168 | 204 | 18.96 |
| 16k | `corrdefault_p64_half_pf_benc_cf0h0_cga2` | 769 | 5721 | 2.23 | 99.6 | 37.0 | – | 18.8 | 20.4 | 7.9 | 0.7 | 32.8 | 1.30 | 9.2 | 6.61 | 0.69 | 0.33 | 0.01 | 0.02 | 0.01 | 38.0 | 14.1 | 71.7 | – | 944 | 168 | 139 | 18.96 |
| 16k | `corrdefault_p64_half_pf_benc_cf0h0rs1_cga2` | 761 | 5777 | 2.24 | 98.2 | 39.0 | – | 19.3 | 9.9 | 7.7 | 0.8 | 26.5 | 1.07 | 11.3 | 8.30 | 0.78 | 0.42 | 0.01 | 0.03 | 0.03 | 38.9 | 13.7 | 76.3 | – | 956 | 168 | 140 | 18.96 |
| 16k | `corrnever_p64_half_pf_benc_cf0h0_cga2` | 783 | 5617 | 2.24 | 99.7 | 36.1 | – | 18.4 | 20.2 | 9.6 | 0.6 | 30.7 | 1.23 | 9.7 | 7.29 | 0.63 | 0.35 | 0.01 | 0.05 | 0.01 | 37.2 | 13.7 | 71.4 | – | 927 | 168 | 139 | 18.96 |
| 16k | `corrnever_p64_half_pf_benc_cf0h0rs1_cga2` | 773 | 5690 | 2.25 | 99.0 | 37.9 | – | 18.7 | 10.1 | 9.4 | 0.6 | 24.9 | 1.00 | 12.1 | 9.43 | 0.68 | 0.43 | 0.01 | 0.05 | 0.02 | 38.2 | 13.1 | 77.4 | – | 940 | 168 | 140 | 18.96 |



## PerfSim (GR100, B=1 H=1 S=4096 no mask, SSAF route)

_pending: traces captured with perfsim/capture.sh, submitted with perfsim/submit.sh; run perfsim/summarize.py --out results/perfsim_RESULTS.md and pass the JSON to --perfsim._

## Files

- `results/*.jsonl` -- bench.py records (one per config x S x mask: all burst samples, clocks/power per round, kernel CFG/kmod flags, validation, rescale emulation)
- `results/ncu_table.{csv,json}` -- ncu_table.py over run_ncu.sh reports (+ `.raw.csv` / `.sass.csv` exports per report)
- `results/unit_sol.{json,md}` -- unit_sol.py
- `results/perfsim_RESULTS.{md,json}` -- perfsim/summarize.py
