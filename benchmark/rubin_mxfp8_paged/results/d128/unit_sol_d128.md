## Per-unit SOL model (per 128x128 step per SM)

Constants: tc_mac_per_clk = 16384 (PerfSim gr100 math-SOL (17408 cyc = 32 kv steps x 544) + silicon 6.9 PF/s @ 45 % tensor-active; Blackwell = 8192); xu_lanes_per_clk = 32 (ncu-consistent (fp32 EX2 at 62-68 % MUFU, 770 clk/step); CUDA guide table says 16 for exp2 on cc 10.x); issue_per_clk = 4 (4 SMSP schedulers x 1 warp-instruction/clk); fma_winst_per_clk = 2 (fmaheavy + fmalite, 16 lanes each per SMSP -> 2 clk per warp-instruction per pipe); alu_winst_per_clk = 2 (16 lanes/clk/SMSP); smem_bytes_per_clk = 128 (32 banks x 4 B (plan.md); TMA writes + UMMA operand reads); tmem_bytes_per_clk = 512 (ASSUMED 128 lanes x 32 bit per clock for tcgen05.ld/st); l2_bytes_per_clk_sm = 64 (ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present); dram_bytes_per_s = 8e+12 (ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present)

Records without an ncu row use the ncu-derived L2 / DRAM peaks of the same head dim: d128: L2 117 B/clk/SM, DRAM 8.0 TB/s

### corralways_p64_half_pf_benc_cf1h1 d128 S=8192 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 137.0 us @ 2364 MHz, 308 steps/SM -> **1051 clk per 128x128 step**; MMA util 25.9 %; binding unit by SOL: tmem (40 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 25.9 % | 25.9 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 24.4 % | 24.4 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 412 | 39.2 % | 39.2 % |  | 1646 warp-instr / 4 (static estimate) |
| fma | 256 | 24.4 % | 24.4 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.9 % | 2.9 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 24.4 % | 24.4 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 24.7 % | 24.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 32.2 % | 32.2 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 418 | 39.8 % | 39.8 % |  | 209 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 6.9 % | 6.9 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 113 | 10.8 % | 10.8 % |  | 118 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf1h1 d128 S=16384 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 464.5 us @ 2364 MHz, 1223 steps/SM -> **898 clk per 128x128 step**; MMA util 30.3 %; binding unit by SOL: issue (52 %); ncu: 468.2 us @ 2.30 GHz, SM-active 98.2 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 30.3 % | 30.9 % | 32.1 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 28.5 % | 29.0 % | 30.8 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 465 | 51.8 % | 52.8 % | 44.6 | 1862 warp-instr / 4 (ncu SASS executed); of which sync/branch 351 |
| fma | 318 | 35.5 % | 36.1 % | 32.8 | 637 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 110 | 12.2 % | 12.4 % | 12.5 | 219 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 14.8 % | 15.1 % |  | 266 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 266 | 29.7 % | 30.2 % | 30.8 | 266 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 37.6 % | 38.3 % | 31.1 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 418 | 46.6 % | 47.4 % | 2.9 | 209 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 8.0 % | 8.2 % | 12.1 | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 6.3 % | 6.5 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.4 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 408 M vs SASS-page executed 492 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corralways_p64_half_pf_benc_cf1h1 d128 S=32768 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 1741.8 us @ 2364 MHz, 4873 steps/SM -> **845 clk per 128x128 step**; MMA util 32.2 %; binding unit by SOL: tmem (49 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 32.2 % | 32.2 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 30.3 % | 30.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 412 | 48.7 % | 48.7 % |  | 1646 warp-instr / 4 (static estimate) |
| fma | 256 | 30.3 % | 30.3 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.6 % | 3.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 30.3 % | 30.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 30.8 % | 30.8 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 40.0 % | 40.0 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 418 | 49.5 % | 49.5 % |  | 209 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 8.5 % | 8.5 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 3.4 % | 3.4 % |  | 472 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf1h1 d128 S=8192 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 249.0 us @ 2364 MHz, 607 steps/SM -> **970 clk per 128x128 step**; MMA util 28.0 %; binding unit by SOL: tmem (43 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 28.0 % | 28.0 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 26.4 % | 26.4 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 412 | 42.4 % | 42.4 % |  | 1646 warp-instr / 4 (static estimate) |
| fma | 256 | 26.4 % | 26.4 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.1 % | 3.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 26.4 % | 26.4 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 26.8 % | 26.8 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 34.8 % | 34.8 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 418 | 43.1 % | 43.1 % |  | 209 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 7.4 % | 7.4 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 5.9 % | 5.9 % |  | 118 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf1h1 d128 S=16384 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 951.4 us @ 2364 MHz, 2427 steps/SM -> **927 clk per 128x128 step**; MMA util 29.4 %; binding unit by SOL: issue (49 %); ncu: 952.1 us @ 2.32 GHz, SM-active 94.6 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 29.4 % | 31.0 % | 31.6 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 27.6 % | 29.2 % | 30.2 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 453 | 48.9 % | 51.6 % | 41.4 | 1811 warp-instr / 4 (ncu SASS executed); of which sync/branch 399 |
| fma | 286 | 30.9 % | 32.6 % | 32.1 | 572 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 102 | 11.0 % | 11.6 % | 10.1 | 204 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 129 | 13.9 % | 14.7 % |  | 258 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 28.1 % | 29.7 % | 30.2 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 36.5 % | 38.6 % | 29.4 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 418 | 45.1 % | 47.7 % | 2.9 | 209 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 7.8 % | 8.3 % | 10.4 | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 3.1 % | 3.3 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.4 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 748 M vs SASS-page executed 950 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corralways_p64_half_pf_benc_cf1h1 d128 S=32768 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 3594.4 us @ 2364 MHz, 9709 steps/SM -> **875 clk per 128x128 step**; MMA util 31.1 %; binding unit by SOL: tmem (48 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 31.1 % | 31.1 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 29.3 % | 29.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 412 | 47.0 % | 47.0 % |  | 1646 warp-instr / 4 (static estimate) |
| fma | 256 | 29.3 % | 29.3 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.4 % | 3.4 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 29.3 % | 29.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 29.7 % | 29.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 38.6 % | 38.6 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 418 | 47.8 % | 47.8 % |  | 209 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 8.2 % | 8.2 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 14 | 1.6 % | 1.6 % |  | 472 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d128 S=8192 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 104.3 us @ 2364 MHz, 308 steps/SM -> **800 clk per 128x128 step**; MMA util 34.0 %; binding unit by SOL: issue (42 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 34.0 % | 34.0 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 32.0 % | 32.0 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 42.4 % | 42.4 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 16.0 % | 16.0 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.7 % | 3.7 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 32.0 % | 32.0 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 32.5 % | 32.5 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 42.2 % | 42.2 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 20.2 % | 20.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 9.0 % | 9.0 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 113 | 14.1 % | 14.1 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d128 S=16384 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 340.9 us @ 2364 MHz, 1223 steps/SM -> **659 clk per 128x128 step**; MMA util 41.3 %; binding unit by SOL: issue (61 %); ncu: 343.6 us @ 2.28 GHz, SM-active 98.0 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 41.3 % | 42.1 % | 44.4 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 38.9 % | 39.7 % | 42.5 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 402 | 61.0 % | 62.3 % | 47.5 | 1608 warp-instr / 4 (ncu SASS executed); of which sync/branch 362 |
| fma | 181 | 27.5 % | 28.1 % | 21.8 | 363 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 147 | 22.3 % | 22.8 % | 16.7 | 294 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 20.2 % | 20.6 % |  | 266 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 266 | 40.4 % | 41.2 % | 42.5 | 266 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 51.3 % | 52.4 % | 42.9 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 24.6 % | 25.1 % | 1.7 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 10.9 % | 11.1 % | 16.3 | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 8.7 % | 8.8 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.8 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 314 M vs SASS-page executed 425 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_half_pf_prod d128 S=32768 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 1255.8 us @ 2364 MHz, 4873 steps/SM -> **609 clk per 128x128 step**; MMA util 44.7 %; binding unit by SOL: issue (56 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 44.7 % | 44.7 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 42.0 % | 42.0 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 55.7 % | 55.7 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 21.0 % | 21.0 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 4.9 % | 4.9 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 42.0 % | 42.0 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 42.7 % | 42.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 55.5 % | 55.5 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 26.6 % | 26.6 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.8 % | 11.8 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 4.7 % | 4.7 % |  | 472 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d128 S=8192 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 172.7 us @ 2364 MHz, 607 steps/SM -> **673 clk per 128x128 step**; MMA util 40.4 %; binding unit by SOL: issue (50 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 40.4 % | 40.4 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 38.0 % | 38.0 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 50.5 % | 50.5 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 19.0 % | 19.0 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 4.5 % | 4.5 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 38.0 % | 38.0 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 38.6 % | 38.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 50.2 % | 50.2 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 24.1 % | 24.1 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 10.7 % | 10.7 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 8.5 % | 8.5 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d128 S=16384 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 641.3 us @ 2364 MHz, 2427 steps/SM -> **625 clk per 128x128 step**; MMA util 43.5 %; binding unit by SOL: issue (58 %); ncu: 641.7 us @ 2.30 GHz, SM-active 94.9 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 43.5 % | 45.9 % | 47.1 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 41.0 % | 43.2 % | 45.0 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 362 | 58.0 % | 61.1 % | 47.1 | 1450 warp-instr / 4 (ncu SASS executed); of which sync/branch 318 |
| fma | 151 | 24.2 % | 25.5 % | 22.7 | 302 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 129 | 20.6 % | 21.7 % | 14.4 | 258 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 129 | 20.7 % | 21.8 % |  | 258 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 41.6 % | 43.9 % | 45.0 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 54.1 % | 57.0 % | 44.0 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 25.9 % | 27.3 % | 1.7 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.6 % | 12.2 % | 14.6 | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 4.6 % | 4.8 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.0 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 572 M vs SASS-page executed 760 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_half_pf_prod d128 S=32768 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 2417.3 us @ 2364 MHz, 9709 steps/SM -> **589 clk per 128x128 step**; MMA util 46.2 %; binding unit by SOL: issue (58 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 46.2 % | 46.2 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 43.5 % | 43.5 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 57.7 % | 57.7 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 21.7 % | 21.7 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 5.1 % | 5.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 43.5 % | 43.5 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 44.2 % | 44.2 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 57.4 % | 57.4 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 27.5 % | 27.5 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 12.2 % | 12.2 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 14 | 2.4 % | 2.4 % |  | 472 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf1h1 d128 S=8192 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 103.8 us @ 2364 MHz, 308 steps/SM -> **797 clk per 128x128 step**; MMA util 34.1 %; binding unit by SOL: issue (43 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 34.1 % | 34.1 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 32.1 % | 32.1 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 42.6 % | 42.6 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 16.1 % | 16.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.8 % | 3.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 32.1 % | 32.1 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 32.6 % | 32.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 42.4 % | 42.4 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 20.3 % | 20.3 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 9.0 % | 9.0 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 113 | 14.2 % | 14.2 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf1h1 d128 S=16384 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 332.8 us @ 2364 MHz, 1223 steps/SM -> **643 clk per 128x128 step**; MMA util 42.3 %; binding unit by SOL: issue (60 %); ncu: 336.7 us @ 2.27 GHz, SM-active 97.9 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 42.3 % | 43.2 % | 45.5 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 39.8 % | 40.6 % | 43.6 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 384 | 59.7 % | 61.0 % | 49.4 | 1537 warp-instr / 4 (ncu SASS executed); of which sync/branch 374 |
| fma | 177 | 27.6 % | 28.2 % | 22.5 | 355 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 114 | 17.7 % | 18.0 % | 18.0 | 227 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 20.7 % | 21.1 % |  | 266 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 266 | 41.4 % | 42.3 % | 43.6 | 266 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 52.5 % | 53.6 % | 43.9 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 25.2 % | 25.7 % | 1.2 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.1 % | 11.4 % | 16.8 | 8.2 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 8.9 % | 9.0 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.8 TB/s = 118 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 319 M vs SASS-page executed 406 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p64_half_pf_benc_cf1h1 d128 S=32768 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 1219.6 us @ 2364 MHz, 4873 steps/SM -> **592 clk per 128x128 step**; MMA util 46.0 %; binding unit by SOL: issue (57 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 46.0 % | 46.0 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 43.3 % | 43.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 57.4 % | 57.4 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 21.6 % | 21.6 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 5.1 % | 5.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 43.3 % | 43.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 43.9 % | 43.9 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 57.1 % | 57.1 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 27.4 % | 27.4 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 12.2 % | 12.2 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 4.8 % | 4.8 % |  | 472 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf1h1 d128 S=8192 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 181.4 us @ 2364 MHz, 607 steps/SM -> **707 clk per 128x128 step**; MMA util 38.5 %; binding unit by SOL: issue (48 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 38.5 % | 38.5 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 36.2 % | 36.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 48.0 % | 48.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 18.1 % | 18.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 4.2 % | 4.2 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 36.2 % | 36.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 36.8 % | 36.8 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 47.8 % | 47.8 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 22.9 % | 22.9 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 10.2 % | 10.2 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 8.1 % | 8.1 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf1h1 d128 S=16384 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 673.1 us @ 2364 MHz, 2427 steps/SM -> **656 clk per 128x128 step**; MMA util 41.5 %; binding unit by SOL: issue (55 %); ncu: 673.6 us @ 2.29 GHz, SM-active 94.6 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 41.5 % | 43.9 % | 45.2 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 39.1 % | 41.3 % | 43.2 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 363 | 55.4 % | 58.6 % | 45.6 | 1452 warp-instr / 4 (ncu SASS executed); of which sync/branch 339 |
| fma | 147 | 22.4 % | 23.7 % | 22.2 | 294 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 127 | 19.4 % | 20.5 % | 14.8 | 255 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 129 | 19.7 % | 20.8 % |  | 258 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 39.7 % | 41.9 % | 43.2 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 51.6 % | 54.5 % | 42.1 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 24.7 % | 26.1 % | 1.2 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.0 % | 11.6 % | 14.3 | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 4.4 % | 4.6 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.2 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 575 M vs SASS-page executed 761 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p64_half_pf_benc_cf1h1 d128 S=32768 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 2516.2 us @ 2364 MHz, 9709 steps/SM -> **613 clk per 128x128 step**; MMA util 44.4 %; binding unit by SOL: issue (55 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 44.4 % | 44.4 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 41.8 % | 41.8 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 55.4 % | 55.4 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 20.9 % | 20.9 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 4.9 % | 4.9 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 41.8 % | 41.8 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 42.4 % | 42.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 55.2 % | 55.2 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 26.4 % | 26.4 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.8 % | 11.8 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 14 | 2.3 % | 2.3 % |  | 472 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf1h1 d128 S=8192 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 103.4 us @ 2364 MHz, 308 steps/SM -> **793 clk per 128x128 step**; MMA util 34.3 %; binding unit by SOL: issue (43 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 34.3 % | 34.3 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 32.3 % | 32.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 42.8 % | 42.8 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 16.1 % | 16.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.8 % | 3.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 32.3 % | 32.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 32.8 % | 32.8 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 42.6 % | 42.6 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 20.4 % | 20.4 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 9.1 % | 9.1 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 113 | 14.3 % | 14.3 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf1h1 d128 S=16384 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 332.7 us @ 2364 MHz, 1223 steps/SM -> **643 clk per 128x128 step**; MMA util 42.3 %; binding unit by SOL: issue (61 %); ncu: 336.9 us @ 2.27 GHz, SM-active 97.8 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 42.3 % | 43.2 % | 45.5 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 39.8 % | 40.7 % | 43.6 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 390 | 60.7 % | 62.1 % | 47.7 | 1561 warp-instr / 4 (ncu SASS executed); of which sync/branch 358 |
| fma | 175 | 27.3 % | 27.9 % | 22.6 | 351 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 139 | 21.7 % | 22.2 % | 17.6 | 279 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 20.7 % | 21.2 % |  | 266 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 266 | 41.4 % | 42.3 % | 43.6 | 266 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 52.6 % | 53.7 % | 43.8 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 25.2 % | 25.8 % | 1.0 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.2 % | 11.4 % | 16.8 | 8.2 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 8.9 % | 9.1 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.8 TB/s = 118 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 309 M vs SASS-page executed 413 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrnever_p64_half_pf_benc_cf1h1 d128 S=32768 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 1221.1 us @ 2364 MHz, 4873 steps/SM -> **592 clk per 128x128 step**; MMA util 45.9 %; binding unit by SOL: issue (57 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 45.9 % | 45.9 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 43.2 % | 43.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 57.3 % | 57.3 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 21.6 % | 21.6 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 5.1 % | 5.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 43.2 % | 43.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 43.9 % | 43.9 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 57.1 % | 57.1 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 27.4 % | 27.4 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 12.2 % | 12.2 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 4.8 % | 4.8 % |  | 472 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf1h1 d128 S=8192 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 169.3 us @ 2364 MHz, 607 steps/SM -> **660 clk per 128x128 step**; MMA util 41.2 %; binding unit by SOL: issue (51 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 41.2 % | 41.2 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 38.8 % | 38.8 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 51.5 % | 51.5 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 19.4 % | 19.4 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 4.5 % | 4.5 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 38.8 % | 38.8 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 39.4 % | 39.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 51.2 % | 51.2 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 24.6 % | 24.6 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 10.9 % | 10.9 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 8.7 % | 8.7 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf1h1 d128 S=16384 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 625.4 us @ 2364 MHz, 2427 steps/SM -> **609 clk per 128x128 step**; MMA util 44.7 %; binding unit by SOL: issue (58 %); ncu: 627.3 us @ 2.29 GHz, SM-active 94.4 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 44.7 % | 47.3 % | 48.7 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 42.0 % | 44.5 % | 46.5 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 352 | 57.9 % | 61.3 % | 47.3 | 1410 warp-instr / 4 (ncu SASS executed); of which sync/branch 325 |
| fma | 145 | 23.8 % | 25.2 % | 23.9 | 290 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 119 | 19.6 % | 20.7 % | 15.5 | 238 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 129 | 21.2 % | 22.4 % |  | 258 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 42.7 % | 45.2 % | 46.5 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 55.5 % | 58.8 % | 45.3 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 26.6 % | 28.2 % | 1.1 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 11.8 % | 12.5 % | 16.0 | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 29 | 4.7 % | 5.0 % |  | 236 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.0 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 554 M vs SASS-page executed 739 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrnever_p64_half_pf_benc_cf1h1 d128 S=32768 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 2364.6 us @ 2364 MHz, 9709 steps/SM -> **576 clk per 128x128 step**; MMA util 47.2 %; binding unit by SOL: issue (59 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 47.2 % | 47.2 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 44.5 % | 44.5 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 59.0 % | 59.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 22.2 % | 22.2 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 5.2 % | 5.2 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 44.5 % | 44.5 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 45.2 % | 45.2 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 58.7 % | 58.7 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 28.1 % | 28.1 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 12.5 % | 12.5 % |  | 8.2 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 14 | 2.5 % | 2.5 % |  | 472 MB whole kernel / 8.0 TB/s |
