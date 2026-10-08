## Per-unit SOL model (per 128x128 step per SM)

Constants: tc_mac_per_clk = 16384 (PerfSim gr100 math-SOL (17408 cyc = 32 kv steps x 544) + silicon 6.9 PF/s @ 45 % tensor-active; Blackwell = 8192); xu_lanes_per_clk = 32 (ncu-consistent (fp32 EX2 at 62-68 % MUFU, 770 clk/step); CUDA guide table says 16 for exp2 on cc 10.x); issue_per_clk = 4 (4 SMSP schedulers x 1 warp-instruction/clk); fma_winst_per_clk = 2 (fmaheavy + fmalite, 16 lanes each per SMSP -> 2 clk per warp-instruction per pipe); alu_winst_per_clk = 2 (16 lanes/clk/SMSP); smem_bytes_per_clk = 128 (32 banks x 4 B (plan.md); TMA writes + UMMA operand reads); tmem_bytes_per_clk = 512 (ASSUMED 128 lanes x 32 bit per clock for tcgen05.ld/st); l2_bytes_per_clk_sm = 64 (ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present); dram_bytes_per_s = 8e+12 (ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present)

### corrdefault_p0_f32_scale_prod d128 S=8192 causal (cta_mma 2, TILES_Q 2, f32 exp, rowsum-MMA 1, corr default, paged 0)
measured 158.6 us @ 2382 MHz, 308 steps/SM -> **1226 clk per 128x128 step**; MMA util 22.2 %; binding unit by SOL: xu_inst (42 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 22.2 % | 22.2 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 512 | 41.8 % | 41.8 % |  | 16384 exp fp32 / 32 lanes |
| issue | 462 | 37.7 % | 37.7 % |  | 1850 warp-instr / 4 (static estimate) |
| fma | 256 | 20.9 % | 20.9 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.4 % | 2.4 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 128 | 10.4 % | 10.4 % |  | 256 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 516 | 42.1 % | 42.1 % |  | 516 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 27.6 % | 27.6 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 13.2 % | 13.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 132 | 10.8 % | 10.8 % |  | 8.2 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 114 | 9.3 % | 9.3 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_f32_scale_prod d128 S=8192 none (cta_mma 2, TILES_Q 2, f32 exp, rowsum-MMA 1, corr default, paged 0)
measured 217.0 us @ 2364 MHz, 607 steps/SM -> **845 clk per 128x128 step**; MMA util 32.2 %; binding unit by SOL: xu_inst (61 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 32.2 % | 32.2 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 512 | 60.6 % | 60.6 % |  | 16384 exp fp32 / 32 lanes |
| issue | 462 | 54.7 % | 54.7 % |  | 1850 warp-instr / 4 (static estimate) |
| fma | 256 | 30.3 % | 30.3 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.5 % | 3.5 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 128 | 15.1 % | 15.1 % |  | 256 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 516 | 61.0 % | 61.0 % |  | 516 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 40.0 % | 40.0 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 19.2 % | 19.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 132 | 15.6 % | 15.6 % |  | 8.2 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 57 | 6.8 % | 6.8 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d128 S=8192 causal (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 128.6 us @ 2382 MHz, 308 steps/SM -> **994 clk per 128x128 step**; MMA util 27.4 %; binding unit by SOL: issue (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 27.4 % | 27.4 % |  | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 25.8 % | 25.8 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 34.2 % | 34.2 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 12.9 % | 12.9 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 3.0 % | 3.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 25.8 % | 25.8 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 26.2 % | 26.2 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 338 | 34.0 % | 34.0 % |  | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 16.3 % | 16.3 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 132 | 13.3 % | 13.3 % |  | 8.2 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 114 | 11.5 % | 11.5 % |  | 118 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d128 S=8192 none (cta_mma 2, TILES_Q 2, f16 exp, rowsum-MMA 1, corr default, paged 0)
measured 172.2 us @ 2364 MHz, 607 steps/SM -> **671 clk per 128x128 step**; MMA util 40.6 %; binding unit by SOL: issue (55 %); ncu: 174.0 us @ 2.27 GHz, SM-active 92.2 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 272 | 40.6 % | 44.0 % | 45.3 | BMM1 2048k + BMM2 2048k + rowsum 256k MAC / 16384 |
| mufu | 256 | 38.2 % | 41.4 % | 43.3 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 367 | 54.7 % | 59.4 % | 45.8 | 1468 warp-instr / 4 (ncu SASS executed); of which sync/branch 321 |
| fma | 154 | 23.0 % | 25.0 % | 22.1 | 309 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 131 | 19.6 % | 21.3 % | 14.2 | 263 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 19.4 % | 21.0 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 38.8 % | 42.1 % | 43.3 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 338 | 50.4 % | 54.7 % | 41.2 | reads 34 KiB (Q 16 + K/2 8 + V/2 8 + ones 2) + TMA writes 8.2 KiB / 128 B |
| tmem | 162 | 24.2 % | 26.2 % | 1.7 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 72 | 10.7 % | 11.6 % | 12.7 | 8.2 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 8.6 % | 9.3 % |  | 118 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.8 TB/s = 118 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 145 M vs SASS-page executed 192 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_f32_scale_prod d256 S=8192 causal (cta_mma 1, TILES_Q 1, f32 exp, rowsum-MMA 0, corr default, paged 0)
measured 214.5 us @ 2364 MHz, 308 steps/SM -> **1646 clk per 128x128 step**; MMA util 31.1 %; binding unit by SOL: smem (79 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 31.1 % | 31.1 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 512 | 31.1 % | 31.1 % |  | 16384 exp fp32 / 32 lanes |
| issue | 462 | 28.1 % | 28.1 % |  | 1850 warp-instr / 4 (static estimate) |
| fma | 256 | 15.6 % | 15.6 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.8 % | 1.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 128 | 7.8 % | 7.8 % |  | 256 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 516 | 31.4 % | 31.4 % |  | 516 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 1296 | 78.7 % | 78.7 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 128 B |
| tmem | 162 | 9.8 % | 9.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 1056 | 64.2 % | 64.2 % |  | 66.0 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 226 | 13.7 % | 13.7 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_f32_scale_prod d256 S=8192 none (cta_mma 1, TILES_Q 1, f32 exp, rowsum-MMA 0, corr default, paged 0)
measured 371.9 us @ 2382 MHz, 607 steps/SM -> **1460 clk per 128x128 step**; MMA util 35.1 %; binding unit by SOL: smem (89 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 35.1 % | 35.1 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 512 | 35.1 % | 35.1 % |  | 16384 exp fp32 / 32 lanes |
| issue | 462 | 31.7 % | 31.7 % |  | 1850 warp-instr / 4 (static estimate) |
| fma | 256 | 17.5 % | 17.5 % |  | 512 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 128 | 8.8 % | 8.8 % |  | 256 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 516 | 35.3 % | 35.3 % |  | 516 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 1296 | 88.8 % | 88.8 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 128 B |
| tmem | 162 | 11.1 % | 11.1 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 1056 | 72.3 % | 72.3 % |  | 66.0 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 116 | 7.9 % | 7.9 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d256 S=8192 causal (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 202.4 us @ 2364 MHz, 308 steps/SM -> **1553 clk per 128x128 step**; MMA util 33.0 %; binding unit by SOL: smem (83 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 33.0 % | 33.0 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 16.5 % | 16.5 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 21.9 % | 21.9 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.2 % | 8.2 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.9 % | 1.9 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 16.5 % | 16.5 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 16.7 % | 16.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 1296 | 83.5 % | 83.5 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 128 B |
| tmem | 162 | 10.4 % | 10.4 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 1056 | 68.0 % | 68.0 % |  | 66.0 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 226 | 14.6 % | 14.6 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d256 S=8192 none (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 357.6 us @ 2382 MHz, 607 steps/SM -> **1404 clk per 128x128 step**; MMA util 36.5 %; binding unit by SOL: smem (92 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 36.5 % | 36.5 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 18.2 % | 18.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.2 % | 24.2 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.1 % | 9.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.2 % | 18.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.5 % | 18.5 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 1296 | 92.3 % | 92.3 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 128 B |
| tmem | 162 | 11.5 % | 11.5 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 1056 | 75.2 % | 75.2 % |  | 66.0 KiB K/V(+SF) per step per SM / 64 B/clk/SM |
| dram | 116 | 8.2 % | 8.2 % |  | 236 MB whole kernel / 8.0 TB/s |
