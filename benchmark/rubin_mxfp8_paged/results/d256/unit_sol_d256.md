## Per-unit SOL model (per 128x128 step per SM)

Constants: tc_mac_per_clk = 16384 (PerfSim gr100 math-SOL (17408 cyc = 32 kv steps x 544) + silicon 6.9 PF/s @ 45 % tensor-active; Blackwell = 8192); xu_lanes_per_clk = 32 (ncu-consistent (fp32 EX2 at 62-68 % MUFU, 770 clk/step); CUDA guide table says 16 for exp2 on cc 10.x); issue_per_clk = 4 (4 SMSP schedulers x 1 warp-instruction/clk); fma_winst_per_clk = 2 (fmaheavy + fmalite, 16 lanes each per SMSP -> 2 clk per warp-instruction per pipe); alu_winst_per_clk = 2 (16 lanes/clk/SMSP); smem_bytes_per_clk = 256 (UTCQMMA operand feed at the 16384 MAC/clk tensor rate (LSU path alone is 128; silicon sustains 138 at d256 cga1); TMA writes + UMMA operand reads); tmem_bytes_per_clk = 512 (ASSUMED 128 lanes x 32 bit per clock for tcgen05.ld/st); l2_bytes_per_clk_sm = 64 (ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present); dram_bytes_per_s = 8e+12 (ASSUMED fallback; replaced by the ncu-derived peak when an ncu row is present)

Records without an ncu row use the ncu-derived L2 / DRAM peaks of the same head dim: d256: L2 118 B/clk/SM, DRAM 8.0 TB/s

### corralways_p64_half_pf_benc_cf0h0_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr always, paged 64)
measured 254.8 us @ 2364 MHz, 308 steps/SM -> **1954 clk per 128x128 step**; MMA util 26.2 %; binding unit by SOL: tmem (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 26.2 % | 26.2 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 13.1 % | 13.1 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 24.7 % | 24.7 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 19.6 % | 19.6 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.5 % | 1.5 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 13.1 % | 13.1 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 13.3 % | 13.3 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 19.9 % | 19.9 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 34.5 % | 34.5 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 14.7 % | 14.7 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 11.6 % | 11.6 % |  | 236 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr always, paged 64)
measured 902.1 us @ 2364 MHz, 1223 steps/SM -> **1744 clk per 128x128 step**; MMA util 29.4 %; binding unit by SOL: issue (47 %); ncu: 902.3 us @ 2.29 GHz, SM-active 99.4 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 29.4 % | 29.5 % | 30.8 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 14.7 % | 14.8 % | 15.7 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 817 | 46.8 % | 47.1 % | 36.5 | 3267 warp-instr / 4 (ncu SASS executed); of which sync/branch 729 |
| fma | 599 | 34.3 % | 34.5 % | 32.5 | 1197 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 308 | 17.6 % | 17.7 % | 6.5 | 615 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 7.6 % | 7.7 % |  | 267 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 15.1 % | 15.1 % | 15.7 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 22.3 % | 22.4 % | 31.6 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 38.7 % | 38.9 % | 2.5 | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 16.5 % | 16.6 % | 23.7 | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 114 | 6.5 % | 6.6 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.2 TB/s = 118 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 646 M vs SASS-page executed 863 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corralways_p64_half_pf_benc_cf0h0_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr always, paged 64)
measured 3405.8 us @ 2364 MHz, 4873 steps/SM -> **1652 clk per 128x128 step**; MMA util 31.0 %; binding unit by SOL: tmem (41 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 31.0 % | 31.0 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 15.5 % | 15.5 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 29.3 % | 29.3 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 23.2 % | 23.2 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.8 % | 1.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 15.5 % | 15.5 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 15.7 % | 15.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 23.5 % | 23.5 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 40.8 % | 40.8 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 17.4 % | 17.4 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 3.5 % | 3.5 % |  | 944 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr always, paged 64)
measured 473.5 us @ 2364 MHz, 607 steps/SM -> **1845 clk per 128x128 step**; MMA util 27.8 %; binding unit by SOL: tmem (37 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 27.8 % | 27.8 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 13.9 % | 13.9 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 26.2 % | 26.2 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 20.8 % | 20.8 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.6 % | 1.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 13.9 % | 13.9 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 14.1 % | 14.1 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 21.0 % | 21.0 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 36.5 % | 36.5 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 15.6 % | 15.6 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 6.2 % | 6.2 % |  | 236 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr always, paged 64)
measured 1717.0 us @ 2364 MHz, 2427 steps/SM -> **1672 clk per 128x128 step**; MMA util 30.6 %; binding unit by SOL: issue (49 %); ncu: 1722.4 us @ 2.30 GHz, SM-active 98.5 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 30.6 % | 31.1 % | 31.8 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 15.3 % | 15.5 % | 16.1 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 816 | 48.8 % | 49.5 % | 37.2 | 3262 warp-instr / 4 (ncu SASS executed); of which sync/branch 764 |
| fma | 578 | 34.6 % | 35.1 % | 33.7 | 1156 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 314 | 18.8 % | 19.1 % | 8.4 | 629 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 7.8 % | 7.9 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 15.6 % | 15.8 % | 16.1 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 23.2 % | 23.6 % | 32.3 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 40.3 % | 40.9 % | 2.6 | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 288 | 17.2 % | 17.5 % | 22.7 | 33.0 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 3.4 % | 3.5 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.3 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 1255 M vs SASS-page executed 1710 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corralways_p64_half_pf_benc_cf0h0_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr always, paged 64)
measured 6681.0 us @ 2364 MHz, 9709 steps/SM -> **1627 clk per 128x128 step**; MMA util 31.5 %; binding unit by SOL: tmem (41 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 31.5 % | 31.5 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 15.7 % | 15.7 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 29.7 % | 29.7 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 23.6 % | 23.6 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.8 % | 1.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 15.7 % | 15.7 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 16.0 % | 16.0 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 23.9 % | 23.9 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 41.4 % | 41.4 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 17.7 % | 17.7 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 1.8 % | 1.8 % |  | 944 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 255.8 us @ 2364 MHz, 308 steps/SM -> **1962 clk per 128x128 step**; MMA util 26.9 %; binding unit by SOL: tmem (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 26.9 % | 26.9 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 13.0 % | 13.0 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 24.6 % | 24.6 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 19.6 % | 19.6 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.5 % | 1.5 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 13.0 % | 13.0 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 13.3 % | 13.3 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 20.2 % | 20.2 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 34.3 % | 34.3 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 14.6 % | 14.6 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 11.5 % | 11.5 % |  | 236 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 908.5 us @ 2364 MHz, 1223 steps/SM -> **1756 clk per 128x128 step**; MMA util 30.1 %; binding unit by SOL: tmem (38 %); ncu: 910.8 us @ 2.29 GHz, SM-active 99.4 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 30.1 % | 30.3 % | 31.4 | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 14.6 % | 14.7 % | 15.5 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 666 | 38.0 % | 38.2 % | 31.3 | 2666 warp-instr / 4 (ncu SASS executed); of which sync/branch 507 |
| fma | 441 | 25.1 % | 25.3 % | 24.4 | 882 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 271 | 15.4 % | 15.5 % | 6.2 | 542 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 7.6 % | 7.6 % |  | 267 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 14.9 % | 15.0 % | 15.5 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 396 | 22.6 % | 22.7 % | 31.7 | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 38.4 % | 38.6 % | 2.6 | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 16.4 % | 16.5 % | 24.1 | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 114 | 6.5 % | 6.5 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.2 TB/s = 118 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 560 M vs SASS-page executed 704 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corralways_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 3443.2 us @ 2364 MHz, 4873 steps/SM -> **1670 clk per 128x128 step**; MMA util 31.6 %; binding unit by SOL: tmem (40 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 31.6 % | 31.6 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 15.3 % | 15.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 28.9 % | 28.9 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 23.0 % | 23.0 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.8 % | 1.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 15.3 % | 15.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 15.6 % | 15.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 23.7 % | 23.7 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 40.4 % | 40.4 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 17.2 % | 17.2 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 3.4 % | 3.4 % |  | 944 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 478.2 us @ 2364 MHz, 607 steps/SM -> **1863 clk per 128x128 step**; MMA util 28.3 %; binding unit by SOL: tmem (36 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 28.3 % | 28.3 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 13.7 % | 13.7 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 26.0 % | 26.0 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 20.6 % | 20.6 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.6 % | 1.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 13.7 % | 13.7 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 14.0 % | 14.0 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 21.3 % | 21.3 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 36.2 % | 36.2 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 15.4 % | 15.4 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 6.2 % | 6.2 % |  | 236 MB whole kernel / 8.0 TB/s |

### corralways_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 1742.9 us @ 2364 MHz, 2427 steps/SM -> **1697 clk per 128x128 step**; MMA util 31.1 %; binding unit by SOL: tmem (40 %); ncu: 1750.1 us @ 2.31 GHz, SM-active 98.5 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 31.1 % | 31.6 % | 32.2 | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 15.1 % | 15.3 % | 15.9 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 649 | 38.3 % | 38.8 % | 31.6 | 2598 warp-instr / 4 (ncu SASS executed); of which sync/branch 485 |
| fma | 424 | 25.0 % | 25.4 % | 25.3 | 848 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 271 | 16.0 % | 16.2 % | 7.9 | 543 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 7.7 % | 7.8 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 15.3 % | 15.6 % | 15.9 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 396 | 23.3 % | 23.7 % | 32.2 | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 39.7 % | 40.3 % | 2.7 | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 289 | 17.0 % | 17.3 % | 23.1 | 33.0 KiB K/V(+SF) per step per SM / 117 B/clk/SM |
| dram | 57 | 3.4 % | 3.4 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.3 TB/s = 117 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 1089 M vs SASS-page executed 1362 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corralways_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr always, paged 64)
measured 6806.2 us @ 2364 MHz, 9709 steps/SM -> **1657 clk per 128x128 step**; MMA util 31.9 %; binding unit by SOL: tmem (41 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 31.9 % | 31.9 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 15.4 % | 15.4 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 484 | 29.2 % | 29.2 % |  | 1934 warp-instr / 4 (static estimate) |
| fma | 384 | 23.2 % | 23.2 % |  | 768 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 1.8 % | 1.8 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 15.4 % | 15.4 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 15.7 % | 15.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 23.9 % | 23.9 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 674 | 40.7 % | 40.7 % |  | 337 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 17.3 % | 17.3 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 1.7 % | 1.7 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_benc_cf0h0_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 178.5 us @ 2364 MHz, 308 steps/SM -> **1369 clk per 128x128 step**; MMA util 37.4 %; binding unit by SOL: tensor (37 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 37.4 % | 37.4 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 18.7 % | 18.7 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.8 % | 24.8 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.3 % | 9.3 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.2 % | 2.2 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.7 % | 18.7 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 19.0 % | 19.0 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 28.3 % | 28.3 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.8 % | 11.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 21.0 % | 21.0 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 16.5 % | 16.5 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_benc_cf0h0_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 678.6 us @ 2364 MHz, 1223 steps/SM -> **1312 clk per 128x128 step**; MMA util 39.0 %; binding unit by SOL: issue (55 %); ncu: 700.8 us @ 2.21 GHz, SM-active 98.6 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 39.0 % | 39.6 % | 41.4 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 19.5 % | 19.8 % | 21.0 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 715 | 54.5 % | 55.3 % | 34.6 | 2860 warp-instr / 4 (ncu SASS executed); of which sync/branch 984 |
| fma | 347 | 26.4 % | 26.8 % | 22.8 | 693 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 294 | 22.4 % | 22.8 % | 8.4 | 589 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 10.2 % | 10.3 % |  | 266 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 20.0 % | 20.3 % | 21.0 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 29.6 % | 30.0 % | 42.1 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 12.4 % | 12.5 % | 0.8 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 279 | 21.3 % | 21.6 % | 16.6 | 33.0 KiB K/V(+SF) per step per SM / 121 B/clk/SM |
| dram | 114 | 8.7 % | 8.8 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.9 TB/s = 121 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 456 M vs SASS-page executed 756 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_half_pf_benc_cf0h0_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 2745.1 us @ 2364 MHz, 4873 steps/SM -> **1332 clk per 128x128 step**; MMA util 38.5 %; binding unit by SOL: tensor (38 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 38.5 % | 38.5 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 19.2 % | 19.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 25.5 % | 25.5 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.6 % | 9.6 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.3 % | 2.3 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 19.2 % | 19.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 19.5 % | 19.5 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 29.1 % | 29.1 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 12.2 % | 12.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 21.6 % | 21.6 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 4.3 % | 4.3 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_benc_cf0h0_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 326.7 us @ 2364 MHz, 607 steps/SM -> **1273 clk per 128x128 step**; MMA util 40.2 %; binding unit by SOL: tensor (40 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 40.2 % | 40.2 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 20.1 % | 20.1 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 26.7 % | 26.7 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 10.1 % | 10.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.4 % | 2.4 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 20.1 % | 20.1 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 20.4 % | 20.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 30.5 % | 30.5 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 12.7 % | 12.7 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 22.6 % | 22.6 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 9.0 % | 9.0 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_benc_cf0h0_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 1112.0 us @ 2364 MHz, 2427 steps/SM -> **1083 clk per 128x128 step**; MMA util 47.3 %; binding unit by SOL: issue (63 %); ncu: 1110.4 us @ 2.15 GHz, SM-active 98.2 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 47.3 % | 48.1 % | 53.0 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 23.6 % | 24.1 % | 26.9 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 687 | 63.4 % | 64.5 % | 42.8 | 2746 warp-instr / 4 (ncu SASS executed); of which sync/branch 950 |
| fma | 326 | 30.1 % | 30.7 % | 28.9 | 652 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 281 | 25.9 % | 26.4 % | 9.6 | 561 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 12.0 % | 12.2 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 24.0 % | 24.4 % | 26.9 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 35.8 % | 36.5 % | 53.7 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 15.0 % | 15.2 % | 1.0 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 275 | 25.4 % | 25.9 % | 26.3 | 33.0 KiB K/V(+SF) per step per SM / 123 B/clk/SM |
| dram | 57 | 5.3 % | 5.4 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.1 TB/s = 123 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 865 M vs SASS-page executed 1440 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_half_pf_benc_cf0h0_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 4224.2 us @ 2364 MHz, 9709 steps/SM -> **1029 clk per 128x128 step**; MMA util 49.8 %; binding unit by SOL: tensor (50 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 49.8 % | 49.8 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 24.9 % | 24.9 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 33.0 % | 33.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 12.4 % | 12.4 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.9 % | 2.9 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 24.9 % | 24.9 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 25.3 % | 25.3 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 37.7 % | 37.7 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 15.8 % | 15.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 27.9 % | 27.9 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 2.8 % | 2.8 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d256 S=8192 causal (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 188.2 us @ 2364 MHz, 308 steps/SM -> **1444 clk per 128x128 step**; MMA util 35.5 %; binding unit by SOL: smem (45 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 35.5 % | 35.5 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 17.7 % | 17.7 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 23.5 % | 23.5 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.9 % | 8.9 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.7 % | 17.7 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.0 % | 18.0 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 648 | 44.9 % | 44.9 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 256 B |
| tmem | 162 | 11.2 % | 11.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 574 | 39.8 % | 39.8 % |  | 66.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 15.7 % | 15.7 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d256 S=16384 causal (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 728.1 us @ 2364 MHz, 1223 steps/SM -> **1407 clk per 128x128 step**; MMA util 36.4 %; binding unit by SOL: issue (55 %); ncu: 744.2 us @ 2.24 GHz, SM-active 98.6 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 36.4 % | 36.9 % | 38.3 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 18.2 % | 18.5 % | 19.4 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 778 | 55.3 % | 56.0 % | 32.0 | 3110 warp-instr / 4 (ncu SASS executed); of which sync/branch 1184 |
| fma | 338 | 24.0 % | 24.4 % | 20.9 | 676 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 325 | 23.1 % | 23.5 % | 7.6 | 651 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 132 | 9.4 % | 9.5 % |  | 264 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.5 % | 18.8 % | 19.4 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 648 | 46.0 % | 46.7 % | 57.8 | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 256 B |
| tmem | 162 | 11.5 % | 11.7 % | 0.8 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 565 | 40.2 % | 40.7 % | 26.1 | 66.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 114 | 8.1 % | 8.2 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.7 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 454 M vs SASS-page executed 822 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_half_pf_prod d256 S=32768 causal (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 2958.4 us @ 2364 MHz, 4873 steps/SM -> **1435 clk per 128x128 step**; MMA util 35.7 %; binding unit by SOL: smem (45 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 35.7 % | 35.7 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 17.8 % | 17.8 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 23.7 % | 23.7 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.9 % | 8.9 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.8 % | 17.8 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.1 % | 18.1 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 648 | 45.2 % | 45.2 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 256 B |
| tmem | 162 | 11.3 % | 11.3 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 574 | 40.0 % | 40.0 % |  | 66.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 4.0 % | 4.0 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d256 S=8192 none (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 356.7 us @ 2364 MHz, 607 steps/SM -> **1390 clk per 128x128 step**; MMA util 36.8 %; binding unit by SOL: smem (47 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 36.8 % | 36.8 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 18.4 % | 18.4 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.4 % | 24.4 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.2 % | 9.2 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.2 % | 2.2 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.4 % | 18.4 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.7 % | 18.7 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 648 | 46.6 % | 46.6 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 256 B |
| tmem | 162 | 11.7 % | 11.7 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 574 | 41.3 % | 41.3 % |  | 66.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 8.3 % | 8.3 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p0_half_pf_prod d256 S=16384 none (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 1230.0 us @ 2364 MHz, 2427 steps/SM -> **1198 clk per 128x128 step**; MMA util 42.7 %; binding unit by SOL: issue (63 %); ncu: 1220.7 us @ 2.20 GHz, SM-active 97.7 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 42.7 % | 43.8 % | 47.5 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 21.4 % | 21.9 % | 24.1 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 755 | 63.0 % | 64.5 % | 38.6 | 3018 warp-instr / 4 (ncu SASS executed); of which sync/branch 1150 |
| fma | 326 | 27.2 % | 27.9 % | 25.9 | 652 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 312 | 26.0 % | 26.7 % | 8.5 | 624 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 10.9 % | 11.1 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 21.7 % | 22.2 % | 24.1 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 648 | 54.1 % | 55.4 % | 71.1 | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 256 B |
| tmem | 162 | 13.5 % | 13.8 % | 0.9 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 563 | 47.0 % | 48.1 % | 39.2 | 66.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 57 | 4.8 % | 4.9 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.0 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 875 M vs SASS-page executed 1582 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p0_half_pf_prod d256 S=32768 none (cta_mma 1, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 0)
measured 4813.8 us @ 2364 MHz, 9709 steps/SM -> **1172 clk per 128x128 step**; MMA util 43.7 %; binding unit by SOL: smem (55 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 43.7 % | 43.7 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 21.8 % | 21.8 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 29.0 % | 29.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 10.9 % | 10.9 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.6 % | 2.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 21.8 % | 21.8 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.2 % | 22.2 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 648 | 55.3 % | 55.3 % |  | reads 96 KiB (Q 32 + K/1 32 + V/1 32) + TMA writes 66.0 KiB / 256 B |
| tmem | 162 | 13.8 % | 13.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 574 | 49.0 % | 49.0 % |  | 66.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 2.5 % | 2.5 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 64)
measured 196.3 us @ 2364 MHz, 308 steps/SM -> **1506 clk per 128x128 step**; MMA util 34.0 %; binding unit by SOL: tensor (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 34.0 % | 34.0 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 17.0 % | 17.0 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 22.5 % | 22.5 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.5 % | 8.5 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.0 % | 17.0 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.3 % | 17.3 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 25.8 % | 25.8 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.8 % | 10.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.1 % | 19.1 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 15.0 % | 15.0 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 64)
measured 768.2 us @ 2364 MHz, 1223 steps/SM -> **1485 clk per 128x128 step**; MMA util 34.5 %; binding unit by SOL: issue (49 %); ncu: 768.8 us @ 2.23 GHz, SM-active 99.6 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 34.5 % | 34.6 % | 37.0 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 17.2 % | 17.3 % | 18.8 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 731 | 49.2 % | 49.4 % | 32.8 | 2923 warp-instr / 4 (ncu SASS executed); of which sync/branch 957 |
| fma | 347 | 23.3 % | 23.4 % | 20.4 | 693 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 337 | 22.7 % | 22.8 % | 7.9 | 674 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 9.0 % | 9.0 % |  | 267 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 17.7 % | 17.8 % | 18.8 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 26.1 % | 26.2 % | 38.0 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.9 % | 11.0 % | 0.7 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 280 | 18.9 % | 19.0 % | 14.1 | 33.0 KiB K/V(+SF) per step per SM / 121 B/clk/SM |
| dram | 114 | 7.7 % | 7.7 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.1 TB/s = 121 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 481 M vs SASS-page executed 772 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p64_half_pf_benc_cf0h0_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 64)
measured 3068.4 us @ 2364 MHz, 4873 steps/SM -> **1488 clk per 128x128 step**; MMA util 34.4 %; binding unit by SOL: tensor (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 34.4 % | 34.4 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 17.2 % | 17.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 22.8 % | 22.8 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.6 % | 8.6 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.2 % | 17.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.5 % | 17.5 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 26.1 % | 26.1 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.9 % | 10.9 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.3 % | 19.3 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 3.8 % | 3.8 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 64)
measured 361.9 us @ 2364 MHz, 607 steps/SM -> **1410 clk per 128x128 step**; MMA util 36.3 %; binding unit by SOL: tensor (36 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 36.3 % | 36.3 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 18.2 % | 18.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.1 % | 24.1 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.1 % | 9.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.2 % | 18.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.4 % | 18.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 27.5 % | 27.5 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.5 % | 11.5 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 20.4 % | 20.4 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 8.1 % | 8.1 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 64)
measured 1200.8 us @ 2364 MHz, 2427 steps/SM -> **1170 clk per 128x128 step**; MMA util 43.8 %; binding unit by SOL: issue (61 %); ncu: 1199.3 us @ 2.22 GHz, SM-active 97.1 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 43.8 % | 45.1 % | 48.1 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 21.9 % | 22.5 % | 24.4 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 712 | 60.9 % | 62.7 % | 41.5 | 2847 warp-instr / 4 (ncu SASS executed); of which sync/branch 936 |
| fma | 326 | 27.9 % | 28.7 % | 26.8 | 652 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 336 | 28.7 % | 29.6 % | 12.7 | 672 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 11.1 % | 11.4 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.2 % | 22.9 % | 24.4 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 33.2 % | 34.2 % | 48.2 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 13.9 % | 14.3 % | 1.0 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 281 | 24.0 % | 24.7 % | 28.2 | 33.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 57 | 4.9 % | 5.1 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.8 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 927 M vs SASS-page executed 1493 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p64_half_pf_benc_cf0h0_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr default, paged 64)
measured 4716.3 us @ 2364 MHz, 9709 steps/SM -> **1148 clk per 128x128 step**; MMA util 44.6 %; binding unit by SOL: tensor (45 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 44.6 % | 44.6 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 22.3 % | 22.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 29.6 % | 29.6 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 11.1 % | 11.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.6 % | 2.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 22.3 % | 22.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.6 % | 22.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 33.8 % | 33.8 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 14.1 % | 14.1 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 25.0 % | 25.0 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 2.5 % | 2.5 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 189.1 us @ 2364 MHz, 308 steps/SM -> **1451 clk per 128x128 step**; MMA util 36.4 %; binding unit by SOL: tensor (36 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 36.4 % | 36.4 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 17.6 % | 17.6 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 23.4 % | 23.4 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.8 % | 8.8 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.6 % | 17.6 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.9 % | 17.9 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 27.3 % | 27.3 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.2 % | 11.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.8 % | 19.8 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 15.6 % | 15.6 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 758.1 us @ 2364 MHz, 1223 steps/SM -> **1465 clk per 128x128 step**; MMA util 36.0 %; binding unit by SOL: issue (38 %); ncu: 761.3 us @ 2.24 GHz, SM-active 98.2 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 36.0 % | 36.7 % | 39.0 | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 17.5 % | 17.8 % | 19.3 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 558 | 38.1 % | 38.8 % | 26.5 | 2231 warp-instr / 4 (ncu SASS executed); of which sync/branch 694 |
| fma | 173 | 11.8 % | 12.0 % | 9.9 | 346 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 296 | 20.2 % | 20.6 % | 7.7 | 592 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 9.1 % | 9.3 % |  | 267 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 17.9 % | 18.2 % | 19.3 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 396 | 27.0 % | 27.5 % | 38.9 | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.1 % | 11.3 % | 0.8 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 281 | 19.2 % | 19.5 % | 13.7 | 33.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 114 | 7.8 % | 7.9 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.1 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 385 M vs SASS-page executed 589 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 3039.8 us @ 2364 MHz, 4873 steps/SM -> **1475 clk per 128x128 step**; MMA util 35.8 %; binding unit by SOL: tensor (36 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 35.8 % | 35.8 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 17.4 % | 17.4 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 23.0 % | 23.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.7 % | 8.7 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.4 % | 17.4 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.6 % | 17.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 26.9 % | 26.9 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.0 % | 11.0 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.5 % | 19.5 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 3.9 % | 3.9 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 363.2 us @ 2364 MHz, 607 steps/SM -> **1415 clk per 128x128 step**; MMA util 37.3 %; binding unit by SOL: tensor (37 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 37.3 % | 37.3 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 18.1 % | 18.1 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.0 % | 24.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.0 % | 9.0 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.1 % | 18.1 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.4 % | 18.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 28.0 % | 28.0 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.4 % | 11.4 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 20.3 % | 20.3 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 8.1 % | 8.1 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrdefault_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 1205.5 us @ 2364 MHz, 2427 steps/SM -> **1174 clk per 128x128 step**; MMA util 45.0 %; binding unit by SOL: issue (47 %); ncu: 1202.1 us @ 2.23 GHz, SM-active 98.1 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 45.0 % | 45.9 % | 48.8 | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 21.8 % | 22.2 % | 24.0 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 554 | 47.2 % | 48.1 % | 32.9 | 2218 warp-instr / 4 (ncu SASS executed); of which sync/branch 712 |
| fma | 156 | 13.3 % | 13.6 % | 12.9 | 312 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 303 | 25.8 % | 26.3 % | 12.0 | 606 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 11.1 % | 11.3 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.1 % | 22.6 % | 24.0 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 396 | 33.7 % | 34.4 % | 48.6 | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 13.8 % | 14.1 % | 0.9 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 281 | 23.9 % | 24.4 % | 26.4 | 33.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 57 | 4.9 % | 5.0 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.0 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 744 M vs SASS-page executed 1163 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrdefault_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr default, paged 64)
measured 4683.0 us @ 2364 MHz, 9709 steps/SM -> **1140 clk per 128x128 step**; MMA util 46.3 %; binding unit by SOL: tensor (46 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 46.3 % | 46.3 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 22.5 % | 22.5 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 29.8 % | 29.8 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 11.2 % | 11.2 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.6 % | 2.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 22.5 % | 22.5 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.8 % | 22.8 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 34.7 % | 34.7 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 14.2 % | 14.2 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 25.2 % | 25.2 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 2.5 % | 2.5 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr never, paged 64)
measured 195.9 us @ 2364 MHz, 308 steps/SM -> **1503 clk per 128x128 step**; MMA util 34.1 %; binding unit by SOL: tensor (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 34.1 % | 34.1 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 17.0 % | 17.0 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 22.6 % | 22.6 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.5 % | 8.5 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.0 % | 17.0 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.3 % | 17.3 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 25.8 % | 25.8 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.8 % | 10.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.1 % | 19.1 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 15.1 % | 15.1 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr never, paged 64)
measured 787.4 us @ 2364 MHz, 1223 steps/SM -> **1522 clk per 128x128 step**; MMA util 33.6 %; binding unit by SOL: issue (48 %); ncu: 783.0 us @ 2.24 GHz, SM-active 99.7 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 33.6 % | 33.7 % | 36.1 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 16.8 % | 16.9 % | 18.4 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 735 | 48.3 % | 48.4 % | 30.7 | 2939 warp-instr / 4 (ncu SASS executed); of which sync/branch 1011 |
| fma | 345 | 22.7 % | 22.7 % | 20.2 | 689 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 326 | 21.4 % | 21.5 % | 9.6 | 652 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 8.8 % | 8.8 % |  | 267 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 17.2 % | 17.3 % | 18.4 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 25.5 % | 25.6 % | 37.2 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.6 % | 10.7 % | 0.6 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 282 | 18.5 % | 18.6 % | 13.7 | 33.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 114 | 7.5 % | 7.5 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.1 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 464 M vs SASS-page executed 776 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrnever_p64_half_pf_benc_cf0h0_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr never, paged 64)
measured 3141.5 us @ 2364 MHz, 4873 steps/SM -> **1524 clk per 128x128 step**; MMA util 33.6 %; binding unit by SOL: tensor (34 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 33.6 % | 33.6 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 16.8 % | 16.8 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 22.3 % | 22.3 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.4 % | 8.4 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 16.8 % | 16.8 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.1 % | 17.1 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 25.5 % | 25.5 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.6 % | 10.6 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 18.8 % | 18.8 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 3.8 % | 3.8 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr never, paged 64)
measured 358.0 us @ 2364 MHz, 607 steps/SM -> **1395 clk per 128x128 step**; MMA util 36.7 %; binding unit by SOL: tensor (37 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 36.7 % | 36.7 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 18.4 % | 18.4 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.3 % | 24.3 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.2 % | 9.2 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.2 % | 2.2 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.4 % | 18.4 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.6 % | 18.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 27.8 % | 27.8 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.6 % | 11.6 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 20.6 % | 20.6 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 8.2 % | 8.2 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr never, paged 64)
measured 1188.5 us @ 2364 MHz, 2427 steps/SM -> **1157 clk per 128x128 step**; MMA util 44.2 %; binding unit by SOL: issue (62 %); ncu: 1190.6 us @ 2.22 GHz, SM-active 97.0 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 44.2 % | 45.6 % | 48.6 | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 22.1 % | 22.8 % | 24.7 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 716 | 61.9 % | 63.8 % | 39.6 | 2864 warp-instr / 4 (ncu SASS executed); of which sync/branch 1018 |
| fma | 324 | 28.0 % | 28.9 % | 26.9 | 648 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 312 | 26.9 % | 27.8 % | 11.7 | 623 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 11.2 % | 11.6 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.5 % | 23.2 % | 24.7 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 388 | 33.5 % | 34.6 % | 48.6 | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 14.0 % | 14.4 % | 0.8 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 280 | 24.2 % | 24.9 % | 28.1 | 33.0 KiB K/V(+SF) per step per SM / 121 B/clk/SM |
| dram | 57 | 5.0 % | 5.1 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.8 TB/s = 121 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 875 M vs SASS-page executed 1502 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrnever_p64_half_pf_benc_cf0h0_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 0, corr never, paged 64)
measured 4663.6 us @ 2364 MHz, 9709 steps/SM -> **1136 clk per 128x128 step**; MMA util 45.1 %; binding unit by SOL: tensor (45 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 512 | 45.1 % | 45.1 % |  | BMM1 4096k + BMM2 4096k MAC / 16384 |
| mufu | 256 | 22.5 % | 22.5 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 29.9 % | 29.9 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 11.3 % | 11.3 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.6 % | 2.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 22.5 % | 22.5 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.9 % | 22.9 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 388 | 34.2 % | 34.2 % |  | reads 64 KiB (Q 32 + K/2 16 + V/2 16) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 14.3 % | 14.3 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 25.3 % | 25.3 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 2.5 % | 2.5 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=8192 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 192.4 us @ 2364 MHz, 308 steps/SM -> **1476 clk per 128x128 step**; MMA util 35.8 %; binding unit by SOL: tensor (36 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 35.8 % | 35.8 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 17.3 % | 17.3 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 23.0 % | 23.0 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.7 % | 8.7 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.3 % | 17.3 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.6 % | 17.6 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 26.8 % | 26.8 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.0 % | 11.0 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.4 % | 19.4 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 226 | 15.3 % | 15.3 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=16384 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 776.9 us @ 2364 MHz, 1223 steps/SM -> **1501 clk per 128x128 step**; MMA util 35.2 %; binding unit by SOL: issue (38 %); ncu: 773.0 us @ 2.25 GHz, SM-active 99.0 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 35.2 % | 35.5 % | 37.9 | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 17.0 % | 17.2 % | 18.7 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 575 | 38.3 % | 38.7 % | 24.9 | 2301 warp-instr / 4 (ncu SASS executed); of which sync/branch 785 |
| fma | 173 | 11.5 % | 11.7 % | 10.1 | 347 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 291 | 19.4 % | 19.6 % | 9.4 | 582 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 133 | 8.9 % | 9.0 % |  | 267 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 262 | 17.5 % | 17.7 % | 18.7 | 262 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 396 | 26.4 % | 26.6 % | 38.2 | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.8 % | 10.9 % | 0.6 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 282 | 18.8 % | 19.0 % | 13.1 | 33.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 114 | 7.6 % | 7.7 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 58.1 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 370 M vs SASS-page executed 608 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrnever_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=32768 causal (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 3089.1 us @ 2364 MHz, 4873 steps/SM -> **1498 clk per 128x128 step**; MMA util 35.2 %; binding unit by SOL: tensor (35 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 35.2 % | 35.2 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 17.1 % | 17.1 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 22.7 % | 22.7 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 8.5 % | 8.5 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.0 % | 2.0 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 17.1 % | 17.1 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 17.4 % | 17.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 26.4 % | 26.4 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 10.8 % | 10.8 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 19.2 % | 19.2 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 57 | 3.8 % | 3.8 % |  | 944 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=8192 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 361.9 us @ 2364 MHz, 607 steps/SM -> **1410 clk per 128x128 step**; MMA util 37.5 %; binding unit by SOL: tensor (37 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 37.5 % | 37.5 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 18.2 % | 18.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 24.1 % | 24.1 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 9.1 % | 9.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.1 % | 2.1 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 18.2 % | 18.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 18.4 % | 18.4 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 28.1 % | 28.1 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 11.5 % | 11.5 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 20.4 % | 20.4 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 115 | 8.2 % | 8.2 % |  | 236 MB whole kernel / 8.0 TB/s |

### corrnever_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=16384 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 1198.6 us @ 2364 MHz, 2427 steps/SM -> **1167 clk per 128x128 step**; MMA util 45.2 %; binding unit by SOL: issue (46 %); ncu: 1199.6 us @ 2.23 GHz, SM-active 96.6 %

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 45.2 % | 46.8 % | 49.6 | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 21.9 % | 22.7 % | 24.4 | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 535 | 45.8 % | 47.4 % | 30.9 | 2140 warp-instr / 4 (ncu SASS executed); of which sync/branch 718 |
| fma | 154 | 13.2 % | 13.7 % | 13.0 | 308 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (ncu SASS executed) |
| alu | 269 | 23.0 % | 23.8 % | 11.1 | 538 integer/logic ALU warp-instr / 2 (ncu SASS executed) |
| cvt | 130 | 11.1 % | 11.5 % |  | 260 F2FP/I2F convert+pack warp-instr / 2 (ncu SASS executed; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.3 % | 23.1 % | 24.4 | 260 MUFU warp-instr x 32 lanes / 32 (ncu SASS executed) |
| smem | 396 | 33.9 % | 35.1 % | 48.7 | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 13.9 % | 14.4 % | 0.8 | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 281 | 24.1 % | 24.9 % | 25.2 | 33.0 KiB K/V(+SF) per step per SM / 120 B/clk/SM |
| dram | 57 | 4.9 % | 5.1 % |  | 472 MB whole kernel / 8.0 TB/s |

ncu-derived L2 sector peak 57.9 TB/s = 120 B/clk/SM (lts__t_sectors / pct_of_peak); ncu smsp__inst_executed.sum 692 M vs SASS-page executed 1122 M (the SASS page counts every pipe of a multi-pipe instruction; issue SOL uses the SASS count = upper bound)

### corrnever_p64_half_pf_benc_cf0h0rs1_cga2 d256 S=32768 none (cta_mma 2, TILES_Q 1, f16 exp, rowsum-MMA 1, corr never, paged 64)
measured 4744.0 us @ 2364 MHz, 9709 steps/SM -> **1155 clk per 128x128 step**; MMA util 45.7 %; binding unit by SOL: tensor (46 %)

| unit | SOL clk/step | util = SOL/measured | util over SM-active | ncu % (cross-check, of SM-active) | how |
|---|---|---|---|---|---|
| tensor | 528 | 45.7 % | 45.7 % |  | BMM1 4096k + BMM2 4096k + rowsum 256k MAC / 16384 |
| mufu | 256 | 22.2 % | 22.2 % |  | 16384 exp as f16x2 pairs / 32 lanes |
| issue | 340 | 29.4 % | 29.4 % |  | 1358 warp-instr / 4 (static estimate) |
| fma | 128 | 11.1 % | 11.1 % |  | 256 FMA-pipe warp-instr (FFMA2/FHADD2/HFMA2) / 2 (static estimate) |
| alu | 30 | 2.6 % | 2.6 % |  | 60 integer/logic ALU warp-instr / 2 (static estimate) |
| cvt | 256 | 22.2 % | 22.2 % |  | 512 F2FP/I2F convert+pack warp-instr / 2 (static estimate; ncu pipe attribution of F2FP differs) |
| xu_inst | 260 | 22.5 % | 22.5 % |  | 260 MUFU warp-instr x 32 lanes / 32 (static estimate) |
| smem | 396 | 34.3 % | 34.3 % |  | reads 66 KiB (Q 32 + K/2 16 + V/2 16 + ones 2) + TMA writes 33.0 KiB / 256 B |
| tmem | 162 | 14.0 % | 14.0 % |  | 81 KiB tcgen05.ld/st / 512 B (assumed) |
| l2 | 287 | 24.9 % | 24.9 % |  | 33.0 KiB K/V(+SF) per step per SM / 118 B/clk/SM |
| dram | 29 | 2.5 % | 2.5 % |  | 944 MB whole kernel / 8.0 TB/s |
