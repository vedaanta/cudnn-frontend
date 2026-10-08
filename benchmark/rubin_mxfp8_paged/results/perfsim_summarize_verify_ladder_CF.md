# Rubin GR100 PerfSim: MXFP8 prefill (B=1, H_q=H_kv=1, S=4096, no mask)

Simulated at nvclk 2350 MHz, 16 CTAs on 16 SMs. Durations are the kernel's grid start-to-end in SM clocks; MMA numbers are the instrumented cluster's (SM0_0_0/1). Trace = 4th launch (3 warm-ups) captured with cuda_apic on w2u1g-lc-0614; ACE via SSAF, SMART + PIC via flow.perfsim latest.

## Timing and top-level SOL

| run | sim cycles | sim µs | Δ first | SOL top-3 | mainloop MMA util | MMA cycles (SOL→actual) | pre-MMA | post-MMA | bottlenecks | GPU SOL full chip | silicon µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gr100_ladder_C_b1h1s4k_none_r1 | 53537 | 22.78 | +0.0% | TPC 48.7%, SM 40.9%, LST 29.9% | 39.5% | 17408→44115 | 5802 | 3620 | GPCMMU 6.7%, MMU 6.7%, FB 3.9% | 2.61 | – |
| gr100_ladder_F_b1h1s4k_none_r1 | 48638 | 20.70 | -9.2% | TPC 52.6%, SM 44.6%, LST 32.7% | 44.3% | 17408→39334 | 5721 | 3583 | GPCMMU 7.3%, MMU 7.3%, FB 4.5% | 2.92 | – |

## Per-unit SOL table (busiest / instrumented instance; % of each unit's peak over its elapsed clocks)

| run | issue % | MMA pipe % (whole kernel) | XU (MUFU) % | FMA-heavy % | FMA-lite % | ALU % | ADU % | uniform pipe % | LSU % | TMEM rd % | TMEM wr % | SMEM wavefronts (LST) % | LST TC-MMA wavefronts % | ICC instr fetch (TPC) % | L2 data-bank accesses % | L2 xbar read bytes % | L2 tag lookups % | DRAM (FB) bytes % | GPCMMU ltp requests % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gr100_ladder_C_b1h1s4k_none_r1 | 40.9 | 30.3 | 29.0 | 18.2 | 15.4 | 31.2 | 12.7 | 3.5 | 1.3 | 17.6 | 16.6 | 29.9 | 29.9 | 48.7 | 0.5 | 0.5 | 0.4 | 0.6 | 0.4 |
| gr100_ladder_F_b1h1s4k_none_r1 | 44.6 | 33.1 | 31.7 | 20.0 | 16.9 | 34.1 | 13.8 | 3.8 | 2.4 | 19.2 | 18.1 | 32.7 | 32.7 | 52.6 | 0.5 | 0.5 | 0.5 | 0.6 | 0.5 |

## Register-file bandwidth (instrumented SM, raw SMART counters)

| run | RF rd ports % (elapsed) | RF rd ports % (mainloop) | RF wr ports % (elapsed) | RF wr ports % (mainloop) | LRF reads/clk | LRF writes/clk | writes coupled / decoupled | reads avoided by reuse cache | rd dispatch stall % (bank0/bank1/f-pipe) | URF rd port % (elapsed / mainloop) | URF wr port % | URF reads / writes | URF rd dispatch stall % | uniform pipe active % | UR instr/clk | TMEM→RF wb stall % | TMEM reads (ldtm / utcmma_c / sf) | TMEM writes (sttm / utcmma / utccp) | issued instr/clk/SM | issued by pipe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gr100_ladder_C_b1h1s4k_none_r1 | 27.7 | 36.1 | 27.0 | 35.2 | 2.22 | 2.16 | 72206 / 51961 | 16243 | 1.05 (2372/2114/2411) | 14.3 / 18.6 | 3.5 | 32785 / 7997 | 0.08 | 3.5 | 0.141 | 0.77 | 81088 (17152 / 52672 / 11264) | 76136 (4360 / 69632 / 2144) | 1.63 | alu 35870, mio 24438, fmaheavy 20846, fmalite 17719, uniform 8083, fe 3632 |
| gr100_ladder_F_b1h1s4k_none_r1 | 30.6 | 40.8 | 29.5 | 39.4 | 2.45 | 2.36 | 72362 / 51713 | 15693 | 0.88 (1484/910/1851) | 15.4 / 20.5 | 3.8 | 32297 / 7997 | 0.10 | 3.8 | 0.154 | 0.56 | 80840 (16904 / 52672 / 11264) | 75880 (4104 / 69632 / 2144) | 1.78 | alu 35849, mio 24184, fmaheavy 21023, fmalite 17719, uniform 8083, fe 3388 |

Normalisation: read ports = LRF operand reads (`register_reads_bank{0,1}_q`, hardware count; the simulator-only count differs by a few %) / (4 SMSP x 2 banks x 1 read/clk) / clocks; write ports = `register_writes_bank{b}_hw0_q` summed over banks / (8/clk) / clocks -- hw0 and hw1 are the two half-warp ports and a 32-lane write is counted once on each (hw0 == hw1 in every run), so this equals the sum over all four hw counters / 16; mainloop = first-MMA-issue..last-MMA-retire window of the math SOL table (same numerator). Coupled writes come from fixed-latency math pipes, decoupled from MIO/MUFU/TMEM/LSU write-backs. Dispatch stalls = cycles an SMSP could not dispatch because of a register-read port/bank conflict, per SMSP-clock. TMEM→RF wb stall = tcgen05.ld write-back stalled against the RF write ports, per SMSP-clock. Uniform RF: `uniform_register_reads_q` / `uniform_register_writes_q` (per-SM sums) / (4 SMSP x 1 uniform read or write per clock) / clocks -- the uniform datapath is one scalar register access per SMSP per clock; `uniform_pipe_active_q*` and `inst_issued_uniform_pipe_q` give the UR pipe's activity for comparison.

## PIC-Smart webviews (perf-inspector)

- gr100_ladder_C_b1h1s4k_none_r1: https://perf-inspector/server/?infoWin=On&panels=%5B%5D&path=/home/scratch.vagarwalla_gpu/perfsim_ladder/perfsim_output/gr100_ladder_C_b1h1s4k_none_r1/perfsim/pic_analysis/run.A.dir.0/861745.861743/pic-analysis/pi/web/full
- gr100_ladder_F_b1h1s4k_none_r1: https://perf-inspector/server/?infoWin=On&panels=%5B%5D&path=/home/scratch.vagarwalla_gpu/perfsim_ladder/perfsim_output/gr100_ladder_F_b1h1s4k_none_r1/perfsim/pic_analysis/run.A.dir.0/861950.861743/pic-analysis/pi/web/full
