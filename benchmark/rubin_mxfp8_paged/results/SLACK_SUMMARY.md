# Slack-ready summary (thread C0C4WN7MTGV / p1790955707752319)

*Rubin GR100 MXFP8 GQA prefill, paged KV page 64, correction always-ON vs always-OFF, dH=128 and dH=256: cuDNN silicon + PerfSim per-unit SOL incl. RF bandwidth*

@Timmy Liu here is the cuDNN-silicon-grounded view I owed you, both head dims, with the full trick stack and both correction bounds.

*Setup*
• Kernels: cuDNN FE FROST cc 10.7 MXFP8 prefill (develop head + bench levers). Q/K/V e4m3 + E8M0/32 scales, O bf16, GQA. Tricks ON in every row: 1/ln2 folded outside (prefolded Q), f16 softmax with the fused shift+convert, row-sum as a ones-MMA (d128 product kernel; d256 as a lever), correction fast path (d128). Paged KV = 64-row pages behind a random block table over NHD pools.
• Correction modes: `default` = product threshold rule (rescale only when the row max moves > 4 log2 units; on these inputs that is zero rescales), `ALWAYS` = FA2 ratchet, O (+Sigma) rescaled every step (exact), `NEVER` = no rescale and no correction arrive (timing-only; exact only at zero rescales, which holds here; K-growth inputs prove it is a different kernel).
• Silicon: w2u1g-lc-0614, 216 SMs @ 2364 MHz, B1 H_q 32 / H_kv 8, S 8k/16k/32k, no mask + causal (LPT), 15 interleaved rounds x 4 graph replays; every cell validated vs an fp32 reference (rel 0.036 none / 0.012 causal = e4m3 P noise, identical across modes).
• PerfSim: GR100 model, B1 H1 S4096 no mask (16 CTAs at d128, 32 at d256), flow.perfsim latest via SSAF, 4th-launch cuda_apic trace; per-unit SOL from the PIC, RF port utilization from the SMART counters.

*Silicon, 16k (µs, TF/s; Δ vs the row above it refers to)*
```
d128 (cga2)                      none            causal (LPT)      32k none TF/s
product dense                    641 (6858)      341 (6451)        7278
page-64 default                  673  +5.0%      333  -2.4%        6992
page-64 correction ALWAYS        951  +41% vs default   465 +40%   4894
page-64 correction NEVER         625  -7.1% vs default  333  0%    7440  (47% of 16384 MAC/clk/SM)

d256                             none            causal (LPT)      32k none TF/s
product dense cga1 (adapter default) 1230 (7152) 728 (6041)        7309
dense cga2 (same body)           1112  -9.6%     679  -6.7%        8329  (49.8% of peak)
page-64 default cga2             1201  +8.0% vs dense cga2   768 +13%   7460
page-64 correction ALWAYS        1717  +43% vs default      902 +17%   5266
page-64 correction NEVER         1189  -1.0% vs default     787 +2.5%  7544
page-64 + rowsum-MMA (def/ALW/NEV) 1206 / 1743 / 1199     758 / 909 / 777   (±1%)
```

*Silicon per-unit util vs SOL (16k none; analytic SOL per 128x128 step, ncu pipe % in parentheses)*
```
d128 page-64 default: tensor 41% (45)  MUFU 39% (43)  SM issue 55% (46)  SMEM 26% (L1 42)  TMEM 25%  L2 11%   -> binding: SM issue
d256 page-64 default: tensor 44% (48)  MUFU 22% (24)  SM issue 61% (42)  SMEM 33% (L1 48)  TMEM 14%  L2 24%   -> binding: SM issue
ALWAYS (both):        tensor 29-31%    FMA 31-35%     TMEM 40-45% (the per-step O round trip)
```

*PerfSim, B1 H1 S4k none (mainloop MMA util = tensor SOL cycles / actual; unit % = of that unit's peak; RF = port utilization over the mainloop)*
```
run                                 cycles  MMA   issue  XU    LST   TMEM rd/wr  RF rd  RF wr  rd-stall
ladder E before (d128 p64, no fused cvt) 55030 41.5  41.1   28.2  29.1  17/16       38.6   37.5   0.6
d128 product dense                  46744   46.8  37.8   32.7  33.7  20/19       32.4   30.6   0.6
d128 p64 default                    49264   46.6  36.9   31.2  32.1  19/18       33.2   31.1   1.1
d128 p64 ALWAYS                     64090   33.7  36.3   24.4  25.1  18/17       41.9   39.8   2.8
d128 p64 NEVER                      46798   50.0  37.4   32.7  33.6  20/19       35.4   33.2   0.8
d256 product dense cga1             53084   38.0  24.9   14.5  43.8  16/16       20.8   17.3   1.2   (LST/SMEM-bound)
d256 p64 default cga2               59139   35.1  24.7   13.2  26.8  14/14       21.1   17.4   1.3
d256 p64 ALWAYS                     64076   32.1  29.9   12.2  24.7  16/16       35.9   31.4   1.9
d256 p64 NEVER                      56826   37.1  24.2   13.6  27.7  14/15       21.9   18.0   1.3
d256 p64 ALWAYS + rowsum-MMA        62482   33.6  27.0   12.6  25.8  17/17       32.1   29.9   0.9
d256 p64 NEVER  + rowsum-MMA        57328   37.2  19.5   13.5  27.8  15/15       15.3   14.2   0.2
```
RF normalization: read ports = LRF operand reads / (4 SMSP x 2 banks) / clocks; write ports = bank writes (hw0) / 8 / clocks; rd-stall = % of SMSP clocks that could not dispatch because of a register read port/bank conflict. Uniform RF ports stay < 11%, TMEM->RF write-back stalls < 3%.

*Takeaways*
1. The trick stack moves the d128 page-64 kernel from the 41.5% mainloop MMA SOL I quoted earlier (ladder E) to 46.6% with the threshold correction and 50.0% with correction OFF (-10.5% / -15% sim cycles). On silicon at 32k that is 6992 / 7440 TF/s = 43 / 47% of the 16384 MAC/clk/SM peak. The SM issue slot is the binding unit in the analytic model (55-61%); PerfSim's top SOLs are TPC/SM instruction issue and LST, not the tensor pipe.
2. Correction always-ON costs +41-48% kernel time on silicon at both head dims (16k-32k) and +30% (d128) / +8% (d256) sim cycles at the tiny PerfSim shape. The unconditional 128- or 256-column TMEM O round trip per step lands on the PV chain: RF read ports 33 -> 42% (d128) and 21 -> 36% (d256), FMA pipes 20 -> 29%, TMEM traffic +20%. Correction always-OFF saves only 7% (d128) / 1% (d256) over the threshold rule at none, and nothing at causal: the threshold rule already skips every rescale here, so what is left is the arrive/wait on the correction warps.
3. Row-sum as MMA vs FMA at d256: no speedup on silicon (±1%), -2.5% sim cycles on the ALWAYS chain only. It does what the limiter model expects on the units: RF read ports 21.9 -> 15.3% and FMA-heavy 17.9 -> 10.0% (NEVER), issue 24 -> 19.5%, at +3% tensor cycles (N=16 ones-MMA) and +1 KiB SMEM per step; the d256 kernel is not RF/FMA-bound, so the freed slots do not turn into time. d128 already has the ones-MMA row-sum in the product kernel.
4. d256 cga: the adapter's cga1 default is LST/SMEM-bound in PerfSim (43.8% LST SOL) and 7-12% slower than the same body at cga2 on silicon (dense 32k: 7309 vs 8329 TF/s); cga2 should become the d256 MXFP8 default (follow-up, not in this drop).
5. Page-64 paging cost over NHD pools: d128 +4-5% (none), -2.5% at causal with LPT; d256 +8-12% (none) / +12-13% (causal). The same kernels over HND pools pay +40% at d256, so the pool layout is first-order for any page-size comparison.
6. Sim vs silicon at B1 H1 S4k: d128 sim is 0.95-0.96x silicon (within 5%); d256 sim is 1.21-1.26x silicon (the LST model is pessimistic for the d256 operand traffic), so use the d256 PerfSim rows for deltas and per-unit shares, not absolute cycles.

Artifacts: perf-inspector webviews for all 11 PICs, the full silicon tables (8k/16k/32k, both masks, ncu per-pipe) and the harness are on branch `vagarwalla/rubin-mxfp8-paged-bench` (vedaanta/cudnn-frontend), `benchmark/rubin_mxfp8_paged/results/{REPORT.md,perfsim_RESULTS.md}`; scratch copies under /home/scratch.vagarwalla_gpu/perfsim_mxb/.
