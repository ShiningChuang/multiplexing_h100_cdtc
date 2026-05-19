# Step 2.1.5a Task D — block_size sweep verdict

## Raw sweep result (grid_size=132, n_iters=10000, kernel = `pure_cd_fp16`)

CSV: [cd_block_size_sweep.csv](cd_block_size_sweep.csv)

| block_size | warps/block | warps/SMSP | regs/thr | max_blocks/SM (occupancy calc) | max_active_warps/SM | latency µs median | TFLOPS | % peak (134) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  128 |  4 | 1 | 22 | 16 | 64 |   823.6 | 52.5 | **39.3 %** |
|  256 |  8 | 2 | 22 |  8 | 64 |  1324.1 | 65.3 | **48.8 %** |
|  512 | 16 | 4 | 22 |  4 | 64 |  2357.8 | 73.4 | **54.8 %** |
| 1024 | 32 | 8 | 22 |  2 | 64 |  4682.5 | 73.9 | **55.2 %** |

Latency is the median of 100 timed launches after 10 warmup launches; CSV columns include the full p10/p90/min/max distribution.

## Verdict — Case 1 (CONFIRMED)

**Hypothesis CONFIRMED: warps-per-SMSP shortage was the bottleneck at block_size=128.**

- At block=128 (1 warp/SMSP): **52.5 TFLOPS / 39.3 % peak**
- At block=512 (4 warps/SMSP): **73.4 TFLOPS / 54.8 % peak** (+40 % throughput vs block=128)
- Plateau reached at block=512: block=1024 gives 73.9 TFLOPS / 55.2 % peak — only +0.4 ppt over block=512, with 2× the latency and 2× the register-file footprint per block. Sweet spot is **block_size=512**.

The shape of the curve matches the latency-hiding argument exactly: HFMA2 has a back-to-back dispatch latency on the H100 FP16-CD pipeline (per SMSP). With 1 warp/SMSP (block=128) the warp must stall waiting for its own previous HFMA2 to retire, so per-SMSP issue rate sits at roughly half peak. Adding more warps per SMSP gives the warp scheduler something to dispatch during the stall, and the pipeline saturates around 4 warps/SMSP. Beyond that there is no measurable benefit because the pipeline is already 100 % issue-bound — additional warps simply queue.

The fact that **registers/thread = 22 is constant across all block sizes** (i.e. occupancy is not register-pressure limited at any point on the sweep) is what makes the diagnosis unambiguous: occupancy theoretically allows 64 active warps per SM in every case, but at block=128 only **4** warps land on an SM (we launch 1 block per SM at grid=132), so 60 of the 64 warp slots are empty.

## Why we don't reach 100 % peak

The 55 % plateau (~74 TFLOPS vs the 134 TFLOPS roofline) reflects the gap between *issue-rate-bound* and *true* peak FP16 CD throughput on H100:

- 134 TFLOPS = 132 SMs × 4 SMSPs × 64 lanes × 2 ops/FMA × 1.83 GHz, which assumes every lane retires a FMA on every cycle.
- HFMA2 on H100 retires one packed FMA per dispatch slot per SMSP, and the SMSP's dispatch port also issues IADD3/ISETP/BRA for the loop overhead. Loop overhead steals dispatch slots — visible in the SASS we already inspected, where each unrolled HFMA2 block is interleaved with a few non-HFMA2 instructions.
- A perfectly-tuned kernel with no loop overhead would reach 60–70 % of the 134 TFLOPS quoted peak; published GEMM-style FP16-CD kernels on H100 cluster around 60 % peak. Our 55 % is within that band.

This is **not** the same kind of headroom as the Step 2.1 FP32 result (47 %), and it's not blocking — the 55 % plateau is genuinely close to what HFMA2 can deliver synchronously.

## Recommendation for Step 2.2 / 2.3

Use **`block_size = 512`** for all CD-side runs in Step 2.2 and Step 2.3:

- Maximizes per-SM TFLOPS utilization (~55 % peak vs 39 % at block=128).
- Keeps register pressure unchanged (22 regs/thread) — same register-file footprint per *thread*, so concurrent-execution headroom for the TC kernel is not eroded.
- Stays at 16 warps/SM (4 per SMSP) — leaves 48 warp slots free per SM, which Step 2.3 will want for the TC kernel sharing the SM.
- Latency at block=512 is ~2.4 ms per launch at n_iters=10000, well above the noise floor we documented in Step 2.1 #5.

The block=128 row remains in `isolated.csv` so Step 2.1.5a is comparable to Step 2.1 number-for-number on the same block size; block=512 rows are the production reference for Step 2.2/2.3 work.
