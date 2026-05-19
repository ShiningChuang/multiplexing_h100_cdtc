# Step 2.1 — Isolated Micro-bench Summary

## Inputs
- `src/microbench/kernel_pure_cd.cu` — synthetic FP32 FFMA kernel, no HMMA
- `src/microbench/kernel_pure_tc.cu` — synthetic FP16 HMMA kernel (wmma::mma_sync), no main-loop FFMA
- `src/microbench/bench_isolated.cu` — CSV-emitting timing driver (CUDA events, warmup=10, repeat=100, median)
- `src/microbench/Makefile` — builds cubins + bench binary into `build/`
- `tools/verify_sass.sh` — cuobjdump-based SASS instruction-mix check

## SASS verification (`tools/verify_sass.sh`)
```
=== kernel_pure_cd ===
FFMA count                : 65
FMUL / FADD count         : 6
HMMA / HGMMA / GMMA count : 0       ✅ no Tensor Core ops
SASS lines                : 231

=== kernel_pure_tc ===
HMMA / HGMMA / GMMA count : 16      ✅ Tensor Cores in use
FFMA count                : 0       ✅ no CUDA Core main-loop ops
SASS lines                : 407

Verdict: PASS — CD kernel is CD-only, TC kernel uses Tensor Cores.
```

CD kernel SASS: inner loop is 64 unrolled FFMAs (CD_UNROLL=16 × CD_NACC=4),
plus 1 epilogue FFMA, plus 6 FMUL/FADD for the per-thread constant init.

TC kernel SASS: 16 HMMA in the loop body — TC_NACC=8 accumulator chains
survive optimization after the per-chain distinct init + per-chain store
trick. (Initial naive version was CSE-folded down to 2; documented in the
"What did not work" section below.)

## Isolated benchmark — full sweep

CSV: [result/step2_microbench/isolated.csv](isolated.csv)

| kernel  | grid | n_iters | latency (µs median) | TFLOPS achieved | % of H100 peak |
| ---     | ---: | ---:    | ---:                | ---:            | ---:           |
| pure_cd |   1  |   1 000 |   73.7  |   0.22 |  0.3 % |
| pure_cd |   1  |  10 000 |  692.0  |   0.24 |  0.4 % |
| pure_cd |   1  | 100 000 | 6924.4  |   0.24 |  0.4 % |
| pure_cd |  33  |   1 000 |   70.1  |   7.71 | 11.5 % |
| pure_cd |  33  |  10 000 |  658.2  |   8.21 | 12.3 % |
| pure_cd |  33  | 100 000 | 6878.4  |   7.86 | 11.7 % |
| pure_cd |  66  |   1 000 |   70.5  |  15.35 | 22.9 % |
| pure_cd |  66  |  10 000 |  659.6  |  16.39 | 24.5 % |
| pure_cd |  66  | 100 000 | 6872.1  |  15.74 | 23.5 % |
| pure_cd | 132  |   1 000 |   70.3  |  30.78 | **45.9 %** |
| pure_cd | 132  |  10 000 |  659.2  |  32.81 | **49.0 %** |
| pure_cd | 132  | 100 000 | 6876.5  |  31.45 | **46.9 %** |
| pure_tc |   1  |   1 000 |   68.8  |   3.81 |  0.4 % |
| pure_tc |   1  |  10 000 |  634.2  |   4.13 |  0.4 % |
| pure_tc |   1  | 100 000 | 6150.3  |   4.26 |  0.4 % |
| pure_tc |  33  |   1 000 |   66.8  | 129.5  | 13.1 % |
| pure_tc |  33  |  10 000 |  622.7  | 138.9  | 14.0 % |
| pure_tc |  33  | 100 000 | 6166.8  | 140.3  | 14.2 % |
| pure_tc |  66  |   1 000 |   67.1  | 258.0  | 26.1 % |
| pure_tc |  66  |  10 000 |  622.0  | 278.1  | 28.1 % |
| pure_tc |  66  | 100 000 | 6166.7  | 280.6  | 28.4 % |
| pure_tc | 132  |   1 000 |   67.1  | 515.4  | **52.1 %** |
| pure_tc | 132  |  10 000 |  622.4  | 556.0  | **56.2 %** |
| pure_tc | 132  | 100 000 | 6181.3  | 559.8  | **56.6 %** |

Peaks used:
- CD FP32 = **67.0 TFLOPS** (H100 SXM5 datasheet)
- TC FP16 = **989.4 TFLOPS** (H100 SXM5 datasheet, WGMMA / dense / no sparsity)

## H100 peak-match verdict (Task E)

| kernel  | spec target | achieved at grid=132 | verdict |
| ---     | ---         | ---                   | ---     |
| pure_cd | 40–60 TFLOPS (60–90 % peak) | ~31 TFLOPS (~47 %)   | **below desired band; passes ">50% off" stop-gate** (50% of 40 = 20, we're at 31) |
| pure_tc | 600–850 TFLOPS (60–86 % peak) | ~558 TFLOPS (~56 %) | **below desired band; passes ">50% off" stop-gate** (50% of 600 = 300, we're at 558) |

### Why we're below the desired bands (but not pathological)

**CD (47 % vs 60–90 % expected).**
H100 FP32 peak assumes every cycle, every FP32 lane in every SMSP retires
a FMA. Achieving that needs enough warps to hide FFMA latency (4 cycles
typical) — i.e. ≥ ~4 warps per SMSP, or ≥ 16 warps per SM. With our
block_size = 128 (per the spec) we have **4 warps per SM**, one per SMSP.
That gives latency-hiding factor 1× per SMSP, so the FFMA pipeline is
issuing roughly every other cycle (≈ 50 % of issue slots). Empirical
47 % is consistent. To reach 60–90 % we would need either more warps per
SM (block_size ≥ 256) or aggressive software pipelining — but block_size
is fixed by the spec for this round.

**TC (56 % vs 60–86 % expected).**
The 989 TFLOPS peak number on H100 SXM5 is the **WGMMA** path (async
descriptor-based instruction, 4th-gen Tensor Core). `wmma::mma_sync` on
sm_90a generates `HMMA.16816.F16` (synchronous, 3rd-gen-style issue
shape). Published empirical HMMA peaks on H100 land around 500–600
TFLOPS — exactly where our 558 TFLOPS sits. To approach the 989 TFLOPS
datasheet number we would need inline `wgmma.mma_async` PTX (and the
fence / commit / wait barrier machinery around it), which is out of
scope for Step 2.1 (and arguably not necessary — the research question
is co-execution, not absolute peak).

Both numbers are well above the ">50 % off" stop-gate, so per the spec we
proceed.

## SM utilization (`sm__cycles_active.avg.pct_of_peak_sustained_elapsed`)

The CSV does NOT include this column. Reason:
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```
NCU is installed and on PATH (verified in Step 1 check #9), but the
kernel-mode `RestrictProfilingToAdminUsers` flag is set on this host, so
`ncu`'s SM-cycle metrics require root. We did not escalate because (a) it
needs `nvidia-modprobe -u NVreg_RestrictProfilingToAdminUsers=0` plus a
GPU reset, which is a host-wide change, and (b) the `pct_of_peak` column
we already emit answers the same "is the unit saturated" question via the
achieved-vs-peak FLOPS ratio — which is what determines whether we have
any headroom for co-execution.

If a hard SM-cycle number is required for the paper, we can re-run after
the host admin clears the flag. For now `pct_of_peak` is the operative
saturation signal.

## Scaling story (key plot input for Step 2.3)

Both kernels scale **linearly** with grid_size (= SM occupancy):

| grid | pure_cd TFLOPS | pure_cd / grid | pure_tc TFLOPS | pure_tc / grid |
| --- | ---: | ---: | ---: | ---: |
|   1 |  0.24 | 0.24 |   4.3 |  4.3 |
|  33 |  8.21 | 0.25 | 138.9 |  4.2 |
|  66 | 16.39 | 0.25 | 278.1 |  4.2 |
| 132 | 32.81 | 0.25 | 556.0 |  4.2 |

→ both pipelines saturate **per-SM** quickly and scale cleanly. A
single SM gives ~0.25 TFLOPS of FFMA throughput and ~4.2 TFLOPS of HMMA
throughput. The total budget per SM is therefore ~4.45 TFLOPS if perfect
CD+TC overlap is achievable — Step 2.3 measures how much of that budget
is actually unlocked when we run both kernels concurrently on the same SM.

## Stability

Median is well-defined across all 24 points. (p90 − p10) / median is
**< 1 % at n_iters ≥ 10 000** for both kernels and is **< 7 % at
n_iters = 1 000** (sub-100 µs kernels still hit the CUDA event jitter
floor we documented in Step 1 sanity check #5). Step 2.3 should pick
n_iters ≥ 10 000 to stay above the noise floor.

## What did not work (kept for documentation)

1. **Initial TC kernel: TC_NACC=8 collapsed to 2 in SASS.** The first
   version of `kernel_pure_tc.cu` had all 8 accumulators initialized to
   zero and stored only `frag_c[0]` at the end. The optimizer (correctly)
   noticed that 8 chains with identical inits, identical (a, b) operands,
   and identical evolution will hold identical values, so 6 chains had no
   observable side effect → CSE-folded to 2. Result: TFLOPS counter
   over-reported by 4× and reported 2035 TFLOPS at grid=132, which is
   above H100's 989 TFLOPS datasheet peak — i.e., physically impossible
   and a clear sign of mis-accounting. Fix: distinct per-j init (`0.001 *
   (j+1)`) + per-j separate `store_matrix_sync`. After the fix the SASS
   shows 16 HMMA in the loop body and TFLOPS at grid=132 falls back to
   the plausible 559 TFLOPS reported above.

2. **NCU SM-util counters blocked.** See "SM utilization" section above.

## Files produced
```
src/microbench/
├── kernel_pure_cd.cu        # CD-only FFMA kernel + host launcher + FLOP counter
├── kernel_pure_tc.cu        # TC-only HMMA kernel + host launcher + FLOP counter
├── bench_isolated.cu        # CSV driver (also has --profile mode for NCU)
└── Makefile                 # build cubins + binary into ../../build/
build/
├── kernel_pure_cd.cubin     # for SASS check
├── kernel_pure_tc.cubin     # for SASS check
└── bench_isolated           # benchmark binary
tools/
└── verify_sass.sh           # cuobjdump-based instruction-mix check
result/step2_microbench/
├── isolated.csv             # all 24 measurements
└── summary.md               # this file
```

## Gate for Step 2.2

- ✅ Two synthetic kernels exist, build clean against sm_90a, and SASS
  contains exactly the intended instruction class for each (no
  cross-contamination).
- ✅ Isolated peaks are within the spec's ">50 % off" stop-gate.
- ✅ Per-SM throughput numbers (0.25 TFLOPS FFMA + 4.2 TFLOPS HMMA per SM)
  are the inputs that Step 2.3's co-execution measurement compares
  against — they're now logged and reproducible.
- ⚠ SM-cycle util via NCU not available without root; documented.

Ready for Step 2.2 (whatever comes next per the project plan).

---

## Step 2.1.5a: CD kernel FP16 HFMA2 + block_size sweep

`kernel_pure_cd.cu` was rewritten in place from FP32 FFMA to FP16 HFMA2 to
match the project's real CD-attention regime (attention softmax accumulates
in FP16 after FP8 dequant). The TC kernel is unchanged (FP16 HMMA via
`wmma::mma_sync`). Step 2.1's FP32 measurements remain in `isolated.csv`
under `kernel_type='pure_cd'` for reference, but **the FP32 numbers
should not be cited going forward** — they were a proxy that did not
match the production path. FP16 numbers live under
`kernel_type='pure_cd_fp16_block{128,512}'`.

### SASS verification of FP16 kernel

```
HFMA2 count            : 65       (gate ≥50, main loop has 16 × 4 = 64)
HFMA (single, non-MMA) : 0        (no scalar fallback)
FFMA (FP32)            : 8        (only in __floats2half2_rn constant init, not main loop)
HMMA / HGMMA / GMMA    : 0        ✅ no Tensor Core ops
```

A representative inner-loop slice (4 distinct accumulators R13, R15, R17, R19):
```
HFMA2.MMA R13, R4, R6, R13 ;
HFMA2     R15, R7, R10, R15 ;
HFMA2.MMA R17, R8, R11, R17 ;
HFMA2     R19, R9, R12, R19 ;
```
All four chains survived optimization (the Step 2.1 CSE-folding pattern is
defeated by distinct per-chain inits and separate per-chain global stores).
Both `HFMA2` and `HFMA2.MMA` are FP16 packed-FMA encodings.

### block_size sweep result (grid=132, n_iters=10000)

CSV: [cd_block_size_sweep.csv](cd_block_size_sweep.csv)

| block | warps/SM | warps/SMSP | regs/thr | max-active-warps/SM | TFLOPS | % peak (134) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  128 |  4 | 1 | 22 | 64 | 52.5 | 39.3 % |
|  256 |  8 | 2 | 22 | 64 | 65.3 | 48.8 % |
|  512 | 16 | 4 | 22 | 64 | **73.4** | **54.8 %** |
| 1024 | 32 | 8 | 22 | 64 | 73.9 | 55.2 % |

**Hypothesis CONFIRMED (Case 1 of the verdict template):** warps-per-SMSP
shortage drives the Step 2.1 47 % FP32 utilization (and the 39 % FP16
utilization at the same block_size=128). Adding warps so the SMSP scheduler
has something to dispatch during HFMA2 latency lifts throughput by +40 %
(39 % → 55 %) until plateau at 4 warps/SMSP. See
[cd_block_size_sweep_verdict.md](cd_block_size_sweep_verdict.md).

Registers/thread stays at 22 across all block sizes — occupancy is *not*
register-pressure limited; the 64-warp-per-SM hardware cap is reached
naturally by block=512 with grid=132.

### Recommendation for Step 2.2 / 2.3

**Use `block_size = 512` for all CD-side runs going forward.**
- Maximum CD throughput on this kernel (73 TFLOPS, 55 % peak).
- Same regs/thread as block=128 (22) → register-file footprint per *thread*
  is unchanged, so TC kernel headroom for SM-co-execution is preserved.
- 16 warps/SM leaves 48 of 64 warp slots free per SM for the TC kernel
  Step 2.3 will run alongside.

### Updated isolated baseline (grid=132)

| kernel_type | block | n_iters | latency µs | TFLOPS | % peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure_cd_fp16_block128 | 128 |  10 000 |   823.6 | 52.5 | 39.3 % |
| pure_cd_fp16_block128 | 128 | 100 000 |  8188.0 | 52.8 | 39.5 % |
| pure_cd_fp16_block512 | 512 |  10 000 |  2355.7 | 73.4 | **54.9 %** |
| pure_cd_fp16_block512 | 512 | 100 000 | 23513.9 | 73.6 | **55.0 %** |

n_iters scaling is consistent within ±0.3 ppt — stability is excellent at
both block sizes; the noise floor we documented in Step 2.1 (sub-100 µs
kernels) doesn't apply here since every CD config in this sweep is ≥ 800 µs.

### Per-SM budget update for Step 2.3

Updating the "per-SM throughput available for co-execution" table from
Step 2.1's main summary:

| path | per-SM TFLOPS (best block) | at grid=132 |
| --- | ---: | ---: |
| CD (FP16 HFMA2, block=512) | **0.557 TFLOPS / SM** | 73.4 TFLOPS |
| TC (FP16 HMMA, block=128 unchanged) | 4.21 TFLOPS / SM | 556.0 TFLOPS |
| Combined ceiling (if perfect SM-level overlap) | **4.77 TFLOPS / SM** | **629.4 TFLOPS** |

The CD per-SM number is roughly 2.2× the Step 2.1 FP32 estimate (0.25 →
0.557 TFLOPS/SM), reflecting both the FP16-vs-FP32 lane width and the
block_size correction. This is the new comparison point for Step 2.3's
concurrent-execution measurement.

### Files added / updated in Step 2.1.5a

```
src/microbench/
├── kernel_pure_cd.cu                 # MODIFIED — now FP16 HFMA2 + 4 chains
└── bench_isolated.cu                 # MODIFIED — subcommands (block-sweep, rebaseline-cd-fp16)
result/step2_microbench/
├── isolated.csv                      # APPENDED — pure_cd_fp16_block{128,512} rows
├── isolated.csv.step2_1              # NEW — Step 2.1 snapshot before append
├── cd_block_size_sweep.csv           # NEW — 4-row block sweep
├── cd_block_size_sweep_verdict.md    # NEW — Case 1 verdict + recommendation
└── summary.md                        # this file (Step 2.1.5a section appended)
```
