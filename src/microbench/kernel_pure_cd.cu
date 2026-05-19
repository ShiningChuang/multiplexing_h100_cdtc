// kernel_pure_cd.cu — synthetic CUDA-Core FP16 HFMA2 kernel.
//
// Step 2.1.5a upgrade: FP16 path to match the project's real CD-attention regime
// (attention softmax accumulates in FP16 after FP8 dequant). FP32 FFMA is
// the wrong proxy because production never runs CD-attention in FP32 here.
//
// Each thread runs (n_iters * UNROLL * NACC) HFMA2 ops in registers; SASS
// must contain HFMA2.* and nothing else compute-side (no FFMA, no HMMA/GMMA).
// CSE protection from Step 2.1 lessons-learned:
//   - each accumulator chain uses a distinct (a, b) operand pair
//   - each chain has a distinct thread-derived init value
//   - each chain stores to its own global offset at the end
//
// Step 2.1.5a / multiplexing_h100_cdtc

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

constexpr int CD_UNROLL = 16;       // inner unroll factor (back-to-back HFMA2s)
constexpr int CD_NACC   = 4;        // independent accumulator chains (ILP)

// Build a small finite per-(thread, chain) constant. Avoids the __int_as_float
// trick in the prompt (which can produce NaN/Inf bit patterns).
__device__ inline __half2 make_half2_unique(int tid, int chain) {
    float x = (float)(tid * 1000 + chain * 7919) * 1e-7f + 1e-4f;
    float y = (float)(tid * 1001 + chain * 7907) * 1e-7f + 2e-4f;
    return __floats2half2_rn(x, y);
}

extern "C" __global__ void kernel_pure_cd(int n_iters, __half2 seed_a, __half2 seed_b,
                                          __half2* out) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    // Four (a, b) operand pairs — distinct so the optimizer cannot merge chains.
    // a* are multipliers near 1.0 so acc stays bounded across n_iters * UNROLL ops.
    // b* are tiny additive offsets so acc grows linearly within FP16 range.
    __half2 a0 = seed_a;
    __half2 b0 = seed_b;
    __half2 a1 = __hadd2(seed_a, __floats2half2_rn(1e-4f, -1e-4f));
    __half2 b1 = __hadd2(seed_b, __floats2half2_rn(2e-5f,  3e-5f));
    __half2 a2 = __hadd2(seed_a, __floats2half2_rn(-2e-4f, 1e-4f));
    __half2 b2 = __hadd2(seed_b, __floats2half2_rn(1e-5f, -2e-5f));
    __half2 a3 = __hadd2(seed_a, __floats2half2_rn(3e-4f, -3e-4f));
    __half2 b3 = __hadd2(seed_b, __floats2half2_rn(-1e-5f, 4e-5f));

    // Distinct per-chain init derived from threadIdx → 4 chains never coincide.
    __half2 acc0 = make_half2_unique(tid, 0);
    __half2 acc1 = make_half2_unique(tid, 1);
    __half2 acc2 = make_half2_unique(tid, 2);
    __half2 acc3 = make_half2_unique(tid, 3);

    #pragma unroll 1                            // outer loop stays rolled
    for (int i = 0; i < n_iters; ++i) {
        #pragma unroll                          // inner unroll → CD_UNROLL × CD_NACC HFMA2 back-to-back
        for (int k = 0; k < CD_UNROLL; ++k) {
            acc0 = __hfma2(a0, b0, acc0);
            acc1 = __hfma2(a1, b1, acc1);
            acc2 = __hfma2(a2, b2, acc2);
            acc3 = __hfma2(a3, b3, acc3);
        }
    }

    // Defeat DCE: each chain stores to its own global slot, so the optimizer
    // cannot prove any chain has no observable side effect.
    out[gid * CD_NACC + 0] = acc0;
    out[gid * CD_NACC + 1] = acc1;
    out[gid * CD_NACC + 2] = acc2;
    out[gid * CD_NACC + 3] = acc3;
}

// Host wrapper. Caller pre-allocates out_device with >= grid_size*block_size*CD_NACC
// __half2 slots (each chain stores per-thread; we allocate per-thread for simplicity).
extern "C" void launch_pure_cd(int grid_size, int block_size, int n_iters,
                               __half2* out_device, cudaStream_t stream = 0) {
    // Multiplier near 1.0 (slight decay) + tiny additive offset = bounded chain.
    __half2 seed_a = __floats2half2_rn(0.9999f, 0.9999f);
    __half2 seed_b = __floats2half2_rn(1e-4f,   1e-4f);
    kernel_pure_cd<<<grid_size, block_size, 0, stream>>>(n_iters, seed_a, seed_b, out_device);
}

// FLOP accounting:
//   per thread per outer iter : CD_UNROLL * CD_NACC HFMA2 = 64 HFMA2
//   per HFMA2                 : 4 FP16 ops (2 lanes × (1 MUL + 1 ADD))
//   per thread per outer iter : 64 * 4 = 256 FP16 ops
//   total FP16 ops            : grid * block * n_iters * 256
extern "C" long long pure_cd_flops(int grid_size, int block_size, int n_iters) {
    return (long long)grid_size * block_size * n_iters
         * CD_UNROLL * CD_NACC          // HFMA2 count per thread per outer iter
         * 4LL;                          // ops per HFMA2 (2 lanes × 2 ops)
}

// Size helper for the driver: how many __half2 elements does `out` need?
extern "C" int pure_cd_out_elems_per_thread() { return CD_NACC; }
