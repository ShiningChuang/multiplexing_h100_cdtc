// kernel_pure_cd.cu — synthetic CUDA-Core FP32 FFMA kernel.
//
// Goal: occupy the CUDA Core (CD) FP32 pipeline only, no Tensor Cores.
// Each thread does (n_iters * UNROLL * NACC) FFMA ops in registers.
// No HBM access in the hot loop. SASS must contain only FFMA / FMUL / FADD
// (NO HMMA, NO GMMA).
//
// Step 2.1 / multiplexing_h100_cdtc

#include <cuda_runtime.h>
#include <cstdio>

// Unroll factor for the inner loop (more unroll -> more FFMA back-to-back).
constexpr int CD_UNROLL = 16;
// Independent accumulator chains (ILP -> hide FFMA latency).
constexpr int CD_NACC   = 4;

extern "C" __global__ void kernel_pure_cd(int n_iters, float seed, float* out) {
    // Build per-thread constants from runtime args so compiler can't fold.
    float a = seed + (float)threadIdx.x * 0.001f;
    float b = a * 1.0001f;
    float c = b + 0.5f;
    float d = c * 0.9999f;

    // Independent accumulator chains -> 4-way ILP (each can issue per-cycle).
    float acc0 = 1.0f;
    float acc1 = 1.0f;
    float acc2 = 1.0f;
    float acc3 = 1.0f;

    #pragma unroll 1                       // keep outer loop rolled
    for (int i = 0; i < n_iters; ++i) {
        #pragma unroll                      // fully unroll inner -> CD_UNROLL FFMAs back to back
        for (int k = 0; k < CD_UNROLL; ++k) {
            acc0 = a * acc0 + b;
            acc1 = c * acc1 + d;
            acc2 = a * acc2 + d;
            acc3 = c * acc3 + b;
        }
    }

    // Defeat DCE: write to global memory exactly once (single thread per grid).
    // The compiler cannot prove this never executes, so it keeps the FFMA chain.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = acc0 + acc1 + acc2 + acc3;
    }
}

// Host wrapper. Caller pre-allocates out_device (>= 1 float).
extern "C" void launch_pure_cd(int grid_size, int block_size, int n_iters,
                               float* out_device, cudaStream_t stream = 0) {
    kernel_pure_cd<<<grid_size, block_size, 0, stream>>>(n_iters, 1.0001f, out_device);
}

// FLOP accounting (host-side helper for the driver):
//   per thread per outer iter : CD_UNROLL * CD_NACC FFMA = 64 FFMA = 128 FP32 ops
//   total FLOPS = grid * block * n_iters * CD_UNROLL * CD_NACC * 2
extern "C" long long pure_cd_flops(int grid_size, int block_size, int n_iters) {
    return (long long)grid_size * block_size * n_iters * CD_UNROLL * CD_NACC * 2LL;
}
