// kernel_pure_tc.cu — synthetic Tensor-Core (HMMA) kernel.
//
// Goal: occupy the Tensor Core pipeline only. Each warp issues mma_sync
// repeatedly into multiple accumulators to pipeline HMMA instructions.
// SASS must contain HMMA.* (and, depending on nvcc choices on sm_90a,
// optionally GMMA.*). Main-loop FFMA count should be ~0 (a few in the
// store_matrix_sync epilogue is acceptable).
//
// Step 2.1 / multiplexing_h100_cdtc

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

// Each warp owns NACC accumulators -> ILP -> hides HMMA latency.
constexpr int TC_NACC   = 8;
constexpr int TC_M      = 16;
constexpr int TC_N      = 16;
constexpr int TC_K      = 16;

extern "C" __global__ void kernel_pure_tc(int n_iters, const __half* data_in, __half* out) {
    using FragA = wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, __half, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, __half, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, __half>;

    FragA frag_a;
    FragB frag_b;
    FragC frag_c[TC_NACC];

    // Shared-memory tiles: one A tile + one B tile, reused every iteration.
    __shared__ __half smem_a[TC_M * TC_K];
    __shared__ __half smem_b[TC_K * TC_N];

    // Initial population: cooperatively load 256 halves each.
    int tid = threadIdx.x;
    if (tid < TC_M * TC_K) smem_a[tid] = data_in[tid];
    if (tid < TC_K * TC_N) smem_b[tid] = data_in[tid + TC_M * TC_K];
    __syncthreads();

    wmma::load_matrix_sync(frag_a, smem_a, TC_K);
    wmma::load_matrix_sync(frag_b, smem_b, TC_K);

    // Distinct init per accumulator so compiler can't CSE-fold the chains
    // (with identical inits and identical (a,b), the optimizer collapses 8 -> 2).
    #pragma unroll
    for (int j = 0; j < TC_NACC; ++j) {
        wmma::fill_fragment(frag_c[j], __float2half(0.001f * (float)(j + 1)));
    }

    // Hot loop: TC_NACC * n_iters HMMAs per warp.
    #pragma unroll 1
    for (int i = 0; i < n_iters; ++i) {
        #pragma unroll
        for (int j = 0; j < TC_NACC; ++j) {
            wmma::mma_sync(frag_c[j], frag_a, frag_b, frag_c[j]);
        }
    }

    // Defeat DCE: store every accumulator separately. Distinct outputs +
    // distinct inits => compiler must keep all NACC HMMA chains.
    if (blockIdx.x == 0 && (tid / 32) == 0) {
        #pragma unroll
        for (int j = 0; j < TC_NACC; ++j) {
            wmma::store_matrix_sync(out + j * (TC_M * TC_N),
                                    frag_c[j], TC_N, wmma::mem_row_major);
        }
    }
}

// Host wrapper. data_device must point to >= 2 * TC_M * TC_K halves (we read
// 16x16 A and 16x16 B = 512 halves total). out_device must hold >= 256 halves.
extern "C" void launch_pure_tc(int grid_size, int block_size, int n_iters,
                               const __half* data_device, __half* out_device,
                               cudaStream_t stream = 0) {
    kernel_pure_tc<<<grid_size, block_size, 0, stream>>>(n_iters, data_device, out_device);
}

// FLOP accounting:
//   per warp per outer iter : TC_NACC HMMA = TC_NACC * (M*N*K*2) FP16 FMA ops
//   warps per block         : block_size / 32
//   total FLOPS = grid * warps_per_block * n_iters * TC_NACC * M*N*K * 2
extern "C" long long pure_tc_flops(int grid_size, int block_size, int n_iters) {
    long long warps = block_size / 32;
    long long mnk2  = (long long)TC_M * TC_N * TC_K * 2LL;
    return (long long)grid_size * warps * n_iters * TC_NACC * mnk2;
}
