// bench_isolated.cu — isolated micro-bench driver (Step 2.1 / 2.1.5a).
//
// Subcommands:
//   default                              -> Step 2.1 sweep: grid × n_iters,
//                                           kernel_type ∈ {pure_cd_fp16, pure_tc},
//                                           block_size=128. Emits Step-2.1-style CSV.
//   block-sweep [n_iters] [grid]         -> Step 2.1.5a: CD-only sweep across
//                                           block_size ∈ {128,256,512,1024} at fixed
//                                           grid (default 132) and fixed n_iters
//                                           (default 10000). Emits sweep CSV with
//                                           occupancy columns.
//   rebaseline-cd-fp16 <block_size>      -> Step 2.1.5a: CD-only sweep over the
//                                           Step-2.1 grid × n_iters matrix at the
//                                           specified block_size. Emits rows ready
//                                           to append to isolated.csv.
//   profile <pure_cd|pure_tc> <grid> <iters> -> Run one config many times for
//                                                ncu attach (no CSV output).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Kernel decls (defined in kernel_pure_cd.cu / kernel_pure_tc.cu).
extern "C" __global__ void kernel_pure_cd(int n_iters, __half2 seed_a, __half2 seed_b, __half2* out);
extern "C" __global__ void kernel_pure_tc(int n_iters, const __half* data_in, __half* out);

extern "C" void      launch_pure_cd(int grid, int block, int n_iters, __half2* out, cudaStream_t s);
extern "C" void      launch_pure_tc(int grid, int block, int n_iters,
                                    const __half* data, __half* out, cudaStream_t s);
extern "C" long long pure_cd_flops(int grid, int block, int n_iters);
extern "C" long long pure_tc_flops(int grid, int block, int n_iters);
extern "C" int       pure_cd_out_elems_per_thread();

#define CK(x) do { cudaError_t _err = (x); if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// --- H100 SXM5 peak references (TFLOPS) ----------------------------------
constexpr double H100_CD_FP32_PEAK = 67.0;    // FP32 FFMA (Step 2.1 reference, no longer used)
constexpr double H100_CD_FP16_PEAK = 133.8;   // FP16 HFMA2 (Step 2.1.5a active reference)
constexpr double H100_TC_FP16_PEAK = 989.4;   // FP16 HMMA/WGMMA dense

// --- timing -------------------------------------------------------------
struct TimingStats {
    float median_us, p10_us, p90_us, min_us, max_us;
};

template <typename Launch>
TimingStats time_kernel(Launch launch, int warmup, int repeat) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));
    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> times;
    times.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        CK(cudaEventRecord(s));
        launch();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, s, e));
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    auto pct = [&](double p) {
        long n = (long)times.size();
        long i = (long)std::round(p * (n - 1));
        if (i < 0) i = 0;
        if (i >= n) i = n - 1;
        return times[(size_t)i] * 1000.0f;
    };
    CK(cudaEventDestroy(s));
    CK(cudaEventDestroy(e));
    return TimingStats{pct(0.50), pct(0.10), pct(0.90), pct(0.00), pct(1.00)};
}

// --- shared allocation state --------------------------------------------
struct Buffers {
    __half2* d_out_cd  = nullptr;   // sized to max grid × block × CD_NACC
    size_t   cd_capacity_h2 = 0;
    __half*  d_in_tc   = nullptr;   // 512 halves (2 × 16×16)
    __half*  d_out_tc  = nullptr;   // 8 × 256 halves
};

void ensure_cd_capacity(Buffers& b, size_t needed_h2) {
    if (needed_h2 <= b.cd_capacity_h2) return;
    if (b.d_out_cd) cudaFree(b.d_out_cd);
    CK(cudaMalloc(&b.d_out_cd, needed_h2 * sizeof(__half2)));
    b.cd_capacity_h2 = needed_h2;
}

Buffers make_buffers() {
    Buffers b;
    CK(cudaMalloc(&b.d_in_tc,  512 * sizeof(__half)));
    CK(cudaMalloc(&b.d_out_tc, 8 * 256 * sizeof(__half)));
    std::vector<__half> h_in(512);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = __float2half(0.01f * (float)(i % 17) + 0.001f);
    CK(cudaMemcpy(b.d_in_tc, h_in.data(), 512 * sizeof(__half), cudaMemcpyHostToDevice));
    return b;
}

void free_buffers(Buffers& b) {
    if (b.d_out_cd) cudaFree(b.d_out_cd);
    cudaFree(b.d_in_tc);
    cudaFree(b.d_out_tc);
}

// --- subcommand: default sweep (Step 2.1 style, CD now FP16) -------------
int run_default(Buffers& b) {
    const int block_size = 128;
    const int warmup     = 10;
    const int repeat     = 100;
    std::vector<int> grids = {1, 33, 66, 132};
    std::vector<int> iters = {1000, 10000, 100000};

    std::printf("kernel_type,grid_size,block_size,n_iters,"
                "latency_us_median,latency_us_p10,latency_us_p90,"
                "latency_us_min,latency_us_max,"
                "tflops_achieved,pct_of_peak\n");
    std::fflush(stdout);

    for (int g : grids) {
        for (int n : iters) {
            // pure_cd_fp16
            size_t need = (size_t)g * block_size * pure_cd_out_elems_per_thread();
            ensure_cd_capacity(b, need);
            TimingStats t = time_kernel(
                [&]{ launch_pure_cd(g, block_size, n, b.d_out_cd, 0); }, warmup, repeat);
            long long fl = pure_cd_flops(g, block_size, n);
            double tflops = (double)fl / (double)t.median_us / 1e6;
            double pct    = 100.0 * tflops / H100_CD_FP16_PEAK;
            std::printf("pure_cd_fp16,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                        g, block_size, n, t.median_us, t.p10_us, t.p90_us,
                        t.min_us, t.max_us, tflops, pct);
            std::fflush(stdout);

            // pure_tc (unchanged from Step 2.1)
            TimingStats tt = time_kernel(
                [&]{ launch_pure_tc(g, block_size, n, b.d_in_tc, b.d_out_tc, 0); },
                warmup, repeat);
            long long flt = pure_tc_flops(g, block_size, n);
            double tflops_t = (double)flt / (double)tt.median_us / 1e6;
            double pct_t    = 100.0 * tflops_t / H100_TC_FP16_PEAK;
            std::printf("pure_tc,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                        g, block_size, n, tt.median_us, tt.p10_us, tt.p90_us,
                        tt.min_us, tt.max_us, tflops_t, pct_t);
            std::fflush(stdout);
        }
    }
    return 0;
}

// --- subcommand: block-sweep (Step 2.1.5a Task C) -----------------------
int run_block_sweep(Buffers& b, int grid_size, int n_iters) {
    const int warmup = 10;
    const int repeat = 100;
    std::vector<int> blocks = {128, 256, 512, 1024};

    // One-time: kernel resource attributes.
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, (const void*)kernel_pure_cd));
    int regs = attr.numRegs;
    int smem_static = (int)attr.sharedSizeBytes;

    std::printf("block_size,warps_per_block,warps_per_smsp,"
                "registers_per_thread,static_smem_bytes,"
                "max_resident_blocks_per_sm,max_active_warps_per_sm,"
                "grid_size,n_iters,"
                "latency_us_median,latency_us_p10,latency_us_p90,"
                "latency_us_min,latency_us_max,"
                "tflops_achieved,pct_of_peak\n");
    std::fflush(stdout);

    for (int bs : blocks) {
        int max_blocks_per_sm = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, (const void*)kernel_pure_cd, bs, 0));
        int warps_per_block = bs / 32;
        int warps_per_smsp  = warps_per_block / 4;  // H100 has 4 SMSPs/SM
        int max_active_warps = max_blocks_per_sm * warps_per_block;

        size_t need = (size_t)grid_size * bs * pure_cd_out_elems_per_thread();
        ensure_cd_capacity(b, need);

        TimingStats t = time_kernel(
            [&]{ launch_pure_cd(grid_size, bs, n_iters, b.d_out_cd, 0); }, warmup, repeat);
        long long fl = pure_cd_flops(grid_size, bs, n_iters);
        double tflops = (double)fl / (double)t.median_us / 1e6;
        double pct    = 100.0 * tflops / H100_CD_FP16_PEAK;

        std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                    bs, warps_per_block, warps_per_smsp, regs, smem_static,
                    max_blocks_per_sm, max_active_warps,
                    grid_size, n_iters,
                    t.median_us, t.p10_us, t.p90_us, t.min_us, t.max_us,
                    tflops, pct);
        std::fflush(stdout);
    }
    return 0;
}

// --- subcommand: rebaseline-cd-fp16 at a given block_size ---------------
int run_rebaseline_cd(Buffers& b, int block_size) {
    const int warmup = 10;
    const int repeat = 100;
    std::vector<int> grids = {1, 33, 66, 132};
    std::vector<int> iters = {10000, 100000};

    std::printf("kernel_type,grid_size,block_size,n_iters,"
                "latency_us_median,latency_us_p10,latency_us_p90,"
                "latency_us_min,latency_us_max,"
                "tflops_achieved,pct_of_peak\n");
    std::fflush(stdout);

    std::string tag = std::string("pure_cd_fp16_block") + std::to_string(block_size);
    for (int g : grids) {
        for (int n : iters) {
            size_t need = (size_t)g * block_size * pure_cd_out_elems_per_thread();
            ensure_cd_capacity(b, need);
            TimingStats t = time_kernel(
                [&]{ launch_pure_cd(g, block_size, n, b.d_out_cd, 0); }, warmup, repeat);
            long long fl = pure_cd_flops(g, block_size, n);
            double tflops = (double)fl / (double)t.median_us / 1e6;
            double pct    = 100.0 * tflops / H100_CD_FP16_PEAK;
            std::printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                        tag.c_str(), g, block_size, n,
                        t.median_us, t.p10_us, t.p90_us,
                        t.min_us, t.max_us, tflops, pct);
            std::fflush(stdout);
        }
    }
    return 0;
}

// --- subcommand: profile (for NCU attach) -------------------------------
int run_profile(Buffers& b, const std::string& kind, int grid, int n_iters) {
    const int block_size = 128;
    for (int i = 0; i < 110; ++i) {
        if (kind == "pure_cd" || kind == "pure_cd_fp16") {
            size_t need = (size_t)grid * block_size * pure_cd_out_elems_per_thread();
            ensure_cd_capacity(b, need);
            launch_pure_cd(grid, block_size, n_iters, b.d_out_cd, 0);
        } else {
            launch_pure_tc(grid, block_size, n_iters, b.d_in_tc, b.d_out_tc, 0);
        }
    }
    CK(cudaDeviceSynchronize());
    fprintf(stderr, "profile mode %s grid=%d iters=%d done\n", kind.c_str(), grid, n_iters);
    return 0;
}

// --- entrypoint ---------------------------------------------------------
int main(int argc, char** argv) {
    Buffers b = make_buffers();

    std::string sub = argc > 1 ? argv[1] : "";
    int rc = 0;
    if (sub == "block-sweep") {
        int n_iters = argc > 2 ? std::atoi(argv[2]) : 10000;
        int grid    = argc > 3 ? std::atoi(argv[3]) : 132;
        rc = run_block_sweep(b, grid, n_iters);
    } else if (sub == "rebaseline-cd-fp16") {
        int bs = argc > 2 ? std::atoi(argv[2]) : 256;
        rc = run_rebaseline_cd(b, bs);
    } else if (sub == "profile") {
        std::string kind = argc > 2 ? argv[2] : "pure_cd_fp16";
        int grid = argc > 3 ? std::atoi(argv[3]) : 132;
        int iters = argc > 4 ? std::atoi(argv[4]) : 10000;
        rc = run_profile(b, kind, grid, iters);
    } else if (sub.empty() || sub == "default") {
        rc = run_default(b);
    } else {
        fprintf(stderr, "unknown subcommand: %s\n", sub.c_str());
        rc = 1;
    }
    free_buffers(b);
    return rc;
}
