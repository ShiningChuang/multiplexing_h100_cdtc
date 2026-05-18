// bench_isolated.cu — Step 2.1 isolated micro-benchmark driver.
//
// Sweeps {grid_size} x {n_iters} for kernel_pure_cd and kernel_pure_tc.
// Times each config with CUDA events (median of `repeat` after `warmup`),
// computes achieved TFLOPS, emits CSV to stdout.
//
// SM-utilization (`sm__cycles_active.avg.pct_of_peak_sustained_elapsed`) is
// NOT computed here — that requires Nsight Compute. See `tools/run_ncu.sh`
// for the post-processing pass that fills the sm_util_pct column.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" void      launch_pure_cd(int grid, int block, int n_iters, float* out, cudaStream_t s);
extern "C" void      launch_pure_tc(int grid, int block, int n_iters,
                                    const __half* data, __half* out, cudaStream_t s);
extern "C" long long pure_cd_flops(int grid, int block, int n_iters);
extern "C" long long pure_tc_flops(int grid, int block, int n_iters);

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s at %s:%d\n", cudaGetErrorString(_ck_err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

struct TimingStats {
    float median_us;
    float p10_us;
    float p90_us;
    float min_us;
    float max_us;
};

template <typename Launch>
TimingStats time_kernel(Launch launch, int warmup, int repeat) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));

    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());

    std::vector<float> times_ms;
    times_ms.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        CK(cudaEventRecord(s));
        launch();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, s, e));
        times_ms.push_back(ms);
    }
    std::sort(times_ms.begin(), times_ms.end());
    auto idx = [&](double p) {
        long n = (long)times_ms.size();
        long i = (long)std::round(p * (n - 1));
        if (i < 0) i = 0;
        if (i >= n) i = n - 1;
        return times_ms[(size_t)i];
    };
    TimingStats out;
    out.min_us    = times_ms.front() * 1000.0f;
    out.max_us    = times_ms.back()  * 1000.0f;
    out.p10_us    = idx(0.10) * 1000.0f;
    out.p90_us    = idx(0.90) * 1000.0f;
    out.median_us = idx(0.50) * 1000.0f;

    CK(cudaEventDestroy(s));
    CK(cudaEventDestroy(e));
    return out;
}

int main(int argc, char** argv) {
    const int block_size = 128;
    const int warmup     = 10;
    const int repeat     = 100;
    std::vector<int> grids = {1, 33, 66, 132};
    std::vector<int> iters = {1000, 10000, 100000};

    // Allow CLI override: --kernels-only or --profile <kernel> <grid> <iters>
    // (the latter is used by run_ncu.sh to isolate one config per launch).
    const char* profile_mode = nullptr;
    int profile_grid = 0, profile_iters = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--profile" && i + 3 < argc) {
            profile_mode  = argv[i + 1];
            profile_grid  = std::atoi(argv[i + 2]);
            profile_iters = std::atoi(argv[i + 3]);
            i += 3;
        }
    }

    // Allocations.
    float* d_out_cd = nullptr;
    CK(cudaMalloc(&d_out_cd, 4 * sizeof(float)));

    __half* d_in_tc  = nullptr;
    __half* d_out_tc = nullptr;
    CK(cudaMalloc(&d_in_tc,  512 * sizeof(__half)));   // 2 * 16*16 (A + B tiles)
    CK(cudaMalloc(&d_out_tc, 8 * 256 * sizeof(__half))); // TC_NACC * 16*16 accumulator outputs
    // Init data_in with non-trivial values so HMMA produces non-zero results.
    std::vector<__half> h_in(512);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = __float2half(0.01f * (float)(i % 17) + 0.001f);
    CK(cudaMemcpy(d_in_tc, h_in.data(), 512 * sizeof(__half), cudaMemcpyHostToDevice));

    // Profile mode: launch one kernel many times (no timing CSV), so NCU can
    // attach and read SM cycles. The wrapping shell script handles the rest.
    if (profile_mode) {
        std::string m = profile_mode;
        for (int i = 0; i < warmup + repeat; ++i) {
            if (m == "pure_cd") launch_pure_cd(profile_grid, block_size, profile_iters, d_out_cd, 0);
            else                launch_pure_tc(profile_grid, block_size, profile_iters, d_in_tc, d_out_tc, 0);
        }
        CK(cudaDeviceSynchronize());
        fprintf(stderr, "profile mode %s grid=%d iters=%d done\n", profile_mode, profile_grid, profile_iters);
        cudaFree(d_out_cd); cudaFree(d_in_tc); cudaFree(d_out_tc);
        return 0;
    }

    // CSV header.
    std::printf("kernel_type,grid_size,block_size,n_iters,"
                "latency_us_median,latency_us_p10,latency_us_p90,"
                "latency_us_min,latency_us_max,"
                "tflops_achieved,pct_of_peak\n");
    std::fflush(stdout);

    // H100 SXM5 peaks for the percent column.
    constexpr double H100_CD_FP32_TFLOPS = 67.0;   // FP32 FFMA peak
    constexpr double H100_TC_FP16_TFLOPS = 989.4;  // FP16 dense peak (WGMMA path)

    for (int g : grids) {
        for (int n : iters) {
            // -------- pure_cd --------
            TimingStats t_cd = time_kernel(
                [&] { launch_pure_cd(g, block_size, n, d_out_cd, 0); },
                warmup, repeat);
            long long flops_cd = pure_cd_flops(g, block_size, n);
            double tflops_cd   = (double)flops_cd / (double)t_cd.median_us / 1e6;
            double pct_cd      = 100.0 * tflops_cd / H100_CD_FP32_TFLOPS;
            std::printf("pure_cd,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                        g, block_size, n,
                        t_cd.median_us, t_cd.p10_us, t_cd.p90_us,
                        t_cd.min_us, t_cd.max_us,
                        tflops_cd, pct_cd);
            std::fflush(stdout);

            // -------- pure_tc --------
            TimingStats t_tc = time_kernel(
                [&] { launch_pure_tc(g, block_size, n, d_in_tc, d_out_tc, 0); },
                warmup, repeat);
            long long flops_tc = pure_tc_flops(g, block_size, n);
            double tflops_tc   = (double)flops_tc / (double)t_tc.median_us / 1e6;
            double pct_tc      = 100.0 * tflops_tc / H100_TC_FP16_TFLOPS;
            std::printf("pure_tc,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                        g, block_size, n,
                        t_tc.median_us, t_tc.p10_us, t_tc.p90_us,
                        t_tc.min_us, t_tc.max_us,
                        tflops_tc, pct_tc);
            std::fflush(stdout);
        }
    }

    cudaFree(d_out_cd);
    cudaFree(d_in_tc);
    cudaFree(d_out_tc);
    return 0;
}
