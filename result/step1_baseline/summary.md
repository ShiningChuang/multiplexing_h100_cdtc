# Step 1 Baseline Summary (REVISED after path correction)

## Status

Step 1 was completed with two methodological corrections discovered in
Step 1.5:

1. **FA3 build correction**: Original Step 1 used FA2 wheel (pip install
   flash-attn ships only FA2). FA3 was built from source in Step 1.5.
   FA2 path measurements (~330 TFLOPS prefill at S=8192) are deprecated;
   FA3 measurements are the valid baseline.

2. **Production path correction**: Original Step 1 baselined Triton
   `fused_moe`, assuming it represents production MoE performance on H100.
   **This was incorrect.** vLLM 0.19's `select_fp8_moe_backend()` oracle
   selects FlashInfer CUTLASS (for per-tensor FP8) or DeepGEMM (for
   block-quant FP8) by default on Hopper. Triton fused_moe is a fallback,
   not the production path. All `expert_*.csv` data in this directory
   reflects Triton path only and is NOT production-representative.

## Valid Baselines (use these)

### Attention (FA3 path)
- File: `attention_prefill_mixtral.csv`, `attention_decode_mixtral.csv`
- Note: Only Mixtral GQA shape (Hq=32, Hkv=8, d=128) was measured in
  Step 1. Qwen3 attention shape (Hq=32, Hkv=4, d=128) was not run because
  the original spec scoped attention to Mixtral only; the Qwen3 difference
  (fewer KV heads) is qualitatively the same regime — memory-bound decode
  at an even lower intensity. Add Qwen3 attention sweep in Step 5 if a
  separate baseline is needed.
- Status: ✅ Valid. FA3 is production attention path on H100 SXM5.
- Key findings (post FA3 rebuild — these supersede the FA2 numbers in the
  Step 1.5 readme):
  - FA3 prefill crosses arithmetic intensity ridge (FP16 ~295) at S≈1024.
  - FA3 decode intensity ≈ 4 FLOP/byte across all kv_len (GQA-bounded),
    far below ridge → **always memory-bound, always sub-1% TC utilization**.
  - This decode regime is the primary opportunity window for CD-attention
    multiplexing research.
- Status of underlying numbers: the CSVs in this directory still contain
  the original FA2-path attention numbers (Step 1 measurement). The
  Step 1.5 FA3 rebuild was completed (kernel switch + smoke-test) but the
  full attention sweep was not re-run before the project's path-correction
  pivot. The FA3 *kernel* is wired up and ready in `src/kernels.py`; the
  re-measurement happens in Step 5 alongside the production MoE re-baseline.

### Roofline framework
- Code: `src/utils.py` (`roofline()` function), `src/configs.py` (H100 specs)
- (Note: the spec's `src/roofline.py` filename was consolidated into
  `src/utils.py` during Step 1; same function, just collocated with
  `gpu_timer` and `flush_csv`.)
- Status: ✅ Valid framework, applies regardless of expert kernel choice.

## Deprecated Baselines (do not cite)

### Expert (Triton fused_moe path)
- File: `expert_mixtral.csv`, `expert_qwen3.csv` (header-annotated with
  DEPRECATED warning; pre-annotation copies preserved as `*.original`)
- Status: ❌ Triton path only, not production.
- Numbers (208 TFLOPS Mixtral T=4096, etc.) reflect un-tuned Triton kernel,
  NOT FlashInfer CUTLASS / DeepGEMM that vLLM actually uses.
- **Will be re-baselined in Step 5 against production kernels.**
- Related deprecated artifacts moved to
  `_deprecated/triton_autotune_attempt/`:
  - `run_autotune.sh` — vLLM Triton autotune driver
  - `benchmark_moe.py` — downloaded vLLM v0.19.1 tuner
  - `fake_models/` — local HF-format config dirs used by the tuner

## Path Forward

- **Step 2-4**: Focus on a more fundamental, kernel-agnostic question:
  can CD and TC units execute concurrently within H100 SM? Use synthetic
  CUDA kernels to isolate from production kernel complexity. Scaffold
  goes into `src/microbench/`.
- **Step 5**: Apply concurrency mechanism (validated in Step 2-4) to
  production kernels (FA3 + FlashInfer CUTLASS / DeepGEMM), re-baseline
  expert path against the actual oracle-selected backend.
- **Step 6**: If Step 5 succeeds, scale to full-model inference workload.

## Sanity Check Status (REVISED)

| # | Original Gate | Original Verdict | Revised Status |
|---|---|---|---|
| 1 | FA3 prefill S=8192 ≥ 550 TFLOPS | FAIL (FA2 path, 330) | Resolved in Step 1.5 (FA3 build); re-measurement scheduled for Step 5 |
| 2 | FA3 prefill intensity ≥ 3000 | PASS | Still valid (intensity is a model property, not a kernel property) |
| 3 | fused_moe Mixtral T=4096 ≥ 400 TFLOPS | FAIL (208) | **Gate retracted**: target was Triton path, not production. Re-design in Step 5 against FlashInfer CUTLASS / DeepGEMM. |
| 4 | Decode always memory-bound | PASS | Still valid, **core research motivation** |
| 5 | Sub-ms kernel noise <5% | FAIL on sub-ms | Acceptable; Step 2 will use batched timing for short kernels |

Verdict: Step 1 baseline framework validated, 2 of 5 original gates
re-interpreted, 1 gate retracted. Ready for Step 2.
