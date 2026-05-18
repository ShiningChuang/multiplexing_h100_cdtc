# multiplexing_h100_cdtc

GPU multiplexing research project on H100 SXM5.

## Path Correction (Step 1.5 Discovery)

This project's original Step 1 plan assumed vLLM's default H100 FP8 MoE
path uses Triton `fused_moe`. Investigation in Step 1.5 revealed this
assumption was wrong:

- vLLM 0.19 uses `select_fp8_moe_backend()` to choose between
  `FLASHINFER_CUTLASS`, `DEEPGEMM`, `TRITON`, and others.
- On Hopper + FP8, Triton is **not the default**. FlashInfer CUTLASS is
  selected for per-tensor FP8 (Mixtral 8x7B), and DeepGEMM is preferred
  for block-quant FP8 (Qwen3-30B-A3B uses `block_shape=[128,128]`).
- Reference: Issue #34249, `vllm/model_executor/layers/fused_moe/oracle/fp8.py`.

**Consequence**: Triton fused_moe autotune (originally planned in Step 1.5)
is **not relevant to production path** and has been removed from the
project. Production MoE baseline will be established in Step 5 using
FlashInfer CUTLASS / DeepGEMM directly.

**Research scope refinement**: Step 2-4 now focus on a more fundamental
question — whether CUDA Core (CD) and Tensor Core (TC) units can execute
concurrently within H100 SM, using synthetic kernels to isolate from
production kernel complexity. This is a hardware-level capability test
that is kernel-framework-agnostic. If Step 2-4 demonstrate concurrency
feasibility, Step 5 composes the mechanism with production kernels
(FA3 + FlashInfer CUTLASS / DeepGEMM).

## 1. Project goal

We are investigating whether running **attention on CUDA Cores** and
**MoE expert GEMMs on Tensor Cores** concurrently on the same SMs (CDTC
co-execution) can lift end-to-end throughput on H100 versus the standard
time-sliced execution where attention and experts both contend for Tensor
Cores. The thesis is that decode-style attention is bandwidth-bound and
leaves the Tensor Cores idle, while MoE prefill expert GEMMs are
compute-bound on Tensor Cores — so the two should fit into the same SM
without stepping on each other.

## 2. Datatype path (revised in Step 1.5)

The project initially targeted FP16 (TC/CD ratio ≈ 7.4×) on the hypothesis
that the smaller compute-gap matches the regime where prior CDTC work
(Plasticine/Tacker on V100/2080Ti) showed gains. **In Step 1.5 this was
re-pointed to FP8** because:

1. Production deployments of Mixtral / Qwen3 on H100 are FP8, not FP16.
2. FP16 makes Mixtral 8x7B exceed single-H100 capacity (94 GB > 80 GB);
   FP8 fits (~47 GB). FP16 multiplexing on a model that doesn't fit
   isn't a real research scenario.
3. Under FP8 the TC/CD ratio is ~14.8×, but CD-path attention is still
   viable in the memory-bound regime (decode, small-S prefill). This
   actually *sharpens* the research framing: "which regimes admit
   CD-multiplexing under aggressive Tensor Core peaks?"

The FP16 baseline numbers in `result/step1_baseline/` remain in place
as the historical record of Step 1 (with their deprecation status
documented), but Step 5 will re-baseline on the FP8 production path.

## 3. Hardware / software stack (this checkout, post Step 1.5)

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (SXM5), 132 SMs |
| Driver | 590.48.01 |
| CUDA Toolkit | 12.4 |
| PyTorch | 2.10.0+cu128 (upgraded from 2.8.0 by vLLM 0.19.1's hard pin) |
| flash-attn / FA3 | 2.8.3 (FA2 ABI-broken after torch upgrade) + `flash_attn_3` 3.0.0 (built from source, exposes `flash_attn_interface`) |
| vLLM | 0.19.1 (spec target) |
| Triton | 3.6.0 (vLLM pin) |
| cuda-python (Green Context API) | 12.9.4 |

## 4. Experiment roadmap (updated in Step 1.5)

| Step | Scope | Status |
| --- | --- | --- |
| 1. FP16 isolated baseline | FA3 prefill/decode + Triton `fused_moe` sweeps, roofline. | done; attention findings retained; Triton expert path deprecated (see Path Correction). |
| 1.5. Environment correction | vLLM 0.19.1 upgrade, FA3 source build, kernels.py rewire, path-discovery investigation. | done. |
| 2. CDTC concurrency micro-bench | Synthetic CD-only and TC-only kernels in `src/microbench/`; measure whether they can co-run inside one H100 SM via Green Context / streams. Kernel-agnostic. | next. |
| 3. Concurrency primitive | Pick the mechanism that worked in Step 2 (Green Context partition, stream priority, or PTB-style cooperative scheduling) and harden it. | blocked on Step 2. |
| 4. Synthetic workload integration | Wire CD-attention proxy + TC-GEMM proxy through the Step 3 primitive end-to-end. | blocked on Step 3. |
| 5. Production-path re-baseline | Apply Step 3 primitive to FA3 + FlashInfer CUTLASS / DeepGEMM (the actual H100 FP8 production kernels). Re-measure expert path against the oracle-selected backend. | blocked on Step 4. |
| 6. (optional) Full-model integration | Compose into a vLLM-served inference workload; measure throughput / latency / energy. | blocked on Step 5. |

## 5. Repo layout

```
multiplexing_h100_cdtc/
├── README.md                    # this file (incl. Path Correction at top)
├── script/
│   ├── env.sh                   # PATH + venv + PYTHONPATH bootstrap (source it)
│   └── verify_env.sh            # 15-item PASS/FAIL/WARN sanity check
├── src/
│   ├── configs.py               # H100 constants + ModelConfig presets + sweep grids
│   ├── utils.py                 # gpu_timer (CUDA events), roofline, flush_csv
│   ├── kernels.py               # FA3 attention; vLLM Triton MoE wrapper retained as comparator
│   ├── bench_baseline_h100_fp16.py   # 4 isolated baseline sweeps (Step 1 driver)
│   └── microbench/              # (empty) — Step 2 CD/TC concurrency micro-bench scaffold
├── result/
│   └── step1_baseline/
│       ├── summary.md           # narrative + sanity-check verdict (revised after Step 1.5)
│       ├── attention_prefill_mixtral.csv  # valid, annotated
│       ├── attention_decode_mixtral.csv   # valid, annotated
│       ├── expert_mixtral.csv             # DEPRECATED, annotated
│       ├── expert_qwen3.csv               # DEPRECATED, annotated
│       ├── *.csv.original                 # pre-annotation backups
│       └── run.log
├── configs/                     # project-managed config artifacts (Step 5 will populate)
├── plan/
│   └── step1_baseline.md        # archived Step 1 prompt
├── tools/                       # (empty) — Triton autotune tools removed in Step 1.5
└── _deprecated/
    └── triton_autotune_attempt/
        ├── run_autotune.sh      # vLLM Triton autotune driver
        ├── benchmark_moe.py     # downloaded vLLM v0.19.1 tuner
        └── fake_models/         # local HF-format config dirs (Mixtral, Qwen3)
```

## 6. How to run (Step 1 artifacts)

```bash
source script/env.sh
bash   script/verify_env.sh      # current: 14 PASS / 1 WARN (FA2 ABI) / 0 FAIL
# Step 1 baseline driver is intact; re-running overwrites annotations.
# Re-run only for development; production re-baseline happens in Step 5.
# python -m src.bench_baseline_h100_fp16
```

## 7. Step-1 outputs

- `result/step1_baseline/attention_prefill_mixtral.csv` (valid baseline)
- `result/step1_baseline/attention_decode_mixtral.csv`  (valid baseline)
- `result/step1_baseline/expert_mixtral.csv`            (deprecated — Triton path)
- `result/step1_baseline/expert_qwen3.csv`              (deprecated — Triton path)
- `result/step1_baseline/summary.md` — revised narrative + sanity-check verdict

## 8. Trigger conditions for step 2

Step 2 begins when:
1. `result/step1_baseline/summary.md` reflects revised verdict (✅ done in Step 1.5 cleanup).
2. The user has reviewed the path-correction rationale and approved
   pivot to synthetic CD/TC concurrency micro-bench instead of
   production-MoE autotune.
