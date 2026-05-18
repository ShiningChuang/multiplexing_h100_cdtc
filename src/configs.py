"""H100 SXM5 hardware constants and model presets for the multiplexing study.

All numbers come from NVIDIA's official H100 SXM5 80GB HBM3 datasheet
(dense throughput, no sparsity).
"""

from dataclasses import dataclass
import torch

# --- H100 SXM5 80GB HBM3 official specs ---------------------------------------
H100_FP16_TENSORCORE_TFLOPS = 989.4   # dense, no sparsity
H100_FP16_CUDACORE_TFLOPS   = 133.8   # peak non-Tensor FP16
H100_FP8_TENSORCORE_TFLOPS  = 1978.9  # dense, no sparsity (for future ref)
H100_HBM_BW_GBS             = 3350.0
H100_SM_COUNT               = 132
H100_L2_CACHE_MB            = 50
H100_SHARED_MEM_PER_SM_KB   = 228

# --- Roofline ridge points (FLOP / byte) --------------------------------------
H100_RIDGE_FP16_TC = H100_FP16_TENSORCORE_TFLOPS * 1e3 / H100_HBM_BW_GBS  # ~295
H100_RIDGE_FP16_CD = H100_FP16_CUDACORE_TFLOPS   * 1e3 / H100_HBM_BW_GBS  # ~40


@dataclass
class ModelConfig:
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    dtype: torch.dtype
    device: str = "cuda"


MIXTRAL_FP16 = ModelConfig(
    name="mixtral-8x7b",
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    hidden=4096,
    intermediate=14336,
    num_experts=8,
    top_k=2,
    dtype=torch.float16,
)

QWEN3_FP16 = ModelConfig(
    name="qwen3-30b-a3b",
    num_q_heads=32,
    num_kv_heads=4,
    head_dim=128,
    hidden=2048,
    intermediate=768,
    num_experts=128,
    top_k=8,
    dtype=torch.float16,
)


# --- Experiment grids ---------------------------------------------------------
PREFILL_SEQ_LIST    = [128, 256, 512, 1024, 2048, 4096, 8192]
DECODE_KV_LIST      = [512, 1024, 2048, 4096, 8192, 16384]
MOE_TOKEN_LIST_MIX  = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
MOE_TOKEN_LIST_QWEN = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
WARMUP = 20
REPEAT = 200
