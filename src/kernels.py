"""kernels.py - kernel factory for attention and expert ops.

All allocations happen once, *outside* the timing loop, so gpu_timer measures
just the kernel launch + execution.

Attention path: FA3 only (flash_attn_interface). FA2 (`flash_attn`) is left
unimported because vllm 0.19.1's torch 2.10 upgrade ABI-broke its compiled
extension; we intentionally do *not* fall back to a non-working library.

⚠️ PATH NOTE (Step 1.5 path correction): The fused_experts wrapper below
calls vLLM's Triton fused_moe kernel. As of vLLM 0.19, this is **NOT** the
default production path on H100 FP8 — production uses FlashInfer CUTLASS
(per-tensor FP8, e.g. Mixtral 8x7B) or DeepGEMM (block-quant FP8, e.g.
Qwen3-30B-A3B with block_shape=[128,128]). See
`vllm/model_executor/layers/fused_moe/oracle/fp8.py::select_fp8_moe_backend`
and Issue #34249.

The Triton path is retained here for two purposes:
  (1) Step 2-4 may use it as a controllable expert proxy (Triton tile
      config is more transparent than CUTLASS persistent kernel).
  (2) Step 5 will benchmark production (FlashInfer/DeepGEMM) vs Triton
      and report relative performance.

For production-representative measurements, use FlashInfer CUTLASS /
DeepGEMM bindings (to be added in Step 5).

API note: vLLM 0.19.x removed the legacy `fused_moe()` wrapper; we compose
`fused_topk` + `fused_experts` directly. Same Triton kernel underneath.
"""

from __future__ import annotations

from typing import Callable

import torch

# --- FA3 (Hopper) — required, no FA2 fallback --------------------------------
try:
    from flash_attn_interface import (
        flash_attn_varlen_func as fa3_varlen_func,
        flash_attn_with_kvcache as fa3_with_kvcache,
    )
    FA3_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "FA3 (flash_attn_interface) is not available in this environment. "
        "This project requires FA3 built from source against the current torch. "
        "Build: cd ~/flash-attention/hopper && "
        "pip install --no-build-isolation . "
        f"(original ImportError: {e})"
    ) from e

# --- vLLM 0.19.x MoE entry points --------------------------------------------
from vllm.model_executor.layers.fused_moe import fused_topk, fused_experts

from .configs import ModelConfig


def make_fa3_prefill_fn(seq_len: int, cfg: ModelConfig) -> Callable[[], torch.Tensor]:
    """FA3 varlen prefill, causal=True, single sequence of length seq_len."""
    assert FA3_AVAILABLE, "FA3 must be importable to call this factory."
    device = cfg.device
    dtype = cfg.dtype

    q = torch.randn(seq_len, cfg.num_q_heads,  cfg.head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        # FA3's varlen returns either Tensor or (Tensor, lse) depending on params;
        # we only care about the timing path here.
        return fa3_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            causal=True,
        )

    return run


def make_fa3_decode_fn(kv_len: int, cfg: ModelConfig) -> Callable[[], torch.Tensor]:
    """FA3 decode: one query token attends over a kv_len-long cache."""
    assert FA3_AVAILABLE, "FA3 must be importable to call this factory."
    device = cfg.device
    dtype = cfg.dtype

    q = torch.randn(1, 1, cfg.num_q_heads, cfg.head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(1, kv_len, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(1, kv_len, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor([kv_len], dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return fa3_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            causal=False,
        )

    return run


def make_vllm_fused_moe_fn(num_tokens: int, cfg: ModelConfig) -> Callable[[], torch.Tensor]:
    """vLLM 0.19.x MoE: fused_topk + fused_experts.

    Composed to match what the legacy `fused_moe()` wrapper did internally, so
    the timing covers (gating softmax + top-k selection + per-expert grouped
    GEMMs + scatter/reduce). Same Triton kernel as legacy 0.10.x.

    First call triggers Triton autotune; callers must warm up >= 20 iters.
    """
    device = cfg.device
    dtype = cfg.dtype

    hidden_states = torch.randn(num_tokens, cfg.hidden, device=device, dtype=dtype)
    w1 = torch.randn(cfg.num_experts, 2 * cfg.intermediate, cfg.hidden, device=device, dtype=dtype)
    w2 = torch.randn(cfg.num_experts, cfg.hidden, cfg.intermediate, device=device, dtype=dtype)
    gating_output = torch.randn(num_tokens, cfg.num_experts, device=device, dtype=dtype)

    top_k = cfg.top_k
    global_num_experts = cfg.num_experts

    def run() -> torch.Tensor:
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, gating_output, top_k, renormalize=True,
        )
        return fused_experts(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            inplace=False,
            global_num_experts=global_num_experts,
        )

    return run
