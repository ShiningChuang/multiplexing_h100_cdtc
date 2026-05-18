"""Step-1 isolated FP16 baseline benchmark on H100 SXM5.

Runs four sweeps and writes one CSV per sweep into result/step1_baseline/.

    1. FA3 attention prefill, Mixtral GQA shape
    2. FA3 attention decode,  Mixtral GQA shape
    3. vLLM Triton fused_moe, Mixtral (8 experts, top-2)
    4. vLLM Triton fused_moe, Qwen3-30B-A3B (128 experts, top-8)

Each row carries timing percentiles, FLOPs, bytes, achieved TFLOPS/BW, intensity,
roofline bound classification, and utilization vs the FP16 TC roofline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from .configs import (
    DECODE_KV_LIST,
    H100_FP16_TENSORCORE_TFLOPS,
    H100_HBM_BW_GBS,
    MIXTRAL_FP16,
    MOE_TOKEN_LIST_MIX,
    MOE_TOKEN_LIST_QWEN,
    PREFILL_SEQ_LIST,
    QWEN3_FP16,
    REPEAT,
    WARMUP,
    ModelConfig,
)
from .kernels import (
    make_fa3_decode_fn,
    make_fa3_prefill_fn,
    make_vllm_fused_moe_fn,
)
from .utils import flush_csv, gpu_timer, roofline


# Bytes per element for fp16.
FP16 = 2

RESULT_DIR = Path(__file__).resolve().parents[1] / "result" / "step1_baseline"


# --- FLOP / byte models -------------------------------------------------------

def attn_prefill_flops_bytes(seq_len: int, cfg: ModelConfig):
    """Causal self-attention prefill, single sequence.

    FLOPs (causal halves the count):
        QK^T :  2 * H_q * S^2 * head_dim
        AV   :  2 * H_q * S^2 * head_dim
        -> total 4 * H_q * S^2 * head_dim,  causal -> /2
    Bytes (Q, K, V, O at fp16, GQA-aware):
        Q + O use num_q_heads, K + V use num_kv_heads
    """
    H_q = cfg.num_q_heads
    H_kv = cfg.num_kv_heads
    d = cfg.head_dim
    S = seq_len

    flops = (2 * S * S * H_q * d + 2 * S * S * H_q * d) / 2  # causal

    q_bytes = S * H_q  * d * FP16
    o_bytes = S * H_q  * d * FP16
    k_bytes = S * H_kv * d * FP16
    v_bytes = S * H_kv * d * FP16
    bytes_ = q_bytes + k_bytes + v_bytes + o_bytes
    return flops, bytes_


def attn_decode_flops_bytes(kv_len: int, cfg: ModelConfig):
    """Single-token decode against a kv_len cache."""
    H_q = cfg.num_q_heads
    H_kv = cfg.num_kv_heads
    d = cfg.head_dim

    flops = 2 * H_q * kv_len * d + 2 * H_q * kv_len * d  # QK^T + AV

    q_bytes = 1 * H_q  * d * FP16
    o_bytes = 1 * H_q  * d * FP16
    k_bytes = kv_len * H_kv * d * FP16
    v_bytes = kv_len * H_kv * d * FP16
    bytes_ = q_bytes + k_bytes + v_bytes + o_bytes
    return flops, bytes_


def moe_flops_bytes(num_tokens: int, cfg: ModelConfig):
    """3 GEMMs per activated expert (gate, up, down)."""
    H = cfg.hidden
    I = cfg.intermediate
    K = cfg.top_k
    T = num_tokens

    flops = T * K * (
        2 * H * I +   # gate_proj
        2 * H * I +   # up_proj
        2 * I * H     # down_proj
    )

    act_in_bytes  = T * H * FP16
    act_out_bytes = T * H * FP16
    # Worst-case weight read: every activated expert touched exactly once.
    weight_bytes_per_expert = (2 * I * H + I * H) * FP16  # w1 (2I*H) + w2 (H*I)
    weight_bytes = K * weight_bytes_per_expert
    bytes_ = act_in_bytes + act_out_bytes + weight_bytes
    return flops, bytes_


# --- Sweep drivers ------------------------------------------------------------

CSV_COLS_ATTN = [
    "seq_len_or_kvlen", "model", "phase",
    "latency_median_ms", "latency_p10_ms", "latency_p90_ms",
    "latency_min_ms", "latency_max_ms", "stability_pct",
    "flops", "bytes",
    "tflops_achieved", "bw_gbs_achieved", "intensity",
    "bound", "util_pct_vs_tc_roofline",
]

CSV_COLS_MOE = [
    "num_tokens", "model", "num_experts", "top_k",
    "latency_median_ms", "latency_p10_ms", "latency_p90_ms",
    "latency_min_ms", "latency_max_ms", "stability_pct",
    "flops", "bytes",
    "tflops_achieved", "bw_gbs_achieved", "intensity",
    "bound", "util_pct_vs_tc_roofline",
]


def _row_from_timing(t: Dict, flops: float, bytes_: float) -> Dict:
    rf = roofline(
        flops=flops,
        bytes_accessed=bytes_,
        latency_ms=t["median_ms"],
        peak_tflops=H100_FP16_TENSORCORE_TFLOPS,
        peak_bw_gbs=H100_HBM_BW_GBS,
    )
    stability = 100.0 * (t["p90_ms"] - t["p10_ms"]) / t["median_ms"]
    return {
        "latency_median_ms": t["median_ms"],
        "latency_p10_ms":    t["p10_ms"],
        "latency_p90_ms":    t["p90_ms"],
        "latency_min_ms":    t["min_ms"],
        "latency_max_ms":    t["max_ms"],
        "stability_pct":     stability,
        "flops":             flops,
        "bytes":             bytes_,
        "tflops_achieved":   rf["tflops"],
        "bw_gbs_achieved":   rf["bw_gbs"],
        "intensity":         rf["intensity"],
        "bound":             rf["bound"],
        "util_pct_vs_tc_roofline": rf["util_pct"],
    }


def bench_attn_prefill(cfg: ModelConfig) -> List[Dict]:
    rows: List[Dict] = []
    for seq_len in PREFILL_SEQ_LIST:
        fn = make_fa3_prefill_fn(seq_len, cfg)
        t = gpu_timer(fn, warmup=WARMUP, repeat=REPEAT)
        flops, bytes_ = attn_prefill_flops_bytes(seq_len, cfg)
        row = {"seq_len_or_kvlen": seq_len, "model": cfg.name, "phase": "prefill"}
        row.update(_row_from_timing(t, flops, bytes_))
        rows.append(row)
        print(f"[prefill {cfg.name} S={seq_len:>5}] "
              f"med={t['median_ms']:.3f}ms  "
              f"{row['tflops_achieved']:7.1f} TFLOPS  "
              f"I={row['intensity']:7.1f} FLOP/B  bound={row['bound']}  "
              f"util={row['util_pct_vs_tc_roofline']:5.1f}%")
        del fn
        torch.cuda.empty_cache()
    return rows


def bench_attn_decode(cfg: ModelConfig) -> List[Dict]:
    rows: List[Dict] = []
    for kv_len in DECODE_KV_LIST:
        fn = make_fa3_decode_fn(kv_len, cfg)
        t = gpu_timer(fn, warmup=WARMUP, repeat=REPEAT)
        flops, bytes_ = attn_decode_flops_bytes(kv_len, cfg)
        row = {"seq_len_or_kvlen": kv_len, "model": cfg.name, "phase": "decode"}
        row.update(_row_from_timing(t, flops, bytes_))
        rows.append(row)
        print(f"[decode  {cfg.name} K={kv_len:>5}] "
              f"med={t['median_ms']:.3f}ms  "
              f"{row['bw_gbs_achieved']:7.1f} GB/s  "
              f"I={row['intensity']:7.2f}  bound={row['bound']}  "
              f"util={row['util_pct_vs_tc_roofline']:5.1f}%")
        del fn
        torch.cuda.empty_cache()
    return rows


def bench_moe(cfg: ModelConfig, token_grid: List[int]) -> List[Dict]:
    rows: List[Dict] = []
    for nt in token_grid:
        fn = make_vllm_fused_moe_fn(nt, cfg)
        # vLLM fused_moe has heavy Triton autotune; bump warmup to be safe.
        t = gpu_timer(fn, warmup=max(WARMUP, 30), repeat=REPEAT)
        flops, bytes_ = moe_flops_bytes(nt, cfg)
        row = {
            "num_tokens": nt,
            "model": cfg.name,
            "num_experts": cfg.num_experts,
            "top_k": cfg.top_k,
        }
        row.update(_row_from_timing(t, flops, bytes_))
        rows.append(row)
        print(f"[moe     {cfg.name} T={nt:>5}] "
              f"med={t['median_ms']:.3f}ms  "
              f"{row['tflops_achieved']:7.1f} TFLOPS  "
              f"I={row['intensity']:7.1f}  bound={row['bound']}  "
              f"util={row['util_pct_vs_tc_roofline']:5.1f}%")
        del fn
        torch.cuda.empty_cache()
    return rows


# --- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode",  action="store_true")
    parser.add_argument("--skip-moe-mix", action="store_true")
    parser.add_argument("--skip-moe-qwen", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Device: {torch.cuda.get_device_name(0)}  "
          f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_prefill:
        print("\n=== Group 1: FA3 prefill (Mixtral GQA) ===")
        rows = bench_attn_prefill(MIXTRAL_FP16)
        flush_csv(rows, RESULT_DIR / "attention_prefill_mixtral.csv", CSV_COLS_ATTN)

    if not args.skip_decode:
        print("\n=== Group 2: FA3 decode (Mixtral GQA) ===")
        rows = bench_attn_decode(MIXTRAL_FP16)
        flush_csv(rows, RESULT_DIR / "attention_decode_mixtral.csv", CSV_COLS_ATTN)

    if not args.skip_moe_mix:
        print("\n=== Group 3: vLLM fused_moe (Mixtral 8x7B) ===")
        rows = bench_moe(MIXTRAL_FP16, MOE_TOKEN_LIST_MIX)
        flush_csv(rows, RESULT_DIR / "expert_mixtral.csv", CSV_COLS_MOE)

    if not args.skip_moe_qwen:
        print("\n=== Group 4: vLLM fused_moe (Qwen3-30B-A3B) ===")
        rows = bench_moe(QWEN3_FP16, MOE_TOKEN_LIST_QWEN)
        flush_csv(rows, RESULT_DIR / "expert_qwen3.csv", CSV_COLS_MOE)

    print("\nDone. CSVs written to", RESULT_DIR)


if __name__ == "__main__":
    main()
