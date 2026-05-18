# Step 1 — Prompt archive

This file preserves the exact prompt that drove step 1 of the multiplexing study so
that future maintainers can see the original framing, scope boundaries, and sanity
gates. Run artifacts produced from this prompt live in `result/step1_baseline/`.

---

我们正在 H100 SXM5 上开展一个新的 GPU multiplexing 研究项目，研究问题是：
**"attention 走 CUDA Core + expert 走 Tensor Core 同 SM 并发是否能提升 throughput"**

这是整个项目的 **第 1 步：基础设施 sanity check 与 FP16 baseline**。
本步骤范围严格限定为"环境验证 + FP16 isolated baseline 复现"，
**不要**做任何 CUDA Core attention 实现或并发实验——那些是后续步骤。

## 路径选型说明（先看，避免选错 kernel）

本项目主路径走 **FP16**，不是 FP8。理由：
- H100 FP16 Tensor Core : CUDA Core ≈ 989 / 134 ≈ 7.4×
- H100 FP8 Tensor Core : (FP16) CUDA Core ≈ 1979 / 134 ≈ 14.8×
- Plasticine/Tacker 在 V100/2080Ti 上的 8× 量级下能拿到 45% overlap 和
  15-40% throughput 提升；FP16 路径与之对齐，FP8 差距太大对 CD 路径不友好。

因此 kernel 选型：
- **Attention**：FA3 FP16，**用 upstream `flash-attn`**（不是 vLLM fork，
  因为 vLLM fork 主要 patch fp8 KV cache，fp16 路径 upstream 更干净）
- **Expert**：vLLM Triton `fused_moe`（FP16），这是 vLLM v0.19.x 对 fp16/bf16
  MoE 的默认选型。FlashInfer cutlass_fused_moe 主要服务 fp8，本步骤不用。

## 项目目录约定

在 `~/multiplexing_h100_cdtc/` 下建立项目：

```
multiplexing_h100_cdtc/
├── README.md
├── script/
│   ├── verify_env.sh
│   └── env.sh
├── src/
│   ├── configs.py
│   ├── utils.py
│   ├── kernels.py
│   └── bench_baseline_h100_fp16.py
├── result/
│   └── step1_baseline/
│       ├── figures/
│       └── *.csv
└── plan/
    └── step1_baseline.md   # 把本 prompt 存进去做存档
```

## 任务 1：环境验证脚本 `script/verify_env.sh`

输出 12 项检查，每项 PASS/FAIL/WARN。失败时打印**具体修复建议**
（例如 "pip install flash-attn --no-build-isolation"）。
脚本应是幂等的，可重复运行。

检查项：
1. GPU 是 H100 SXM5 80GB HBM3（用 `nvidia-smi --query-gpu=name,memory.total --format=csv`）
2. CUDA Toolkit ≥ 12.4（`nvcc --version`）
3. NVIDIA driver ≥ 555（`nvidia-smi`）
4. PyTorch ≥ 2.5，CUDA build matching
5. vLLM 版本是 v0.19.x（`python3 -c "import vllm; print(vllm.__version__)"`）
6. **upstream FA3** 能 import：`from flash_attn import flash_attn_varlen_func,
   flash_attn_with_kvcache`，并且 `flash_attn.__version__ >= "2.7"`
   （注意：FA3 在 upstream `flash-attn` 包内，不需要 vllm_flash_attn）
7. **vLLM Triton fused_moe** 能 import：`from vllm.model_executor.layers.fused_moe
   import fused_moe`
8. Triton ≥ 3.2
9. `nsys` 和 `ncu` 可执行（仅检查 `which`，不要试运行）
10. CUDA Green Context API 可用：`from cuda.bindings import driver as cu` 且
    `cu.cuDevSmResourceSplitByCount` 存在
11. H100 SM 数 = 132：
    `torch.cuda.get_device_properties(0).multi_processor_count == 132`
12. 简单 Triton sanity：一个最小 Triton kernel（element-wise add）能跑通

## 任务 2：`src/configs.py`

定义 H100 硬件常量（**严格用下面这些数字**，来自 NVIDIA 官方 datasheet）：

```python
# H100 SXM5 80GB HBM3 official specs
H100_FP16_TENSORCORE_TFLOPS = 989.4   # dense, no sparsity
H100_FP16_CUDACORE_TFLOPS   = 133.8   # peak non-Tensor FP16
H100_FP8_TENSORCORE_TFLOPS  = 1978.9  # dense, no sparsity (for future ref)
H100_HBM_BW_GBS             = 3350.0
H100_SM_COUNT               = 132
H100_L2_CACHE_MB            = 50
H100_SHARED_MEM_PER_SM_KB   = 228

# Roofline ridge points (FLOP / byte)
H100_RIDGE_FP16_TC = H100_FP16_TENSORCORE_TFLOPS * 1e3 / H100_HBM_BW_GBS  # ≈ 295
H100_RIDGE_FP16_CD = H100_FP16_CUDACORE_TFLOPS   * 1e3 / H100_HBM_BW_GBS  # ≈ 40
```

定义 `ModelConfig` dataclass 和两个 preset（**严格按下面的数字**）：

```python
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
    name="mixtral-8x7b", num_q_heads=32, num_kv_heads=8, head_dim=128,
    hidden=4096, intermediate=14336, num_experts=8, top_k=2,
    dtype=torch.float16,
)

QWEN3_FP16 = ModelConfig(
    name="qwen3-30b-a3b", num_q_heads=32, num_kv_heads=4, head_dim=128,
    hidden=2048, intermediate=768, num_experts=128, top_k=8,
    dtype=torch.float16,
)
```

也定义实验网格常量：

```python
PREFILL_SEQ_LIST     = [128, 256, 512, 1024, 2048, 4096, 8192]
DECODE_KV_LIST       = [512, 1024, 2048, 4096, 8192, 16384]
MOE_TOKEN_LIST_MIX   = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
MOE_TOKEN_LIST_QWEN  = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
WARMUP = 20
REPEAT = 200
```

## 任务 3：`src/utils.py`

实现：

### `gpu_timer(fn, warmup=20, repeat=200) -> Dict`
用 CUDA event 计时。warmup 不计入。返回：
- `min_ms`, `max_ms`, `median_ms`, `p10_ms`, `p90_ms`, `mean_ms`
- 同时返回 `times_ms` raw list 供 CSV 存档

### `roofline(flops, bytes_accessed, latency_ms, peak_tflops, peak_bw_gbs) -> Dict`
返回：
- `tflops`: achieved
- `bw_gbs`: achieved
- `intensity`: FLOP/byte
- `bound`: "memory" 或 "compute"（基于 intensity vs ridge point）
- `util_pct`: 相对 roofline 模型的利用率

注意 `peak_tflops` 可以传 `H100_FP16_TENSORCORE_TFLOPS`（用于 TC kernel）
也可以传 `H100_FP16_CUDACORE_TFLOPS`（用于后续 CD kernel）。
本步骤所有 baseline 都用 TC peak。

### `flush_csv(rows: List[Dict], path: Path)`
写入 CSV，column 顺序固定。

## 任务 4：`src/kernels.py`

提供两个 factory function（返回可调用的 closure，输入数据预分配在 device 上，
避免计时时把 allocation 也算进去）：

### `make_fa3_prefill_fn(seq_len, cfg) -> Callable`
内部用 `flash_attn_varlen_func`，causal=True，dtype=cfg.dtype。
预分配 Q/K/V tensors，shape:
- Q: `[seq_len, cfg.num_q_heads, cfg.head_dim]`
- K: `[seq_len, cfg.num_kv_heads, cfg.head_dim]`
- V: `[seq_len, cfg.num_kv_heads, cfg.head_dim]`
需要 `cu_seqlens_q` 和 `cu_seqlens_k`：`torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')`，
`max_seqlen_q = max_seqlen_k = seq_len`。

### `make_fa3_decode_fn(kv_len, cfg) -> Callable`
内部用 `flash_attn_with_kvcache`，单 token decode（query length=1，
attend 到 kv_len 个 cached tokens）。
预分配：
- Q: `[1, 1, cfg.num_q_heads, cfg.head_dim]`  (batch=1, seqlen_q=1)
- K_cache, V_cache: `[1, kv_len, cfg.num_kv_heads, cfg.head_dim]`
- cache_seqlens: `torch.tensor([kv_len], dtype=torch.int32, device='cuda')`

### `make_vllm_fused_moe_fn(num_tokens, cfg) -> Callable`
内部用 `vllm.model_executor.layers.fused_moe.fused_moe`。
**注意 vLLM fused_moe 的 API 签名**（v0.19.x）：

```python
fused_moe(
    hidden_states,  # [num_tokens, hidden]
    w1,             # [num_experts, 2*intermediate, hidden]  (gate + up concatenated)
    w2,             # [num_experts, hidden, intermediate]    (down)
    gating_output,  # [num_tokens, num_experts]              (routing logits)
    topk=cfg.top_k,
    renormalize=True,
    inplace=False,
)
```

预分配：
- `hidden_states`: `[num_tokens, hidden]`，randn fp16
- `w1`: `[num_experts, 2*intermediate, hidden]`，randn fp16
- `w2`: `[num_experts, hidden, intermediate]`，randn fp16
- `gating_output`: `[num_tokens, num_experts]`，randn fp16（任意值，
  内部会做 softmax + top-k 选择）

**重要**：第一次调用会触发 Triton autotune，warmup 必须足够长（≥20 次）
让 autotune 完成。

## 任务 5：`src/bench_baseline_h100_fp16.py`

跑 4 组 isolated baseline：

### 组 1：Attention Prefill (Mixtral)
扫 `seq_len ∈ PREFILL_SEQ_LIST`。
每行 CSV：
```
seq_len, latency_median_ms, latency_p10_ms, latency_p90_ms,
flops, bytes, tflops_achieved, bw_gbs_achieved, intensity,
bound, util_pct_vs_tc_roofline
```
FLOPs 计算（causal attention）：
```
flops = 2 * seq_len^2 * num_q_heads * head_dim   # QK^T
      + 2 * seq_len^2 * num_q_heads * head_dim   # AV
      / 2                                         # causal mask halves the work
```
bytes：Q, K, V, O 四个 fp16 张量 = 2 * (Q + K + V + O) bytes
（注意 GQA：K 和 V 用 num_kv_heads, Q 和 O 用 num_q_heads）

### 组 2：Attention Decode (Mixtral)
扫 `kv_len ∈ DECODE_KV_LIST`。
FLOPs：`2 * H_q * kv_len * head_dim` (QK^T) + `2 * H_q * kv_len * head_dim` (AV)
bytes：Q (1 token) + K_cache + V_cache + O (1 token) 的 fp16 字节数

### 组 3：Expert / MoE (Mixtral)
扫 `num_tokens ∈ MOE_TOKEN_LIST_MIX`。
FLOPs（**注意 MoE 的 FLOP 计算**）：
```
# 每个 token 激活 top_k 个 expert，每个 expert 做 3 个 GEMM：
# gate_proj: hidden -> intermediate
# up_proj:   hidden -> intermediate
# down_proj: intermediate -> hidden
flops = num_tokens * top_k * (
    2 * hidden * intermediate +    # gate_proj
    2 * hidden * intermediate +    # up_proj
    2 * intermediate * hidden      # down_proj
)
```
bytes：
- activations: `num_tokens * hidden * 2` (input) + `num_tokens * hidden * 2` (output)
- weights：top_k experts 的 w1 + w2 = `top_k * (2*hidden*intermediate + intermediate*hidden) * 2` bytes
  （**注意**：实际只有被激活的 expert weights 会被读取，但因为是 randomized routing，
  按 top_k 算 worst case 是合理的）

### 组 4：Expert / MoE (Qwen3)
同上，扫 `num_tokens ∈ MOE_TOKEN_LIST_QWEN`。

## 任务 6：输出文件

```
result/step1_baseline/
├── attention_prefill_mixtral.csv
├── attention_decode_mixtral.csv
├── expert_mixtral.csv
├── expert_qwen3.csv
└── summary.md         # 4 个 CSV 的摘要 + roofline 解读 + sanity check 结果
```

`summary.md` 内容：
1. 硬件软件栈表格
2. 4 个 baseline 的摘要表（每行一个 workload size）
3. Roofline 解读：哪些 operating point 是 memory-bound / compute-bound、
   ridge 拐点在哪
4. **本步骤 sanity check 卡点结果**（见任务 7）
5. 与之前 H100 FlashInfer baseline 数据的对比（如果可获取）

## 任务 7：Sanity check 硬指标（必须通过才进入第 2 步）

在 `summary.md` 里明确记录以下硬指标，**若不通过先停下来调试**，
不要硬继续：

1. **FA3 Prefill seq=8192 时 TFLOPS ≥ 550**
   （= 55% of 989 TFLOPS peak。之前 FlashInfer fp16 在 H100 上是 510，
   FA3 应明显更高。如果低于 550，说明 FA3 没正确启用或没走 SM90 path）

2. **FA3 Prefill 跨过 ridge 点**：seq_len 越大 intensity 应该越高，
   seq=8192 时 intensity 应 ≥ 3000 FLOP/byte（远超 295 ridge），
   bound 应为 "compute"

3. **vLLM Triton fused_moe Mixtral num_tokens=4096 时 TFLOPS ≥ 400**
   （= 40% peak。Triton kernel 通常比 cuBLAS-class kernel 利用率略低，
   400 是合理下限。如果低于 400，可能 autotune 没完成或选了次优 config）

4. **Decode attention 恒为 memory-bound**：所有 kv_len 下 bound 应为
   "memory"，intensity 应远低于 295

5. **测量稳定性**：每个 operating point 的 (p90 - p10) / median < 5%，
   表示测量噪声受控

任何一项不过，**停下来报告**，列出可能原因：
- FA3 fallback 到 FA2（检查 `flash_attn.__version__` 和编译时 SM90 支持）
- vLLM fused_moe 走了未调优的 Triton config
- GPU 上有其他进程（`nvidia-smi` 查）
- Power throttling（`nvidia-smi -q -d POWER` 查 power state）

## 任务 8：README

在 `~/multiplexing_h100_cdtc/README.md` 写：

1. **项目目标**（一段话）：研究 CUDA Core attention + Tensor Core expert 同 SM 并发
2. **路径选型说明**：为什么走 FP16 不走 FP8（粘贴本 prompt 开头的解释）
3. **硬件软件栈表格**：GPU / CUDA / PyTorch / vLLM / flash-attn / Triton 版本
4. **整体实验路线图**：6 步规划（step1 baseline → step2 micro-bench TC+CD overlap →
   step3 CUDA Core attention → step4 stream concurrency → step5 决策图 →
   step6 可选 fused kernel）
5. **本步骤产出**：4 个 CSV + summary.md
6. **下一步触发条件**：所有 sanity check 通过 + 用户审阅 baseline 数据后批准

## 严格的禁止事项

- ❌ **不要**写任何 CUDA Core attention 代码（Triton 或 CUDA）
- ❌ **不要**做并发实验、不要起两个 stream
- ❌ **不要**碰 Green Context API（虽然 sanity check 验证它可用，但不要用）
- ❌ **不要**改 flash-attn 或 vLLM 源码
- ❌ **不要**碰 vLLM 的 LLMEngine、scheduler 等 engine layer
- ❌ **不要**做 Nsight Compute profiling（留到后续步骤）
- ❌ **不要**做 Plasticine/Tacker 风格的 PTB 改造

## 完成后请汇报

1. 12 项 verify_env 检查结果（PASS/FAIL/WARN）
2. 4 个 CSV 的摘要（贴关键行即可，不要全贴）
3. 5 个 sanity check 卡点是否通过
4. 任何 anomaly（即使不阻塞也要报告）
5. 你认为下一步（micro-benchmark 验证 H100 上 TC+CD 物理并行可行性，
   即 Plasticine Bench-A 风格实验）是否可以开始

完成时间预期：在 H100 上 1-2 小时内应该跑完所有 baseline。
如果超过 3 小时还没跑完，停下来报告原因。
