#!/usr/bin/env bash
# Step-1 environment sanity check.
# Prints 12 PASS/FAIL/WARN lines. Idempotent; safe to re-run.

set -u

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${THIS_DIR}/env.sh"

PASS=0; WARN=0; FAIL=0
status() {
    case "$1" in
        PASS) PASS=$((PASS+1)); printf '[\033[32mPASS\033[0m] %s\n' "$2" ;;
        WARN) WARN=$((WARN+1)); printf '[\033[33mWARN\033[0m] %s\n' "$2" ;;
        FAIL) FAIL=$((FAIL+1)); printf '[\033[31mFAIL\033[0m] %s\n' "$2" ;;
    esac
    if [ -n "${3:-}" ]; then
        printf '       fix: %s\n' "$3"
    fi
}

# --- 1. GPU is H100 SXM5 80GB HBM3 ----------------------------------------------
gpu_line=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
if echo "$gpu_line" | grep -q "H100" && echo "$gpu_line" | grep -qE "(80|81)[0-9]{3} MiB"; then
    status PASS "1. GPU is H100 80GB HBM3 (${gpu_line})"
else
    status FAIL "1. Expected H100 80GB HBM3, got: ${gpu_line:-none}" \
        "run on a real H100 SXM5 node"
fi

# --- 2. CUDA Toolkit >= 12.4 ----------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
    status FAIL "2. nvcc not on PATH" "source script/env.sh to add /usr/local/cuda/bin"
else
    cuda_ver=$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    cuda_major=$(echo "$cuda_ver" | cut -d. -f1)
    cuda_minor=$(echo "$cuda_ver" | cut -d. -f2)
    if [ "$cuda_major" -gt 12 ] || { [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 4 ]; }; then
        status PASS "2. CUDA Toolkit $cuda_ver (>=12.4)"
    else
        status FAIL "2. CUDA Toolkit $cuda_ver < 12.4" \
            "install CUDA 12.4+ from developer.nvidia.com/cuda-downloads"
    fi
fi

# --- 3. NVIDIA driver >= 555 ----------------------------------------------------
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
drv_major=$(echo "$drv" | cut -d. -f1)
if [ -n "$drv_major" ] && [ "$drv_major" -ge 555 ]; then
    status PASS "3. NVIDIA driver $drv (>=555)"
else
    status FAIL "3. NVIDIA driver $drv < 555" \
        "apt install nvidia-driver-555 (or newer) and reboot"
fi

# --- 4. PyTorch >= 2.5, CUDA build matching -------------------------------------
torch_check=$(python3 - <<'PY' 2>&1
import torch, sys
print(f"torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
ok = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 5)
print('VERSION_OK' if ok else 'VERSION_BAD')
print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_BAD')
PY
)
if echo "$torch_check" | grep -q VERSION_OK && echo "$torch_check" | grep -q CUDA_OK; then
    info=$(echo "$torch_check" | head -1)
    status PASS "4. PyTorch ($info)"
else
    status FAIL "4. PyTorch check failed: $torch_check" \
        "pip install --upgrade torch (cu12x wheel)"
fi

# --- 5. vLLM v0.19.x ------------------------------------------------------------
vllm_ver=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
case "$vllm_ver" in
    0.19.*)
        status PASS "5. vLLM $vllm_ver"
        ;;
    "")
        status FAIL "5. vLLM not importable" "pip install vllm==0.19.1"
        ;;
    *)
        status FAIL "5. vLLM $vllm_ver (expected 0.19.x)" \
            "pip install vllm==0.19.1 (note: pins torch==2.10.0)"
        ;;
esac

# --- 6. FA3 (flash_attn_interface) imports --------------------------------------
fa3_check=$(python3 -c "from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache; print('OK')" 2>&1 | tail -1)
if [ "$fa3_check" = "OK" ]; then
    status PASS "6. FA3 (flash_attn_interface) varlen_func + with_kvcache importable"
else
    status FAIL "6. flash_attn_interface (FA3) import failed" \
        "cd ~/flash-attention/hopper && pip install --no-build-isolation ."
fi

# --- 7. vLLM Triton fused_topk + fused_experts (0.19.x API) ---------------------
moe_check=$(python3 -c "from vllm.model_executor.layers.fused_moe import fused_topk, fused_experts; assert callable(fused_topk) and callable(fused_experts); print('OK')" 2>&1 | tail -1)
if [ "$moe_check" = "OK" ]; then
    status PASS "7. vllm.…fused_moe.fused_topk + fused_experts importable & callable"
else
    status FAIL "7. fused_topk/fused_experts import failed: $moe_check" \
        "reinstall vllm 0.19.x; legacy fused_moe() wrapper was removed"
fi

# --- 8. Triton >= 3.2 -----------------------------------------------------------
triton_ver=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null)
if [ -z "$triton_ver" ]; then
    status FAIL "8. triton not importable" "pip install triton"
else
    t_major=$(echo "$triton_ver" | cut -d. -f1)
    t_minor=$(echo "$triton_ver" | cut -d. -f2)
    if [ "$t_major" -gt 3 ] || { [ "$t_major" -eq 3 ] && [ "$t_minor" -ge 2 ]; }; then
        status PASS "8. triton $triton_ver (>=3.2)"
    else
        status FAIL "8. triton $triton_ver < 3.2" "pip install --upgrade triton"
    fi
fi

# --- 9. nsys and ncu executable -------------------------------------------------
nsys_path=$(command -v nsys 2>/dev/null)
ncu_path=$(command -v ncu 2>/dev/null)
if [ -n "$nsys_path" ] && [ -n "$ncu_path" ]; then
    status PASS "9. nsys=$nsys_path  ncu=$ncu_path"
else
    miss=""
    [ -z "$nsys_path" ] && miss+="nsys "
    [ -z "$ncu_path" ]  && miss+="ncu "
    status FAIL "9. missing on PATH: $miss" \
        "source script/env.sh (adds /usr/local/cuda/bin) or install Nsight CLI"
fi

# --- 10. CUDA Green Context API -------------------------------------------------
gctx=$(python3 - <<'PY' 2>&1
from cuda.bindings import driver as cu
assert hasattr(cu, 'cuDevSmResourceSplitByCount'), 'symbol missing'
print('OK')
PY
)
if echo "$gctx" | tail -1 | grep -q OK; then
    status PASS "10. cuda.bindings.driver.cuDevSmResourceSplitByCount available"
else
    status FAIL "10. Green Context API not available: $gctx" \
        "pip install cuda-python (>=12.4)"
fi

# --- 11. H100 SM count = 132 ----------------------------------------------------
sm_count=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)" 2>/dev/null)
if [ "$sm_count" = "132" ]; then
    status PASS "11. SM count = 132"
else
    status FAIL "11. SM count = ${sm_count:-?} (expected 132)" \
        "verify GPU is H100 SXM5 (PCIe variant has 114 SMs)"
fi

# --- 12. Triton element-wise add sanity ----------------------------------------
# triton.jit needs inspect.getsourcelines, which fails on stdin-fed scripts —
# write to a real temp file then exec it.
triton_tmp=$(mktemp --suffix=.py)
cat >"$triton_tmp" <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)

n = 1 << 16
x = torch.randn(n, device='cuda', dtype=torch.float32)
y = torch.randn(n, device='cuda', dtype=torch.float32)
o = torch.empty_like(x)
BLOCK = 1024
grid = (triton.cdiv(n, BLOCK),)
_add[grid](x, y, o, n, BLOCK=BLOCK)
torch.cuda.synchronize()
assert torch.allclose(o, x + y), 'mismatch'
print('OK')
PY
triton_sanity=$(python3 "$triton_tmp" 2>&1)
rm -f "$triton_tmp"
if echo "$triton_sanity" | tail -1 | grep -q OK; then
    status PASS "12. Triton element-wise add kernel runs and matches torch"
else
    status FAIL "12. Triton sanity failed: $triton_sanity" \
        "check triton + CUDA install consistency"
fi

# --- 13. cuda-python importable (Green Context bindings, broader than #10) -----
cudapy_check=$(python3 -c "from cuda.bindings import driver; print('OK')" 2>&1 | tail -1)
if [ "$cudapy_check" = "OK" ]; then
    status PASS "13. cuda-python (cuda.bindings.driver) importable"
else
    status FAIL "13. cuda-python import failed: $cudapy_check" \
        "pip install cuda-python"
fi

# --- 14. FA3 build artifact lives at a sane location ---------------------------
fa3_path=$(python3 -c "import flash_attn_interface; print(flash_attn_interface.__file__)" 2>/dev/null)
if [ -z "$fa3_path" ]; then
    status FAIL "14. flash_attn_interface not importable" \
        "build FA3 (see check 6)"
else
    case "$fa3_path" in
        */site-packages/*|*/flash-attention/hopper/*)
            status PASS "14. FA3 module at $fa3_path"
            ;;
        *)
            status WARN "14. FA3 module at unexpected path: $fa3_path"
            ;;
    esac
fi

# --- 15. FA2 (flash_attn) is expected to be broken after torch 2.10 upgrade ----
fa2_check=$(python3 -c "from flash_attn import flash_attn_func; print('OK')" 2>&1 | tail -1)
if [ "$fa2_check" = "OK" ]; then
    status PASS "15. flash_attn (FA2) importable — coexists with FA3"
else
    status WARN "15. FA2 (flash_attn) ABI-broken after torch 2.10 upgrade. Expected; project uses FA3 exclusively. Leaving FA2 installed avoids tripping vLLM dep checks."
fi

echo
printf 'Summary: %d PASS / %d WARN / %d FAIL\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
