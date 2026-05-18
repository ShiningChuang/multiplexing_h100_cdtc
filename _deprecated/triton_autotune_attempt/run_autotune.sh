#!/usr/bin/env bash
# Step 1.5 Task 2 — autotune vLLM Triton fused_moe for our two MoE shapes.
# Saves JSON into project ./configs/ and copies into vllm's configs/ dir so
# fused_moe picks them up automatically.

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJ}/script/env.sh"

TUNER="${PROJ}/tools/benchmark_moe.py"
LOCAL_OUT="${PROJ}/configs"
VLLM_CFG_DIR=$(python3 -c "import vllm.model_executor.layers.fused_moe as m, os; print(os.path.join(os.path.dirname(m.__file__),'configs'))")
echo "vllm configs dir: ${VLLM_CFG_DIR}"
mkdir -p "${LOCAL_OUT}"

# --- Mixtral 8x7B: E=8, top-2, intermediate=14336, hidden=4096, fp16 ----------
# Tight grid (5 sizes): launch-bound / ridge transition / compute-bound.
# vLLM picks the nearest tuned config at runtime for non-tuned batch sizes.
echo
echo "=== tuning Mixtral 8x7B (E=8, N=14336) ==="
time python3 "${TUNER}" \
    --model "${PROJ}/tools/fake_models/mixtral_8x7b" \
    --tp-size 1 \
    --save-dir "${LOCAL_OUT}" \
    --tune \
    --batch-size 16 256 1024 4096 8192 \
    2>&1 | tee "${PROJ}/result/step1_baseline/autotune_mixtral.log" | tail -50

# --- Qwen3-30B-A3B: E=128, top-8, moe_intermediate=768, hidden=2048, fp16 -----
# Tight grid (4 sizes): Qwen3 has smaller intermediate, narrower roofline transition.
echo
echo "=== tuning Qwen3-30B-A3B (E=128, N=768) ==="
time python3 "${TUNER}" \
    --model "${PROJ}/tools/fake_models/qwen3_30b_a3b" \
    --tp-size 1 \
    --save-dir "${LOCAL_OUT}" \
    --tune \
    --batch-size 16 256 1024 4096 \
    2>&1 | tee "${PROJ}/result/step1_baseline/autotune_qwen3.log" | tail -50

# --- Distribute JSONs into vllm so fused_moe loads them ----------------------
echo
echo "=== installing tuned configs into vllm ==="
for f in "${LOCAL_OUT}"/E=8*device_name=NVIDIA_H100_80GB_HBM3*.json \
         "${LOCAL_OUT}"/E=128*device_name=NVIDIA_H100_80GB_HBM3*.json; do
    [ -f "$f" ] || continue
    dest="${VLLM_CFG_DIR}/$(basename "$f")"
    cp -v "$f" "$dest"
done

echo
echo "Done. Tuned JSONs archived in ${LOCAL_OUT} and installed into ${VLLM_CFG_DIR}."
