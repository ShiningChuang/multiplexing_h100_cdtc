# Source this file to load the project environment.
# Usage:  source script/env.sh

# Add CUDA toolchain to PATH (nvcc, ncu, etc).
if [ -d /usr/local/cuda/bin ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
if [ -d /usr/local/cuda/lib64 ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

# Activate the shared venv (created under llm_multiplexing_stage_2/.venv).
PROJECT_VENV="${HOME}/llm_multiplexing_stage_2/.venv"
if [ -f "${PROJECT_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${PROJECT_VENV}/bin/activate"
fi

# Make `src/` importable as `from src...` from the project root.
export PYTHONPATH="${HOME}/multiplexing_h100_cdtc:${PYTHONPATH:-}"

# Deterministic Triton autotune cache directory (so warmup costs amortize).
export TRITON_CACHE_DIR="${HOME}/.triton/cache_multiplexing_h100_cdtc"
mkdir -p "$TRITON_CACHE_DIR"
