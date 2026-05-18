#!/usr/bin/env bash
# Verify SASS instruction mix in the two synthetic micro-bench kernels.
# Step 2.1 / multiplexing_h100_cdtc.

set -u
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJ_ROOT}/script/env.sh"
BUILD_DIR="${BUILD_DIR:-$PROJ_ROOT/build}"

if [ ! -f "$BUILD_DIR/kernel_pure_cd.cubin" ] || [ ! -f "$BUILD_DIR/kernel_pure_tc.cubin" ]; then
    echo "missing cubins in $BUILD_DIR. run 'make -C src/microbench' first." >&2
    exit 1
fi

CD_SASS=$(mktemp)
TC_SASS=$(mktemp)
trap 'rm -f "$CD_SASS" "$TC_SASS"' EXIT

cuobjdump --dump-sass "$BUILD_DIR/kernel_pure_cd.cubin" > "$CD_SASS"
cuobjdump --dump-sass "$BUILD_DIR/kernel_pure_tc.cubin" > "$TC_SASS"

count() { grep -cE "$1" "$2"; }

echo "=== kernel_pure_cd SASS check ==="
echo "FFMA count                      : $(count '\bFFMA\b' $CD_SASS)"
echo "FMUL / FADD count               : $(count '\bF(MUL|ADD)\b' $CD_SASS)"
echo "HMMA / HGMMA / GMMA count (==0) : $(count '\b(HMMA|HGMMA|GMMA)\b' $CD_SASS)"
echo "Total SASS lines                : $(wc -l < $CD_SASS)"

echo
echo "=== kernel_pure_tc SASS check ==="
echo "HMMA / HGMMA / GMMA count       : $(count '\b(HMMA|HGMMA|GMMA)\b' $TC_SASS)"
echo "FFMA count (main loop should be 0 or very small <10):"
echo "  $(count '\bFFMA\b' $TC_SASS)"
echo "Total SASS lines                : $(wc -l < $TC_SASS)"

echo
echo "=== verdict ==="
cd_hmma=$(count '\b(HMMA|HGMMA|GMMA)\b' "$CD_SASS")
tc_hmma=$(count '\b(HMMA|HGMMA|GMMA)\b' "$TC_SASS")
tc_ffma=$(count '\bFFMA\b' "$TC_SASS")

ok=1
if [ "$cd_hmma" -ne 0 ]; then echo "FAIL: pure_cd has $cd_hmma HMMA/GMMA"; ok=0; fi
if [ "$tc_hmma" -lt 1 ]; then echo "FAIL: pure_tc has $tc_hmma HMMA/GMMA, expected >0"; ok=0; fi
if [ "$tc_ffma" -gt 32 ]; then echo "WARN: pure_tc FFMA=$tc_ffma > 32 — may indicate accidental CD work in TC path"; fi
if [ "$ok" -eq 1 ]; then echo "PASS: CD kernel is CD-only, TC kernel uses Tensor Cores"; exit 0; else exit 1; fi
