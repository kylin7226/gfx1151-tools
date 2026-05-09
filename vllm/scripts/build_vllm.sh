#!/bin/bash
# vLLM build wrapper — captures full output on failure so CI logs show
# the actual cmake/python error instead of a truncated tail.
set -euo pipefail

LOG=/tmp/vllm_build.log

echo "=== Build Environment ==="
echo "HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit)"
echo "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-<unset>}"
echo "HIP_ARCHITECTURES=${HIP_ARCHITECTURES:-<unset>}"
echo "GPU_TARGETS=${GPU_TARGETS:-<unset>}"
echo "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-<unset>}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-<unset>}"
echo "MAX_JOBS=${MAX_JOBS:-<unset>}"
echo "CC=${CC:-<unset>}"
echo "CXX=${CXX:-<unset>}"
cmake --version 2>/dev/null | head -1
python --version 2>/dev/null
echo "ROCM_PATH=${ROCM_PATH:-<unset>}"
echo "Disk:" && df -h /tmp | tail -1

# Quick smoke test: verify cmake can detect HIP compiler
echo "=== HIP compiler check ==="
cmake --system-information 2>/dev/null | grep -i "HIP_COMPILER\|HIP_COMPILER_ID" | head -5 || true
if [ -x "${CXX:-}" ]; then
    echo "CXX compiler: $(${CXX} --version 2>/dev/null | head -1)"
else
    echo "WARNING: CXX compiler not found at ${CXX:-<unset>}"
fi

# Clean any stale build artifacts
rm -rf /opt/vllm/build /opt/vllm/.deps

echo "=== Starting vLLM build ==="

SKBUILD_BUILD_VERBOSE=true \
uv pip install --no-build-isolation --no-deps . >"$LOG" 2>&1 || {
    rc=$?
    lines=$(wc -l < "$LOG")

    echo ""
    echo "========================================"
    echo "  vLLM BUILD FAILED (exit $rc, $lines lines)"
    echo "========================================"
    echo ""

    # --- ERROR SUMMARY: grep for the actual failure ---
    echo "--- error summary ---"
    grep -inE "CMake Error|FATAL ERROR|fatal error|: error:|Error :|cmake: error" "$LOG" | tail -20 || true
    echo "--- end error summary ---"
    echo ""

    # --- LAST 200 LINES: guaranteed to survive truncation ---
    echo "--- last 200 lines ---"
    tail -200 "$LOG"
    echo "--- end last 200 lines ---"
    echo ""
    echo "========================================"
    echo "  END ERROR REPORT"
    echo "========================================"

    exit $rc
}

rm -f "$LOG"
echo "=== vLLM build successful ==="
