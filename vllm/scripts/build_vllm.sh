#!/bin/bash
# vLLM build wrapper — captures full output on failure so CI logs show
# the actual cmake/python error instead of a truncated tail.
set -euo pipefail

LOG=/tmp/vllm_build.log

export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit)
echo "Building with bitcode: $HIP_DEVICE_LIB_PATH"
rm -rf /opt/vllm/build /opt/vllm/.deps
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATH=/opt/rocm -DGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151"

SKBUILD_BUILD_VERBOSE=true \
uv pip install --no-build-isolation --no-deps . >"$LOG" 2>&1 || {
    echo "=== vLLM BUILD FAILED ==="
    echo "=== Last 50 lines ==="
    tail -50 "$LOG"
    echo ""
    echo "=== Full log ==="
    cat "$LOG"
    exit 1
}

rm -f "$LOG"
echo "=== vLLM build successful ==="
