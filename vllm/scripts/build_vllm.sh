#!/bin/bash
# vLLM build wrapper — captures full output on failure so CI logs show
# the actual cmake/python error instead of a truncated tail.
set -euo pipefail

LOG=/tmp/vllm_build.log

export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit)
echo "=== ROCm bitcode: $HIP_DEVICE_LIB_PATH ==="
rm -rf /opt/vllm/build /opt/vllm/.deps

# Step 1: Run cmake configure explicitly to get the real error message.
# vLLM 0.20.1 uses scikit-build-core which calls cmake internally.
# We call it the same way so the error is visible.
echo "=== Running cmake configure ==="
cmake /opt/vllm -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DVLLM_TARGET_DEVICE=rocm \
    -DVLLM_PYTHON_EXECUTABLE=$(which python) \
    -DFETCHCONTENT_BASE_DIR=/opt/vllm/.deps \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
    -DCMAKE_JOB_POOLS:STRING=compile=8 \
    -DROCM_PATH=/opt/rocm \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS=gfx1151 \
    -DHIP_ARCHITECTURES=gfx1151 2>&1 | tee /tmp/cmake_configure.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "=== CMAKE CONFIGURE FAILED ==="
    echo "=== Full cmake output ==="
    cat /tmp/cmake_configure.log
    exit 1
fi

echo "=== cmake configure succeeded ==="
rm -rf /opt/vllm/build /opt/vllm/.deps

# Step 2: Build and install via pip
echo "=== Running uv pip install ==="
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATH=/opt/rocm -DGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151"

SKBUILD_BUILD_VERBOSE=true \
uv pip install --no-build-isolation --no-deps . >"$LOG" 2>&1 || {
    echo "=== vLLM BUILD FAILED ==="
    echo "=== Last 100 lines ==="
    tail -100 "$LOG"
    echo ""
    echo "=== Full log ==="
    cat "$LOG"
    exit 1
}

rm -f "$LOG" /tmp/cmake_configure.log
echo "=== vLLM build successful ==="
