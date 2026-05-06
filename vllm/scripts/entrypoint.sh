#!/usr/bin/env bash
# Entrypoint wrapper: validate profile cache directory and guard LD_PRELOAD
# before starting vllm.
set -euo pipefail

# ── tcmalloc guard ─────────────────────────────────────────────────────
# LD_PRELOAD is not set in the Dockerfile ENV (container won't start if
# the library is missing). Check at runtime and only enable if present.
TCMALLOC="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
if [[ -f "$TCMALLOC" ]]; then
    export LD_PRELOAD="$TCMALLOC"
else
    echo "[entrypoint] tcmalloc not found at $TCMALLOC — skipping LD_PRELOAD."
fi

# ── profile cache validation ───────────────────────────────────────────

if [[ "${VLLM_SKIP_MEMORY_PROFILING:-0}" == "1" ]]; then
    CACHE_DIR="${VLLM_PROFILE_CACHE_DIR:-/root/.cache/vllm-profile}"
    if [[ ! -d "$CACHE_DIR" ]]; then
        echo "[entrypoint] WARNING: profile cache dir $CACHE_DIR does not exist."
        echo "  Creating it — profile cache will work from next restart."
        mkdir -p "$CACHE_DIR"
    elif [[ ! -w "$CACHE_DIR" ]]; then
        echo "[entrypoint] WARNING: profile cache dir $CACHE_DIR is not writable."
        echo "  Falling back to full memory profiling this boot."
        export VLLM_SKIP_MEMORY_PROFILING=0
    else
        echo "[entrypoint] Profile cache dir $CACHE_DIR OK."
    fi
fi

# Execute the original CMD.
exec "$@"
