# vLLM-Omni layer on top of the rocm_gfx1151_vllm builder image.
#
# Installs vllm-omni v0.20.0 (rebased on upstream vLLM v0.20.0) and
# applies gfx1151-specific patches to the vllm_omni package so that
# attention backends and ROCm platform detection work correctly on AMD
# Strix Halo (RDNA 3.5 / gfx1151).
#
# Build order:
#   1. podman-compose build vllm       (builds the base builder image)
#   2. podman-compose build vllm-omni  (layers vllm-omni on top)

ARG BASE_IMAGE=rocm_gfx1151_vllm:v0.20.1
FROM ${BASE_IMAGE}

# 1. vllm-omni runtime deps (not in base builder image).
# espeak-ng: TTS phonemization; ffmpeg: audio/video decoding/transcoding.
# jq: JSON utility (diagnostics). Matches upstream Dockerfile.rocm.
RUN apt-get update && apt-get install -y --no-install-recommends \
      espeak-ng ffmpeg jq \
    && rm -rf /var/lib/apt/lists/*

# 2. Install vllm-omni v0.20.0 from source.
# setup.py auto-detects ROCm via torch.version.hip and installs
# onnxruntime-rocm (auto-uninstalling vanilla onnxruntime first).
ARG VLLM_OMNI_COMMIT=v0.20.0
RUN git clone --depth 1 --branch "${VLLM_OMNI_COMMIT}" \
      https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni

WORKDIR /opt/vllm-omni
RUN VLLM_OMNI_TARGET_DEVICE=rocm uv pip install --no-build-isolation -e . && \
    rm -rf /root/.cache/uv /root/.cache/pip

# 2. Apply gfx1151 patches to the installed vllm_omni package.
# These patch RocmOmniPlatform to enable TRITON_ATTN for gfx1151
# diffusion backends (upstream only gates aiter to gfx94x/gfx95x).
COPY scripts/patch_omni.py /opt/vllm-omni/patch_omni.py
RUN python /opt/vllm-omni/patch_omni.py

WORKDIR /opt

# 3. Runtime env overrides — inherited from the base builder image
# (FLASH_ATTENTION_TRITON_AMD_ENABLE, etc.) plus vllm-omni-specific additions.
ENV VLLM_OMNI_TARGET_DEVICE=rocm

CMD ["/bin/bash"]
