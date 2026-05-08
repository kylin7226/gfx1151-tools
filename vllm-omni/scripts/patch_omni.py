"""
gfx1151 (Strix Halo / RDNA 3.5) patches for vllm-omni.

vllm-omni is built on top of the already-patched rocm_gfx1151_vllm builder
image, so all vLLM-level patches (platform detection, AITER guards, VRAM,
Triton SDPA caps, etc.) are inherited automatically.

The diffusion attention subsystem (vllm_omni/diffusion/attention/) is
vllm-omni's own implementation and does not expose a TRITON_ATTN backend
(only FLASH_ATTN / TORCH_SDPA / SAGE_ATTN). On gfx1151 diffusion falls
back to TORCH_SDPA — no runtime patch is needed.

This script currently only performs a runtime verification check (Patch 2).
"""
import site
from pathlib import Path


def patch_omni():
    print("Checking vllm-omni for gfx1151...")

    # ----------------------------------------------------------------
    # Patch 2: Ensure onnxruntime conflict is resolved
    #
    # vllm-omni setup.py should have auto-uninstalled vanilla onnxruntime
    # and installed onnxruntime-rocm. Verify no stale imports remain.
    # ----------------------------------------------------------------
    for sp in site.getsitepackages():
        ort_dir = Path(sp) / "onnxruntime"
        if ort_dir.exists():
            # Check if it's the ROCm variant or vanilla
            try:
                import onnxruntime
                if hasattr(onnxruntime, '__file__') and 'rocm' not in str(onnxruntime.__file__).lower():
                    print(f" -> Warning: vanilla onnxruntime found at {onnxruntime.__file__}")
            except ImportError:
                pass

    print("vllm-omni gfx1151 check complete.")


if __name__ == "__main__":
    patch_omni()
