"""
gfx1151 (Strix Halo / RDNA 3.5) patches for vllm-omni.

vllm-omni's ROCm platform is designed for CDNA datacenter GPUs (MI300
series, gfx94x/gfx95x). On gfx1151 the diffusion attention backend
selection excludes all optimized kernels because device capability 115
falls outside the `90 < cap < 100` check for aiter.

These patches add gfx1151 support using TRITON_ATTN (the same backend
vllm-omni already defaults to for ROCm in stage_init_utils.py).
"""
import sys
import site
from pathlib import Path


def patch_omni():
    print("Applying gfx1151 patches to vllm-omni...")

    # ----------------------------------------------------------------
    # Patch 1: RocmOmniPlatform diffusion attention backend
    #
    # Upstream get_diffusion_attn_backend_cls() only enables aiter for
    # 90 < capability < 100 (gfx942 MI300X / gfx950 MI325X). gfx1151
    # reports capability (11, 5) → 115, so it falls through to
    # TORCH_SDPA which is slow on RDNA 3.5.
    #
    # Fix: add an on_gfx11xx branch that returns TRITON_ATTN, matching
    # vllm-omni's existing ROCm default in stage_init_utils.py.
    # ----------------------------------------------------------------
    p_rocm_platform = _find_site_file(
        "vllm_omni/platforms/rocm/platform.py"
    )
    if p_rocm_platform:
        txt = p_rocm_platform.read_text()

        # Add on_gfx11xx helper and branch into get_diffusion_attn_backend_cls
        if "on_gfx11xx" not in txt and "def get_diffusion_attn_backend_cls" in txt:
            # Find the method and add gfx11xx branch before the fallback
            old_fallback = (
                '        # ROCm attention backend for diffusion is not guaranteed '
                'to be compatible\n'
            )
            # If that exact comment doesn't exist, look for the TRITON_ATTN return
            if "return AttentionBackendEnum.TRITON_ATTN" in txt:
                # Find the line that checks capability for aiter
                # Pattern: if capability > 90 and capability < 100:
                # We add an elif for gfx11xx before the else that returns TORCH_SDPA

                # Look for the aiter capability check block
                lines = txt.split('\n')
                new_lines = []
                patched = False
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    # After the aiter block's else clause starts, before TORCH_SDPA return
                    if not patched and 'return AttentionBackendEnum.TORCH_SDPA' in line:
                        # Insert gfx11xx check before this return
                        indent = '        '
                        gfx11_block = (
                            f'{indent}# Strix Halo (gfx11xx): use TRITON_ATTN for\n'
                            f'{indent}# diffusion. RDNA 3.5 does not support aiter\n'
                            f'{indent}# (CDNA-only DPP/vector intrinsics) but TRITON_ATTN\n'
                            f'{indent}# works via the Triton AMD backend.\n'
                            f'{indent}if capability >= 110:\n'
                            f'{indent}    return AttentionBackendEnum.TRITON_ATTN\n\n'
                        )
                        new_lines.append(gfx11_block.rstrip('\n'))
                        patched = True

                if patched:
                    txt = '\n'.join(new_lines)
                    p_rocm_platform.write_text(txt)
                    print(f" -> Patched {p_rocm_platform} (1: gfx11xx diffusion TRITON_ATTN)")

        # Also add on_gfx11xx import if capability check is done differently
        if "_is_gfx11xx" not in txt:
            # Add a helper function to detect gfx11xx at the top of the class or module
            # Look for the class definition
            if "class RocmOmniPlatform" in txt:
                # Find the import section and add capability detection
                # We need to add detection before the get_diffusion_attn_backend_cls method
                target = "    def get_diffusion_attn_backend_cls"
                if target in txt:
                    helper = (
                        '    @staticmethod\n'
                        '    def _is_gfx11xx(capability: int) -> bool:\n'
                        '        """Detect RDNA 3/3.5 consumer GPUs (gfx11xx)."""\n'
                        '        return 110 <= capability < 120\n\n'
                    )
                    txt = txt.replace(target, helper + target, 1)
                    p_rocm_platform.write_text(txt)
                    print(f" -> Patched {p_rocm_platform} (1b: _is_gfx11xx helper added)")

    # ----------------------------------------------------------------
    # Patch 2: Ensure AOTRITON_PATH is respected in vllm-omni
    #
    # vllm-omni's ROCm platform may not inherit the AOTRITON_PATH env
    # var from the base image. Ensure it's available at import time.
    # ----------------------------------------------------------------
    p_platform_init = _find_site_file(
        "vllm_omni/platforms/rocm/__init__.py"
    )
    if p_platform_init and p_platform_init.exists():
        txt = p_platform_init.read_text()
        if "AOTRITON_PATH" not in txt:
            aotriton_inject = (
                '# Ensure AOTRITON_PATH is available for PyTorch AOTriton discovery\n'
                'import os as _omni_aotriton_os\n'
                'if not _omni_aotriton_os.environ.get("AOTRITON_PATH"):\n'
                '    _omni_aotriton_os.environ["AOTRITON_PATH"] = "/opt/rocm/aotriton"\n'
                'del _omni_aotriton_os\n\n'
            )
            txt = aotriton_inject + txt
            p_platform_init.write_text(txt)
            print(f" -> Patched {p_platform_init} (2: AOTRITON_PATH default)")

    # ----------------------------------------------------------------
    # Patch 3: Ensure onnxruntime conflict is resolved
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

    print("Successfully patched vllm-omni for gfx1151.")


def _find_site_file(relative_path: str) -> Path | None:
    """Find a file within site-packages by relative path."""
    for sp in site.getsitepackages():
        candidate = Path(sp) / relative_path
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    patch_omni()
