"""
Strix Halo (gfx1151) patch bundle for vLLM source builds.

Applies 19 patches (numbered 1-19 sequentially) at build time to make
vLLM functional on RDNA 3.5 (gfx1151 / Strix Halo) where upstream
support is incomplete.

Patch summary:
  1   amdsmi import bypass (platforms/__init__.py)
  2   on_gfx1x() helper injection (platforms/rocm.py)
  3   force gfx1151 arch + device_name (platforms/rocm.py)
  4   AITER ops: gfx1x support, disable FP8 linear/RMSNorm/MoE (_aiter_ops.py)
  5   AITER FA backend gfx1x support (rocm_aiter_fa.py)
  6   block AITER MoE forced override (unquantized.py)
  7   custom_ops RMSNorm bypass on gfx1x (platforms/rocm.py)
  8   aiter JIT __path__ fix + aiter fusion dedup fix
  9   flash_attn soft import for aiter resilience
  10  Triton MoE capability cap for gfx11xx
  11  APU VRAM dynamic margin (ROCM-21812, guarded — removable if ROCm PR #5113 merged)
  12  hipCtx deprecation warning suppression
  13  chat_template_kwargs through /v1/responses (protocol.py — upstream candidate)
  14  AWQ-INT4 MMQ HIP kernel registration
  15  atomicAdd half/half2 polyfill removal on ROCm
  16  profile_run cache (gpu_worker.py — skip ~7 min on restart)
  17  non-streaming /v1/responses enable_thinking=false fix (serving.py)
  18  Triton softmax segments tuning 16→32 (triton_attn.py)
  19  Triton SDPA shared memory cap for gfx11 (triton_unified_attention.py)

Categories (for reference/reuse):
  Hardware enablement:  1-3    (ROCm APUs, gfx1151-specific)
  AITER compatibility:  4-9    (CDNA-only kernel guards, JIT fixes)
  ROCm SDK bugs:        10-12  (MoE cap, VRAM clamp, hipCtx deprecation)
  Upstream candidates:  13,17  (generic vLLM bugs, not gfx1151-specific)
  Local features:       14-16  (AWQ MMQ, profile cache, gfx1151 tuning)
  gfx1151 performance:  18-19  (softmax tuning, Triton LDS caps)

Historical mapping (old → new):
  1   → 1,  1.25 → 2,  1.5 → 3,  2 → 4,  3 → 5,  3.5 → 6
  5   → 7,  6 → 8a, 7 (attrs) → 8b, 7 (aiter) → 8c, 8 → 9
  9   → 10, 10 → 11, 11 → 12, 12 (GGUF) → REMOVED
  13a → 13a, 13b → 13b, 14 (MMQ) → 14, 13 (atomicAdd) → 15
  14a → 16a, 14b → 16b, 14 (enable_thinking) → 17a+17b, 13 (softmax) → 18

Every patch is gfx1151-driven, not quant-driven. The AWQ-INT4 model
exercises the same RDNA 3.5 code paths as the BF16 model.

If a future tool-call / reasoning-parser PR is needed before merging
upstream, add it here as a new numbered patch and rebuild — the existing
patch numbers stay stable so cross-repo references don't drift.
"""
import sys
import re
import site
import os
import shutil
from pathlib import Path

def patch_vllm(vllm_root=None, dry_run=False):
    if vllm_root is not None:
        os.chdir(vllm_root)
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Applying Strix Halo patches to vLLM (ai-notes modernization)...")

    # Patch 1: vllm/platforms/__init__.py (amdsmi monkey patch  -  PROVEN working for 5 months)
    # Comment out real amdsmi imports and replace with pass stubs.
    # The actual amdsmi library doesn't work on Strix Halo APUs in containers.
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py (amdsmi disabled, is_rocm forced True)")

    # Patch 2 (was 1.25): inject on_gfx1x() into vllm/platforms/rocm.py.
    # v0.20.0 from vllm-project does NOT ship this function (only in
    # ROCm/vllm gfx11 branch). Multiple patches below depend on it
    # (`from vllm.platforms.rocm import on_gfx1x`), so define it first.
    # Uses current_platform.device_name (set by Patch 3 to "gfx1151")
    # rather than _get_gcn_arch_name() whose name varies across vLLM versions.
    p_rocm_plat_gfx = Path('vllm/platforms/rocm.py')
    if p_rocm_plat_gfx.exists():
        txt = p_rocm_plat_gfx.read_text()
        if "def on_gfx1x()" not in txt:
            injection = (
                "\n\ndef on_gfx1x() -> bool:\n"
                '    """Return True on RDNA 3.5 (gfx115x / Strix Halo)."""\n'
                "    try:\n"
                "        from vllm.platforms import current_platform as _cp\n"
                "        return getattr(_cp, 'device_name', '').startswith('gfx115')\n"
                "    except Exception:\n"
                "        return False\n"
            )
            # Insert after the first function definition (or class) so it's
            # at module level but after imports.
            m = re.search(r'\n(?:class |def )', txt)
            if m:
                txt = txt[:m.start()] + injection + txt[m.start():]
            else:
                txt += injection
            p_rocm_plat_gfx.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (2: injected on_gfx1x helper)")

    # Patch 3 (was 1.5): vllm/platforms/rocm.py (MagicMock amdsmi + force gfx1151)
    # Prepend MagicMock so any remaining amdsmi references in rocm.py silently succeed.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        # Add MagicMock header if not already present
        if 'sys.modules["amdsmi"] = MagicMock()' not in txt:
            header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
            txt = header + txt
        # Force arch detection
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        p_rocm_plat.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (MagicMock amdsmi + forced gfx1151)")

    # Patch 4 (was 2): _aiter_ops.py (Enable AITER on gfx1x, disable FP8 linear)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()

        # Ensure on_gfx1x is available globally for our patches below
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms import current_platform",
                              "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x")

        # Extend is_aiter_found_and_supported
        if "or on_gfx1x()" not in txt:
            txt = txt.replace("import on_mi3xx", "import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")

        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n',
                r'\1 False\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER RMSNorm on gfx1x (CUDA Graph hang)
        if "is_rmsnorm_enabled" in txt:
            txt = re.sub(
                r'(def is_rmsnorm_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._RMSNORM_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER Fused MoE on gfx1x (due to hundreds of CDNA-specific dpp_mov assembly conflicts)
        if "is_fused_moe_enabled" in txt:
            txt = re.sub(
                r'(def is_fused_moe_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._FMOE_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1x support, FP8 linear empty, MoE disabled)")

    # Patch 5 (was 3): rocm_aiter_fa.py
    p_fa = Path('vllm/v1/attention/backends/rocm_aiter_fa.py')
    if p_fa.exists():
        txt = p_fa.read_text()
        if "on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms.rocm import on_mi3xx", "from vllm.platforms.rocm import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")
            p_fa.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_fa.py (gfx1x support)")

    # Patch 6 (was 3.5): unquantized.py (Hard-block AITER MoE forced override on gfx1x)
    p_unquant = Path('vllm/model_executor/layers/fused_moe/oracle/unquantized.py')
    if p_unquant.exists():
        txt = p_unquant.read_text()
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                'if envs.is_set("VLLM_ROCM_USE_AITER")',
                'from vllm.platforms.rocm import on_gfx1x\n    if envs.is_set("VLLM_ROCM_USE_AITER")'
            )
            txt = txt.replace(
                'if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:',
                'if getattr(on_gfx1x, "__call__", lambda: False)() or not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:'
            )
            p_unquant.write_text(txt)
            print(" -> Patched unquantized.py (Blocked AITER MoE override on gfx1x)")


    # Patch 7 (was 5): custom_ops RMSNorm block on gfx1x (Full CUDA Graph capture)
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()

        # Legacy vLLM < 0.19 fallback
        if "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")" in txt:
            txt = txt.replace(
                "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")",
                "if is_aiter_found_and_supported() and not getattr(self, 'on_gfx1x', lambda: False)():\n            custom_ops.append(\"+rms_norm\")"
            )

        # Modern vLLM 0.19+ struct (compilation_config.custom_ops)
        elif "compilation_config.custom_ops.append(\"+rms_norm\")" in txt:
            if "if not getattr(self, \"on_gfx1x\", lambda: False)():" not in txt:
                txt = re.sub(
                    r'(\s+)compilation_config\.custom_ops\.append\("\+rms_norm"\)',
                    r'\1if not getattr(self, "on_gfx1x", lambda: False)():\n\1    compilation_config.custom_ops.append("+rms_norm")',
                    txt
                )

        # Modern vLLM 0.19.2rc1+ IrOpPriorityConfig bypass
        if 'rms_norm = ["aiter"] + default' in txt:
            txt = txt.replace(
                'rms_norm = ["aiter"] + default',
                'rms_norm = ["aiter"] + default if not on_gfx1x() else default'
            )

        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (custom_ops & IrOpPriorityConfig rms_norm bypassed on gfx1x)")

    # Patch 8a (was 6): rocm_aiter_fusion.py (duplicate pattern bypass)
    p_fusion = Path('vllm/compilation/passes/fusion/rocm_aiter_fusion.py')
    if p_fusion.exists():
        txt = p_fusion.read_text()
        if "skip_duplicates=True" not in txt:
            txt = re.sub(
                r"(pm\.register_replacement\s*\((?:(?!\bpm\.register_replacement\b).)*?)pm_pass(\s*[\),])",
                r"\1pm_pass, skip_duplicates=True\2",
                txt, flags=re.DOTALL
            )
            p_fusion.write_text(txt)
            print(" -> Patched vllm/compilation/passes/fusion/rocm_aiter_fusion.py (skip_duplicates)")

    # Patch 8b (was 7): Triton backend AttrsDescriptor repr
    for sp in site.getsitepackages():
        triton_compiler = Path(sp) / "triton/backends/compiler.py"
        if triton_compiler.exists():
            txt = triton_compiler.read_text()
            if "def __repr__(self):" not in txt:
                txt = txt.replace(
                    "def to_dict(self):",
                    "def __repr__(self):\n        return f'AttrsDescriptor.from_dict({self.to_dict()!r})'\n\n    def to_dict(self):"
                )
                triton_compiler.write_text(txt)
                print(f" -> Patched {triton_compiler} (AttrsDescriptor repr)")

    # Patch 8c (was 7): aiter JIT path fix  -  aiter builds .so files into ~/.aiter/jit/
    # but importlib.import_module("aiter.jit.<module>") only looks in the
    # installed package directory. Fix by adding the JIT cache to __path__.
    for sp in site.getsitepackages():
        aiter_jit_init = Path(sp) / "aiter/jit/__init__.py"
        if aiter_jit_init.exists():
            txt = aiter_jit_init.read_text()
            if "# PATCHED: JIT cache path" not in txt:
                jit_path_fix = '''
# PATCHED: JIT cache path for Strix Halo
# aiter's JIT compiles .so modules into ~/.aiter/jit/ but importlib looks
# in the installed package directory. Add the JIT cache to __path__.
import os as _os
_jit_cache = _os.path.join(_os.path.expanduser("~"), ".aiter", "jit")
if _os.path.isdir(_jit_cache) and _jit_cache not in __path__:
    __path__.append(_jit_cache)
'''
                txt += jit_path_fix
                aiter_jit_init.write_text(txt)
                print(f" -> Patched {aiter_jit_init} (JIT cache added to __path__)")

    # Patch 9 (was 8): flash_attn_interface.py  -  make aiter import soft as safety net.
    # If aiter JIT fails for any reason, flash_attn should still load (TRITON_ATTN works).
    # ROCM_ATTN will also work when aiter JIT succeeds (patch 7 fixes the path).
    hard_import_bare = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"

    def _patch_flash_interface(fa_iface):
        txt = fa_iface.read_text()
        if hard_import_bare not in txt or "except (ImportError" in txt:
            return False
        # Detect indentation of the original import line
        m = re.search(r'^( *)' + re.escape(hard_import_bare), txt, re.MULTILINE)
        if not m:
            return False
        indent = m.group(1)
        original_line = indent + hard_import_bare
        soft_import = (
            f"{indent}try:\n"
            f"{indent}    {hard_import_bare}\n"
            f"{indent}except (ImportError, KeyError, ModuleNotFoundError):\n"
            f"{indent}    flash_attn_gpu = None"
        )
        txt = txt.replace(original_line, soft_import)
        fa_iface.write_text(txt)
        print(f" -> Patched {fa_iface} (aiter import made resilient)")
        return True

    for sp in site.getsitepackages():
        for fa_egg in Path(sp).glob("flash_attn*.egg"):
            fa_iface = fa_egg / "flash_attn/flash_attn_interface.py"
            if fa_iface.exists():
                _patch_flash_interface(fa_iface)
        # Also check non-egg installs
        fa_iface = Path(sp) / "flash_attn/flash_attn_interface.py"
        if fa_iface.exists():
            _patch_flash_interface(fa_iface)

    # Patch 10 (was 9): Allow Triton MoE kernels on gfx11xx (Strix Halo)
    # vLLM recently capped MXFP4 Triton MoE kernels to < (11, 0) which excludes RDNA3.5 (11.x)
    for p_triton in [
        Path('vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py'),
        Path('vllm/model_executor/layers/fused_moe/oracle/mxfp4.py')
    ]:
        if p_triton.exists():
            txt = p_triton.read_text()
            if "cap.minor) < (11, 0)" in txt:
                txt = txt.replace("cap.minor) < (11, 0)", "cap.minor) < (12, 0)")
            if "capability() < (11, 0)" in txt:
                txt = txt.replace("capability() < (11, 0)", "capability() < (12, 0)")
            p_triton.write_text(txt)
            print(f" -> Patched {p_triton} (Triton MoE on gfx11xx)")

    # Patch 11 (was 10): ROCM-21812 APU VRAM Dynamic Margin Patch
    #
    # Explanation: ROCm nightly builds introduced a 50% APU VRAM clamp to prevent
    # OOM kernel panics on headless hosts. This broke vLLM large model loading.
    # This patch intercepts PyTorch memory bounds and dynamically proxies the
    # real amdgpu hardware GTT limits, minus a strict 8GB OS safety margin.
    # By symmetrically carving the OS margin from the top of the GTT ceiling,
    # vLLM's memory profiler allocates flawlessly while guaranteeing the OS stays alive,
    # regardless of the specific GTT allocation size on the host.
    # Ref: https://github.com/ROCm/rocm-systems/pull/5113
    #
    # GUARD: If the upstream ROCm PR #5113 fix has landed in the installed
    # ROCm SDK (detected by checking whether ROCm reports total > 70 GiB
    # without clamping), this patch becomes a no-op and should be removed.
    # TODO: Remove this patch block once PR #5113 is in the nightly tarballs.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        if "_patched_mem_info" not in txt:
            mem_patch = '''
# --- ROCM-21812 VRAM DYNAMIC PATCH ---
import torch
import glob
import os

try:
    _orig_mem_info = torch.cuda.mem_get_info
    _orig_get_dev_prop = torch.cuda.get_device_properties

    class MockCudaDeviceProperties:
        def __init__(self, prop, override_total):
            self._prop = prop
            self.total_memory = override_total
        def __getattr__(self, name):
            return getattr(self._prop, name)
        def __dir__(self):
            return dir(self._prop)

    def _patched_mem_info(device=None):
        free, total = _orig_mem_info(device)
        try:
            # On APUs, ROCm clamps total to 50% limit. We need the real GTT limits.
            if total < 70 * 1024**3:
                drm_cards = glob.glob('/sys/class/drm/card*/device/mem_info_gtt_total')
                if drm_cards:
                    card_dir = os.path.dirname(drm_cards[0])
                    with open(os.path.join(card_dir, 'mem_info_gtt_total'), 'r') as f:
                        gtt_total = int(f.read().strip())
                    with open(os.path.join(card_dir, 'mem_info_gtt_used'), 'r') as f:
                        gtt_used = int(f.read().strip())

                    # Symmetrically carve 8GB off the TOP of the device perfectly.
                    safe_ceiling = gtt_total - (8 * 1024**3)

                    real_total = safe_ceiling
                    real_free = max(0, safe_ceiling - gtt_used)

                    total = max(total, real_total)
                    free = real_free
        except Exception as e:
            pass
        return int(free), int(total)

    def _patched_get_dev_prop(device=None):
        prop = _orig_get_dev_prop(device)
        free, total = _patched_mem_info(device)
        if hasattr(prop, 'total_memory') and prop.total_memory < total:
            return MockCudaDeviceProperties(prop, total)
        return prop

    torch.cuda.mem_get_info = _patched_mem_info
    torch.cuda.get_device_properties = _patched_get_dev_prop
except Exception:
    pass
# ---------------------------
'''
            txt = mem_patch + txt
            p_rocm_plat.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (ROCM-21812 APU VRAM Dynamic Margin)")

    # Patch 12 (was 11): silence hipCtx* deprecation warnings in
    # csrc/cumem_allocator_compat.h. vLLM still uses hipCtxGetCurrent /
    # hipCtxSetCurrent / hipDevicePrimaryCtxRetain for CUDA-compat context
    # management; HIP marked these deprecated but there is no clean
    # replacement for the use case, and upstream vLLM hasn't migrated yet.
    # Suppressing the warning class for that file keeps our build clean.
    p_cumem = Path('csrc/cumem_allocator_compat.h')
    if p_cumem.exists():
        txt = p_cumem.read_text()
        marker = '#pragma clang diagnostic ignored "-Wdeprecated-declarations"'
        if marker not in txt:
            txt = marker + "\n" + txt
            p_cumem.write_text(txt)
            print(" -> Patched csrc/cumem_allocator_compat.h (suppress hipCtx* deprecations)")

    # Patch 13 (upstream candidate): thread chat_template_kwargs through
    # /v1/responses. Without this, ResponsesRequest.to_chat_params() builds
    # chat_template_kwargs from a hardcoded dict and ignores the request body.
    # 13a: add the field to ResponsesRequest
    # 13b: pass it as `defaults` to merge_kwargs()
    p_responses_proto = Path('vllm/entrypoints/openai/responses/protocol.py')
    if p_responses_proto.exists():
        txt = p_responses_proto.read_text()

        # 13a: add chat_template_kwargs field, sandwiched between `user` (last
        # OpenAI-spec field) and `skip_special_tokens` (first vLLM extension).
        field_anchor = "    user: str | None = None\n    skip_special_tokens: bool = True\n"
        field_replacement = (
            "    user: str | None = None\n"
            "    chat_template_kwargs: dict[str, Any] | None = None\n"
            "    skip_special_tokens: bool = True\n"
        )
        if "chat_template_kwargs: dict[str, Any] | None = None" not in txt and field_anchor in txt:
            txt = txt.replace(field_anchor, field_replacement, 1)
            print(" -> Patched protocol.py (13a: ResponsesRequest gains chat_template_kwargs field)")

        # 13b: in to_chat_params(), feed the user kwargs into merge_kwargs as
        # the `defaults` argument. The hardcoded dict stays as `overrides` so
        # vLLM-managed keys (add_generation_prompt, continue_final_message,
        # reasoning_effort) keep precedence, while user-supplied keys
        # (enable_thinking, etc.) flow through to the chat template renderer.
        # Indents: the call sits inside `return ChatParams(` so it's 12 spaces
        # for the kwarg line and 16 spaces for the inner positional args.
        merge_anchor = (
            "            chat_template_kwargs=merge_kwargs(  # To remove unset values\n"
            "                {},\n"
            "                dict(\n"
            "                    add_generation_prompt=not continue_final,\n"
        )
        merge_replacement = (
            "            chat_template_kwargs=merge_kwargs(  # To remove unset values\n"
            "                self.chat_template_kwargs or {},\n"
            "                dict(\n"
            "                    add_generation_prompt=not continue_final,\n"
        )
        if "self.chat_template_kwargs or {}" not in txt and merge_anchor in txt:
            txt = txt.replace(merge_anchor, merge_replacement, 1)
            print(" -> Patched protocol.py (13b: to_chat_params merges user chat_template_kwargs)")

        p_responses_proto.write_text(txt)

    # Patch 14 (local, from hec-ovi/vllm-awq4-qwen): register the AWQ-INT4 MMQ
    # HIP custom op into vLLM's mixed-precision kernel dispatcher so it's
    # picked ahead of TritonW4A16 for the W4A16 g32 path on gfx1151.
    # The .so is built from csrc/awq_mmq_gfx1151/ (host-mounted at /root/csrc/)
    # and imports lazily at module-load time.
    #
    # Implementation: append a registration block to the dispatcher's
    # __init__.py. On load the block adds the package dir to sys.path,
    # imports our RocmMmqQ4LinearKernel, and inserts it at position 0 of
    # _POSSIBLE_KERNELS[ROCM]. If the import fails (e.g. .so not built yet),
    # the kernel list is left untouched and TritonW4A16 keeps its slot.
    #
    # apply_weights internally dispatches: M >= 32 (prefill) -> our HIP
    # kernel, M < 32 (decode) -> TritonW4A16 fallback. Both paths use the
    # same layer's weight tensors via the dual-storage process_weights step.
    p_dispatch = Path('vllm/model_executor/kernels/linear/__init__.py')
    if p_dispatch.exists():
        txt = p_dispatch.read_text()
        if "Patch 14" not in txt:
            injection = (
                "\n\n# --- Patch 14: AWQ-INT4 MMQ HIP custom op for gfx1151 (Strix Halo) ---\n"
                "import sys as _sys\n"
                "import os as _os\n"
                "_AWQ_MMQ_DIR = '/root/csrc/awq_mmq_gfx1151'\n"
                "if _os.path.exists(_AWQ_MMQ_DIR) and _AWQ_MMQ_DIR not in _sys.path:\n"
                "    _sys.path.insert(0, _AWQ_MMQ_DIR)\n"
                "try:\n"
                "    from awq_mmq_gfx1151.vllm_kernel import RocmMmqQ4LinearKernel as _RocmMmqQ4\n"
                "    if _RocmMmqQ4 not in _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, []):\n"
                "        _POSSIBLE_KERNELS[PlatformEnum.ROCM].insert(0, _RocmMmqQ4)\n"
                "        logger.info('Patch 14: RocmMmqQ4LinearKernel registered at _POSSIBLE_KERNELS[ROCM][0]')\n"
                "except Exception as _e:\n"
                "    logger.warning('Patch 14: failed to register RocmMmqQ4LinearKernel: %s', _e)\n"
                "# --- end Patch 14 ---\n"
            )
            txt += injection
            p_dispatch.write_text(txt)
            print(" -> Patched vllm/model_executor/kernels/linear/__init__.py (Patch 14: AWQ-INT4 MMQ HIP)")

    # Patch 15 (was 13, from hec-ovi/vllm-awq4-qwen): drop vLLM's half/half2
    # atomicAdd polyfills on ROCm.
    #
    # csrc/quantization/gptq/compat.cuh ships polyfills
    #   __device__ void atomicAdd(half*  address, half  val)
    #   __device__ void atomicAdd(half2* address, half2 val)
    # gated on `#if defined(__CUDA_ARCH__) || defined(USE_ROCM)`. ROCm 7.13
    # nightlies (post 7.13.0a20260426) added builtins
    #   __device__ __half  atomicAdd(__half*  const, const __half)   @ amd_hip_fp16.h:869
    #   __device__ __half2 atomicAdd(__half2* const, const __half2)  @ amd_hip_fp16.h:875
    # With both the polyfill and the builtin visible, clang reports
    # "call to 'atomicAdd' is ambiguous" in q_gemm.hip (10 sites).
    #
    # Fix: change the outermost guard to drop the entire ROCm path through
    # this overload region. The polyfills are now CUDA-only; ROCm uses the
    # HIP builtins exclusively. The named helpers atomicAdd_half /
    # atomicAdd_half2 (defined above the guard) are untouched in case any
    # other vLLM source calls them by name.
    p_compat = Path('csrc/quantization/gptq/compat.cuh')
    if p_compat.exists():
        txt = p_compat.read_text()
        old_guard = "#if defined(__CUDA_ARCH__) || defined(USE_ROCM)\n"
        new_guard = "#if defined(__CUDA_ARCH__)\n"
        if old_guard in txt:
            txt = txt.replace(old_guard, new_guard, 1)
            p_compat.write_text(txt)
            print(" -> Patched csrc/quantization/gptq/compat.cuh (Patch 15: drop atomicAdd half/half2 polyfills on ROCm)")

    # ----------------------------------------------------------------
    # Patch 16 (was 14): Cache profile_run results to skip ~7 min memory
    # profiling on restart.
    #
    # vLLM runs synthetic forward passes every boot to size the KV cache.
    # On Strix Halo with --enforce-eager, the dominant cost is Triton JIT
    # compilation + dummy runs, not cudagraph capture. Since the Triton
    # cache is already persisted to disk, the profile result is stable
    # across restarts for the same config.
    #
    # This patch checks for a cached KV cache memory value at the top of
    # GPUWorker.determine_available_memory(). If found, it returns
    # immediately (skipping profile_run). Otherwise it runs normal
    # profiling and caches the result for the next restart.
    #
    # Controlled by VLLM_SKIP_MEMORY_PROFILING=1 (opt-in) and
    # VLLM_PROFILE_CACHE_DIR (defaults to /root/.cache/vllm-profile).
    # The cache module (vllm_profile_cache.py) is copied into the image
    # at /opt/vllm_profile_cache.py by the Dockerfile.
    # ----------------------------------------------------------------
    p_gpu_worker = Path('vllm/v1/worker/gpu_worker.py')
    if p_gpu_worker.exists():
        txt = p_gpu_worker.read_text()

        # 16a: Insert cache read at the top of determine_available_memory().
        read_anchor = (
            '        """\n'
            '        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n'
            '            # still need a profile run which compiles the model for\n'
            '            # max_num_batched_tokens\n'
            '            self.model_runner.profile_run()\n'
        )
        read_replacement = (
            '        """\n'
            '        # Strix Halo Patch 16: Try to use cached profile result\n'
            '        # to skip ~7 min memory profiling on restart.\n'
            '        if envs.VLLM_SKIP_MEMORY_PROFILING:\n'
            '            try:\n'
            '                import vllm_profile_cache as _vpc\n'
            '                cache_dir = envs.VLLM_PROFILE_CACHE_DIR or "/root/.cache/vllm-profile"\n'
            '                cached = _vpc.read_cached_kv_cache_memory_bytes(cache_dir, self.vllm_config)\n'
            '                if cached is not None:\n'
            '                    logger.info(\n'
            '                        "Using cached KV cache memory from profile_cache: %s GiB",\n'
            '                        format_gib(cached),\n'
            '                    )\n'
            '                    return cached\n'
            '            except Exception:\n'
            '                logger.debug("Profile cache read failed; falling back to profiling")\n\n'
            '        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n'
            '            # still need a profile run which compiles the model for\n'
            '            # max_num_batched_tokens\n'
            '            self.model_runner.profile_run()\n'
        )
        if "VLLM_SKIP_MEMORY_PROFILING" not in txt and read_anchor in txt:
            txt = txt.replace(read_anchor, read_replacement, 1)
            print(" -> Patched vllm/v1/worker/gpu_worker.py (16a: cache read at top of determine_available_memory)")

        # 16b: Insert cache write after computing available_kv_cache_memory_bytes.
        write_anchor = (
            '        self.available_kv_cache_memory_bytes = (\n'
            '            self.requested_memory\n'
            '            - profile_result.non_kv_cache_memory\n'
            '            - cudagraph_memory_estimate_applied\n'
            '        )\n\n'
            '        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory\n'
        )
        write_replacement = (
            '        self.available_kv_cache_memory_bytes = (\n'
            '            self.requested_memory\n'
            '            - profile_result.non_kv_cache_memory\n'
            '            - cudagraph_memory_estimate_applied\n'
            '        )\n\n'
            '        # Strix Halo Patch 16: Cache the result for future restarts.\n'
            '        if envs.VLLM_SKIP_MEMORY_PROFILING:\n'
            '            try:\n'
            '                import vllm_profile_cache as _vpc\n'
            '                cache_dir = envs.VLLM_PROFILE_CACHE_DIR or "/root/.cache/vllm-profile"\n'
            '                _vpc.write_cached_kv_cache_memory_bytes(\n'
            '                    cache_dir, self.available_kv_cache_memory_bytes, self.vllm_config,\n'
            '                )\n'
            '                logger.info(\n'
            '                    "Cached KV cache memory to profile_cache: %s GiB",\n'
            '                    format_gib(self.available_kv_cache_memory_bytes),\n'
            '                )\n'
            '            except Exception:\n'
            '                logger.debug("Profile cache write failed (non-fatal)")\n\n'
            '        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory\n'
        )
        if "Strix Halo Patch 16" not in txt and write_anchor in txt:
            txt = txt.replace(write_anchor, write_replacement, 1)
            print(" -> Patched vllm/v1/worker/gpu_worker.py (16b: cache write after profiling)")

        p_gpu_worker.write_text(txt)

    # ----------------------------------------------------------------
    # Patch 17 (was 13, PR #40334 cherry-pick) — dtype cast in
    # combine_hidden_states for mixed-precision targets.
    #
    # AWQ-quantized models with unquantized attention layers can output
    # float32 activations that get passed to the draft head's fc layer
    # (which expects params_dtype, typically bfloat16). Without this
    # cast, generation crashes with:
    #   RuntimeError: expected scalar type Float but found Half
    #
    # Upstream PR: https://github.com/vllm-project/vllm/pull/40334
    # (OPEN as of 2026-04-30)
    # ----------------------------------------------------------------
    p_qwen3_dflash = Path('vllm/model_executor/models/qwen3_dflash.py')
    if p_qwen3_dflash.exists():
        txt = p_qwen3_dflash.read_text()

        old_block = (
            '        if not self.model.use_aux_hidden_state:\n'
            '            return hidden_states\n'
            '        needs_squeeze = hidden_states.dim() == 1\n'
            '        if needs_squeeze:\n'
            '            hidden_states = hidden_states.unsqueeze(0)\n'
            '        result = self.model.fc(hidden_states)\n'
        )
        new_block = (
            '        if not self.model.use_aux_hidden_state:\n'
            '            return hidden_states\n'
            '        # Cast to fc params_dtype to handle mixed-precision\n'
            '        # targets (e.g. AWQ with unquantized attention layers\n'
            '        # that output float32 activations).\n'
            '        if hidden_states.dtype != self.model.fc.params_dtype:\n'
            '            hidden_states = hidden_states.to(self.model.fc.params_dtype)\n'
            '        needs_squeeze = hidden_states.dim() == 1\n'
            '        if needs_squeeze:\n'
            '            hidden_states = hidden_states.unsqueeze(0)\n'
            '        result = self.model.fc(hidden_states)\n'
        )
        if "hidden_states.dtype != self.model.fc.params_dtype" not in txt and old_block in txt:
            txt = txt.replace(old_block, new_block, 1)
            p_qwen3_dflash.write_text(txt)
            print(" -> Patched vllm/model_executor/models/qwen3_dflash.py (17: PR #40334 dtype cast in combine_hidden_states)")

    # ----------------------------------------------------------------
    # Patch 17 (upstream candidate): Fix non-streaming /v1/responses with
    # enable_thinking=false. Patch 13 wired chat_template_kwargs through the
    # request model, but _make_response_output_items() (non-streaming path)
    # creates the parser without passing it. 17a: pass chat_template_kwargs to
    # the parser; 17b: add is_reasoning_end check on prompt_token_ids as safety
    # net.
    # ----------------------------------------------------------------
    p_responses_serving = Path('vllm/entrypoints/openai/responses/serving.py')
    if p_responses_serving.exists():
        txt = p_responses_serving.read_text()

        # 17a: Pass chat_template_kwargs to the parser.
        parser_call_old = (
            '        # Use parser to extract and create response output items\n'
            '        if self.parser:\n'
            '            parser = self.parser(tokenizer, request.tools)\n'
            '            return parser.extract_response_outputs('
        )
        parser_call_new = (
            '        # Use parser to extract and create response output items\n'
            '        if self.parser:\n'
            '            parser = self.parser(\n'
            '                tokenizer,\n'
            '                request.tools,\n'
            '                chat_template_kwargs=self._effective_chat_template_kwargs(request),\n'
            '            )\n'
            '            return parser.extract_response_outputs('
        )
        if "self._effective_chat_template_kwargs(request)" not in txt and parser_call_old in txt:
            txt = txt.replace(parser_call_old, parser_call_new, 1)
            print(" -> Patched vllm/entrypoints/openai/responses/serving.py (17a: pass chat_template_kwargs to parser)")

        # 17b: Safety net — skip parser when reasoning already ended in prompt.
        # Insert right before "if self.parser:" block.
        safety_anchor = (
            '        # Use parser to extract and create response output items\n'
            '        if self.parser:\n'
        )
        safety_replacement = (
            '        # Strix Halo Patch 17b: If reasoning already ended in the\n'
            '        # prompt (enable_thinking=false pre-fills <think>\\n\\n</think>),\n'
            '        # skip the parser and treat the full output as content.\n'
            '        # This mirrors the streaming path behavior.\n'
            '        reasoning_ended_in_prompt = False\n'
            '        if (\n'
            '            self.parser is not None\n'
            '            and self.parser.reasoning_parser_cls is not None\n'
            '            and final_res.prompt_token_ids is not None\n'
            '        ):\n'
            '            try:\n'
            '                reasoning_parser = self.parser.reasoning_parser_cls(\n'
            '                    tokenizer,\n'
            '                    chat_template_kwargs=self._effective_chat_template_kwargs(request),\n'
            '                )\n'
            '                reasoning_ended_in_prompt = reasoning_parser.is_reasoning_end(\n'
            '                    final_res.prompt_token_ids\n'
            '                )\n'
            '            except Exception:\n'
            '                pass\n'
            '        if reasoning_ended_in_prompt:\n'
            '            return [\n'
            '                ResponseOutputMessage(\n'
            '                    id=f"msg_{random_uuid()}",\n'
            '                    content=[\n'
            '                        ResponseOutputText(\n'
            '                            text=final_output.text,\n'
            '                            annotations=[],\n'
            '                            type="output_text",\n'
            '                            logprobs=logprobs,\n'
            '                        )\n'
            '                    ] if final_output.text else [],\n'
            '                    role="assistant",\n'
            '                    status="completed",\n'
            '                    type="message",\n'
            '                )\n'
            '            ]\n\n'
            '        # Use parser to extract and create response output items\n'
            '        if self.parser:\n'
        )
        if "Strix Halo Patch 17b" not in txt and safety_anchor in txt:
            txt = txt.replace(safety_anchor, safety_replacement, 1)
            print(" -> Patched vllm/entrypoints/openai/responses/serving.py (17b: is_reasoning_end safety net)")

        p_responses_serving.write_text(txt)

    # Patch 18 (was 13, from ROCm/vllm gfx11): Strix Halo softmax segments tuning.
    # On gfx1151 with MQA (num_kv_heads == 1) or large head sizes (>= 224),
    # increasing num_par_softmax_segments from 16 to 32 shows measurable
    # gains in the Triton attention backend (ROCm/vllm gfx11 branch).
    p_triton_attn = Path('vllm/v1/attention/backends/triton_attn.py')
    if p_triton_attn.exists():
        txt = p_triton_attn.read_text()
        applied = False

        # Add import of on_gfx1x and get_current_platform at the top.
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                "from vllm.platforms import current_platform",
                "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x",
                1,
            )
            applied = True

        # Inject gfx1151 softmax segments override in TritonAttentionMetadataBuilder.
        # Look for the default assignment in __init__ and add a conditional override.
        old_default = "self.num_par_softmax_segments = 16"
        new_override = (
            "self.num_par_softmax_segments = 16\n"
            "        # Strix Halo (gfx1151): 32 segments shows gains for MQA or large heads.\n"
            "        if on_gfx1x() and self.mq_head_size >= 224:\n"
            "            self.num_par_softmax_segments = 32\n"
        )
        if "Strix Halo (gfx1151): 32 segments" not in txt and old_default in txt:
            txt = txt.replace(old_default, new_override, 1)
            applied = True

        if applied:
            p_triton_attn.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/triton_attn.py (Patch 18: gfx1151 softmax segments 16→32)")

    # Patch 19 (ROCm/vllm gfx11 PR #919, #911): cap Triton SDPA shared memory
    # usage on gfx11 (RDNA 3/3.5) to prevent OutOfResources errors.
    #
    # RDNA3 LDS (shared memory) is 64 KB per CU. The unified attention kernel's
    # Q-tile = BLOCK_M * head_size * element_size can exceed this for large
    # head_size (>= 256) or when the autotuner picks BLOCK_M=128 with
    # head_size >= 256.
    #
    # Upstream commits: 8943cfb2 (TILE_SIZE cap), 2e4ab9fe (BLOCK_M + stages).
    p_unified = Path('vllm/v1/attention/ops/triton_unified_attention.py')
    if p_unified.exists():
        txt = p_unified.read_text()

        # 19a: Add on_gfx1x import.
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                "from vllm.platforms import current_platform",
                "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x",
                1,
            )
            print(" -> Patched triton_unified_attention.py (19a: import on_gfx1x)")

        # 19b: Cap BLOCK_M / max_num_stages / TILE_SIZE after they are computed
        #      but before the kernel launch.
        #
        # Injection point: right after the TILE_SIZE_DECODE assignment line.
        inject_after = "    TILE_SIZE_DECODE = _get_tile_size(\n        head_size, sliding_window_val, q.element_size(), is_prefill=False\n    )"
        injection = (
            inject_after
            + "\n"
            + """
    # --- Strix Halo Patch 19b: gfx11 shared memory (LDS) caps ---
    # RDNA3 LDS = 64 KB. Q-tile = BLOCK_M * head_size * element_size must fit
    # with headroom for K/V tiles and intermediate buffers.
    element_size = q.element_size()
    if on_gfx1x():
        # Cap TILE_SIZE for large head_size to avoid 128 KB LDS requirement
        # (TILE_SIZE=256 needs 128 KB for a single tile).
        if head_size > 128:
            TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, 128)
            TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, 128)

        # Dynamically cap BLOCK_M when Q-tile would exceed shared memory budget.
        # Conservative budget: 16 KB for Q, leaving ~48 KB for K/V tiles +
        # score/accumulator intermediates within the 64 KB LDS limit.
        lds_budget = 65536  # bytes
        q_tile_budget = lds_budget // 4  # 16 KB
        q_tile_bytes = BLOCK_M * head_size * element_size
        if q_tile_bytes > q_tile_budget:
            max_block_m = q_tile_budget // (head_size * element_size)
            BLOCK_M = max(16, triton.next_power_of_2(max_block_m) // 2)
    # --- end Strix Halo Patch 19b ---"""
        )

        if "Strix Halo Patch 19b" not in txt and inject_after in txt:
            txt = txt.replace(inject_after, injection, 1)
            print(" -> Patched triton_unified_attention.py (19b: gfx11 BLOCK_M/TILE_SIZE caps)")

        p_unified.write_text(txt)

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    # --check <vllm_dir>: clone target vLLM and verify patches apply cleanly.
    if "--check" in sys.argv:
        import tempfile
        idx = sys.argv.index("--check")
        vllm_src = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not vllm_src:
            print("Usage: patch_strix.py --check <vllm_source_dir>", file=sys.stderr)
            sys.exit(1)

        with tempfile.TemporaryDirectory() as tmp:
            vllm_dst = os.path.join(tmp, "vllm")
            shutil.copytree(vllm_src, vllm_dst)
            try:
                patch_vllm(vllm_root=vllm_dst)
                print("All patches applied successfully to " + vllm_src)
            except Exception as e:
                print(f"Patch application FAILED on {vllm_src}: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        patch_vllm()
