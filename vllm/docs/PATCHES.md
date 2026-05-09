# 补丁分析报告

vLLM v0.20.1 + gfx1151（Strix Halo / RDNA 3.5）适配补丁的详细分析。

## 补丁总览

| 补丁 | 分类 | 文件 | 行变更 | 效果 |
|------|------|------|--------|------|
| 1 | 硬件使能 | `vllm/platforms/__init__.py` | ~5 行 | 禁用 amdsmi，强制 `is_rocm=True` |
| 2 | 基础设施 | `vllm/platforms/rocm.py` | ~10 行 | 注入 `on_gfx1x()` 辅助函数 |
| 3 | 硬件使能 | `vllm/platforms/rocm.py` | ~3 行 | 强制 device_name="gfx1151" |
| 4 | AITER 兼容 | `vllm/_aiter_ops.py` | ~8 行 | 禁用 FP8/RMSNorm/MoE AITER |
| 5 | AITER 兼容 | `vllm/v1/attention/backends/rocm_aiter_fa.py` | ~2 行 | AITER FA 对 gfx1x 开放 |
| 6 | AITER 兼容 | `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` | ~2 行 | 阻止 AITER MoE 强制覆盖 |
| 7 | AITER 兼容 | `vllm/platforms/rocm.py` | ~5 行 | 绕过 gfx1x 的 custom_ops RMSNorm |
| 8a | AITER 兼容 | `vllm/compilation/passes/fusion/rocm_aiter_fusion.py` | ~1 行 | 修复 aiter fusion 重复模式 |
| 8b | AITER 兼容 | `triton/backends/compiler.py`（site-packages） | ~3 行 | AttrsDescriptor repr |
| 8c | AITER 兼容 | `aiter/jit/__init__.py`（site-packages） | ~8 行 | JIT 缓存路径修复 |
| 9 | AITER 兼容 | `flash_attn/flash_attn_interface.py`（site-packages） | ~6 行 | aiter import 软化 |
| 10 | ROCm 修复 | `gpt_oss_triton_kernels_moe.py`, `mxfp4.py` | ~3 行 | Triton MoE 能力上限 (11→12) |
| 11 | ROCm 修复 | `vllm/platforms/rocm.py` | ~55 行 | APU VRAM 动态余量（ROCM-21812） |
| 12 | 构建修复 | `csrc/cumem_allocator_compat.h` | ~1 行 | 抑制 hipCtx 弃用警告 |
| 13a | API 修复 | `vllm/entrypoints/openai/responses/protocol.py` | ~1 行 | ResponsesRequest 添加 chat_template_kwargs |
| 13b | API 修复 | `vllm/entrypoints/openai/responses/protocol.py` | ~1 行 | to_chat_params 合并用户 kwargs |
| 14 | 特性 | `vllm/model_executor/kernels/linear/__init__.py` | ~15 行 | AWQ-INT4 MMQ HIP 核注册 |
| 15 | ROCm 修复 | `csrc/quantization/gptq/compat.cuh` | ~1 行 | 移除 atomicAdd half polyfills |
| 16a | 性能优化 | `vllm/v1/worker/gpu_worker.py` | ~14 行 | profile 缓存读取（跳过 ~7 min） |
| 16b | 性能优化 | `vllm/v1/worker/gpu_worker.py` | ~14 行 | profile 缓存写入 |
| 17 | Bug 修复 | `vllm/model_executor/models/qwen3_dflash.py` | ~4 行 | combine_hidden_states dtype 修复 |
| 17b | Bug 修复 | `vllm/entrypoints/openai/responses/serving.py` | ~40 行 | enable_thinking=false 非流式修复 |
| 18 | 性能优化 | `vllm/v1/attention/backends/triton_attn.py` | ~4 行 | softmax segments 16→32 |
| 19a | 性能优化 | `vllm/v1/attention/ops/triton_unified_attention.py` | ~1 行 | on_gfx1x 导入 |
| 19b | 性能优化 | `vllm/v1/attention/ops/triton_unified_attention.py` | ~15 行 | BLOCK_M/TILE_SIZE LDS 上限 |
| 20 | 构建修复 | `CMakeLists.txt` | ~12 行 | 显式检测并启用 HIP 语言 |

## 分类统计

| 分类 | 补丁数 | 变更行数 | 运行时影响 |
|------|--------|----------|-----------|
| 硬件使能 | 2（1,3） | ~8 | 启动必需 |
| 基础设施 | 1（2） | ~10 | 下游依赖 |
| AITER 兼容 | 7（4-9） | ~33 | 禁用 CDNA 特性 |
| ROCm 修复 | 3（10-12） | ~59 | 修复 SDK Bug |
| 构建修复 | 2（12,20） | ~13 | 修复 SDK Bug + CMake HIP 检测 |
| API 修复 | 3（13a,13b,17b） | ~42 | 修复 /v1/responses |
| 特性 | 2（14,17） | ~19 | AWQ MMQ + dtype |
| 性能优化 | 4（16a,16b,18,19） | ~47 | 重启加速 + 注意力调优 |
| **合计** | **23 操作 / 20 补丁** | **~230 行** | — |

## 逐补丁详细分析

### 硬件使能层（Patch 1-3）

#### Patch 1: amdsmi 禁用

- **修改前**：`import amdsmi` → 容器内 APU 上不可用，启动即崩溃
- **修改后**：注释导入 + `pass` 占位 + `is_rocm = True` 强制
- **预期效果**：vLLM 成功启动，平台检测正确识别为 ROCm
- **可移除性**：当 amdsmi 在 APU 容器内可用时移除

#### Patch 2: on_gfx1x() 辅助函数

- **修改前**：`vllm/platforms/rocm.py` 无 gfx1x 检测方法
- **修改后**：注入 `on_gfx1x() → bool`，检查 `device_name.startswith('gfx115')`
- **预期效果**：下游补丁（4-7, 18, 19）共享此检测函数
- **上游状态**：仅在 ROCm/vllm gfx11 分支存在，上游未合并

#### Patch 3: 强制 gfx1151 架构

- **修改前**：device_name 依赖 amdsmi 或系统检测（APU 上失败）
- **修改后**：`device_name = "gfx1151"`，`device_type = "rocm"`
- **预期效果**：Triton JIT 编译器能正确选择 gfx1151 内核变体
- **v0.20.1 修复**：正则表达式增加类型注解兼容（`device_name: str = ...`）

### AITER 兼容层（Patch 4-9）

AITER 使用 CDNA 专属指令（DPP 数据并行原语、向量打包指令），这些在 RDNA 3.5 上**不存在**。这组补丁确保 vLLM 在 AITER 存在但不完全兼容的情况下稳定运行。

#### Patch 4: AITER 操作门控

- **修改前**：`is_aiter_found_and_supported()` 在 gfx1151 上返回 False
- **修改后**：扩展为 `(on_mi3xx() or on_gfx1x())`，但禁用 FP8 linear / RMSNorm / MoE
- **预期效果**：AITER 基础路径可用，CDNA 专属特性安全绕过

#### Patch 5-6: AITER FA 和 MoE

- **Patch 5**：AITER Flash Attention 后端对 gfx1x 开放注册
- **Patch 6**：在 unquantized.py 中硬阻止 AITER MoE 强制覆盖（防止绕过 Patch 4）

#### Patch 7: custom_ops RMSNorm 绕过

- **修改前**：`compilation_config.custom_ops.append("+rms_norm")` 在 gfx1x 上导致 CUDA Graph hang
- **修改后**：添加 `on_gfx1x()` 守卫
- **预期效果**：CUDA Graph 捕获完整，推理时不卡死

#### Patch 8: AITER JIT 修复（3 个子补丁）

- **8a**：fusion 注册添加 `skip_duplicates=True`，防止重复模式崩溃
- **8b**：AttrsDescriptor 添加 `__repr__`，JIT 缓存序列化正常
- **8c**：JIT 编译的 .so 文件在 `~/.aiter/jit/`，添加 `__path__` 使其可导入

#### Patch 9: flash_attn 软导入

- **修改前**：`from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_gpu` 硬导入
- **修改后**：try/except 包裹，失败时 `flash_attn_gpu = None`
- **预期效果**：即使 aiter JIT 失败，TRITON_ATTN 路径仍可工作

### ROCm SDK 修复（Patch 10-12）

#### Patch 10: Triton MoE 能力上限

- **修改前**：`device_capability < (11, 0)` 排除了 gfx1151（cap = 11.5）
- **修改后**：`(11, 0)` → `(12, 0)`，允许 gfx11xx 使用 Triton MoE
- **影响文件**：`gpt_oss_triton_kernels_moe.py` + `mxfp4.py`
- **v0.20.1 修复**：新增 `device_capability < (11, 0)` 变量模式匹配

#### Patch 11: APU VRAM 动态余量（ROCM-21812）

- **修改前**：ROCm 在 APU 上将 VRAM clamp 到 50%，27B 模型无法加载
- **修改后**：Mock `torch.cuda.mem_get_info` 和 `get_device_properties`，从 GTT 读取真实值，保留 8GB OS 安全余量
- **预期效果**：可用 VRAM 从 ~58GB 提升到 ~108GB（128GB UMA - 8GB - 系统占用）
- **可移除性**：当 ROCm PR #5113 合入 nightly tarball 后移除

#### Patch 12: hipCtx 弃用警告

- **修改前**：clang 编译时产生大量 deprecation warnings
- **修改后**：添加 `#pragma clang diagnostic ignored`
- **预期效果**：构建日志清爽，不影响功能

### API 修复（Patch 13, 17b）

#### Patch 13a/13b: chat_template_kwargs 传入 /v1/responses

- **修改前**：`ResponsesRequest.to_chat_params()` 使用硬编码 dict，忽略请求体中的 chat_template_kwargs
- **修改后**：
  - 13a：`ResponsesRequest` 添加 `chat_template_kwargs` 字段
  - 13b：`to_chat_params()` 通过 `merge_kwargs()` 合并用户 kwargs
- **预期效果**：`enable_thinking` 等参数在 `/v1/responses` 路径生效
- **上游状态**：通用 vLLM bug，非 gfx1151 专属，可贡献上游

#### Patch 17b: enable_thinking=false 非流式修复

- **修改前**：非流式 `/v1/responses` 中，`enable_thinking=false` 时 reasoning parser 错误处理预填充的 `<|assistant|>
</think>

` 标记
- **修改后**：添加 `is_reasoning_end` 安全检查，检测 reasoning 已在 prompt 中结束时跳过 parser
- **预期效果**：非流式模式下 `enable_thinking=false` 返回完整输出而非截断

### 特性（Patch 14, 17）

#### Patch 14: AWQ-INT4 MMQ HIP 核注册

- **修改前**：gfx1151 上 AWQ-INT4 使用 TritonW4A16 路径（性能较差）
- **修改后**：注册 `RocmMmqQ4LinearKernel` 到 `_POSSIBLE_KERNELS[ROCM]` 首位
- **预期效果**：预填充阶段（M ≥ 32）使用 INT8 WMMA HIP 核，从 ~38 t/s 提升到 ~130 t/s
- **注意**：本项目已移除 DFlash 推测解码，此补丁保留用于 AWQ 预填充加速
- **注意**：AWQ MMQ HIP 核的 .so 需要单独构建（csrc/awq_mmq_gfx1151/）

#### Patch 17: combine_hidden_states dtype 修复（PR #40334）

- **修改前**：AWQ 量化模型的未量化注意力层输出 float32，传入 draft head 的 fc 层（期望 bfloat16）→ `RuntimeError: expected scalar type Float but found Half`
- **修改后**：在 `self.model.fc()` 前添加 dtype 转换
- **预期效果**：混合精度模型生成不再崩溃
- **上游状态**：PR #40334 仍在 OPEN 状态

### 性能优化（Patch 16, 18, 19）

#### Patch 16a/16b: Profile 缓存

- **修改前**：每次 vLLM 重启都运行 ~7 分钟的 synthetic forward passes 来确定 KV cache 大小
- **修改后**：
  - 16a：启动时检查缓存，命中则直接返回
  - 16b：正常 profiling 后将结果写入缓存
- **预期效果**：首次启动 ~7 min → 后续重启 < 10 s
- **v0.20.1 修复**：Patch 16b 的幂等守卫从 `"Strix Halo Patch 16"` 改为 `"write_cached_kv_cache_memory_bytes"`，避免与 16a 注入的字符串冲突
- **控制**：`VLLM_SKIP_MEMORY_PROFILING=1` 开启，`VLLM_PROFILE_CACHE_DIR` 指定路径

#### Patch 18: Softmax segments调优

- **修改前**：`num_par_softmax_segments = 16`（固定值，后改为常量 `NUM_PAR_SOFTMAX_SEGMENTS = 16`）
- **修改后**：gfx1151 + MQA/large heads（≥ 224）时提升到 32
- **预期效果**：Triton 注意力瓦片化软计算的并行度提高，MQA 和大 head size 场景下 measurable 增益
- **v0.20.1 修复**：新增对 `NUM_PAR_SOFTMAX_SEGMENTS` 常量形式的匹配（上游已将字面 `16` 替换为常量）

#### Patch 19a/19b: Triton SDPA 共享内存上限

- **修改前**：Triton 自动调优器可能选择 BLOCK_M=128 + head_size=256 → Q-tile 需要 128KB LDS，超过 RDNA3 的 64KB/CU
- **修改后**：
  - 大 head_size（> 128）时：TILE_SIZE 上限 128
  - Q-tile 超出 16KB 预算时：动态降低 BLOCK_M
- **预期效果**：避免 `OutOfResources` 编译错误，大模型/大 head size 场景稳定运行
- **来源**：ROCm/vllm gfx11 PR #919, #911

#### Patch 20: 显式检测并启用 HIP 语言

- **修改前**：vLLM CMakeLists.txt 依赖 PyTorch `find_package(Torch)` 设置 `HIP_FOUND`，但 CMake 3.31+ 下 PyTorch 的 TorchConfig.cmake 内部启用 HIP 却不导出该变量，导致 "Can't find CUDA or HIP installation" 致命错误
- **修改后**：在 `find_package(Torch REQUIRED)` 后插入 `check_language(HIP)` + `enable_language(HIP)` + `set(HIP_FOUND TRUE)`
- **预期效果**：vLLM 源码构建不再因 CMake 版本不兼容而失败
- **触发条件**：仅当 `CUDA_FOUND` 为假时执行（即 ROCm 路径）
- **可移除性**：当上游 vLLM 的 CMakeLists.txt 自行处理此检测时移除

## 补丁与上游 vLLM 的兼容性

| 补丁 | v0.20.0 | v0.20.1 | 上游 PR | 状态 |
|------|---------|---------|---------|------|
| 1 | ✅ | ✅ | — | 本地必需 |
| 2 | ✅ | ✅ | ROCm/vllm gfx11 | 本地必需 |
| 3 | ✅ | ✅（已修复正则） | — | 本地必需 |
| 4 | ✅ | ✅ | — | 本地必需 |
| 5 | ✅ | ✅ | — | 本地必需 |
| 6 | ✅ | ✅ | — | 本地必需 |
| 7 | ✅ | ✅ | — | 本地必需 |
| 8 | ✅ | ✅ | — | 本地必需 |
| 9 | ✅ | ✅ | — | 安全网 |
| 10 | ✅ | ✅（已修复变量模式） | — | 本地必需 |
| 11 | ✅ | ✅ | ROCm #5113 | 可移除 |
| 12 | ✅ | ✅ | — | 构建清理 |
| 13 | ✅ | ✅ | 候选上游 | 通用修复 |
| 14 | ✅ | ✅ | — | 本地特性 |
| 15 | ✅ | ✅ | — | ROCm 兼容 |
| 16 | ✅ | ✅（已修复 16b 守卫） | — | 本地优化 |
| 17 | ✅ | ✅ | PR #40334 | 候选上游 |
| 18 | ✅ | ✅（已修复常量匹配） | ROCm/vllm gfx11 | 性能调优 |
| 19 | ✅ | ✅ | ROCm/vllm #919 | 稳定性 |
| 20 | ✅ | ✅ | — | 本地必需 |

**所有 20 个补丁（23 个操作）在 vLLM v0.20.1 上验证通过。**

## vllm-omni 补丁继承

vllm-omni 基于 `rocm_gfx1151_vllm:v0.20.1` builder 镜像构建，自动继承所有 20 个 vLLM 级别补丁。

vllm-omni 自身的 `patch_omni.py` 仅执行 Patch 2（onnxruntime 验证），无需额外 vLLM 级补丁。

扩散注意力子系统使用 `TORCH_SDPA`（gfx1151 fallback），无需 Triton 路径补丁。

## 补丁可移除性评估

| 补丁 | 可移除条件 | 优先级 |
|------|-----------|--------|
| 1-3 | gfx1151 获上游官方支持 | 低（消费卡可能长期需要） |
| 4-9 | AITER 修复 RDNA 3.5 指令兼容 | 中（AITER PR #1498 仅注册架构 ID） |
| 10 | vLLM 修复 Triton MoE 能力检查 | 高（明显是 Bug） |
| 11 | ROCm PR #5113 合入 | 高（已在跟踪中） |
| 12 | vLLM 迁移到非弃用 HIP API | 低（纯构建警告） |
| 13, 17b | 上游 /v1/responses 修复 | 中（通用 Bug） |
| 17 | PR #40334 合并 | 高（已在 OPEN 状态） |
| 16 | vLLM 原生支持 profile 缓存 | 低（本地优化） |
| 18, 19 | 上游 Triton 注意力优化 | 低（gfx1151 特定调优） |
