# vLLM-on-Strix-Halo

在 AMD Strix Halo（gfx1151 / RDNA 3.5 核显）上运行 vLLM 推理服务的完整项目集。包含两个子项目：

| 子项目 | 说明 |
|--------|------|
| [vllm/](vllm/) | vLLM 文本大模型推理（Qwen 3.6-27B AWQ-INT4） |
| [vllm-omni/](vllm-omni/) | vLLM-Omni 全模态推理（文本/图像/视频/音频输入输出） |

---

## 硬件平台

### AMD Ryzen AI MAX+ 395（Strix Halo）

| 规格 | 值 |
|------|-----|
| **架构** | Strix Halo (RDNA 3.5 iGPU + Zen 5 CPU) |
| **CPU** | 16 核 32 线程 Zen 5，基础 3.0 GHz，加速 5.1 GHz |
| **L3 缓存** | 64 MB |
| **GPU 型号** | Radeon 8060S（gfx1151 / RDNA 3.5） |
| **GPU 计算单元** | 40 CU（20 WGP） |
| **GPU 频率** | 最高 2.9 GHz |
| **FP16/BF16 算力** | ~59.4 TFLOPS |
| **内存** | LPDDR5x-8000，8 通道，最高 128 GB UMA |
| **理论带宽** | 256 GB/s |
| **实测带宽** | ~215 GB/s |
| **NPU** | XDNA 2，50+ TOPS |
| **制程** | TSMC 4nm |

Strix Halo 是一颗 chiplet APU：CPU 和 GPU 通过 Infinity Fabric 互联，共享统一内存池。核显拥有 40 个计算单元，规模是 Strix Point（16 CU）的 2.5 倍，接近入门级独显的性能。统一内存架构使得 128 GB LPDDR5x 可全部被 GPU 访问，这是在此硬件上运行 27B 级大模型的前提条件。

---

## 物理机操作系统配置

### 宿主机环境

| 项目 | 值 |
|------|-----|
| **操作系统** | Ubuntu 26.04 LTS (Resolute Raccoon) |
| **内核** | Linux 7.0.0-15-generic |
| **容器运行时** | Podman |
| **架构** | x86_64 |

### 关键内核参数

```
# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1"
```

- `ttm.pages_limit=30408704`：将 GPU 可映射的 GTT 页限制提高至约 116 GiB，使核显能通过 GTT 按需分配大内存
- `amdgpu.noretry=0`：允许 GPU 页错误重试（AWQ 模型加载需要）
- `amdgpu.gpu_recovery=1`：启用 GPU 错误自动恢复

### BIOS 设置

将 UEFI 中的 **UMA Frame Buffer Size**（或 iGPU Memory / GPU Shared Memory）设为最小值 **2 GB**。Strix Halo 的 GTT 按需分配机制意味着不需要预分配大段显存，设为最小值反而能获得最大的可用内存。

### Podman 部署

本项目使用 Podman 作为容器运行时（替代 Docker），Podman 兼容 docker-compose 语法。所有服务通过 `podman-compose` 编排：

```bash
# 构建并启动 vLLM 服务
podman-compose build vllm
podman-compose up -d vllm

# 构建并启动 vLLM-Omni 服务
podman-compose build vllm-omni
podman-compose up -d vllm-omni
```

Podman 以 rootless 模式运行，容器通过 `--privileged` + `/dev/kfd:/dev/kfd` + `/dev/dri:/dev/dri` 直接访问 GPU 设备节点。

---

## 项目概览

本项目集的目标是在 AMD Strix Halo 消费级 APU 上运行大语言模型和全模态模型，填补 ROCm 对 RDNA 3.5 消费卡支持的空白。所有组件均从源码构建，并带有针对 gfx1151 硬件特性的补丁。

### 技术栈

| 层 | 组件 | 版本 |
|---|---|---|
| 操作系统 | Ubuntu 26.04 LTS | — |
| 容器运行时 | Podman | — |
| ROCm SDK | pip rocm[devel,libraries] 7.13 nightly | gfx1151 (site-packages) |
| 深度学习框架 | PyTorch 2.10 + Triton 3.6 | ROCm nightly |
| 推理引擎 | vLLM 0.20.1 / vLLM-Omni 0.20.0rc1 | 源码 + 补丁 |
| 注意力 | Triton AMD SDPA (JIT 运行时编译) | — |
| 量化 | AWQ-INT4 W4A16 g32（compressed-tensors） | — |

### 性能数据

| 场景 | 吞吐量 | 说明 |
|------|--------|------|
| vLLM 单流解码（基线） | ~5.6 t/s | Qwen 3.6-27B AWQ4，全 256K 上下文 |
| vLLM 预填充 | **33-38 t/s** 均值 | 包含 prompt-with-tools 场景 |

### 目录结构

```
.
├── README.md              ← 本文件：硬件、系统、项目概览
├── vllm/
│   ├── README.md          ← vLLM 子项目文档
│   ├── Dockerfile         ← vLLM 镜像构建
│   ├── docker-compose.yml ← vLLM 服务编排
│   ├── .env.template      ← vLLM 环境变量模板
│   ├── glados.py          ← CLI 客户端
│   ├── docs/              ← vLLM 文档（LLM/ASR/使用指南）
│   ├── scripts/           ← vLLM 构建脚本和补丁 (patch_strix.py + vllm_profile_cache.py)
│   └── test/              ← vLLM 测试和基准
└── vllm-omni/
    ├── README.md          ← vLLM-Omni 子项目文档
    ├── Dockerfile         ← vLLM-Omni 镜像构建（基于 vLLM builder）
    └── scripts/           ← vLLM-Omni gfx1151 适配补丁
```

### GitHub Actions 自动构建

推送 main 分支或打 tag 后，GitHub Actions 自动构建 Docker 镜像并推送到 GHCR：

| 镜像 | GHCR 地址 | 说明 |
|------|-----------|------|
| vLLM | `ghcr.io/<owner>/<repo>/rocm_gfx1151_vllm_v0.20.1:<日期时间>` | 主标签 + `<分支名>` |
| vLLM-Omni | `ghcr.io/<owner>/<repo>/rocm_gfx1151_vllm-omni_v0.20.0rc1:<日期时间>` | 主标签 + `<分支名>` |

日期时间格式为 `YYYYMMDDHHmmSS`，例如 `20260502163025`。

### 已知限制

- **AOTriton**：Ubuntu 26.04 自带 CMake 4.2，AOTriton 构建不兼容，改用 Triton JIT 运行时编译
- **Flash-Attention（Dao-AILab）**：在 gfx1151 上导致 ViT 2.2-3.7x 性能退化，未安装
- **AITER 自定义核**：CDNA 专属指令（DPP/向量打包）在 RDNA 上不存在，运行时禁用
- **HIP Graph**：gfx1151 上的冻结类问题，使用 `--enforce-eager`
- **AITER Flash Attention**：仅支持 gfx94x/gfx95x（MI300 系列），gfx1151 使用 Triton 路径

---

## 相关项目比较

gfx1151（RDNA 3.5）在 vLLM 生态中仍属"非官方支持"架构。本项目基于对以下五个项目的调研和实践：

| 项目 | 定位 | vLLM 版本 | 关键特性 |
|------|------|-----------|----------|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 上游官方 | 主干/发布标签 | 基础推理引擎，gfx1151 未列入官方支持列表 |
| [ROCm/vllm gfx11](https://github.com/ROCm/vllm/tree/gfx11) | AMD ROCm 官方分支 | 主干（跟踪上游） | 自定义 HIP 内核、AWQ MoE、gfx1151 CI |
| [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) | Fedora Toolbx 容器 | HEAD + 补丁 | 多节点 RDMA 集群、bitsandbytes、TUI 向导 |
| [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen) | AWQ-INT4 + DFlash | v0.20.0 + 补丁 | DFlash 推测解码、AWQ MMQ HIP 预填充核 |
| **本项目（gfx1151-tools）** | Ubuntu 生产部署 | **v0.20.1 + 补丁** | 多服务编排（LLM+ASR+Omni）、pip ROCm SDK |

### 1. 上游 vLLM（vllm-project/vllm）

vLLM 是本项目的基础推理引擎。截至 v0.20.1，**gfx1151 仍未列入官方支持的 GPU 列表**。上游代码中已有部分 RDNA 3.5 设备 ID 检测和 Triton 注意力修复，但不足以直接运行：

- **设备检测**：v0.20.0 起包含 gfx1150/1151/1201 设备 ID 识别（来自 ROCm/vllm 的合并）
- **AWQ 路由**：PR #36505 使 AWQ 通过 AWQMarlinLinearMethod → ConchLinearKernel 在 ROCm 上生效
- **缺失**：`amdsmi` 在 APU 上不可用、ROCm 平台 VRAM 检测 bug、AITer CDNA 指令不兼容、Triton MoE 设备能力检查等问题仍需自行修复

**结论**：上游提供了基础框架，但直接安装 `pip install vllm` 无法在 Strix Halo 上运行，需要额外补丁。

### 2. ROCm/vllm gfx11 分支

AMD ROCm 团队维护的 vLLM 功能分支，是 gfx1151 支持的最上游来源：

- **自定义内核**：`csrc/rocm/` 下的 HIP/C++ 内核（注意力、skinny GEMM、AWQ GEMV）
- **量化支持**：AWQ GEMV HIP 内核、W4A16 MoE 混合内核、W8A8 skinny GEMM
- **gfx1151 专项优化**：2026 年 5 月提交了 bf16 dequant DOT2C 优化、Triton 共享内存溢出修复
- **AITer 集成**：RMSNorm、MoE、FP8 GEMM、paged attention 等大量 ROCm 特有环境变量
- **CI**：有真实的 Strix Halo 硬件 CI runner

**与本项目关系**：本项目的 19 个补丁中有 12 个来自 kyuz0 的 toolboxes 项目，而 kyuz0 的工作又基于 ROCm/vllm 分支的功能。ROCm/vllm 是底层内核的源头，但本项目的补丁更聚焦于让生产环境（Docker + 多服务）稳定运行，而非提供完整的内核实现。

### 3. kyuz0/amd-strix-halo-vllm-toolboxes

kyuz0（Donato Capitella）的 Fedora Toolbx 容器项目，是最早的 Strix Halo vLLM 实践之一：

- **容器方案**：Fedora 43 + TheRock ROCm tarball + Toolbx 容器
- **多节点集群**：通过 RDMA/RoCE v2（Intel E810 网卡）或 Thunderbolt 4 实现跨节点张量并行
- **补丁覆盖**：amdsmi 禁用、gfx1151 强制检测、AITer CDNA 指令替换、VRAM 动态余量补丁
- **量化**：支持 BF16 和 AWQ（通过 bitsandbytes ROCm fork）
- **测试模型**：Llama 3.1 8B、Gemma 4、GPT-OSS、Qwen 3.5/3.6

**与本项目差异**：
| 维度 | kyuz0 toolboxes | 本项目 |
|------|-----------------|--------|
| OS | Fedora 43 + TheRock tarball | Ubuntu 26.04 + pip ROCm SDK |
| 容器 | Toolbx（交互式桌面环境） | Podman（headless 服务） |
| 量化 | BF16 + AWQ（bitsandbytes） | AWQ-INT4（compressed-tensors） |
| 网络 | RDMA/RoCE 多节点集群 | 单节点 |
| 服务 | 单 vLLM 实例 | LLM + ASR + Omni 三服务 |
| 补丁数量 | 10+（含 AITER 头文件替换） | 19（含 profile 缓存、API 修复） |

**结论**：kyuz0 的项目适合需要多节点集群或交互式桌面的场景。本项目面向单节点、多服务、headless 生产的场景。

### 4. hec-ovi/vllm-awq4-qwen

hec-ovi 的 AWQ-INT4 + DFlash 项目，是本项目最初的基础参考：

- **DFlash 推测解码**：在 Strix Halo 上实现最高 24.8 t/s（N=8，51-67% 接受率）
- **AWQ MMQ HIP 内核**：gfx1151 专属 INT8 WMMA 预填充核，4K 上下文预填充从 ~38 t/s 提升到 ~130 t/s
- **18 个补丁**：Patches 1-12 来自 kyuz0，Patches 13-18 为 DFlash、thinking 修复、atomicAdd polyfills 等

**与本项目差异**：
| 维度 | hec-ovi | 本项目 |
|------|---------|--------|
| vLLM 版本 | v0.20.0 | v0.20.1 |
| 推测解码 | DFlash（最高 24.8 t/s） | 已移除（见下方说明） |
| AWQ MMQ | INT8 WMMA 预填充核 | 已移除 |
| 服务编排 | 单 vLLM 服务 | LLM + ASR + Omni 三服务 |
| OS | Ubuntu 26.04 | Ubuntu 26.04 |
| ROCm SDK | pip 7.13 nightly | pip 7.13 nightly |
| 文档 | 英文 | 中文 |

**DFlash 移除说明**：本项目移除了 DFlash 相关内容，原因包括：(1) DFlash 推测解码在 gfx1151 上仍属早期上游功能，5+ 个 bug 修复 PR 未合并；(2) 非流式工具调用和流式推理的稳定性问题；(3) DFlash 接受率在 N>8 时急剧下降，实际生产收益不稳定。

### 5. 本项目的独特定位

在上述四个参考项目的基础上，本项目的差异化定位是：

- **面向生产部署**：headless Podman 服务、docker-compose 编排、GHCR 自动构建
- **多服务共存**：LLM（8000）+ ASR（8001）+ Omni（8002）三服务独立运行，各自优化
- **pip ROCm SDK**：采用 `uv pip install rocm[devel,libraries]` 替代 TheRock tarball，更轻量、更可复现
- **中文文档**：全流程中文文档，包含硬件配置、部署指南、故障排查
- **AWQ-INT4 专注**：不做 DFlash 推测解码，专注 compressed-tensors AWQ-INT4 路径的稳定性和性能
- **vLLM v0.20.1**：跟踪最新上游补丁版本，减少补丁数量

### 参考链接

- [vLLM 上游仓库](https://github.com/vllm-project/vllm)
- [ROCm/vllm gfx11 分支](https://github.com/ROCm/vllm/tree/gfx11)
- [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes)
- [kyuz0 的 Strix Halo vLLM 基准测试](https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/)
- [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen)
- [vLLM Issue #16621 — Strix Halo 支持请求](https://github.com/vllm-project/vllm/issues/16621)
