# vLLM-on-Strix-Halo

在 AMD Strix Halo（gfx1151 / RDNA 3.5 核显）上运行 vLLM 推理服务的完整项目集。包含两个子项目：

| 子项目 | 说明 |
|--------|------|
| [vllm/](vllm/) | vLLM 文本大模型推理（Qwen 3.6-27B AWQ-INT4 + DFlash 推测解码） |
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
| ROCm SDK | TheRock ROCm 7.13 nightly | gfx1151 |
| 深度学习框架 | PyTorch 2.10 + Triton 3.6 | ROCm nightly |
| 推理引擎 | vLLM 0.20.0 / vLLM-Omni 0.20.0rc1 | 源码 + 补丁 |
| 注意力优化 | AOTriton（预编译 gfx1151 核） + Triton AMD SDPA | — |
| 量化 | AWQ-INT4 W4A16 g32（compressed-tensors） | — |

### 性能数据

| 场景 | 吞吐量 | 说明 |
|------|--------|------|
| vLLM 单流解码（DFlash N=8） | **24.8 t/s**（峰值） | Qwen 3.6-27B AWQ4，全 256K 上下文 |
| vLLM 无推测基线 | 5.6 t/s | 同上配置，无 DFlash |
| vLLM 3 流并发 | **27-41 t/s** 聚合 | 每流 ~13.5 t/s |
| vLLM 预填充 | **33-38 t/s** 均值 | 包含 prompt-with-tools 场景 |

> +340% 提升来自 DFlash 推测解码（5.6 → 24.8 t/s），在扇冷式核显上实现。

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
│   ├── scripts/           ← vLLM 构建脚本和补丁
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
| vLLM | `ghcr.io/rocm_gfx1151_vllm_v0.20.0:<日期时间>` | 主标签 + `<分支名>` |
| vLLM-Omni | `ghcr.io/rocm_gfx1151_vllm-omni_v0.20.0rc1:<日期时间>` | 主标签 + `<分支名>` |

日期时间格式为 `YYYYMMDDHHmmSS`，例如 `20260502163025`。

### 已知限制

- **Flash-Attention（Dao-AILab）**：在 gfx1151 上导致 ViT 2.2-3.7x 性能退化，未安装
- **AITER 自定义核**：CDNA 专属指令（DPP/向量打包）在 RDNA 上不存在，运行时禁用
- **HIP Graph**：gfx1151 上的冻结类问题，使用 `--enforce-eager`
- **AITER Flash Attention**：仅支持 gfx94x/gfx95x（MI300 系列），gfx1151 使用 Triton 路径
