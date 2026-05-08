# vLLM-Omni 全模态推理

在 AMD Strix Halo（gfx1151 / RDNA 3.5）上运行 vLLM-Omni 全模态推理服务，支持文本、图像、视频、音频作为输入和输出。

## 概述

vLLM-Omni 是 vLLM 的扩展，增加了 any-to-any 多模态能力。它基于上游 vLLM v0.20.0 构建，在其之上添加了扩散引擎、阶段管道（stage pipeline）和 OmniConnector。

本项目在 vLLM-Omni v0.20.0 上添加了 gfx1151 适配补丁，使其能在 RDNA 3.5 消费级核显上运行。

## 模型

| 项目 | 值 |
|------|-----|
| 目标模型 | `Qwen/Qwen3-Omni-MoE-27B` |
| 输入模态 | 文本、图像、视频、音频 |
| 输出模态 | 文本、音频 |
| 架构 | MoE（混合专家），27B 总参数 |
| 磁盘占用 | ~60 GB（需预下载） |
| HuggingFace | [Qwen/Qwen3-Omni-MoE-27B](https://huggingface.co/Qwen/Qwen3-Omni-MoE-27B) |

## 快速开始

### 前置准备

1. 完成 vLLM 子项目的构建（vllm-omni 基于 builder 镜像）
2. 下载模型：

```bash
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' ../vllm/.env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-Omni-MoE-27B --cache-dir "$VLLM_HOST_MODELS_DIR/hub"
```

### 构建与启动

```bash
# 1. 在项目根目录中：

# 2. 构建 vllm-omni 镜像（依赖 rocm_gfx1151_vllm:v0.20.1）
podman-compose build vllm-omni

# 3. 启动服务
podman-compose up -d vllm-omni

# 4. 验证
curl http://127.0.0.1:8002/v1/models
```

### 环境变量

在 `vllm/.env` 中配置以下变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_OMNI_MODEL_ID` | `Qwen/Qwen3-Omni-MoE-27B` | 模型 ID 或本地路径 |
| `VLLM_OMNI_SERVED_MODEL_NAME` | `Qwen3-Omni-MoE-27B` | API 中显示的模型名 |
| `VLLM_OMNI_HOST_PORT` | `8002` | 服务端口 |
| `VLLM_OMNI_GPU_MEMORY_UTIL` | `0.9` | GPU 显存利用率 |
| `VLLM_OMNI_MAX_MODEL_LEN` | `8192` | 最大上下文长度 |
| `VLLM_OMNI_MAX_NUM_SEQS` | `1` | 最大并发流数 |
| `VLLM_OMNI_COMMIT` | `v0.20.0` | vllm-omni 版本 |

> **注意**：`VLLM_OMNI_COMMIT` 变更后需要重新构建镜像。

## 架构

```
┌──────────────────────────────────────────────────┐
│  docker-compose.yml                              │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  vllm-omni (8002)                        │    │
│  │  Qwen3-Omni-MoE-27B                      │    │
│  │  文本/图像/视频/音频 → 文本/音频          │    │
│  │  any-to-any 全模态                        │    │
│  └──────────────────────────────────────────┘    │
│                    │                              │
│              128 GB UMA                           │
│         AMD Strix Halo (gfx1151)                  │
└──────────────────────────────────────────────────┘
```

构建依赖链：

```
rocm_gfx1151_vllm:v0.20.1 (vLLM v0.20.1 + PyTorch + pip ROCm SDK)
    └── rocm_gfx1151_vllm-omni:v0.20.0 (+ vllm-omni v0.20.0 + gfx1151 patches)
```

## gfx1151 适配补丁

vllm-omni 上游面向 CDNA 数据中心 GPU（MI300/MI325，gfx94x/gfx95x）。vllm-omni 基于已打补丁的 builder 镜像（`rocm_gfx1151_vllm:v0.20.1`）构建，vLLM 级别的 19 个补丁全部自动继承。

扩散注意力子系统（`vllm_omni/diffusion/attention/`）是 vllm-omni 的独立实现，不暴露 `TRITON_ATTN` 后端（仅 `FLASH_ATTN` / `TORCH_SDPA` / `SAGE_ATTN`）。gfx1151 上扩散阶段 fallback 到 `TORCH_SDPA`，无需额外补丁。

`scripts/patch_omni.py` 仅在构建时做运行时验证检查：

| 补丁 | 内容 |
|------|------|
| Patch 2 | 确认 onnxruntime-rocm 已正确安装（无 vanilla onnxruntime 冲突） |

### 为什么不用 AITER？

AITER 使用了 CDNA 专属指令（DPP 数据并行原语、`v_pk_mul_f32`/`v_cvt_pk_fp8_f32` 等向量打包指令），这些指令在 RDNA 3.5 上**不存在**。AITer PR #1498 仅注册了 gfx11XX 架构 ID，但未修复底层指令不兼容问题，37/48 测试仍然因非法指令失败。因此必须禁用 AITER（`VLLM_ROCM_USE_AITER=0`）。

### 注意力后端选择

| 组件 | 后端 | 原因 |
|------|------|------|
| 扩散注意力 | `TORCH_SDPA` | 无 TRITON_ATTN 后端，fallback 到 PyTorch SDPA |
| 视觉编码器 | `TRITON_ATTN` | vllm-omni 默认行为，与 vLLM 一致 |

## API 端点

| 端点 | 用途 |
|------|------|
| `POST /v1/chat/completions` | 多模态聊天（文本 + 图像/视频/音频） |
| `POST /v1/completions` | 原始文本补全 |
| `GET /v1/models` | 列出可用模型 |

> vllm-omni 的 API 与 vLLM 兼容，扩展了多模态输入支持。

## 与 vLLM 子项目的关系

| 维度 | vLLM 子项目 | vLLM-Omni 子项目 |
|------|-------------|-------------------|
| 基础 | vLLM v0.20.1 | vllm-omni v0.20.0 (基于 vLLM v0.20.0) |
| 端口 | 8000 (LLM) + 8001 (ASR) | 8002 |
| 模型 | Qwen3.6-27B-AWQ4 | Qwen3-Omni-MoE-27B |
| 模态 | 文本 + 视觉（输入） | 文本/图像/视频/音频（输入/输出） |
| 构建依赖 | 独立 | 依赖 rocm_gfx1151_vllm:v0.20.1 |

两个子项目共享同一个 builder 镜像，omni 在其之上叠加安装，互不干扰。

## 目录结构

```
vllm-omni/
├── README.md              ← 本文件
├── Dockerfile             ← vLLM-Omni 层（基于 builder 镜像）
└── scripts/
    └── patch_omni.py      ← gfx1151 适配补丁
```

## 技术栈

| 层 | 组件 | 版本 |
|---|------|------|
| 推理引擎 | vLLM-Omni | v0.20.0 |
| 基础 vLLM | vLLM | v0.20.1 |
| ROCm SDK | TheRock 7.13 nightly tarball | /opt/rocm |
| PyTorch | torch + triton | 2.10 + 3.6 |
| 注意力 | Triton SDPA (JIT 运行时编译) | — |
| 音频编解码 | onnxruntime-rocm | — |

## 已知限制

- **AOTriton**：Ubuntu 26.04 自带 CMake 4.2 与 AOTriton 构建系统不兼容，改用 Triton JIT 运行时编译
- **AITER**：CDNA 专属指令在 RDNA 上不存在，完全禁用
- **HIP Graph**：gfx1151 上存在冻结类问题，使用 `--enforce-eager`
- **Flash-Attention（Dao-AILab）**：gfx1151 上编译失败，使用 Triton SDPA
- **性能**：Omni MoE 模型在核显上的性能尚未进行系统级基准测试，预期低于 AWQ4 文本 LLM

## 相关文档

- [../README.md](../README.md) — 项目集总览（硬件、系统配置、Podman 部署）
- [../vllm/README.md](../vllm/README.md) — vLLM 文本大模型子项目
- [../vllm/docs/GUIDE.md](../vllm/docs/GUIDE.md) — 全流程使用指南
- [../vllm/docs/PATCHES.md](../vllm/docs/PATCHES.md) — 19 个 vLLM 补丁逐条分析
