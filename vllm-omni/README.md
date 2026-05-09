# vLLM-Omni 全模态推理

在 AMD Strix Halo（gfx1151 / RDNA 3.5）上运行 vLLM-Omni 全模态推理服务，支持文本、图像、视频、音频作为输入和输出。

## 概述

vLLM-Omni 是 vLLM 的扩展，增加了 any-to-any 多模态能力。它基于上游 vLLM v0.20.0rc1 构建，在其之上添加了扩散引擎、阶段管道（stage pipeline）和 OmniConnector。

本项目在 vLLM-Omni v0.20.0rc1 上添加了 gfx1151 适配补丁，使其能在 RDNA 3.5 消费级核显上运行。

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
| `VLLM_OMNI_COMMIT` | `v0.20.0rc1` | vllm-omni 版本 |

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
    └── rocm_gfx1151_vllm-omni:v0.20.0rc1 (+ vllm-omni v0.20.0rc1 + gfx1151 patches)
```

## gfx1151 适配补丁

vllm-omni 上游面向 CDNA 数据中心 GPU（MI300/MI325，gfx94x/gfx95x）。vllm-omni 基于已打补丁的 builder 镜像（`rocm_gfx1151_vllm:v0.20.1`）构建，vLLM 级别的 20 个补丁全部自动继承。

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
| 基础 | vLLM v0.20.1 | vllm-omni v0.20.0rc1 (基于 vLLM v0.20.0rc1) |
| 端口 | 8000 (LLM) + 8001 (ASR) + 8003 (TTS) | 8002 |
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
| 推理引擎 | vLLM-Omni | v0.20.0rc1 |
| 基础 vLLM | vLLM | v0.20.1 |
| ROCm SDK | TheRock 7.13 nightly tarball | /opt/rocm |
| PyTorch | torch + triton | 2.10 + 3.6 |
| 注意力 | Triton SDPA (JIT 运行时编译) | — |
| 音频编解码 | onnxruntime-rocm | — |

## Qwen3-TTS 语音合成

vLLM-Omni 支持运行 Qwen3-TTS 系列模型，提供 OpenAI 兼容的 `/v1/audio/speech` API。

### 模型概览

| 模型 | 参数量 | 用途 | 说明 |
|------|--------|------|------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | 预设音色语音合成 | 默认音色（如 "vivian"），支持风格指令 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | 音色设计 | 通过文字描述创建全新音色 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | 声音克隆 | 提供参考音频克隆任意声音 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | 轻量级语音合成 | 更快、更小显存占用 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | 轻量级声音克隆 | 同上 |

- 协议：Apache 2.0
- 支持语言：中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语
- 最低延迟：~97ms
- 支持 3 秒声音克隆

### 启动方式

Qwen3-TTS 需要 `--omni` 标志和专用部署配置，因此建议作为独立容器运行：

```bash
# 下载模型
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' ../vllm/.env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --cache-dir "$VLLM_HOST_MODELS_DIR/hub"

# 在 docker-compose.yml 中添加 vllm-tts 服务后：
podman-compose up -d vllm-tts
```

**podman 直接运行：**

```bash
podman run -d \
  --name vllm-tts \
  --privileged \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --ipc host \
  --shm-size 16gb \
  -p 8003:8000 \
  -v "$VLLM_HOST_MODELS_DIR":/models:ro \
  -v "$PWD/.vllm-cache":/root/.cache/vllm \
  rocm_gfx1151_vllm:v0.20.1 \
  vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 4
```

> gfx1151 固定环境变量（`HSA_OVERRIDE_GFX_VERSION`、`VLLM_ROCM_USE_AITER` 等 21 项）已写入镜像 ENV，无需在运行时重复传递。

| 参数 | 值 | 原因 |
|------|------|------|
| `--deploy-config` | `vllm_omni/deploy/qwen3_tts.yaml` | TTS 专用部署配置（定义多阶段管道） |
| `--omni` | — | 启用 vLLM-Omni 扩展模态支持 |
| `--enforce-eager` | — | gfx1151 上 HIP Graph 会冻结 |
| `--max-model-len` | `4096` | TTS 不需要长上下文 |
| `--gpu-memory-utilization` | `0.7` | TTS 模型较小，不需要高显存利用率 |
| `--max-num-seqs` | `4` | 支持并发合成 |

### API 端点

| 端点 | 用途 |
|------|------|
| `POST /v1/audio/speech` | 语音合成（OpenAI 兼容） |
| `GET /v1/audio/voices` | 列出可用音色 |
| `POST /v1/audio/voices` | 上传音色（声音克隆） |
| `DELETE /v1/audio/voices/{name}` | 删除上传的音色 |
| `POST /v1/audio/speech/batch` | 批量合成（1-32 项） |
| `WS /v1/audio/speech/stream` | WebSocket 流式输入 |

### 快速测试

```bash
# 验证服务就绪
curl http://127.0.0.1:8003/v1/audio/voices

# 基础语音合成
curl -X POST http://127.0.0.1:8003/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "你好，欢迎使用 Qwen3 语音合成。",
        "voice": "vivian",
        "language": "Chinese"
    }' --output output.wav

# 带风格指令的语音合成
curl -X POST http://127.0.0.1:8003/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "太棒了！我们成功了！",
        "voice": "vivian",
        "instructions": "用非常兴奋的语气说"
    }' --output excited.wav

# 批量合成
curl -X POST http://127.0.0.1:8003/v1/audio/speech/batch \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"input": "第一句话。"},
            {"input": "第二句话。"}
        ],
        "voice": "vivian",
        "language": "Chinese"
    }'
```

### 声音克隆（Base 模式）

```bash
curl -X POST http://127.0.0.1:8003/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "你好，这是用我的声音合成的。",
        "task_type": "Base",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "参考音频的原始文字"
    }' --output cloned.wav
```

**上传音色后使用：**

```bash
# 上传参考音频
curl -X POST http://127.0.0.1:8003/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=my_voice" \
  -F "ref_text=参考音频的完整文字"

# 使用上传的音色
curl -X POST http://127.0.0.1:8003/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "你好世界", "voice": "my_voice", "language": "Chinese"}' \
    --output my_voice.wav

# 删除上传的音色
curl -X DELETE http://127.0.0.1:8003/v1/audio/voices/my_voice
```

### Python 客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="none")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="vivian",
    input="你好，欢迎使用 Qwen3 语音合成。",
)
response.stream_to_file("output.wav")
```

### 与 vLLM-Omni 共存

| 服务 | 端口 | 模型 | GPU 占用 |
|------|------|------|----------|
| vllm-omni | 8002 | Qwen3-Omni-MoE-27B | ~60 GiB |
| vllm-tts | 8003 | Qwen3-TTS-1.7B | ~8 GiB |

TTS 模型参数很小（1.7B），与 Omni 服务共存时 128 GB UMA 池完全够用。建议将 TTS 的 `gpu_memory_utilization` 设为 0.7 以留出余量。

## 已知限制

- **AOTriton**：Ubuntu 24.04 自带 CMake 3.28 与 AOTriton 构建系统不兼容，改用 Triton JIT 运行时编译
- **AITER**：CDNA 专属指令在 RDNA 上不存在，完全禁用
- **HIP Graph**：gfx1151 上存在冻结类问题，使用 `--enforce-eager`
- **Flash-Attention（Dao-AILab）**：gfx1151 上编译失败，使用 Triton SDPA
- **性能**：Omni MoE 模型在核显上的性能尚未进行系统级基准测试，预期低于 AWQ4 文本 LLM

## 相关文档

- [../README.md](../README.md) — 项目集总览（硬件、系统配置、Podman 部署）
- [../vllm/README.md](../vllm/README.md) — vLLM 文本大模型子项目
- [../vllm/docs/GUIDE.md](../vllm/docs/GUIDE.md) — 全流程使用指南
- [../vllm/docs/PATCHES.md](../vllm/docs/PATCHES.md) — 20 个 vLLM 补丁逐条分析
