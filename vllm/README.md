# vLLM 文本大模型推理

在 AMD Strix Halo（gfx1151 / RDNA 3.5）上运行 Qwen 3.6-27B AWQ-INT4 量化模型，支持视觉输入、工具调用和 Qwen3-ASR 语音识别。

## 快速开始

### 前置准备

1. **硬件**：AMD Ryzen AI MAX+ 395，128 GB UMA，BIOS 中 UMA Frame Buffer 设为最小 2 GB
2. **GRUB 配置**：`ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1`
3. **Podman**：rootless 模式，支持 `--privileged` + `/dev/kfd` + `/dev/dri` 暴露
4. **模型下载**：

```bash
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' .env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download cyankiwi/Qwen3.6-27B-AWQ-INT4 --cache-dir "$VLLM_HOST_MODELS_DIR/hub"
```

### 构建与启动

```bash
# 1. 复制并编辑环境变量
cp .env.template .env
nano .env  # 至少填写 VLLM_HOST_MODELS_DIR 和 HF_TOKEN

# 2. 构建镜像（首次约 25-35 分钟）
podman-compose build vllm

# 3. 启动服务
podman-compose up -d vllm

# 4. 验证
curl http://127.0.0.1:8000/v1/models
```

> 详细指南见 [docs/GUIDE.md](docs/GUIDE.md)

## 服务

| 服务 | 端口 | 模型 | 说明 |
|------|------|------|------|
| vllm | 8000 | Qwen3.6-27B-AWQ4 | 文本 LLM + 视觉 + 工具调用 |
| vllm-asr | 8001 | Qwen3-ASR-8B | 语音转文字（非流式/SSE/WebSocket） |

## 性能

| 场景 | 吞吐量 | 说明 |
|------|--------|------|
| 单流解码（基线） | ~5.6 t/s | Qwen 3.6-27B AWQ4，256K 上下文 |
| 预填充 | **33-38 t/s** 均值 | 包含 prompt-with-tools 场景 |

## 模型

| 项目 | 值 |
|------|-----|
| 目标模型 | `cyankiwi/Qwen3.6-27B-AWQ-INT4` |
| 量化方案 | AWQ-INT4, W4A16, group_size 32, compressed-tensors |
| 磁盘占用 | ~14 GiB |

## API 端点

| 端点 | 用途 |
|------|------|
| `POST /v1/chat/completions` | 标准聊天，支持 thinking/视觉/工具调用 |
| `POST /v1/responses` | OpenAI Responses API，SSE 流式分离 reasoning/output |
| `POST /v1/completions` | 原始文本补全 |

### 快速测试

```bash
# 基础聊天
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-AWQ4","messages":[{"role":"user","content":"你好"}]}'

# 使用内置 CLI
./glados.py "解释量子纠缠"
```

## 推荐配置

| 配置 | MAX_NUM_SEQS | MAX_MODEL_LEN | GPU_MEMORY_UTIL | 使用场景 |
|------|-------------|---------------|-----------------|---------|
| **单用户，最大上下文** | `1` | `262144` | `0.9` | 单聊，完整 256K |
| **3 代理多流** | `3` | `131072` | `0.5` | 3 并发客户端，为 RAG/TTS 留空间 |

## 目录结构

```
vllm/
├── README.md              ← 本文件
├── Dockerfile             ← vLLM + PyTorch + pip ROCm SDK 构建
├── .env.template          ← 环境变量模板
├── glados.py              ← CLI 客户端（纯标准库 REPL）
├── docs/
│   ├── GUIDE.md           ← 全流程使用指南
│   ├── LLM.md             ← 文本大模型详细部署指南
│   └── ASR.md             ← 语音识别详细部署指南
├── scripts/
│   ├── patch_strix.py     ← gfx1151 适配补丁 (19 个)
│   ├── vllm_profile_cache.py ← profile 缓存优化
│   └── dump_logs.sh       ← 日志诊断导出
└── test/
    ├── bench.py           ← 5 端点扫描
    ├── bench_full.py      ← 全功能测试 + 工具调用 + 图像
    ├── bench_longctx.py   ← 长上下文基准测试
    └── verify_responses_streaming.py ← SSE 追踪验证
```

> docker-compose.yml 在项目根目录，包含 vllm/vllm-asr/vllm-omni 三个服务。

## 技术栈

| 层 | 组件 | 版本 |
|---|------|------|
| 推理引擎 | vLLM | v0.20.1 |
| ROCm SDK | pip rocm[devel,libraries] 7.13 nightly | gfx1151 (site-packages) |
| PyTorch | torch + triton | 2.10 + 3.6 |
| 量化 | AWQ-INT4 W4A16 g32 (compressed-tensors) | — |
| 注意力 | Triton SDPA (JIT 运行时编译) | — |

## 已知限制

- **AOTriton**：Ubuntu 26.04 自带 CMake 4.2，AOTriton 构建不兼容，改用 Triton JIT 运行时编译
- **Flash-Attention（Dao-AILab）**：gfx1151 上编译失败，使用 Triton SDPA 路径
- **AITER 自定义核**：CDNA 专属指令（DPP/向量打包）在 RDNA 上不存在
- **HIP Graph**：gfx1151 上的冻结类问题，使用 `--enforce-eager`
- **流式工具调用**：上游解析器 PR 未合并，推荐使用非流式或 `/v1/responses` 路径

## 相关文档

- [docs/GUIDE.md](docs/GUIDE.md) — 从零开始的全流程使用指南
- [docs/LLM.md](docs/LLM.md) — Qwen3.6-27B 文本大模型详细部署指南
- [docs/ASR.md](docs/ASR.md) — Qwen3-ASR 语音识别详细部署指南
- [../README.md](../README.md) — 项目集总览（硬件、系统配置、Podman 部署）
