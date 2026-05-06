# 项目使用指南

Qwen 3.6-27B AWQ-INT4 + Qwen3-ASR 语音识别，运行于 AMD Strix Halo (gfx1151)。

## 快速导航

| 文档 | 内容 |
|---|---|
| [LLM.md](LLM.md) | Qwen3.6-27B 文本大模型部署指南 |
| [ASR.md](ASR.md) | Qwen3-ASR 语音识别部署指南 |
| 本文档 | 从零开始的全流程使用指南 |

## 一、环境准备

### 1.1 硬件要求

- AMD Ryzen AI Max+ 395 "Strix Halo" 或兼容 gfx1151 iGPU
- 128 GB UMA 内存
- ≥ 100 GB 磁盘空间（LLM + ASR 模型 + 缓存）

### 1.2 BIOS 设置

将专用 GPU VRAM 设置为 **最小值（2 GB / 2048 MB）**。菜单名称可能为 `UMA Frame Buffer Size`、`iGPU Memory` 或 `GPU Shared Memory`。目标是 GTT-on-demand 模式，而非固定预分配。

### 1.3 GRUB 配置

```bash
sudo nano /etc/default/grub
# 确保包含以下参数：
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1"
sudo update-grub
sudo reboot

# 验证：
cat /sys/class/drm/card1/device/mem_info_gtt_total
# 期望值：~124554670080（≈ 116 GiB）
```

### 1.4 安装 Podman

```bash
# Ubuntu/Debian
sudo apt-get install -y podman
# 或使用官方安装脚本：
curl -fsSL https://get.podman.io/ | sh
sudo usermod -aG podman $USER
# 重新登录后生效

# 可选：安装 podman-compose（或使用 docker-compose 兼容层）
pip install podman-compose
```

> **注意**：本文档中的 `docker compose` 命令在 Podman 环境下可通过以下方式兼容：
> - 安装 `podman-docker` 包（提供 `docker` → `podman` 兼容层）：`sudo apt install podman-docker`
> - 或设置别名：`alias docker=podman`
> - 或直接使用 `podman-compose` 替代 `docker compose`

## 二、克隆项目

```bash
git clone <repo-url>
cd <repo-dir>
```

## 三、配置

```bash
cp .env.template .env
nano .env
```

必填字段：

| 变量 | 说明 |
|---|---|
| `VLLM_HOST_MODELS_DIR` | HuggingFace 模型缓存路径的绝对路径 |
| `HF_TOKEN` | HuggingFace 读取令牌 |

按需调整的参数：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_HOST_PORT` | `8000` | 文本 LLM 服务端口 |
| `VLLM_MODEL_ID` | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | 目标模型 |
| `VLLM_SERVED_MODEL_NAME` | `Qwen3.6-27B-AWQ4` | API 中显示的模型名 |
| `VLLM_MAX_NUM_SEQS` | `1` | 最大并发流数 |
| `VLLM_MAX_MODEL_LEN` | `262144` | 最大上下文长度（256K） |
| `VLLM_GPU_MEMORY_UTIL` | `0.9` | 显存利用率 |
| `VLLM_ASR_MODEL_ID` | `Qwen/Qwen3-ASR-8B` | ASR 模型 |
| `VLLM_ASR_HOST_PORT` | `8001` | ASR 服务端口 |
| `VLLM_ASR_GPU_MEMORY_UTIL` | `0.9` | ASR 显存利用率 |
| `VLLM_ASR_MAX_MODEL_LEN` | `8192` | ASR 最大上下文 |

## 四、下载模型

### 4.1 文本 LLM（必需）

```bash
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' .env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download cyankiwi/Qwen3.6-27B-AWQ-INT4 --cache-dir "$VLLM_HOST_MODELS_DIR/hub"
```

### 4.2 ASR 模型（可选）

ASR 模型也是 gated model，需要先访问 https://huggingface.co/Qwen/Qwen3-ASR-8B 接受条款。

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-ASR-8B --cache-dir "$VLLM_HOST_MODELS_DIR/hub"
```

> 两个模型可以并行下载。磁盘占用：LLM ~14 GB + ASR ~16 GB = ~30 GB。

## 五、构建镜像

```bash
docker compose build
```

首次构建约 25-35 分钟。Docker 会缓存中间层，后续构建（仅修改补丁）约 3-5 分钟。

## 六、启动服务

### 6.1 启动全部服务（LLM + ASR）

```bash
docker compose up -d vllm vllm-asr
```

### 6.2 仅启动文本 LLM

```bash
docker compose up -d vllm
```

### 6.3 仅启动 ASR

```bash
docker compose up -d vllm-asr
```

### 6.4 查看日志

```bash
# 文本 LLM
docker logs -f rocm_gfx1151_vllm

# ASR
docker logs -f vllm-asr
```

**等待 `Application startup complete` 信号**，表示服务就绪。

### 6.5 验证

```bash
# 文本 LLM
curl http://127.0.0.1:8000/v1/models

# ASR
curl http://127.0.0.1:8001/v1/models
```

## 七、快速测试

### 7.1 文本聊天

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-AWQ4","messages":[{"role":"user","content":"你好"}]}'
```

### 7.2 使用 glados.py CLI

项目内置了一个纯 Python 标准库的命令行工具，提供 REPL 和单次对话模式：

```bash
# 交互式 REPL
./glados.py

# 单次对话
./glados.py "explain mitosis"
```

REPL 中会实时流式显示 thinking 内容和回答，并在结束后显示统计信息（token 数、耗时、吞吐量）。

### 7.3 语音识别

```bash
# 非流式转录
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "model=Qwen/Qwen3-ASR-8B"

# SSE 流式
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@long_audio.wav" -F "model=Qwen/Qwen3-ASR-8B" -F "stream=true"
```

## 八、推荐配置方案

### 8.1 单用户最大上下文

适合一个人用，一次一个对话，需要完整 256K 上下文。

| 变量 | 值 |
|---|---|
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_MAX_MODEL_LEN` | `262144` |
| `VLLM_GPU_MEMORY_UTIL` | `0.9` |

实测空闲显存 ~36 GiB。

### 8.2 3 代理多流（推荐）

适合 3 个并发客户端（聊天 UI + RAG API + 自动化客户端），128K 上下文。

| 变量 | 值 |
|---|---|
| `VLLM_MAX_NUM_SEQS` | `3` |
| `VLLM_MAX_MODEL_LEN` | `131072` |
| `VLLM_GPU_MEMORY_UTIL` | `0.5` |

实测空闲显存 ~50 GiB，为 CPU 服务（RAG/TTS）留出 ~75 GiB UMA 空间。

### 8.3 同时运行 LLM + ASR

| 资源 | 文本 LLM | ASR | 总计 |
|---|---|---|---|
| 模型权重 | ~28 GiB | ~16 GiB | ~44 GiB |
| KV cache | ~24 GiB | ~2 GiB | ~26 GiB |
| GTT 总计 | ~50 GiB | ~20 GiB | ~70 GiB |

128 GB UMA 池完全够用。如果使用 3 代理配置，建议将 LLM 的 `gpu_memory_utilization` 降到 0.5。

## 九、端口总览

| 服务 | 端口 | 端点 |
|---|---|---|
| 文本 LLM | 8000 | `/v1/chat/completions`, `/v1/responses`, `/v1/completions` |
| ASR | 8001 | `/v1/audio/transcriptions`, `/v1/realtime` (WebSocket) |

## 十、补丁说明

项目对 vLLM v0.20.1 应用了 18 个补丁：

| 补丁 | 来源 | 内容 |
|---|---|---|
| 1-12 | kyuz0 (amd-strix-halo-vllm-toolboxes) | gfx1151 硬件使能（amdsmi 禁用、架构检测、CDNA-only 特性保护、JIT 路径修复、APU VRAM 余量等） |
| 13 | 本地 | 将 `chat_template_kwargs` 传入 `/v1/responses` 流式路径 |
| 14 | hec-ovi/vllm-awq4-qwen | AWQ-INT4 MMQ HIP 自定义核注册 |
| 15 | hec-ovi/vllm-awq4-qwen | 移除 atomicAdd half/half2 polyfills（ROCm 7.13 兼容性） |
| 16 | 本地 | 缓存 profile_run 结果，跳过 ~7 分钟内存分析 |
| 17 | 本地 + PR #40334 | `combine_hidden_states` dtype 修复 + 非流式 `/v1/responses` enable_thinking 修复 |
| 18 | ROCm/vllm gfx11 | Strix Halo softmax segments 调优（16→32） |

## 十一、运行基准测试

```bash
# SSE 追踪 T1-T5 推理/工具调用验证（Patch 13+17 验证）
python3 test/verify_responses_streaming.py

# 5 端点扫描 + 工具 + 图像 + Three.js 代码生成
python3 test/bench_full.py
# -> test/bench_full_results.json + test/bench_full_threejs.html

# 长上下文（~25K token，硬合成问题）
python3 -u test/bench_longctx.py
# -> test/bench_longctx_result.json

# 原始 5 端点测试（向后兼容）
python3 test/bench.py
# -> test/bench_results.json
```

所有脚本都是自包含的，直接打印结果到 stdout。引擎必须已经在运行。

## 十二、常见问题

### `.env` 修改后不生效

需要使用 `docker compose down && up -d`，**不能**用 `docker compose restart`。`restart` 复用运行中的容器，不会重新读取环境变量。

### 服务启动后端口被占用

修改 `.env` 中的 `VLLM_HOST_PORT` 或 `VLLM_ASR_HOST_PORT` 为其他可用端口。

### 模型加载失败 "Access denied"

1. 确认 `.env` 中的 `HF_TOKEN` 有效
2. 确认已接受 Gated model 条款（ASR 和 Omni 模型都需要）
3. 重新运行下载命令

### 工具调用在流式模式下不完整

使用 `/v1/chat/completions` + `stream: false`，或 `/v1/responses` + `stream: true`。

## 十三、停止和重启

```bash
# 停止全部
docker compose down

# 仅停止 LLM
docker compose stop vllm

# 仅停止 ASR
docker compose stop vllm-asr

# 完整重建（修改代码/补丁后）
docker compose down
docker compose build
docker compose up -d vllm vllm-asr
```

## 十四、GPU 监控

```bash
# 实时 GTT/VRAM 使用量
watch -n 1 'cat /sys/class/drm/card1/device/mem_info_gtt_used; cat /sys/class/drm/card1/device/mem_info_vram_used'

# ROCm SMI 工具
rocm-smi

# 进程级 GPU 占用
rocm-smi --showpidinfo
```

## 十五、日志诊断

```bash
# 查看所有日志
docker logs rocm_gfx1151_vllm 2>&1 | tail -100

# 查看关键信息
docker logs rocm_gfx1151_vllm 2>&1 | grep -E "Application startup|Available KV|Cached KV|ERROR"

# 导出诊断信息（重启前运行）
./scripts/dump_logs.sh
```

> **重要**：遇到卡死状态，先运行 `./scripts/dump_logs.sh stuck-state` 保存诊断信息，再重启容器。

## 十六、GitHub Actions 自动构建推送

推送 main 分支或打 tag 后，GitHub Actions 会自动构建 Docker 镜像并推送到 GitHub Container Registry (GHCR)：

- 推送到 `main`：标签为 `:main` + `<日期时间>`
- 推送 `v*` tag：标签为 `:<版本号>` + `<日期时间>`
- 也可通过 `workflow_dispatch` 手动触发，可自定义 `VLLM_COMMIT`

镜像地址：`ghcr.io/<owner>/<repo>/rocm_gfx1151_vllm_v0.20.1:<YYYYMMDDhhmmss>`

## 十七、相关文档

- [docs/LLM.md](LLM.md) — Qwen3.6-27B 文本大模型详细部署指南
- [docs/ASR.md](ASR.md) — Qwen3-ASR 语音识别详细部署指南
- README.md — 项目概览、性能数据、vs 官方 vLLM 对比
