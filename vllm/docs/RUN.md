# 容器运行与模型启动速查

本文档汇总了所有通过 Docker/Podman 运行镜像和启动模型的操作方式，涵盖 docker compose、独立 `docker run`/`podman run` 以及 GHCR 预构建镜像拉取三种路径。

## 一、前置条件

### 1.1 硬件与系统

| 项目 | 要求 |
|------|------|
| GPU | AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151 / RDNA 3.5) 或兼容 iGPU |
| 内存 | 128 GB UMA |
| BIOS | UMA Frame Buffer Size 设为最小值 **2 GB** |
| GRUB | `ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1` |
| 磁盘 | ≥ 100 GB（LLM ~14 GB + ASR ~16 GB + Omni ~60 GB + 缓存） |

### 1.2 容器运行时

**Podman**（推荐，rootless 模式）：

```bash
sudo apt-get install -y podman
sudo usermod -aG podman $USER  # 重新登录后生效
```

**Docker**：

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # 重新登录后生效
```

> `podman` 和 `docker` 命令在下文中可以互换使用。Podman 兼容 Docker 语法。

### 1.3 模型预下载

```bash
# 配置环境变量
cp vllm/.env.template vllm/.env
nano vllm/.env   # 至少填写 VLLM_HOST_MODELS_DIR 和 HF_TOKEN

export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' vllm/.env | xargs)

# LLM 模型
HF_HUB_ENABLE_HF_TRANSFER=1 hf download cyankiwi/Qwen3.6-27B-AWQ-INT4 --cache-dir "$VLLM_HOST_MODELS_DIR/hub"

# ASR 模型（gated，需先接受条款）
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-ASR-8B --cache-dir "$VLLM_HOST_MODELS_DIR/hub"

# Omni 模型（gated，需先接受条款）
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-Omni-MoE-27B --cache-dir "$VLLM_HOST_MODELS_DIR/hub"
```

---

## 二、方式一：docker compose / podman-compose（推荐）

适合大多数用户，一键编排全部服务。

### 2.1 构建镜像

```bash
# 进入项目根目录
cd gfx1151-tools

# 构建 vLLM 基础镜像（首次约 25-35 分钟）
podman-compose build vllm

# 构建 vLLM-Omni 镜像（依赖 vllm，约 5-10 分钟）
podman-compose build vllm-omni

# 或一次性构建全部
podman-compose build
```

### 2.2 启动服务

```bash
# 全部服务（LLM + ASR + Omni）
podman-compose up -d

# 仅 LLM（文本推理）
podman-compose up -d vllm

# 仅 ASR（语音识别）
podman-compose up -d vllm-asr

# 仅 Omni（全模态）
podman-compose up -d vllm-omni

# LLM + ASR（常见组合）
podman-compose up -d vllm vllm-asr
```

### 2.3 验证

```bash
# 检查容器状态
podman ps

# 等待 Application startup complete 后再测试
podman logs -f rocm_gfx1151_vllm   # LLM
podman logs -f vllm-asr             # ASR
podman logs -f vllm-omni            # Omni
```

| 服务 | 验证命令 | 期望 |
|------|----------|------|
| LLM | `curl http://127.0.0.1:8000/v1/models` | 返回 `Qwen3.6-27B-AWQ4` |
| ASR | `curl http://127.0.0.1:8001/v1/models` | 返回 `Qwen/Qwen3-ASR-8B` |
| Omni | `curl http://127.0.0.1:8002/v1/models` | 返回 `Qwen3-Omni-MoE-27B` |

### 2.4 停止与重启

```bash
# 停止全部
podman-compose down

# 停止单个服务
podman-compose stop vllm

# 重启单个服务
podman-compose start vllm

# 完整重建（修改代码/补丁后）
podman-compose down && podman-compose build && podman-compose up -d
```

> **注意**：`.env` 变更后需要 `down && up -d`，仅 `restart` 不会重新读取环境变量。

---

## 三、方式二：独立 docker/podman run

适合调试、自定义参数、或不在 docker compose 管理的场景。

### 3.1 LLM 服务

```bash
# 先创建缓存目录
mkdir -p ./.triton-cache ./.vllm-cache/profile

podman run -d \
  --name rocm_gfx1151_vllm \
  --privileged \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --ipc host \
  --shm-size 16gb \
  -p 8000:8000 \
  -v "$VLLM_HOST_MODELS_DIR":/models:ro \
  -v "$PWD/.triton-cache":/root/.triton/cache \
  -v "$PWD/.vllm-cache":/root/.cache/vllm \
  -v "$PWD/.vllm-cache/profile":/root/.cache/vllm-profile \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e MIOPEN_FIND_MODE=FAST \
  -e VLLM_SKIP_MEMORY_PROFILING=1 \
  -e VLLM_PROFILE_CACHE_DIR=/root/.cache/vllm-profile \
  rocm_gfx1151_vllm:v0.20.1 \
  vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name Qwen3.6-27B-AWQ4 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1 \
    --max-model-len 262144
```

### 3.2 ASR 服务

```bash
mkdir -p ./.vllm-cache/profile

podman run -d \
  --name vllm-asr \
  --privileged \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --ipc host \
  --shm-size 16gb \
  -p 8001:8000 \
  -v "$VLLM_HOST_MODELS_DIR":/models:ro \
  -v "$PWD/.vllm-cache":/root/.cache/vllm \
  -v "$PWD/.vllm-cache/profile":/root/.cache/vllm-profile \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e MIOPEN_FIND_MODE=FAST \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=25 \
  -e VLLM_SKIP_MEMORY_PROFILING=1 \
  -e VLLM_PROFILE_CACHE_DIR=/root/.cache/vllm-profile \
  rocm_gfx1151_vllm:v0.20.1 \
  vllm serve Qwen/Qwen3-ASR-8B \
    --host 0.0.0.0 --port 8000 \
    --supported-tasks transcription,realtime \
    --enforce-eager \
    --max-model-len 8192 \
    --mm-encoder-attn-backend TRITON_ATTN \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1
```

### 3.3 Omni 服务

```bash
mkdir -p ./.triton-cache ./.vllm-cache

podman run -d \
  --name vllm-omni \
  --privileged \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --ipc host \
  --shm-size 16gb \
  -p 8002:8000 \
  -v "$VLLM_HOST_MODELS_DIR":/models:ro \
  -v "$PWD/.triton-cache":/root/.triton/cache \
  -v "$PWD/.vllm-cache":/root/.cache/vllm \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e MIOPEN_FIND_MODE=FAST \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_OMNI_TARGET_DEVICE=rocm \
  rocm_gfx1151_vllm-omni:v0.20.0rc1 \
  vllm-omni serve Qwen/Qwen3-Omni-MoE-27B \
    --host 0.0.0.0 --port 8000 \
    --served-model-name Qwen3-Omni-MoE-27B \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1
```

### 3.4 关键参数速查

| 参数 | 作用 | 何时调整 |
|------|------|----------|
| `--gpu-memory-utilization` | UMA 池利用率上限 | OOM 时降低（0.9→0.7→0.5） |
| `--max-num-seqs` | 最大并发序列数 | 多用户并发时增大 |
| `--max-model-len` | 最大上下文长度 | 需要超长上下文时增大，内存紧张时减小 |
| `--enforce-eager` | 禁用 HIP 图捕获 | **gfx1151 上必须开启** |
| `--attention-backend TRITON_ATTN` | Triton 注意力 | **gfx1151 上 LLM 必须** |
| `--mm-encoder-attn-backend TRITON_ATTN` | 多模态编码器注意力 | **gfx1151 上必须**（TORCH_SDPA 会产生 NaN） |
| `VLLM_SKIP_MEMORY_PROFILING=1` | 跳过 ~7 分钟 profile | 配合 `VLLM_PROFILE_CACHE_DIR` 加速重启 |

---

## 四、方式三：GHCR 预构建镜像

推送 main 分支或打 tag 后，GitHub Actions 自动构建并推送镜像到 GitHub Container Registry。无需本地构建。

### 4.1 拉取镜像

```bash
# 查看可用标签
docker pull ghcr.io/<owner>/gfx1151-tools/rocm_gfx1151_vllm_v0.20.1:main
docker pull ghcr.io/<owner>/gfx1151-tools/rocm_gfx1151_vllm-omni_v0.20.0rc1:main

# 或使用带时间戳的版本（更稳定）
docker pull ghcr.io/<owner>/gfx1151-tools/rocm_gfx1151_vllm_v0.20.1:20260509123456
```

> 将 `<owner>` 替换为你的 GitHub 用户名。标签格式为 `<分支名>` 或 `YYYYMMDDHHmmSS` 时间戳。

### 4.2 使用预构建镜像运行

拉取后直接使用 [方式二](#三方式二独立-dockerpodman-run) 中的 `podman run` 命令，只需将镜像名替换为 GHCR 地址即可。

### 4.3 替换 docker-compose.yml 中的镜像

将 `docker-compose.yml` 中的 `image:` 字段改为 GHCR 地址，然后执行 `podman-compose up -d`。由于镜像已存在，Compose 不会尝试本地构建。

---

## 五、模型启动排障

### 5.1 启动流程

```
容器启动 → entrypoint 校验 → vllm serve 启动 → 模型加载 → profile_run → 服务就绪
  (~1s)        (~1s)           (~2s)            (~60s)      (~7min, 有缓存 <10s)   (~5s)
```

- **冷启动**（首次）：约 8-10 分钟（模型加载 + ~7 分钟 profile_run）
- **热启动**（有 profile 缓存）：约 90-100 秒

### 5.2 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| 启动后 curl 无响应 | 仍在 profile_run 阶段 | 等待 `Application startup complete` 日志 |
| OOM / 显存不足 | `--gpu-memory-utilization` 过高 | 降到 0.5-0.7，或减少 `--max-model-len` |
| 端口被占用 | 其他服务已占用端口 | 修改 `-p` 映射或 `.env` 中的端口变量 |
| 模型加载失败 "Access denied" | HF Token 无效或未接受 gated model 条款 | 检查 `HF_TOKEN`，登录 HF 接受条款后重新下载 |
| 视觉输入返回乱码 | 使用了错误的注意力后端 | 确保 `--mm-encoder-attn-backend TRITON_ATTN` |
| 工具调用流式不完整 | 上游流式解析器 bug | 使用非流式或 `/v1/responses` + `stream: true` |
| DEBUG 日志导致极慢 | `VLLM_LOGGING_LEVEL=DEBUG` 使每个 op 格式化参数 | **不要设置 DEBUG 级别** |

### 5.3 诊断命令

```bash
# 查看容器日志
podman logs rocm_gfx1151_vllm 2>&1 | tail -100

# 关键信息筛选
podman logs rocm_gfx1151_vllm 2>&1 | grep -E "Application startup|Available KV|Cached KV|ERROR"

# GPU 使用情况
watch -n 1 'cat /sys/class/drm/card1/device/mem_info_gtt_used; cat /sys/class/drm/card1/device/mem_info_vram_used'

# 进入容器调试
podman exec -it rocm_gfx1151_vllm bash

# 导出诊断日志（卡死时先运行此命令）
./vllm/scripts/dump_logs.sh stuck-state
```

### 5.4 推荐配置方案

| 场景 | `--max-num-seqs` | `--max-model-len` | `--gpu-memory-util` | 预计 GTT 占用 |
|------|------------------|-------------------|---------------------|--------------|
| 单用户，最大上下文 | 1 | 262144 | 0.9 | ~115 GiB |
| 3 代理多流（推荐） | 3 | 131072 | 0.5 | ~64 GiB |
| LLM + ASR 共存 | 1 + 1 | 262144 + 8192 | 0.9 + 0.9 | ~70 GiB |

---

## 六、端口总览

| 服务 | 宿主机端口 | 容器端口 | 用途 |
|------|-----------|---------|------|
| vllm (LLM) | `${VLLM_HOST_PORT:-8000}` | 8000 | 文本聊天 / 视觉 / 工具调用 |
| vllm-asr (ASR) | `${VLLM_ASR_HOST_PORT:-8001}` | 8000 | 语音转文字（HTTP + WebSocket） |
| vllm-omni (Omni) | `${VLLM_OMNI_HOST_PORT:-8002}` | 8000 | 全模态 any-to-any |

> ASR 和 Omni 的容器内端口固定为 8000，通过 `-p` 映射到不同的宿主机端口。
