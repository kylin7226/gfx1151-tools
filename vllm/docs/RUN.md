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

### 3.4 容器运行时参数

这些是 `docker run` / `podman run` 的通用参数，三个服务共用。

| 参数 | 说明 |
|------|------|
| `--name <name>` | 容器名称，用于 `docker logs`、`docker exec` 等命令引用 |
| `--privileged` | 赋予容器几乎全部宿主机权限，使容器内能直接访问 GPU 设备节点 |
| `--device /dev/kfd:/dev/kfd` | 映射 AMD GPU 的 KFD 设备节点（Kernel Fusion Driver），ROCm 运行时必需 |
| `--device /dev/dri:/dev/dri` | 映射 DRI（Direct Rendering Infrastructure）设备节点，用于 DRM/GEM 内存管理 |
| `--ipc host` | 共享宿主机 IPC 命名空间。vLLM 的多进程张量并行依赖共享内存通信，必须设置为 `host` |
| `--shm-size 16gb` | 设置 `/dev/shm` 共享内存大小。PyTorch DataLoader 和 Triton JIT 编译使用共享内存，默认 64MB 不够 |
| `-p <host>:<container>` | 端口映射。ASR 和 Omni 的容器内端口固定为 8000，通过映射到不同宿主机端口实现多服务共存 |
| `-v <host>:<container>:ro` | 卷挂载。`:ro` 表示只读（模型目录），防止容器内误写模型文件 |

### 3.5 环境变量详解

#### 3.5.1 HuggingFace 相关

| 变量 | 示例值 | 说明 |
|------|--------|------|
| `HF_HOME` | `/models` | HF 缓存根目录。容器内指向挂载的模型目录，避免重复下载 |
| `HF_HUB_OFFLINE` | `1` | 离线模式。已预下载模型时设为 1，跳过网络检查加速启动 |
| `HF_TOKEN` | `hf_xxx...` | HuggingFace 访问令牌。访问 gated model（ASR、Omni）时必须。通过 `.env` 或 `-e` 传入 |

#### 3.5.2 ROCm/GPU 相关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HIP_VISIBLE_DEVICES` | `0` | 可见 GPU 设备索引。单 iGPU 设为 0；如需隐藏 GPU 设为空字符串 |
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | 强制覆盖 GFX IP 版本。ASR 服务使用此变量确保 ROCm 识别 gfx1151。格式为 `major.minor.stepping` |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | 禁止 HSA 运行时回收 GPU scratch 内存。gfx1151 上 AWQ 张量加载时不设置会触发段错误（vllm#37151） |
| `MIOPEN_FIND_MODE` | `FAST` | MIOpen 卷积算法搜索模式。设为 `FAST` 跳过穷举搜索，使用启发式选择。gfx1151 无预编译求解器数据库，默认 exhaustive 模式会在 ViT 卷积层卡住 |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | 启用 Triton AMD FlashAttention 路径。gfx1151 上唯一可行的 FlashAttention 实现 |
| `VLLM_ROCM_USE_AITER` | `0` | 禁用 AITER 自定义内核。AITER 使用 CDNA 专属指令（DPP、向量打包），在 RDNA 3.5 上不存在，启用即崩溃 |

#### 3.5.3 vLLM 功能开关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_USE_TRITON_AWQ` | `1` | 强制使用 Triton AWQ 内核路径。AWQ-INT4 模型通过 AWQMarlin → Conch 内核执行，比 legacy `ops.awq_gemm` 快 57-73% |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | 禁用 TorchDynamo 编译缓存。gfx1151 上冷启动时 EngineCore 序列化有 bug，设为 1 避免 |
| `VLLM_SKIP_MEMORY_PROFILING` | `1` | 跳过内存 profile_run。vLLM 默认启动时运行 ~7 分钟合成推理来确定 KV cache 大小。设为 1 配合 profile 缓存机制可将重启从 ~9 min 降到 ~90 s |
| `VLLM_PROFILE_CACHE_DIR` | `/root/.cache/vllm-profile` | Profile 缓存写入/读取目录。首次启动后缓存 KV cache 大小结果，后续启动直接读取。需挂载持久化卷 |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | ASR 专用。单个音频片段大小上限（MB）。25 MB 约等于 30 分钟 16kHz 单声道 PCM16 音频 |
| `VLLM_OMNI_TARGET_DEVICE` | `rocm` | Omni 专用。强制 vllm-omni 使用 ROCm 平台检测而非 CUDA |

### 3.6 vLLM serve 命令行参数

这些是 `vllm serve` 命令的参数，通过 docker-compose.yml 的 `command:` 或独立 `podman run` 传入。

#### 3.6.1 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务监听地址。容器内必须为 `0.0.0.0` 以接受外部连接 |
| `--port` | `8000` | 服务监听端口。容器内固定 8000，通过 `-p` 映射到宿主机不同端口 |
| `--served-model-name` | 自动检测 | API 中展示的模型名称。`/v1/models` 和聊天请求中的 `model` 字段需匹配此值 |
| `--enforce-eager` | 关闭 | 禁用 HIP Graph（CUDA Graph 的 ROCm 等效物）。gfx1151 上 HIP Graph 捕获会导致冻结，**必须开启** |
| `--gpu-memory-utilization` | `0.9` | GPU 内存利用率上限。vLLM 用此值 × 总可用内存 = KV cache + 权重的预算上限。OOM 时降低（0.9→0.7→0.5） |
| `--max-num-seqs` | `256` | 最大并发序列数。单用户设为 1；多用户并发时增大。增大后 KV cache 预算更紧张 |
| `--max-model-len` | 模型默认 | 最大上下文长度（token 数）。Qwen 3.6 原生 256K。减小此值可直接降低 KV cache 内存占用 |

#### 3.6.2 注意力后端参数

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--attention-backend` | `TRITON_ATTN`（推荐）、`ROCM_ATTN`、`TORCH_SDPA` | LLM 主注意力后端。gfx1151 上 `TRITON_ATTN` 性能最优，`ROCM_ATTN` 的 paged attention kernel 内部回退到 Triton 有额外开销 |
| `--mm-encoder-attn-backend` | `TRITON_ATTN`（必须）、`TORCH_SDPA` | 多模态编码器（视觉/音频）的注意力后端。**gfx1151 上必须为 `TRITON_ATTN`**，`TORCH_SDPA` 在 gfx1151 上会产生 NaN/Inf 导致视觉/音频输入返回乱码 |

#### 3.6.3 模型专用参数

| 参数 | 适用服务 | 说明 |
|------|----------|------|
| `--reasoning-parser qwen3` | LLM | 解析 Qwen3 的 `` 推理 token，使 `/v1/responses` 能分离 reasoning 和 output 内容 |
| `--tool-call-parser qwen3_coder` | LLM | Qwen3.5/3.6 工具调用解析器。配合 `--enable-auto-tool-choice` 使用。**注意**：流式模式下有上游 bug（3 个 PR 未合并），建议工具调用使用非流式 |
| `--enable-auto-tool-choice` | LLM | 自动检测并执行工具调用。需配合 `--tool-call-parser` |
| `--supported-tasks transcription,realtime` | ASR | 注册 ASR 端点：`/v1/audio/transcriptions`（非流式/SSE 流式）和 `/v1/realtime`（WebSocket 实时） |

### 3.7 .env 环境变量

docker-compose.yml 通过 `${VAR:-default}` 语法从 `.env` 文件读取以下变量。

| 变量 | 默认值 | 适用服务 | 说明 |
|------|--------|----------|------|
| `VLLM_HOST_MODELS_DIR` | — | 全部 | **必填**。宿主机上 HF 缓存的绝对路径。通过只读挂载到容器内 `/models` |
| `HF_TOKEN` | — | 全部 | **必填**。HuggingFace 访问令牌 |
| `VLLM_COMMIT` | `v0.20.1` | LLM, ASR | vLLM git 版本引用。修改后需重新构建镜像 |
| `VLLM_MODEL_ID` | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | LLM | 模型 ID 或本地路径 |
| `VLLM_SERVED_MODEL_NAME` | `Qwen3.6-27B-AWQ4` | LLM | API 中展示的模型名 |
| `VLLM_HOST_PORT` | `8000` | LLM | LLM 服务宿主机端口 |
| `VLLM_MAX_NUM_SEQS` | `1` | LLM | LLM 最大并发序列数 |
| `VLLM_MAX_MODEL_LEN` | `262144` | LLM | LLM 最大上下文长度 |
| `VLLM_GPU_MEMORY_UTIL` | `0.9` | LLM | LLM GPU 内存利用率 |
| `VLLM_ASR_MODEL_ID` | `Qwen/Qwen3-ASR-8B` | ASR | ASR 模型 ID |
| `VLLM_ASR_HOST_PORT` | `8001` | ASR | ASR 服务宿主机端口 |
| `VLLM_ASR_GPU_MEMORY_UTIL` | `0.9` | ASR | ASR GPU 内存利用率 |
| `VLLM_ASR_MAX_MODEL_LEN` | `8192` | ASR | ASR 最大上下文长度 |
| `VLLM_ASR_MAX_NUM_SEQS` | `1` | ASR | ASR 最大并发序列数 |
| `VLLM_OMNI_MODEL_ID` | `Qwen/Qwen3-Omni-MoE-27B` | Omni | Omni 模型 ID |
| `VLLM_OMNI_SERVED_MODEL_NAME` | `Qwen3-Omni-MoE-27B` | Omni | Omni API 中展示的模型名 |
| `VLLM_OMNI_HOST_PORT` | `8002` | Omni | Omni 服务宿主机端口 |
| `VLLM_OMNI_GPU_MEMORY_UTIL` | `0.9` | Omni | Omni GPU 内存利用率 |
| `VLLM_OMNI_MAX_MODEL_LEN` | `8192` | Omni | Omni 最大上下文长度 |
| `VLLM_OMNI_MAX_NUM_SEQS` | `1` | Omni | Omni 最大并发序列数 |
| `VLLM_OMNI_COMMIT` | `v0.20.0rc1` | Omni | vllm-omni git 版本引用 |
| `VLLM_HOST_TRITON_CACHE` | `./.triton-cache` | LLM, Omni | Triton JIT 缓存宿主机路径 |
| `VLLM_HOST_VLLM_CACHE` | `./.vllm-cache` | 全部 | vLLM 缓存宿主机路径 |

### 3.8 卷挂载详解

| 宿主机路径 | 容器内路径 | 权限 | 用途 |
|------------|-----------|------|------|
| `$VLLM_HOST_MODELS_DIR` | `/models` | 只读 (`ro`) | HuggingFace 模型缓存。包含 `hub/` 子目录，内有各模型 snapshot |
| `$VLLM_HOST_TRITON_CACHE` | `/root/.triton/cache` | 读写 | Triton JIT 编译缓存。持久化后重启容器无需重新编译内核 |
| `$VLLM_HOST_VLLM_CACHE` | `/root/.cache/vllm` | 读写 | vLLM 内部缓存（如 torch compile cache，当 `VLLM_DISABLE_COMPILE_CACHE=0` 时） |
| `$VLLM_HOST_VLLM_CACHE/profile` | `/root/.cache/vllm-profile` | 读写 | Profile 缓存。记录 KV cache 内存大小测量结果，加速后续启动 |

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

| 症状　　　　　　　　　　　　 | 原因　　　　　　　　　　　　　　　　　　　　　　| 解决　　　　　　　　　　　　　　　　　　　　　|
| ------------------------------| -------------------------------------------------| -----------------------------------------------|
| 启动后 curl 无响应　　　　　 | 仍在 profile_run 阶段　　　　　　　　　　　　　 | 等待 `Application startup complete` 日志　　　|
| OOM / 显存不足　　　　　　　 | `--gpu-memory-utilization` 过高　　　　　　　　 | 降到 0.5-0.7，或减少 `--max-model-len`　　　　|
| 端口被占用　　　　　　　　　 | 其他服务已占用端口　　　　　　　　　　　　　　　| 修改 `-p` 映射或 `.env` 中的端口变量　　　　　|
| 模型加载失败 "Access denied" | HF Token 无效或未接受 gated model 条款　　　　　| 检查 `HF_TOKEN`，登录 HF 接受条款后重新下载　 |
| 视觉输入返回乱码　　　　　　 | 使用了错误的注意力后端　　　　　　　　　　　　　| 确保 `--mm-encoder-attn-backend TRITON_ATTN`　|
| 工具调用流式不完整　　　　　 | 上游流式解析器 bug　　　　　　　　　　　　　　　| 使用非流式或 `/v1/responses` + `stream: true` |
| DEBUG 日志导致极慢　　　　　 | `VLLM_LOGGING_LEVEL=DEBUG` 使每个 op 格式化参数 | **不要设置 DEBUG 级别**　　　　　　　　　　　 |

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
