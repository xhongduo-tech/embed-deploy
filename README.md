# 离线环境 9 节点向量化/重排序模型部署指南

## 📑 项目背景与环境说明

本项目旨在为行内离线内网环境完整部署 **9** 个推理容器节点（向量化及重排序 / Reranker）与 Nginx 网关。由于物理机环境的特殊性，整个实施过程必须在严格的离线状态下进行。

**基础环境约束：**

1. **网络状态：** 100% 纯离线内网环境
2. **硬件基础：** 纯物理机部署（搭载 V100 计算卡）
3. **初始状态：** 裸机环境，无预装 NVIDIA 显卡驱动

---

## 🤗 模型下载（Hugging Face）

本项目使用 BGE 系列与 Qwen3 系列向量化与重排序模型，**模型文件未纳入 Git 仓库**，请在有网环境从 Hugging Face 下载后放入 **`models/`** 目录（小写，与 `docker-compose.yml` 卷挂载一致），再拷贝至离线环境。

### 引擎与格式说明

| 引擎 | 服务对象 | 模型格式 |
|------|----------|----------|
| **Infinity** | BGE-M3、BGE-Reranker-V2-M3 | HuggingFace 原始格式（safetensors） |
| **llama.cpp** | 所有 Qwen3 模型 | GGUF 量化格式 |

> **关于 Qwen3-VL 系列**：名称中的"VL"表示支持多模态（视觉+文本）输入。`Qwen3-VL-Embedding-2B` 与 `Qwen3-VL-Reranker-2B` 在 llama.cpp 中均需 **主模型 GGUF + mmproj（多模态投影）** 一并加载，图像/图文输入才能走视觉分支；`docker-compose.yml` 已对两个 VL 服务配置 `--mmproj`。若仅下载主 GGUF、缺少同目录下的 mmproj 文件，容器启动会失败。

### 模型选型与 VRAM 评估

基于 V100 16GB 显存，各节点实际占用如下：

| GPU | 容器 | 模型 | VRAM（单容器参考） |
|-----|------|------|------|
| GPU 0 | embed-1 / embed-2 / embed-3 | BGE-M3 ×3（Infinity） | ~3.6 GB |
| GPU 0 | embed-qwen-8b | Qwen3-Embedding-8B Q4_K_M ×1 | 4.68 GB |
| GPU 0 | embed-qwen-vl | Qwen3-VL-Embedding-2B Q4_K_M + mmproj F16 | ~1.9 GB |
| **GPU 0 合计** | | | **同卡多实例时以 `nvidia-smi` 实测为准** |
| GPU 1 | reranker-1 / reranker-2 | BGE-Reranker-V2-M3 ×2（Infinity） | ~1.2 GB |
| GPU 1 | reranker-qwen | Qwen3-Reranker-8B Q4_K_M | 5.03 GB |
| GPU 1 | reranker-qwen-vl | Qwen3-VL-Reranker-2B Q4_K_M + mmproj F16 | ~2.0 GB |
| **GPU 1 合计** | | | **~8.2 GB / 16 GB** |

> BGE 与 Qwen 文本向量化在同一张 GPU 上多实例并行时，显存叠加明显；部署后务必用 `nvidia-smi` 观察，必要时调整实例数或换更大显存 GPU。

### 模型清单与下载命令

| 模型 | 用途 | GGUF 来源 | HuggingFace 仓库 |
|------|------|-----------|-----------------|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 文本向量化 ×3（HF 格式） | — | 官方 |
| [Qwen/Qwen3-Embedding-8B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF) | 文本向量化 ×1 | **官方** GGUF | 官方 |
| [DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF) | 多模态向量化 ×1 | 社区 GGUF | 社区 |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 文本重排序 ×2（HF 格式） | — | 官方 |
| [QuantFactory/Qwen3-Reranker-8B-GGUF](https://huggingface.co/QuantFactory/Qwen3-Reranker-8B-GGUF) | 文本重排序 ×1 | 社区 GGUF | 社区 |
| [mradermacher/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-GGUF) | 多模态重排序 ×1 | 社区 GGUF | 社区 |

### BGE 系列（HF 格式，直接下载）

```bash
huggingface-cli download BAAI/bge-m3              --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3  --local-dir models/bge-reranker-v2-m3
```

### Qwen3-Embedding-8B（官方 GGUF，直接下载）

```bash
# 官方仓库常见文件名为 model-Q4_K_M.gguf；若与 compose 中路径一致，可复制为 models/Qwen3-Embedding-8B-Q4_K_M.gguf（见下文「模型目录结构」）
huggingface-cli download Qwen/Qwen3-Embedding-8B-GGUF \
    --include "model-Q4_K_M.gguf" \
    --local-dir models/qwen3-embedding-8b
```

### Qwen3-VL-Embedding-2B（社区 GGUF，直接下载）

```bash
# DevQuasar 仓库文件名常以 "Qwen." 开头；若采用根目录扁平布局，请重命名为 compose 中路径（如 Qwen3-VL-Embedding-2B-Q4_K_M.gguf 与 Qwen3-VL-Embedding-2B-mmproj_f16.gguf）
huggingface-cli download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \
    --include "Qwen.Qwen3-VL-Embedding-2B-Q4_K_M.gguf" \
    --include "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf" \
    --local-dir models/Qwen3-VL-Embedding-2B
```

### Qwen3-Reranker-8B（社区 GGUF，直接下载）

```bash
huggingface-cli download QuantFactory/Qwen3-Reranker-8B-GGUF \
    --include "Qwen3-Reranker-8B-Q4_K_M.gguf" \
    --local-dir models/qwen3-reranker-8b
```

### Qwen3-VL-Reranker-2B（社区 GGUF，直接下载）

```bash
# 下载后若采用根目录扁平布局，文件名需与 compose 一致（如 Qwen3-VL-Reranker-2B-Q4_K_M.gguf、Qwen3-VL-Reranker-2B-mmproj-_f16.gguf，注意个别社区包 mmproj 文件名含连字符）
huggingface-cli download mradermacher/Qwen3-VL-Reranker-2B-GGUF \
    --include "Qwen3-VL-Reranker-2B.Q4_K_M.gguf" \
    --include "Qwen3-VL-Reranker-2B.mmproj-f16.gguf" \
    --local-dir models/Qwen3-VL-Reranker-2B
```

> 💡 多模态能力依赖 mmproj：仓库还提供 `Qwen3-VL-Reranker-2B.mmproj-Q8_0.gguf`（体积更小）。若改用 Q8_0，请将 `docker-compose.yml` 中 `reranker-qwen-vl` 的 `--mmproj` 路径改为对应文件名。

---

## 🛠️ 第一阶段：系统内核（Kernel）一致性确认

NVIDIA 驱动需要挂载至系统内核，因此确保内核（Kernel）、开发包（Kernel-devel）与头文件（Kernel-headers）的版本 100% 匹配是第一要务。

### 1. 离线资源准备

请提前准备或在介质文件夹 `Base/Kernel_Base` 中确认以下三个核心 RPM 包（针对 `kernel-4.19.90-23.60.v2101.ky10.x86_64`）：

| 组件 | 文件名 | 下载链接 |
|------|--------|----------|
| Kernel 本体 | `kernel-4.19.90-23.60.v2101.ky10.x86_64.rpm` | [麒麟 V10SP1.1 更新源](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/kernel-4.19.90-23.60.v2101.ky10.x86_64.rpm) |
| Kernel-devel | `kernel-devel-4.19.90-23.60.v2101.ky10.x86_64.rpm` | [麒麟 V10SP1.1 更新源](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/kernel-devel-4.19.90-23.60.v2101.ky10.x86_64.rpm) |
| Kernel-headers | `kernel-headers-4.19.90-23.60.v2101.ky10.x86_64.rpm` | [麒麟 V10SP1.1 更新源](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/kernel-headers-4.19.90-23.60.v2101.ky10.x86_64.rpm) |

> 备用：若上述链接不可用，可访问目录 [Packages 索引](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/) 手动查找对应版本。

### 2. 版本一致性校验

在安装驱动前，必须通过以下命令核对内核版本：

```bash
# 1. 查看当前正在运行的内核版本
uname -r

# 2. 查看系统中实际安装好的开发包版本
rpm -qa | grep kernel-devel
```

### 3. 冲突修复（如版本不一致）

如果上述两条命令输出的版本号不一致，必须进行强制替换：

```bash
# 强制卸载错误的高版本 (请将命令中的输出替换为你查出的实际错误版本号)
sudo rpm -e --nodeps $(rpm -qa | grep kernel-devel)

# 强制安装 100% 匹配的版本（在离线资源目录下执行）
sudo rpm -ivh kernel-devel-*.rpm kernel-headers-*.rpm
```

至此，内核底层依赖准备完毕。

---

## 🚫 第二阶段：彻底禁用 Nouveau 开源驱动

在安装官方 NVIDIA 驱动前，必须彻底封杀系统自带的 Nouveau 驱动。

> **💡 核心原理解析：为什么要「死磕」Nouveau？**
>
> - **一山不容二虎：** Nouveau 是 Linux 默认的开源显卡驱动，开机即接管 GPU。官方闭源驱动（`nvidia.ko`）同样需要底层硬件的独占控制权。若 Nouveau 已在运行，GPU 资源（显存、寄存器等）会被「锁死」，官方驱动直接挂载失败。
> - **深度绑定无法热卸载：** Nouveau 通常被编译进初始内存盘（`initramfs`），在真正的根文件系统挂载前就已驻留内存。因此，简单的 `rmmod` 命令无效，必须通过修改内核引导参数并 **重启物理机** 才能彻底拔除。

### 方案 A：常规封杀法（优先尝试）

```bash
# 1. 屏蔽 nouveau，写入黑名单配置文件
echo -e "blacklist nouveau\noptions nouveau modeset=0" \
    | sudo tee /etc/modprobe.d/blacklist-nouveau.conf

# 2. 更新内核引导镜像
sudo dracut --force

# 3. 重启服务器
sudo reboot
```

重启后检查：执行 `lsmod | grep nouveau`。如果 **没有任何输出**，说明屏蔽成功，可直接跳至 **第三阶段**。如果仍有输出，请执行方案 B。

### 方案 B：GRUB 核心底层封杀法（方案 A 失败时使用）

如果 Nouveau 依然诈尸，我们需要直接修改内核启动参数进行精准打击。

#### 第一步：修改 GRUB 核心引导文件

```bash
sudo vi /etc/default/grub
```

找到以 `GRUB_CMDLINE_LINUX=` 开头的那一行。在行末的双引号 `"` 前加一个空格，并补充以下参数：

```
nouveau.modeset=0 rd.driver.blacklist=nouveau
```

修改后示例：

```
GRUB_CMDLINE_LINUX="...原来的参数... nouveau.modeset=0 rd.driver.blacklist=nouveau"
```

保存并退出（`:wq`）。

#### 第二步：重新编译生成引导菜单（兼容传统 BIOS 与 UEFI）

```bash
# 刷新传统 BIOS 引导配置
sudo grub2-mkconfig -o /boot/grub2/grub.cfg

# 刷新 UEFI 引导配置（若提示找不到路径请忽略）
sudo grub2-mkconfig -o /boot/efi/EFI/kylin/grub.cfg
```

#### 第三步：针对当前内核的精确打包

强制指定当前运行的内核版本重新构建 `initramfs`：

```bash
sudo dracut --force -v /boot/initramfs-$(uname -r).img $(uname -r)
```

#### 第四步：重启验证

```bash
sudo reboot
```

重启后再次运行 `lsmod | grep nouveau`，此时应彻底干净无输出。

---

## 🖥️ 第三阶段：安装 NVIDIA 官方闭源驱动

确认 Nouveau 死亡后，开始安装官方驱动。

**离线资源：** 将驱动文件置于 `Base/` 目录，本项目已包含 `NVIDIA-Linux-x86_64-550.163.01.run`。

| 组件 | 文件名 | 下载链接 |
|------|--------|----------|
| NVIDIA 驱动 (V100 推荐) | `NVIDIA-Linux-x86_64-550.163.01.run` | [NVIDIA 数据中心驱动 550.163.01](https://www.nvidia.com/download/driverResults.aspx/243537/en-us/) · [驱动归档](https://developer.nvidia.com/datacenter-driver-archive) |

```bash
# 请将文件名替换为实际传输到服务器的 .run 文件（本项目为 550.163.01）
sudo ./NVIDIA-Linux-x86_64-550.163.01.run -no-x-check -no-nouveau-check -no-opengl-files
```

### 安装过程中的关键交互选项指南

| 安装程序提示 | 应答选择 | 原因说明 |
|-------------|----------|----------|
| *The distribution-provided pre-install script failed...* | **Continue installation** | 发行版自带脚本失败是正常的，直接跳过。 |
| *Register the kernel module sources with DKMS?* | **No** | 纯内网离线环境不折腾动态内核模块支持。 |
| *Install NVIDIA's 32-bit compatibility libraries?* | **No** | 现代 AI 模型和容器环境无需 32 位支持。 |
| *Run the nvidia-xconfig utility?* | **No** | V100 是纯计算卡，千万不要让它接管图形桌面环境。 |
| *Would you like to rebuild the initramfs?* | **Yes** | **极度重要！** 这会将防冲突配置和官方模块深层注入启动镜像，给驱动加上「开机即生效」的保险，彻底防止 Nouveau 重启诈尸。同时也确保内核在启动早期就能识别硬件 ID。 |

### 安装完成后验证

```bash
nvidia-smi
```

若成功输出显卡信息面板，驱动环节全部结束。

---

## 🐳 第四阶段：容器化环境构建

为确保稳定性并避开复杂的 RPM 依赖，我们采用 Docker 官方静态二进制包进行离线部署，并安装 NVIDIA Container Toolkit 以透传 GPU 给容器。

### 1. 准备离线资源

请在 `Base/Docker_Base` 目录下准备以下文件（或通过给定的外网链接提前下载备用）：

| 组件 | 文件名 | 下载链接 |
|------|--------|----------|
| Docker 静态包 | `docker-24.0.9.tgz` | [Docker 官方静态二进制](https://download.docker.com/linux/static/stable/x86_64/docker-24.0.9.tgz) |
| Docker Compose | `docker-compose-linux-x86_64` | [GitHub Releases](https://github.com/docker/compose/releases)（如 [v2.24.0](https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64)） |
| NVIDIA Container Toolkit | 见下表 | 见下表 |

**NVIDIA Container Toolkit 依赖包（4 个）：**

| 文件名 | 下载说明 |
|--------|----------|
| `libnvidia-container1-1.14.6-1.x86_64.rpm` | 配置 [NVIDIA 仓库](https://nvidia.github.io/libnvidia-container/) 后执行 `yumdownloader` 离线拉取，或从已有 `Base/Docker_Base` 获取 |
| `libnvidia-container-tools-1.14.6-1.x86_64.rpm` | 同上 |
| `nvidia-container-toolkit-base-1.14.6-1.x86_64.rpm` | 同上 |
| `nvidia-container-toolkit-1.14.6-1.x86_64.rpm` | 同上 |

> **离线获取 NVIDIA Container Toolkit：** 在有网环境配置 `https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo` 后，使用 `yum install --downloadonly` 或 `dnf download` 将 4 个 RPM 下载至 `Base/Docker_Base`。

### 2. 安装 Docker 与注册系统服务

```bash
cd /root/
# 解压二进制包
tar -zxvf docker-24.0.9.tgz

# 迁移可执行文件到系统路径
sudo cp docker/* /usr/bin/

# 注册为 Systemd 系统服务
cat <<EOF | sudo tee /etc/systemd/system/docker.service
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target firewalld.service
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/bin/dockerd
ExecReload=/bin/kill -s HUP \$MAINPID
LimitNOFILE=infinity
LimitNPROC=infinity
TimeoutStartSec=0
Delegate=yes
KillMode=process
Restart=on-failure
StartLimitBurst=3
StartLimitInterval=60s

[Install]
WantedBy=multi-user.target
EOF

# 启动 Docker 并设置开机自启
sudo systemctl daemon-reload
sudo systemctl start docker
sudo systemctl enable docker
```

### 3. 安装 Docker Compose

```bash
# 复制并重命名
sudo cp /root/docker-compose-linux-x86_64 /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

### 4. 安装 NVIDIA Container Toolkit 并配置运行时

此步骤用于打通 Docker 与 V100 物理硬件的壁垒。进入存放 4 个 RPM 包的目录执行：

```bash
# 批量离线安装工具包
sudo rpm -ivh *.rpm

# 配置 Docker 使用 NVIDIA 作为默认运行时
sudo nvidia-ctk runtime configure --runtime=docker

# 重启 Docker 服务使配置生效
sudo systemctl restart docker
```

至此，底层的系统、驱动与容器化环境已全部打通并固化。

---

## 🚀 第五阶段：Docker 镜像下载与推理服务部署

本阶段采用**双引擎架构**，**9** 个推理容器节点按模型系列使用不同引擎（另含 Nginx 网关）：

| 引擎 | 镜像来源 | 服务对象 | 选型原因 |
|------|----------|----------|----------|
| **Infinity** | Docker Hub | BGE-M3、BGE-Reranker-V2-M3 | 专为 BERT 系 Embedding/Reranker 深度优化，动态 Batching，高 QPS |
| **llama.cpp** | GitHub Container Registry | 所有 Qwen3 模型 | 原生 GGUF 量化支持，Qwen3 架构完整适配，部署最稳定 |

> **现有用户无感**：BGE 系列的 `/v1/embeddings`、`/v1/rerank` 路径完全不变；Qwen3 系列通过独立路径接入，两套引擎互不干扰。

---

### 1. 推理引擎简介

#### Infinity（BGE 系列）

**Infinity**（[michaelfeil/infinity](https://github.com/michaelfeil/infinity)）是专为高吞吐、低延迟的文本 Embedding 和 Reranking 设计的推理服务框架。

| 特性 | 说明 |
|------|------|
| **动态批处理** | 高并发请求自动合并为大 Batch 送入 GPU，压榨 V100 并行算力 |
| **专精检索模型** | 针对 BGE-M3 等 BERT 系模型深度优化，高 QPS |
| **OpenAI 兼容** | 暴露标准 `/v1/embeddings`、`/v1/rerank`，LangChain 等框架可直接对接 |

#### llama.cpp（Qwen3 系列）

**llama.cpp**（[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)）是 C++ 实现的高效推理引擎，对 Qwen3 架构有完整原生支持。

| 特性 | 说明 |
|------|------|
| **原生 GGUF** | llama.cpp 只支持 GGUF 格式，这是引擎的硬性要求，非量化选择 |
| **量化可选** | `F16` GGUF 零精度损失；`Q4_K_M` 约占原模型 25% 显存，按实际显存余量选择 |
| **Qwen3 全系支持** | 文本模型单 GGUF；VL Embedding / VL Reranker 需主 GGUF + mmproj（与 `docker-compose.yml` 一致） |
| **OpenAI 兼容** | Server 模式在容器内暴露 `/v1/embeddings`（`--embedding`）与 `/v1/rerank`（`--rerank`）；经网关的 Qwen 路径由 Nginx 转发到对应上游（见「API 路由说明」） |

---

### 2. 镜像下载（有网环境）

在 **Mac** 上拉取时需指定 `--platform linux/amd64`，以生成适用于内网 Linux 服务器的镜像：

```bash
# ── Infinity（BGE 系列引擎，约 9GB，Docker Hub）──────────────────────
docker pull --platform linux/amd64 michaelf34/infinity:latest

# ── llama.cpp CUDA 版（Qwen3 系列引擎，约 6GB，GitHub Container Registry）──
docker pull --platform linux/amd64 ghcr.io/ggml-org/llama.cpp:server-cuda

# ── Nginx 负载均衡 ────────────────────────────────────────────────────
docker pull --platform linux/amd64 nginx:latest
```

> **注意**：`ghcr.io`（GitHub Container Registry）与 Docker Hub 是不同的镜像源，需分别拉取，各自导出 tar。

### 3. 导出为 .tar 文件（便于 U 盘拷贝）

```bash
docker save -o infinity_latest_amd64.tar          michaelf34/infinity:latest
docker save -o llamacpp_server_cuda_amd64.tar     ghcr.io/ggml-org/llama.cpp:server-cuda
docker save -o nginx_latest_amd64.tar             nginx:latest
```

将 3 个 `.tar` 文件拷贝至 U 盘。

### 4. 内网服务器导入镜像

```bash
docker load -i infinity_latest_amd64.tar
docker load -i llamacpp_server_cuda_amd64.tar
docker load -i nginx_latest_amd64.tar
```

验证导入：

```bash
docker images | grep -E "infinity|llama|nginx"
```

---

### 5. 模型目录结构

所有模型放入 **`models/`** 目录（与 `docker-compose.yml` 中 `./models:/models` 一致）。当前仓库中的 **`docker-compose.yml` 采用「BGE 仍为子目录 + Qwen GGUF 放在 `models/` 根目录」的扁平命名**，便于与多源下载的文件名对齐，例如：

```
embed-deploy/
├── models/
│   ├── bge-m3/                               # BGE 文本向量化 ×3（HF，Infinity）
│   ├── bge-reranker-v2-m3/                   # BGE 文本重排序 ×2（HF，Infinity）
│   ├── Qwen3-Embedding-8B-Q4_K_M.gguf        # Qwen3 文本向量化 ×1（llama.cpp）
│   ├── Qwen3-VL-Embedding-2B-Q4_K_M.gguf     # VL 主模型
│   ├── Qwen3-VL-Embedding-2B-mmproj_f16.gguf # VL mmproj（文件名需与 compose 中 --mmproj 一致）
│   ├── Qwen3-Reranker-8B-Q4_K_M.gguf
│   ├── Qwen3-VL-Reranker-2B-Q4_K_M.gguf
│   └── Qwen3-VL-Reranker-2B-mmproj-_f16.gguf # 个别社区包文件名含额外连字符，以磁盘实际为准
├── web/
│   └── index.html
├── nginx/
│   └── nginx.conf
├── docker-compose.yml
└── README.md
```

若你更习惯「每个 Qwen 模型单独子目录」，可将 GGUF 放入子目录并**相应修改** `docker-compose.yml` 里的 `--model` / `--mmproj` 路径；只要容器内路径与文件一致即可。

> ⚠️ **文件名确认**：`docker-compose.yml` 中 `--model` 与 VL 服务的 `--mmproj` 路径必须与实际文件名一致（不同 Hugging Face 仓库命名可能不同）。若文件名不同，直接修改 `docker-compose.yml` 中对应行即可。

**llama.cpp 侧常用参数（已在 compose 中配置）**：文本与 VL 的 Embedding/Rerank 使用 **`--ctx-size 16384`**；两个 VL 服务额外使用 **`--pooling mean`**。

---

### 6. 启动服务

在项目根目录执行：

```bash
docker-compose up -d
```

所有节点显示为绿色（Up）即表示启动成功。Nginx 默认将宿主机 **80** 映射到容器内 **8080**（见 `docker-compose.yml` 的 `ports`），因此浏览器或 `curl` 可使用 **`http://<服务器IP>/`**（等价于 `:80`）。

查看各节点状态：

```bash
docker-compose ps
```

查看某节点启动日志（如排查 GGUF 文件名错误）：

```bash
docker logs embed-qwen-8b
docker logs embed-qwen-vl
docker logs reranker-qwen
docker logs reranker-qwen-vl
```

---

### 7. API 路由说明

由于各节点运行不同模型，Nginx 通过 **URL 路径** 区分模型，内部转发至对应引擎：

| 请求路径（对外） | 内部引擎 | 路由目标 | 对应模型 |
|-----------------|----------|----------|----------|
| `POST /v1/embeddings` | Infinity | bge_embed_pool（**3** 节点） | BGE-M3 |
| `POST /v1/qwen-embed` | llama.cpp | embed-qwen-8b（1 节点） | Qwen3-Embedding-8B |
| `POST /v1/qwen-vl-embed` | llama.cpp | embed-qwen-vl | Qwen3-VL-Embedding-2B |
| `POST /v1/rerank` | Infinity | bge_rerank_pool（2 节点） | BGE-Reranker-V2-M3 |
| `POST /v1/qwen-rerank` | llama.cpp | reranker-qwen | Qwen3-Reranker-8B |
| `POST /v1/qwen-vl-rerank` | llama.cpp | reranker-qwen-vl | Qwen3-VL-Reranker-2B |

**网关路径重写（`nginx/nginx.conf`）**：在 `location /v1/embeddings` 与 `location /v1/rerank` 内，先将 URI 重写为 `/embeddings`、`/rerank` 再转发到上游（`rewrite ... break`），以便与后端实际监听路径一致。Qwen 专用路径（`/v1/qwen-embed` 等）为 **location 前缀匹配**，仍按表中方式 `proxy_pass` 到各 llama.cpp 容器的 `/v1/embeddings` 或 `/v1/rerank`。

**VL 请求体（经联调约定）**：向量化使用 **`input`（字符串）** + **`image_data`（完整 Data URL）**；多模态重排使用 **`query`（字符串）** + **`image_data`** + **`documents`（字符串数组）**。详见仓库内 `web/index.html` 文档页示例。

---

### 8. 验证与调用

以下示例默认访问 **本机 Nginx**（`docker-compose` 中 **`80:8080`**，故使用 `http://localhost`；若你改为映射 `8080:8080`，请把主机名改为 `localhost:8080`）。

```bash
# BGE-M3 向量化（Infinity）
curl -X POST http://localhost/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "测试文本", "model": "bge-m3"}'

# Qwen3-Embedding-8B 向量化（llama.cpp）
curl -X POST http://localhost/v1/qwen-embed \
  -H "Content-Type: application/json" \
  -d '{"input": ["测试文本"], "model": "qwen3-embedding-8b"}'

# Qwen3-VL-Embedding-2B 多模态向量化（llama.cpp，扁平字段）
curl -X POST http://localhost/v1/qwen-vl-embed \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-Embedding-2B","input":"请结合图片与这句话生成向量：产品是否在画面中？","image_data":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}'

# BGE-Reranker 重排序（Infinity）
curl -X POST http://localhost/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "查询", "documents": ["文档1", "文档2"], "model": "bge-reranker-v2-m3"}'

# Qwen3-Reranker-8B 重排序（llama.cpp）
curl -X POST http://localhost/v1/qwen-rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "查询", "documents": ["文档1", "文档2"], "model": "qwen3-reranker-8b"}'

# Qwen3-VL-Reranker-2B 多模态重排序（llama.cpp，扁平字段）
curl -X POST http://localhost/v1/qwen-vl-rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-Reranker-2B","query":"用户上传的截图里有没有错误提示？","image_data":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==","documents":["界面显示「连接超时」，属于网络类报错。","今天天气不错，与截图无关。"]}'
```

### 9. 页端访问

启动后，在浏览器访问 **`http://<服务器IP>/`**（默认 **80** 端口）即可打开 **API 使用文档页**。文档源码在仓库 **`web/index.html`**；若你使用的 `docker-compose.yml` 将站点根目录挂载为其他路径（例如单文件 `nginx.html`），请保持挂载与编辑的源文件一致。页内提供 cURL、Requests、OpenAI SDK、Node.js、LangChain 等示例；示例域名为 `http://api-embed.cs.icbc`，实际部署时请替换为内网域名或 IP。

> **注意：** 若使用离线导入的镜像，需确保 `docker-compose.yml` 中的 `image` 名称与 `docker load` 后的镜像名一致。

---

## 📋 实施检查清单

| 阶段 | 检查项 | 验证命令 |
|------|--------|----------|
| 一 | 内核版本一致 | `uname -r` 与 `rpm -qa \| grep kernel-devel` 输出一致 |
| 二 | Nouveau 已禁用 | `lsmod \| grep nouveau` 无输出 |
| 三 | NVIDIA 驱动正常 | `nvidia-smi` 正常输出 |
| 四 | Docker 运行正常 | `docker run --rm hello-world` |
| 四 | GPU 透传可用 | `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi` |
| 五 | 推理服务启动 | `docker-compose ps` 全部 Up |
| 五 | 页端可访问 | 浏览器打开 `http://<IP>/`（默认端口 80） |
| 五 | BGE Embedding 可用 | `curl -X POST http://localhost/v1/embeddings ...` |
| 五 | Qwen Embedding 可用 | `curl -X POST http://localhost/v1/qwen-embed ...` |
| 五 | Qwen VL Embedding 可用 | `curl -X POST http://localhost/v1/qwen-vl-embed ...`（含 `image_data`） |
| 五 | BGE Rerank 可用 | `curl -X POST http://localhost/v1/rerank ...` |
| 五 | Qwen Rerank 可用 | `curl -X POST http://localhost/v1/qwen-rerank ...` |
| 五 | Qwen VL Rerank 可用 | `curl -X POST http://localhost/v1/qwen-vl-rerank ...`（含 `image_data`） |

---

*本文档适用于麒麟 V10 SP1.1 操作系统，在纯离线物理机环境中实施。任何接手的工程师均可按阶段顺序执行，知其然，更知其所以然。*

---

## 📥 下载资源总览

| 类别 | 本地路径 | 资源 | 下载链接 |
|------|----------|------|----------|
| **模型** | `models/bge-m3/` | BGE-M3 文本向量化 ×3（HF 格式） | [Hugging Face: BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| **模型** | `models/bge-reranker-v2-m3/` | BGE-Reranker-V2-M3 文本重排序 ×2（HF 格式） | [Hugging Face: BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| **模型** | `models/*.gguf`（见上节目录树） | Qwen3 系列 GGUF（Embedding、Rerank、VL+mmproj） | 见上文「模型清单与下载命令」 |
| **镜像** | `Images/` | Infinity 推理引擎（BGE 系列） | `docker pull --platform linux/amd64 michaelf34/infinity:latest` |
| **镜像** | `Images/` | llama.cpp CUDA Server（Qwen3 系列） | `docker pull --platform linux/amd64 ghcr.io/ggml-org/llama.cpp:server-cuda` |
| **镜像** | `Images/` | Nginx 负载均衡 | `docker pull --platform linux/amd64 nginx:latest` |
| 一 | `Base/Kernel_Base/` | kernel / kernel-devel / kernel-headers | [麒麟 V10SP1.1 Packages](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/) |
| 三 | `Base/` | NVIDIA-Linux-x86_64-550.163.01.run | [NVIDIA 数据中心驱动](https://www.nvidia.com/download/driverResults.aspx/243537/en-us/) |
| 四 | `Base/Docker_Base/` | docker-24.0.9.tgz | [Docker 静态包](https://download.docker.com/linux/static/stable/x86_64/docker-24.0.9.tgz) |
| 四 | `Base/Docker_Base/` | docker-compose-linux-x86_64 | [Docker Compose](https://github.com/docker/compose/releases) |
| 四 | `Base/Docker_Base/` | NVIDIA Container Toolkit (4×RPM) | [NVIDIA 仓库](https://nvidia.github.io/libnvidia-container/) + `yum download` |
