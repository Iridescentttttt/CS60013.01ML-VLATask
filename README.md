# Octo-Small 1.5 在 SimplerEnv WidowX 中的部署与评估

本仓库提供一套已经在云端 NVIDIA A16 上实际跑通的 Octo-Small 1.5 推理流程。它从本地检查点加载模型，在 SimplerEnv 的 WidowX 任务中执行策略，并保存逐回合记录、汇总结果和 MP4 视频。

本文档面向第一次接手项目的合作者，包含从空白 Ubuntu 22.04、Python 3.10 和 CUDA 12.4 镜像开始的完整配置步骤。

## 1. 当前方案

| 项目 | 选择 |
| --- | --- |
| 模型 | `rail-berkeley/octo-small-1.5` |
| 检查点 step | `300000` |
| 机器人 | WidowX |
| Octo 动作统计 | `bridge_dataset` |
| SimplerEnv policy setup | `widowx_bridge` |
| 默认任务 | `widowx_spoon_on_towel` |
| 另一个支持的任务 | `widowx_put_eggplant_in_basket` |
| 模型目录 | `/root/models/octo-small-1.5` |
| 运行结果目录 | `/root/vlalora/results` |

当前代码执行的是预训练 Octo-Small 1.5 的仿真评估，不需要单独下载 BridgeData RLDS 训练集。WidowX 所需的 `bridge_dataset` 动作均值和标准差已经保存在模型的 `dataset_statistics.json` 中。只有继续做微调时才需要准备 RLDS 数据集。

## 2. 已验证的平台和版本

以下组合已于 2026-10-16 在云端完整执行一回合。

| 组件 | 已验证版本 |
| --- | --- |
| Ubuntu | 22.04.4 LTS |
| Python | 3.10.16，Miniconda 默认 `base` 环境 |
| CUDA toolkit | 12.4 |
| NVIDIA GPU | A16，驱动 580.65.06 |
| NumPy | 1.24.4 |
| TensorFlow | 2.15.0 |
| JAX | 0.4.20 |
| jaxlib | 0.4.20+cuda12.cudnn89 |
| cuDNN Python wheel | 8.9.7.29 |
| Flax | 0.7.5 |
| Transformers | 4.55.4 |
| SimplerEnv commit | `c701ea271b16b80616d6fed9713c4a50a0adea3b` |
| ManiSkill2_real2sim commit | `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` |
| Octo commit | `241fb3514b7c40957a86d869fecb7c7fc353f540` |

## 3. 最终目录结构

不要把仓库、模型和运行结果放到只读的 `/root/eb-public`。当前服务器的可写目录布局如下：

```text
/root/
├── SimplerEnv/                    # SimplerEnv 与 ManiSkill2_real2sim 子模块
├── octo/                          # Octo 源码
├── models/
│   └── octo-small-1.5/            # Hugging Face 模型检查点
└── vlalora/                       # 本 Gitee 项目
    ├── config/nvidia_icd.json
    ├── scripts/evaluate_octo_simpler.py
    ├── scripts/run_cloud_once.sh
    └── results/
```

`/root` 位于当前容器的 30 GB overlay 文件系统中。Octo-Small 检查点约 523 MB；安装 TensorFlow、JAX、cuDNN、cuBLAS、Vulkan 和 SimplerEnv 后需要预留数 GB 空间。

## 4. 从零部署

以下命令均在服务器的 Miniconda 默认 `base` 环境中执行，不新建 conda 环境。

### 4.1 安装系统图形库

SAPIEN 需要 Vulkan、EGL 和视频编码库：

```bash
apt-get update
apt-get install -y \
  libvulkan1 \
  vulkan-tools \
  libegl1 \
  libgl1 \
  libglvnd-dev \
  libxext6 \
  libxrender1 \
  libsm6 \
  ffmpeg
```

### 4.2 下载三个代码仓库

先下载本项目：

```bash
git clone https://gitee.com/karlshaus/vlalora.git /root/vlalora
```

该云端无法稳定直连 GitHub。下面使用本次实际成功的 `ghfast.top` 前缀下载官方仓库，并固定到已经验证的提交：

```bash
git clone --recurse-submodules \
  https://ghfast.top/https://github.com/allenai/SimplerEnv.git \
  /root/SimplerEnv

git -C /root/SimplerEnv checkout c701ea271b16b80616d6fed9713c4a50a0adea3b
git -C /root/SimplerEnv submodule update --init --recursive

git clone \
  https://ghfast.top/https://github.com/octo-models/octo.git \
  /root/octo

git -C /root/octo checkout 241fb3514b7c40957a86d869fecb7c7fc353f540
```

如果所在机器可以直连 GitHub，只需从上述 URL 中删除 `https://ghfast.top/` 前缀。

### 4.3 配置 Python 镜像并固定基础数值包

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-cache-dir \
  numpy==1.24.4 \
  scipy==1.12.0 \
  opencv-python==4.10.0.84 \
  protobuf==3.20.3
```

必须使用 `numpy==1.24.4`，不要只写 `numpy<2.0`，也不要让后续安装把它自动升级到 NumPy 2.x。

### 4.4 安装 SimplerEnv

```bash
python -m pip install -e /root/SimplerEnv/ManiSkill2_real2sim
python -m pip install -e /root/SimplerEnv
```

当前实际使用的关键仿真包版本为：

```text
gymnasium==0.29.1
sapien==2.2.2
coacd==1.0.14
```

### 4.5 安装 TensorFlow 和数据处理依赖

```bash
python -m pip install --no-cache-dir \
  numpy==1.24.4 \
  tensorflow==2.15.0 \
  tensorflow-datasets==4.9.4 \
  tensorflow-hub==0.16.0 \
  tensorflow-metadata==1.15.0 \
  tensorflow-probability==0.23.0 \
  tensorflow-text==2.15.0 \
  tensorflow-graphics==2021.12.3 \
  tf-keras==2.15.0 \
  rlds==0.1.8 \
  protobuf==3.20.3 \
  mediapy==1.2.0 \
  IPython==8.22.2
```

这里固定 `tensorflow-metadata==1.15.0` 和 `protobuf==3.20.3`。安装最新的 `tensorflow-metadata` 会在导入 TFDS 时产生 `cannot import name 'runtime_version' from google.protobuf`。

本推理流程不需要安装以下训练依赖：

```text
tf-agents
dm-reverb
apache-beam
tfds-nightly
tfp-nightly
wandb
```

这些包会引入额外的 TensorFlow、JAX 和 protobuf 约束，是此前依赖解析失败的主要来源。当前推理脚本不导入 wandb；需要进行微调训练时再为训练环境选择与 protobuf 约束相容的 wandb 版本。

### 4.6 安装 GPU 版 JAX

先安装与 Octo 当前代码匹配的 JAX，再单独安装 CUDA 12 jaxlib wheel：

```bash
python -m pip install --no-cache-dir jax==0.4.20

python -m pip install --no-cache-dir --no-deps \
  'https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.20%2Bcuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl'
```

基础镜像虽然包含 CUDA 12.4 和 NVIDIA 驱动，但没有完整 cuDNN。继续安装：

```bash
python -m pip install --no-cache-dir \
  nvidia-cublas-cu12==12.9.2.10 \
  nvidia-cuda-nvrtc-cu12==12.9.86 \
  nvidia-cudnn-cu12==8.9.7.29
```

这些 wheel 的动态库位于：

```text
/root/miniconda3/lib/python3.10/site-packages/nvidia/cublas/lib
/root/miniconda3/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
/root/miniconda3/lib/python3.10/site-packages/nvidia/cudnn/lib
```

`scripts/run_cloud_once.sh` 会自动把它们加入 `LD_LIBRARY_PATH`，并设置 `JAX_PLATFORMS=cuda`。

### 4.7 安装 Octo 依赖和源码

```bash
python -m pip install --no-cache-dir \
  flax==0.7.5 \
  chex==0.1.85 \
  optax==0.1.5 \
  distrax==0.1.5 \
  ml-dtypes==0.2.0 \
  einops \
  ml-collections \
  transformers==4.55.4 \
  tokenizers==0.21.4 \
  huggingface-hub==0.34.4 \
  plotly==7.1.0

python -m pip install --no-deps \
  orbax-checkpoint==0.4.8 \
  tensorstore==0.1.45

git clone \
  https://ghfast.top/https://github.com/kvablack/dlimp.git \
  /root/dlimp
git -C /root/dlimp checkout 5edaa4691567873d495633f2708982b42edf1972
python -m pip install -e /root/dlimp

python -m pip install -e /root/octo --no-deps
```

上述 dlimp 提交就是当前云端实际安装的版本。这里也使用 GitHub 代理，避免 `pip install git+https://github.com/...` 在克隆阶段超时。

最后再次固定几个容易被间接依赖改写的包：

```bash
python -m pip install --no-cache-dir --no-deps \
  numpy==1.24.4 \
  protobuf==3.20.3 \
  tensorflow-metadata==1.15.0
```

### 4.8 下载 Octo-Small 1.5

```bash
mkdir -p /root/models

HF_ENDPOINT=https://hf-mirror.com python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='rail-berkeley/octo-small-1.5', local_dir='/root/models/octo-small-1.5')"
```

下载成功后至少应包含：

```text
/root/models/octo-small-1.5/config.json
/root/models/octo-small-1.5/dataset_statistics.json
/root/models/octo-small-1.5/example_batch.msgpack
/root/models/octo-small-1.5/300000/default/checkpoint
```

其中 checkpoint 文件约 546 MB。

Octo 还需要 T5 tokenizer 和配置。运行脚本会通过 `HF_ENDPOINT=https://hf-mirror.com` 自动下载，也可以预先缓存：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='t5-base', allow_patterns=['config.json','tokenizer_config.json','special_tokens_map.json','spiece.model','tokenizer.json'])"
```

## 5. 本项目做了哪些适配

### 5.1 本地模型加载

`scripts/evaluate_octo_simpler.py` 不让 SimplerEnv 在线下载模型，而是执行：

```python
OctoModel.load_pretrained('/root/models/octo-small-1.5', step=300000)
```

因此模型下载完成后可以重复离线使用。

### 5.2 WidowX 与 Bridge 动作统计

策略适配器固定为：

```python
dataset_id='bridge_dataset'
policy_setup='widowx_bridge'
```

这会使用 Bridge 数据中的动作均值和标准差反归一化 Octo 输出，并执行以下转换：

- 三维末端位移直接映射为 `world_vector`。
- 欧拉角旋转增量转换为 axis-angle `rot_axangle`。
- Octo 的开合夹爪输出二值化为 WidowX 使用的 `-1/1` 命令。

### 5.3 SimplerEnv 与当前 Octo 的 pad mask 接口

当前 SimplerEnv 的 `OctoInference` 传入旧字段 `pad_mask`，当前 Octo 的 `sample_actions` 接口需要 `timestep_pad_mask`。如果直接组合会出现：

```text
KeyError: 'timestep_pad_mask'
```

本项目的 `SimplerOctoCompatibility` 包装器会移除旧字段，并通过 `timestep_pad_mask=` 参数传给 Octo。不要删除这个兼容层。

### 5.4 无显示器 Vulkan 渲染

SAPIEN 需要 Vulkan 才能创建相机和场景。服务器没有桌面显示器，因此运行脚本会设置：

```bash
DISPLAY=""
XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
VK_ICD_FILENAMES=/root/vlalora/config/nvidia_icd.json
```

`config/nvidia_icd.json` 指向容器已有的 `libGLX_nvidia.so.0`，让 SAPIEN 使用 NVIDIA A16。

不要把 `VK_ICD_FILENAMES` 指向 `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`。llvmpipe 可以被 `vulkaninfo` 识别，但与当前 SAPIEN renderer 不兼容，会报错后 segmentation fault。

### 5.5 GPU 动态库路径

运行脚本把 pip 安装的 cuDNN、cuBLAS、NVRTC 和系统 CUDA 加入 `LD_LIBRARY_PATH`。如果缺少这一步，JAX 会回退到 CPU，并显示：

```text
Found cuDNN version 0
[CpuDevice(id=0)]
```

正确配置后 JAX 使用 `cuda(id=0)`，TensorFlow 也能创建 NVIDIA A16 GPU device。

## 6. 运行模型

### 6.1 默认 smoke run

```bash
cd /root/vlalora
git pull --ff-only
bash scripts/run_cloud_once.sh
```

不传参数时，脚本运行一次 `widowx_spoon_on_towel`，结果写入：

```text
/root/vlalora/results/smoke_spoon
```

### 6.2 自定义任务和回合数

启动脚本现在会把命令行参数直接转发给 Python 评估程序，因此仍会自动配置 Vulkan、Hugging Face 镜像和 GPU 动态库。

运行 10 回合 spoon-on-towel：

```bash
bash /root/vlalora/scripts/run_cloud_once.sh \
  --checkpoint /root/models/octo-small-1.5 \
  --checkpoint-step 300000 \
  --task widowx_spoon_on_towel \
  --episodes 10 \
  --seed 0 \
  --output-dir /root/vlalora/results/base_spoon
```

运行 10 回合 eggplant-in-basket：

```bash
bash /root/vlalora/scripts/run_cloud_once.sh \
  --checkpoint /root/models/octo-small-1.5 \
  --checkpoint-step 300000 \
  --task widowx_put_eggplant_in_basket \
  --episodes 10 \
  --seed 0 \
  --output-dir /root/vlalora/results/base_eggplant
```

`--tf-memory-limit` 的默认值为 3072 MB。需要调整 TensorFlow 显存上限时追加，例如：

```bash
--tf-memory-limit 4096
```

## 7. 输出文件

每个输出目录包含：

```text
episode_000_success_True.mp4       # 或 success_False
episode_001_success_False.mp4
episodes.jsonl                     # 每回合指令、成功状态、步数、耗时和场景信息
summary.json                       # 总回合数、成功数和成功率
```

一次完整运行结束并生成 `summary.json` 和 MP4，表示模型加载、Octo 推理、环境控制、Vulkan 渲染和视频编码链路已经跑通。单个随机回合的 `success: false` 表示机器人没有完成该回合任务，不表示程序运行失败。

## 8. 已验证运行记录

验证任务：

```text
task: widowx_spoon_on_towel
instruction: put the spoon on the towel
episodes: 1
control_steps: 60
elapsed_seconds: 85.435
GPU: NVIDIA A16
```

该回合成功数为 0，但完整执行了 60 个控制步，并生成：

```text
/root/vlalora/results/smoke_spoon/episode_000_success_False.mp4
/root/vlalora/results/smoke_spoon/episodes.jsonl
/root/vlalora/results/smoke_spoon/summary.json
```

## 9. 常见错误与对应处理

| 现象 | 原因 | 本项目采用的处理 |
| --- | --- | --- |
| `git clone` 连接 GitHub 443 超时 | 云端无法直连 GitHub | 使用 `https://ghfast.top/https://github.com/...` |
| NumPy 2.x 导致 TensorFlow/JAX 兼容问题 | 依赖解析自动升级 NumPy | 开始和结束都固定 `numpy==1.24.4` |
| `cannot import name runtime_version from google.protobuf` | 最新 tensorflow-metadata 与 protobuf 版本不匹配 | 固定 `tensorflow-metadata==1.15.0`、`protobuf==3.20.3` |
| `libvulkan.so.1` 不存在 | 没有 Vulkan loader | apt 安装 `libvulkan1` 和相关图形库 |
| `enumeratePhysicalDevices: ErrorInitializationFailed` | 容器没有 NVIDIA ICD JSON | 使用仓库内 `config/nvidia_icd.json` |
| Vulkan incompatible 后 segmentation fault | 错用了 Mesa llvmpipe | 使用 `libGLX_nvidia.so.0` 对应的 NVIDIA ICD |
| JAX 只显示 CPU | 缺少 cuDNN 或动态库路径 | 安装 NVIDIA Python wheels，并通过启动脚本设置 `LD_LIBRARY_PATH` |
| 无法访问 `huggingface.co/t5-base` | Hugging Face 主站不可达 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `KeyError: timestep_pad_mask` | SimplerEnv 和 Octo API 版本不同 | 保留 `SimplerOctoCompatibility` 包装器 |
| `GLFW error: X11: Failed to open display` | 无显示器服务器 | 这是 headless 模式的预期提示；后续出现 `Continue without GLFW` 即可 |
| `Unable to register cuDNN/cuFFT/cuBLAS factory` | TensorFlow 组件重复注册日志 | 本次运行中为非致命日志，不影响后续 GPU 创建和推理 |
| 提示缺少 `image_wrist`、`task_completed` 等 observation | Octo 检查点保存了更多训练模态 | 当前单主相机推理会跳过缺失的可选模态 |

## 10. 合作者最短操作清单

已经配置好服务器后，合作者只需：

```bash
cd /root/vlalora
git pull --ff-only
bash scripts/run_cloud_once.sh
```

在新服务器上则按顺序完成：

1. 安装 Vulkan/EGL/FFmpeg 系统包。
2. 下载本项目、固定提交的 SimplerEnv 和 Octo。
3. 按本文固定版本安装 NumPy、TensorFlow、JAX 和 Octo 依赖。
4. 安装 cuDNN、cuBLAS 和 NVRTC Python wheels。
5. 将 Octo-Small 1.5 下载到 `/root/models/octo-small-1.5`。
6. 始终通过 `scripts/run_cloud_once.sh` 启动评估，使 GPU、Vulkan 和镜像设置自动生效。
