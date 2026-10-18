# Octo-Small 1.5 × SimplerEnv × WidowX

本项目提供 Octo-Small 1.5 在 SimplerEnv WidowX 任务中的推理与评估代码。项目从本地模型目录加载检查点，通过兼容层连接 Octo 与 SimplerEnv，执行视觉语言控制策略，并输出逐回合记录、成功率汇总和视频。

课程 B 类的模型依据、BridgeData V2 分片方案、轻量微调设计和正式评测协议见 [`docs/B_TASK_REPORT.md`](docs/B_TASK_REPORT.md)。

## 1. 方案说明

| 项目 | 当前选择 |
| --- | --- |
| 模型 | `rail-berkeley/octo-small-1.5` |
| checkpoint step | `300000` |
| 机器人 | WidowX |
| 动作统计 | `bridge_dataset` |
| policy setup | `widowx_bridge` |
| 默认任务 | `widowx_spoon_on_towel` |
| 另一支持任务 | `widowx_put_eggplant_in_basket` |

当前流程用于评估预训练模型。`bridge_dataset` 的动作均值和标准差已经包含在 Octo-Small 1.5 的 `dataset_statistics.json` 中，因此推理阶段不需要下载完整 BridgeData RLDS 数据集。进行微调训练时才需要另外准备 RLDS 数据。

## 2. 推荐文件架构

将本项目、Octo、SimplerEnv 和模型放在同一个工作目录中：

```text
vla_workspace/
├── vlalora/                       # 本项目
│   ├── config/
│   │   └── nvidia_icd.json
│   ├── scripts/
│   │   ├── evaluate_octo_simpler.py
│   │   └── run_octo_simpler.sh
│   └── results/
├── SimplerEnv/                    # SimplerEnv 与 ManiSkill2_real2sim
├── octo/                          # Octo 源码
└── models/
    └── octo-small-1.5/
        ├── config.json
        ├── dataset_statistics.json
        ├── example_batch.msgpack
        └── 300000/default/checkpoint
```

启动脚本会根据自身位置自动确定 `vlalora` 和工作目录，不依赖固定的绝对路径。默认约定如下：

```text
PROJECT_DIR    = run_octo_simpler.sh 所在项目的根目录
WORKSPACE_DIR  = PROJECT_DIR 的父目录
SIMPLERENV_DIR = WORKSPACE_DIR/SimplerEnv
OCTO_DIR       = WORKSPACE_DIR/octo
MODEL_DIR      = WORKSPACE_DIR/models/octo-small-1.5
```

如果采用不同目录结构，可以在运行时覆盖这些变量：

```bash
WORKSPACE_DIR=/data/vla \
SIMPLERENV_DIR=/data/repos/SimplerEnv \
OCTO_DIR=/data/repos/octo \
MODEL_DIR=/data/models/octo-small-1.5 \
bash scripts/run_octo_simpler.sh
```

## 3. 已验证的软件组合

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| CUDA toolkit | 12.4 |
| NumPy | 1.24.4 |
| TensorFlow | 2.15.0 |
| JAX | 0.4.20 |
| jaxlib | 0.4.20+cuda12.cudnn89 |
| cuDNN Python wheel | 8.9.7.29 |
| Flax | 0.7.5 |
| Transformers | 4.55.4 |
| Gymnasium | 0.29.1 |
| SAPIEN | 2.2.2 |
| SimplerEnv | `c701ea271b16b80616d6fed9713c4a50a0adea3b` |
| ManiSkill2_real2sim | `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` |
| Octo | `241fb3514b7c40957a86d869fecb7c7fc353f540` |
| dlimp | `5edaa4691567873d495633f2708982b42edf1972` |

## 4. 下载代码

创建工作目录并下载四个仓库：

```bash
export WORKSPACE="$HOME/vla_workspace"
mkdir -p "$WORKSPACE"

git clone https://gitee.com/karlshaus/vlalora.git "$WORKSPACE/vlalora"

git clone --recurse-submodules \
  https://github.com/allenai/SimplerEnv.git \
  "$WORKSPACE/SimplerEnv"
git -C "$WORKSPACE/SimplerEnv" checkout c701ea271b16b80616d6fed9713c4a50a0adea3b
git -C "$WORKSPACE/SimplerEnv" submodule update --init --recursive

git clone https://github.com/octo-models/octo.git "$WORKSPACE/octo"
git -C "$WORKSPACE/octo" checkout 241fb3514b7c40957a86d869fecb7c7fc353f540

git clone https://github.com/kvablack/dlimp.git "$WORKSPACE/dlimp"
git -C "$WORKSPACE/dlimp" checkout 5edaa4691567873d495633f2708982b42edf1972
```

## 5. 安装依赖

以下命令使用当前 Python 3.10 环境。可以使用 Miniconda `base`，也可以使用单独的 Python 3.10 环境。

### 5.1 系统图形与视频库

```bash
sudo apt-get update
sudo apt-get install -y \
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

### 5.2 基础数值包

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-cache-dir \
  numpy==1.24.4 \
  scipy==1.12.0 \
  opencv-python==4.10.0.84 \
  protobuf==3.20.3
```

必须固定 `numpy==1.24.4`。仅写 `numpy<2.0` 可能让不同安装顺序产生不同的 NumPy 1.x 版本。

### 5.3 SimplerEnv

```bash
python -m pip install -e "$WORKSPACE/SimplerEnv/ManiSkill2_real2sim"
python -m pip install -e "$WORKSPACE/SimplerEnv"
```

### 5.4 TensorFlow 与数据处理包

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

`tensorflow-metadata==1.15.0` 与 `protobuf==3.20.3` 必须成对固定。使用不匹配的版本会在导入 TFDS 时出现：

```text
cannot import name 'runtime_version' from google.protobuf
```

### 5.5 GPU 版 JAX

```bash
python -m pip install --no-cache-dir jax==0.4.20

python -m pip install --no-cache-dir --no-deps \
  'https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.20%2Bcuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl'

python -m pip install --no-cache-dir \
  nvidia-cublas-cu12==12.9.2.10 \
  nvidia-cuda-nvrtc-cu12==12.9.86 \
  nvidia-cudnn-cu12==8.9.7.29
```

启动脚本会根据当前 Python 的 `site-packages` 自动找到 NVIDIA 动态库，不依赖某个 Miniconda 安装路径。

### 5.6 Octo 依赖

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

python -m pip install -e "$WORKSPACE/dlimp"
python -m pip install -e "$WORKSPACE/octo" --no-deps
```

安装结束后再次固定容易被间接依赖改写的包：

```bash
python -m pip install --no-cache-dir --no-deps \
  numpy==1.24.4 \
  protobuf==3.20.3 \
  tensorflow-metadata==1.15.0
```

当前推理脚本不需要以下训练依赖：

```text
tf-agents
dm-reverb
apache-beam
tfds-nightly
tfp-nightly
wandb
```

需要进行微调时，再为训练代码建立相应的依赖组合。

本项目的训练脚本使用 W&B 离线日志。当前 TensorFlow/Protobuf 组合对应：

```bash
python -m pip install wandb==0.16.6
python -m pip install gym==0.26.2
```

不要安装 `wandb==0.28.0`；它与这里固定的 `protobuf==3.20.3` 组合无法正常导入。`gym==0.26.2` 由 Octo 官方训练可视化回调导入，SimplerEnv 仍使用 Gymnasium。

## 6. 下载模型

将 Octo-Small 1.5 下载到文件架构中的模型目录：

```bash
mkdir -p "$WORKSPACE/models"

MODEL_DIR="$WORKSPACE/models/octo-small-1.5" python -c \
  "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='rail-berkeley/octo-small-1.5', local_dir=os.environ['MODEL_DIR'])"
```

Octo 还需要 T5 tokenizer 和配置。首次运行时会自动缓存，也可以提前下载：

```bash
python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='t5-base', allow_patterns=['config.json','tokenizer_config.json','special_tokens_map.json','spiece.model','tokenizer.json'])"
```

模型目录应至少包含：

```text
octo-small-1.5/
├── config.json
├── dataset_statistics.json
├── example_batch.msgpack
└── 300000/
    └── default/
        └── checkpoint
```

## 7. 本项目的适配内容

### 7.1 从本地文件架构加载模型

评估代码使用：

```python
OctoModel.load_pretrained(checkpoint_path, step=300000)
```

模型路径由 `--checkpoint` 提供。启动脚本默认使用工作目录中的 `models/octo-small-1.5`，也可以通过 `MODEL_DIR` 覆盖。

### 7.2 WidowX 动作适配

创建策略时使用：

```python
OctoInference(
    model=model,
    dataset_id="bridge_dataset",
    policy_setup="widowx_bridge",
)
```

这一设置完成以下处理：

- 使用 `bridge_dataset` 的统计量反归一化动作。
- 将平移输出映射为 `world_vector`。
- 将欧拉角旋转增量转换为 axis-angle `rot_axangle`。
- 将夹爪输出转换成 WidowX 环境使用的 `-1/1` 命令。

### 7.3 SimplerEnv 与 Octo 接口适配

当前 SimplerEnv 的 `OctoInference` 使用旧字段 `pad_mask`，当前 Octo 的 `sample_actions` 需要 `timestep_pad_mask`。直接组合会出现：

```text
KeyError: 'timestep_pad_mask'
```

`scripts/evaluate_octo_simpler.py` 中的 `SimplerOctoCompatibility` 会将 mask 作为 `timestep_pad_mask=` 参数传给 Octo。该兼容层是当前两个固定提交能够共同运行的必要部分。

### 7.4 文件路径适配

`scripts/run_octo_simpler.sh` 会自动计算：

- 项目目录；
- SimplerEnv 源码目录；
- Octo 源码目录；
- Octo-Small 模型目录；
- 当前 Python 的 NVIDIA 动态库目录。

因此仓库可以放在任意工作目录中。需要特殊布局时使用 `WORKSPACE_DIR`、`SIMPLERENV_DIR`、`OCTO_DIR` 和 `MODEL_DIR` 显式指定。

### 7.5 Vulkan 与 GPU 适配

`config/nvidia_icd.json` 指向 `libGLX_nvidia.so.0`，用于让 SAPIEN renderer 选择 NVIDIA Vulkan 驱动。启动脚本同时设置：

```bash
VK_ICD_FILENAMES="$PROJECT_DIR/config/nvidia_icd.json"
JAX_PLATFORMS=cuda
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

不要将 `VK_ICD_FILENAMES` 指向 Mesa 的 `lvp_icd.x86_64.json`。当前 SAPIEN 版本与 llvmpipe renderer 不兼容。

## 8. 使用模型

### 8.1 默认运行一次

```bash
cd "$WORKSPACE/vlalora"
bash scripts/run_octo_simpler.sh
```

默认执行：

```text
checkpoint: models/octo-small-1.5
checkpoint step: 300000
task: widowx_spoon_on_towel
episodes: 1
seed: 0
output: results/smoke_spoon
```

### 8.2 运行 spoon-on-towel

```bash
cd "$WORKSPACE/vlalora"

bash scripts/run_octo_simpler.sh \
  --checkpoint "$WORKSPACE/models/octo-small-1.5" \
  --checkpoint-step 300000 \
  --task widowx_spoon_on_towel \
  --episodes 10 \
  --seed 0 \
  --output-dir results/base_spoon
```

### 8.3 运行 eggplant-in-basket

```bash
cd "$WORKSPACE/vlalora"

bash scripts/run_octo_simpler.sh \
  --checkpoint "$WORKSPACE/models/octo-small-1.5" \
  --checkpoint-step 300000 \
  --task widowx_put_eggplant_in_basket \
  --episodes 10 \
  --seed 0 \
  --output-dir results/base_eggplant
```

`--tf-memory-limit` 默认值为 3072 MB，可以根据可用显存修改：

```bash
bash scripts/run_octo_simpler.sh \
  --checkpoint "$WORKSPACE/models/octo-small-1.5" \
  --task widowx_spoon_on_towel \
  --episodes 1 \
  --tf-memory-limit 4096 \
  --output-dir results/smoke_4gb
```

## 9. 输出文件

每个评估目录包含：

```text
results/example_run/
├── episode_000_success_True.mp4
├── episode_001_success_False.mp4
├── episodes.jsonl
└── summary.json
```

`episodes.jsonl` 记录每回合的：

- 语言指令；
- 是否成功；
- 控制步数；
- 运行时间；
- 环境初始化信息；
- SimplerEnv 的 episode statistics；
- 视频路径。

`summary.json` 记录：

- checkpoint 路径和 step；
- 任务名称；
- 总回合数；
- 成功回合数；
- 成功率；
- 随机种子。

生成 `summary.json`、`episodes.jsonl` 和 MP4 表示模型加载、策略推理、环境控制、渲染和视频保存流程已经完成。`success: false` 表示该回合没有完成任务，不代表程序执行失败。

## 10. 关键兼容问题

| 现象 | 原因 | 处理方式 |
| --- | --- | --- |
| NumPy 2.x 导致导入错误 | TensorFlow 2.15 和当前依赖组合不支持 NumPy 2.x | 固定 `numpy==1.24.4` |
| `cannot import name runtime_version` | tensorflow-metadata 与 protobuf 不匹配 | 固定 metadata 1.15.0 和 protobuf 3.20.3 |
| `libvulkan.so.1` 不存在 | 缺少 Vulkan loader | 安装 `libvulkan1` |
| `enumeratePhysicalDevices` 失败 | Vulkan ICD 未指向 NVIDIA 驱动 | 使用 `config/nvidia_icd.json` |
| Vulkan incompatible 或 segmentation fault | 使用了 llvmpipe | 使用 NVIDIA ICD，不使用 `lvp_icd` |
| JAX 显示 `[CpuDevice(id=0)]` | cuDNN 未安装或动态库路径缺失 | 安装 NVIDIA wheels，并通过启动脚本设置路径 |
| `KeyError: timestep_pad_mask` | SimplerEnv 与 Octo 接口版本不同 | 保留 `SimplerOctoCompatibility` |
| 提示缺少 `image_wrist` 等 observation | 检查点包含额外训练模态 | 当前主相机推理会跳过未提供的可选模态 |
| `GLFW: Failed to open display` | headless 渲染没有桌面窗口 | SAPIEN 继续使用 Vulkan 渲染时该提示不影响执行 |

## 11. 合作者使用顺序

1. 按推荐文件架构下载 `vlalora`、`SimplerEnv`、`octo` 和 `dlimp`。
2. 固定三个上游项目的提交号。
3. 按本文版本安装 TensorFlow、JAX、Octo 和 SimplerEnv 依赖。
4. 将 Octo-Small 1.5 放入 `models/octo-small-1.5`。
5. 始终通过 `scripts/run_octo_simpler.sh` 启动，使路径、GPU 和 Vulkan 配置统一生效。
6. 根据实验任务设置回合数、随机种子和结果目录。

## 12. BridgeData V2 分片试验

完整 BridgeData V2 RLDS 约 124GB。本项目先用种子 42 固定抽取 20 个训练分片和 4 个验证分片，并限制总下载量不超过 3GB：

```bash
cd "$WORKSPACE/vlalora"
python scripts/download_bridge_subset.py \
  --output-root "$WORKSPACE/data/bridge_course_subset" \
  --train-shards 20 \
  --val-shards 4 \
  --seed 42 \
  --max-bytes 3000000000
```

分片来源编号保存在：

```text
$WORKSPACE/data/bridge_course_subset/bridge_dataset/1.0.0/subset_manifest.json
```

生成语言清单、类别统计和课程要求的图像/指令/动作样本：

```bash
python scripts/inspect_bridge_subset.py \
  --data-dir "$WORKSPACE/data/bridge_course_subset" \
  --output-dir artifacts/data_report
```

当前固定分片的实际结果为 1,252 条轨迹、46,479 个时间步。语言筛选得到 188 条训练轨迹（85 条表面放置、103 条容器放置）和 48 条验证轨迹（19 条、29 条）。其余 1,016 条轨迹只保留在下载试验中，不进入动作头训练。

## 13. 动作头微调

训练配置为 `head_only,language_conditioned`，冻结 `octo_transformer.*`，使用两帧历史和四步动作分块：

```bash
DATA_DIR="$WORKSPACE/data/bridge_course_subset" \
MODEL_DIR="$WORKSPACE/models/octo-small-1.5" \
SAVE_DIR="$WORKSPACE/checkpoints" \
bash scripts/run_finetune.sh
```

统计总参数、冻结参数和可训练参数：

```bash
python scripts/count_trainable_params.py \
  --checkpoint "$WORKSPACE/models/octo-small-1.5" \
  --checkpoint-step 300000
```

## 14. B 类正式评测

Final 根据验证损失确定后，设置其目录和 step。脚本会分别启动 Base 和 Final，在两个任务上各运行 10 次：

```bash
FINAL_MODEL_DIR="$WORKSPACE/checkpoints/<selected-run>" \
FINAL_STEP=8000 \
bash scripts/run_b_eval.sh

python scripts/summarize_b_results.py \
  --results-dir results/formal \
  --output-dir artifacts/results
```

固定的正式初始状态在 `config/eval_states_b.json`。Base 与 Final 使用相同环境 seed、模型 seed 规则、5Hz 控制频率和 60 步上限。
