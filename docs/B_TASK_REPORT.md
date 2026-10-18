# Octo-Small 1.5 在 BridgeData V2 上的轻量微调与 SimplerEnv 评测方案

## 1. 项目概述

本项目选择课程 B 类任务。目标是以公开 VLA 模型 `rail-berkeley/octo-small-1.5` 为 Base，使用 BridgeData V2 的 WidowX 轨迹微调语言条件动作头，并在 SimplerEnv WidowX Visual Matching 的两个指定任务上比较 Base 与 Final：

- T1：`widowx_spoon_on_towel`；
- T2：`widowx_put_eggplant_in_basket`。

正式实验包含 `2 个模型 × 2 个任务 × 10 次 = 40` 个 episode。训练数据、开发状态和正式评测状态相互隔离；Final 只根据验证集损失选择。

## 2. 模型与 WidowX 训练证据

Octo 是基于 Transformer 的扩散机器人策略。Octo-Small 1.5 约有 2700 万参数，使用两帧观察历史，输出未来 4 步的 7 维动作。主相机输入为 256×256 RGB，语言由 T5 tokenizer 和 T5-Base 编码器处理。[Octo-Small 1.5 模型卡](https://huggingface.co/rail-berkeley/octo-small-1.5)

该模型卡列出的预训练数据混合中，Bridge 占训练 batch 的 17%。BridgeData V2 全部由 WidowX 250 机器人采集，因此这个 checkpoint 满足课程“必须使用过 WidowX 数据训练”的要求。Octo 官方仓库还提供了 WidowX 机器人评测和 `head_only`、`head_mlp_only`、`full` 三种微调方式。[Octo 官方仓库](https://github.com/octo-models/octo)

本项目固定：

| 项目 | 设置 |
| --- | --- |
| Base | `rail-berkeley/octo-small-1.5` |
| Base step | `300000` |
| 参数规模 | 27M |
| 输入历史 | 2 帧 |
| 动作分块 | 4 步 |
| 动作维数 | 7 |
| 推理统计量 | `bridge_dataset` |
| SimplerEnv policy setup | `widowx_bridge` |

## 3. SimplerEnv 已有结果与本实验关系

SimplerEnv 官方论文 Table V 报告，Octo-Small 在 WidowX Visual Matching 中，T1 的最终成功率为 47.2%，T2 为 56.9%；相应抓取率为 77.8% 和 87.5%。这些结果说明 Octo-Small 已有公开的 SimplerEnv 适配依据。[SimplerEnv 论文](https://arxiv.org/pdf/2405.05941)

论文结果采用的 episode 数、随机种子和统计方式与本课程不同，只作为运行合理性的参考。本项目的 Base 与 Final 必须在同一版本、同一组 10 个初始状态和相同控制设置下重新评测。

## 4. BridgeData V2

BridgeData V2 包含 60,096 条轨迹，其中 50,365 条为人工遥操作示范，9,731 条来自脚本 pick-and-place 策略；覆盖 24 个环境和 13 类技能。数据由 WidowX 250 采集，控制频率为 5Hz，平均轨迹长度约 38 步，原始图像为 640×480，每条轨迹带自然语言任务标注，数据许可为 CC BY 4.0。[BridgeData V2 官方页面](https://rail-berkeley.github.io/bridgedata/)

本项目使用 Berkeley 发布的新版 RLDS 数据，而非过时的 OXE Bridge 副本。完整 RLDS 约 124GB，超过当前实验空间，因此先进行固定分片试验。官方 RLDS 包含 1024 个训练分片和 128 个验证分片，每片约 90–130MB；服务器没有按语言或任务提供轨迹索引。[BridgeData V2 RLDS 目录](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/)

### 4.1 第一阶段分片试验

第一阶段采用固定随机种子 42，下载：

- 20 个训练分片；
- 4 个官方验证分片；
- 总预算不超过 3,000,000,000 bytes。

下载脚本把远端分片编号、原始文件名、本地文件名和字节数写入 `subset_manifest.json`。实际分片编号和大小由本次下载清单填写，确保其他合作者可以复现完全相同的数据。

本次实际下载结果：

```text
train 分片编号：51, 54, 61, 65, 178, 191, 209, 228, 285, 407,
               447, 451, 457, 476, 501, 563, 569, 859, 864, 919
val 分片编号：  1, 40, 87, 108
train 大小：    2,263,510,335 bytes
val 大小：        476,457,537 bytes
总大小：        2,739,967,872 bytes，即 2.552 GiB
```

完整逐文件记录保存在 `artifacts/data_report/subset_manifest.json`。

### 4.2 任务相关筛选

RLDS 分片混合存放不同任务，不能在下载前知道每片的语言组成。下载完成后按完整轨迹读取语言指令，并划分为：

1. `on_surface`：`put/place/move ... on/onto ... towel/cloth/plate/tray`；
2. `in_container`：`put/place ... in/into ... basket/bowl/pot/pan/container`；
3. `other`：保留在本次下载试验中用于统计，但不进入训练。

优先保留包含 spoon、towel、cloth、eggplant、basket 的轨迹；完全相同的对象组合不足时，使用相同空间关系和操作技能的其他物体轨迹。第一阶段的目的包括统计目标轨迹密度，由统计结果决定是否继续增加分片。

训练和验证沿用官方独立 split，不按帧重新随机划分，避免同一轨迹的相邻帧跨越两侧。

## 5. 数据字段与动作处理

Octo 官方 Bridge 微调配置采用以下字段：

| 原始字段 | Octo 字段 | 处理 |
| --- | --- | --- |
| `observation.image_0` | `image_primary` | resize 到 256×256 |
| `observation.state` | `proprio` | `bridge_dataset_transform` 映射 |
| `language_instruction` | `language_instruction` | 语言条件 |
| `action` | 7 维动作 | Bridge 标准化与动作重标记 |

前六维动作使用训练集均值和标准差归一化，夹爪维度不归一化：

```text
action_normalization_mask = [True, True, True, True, True, True, False]
```

推理时 SimplerEnv 的 `OctoInference` 使用相应统计量反归一化，将平移输出映射为 `world_vector`，把旋转增量转换为 axis-angle，并把夹爪值转换为 WidowX 控制命令。项目中的兼容层还把旧版 SimplerEnv 的 `pad_mask` 转为 Octo 当前要求的 `timestep_pad_mask`。

数据检查脚本输出：

- `data_manifest.csv`：split、episode 编号、类别、指令、轨迹步数；
- `summary.json`：轨迹数、总步数和类别统计；
- `sample_image.png`：一个样本图像；
- `sample.json`：对应指令、动作、轨迹长度和图像形状。

## 6. 轻量微调

本项目使用 Octo 官方的 `head_only,language_conditioned` 模式。冻结规则为 `octo_transformer.*`，训练扩散动作头及其非冻结参数。该设置实际更新可学习参数，符合课程对动作头或其他轻量模块的要求。

初始训练配置：

| 参数 | 数值 |
| --- | ---: |
| batch size | 16 |
| 历史窗口 | 2 |
| action horizon | 4 |
| 总步数 | 10,000 |
| 峰值学习率 | 1e-4 |
| warmup | 500 steps |
| 学习率 | cosine decay |
| weight decay | 0.01 |
| gradient clip | 1.0 |
| 训练 seed | 42 |
| 验证间隔 | 1,000 steps |
| 保存间隔 | 2,000 steps |

图像增强包括随机裁剪、亮度、对比度、饱和度和色调变化。训练日志使用 W&B offline 模式保存，不向外部服务发送实验数据；同时把训练 loss、验证 loss、学习率和梯度范数镜像到便于提交的 `logs/train_metrics.jsonl`。

候选 checkpoint 为 2k、4k、6k、8k、10k。Final 选择规则预先确定为：选择验证集扩散动作损失最低的 checkpoint；若差异很小则选择更早的 checkpoint。正式 SimplerEnv 结果不得用于选择 Final。

## 7. 正式评测协议

项目在 `config/eval_states_b.json` 中预先固定每项任务的 10 个环境 seed：

- T1：1100–1109；
- T2：2200–2209。

Base 和 Final 在独立进程中加载，使用相同状态、相同模型 seed 规则、5Hz 控制频率和 60 步上限。每个 episode 独立重置扩散随机数发生器，episode 的模型 seed 为 `base_seed + episode_id`。不进行人工动作干预，不删除失败轨迹；策略异常和超时按失败记录。

逐 episode 记录包括模型、checkpoint、任务、环境 seed、模型 seed、成功标记、控制步数、运行时间、环境统计和视频路径。

## 8. 结果指标

单任务成功率为成功数除以 10：

```text
B-SimplerScore = (T1 成功率 + T2 成功率) / 2
```

最终表格：

| 模型 | 训练模块 | T1 成功数 | T2 成功数 | B-SimplerScore |
| --- | --- | ---: | ---: | ---: |
| Base | 无本次微调 | /10 | /10 | 待评测 |
| Final | Octo diffusion action head | /10 | /10 | 待评测 |

报告同时给出 T1、T2 和 B-SimplerScore 的 Final−Base 百分点变化，并分析至少两个代表性 episode。分析把视频直接观察与可能原因分开陈述。

## 9. 复现流程与提交物

执行顺序为：下载固定分片、生成数据清单和样本、统计目标轨迹、动作头微调、按验证损失确定 Final、分别运行 Base/Final 正式评测、汇总 CSV 和选择代表视频。

公开仓库应包含依赖版本、数据清单、训练配置、训练日志、固定评测状态、Base/Final 命令、`results.csv`、`episodes.csv`、Final checkpoint 下载链接和 2–4 个视频。大体积 RLDS、模型权重和全部视频不进入 Git 仓库。

## 10. 局限

Octo-Small 已在包含 Bridge 的混合数据上预训练，本实验测量的是任务相关继续适配效果，不能表述为首次适配 WidowX。第一阶段只采样约 3GB RLDS 分片，轨迹分布可能不能代表完整 BridgeData V2；需要在清单生成后报告实际类别数量。课程正式评测每项仅 10 个 episode，较小的成功率变化具有较大随机波动。

## 参考资料

1. [Octo-Small 1.5 模型卡](https://huggingface.co/rail-berkeley/octo-small-1.5)
2. [Octo 官方仓库](https://github.com/octo-models/octo)
3. [Octo 官方微调配置](https://github.com/octo-models/octo/blob/main/scripts/configs/finetune_config.py)
4. [BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
5. [BridgeData V2 代码仓库](https://github.com/rail-berkeley/bridge_data_v2)
6. [SimplerEnv 仓库](https://github.com/simpler-env/SimplerEnv)
7. [SimplerEnv 论文](https://arxiv.org/pdf/2405.05941)
