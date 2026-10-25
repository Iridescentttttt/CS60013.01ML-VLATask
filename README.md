# OpenVLA-7B LoRA × SimplerEnv × WidowX

机器学习课程 **B 类**作业。Base 是公开模型 `openvla/openvla-7b`，Final 是在 BridgeData V2 子集上用 LoRA 微调后的 v3。闭环评测使用 SimplerEnv 的 WidowX Visual Matching。

完整分析见 [`docs/实验报告.md`](docs/实验报告.md)，PDF 见 [`docs/实验报告.pdf`](docs/实验报告.pdf)。

## 最终结果

正式初始状态在 `config/eval_states_b_formal.json`：T1 为种子 1110–1119，T2 为种子 2210–2219。Base 与 Final 各评测 10 次。

| 模型 | T1 勺子放到毛巾 | T2 茄子放进篮子 | B-SimplerScore |
| --- | ---: | ---: | ---: |
| Base | 1/10 | 1/10 | 10.0 |
| Final（v3，step 300） | 0/10 | 5/10 | 25.0 |

相对 Base：T1 −10.0 个百分点，T2 +40.0 个百分点，平均分 +15.0 个百分点。提升集中在 T2。每个任务只有 10 次，这个差距还有波动。逐条记录在 `results/b_formal/episodes.csv`。

同一协议下还比较过 v1、v2 和一次全量数据探究。全量数据探究保留了与 v3 相同轨迹的全部时间步，结果为 0/20，没有替换 Final。

| 模型 | T1 | T2 | B-SimplerScore |
| --- | ---: | ---: | ---: |
| v1 | 0/10 | 0/10 | 0.0 |
| v2 | 0/10 | 2/10 | 10.0 |
| v3 / Final | 0/10 | 5/10 | 25.0 |
| 全量数据探究 | 0/10 | 0/10 | 0.0 |

## 怎么运行

在仓库根目录执行。脚本使用本机的两个环境：

- 模型：`/home/lizhenkai/anaconda3/envs/openvla/bin/python`
- 仿真：`/home/lizhenkai/anaconda3/envs/simpler_env/bin/python`

Base 权重不在仓库里，需要放到 `models/openvla-7b`，来源是 <https://huggingface.co/openvla/openvla-7b>。Final 的 LoRA 在 `checkpoints/openvla_bridge_lora_v3/best_adapter`，下载地址见 `checkpoint_link.txt`。Bridge 原始分片在 `data/`，不进入 Git；分片清单在 `artifacts/`。

复现 Base 与 Final 的正式评测，不重新训练：

```bash
bash scripts/run_b_formal.sh
python scripts/summarize_openvla_formal.py
```

重新训练 Final。默认即 v3：关键词轨迹、去掉静止帧、最多 1,200 步，按验证损失保存。训练结束后脚本会再评测一轮，那一轮不替代 `results/b_formal/`。

```bash
bash scripts/run_b_openvla_v3.sh
```

用同一套正式种子比较 Base、v1、v2、v3：

```bash
bash scripts/run_b_unified.sh
```

全量数据探究会保留静止帧并单独评测，不作为 Final：

```bash
bash scripts/run_b_openvla_v4.sh
```

## 实验要点

- **模型。** OpenVLA-7B 已在含 Bridge 的数据上训练。BridgeData V2 由 WidowX 250 采集。推理使用 `unnorm_key=bridge_orig`，贪心解码。
- **Final。** 只训练语言模型的 `q_proj` 和 `v_proj`，rank 8，alpha 16。可训练参数 4,194,304，约占 0.056%。学习率 1e-4，有效 batch 8，种子 42。验证损失每降 0.02 才保存；step 300 的验证损失为 3.478，随后早停。正式评测没有参与这次选择。
- **数据。** BridgeData V2，CC BY 4.0。种子 42 抽取 60 个训练分片和 12 个验证分片，再加 80 个只用于训练的分片。轨迹按官方 split 整段划分。v3 只保留指令中含 spoon、towel、eggplant、basket 的轨迹，并去掉平移、旋转和夹爪都几乎不动的静止帧。训练样本 12,802 步，来自 418 条轨迹；验证样本 535 步。动作前 6 维用 `bridge_orig` 的 q01/q99 归一化，夹爪保持 0/1。
- **评测。** SimplerEnv `eddb569`，控制频率 5 Hz，每回合最多 60 步。图像用 Lanczos 缩到 256×256。欧拉角增量转为轴角，夹爪按 0.5 阈值变成 +1 或 −1。成功以环境返回的 success 为准，超时记失败。开发种子 1100–1109 和 2200–2209 不进入正式结果。
- **观察。** v3 在 T2 上每次连续抓取都放进了篮子。T1 上 Base 有 1 次成功，v3 为 0/10。全量数据探究的 20 次都没有闭合夹爪，说明保留静止帧后策略几乎不再移动。

四个代表视频在 `results/b_formal/representative/`，覆盖两个任务，并包含失败样例。
