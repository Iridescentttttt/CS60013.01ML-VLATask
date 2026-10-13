方案 B 规划：用 OpenVLA 做 WidowX 轻量微调与 SimplerEnv 评测
结合项目名 vlalora 和课程要求，我建议把方案 B 收敛为：用 BridgeData V2 的 WidowX 轨迹对 OpenVLA 做 LoRA 微调，再在 SimplerEnv 的两个 WidowX 任务上比较 Base 与 Final。 当前仓库里还没有项目代码，因此可以按下面的结构开始搭建。
1. 模型和数据
- 基础模型： OpenVLA 官方 openvla-7b checkpoint。OpenVLA 官方仓库提供 LoRA 微调流程，并说明了 BridgeData V2 WidowX 的评测设置；其训练数据包含 BridgeData V2。报告中要保存具体 checkpoint 标识和下载地址，并引用模型卡或官方仓库作为使用 WidowX 数据训练的依据。OpenVLA 官方仓库
- 训练数据： BridgeData V2 中的 WidowX 轨迹。先下载或筛选一个可管理的子集，例如 约 2,000 条完整轨迹；实际数量以下载后统计为准。按完整轨迹划分训练集和验证集，例如 90%/10%，避免同一轨迹的相邻帧同时进入两侧。BridgeData V2
- 微调方式： LoRA，先尝试 rank 16；冻结基础模型主体，仅训练 LoRA 参数。保存配置、训练日志、loss 曲线和 Final checkpoint。正式评测不参与 checkpoint 选择。
- 任务： T1 widowx_spoon_on_towel 和 T2 widowx_put_eggplant_in_basket。每个任务分别对 Base 和 Final 跑 10 个 episode，总计 40 个正式评测 episode。
训练数据子集应先检查轨迹质量和动作字段，再定最终规模。课程没有要求固定轨迹数；重要的是如实报告数据来源、轨迹数、样本数和筛选规则。
2. 实施顺序
1. 建立可复现环境
   记录 SimplerEnv 版本或提交号、OpenVLA 代码版本、依赖和硬件。先验证环境能启动、模型能加载、WidowX 图像和指令能送入模型。
2. 核对动作接口
   对照所用环境版本和 OpenVLA 适配代码，明确动作维度、坐标系、旋转表示、动作单位、夹爪开闭符号、相机输入及归一化统计量。不要仅凭“都是 7 维动作”就直接接线。
3. 处理数据
   编写数据检查和转换脚本，记录图像、语言指令、动作的字段映射；统计完整轨迹数、训练样本数、无效数据处理规则、训练/验证划分和动作归一化。报告中展示至少一个样本的图像、指令和动作格式。
4. 跑通 Base 基线
   在开发初始状态上加载未经本次训练的 Base，验证推理、动作转换、环境步进和环境原生成功判定。开发状态与后续正式评测状态分开保存。
5. LoRA 微调并选定 Final
   记录训练配置、可训练模块与参数量、训练步数、资源占用和 loss。按预先确定的规则（如固定训练步数，或验证集指标）选定 Final checkpoint。
6. 正式评测 Base 和 Final
   预先保存 T1、T2 各 10 个初始状态及环境 seed；Base 与 Final 使用相同清单和相同评测设置。异常、超时和失败 episode 都要记录，策略导致的失败不能筛掉。评测代码应在独立进程中加载 Base 和 Final，检查 checkpoint 与归一化配置正确。
7. 汇总并分析
   计算每项任务的成功数、成功率，按课程公式计算
   B-SimplerScore = (T1 成功率 + T2 成功率) / 2，并报告 Final − Base 的百分点变化。挑选至少两个代表性 episode，结合视频和训练日志分析观察到的变化，并把推测与直接观察到的证据区分开。
3. 仓库和提交材料
建议把代码组织成：
README.md
configs/
  train.yaml
  eval.yaml
scripts/
  prepare_data.py
  train.py
  evaluate.py
data_manifest.md
train_log.txt
results.csv
episodes.csv
checkpoint_link.txt
videos/
report.pdf
episodes.csv 至少包含模型或设置、任务、episode 编号、环境 seed 或初始化标识、模型 seed、成功标记、控制步数和异常信息。README 写清依赖安装、基础模型版本、数据与训练配置、Base/Final 加载方式及运行命令。权重可以放在公开仓库之外，但要给出有效下载链接和加载说明。
按 md 要求，最终还要提交：
- 不超过 10 页的 PDF 报告，包含参考文献、复现链接和组员分工；
- 公开代码仓库链接、Final checkpoint 下载链接和逐 episode 结果；
- 2–4 个视频，覆盖任务并在有失败时包含失败案例；
- 报告中的至少两个 episode 案例分析。
4. 主要风险与先决检查
- SimplerEnv 与所选 OpenVLA checkpoint 是否能直接对接，需要尽早验证。 先确认 WidowX 控制器、相机输入和动作归一化兼容；若需适配，记录代码和验证过程。SimplerEnv 官方仓库列出环境与任务，OpenVLA 官方仓库也提供 WidowX 评测参考，可作为实现依据。SimplerEnv · OpenVLA WidowX 评测说明
- 硬件可能限制微调规模。 OpenVLA 是 7B 模型；先用少量数据和短训练确认显存与训练流程，再扩展到计划的数据子集。不要把“能加载模型”误当成“已完成微调”。
- 正式初始状态要与开发、调参和 checkpoint 选择隔离。 如果某组状态曾用于调试或调参，就不要把它作为正式评测结果提交。
这条路线的重点是先完成一条完整、可复现的 数据处理 → LoRA 微调 → 独立加载 → 40 个 episode 评测 → 视频和报告 流程；方案 B 不需要额外消融或多组训练随机种子。