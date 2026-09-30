# SARdetect 竞赛操作指南

**标签格式已统一：训练为“类别 四点坐标”，预测为“类别 四点坐标 置信度”。** 目录、类别映射和逐图TXT输出位置见 [dataset_format.md](dataset_format.md)。

二级标签训练属于可选模式：在 `competition.json` 添加 `"classifier": {"mode": "level2"}`，第 4 步会用二级训练目标重建检索库；候选分类器默认输出二级。训练集需最高二级且含二级目标，验证集可保留原始混合深度。缺省 `mixed` 模式继续按原流程训练和逐目标选层级。用二级合并数据时，`depth_prior` 应参考第 1 步报告中的“原样训练标签声明深度统计”。补丁说明见 [LEVEL2_PATCH.md](LEVEL2_PATCH.md)。

> **先检查新数据集能否跑通，再完成一批全部检测模型微调。**
>
> 微调与常规训练使用不同脚本；分类器单独训练。先看验证结果，再决定是否需要常规检测训练。

---

## 执行顺序

| 顺序 | 脚本 | 本步完成什么 |
|:---|:---|:---|
| **1** | `1.data_analyze.sh` | 检查数据、划分集合、填写先验 |
| **2** | `2.check_models.sh` | 现役模型在新数据集上的可用性检查 |
| **3** | `3.finetune_all.sh` | 连续微调全部五个检测模型及各自OBB |
| **4** | `4.train_classifier.sh` | 完整训练分类器五个分支和检索库 |
| **5** | `5.validate_all.sh` | 对微调结果做全量验证、选择并冻结 |
| **6** | `6.test_all.sh` | 按冻结方案测试全部已验证模型 |
| **7** | `7.infer.sh` | 官方无标签测试数据推理 |
| **8** | `8.visualize.sh` | 对已有预测重新抽样查看 |
| 辅助 | `9.status.sh` / `10.resume.sh` | 状态、日志与中断续训 |
| 可选 | `11.train_detectors.sh` | 微调不足时，另开常规检测训练 |

**首轮主线：1 → 2 → 3 → 4 → 5。** 不先做长周期检测训练。

这里的“一轮全部微调”指依次完成五种检测架构的一批实验：默认每个检测器5个epoch、对应OBB 3个epoch，并非每个模型只训练1个epoch。

---

## 开始前：目录、环境与命令

先自行初始化 Python/CANN/torch-npu 环境，再进入竞赛目录：

```bash
cd /workspace/SARdetect
```

本文使用 `contest` 作为数据集名称。实际执行时，把命令中的 `contest` 换成自己的名称即可。

```bash
# 命令形式：脚本 + 数据集名 + 可选参数
bash 3.finetune_all.sh contest --background
```

脚本自动读取 `C.results/0.settings/contest/competition.json`，通常无需填写 `--config`，也不需要切换 `mode`。

| 可选参数 | 作用 |
|:---|:---|
| `--dry-run` | 只打印计划；不会生成“检查已通过”的标记 |
| `--background` | 后台运行，断开SSH后继续；返回PID和日志路径 |
| `--epochs N` | 本次微调/训练的epoch数；不修改配置文件 |
| `--batch N` | 本次微调/训练的batch；不修改配置文件 |

同一配置目录一次只运行一个任务。第3步结束后再启动第4步；连发两条后台命令不会自动排队。

---

## 1. 数据分析与先验

**目标：** 确认标签、划分及分布正确，为后续分类决策提供先验。

### 放置数据

```text
A.dataset/contest/
├── images/
│   ├── scene001_001.png
│   └── scene002_001.png
└── labels/
    ├── scene001_001.txt
    └── scene002_001.txt
```

标签与图像同名。每个目标一行，四点坐标为像素坐标：

```text
A-A1-A11 x1 y1 x2 y2 x3 y3 x4 y4
```

类别始终放在第一列，可为编号或名称；空图提供空TXT。现役分类器的一级/二级/三级名称分别写 `A-A-A`、`A-A1-A1`、`A-A1-A11`。编号与现役分类类别的对应关系由可选 `class_map.json` 明确提供，不能凭编号猜测。

类别名含空格时，用双引号包住整个标签，例如 `"Car-Medium_Car-Chery_Arrizo 5"`。目录可附带 `taxonomy.json`；TXT数据无需提供COCO标注。

### 新数据：分析并划分

```bash
bash 1.data_analyze.sh contest \
  --ratios 0.8 0.1 0.1 \
  --group-regex '^(scene[0-9]+)_'
```

上述正则匹配示例中的场景编号；按实际命名修改。同源切片应归为同组。省略正则时，只自动合并内容完全相同的图像，不能保证场景隔离。

### 已有划分：只检查

```bash
bash 1.data_analyze.sh contest
```

此时保留已有 `train.txt`、`val.txt`、`test.txt`。每行是相对数据集目录的路径，例如 `images/scene001_001.png`；三个集合不能重叠。

### 只需修改三个字段

初次执行生成：`C.results/0.settings/contest/competition.json`。

```json
{
  "dataset": "contest",
  "depth_prior": null,
  "cards": "0,1"
}
```

| 字段 | 怎么填 |
|:---|:---|
| `dataset` | 自动生成的数据集名，通常无需修改 |
| `depth_prior` | 根据 **train** 的深度统计手动填写，例如 `[0.2, 0.3, 0.5]` |
| `cards` | 可用NPU物理编号；单卡填 `"0"`，两卡填 `"0,1"` |

查看终端给出的 `analysis.md` 和 `analysis.json`。先验依次为一级、二级、三级**声明深度**的比例，非各车型比例；三个数非负且和为1。不使用test分布调先验。

**完成标志：** 划分检查通过，统计合理，`depth_prior` 不再为 `null`。

已有配置不会被覆盖。旧版多字段配置仍可读取，但脚本决定微调或训练模式；如需简化，可保留上述三个字段，轮数和batch改用命令参数。

---

## 2. 先检查模型在新数据集上的可用性

**目标：** 在花时间微调前，检查数据读取、模型加载、检测、OBB与可用的分类流程。

默认使用val前8张图，逐一检查现役模型1–5：

```bash
bash 2.check_models.sh contest --background
```

需要更多样本时：

```bash
bash 2.check_models.sh contest --samples 20 --background
```

| 分类标签情况 | 检查内容 |
|:---|:---|
| 与现役分类树兼容 | 检测 → OBB → 分类完整流程 |
| 不兼容，或没有可确认的分类真值 | 检测 → OBB；单独报告分类兼容性，不计算分类分 |

某模型失败时，继续检查其余模型，最后汇总失败项。查看后台日志给出的 `check-时间戳/` 汇总目录和各模型报告。

**完成标志：** `availability.json` 为 `completed`，需要微调的五个模型均检查成功。第3步会检查该记录；数据标签、划分或现役模型登记变化后，需要重新检查。

> 这是少量样本的功能检查，不代表新数据集的全量精度。类别不兼容时，“完成”只表示检测/OBB可运行，分类仍需适配。当前分类器绑定4/21/40代理类别树，不能将新类别强行套入旧树。

可选检查现役十模型（含合成初始化变体）：

```bash
bash 2.check_models.sh contest --models all-ten --background
```

后续默认微调的仍是五种普通检测架构1–5。

---

## 训练前：按内存选择 batch

**推荐起步：主机内存≥64GiB、单卡显存≥40GiB时，检测batch用 `4 / 2 / 4 / 2 / 8`，分类用 `32`。** 本机虽为64GiB显存，也建议先从此档试跑。

以下是**待试跑的配置建议，不是已验证的容量上限或最快配置**。c119已实跑检测batch=1、model-4常规训练batch=4、OBB/分类batch=8；其他大batch档位和16GiB主机竞赛整机尚未验收。

### 配置表（每个模型、每张卡的 batch）

| 档位 | 主机内存至少 | 单卡显存至少 | model-1 | model-2 | model-3 | model-4 | model-5 | 分类 | jobs | memory_gb |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 保守起步 | 16GiB | 16GiB | 1 | 1 | 1 | 1 | 1 | 8 | 1 | 12 |
| 小幅提速 | 32GiB | 24GiB | 2 | 1 | 2 | 1 | 4 | 16 | 1 | 24 |
| **优先试跑** | **64GiB** | **40GiB** | **4** | **2** | **4** | **2** | **8** | **32** | **2** | **48** |
| 通过上一档后再试 | 128GiB | 64GiB | 8 | 4 | 8 | 4 | 16 | 64 | 2 | 96 |

按**主机可用内存和单卡显存同时满足**的档位选，容器限额更低时按容器额度。显存不能跨卡相加；`jobs=2`指两个模型各用一张卡，单模型batch不会翻倍。

- `memory_gb`：全部训练子进程合计的**主机内存软预算**，不是显存限制，也不预先分配这些内存；仍需留系统及其他程序余量。
- 所有档位的OBB batch仍为 **8**，编号入口当前固定此值；不支持在JSON中设置 `obb_batch`。分类batch作用于全部五个分类分支，需以最吃显存的分支为准。
- 编号入口当前固定 `workers=0`，不要添加无效的workers字段。表中batch建议沿用当前图像尺寸和模型配方；输入尺寸、目标密度增大时应降档。

### 手动修改位置

编辑 `C.results/0.settings/contest/competition.json`，**保留已有 `dataset` 和 `depth_prior`**，合并以下字段（优先试跑档）：

```json
{
  "cards": "0,1",
  "jobs": 2,
  "memory_gb": 48,
  "detectors": {
    "model-1": {"batch": 4},
    "model-2": {"batch": 2},
    "model-3": {"batch": 4},
    "model-4": {"batch": 2},
    "model-5": {"batch": 8}
  },
  "classifier": {"batch": 32}
}
```

已有 `detectors` / `classifier` 时只改对应batch，不删除原有轮数等设置。第3、11步读取同一组检测batch，第4步读取分类batch。

```bash
# 检查实际计划中的 --batch-size；不会启动训练
bash 3.finetune_all.sh contest --dry-run
bash 4.train_classifier.sh contest --dry-run
```

**使用逐模型配置时，不再加 `--batch`。** 命令行 `--batch N` 会把该次全部检测器统一改成N；用于第4步则统一覆盖分类batch，不影响OBB。

### 试跑与调整

1. 先用独立试跑配置/候选完成1轮训练、验证和保存，再跑正式批次。第2步可用性检查不能验证训练显存；第3步 `--epochs 1` 只缩短检测轮数，OBB仍按自身轮数执行。
2. 同时观察主机内存、NPU占用与每秒处理图像数；双任务需在并行时复核。显存建议留约20%余量，这只是调参余量，不是通过标准。
3. 发生显存不足：将该模型batch减半。主机内存不足：先将jobs降为1，再减batch。有余量且吞吐仍提升时，再试batch翻倍；无提速就回退。
4. 不在原队列续训时改batch。新batch用新任务验证，不能沿用上面的实测耗时估算；每轮优化器更新次数会变化，先保留学习率并看val结果，不自动按batch倍增学习率。

调参依据：[PyTorch性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)说明batch及数据加载会影响利用率；**上表具体数值是本项目的建议起点，非官方认证值**。

---

## 3. 完成一批全部检测模型微调

**目标：** 用较短周期验证现役检测权重能否适应新数据集。

先看计划：

```bash
bash 3.finetune_all.sh contest --dry-run
```

确认后后台执行全部五个模型：

```bash
bash 3.finetune_all.sh contest --background
```

| 项目 | 默认行为 |
|:---|:---|
| 检测器 | 每个5轮，batch=1 |
| OBB精修器 | 每个3轮 |
| 学习率 | 对应原配方的0.1倍 |
| 初始化 | 各模型现役检测权重和对应OBB权重 |
| 排队 | 串行执行；按 `cards` 交替使用卡 |

五种架构依次为：1 MSFA/Faster R-CNN、2 DDQ-DETR、3 RT-DETRv2、4 DINO、5 YOLOv8m。

例如将本批检测微调缩短为3轮：

```bash
bash 3.finetune_all.sh contest --epochs 3 --background
```

`--epochs` 控制检测器轮数；OBB仍使用微调默认3轮。微调会创建新优化器并缩短预热，输出候选，不覆盖现役权重。

**完成标志：** 整队列 `completed`，配置目录生成 `candidates.finetune.json`。队列任一任务失败会停止，按第10步恢复。

---

## 4. 分类器单独训练

**目标：** 训练五个分类分支，并重建仅使用训练集的检索库。

```bash
bash 4.train_classifier.sh contest --dry-run
```

```bash
bash 4.train_classifier.sh contest --background
```

这里默认使用常规分类训练配方：**12轮、batch=8、基础学习率3e-5**。允许比检测微调更充分地拟合新数据；仍从现役网络权重初始化，不是随机初始化从零训练。

也可单独指定：

```bash
bash 4.train_classifier.sh contest --epochs 20 --batch 4 --background
```

三个EfficientNet-B0、ConvNeXt-Tiny、DINOv2-Base均参与训练。DINO分支更新后重建检索特征；当前流程要求训练集有三级细标签目标。

**完成标志：** 任务 `completed`，生成 `candidates.classifier.json`。

如果第2步提示分类树不兼容，先适配taxonomy、输出头和初始化权重，再执行本步。仅增加轮数不能解决类别定义不同的问题。

---

## 5. 全量验证，再选择方案

默认组合为：**第3步微调的检测器/OBB + 第4步训练的分类器**。

```bash
bash 5.validate_all.sh contest --background
```

在完整val上评测；不是第2步的小样本检查。候选缺失会报错，不会自动使用旧权重代替。

### 查看结果

后台日志给出 `validate-时间戳/` 目录：

- `summary.csv`：各模型的总分、检测、分类、FN/FP。
- `evaluation.json`：各模型完整报告的路径。
- 各运行目录中的 `report.html`：逐图、逐类、错误混淆与样例。

如需同数据集现役基线，可另跑：

```bash
bash 5.validate_all.sh contest --current --background
```

### 结果足够好：冻结

例如，按val选中模型5用于最终推理：

```bash
bash 5.validate_all.sh contest \
  --validation C.results/0.settings/contest/validate-实际时间戳 \
  --select 5
```

生成 `frozen.json`。第6步测试全部已验证模型；第7步使用 `--select` 指定的模型。只有完整val结果可以冻结，第2步功能检查不可以。

### 微调仍不足：再考虑常规检测训练

先进入后面的**可选第11步**，再回到本步验证；不因微调效果不明就默认启动长周期训练。

---

## 6. 有标签test集测试

```bash
bash 6.test_all.sh contest --background
```

按冻结方案测试全部已验证模型，不再修改先验、阈值或模型选择。

**结果：** `C.results/2.test/实验名/result.txt`，记录检测分、分类分、综合分、FP/FN。无车型真值时分类/总分为 `N/A`。

---

## 7. 官方无标签数据推理

```bash
bash 7.infer.sh contest \
  --source /data/official_test/images \
  --background
```

将 `--source` 改为实际图像目录，也可传单张图像。使用已冻结、已选择的模型；不计算无标签数据的分数。

**结果：** `C.results/3.infer/实验名/result.txt`，末尾的预测 JSON 包含四点坐标、置信度和层级标签。无标签输入不计算分数；最终提交文件仍须按赛方格式转换。

---

## 8. 抽样查看预测

```bash
bash 8.visualize.sh \
  --run C.results/3.infer/实际运行名 \
  --examples 8 \
  --seed 42
```

使用具体评测/推理目录，不使用 `validate-时间戳` 汇总目录。

- 有真值：从高分、低分四分位池各随机抽最多8张。
- 无真值：随机抽最多8张，不按成绩排序。
- 换 `--seed` 可查看另一批图；无需重新推理。

新报告位于原目录的 `visualization-时间戳/report.html`。绿框为真值，红框为预测。

---

## 9. 状态与日志

```bash
bash 9.status.sh contest
```

查看 `running / completed / failed / interrupted`、PID、日志路径及最近训练队列。

```bash
# 替换为启动时返回的实际日志
 tail -f C.results/0.settings/contest/job-实际时间戳.log
```

机器重启或进程被强制杀死后，持久记录可能仍显示 `running`，需结合日志和实际PID确认。

---

## 10. 中断续训

使用启动时打印、或第9步 `QUEUE` 显示的准确路径：

```bash
bash 10.resume.sh \
  --queue .runtime/queues/实际名称.training_queue.json \
  --background
```

保留原batch、轮数、学习率和数据划分，跳过已完成模型。首个检查点尚未生成时，从第0轮按原请求重启。

要修改batch或轮数，应新开任务。评测和推理失败使用对应入口重跑，不使用训练续训脚本。

---

## 11. 可选：常规检测训练

**这是独立训练入口，不属于首轮微调主线。**

```bash
bash 11.train_detectors.sh contest --dry-run
```

```bash
bash 11.train_detectors.sh contest --epochs 30 --background
```

| 对比项 | 第3步微调 | 第11步常规训练 |
|:---|:---|:---|
| 检测器默认轮数 | 5 | 30 |
| OBB默认轮数 | 3 | 12 |
| 学习率 | 原配方0.1倍 | 原配方基础学习率 |
| 调度 | 短程微调预热 | 原训练配方 |
| 候选索引 | `candidates.finetune.json` | `candidates.trained.json` |

两者都从现役检测/OBB权重初始化，不随机重建检测网络；各自结果独立保存。常规训练不会覆盖微调候选。

验证常规训练结果时，明确加 `--trained`：

```bash
bash 5.validate_all.sh contest --trained --background
```

分类器仍使用第4步候选。按该次val结果选择并冻结后，再执行第6、7步。已有冻结文件不覆盖；已冻结后开展新方案，请另用配置目录重新验证。

---

## 实测耗时与显存速查

**2026-09-26 c119 演练，训练已全部完成。** Ascend 910B3（64GiB/卡），仅使用0、1；每个模型单卡，最多两模型并行。完整v3：训练2312图、验证72图；检测batch=1（model-4常规训练为4），OBB/分类batch=8，workers=0。

### 检测器

| 模型 | 微调：分钟/轮 | 常规训练：分钟/轮 | 检测阶段显存记录 |
|:---|---:|---:|:---|
| model-1 | 11.1 | 11.8 | 未记录 |
| model-2 | 53.4 | 53.9 | 未记录 |
| model-3 | 13.6 | 13.9 | 未记录 |
| model-4 | 40.6（batch1） | 30.0（batch4） | batch4进程采样约31.5GiB，非峰值 |
| model-5 | 5.3 | 6.1 | 微调1.46 / 常规1.44 GiB |

**每个检测器另加OBB训练：约3.4–3.9分钟/轮，5轮约17–20分钟。** 17:10的model-2 OBB实测进程显存667MB，整卡HBM占用4008MB；这是即时采样，不是峰值。

**model-4 batch4已完成5轮。** 第2–5轮每轮约30.0分钟；启动后即时采样进程显存约31.5GiB、整卡约34.8GiB，非全程峰值。微调与常规训练配方不同，不能将两列差值当作严格batch加速比。

### 分类器（五分支依次训练）

| 分支 | 分钟/轮 | 显存记录 |
|:---|---:|:---|
| baseline | 4.5 | 未记录 |
| aligned | 4.7 | 未记录 |
| aligned3924 | 4.2 | 未记录 |
| convnext | 3.5 | 未记录 |
| dino | 4.0 | 未记录 |

五分支各5轮，训练约105分钟，另计初始化及特征库构建。

- 耗时取连续已完成轮的检查点时间差均值，通常为第2–5轮，包含期间验证与保存；排除首轮初始化及跨中断区间。model-4只统计微调第2–4轮。
- YOLO显存为日志中框架**预留内存最大观测值**，不等于整卡HBM峰值；其他训练器未记录历史NPU峰值，不能用主机内存替代。
- 微调前半段串行，常规训练可双任务并行；数值用于排期，不作为严格速度对比。数据规模、batch及并发变化后需重新测量。
- 复算依据：研究目录 `3.experiments/stage15/s15-c119-full-rehearsal/resource-timing.json`；训练日志统一查阅各实验目录的 `record.txt`。

---

## 实验结果：每个目录只看四项

检测器、OBB、分类器使用同一输出格式；测试和推理也采用此目录结构。

```text
C.results/1.train/数据集_model5_cfg00_时间戳/
├── config/          实际配置：数据规模、模型、batch、轮数、学习率
├── record.txt       逐轮loss与验证指标、完整训练输出、失败与续训记录
├── result.txt       最佳验证轮次/指标，以及已关联的最终val/test成绩
└── visualization/   loss与验证曲线PNG、对应逐轮CSV
```

| 要查什么 | 打开哪里 |
|:---|:---|
| 训练是否正常、损失是否下降 | `record.txt`；运行中可 `tail -f 实验目录/record.txt` |
| 最佳轮次与验证成绩 | `result.txt`，检测、OBB、分类分支分别列出 |
| 最终检测分、分类分、综合分 | 第5/6步实验的 `result.txt`；自动关联回对应训练实验 |
| 曲线与可视化样例 | `visualization/`；第8步只更新样例，不改原日志或分数 |
| 复现实际训练参数 | `config/run.json` 和各模型/分支配置 |

**指标口径：** 分数为0～1，乘100即百分比。训练mAP、OBB IoU、分类分支得分分别展示；完整流水线综合分在第5/6步运行后生成。未执行的评测明确写“尚未执行”，不填0。MMEngine曲线中的loss是日志采样平滑值均值，YOLO保留box/cls/dfl三条曲线。

**参数修改：** `config/` 是本次实验的配置留档；下次实验请修改 `C.results/0.settings/数据集名/competition.json`，不要改留档冒充原运行参数。

### 其他目录

| 内容 | 位置 |
|:---|:---|
| 配置、可用性记录、各类候选索引、冻结文件、后台日志 | `C.results/0.settings/数据集名/` |
| 队列JSON与续训入口 | `.runtime/queues/*.training_queue.json`（第3/4/11步会打印准确路径） |
| 训练结果 | `C.results/1.train/实验名/` |
| 有标签评测 / 无标签预测 | `C.results/2.test/` / `C.results/3.infer/` |
| 内部状态、后端原始产物、候选权重 | `.runtime/experiments/`、`.runtime/checkpoints/`，由脚本维护 |

`.runtime/` 用于续训、候选评测与追溯，不能在任务进行中删除。后端临时标注、缓存、事件JSON与检查点不再散落到公开实验目录。现役部署权重保留在 `B.models/`，训练产生的候选不会自动替换它。

需要自定义配置目录时，使用 `--config /配置目录/competition.json`；该文件仍可只含三个基本字段。默认训练主存软预算12GiB，workers=0；双NPU不等于双倍主存，16GB赛事整机仍需实测。

主机内存充足时，可在 `competition.json` 增加 `"jobs": 2, "memory_gb": 0`，保留 `"cards": "0,1"`。两个独立检测模型分别使用一张卡；0表示放开主存预算至主机/容器容量，保留2GiB系统余量。分类队列只有一个任务时仍串行。运行中的队列不会热更新配置，须停止后用原队列与检查点续训。

评测默认只使用 `evaluation.device` 一张卡。需要并行测试独立模型时，可增加 `"evaluation": {"devices": "0,1"}`；八卡机器可填 `"0,1,2,3,4,5,6,7"`。每张卡同一时间只跑一个模型组；融合方案仍在一张卡上执行。请按可用主存选择卡数，训练和评测共用卡锁，避免抢占正在运行的模型。

训练队列也支持最多八卡：例如五种检测器同时训练，设置 `"cards": "0,1,2,3,4", "jobs": 5, "memory_gb": 0`。默认仍为双卡配置；每个模型只占一张卡，分类器的五个分支仍在单任务内依次训练。

旧 `2.train_single.sh`、`3.train_detectors.sh` 已替换为新的第2、3步。旧配置和历史结果不删除，但旧 `candidates.json` 不会自动并入新的微调/训练候选集合。
