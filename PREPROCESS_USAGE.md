# SAR TIFF 预处理与二级分类模式

## 现场前先核对

项目中的 `源数据标注可能形式示例.txt` 来自 SARdetect 的样例数据，尚不能证明比赛方大图 TXT 的格式。先拿一组真实 TIFF 与 TXT 核对：文件对应关系、编码、列顺序、分隔符、坐标系、波段与类别树。`sar_preprocess_config.py` 默认将 `SOURCE_SCHEMA_CONFIRMED` 设为 `False`，核对并填写配置后才改为 `True`。当前框架现役分类头绑定其 `taxonomy.json`；若正式类别树不同，预处理会在未知标签处报错，分类头也需要另行适配。

预处理环境需要 Python 3.9+、NumPy、Pillow、Rasterio。比赛现场离线，先在比赛镜像中确认这些包可用。

## 上传前手动验收

在项目根目录，先使用 Python 3.9+ 的环境运行合成数据测试：

```bash
python -c "import numpy, PIL, rasterio; print('预处理依赖可用')"
python -m unittest discover -s tests -p 'test_preprocess_sar.py' -v
python -m unittest discover -s tests -p 'test_level2_mode.py' -v
```

预处理测试会调用真实的 `preprocess_sar.py --config ...` 命令，随后让 SARdetect 的数据适配器和第 1 步分析代码读取结果。两组测试各有 3 项，应显示 `OK`。测试依赖项目内的 `SARdetect示例/A.dataset/SIVED_ATR_synthetic_v3/taxonomy.json`。

拿到真实 TIFF/TXT 后，用一两张小样本先核对输入格式，再在配置中填路径并将 `SOURCE_SCHEMA_CONFIRMED=True`。执行下文预处理命令后，检查 `preprocess_manifest.json` 的训练/验证数量、`test.txt` 为空、每张 PNG 的尺寸、`labels/` 与 `labels_original/` 的类别及边缘框坐标。训练集应至少含一个二级目标，验证集应保留真实声明深度。检查通过后，在 SARdetect 所在环境运行：

```bash
SARDETECT_PYTHON=/path/to/framework/python bash SARdetect/1.data_analyze.sh <数据集名>
```

如果当前目录名为 `SARdetect示例`，就把命令中的 `SARdetect/` 改为 `SARdetect示例/`。第 1 步需要框架自己的依赖，包括 OpenCV；若已激活正确环境，可省略 `SARDETECT_PYTHON=...`。在 `analysis.json` 核对 `original_train_label_depth.depth_ratio`。现场具备现役权重与 NPU 时，再运行 `4.train_classifier.sh <数据集名> --dry-run` 检查命令中出现 `--classifier-mode level2`，随后才开始真实训练。当前工作区没有权重，无法代替这一步。

## 生成数据集

编辑 [`sar_preprocess_config.py`](sar_preprocess_config.py) 中的原图、原标注、输出目录、解析规则及 `TAXONOMY_JSON`，然后运行：

```bash
python preprocess_sar.py --config sar_preprocess_config.py
```

输出目录必须尚不存在。默认按 512 像素切片、步长 512、边缘补零，使用每张 TIFF 的 1% 与 99% 分位统一拉伸为 8 位 PNG。跨切片框裁剪后保留面积至少 50% 才输出；多于四点的裁后多边形使用面积最大的内接四边形近似。`MIN_VISIBLE_AREA`、灰度分位、负样本比例都在配置中可改。建议拿真实小样例逐图检查裁剪框。

生成的 `labels/` 是训练用标签：`TRAIN_DEPTH_LIMIT=2` 会把三级目标合并到二级，一级目标仍是一级。验证集的 `labels/` 保留原始声明深度。`labels_original/` 保存所有保留切片的原始类别，`source_labels_original/` 逐字节保存原始 TXT。若配置 `CLASS_ALIASES`，训练标签使用模型类别、原样标签使用比赛方类别，并生成覆盖完整类别树的 `class_map.json` 用于预测反向导出。`train.txt`、`val.txt` 按原图分组；`test.txt` 为空，不划出人工测试集。`preprocess_manifest.json` 记录划分和数量。

## 在 SARdetect 中使用

将输出目录放在 `SARdetect/A.dataset/<数据集名>/`，运行原有第 1 步检查：

```bash
cd SARdetect
bash 1.data_analyze.sh <数据集名>
```

已有 `train.txt`、`val.txt`、`test.txt` 时，第 1 步只检查现有划分。分析结果同时给出训练标签和 `labels_original/` 中原始训练标签的声明深度比例。填写 `C.results/0.settings/<数据集名>/competition.json` 的 `depth_prior` 时，使用**原始训练标签**比例；二级合并后的训练标签比例不是原始深度先验。

随后按框架顺序运行第 2 步检查和第 3 步检测器微调。第 3 步分别训练五个检测器及对应旋转框精修器；第 4 步单独训练分类器的五个分支并重建检索库。若采用本补丁新增的二级模式，在同一配置文件中加入：

```json
"classifier": {"mode": "level2"}
```

然后运行：

```bash
bash 4.train_classifier.sh <数据集名>
```

该模式要求训练标签最高为二级，且至少有一个二级目标。检索库使用二级目标；生成的分类器候选清单默认固定输出二级。验证标签仍使用原始声明深度。原有模式为 `"mixed"`，保留三级检索库与逐目标期望收益选层级；它需要训练集中至少有一个三级目标。

`TRAIN_DEPTH_LIMIT=1` 可以生成只含一级训练目标的数据集，但本次框架补丁仅新增了二级分类模式；一级分类训练需要另外实现对应检索策略。

做三组比较时，分别验证：原混合模式动态输出、同一原模型强制二级输出、二级标签重训候选固定二级输出。前两组需使用含原始三级训练标签的原模型；强制二级输出可在配置中设 `"evaluation": {"output_depth": 2}`。每组使用同一份原始验证真值和相同的检测器、旋转框权重。当前代码提供的是 `proxy-v3.2` 本地评分，需以比赛方正式评分为准。

## 本地核验范围

已用合成 TIFF 验证预处理 CLI、切片、补齐、标签合并、原标注备份、空测试集，以及输出由 SARdetect 的 `dataset_adapter` 读入。项目自带 2455 张样例图已通过 `1.data_analyze.sh`；分类训练队列也已确认会传递 `--classifier-mode level2`。已验证旧解码器行为不变、新解码器固定二级和二级检索概率投影。项目内的 `B.models` 为空，当前环境也没有比赛所需 NPU 与权重，因此尚未执行完整训练或真实数据验证。
