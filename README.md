# SAR 大图预处理与 SARdetect 二级分类补丁

这个项目提供两部分代码：

1. [`preprocess_sar.py`](preprocess_sar.py) 将带四点目标标注的 SAR TIFF 大图切成 SARdetect 可读取的 512×512 PNG 数据集。训练标签可以限制到指定层级；验证标签保留原始声明深度。
2. `SARdetect示例/D.codes/` 中的可选补丁保留原有混合层级分类模式，增加用二级标签训练、二级检索并固定输出二级的模式。覆盖文件清单见 [`SARdetect示例/LEVEL2_PATCH.md`](SARdetect示例/LEVEL2_PATCH.md)。

操作步骤、输出结构和三组验证对照见 [`PREPROCESS_USAGE.md`](PREPROCESS_USAGE.md)。全部可调参数集中在 [`sar_preprocess_config.py`](sar_preprocess_config.py)，含中文注释。

## 快速检查

使用 Python 3.9+，安装 NumPy、Pillow、Rasterio。若要让 SARdetect 的分析入口读取数据，还需要框架环境的 OpenCV 等依赖。

```bash
python -m unittest discover -s tests -p 'test_preprocess_sar.py' -v
python -m unittest discover -s tests -p 'test_level2_mode.py' -v
```

预处理测试会生成临时合成 TIFF 并真正调用预处理命令；如果本地有完整的 `SARdetect示例`，还会检查数据适配器和第 1 步分析结果。`tests/fixtures/taxonomy.json` 仅用于合成数据测试，不代表比赛类别树。测试不会启动训练。

## 使用边界

项目里的原始 TXT 示例来自 SARdetect 样例，尚未用比赛方真实 TIFF/TXT 核对。预处理配置默认设置 `SOURCE_SCHEMA_CONFIRMED=False`；拿到真实样例核对标注顺序、坐标、编码及类别树后再启用。当前工作区缺少模型权重和 NPU，完整训练与最终推理仍须在比赛镜像中验证。

数据集、模型权重、缓存及实验结果不纳入源码仓库。项目中的 ZIP 是便于复制的交付包；修改源码后需要重新生成 ZIP，源码文件才是版本依据。
