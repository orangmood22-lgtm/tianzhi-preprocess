"""SAR 大图预处理配置。现场只需修改本文件，再运行 preprocess_sar.py。"""

# 比赛方原始数据目录。图片和标签默认按相对子目录及文件主名对应。
# 例如 RAW_IMAGES/sub/a.tif 对应 RAW_LABELS/sub/a.txt。
RAW_IMAGES = "/path/to/raw/images"
RAW_LABELS = "/path/to/raw/labels"
OUTPUT_DATASET = "/path/to/SARdetect/A.dataset/contest"

# 原始 TXT 解析规则必须拿真实样例核验；当前值只匹配项目中的“可能形式示例”。
# 字段名只允许 class、x1、y1、x2、y2、x3、y3、x4、y4，各用一次。
LABEL_FIELDS = ("class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
LABEL_DELIMITER = "whitespace"  # 可改为 comma；不自动猜测分隔符
LABEL_ENCODING = "utf-8-sig"
LABEL_RELATIVE_TEMPLATE = "{parent}{stem}.txt"  # parent 含末尾 / 或为空；与 RAW_LABELS 拼接
COORDINATE_SYSTEM = "pixel"  # pixel / normalized / world；world 使用 TIFF 的仿射变换
SOURCE_SCHEMA_CONFIRMED = False  # 拿到真实 TIFF+TXT 核验后改为 True；避免把“可能示例”当作实证格式

# TIFF 读取：多波段时明确选择一个波段；NoData 与非有限值不参加百分位统计。
TIFF_BAND = 1
STRETCH_PERCENTILES = (1.0, 99.0)  # 每张原始 TIFF 统一统计，不在切片内各自拉伸
STATS_MAX_SAMPLES = 1_000_000  # 规则网格采样的近似上限，避免整张大图读入内存
IO_WINDOW = 1024  # 统计时的分块宽/高；切片读取仍按 TILE_SIZE

TILE_SIZE = 512
STRIDE = 512  # 小于 TILE_SIZE 会产生重叠切片
EDGE_MODE = "pad"  # pad: 边缘补零至 TILE_SIZE；drop: 丢弃不足尺寸的边缘区域
MIN_VISIBLE_AREA = 0.50  # 裁后多边形面积 / 原目标面积，达到阈值才保留
NEGATIVE_KEEP_RATIO = 0.20  # 无目标切片保留比例；改动会影响类别/背景平衡
SEED = 3929  # 控制无目标切片抽样和无官方划分时的原图分组划分

# 1/2/3 为训练标签允许的最细层级。三级目标向上合并，较粗的源标签保持原样。
# 验证集标签始终保留原始声明深度；labels_original/ 保存所有保留切片的原样标签。
TRAIN_DEPTH_LIMIT = 2
TRAIN_RATIO = 0.90  # 无官方划分时按原始 TIFF 分组；至少需要两张原图
OFFICIAL_SPLIT_DIR = None  # 如已有官方划分，填含 train.txt 与 val.txt 的目录；每行原始 TIFF 相对路径

# 必须与现场分类器权重的类别树一致。若还未获得正式类别树，可先用示例树试运行，
# 真实比赛类别不在树中时脚本会报错，不会猜测或伪造三级类别。
TAXONOMY_JSON = None
CLASS_ALIASES = {}  # 官方名称→模型完整标签；非空时生成覆盖全部模型类别的 class_map.json 供预测反向导出

# 安全策略：已有输出目录时拒绝覆盖，避免不同层级的标签混入同一数据集。
