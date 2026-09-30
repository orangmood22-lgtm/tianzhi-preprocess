#!/usr/bin/env python3
"""将 SAR TIFF 大图切成 SARdetect 可读数据集；参数见 sar_preprocess_config.py。"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote


FIELDS = ("class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
TIFF_EXTS = {".tif", ".tiff"}


def config_from(path: Path):
    spec = importlib.util.spec_from_file_location("sar_preprocess_config", path)
    if not spec or not spec.loader:
        raise ValueError(f"无法加载配置文件：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_config(cfg):
    if not cfg.SOURCE_SCHEMA_CONFIRMED:
        raise ValueError("尚未用比赛方真实 TIFF+TXT 核验原始格式；核验后将 SOURCE_SCHEMA_CONFIRMED 改为 True")
    if sorted(cfg.LABEL_FIELDS) != sorted(FIELDS):
        raise ValueError("LABEL_FIELDS 必须包含 class 和八个坐标字段，且各用一次")
    if cfg.LABEL_DELIMITER not in ("whitespace", "comma"):
        raise ValueError("LABEL_DELIMITER 只能是 whitespace 或 comma")
    if cfg.COORDINATE_SYSTEM not in ("pixel", "normalized", "world"):
        raise ValueError("COORDINATE_SYSTEM 只能是 pixel、normalized 或 world")
    if cfg.EDGE_MODE not in ("pad", "drop"):
        raise ValueError("EDGE_MODE 只能是 pad 或 drop")
    if cfg.TRAIN_DEPTH_LIMIT not in (1, 2, 3):
        raise ValueError("TRAIN_DEPTH_LIMIT 只能是 1、2、3")
    if not 0 <= cfg.NEGATIVE_KEEP_RATIO <= 1 or not 0 <= cfg.MIN_VISIBLE_AREA <= 1:
        raise ValueError("负样本保留比例与面积阈值必须在 0 到 1 之间")
    if not 0 < cfg.TRAIN_RATIO < 1:
        raise ValueError("TRAIN_RATIO 必须严格在 0 到 1 之间")
    if cfg.TILE_SIZE < 1 or not 1 <= cfg.STRIDE <= cfg.TILE_SIZE:
        raise ValueError("TILE_SIZE 必须为正数；STRIDE 必须在 1 到 TILE_SIZE 之间")
    if cfg.TIFF_BAND < 1 or cfg.IO_WINDOW < 1 or cfg.STATS_MAX_SAMPLES < 1:
        raise ValueError("波段、分块尺寸、统计样本数必须为正数")
    low, high = cfg.STRETCH_PERCENTILES
    if not 0 <= low < high <= 100:
        raise ValueError("STRETCH_PERCENTILES 必须满足 0 <= 下界 < 上界 <= 100")
    if not cfg.TAXONOMY_JSON:
        raise ValueError("请配置与现场分类器权重一致的 TAXONOMY_JSON；不能猜测类别树")
    if not isinstance(cfg.CLASS_ALIASES, dict) or any(not isinstance(k, str) or not k or
                                                      not isinstance(v, str) or not v
                                                      for k, v in cfg.CLASS_ALIASES.items()):
        raise ValueError("CLASS_ALIASES 必须是非空字符串到非空字符串的字典")
    if len(set(cfg.CLASS_ALIASES.values())) != len(cfg.CLASS_ALIASES):
        raise ValueError("CLASS_ALIASES 不能把多个外部标签映射到同一模型标签")


def encode_part(value: str) -> str:
    return value.replace("%", "%25").replace("-", "%2D")


def label_parts(label: str):
    raw = label.split("-")
    parts = tuple(unquote(x, errors="strict") for x in raw)
    if len(parts) != 3 or any(not p or p.strip() != p or encode_part(p) != r for p, r in zip(parts, raw)):
        raise ValueError(f"非法三段类别名：{label}")
    a, b, c = parts
    if a == b == c:
        return 1, parts
    if a == b:
        raise ValueError(f"一级占位应为 A-A-A：{label}")
    return (2 if b == c else 3), parts


def taxonomy_labels(tax):
    """由明确的类别树建立路径表；不根据名称前缀猜父子关系。"""
    lookup = {}
    roots = tax["level1"]
    parents = tax["level2"]
    for root in roots:
        label = "-".join([encode_part(root)] * 3)
        lookup[label] = (1, label, label, label)
    for row in parents:
        a, b = roots[row["parent"]], row["name"]
        label = "-".join(map(encode_part, (a, b, b)))
        lookup[label] = (2, "-".join([encode_part(a)] * 3), label, label)
    for row in tax["leaves"]:
        a, b, c = roots[row["l1"]], parents[row["l2"]]["name"], row["level3"]
        if parents[row["l2"]]["parent"] != row["l1"]:
            raise ValueError("taxonomy 中三级类别的父路径不一致")
        label = "-".join(map(encode_part, (a, b, c)))
        root_label = "-".join([encode_part(a)] * 3)
        parent_label = "-".join(map(encode_part, (a, b, b)))
        lookup[label] = (3, root_label, parent_label, label)
    return lookup


def polygon_area(points):
    return abs(sum(x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
                   for i, (x, y) in enumerate(points))) / 2 if len(points) >= 3 else 0.0


def validate_quad(points):
    if len(points) != 4 or any(not math.isfinite(v) for p in points for v in p):
        raise ValueError("四点框必须含八个有限坐标")
    cross = []
    for i in range(4):
        a, b, c = (points[(i + j) % 4] for j in range(3))
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(v > 1e-9 for v in cross) or all(v < -1e-9 for v in cross)):
        raise ValueError("四点必须沿边界排列并形成非退化凸四边形")


def clip_polygon(points, left, top, right, bottom):
    """Sutherland-Hodgman 裁剪，边界点坐标可等于图像宽/高。"""
    result = list(points)
    edges = ((0, left, True), (0, right, False), (1, top, True), (1, bottom, False))
    for axis, bound, keep_greater in edges:
        output = []
        for a, b in zip(result, result[1:] + result[:1]):
            av, bv = a[axis], b[axis]
            ain = av >= bound - 1e-9 if keep_greater else av <= bound + 1e-9
            bin = bv >= bound - 1e-9 if keep_greater else bv <= bound + 1e-9
            if ain != bin:
                t = (bound - av) / (bv - av)
                output.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if bin:
                output.append(b)
        result = output
        if not result:
            return []
    clean = []
    for point in result:
        if not clean or math.dist(point, clean[-1]) > 1e-7:
            clean.append(point)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-7:
        clean.pop()
    return clean


def four_point_approximation(points):
    """裁后凸多边形取最大面积内接四点，保证坐标仍在切片内。"""
    if len(points) < 3 or polygon_area(points) <= 1e-9:
        return None
    if len(points) == 3:
        a, b, c = points
        t = 1e-3
        points = [(a[0] * (1 - t) + c[0] * t, a[1] * (1 - t) + c[1] * t),
                  (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t), b, c]
    elif len(points) > 4:
        points = max((tuple(points[i] for i in choice) for choice in itertools.combinations(range(len(points)), 4)),
                     key=polygon_area)
    try:
        validate_quad(points)
    except ValueError:
        return None
    return points


def parse_source_line(line, cfg, src, context):
    try:
        cells = split_whitespace_tokens(line) if cfg.LABEL_DELIMITER == "whitespace" else next(csv.reader([line]))
    except (ValueError, csv.Error) as exc:
        raise ValueError(f"{context}: 原始标签字段解析失败：{exc}") from exc
    if len(cells) != 9:
        raise ValueError(f"{context}: 应有类别和八个坐标，实际 {len(cells)} 个字段")
    row = dict(zip(cfg.LABEL_FIELDS, cells))
    original_label = row["class"]
    label = cfg.CLASS_ALIASES.get(original_label, original_label)
    try:
        depth, _ = label_parts(label)
        coords = [float(row[f"{axis}{i}"]) for i in range(1, 5) for axis in "xy"]
        points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        if cfg.COORDINATE_SYSTEM == "normalized":
            points = [(x * src.width, y * src.height) for x, y in points]
        elif cfg.COORDINATE_SYSTEM == "world":
            inverse = ~src.transform
            points = [inverse * (x, y) for x, y in points]
        validate_quad(points)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: {exc}") from exc
    return original_label, label, depth, points


def split_whitespace_tokens(line):
    """仅双引号包住含空格类别；名称中的单引号（如 Chang'an）是普通字符。"""
    text = line.strip()
    cells = []
    while text:
        if text.startswith('"'):
            value, end = json.JSONDecoder().raw_decode(text)
            if not isinstance(value, str) or (text[end:] and not text[end].isspace()):
                raise ValueError("双引号类别字段格式错误")
            cells.append(value)
            text = text[end:].lstrip()
        else:
            parts = text.split(maxsplit=1)
            cells.append(parts[0])
            text = parts[1] if len(parts) == 2 else ""
    return cells


def label_path(relative_image, label_root, cfg):
    parent = "" if relative_image.parent == Path(".") else relative_image.parent.as_posix() + "/"
    rendered = cfg.LABEL_RELATIVE_TEMPLATE.format(parent=parent, stem=relative_image.stem,
                                                   name=relative_image.name)
    target = (label_root / rendered).resolve()
    if not target.is_relative_to(label_root):
        raise ValueError(f"标注路径越界：{rendered}")
    return target


def source_labels(path, cfg, src, lookup):
    if not path.is_file():
        raise FileNotFoundError(f"缺少原始标注：{path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding=cfg.LABEL_ENCODING).splitlines(), 1):
        if not line.strip():
            continue
        original_label, label, depth, points = parse_source_line(line, cfg, src, f"{path}:{number}")
        if label not in lookup or lookup[label][0] != depth:
            raise ValueError(f"{path}:{number}: 类别 {label} 不在配置的 taxonomy 中")
        rows.append((original_label, label, points, polygon_area(points)))
    return rows


def read_official_splits(images, image_root, split_dir):
    known = {p.relative_to(image_root).as_posix(): p for p in images}
    memberships = {}
    for split in ("train", "val"):
        file = split_dir / f"{split}.txt"
        if not file.is_file():
            raise FileNotFoundError(f"缺少官方划分文件：{file}")
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            name = line.strip().replace("\\", "/")
            if not name:
                continue
            if name not in known or name in memberships:
                raise ValueError(f"官方划分包含未知或重复原图：{name}")
            memberships[name] = split
    if set(memberships) != set(known) or not {"train", "val"} <= set(memberships.values()):
        raise ValueError("官方划分必须覆盖全部原图，且训练/验证均非空")
    return memberships


def split_scenes(images, image_root, cfg):
    if cfg.OFFICIAL_SPLIT_DIR:
        return read_official_splits(images, image_root, Path(cfg.OFFICIAL_SPLIT_DIR).resolve())
    if len(images) < 2:
        raise ValueError("无官方划分且原始 TIFF 少于两张，无法按原图分组得到非空训练/验证集")
    names = [p.relative_to(image_root).as_posix() for p in images]
    rng = random.Random(cfg.SEED)
    rng.shuffle(names)
    val_count = min(len(names) - 1, max(1, round(len(names) * (1 - cfg.TRAIN_RATIO))))
    val_names = set(names[:val_count])
    return {name: "val" if name in val_names else "train" for name in names}


def positions(length, cfg):
    if cfg.EDGE_MODE == "pad":
        return list(range(0, length, cfg.STRIDE))
    return list(range(0, max(0, length - cfg.TILE_SIZE + 1), cfg.STRIDE))


def stretch_bounds(src, cfg, np):
    step = max(1, math.ceil(math.sqrt(src.width * src.height / cfg.STATS_MAX_SAMPLES)))
    samples = []
    from rasterio.windows import Window
    for top in range(0, src.height, cfg.IO_WINDOW):
        for left in range(0, src.width, cfg.IO_WINDOW):
            width = min(cfg.IO_WINDOW, src.width - left)
            height = min(cfg.IO_WINDOW, src.height - top)
            tile = src.read(cfg.TIFF_BAND, window=Window(left, top, width, height), masked=True)
            y0, x0 = (-top) % step, (-left) % step
            sample = tile[y0::step, x0::step]
            valid = np.asarray(sample.compressed(), dtype=np.float64)
            if valid.size:
                samples.append(valid[np.isfinite(valid)])
    if not samples:
        raise ValueError("图像没有可用像素用于灰度拉伸")
    values = np.concatenate(samples)
    if not values.size:
        raise ValueError("图像有效像素全为 NaN/Inf")
    low, high = np.percentile(values, cfg.STRETCH_PERCENTILES)
    if high <= low:
        raise ValueError("百分位上下界相同；请检查 TIFF 数值范围或修改拉伸配置")
    return float(low), float(high), int(values.size), step


def grayscale(window, low, high, np):
    valid = ~np.ma.getmaskarray(window)
    raw = np.asarray(window.data, dtype=np.float32)
    valid &= np.isfinite(raw)
    result = np.zeros(raw.shape, dtype=np.uint8)
    if np.any(valid):
        result[valid] = np.clip((raw[valid] - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return result


def write_label(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for label, points in rows:
        token = json.dumps(label, ensure_ascii=False) if re.search(r'\s|["\\]', label) or label.startswith("#") else label
        lines.append(token + " " + " ".join(format(v, ".9g") for point in points for v in point))
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + ("\n" if lines else ""))


def keep_negative(seed, scene, row, col, ratio):
    key = f"{seed}\0{scene}\0{row}\0{col}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return value < ratio


def run(cfg):
    validate_config(cfg)
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("预处理依赖 numpy、rasterio、Pillow；请在离线服务器预先安装并验证") from exc
    image_root = Path(cfg.RAW_IMAGES).resolve(strict=True)
    label_root = Path(cfg.RAW_LABELS).resolve(strict=True)
    output = Path(cfg.OUTPUT_DATASET).resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in TIFF_EXTS)
    if not images:
        raise ValueError(f"未找到 TIFF：{image_root}")
    taxonomy = json.loads(Path(cfg.TAXONOMY_JSON).read_text(encoding="utf-8-sig"))
    lookup = taxonomy_labels(taxonomy)
    aliases = cfg.CLASS_ALIASES
    if any(value not in lookup for value in aliases.values()):
        raise ValueError("CLASS_ALIASES 的模型标签必须属于 TAXONOMY_JSON")
    inverse_aliases = {value: key for key, value in aliases.items()}
    full_class_map = {inverse_aliases.get(label, label): label for label in lookup}
    if len(full_class_map) != len(lookup):
        raise ValueError("CLASS_ALIASES 的外部名称与其他模型标签冲突")
    memberships = split_scenes(images, image_root, cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    lists = {"train": [], "val": []}
    totals = collections.Counter()
    scene_report = []
    try:
        for image_path in images:
            relative = image_path.relative_to(image_root)
            scene = relative.as_posix()
            split = memberships[scene]
            identifier = re.sub(r"[^A-Za-z0-9_-]", "_", relative.with_suffix("").as_posix())
            identifier += "_" + hashlib.sha1(scene.encode("utf-8")).hexdigest()[:8]
            with rasterio.open(image_path) as src:
                if cfg.TIFF_BAND > src.count:
                    raise ValueError(f"{scene}: 只有 {src.count} 个波段，配置选择了第 {cfg.TIFF_BAND} 波段")
                if np.issubdtype(np.dtype(src.dtypes[cfg.TIFF_BAND - 1]), np.complexfloating):
                    raise ValueError(f"{scene}: 复数 TIFF 需要先确定幅度/强度处理规则")
                original_label_file = label_path(relative, label_root, cfg)
                labels = source_labels(original_label_file, cfg, src, lookup)
                source_backup = temporary / "source_labels_original" / relative.with_suffix(".txt")
                source_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_label_file, source_backup)
                low, high, sample_count, sample_step = stretch_bounds(src, cfg, np)
                xs, ys = positions(src.width, cfg), positions(src.height, cfg)
                by_row = [[] for _ in ys]
                for original_label, label, points, area in labels:
                    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
                    first = max(0, bisect.bisect_right(ys, ymin - cfg.TILE_SIZE))
                    last = bisect.bisect_left(ys, ymax)
                    for index in range(first, last):
                        by_row[index].append((original_label, label, points, area))
                counts = collections.Counter()
                for row_index, top in enumerate(ys):
                    height = min(cfg.TILE_SIZE, src.height - top)
                    for col_index, left in enumerate(xs):
                        width = min(cfg.TILE_SIZE, src.width - left)
                        original_rows = []
                        training_rows = []
                        for original_label, label, points, area in by_row[row_index]:
                            if max(x for x, _ in points) <= left or min(x for x, _ in points) >= left + width:
                                continue
                            clipped = clip_polygon(points, left, top, left + width, top + height)
                            if polygon_area(clipped) / area + 1e-12 < cfg.MIN_VISIBLE_AREA:
                                continue
                            quad = four_point_approximation(clipped)
                            if quad is None:
                                counts["unrepresentable"] += 1
                                continue
                            local = [(x - left, y - top) for x, y in quad]
                            depth, root_label, parent_label, full_label = lookup[label]
                            training_label = (root_label, parent_label, full_label)[min(depth, cfg.TRAIN_DEPTH_LIMIT) - 1]
                            original_rows.append((original_label, local))
                            training_rows.append((training_label if split == "train" else original_label, local))
                        if not original_rows and not keep_negative(cfg.SEED, scene, row_index, col_index,
                                                                   cfg.NEGATIVE_KEEP_RATIO):
                            counts["negative_dropped"] += 1
                            continue
                        tile = grayscale(src.read(cfg.TIFF_BAND, window=Window(left, top, width, height), masked=True),
                                         low, high, np)
                        if cfg.EDGE_MODE == "pad" and (height < cfg.TILE_SIZE or width < cfg.TILE_SIZE):
                            padded = np.zeros((cfg.TILE_SIZE, cfg.TILE_SIZE), dtype=np.uint8)
                            padded[:height, :width] = tile
                            tile = padded
                        stem = f"{identifier}_r{row_index:05d}_c{col_index:05d}"
                        rel_image = Path("images") / f"{stem}.png"
                        rel_label = Path("labels") / f"{stem}.txt"
                        (temporary / rel_image).parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(tile).save(temporary / rel_image)
                        write_label(temporary / rel_label, training_rows)
                        write_label(temporary / "labels_original" / f"{stem}.txt", original_rows)
                        lists[split].append(rel_image.as_posix())
                        counts["tiles"] += 1
                        counts["targets"] += len(original_rows)
                        counts["positive" if original_rows else "negative_kept"] += 1
                scene_report.append(dict(source=scene, split=split, width=src.width, height=src.height,
                                         source_targets=len(labels), stretch=[low, high], sample_count=sample_count,
                                         sample_step=sample_step, **counts))
                totals.update(counts)
        if not lists["train"] or not lists["val"]:
            raise ValueError("切片后训练/验证集合为空；请检查划分、切片大小及负样本比例")
        if not any(s["split"] == "train" and s.get("targets", 0) > 0 for s in scene_report):
            raise ValueError("训练集没有目标标注")
        if not any(s["split"] == "val" and s.get("targets", 0) > 0 for s in scene_report):
            raise ValueError("验证集没有目标标注；请更换原图分组种子或官方划分")
        for split in ("train", "val"):
            (temporary / f"{split}.txt").write_text("\n".join(lists[split]) + "\n", encoding="utf-8")
        (temporary / "test.txt").write_text("", encoding="utf-8")
        shutil.copyfile(cfg.TAXONOMY_JSON, temporary / "taxonomy.json")
        if aliases:
            (temporary / "class_map.json").write_text(json.dumps(full_class_map, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
        manifest = dict(status="complete", train_depth_limit=cfg.TRAIN_DEPTH_LIMIT,
                        validation_labels="original mixed depth", original_labels="labels_original/",
                        source_labels_backup="source_labels_original/",
                        split_images={k: len(v) for k, v in lists.items()}, totals=dict(totals),
                        source_schema=dict(fields=cfg.LABEL_FIELDS, delimiter=cfg.LABEL_DELIMITER,
                                           coordinates=cfg.COORDINATE_SYSTEM, confirmed=True), scenes=scene_report)
        (temporary / "preprocess_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                                encoding="utf-8")
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sar_preprocess_config.py"))
    args = parser.parse_args()
    cfg = config_from(args.config.resolve(strict=True))
    result = run(cfg)
    print(json.dumps(dict(output=cfg.OUTPUT_DATASET, split_images=result["split_images"],
                          totals=result["totals"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
