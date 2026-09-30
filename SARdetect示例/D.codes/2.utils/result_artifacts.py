#!/usr/bin/env python3
"""Create compact, human-readable result artifacts and remove framework clutter."""

from __future__ import annotations

import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = ((43, 108, 176), (49, 151, 149), (237, 137, 54), (128, 90, 213))


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _source_name(image_name: str) -> str:
    return Path(image_name).stem.split("_", 1)[0]


def dataset_distribution(dataset_dir: Path, split: str) -> dict[str, Any]:
    split_file = dataset_dir / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(split_file)
    sources: dict[str, dict[str, int]] = defaultdict(lambda: {"images": 0, "instances": 0})
    instance_counts: list[int] = []
    for raw_name in split_file.read_text(encoding="utf-8").splitlines():
        raw_name = raw_name.strip()
        if not raw_name:
            continue
        name = Path(raw_name).name
        label = dataset_dir / "labels" / f"{Path(name).stem}.txt"
        instances = sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
        source = _source_name(name)
        sources[source]["images"] += 1
        sources[source]["instances"] += instances
        instance_counts.append(instances)
    return {
        "split": split,
        "images": len(instance_counts),
        "instances": sum(instance_counts),
        "instances_per_image": round(sum(instance_counts) / max(1, len(instance_counts)), 3),
        "sources": dict(sorted(sources.items())),
    }


def _bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=(31, 41, 55), font=_font(28, True))
    axis_top = top + 55
    axis_bottom = bottom - 55
    draw.line((left + 70, axis_top, left + 70, axis_bottom), fill=(75, 85, 99), width=2)
    draw.line((left + 70, axis_bottom, right, axis_bottom), fill=(75, 85, 99), width=2)
    maximum = max(values or [1])
    count = max(1, len(values))
    usable = right - left - 100
    slot = usable / count
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_width = min(90, slot * 0.58)
        x0 = left + 85 + index * slot + (slot - bar_width) / 2
        x1 = x0 + bar_width
        height = (axis_bottom - axis_top - 35) * value / maximum if maximum else 0
        y0 = axis_bottom - height
        draw.rectangle((x0, y0, x1, axis_bottom), fill=COLORS[index % len(COLORS)])
        value_text = f"{value:.3f}" if maximum <= 1.0 else f"{int(value):,}"
        draw.text((x0, max(axis_top, y0 - 28)), value_text, fill=(31, 41, 55), font=_font(19))
        draw.text((x0, axis_bottom + 12), label, fill=(31, 41, 55), font=_font(18))


def write_distribution_plot(distribution: dict[str, Any], destination: Path) -> None:
    image = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(image)
    split = str(distribution["split"]).upper()
    draw.text((60, 35), f"SIVED {split} sample distribution", fill=(17, 24, 39), font=_font(38, True))
    source_rows = distribution["sources"]
    labels = list(source_rows)
    _bar_panel(draw, (60, 120, 735, 700), labels, [row["images"] for row in source_rows.values()], "Images by source")
    _bar_panel(draw, (795, 120, 1440, 700), labels, [row["instances"] for row in source_rows.values()], "Instances by source")
    summary = (
        f"Images: {distribution['images']:,}    Instances: {distribution['instances']:,}    "
        f"Instances / image: {distribution['instances_per_image']:.3f}"
    )
    draw.text((60, 750), summary, fill=(55, 65, 81), font=_font(24))
    image.save(destination, format="PNG")


def read_epoch_losses(run_dir: Path) -> tuple[list[int], list[float], str]:
    epoch_values: dict[int, list[float]] = defaultdict(list)
    metric = "loss"
    scalar_files = sorted(run_dir.rglob("scalars.json"))
    for path in scalar_files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in row and "loss" in row:
                epoch_values[int(row["epoch"])].append(float(row["loss"]))
    if not epoch_values:
        metric = "train_loss"
        for path in sorted(run_dir.rglob("log.txt")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "epoch" in row and "train_loss" in row:
                    epoch_values[int(row["epoch"]) + 1].append(float(row["train_loss"]))
    if not epoch_values:
        raise RuntimeError(f"no structured training loss found in {run_dir}")
    epochs = sorted(epoch_values)
    values = [sum(epoch_values[epoch]) / len(epoch_values[epoch]) for epoch in epochs]
    return epochs, values, metric


def write_loss_plot(epochs: list[int], losses: list[float], metric: str, destination: Path) -> None:
    image = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), "Training loss by epoch", fill=(17, 24, 39), font=_font(38, True))
    left, top, right, bottom = 120, 130, 1420, 700
    draw.line((left, top, left, bottom), fill=(75, 85, 99), width=3)
    draw.line((left, bottom, right, bottom), fill=(75, 85, 99), width=3)
    minimum, maximum = min(losses), max(losses)
    spread = max(maximum - minimum, max(abs(maximum), 1.0) * 0.05)
    low, high = max(0.0, minimum - spread * 0.15), maximum + spread * 0.15
    points: list[tuple[float, float]] = []
    for index, (epoch, loss) in enumerate(zip(epochs, losses)):
        x = left if len(epochs) == 1 else left + index * (right - left) / (len(epochs) - 1)
        y = bottom - (loss - low) * (bottom - top) / max(high - low, 1e-9)
        points.append((x, y))
        draw.line((x, bottom, x, bottom + 8), fill=(75, 85, 99), width=2)
        draw.text((x - 8, bottom + 14), str(epoch), fill=(55, 65, 81), font=_font(18))
    if len(points) > 1:
        draw.line(points, fill=(43, 108, 176), width=5, joint="curve")
    for (x, y), loss in zip(points, losses):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(43, 108, 176))
        draw.text((x - 20, max(top, y - 32)), f"{loss:.3f}", fill=(31, 41, 55), font=_font(17))
    draw.text((670, 760), "Epoch", fill=(31, 41, 55), font=_font(24, True))
    draw.text((45, 90), metric, fill=(31, 41, 55), font=_font(22, True))
    image.save(destination, format="PNG")


def write_metrics_plot(metrics: dict[str, Any], destination: Path) -> None:
    keys = (
        ("AP", "coco/bbox_mAP"),
        ("AP50", "coco/bbox_mAP_50"),
        ("AP75", "coco/bbox_mAP_75"),
        ("AP90", "coco/bbox_mAP_90"),
        ("Recall", "coco/bbox_AR_300"),
        ("AP-small", "coco/bbox_mAP_s"),
        ("AP-medium", "coco/bbox_mAP_m"),
    )
    labels = [label for label, key in keys if isinstance(metrics.get(key), (int, float)) and metrics[key] >= 0]
    values = [float(metrics[key]) for _, key in keys if isinstance(metrics.get(key), (int, float)) and metrics[key] >= 0]
    image = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), "SIVED test metrics", fill=(17, 24, 39), font=_font(38, True))
    _bar_panel(draw, (100, 120, 1420, 720), labels, values, "COCO bounding-box metrics")
    image.save(destination, format="PNG")


def _tensor_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _coco_predictions(run_dir: Path, annotation: dict[str, Any]) -> list[dict[str, Any]]:
    coco_path = run_dir / "predictions.coco.json"
    if coco_path.is_file():
        return json.loads(coco_path.read_text(encoding="utf-8"))
    pickle_path = run_dir / "predictions.pkl"
    if not pickle_path.is_file():
        raise FileNotFoundError(f"prediction dump not found in {run_dir}")
    with pickle_path.open("rb") as stream:
        rows = pickle.load(stream)
    image_id_by_name = {
        Path(row["file_name"]).name: int(row["id"]) for row in annotation["images"]
    }
    ordered_ids = [int(row["id"]) for row in annotation["images"]]
    category_id = int(annotation["categories"][0]["id"])
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        if not isinstance(row, dict):
            raise TypeError(f"unsupported prediction record: {type(row)}")
        raw_image_id = row.get("img_id")
        if raw_image_id is not None:
            image_id = int(raw_image_id)
        elif row.get("img_path") is not None:
            image_id = image_id_by_name[Path(row["img_path"]).name]
        else:
            image_id = ordered_ids[index]
        instances = row.get("pred_instances", {})
        if hasattr(instances, "to_dict"):
            instances = instances.to_dict()
        boxes = _tensor_list(instances.get("bboxes", []))
        scores = _tensor_list(instances.get("scores", []))
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = (float(value) for value in box)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    return results


def supplement_coco_metrics(
    run_dir: Path,
    annotation_path: Path,
    max_detections: int = 300,
) -> dict[str, Any]:
    """Add AP@0.90 and COCO AR at the configured maximum detections."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    predictions = _coco_predictions(run_dir, annotation)
    if not predictions:
        raise RuntimeError(f"no detections were produced in {run_dir}")
    coco_ground_truth = COCO(str(annotation_path))
    coco_detections = coco_ground_truth.loadRes(predictions)
    evaluator = COCOeval(coco_ground_truth, coco_detections, "bbox")
    evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
    evaluator.params.maxDets = [1, 10, int(max_detections)]
    evaluator.evaluate()
    evaluator.accumulate()
    iou_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.90))[0])
    precision = evaluator.eval["precision"][iou_index, :, :, 0, 2]
    valid_precision = precision[precision > -1]
    recall = evaluator.eval["recall"][:, :, 0, 2]
    valid_recall = recall[recall > -1]
    if not valid_precision.size or not valid_recall.size:
        raise RuntimeError("COCO evaluator returned no valid precision/recall values")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["coco/bbox_mAP_90"] = float(np.mean(valid_precision))
    metrics["coco/bbox_AR_300"] = float(np.mean(valid_recall))
    metrics["extended_metric_protocol"] = {
        "mAP90": "COCO AP at IoU=0.90, area=all, max_detections=300",
        "Recall": "COCO AR averaged over IoU=0.50:0.95, area=all, max_detections=300",
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def _copy_log(run_dir: Path, destination_name: str) -> None:
    source = run_dir / "console.log"
    if source.is_file():
        shutil.copy2(source, run_dir / destination_name)


def _prune(run_dir: Path, keep: set[str]) -> None:
    resolved = run_dir.resolve(strict=True)
    for path in run_dir.iterdir():
        if path.name in keep:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    if run_dir.resolve(strict=True) != resolved:
        raise RuntimeError("result directory changed while pruning")


def finalize_training(run_dir: Path, dataset_dir: Path, metadata: dict[str, Any] | None = None) -> None:
    distribution = dataset_distribution(dataset_dir, "train")
    epochs, losses, metric = read_epoch_losses(run_dir)
    write_distribution_plot(distribution, run_dir / "sample_distribution.png")
    write_loss_plot(epochs, losses, metric, run_dir / "loss_curve.png")
    summary = {
        "run_id": run_dir.name,
        "dataset": dataset_dir.name,
        "input_size": [1024, 1024],
        "sample_distribution": distribution,
        "loss_by_epoch": [
            {"epoch": epoch, metric: round(loss, 8)} for epoch, loss in zip(epochs, losses)
        ],
    }
    if metadata:
        summary.update(metadata)
    (run_dir / "train_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _copy_log(run_dir, "train.log")
    _prune(run_dir, {"sample_distribution.png", "loss_curve.png", "train.log", "train_results.json"})


def finalize_test(run_dir: Path, dataset_dir: Path, split: str = "test") -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop("coco_eval_artifact", None)
    metrics.pop("predictions", None)
    metrics["dataset"] = dataset_dir.name
    metrics["split"] = split
    metrics["input_size"] = [1024, 1024]
    distribution = dataset_distribution(dataset_dir, split)
    metrics["sample_distribution"] = distribution
    write_distribution_plot(distribution, run_dir / "sample_distribution.png")
    write_metrics_plot(metrics, run_dir / "test_metrics.png")
    (run_dir / "test_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _copy_log(run_dir, "test.log")
    _prune(run_dir, {"sample_distribution.png", "test_metrics.png", "test.log", "test_results.json"})
    return metrics


def finalize_inference(run_dir: Path) -> None:
    current = run_dir / "inference_results.json"
    if not current.is_file():
        raise FileNotFoundError(current)
    _copy_log(run_dir, "inference.log")
    keep = {"inference_results.json", "inference.log"}
    keep.update(path.name for path in run_dir.glob("*.jpg"))
    _prune(run_dir, keep)
