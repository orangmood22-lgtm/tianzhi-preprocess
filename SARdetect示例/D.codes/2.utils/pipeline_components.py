#!/usr/bin/env python3
"""Optional second-stage and SOC40 inference components for SARdetect."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'1.model/modelc-ResCNN'),str(Path(__file__).resolve().parents[1]/'1.model/modeld-ResidualRefiner')]
from residual_encoder import ResidualBlock,LocalEncoder
from rescnn import ATRClassifier
from residual_refiner import ResidualPixelRefiner
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_device(name: str) -> torch.device:
    if name.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(name)
    return torch.device(name)


def image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    low, high = np.percentile(array, (1.0, 99.0))
    if high > low + 1e-6:
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)










class ResNet18HbbClassifier(nn.Module):
    """Stage 10 single-view, one-channel torchvision ResNet-18 classifier."""

    def __init__(self, classes: int):
        super().__init__()
        from torchvision.models import resnet18

        network = resnet18(weights=None)
        network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        network.fc = nn.Identity()
        self.encoder = network
        self.head = nn.Linear(512, classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))


class EfficientNetB0HbbClassifier(nn.Module):
    """Stage 11 one-channel EfficientNet-B0 with an optional stride-1 stem."""

    def __init__(self, classes: int, *, small_stem: bool = False, dropout: float = 0.2):
        super().__init__()
        from torchvision.models import efficientnet_b0

        network = efficientnet_b0(weights=None)
        network.features[0][0] = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            stride=1 if small_stem else 2,
            padding=1,
            bias=False,
        )
        channels = network.classifier[1].in_features
        network.classifier = nn.Identity()
        self.encoder = network
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, classes))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def square_crop(box: list[float], context: float) -> list[float]:
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    side = max(box[2] - box[0], box[3] - box[1]) * context
    return [
        center_x - side / 2.0,
        center_y - side / 2.0,
        center_x + side / 2.0,
        center_y + side / 2.0,
    ]


def encode_box(box: list[float], crop: list[float]) -> list[float]:
    width = crop[2] - crop[0]
    height = crop[3] - crop[1]
    center_x = ((box[0] + box[2]) / 2.0 - crop[0]) / width
    center_y = ((box[1] + box[3]) / 2.0 - crop[1]) / height
    box_width = max(1e-3, box[2] - box[0]) / width
    box_height = max(1e-3, box[3] - box[1]) / height
    return [center_x, center_y, math.log(box_width), math.log(box_height)]


def decode_box(values: list[float], crop: list[float]) -> list[float]:
    center_x = min(max(float(values[0]), -0.5), 1.5)
    center_y = min(max(float(values[1]), -0.5), 1.5)
    width = math.exp(min(max(float(values[2]), -4.5), 0.7))
    height = math.exp(min(max(float(values[3]), -4.5), 0.7))
    crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
    absolute_x = crop[0] + center_x * crop_width
    absolute_y = crop[1] + center_y * crop_height
    absolute_width, absolute_height = width * crop_width, height * crop_height
    return [
        absolute_x - absolute_width / 2.0,
        absolute_y - absolute_height / 2.0,
        absolute_x + absolute_width / 2.0,
        absolute_y + absolute_height / 2.0,
    ]


COMPONENT_STORAGE_ALIASES = {
    "hivit-base-orientation-hbb224-e259": "model-c-s",
    "rescnn256-1ch-hbb128": "model-c",
    "resnet18-1ch-hbb192": "model-c-s",
    "efficientnet-b0-1ch-hbb192-ensemble3": "model-c-s",
}


def component_storage_key(model_key: str) -> str:
    return COMPONENT_STORAGE_ALIASES.get(model_key, model_key)


def latest_component_record(project_root: Path, model_key: str) -> tuple[dict[str, Any], Path]:
    storage_key = component_storage_key(model_key)
    registry_path = project_root / "B.models" / storage_key / "modellist.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = [
        row for row in registry.get("models", [])
        if row.get("status") in {"trained", "ready", "current", "candidate"}
    ]
    if not records:
        raise FileNotFoundError(f"no usable checkpoint registered for {model_key}")
    exact = [r for r in records if model_key in {r.get("name"), r.get("canonical_name")}]
    record = exact[-1] if exact else records[-1]
    members = record.get("members")
    if isinstance(members, list) and members:
        definition = (project_root / record["ensemble_path"]).resolve(strict=True)
        expected_definition = record.get("ensemble_sha256")
        if expected_definition and sha256_file(definition) != expected_definition:
            raise ValueError(f"ensemble definition hash mismatch for {model_key}: {definition}")
        for member in members:
            checkpoint = (project_root / member["checkpoint_path"]).resolve(strict=True)
            expected = member.get("checkpoint_sha256")
            if expected and sha256_file(checkpoint) != expected:
                raise ValueError(f"checkpoint hash mismatch for {model_key}: {checkpoint}")
        return record, definition
    checkpoint = (project_root / record["checkpoint_path"]).resolve(strict=True)
    expected = record.get("checkpoint_sha256")
    if expected and sha256_file(checkpoint) != expected:
        raise ValueError(f"checkpoint hash mismatch for {model_key}: {checkpoint}")
    return record, checkpoint


def classifier_model(
    architecture: str,
    classes: int,
    *,
    small_stem: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    if architecture in {"localencoder_residual_cnn_1channel_256", "atrclassifier"}:
        return ATRClassifier(classes)
    if architecture == "torchvision_resnet18_1channel":
        return ResNet18HbbClassifier(classes)
    if architecture in {"efficientnet_b0", "torchvision_efficientnet_b0_1channel"}:
        return EfficientNetB0HbbClassifier(
            classes,
            small_stem=small_stem,
            dropout=dropout,
        )
    raise ValueError(f"unsupported classifier architecture: {architecture}")


def _load_image(image_root: Path, image_rows: dict[int, dict], image_id: int) -> Image.Image:
    path = image_root / Path(image_rows[image_id]["file_name"]).name
    image = Image.open(path)
    image.load()
    return image


def refine_predictions(
    project_root: Path,
    model_key: str,
    predictions: list[dict[str, Any]],
    coco_payload: dict[str, Any],
    image_root: Path,
    device_name: str,
    batch_size: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record, checkpoint_path = latest_component_record(project_root, model_key)
    options = record.get("inference", {})
    image_size = int(options.get("image_size", 128))
    context = float(options.get("context", 1.75))
    threshold = float(options.get("score_threshold", 0.02))
    alpha = float(options.get("alpha", 1.0))
    score_mode = str(options.get("score_mode", "original"))
    if score_mode not in {"original", "verified"}:
        raise ValueError(f"unsupported refiner score mode: {score_mode}")

    device = setup_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResidualPixelRefiner().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    image_rows = {int(row["id"]): row for row in coco_payload["images"]}

    eligible: list[tuple[int, int, list[float], list[float]]] = []
    for index, row in enumerate(predictions):
        if float(row["score"]) < threshold:
            continue
        original = xywh_to_xyxy(row["bbox"])
        crop = square_crop(original, context)
        image_id = int(row["image_id"])
        eligible.append((index, image_id, crop, encode_box(original, crop)))

    decoded: dict[int, tuple[list[float], float]] = {}
    with torch.no_grad():
        for start in range(0, len(eligible), batch_size):
            stop = min(start + batch_size, len(eligible))
            tensors = []
            for _, image_id, crop, _ in eligible[start:stop]:
                with _load_image(image_root, image_rows, image_id) as image:
                    tensors.append(image_tensor(image.crop(tuple(crop)), image_size))
            images = torch.stack(tensors).to(device)
            geometry = torch.tensor(
                [eligible[index][3] for index in range(start, stop)],
                dtype=torch.float32,
                device=device,
            )
            boxes, logits = model(images, geometry)
            for source, values, object_score in zip(
                eligible[start:stop], boxes.float().cpu().numpy(),
                torch.sigmoid(logits).float().cpu().numpy(),
            ):
                decoded[source[0]] = (decode_box(values, source[2]), float(object_score))

    output: list[dict[str, Any]] = []
    for index, item in enumerate(predictions):
        row = dict(item)
        if index in decoded:
            original = xywh_to_xyxy(item["bbox"])
            refined, object_score = decoded[index]
            mixed = [(1.0 - alpha) * left + alpha * right for left, right in zip(original, refined)]
            info = image_rows[int(item["image_id"])]
            mixed[0] = min(max(mixed[0], 0.0), float(info["width"]) - 1.0)
            mixed[1] = min(max(mixed[1], 0.0), float(info["height"]) - 1.0)
            mixed[2] = min(max(mixed[2], mixed[0] + 1.0), float(info["width"]))
            mixed[3] = min(max(mixed[3], mixed[1] + 1.0), float(info["height"]))
            row["bbox"] = [mixed[0], mixed[1], mixed[2] - mixed[0], mixed[3] - mixed[1]]
            if score_mode == "verified":
                row["score"] = float(row["score"]) * object_score
        output.append(row)

    summary = {
        "model": model_key,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "eligible_detections": len(eligible),
        "inference": {
            "image_size": image_size,
            "context": context,
            "score_threshold": threshold,
            "alpha": alpha,
            "score_mode": score_mode,
        },
    }
    return output, summary


def _class_names(checkpoint: dict[str, Any], record: dict[str, Any]) -> list[str]:
    classes = checkpoint.get("metadata", {}).get("classes") or record.get("classes")
    if isinstance(classes, dict):
        result = [None] * len(classes)
        for name, index in classes.items():
            result[int(index)] = str(name)
        if any(name is None for name in result):
            raise ValueError("classifier class mapping is not contiguous")
        return [str(name) for name in result]
    if isinstance(classes, list) and classes:
        return [str(name) for name in classes]
    raise ValueError("classifier checkpoint does not contain a class mapping")


def classify_predictions(
    project_root: Path,
    model_key: str,
    predictions: list[dict[str, Any]],
    coco_payload: dict[str, Any],
    image_root: Path,
    device_name: str,
    score_threshold: float = 0.25,
    batch_size: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record, artifact_path = latest_component_record(project_root, model_key)
    if record.get("architecture") == "hivit_base_orientation_e259":
        import importlib.util
        spec = importlib.util.spec_from_file_location("sardetect_e259", project_root / record["implementation_path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.classify_predictions(project_root, record, artifact_path, predictions,
            coco_payload, image_root, device_name, score_threshold, batch_size, model_key)
    options = record.get("inference", {})
    image_size = int(options.get("image_size", 128))
    context = float(options.get("context", 2.0))
    device = setup_device(device_name)
    member_specs = record.get("members")
    if not isinstance(member_specs, list) or not member_specs:
        member_specs = [
            {
                "run_id": record.get("source_experiment", record.get("name", model_key)),
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record.get("checkpoint_sha256"),
                "architecture": record.get("architecture"),
                "weight": 1.0,
            }
        ]
    loaded_members = []
    class_names = None
    for member in member_specs:
        checkpoint_path = (project_root / member["checkpoint_path"]).resolve(strict=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        member_classes = _class_names(checkpoint, record)
        if class_names is None:
            class_names = member_classes
        elif member_classes != class_names:
            raise ValueError("classifier ensemble members use different class mappings")
        metadata = checkpoint.get("metadata", {})
        config = metadata.get("config", {})
        architecture = str(
            member.get("architecture")
            or metadata.get("architecture")
            or record.get("architecture")
            or "atrclassifier"
        )
        small_stem = bool(member.get("small_stem", config.get("small_stem", False)))
        dropout = float(member.get("dropout", config.get("dropout", 0.2)))
        model = classifier_model(
            architecture,
            len(member_classes),
            small_stem=small_stem,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        loaded_members.append(
            {
                "model": model,
                "run_id": str(member.get("run_id", checkpoint_path.stem)),
                "weight": float(member.get("weight", 1.0)),
                "architecture": architecture,
                "small_stem": small_stem,
                "checkpoint": checkpoint_path,
            }
        )
    assert class_names is not None
    weight_sum = sum(item["weight"] for item in loaded_members)
    if weight_sum <= 0.0:
        raise ValueError("classifier ensemble weights must sum to a positive value")
    canonical_name = str(record.get("canonical_name") or model_key)
    image_rows = {int(row["id"]): row for row in coco_payload["images"]}

    eligible: list[tuple[int, int, list[float]]] = []
    for index, row in enumerate(predictions):
        if float(row["score"]) < score_threshold:
            continue
        crop = square_crop(xywh_to_xyxy(row["bbox"]), context)
        eligible.append((index, int(row["image_id"]), crop))

    classifications: dict[int, tuple[str, float, list[dict[str, Any]]]] = {}
    with torch.no_grad():
        for start in range(0, len(eligible), batch_size):
            stop = min(start + batch_size, len(eligible))
            tensors = []
            for _, image_id, crop in eligible[start:stop]:
                with _load_image(image_root, image_rows, image_id) as image:
                    tensors.append(image_tensor(image.crop(tuple(crop)), image_size))
            images = torch.stack(tensors).to(device)
            probabilities = sum(
                item["weight"] * torch.softmax(item["model"](images), dim=1)
                for item in loaded_members
            ) / weight_sum
            top_scores, top_indices = probabilities.topk(min(5, len(class_names)), dim=1)
            for source, indices, scores in zip(
                eligible[start:stop], top_indices.cpu().tolist(), top_scores.cpu().tolist()
            ):
                top5 = [
                    {"label": class_names[int(class_index)], "score": float(class_score)}
                    for class_index, class_score in zip(indices, scores)
                ]
                classifications[source[0]] = (top5[0]["label"], top5[0]["score"], top5)

    output = []
    for index, item in enumerate(predictions):
        row = dict(item)
        if index in classifications:
            class_name, class_score, top5 = classifications[index]
            row["classification"] = {
                "model": canonical_name,
                "label": class_name,
                "score": class_score,
                "top5": top5,
            }
        output.append(row)
    summary = {
        "model": canonical_name,
        "requested_model": model_key,
        "storage_directory": component_storage_key(model_key),
        "display_name": record.get("display_name", canonical_name),
        "architecture": record.get("architecture", loaded_members[0]["architecture"]),
        "artifact": str(artifact_path.relative_to(project_root)),
        "artifact_sha256": sha256_file(artifact_path),
        "members": [
            {
                "run_id": item["run_id"],
                "weight": item["weight"] / weight_sum,
                "architecture": item["architecture"],
                "small_stem": item["small_stem"],
                "checkpoint": str(item["checkpoint"].relative_to(project_root)),
                "checkpoint_sha256": sha256_file(item["checkpoint"]),
            }
            for item in loaded_members
        ],
        "classified_detections": len(classifications),
        "score_threshold": score_threshold,
        "image_size": image_size,
        "context": context,
        "classes": class_names,
    }
    return output, summary


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_classification(
    predictions: list[dict[str, Any]],
    coco_payload: dict[str, Any],
    manifest_path: Path,
    split: str,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Evaluate classified detections against synthetic donor labels."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_to_index = {str(name): int(index) for name, index in manifest["class_to_index"].items()}
    samples = manifest.get("split_reports", {}).get(split, {}).get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"classification manifest has no {split!r} samples: {manifest_path}")
    image_id_by_name = {
        Path(str(row["file_name"])).name: int(row["id"])
        for row in coco_payload["images"]
    }
    donors_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        image_name = Path(str(sample["file_name"])).name
        if image_name not in image_id_by_name:
            raise ValueError(f"classification manifest image is absent from COCO ground truth: {image_name}")
        image_id = image_id_by_name[image_name]
        for donor in sample.get("donors", []):
            category = str(donor["category"])
            if category not in class_to_index:
                raise ValueError(f"unknown donor class {category!r} in {manifest_path}")
            donors_by_image[image_id].append(donor)

    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if isinstance(row.get("classification"), dict):
            predictions_by_image[int(row["image_id"])].append(row)

    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for image_id, donors in donors_by_image.items():
        used: set[int] = set()
        rows = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        for prediction in rows:
            prediction_box = xywh_to_xyxy(prediction["bbox"])
            choices = [
                (box_iou(prediction_box, donor["bbox"]), index)
                for index, donor in enumerate(donors)
                if index not in used
            ]
            if not choices:
                continue
            iou, index = max(choices)
            if iou < match_iou:
                continue
            used.add(index)
            matches.append((prediction, donors[index], iou))

    top1_correct = 0
    top5_correct = 0
    class_total = {name: 0 for name in class_to_index}
    class_correct = {name: 0 for name in class_to_index}
    for prediction, donor, _ in matches:
        target = str(donor["category"])
        classification = prediction["classification"]
        top1 = str(classification["label"])
        top5 = [str(item["label"]) for item in classification.get("top5", [])]
        if not top5:
            top5 = [top1]
        class_total[target] += 1
        if top1 == target:
            top1_correct += 1
            class_correct[target] += 1
        if target in top5:
            top5_correct += 1

    total_ground_truth = sum(len(rows) for rows in donors_by_image.values())
    matched_count = len(matches)
    supported_classes = [name for name, count in class_total.items() if count > 0]
    balanced_accuracy = (
        sum(class_correct[name] / class_total[name] for name in supported_classes) / len(supported_classes)
        if supported_classes else 0.0
    )
    return {
        "status": "evaluated",
        "protocol": "greedy score-ordered detection-to-donor matching",
        "split": split,
        "match_iou": float(match_iou),
        "ground_truth_instances": total_ground_truth,
        "matched_detections": matched_count,
        "matched_recall": matched_count / max(total_ground_truth, 1),
        "matched_mean_iou": sum(item[2] for item in matches) / max(matched_count, 1),
        "matched_top1_accuracy": top1_correct / max(matched_count, 1),
        "matched_top5_accuracy": top5_correct / max(matched_count, 1),
        "matched_balanced_accuracy": balanced_accuracy,
        "joint_detection_top1_recall": top1_correct / max(total_ground_truth, 1),
        "evaluated_classes": len(supported_classes),
        "classification_manifest": str(manifest_path),
        "classification_manifest_sha256": sha256_file(manifest_path),
    }
