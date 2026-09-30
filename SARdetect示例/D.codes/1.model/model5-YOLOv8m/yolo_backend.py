#!/usr/bin/env python3
"""Run the registered Ultralytics YOLO detector on one Ascend NPU."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = PROJECT_ROOT / "D.codes/1.model/model5-YOLOv8m"
ANALYSIS_RESEARCH = PROJECT_ROOT / "D.codes/1.model/model5-YOLOv8m/support"
ANALYSIS_CODE = PROJECT_ROOT / "D.codes/1.model/model5-YOLOv8m/support"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dump-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evaluate(annotation_path: Path, predictions: list[dict[str, Any]], max_det: int) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not predictions:return {"status":"completed","coco/bbox_mAP":0.,"coco/bbox_mAP_50":0.}
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
        detected = ground_truth.loadRes(predictions)
        evaluator = COCOeval(ground_truth, detected, "bbox")
        evaluator.params.maxDets = [1, 100, max_det]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats.tolist()
    return {
        "status": "completed",
        "coco/bbox_mAP": float(stats[0]),
        "coco/bbox_mAP_50": float(stats[1]),
        "coco/bbox_mAP_75": float(stats[2]),
        "coco/bbox_mAP_s": float(stats[3]),
        "coco/bbox_mAP_m": float(stats[4]),
        "coco/bbox_mAP_l": float(stats[5]),
        "coco_eval_stats": stats,
    }


def main() -> None:
    args = parse_args()
    if args.mode != "test":
        raise ValueError("Use SARdetect/train.py for YOLO training")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = Path(raw["run"]["output_root"]).resolve()
    run_id = args.run_id or str(raw["run"]["id"])
    run_dir = output_root / run_id
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "architecture": raw["model"]["architecture"]}, indent=2))
        return
    # The shared runner creates ``run_dir`` first so it can stream console.log.
    # Accept that empty/log-only directory, while refusing to overwrite a
    # completed prediction export.
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "predictions.coco.json").exists() or (run_dir / "metrics.json").exists():
        raise FileExistsError(f"YOLO result already exists: {run_dir}")
    checkpoint = args.checkpoint.resolve(strict=True)
    data_root = Path(raw["data"]["root"]).resolve(strict=True)
    annotation_name = raw["data"][f"{args.split}_annotations"]
    annotation_path = (data_root / annotation_name).resolve(strict=True)
    image_root = Path(raw["data"]["image_prefix"]).resolve(strict=True)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in annotation["images"]}
    image_ids_by_name = {str((image_root / row["file_name"]).resolve()): image_id for image_id, row in image_rows.items()}
    image_paths = [str((image_root / row["file_name"]).resolve(strict=True)) for row in annotation["images"]]

    model_cfg = raw["model"]
    size = int(model_cfg.get("input_size", [1024, 1024])[0])
    score = float(model_cfg.get("score_threshold", 0.001))
    nms_iou = float(model_cfg.get("nms_iou_threshold", 0.7))
    max_det = int(model_cfg.get("max_detections", 300))
    half = bool(model_cfg.get("half", False))
    # This vendored Ultralytics build accepts the Ascend spelling directly.
    # Physical visibility is still constrained by ASCEND_RT_VISIBLE_DEVICES.
    device = args.device

    sys.path[:0] = [str(VENDOR_ROOT), str(ANALYSIS_RESEARCH), str(ANALYSIS_CODE)]
    from runtime_paths import prepare_yolo_environment
    prepare_yolo_environment(run_dir, PROJECT_ROOT)
    from ultralytics import YOLO
    from runtime import complete_nms_batches

    complete_nms_batches()
    model = YOLO(str(checkpoint))
    started = time.monotonic()
    predictions: list[dict[str, Any]] = []
    seen: set[int] = set()
    results = model.predict(
        source=image_paths,
        stream=True,
        device=device,
        imgsz=size,
        batch=1,
        conf=score,
        iou=nms_iou,
        max_det=max_det,
        nms=True,
        augment=False,
        save=False,
        verbose=False,
        half=half,
        project=str(run_dir),
        name="predict",
    )
    for result in results:
        name = str(Path(result.path).resolve())
        image_id = image_ids_by_name[name]
        if image_id in seen:
            raise ValueError(f"duplicate prediction image: {name}")
        seen.add(image_id)
        for x1, y1, x2, y2, confidence, category in result.boxes.data.cpu().tolist():
            values = [x1, y1, x2, y2, confidence]
            if int(category) != 0 or x2 <= x1 or y2 <= y1 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"invalid YOLO prediction for {name}: {values + [category]}")
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": confidence,
                }
            )
    if seen != set(image_rows):
        raise ValueError(f"YOLO output coverage mismatch: {len(seen)} != {len(image_rows)}")

    prediction_path = run_dir / "predictions.coco.json"
    atomic_json(prediction_path, predictions)
    metrics = {"labels_available": False} if os.environ.get("SARDETECT_UNLABELED")=="1" else evaluate(annotation_path, predictions, max_det)
    metrics["predictions"] = str(prediction_path)
    atomic_json(run_dir / "metrics.json", metrics)
    atomic_json(
        run_dir / "receipt.json",
        {
            "split": args.split,
            "images": len(seen),
            "detections": len(predictions),
            "seconds": time.monotonic() - started,
            "weights": str(checkpoint),
            "weights_sha256": sha256_file(checkpoint),
            "annotation_sha256": sha256_file(annotation_path),
            "prediction_sha256": sha256_file(prediction_path),
            "imgsz": size,
            "half": half,
            "conf": score,
            "iou": nms_iou,
            "max_det": max_det,
            "nms": True,
            "augment": False,
        },
    )
    print(json.dumps({"status": "completed", "images": len(seen), "detections": len(predictions), **metrics}, indent=2))


if __name__ == "__main__":
    main()
