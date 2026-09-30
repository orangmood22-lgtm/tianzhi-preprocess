#!/usr/bin/env python3
"""Evaluate a detector checkpoint with the test split in a YAML config."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = Path(__file__).resolve().parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

sys.path.insert(0,str(PROJECT_ROOT/'D.codes/2.utils'))

from common import load_yaml, prepare_run_directory, resolve_path, sha256_file, write_json  # noqa: E402
from faster_backend import build_runner_cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override run.output_root without modifying the training YAML",
    )
    parser.add_argument("--launcher", choices=("none", "pytorch"), default=None)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--dump-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.load_from = None
    args.resume = False
    raw = load_yaml(args.config)
    if args.output_root is not None:
        raw["run"]["output_root"] = str(args.output_root)
    if args.split == "val":
        raw["data"]["test_annotations"] = raw["data"]["val_annotations"]
        raw["data"]["test_image_prefix"] = raw["data"].get(
            "val_image_prefix", raw["data"].get("image_prefix", "images/")
        )
    if args.score_threshold is not None:
        if not 0.0 <= args.score_threshold <= 1.0:
            raise ValueError("--score-threshold must be in [0, 1]")
        raw["model"]["score_threshold"] = args.score_threshold
    if args.nms_iou_threshold is not None:
        if not 0.0 < args.nms_iou_threshold <= 1.0:
            raise ValueError("--nms-iou-threshold must be in (0, 1]")
        raw["model"]["nms_iou_threshold"] = args.nms_iou_threshold
    output_root = resolve_path(str(raw["run"].get("output_root", "3.experiments")))
    assert output_root is not None
    run_dir = prepare_run_directory(output_root, args.run_id, raw)
    cfg = build_runner_cfg(raw, run_dir, args)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg["load_from"] = str(checkpoint)
    write_json(run_dir / "mmdet_config.resolved.json", cfg)
    write_json(
        run_dir / "evaluation_checkpoint.json",
        {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    )
    write_json(
        run_dir / "evaluation_protocol.json",
        {
            "split": args.split,
            "score_threshold": float(raw["model"].get("score_threshold", 0.01)),
            "nms_iou_threshold": float(raw["model"].get("nms_iou_threshold", 0.5)),
            "max_detections": int(raw["model"].get("max_detections", 300)),
        },
    )
    if args.dump_predictions:
        cfg["test_evaluator"] = [
            cfg["test_evaluator"],
            dict(type="DumpDetResults", out_file_path=str(run_dir / "predictions.pkl")),
        ]

    reference_root = resolve_path(str(raw["references"]["sardet_100k"]), required=True)
    plugin_root = reference_root
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    importlib.import_module("msfa")
    if bool(raw.get("data", {}).get("sar_domain_randomization", {}).get("enabled", False)):
        model_root = PROJECT_ROOT / "D.codes" / "1.model"
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        importlib.import_module("model.sar_transforms")
    import torch_npu  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner

    runner = Runner.from_cfg(Config(cfg))
    metrics = runner.test()
    write_json(run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
