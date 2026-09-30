#!/usr/bin/env python3
"""Run the official RT-DETRv2-R34 implementation on one Ascend NPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = Path(__file__).resolve().parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

sys.path.insert(0,str(PROJECT_ROOT/'D.codes/2.utils'))

from common import (  # noqa: E402
    load_yaml,
    prepare_run_directory,
    resolve_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dump-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _transforms(size: int, train: bool) -> Dict[str, Any]:
    ops = []
    if train:
        ops.append(dict(type="RandomHorizontalFlip"))
    ops.extend(
        [
            dict(type="Resize", size=[size, size]),
            dict(type="SanitizeBoundingBoxes", min_size=1),
            dict(type="ConvertPILImage", dtype="float32", scale=True),
            dict(type="ConvertBoxes", fmt="cxcywh", normalize=True),
        ]
    )
    return dict(type="Compose", ops=ops)


def build_overrides(raw: Dict[str, Any], run_dir: Path, args: argparse.Namespace) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    refs = raw["references"]
    root = resolve_path(str(refs["rtdetrv2"]), required=True)
    assert root is not None
    source_root = root / "rtdetrv2_pytorch"
    base_config = source_root / "configs" / "rtdetrv2" / "rtdetrv2_r34vd_120e_coco.yml"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    data_root = resolve_path(str(raw["data"]["root"]), required=True)
    checkpoint = resolve_path(str(raw["model"]["pretrained_detector"]), required=True)
    assert data_root is not None and checkpoint is not None
    image_root = data_root / str(raw["data"].get("image_prefix", "JPEGImages/"))
    split_key = "val_annotations" if args.split == "val" else "test_annotations"
    val_annotation = str(raw["data"][split_key] if args.mode == "test" else raw["data"]["val_annotations"])
    train_cfg = raw["train"]
    size = int(raw["model"].get("input_size", [800, 800])[0])
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    backbone_lr_multiplier = float(train_cfg.get("backbone_lr_multiplier", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    milestones = [int(value) for value in train_cfg.get("milestones", [16, 22])]
    warmup_steps = int(train_cfg.get("warmup_iters", 300))
    multiscale_sizes = [int(value) for value in raw["data"].get("multiscale_sizes", [])]
    if any(value <= 0 or value % 32 for value in multiscale_sizes):
        raise ValueError("data.multiscale_sizes must contain positive multiples of 32")
    use_ema = bool(train_cfg.get("use_ema", False))
    overrides: Dict[str, Any] = dict(
        device=args.device,
        seed=int(raw.get("runtime", {}).get("seed", 3407)),
        output_dir=str(run_dir),
        num_classes=1,
        # Official COCO configs cache evaluation positional embeddings for
        # 640x640.  This experiment is locked to 800x800; leaving the cache at
        # 640 produces a 25x25-vs-20x20 token mismatch only during validation.
        eval_spatial_size=[size, size],
        remap_mscoco_category=True,
        epoches=int(train_cfg.get("epochs", 24)),
        clip_max_norm=float(train_cfg.get("clip_grad_norm", 1.0)),
        checkpoint_freq=int(train_cfg.get("checkpoint_interval", 1)),
        print_freq=int(train_cfg.get("log_interval", 20)),
        sync_bn=False,
        use_amp=False,
        use_ema=use_ema,
        ema=dict(
            type="ModelEMA",
            decay=float(train_cfg.get("ema_decay", 0.9999)),
            warmups=int(train_cfg.get("ema_warmups", 2000)),
        ),
        PResNet=dict(pretrained=False),
        train_dataloader=dict(
            dataset=dict(
                img_folder=str(image_root),
                ann_file=str(data_root / str(raw["data"]["train_annotations"])),
                transforms=_transforms(size, train=True),
            ),
            total_batch_size=int(train_cfg.get("batch_size", 2)),
            num_workers=int(train_cfg.get("workers", 4)),
            drop_last=False,
            collate_fn=dict(
                type="BatchImageCollateFunction",
                scales=multiscale_sizes or None,
                stop_epoch=int(train_cfg.get("multiscale_stop_epoch", 1000000)),
            ),
        ),
        val_dataloader=dict(
            dataset=dict(
                img_folder=str(image_root),
                ann_file=str(data_root / val_annotation),
                transforms=_transforms(size, train=False),
            ),
            total_batch_size=int(train_cfg.get("val_batch_size", 1)),
            num_workers=int(train_cfg.get("workers", 4)),
            drop_last=False,
            collate_fn=dict(type="BatchImageCollateFunction", scales=None, stop_epoch=1000000),
        ),
        optimizer=dict(
            params=[
                dict(params=r"^(?=.*backbone)(?!.*norm|bn).*$", lr=base_lr * backbone_lr_multiplier),
                dict(
                    params=r"^(?=.*backbone)(?=.*norm|bn).*$",
                    lr=base_lr * backbone_lr_multiplier,
                    weight_decay=0.0,
                ),
                dict(
                    params=r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$",
                    weight_decay=0.0,
                ),
            ],
            lr=base_lr,
            weight_decay=weight_decay,
        ),
        lr_scheduler=dict(type="MultiStepLR", milestones=milestones, gamma=0.1),
        lr_warmup_scheduler=dict(type="LinearWarmup", warmup_duration=warmup_steps),
    )
    if args.mode == "train":
        overrides["tuning"] = str(checkpoint)
        overrides["resume"] = None
        if raw.get('training_control',{}).get('resume'):
            overrides['resume']=str(Path(raw['training_control']['checkpoint_dir'])/'last.pth')
            overrides['tuning']=None
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in test mode")
        eval_checkpoint = args.checkpoint.resolve()
        if not eval_checkpoint.is_file():
            raise FileNotFoundError(eval_checkpoint)
        overrides["tuning"] = None
        overrides["resume"] = str(eval_checkpoint)
    provenance = dict(
        architecture="rtdetrv2_r34",
        implementation="official lyuwenyu/RT-DETR rtdetrv2_pytorch",
        source_root=str(root),
        source_commit="068dfde65f2667ad6555883c69d73de886518cad",
        official_config=str(base_config),
        official_config_sha256=sha256_file(base_config),
        pretrained_checkpoint=str(checkpoint),
        pretrained_sha256=sha256_file(checkpoint),
    )
    if args.mode == "test" and args.checkpoint is not None:
        provenance["evaluation_checkpoint"] = str(args.checkpoint.resolve())
        provenance["evaluation_checkpoint_sha256"] = sha256_file(args.checkpoint.resolve())
    return source_root, overrides, provenance


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    output_root = resolve_path(str(raw["run"].get("output_root", "3.experiments")))
    assert output_root is not None
    base_id = args.run_id or str(raw["run"]["id"])
    run_id = base_id
    run_dir = prepare_run_directory(output_root, run_id, raw)
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "architecture": "rtdetrv2_r34"}, indent=2))
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("This experiment must run on the remote Ascend host") from error
    source_root, overrides, provenance = build_overrides(raw, run_dir, args)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS

    base_config = Path(provenance["official_config"])
    write_json(run_dir / "rtdetrv2_overrides.resolved.json", overrides)
    write_json(run_dir / "architecture_provenance.json", provenance)
    dist_utils.setup_distributed(seed=int(overrides["seed"]))
    cfg = YAMLConfig(str(base_config), **overrides)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.mode == "train":
        if 'training_control' in raw:
            from training_native import rtdetr_fit
            rtdetr_fit(solver,raw)
        else:solver.fit()
    else:
        solver.val()
        metrics_path = run_dir / "eval.pth"
        stats = solver.evaluator.coco_eval["bbox"].stats.tolist()
        predictions_path = None
        if args.dump_predictions:
            import torch

            module = solver.ema.module if solver.ema else solver.model
            module.eval()
            predictions = []
            with torch.no_grad():
                for samples, targets in solver.val_dataloader:
                    samples = samples.to(solver.device)
                    targets = [{key: value.to(solver.device) for key, value in target.items()} for target in targets]
                    outputs = module(samples)
                    original_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
                    results = solver.postprocessor(outputs, original_sizes)
                    for target, result in zip(targets, results):
                        image_id = int(target["image_id"].item())
                        # This is a one-class detector.  RT-DETRv2's COCO
                        # postprocessor may expose the foreground label as 0
                        # or remap it to the COCO category id 1, depending on
                        # ``remap_mscoco_category``.  Both represent the same
                        # vehicle class here, so do not discard label 1 while
                        # serializing predictions.
                        for box, score, _label in zip(result["boxes"], result["scores"], result["labels"]):
                            x1, y1, x2, y2 = (float(value) for value in box.detach().cpu().tolist())
                            predictions.append(
                                {
                                    "image_id": image_id,
                                    "category_id": 1,
                                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                    "score": float(score.detach().cpu().item()),
                                }
                            )
            predictions_path = run_dir / "predictions.coco.json"
            predictions_path.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
        write_json(
            run_dir / "metrics.json",
            {
                "status": "completed",
                "coco/bbox_mAP": float(stats[0]),
                "coco/bbox_mAP_50": float(stats[1]),
                "coco/bbox_mAP_75": float(stats[2]),
                "coco/bbox_mAP_s": float(stats[3]),
                "coco/bbox_mAP_m": float(stats[4]),
                "coco/bbox_mAP_l": float(stats[5]),
                "coco_eval_stats": stats,
                "coco_eval_artifact": str(metrics_path),
                "predictions": str(predictions_path) if predictions_path else None,
            },
        )
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
