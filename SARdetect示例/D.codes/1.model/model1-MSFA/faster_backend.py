#!/usr/bin/env python3
"""Train the class-agnostic MSFA Faster R-CNN with MMDetection 3.x."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "D.codes" / "1.model" / "model1-MSFA"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from model.detector import DetectorBuildOptions, build_detector_model_cfg  # noqa: E402

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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-from", type=Path, default=None)
    parser.add_argument("--launcher", choices=("none", "pytorch"), default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and dump config without importing MMDetection")
    return parser.parse_args()


def _register_msfa(reference_root: Path) -> None:
    plugin_root = reference_root
    if not plugin_root.is_dir():
        raise FileNotFoundError(f"MSFA plugin root not found: {plugin_root}")
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    importlib.import_module("msfa")


def _dataset_cfg(
    data_root: Path,
    annotation: str,
    image_prefix: str,
    pipeline: list,
    classes: list,
    batch_size: int,
    workers: int,
    train: bool,
) -> Dict[str, Any]:
    dataset = dict(
        type="CocoDataset",
        data_root=str(data_root) + os.sep,
        ann_file=annotation,
        data_prefix=dict(img=image_prefix),
        metainfo=dict(classes=tuple(classes)),
        test_mode=not train,
        pipeline=pipeline,
        backend_args=None,
    )
    if train:
        dataset["filter_cfg"] = dict(filter_empty_gt=False, min_size=8)
    return dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=train),
        batch_sampler=dict(type="AspectRatioBatchSampler") if train else None,
        dataset=dataset,
    )


def build_runner_cfg(raw: Dict[str, Any], run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = raw["data"]
    model_cfg = raw["model"]
    train_cfg = raw["train"]
    runtime_cfg = raw.get("runtime", {})

    data_root = resolve_path(str(data_cfg["root"]), required=True)
    assert data_root is not None
    classes = [str(value) for value in data_cfg.get("classes", ["vehicle"])]
    if classes != ["vehicle"]:
        raise ValueError("The stage-one baseline is intentionally class-agnostic and requires classes: [vehicle]")
    input_size = tuple(int(value) for value in model_cfg.get("input_size", [1024, 1024]))
    options = DetectorBuildOptions(
        architecture=str(model_cfg.get("architecture", "faster_rcnn")),
        num_queries=(
            int(model_cfg["num_queries"])
            if model_cfg.get("num_queries") is not None
            else None
        ),
        input_size=input_size,
        feature_mode=str(model_cfg.get("feature_mode", "wavelet")),
        num_classes=1,
        fpn_channels=int(model_cfg.get("fpn_channels", 256)),
        anchor_scales=tuple(int(value) for value in model_cfg.get("anchor_scales", [8])),
        anchor_ratios=tuple(float(value) for value in model_cfg.get("anchor_ratios", [0.5, 1.0, 2.0])),
        score_threshold=float(model_cfg.get("score_threshold", 0.01)),
        nms_iou_threshold=float(model_cfg.get("nms_iou_threshold", 0.5)),
        max_detections=int(model_cfg.get("max_detections", 300)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.4)),
        roi_bbox_loss_type=str(model_cfg.get("roi_bbox_loss_type", "L1Loss")),
        roi_bbox_loss_weight=float(model_cfg.get("roi_bbox_loss_weight", 1.0)),
    )
    height, width = input_size
    train_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="Resize", scale=(width, height), keep_ratio=False),
    ]
    domain_aug = data_cfg.get("sar_domain_randomization", {})
    if bool(domain_aug.get("enabled", False)):
        train_pipeline.append(
            dict(
                type="SARDomainRandomization",
                prob=float(domain_aug.get("probability", 0.8)),
                gamma_range=list(domain_aug.get("gamma_range", [0.65, 1.55])),
                contrast_range=list(domain_aug.get("contrast_range", [0.75, 1.30])),
                gain_range=list(domain_aug.get("gain_range", [0.85, 1.15])),
                speckle_std_range=list(
                    domain_aug.get("speckle_std_range", [0.0, 0.12])
                ),
                log_strength_range=list(
                    domain_aug.get("log_strength_range", [0.0, 8.0])
                ),
            )
        )
    train_pipeline.extend(
        [
            dict(type="RandomFlip", prob=float(data_cfg.get("flip_probability", 0.5))),
            dict(type="PackDetInputs"),
        ]
    )
    val_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="Resize", scale=(width, height), keep_ratio=False),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    workers = int(train_cfg.get("workers", 4))
    shared_image_prefix = str(data_cfg.get("image_prefix", "images/"))
    train_dataloader = _dataset_cfg(
        data_root,
        str(data_cfg["train_annotations"]),
        str(data_cfg.get("train_image_prefix", shared_image_prefix)),
        train_pipeline,
        classes,
        int(train_cfg.get("batch_size", 2)),
        workers,
        True,
    )
    val_dataloader = _dataset_cfg(
        data_root,
        str(data_cfg["val_annotations"]),
        str(data_cfg.get("val_image_prefix", shared_image_prefix)),
        val_pipeline,
        classes,
        int(train_cfg.get("val_batch_size", 1)),
        workers,
        False,
    )
    test_annotation_name = str(data_cfg.get("test_annotations", data_cfg["val_annotations"]))
    test_image_prefix = str(
        data_cfg.get(
            "test_image_prefix",
            data_cfg.get("val_image_prefix", shared_image_prefix),
        )
    )
    test_dataloader = _dataset_cfg(
        data_root,
        test_annotation_name,
        test_image_prefix,
        val_pipeline,
        classes,
        int(train_cfg.get("test_batch_size", train_cfg.get("val_batch_size", 1))),
        workers,
        False,
    )
    val_annotation = data_root / str(data_cfg["val_annotations"])
    if not val_annotation.is_file():
        raise FileNotFoundError(val_annotation)
    test_annotation = data_root / test_annotation_name
    if not test_annotation.is_file():
        raise FileNotFoundError(test_annotation)

    epochs = int(train_cfg.get("epochs", 24))
    milestones = [int(value) for value in train_cfg.get("milestones", [16, 22])]
    amp = bool(train_cfg.get("amp", False))
    optim_type = "AmpOptimWrapper" if amp else "OptimWrapper"
    accumulative_counts = int(train_cfg.get("accumulative_counts", 1))
    if accumulative_counts < 1:
        raise ValueError("train.accumulative_counts must be positive")
    optim_wrapper_cfg: Dict[str, Any] = dict(
        type=optim_type,
        optimizer=dict(
            type="AdamW",
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        ),
        clip_grad=dict(
            max_norm=float(train_cfg.get("clip_grad_norm", 1.0)), norm_type=2
        ),
        accumulative_counts=accumulative_counts,
    )
    if train_cfg.get("backbone_lr_multiplier") is not None:
        optim_wrapper_cfg["paramwise_cfg"] = dict(
            custom_keys=dict(
                backbone=dict(
                    lr_mult=float(train_cfg["backbone_lr_multiplier"]),
                    decay_mult=float(train_cfg.get("backbone_decay_multiplier", 1.0)),
                )
            )
        )
    param_schedulers = []
    warmup_iters = int(train_cfg.get("warmup_iters", 1000))
    if warmup_iters > 0:
        param_schedulers.append(
            dict(
                type="LinearLR",
                start_factor=0.001,
                by_epoch=False,
                begin=0,
                end=warmup_iters,
            )
        )
    param_schedulers.append(
        dict(
            type="MultiStepLR",
            begin=0,
            end=epochs,
            by_epoch=True,
            milestones=milestones,
            gamma=0.1,
        )
    )
    launcher = args.launcher or ("pytorch" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "none")
    resume_requested = bool(args.resume or train_cfg.get("resume", False))
    if resume_requested and args.load_from is None:
        # Auto-resume must discover work_dir/last_checkpoint.  Keeping the
        # initialization checkpoint in load_from makes MMEngine try to resume
        # optimizer/epoch state from a weights-only artifact.
        load_from = None
    else:
        load_from = args.load_from or resolve_path(model_cfg.get("pretrained_detector"), required=False)

    cfg: Dict[str, Any] = dict(
        default_scope="mmdet",
        model=build_detector_model_cfg(options),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        val_evaluator=dict(
            type="CocoMetric",
            ann_file=str(val_annotation),
            metric="bbox",
            classwise=True,
            format_only=False,
            backend_args=None,
        ),
        test_evaluator=dict(
            type="CocoMetric",
            ann_file=str(test_annotation),
            metric="bbox",
            classwise=True,
            format_only=False,
            backend_args=None,
        ),
        train_cfg=dict(
            type="EpochBasedTrainLoop",
            max_epochs=epochs,
            val_interval=int(train_cfg.get("val_interval", 1)),
        ),
        val_cfg=dict(type="ValLoop"),
        test_cfg=dict(type="TestLoop"),
        optim_wrapper=optim_wrapper_cfg,
        param_scheduler=param_schedulers,
        auto_scale_lr=dict(enable=False, base_batch_size=16),
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=int(train_cfg.get("log_interval", 50))),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(
                type="CheckpointHook",
                interval=int(train_cfg.get("checkpoint_interval", 1)),
                max_keep_ckpts=int(train_cfg.get("max_keep_checkpoints", 3)),
                save_best="coco/bbox_mAP",
                rule="greater",
            ),
            sampler_seed=dict(type="DistSamplerSeedHook"),
            visualization=dict(type="DetVisualizationHook", draw=False),
        ),
        custom_hooks=[],
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
            dist_cfg=dict(backend=str(runtime_cfg.get("distributed_backend", "hccl"))),
        ),
        vis_backends=[dict(type="LocalVisBackend")],
        visualizer=dict(type="DetLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")], name="visualizer"),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        log_level="INFO",
        launcher=launcher,
        work_dir=str(run_dir),
        load_from=str(load_from) if load_from is not None else None,
        resume=resume_requested,
        randomness=dict(
            seed=int(runtime_cfg.get("seed", 3407)),
            deterministic=bool(runtime_cfg.get("deterministic", False)),
        ),
    )
    if amp:
        cfg["optim_wrapper"]["loss_scale"] = "dynamic"
    return cfg


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    output_root = resolve_path(str(raw["run"].get("output_root", "3.experiments")))
    assert output_root is not None
    run_id = args.run_id or str(raw["run"]["id"])
    run_dir = prepare_run_directory(output_root, run_id, raw)
    cfg_dict = build_runner_cfg(raw, run_dir, args)
    from training_native import mmengine_config
    mmengine_config(cfg_dict,raw)
    write_json(run_dir / "mmdet_config.resolved.json", cfg_dict)

    reference_root = resolve_path(
        str(raw.get("references", {}).get("sardet_100k", "4.refer/SARDet_100K")), required=True
    )
    assert reference_root is not None
    reference_commit = "unknown"
    try:
        reference_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(reference_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    write_json(
        run_dir / "reference_provenance.json",
        {"sardet_100k": str(reference_root), "commit": reference_commit},
    )
    load_from = cfg_dict.get("load_from")
    if load_from:
        checkpoint_path = Path(load_from)
        write_json(
            run_dir / "initialization_provenance.json",
            {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        )
    val_ann = resolve_path(str(raw["data"]["root"]), required=True) / str(raw["data"]["val_annotations"])
    (run_dir / "val_annotations_sha256.txt").write_text(sha256_file(val_ann), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(cfg_dict, ensure_ascii=False, indent=2, default=str))
        return

    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Detector training is restricted to the remote Ascend host with torch-npu installed") from error
    _register_msfa(reference_root)
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as error:
        raise RuntimeError("MMEngine/MMDetection 3.x is required; run preflight_ascend.py before installing blindly") from error
    if bool(raw.get("data", {}).get("sar_domain_randomization", {}).get("enabled", False)):
        importlib.import_module("model.sar_transforms")

    runner = Runner.from_cfg(Config(cfg_dict))
    runner.train()


if __name__ == "__main__":
    main()
