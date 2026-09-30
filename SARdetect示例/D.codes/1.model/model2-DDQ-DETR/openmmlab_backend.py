#!/usr/bin/env python3
"""Train/evaluate official OpenMMLab detectors on the fixed SIVED protocol.

This entry point intentionally changes the detector architecture while keeping
the dataset, resize/flip pipeline, optimiser budget, and evaluation protocol
constant.  It supports MMDetection models and the MMYOLO YOLOv8-M model.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

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


ARCHITECTURES: Dict[str, Dict[str, str]] = {
    "rtmdet_m": {
        "framework": "mmdet",
        "config": "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py",
    },
    "ddq_detr_4scale_r50": {
        "framework": "mmdet",
        "config": "configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py",
    },
    "dino_4scale_r50": {
        "framework": "mmdet",
        "config": "configs/dino/dino-4scale_r50_8xb2-12e_coco.py",
    },
    "cascade_rcnn_r50": {
        "framework": "mmdet",
        "config": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_20e_coco.py",
    },
    "yolov8_m": {
        "framework": "mmyolo",
        "config": "configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dump-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _set_num_classes(value: Any, num_classes: int) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "num_classes":
                value[key] = num_classes
            else:
                _set_num_classes(child, num_classes)
    elif isinstance(value, list):
        for child in value:
            _set_num_classes(child, num_classes)


def _clear_backbone_initializers(model: Dict[str, Any]) -> None:
    backbone = model.get("backbone")
    if isinstance(backbone, dict):
        backbone["init_cfg"] = None


def _set_test_thresholds(value: Any, score_thr: float, nms_iou: float, max_det: int) -> None:
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        if key in {"score_thr", "score_threshold"}:
            value[key] = score_thr
        elif key in {"max_per_img", "max_detections"}:
            value[key] = max_det
        elif key == "nms" and isinstance(child, dict) and "iou_threshold" in child:
            child["iou_threshold"] = nms_iou
        else:
            _set_test_thresholds(child, score_thr, nms_iou, max_det)


def _dataset_cfg(
    *,
    framework: str,
    data_root: Path,
    annotation: str,
    image_prefix: str,
    pipeline: list,
    batch_size: int,
    workers: int,
    train: bool,
) -> Dict[str, Any]:
    dataset_type = "YOLOv5CocoDataset" if framework == "mmyolo" else "CocoDataset"
    dataset: Dict[str, Any] = dict(
        type=dataset_type,
        data_root=str(data_root) + os.sep,
        ann_file=annotation,
        data_prefix=dict(img=image_prefix),
        metainfo=dict(classes=("vehicle",)),
        test_mode=not train,
        pipeline=pipeline,
    )
    if train:
        dataset["filter_cfg"] = dict(filter_empty_gt=False, min_size=8)
    loader: Dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=train),
        dataset=dataset,
    )
    if framework == "mmyolo" and train:
        loader["collate_fn"] = dict(type="yolov5_collate")
    elif train:
        loader["batch_sampler"] = dict(type="AspectRatioBatchSampler")
    return loader


def _pipelines(
    framework: str,
    input_size: Iterable[int],
    flip_probability: float,
    multiscale_sizes: Iterable[int] | None = None,
    vertical_flip_probability: float = 0.0,
) -> tuple[list, list]:
    height, width = (int(v) for v in input_size)
    prefix = "mmdet." if framework == "mmyolo" else ""
    resize: Dict[str, Any]
    scales = [int(value) for value in (multiscale_sizes or [])]
    if scales:
        if any(value <= 0 or value % 32 for value in scales):
            raise ValueError("data.multiscale_sizes must contain positive multiples of 32")
        resize = dict(
            type=f"{prefix}RandomChoiceResize",
            scales=[(value, value) for value in scales],
            keep_ratio=False,
        )
    else:
        resize = dict(type=f"{prefix}Resize", scale=(width, height), keep_ratio=False)
    train = [
        dict(type=f"{prefix}LoadImageFromFile", backend_args=None),
        dict(type=f"{prefix}LoadAnnotations", with_bbox=True),
        resize,
        dict(type=f"{prefix}RandomFlip", prob=flip_probability),
    ]
    if vertical_flip_probability > 0:
        train.append(
            dict(
                type=f"{prefix}RandomFlip",
                prob=vertical_flip_probability,
                direction="vertical",
            )
        )
    train.append(dict(type=f"{prefix}PackDetInputs"))
    val = [
        dict(type=f"{prefix}LoadImageFromFile", backend_args=None),
        dict(type=f"{prefix}Resize", scale=(width, height), keep_ratio=False),
        dict(type=f"{prefix}LoadAnnotations", with_bbox=True),
        dict(
            type=f"{prefix}PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    return train, val


def _register_frameworks(mmdet_root: Path, mmyolo_root: Path | None) -> None:
    for root in reversed([p for p in (mmdet_root, mmyolo_root) if p is not None]):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    importlib.import_module("mmdet.utils").register_all_modules(init_default_scope=True)
    if mmyolo_root is not None:
        importlib.import_module("mmyolo.utils").register_all_modules(init_default_scope=True)


def build_runner_cfg(raw: Dict[str, Any], run_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    from mmengine.config import Config

    architecture = str(raw["model"]["architecture"])
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    spec = ARCHITECTURES[architecture]
    refs = raw["references"]
    mmdet_root = resolve_path(str(refs["mmdetection"]), required=True)
    assert mmdet_root is not None
    mmyolo_root = None
    config_root = mmdet_root
    if spec["framework"] == "mmyolo":
        mmyolo_root = resolve_path(str(refs["mmyolo"]), required=True)
        assert mmyolo_root is not None
        config_root = mmyolo_root
    _register_frameworks(mmdet_root, mmyolo_root)

    official_config = (config_root / spec["config"]).resolve()
    if not official_config.is_file():
        raise FileNotFoundError(official_config)
    official = Config.fromfile(str(official_config)).to_dict()
    model = copy.deepcopy(official["model"])
    _set_num_classes(model, 1)
    _clear_backbone_initializers(model)
    _set_test_thresholds(
        model,
        float(raw["model"].get("score_threshold", 0.001)),
        float(raw["model"].get("nms_iou_threshold", 0.8)),
        int(raw["model"].get("max_detections", 300)),
    )

    data = raw["data"]
    train_cfg = raw["train"]
    runtime = raw.get("runtime", {})
    data_root = resolve_path(str(data["root"]), required=True)
    assert data_root is not None
    train_pipeline, val_pipeline = _pipelines(
        spec["framework"],
        raw["model"].get("input_size", [800, 800]),
        float(data.get("flip_probability", 0.5)),
        data.get("multiscale_sizes"),
        float(data.get("vertical_flip_probability", 0.0)),
    )
    image_prefix = str(data.get("image_prefix", "JPEGImages/"))
    workers = int(train_cfg.get("workers", 4))
    train_loader = _dataset_cfg(
        framework=spec["framework"], data_root=data_root,
        annotation=str(data["train_annotations"]), image_prefix=image_prefix,
        pipeline=train_pipeline, batch_size=int(train_cfg.get("batch_size", 2)),
        workers=workers, train=True,
    )
    val_loader = _dataset_cfg(
        framework=spec["framework"], data_root=data_root,
        annotation=str(data["val_annotations"]), image_prefix=image_prefix,
        pipeline=val_pipeline, batch_size=int(train_cfg.get("val_batch_size", 1)),
        workers=workers, train=False,
    )
    test_loader = _dataset_cfg(
        framework=spec["framework"], data_root=data_root,
        annotation=str(data["test_annotations"]), image_prefix=image_prefix,
        pipeline=val_pipeline, batch_size=int(train_cfg.get("test_batch_size", 1)),
        workers=workers, train=False,
    )
    epochs = int(train_cfg.get("epochs", 24))
    warmup_iters = int(train_cfg.get("warmup_iters", 300))
    schedulers = []
    if warmup_iters > 0:
        schedulers.append(dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=warmup_iters))
    schedulers.append(dict(
        type="MultiStepLR", begin=0, end=epochs, by_epoch=True,
        milestones=[int(v) for v in train_cfg.get("milestones", [16, 22])], gamma=0.1,
    ))
    amp = bool(train_cfg.get("amp", False))
    optim_wrapper: Dict[str, Any] = dict(
        type="AmpOptimWrapper" if amp else "OptimWrapper",
        optimizer=dict(
            type="AdamW", lr=float(train_cfg.get("learning_rate", 1e-4)),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        ),
        clip_grad=dict(max_norm=float(train_cfg.get("clip_grad_norm", 1.0)), norm_type=2),
        accumulative_counts=int(train_cfg.get("accumulative_counts", 1)),
        paramwise_cfg=dict(
            custom_keys={"backbone": dict(lr_mult=float(train_cfg.get("backbone_lr_multiplier", 0.1)))},
            # CSPNeXt/RTMDet deliberately shares some convolution parameters.
            bypass_duplicate=True,
        ),
    )
    if amp:
        optim_wrapper["loss_scale"] = "dynamic"
    checkpoint = resolve_path(str(raw["model"]["pretrained_detector"]), required=True)
    assert checkpoint is not None
    scope = "mmyolo" if spec["framework"] == "mmyolo" else "mmdet"
    val_ann = data_root / str(data["val_annotations"])
    test_ann = data_root / str(data["test_annotations"])
    cfg: Dict[str, Any] = dict(
        default_scope=scope,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        val_evaluator=dict(type="mmdet.CocoMetric" if scope == "mmyolo" else "CocoMetric", ann_file=str(val_ann), metric="bbox", classwise=True),
        test_evaluator=dict(type="mmdet.CocoMetric" if scope == "mmyolo" else "CocoMetric", ann_file=str(test_ann), metric="bbox", classwise=True),
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=epochs, val_interval=int(train_cfg.get("val_interval", 1))),
        val_cfg=dict(type="ValLoop"),
        test_cfg=dict(type="TestLoop"),
        optim_wrapper=optim_wrapper,
        param_scheduler=schedulers,
        auto_scale_lr=dict(enable=False, base_batch_size=16),
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=int(train_cfg.get("log_interval", 20))),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(type="CheckpointHook", interval=int(train_cfg.get("checkpoint_interval", 1)), max_keep_ckpts=int(train_cfg.get("max_keep_checkpoints", 2)), save_best="coco/bbox_mAP", rule="greater"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
            visualization=dict(type="mmdet.DetVisualizationHook" if scope == "mmyolo" else "DetVisualizationHook", draw=False),
        ),
        custom_hooks=[],
        env_cfg=dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="hccl")),
        vis_backends=[dict(type="LocalVisBackend")],
        visualizer=dict(type="mmdet.DetLocalVisualizer" if scope == "mmyolo" else "DetLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")], name="visualizer"),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        log_level="INFO",
        launcher="none",
        work_dir=str(run_dir),
        load_from=str(checkpoint),
        resume=False,
        randomness=dict(seed=int(runtime.get("seed", 3407)), deterministic=bool(runtime.get("deterministic", False))),
    )
    provenance = dict(
        architecture=architecture,
        framework=spec["framework"],
        official_config=str(official_config),
        official_config_sha256=sha256_file(official_config),
        mmdetection_commit=_git_commit(mmdet_root),
        mmyolo_commit=_git_commit(mmyolo_root) if mmyolo_root else None,
        pretrained_checkpoint=str(checkpoint),
        pretrained_sha256=sha256_file(checkpoint),
    )
    return cfg, provenance


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    # Bootstrap the pinned source tree before environment_snapshot imports
    # MMDetection.  Otherwise Python caches the older site-package (3.1.0),
    # which does not contain DDQ-DETR.
    bootstrap_mmdet = resolve_path(str(raw["references"]["mmdetection"]), required=True)
    assert bootstrap_mmdet is not None
    bootstrap_mmyolo = None
    architecture = str(raw["model"]["architecture"])
    if ARCHITECTURES.get(architecture, {}).get("framework") == "mmyolo":
        bootstrap_mmyolo = resolve_path(str(raw["references"]["mmyolo"]), required=True)
    _register_frameworks(bootstrap_mmdet, bootstrap_mmyolo)
    output_root = resolve_path(str(raw["run"].get("output_root", "3.experiments")))
    assert output_root is not None
    base_id = args.run_id or str(raw["run"]["id"])
    run_id = base_id
    run_dir = prepare_run_directory(output_root, run_id, raw)

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "architecture": raw["model"]["architecture"]}, indent=2))
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("This experiment must run on the remote Ascend host") from error

    cfg, provenance = build_runner_cfg(raw, run_dir)
    if args.mode == 'train':
        from training_native import mmengine_config
        mmengine_config(cfg,raw)
    if args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in test mode")
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        cfg["load_from"] = str(checkpoint)
        if args.split == "val":
            cfg["test_dataloader"] = copy.deepcopy(cfg["val_dataloader"])
            cfg["test_evaluator"] = copy.deepcopy(cfg["val_evaluator"])
        if args.dump_predictions:
            cfg["test_evaluator"] = [
                cfg["test_evaluator"],
                dict(type="mmdet.DumpDetResults" if cfg["default_scope"] == "mmyolo" else "DumpDetResults", out_file_path=str(run_dir / "predictions.pkl")),
            ]
        provenance["evaluation_checkpoint"] = str(checkpoint)
        provenance["evaluation_checkpoint_sha256"] = sha256_file(checkpoint)
    write_json(run_dir / "mmdet_config.resolved.json", cfg)
    write_json(run_dir / "architecture_provenance.json", provenance)
    from mmengine.config import Config
    from mmengine.runner import Runner
    runner = Runner.from_cfg(Config(cfg))
    if args.mode == "train":
        runner.train()
    else:
        metrics = runner.test()
        write_json(run_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
