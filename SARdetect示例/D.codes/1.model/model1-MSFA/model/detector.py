"""MMDetection model definition for the class-agnostic MSFA Faster R-CNN.

The official MSFA implementation is registered from the read-only reference
checkout under ``4.refer/SARDet_100K/MSFA``. This file owns the competition
model configuration; the reference repository is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple


@dataclass(frozen=True)
class DetectorBuildOptions:
    architecture: str = "faster_rcnn"
    num_queries: int | None = None
    input_size: Tuple[int, int] = (1024, 1024)
    feature_mode: str = "wavelet"
    num_classes: int = 1
    fpn_channels: int = 256
    anchor_scales: Sequence[int] = (8,)
    anchor_ratios: Sequence[float] = (0.5, 1.0, 2.0)
    score_threshold: float = 0.01
    nms_iou_threshold: float = 0.5
    max_detections: int = 300
    drop_path_rate: float = 0.4
    roi_bbox_loss_type: str = "L1Loss"
    roi_bbox_loss_weight: float = 1.0

    def validate(self) -> None:
        if self.architecture not in {"faster_rcnn", "detr", "yolox", "fcos"}:
            raise ValueError(
                "architecture must be one of: faster_rcnn, detr, yolox, fcos"
            )
        if self.feature_mode not in {"wavelet", "raw"}:
            raise ValueError("feature_mode must be 'wavelet' or 'raw'")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if self.num_queries is not None and self.num_queries < 1:
            raise ValueError("num_queries must be positive when specified")
        if min(self.input_size) <= 0:
            raise ValueError("input_size values must be positive")
        if not self.anchor_scales or not self.anchor_ratios:
            raise ValueError("anchor scales and ratios cannot be empty")
        if self.roi_bbox_loss_type not in {"L1Loss", "SmoothL1Loss", "IoULoss", "GIoULoss", "DIoULoss"}:
            raise ValueError("unsupported roi_bbox_loss_type")


def _data_preprocessor(pad_size_divisor: int = 32) -> Dict[str, Any]:
    return dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=pad_size_divisor,
    )


def _msfa_convnext_backbone(
    options: DetectorBuildOptions, out_indices: Sequence[int]
) -> Dict[str, Any]:
    height, width = options.input_size
    return dict(
        type="MSFA",
        use_sar=True,
        use_wavelet=options.feature_mode == "wavelet",
        input_size=(height, width),
        backbone=dict(
            type="ConvNeXt",
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=options.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=list(out_indices),
        ),
    )


def _build_detr_cfg(options: DetectorBuildOptions) -> Dict[str, Any]:
    num_queries = options.num_queries or options.max_detections
    return dict(
        type="DETR",
        num_queries=num_queries,
        data_preprocessor=_data_preprocessor(pad_size_divisor=1),
        backbone=_msfa_convnext_backbone(options, [3]),
        neck=dict(
            type="ChannelMapper",
            in_channels=[768],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=None,
            num_outs=1,
        ),
        encoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
        ),
        decoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
            return_intermediate=True,
        ),
        positional_encoding=dict(num_feats=128, normalize=True),
        bbox_head=dict(
            type="DETRHead",
            num_classes=options.num_classes,
            embed_dims=256,
            loss_cls=dict(
                type="CrossEntropyLoss",
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        ),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                match_costs=[
                    dict(type="ClassificationCost", weight=1.0),
                    dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                    dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ],
            )
        ),
        test_cfg=dict(max_per_img=min(options.max_detections, num_queries)),
    )


def _build_yolox_cfg(options: DetectorBuildOptions) -> Dict[str, Any]:
    return dict(
        type="YOLOX",
        data_preprocessor=_data_preprocessor(),
        backbone=_msfa_convnext_backbone(options, [1, 2, 3]),
        neck=dict(
            type="YOLOXPAFPN",
            in_channels=[192, 384, 768],
            out_channels=256,
            num_csp_blocks=1,
            use_depthwise=False,
            upsample_cfg=dict(scale_factor=2, mode="nearest"),
            norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
            act_cfg=dict(type="Swish"),
        ),
        bbox_head=dict(
            type="YOLOXHead",
            num_classes=options.num_classes,
            in_channels=256,
            feat_channels=256,
            stacked_convs=2,
            strides=(8, 16, 32),
            use_depthwise=False,
            norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
            act_cfg=dict(type="Swish"),
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="sum",
                loss_weight=1.0,
            ),
            loss_bbox=dict(
                type="IoULoss",
                mode="square",
                eps=1e-16,
                reduction="sum",
                loss_weight=5.0,
            ),
            loss_obj=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="sum",
                loss_weight=1.0,
            ),
            loss_l1=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
        ),
        train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
        test_cfg=dict(
            score_thr=options.score_threshold,
            nms=dict(type="nms", iou_threshold=options.nms_iou_threshold),
            max_per_img=options.max_detections,
        ),
    )


def _build_fcos_cfg(options: DetectorBuildOptions) -> Dict[str, Any]:
    return dict(
        type="FCOS",
        data_preprocessor=_data_preprocessor(),
        backbone=_msfa_convnext_backbone(options, [0, 1, 2, 3]),
        neck=dict(
            type="FPN",
            in_channels=[96, 192, 384, 768],
            out_channels=options.fpn_channels,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
            relu_before_extra_convs=True,
        ),
        bbox_head=dict(
            type="FCOSHead",
            num_classes=options.num_classes,
            in_channels=options.fpn_channels,
            stacked_convs=4,
            feat_channels=options.fpn_channels,
            strides=[8, 16, 32, 64, 128],
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="IoULoss", loss_weight=1.0),
            loss_centerness=dict(
                type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
            ),
        ),
        test_cfg=dict(
            nms_pre=3000,
            min_bbox_size=0,
            score_thr=options.score_threshold,
            nms=dict(type="nms", iou_threshold=options.nms_iou_threshold),
            max_per_img=options.max_detections,
        ),
    )


def build_detector_model_cfg(options: DetectorBuildOptions) -> Dict[str, Any]:
    """Return an MMDetection 3.x model config.

    ``wavelet`` reproduces the SAR+WST branch of the official MSFA model. It
    creates 81 scattering channels plus one SAR channel and therefore must pass
    the NPU operator preflight. ``raw`` retains the same detector/backbone but
    consumes the three-channel SAR image directly as the compatibility fallback.
    """

    options.validate()
    if options.architecture == "detr":
        return _build_detr_cfg(options)
    if options.architecture == "yolox":
        return _build_yolox_cfg(options)
    if options.architecture == "fcos":
        return _build_fcos_cfg(options)
    use_wavelet = options.feature_mode == "wavelet"
    height, width = options.input_size

    decoded_roi_bbox = options.roi_bbox_loss_type in {"IoULoss", "GIoULoss", "DIoULoss"}
    roi_bbox_loss: Dict[str, Any] = dict(
        type=options.roi_bbox_loss_type,
        loss_weight=options.roi_bbox_loss_weight,
    )
    if options.roi_bbox_loss_type == "SmoothL1Loss":
        roi_bbox_loss["beta"] = 1.0

    return dict(
        type="FasterRCNN",
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=False,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type="MSFA",
            use_sar=True,
            use_wavelet=use_wavelet,
            input_size=(height, width),
            backbone=dict(
                type="ConvNeXt",
                depths=[3, 3, 27, 3],
                dims=[96, 192, 384, 768],
                drop_path_rate=options.drop_path_rate,
                layer_scale_init_value=1e-6,
                out_indices=[0, 1, 2, 3],
            ),
        ),
        neck=dict(
            type="FPN",
            in_channels=[96, 192, 384, 768],
            out_channels=options.fpn_channels,
            num_outs=5,
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=options.fpn_channels,
            feat_channels=options.fpn_channels,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=list(options.anchor_scales),
                ratios=list(options.anchor_ratios),
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        roi_head=dict(
            type="StandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=options.fpn_channels,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=options.fpn_channels,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=options.num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=decoded_roi_bbox,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=roi_bbox_loss,
            ),
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=3000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=3000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=options.score_threshold,
                nms=dict(type="nms", iou_threshold=options.nms_iou_threshold),
                max_per_img=options.max_detections,
            ),
        ),
    )
