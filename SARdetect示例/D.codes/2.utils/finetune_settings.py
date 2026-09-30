"""Bound warmup for short warm-start runs; retain explicit optimizer learning rates."""
import math


def configure_finetune(raw, images):
    tr = raw['train']
    steps = max(1, math.ceil(images / tr['batch_size']) * tr['epochs'])
    tr['warmup_iters'] = min(50, max(1, steps // 10))
    tr['warmup_epochs'] = min(.25, tr['epochs'] / 10)
    tr['close_mosaic'] = min(tr.get('close_mosaic', 0), max(0, tr['epochs'] - 1))
    # YOLO's default bias warmup LR is 0.1, much too large for a small-LR fine-tune.
    tr['warmup_bias_lr'] = 0.
    tr['finetune'] = True
    return raw
