"""Documented evaluation correctness adjustment for Ascend CPU-fallback NMS."""
import functools


def complete_nms_batches():
    from ultralytics.utils import nms
    original = nms.non_max_suppression
    if getattr(original, '_complete_batches', False):
        return

    @functools.wraps(original)
    def complete(*args, **kwargs):
        # NPU cold compilation/CPU fallback can exceed Ultralytics' wall-clock
        # guard. The guard silently drops remaining images in a batch. Disable
        # only that guard; keep thresholds, sorting and suppression unchanged.
        kwargs['max_time_img'] = float('inf')
        return original(*args, **kwargs)

    complete._complete_batches = True
    nms.non_max_suppression = complete
