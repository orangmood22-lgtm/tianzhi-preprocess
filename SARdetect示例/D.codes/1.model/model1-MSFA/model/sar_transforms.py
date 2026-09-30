"""MMDetection transforms for sensor/style-robust SAR training."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SARDomainRandomization(BaseTransform):
    """Randomize monotonic intensity mapping and multiplicative speckle.

    The same transform is applied to every image channel so grayscale SAR
    structure is preserved.  Geometry and boxes are unchanged.  This is a
    source-agnostic domain-generalization transform: it requires no sensor label
    at inference time and therefore remains usable on an unseen competition
    source.
    """

    def __init__(
        self,
        prob: float = 0.8,
        gamma_range: Sequence[float] = (0.65, 1.55),
        contrast_range: Sequence[float] = (0.75, 1.30),
        gain_range: Sequence[float] = (0.85, 1.15),
        speckle_std_range: Sequence[float] = (0.0, 0.12),
        log_strength_range: Sequence[float] = (0.0, 8.0),
    ) -> None:
        self.prob = float(prob)
        self.gamma_range = tuple(float(value) for value in gamma_range)
        self.contrast_range = tuple(float(value) for value in contrast_range)
        self.gain_range = tuple(float(value) for value in gain_range)
        self.speckle_std_range = tuple(float(value) for value in speckle_std_range)
        self.log_strength_range = tuple(float(value) for value in log_strength_range)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        for name, values in (
            ("gamma_range", self.gamma_range),
            ("contrast_range", self.contrast_range),
            ("gain_range", self.gain_range),
            ("speckle_std_range", self.speckle_std_range),
            ("log_strength_range", self.log_strength_range),
        ):
            if len(values) != 2 or values[0] > values[1] or values[0] < 0:
                raise ValueError(f"invalid {name}: {values}")

    @staticmethod
    def _sample(values: tuple[float, float]) -> float:
        return float(np.random.uniform(values[0], values[1]))

    def transform(self, results: dict) -> dict:
        if np.random.random() >= self.prob:
            return results
        original = results["img"]
        image = original.astype(np.float32) / 255.0

        gamma = self._sample(self.gamma_range)
        contrast = self._sample(self.contrast_range)
        gain = self._sample(self.gain_range)
        speckle_std = self._sample(self.speckle_std_range)
        log_strength = self._sample(self.log_strength_range)

        image = np.power(np.clip(image, 0.0, 1.0), gamma)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast + mean
        image *= gain
        if log_strength > 1e-6:
            image = np.log1p(log_strength * np.clip(image, 0.0, None)) / np.log1p(
                log_strength
            )
        if speckle_std > 1e-6:
            noise_shape = image.shape[:2] + (1,) if image.ndim == 3 else image.shape
            speckle = np.random.normal(1.0, speckle_std, size=noise_shape).astype(
                np.float32
            )
            image *= speckle

        image = np.clip(image * 255.0, 0.0, 255.0)
        results["img"] = image.astype(original.dtype, copy=False)
        results["sar_domain_randomization"] = {
            "gamma": gamma,
            "contrast": contrast,
            "gain": gain,
            "speckle_std": speckle_std,
            "log_strength": log_strength,
        }
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"gamma_range={self.gamma_range}, contrast_range={self.contrast_range}, "
            f"gain_range={self.gain_range}, "
            f"speckle_std_range={self.speckle_std_range}, "
            f"log_strength_range={self.log_strength_range})"
        )
