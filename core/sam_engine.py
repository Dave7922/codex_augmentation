from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SamResult:
    mask: np.ndarray
    score: float


class SamEngine:
    """SAM wrapper with OpenCV fallback when SAM is unavailable."""

    def __init__(self) -> None:
        self._predictor = None
        self._sam_loaded = False
        self._image_rgb: Optional[np.ndarray] = None

    @property
    def sam_loaded(self) -> bool:
        return self._sam_loaded

    def load_checkpoint(self, checkpoint_path: str, model_type: str = "vit_h") -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise RuntimeError(
                "segment-anything not installed. Please install it before loading SAM."
            ) from exc

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self._predictor = SamPredictor(sam)
        self._sam_loaded = True

    def set_image(self, bgr_image: np.ndarray) -> None:
        self._image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self._sam_loaded and self._predictor is not None:
            self._predictor.set_image(self._image_rgb)

    def predict(
        self,
        points: List[Tuple[int, int]],
        labels: List[int],
    ) -> List[SamResult]:
        if self._image_rgb is None:
            raise RuntimeError("No image set for segmentation")

        if self._sam_loaded and self._predictor is not None:
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            return [SamResult(mask=m.astype(np.uint8), score=float(s)) for m, s in zip(masks, scores)]

        # Fallback: GrabCut from prompts.
        # Use positive points bounding region as initialization.
        h, w = self._image_rgb.shape[:2]
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        for (x, y), lb in zip(points, labels):
            if 0 <= x < w and 0 <= y < h:
                gc_mask[y, x] = cv2.GC_FGD if lb == 1 else cv2.GC_BGD

        pos = [(x, y) for (x, y), lb in zip(points, labels) if lb == 1]
        if not pos:
            raise RuntimeError("Need at least one positive point")

        xs, ys = zip(*pos)
        pad = 24
        x1, x2 = max(0, min(xs) - pad), min(w - 1, max(xs) + pad)
        y1, y2 = max(0, min(ys) - pad), min(h - 1, max(ys) + pad)
        rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        bgr = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(
            bgr,
            gc_mask,
            rect,
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )

        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        return [SamResult(mask=out, score=0.5)]
