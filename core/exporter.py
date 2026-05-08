from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .image_ops import mask_to_polygon


@dataclass
class ExportRecord:
    image: np.ndarray
    class_name: str
    file_stem: str
    masks: List[np.ndarray] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


def export_coco(output_dir: str, record: ExportRecord) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_path = out / f"{record.file_stem}.png"
    ann_path = out / f"{record.file_stem}.json"

    cv2.imwrite(str(image_path), record.image)

    h, w = record.image.shape[:2]
    annotations: List[Dict] = []
    for idx, (mask, bbox) in enumerate(zip(record.masks, record.bboxes), start=1):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        area = int(np.count_nonzero(mask))
        polygons = mask_to_polygon(mask)
        annotations.append(
            {
                "id": idx,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "area": area,
                "iscrowd": 0,
                "segmentation": polygons,
            }
        )

    coco: Dict[str, List[Dict]] = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": w,
                "height": h,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": record.class_name}],
    }

    ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return ann_path
