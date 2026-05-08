from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def refine_mask(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(m)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 1, -1)

    cleaned = cv2.GaussianBlur(cleaned.astype(np.float32), (0, 0), sigmaX=1.0)
    cleaned = (cleaned > 0.3).astype(np.uint8)
    return cleaned


def extract_patch(source_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Mask is empty")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    crop = source_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1].copy().astype(np.uint8)
    return crop, crop_mask, (x1, y1, x2, y2)


def transform_patch(
    patch_bgr: np.ndarray,
    patch_mask: np.ndarray,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    src = patch_bgr.copy()
    msk = patch_mask.copy().astype(np.uint8)

    if flip_x:
        src = cv2.flip(src, 1)
        msk = cv2.flip(msk, 1)
    if flip_y:
        src = cv2.flip(src, 0)
        msk = cv2.flip(msk, 0)

    h, w = src.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), rotation_deg, max(0.05, float(scale)))

    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    mat[0, 2] += (new_w / 2.0) - cx
    mat[1, 2] += (new_h / 2.0) - cy

    out_patch = cv2.warpAffine(
        src,
        mat,
        (max(1, new_w), max(1, new_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    out_mask = cv2.warpAffine(
        msk,
        mat,
        (max(1, new_w), max(1, new_h)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out_mask = (out_mask > 0).astype(np.uint8)
    return out_patch, out_mask


def match_patch_statistics(
    patch_bgr: np.ndarray,
    patch_mask: np.ndarray,
    target_roi_bgr: np.ndarray,
) -> np.ndarray:
    """Align patch color/brightness to target ROI using masked channel statistics."""
    out = patch_bgr.astype(np.float32).copy()
    m = patch_mask > 0
    if not np.any(m):
        return patch_bgr

    for c in range(3):
        pvals = out[..., c][m]
        tvals = target_roi_bgr[..., c].astype(np.float32).reshape(-1)
        if pvals.size < 4 or tvals.size < 4:
            continue

        p_mean, p_std = float(np.mean(pvals)), float(np.std(pvals))
        t_mean, t_std = float(np.mean(tvals)), float(np.std(tvals))
        if p_std < 1e-3:
            out[..., c][m] = np.clip(t_mean, 0, 255)
            continue

        scale = t_std / max(p_std, 1e-3)
        shifted = (out[..., c][m] - p_mean) * scale + t_mean
        out[..., c][m] = np.clip(shifted, 0, 255)

    return out.astype(np.uint8)


def place_patch(
    target_bgr: np.ndarray,
    patch_bgr: np.ndarray,
    patch_mask: np.ndarray,
    center_xy: Tuple[int, int],
    blend_mode: str = "poisson-mixed",
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    h, w = target_bgr.shape[:2]
    ph, pw = patch_bgr.shape[:2]
    cx, cy = center_xy

    x1 = int(cx - pw / 2)
    y1 = int(cy - ph / 2)
    x2 = x1 + pw
    y2 = y1 + ph

    tx1, ty1 = max(0, x1), max(0, y1)
    tx2, ty2 = min(w, x2), min(h, y2)
    if tx1 >= tx2 or ty1 >= ty2:
        raise RuntimeError("Patch placement is outside target")

    px1, py1 = tx1 - x1, ty1 - y1
    px2, py2 = px1 + (tx2 - tx1), py1 + (ty2 - ty1)

    patch_roi = patch_bgr[py1:py2, px1:px2]
    mask_roi = (patch_mask[py1:py2, px1:px2] > 0).astype(np.uint8) * 255

    result = target_bgr.copy()

    if blend_mode == "alpha":
        region = result[ty1:ty2, tx1:tx2].astype(np.float32)
        src = patch_roi.astype(np.float32)
        m = (mask_roi.astype(np.float32) / 255.0)[..., None] * alpha
        blended = src * m + region * (1.0 - m)
        result[ty1:ty2, tx1:tx2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        canvas = np.zeros_like(target_bgr)
        mask_canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[ty1:ty2, tx1:tx2] = patch_roi
        mask_canvas[ty1:ty2, tx1:tx2] = mask_roi
        clone_flag = cv2.MIXED_CLONE if blend_mode == "poisson-mixed" else cv2.NORMAL_CLONE
        result = cv2.seamlessClone(canvas, target_bgr, mask_canvas, (cx, cy), clone_flag)

    placed_mask = np.zeros((h, w), dtype=np.uint8)
    placed_mask[ty1:ty2, tx1:tx2] = (mask_roi > 0).astype(np.uint8)

    ys, xs = np.where(placed_mask > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return result, placed_mask, bbox


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = c.reshape(-1, 2).astype(float).flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons
