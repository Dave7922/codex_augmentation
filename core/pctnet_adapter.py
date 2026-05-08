from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


class PCTNetAdapter:
    """
    PCT-Net harmonization adapter.

    Uses official repository code if present; otherwise falls back to a seam
    smoothing pass that is still better than no post-processing.
    """

    def __init__(self, repo_dir: str = "third_party/PCT-Net-Image-Harmonization-main") -> None:
        self.repo_dir = Path(repo_dir)
        if not self.repo_dir.exists():
            alt = Path("codex_augmentation") / repo_dir
            if alt.exists():
                self.repo_dir = alt
        if self.repo_dir.exists():
            self.repo_dir = self.repo_dir.resolve()
        self.available = self.repo_dir.exists()
        self._predictor = None
        self._init_error = None

    def _init_predictor(self) -> None:
        if self._predictor is not None or self._init_error is not None:
            return
        if not self.available:
            self._init_error = "PCT-Net repository directory not found"
            return

        repo_path = str(self.repo_dir.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            old_cwd = os.getcwd()
            os.chdir(repo_path)
            from iharm.inference.predictor import Predictor
            from iharm.inference.utils import load_model
            from iharm.mconfigs import ALL_MCONFIGS

            weights = self.repo_dir.resolve() / "pretrained_models" / "PCTNet_ViT.pth"
            if not weights.exists():
                self._init_error = f"Missing weight file: {weights}"
                return

            model_type = "ViT_pct"
            model = load_model(model_type, str(weights), verbose=False)
            use_attn = ALL_MCONFIGS[model_type]["params"]["use_attn"]
            norm = ALL_MCONFIGS[model_type]["params"]["input_normalization"]
            device = torch.device("cpu")
            self._predictor = Predictor(
                model,
                device,
                use_attn=use_attn,
                mean=norm["mean"],
                std=norm["std"],
            )
        except Exception as exc:
            self._init_error = str(exc)
        finally:
            # Restore caller process cwd.
            os.chdir(old_cwd)

    def harmonize(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._init_predictor()
        if self._predictor is not None:
            try:
                comp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                m = (mask > 0).astype(np.float32)
                comp_lr = cv2.resize(comp, (256, 256))
                mask_lr = cv2.resize(m, (256, 256))
                _, pred_img = self._predictor.predict(comp_lr, comp, mask_lr, m)
                if pred_img is None:
                    return self._seam_fallback(image_bgr, mask)
                arr = np.asarray(pred_img)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    return self._seam_fallback(image_bgr, mask)
                if arr.dtype != np.uint8:
                    arr = np.nan_to_num(arr)
                    # Many harmonization models output float in [0,1].
                    if arr.max() <= 1.5:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                # If model inference fails, use seam-only fallback.
                pass
        return self._seam_fallback(image_bgr, mask)

    @staticmethod
    def _seam_fallback(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)
        if np.count_nonzero(m) == 0:
            return image_bgr
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        band = (cv2.dilate(m, kernel, iterations=1) - cv2.erode(m, kernel, iterations=1)) > 0
        blur = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=20, sigmaSpace=15)
        out = image_bgr.copy()
        out[band] = blur[band]
        return out
