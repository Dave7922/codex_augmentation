from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import random
from tkinter import BOTH, LEFT, RIGHT, TOP, BooleanVar, Button, Canvas, DoubleVar, Frame, IntVar, Label, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.exporter import ExportRecord, export_coco
from core.image_ops import extract_patch, match_patch_statistics, place_patch, refine_mask, transform_patch
from core.pctnet_adapter import PCTNetAdapter
from core.sam_engine import SamEngine


class AugApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("PCB Defect Augmentation - SAM + Advanced Blending")
        self.root.geometry("1500x860")

        self.sam = SamEngine()
        self.pctnet = PCTNetAdapter()

        self.source_bgr = None
        self.source_mask = None
        self.target_bgr = None
        self.result_bgr = None
        self.patch_bgr = None
        self.patch_mask = None
        self.placed_mask = None
        self.last_bbox = None
        self.all_masks = []
        self.all_bboxes = []
        self.src_offset_x = 0
        self.src_offset_y = 0
        self.tgt_offset_x = 0
        self.tgt_offset_y = 0
        self._src_pan_anchor = None
        self._tgt_pan_anchor = None
        self._src_panning = False
        self._tgt_panning = False

        self.points = []
        self.labels = []
        self.paste_center = None
        self.last_target_dir = None

        self.class_name = StringVar(value="defect")
        self.blend_mode = StringVar(value="poisson-mixed")
        self.scale_var = DoubleVar(value=1.0)
        self.rotation_var = DoubleVar(value=0.0)
        self.flip_x_var = BooleanVar(value=False)
        self.flip_y_var = BooleanVar(value=False)
        self.color_match_var = BooleanVar(value=True)
        self.harmonize_var = StringVar(value="off")
        self.batch_count_var = IntVar(value=20)
        self.src_zoom_var = DoubleVar(value=1.0)
        self.tgt_zoom_var = DoubleVar(value=1.0)
        self.status = StringVar(value="Ready")

        self._build_ui()
        self._auto_load_default_sam()

    @staticmethod
    def _image_filetypes():
        # macOS Tk can crash if a single filetype string contains semicolon-separated patterns.
        return [("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"))]

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side=TOP, fill="x")

        source_ops = ttk.LabelFrame(toolbar, text="1) Source & Mask", padding=(8, 6))
        source_ops.pack(side=LEFT, padx=(0, 8))
        Button(source_ops, text="Load Source", command=self.load_source).pack(side=LEFT, padx=3)
        Button(source_ops, text="Load SAM Checkpoint", command=self.load_sam_checkpoint).pack(side=LEFT, padx=3)
        Button(source_ops, text="Reset Source", command=self.reset_source_state).pack(side=LEFT, padx=3)
        Button(source_ops, text="Segment", command=self.run_segment).pack(side=LEFT, padx=3)
        Button(source_ops, text="Undo Point", command=self.undo_last_point).pack(side=LEFT, padx=3)
        Button(source_ops, text="Clear Points", command=self.clear_points).pack(side=LEFT, padx=3)
        Button(source_ops, text="Refine Mask", command=self.run_refine).pack(side=LEFT, padx=3)
        Button(source_ops, text="Extract Patch", command=self.run_extract_patch).pack(side=LEFT, padx=3)

        synth_ops = ttk.LabelFrame(toolbar, text="2) Synthesize & Export", padding=(8, 6))
        synth_ops.pack(side=LEFT, padx=(0, 8))
        Button(synth_ops, text="Load Target", command=self.load_target).pack(side=LEFT, padx=3)
        Button(synth_ops, text="Synthesize", command=self.run_synthesize).pack(side=LEFT, padx=3)
        Button(synth_ops, text="Batch Synthesize", command=self.run_batch_synthesize).pack(side=LEFT, padx=3)
        Button(synth_ops, text="Reset Result", command=self.reset_result).pack(side=LEFT, padx=3)
        Button(synth_ops, text="Export COCO", command=self.run_export).pack(side=LEFT, padx=3)

        settings = ttk.LabelFrame(self.root, text="Settings", padding=(10, 8))
        settings.pack(side=TOP, fill="x", padx=10, pady=(0, 8))

        ttk.Label(settings, text="Blend").pack(side=LEFT, padx=(0, 4))
        ttk.Combobox(settings, textvariable=self.blend_mode, values=["alpha", "poisson-normal", "poisson-mixed"], width=16, state="readonly").pack(side=LEFT, padx=(0, 10))
        ttk.Label(settings, text="Class").pack(side=LEFT, padx=(0, 4))
        ttk.Entry(settings, textvariable=self.class_name, width=14).pack(side=LEFT, padx=(0, 14))
        ttk.Label(settings, text="Scale").pack(side=LEFT, padx=(0, 4))
        ttk.Scale(settings, variable=self.scale_var, from_=0.4, to=2.5, orient="horizontal", length=120).pack(side=LEFT, padx=(0, 10))
        ttk.Label(settings, text="Rotation").pack(side=LEFT, padx=(0, 4))
        ttk.Scale(settings, variable=self.rotation_var, from_=-180, to=180, orient="horizontal", length=140).pack(side=LEFT, padx=(0, 10))
        ttk.Checkbutton(settings, text="Flip X", variable=self.flip_x_var).pack(side=LEFT, padx=3)
        ttk.Checkbutton(settings, text="Flip Y", variable=self.flip_y_var).pack(side=LEFT, padx=3)
        ttk.Checkbutton(settings, text="Color Match", variable=self.color_match_var).pack(side=LEFT, padx=(6, 8))
        ttk.Label(settings, text="Harmonize").pack(side=LEFT, padx=(0, 4))
        ttk.Combobox(settings, textvariable=self.harmonize_var, values=["off", "pctnet"], width=8, state="readonly").pack(side=LEFT, padx=(0, 10))
        ttk.Label(settings, text="Batch N").pack(side=LEFT, padx=(0, 4))
        ttk.Spinbox(settings, from_=1, to=1000, textvariable=self.batch_count_var, width=6).pack(side=LEFT, padx=(0, 10))
        ttk.Label(settings, text="Src Zoom").pack(side=LEFT, padx=(0, 4))
        ttk.Scale(settings, variable=self.src_zoom_var, from_=0.5, to=4.0, orient="horizontal", length=90, command=lambda _: self._draw_source()).pack(side=LEFT, padx=(0, 8))
        ttk.Label(settings, text="Tgt Zoom").pack(side=LEFT, padx=(0, 4))
        ttk.Scale(settings, variable=self.tgt_zoom_var, from_=0.5, to=4.0, orient="horizontal", length=90, command=lambda _: self._draw_target()).pack(side=LEFT, padx=(0, 2))

        middle = ttk.Frame(self.root)
        middle.pack(fill=BOTH, expand=True)

        left_panel = ttk.LabelFrame(middle, text="Source View", padding=(6, 6))
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=8, pady=6)
        Label(left_panel, text="Left-click=FG | Shift+Left-click=BG | Right-click=remove nearest | Ctrl+Drag=pan").pack(anchor="w", pady=(0, 4))
        self.source_canvas = Canvas(left_panel, bg="#202020", width=700, height=740)
        self.source_canvas.pack(fill=BOTH, expand=True)
        self.source_canvas.bind("<Button-1>", self.on_source_click_fg)
        self.source_canvas.bind("<Shift-Button-1>", self.on_source_click_bg)
        self.source_canvas.bind("<Button-3>", self.on_source_right_click)
        self.source_canvas.bind("<Control-ButtonPress-1>", self.on_source_pan_start)
        self.source_canvas.bind("<Control-B1-Motion>", self.on_source_pan_move)
        self.source_canvas.bind("<Control-ButtonRelease-1>", self.on_source_pan_end)

        right_panel = ttk.LabelFrame(middle, text="Target / Result View", padding=(6, 6))
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=8, pady=6)
        Label(right_panel, text="Left-click=set paste center | Ctrl+Drag=pan").pack(anchor="w", pady=(0, 4))
        self.target_canvas = Canvas(right_panel, bg="#202020", width=700, height=740)
        self.target_canvas.pack(fill=BOTH, expand=True)
        self.target_canvas.bind("<Button-1>", self.on_target_click)
        self.target_canvas.bind("<Control-ButtonPress-1>", self.on_target_pan_start)
        self.target_canvas.bind("<Control-B1-Motion>", self.on_target_pan_move)
        self.target_canvas.bind("<Control-ButtonRelease-1>", self.on_target_pan_end)

        bar = ttk.Label(self.root, textvariable=self.status, anchor="w", relief="sunken", padding=(8, 4))
        bar.pack(fill="x", padx=10, pady=(0, 8))

    def _auto_load_default_sam(self) -> None:
        # Prefer the smaller ViT-B checkpoint for faster startup/inference.
        candidates = [
            Path("checkpoints/sam_vit_b_01ec64.pth"),
            Path("codex_augmentation/checkpoints/sam_vit_b_01ec64.pth"),
        ]
        for ckpt in candidates:
            if ckpt.exists():
                try:
                    self.sam.load_checkpoint(str(ckpt), model_type="vit_b")
                    self.status.set(f"Auto-loaded SAM: {ckpt}")
                except Exception as exc:
                    self.status.set(f"SAM auto-load failed: {exc}")
                return

    @staticmethod
    def _fit_for_canvas(img_bgr: np.ndarray, cw: int, ch: int, zoom: float = 1.0):
        h, w = img_bgr.shape[:2]
        scale = min(cw / max(1, w), ch / max(1, h)) * max(0.1, zoom)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _clamp_offset(off_x: float, off_y: float, img_w: int, img_h: int, cw: int, ch: int):
        min_x = min(0, cw - img_w)
        min_y = min(0, ch - img_h)
        x = int(min(0, max(min_x, off_x)))
        y = int(min(0, max(min_y, off_y)))
        return x, y

    def _draw_source(self) -> None:
        if self.source_bgr is None:
            return
        self.source_canvas.delete("all")
        cw = max(1, self.source_canvas.winfo_width())
        ch = max(1, self.source_canvas.winfo_height())
        disp, self.src_scale = self._fit_for_canvas(self.source_bgr, cw, ch, float(self.src_zoom_var.get()))
        self.src_offset_x, self.src_offset_y = self._clamp_offset(
            self.src_offset_x, self.src_offset_y, disp.shape[1], disp.shape[0], cw, ch
        )

        overlay = disp.copy()
        if self.source_mask is not None:
            m = cv2.resize(self.source_mask.astype(np.uint8), (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0] = (0, 0, 255)
            disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.src_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.source_canvas.create_image(self.src_offset_x, self.src_offset_y, image=self.src_photo, anchor="nw")

        for (x, y), lb in zip(self.points, self.labels):
            dx, dy = int(x * self.src_scale), int(y * self.src_scale)
            dx += self.src_offset_x
            dy += self.src_offset_y
            color = "#00ff00" if lb == 1 else "#ff4040"
            self.source_canvas.create_oval(dx - 4, dy - 4, dx + 4, dy + 4, fill=color, outline="")

    def _draw_target(self) -> None:
        img = self.result_bgr if self.result_bgr is not None else self.target_bgr
        if img is None:
            return
        self.target_canvas.delete("all")
        cw = max(1, self.target_canvas.winfo_width())
        ch = max(1, self.target_canvas.winfo_height())
        disp, self.tgt_scale = self._fit_for_canvas(img, cw, ch, float(self.tgt_zoom_var.get()))
        self.tgt_offset_x, self.tgt_offset_y = self._clamp_offset(
            self.tgt_offset_x, self.tgt_offset_y, disp.shape[1], disp.shape[0], cw, ch
        )
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.tgt_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.target_canvas.create_image(self.tgt_offset_x, self.tgt_offset_y, image=self.tgt_photo, anchor="nw")

        if self.paste_center is not None:
            x, y = self.paste_center
            dx, dy = int(x * self.tgt_scale), int(y * self.tgt_scale)
            dx += self.tgt_offset_x
            dy += self.tgt_offset_y
            self.target_canvas.create_line(dx - 8, dy, dx + 8, dy, fill="#00ffff", width=2)
            self.target_canvas.create_line(dx, dy - 8, dx, dy + 8, fill="#00ffff", width=2)

    def load_source(self) -> None:
        path = filedialog.askopenfilename(filetypes=self._image_filetypes())
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read source image")
            return

        self.source_bgr = img
        self.sam.set_image(img)
        self.source_mask = None
        self.points, self.labels = [], []
        self.src_offset_x = 0
        self.src_offset_y = 0
        self.status.set(f"Loaded source: {Path(path).name}")
        self._draw_source()

    def reset_source_state(self) -> None:
        if self.source_bgr is None:
            self.status.set("No source image loaded")
            return
        self.source_mask = None
        self.points = []
        self.labels = []
        self.patch_bgr = None
        self.patch_mask = None
        self.src_offset_x = 0
        self.src_offset_y = 0
        self.status.set("Source reset: cleared prompts, mask, and extracted patch")
        self._draw_source()

    def load_sam_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("SAM checkpoint", "*.pth")])
        if not path:
            return
        try:
            lower = Path(path).name.lower()
            model_type = "vit_h"
            if "vit_b" in lower:
                model_type = "vit_b"
            elif "vit_l" in lower:
                model_type = "vit_l"
            self.sam.load_checkpoint(path, model_type=model_type)
            if self.source_bgr is not None:
                self.sam.set_image(self.source_bgr)
            self.status.set(f"SAM checkpoint loaded ({model_type})")
        except Exception as exc:
            messagebox.showerror("SAM error", str(exc))

    def run_segment(self) -> None:
        if self.source_bgr is None:
            messagebox.showwarning("Warning", "Load source image first")
            return
        if not self.points:
            messagebox.showwarning("Warning", "Please click at least one foreground point")
            return
        fg_count = sum(1 for lb in self.labels if lb == 1)
        if fg_count == 0:
            messagebox.showwarning("Warning", "Need at least one foreground point (normal left-click)")
            return
        try:
            self.status.set(f"Running segmentation... points={len(self.points)}, fg={fg_count}")
            self.root.update_idletasks()
            res = self.sam.predict(self.points, self.labels)
            best = max(res, key=lambda x: x.score)
            self.source_mask = best.mask.astype(np.uint8)
            self.status.set(f"Segmentation done. score={best.score:.3f} ({'SAM' if self.sam.sam_loaded else 'fallback'})")
            self._draw_source()
        except Exception as exc:
            self.status.set(f"Segmentation failed: {exc}")
            messagebox.showerror("Segmentation error", str(exc))

    def run_refine(self) -> None:
        if self.source_mask is None:
            messagebox.showwarning("Warning", "Please segment first")
            return
        self.source_mask = refine_mask(self.source_mask)
        self.status.set("Mask refined with contour + morphology")
        self._draw_source()

    def run_extract_patch(self) -> None:
        if self.source_bgr is None or self.source_mask is None:
            messagebox.showwarning("Warning", "Need source and mask")
            return
        try:
            self.patch_bgr, self.patch_mask, _ = extract_patch(self.source_bgr, self.source_mask)
            self.status.set("Patch extracted")
        except Exception as exc:
            messagebox.showerror("Patch error", str(exc))

    def load_target(self) -> None:
        path = filedialog.askopenfilename(filetypes=self._image_filetypes())
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read target image")
            return
        self.target_bgr = img
        self.last_target_dir = str(Path(path).parent)
        self.result_bgr = None
        self.paste_center = None
        self.tgt_offset_x = 0
        self.tgt_offset_y = 0
        self.all_masks = []
        self.all_bboxes = []
        self.status.set(f"Loaded target: {Path(path).name}")
        self._draw_target()

    def run_synthesize(self) -> None:
        if self.target_bgr is None:
            messagebox.showwarning("Warning", "Load target first")
            return
        if self.patch_bgr is None or self.patch_mask is None:
            messagebox.showwarning("Warning", "Extract patch first")
            return
        if self.paste_center is None:
            messagebox.showwarning("Warning", "Click target to set paste center")
            return

        try:
            mode = self.blend_mode.get()
            real_mode = "poisson-mixed"
            if mode == "alpha":
                real_mode = "alpha"
            elif mode == "poisson-normal":
                real_mode = "poisson-normal"

            t_patch, t_mask = transform_patch(
                self.patch_bgr,
                self.patch_mask,
                scale=float(self.scale_var.get()),
                rotation_deg=float(self.rotation_var.get()),
                flip_x=bool(self.flip_x_var.get()),
                flip_y=bool(self.flip_y_var.get()),
            )
            t_patch = self._apply_color_match_if_needed(self.target_bgr, t_patch, t_mask, self.paste_center)
            base = self.result_bgr if self.result_bgr is not None else self.target_bgr
            self.result_bgr, self.placed_mask, self.last_bbox = place_patch(
                base,
                t_patch,
                t_mask,
                self.paste_center,
                blend_mode=real_mode,
            )
            if self.harmonize_var.get() == "pctnet":
                self.result_bgr = self.pctnet.harmonize(self.result_bgr, self.placed_mask)
                if self.pctnet._init_error is not None:
                    self.status.set(f"Synthesis done: {real_mode} | pctnet fallback ({self.pctnet._init_error})")
                    self._draw_target()
                    return
            self.all_masks.append(self.placed_mask.copy())
            self.all_bboxes.append(self.last_bbox)
            self.status.set(f"Synthesis done: {real_mode}")
            self._draw_target()
        except Exception as exc:
            messagebox.showerror("Synthesis error", str(exc))

    def run_export(self) -> None:
        if self.result_bgr is None or self.placed_mask is None or self.last_bbox is None:
            messagebox.showwarning("Warning", "Synthesize before export")
            return

        initial_dir = self.last_target_dir if self.last_target_dir else os.getcwd()
        out_dir = filedialog.askdirectory(
            initialdir=initial_dir,
            title="Select export directory (defaults to target image folder)",
        )
        if not out_dir:
            return

        file_stem = f"synth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = ExportRecord(
            image=self.result_bgr,
            class_name=self.class_name.get().strip() or "defect",
            file_stem=file_stem,
            masks=self.all_masks if self.all_masks else [self.placed_mask],
            bboxes=self.all_bboxes if self.all_bboxes else [self.last_bbox],
        )
        ann_path = export_coco(out_dir, record)
        self.status.set(f"Exported: {ann_path}")
        messagebox.showinfo("Export done", f"Saved to:\n{out_dir}")

    def reset_result(self) -> None:
        self.result_bgr = None
        self.placed_mask = None
        self.last_bbox = None
        self.all_masks = []
        self.all_bboxes = []
        self.status.set("Result reset; synth will start from original target")
        self._draw_target()

    def run_batch_synthesize(self) -> None:
        if self.patch_bgr is None or self.patch_mask is None:
            messagebox.showwarning("Warning", "Extract patch first")
            return

        src_dir = filedialog.askdirectory(title="Select target image directory")
        if not src_dir:
            return
        out_dir = filedialog.askdirectory(title="Select output directory")
        if not out_dir:
            return

        img_paths = sorted(
            p for p in Path(src_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if not img_paths:
            messagebox.showwarning("Warning", "No target images found in directory")
            return

        total = int(max(1, self.batch_count_var.get()))
        done = 0
        for i in range(total):
            t_path = random.choice(img_paths)
            target = cv2.imread(str(t_path))
            if target is None:
                continue
            try:
                synth, msk, bbox = self._synthesize_one_random(target)
            except Exception:
                continue
            stem = f"{t_path.stem}_synth_{i:04d}"
            record = ExportRecord(
                image=synth,
                class_name=self.class_name.get().strip() or "defect",
                file_stem=stem,
                masks=[msk],
                bboxes=[bbox],
            )
            export_coco(out_dir, record)
            done += 1

        self.status.set(f"Batch done: {done}/{total} exported to {out_dir}")
        messagebox.showinfo("Batch done", f"Exported {done}/{total} samples")

    def _apply_color_match_if_needed(
        self,
        target_bgr: np.ndarray,
        patch_bgr: np.ndarray,
        patch_mask: np.ndarray,
        center_xy: tuple[int, int],
    ) -> np.ndarray:
        if not self.color_match_var.get():
            return patch_bgr

        h, w = target_bgr.shape[:2]
        ph, pw = patch_bgr.shape[:2]
        cx, cy = center_xy
        x1 = max(0, int(cx - pw / 2))
        y1 = max(0, int(cy - ph / 2))
        x2 = min(w, x1 + pw)
        y2 = min(h, y1 + ph)
        if x2 <= x1 or y2 <= y1:
            return patch_bgr

        patch_crop = patch_bgr[: y2 - y1, : x2 - x1]
        mask_crop = patch_mask[: y2 - y1, : x2 - x1]
        target_roi = target_bgr[y1:y2, x1:x2]
        matched = match_patch_statistics(patch_crop, mask_crop, target_roi)

        out = patch_bgr.copy()
        out[: y2 - y1, : x2 - x1] = matched
        return out

    def _synthesize_one_random(self, target_bgr: np.ndarray):
        h, w = target_bgr.shape[:2]
        mode = self.blend_mode.get()
        real_mode = "poisson-mixed"
        if mode == "alpha":
            real_mode = "alpha"
        elif mode == "poisson-normal":
            real_mode = "poisson-normal"

        scale = float(self.scale_var.get()) * random.uniform(0.75, 1.25)
        rotation = float(self.rotation_var.get()) + random.uniform(-35, 35)
        flip_x = bool(self.flip_x_var.get()) if random.random() < 0.5 else not bool(self.flip_x_var.get())
        flip_y = bool(self.flip_y_var.get()) if random.random() < 0.2 else bool(self.flip_y_var.get())
        t_patch, t_mask = transform_patch(
            self.patch_bgr,
            self.patch_mask,
            scale=scale,
            rotation_deg=rotation,
            flip_x=flip_x,
            flip_y=flip_y,
        )

        ph, pw = t_patch.shape[:2]
        if pw >= w or ph >= h:
            shrink = min((w - 2) / max(1, pw), (h - 2) / max(1, ph), 0.95)
            t_patch, t_mask = transform_patch(t_patch, t_mask, scale=max(0.1, shrink))
            ph, pw = t_patch.shape[:2]

        min_cx, max_cx = pw // 2, w - (pw - pw // 2)
        min_cy, max_cy = ph // 2, h - (ph - ph // 2)
        if min_cx >= max_cx or min_cy >= max_cy:
            raise RuntimeError("Transformed patch cannot fit target image")
        cx = random.randint(min_cx, max_cx)
        cy = random.randint(min_cy, max_cy)
        t_patch = self._apply_color_match_if_needed(target_bgr, t_patch, t_mask, (cx, cy))
        out_img, out_mask, out_bbox = place_patch(target_bgr, t_patch, t_mask, (cx, cy), blend_mode=real_mode)
        if self.harmonize_var.get() == "pctnet":
            out_img = self.pctnet.harmonize(out_img, out_mask)
        return out_img, out_mask, out_bbox

    def _add_prompt_point(self, event, label: int) -> None:
        if self.source_bgr is None:
            return
        if self._src_panning:
            self._src_panning = False
            return
        x = int((event.x - self.src_offset_x) / max(self.src_scale, 1e-6))
        y = int((event.y - self.src_offset_y) / max(self.src_scale, 1e-6))
        h, w = self.source_bgr.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        self.points.append((x, y))
        self.labels.append(label)
        fg_n = sum(1 for lb in self.labels if lb == 1)
        self.status.set(f"Point added ({'BG' if label == 0 else 'FG'}): ({x}, {y}) | FG={fg_n}, Total={len(self.points)}")
        self._draw_source()

    def on_source_click_fg(self, event) -> None:
        self._add_prompt_point(event, label=1)

    def on_source_click_bg(self, event) -> None:
        self._add_prompt_point(event, label=0)

    def on_source_right_click(self, event) -> None:
        if not self.points:
            return
        x = int((event.x - self.src_offset_x) / max(self.src_scale, 1e-6))
        y = int((event.y - self.src_offset_y) / max(self.src_scale, 1e-6))
        # Remove nearest prompt point for quick correction.
        d2 = [((px - x) ** 2 + (py - y) ** 2) for (px, py) in self.points]
        idx = int(np.argmin(np.array(d2)))
        px, py = self.points.pop(idx)
        lb = self.labels.pop(idx)
        self.status.set(f"Point removed ({'BG' if lb == 0 else 'FG'}): ({px}, {py})")
        self._draw_source()

    def undo_last_point(self) -> None:
        if not self.points:
            self.status.set("No prompt points to undo")
            return
        px, py = self.points.pop()
        lb = self.labels.pop()
        self.status.set(f"Undo last point ({'BG' if lb == 0 else 'FG'}): ({px}, {py})")
        self._draw_source()

    def clear_points(self) -> None:
        self.points = []
        self.labels = []
        self.status.set("Cleared all prompt points")
        self._draw_source()

    def on_target_click(self, event) -> None:
        if self.target_bgr is None and self.result_bgr is None:
            return
        if self._tgt_panning:
            self._tgt_panning = False
            return
        img = self.result_bgr if self.result_bgr is not None else self.target_bgr
        x = int((event.x - self.tgt_offset_x) / max(self.tgt_scale, 1e-6))
        y = int((event.y - self.tgt_offset_y) / max(self.tgt_scale, 1e-6))
        h, w = img.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        self.paste_center = (x, y)
        self.status.set(f"Paste center set: ({x}, {y})")
        self._draw_target()

    def on_source_pan_start(self, event) -> None:
        self._src_panning = True
        self._src_pan_anchor = (event.x, event.y)
        return "break"

    def on_source_pan_move(self, event) -> None:
        if self._src_pan_anchor is None or self.source_bgr is None:
            return
        dx = event.x - self._src_pan_anchor[0]
        dy = event.y - self._src_pan_anchor[1]
        self._src_pan_anchor = (event.x, event.y)
        self.src_offset_x += dx
        self.src_offset_y += dy
        self._draw_source()

    def on_target_pan_start(self, event) -> None:
        self._tgt_panning = True
        self._tgt_pan_anchor = (event.x, event.y)
        return "break"

    def on_target_pan_move(self, event) -> None:
        if self._tgt_pan_anchor is None or (self.target_bgr is None and self.result_bgr is None):
            return
        dx = event.x - self._tgt_pan_anchor[0]
        dy = event.y - self._tgt_pan_anchor[1]
        self._tgt_pan_anchor = (event.x, event.y)
        self.tgt_offset_x += dx
        self.tgt_offset_y += dy
        self._draw_target()

    def on_source_pan_end(self, _event) -> None:
        self._src_pan_anchor = None

    def on_target_pan_end(self, _event) -> None:
        self._tgt_pan_anchor = None


def main() -> None:
    # Prevent overly blurry rendering on some HiDPI displays.
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
    root = Tk()
    app = AugApp(root)
    app._draw_source()
    app._draw_target()
    root.mainloop()


if __name__ == "__main__":
    main()
