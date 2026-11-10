#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage5_ai_inpaint_integration.py
===============================================================================
V8 / STAGE 5 — AI INPAINT MATERIAL / TEXTURE / LIGHT INTEGRATION

AMAÇ
----
Stage 4'te uzman tarafından kesinleşmiş geometriyi BOZMADAN, yalnız hasarlı
bölge içinde, yapının kendi gözlenen dilinden öğrenerek AI inpaint ile
malzeme / doku / ışık bütünleştirmesi yapmak.

BU AŞAMA NE YAPAR?
------------------
- Stage 4 planını ve eleman maskelerini okur.
- Her onaylı eksik elemanı AYRI AYRI crop bazlı işler.
- Donör elemanı birebir yapıştırmaz; geometri rehberli bir guide-composite
  üretir ve onu ControlNet için yönlendirici olarak kullanır.
- ControlNet lineart + depth ile geometriyi sabitler.
- Inpaint yalnız damage-limited mask içinde çalışır.
- Maske dışı pikseller ana görselde manuel kompozitle KESİN korunur.

BU AŞAMA NE YAPMAZ?
-------------------
- Yeni hipotez üretmez.
- Stage 4 geometrisini değiştirmez.
- Uzman onayını geçersiz kılmaz.
- Rastgele başka tarihi yapı motifi eklemez.
- Tüm görüntüyü yeniden render etmez.

ÇIKTI FİKİRİ
------------
Stage 5'in hedefi nihai poster/görsel değil; Stage 4'ün geometrisini yapının
kendi diliyle AI destekli olarak yüzeyde bütünleştirmektir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# -----------------------------------------------------------------------------
# ENV / warning cleanup
# -----------------------------------------------------------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from controlnet_aux.lineart import LineartDetector
from transformers import AutoImageProcessor, DPTForDepthEstimation

try:
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()
except Exception:
    pass
try:
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
except Exception:
    pass


SCRIPT_NAME = "ImageToRender2Restoration_v8_stage5_ai_inpaint_integration.py"
STAGE_NAME = "v8_stage5_ai_inpaint_integration"
STAGE_TITLE = "Stage 5 — AI Inpaint Material / Texture / Light Integration"


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    # Intent: original_state | conservation
    intent: str = "original_state"

    # Model IDs
    base_model_id: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"
    controlnet_depth_id: str = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet_lineart_id: str = "lllyasviel/control_v11p_sd15_lineart"
    depth_model_id: str = "Intel/dpt-hybrid-midas"
    lineart_model_id: str = "lllyasviel/Annotators"

    # Crop / geometry
    crop_context_px: int = 36
    max_crop_side: int = 512
    min_crop_side: int = 256
    mask_threshold: int = 128
    fill_blur_ksize: int = 11

    # Generation
    num_inference_steps: int = 28
    guidance_scale: float = 6.0
    strength: float = 0.50
    controlnet_conditioning_scale_depth: float = 0.50
    controlnet_conditioning_scale_lineart: float = 0.65
    control_guidance_end_depth: float = 0.75
    control_guidance_end_lineart: float = 0.90
    base_seed: int = 20260804

    # Resource / memory
    prefer_xformers: bool = True
    use_cpu_offload: bool = True
    attention_slicing: bool = True
    vae_slicing: bool = True
    vae_tiling: bool = True

    # Save debug assets
    save_debug_assets: bool = True


REQUIRED_PARENT_FILES = [
    "input.png",
    "completion_mask.png",
    "source_expert_validated_inventory.json",
    "stage4_geometric_reconstruction_plan.json",
    "stage4_donor_selection.json",
    "stage4_geometry_damage_limited_mask.png",
    "reconstruction_element_masks",
]

OPTIONAL_PARENT_FILES = [
    "stage4_source_map.json",
    "stage4_report.txt",
    "stage4_geometric_reconstruction_overlay.png",
    "stage4_donor_selection_overlay.png",
    "stage4_source_map.png",
    "stage4_donor_reference_sheet.png",
    "source_expert_approved_reconstruction_hypotheses.json",
    "source_stage3c_b_facade_sectors.json",
    "source_stage3c_b_support_axis_sequence.json",
    "source_stage3c_b_ordered_bay_sequence.json",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def try_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def create_run_dir(output_root: Path) -> tuple[str, Path]:
    run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    try_mkdir(run_dir)
    return run_id, run_dir


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"stage5_ai_{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_traceback(run_dir: Path) -> None:
    (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")


def load_font(size: int):
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def copy_with_hash(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_dir():
        shutil.copytree(source, destination)
        return {
            "source": str(source.resolve()),
            "copied_to": str(destination.resolve()),
            "type": "directory",
        }
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "copied_to": str(destination.resolve()),
        "type": "file",
        "sha256": sha256_of_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def as_int_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_width(box: Sequence[int]) -> int:
    return int(box[2]) - int(box[0])


def box_height(box: Sequence[int]) -> int:
    return int(box[3]) - int(box[1])


def box_center(box: Sequence[int]) -> tuple[int, int]:
    return (int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2)))


def expand_box(box: Sequence[int], expand: int, width: int, height: int) -> list[int]:
    x1 = max(0, box[0] - expand)
    y1 = max(0, box[1] - expand)
    x2 = min(width, box[2] + expand)
    y2 = min(height, box[3] + expand)
    return [x1, y1, x2, y2]


def crop_array(img: np.ndarray, box: Sequence[int]) -> np.ndarray:
    return img[box[1]:box[3], box[0]:box[2]].copy()


def crop_pil(img: Image.Image, box: Sequence[int]) -> Image.Image:
    return img.crop((box[0], box[1], box[2], box[3]))


def normalize_mask(mask: np.ndarray, threshold: int = 128) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.where(mask >= threshold, 255, 0).astype(np.uint8)


def resize_to_model_dims(arr: np.ndarray, max_side: int, min_side: int, is_mask: bool = False) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Returns resized_arr, original_wh, resized_wh"""
    h, w = arr.shape[:2]
    orig_wh = (w, h)
    long_side = max(w, h)
    scale = min(1.0, float(max_side) / float(long_side)) if long_side > 0 else 1.0
    new_w = max(min_side if w < min_side and h < min_side else 1, int(round(w * scale)))
    new_h = max(min_side if w < min_side and h < min_side else 1, int(round(h * scale)))

    # Keep aspect when upscaling tiny crops
    if w >= h and new_w < min_side:
        scale2 = float(min_side) / max(1, new_w)
        new_w = min_side
        new_h = int(round(new_h * scale2))
    elif h > w and new_h < min_side:
        scale2 = float(min_side) / max(1, new_h)
        new_h = min_side
        new_w = int(round(new_w * scale2))

    # multiples of 8
    new_w = max(8, int(round(new_w / 8.0)) * 8)
    new_h = max(8, int(round(new_h / 8.0)) * 8)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
    if is_mask:
        resized = normalize_mask(resized)
    return resized, orig_wh, (new_w, new_h)


def restore_from_model_dims(arr: np.ndarray, original_wh: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    w, h = original_wh
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    restored = cv2.resize(arr, (w, h), interpolation=interp)
    if is_mask:
        restored = normalize_mask(restored)
    return restored


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# =============================================================================
# PARENT RUN
# =============================================================================
def is_valid_parent_run(path: Path) -> bool:
    if not path.is_dir() or not path.name.startswith("restore_"):
        return False
    for name in REQUIRED_PARENT_FILES:
        if not (path / name).exists():
            return False
    return True


def find_parent_run(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        path = Path(cfg.parent_run_dir)
        if not is_valid_parent_run(path):
            raise FileNotFoundError(f"Verilen parent run uygun Stage 4 çıktıları içermiyor: {path}")
        return path

    output_root = Path(cfg.output_root)
    candidates = sorted(
        [p for p in output_root.glob("restore_*") if is_valid_parent_run(p)],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Uygun Stage 4 run bulunamadı. Önce Stage 4 çalıştırılmalı.")
    return candidates[0]


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================
def normalize_observed_inventory(doc: dict[str, Any], width: int, height: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in doc.get("elements", []):
        clone = dict(item)
        clone["bbox_xyxy"] = as_int_box(item["bbox_xyxy"], width, height)
        out[item["id"]] = clone
    return out


def normalize_stage4_plan(doc: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    records = []
    for item in doc.get("elements", []):
        rec = dict(item)
        rec["expert_approved_bbox_xyxy"] = as_int_box(item["expert_approved_bbox_xyxy"], width, height)
        metrics = rec.get("geometry_metrics", {})
        if int(metrics.get("damage_limited_area_px", 0)) <= 0:
            continue
        records.append(rec)
    return records


def normalize_donor_selection(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in doc.get("selections", []):
        if isinstance(item, dict) and item.get("hypothesis_id"):
            out[item["hypothesis_id"]] = item
    return out


# =============================================================================
# PROMPTS
# =============================================================================
def class_prompt_fragment(class_name: str) -> str:
    mapping = {
        "arch": "continue the approved arch geometry, with consistent voussoir or masonry rhythm and believable historical construction logic",
        "pilaster": "continue the approved pilaster geometry, shaft and edge logic, matching the facade's architectural order",
        "cornice": "continue the approved cornice segment, maintaining the same horizontal profile and material logic",
        "base": "continue the approved base or plinth segment with the same horizontal massing and masonry logic",
    }
    return mapping.get(class_name, "continue the approved architectural geometry faithfully")


def build_prompt(cfg: Config, class_name: str) -> tuple[str, str]:
    base = (
        "photorealistic historical facade reconstruction, use only the style, material, texture, weathering, "
        "mortar logic, and lighting language of this exact building, reconstruct only inside the masked region, "
        "do not alter unmasked regions, no unrelated motifs, no extra windows, no extra columns"
    )
    class_frag = class_prompt_fragment(class_name)

    if cfg.intent == "conservation":
        prompt = (
            base + ", " + class_frag + ", preserve surrounding patina, produce a respectful conservation-aware completion, "
            "harmonious with existing historic fabric, slightly restrained, not over-restored, not brand new"
        )
    else:
        prompt = (
            base + ", " + class_frag + ", reconstruct as a coherent original-state continuation, "
            "matching the same building's own material and architectural language"
        )

    negative = (
        "modern architecture, fantasy, random historical motifs, unrelated ornament, extra columns, extra arches, "
        "new windows, brand new facade, glossy surfaces, concrete, glass curtain wall, cartoon, blurry, low quality, "
        "distorted structure, changing the unmasked area"
    )
    return prompt, negative


# =============================================================================
# GUIDE COMPOSITE / CONTROL IMAGES
# =============================================================================
def local_region_stats(img_bgr: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = img_bgr[region_mask > 0]
    if pixels.size == 0:
        mean = np.array([140.0, 140.0, 140.0], dtype=np.float32)
        std = np.array([18.0, 18.0, 18.0], dtype=np.float32)
        return mean, std
    return pixels.mean(axis=0).astype(np.float32), pixels.std(axis=0).astype(np.float32)


def build_ring_mask(full_mask: np.ndarray, crop_box: Sequence[int], target_box_global: Sequence[int]) -> np.ndarray:
    crop_mask = crop_array(full_mask, crop_box)
    h, w = crop_mask.shape[:2]
    ring = np.ones((h, w), dtype=np.uint8) * 255

    # inner protect zone around target bbox in crop coordinates
    lx1 = target_box_global[0] - crop_box[0]
    ly1 = target_box_global[1] - crop_box[1]
    lx2 = target_box_global[2] - crop_box[0]
    ly2 = target_box_global[3] - crop_box[1]
    ex = 6
    ring[max(0, ly1 - ex):min(h, ly2 + ex), max(0, lx1 - ex):min(w, lx2 + ex)] = 0
    ring[crop_mask > 0] = 0
    return ring


def match_patch_to_local_context(patch_bgr: np.ndarray, local_mean: np.ndarray) -> np.ndarray:
    patch = patch_bgr.astype(np.float32)
    cur = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
    delta = np.clip(local_mean - cur, -35.0, 35.0)
    patch += delta
    return np.clip(patch, 0, 255).astype(np.uint8)


def draw_geometry_outline(guide_bgr: np.ndarray, geometry_mask_crop: np.ndarray, color=(55, 55, 55)) -> np.ndarray:
    contours, _ = cv2.findContours(geometry_mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = guide_bgr.copy()
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def build_guide_composite(
    composite_bgr: np.ndarray,
    original_bgr: np.ndarray,
    crop_box: Sequence[int],
    target_box_global: Sequence[int],
    damage_mask_full: np.ndarray,
    full_geometry_mask_full: np.ndarray,
    donor_box: Sequence[int] | None,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    crop = crop_array(composite_bgr, crop_box)
    damage_crop = crop_array(damage_mask_full, crop_box)
    geometry_crop = crop_array(full_geometry_mask_full, crop_box)
    guide = crop.copy()

    ring_mask = build_ring_mask(damage_mask_full, crop_box, target_box_global)
    local_mean, local_std = local_region_stats(crop, ring_mask)

    lx1 = target_box_global[0] - crop_box[0]
    ly1 = target_box_global[1] - crop_box[1]
    lx2 = target_box_global[2] - crop_box[0]
    ly2 = target_box_global[3] - crop_box[1]

    fill_method = "local_mean_fill"
    if donor_box is not None:
        donor_crop = crop_array(original_bgr, donor_box)
        tw = max(1, lx2 - lx1)
        th = max(1, ly2 - ly1)
        donor_resized = cv2.resize(donor_crop, (tw, th), interpolation=cv2.INTER_CUBIC)
        donor_resized = match_patch_to_local_context(donor_resized, local_mean)
        geom_local = geometry_crop[ly1:ly2, lx1:lx2]
        roi = guide[ly1:ly2, lx1:lx2]
        roi[geom_local > 0] = donor_resized[geom_local > 0]
        guide[ly1:ly2, lx1:lx2] = roi
        fill_method = "donor_guided_fill"
    else:
        fill_color = np.array(local_mean, dtype=np.uint8)
        guide[geometry_crop > 0] = fill_color

    guide = draw_geometry_outline(guide, geometry_crop)

    # Slight blur only inside geometry to avoid hard donor cut edges in guide image
    blur_k = cfg.fill_blur_ksize if cfg.fill_blur_ksize % 2 == 1 else cfg.fill_blur_ksize + 1
    blurred = cv2.GaussianBlur(guide, (blur_k, blur_k), 0)
    alpha_mask = np.clip(geometry_crop.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_mask = cv2.GaussianBlur(alpha_mask, (blur_k, blur_k), 0)
    alpha3 = np.repeat(alpha_mask[..., None], 3, axis=2)
    guide = np.clip(blurred.astype(np.float32) * alpha3 + guide.astype(np.float32) * (1.0 - alpha3), 0, 255).astype(np.uint8)

    meta = {
        "guide_fill_method": fill_method,
        "local_mean_bgr": [round(float(v), 4) for v in local_mean.tolist()],
        "local_std_bgr": [round(float(v), 4) for v in local_std.tolist()],
    }
    return guide, meta


# =============================================================================
# CONTROLNET PREP
# =============================================================================
class ControlPreprocessors:
    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.device = device
        self.lineart = LineartDetector.from_pretrained(cfg.lineart_model_id)
        self.depth_model = DPTForDepthEstimation.from_pretrained(cfg.depth_model_id)
        self.depth_processor = AutoImageProcessor.from_pretrained(cfg.depth_model_id)
        self.depth_model = self.depth_model.to(device)
        self.depth_model.eval()

    def prepare_lineart(self, pil_img: Image.Image) -> Image.Image:
        out = self.lineart(pil_img)
        return out.convert("RGB") if out.mode != "RGB" else out

    def prepare_depth(self, pil_img: Image.Image) -> Image.Image:
        inputs = self.depth_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            pred = self.depth_model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=pil_img.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        arr = pred.squeeze().detach().cpu().numpy()
        arr = arr - arr.min()
        arr = arr / max(1e-8, arr.max())
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


# =============================================================================
# PIPELINE LOAD
# =============================================================================
def build_pipe(cfg: Config, device: str, dtype: torch.dtype):
    cn_depth = ControlNetModel.from_pretrained(cfg.controlnet_depth_id, torch_dtype=dtype)
    cn_lineart = ControlNetModel.from_pretrained(cfg.controlnet_lineart_id, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_id, torch_dtype=dtype)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        cfg.base_model_id,
        controlnet=[cn_depth, cn_lineart],
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    if cfg.prefer_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if cfg.attention_slicing:
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pass
    if cfg.vae_slicing:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    if cfg.vae_tiling:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    if cfg.use_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
    return pipe


# =============================================================================
# GENERATION
# =============================================================================
def generate_element_result(
    pipe,
    preprocessors: ControlPreprocessors,
    composite_bgr: np.ndarray,
    original_bgr: np.ndarray,
    crop_box: Sequence[int],
    target_box_global: Sequence[int],
    damage_mask_full: np.ndarray,
    full_geometry_mask_full: np.ndarray,
    donor_box: Sequence[int] | None,
    class_name: str,
    seed: int,
    cfg: Config,
    debug_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    crop_bgr = crop_array(composite_bgr, crop_box)
    damage_crop = crop_array(damage_mask_full, crop_box)
    full_geom_crop = crop_array(full_geometry_mask_full, crop_box)

    guide_bgr, guide_meta = build_guide_composite(
        composite_bgr=composite_bgr,
        original_bgr=original_bgr,
        crop_box=crop_box,
        target_box_global=target_box_global,
        damage_mask_full=damage_mask_full,
        full_geometry_mask_full=full_geometry_mask_full,
        donor_box=donor_box,
        cfg=cfg,
    )

    # Resize all assets consistently for the diffusion model
    crop_resized, orig_wh, _ = resize_to_model_dims(crop_bgr, cfg.max_crop_side, cfg.min_crop_side, is_mask=False)
    guide_resized, _, _ = resize_to_model_dims(guide_bgr, cfg.max_crop_side, cfg.min_crop_side, is_mask=False)
    damage_resized, _, _ = resize_to_model_dims(damage_crop, cfg.max_crop_side, cfg.min_crop_side, is_mask=True)
    full_geom_resized, _, _ = resize_to_model_dims(full_geom_crop, cfg.max_crop_side, cfg.min_crop_side, is_mask=True)

    # Enforce identical spatial size after possible rounding differences
    model_w, model_h = crop_resized.shape[1], crop_resized.shape[0]
    guide_resized = cv2.resize(guide_resized, (model_w, model_h), interpolation=cv2.INTER_CUBIC)
    damage_resized = normalize_mask(cv2.resize(damage_resized, (model_w, model_h), interpolation=cv2.INTER_NEAREST))
    full_geom_resized = normalize_mask(cv2.resize(full_geom_resized, (model_w, model_h), interpolation=cv2.INTER_NEAREST))

    prompt, negative = build_prompt(cfg, class_name)

    crop_pil_img = bgr_to_pil(crop_resized)
    guide_pil_img = bgr_to_pil(guide_resized)
    mask_pil_img = Image.fromarray(damage_resized, mode="L")
    lineart_pil = preprocessors.prepare_lineart(guide_pil_img)
    depth_pil = preprocessors.prepare_depth(guide_pil_img)

    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=crop_pil_img,
        mask_image=mask_pil_img,
        control_image=[depth_pil, lineart_pil],
        controlnet_conditioning_scale=[
            cfg.controlnet_conditioning_scale_depth,
            cfg.controlnet_conditioning_scale_lineart,
        ],
        control_guidance_end=[
            cfg.control_guidance_end_depth,
            cfg.control_guidance_end_lineart,
        ],
        strength=cfg.strength,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    ).images[0]

    result_bgr = pil_to_bgr(result)
    result_bgr = restore_from_model_dims(result_bgr, orig_wh, is_mask=False)

    damage_restored = restore_from_model_dims(damage_resized, orig_wh, is_mask=True)
    full_geom_restored = restore_from_model_dims(full_geom_resized, orig_wh, is_mask=True)
    lineart_restored = restore_from_model_dims(pil_to_bgr(lineart_pil), orig_wh, is_mask=False)
    depth_restored = restore_from_model_dims(pil_to_bgr(depth_pil), orig_wh, is_mask=False)
    guide_restored = restore_from_model_dims(guide_resized, orig_wh, is_mask=False)

    # Manual strict composite: only mask region changes.
    composited_crop = crop_bgr.copy()
    composited_crop[damage_restored > 0] = result_bgr[damage_restored > 0]

    debug_meta = {
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "model_crop_size": [model_w, model_h],
        "crop_box_xyxy": list(map(int, crop_box)),
        "target_box_global_xyxy": list(map(int, target_box_global)),
        "mask_area_px": int(np.count_nonzero(damage_restored)),
        "full_geometry_area_px": int(np.count_nonzero(full_geom_restored)),
        **guide_meta,
    }

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        bgr_to_pil(crop_bgr).save(debug_dir / "01_crop_before.png")
        bgr_to_pil(guide_restored).save(debug_dir / "02_guide_composite.png")
        Image.fromarray(full_geom_restored, mode="L").save(debug_dir / "03_full_geometry_mask.png")
        Image.fromarray(damage_restored, mode="L").save(debug_dir / "04_damage_mask.png")
        lineart_pil.resize(orig_wh, Image.NEAREST).save(debug_dir / "05_lineart.png")
        depth_pil.resize(orig_wh, Image.BICUBIC).save(debug_dir / "06_depth.png")
        bgr_to_pil(result_bgr).save(debug_dir / "07_ai_generated_crop.png")
        bgr_to_pil(composited_crop).save(debug_dir / "08_composited_crop.png")
        write_json(debug_dir / "debug_meta.json", debug_meta)

    return composited_crop, debug_meta


# =============================================================================
# VISUALIZATION / REPORT
# =============================================================================
def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71), (65, 105, 225), (50, 205, 50), (255, 165, 0),
        (186, 85, 211), (220, 20, 60), (70, 130, 180), (255, 215, 0),
        (0, 191, 255), (218, 112, 214),
    ]
    return palette[(index - 1) % len(palette)]


def draw_stage5_overlay(base_img: Image.Image, records: list[dict[str, Any]], title: str) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    small = load_font(12)
    draw.rectangle([0, 0, img.width, 24], fill=(18, 18, 18))
    draw.text((8, 4), title, fill=(255, 255, 255), font=font)

    for i, rec in enumerate(records, start=1):
        box = rec["expert_approved_bbox_xyxy"]
        color = color_for_index(i)
        draw.rectangle(box, outline=color, width=3)
        label = f"#{i} {rec['class']} seed={rec['ai_integration']['seed']}"
        tx = box[0]
        ty = max(26, box[1] - 16)
        tw = max(100, len(label) * 7)
        draw.rectangle([tx, ty, tx + tw, ty + 14], fill=color)
        draw.text((tx + 2, ty + 1), label, fill=(0, 0, 0), font=small)
    return img


def build_intervention_only(image_shape: tuple[int, int, int], final_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = np.full(image_shape, 245, dtype=np.uint8)
    sel = mask > 0
    canvas[sel] = final_bgr[sel]
    return canvas


def build_source_map(image_shape: tuple[int, int, int], records: list[dict[str, Any]], masks: list[np.ndarray]) -> np.ndarray:
    canvas = np.zeros(image_shape, dtype=np.uint8)
    for i, mask in enumerate(masks, start=1):
        color = np.array(color_for_index(i), dtype=np.uint8)
        canvas[mask > 0] = color
    return canvas


def build_donor_reference_sheet(original_rgb: Image.Image, records: list[dict[str, Any]], observed: dict[str, dict[str, Any]]) -> Image.Image:
    card_w = 320
    card_h = 220
    cols = 3
    cards: list[Image.Image] = []
    font = load_font(14)
    small = load_font(12)

    for idx, rec in enumerate(records, start=1):
        card = Image.new("RGB", (card_w, card_h), (250, 250, 250))
        draw = ImageDraw.Draw(card)
        color = color_for_index(idx)
        draw.rectangle([0, 0, card_w - 1, card_h - 1], outline=(90, 90, 90), width=1)
        draw.rectangle([0, 0, card_w, 24], fill=color)
        draw.text((8, 4), f"#{idx} {rec['hypothesis_id']} | {rec['class']}", fill=(0, 0, 0), font=font)
        donor_id = rec.get("donor_selection", {}).get("primary_donor_id")
        if donor_id and donor_id in observed:
            donor_box = observed[donor_id]["bbox_xyxy"]
            crop = crop_pil(original_rgb, donor_box)
            crop.thumbnail((card_w - 16, 150))
            px = 8 + (card_w - 16 - crop.width) // 2
            py = 36 + max(0, (140 - crop.height) // 2)
            card.paste(crop, (px, py))
            draw.text((8, card_h - 38), f"donor={donor_id}", fill=(25, 25, 25), font=small)
            draw.text((8, card_h - 22), f"selection={rec.get('donor_selection', {}).get('selection_state')}", fill=(25, 25, 25), font=small)
        else:
            draw.text((8, 82), "Birincil donor yok.\nGuide composite + local facade context", fill=(120, 40, 40), font=font)
        cards.append(card)

    rows = math.ceil(len(cards) / cols) if cards else 1
    sheet = Image.new("RGB", (cols * card_w, rows * card_h), (225, 225, 225))
    for idx, card in enumerate(cards):
        x = (idx % cols) * card_w
        y = (idx // cols) * card_h
        sheet.paste(card, (x, y))
    return sheet


def build_report(run_id: str, parent_run: Path, cfg: Config, records: list[dict[str, Any]], union_mask: np.ndarray) -> str:
    classes: dict[str, int] = {}
    donor_count = 0
    for rec in records:
        classes[rec["class"]] = classes.get(rec["class"], 0) + 1
        if rec.get("donor_selection", {}).get("primary_donor_id"):
            donor_count += 1
    lines = [
        "STAGE 5 — AI INPAINT MATERIAL / TEXTURE / LIGHT INTEGRATION",
        f"Run: {run_id}",
        f"Parent: {parent_run.resolve()}",
        f"Intent: {cfg.intent}",
        "",
        "POLICY:",
        "- Stage 4 geometry is authoritative.",
        "- AI is allowed only inside damage-limited masks.",
        "- Unmasked pixels are manually preserved after generation.",
        "- Donors are guidance references, not direct final copy-paste outputs.",
        "",
        "SUMMARY:",
        f"- Processed elements: {len(records)}",
        f"- Class counts: {classes}",
        f"- Elements with observed donors: {donor_count}",
        f"- Elements without observed donors: {len(records) - donor_count}",
        f"- Intervention pixels: {int(np.count_nonzero(union_mask))}",
        "",
        "ELEMENTS:",
    ]
    for rec in records:
        ai = rec.get("ai_integration", {})
        lines.append(
            f"- {rec['hypothesis_id']} | {rec['class']} | donor={rec.get('donor_selection', {}).get('primary_donor_id') or 'NONE'} | "
            f"seed={ai.get('seed')} | mask_px={ai.get('mask_area_px')} | crop={ai.get('crop_box_xyxy')}"
        )
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN STAGE
# =============================================================================
def run_stage(cfg: Config) -> Path:
    parent_run = find_parent_run(cfg)
    run_id, run_dir = create_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started_at = now_iso()
    perf0 = time.perf_counter()

    try:
        logger.info("=" * 92)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info(f"[STAGE] {STAGE_TITLE}")
        logger.info(f"[INTENT] {cfg.intent}")
        logger.info("[RULE] geometry fixed, AI inpaint only inside damage-limited masks")
        logger.info("=" * 92)

        input_path = parent_run / "input.png"
        completion_mask_path = parent_run / "completion_mask.png"
        inventory_path = parent_run / "source_expert_validated_inventory.json"
        plan_path = parent_run / "stage4_geometric_reconstruction_plan.json"
        donor_selection_path = parent_run / "stage4_donor_selection.json"
        union_mask_path = parent_run / "stage4_geometry_damage_limited_mask.png"

        original_rgb = Image.open(input_path).convert("RGB")
        width, height = original_rgb.size
        original_bgr = pil_to_bgr(original_rgb)
        working_bgr = original_bgr.copy()

        completion_mask = normalize_mask(
            np.asarray(Image.open(completion_mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)),
            cfg.mask_threshold,
        )
        stage4_union_mask = normalize_mask(
            np.asarray(Image.open(union_mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)),
            cfg.mask_threshold,
        )

        observed = normalize_observed_inventory(read_json(inventory_path), width, height)
        plan_records = normalize_stage4_plan(read_json(plan_path), width, height)
        donor_selections = normalize_donor_selection(read_json(donor_selection_path))

        if not plan_records:
            raise RuntimeError("Stage 4 planında işlenecek uygun eleman bulunamadı.")

        # Copy trace inputs
        copied_inputs: dict[str, Any] = {
            "input": copy_with_hash(input_path, run_dir / "input.png"),
            "completion_mask": copy_with_hash(completion_mask_path, run_dir / "completion_mask.png"),
            "expert_inventory": copy_with_hash(inventory_path, run_dir / "source_expert_validated_inventory.json"),
            "stage4_plan": copy_with_hash(plan_path, run_dir / "source_stage4_geometric_reconstruction_plan.json"),
            "stage4_donor_selection": copy_with_hash(donor_selection_path, run_dir / "source_stage4_donor_selection.json"),
            "stage4_union_mask": copy_with_hash(union_mask_path, run_dir / "source_stage4_geometry_damage_limited_mask.png"),
            "reconstruction_element_masks": copy_with_hash(parent_run / "reconstruction_element_masks", run_dir / "source_reconstruction_element_masks"),
        }
        for name in OPTIONAL_PARENT_FILES:
            src = parent_run / name
            if src.exists():
                dst = run_dir / (name if name.startswith("source_") else f"source_{name}")
                copied_inputs[name] = copy_with_hash(src, dst)

        # Device / models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"[DEVICE] {device}")
        logger.info("[MODEL] Loading ControlNet inpaint pipeline and preprocessors...")
        pipe = build_pipe(cfg, device, dtype)
        preprocessors = ControlPreprocessors(cfg, device)
        logger.info("[MODEL] Ready")

        element_debug_root = run_dir / "stage5_element_debug"
        if cfg.save_debug_assets:
            element_debug_root.mkdir(parents=True, exist_ok=True)

        final_records: list[dict[str, Any]] = []
        per_element_masks: list[np.ndarray] = []
        union_intervention = np.zeros((height, width), dtype=np.uint8)

        for idx, rec in enumerate(plan_records, start=1):
            hyp_id = rec["hypothesis_id"]
            class_name = rec["class"]
            target_box = as_int_box(rec["expert_approved_bbox_xyxy"], width, height)
            donor_sel = donor_selections.get(hyp_id, rec.get("donor_selection", {}))

            # Load masks
            damage_rel = rec["output_masks"]["damage_limited"]
            full_rel = rec["output_masks"]["full_geometry"]
            damage_full = normalize_mask(
                np.asarray(Image.open(parent_run / damage_rel).convert("L").resize((width, height), Image.Resampling.NEAREST)),
                cfg.mask_threshold,
            )
            full_geom_full = normalize_mask(
                np.asarray(Image.open(parent_run / full_rel).convert("L").resize((width, height), Image.Resampling.NEAREST)),
                cfg.mask_threshold,
            )

            # strict policy intersection
            damage_full = cv2.bitwise_and(damage_full, completion_mask)
            damage_full = cv2.bitwise_and(damage_full, stage4_union_mask)
            full_geom_full = cv2.bitwise_and(full_geom_full, completion_mask)
            if np.count_nonzero(damage_full) == 0:
                logger.info(f"[SKIP] {hyp_id} mask empty after policy intersection")
                continue

            crop_box = expand_box(target_box, cfg.crop_context_px, width, height)
            donor_id = donor_sel.get("primary_donor_id") if isinstance(donor_sel, dict) else None
            donor_box = observed[donor_id]["bbox_xyxy"] if donor_id and donor_id in observed else None
            seed = cfg.base_seed + idx

            debug_dir = (element_debug_root / hyp_id) if cfg.save_debug_assets else None
            composited_crop, ai_meta = generate_element_result(
                pipe=pipe,
                preprocessors=preprocessors,
                composite_bgr=working_bgr,
                original_bgr=original_bgr,
                crop_box=crop_box,
                target_box_global=target_box,
                damage_mask_full=damage_full,
                full_geometry_mask_full=full_geom_full,
                donor_box=donor_box,
                class_name=class_name,
                seed=seed,
                cfg=cfg,
                debug_dir=debug_dir,
            )

            # Composite back strictly inside element damage mask crop
            crop_mask = crop_array(damage_full, crop_box)
            current_crop = crop_array(working_bgr, crop_box)
            current_crop[crop_mask > 0] = composited_crop[crop_mask > 0]
            working_bgr[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = current_crop

            union_intervention = cv2.bitwise_or(union_intervention, damage_full)
            per_element_masks.append(damage_full.copy())

            rec_out = dict(rec)
            rec_out["donor_selection"] = donor_sel
            rec_out["ai_integration"] = {
                **ai_meta,
                "intent": cfg.intent,
                "base_model_id": cfg.base_model_id,
                "vae_model_id": cfg.vae_model_id,
                "controlnet_depth_id": cfg.controlnet_depth_id,
                "controlnet_lineart_id": cfg.controlnet_lineart_id,
                "strength": cfg.strength,
                "num_inference_steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale,
                "controlnet_conditioning_scale": {
                    "depth": cfg.controlnet_conditioning_scale_depth,
                    "lineart": cfg.controlnet_conditioning_scale_lineart,
                },
            }
            final_records.append(rec_out)
            logger.info(
                "[ELEMENT] %s class=%s donor=%s seed=%d mask_px=%d",
                hyp_id,
                class_name,
                donor_id or "NONE",
                seed,
                ai_meta["mask_area_px"],
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not final_records:
            raise RuntimeError("Hiçbir eleman Stage 5'te işlenemedi.")

        # Outputs
        final_rgb = bgr_to_pil(working_bgr)
        final_rgb.save(run_dir / "stage5_ai_integrated.png")

        intervention_only = build_intervention_only(working_bgr.shape, working_bgr, union_intervention)
        bgr_to_pil(intervention_only).save(run_dir / "stage5_ai_intervention_only.png")

        overlay = draw_stage5_overlay(final_rgb, final_records, f"{STAGE_TITLE} | {cfg.intent}")
        overlay.save(run_dir / "stage5_ai_overlay.png")

        source_map = build_source_map(working_bgr.shape, final_records, per_element_masks)
        bgr_to_pil(source_map).save(run_dir / "stage5_ai_source_map.png")

        donor_sheet = build_donor_reference_sheet(original_rgb, final_records, observed)
        donor_sheet.save(run_dir / "stage5_ai_donor_reference_sheet.png")

        Image.fromarray(union_intervention, mode="L").save(run_dir / "stage5_ai_union_mask.png")

        plan_doc = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "title": STAGE_TITLE,
            "status": "completed",
            "parent_run": {
                "run_id": parent_run.name,
                "path": str(parent_run.resolve()),
            },
            "policy": {
                "stage4_geometry_authoritative": True,
                "ai_inpaint_used": True,
                "inpaint_only_inside_damage_mask": True,
                "unmasked_pixels_manually_preserved": True,
                "donor_is_reference_not_final_copy": True,
            },
            "summary": {
                "processed_element_count": len(final_records),
                "intervention_area_px": int(np.count_nonzero(union_intervention)),
                "with_observed_donor_count": sum(1 for r in final_records if r.get("donor_selection", {}).get("primary_donor_id")),
                "without_observed_donor_count": sum(1 for r in final_records if not r.get("donor_selection", {}).get("primary_donor_id")),
            },
            "output_files": {
                "integrated_image": "stage5_ai_integrated.png",
                "intervention_only": "stage5_ai_intervention_only.png",
                "overlay": "stage5_ai_overlay.png",
                "source_map": "stage5_ai_source_map.png",
                "donor_reference_sheet": "stage5_ai_donor_reference_sheet.png",
                "union_mask": "stage5_ai_union_mask.png",
            },
            "elements": final_records,
        }
        write_json(run_dir / "stage5_ai_integration_plan.json", plan_doc)
        write_json(run_dir / "stage5_ai_element_records.json", final_records)

        report = build_report(run_id, parent_run, cfg, final_records, union_intervention)
        (run_dir / "stage5_ai_report.txt").write_text(report, encoding="utf-8")

        finished_at = now_iso()
        duration = time.perf_counter() - perf0
        run_meta = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "title": STAGE_TITLE,
            "script_name": SCRIPT_NAME,
            "timestamp_started": started_at,
            "timestamp_finished": finished_at,
            "duration_seconds": round(duration, 3),
            "parent_run": str(parent_run.resolve()),
            "intent": cfg.intent,
            "counts": {
                "processed_element_count": len(final_records),
                "intervention_area_px": int(np.count_nonzero(union_intervention)),
                "with_observed_donor_count": sum(1 for r in final_records if r.get("donor_selection", {}).get("primary_donor_id")),
                "without_observed_donor_count": sum(1 for r in final_records if not r.get("donor_selection", {}).get("primary_donor_id")),
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": device,
            },
            "config": asdict(cfg),
            "copied_inputs": copied_inputs,
        }
        write_json(run_dir / "run_meta.json", run_meta)

        logger.info("[OK] Stage 5 AI tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            "[COUNT] processed=%d intervention_px=%d donors=%d no_donor=%d",
            len(final_records),
            int(np.count_nonzero(union_intervention)),
            sum(1 for r in final_records if r.get("donor_selection", {}).get("primary_donor_id")),
            sum(1 for r in final_records if not r.get("donor_selection", {}).get("primary_donor_id")),
        )
        logger.info("[FILES] stage5_ai_integrated.png, stage5_ai_intervention_only.png, stage5_ai_integration_plan.json")
        return run_dir

    except Exception:
        save_traceback(run_dir)
        logger.exception("[FAIL] Stage 5 AI başarısız oldu")
        raise


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=STAGE_TITLE)
    parser.add_argument("--parent-run", dest="parent_run_dir", default="", help="Belirli Stage 4 run klasörü")
    parser.add_argument("--output-root", dest="output_root", default="outputs", help="Çıktı kök klasörü")
    parser.add_argument("--intent", dest="intent", default="original_state", choices=["original_state", "conservation"], help="Prompt niyeti")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = Config(
        output_root=args.output_root,
        parent_run_dir=args.parent_run_dir,
        intent=args.intent,
    )
    run_dir = run_stage(cfg)
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()