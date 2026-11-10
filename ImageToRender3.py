#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage1c_cleanup.py
================================================
V8 / STAGE 1C — Çok ölçekli eleman tespiti + mimari temizleme

Amaç
-----
Stage 1B'nin ürettiği gürültülü tespitleri daha mimari ve daha kullanılabilir hale getirmek.
Bu sürüm şunları yapar:

1. image.png + completion_mask.png alır.
2. Çok ölçekli OWL-ViT ile aday kutular bulur.
3. SAM ile kutuların piksel maskelerini çıkarır.
4. Yalnızca ana sınıfları tutar:
      arch, pilaster, window, door
5. Sınıf bazlı geometri filtreleri uygular.
6. Hasar alanıyla fazla çakışan ve çok düşük güvenli adayları ayıklar.
7. Tekrarlayan mimari elemanları gruplar:
      - arch rows
      - pilaster families
      - olası pilaster-arch-pilaster ilişkileri
8. Çıktı olarak temiz elemanlar ve örüntü raporu üretir.

Bu aşamada hâlâ restorasyon üretimi yoktur.
Stable Diffusion / ControlNet YOK.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node.*",
    category=FutureWarning,
)

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor, SamModel, SamProcessor

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# =============================================================================
# CONFIG
# =============================================================================
@dataclass(frozen=True)
class PromptSpec:
    text: str
    canonical_class: str


@dataclass
class Config:
    input_photo: str = "image.png"
    completion_mask: str = "completion_mask.png"
    output_root: str = "outputs"

    detector_backend: str = "owlvit_multiscale_cleanup"
    detector_model_id: str = "google/owlvit-base-patch32"
    sam_model_id: str = "facebook/sam-vit-base"

    prompt_specs: list[PromptSpec] = field(
        default_factory=lambda: [
            PromptSpec("arch", "arch"),
            PromptSpec("stone arch", "arch"),
            PromptSpec("arched opening", "arch"),
            PromptSpec("arcade", "arch"),
            PromptSpec("pilaster", "pilaster"),
            PromptSpec("wall pillar", "pilaster"),
            PromptSpec("rectangular column", "pilaster"),
            PromptSpec("column", "pilaster"),
            PromptSpec("window", "window"),
            PromptSpec("window opening", "window"),
            PromptSpec("door", "door"),
            PromptSpec("doorway", "door"),
        ]
    )

    use_full_image_view: bool = True
    use_damage_context_view: bool = True
    damage_context_expand: float = 2.4
    damage_context_min_fraction: float = 0.50
    tile_fraction_of_short_side: float = 0.68
    tile_overlap: float = 0.38
    max_detection_views: int = 7
    save_detection_views: bool = True

    absolute_score_floor: float = 0.008
    relative_score_ratio: float = 0.58
    fallback_score_floor: float = 0.0035
    max_candidates_per_prompt_per_view: int = 3
    max_candidates_per_class_before_nms: int = 20

    nms_iou_threshold: float = 0.42
    max_detections_per_class: int = 12
    max_total_detections: int = 42

    min_box_area_ratio_in_view: float = 0.0010
    max_box_area_ratio_in_view: float = 0.70

    sam_batch_size: int = 6

    max_damage_overlap_for_usable: float = 0.15
    min_element_area_ratio: float = 0.00010
    pattern_score_floor: float = 0.012

    # Stage 1C temizleme
    kept_classes: tuple[str, ...] = ("arch", "pilaster", "window", "door")
    allow_low_quality_for_kept_classes: bool = False
    max_damage_overlap_for_clean: float = 0.28
    min_confidence_for_clean: float = 0.010

    # Mimari gramer için satır / aile toleransları
    arch_row_y_tolerance_ratio: float = 0.12   # benzer merkez y
    arch_row_height_tolerance_ratio: float = 0.25
    pilaster_width_tolerance_ratio: float = 0.45
    pilaster_height_tolerance_ratio: float = 0.35

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16_if_cuda: bool = True


@dataclass(frozen=True)
class DetectionView:
    view_id: str
    box_xyxy: tuple[int, int, int, int]
    reason: str


# =============================================================================
# UTILITIES
# =============================================================================
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(cfg: Config) -> tuple[str, Path]:
    run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
    return run_id, ensure_dir(Path(cfg.output_root) / run_id)


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"v8_stage1c_cleanup_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def collect_environment() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "transformers": safe_version("transformers"),
        "huggingface_hub": safe_version("huggingface-hub"),
        "opencv_python": safe_version("opencv-python"),
        "pillow": safe_version("Pillow"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        data["gpu_name"] = torch.cuda.get_device_name(0)
        data["cuda_runtime"] = torch.version.cuda
    return data


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    mask = Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask)
    return np.where(array >= 128, 255, 0).astype(np.uint8)


def mask_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def damage_overlap_ratio(element_mask: np.ndarray, damage_mask: np.ndarray) -> float:
    element = element_mask > 0
    area = int(np.count_nonzero(element))
    if area == 0:
        return 0.0
    overlap = int(np.count_nonzero(element & (damage_mask > 0)))
    return float(overlap) / float(area)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    binary = (mask > 0).astype(np.float32)[..., None]
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out = out * (1.0 - binary * alpha) + tint * (binary * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_") or "element"


def clamp_box_xyxy(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(np.clip(round(x1), 0, max(0, width - 1)))
    y1 = int(np.clip(round(y1), 0, max(0, height - 1)))
    x2 = int(np.clip(round(x2), 0, max(0, width - 1)))
    y2 = int(np.clip(round(y2), 0, max(0, height - 1)))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def box_area_ratio(box: Iterable[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, float(width * height)))


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def move_batch_to_device(batch: dict[str, Any], device: str, fp16: bool) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if fp16 and value.dtype.is_floating_point:
                value = value.half()
        moved[key] = value
    return moved


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# DETECTION VIEWS
# =============================================================================
def evenly_spaced_positions(full: int, crop: int, overlap: float) -> list[int]:
    if crop >= full:
        return [0]
    step = max(1, int(round(crop * (1.0 - overlap))))
    positions = list(range(0, max(1, full - crop + 1), step))
    last = full - crop
    if not positions or positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


def bbox_from_binary(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def expanded_square_crop(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    expand: float,
    minimum_side: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * expand
    side = max(float(minimum_side), side)
    side = min(side, float(max(image_width, image_height)))

    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = nx1 + int(round(side))
    ny2 = ny1 + int(round(side))

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > image_width:
        shift = nx2 - image_width
        nx1 -= shift
        nx2 = image_width
    if ny2 > image_height:
        shift = ny2 - image_height
        ny1 -= shift
        ny2 = image_height

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(image_width, nx2)
    ny2 = min(image_height, ny2)
    return nx1, ny1, nx2, ny2


def deduplicate_views(views: list[DetectionView], iou_threshold: float = 0.92) -> list[DetectionView]:
    kept: list[DetectionView] = []
    for view in views:
        if any(box_iou(view.box_xyxy, old.box_xyxy) >= iou_threshold for old in kept):
            continue
        kept.append(view)
    return kept


def build_detection_views(width: int, height: int, damage_mask: np.ndarray, cfg: Config) -> list[DetectionView]:
    views: list[DetectionView] = []

    if cfg.use_full_image_view:
        views.append(DetectionView("full", (0, 0, width, height), "full_image"))

    damage_box = bbox_from_binary(damage_mask)
    if cfg.use_damage_context_view and damage_box is not None:
        minimum_side = int(round(min(width, height) * cfg.damage_context_min_fraction))
        context_box = expanded_square_crop(
            damage_box, width, height, cfg.damage_context_expand, minimum_side
        )
        views.append(DetectionView("damage_context", context_box, "expanded_damage_context"))

    short_side = min(width, height)
    tile_side = int(round(short_side * cfg.tile_fraction_of_short_side))
    tile_side = max(192, min(tile_side, width, height))

    x_positions = evenly_spaced_positions(width, tile_side, cfg.tile_overlap)
    y_positions = evenly_spaced_positions(height, tile_side, cfg.tile_overlap)

    tile_index = 0
    for y in y_positions:
        for x in x_positions:
            tile_index += 1
            views.append(
                DetectionView(
                    f"tile_{tile_index:02d}",
                    (x, y, x + tile_side, y + tile_side),
                    "overlapping_square_tile",
                )
            )

    views = deduplicate_views(views)
    return views[: max(1, cfg.max_detection_views)]


def save_views_preview(original: np.ndarray, views: list[DetectionView], output_path: Path) -> None:
    preview = original.copy()
    for index, view in enumerate(views, start=1):
        x1, y1, x2, y2 = view.box_xyxy
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 220, 40), 2)
        cv2.putText(
            preview,
            f"{index}:{view.view_id}",
            (x1 + 4, min(preview.shape[0] - 8, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 220, 40),
            1,
            cv2.LINE_AA,
        )
    save_rgb(output_path, preview)


# =============================================================================
# NMS
# =============================================================================
def class_aware_nms(detections: list[dict[str, Any]], iou_threshold: float, max_per_class: int, max_total: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        grouped.setdefault(str(det["canonical_class"]), []).append(det)

    kept_all: list[dict[str, Any]] = []
    for canonical_class, group in grouped.items():
        group = sorted(group, key=lambda item: float(item["score"]), reverse=True)
        kept_group: list[dict[str, Any]] = []
        for candidate in group:
            if all(box_iou(candidate["box"], accepted["box"]) < iou_threshold for accepted in kept_group):
                kept_group.append(candidate)
            if len(kept_group) >= max_per_class:
                break
        kept_all.extend(kept_group)

    kept_all.sort(key=lambda item: float(item["score"]), reverse=True)
    return kept_all[:max_total]


# =============================================================================
# VISUALIZATION
# =============================================================================
def draw_box_and_label(rgb: np.ndarray, box: list[int], label: str, score: float, color: tuple[int, int, int], suffix: str = "") -> np.ndarray:
    out = rgb.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text = f"{label} {score:.3f}{suffix}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    top = max(0, y1 - th - 8)
    cv2.rectangle(out, (x1, top), (min(out.shape[1] - 1, x1 + tw + 6), y1), color, -1)
    cv2.putText(out, text, (x1 + 3, max(th + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# =============================================================================
# OWL-VIT MULTISCALE DETECTION
# =============================================================================
def select_view_candidates(
    scores: list[float],
    labels: list[int],
    boxes: list[list[float]],
    crop_width: int,
    crop_height: int,
    view: DetectionView,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_prompt: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(cfg.prompt_specs))}

    for score, label_index, local_box in zip(scores, labels, boxes):
        label_index = int(label_index)
        if label_index < 0 or label_index >= len(cfg.prompt_specs):
            continue

        area_ratio = box_area_ratio(local_box, crop_width, crop_height)
        if area_ratio < cfg.min_box_area_ratio_in_view or area_ratio > cfg.max_box_area_ratio_in_view:
            continue

        by_prompt[label_index].append(
            {
                "score": float(score),
                "local_box": [float(v) for v in local_box],
                "area_ratio_in_view": float(area_ratio),
            }
        )

    selected: list[dict[str, Any]] = []
    prompt_diagnostics: list[dict[str, Any]] = []
    vx1, vy1, _, _ = view.box_xyxy

    for prompt_index, spec in enumerate(cfg.prompt_specs):
        candidates = sorted(by_prompt[prompt_index], key=lambda item: float(item["score"]), reverse=True)
        top_scores = [round(float(item["score"]), 6) for item in candidates[:5]]
        best_score = float(candidates[0]["score"]) if candidates else 0.0
        dynamic_threshold = max(cfg.absolute_score_floor, best_score * cfg.relative_score_ratio)

        primary = [item for item in candidates if float(item["score"]) >= dynamic_threshold][: cfg.max_candidates_per_prompt_per_view]
        quality = "primary"
        chosen = primary

        if not chosen and candidates and best_score >= cfg.fallback_score_floor:
            chosen = [candidates[0]]
            quality = "fallback"

        prompt_diagnostics.append(
            {
                "prompt_index": prompt_index,
                "prompt": spec.text,
                "canonical_class": spec.canonical_class,
                "best_score": round(best_score, 6),
                "dynamic_threshold": round(dynamic_threshold, 6),
                "top_scores": top_scores,
                "selected_count": len(chosen),
                "selection_quality": quality if chosen else "none",
            }
        )

        for item in chosen:
            lx1, ly1, lx2, ly2 = item["local_box"]
            global_box = [float(lx1 + vx1), float(ly1 + vy1), float(lx2 + vx1), float(ly2 + vy1)]
            selected.append(
                {
                    "label": spec.text,
                    "canonical_class": spec.canonical_class,
                    "prompt_index": prompt_index,
                    "score": float(item["score"]),
                    "box": global_box,
                    "local_box": item["local_box"],
                    "view_id": view.view_id,
                    "view_reason": view.reason,
                    "quality": quality,
                    "area_ratio_in_view": item["area_ratio_in_view"],
                }
            )

    diagnostics = {
        "view_id": view.view_id,
        "reason": view.reason,
        "box_xyxy": list(view.box_xyxy),
        "crop_size": [crop_width, crop_height],
        "selected_count": len(selected),
        "prompts": prompt_diagnostics,
    }
    return selected, diagnostics


def detect_multiscale_with_owlvit(
    original: Image.Image,
    views: list[DetectionView],
    cfg: Config,
    logger: logging.Logger,
    view_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logger.info(f"[MODEL] OWL-ViT yükleniyor: {cfg.detector_model_id}")
    processor = OwlViTProcessor.from_pretrained(cfg.detector_model_id)
    model = OwlViTForObjectDetection.from_pretrained(cfg.detector_model_id)
    model.to(cfg.device)
    model.eval()

    fp16 = bool(cfg.device == "cuda" and cfg.use_fp16_if_cuda)
    if fp16:
        model.half()

    prompt_texts = [spec.text for spec in cfg.prompt_specs]
    all_selected: list[dict[str, Any]] = []
    view_diagnostics: list[dict[str, Any]] = []

    for view_index, view in enumerate(views, start=1):
        x1, y1, x2, y2 = view.box_xyxy
        crop = original.crop((x1, y1, x2, y2)).convert("RGB")
        if cfg.save_detection_views:
            crop.save(view_dir / f"{view_index:02d}_{view.view_id}.png")

        logger.info(f"[VIEW] {view_index}/{len(views)} {view.view_id} box={view.box_xyxy} size={crop.size}")

        inputs = processor(text=[prompt_texts], images=crop, return_tensors="pt")
        inputs = move_batch_to_device(dict(inputs), cfg.device, fp16)

        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([crop.size[::-1]], dtype=outputs.pred_boxes.dtype, device=outputs.pred_boxes.device)
        result = processor.post_process_object_detection(outputs=outputs, threshold=0.0, target_sizes=target_sizes)[0]
        boxes = result["boxes"].detach().float().cpu().tolist()
        scores = result["scores"].detach().float().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()

        selected, diagnostics = select_view_candidates(scores, labels, boxes, crop.size[0], crop.size[1], view, cfg)
        all_selected.extend(selected)
        view_diagnostics.append(diagnostics)

        top_any = max(scores) if scores else 0.0
        logger.info(f"[VIEW] {view.view_id}: top={top_any:.5f}, seçilen={len(selected)}")

        del inputs, outputs, result
        clear_cuda()

    reduced: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in all_selected:
        grouped.setdefault(str(item["canonical_class"]), []).append(item)
    for canonical_class, group in grouped.items():
        group.sort(key=lambda item: float(item["score"]), reverse=True)
        reduced.extend(group[: cfg.max_candidates_per_class_before_nms])

    final = class_aware_nms(reduced, cfg.nms_iou_threshold, cfg.max_detections_per_class, cfg.max_total_detections)

    diagnostics_document = {
        "detector_backend": cfg.detector_backend,
        "detector_model_id": cfg.detector_model_id,
        "absolute_score_floor": cfg.absolute_score_floor,
        "relative_score_ratio": cfg.relative_score_ratio,
        "fallback_score_floor": cfg.fallback_score_floor,
        "views_count": len(views),
        "selected_before_class_limit": len(all_selected),
        "selected_before_nms": len(reduced),
        "selected_after_nms": len(final),
        "views": view_diagnostics,
    }

    logger.info(f"[DETECT] çok ölçekli aday={len(all_selected)}, NMS sonrası={len(final)}")

    del model, processor
    clear_cuda()
    return final, diagnostics_document


# =============================================================================
# SAM
# =============================================================================
def segment_boxes_with_sam(image: Image.Image, boxes: list[list[int]], cfg: Config, logger: logging.Logger) -> list[np.ndarray]:
    if not boxes:
        return []

    logger.info(f"[MODEL] SAM yükleniyor: {cfg.sam_model_id}")
    processor = SamProcessor.from_pretrained(cfg.sam_model_id)
    model = SamModel.from_pretrained(cfg.sam_model_id)
    model.to(cfg.device)
    model.eval()

    fp16 = bool(cfg.device == "cuda" and cfg.use_fp16_if_cuda)
    if fp16:
        model.half()

    base_inputs = processor(images=image, return_tensors="pt")
    pixel_values = base_inputs["pixel_values"].to(cfg.device)
    if fp16:
        pixel_values = pixel_values.half()

    logger.info("[SAM] Görüntü gömmesi hesaplanıyor")
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(pixel_values)

    original_sizes = base_inputs["original_sizes"].detach().cpu()
    reshaped_sizes = base_inputs["reshaped_input_sizes"].detach().cpu()
    masks_out: list[np.ndarray] = []

    batch_size = max(1, cfg.sam_batch_size)
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start:start + batch_size]
        logger.info(f"[SAM] Kutular {start + 1}-{start + len(batch_boxes)} / {len(boxes)}")

        prompt_inputs = processor(images=image, input_boxes=[[list(map(float, box)) for box in batch_boxes]], return_tensors="pt")
        input_boxes = prompt_inputs["input_boxes"].to(cfg.device)
        if fp16:
            input_boxes = input_boxes.half()

        with torch.inference_mode():
            outputs = model(image_embeddings=image_embeddings, input_boxes=input_boxes, multimask_output=True)

        post_masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            original_sizes,
            reshaped_sizes,
            binarize=True,
        )[0]
        iou_scores = outputs.iou_scores.detach().float().cpu()[0]

        for local_index in range(len(batch_boxes)):
            best_mask_index = int(torch.argmax(iou_scores[local_index]).item())
            selected = post_masks[local_index, best_mask_index]
            mask = np.where(selected.numpy() > 0, 255, 0).astype(np.uint8)
            masks_out.append(mask)

        del prompt_inputs, input_boxes, outputs, post_masks, iou_scores
        clear_cuda()

    del image_embeddings, pixel_values, base_inputs, model, processor
    clear_cuda()
    return masks_out


# =============================================================================
# CLEANUP / ARCHITECTURAL FILTERING
# =============================================================================
def geometry_features(record: dict[str, Any], image_width: int, image_height: int) -> dict[str, float]:
    width = max(1, float(record["width"]))
    height = max(1, float(record["height"]))
    cx, cy = record["center"]
    return {
        "aspect_w_h": width / height,
        "aspect_h_w": height / width,
        "width_ratio": width / image_width,
        "height_ratio": height / image_height,
        "center_x_ratio": float(cx) / image_width,
        "center_y_ratio": float(cy) / image_height,
        "top_ratio": float(record["bbox_xyxy"][1]) / image_height,
        "bottom_ratio": float(record["bbox_xyxy"][3]) / image_height,
        "area_ratio": float(record["mask_area_ratio"]),
    }


def passes_class_geometry_filters(record: dict[str, Any], image_width: int, image_height: int) -> tuple[bool, str]:
    cls = str(record["class"])
    feat = geometry_features(record, image_width, image_height)
    score = float(record["confidence"])
    overlap = float(record["damage_overlap"])
    quality = str(record["quality"])

    if cls not in ("arch", "pilaster", "window", "door"):
        return False, "class_not_kept"
    if overlap > 0.28:
        return False, "damage_overlap_too_high"
    if score < 0.010:
        return False, "score_too_low"
    if quality != "primary":
        return False, "fallback_candidate"

    if cls == "arch":
        if not (0.18 <= feat["width_ratio"] <= 0.55):
            return False, "arch_bad_width_ratio"
        if not (0.07 <= feat["height_ratio"] <= 0.30):
            return False, "arch_bad_height_ratio"
        if not (0.8 <= feat["aspect_w_h"] <= 5.0):
            return False, "arch_bad_aspect"
        if not (0.18 <= feat["center_y_ratio"] <= 0.78):
            return False, "arch_bad_vertical_position"
        return True, "ok"

    if cls == "pilaster":
        if not (0.012 <= feat["width_ratio"] <= 0.14):
            return False, "pilaster_bad_width_ratio"
        if not (0.12 <= feat["height_ratio"] <= 0.70):
            return False, "pilaster_bad_height_ratio"
        if not (1.7 <= feat["aspect_h_w"] <= 18.0):
            return False, "pilaster_bad_aspect"
        if feat["top_ratio"] > 0.82:
            return False, "pilaster_too_low"
        return True, "ok"

    if cls == "window":
        if not (0.03 <= feat["width_ratio"] <= 0.22):
            return False, "window_bad_width_ratio"
        if not (0.05 <= feat["height_ratio"] <= 0.35):
            return False, "window_bad_height_ratio"
        if not (0.5 <= feat["aspect_h_w"] <= 5.5):
            return False, "window_bad_aspect"
        return True, "ok"

    if cls == "door":
        if not (0.03 <= feat["width_ratio"] <= 0.25):
            return False, "door_bad_width_ratio"
        if not (0.08 <= feat["height_ratio"] <= 0.55):
            return False, "door_bad_height_ratio"
        if not (1.0 <= feat["aspect_h_w"] <= 8.0):
            return False, "door_bad_aspect"
        if feat["center_y_ratio"] < 0.35:
            return False, "door_too_high"
        return True, "ok"

    return False, "unknown_class"


def clean_architectural_records(records: list[dict[str, Any]], image_width: int, image_height: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    for record in records:
        ok, reason = passes_class_geometry_filters(record, image_width, image_height)
        record["geometry_filter_reason"] = reason
        if ok:
            clean.append(record)
        else:
            rejected.append({
                "id": record["id"],
                "class": record["class"],
                "reason": reason,
                "confidence": record["confidence"],
                "bbox_xyxy": record["bbox_xyxy"],
            })
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    by_class: dict[str, list[dict[str, Any]]] = {}
    for record in clean:
        by_class.setdefault(str(record["class"]), []).append(record)

    for cls, group in by_class.items():
        group.sort(key=lambda item: float(item["confidence"]), reverse=True)
        if cls == "arch":
            by_class[cls] = group[:8]
        elif cls == "pilaster":
            by_class[cls] = group[:10]
        elif cls == "window":
            by_class[cls] = group[:6]
        elif cls == "door":
            by_class[cls] = group[:4]

    final_clean: list[dict[str, Any]] = []
    for cls in ("arch", "pilaster", "window", "door"):
        final_clean.extend(by_class.get(cls, []))
    final_clean.sort(key=lambda item: (str(item["class"]), float(item["center"][0])))

    diagnostics = {
        "clean_count": len(final_clean),
        "rejected_count": len(rejected),
        "rejection_counts": rejection_counts,
        "rejected_examples": rejected[:25],
        "class_counts_clean": {cls: len(group) for cls, group in by_class.items()},
    }
    return final_clean, diagnostics


# =============================================================================
# SIMPLE PATTERN ANALYZER
# =============================================================================
def build_arch_rows(archs: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    if not archs:
        return []
    archs = sorted(archs, key=lambda item: float(item["center"][1]))
    rows: list[list[dict[str, Any]]] = []
    for arch in archs:
        placed = False
        for row in rows:
            mean_y = float(np.mean([float(item["center"][1]) for item in row]))
            mean_h = float(np.mean([float(item["height"]) for item in row]))
            if abs(float(arch["center"][1]) - mean_y) <= max(18.0, mean_h * cfg.arch_row_y_tolerance_ratio) and \
               abs(float(arch["height"]) - mean_h) <= max(18.0, mean_h * cfg.arch_row_height_tolerance_ratio):
                row.append(arch)
                placed = True
                break
        if not placed:
            rows.append([arch])

    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        row = sorted(row, key=lambda item: float(item["center"][0]))
        xs = [float(item["center"][0]) for item in row]
        ys = [float(item["center"][1]) for item in row]
        ws = [float(item["width"]) for item in row]
        hs = [float(item["height"]) for item in row]
        gaps = [round(xs[i + 1] - xs[i], 2) for i in range(len(xs) - 1)]
        result.append(
            {
                "row_id": f"arch_row_{index:02d}",
                "member_ids": [item["id"] for item in row],
                "count": len(row),
                "mean_center_y": round(float(np.mean(ys)), 2),
                "mean_width": round(float(np.mean(ws)), 2),
                "mean_height": round(float(np.mean(hs)), 2),
                "x_centers": [round(x, 2) for x in xs],
                "x_gaps": gaps,
            }
        )
    return result


def build_pilaster_families(pilasters: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    if not pilasters:
        return []
    pilasters = sorted(pilasters, key=lambda item: float(item["width"]))
    families: list[list[dict[str, Any]]] = []
    for p in pilasters:
        placed = False
        for fam in families:
            mean_w = float(np.mean([float(item["width"]) for item in fam]))
            mean_h = float(np.mean([float(item["height"]) for item in fam]))
            if abs(float(p["width"]) - mean_w) <= max(8.0, mean_w * cfg.pilaster_width_tolerance_ratio) and \
               abs(float(p["height"]) - mean_h) <= max(24.0, mean_h * cfg.pilaster_height_tolerance_ratio):
                fam.append(p)
                placed = True
                break
        if not placed:
            families.append([p])

    result: list[dict[str, Any]] = []
    for index, fam in enumerate(families, start=1):
        fam = sorted(fam, key=lambda item: float(item["center"][0]))
        xs = [float(item["center"][0]) for item in fam]
        result.append(
            {
                "family_id": f"pilaster_family_{index:02d}",
                "member_ids": [item["id"] for item in fam],
                "count": len(fam),
                "mean_width": round(float(np.mean([float(item["width"]) for item in fam])), 2),
                "mean_height": round(float(np.mean([float(item["height"]) for item in fam])), 2),
                "x_centers": [round(x, 2) for x in xs],
            }
        )
    return result


def infer_arch_pilaster_modules(archs: list[dict[str, Any]], pilasters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not archs or len(pilasters) < 2:
        return []
    pilasters = sorted(pilasters, key=lambda item: float(item["center"][0]))
    modules: list[dict[str, Any]] = []

    for arch in archs:
        ax = float(arch["center"][0])
        ay = float(arch["center"][1])
        a_top = float(arch["bbox_xyxy"][1])
        a_bottom = float(arch["bbox_xyxy"][3])

        left_candidates = [p for p in pilasters if float(p["center"][0]) < ax]
        right_candidates = [p for p in pilasters if float(p["center"][0]) > ax]
        if not left_candidates or not right_candidates:
            continue

        left = left_candidates[-1]
        right = right_candidates[0]
        lx = float(left["center"][0])
        rx = float(right["center"][0])
        if not (lx < ax < rx):
            continue

        # kemer ile pilasterlar arasında kabaca ilişkili bir yatay açıklık olsun
        span = rx - lx
        if span < 20:
            continue
        if not (0.6 * span <= float(arch["width"]) <= 1.6 * span):
            continue

        # pilaster yükseklikleri çok farklı olmasın
        if abs(float(left["height"]) - float(right["height"])) > max(28.0, 0.4 * max(float(left["height"]), float(right["height"]))):
            continue

        modules.append(
            {
                "module_type": "pilaster_arch_pilaster",
                "arch_id": arch["id"],
                "left_pilaster_id": left["id"],
                "right_pilaster_id": right["id"],
                "span_px": round(span, 2),
                "arch_center": [round(ax, 2), round(ay, 2)],
                "arch_top": round(a_top, 2),
                "arch_bottom": round(a_bottom, 2),
            }
        )
    return modules


def draw_clean_preview(original: np.ndarray, records: list[dict[str, Any]], output_path: Path) -> None:
    preview = original.copy()
    class_colors = {
        "arch": (60, 220, 90),
        "pilaster": (70, 180, 255),
        "window": (255, 190, 50),
        "door": (255, 120, 200),
    }
    for record in records:
        color = class_colors.get(str(record["class"]), (255, 255, 255))
        preview = draw_box_and_label(
            preview,
            list(record["bbox_xyxy"]),
            str(record["class"]),
            float(record["confidence"]),
            color,
            "",
        )
    save_rgb(output_path, preview)


def draw_grammar_preview(original: np.ndarray, clean_records: list[dict[str, Any]], arch_rows: list[dict[str, Any]], modules: list[dict[str, Any]], output_path: Path) -> None:
    preview = original.copy()
    record_map = {record["id"]: record for record in clean_records}

    # Kemer satırlarını mor çizgilerle göster
    for row in arch_rows:
        member_ids = row["member_ids"]
        if len(member_ids) >= 2:
            points = []
            for rid in member_ids:
                rec = record_map.get(rid)
                if rec:
                    points.append((int(rec["center"][0]), int(rec["center"][1])))
            if len(points) >= 2:
                cv2.polylines(preview, [np.array(points, dtype=np.int32)], False, (180, 80, 255), 2)

    # pilaster-arch-pilaster modüllerini göster
    for mod in modules:
        left = record_map.get(mod["left_pilaster_id"])
        arch = record_map.get(mod["arch_id"])
        right = record_map.get(mod["right_pilaster_id"])
        if not (left and arch and right):
            continue
        p1 = (int(left["center"][0]), int(left["center"][1]))
        p2 = (int(arch["center"][0]), int(arch["center"][1]))
        p3 = (int(right["center"][0]), int(right["center"][1]))
        cv2.line(preview, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(preview, p2, p3, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(preview, p2, 4, (255, 255, 0), -1)

    save_rgb(output_path, preview)


# =============================================================================
# MAIN
# =============================================================================
def run_stage1c(cfg: Config) -> Path:
    input_path = Path(cfg.input_photo)
    damage_path = Path(cfg.completion_mask)

    if not input_path.exists():
        raise FileNotFoundError(f"Ana fotoğraf bulunamadı: {input_path.resolve()}")
    if not damage_path.exists():
        raise FileNotFoundError(
            f"Hasar maskesi bulunamadı: {damage_path.resolve()}\n"
            "Beyaz=hasarlı/tamamlanacak, siyah=korunacak alan."
        )

    run_id, run_dir = make_run_dir(cfg)
    element_mask_dir = ensure_dir(run_dir / "element_masks")
    view_dir = ensure_dir(run_dir / "detection_views")
    logger = setup_logger(run_dir, run_id)
    run_meta_path = run_dir / "run_meta.json"
    elements_path = run_dir / "elements.json"
    diagnostics_path = run_dir / "detection_diagnostics.json"
    cleanup_diag_path = run_dir / "cleanup_diagnostics.json"
    pattern_groups_path = run_dir / "pattern_groups.json"
    started = time.perf_counter()

    source_path = Path(__file__)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage1c_cleanup_and_pattern_prep",
        "status": "running",
        "started_at": now_iso(),
        "source_script": {
            "filename": source_path.name,
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        },
        "environment": collect_environment(),
        "config": asdict(cfg),
        "input": {},
        "detection_views": [],
        "detections": [],
        "clean_detections": [],
        "summary": {},
        "outputs": [],
    }

    def flush_meta() -> None:
        write_json(run_meta_path, metadata)

    try:
        logger.info("=" * 72)
        logger.info(f"[RUN] {run_id}")
        logger.info("[STAGE] V8 Stage 1C — Çok ölçekli tespit + temizleme")
        logger.info(f"[ENV] transformers={metadata['environment']['transformers']}")
        logger.info(f"[DEVICE] {cfg.device}")
        logger.info("=" * 72)

        original = read_rgb(input_path)
        height, width = original.shape[:2]
        damage_mask = load_binary_mask(damage_path, (width, height))
        damage_area = mask_ratio(damage_mask)
        if damage_area < 0.0005:
            raise ValueError("completion_mask.png neredeyse boş")

        image = Image.fromarray(original, mode="RGB")
        save_rgb(run_dir / "input.png", original)
        Image.fromarray(damage_mask, mode="L").save(run_dir / "completion_mask.png")
        save_rgb(run_dir / "completion_mask_preview.png", overlay_mask(original, damage_mask, color=(255, 30, 30), alpha=0.42))

        views = build_detection_views(width, height, damage_mask, cfg)
        save_views_preview(original, views, run_dir / "detection_views_preview.png")
        metadata["input"] = {
            "photo": {
                "path": str(input_path.resolve()),
                "filename": input_path.name,
                "sha256": sha256_file(input_path),
                "size": [width, height],
                "run_copy": "input.png",
            },
            "completion_mask": {
                "path": str(damage_path.resolve()),
                "filename": damage_path.name,
                "sha256": sha256_file(damage_path),
                "area_ratio": round(damage_area, 6),
                "run_copy": "completion_mask.png",
            },
        }
        metadata["detection_views"] = [{"view_id": v.view_id, "box_xyxy": list(v.box_xyxy), "reason": v.reason} for v in views]
        flush_meta()

        raw_detections, diagnostics = detect_multiscale_with_owlvit(image, views, cfg, logger, view_dir)
        write_json(diagnostics_path, diagnostics)

        boxes = [clamp_box_xyxy(item["box"], width, height) for item in raw_detections]
        element_masks = segment_boxes_with_sam(image, boxes, cfg, logger)
        if len(element_masks) != len(raw_detections):
            raise RuntimeError(f"SAM maske sayısı uyumsuz: {len(element_masks)} != {len(raw_detections)}")

        detection_candidates = original.copy()
        all_visual = original.copy()
        filtered_visual = overlay_mask(original, damage_mask, color=(255, 30, 30), alpha=0.20)

        for det, box in zip(raw_detections, boxes):
            color = (60, 220, 90) if det["quality"] == "primary" else (255, 190, 50)
            detection_candidates = draw_box_and_label(detection_candidates, box, str(det["canonical_class"]), float(det["score"]), color)
        save_rgb(run_dir / "detection_candidates.png", detection_candidates)

        class_counts_all: dict[str, int] = {}
        class_counts_usable: dict[str, int] = {}
        usable_count = 0
        pattern_eligible_count = 0
        skipped_small = 0

        for index, (det, box, element_mask) in enumerate(zip(raw_detections, boxes, element_masks), start=1):
            area = mask_ratio(element_mask)
            if area < cfg.min_element_area_ratio:
                skipped_small += 1
                logger.info(f"[SKIP] {det['canonical_class']} #{index}: maske çok küçük ({area:.6f})")
                continue

            overlap = damage_overlap_ratio(element_mask, damage_mask)
            usable = overlap <= cfg.max_damage_overlap_for_usable
            pattern_eligible = usable and float(det["score"]) >= cfg.pattern_score_floor and str(det["quality"]) == "primary"

            cls = str(det["canonical_class"])
            class_counts_all[cls] = class_counts_all.get(cls, 0) + 1
            if usable:
                usable_count += 1
                class_counts_usable[cls] = class_counts_usable.get(cls, 0) + 1
            if pattern_eligible:
                pattern_eligible_count += 1

            element_id = f"{slug(cls)}_{index:03d}"
            mask_filename = f"{element_id}.png"
            Image.fromarray(element_mask, mode="L").save(element_mask_dir / mask_filename)

            x1, y1, x2, y2 = box
            record = {
                "id": element_id,
                "class": cls,
                "matched_prompt": str(det["label"]),
                "prompt_index": int(det["prompt_index"]),
                "confidence": round(float(det["score"]), 6),
                "quality": str(det["quality"]),
                "source_view": str(det["view_id"]),
                "source_view_reason": str(det["view_reason"]),
                "bbox_xyxy": box,
                "center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "mask_area_ratio": round(area, 6),
                "damage_overlap": round(overlap, 6),
                "usable_as_reference": usable,
                "eligible_for_pattern_analysis": pattern_eligible,
                "mask_file": str(Path("element_masks") / mask_filename),
            }
            metadata["detections"].append(record)

            if not usable:
                color = (255, 90, 90)
                suffix = " [HASAR]"
            elif str(det["quality"]) != "primary":
                color = (255, 190, 50)
                suffix = " [DUSUK]"
            else:
                color = (60, 220, 90)
                suffix = ""

            all_visual = draw_box_and_label(all_visual, box, cls, float(det["score"]), color, suffix)
            filtered_visual = overlay_mask(filtered_visual, element_mask, color=color, alpha=0.25 if usable else 0.20)
            filtered_visual = draw_box_and_label(filtered_visual, box, cls, float(det["score"]), color, suffix)

        save_rgb(run_dir / "detected_elements.png", all_visual)
        save_rgb(run_dir / "detected_elements_filtered.png", filtered_visual)

        # Stage 1C temizleme
        clean_records, cleanup_diag = clean_architectural_records(metadata["detections"], width, height)
        metadata["clean_detections"] = clean_records
        write_json(cleanup_diag_path, cleanup_diag)

        clean_archs = [rec for rec in clean_records if rec["class"] == "arch"]
        clean_pilasters = [rec for rec in clean_records if rec["class"] == "pilaster"]
        clean_windows = [rec for rec in clean_records if rec["class"] == "window"]
        clean_doors = [rec for rec in clean_records if rec["class"] == "door"]

        arch_rows = build_arch_rows(clean_archs, cfg)
        pilaster_families = build_pilaster_families(clean_pilasters, cfg)
        modules = infer_arch_pilaster_modules(clean_archs, clean_pilasters)

        pattern_groups = {
            "summary": {
                "clean_elements": len(clean_records),
                "arches": len(clean_archs),
                "pilasters": len(clean_pilasters),
                "windows": len(clean_windows),
                "doors": len(clean_doors),
                "arch_rows": len(arch_rows),
                "pilaster_families": len(pilaster_families),
                "pilaster_arch_pilaster_modules": len(modules),
            },
            "arch_rows": arch_rows,
            "pilaster_families": pilaster_families,
            "pilaster_arch_pilaster_modules": modules,
        }
        write_json(pattern_groups_path, pattern_groups)

        draw_clean_preview(original, clean_records, run_dir / "clean_detected_elements.png")
        draw_grammar_preview(original, clean_records, arch_rows, modules, run_dir / "grammar_preview.png")

        elements_document = {
            "run_id": run_id,
            "image_size": [width, height],
            "completion_mask_area_ratio": round(damage_area, 6),
            "all_detections_summary": {
                "saved_elements": len(metadata["detections"]),
                "usable_reference_elements": usable_count,
                "eligible_for_pattern_analysis": pattern_eligible_count,
                "class_counts_all": class_counts_all,
                "class_counts_usable": class_counts_usable,
            },
            "clean_summary": cleanup_diag,
            "elements": clean_records,
        }
        write_json(elements_path, elements_document)

        metadata["summary"] = {
            "detector_backend": cfg.detector_backend,
            "detector_model_id": cfg.detector_model_id,
            "views_count": len(views),
            "after_multiscale_nms": len(raw_detections),
            "saved_elements": len(metadata["detections"]),
            "usable_reference_elements": usable_count,
            "eligible_for_pattern_analysis": pattern_eligible_count,
            "clean_detected_elements": len(clean_records),
            "clean_arches": len(clean_archs),
            "clean_pilasters": len(clean_pilasters),
            "clean_windows": len(clean_windows),
            "clean_doors": len(clean_doors),
            "arch_rows": len(arch_rows),
            "pilaster_families": len(pilaster_families),
            "pilaster_arch_pilaster_modules": len(modules),
            "skipped_small_masks": skipped_small,
            "class_counts_all": class_counts_all,
            "class_counts_usable": class_counts_usable,
            "damage_mask_area_ratio": round(damage_area, 6),
            "note": (
                "Bu aşama restorasyon üretmez. clean_detected_elements.png ve pattern_groups.json, "
                "bir sonraki gramer ve aday inşa aşaması için temizlenmiş veriyi sağlar."
            ),
        }

        metadata["outputs"] = [
            {"file": "input.png", "type": "input_copy"},
            {"file": "completion_mask.png", "type": "damage_mask_copy"},
            {"file": "completion_mask_preview.png", "type": "damage_mask_preview"},
            {"file": "detection_views_preview.png", "type": "multiscale_view_layout"},
            {"file": "detection_views/", "type": "individual_detection_crops"},
            {"file": "detection_diagnostics.json", "type": "per_view_per_prompt_scores"},
            {"file": "detection_candidates.png", "type": "boxes_before_sam_filtering"},
            {"file": "detected_elements.png", "type": "all_boxes_after_sam"},
            {"file": "detected_elements_filtered.png", "type": "all_boxes_with_damage_context"},
            {"file": "cleanup_diagnostics.json", "type": "cleanup_filter_report"},
            {"file": "clean_detected_elements.png", "type": "cleaned_elements_overlay"},
            {"file": "grammar_preview.png", "type": "simple_pattern_preview"},
            {"file": "pattern_groups.json", "type": "simple_architecture_pattern_groups"},
            {"file": "elements.json", "type": "clean_element_records"},
            {"file": "element_masks/", "type": "per_element_masks"},
        ]

        metadata["status"] = "completed" if clean_records else "completed_no_clean_elements"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush_meta()

        logger.info("[OK] Stage 1C tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            f"[COUNT] kaydedilen={len(metadata['detections'])}, temiz={len(clean_records)}, "
            f"arch={len(clean_archs)}, pilaster={len(clean_pilasters)}, modül={len(modules)}"
        )
        logger.info("[FILES] clean_detected_elements.png, grammar_preview.png, pattern_groups.json")
        return run_dir

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        metadata["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        flush_meta()
        logger.error(f"[FAIL] {exc}")
        logger.error(traceback.format_exc())
        raise
    finally:
        clear_cuda()


def main() -> None:
    cfg = Config()
    run_dir = run_stage1c(cfg)
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()