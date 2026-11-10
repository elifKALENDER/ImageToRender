#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage1_multiscale.py
=================================================
V8 / STAGE 1B — Çok ölçekli mimari eleman tespiti

Amaç
-----
Kullanıcı yalnızca:
    image.png
    completion_mask.png
verir.

Sistem:
1. Fotoğrafı tam görüntü ve yakın plan örtüşen parçalarda tarar.
2. OWL-ViT ile mimari eleman adaylarını zero-shot olarak bulur.
3. Düşük OWL-ViT puanlarını mutlak + göreli eşikle değerlendirir.
4. Kırpma koordinatlarını ana fotoğrafa geri taşır.
5. Aynı sınıftaki tekrar kutularını NMS ile birleştirir.
6. SAM ile seçilen kutuların piksel maskelerini çıkarır.
7. Hasar maskesiyle çakışmayı hesaplar.
8. Teşhis raporları ve görsel çıktılar üretir.

Bu aşamada Stable Diffusion, ControlNet veya restorasyon üretimi YOKTUR.

Mevcut ortamı korur:
- torch 2.4.x
- transformers 4.33.x
- OWL-ViT + SAM
- diffusers yükseltmesi gerekmez

Çıktı klasörü adı değişmez:
    outputs/restore_YYYYMMDD_HHMMSS
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

# Hugging Face ayarları transformers importundan önce olmalı.
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
from transformers import (
    OwlViTForObjectDetection,
    OwlViTProcessor,
    SamModel,
    SamProcessor,
)

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

    detector_backend: str = "owlvit_multiscale"
    detector_model_id: str = "google/owlvit-base-patch32"

    # Basit ve yaygın İngilizce sözcükler OWL-ViT'te uzun teknik istemlerden
    # daha iyi sonuç verebilir. Aynı mimari sınıfa ait eş anlamlılar daha sonra
    # canonical_class altında birleştirilir.
    prompt_specs: list[PromptSpec] = field(
        default_factory=lambda: [
            PromptSpec("arch", "arch"),
            PromptSpec("stone arch", "arch"),
            PromptSpec("arched opening", "arch"),
            PromptSpec("arcade", "arch"),
            PromptSpec("pilaster", "pilaster"),
            PromptSpec("wall pillar", "pilaster"),
            PromptSpec("rectangular column", "pilaster"),
            PromptSpec("column", "column"),
            PromptSpec("stone column", "column"),
            PromptSpec("column capital", "capital"),
            PromptSpec("column base", "base"),
            PromptSpec("window", "window"),
            PromptSpec("window opening", "window"),
            PromptSpec("door", "door"),
            PromptSpec("doorway", "door"),
            PromptSpec("niche", "niche"),
            PromptSpec("cornice", "cornice"),
            PromptSpec("wall molding", "molding"),
            PromptSpec("wall panel", "wall_panel"),
            PromptSpec("stone wall", "wall"),
            PromptSpec("buttress", "buttress"),
        ]
    )

    # Çok ölçekli tarama.
    use_full_image_view: bool = True
    use_damage_context_view: bool = True
    damage_context_expand: float = 2.40
    damage_context_min_fraction: float = 0.48
    tile_fraction_of_short_side: float = 0.68
    tile_overlap: float = 0.38
    max_detection_views: int = 10
    save_detection_views: bool = True

    # OWL-ViT puanları genellikle düşük olabilir.
    # Her istem için:
    #   seçim eşiği = max(absolute_score_floor, en_iyi_puan * relative_score_ratio)
    absolute_score_floor: float = 0.008
    relative_score_ratio: float = 0.58
    fallback_score_floor: float = 0.0035
    max_candidates_per_prompt_per_view: int = 3
    max_candidates_per_class_before_nms: int = 24

    # Birleştirme.
    nms_iou_threshold: float = 0.42
    max_detections_per_class: int = 10
    max_total_detections: int = 48

    # Aşırı küçük/büyük kutuları süz.
    min_box_area_ratio_in_view: float = 0.0010
    max_box_area_ratio_in_view: float = 0.82

    # SAM
    sam_model_id: str = "facebook/sam-vit-base"
    sam_batch_size: int = 6

    # Hasar ve kalite.
    max_damage_overlap_for_usable: float = 0.15
    min_element_area_ratio: float = 0.00010
    pattern_score_floor: float = 0.012

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16_if_cuda: bool = True


@dataclass(frozen=True)
class DetectionView:
    view_id: str
    box_xyxy: tuple[int, int, int, int]
    reason: str


# =============================================================================
# GENERAL UTILITIES
# =============================================================================
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(cfg: Config) -> tuple[str, Path]:
    run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
    return run_id, ensure_dir(Path(cfg.output_root) / run_id)


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"v8_stage1_multiscale_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
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
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda_runtime"] = torch.version.cuda
    return env


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    mask = Image.open(path).convert("L").resize(
        (width, height), Image.Resampling.NEAREST
    )
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


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    binary = (mask > 0).astype(np.float32)[..., None]
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    output = output * (1.0 - binary * alpha) + tint * (binary * alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


def slug(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return clean or "element"


def clamp_box_xyxy(
    box: Iterable[float],
    width: int,
    height: int,
) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in box]
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
    x1, y1, x2, y2 = [float(value) for value in box]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, float(width * height)))


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def move_batch_to_device(
    batch: dict[str, Any],
    device: str,
    fp16: bool,
) -> dict[str, Any]:
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


def deduplicate_views(
    views: list[DetectionView],
    iou_threshold: float = 0.92,
) -> list[DetectionView]:
    kept: list[DetectionView] = []
    for view in views:
        if any(box_iou(view.box_xyxy, old.box_xyxy) >= iou_threshold for old in kept):
            continue
        kept.append(view)
    return kept


def build_detection_views(
    image_width: int,
    image_height: int,
    damage_mask: np.ndarray,
    cfg: Config,
) -> list[DetectionView]:
    views: list[DetectionView] = []

    if cfg.use_full_image_view:
        views.append(
            DetectionView(
                view_id="full",
                box_xyxy=(0, 0, image_width, image_height),
                reason="full_image",
            )
        )

    damage_box = bbox_from_binary(damage_mask)
    if cfg.use_damage_context_view and damage_box is not None:
        minimum_side = int(
            round(min(image_width, image_height) * cfg.damage_context_min_fraction)
        )
        context_box = expanded_square_crop(
            damage_box,
            image_width,
            image_height,
            cfg.damage_context_expand,
            minimum_side,
        )
        views.append(
            DetectionView(
                view_id="damage_context",
                box_xyxy=context_box,
                reason="expanded_damage_context",
            )
        )

    short_side = min(image_width, image_height)
    tile_side = int(round(short_side * cfg.tile_fraction_of_short_side))
    tile_side = max(192, min(tile_side, image_width, image_height))

    x_positions = evenly_spaced_positions(
        image_width, tile_side, cfg.tile_overlap
    )
    y_positions = evenly_spaced_positions(
        image_height, tile_side, cfg.tile_overlap
    )

    tile_index = 0
    for y in y_positions:
        for x in x_positions:
            tile_index += 1
            views.append(
                DetectionView(
                    view_id=f"tile_{tile_index:02d}",
                    box_xyxy=(x, y, x + tile_side, y + tile_side),
                    reason="overlapping_square_tile",
                )
            )

    views = deduplicate_views(views)
    return views[: max(1, cfg.max_detection_views)]


def save_views_preview(
    original: np.ndarray,
    views: list[DetectionView],
    output_path: Path,
) -> None:
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
def class_aware_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float,
    max_per_class: int,
    max_total: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        grouped.setdefault(str(detection["canonical_class"]), []).append(detection)

    kept_all: list[dict[str, Any]] = []
    for canonical_class, group in grouped.items():
        group = sorted(group, key=lambda item: float(item["score"]), reverse=True)
        kept_group: list[dict[str, Any]] = []
        for candidate in group:
            if all(
                box_iou(candidate["box"], accepted["box"]) < iou_threshold
                for accepted in kept_group
            ):
                kept_group.append(candidate)
            if len(kept_group) >= max_per_class:
                break
        kept_all.extend(kept_group)

    kept_all.sort(key=lambda item: float(item["score"]), reverse=True)
    return kept_all[:max_total]


# =============================================================================
# VISUALIZATION
# =============================================================================
def draw_box_and_label(
    rgb: np.ndarray,
    box: list[int],
    label: str,
    score: float,
    usable: bool,
    quality: str,
) -> np.ndarray:
    output = rgb.copy()
    x1, y1, x2, y2 = box

    if not usable:
        color = (255, 90, 90)
    elif quality == "primary":
        color = (60, 220, 90)
    else:
        color = (255, 190, 50)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    suffix = ""
    if not usable:
        suffix = " [HASAR]"
    elif quality != "primary":
        suffix = " [DUSUK]"

    text = f"{label} {score:.3f}{suffix}"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1
    )
    label_top = max(0, y1 - text_height - 8)
    cv2.rectangle(
        output,
        (x1, label_top),
        (min(output.shape[1] - 1, x1 + text_width + 6), y1),
        color,
        -1,
    )
    cv2.putText(
        output,
        text,
        (x1 + 3, max(text_height + 1, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return output


# =============================================================================
# OWL-ViT MULTISCALE DETECTION
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
    by_prompt: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(cfg.prompt_specs))
    }

    for score, label_index, local_box in zip(scores, labels, boxes):
        label_index = int(label_index)
        if label_index < 0 or label_index >= len(cfg.prompt_specs):
            continue

        area_ratio = box_area_ratio(local_box, crop_width, crop_height)
        if area_ratio < cfg.min_box_area_ratio_in_view:
            continue
        if area_ratio > cfg.max_box_area_ratio_in_view:
            continue

        by_prompt[label_index].append(
            {
                "score": float(score),
                "local_box": [float(value) for value in local_box],
                "area_ratio_in_view": float(area_ratio),
            }
        )

    selected: list[dict[str, Any]] = []
    prompt_diagnostics: list[dict[str, Any]] = []

    vx1, vy1, _, _ = view.box_xyxy

    for prompt_index, spec in enumerate(cfg.prompt_specs):
        candidates = sorted(
            by_prompt[prompt_index],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        top_scores = [round(float(item["score"]), 6) for item in candidates[:5]]
        best_score = float(candidates[0]["score"]) if candidates else 0.0
        dynamic_threshold = max(
            cfg.absolute_score_floor,
            best_score * cfg.relative_score_ratio,
        )

        primary = [
            item for item in candidates
            if float(item["score"]) >= dynamic_threshold
        ][: cfg.max_candidates_per_prompt_per_view]

        quality = "primary"
        chosen = primary

        # Bu istem için birincil aday yoksa ama model tamamen sıfır da değilse,
        # yalnız en iyi kutuyu düşük güvenli teşhis adayı olarak tut.
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
            global_box = [
                float(lx1 + vx1),
                float(ly1 + vy1),
                float(lx2 + vx1),
                float(ly2 + vy1),
            ]
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

        logger.info(
            f"[VIEW] {view_index}/{len(views)} {view.view_id} "
            f"box={view.box_xyxy} size={crop.size}"
        )

        inputs = processor(
            text=[prompt_texts],
            images=crop,
            return_tensors="pt",
        )
        inputs = move_batch_to_device(dict(inputs), cfg.device, fp16)

        with torch.inference_mode():
            outputs = model(**inputs)

        # transformers 4.33.x device uyumluluğu.
        target_sizes = torch.tensor(
            [crop.size[::-1]],
            dtype=outputs.pred_boxes.dtype,
            device=outputs.pred_boxes.device,
        )
        result = processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target_sizes,
        )[0]

        boxes = result["boxes"].detach().float().cpu().tolist()
        scores = result["scores"].detach().float().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()

        selected, diagnostics = select_view_candidates(
            scores=scores,
            labels=labels,
            boxes=boxes,
            crop_width=crop.size[0],
            crop_height=crop.size[1],
            view=view,
            cfg=cfg,
        )
        all_selected.extend(selected)
        view_diagnostics.append(diagnostics)

        top_any = max(scores) if scores else 0.0
        logger.info(
            f"[VIEW] {view.view_id}: top={top_any:.5f}, "
            f"seçilen={len(selected)}"
        )

        del inputs, outputs, result
        clear_cuda()

    # Her sınıfta çok fazla tile tekrarı oluşmadan önce kaba sınır.
    reduced_before_nms: list[dict[str, Any]] = []
    class_groups: dict[str, list[dict[str, Any]]] = {}
    for item in all_selected:
        class_groups.setdefault(str(item["canonical_class"]), []).append(item)
    for canonical_class, group in class_groups.items():
        group.sort(key=lambda item: float(item["score"]), reverse=True)
        reduced_before_nms.extend(
            group[: cfg.max_candidates_per_class_before_nms]
        )

    final = class_aware_nms(
        reduced_before_nms,
        iou_threshold=cfg.nms_iou_threshold,
        max_per_class=cfg.max_detections_per_class,
        max_total=cfg.max_total_detections,
    )

    diagnostics_document = {
        "detector_backend": cfg.detector_backend,
        "detector_model_id": cfg.detector_model_id,
        "absolute_score_floor": cfg.absolute_score_floor,
        "relative_score_ratio": cfg.relative_score_ratio,
        "fallback_score_floor": cfg.fallback_score_floor,
        "views_count": len(views),
        "selected_before_class_limit": len(all_selected),
        "selected_before_nms": len(reduced_before_nms),
        "selected_after_nms": len(final),
        "views": view_diagnostics,
    }

    logger.info(
        f"[DETECT] çok ölçekli aday={len(all_selected)}, "
        f"NMS sonrası={len(final)}"
    )

    del model, processor
    clear_cuda()
    return final, diagnostics_document


# =============================================================================
# SAM SEGMENTATION
# =============================================================================
def segment_boxes_with_sam(
    image: Image.Image,
    boxes: list[list[int]],
    cfg: Config,
    logger: logging.Logger,
) -> list[np.ndarray]:
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
        batch_boxes = boxes[start : start + batch_size]
        logger.info(
            f"[SAM] Kutular {start + 1}-{start + len(batch_boxes)} / {len(boxes)}"
        )

        prompt_inputs = processor(
            images=image,
            input_boxes=[[list(map(float, box)) for box in batch_boxes]],
            return_tensors="pt",
        )
        input_boxes = prompt_inputs["input_boxes"].to(cfg.device)
        if fp16:
            input_boxes = input_boxes.half()

        with torch.inference_mode():
            sam_outputs = model(
                image_embeddings=image_embeddings,
                input_boxes=input_boxes,
                multimask_output=True,
            )

        post_masks = processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().float().cpu(),
            original_sizes,
            reshaped_sizes,
            binarize=True,
        )[0]
        iou_scores = sam_outputs.iou_scores.detach().float().cpu()[0]

        for local_index in range(len(batch_boxes)):
            best_mask_index = int(torch.argmax(iou_scores[local_index]).item())
            selected = post_masks[local_index, best_mask_index]
            mask = np.where(selected.numpy() > 0, 255, 0).astype(np.uint8)
            masks_out.append(mask)

        del prompt_inputs, input_boxes, sam_outputs, post_masks, iou_scores
        clear_cuda()

    del image_embeddings, pixel_values, base_inputs, model, processor
    clear_cuda()
    return masks_out


# =============================================================================
# MAIN
# =============================================================================
def run_stage1(cfg: Config) -> Path:
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
    meta_path = run_dir / "run_meta.json"
    elements_path = run_dir / "elements.json"
    diagnostics_path = run_dir / "detection_diagnostics.json"
    started = time.perf_counter()

    source_path = Path(__file__)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage1b_multiscale_architectural_detection",
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
        "summary": {},
        "outputs": [],
    }

    def flush_metadata() -> None:
        write_json(meta_path, metadata)

    try:
        logger.info("=" * 72)
        logger.info(f"[RUN] {run_id}")
        logger.info("[STAGE] V8 Stage 1B — Çok ölçekli OWL-ViT + SAM")
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
        save_rgb(
            run_dir / "completion_mask_preview.png",
            overlay_mask(original, damage_mask, color=(255, 30, 30), alpha=0.42),
        )

        views = build_detection_views(width, height, damage_mask, cfg)
        save_views_preview(
            original,
            views,
            run_dir / "detection_views_preview.png",
        )

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
        metadata["detection_views"] = [
            {
                "view_id": view.view_id,
                "box_xyxy": list(view.box_xyxy),
                "reason": view.reason,
            }
            for view in views
        ]
        flush_metadata()

        raw_detections, diagnostics = detect_multiscale_with_owlvit(
            image,
            views,
            cfg,
            logger,
            view_dir,
        )
        write_json(diagnostics_path, diagnostics)

        boxes = [
            clamp_box_xyxy(item["box"], width, height)
            for item in raw_detections
        ]

        # SAM yalnız NMS sonrası kutular için yüklenir.
        element_masks = segment_boxes_with_sam(image, boxes, cfg, logger)
        if len(element_masks) != len(raw_detections):
            raise RuntimeError(
                f"SAM maske sayısı uyumsuz: "
                f"{len(element_masks)} != {len(raw_detections)}"
            )

        candidate_visual = original.copy()
        all_visual = original.copy()
        filtered_visual = overlay_mask(
            original, damage_mask, color=(255, 30, 30), alpha=0.20
        )

        # SAM öncesi aday kutuları da ayrıca göster.
        for detection, box in zip(raw_detections, boxes):
            candidate_visual = draw_box_and_label(
                candidate_visual,
                box,
                str(detection["canonical_class"]),
                float(detection["score"]),
                usable=True,
                quality=str(detection["quality"]),
            )
        save_rgb(run_dir / "detection_candidates.png", candidate_visual)

        class_counts_all: dict[str, int] = {}
        class_counts_usable: dict[str, int] = {}
        usable_count = 0
        pattern_eligible_count = 0
        skipped_small = 0

        for index, (detection, box, element_mask) in enumerate(
            zip(raw_detections, boxes, element_masks),
            start=1,
        ):
            canonical_class = str(detection["canonical_class"])
            prompt_text = str(detection["label"])
            score = float(detection["score"])
            quality = str(detection["quality"])
            area = mask_ratio(element_mask)

            if area < cfg.min_element_area_ratio:
                skipped_small += 1
                logger.info(
                    f"[SKIP] {canonical_class} #{index}: "
                    f"maske çok küçük ({area:.6f})"
                )
                continue

            overlap = damage_overlap_ratio(element_mask, damage_mask)
            usable = overlap <= cfg.max_damage_overlap_for_usable
            pattern_eligible = (
                usable
                and score >= cfg.pattern_score_floor
                and quality == "primary"
            )

            class_counts_all[canonical_class] = (
                class_counts_all.get(canonical_class, 0) + 1
            )
            if usable:
                usable_count += 1
                class_counts_usable[canonical_class] = (
                    class_counts_usable.get(canonical_class, 0) + 1
                )
            if pattern_eligible:
                pattern_eligible_count += 1

            element_id = f"{slug(canonical_class)}_{index:03d}"
            mask_filename = f"{element_id}.png"
            Image.fromarray(element_mask, mode="L").save(
                element_mask_dir / mask_filename
            )

            x1, y1, x2, y2 = box
            record = {
                "id": element_id,
                "class": canonical_class,
                "matched_prompt": prompt_text,
                "prompt_index": int(detection["prompt_index"]),
                "confidence": round(score, 6),
                "quality": quality,
                "source_view": detection["view_id"],
                "source_view_reason": detection["view_reason"],
                "bbox_xyxy": box,
                "center": [
                    round((x1 + x2) / 2.0, 2),
                    round((y1 + y2) / 2.0, 2),
                ],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "mask_area_ratio": round(area, 6),
                "damage_overlap": round(overlap, 6),
                "usable_as_reference": usable,
                "eligible_for_pattern_analysis": pattern_eligible,
                "mask_file": str(Path("element_masks") / mask_filename),
            }
            metadata["detections"].append(record)

            all_visual = draw_box_and_label(
                all_visual,
                box,
                canonical_class,
                score,
                usable,
                quality,
            )

            if not usable:
                tint = (255, 100, 100)
            elif pattern_eligible:
                tint = (60, 220, 90)
            else:
                tint = (255, 190, 50)

            filtered_visual = overlay_mask(
                filtered_visual,
                element_mask,
                color=tint,
                alpha=0.25,
            )
            filtered_visual = draw_box_and_label(
                filtered_visual,
                box,
                canonical_class,
                score,
                usable,
                quality,
            )

        save_rgb(run_dir / "detected_elements.png", all_visual)
        save_rgb(run_dir / "detected_elements_filtered.png", filtered_visual)

        metadata["summary"] = {
            "detector_backend": cfg.detector_backend,
            "detector_model_id": cfg.detector_model_id,
            "views_count": len(views),
            "after_multiscale_nms": len(raw_detections),
            "saved_elements": len(metadata["detections"]),
            "usable_reference_elements": usable_count,
            "eligible_for_pattern_analysis": pattern_eligible_count,
            "skipped_small_masks": skipped_small,
            "class_counts_all": class_counts_all,
            "class_counts_usable": class_counts_usable,
            "damage_mask_area_ratio": round(damage_area, 6),
            "note": (
                "Turuncu kutular düşük güvenli teşhis adaylarıdır. "
                "Yeşil kutular birincil ve hasarla az çakışan adaylardır. "
                "Kırmızı kutular hasar alanıyla fazla çakışır."
            ),
        }

        elements_document = {
            "run_id": run_id,
            "detector": {
                "backend": cfg.detector_backend,
                "model_id": cfg.detector_model_id,
                "prompt_specs": [asdict(spec) for spec in cfg.prompt_specs],
            },
            "image_size": [width, height],
            "completion_mask_area_ratio": round(damage_area, 6),
            "summary": metadata["summary"],
            "elements": metadata["detections"],
        }
        write_json(elements_path, elements_document)

        metadata["outputs"] = [
            {"file": "input.png", "type": "input_copy"},
            {"file": "completion_mask.png", "type": "damage_mask_copy"},
            {
                "file": "completion_mask_preview.png",
                "type": "damage_mask_preview",
            },
            {
                "file": "detection_views_preview.png",
                "type": "multiscale_view_layout",
            },
            {
                "file": "detection_views/",
                "type": "individual_detection_crops",
            },
            {
                "file": "detection_diagnostics.json",
                "type": "per_view_per_prompt_scores",
            },
            {
                "file": "detection_candidates.png",
                "type": "boxes_before_sam_filtering",
            },
            {"file": "detected_elements.png", "type": "final_boxes"},
            {
                "file": "detected_elements_filtered.png",
                "type": "final_masks_boxes_damage_context",
            },
            {"file": "elements.json", "type": "element_records"},
            {"file": "element_masks/", "type": "per_element_masks"},
        ]

        if len(metadata["detections"]) == 0:
            metadata["status"] = "completed_no_detections"
            logger.warning(
                "[RESULT] Yine eleman bulunamadı. "
                "detection_diagnostics.json içindeki en yüksek puanlara bak."
            )
        else:
            metadata["status"] = "completed"

        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush_metadata()

        logger.info("[OK] Stage 1B tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            f"[COUNT] kaydedilen={len(metadata['detections'])}, "
            f"kullanılabilir={usable_count}, "
            f"örüntü_adayı={pattern_eligible_count}"
        )
        logger.info(
            "[FILES] detection_candidates.png, detected_elements.png, "
            "detected_elements_filtered.png, detection_diagnostics.json"
        )
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
        flush_metadata()
        logger.error(f"[FAIL] {exc}")
        logger.error(traceback.format_exc())
        raise
    finally:
        clear_cuda()


def main() -> None:
    cfg = Config()
    run_dir = run_stage1(cfg)
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
