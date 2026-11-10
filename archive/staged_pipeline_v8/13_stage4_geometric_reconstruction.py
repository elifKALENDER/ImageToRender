#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ImageToRender2Restoration_v8_stage4_geometric_reconstruction.py
================================================================
V8 / STAGE 4 — Uzman onaylı hipotezlerden geometrik rekonstrüksiyon planı

AMAÇ
----
Stage 3C-C tarafından kilitlenmiş uzman onaylı rekonstrüksiyon hipotezlerini
okur ve her hipotez için:

- aynı sınıftaki gözlenen elemanlar arasından donör/prototip adaylarını sıralar,
- uzman tarafından onaylanmış hedef kutuyu otorite kabul eder,
- sınıfa uygun parametrik geometri üretir,
- tam geometri maskesini ve yalnız completion_mask içinde kalan müdahale
  maskesini ayrı ayrı kaydeder,
- her piksel/eleman için kaynak ve dönüşüm izini JSON olarak yazar.

Bu aşama:
- gözlenen uzman envanterini değiştirmez,
- Stage 3C-C kararlarını değiştirmez,
- yeni hipotez üretmez,
- Stable Diffusion / inpaint kullanmaz,
- taş dokusu veya ışık üretmez,
- yalnız GEOMETRİ + DONÖR PLANI üretir.

GİRDİLER (Stage 3C-C kilitli run)
---------------------------------
input.png
completion_mask.png
source_expert_validated_inventory.json
expert_approved_reconstruction_hypotheses.json

Varsa ek bağlam:
source_stage3c_b_facade_sectors.json
source_stage3c_b_support_axis_sequence.json
source_stage3c_b_ordered_bay_sequence.json
source_stage3c_b_arch_role_subfamilies.json
source_stage3c_b_pilaster_role_subfamilies.json
expert_hypothesis_review_actions.json
stage3c_c_report.txt

ÇIKTILAR
--------
Her çalıştırmada yeni klasör:
    outputs/restore_YYYYMMDD_HHMMSS

source_* girdiler
stage4_geometric_reconstruction_plan.json
stage4_donor_selection.json
stage4_source_map.json
stage4_geometry_full_mask.png
stage4_geometry_damage_limited_mask.png
stage4_geometric_reconstruction_outline.png
stage4_geometric_reconstruction_overlay.png
stage4_donor_selection_overlay.png
stage4_source_map.png
stage4_donor_reference_sheet.png
reconstruction_element_masks/
stage4_report.txt
run.log
run_meta.json
traceback.txt  (yalnız hata olursa)

KULLANIM
--------
Windows:
    C:\Users\ekale.ELIFK\.conda\envs\ImageToRender_GPU\python.exe \
    E:\ImageToRender\ImageToRenderStage4.py

Belirli parent run ile:
    python ImageToRenderStage4.py --parent-run outputs\restore_YYYYMMDD_HHMMSS
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STAGE_NAME = "v8_stage4_geometric_reconstruction"
STAGE_TITLE = "Stage 4 — Geometrik Rekonstrüksiyon"
SCRIPT_NAME = "ImageToRender2Restoration_v8_stage4_geometric_reconstruction.py"


@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    require_locked_hypotheses: bool = True
    require_no_unresolved_review: bool = True
    require_locked_observed_inventory: bool = True

    allowed_classes: tuple[str, ...] = ("arch", "pilaster", "cornice", "base")
    max_donor_candidates: int = 5

    # Donör puanları
    donor_hint_bonus: float = 3.0
    related_id_bonus: float = 0.45
    intact_bonus: float = 1.20
    fragmented_bonus: float = 0.30
    same_sector_bonus: float = 0.45
    low_damage_weight: float = 0.80
    size_similarity_weight: float = 1.20
    aspect_similarity_weight: float = 1.10
    spatial_proximity_weight: float = 0.25

    # Geometri
    arch_band_ratio: float = 0.13
    arch_min_thickness_px: int = 4
    arch_max_thickness_ratio: float = 0.24
    geometry_close_kernel_px: int = 3
    damage_threshold: int = 128

    # Görselleştirme
    observed_box_color: tuple[int, int, int] = (120, 120, 120)
    target_box_color: tuple[int, int, int] = (255, 140, 0)
    donor_box_color: tuple[int, int, int] = (40, 190, 255)
    overlay_alpha: float = 0.55
    header_height: int = 38
    label_font_size: int = 13
    donor_sheet_card_width: int = 280
    donor_sheet_card_height: int = 180
    donor_sheet_columns: int = 3


# =============================================================================
# GENEL YARDIMCILAR
# =============================================================================

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_run_dir(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            time.sleep(1.05)


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"stage4.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def copy_with_hash(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "copied_to": destination.name,
        "source_sha256": sha256_file(source),
        "run_copy_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def source_script_info() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "filename": source.name,
        "path": str(source),
        "sha256": sha256_file(source),
    }


def clip_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_width(box: Sequence[float]) -> float:
    return max(1.0, float(box[2]) - float(box[0]))


def box_height(box: Sequence[float]) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def box_area(box: Sequence[float]) -> float:
    return box_width(box) * box_height(box)


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def safe_log_ratio(first: float, second: float) -> float:
    first = max(1e-6, float(first))
    second = max(1e-6, float(second))
    return abs(math.log(first / second))


def similarity_from_log_distance(distance: float) -> float:
    return math.exp(-max(0.0, float(distance)))


def mask_ratio(mask: np.ndarray, box: Sequence[float]) -> float:
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.mean(crop > 0))


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def class_color(class_name: str) -> tuple[int, int, int]:
    palette = {
        "arch": (236, 72, 153),
        "pilaster": (56, 189, 248),
        "cornice": (250, 204, 21),
        "base": (34, 197, 94),
    }
    return palette.get(class_name, (168, 85, 247))


def source_color(index: int) -> tuple[int, int, int]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ]
    return palette[(index - 1) % len(palette)]


# =============================================================================
# PARENT BULMA / DOĞRULAMA
# =============================================================================

REQUIRED_PARENT_FILES = (
    "input.png",
    "completion_mask.png",
    "source_expert_validated_inventory.json",
    "expert_approved_reconstruction_hypotheses.json",
)

OPTIONAL_PARENT_FILES = (
    "source_stage3c_b_facade_sectors.json",
    "source_stage3c_b_support_axis_sequence.json",
    "source_stage3c_b_ordered_bay_sequence.json",
    "source_stage3c_b_arch_role_subfamilies.json",
    "source_stage3c_b_pilaster_role_subfamilies.json",
    "expert_hypothesis_review_actions.json",
    "stage3c_c_report.txt",
    "run_meta.json",
)


def is_compatible_parent(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_PARENT_FILES)


def find_parent_run(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not is_compatible_parent(parent):
            missing = [name for name in REQUIRED_PARENT_FILES if not (parent / name).exists()]
            raise FileNotFoundError(
                "Belirtilen parent run Stage 4 için uyumlu değil. Eksikler: "
                + ", ".join(missing)
            )
        return parent

    candidates = sorted(
        (path for path in Path(cfg.output_root).glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if is_compatible_parent(candidate):
            return candidate
    raise FileNotFoundError(
        "Kilitli Stage 3C-C çıktılarını içeren uyumlu restore_* klasörü bulunamadı."
    )


def normalize_observed_inventory(
    inventory_doc: dict[str, Any],
    width: int,
    height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    if cfg.require_locked_observed_inventory:
        if not inventory_doc.get("locked") or inventory_doc.get("inventory_state") != "finalized":
            raise RuntimeError("Stage 4 yalnız kilitlenmiş gözlenen uzman envanteriyle çalışır.")
    if inventory_doc.get("unresolved_review"):
        raise RuntimeError("Gözlenen uzman envanterinde çözümlenmemiş review kaydı var.")

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(inventory_doc.get("elements", []), start=1):
        item_id = str(item.get("id") or f"observed_{index:03d}")
        if item_id in seen_ids:
            raise RuntimeError(f"Gözlenen envanterde yinelenen ID: {item_id}")
        seen_ids.add(item_id)
        class_name = str(item.get("class", "other")).strip().lower()
        status = str(item.get("status", "observed_intact"))
        bbox = clip_box(item.get("bbox_xyxy", [0, 0, 1, 1]), width, height)
        record = dict(item)
        record.update(
            {
                "id": item_id,
                "class": class_name,
                "status": status,
                "bbox_xyxy": bbox,
                "center": list(box_center(bbox)),
                "width": box_width(bbox),
                "height": box_height(bbox),
            }
        )
        out.append(record)
    return out


def normalize_approved_hypotheses(
    hypothesis_doc: dict[str, Any],
    width: int,
    height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    if cfg.require_locked_hypotheses:
        if not hypothesis_doc.get("locked") or hypothesis_doc.get("inventory_state") != "finalized":
            raise RuntimeError("Stage 4 yalnız kilitlenmiş Stage 3C-C hipotezleriyle çalışır.")
    if cfg.require_no_unresolved_review and hypothesis_doc.get("unresolved_review"):
        raise RuntimeError("Stage 3C-C çıktısında çözümlenmemiş hipotez bulunuyor.")

    raw = hypothesis_doc.get("approved_hypotheses")
    if raw is None:
        raw = hypothesis_doc.get("hypotheses", [])

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw, start=1):
        item_id = str(item.get("id") or f"approved_hypothesis_{index:03d}")
        if item_id in seen_ids:
            raise RuntimeError(f"Onaylı hipotezlerde yinelenen ID: {item_id}")
        seen_ids.add(item_id)
        status = str(item.get("status", "expert_approved"))
        if status != "expert_approved":
            raise RuntimeError(
                f"Stage 4 girdisinde expert_approved olmayan kayıt var: {item_id} / {status}"
            )
        class_name = str(item.get("class", "other")).strip().lower()
        bbox = clip_box(item.get("bbox_xyxy", [0, 0, 1, 1]), width, height)
        record = dict(item)
        record.update(
            {
                "id": item_id,
                "class": class_name,
                "status": status,
                "bbox_xyxy": bbox,
                "center": list(box_center(bbox)),
                "width": box_width(bbox),
                "height": box_height(bbox),
            }
        )
        out.append(record)
    return out


# =============================================================================
# DONÖR SEÇİMİ
# =============================================================================

def element_to_sector_map(parent_run: Path) -> dict[str, str]:
    path = parent_run / "source_stage3c_b_facade_sectors.json"
    if not path.exists():
        return {}
    doc = read_json(path)
    mapping = doc.get("element_to_sector", {})
    return {str(key): str(value) for key, value in mapping.items()}


def score_donor(
    hypothesis: dict[str, Any],
    candidate: dict[str, Any],
    damage_mask: np.ndarray,
    image_size: tuple[int, int],
    sector_map: dict[str, str],
    cfg: Config,
) -> dict[str, Any]:
    width, height = image_size
    target_box = hypothesis["bbox_xyxy"]
    donor_box = candidate["bbox_xyxy"]

    target_w = box_width(target_box)
    target_h = box_height(target_box)
    donor_w = box_width(donor_box)
    donor_h = box_height(donor_box)

    target_ar = target_w / max(1.0, target_h)
    donor_ar = donor_w / max(1.0, donor_h)

    size_distance = safe_log_ratio(donor_w, target_w) + safe_log_ratio(donor_h, target_h)
    aspect_distance = safe_log_ratio(donor_ar, target_ar)
    size_similarity = similarity_from_log_distance(size_distance)
    aspect_similarity = similarity_from_log_distance(aspect_distance)

    target_center = box_center(target_box)
    donor_center = box_center(donor_box)
    diagonal = max(1.0, math.hypot(width, height))
    normalized_distance = min(
        1.0,
        math.hypot(target_center[0] - donor_center[0], target_center[1] - donor_center[1]) / diagonal,
    )
    proximity = 1.0 - normalized_distance

    donor_damage = mask_ratio(damage_mask, donor_box)
    low_damage = 1.0 - donor_damage

    hinted_ids = {str(value) for value in hypothesis.get("donor_ids", []) if value}
    related_ids = {str(value) for value in hypothesis.get("related_observed_ids", []) if value}
    hinted = candidate["id"] in hinted_ids
    related = candidate["id"] in related_ids

    status = candidate.get("status")
    status_bonus = 0.0
    if status == "observed_intact":
        status_bonus = cfg.intact_bonus
    elif status == "observed_fragmented":
        status_bonus = cfg.fragmented_bonus

    target_sector = hypothesis.get("sector_id")
    donor_sector = sector_map.get(candidate["id"])
    same_sector = bool(target_sector and donor_sector and target_sector == donor_sector)

    score = 0.0
    score += cfg.donor_hint_bonus if hinted else 0.0
    score += cfg.related_id_bonus if related else 0.0
    score += status_bonus
    score += cfg.same_sector_bonus if same_sector else 0.0
    score += cfg.low_damage_weight * low_damage
    score += cfg.size_similarity_weight * size_similarity
    score += cfg.aspect_similarity_weight * aspect_similarity
    score += cfg.spatial_proximity_weight * proximity

    return {
        "donor_id": candidate["id"],
        "class": candidate["class"],
        "status": status,
        "bbox_xyxy": donor_box,
        "sector_id": donor_sector,
        "score": round(float(score), 6),
        "score_components": {
            "hinted_by_stage3c_c": hinted,
            "related_observed_id": related,
            "same_sector": same_sector,
            "status_bonus": round(status_bonus, 6),
            "donor_damage_ratio": round(donor_damage, 6),
            "low_damage_score": round(low_damage, 6),
            "size_similarity": round(size_similarity, 6),
            "aspect_similarity": round(aspect_similarity, 6),
            "spatial_proximity": round(proximity, 6),
        },
    }


def select_donors(
    hypothesis: dict[str, Any],
    observed_elements: list[dict[str, Any]],
    damage_mask: np.ndarray,
    image_size: tuple[int, int],
    sector_map: dict[str, str],
    cfg: Config,
) -> dict[str, Any]:
    same_class = [
        item
        for item in observed_elements
        if item.get("class") == hypothesis.get("class")
        and item.get("status") in {"observed_intact", "observed_fragmented"}
    ]

    scored = [
        score_donor(hypothesis, candidate, damage_mask, image_size, sector_map, cfg)
        for candidate in same_class
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    scored = scored[: max(1, cfg.max_donor_candidates)]

    primary = scored[0] if scored else None
    target_box = hypothesis["bbox_xyxy"]
    transform = None
    if primary is not None:
        donor_box = primary["bbox_xyxy"]
        donor_center = box_center(donor_box)
        target_center = box_center(target_box)
        transform = {
            "source_bbox_xyxy": donor_box,
            "target_bbox_xyxy": target_box,
            "scale_x": round(box_width(target_box) / max(1.0, box_width(donor_box)), 6),
            "scale_y": round(box_height(target_box) / max(1.0, box_height(donor_box)), 6),
            "translation_xy": [
                round(target_center[0] - donor_center[0], 6),
                round(target_center[1] - donor_center[1], 6),
            ],
            "rotation_deg": 0.0,
            "texture_warp_applied": False,
            "note": "Stage 4 yalnız geometrik dönüşüm planını kaydeder; doku Stage 5'e bırakılır.",
        }

    return {
        "hypothesis_id": hypothesis["id"],
        "class": hypothesis["class"],
        "target_bbox_xyxy": target_box,
        "primary_donor_id": None if primary is None else primary["donor_id"],
        "primary_donor": primary,
        "candidate_count": len(scored),
        "ranked_candidates": scored,
        "transform_plan": transform,
        "selection_state": "donor_selected" if primary is not None else "parametric_geometry_only_no_donor",
    }


# =============================================================================
# PARAMETRİK GEOMETRİ
# =============================================================================

def draw_arch_mask(
    canvas: np.ndarray,
    bbox: Sequence[int],
    cfg: Config,
) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = int(round((x1 + x2) / 2.0))
    spring_y = max(y1 + 1, y2 - 1)
    radius_x = max(1, int(round(width / 2.0)))
    rise_y = max(1, height)
    thickness = max(cfg.arch_min_thickness_px, int(round(min(width, height) * cfg.arch_band_ratio)))
    thickness = min(thickness, max(cfg.arch_min_thickness_px, int(round(min(width, height) * cfg.arch_max_thickness_ratio))))

    cv2.ellipse(
        canvas,
        (center_x, spring_y),
        (radius_x, rise_y),
        0.0,
        180.0,
        360.0,
        255,
        thickness,
        cv2.LINE_AA,
    )
    # Bbox dışına taşan anti-alias izlerini kesin.
    clipped = np.zeros_like(canvas)
    clipped[y1:y2, x1:x2] = canvas[y1:y2, x1:x2]
    canvas[:, :] = clipped

    return {
        "primitive": "elliptical_arch_band",
        "center_xy": [center_x, spring_y],
        "radius_x": radius_x,
        "rise_y": rise_y,
        "angle_start_deg": 180.0,
        "angle_end_deg": 360.0,
        "band_thickness_px": thickness,
        "bbox_xyxy": list(map(int, bbox)),
    }


def draw_rectangular_mask(
    canvas: np.ndarray,
    bbox: Sequence[int],
    primitive: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(canvas, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), 255, -1)
    return {
        "primitive": primitive,
        "bbox_xyxy": list(map(int, bbox)),
    }


def build_element_geometry_mask(
    hypothesis: dict[str, Any],
    image_size: tuple[int, int],
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    width, height = image_size
    canvas = np.zeros((height, width), dtype=np.uint8)
    bbox = clip_box(hypothesis["bbox_xyxy"], width, height)
    class_name = hypothesis["class"]

    if class_name == "arch":
        descriptor = draw_arch_mask(canvas, bbox, cfg)
    elif class_name == "pilaster":
        descriptor = draw_rectangular_mask(canvas, bbox, "rectangular_vertical_member")
    elif class_name == "cornice":
        descriptor = draw_rectangular_mask(canvas, bbox, "horizontal_cornice_segment")
    elif class_name == "base":
        descriptor = draw_rectangular_mask(canvas, bbox, "horizontal_base_segment")
    else:
        descriptor = draw_rectangular_mask(canvas, bbox, "fallback_expert_bbox_region")
        descriptor["warning"] = "Sınıf için özel parametrik primitive yok; uzman bbox bölgesi kullanıldı."

    if cfg.geometry_close_kernel_px >= 3:
        kernel_size = cfg.geometry_close_kernel_px
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    canvas = np.where(canvas > 0, 255, 0).astype(np.uint8)
    descriptor["class"] = class_name
    descriptor["geometry_authority"] = "expert_approved_target_bbox"
    descriptor["texture_generated"] = False
    return canvas, descriptor


# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================

def add_header(image: Image.Image, text: str, cfg: Config) -> Image.Image:
    base = image.convert("RGB")
    output = Image.new("RGB", (base.width, base.height + cfg.header_height), (20, 20, 20))
    output.paste(base, (0, cfg.header_height))
    draw = ImageDraw.Draw(output)
    draw.text((8, 9), text, fill=(255, 255, 255), font=load_font(cfg.label_font_size))
    return output


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=(18, 18, 18), outline=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def make_outline_overlay(
    image: Image.Image,
    observed: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    descriptors: dict[str, dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(image, "STAGE 4 — GEOMETRIC RECONSTRUCTION OUTLINE / NO TEXTURE", cfg)
    draw = ImageDraw.Draw(canvas)
    font = load_font(cfg.label_font_size)
    shift = cfg.header_height

    for element in observed:
        x1, y1, x2, y2 = element["bbox_xyxy"]
        draw.rectangle([x1, y1 + shift, x2, y2 + shift], outline=cfg.observed_box_color, width=1)

    for index, hypothesis in enumerate(hypotheses, start=1):
        x1, y1, x2, y2 = hypothesis["bbox_xyxy"]
        color = class_color(hypothesis["class"])
        draw.rectangle([x1, y1 + shift, x2, y2 + shift], outline=color, width=3)
        desc = descriptors[hypothesis["id"]]
        if desc["primitive"] == "elliptical_arch_band":
            center = (desc["center_xy"][0], desc["center_xy"][1] + shift)
            rx = desc["radius_x"]
            ry = desc["rise_y"]
            box = [center[0] - rx, center[1] - ry, center[0] + rx, center[1] + ry]
            draw.arc(box, start=180, end=360, fill=color, width=max(2, int(desc["band_thickness_px"])))
        draw_label(draw, (x1 + 3, y1 + shift + 3), f"H{index} {hypothesis['id']} {hypothesis['class']}", color, font)
    return canvas


def tint_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    selected = mask > 0
    tint = np.zeros_like(base)
    tint[:, :] = np.asarray(color, dtype=np.float32)
    base[selected] = base[selected] * (1.0 - alpha) + tint[selected] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def make_geometry_overlay(
    image: Image.Image,
    damage_limited_masks: dict[str, np.ndarray],
    hypotheses: list[dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    output = image.convert("RGB")
    for hypothesis in hypotheses:
        output = tint_mask(
            output,
            damage_limited_masks[hypothesis["id"]],
            class_color(hypothesis["class"]),
            cfg.overlay_alpha,
        )
    output = add_header(output, "STAGE 4 — DAMAGE-LIMITED GEOMETRIC RECONSTRUCTION", cfg)
    draw = ImageDraw.Draw(output)
    font = load_font(cfg.label_font_size)
    shift = cfg.header_height
    for hypothesis in hypotheses:
        x1, y1, x2, y2 = hypothesis["bbox_xyxy"]
        color = class_color(hypothesis["class"])
        draw.rectangle([x1, y1 + shift, x2, y2 + shift], outline=color, width=2)
        draw_label(draw, (x1 + 3, y1 + shift + 3), hypothesis["id"], color, font)
    return output


def make_donor_overlay(
    image: Image.Image,
    hypotheses: list[dict[str, Any]],
    donor_docs: dict[str, dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(image, "STAGE 4 — TARGET / PRIMARY DONOR LINKS", cfg)
    draw = ImageDraw.Draw(canvas)
    font = load_font(cfg.label_font_size)
    shift = cfg.header_height

    for hypothesis in hypotheses:
        target = hypothesis["bbox_xyxy"]
        tx1, ty1, tx2, ty2 = target
        tcenter = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0 + shift)
        draw.rectangle([tx1, ty1 + shift, tx2, ty2 + shift], outline=cfg.target_box_color, width=3)
        donor_doc = donor_docs[hypothesis["id"]]
        primary = donor_doc.get("primary_donor")
        if primary:
            dx1, dy1, dx2, dy2 = primary["bbox_xyxy"]
            dcenter = ((dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0 + shift)
            draw.rectangle([dx1, dy1 + shift, dx2, dy2 + shift], outline=cfg.donor_box_color, width=2)
            draw.line([dcenter, tcenter], fill=cfg.donor_box_color, width=2)
            label = f"{hypothesis['id']} <- {primary['donor_id']}"
        else:
            label = f"{hypothesis['id']} <- PARAMETRIC ONLY"
        draw_label(draw, (tx1 + 3, ty1 + shift + 3), label, cfg.target_box_color, font)
    return canvas


def make_donor_reference_sheet(
    image: Image.Image,
    hypotheses: list[dict[str, Any]],
    donor_docs: dict[str, dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    cards = []
    font = load_font(cfg.label_font_size)
    for hypothesis in hypotheses:
        donor_doc = donor_docs[hypothesis["id"]]
        primary = donor_doc.get("primary_donor")
        card = Image.new("RGB", (cfg.donor_sheet_card_width, cfg.donor_sheet_card_height), (245, 245, 245))
        draw = ImageDraw.Draw(card)
        title = f"{hypothesis['id']} | {hypothesis['class']}"
        draw.text((8, 6), title, fill=(20, 20, 20), font=font)
        if primary:
            crop = image.crop(tuple(primary["bbox_xyxy"])).convert("RGB")
            max_w = cfg.donor_sheet_card_width - 16
            max_h = cfg.donor_sheet_card_height - 48
            scale = min(max_w / max(1, crop.width), max_h / max(1, crop.height))
            resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
            px = (cfg.donor_sheet_card_width - resized.width) // 2
            py = 34 + (max_h - resized.height) // 2
            card.paste(resized, (px, py))
            draw.text((8, cfg.donor_sheet_card_height - 18), f"donor: {primary['donor_id']} score={primary['score']:.2f}", fill=(20, 20, 20), font=font)
        else:
            draw.text((8, 60), "Uygun gözlenen donör bulunamadı.\nParametrik geometri kullanılacak.", fill=(120, 30, 30), font=font)
        cards.append(card)

    if not cards:
        return Image.new("RGB", (cfg.donor_sheet_card_width, cfg.donor_sheet_card_height), (245, 245, 245))

    cols = max(1, cfg.donor_sheet_columns)
    rows = math.ceil(len(cards) / cols)
    sheet = Image.new(
        "RGB",
        (cols * cfg.donor_sheet_card_width, rows * cfg.donor_sheet_card_height),
        (225, 225, 225),
    )
    for index, card in enumerate(cards):
        x = (index % cols) * cfg.donor_sheet_card_width
        y = (index // cols) * cfg.donor_sheet_card_height
        sheet.paste(card, (x, y))
    return sheet


# =============================================================================
# RAPOR
# =============================================================================

def build_report(
    run_id: str,
    parent_run: Path,
    hypotheses: list[dict[str, Any]],
    plan_records: list[dict[str, Any]],
    full_union: np.ndarray,
    damage_union: np.ndarray,
) -> str:
    class_counts: dict[str, int] = {}
    no_donor = []
    empty_damage = []
    for record in plan_records:
        class_counts[record["class"]] = class_counts.get(record["class"], 0) + 1
        if record["donor_selection"]["primary_donor_id"] is None:
            no_donor.append(record["hypothesis_id"])
        if record["geometry_metrics"]["damage_limited_area_px"] == 0:
            empty_damage.append(record["hypothesis_id"])

    lines = [
        "STAGE 4 — GEOMETRIC RECONSTRUCTION",
        f"Run: {run_id}",
        f"Parent: {parent_run.resolve()}",
        "",
        "POLICY:",
        "- Stage 3C-C expert-approved hypothesis boxes are authoritative.",
        "- Observed expert inventory was not modified.",
        "- No new hypothesis was generated.",
        "- Full geometry and completion-mask-limited intervention geometry are separate.",
        "- No Stable Diffusion, inpaint, material or lighting generation was used.",
        "",
        "SUMMARY:",
        f"- Approved hypotheses processed: {len(hypotheses)}",
        f"- Class counts: {class_counts}",
        f"- Full geometry pixels: {int(np.count_nonzero(full_union))}",
        f"- Damage-limited geometry pixels: {int(np.count_nonzero(damage_union))}",
        f"- Hypotheses without observed donor: {len(no_donor)}",
        f"- Hypotheses with zero completion-mask intersection: {len(empty_damage)}",
        "",
        "DONOR WARNINGS:",
        "- " + (", ".join(no_donor) if no_donor else "none"),
        "",
        "ZERO-DAMAGE-INTERSECTION WARNINGS:",
        "- " + (", ".join(empty_damage) if empty_damage else "none"),
        "",
        "ELEMENTS:",
    ]
    for record in plan_records:
        donor = record["donor_selection"]["primary_donor_id"] or "PARAMETRIC_ONLY"
        metrics = record["geometry_metrics"]
        lines.append(
            f"- {record['hypothesis_id']} | {record['class']} | donor={donor} | "
            f"full_px={metrics['full_geometry_area_px']} | damage_px={metrics['damage_limited_area_px']} | "
            f"mask_overlap={metrics['completion_mask_overlap_ratio']:.3f}"
        )
    return "\n".join(lines) + "\n"


# =============================================================================
# ANA ÇALIŞMA
# =============================================================================

def run_stage(cfg: Config) -> Path:
    parent_run = find_parent_run(cfg)
    run_id, run_dir = create_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started_at = now_iso()
    started = time.perf_counter()

    try:
        logger.info("=" * 88)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info(f"[STAGE] {STAGE_TITLE}")
        logger.info("[RULE] expert-approved hypotheses authoritative / geometry only / no texture")
        logger.info("=" * 88)

        input_path = parent_run / "input.png"
        mask_path = parent_run / "completion_mask.png"
        inventory_path = parent_run / "source_expert_validated_inventory.json"
        hypotheses_path = parent_run / "expert_approved_reconstruction_hypotheses.json"

        image = Image.open(input_path).convert("RGB")
        width, height = image.size
        damage_mask = np.asarray(Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST))
        damage_mask = np.where(damage_mask >= cfg.damage_threshold, 255, 0).astype(np.uint8)

        inventory_doc = read_json(inventory_path)
        hypothesis_doc = read_json(hypotheses_path)
        observed = normalize_observed_inventory(inventory_doc, width, height, cfg)
        hypotheses = normalize_approved_hypotheses(hypothesis_doc, width, height, cfg)
        if not hypotheses:
            raise RuntimeError("Stage 3C-C kilitli dosyasında onaylı rekonstrüksiyon hipotezi yok.")

        logger.info(f"[INPUT] image={width}x{height} observed={len(observed)} approved_hypotheses={len(hypotheses)}")

        copied_inputs: dict[str, Any] = {
            "input": copy_with_hash(input_path, run_dir / "input.png"),
            "completion_mask": copy_with_hash(mask_path, run_dir / "completion_mask.png"),
            "expert_inventory": copy_with_hash(inventory_path, run_dir / "source_expert_validated_inventory.json"),
            "approved_hypotheses": copy_with_hash(
                hypotheses_path,
                run_dir / "source_expert_approved_reconstruction_hypotheses.json",
            ),
        }

        for name in OPTIONAL_PARENT_FILES:
            source = parent_run / name
            if source.exists():
                destination_name = name if name.startswith("source_") else f"source_{name}"
                copied_inputs[name] = copy_with_hash(source, run_dir / destination_name)

        sector_map = element_to_sector_map(parent_run)
        element_mask_dir = run_dir / "reconstruction_element_masks"
        element_mask_dir.mkdir(parents=True, exist_ok=False)

        full_union = np.zeros((height, width), dtype=np.uint8)
        damage_union = np.zeros((height, width), dtype=np.uint8)
        source_map_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        donor_docs: dict[str, dict[str, Any]] = {}
        descriptors: dict[str, dict[str, Any]] = {}
        full_masks: dict[str, np.ndarray] = {}
        damage_masks: dict[str, np.ndarray] = {}
        source_map_records: list[dict[str, Any]] = []
        plan_records: list[dict[str, Any]] = []

        for index, hypothesis in enumerate(hypotheses, start=1):
            donor_doc = select_donors(
                hypothesis,
                observed,
                damage_mask,
                (width, height),
                sector_map,
                cfg,
            )
            donor_docs[hypothesis["id"]] = donor_doc

            full_mask, descriptor = build_element_geometry_mask(hypothesis, (width, height), cfg)
            damage_limited = cv2.bitwise_and(full_mask, damage_mask)
            descriptors[hypothesis["id"]] = descriptor
            full_masks[hypothesis["id"]] = full_mask
            damage_masks[hypothesis["id"]] = damage_limited

            full_name = f"{hypothesis['id']}_full_geometry.png"
            damage_name = f"{hypothesis['id']}_damage_limited.png"
            Image.fromarray(full_mask, mode="L").save(element_mask_dir / full_name)
            Image.fromarray(damage_limited, mode="L").save(element_mask_dir / damage_name)

            full_union = cv2.bitwise_or(full_union, full_mask)
            damage_union = cv2.bitwise_or(damage_union, damage_limited)

            color = source_color(index)
            selected = damage_limited > 0
            source_map_rgb[selected] = np.asarray(color, dtype=np.uint8)

            full_area = int(np.count_nonzero(full_mask))
            damage_area = int(np.count_nonzero(damage_limited))
            completion_overlap = damage_area / max(1, full_area)

            source_map_records.append(
                {
                    "source_index": index,
                    "rgb": list(color),
                    "hypothesis_id": hypothesis["id"],
                    "class": hypothesis["class"],
                    "primary_donor_id": donor_doc["primary_donor_id"],
                    "element_mask_full": f"reconstruction_element_masks/{full_name}",
                    "element_mask_damage_limited": f"reconstruction_element_masks/{damage_name}",
                    "paint_priority": index,
                    "overlap_policy": "later_index_visible_in_rgb_source_map; all masks remain separate",
                }
            )

            plan_record = {
                "hypothesis_id": hypothesis["id"],
                "display_number": hypothesis.get("display_number"),
                "class": hypothesis["class"],
                "hypothesis_type": hypothesis.get("hypothesis_type"),
                "expert_approved_bbox_xyxy": hypothesis["bbox_xyxy"],
                "expert_note": hypothesis.get("expert_note", ""),
                "confidence": hypothesis.get("confidence"),
                "evidence": hypothesis.get("evidence", []),
                "related_observed_ids": hypothesis.get("related_observed_ids", []),
                "stage3c_c_donor_hints": hypothesis.get("donor_ids", []),
                "sector_id": hypothesis.get("sector_id"),
                "bay_id": hypothesis.get("bay_id"),
                "course_id": hypothesis.get("course_id"),
                "geometry_basis_from_stage3c_c": hypothesis.get("geometry_basis", {}),
                "donor_selection": donor_doc,
                "geometry_descriptor": descriptor,
                "geometry_metrics": {
                    "full_geometry_area_px": full_area,
                    "damage_limited_area_px": damage_area,
                    "completion_mask_overlap_ratio": round(completion_overlap, 6),
                    "stage5_intervention_mask_ready": damage_area > 0,
                },
                "output_masks": {
                    "full_geometry": f"reconstruction_element_masks/{full_name}",
                    "damage_limited": f"reconstruction_element_masks/{damage_name}",
                },
                "policy": {
                    "expert_bbox_authoritative": True,
                    "observed_inventory_modified": False,
                    "new_hypothesis_generated": False,
                    "texture_generated": False,
                    "stable_diffusion_used": False,
                },
            }
            plan_records.append(plan_record)

            logger.info(
                "[ELEMENT] %s class=%s donor=%s full_px=%d damage_px=%d",
                hypothesis["id"],
                hypothesis["class"],
                donor_doc["primary_donor_id"] or "PARAMETRIC_ONLY",
                full_area,
                damage_area,
            )

        # JSON çıktıları
        donor_selection_doc = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "parent_run": str(parent_run.resolve()),
            "donor_count": sum(1 for item in donor_docs.values() if item["primary_donor_id"]),
            "hypothesis_count": len(hypotheses),
            "selections": [donor_docs[hypothesis["id"]] for hypothesis in hypotheses],
        }
        write_json(run_dir / "stage4_donor_selection.json", donor_selection_doc)

        source_map_doc = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "image_size": [width, height],
            "source_map_file": "stage4_source_map.png",
            "records": source_map_records,
        }
        write_json(run_dir / "stage4_source_map.json", source_map_doc)

        plan_doc = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "status": "completed",
            "parent_run": {
                "run_id": parent_run.name,
                "path": str(parent_run.resolve()),
            },
            "policy": {
                "stage3c_c_expert_approvals_are_authoritative": True,
                "observed_inventory_untouched": True,
                "new_hypotheses_generated": False,
                "geometry_only": True,
                "full_and_damage_limited_masks_separate": True,
                "stable_diffusion_used": False,
                "material_generated": False,
                "lighting_generated": False,
            },
            "summary": {
                "observed_element_count": len(observed),
                "approved_hypothesis_count": len(hypotheses),
                "primary_donor_selected_count": sum(1 for item in donor_docs.values() if item["primary_donor_id"]),
                "parametric_only_count": sum(1 for item in donor_docs.values() if not item["primary_donor_id"]),
                "full_geometry_area_px": int(np.count_nonzero(full_union)),
                "damage_limited_geometry_area_px": int(np.count_nonzero(damage_union)),
                "zero_damage_intersection_count": sum(
                    record["geometry_metrics"]["damage_limited_area_px"] == 0 for record in plan_records
                ),
            },
            "elements": plan_records,
        }
        write_json(run_dir / "stage4_geometric_reconstruction_plan.json", plan_doc)

        # Maskeler
        Image.fromarray(full_union, mode="L").save(run_dir / "stage4_geometry_full_mask.png")
        Image.fromarray(damage_union, mode="L").save(run_dir / "stage4_geometry_damage_limited_mask.png")
        Image.fromarray(source_map_rgb, mode="RGB").save(run_dir / "stage4_source_map.png")

        # Overlayler
        make_outline_overlay(image, observed, hypotheses, descriptors, cfg).save(
            run_dir / "stage4_geometric_reconstruction_outline.png"
        )
        make_geometry_overlay(image, damage_masks, hypotheses, cfg).save(
            run_dir / "stage4_geometric_reconstruction_overlay.png"
        )
        make_donor_overlay(image, hypotheses, donor_docs, cfg).save(
            run_dir / "stage4_donor_selection_overlay.png"
        )
        make_donor_reference_sheet(image, hypotheses, donor_docs, cfg).save(
            run_dir / "stage4_donor_reference_sheet.png"
        )

        report = build_report(run_id, parent_run, hypotheses, plan_records, full_union, damage_union)
        (run_dir / "stage4_report.txt").write_text(report, encoding="utf-8")

        warnings = []
        no_donor = [record["hypothesis_id"] for record in plan_records if record["donor_selection"]["primary_donor_id"] is None]
        zero_damage = [record["hypothesis_id"] for record in plan_records if record["geometry_metrics"]["damage_limited_area_px"] == 0]
        if no_donor:
            warnings.append({"type": "no_observed_donor", "hypothesis_ids": no_donor})
        if zero_damage:
            warnings.append({"type": "zero_completion_mask_intersection", "hypothesis_ids": zero_damage})

        metadata = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "status": "completed_with_warnings" if warnings else "completed",
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "parent_run": {
                "run_id": parent_run.name,
                "path": str(parent_run.resolve()),
            },
            "source_script": source_script_info(),
            "inputs": copied_inputs,
            "config": asdict(cfg),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "pillow": getattr(Image, "__version__", "unknown"),
            },
            "policy": plan_doc["policy"],
            "summary": plan_doc["summary"],
            "warnings": warnings,
            "outputs": sorted(path.name for path in run_dir.iterdir()),
        }
        write_json(run_dir / "run_meta.json", metadata)

        logger.info(
            "[COUNT] observed=%d hypotheses=%d donors=%d parametric_only=%d",
            len(observed),
            len(hypotheses),
            plan_doc["summary"]["primary_donor_selected_count"],
            plan_doc["summary"]["parametric_only_count"],
        )
        logger.info(
            "[MASK] full_px=%d damage_limited_px=%d",
            plan_doc["summary"]["full_geometry_area_px"],
            plan_doc["summary"]["damage_limited_geometry_area_px"],
        )
        if warnings:
            logger.warning(f"[WARN] {warnings}")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info("[OK] Stage 4 geometrik rekonstrüksiyon planı tamamlandı")
        return run_dir

    except Exception as exc:
        logger.exception("[FAIL] Stage 4 failed")
        traceback_text = traceback.format_exc()
        (run_dir / "traceback.txt").write_text(traceback_text, encoding="utf-8")
        failure_meta = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "status": "failed",
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "parent_run": str(parent_run.resolve()),
            "source_script": source_script_info(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback_text,
            "config": asdict(cfg),
        }
        write_json(run_dir / "run_meta.json", failure_meta)
        raise


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=STAGE_TITLE)
    parser.add_argument("--parent-run", default="", help="Kilitli Stage 3C-C run klasörü")
    parser.add_argument("--output-root", default="outputs", help="Yeni restore_* klasörlerinin kökü")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(output_root=args.output_root, parent_run_dir=args.parent_run)
    try:
        run_dir = run_stage(cfg)
        print(f"\nStage 4 tamamlandı: {run_dir.resolve()}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
