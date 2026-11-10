#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3b_v3_family_cleanup.py
=========================================================
V8 / STAGE 3B-v3 — Mimari aile temizleme ve güvenilir ritim kontrolü

Bu sürüm Stage 3B-v2 çıktısını yeniden değerlendirir ve şu sorunları düzeltir:

1. Aynı mimari elemanı temsil eden iç içe / aşırı uzamış kutuları birleştirir.
   Örnek: küçük gerçek kemer kutusuyla aynı üst konturu paylaşan, fakat aşağı doğru
   gereksiz uzayan ikinci bir kemer kutusu prototip üyesi yapılmaz.
2. Üst kat ile alt cephe kemerlerini aynı aileye karıştırmaz.
3. Sağlam kemer prototipini aynı kotta bulunan benzersiz ve düşük hasarlı üyelerden kurar.
4. Merkezdeki ağır hasarlı açıklığı sağlam prototip üyesi yapmaz.
5. Az veya düzensiz üyeden sahte bir "ortalama ritim" üretmez.
6. Yeni kemer, pilaster, eksen, modül veya kayıp eleman OLUŞTURMAZ.
7. Her çalıştırmada yeni outputs/restore_YYYYMMDD_HHMMSS klasörü açar.

Ana çıktılar
------------
stage3b_v3_cleanup_diagnostics.png
stage3b_v3_clean_families.png
stage3b_v3_family_schematic.png
stage3b_v3_arch_families.json
stage3b_v3_pilaster_families.json
stage3b_v3_openings.json
stage3b_v3_duplicate_relations.json
stage3b_v3_rhythm_analysis.json
stage3b_v3_report.txt
run.log
run_meta.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    # Parent run içinde bulunması gereken temel dosyalar.
    required_parent_files: tuple[str, ...] = (
        "input.png",
        "completion_mask.png",
        "source_validated_elements.json",
        "source_tentative_elements.json",
        "source_stage2d_rhythm_groups.json",
        "stage3a_existing_evidence_skeleton.json",
        "stage3b_v2_arch_families.json",
        "stage3b_v2_pilaster_families.json",
        "stage3b_v2_rhythm_analysis.json",
    )

    # Temel skor eşikleri.
    min_arch_score: float = 0.40
    min_pilaster_score: float = 0.40
    min_opening_score: float = 0.35

    # Hasar durumları.
    intact_damage_max: float = 0.18
    fragmented_damage_max: float = 0.75
    central_heavy_damage_min: float = 0.55

    # Aynı kemerin alternatif kutularını bulma.
    arch_duplicate_center_x_tol_ratio: float = 0.045
    arch_duplicate_top_y_tol_ratio: float = 0.025
    arch_duplicate_width_log_tol: float = 0.38
    arch_duplicate_x_overlap_min: float = 0.62
    arch_overextended_height_ratio: float = 1.75
    general_duplicate_iou_min: float = 0.72

    # Üst bölümdeki büyük yanlış kemer benzeri kutu.
    upper_outlier_center_y_max_ratio: float = 0.34
    upper_outlier_width_min_ratio: float = 0.20
    upper_outlier_height_min_ratio: float = 0.15

    # Aynı cephe bandındaki kemer ailesi.
    arch_springline_cluster_tol_ratio: float = 0.035
    arch_center_y_cluster_tol_ratio: float = 0.060
    arch_size_log_distance_max: float = 0.58
    arch_aspect_ratio_min: float = 0.45
    arch_aspect_ratio_max: float = 1.55

    # Merkez hasar bölgesini anlamak için.
    central_x_tolerance_ratio: float = 0.18

    # Pilaster ailesi.
    pilaster_size_log_distance_max: float = 0.34
    pilaster_center_y_tol_ratio: float = 0.085

    # Ritim güvenilirliği.
    rhythm_min_members: int = 4
    rhythm_max_spacing_cv: float = 0.25
    rhythm_spacing_outlier_z: float = 2.5

    # Görselleştirme.
    box_thickness: int = 2
    text_scale: float = 0.42


# =============================================================================
# CONSTANTS
# =============================================================================
COLORS: dict[str, tuple[int, int, int]] = {
    "arch_intact_family": (45, 215, 75),
    "arch_fragmented_assigned": (255, 180, 45),
    "arch_central_fragment": (255, 135, 35),
    "arch_upper_outlier": (170, 170, 170),
    "arch_uncertain": (235, 210, 45),
    "pilaster_intact": (220, 75, 245),
    "pilaster_fragmented": (255, 145, 245),
    "opening": (50, 180, 255),
    "duplicate_removed": (120, 120, 120),
}


# =============================================================================
# FILE / LOG / METADATA
# =============================================================================
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_new_run_dir(output_root: Path) -> tuple[str, Path]:
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
    logger = logging.getLogger(f"stage3b_v3_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def find_parent_run(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not parent.is_dir():
            raise FileNotFoundError(f"Parent run klasörü yok: {parent.resolve()}")
        missing = [name for name in cfg.required_parent_files if not (parent / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Parent run içinde eksik dosyalar: " + ", ".join(missing)
            )
        return parent

    output_root = Path(cfg.output_root)
    candidates = sorted(
        (path for path in output_root.glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / name).exists() for name in cfg.required_parent_files):
            return candidate

    raise FileNotFoundError(
        "Stage 3B-v2 girdilerini içeren uygun restore_* klasörü bulunamadı: "
        f"{output_root.resolve()}"
    )


def copy_parent_inputs(parent: Path, run_dir: Path, cfg: Config) -> dict[str, Any]:
    copy_names = [
        *cfg.required_parent_files,
        "stage3b_v2_element_families.png",
        "stage3b_v2_family_schematic.png",
        "stage3b_v2_family_report.txt",
    ]

    copied: dict[str, Any] = {}
    for name in copy_names:
        source = parent / name
        if not source.exists():
            continue
        target_name = name if (name in ("input.png", "completion_mask.png") or name.startswith("source_")) else f"source_{name}"
        target = run_dir / target_name
        shutil.copy2(source, target)
        copied[name] = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "run_copy": target_name,
            "run_copy_sha256": sha256_file(target),
        }
    return copied


# =============================================================================
# IMAGE / GEOMETRY HELPERS
# =============================================================================
def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L").resize(
        (width, height),
        Image.Resampling.NEAREST,
    )
    array = np.asarray(image)
    return np.where(array >= 128, 255, 0).astype(np.uint8)


def clip_box(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_width(box: Iterable[float]) -> float:
    box = list(box)
    return max(0.0, float(box[2]) - float(box[0]))


def box_height(box: Iterable[float]) -> float:
    box = list(box)
    return max(0.0, float(box[3]) - float(box[1]))


def box_center(box: Iterable[float]) -> tuple[float, float]:
    box = list(box)
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def box_area(box: Iterable[float]) -> float:
    return box_width(box) * box_height(box)


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def horizontal_overlap_ratio(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, _, ax2, _ = [float(value) for value in box_a]
    bx1, _, bx2, _ = [float(value) for value in box_b]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    denominator = max(1e-6, min(ax2 - ax1, bx2 - bx1))
    return intersection / denominator


def logarithmic_difference(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(math.log(a / b))


def damage_fraction(box: list[int], mask: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi > 0) / roi.size)


def largest_damage_component(mask: np.ndarray) -> dict[str, Any]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        height, width = mask.shape[:2]
        return {
            "bbox_xyxy": [0, 0, width, height],
            "center": [width / 2.0, height / 2.0],
            "area": 0,
        }
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[label]
    return {
        "bbox_xyxy": [x, y, x + width, y + height],
        "center": [round(float(cx), 3), round(float(cy), 3)],
        "area": int(stats[label, cv2.CC_STAT_AREA]),
    }


# =============================================================================
# SOURCE ELEMENT NORMALIZATION
# =============================================================================
def flatten_elements(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]
    if isinstance(document, dict):
        for key in ("elements", "validated_elements", "tentative_elements", "items"):
            value = document.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def canonical_class(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    if "arch" in label:
        return "arch"
    if "pilaster" in label or "column" in label:
        return "pilaster"
    if "window" in label:
        return "window"
    if "door" in label:
        return "door"
    return label


def element_class(item: dict[str, Any]) -> str:
    for key in ("class", "label", "class_name", "category", "type"):
        if item.get(key):
            return canonical_class(item[key])
    return "unknown"


def element_score(item: dict[str, Any]) -> float:
    for key in ("final_score", "fused_score", "score", "confidence"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def element_box(item: dict[str, Any]) -> list[float] | None:
    for key in ("bbox_xyxy", "box_xyxy", "bbox", "box"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return [float(part) for part in value]
    return None


def normalize_elements(
    validated_document: Any,
    tentative_document: Any,
    damage_mask: np.ndarray,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source_group, document in (
        ("validated", validated_document),
        ("tentative", tentative_document),
    ):
        for source_index, raw in enumerate(flatten_elements(document)):
            box_value = element_box(raw)
            if box_value is None:
                continue
            box = clip_box(box_value, width, height)
            class_name = element_class(raw)
            score = element_score(raw)
            cx, cy = box_center(box)
            bw = box_width(box)
            bh = box_height(box)
            normalized.append(
                {
                    "id": str(raw.get("source_id") or raw.get("id") or f"{source_group}_{source_index:03d}"),
                    "source_group": source_group,
                    "source_index": source_index,
                    "class": class_name,
                    "score": round(score, 6),
                    "raw_status": str(raw.get("status") or raw.get("raw_status") or source_group),
                    "bbox_xyxy": box,
                    "center": [round(cx, 3), round(cy, 3)],
                    "width": round(bw, 3),
                    "height": round(bh, 3),
                    "aspect_height_over_width": round(bh / max(1.0, bw), 6),
                    "damage_fraction": round(damage_fraction(box, damage_mask), 6),
                    "raw": raw,
                }
            )
    return normalized


# =============================================================================
# DUPLICATE / ALTERNATIVE BOX CLEANUP
# =============================================================================
def source_priority(element: dict[str, Any]) -> tuple[int, int, float]:
    status = str(element.get("raw_status", "")).lower()
    group = str(element.get("source_group", "")).lower()
    status_rank = {
        "validated": 4,
        "fragment_supported": 3,
        "tentative": 2,
    }.get(status, 1)
    group_rank = 2 if group == "validated" else 1
    return status_rank, group_rank, float(element.get("score", 0.0))


def same_arch_top_hypothesis(
    first: dict[str, Any],
    second: dict[str, Any],
    width: int,
    height: int,
    cfg: Config,
) -> bool:
    first_box = first["bbox_xyxy"]
    second_box = second["bbox_xyxy"]
    first_cx = float(first["center"][0])
    second_cx = float(second["center"][0])
    center_close = abs(first_cx - second_cx) <= width * cfg.arch_duplicate_center_x_tol_ratio
    top_close = abs(float(first_box[1]) - float(second_box[1])) <= height * cfg.arch_duplicate_top_y_tol_ratio
    widths_close = logarithmic_difference(float(first["width"]), float(second["width"])) <= cfg.arch_duplicate_width_log_tol
    x_overlap = horizontal_overlap_ratio(first_box, second_box) >= cfg.arch_duplicate_x_overlap_min
    return center_close and top_close and widths_close and x_overlap


def choose_arch_representative(first: dict[str, Any], second: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], dict[str, Any], str]:
    first_height = float(first["height"])
    second_height = float(second["height"])
    taller, shorter = (first, second) if first_height >= second_height else (second, first)

    if float(taller["height"]) >= float(shorter["height"]) * cfg.arch_overextended_height_ratio:
        return shorter, taller, "overextended_alternative_same_arch_top"

    first_priority = source_priority(first)
    second_priority = source_priority(second)
    if first_priority >= second_priority:
        return first, second, "lower_priority_overlapping_alternative"
    return second, first, "lower_priority_overlapping_alternative"


def cleanup_arch_duplicates(
    arches: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active = [dict(item) for item in arches]
    removed_ids: set[str] = set()
    relations: list[dict[str, Any]] = []

    for first_index in range(len(active)):
        first = active[first_index]
        if first["id"] in removed_ids:
            continue
        for second_index in range(first_index + 1, len(active)):
            second = active[second_index]
            if second["id"] in removed_ids:
                continue

            same_top = same_arch_top_hypothesis(first, second, width, height, cfg)
            strong_iou = box_iou(first["bbox_xyxy"], second["bbox_xyxy"]) >= cfg.general_duplicate_iou_min
            if not same_top and not strong_iou:
                continue

            kept, removed, reason = choose_arch_representative(first, second, cfg)
            removed_ids.add(str(removed["id"]))
            relations.append(
                {
                    "class": "arch",
                    "kept_id": kept["id"],
                    "removed_id": removed["id"],
                    "reason": reason,
                    "kept_box": kept["bbox_xyxy"],
                    "removed_box": removed["bbox_xyxy"],
                    "iou": round(box_iou(kept["bbox_xyxy"], removed["bbox_xyxy"]), 6),
                    "horizontal_overlap": round(horizontal_overlap_ratio(kept["bbox_xyxy"], removed["bbox_xyxy"]), 6),
                }
            )
            if removed["id"] == first["id"]:
                first = kept
                active[first_index] = kept
                break

    cleaned = [item for item in active if item["id"] not in removed_ids]
    return cleaned, relations


def cleanup_general_duplicates(
    elements: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for element in sorted(elements, key=source_priority, reverse=True):
        duplicate_of: dict[str, Any] | None = None
        for previous in kept:
            if previous["class"] != element["class"]:
                continue
            if box_iou(previous["bbox_xyxy"], element["bbox_xyxy"]) >= cfg.general_duplicate_iou_min:
                duplicate_of = previous
                break
        if duplicate_of is None:
            kept.append(element)
        else:
            relations.append(
                {
                    "class": element["class"],
                    "kept_id": duplicate_of["id"],
                    "removed_id": element["id"],
                    "reason": "same_class_high_iou_duplicate",
                    "kept_box": duplicate_of["bbox_xyxy"],
                    "removed_box": element["bbox_xyxy"],
                    "iou": round(box_iou(duplicate_of["bbox_xyxy"], element["bbox_xyxy"]), 6),
                }
            )
    return kept, relations


def cleanup_opening_aliases(openings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for element in sorted(openings, key=source_priority, reverse=True):
        matched: dict[str, Any] | None = None
        for previous in kept:
            if box_iou(previous["bbox_xyxy"], element["bbox_xyxy"]) >= 0.80:
                matched = previous
                break
        if matched is None:
            canonical = dict(element)
            canonical["class"] = "opening"
            canonical["class_aliases"] = [element["class"]]
            kept.append(canonical)
        else:
            if element["class"] not in matched["class_aliases"]:
                matched["class_aliases"].append(element["class"])
            aliases.append(
                {
                    "class": "opening",
                    "kept_id": matched["id"],
                    "removed_id": element["id"],
                    "reason": "same_opening_cross_class_alias",
                    "kept_box": matched["bbox_xyxy"],
                    "removed_box": element["bbox_xyxy"],
                    "iou": round(box_iou(matched["bbox_xyxy"], element["bbox_xyxy"]), 6),
                    "aliases": sorted(set(matched["class_aliases"] + [element["class"]])),
                }
            )
    return kept, aliases


# =============================================================================
# ARCH FAMILY CLEANUP
# =============================================================================
def is_upper_large_outlier(arch: dict[str, Any], width: int, height: int, cfg: Config) -> bool:
    center_y_ratio = float(arch["center"][1]) / max(1, height)
    width_ratio = float(arch["width"]) / max(1, width)
    height_ratio = float(arch["height"]) / max(1, height)
    return (
        center_y_ratio <= cfg.upper_outlier_center_y_max_ratio
        and width_ratio >= cfg.upper_outlier_width_min_ratio
        and height_ratio >= cfg.upper_outlier_height_min_ratio
    )


def arch_integrity(arch: dict[str, Any], cfg: Config) -> str:
    damage = float(arch["damage_fraction"])
    if damage <= cfg.intact_damage_max:
        return "intact"
    if damage <= cfg.fragmented_damage_max:
        return "fragmented"
    return "heavily_fragmented"


def arch_shape_plausible(arch: dict[str, Any], cfg: Config) -> bool:
    aspect = float(arch["aspect_height_over_width"])
    return cfg.arch_aspect_ratio_min <= aspect <= cfg.arch_aspect_ratio_max


def cluster_by_springline(
    arches: list[dict[str, Any]],
    image_height: int,
    cfg: Config,
) -> list[list[dict[str, Any]]]:
    tolerance = image_height * cfg.arch_springline_cluster_tol_ratio
    clusters: list[list[dict[str, Any]]] = []
    for arch in sorted(arches, key=lambda item: float(item["bbox_xyxy"][3])):
        springline = float(arch["bbox_xyxy"][3])
        placed = False
        for cluster in clusters:
            median_springline = float(np.median([float(item["bbox_xyxy"][3]) for item in cluster]))
            if abs(springline - median_springline) <= tolerance:
                cluster.append(arch)
                placed = True
                break
        if not placed:
            clusters.append([arch])
    return clusters


def choose_primary_arch_band(clusters: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not clusters:
        return []
    return max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            float(np.median([float(item["bbox_xyxy"][3]) for item in cluster])),
        ),
    )


def robust_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "std": None, "mad": None}
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return {
        "count": int(array.size),
        "median": round(median, 6),
        "mean": round(float(np.mean(array)), 6),
        "std": round(float(np.std(array)), 6),
        "mad": round(mad, 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
    }


def arch_size_distance(arch: dict[str, Any], median_width: float, median_height: float) -> float:
    return math.sqrt(
        logarithmic_difference(float(arch["width"]), median_width) ** 2
        + logarithmic_difference(float(arch["height"]), median_height) ** 2
    )


def build_arch_family_cleanup(
    arches: list[dict[str, Any]],
    damage_component: dict[str, Any],
    width: int,
    height: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    upper_outliers: list[dict[str, Any]] = []
    intact_candidates: list[dict[str, Any]] = []
    fragmented_candidates: list[dict[str, Any]] = []

    for arch in arches:
        item = dict(arch)
        item["integrity"] = arch_integrity(item, cfg)
        if is_upper_large_outlier(item, width, height, cfg):
            item["classification"] = "isolated_upper_outlier_candidate"
            upper_outliers.append(item)
        elif item["integrity"] == "intact" and arch_shape_plausible(item, cfg):
            intact_candidates.append(item)
        else:
            fragmented_candidates.append(item)

    springline_clusters = cluster_by_springline(intact_candidates, height, cfg)
    primary_band = choose_primary_arch_band(springline_clusters)

    primary_ids = {str(item["id"]) for item in primary_band}
    other_intact = [item for item in intact_candidates if str(item["id"]) not in primary_ids]

    median_width = float(np.median([float(item["width"]) for item in primary_band])) if primary_band else 0.0
    median_height = float(np.median([float(item["height"]) for item in primary_band])) if primary_band else 0.0
    median_springline = float(np.median([float(item["bbox_xyxy"][3]) for item in primary_band])) if primary_band else 0.0

    family_members: list[dict[str, Any]] = []
    family_rejected_size: list[dict[str, Any]] = []
    for item in primary_band:
        size_distance = arch_size_distance(item, median_width, median_height) if median_width and median_height else 0.0
        enriched = dict(item)
        enriched["family_size_log_distance"] = round(size_distance, 6)
        enriched["normalized_width_by_family_median"] = round(float(item["width"]) / max(1e-6, median_width), 6)
        enriched["normalized_height_by_family_median"] = round(float(item["height"]) / max(1e-6, median_height), 6)
        enriched["springline_offset_px"] = round(float(item["bbox_xyxy"][3]) - median_springline, 3)
        if size_distance <= cfg.arch_size_log_distance_max:
            enriched["classification"] = "intact_prototype_member"
            family_members.append(enriched)
        else:
            enriched["classification"] = "same_level_size_outlier"
            family_rejected_size.append(enriched)

    family_center_y = float(np.median([float(item["center"][1]) for item in family_members])) if family_members else 0.0
    family_springline = float(np.median([float(item["bbox_xyxy"][3]) for item in family_members])) if family_members else 0.0

    damage_center_x = float(damage_component["center"][0])
    fragmented_assigned: list[dict[str, Any]] = []
    central_fragments: list[dict[str, Any]] = []
    uncertain_fragments: list[dict[str, Any]] = []

    for item in fragmented_candidates:
        enriched = dict(item)
        center_x = float(item["center"][0])
        damage = float(item["damage_fraction"])
        springline_gap = abs(float(item["bbox_xyxy"][3]) - family_springline) / max(1, height) if family_members else 1.0
        center_y_gap = abs(float(item["center"][1]) - family_center_y) / max(1, height) if family_members else 1.0
        size_distance = arch_size_distance(item, median_width, median_height) if median_width and median_height else float("inf")

        enriched["family_size_log_distance"] = None if not math.isfinite(size_distance) else round(size_distance, 6)
        enriched["family_springline_gap_ratio"] = round(springline_gap, 6)
        enriched["family_center_y_gap_ratio"] = round(center_y_gap, 6)

        near_damage_center = abs(center_x - damage_center_x) <= width * cfg.central_x_tolerance_ratio
        if damage >= cfg.central_heavy_damage_min and near_damage_center:
            enriched["classification"] = "central_heavily_fragmented_arch_candidate"
            central_fragments.append(enriched)
        elif (
            family_members
            and springline_gap <= cfg.arch_springline_cluster_tol_ratio
            and center_y_gap <= cfg.arch_center_y_cluster_tol_ratio
            and math.isfinite(size_distance)
            and size_distance <= cfg.arch_size_log_distance_max * 1.25
        ):
            enriched["classification"] = "fragmented_member_of_primary_family"
            fragmented_assigned.append(enriched)
        else:
            enriched["classification"] = "fragmented_unassigned_arch_candidate"
            uncertain_fragments.append(enriched)

    for item in other_intact:
        enriched = dict(item)
        enriched["classification"] = "separate_level_intact_arch_candidate"
        uncertain_fragments.append(enriched)
    uncertain_fragments.extend(family_rejected_size)

    family = {
        "family_id": "arch_family_lower_01",
        "family_type": "arch_family",
        "prototype_status": "clean_intact_same_level_family" if family_members else "no_clean_prototype",
        "intact_count": len(family_members),
        "fragmented_assigned_count": len(fragmented_assigned),
        "prototype": {
            "source": "unique_low_damage_same_springline_members_only",
            "width": robust_stats([float(item["width"]) for item in family_members]),
            "height": robust_stats([float(item["height"]) for item in family_members]),
            "center_y": robust_stats([float(item["center"][1]) for item in family_members]),
            "springline_y": robust_stats([float(item["bbox_xyxy"][3]) for item in family_members]),
            "aspect_height_over_width": robust_stats([
                float(item["aspect_height_over_width"]) for item in family_members
            ]),
        },
        "members_intact": family_members,
        "members_fragmented_assigned": fragmented_assigned,
    }

    categories = {
        "primary_family": family,
        "central_fragmented_arches": central_fragments,
        "upper_outlier_arches": upper_outliers,
        "other_arch_candidates": uncertain_fragments,
        "springline_clusters": [
            {
                "member_ids": [item["id"] for item in cluster],
                "member_count": len(cluster),
                "springline_median": round(float(np.median([float(item["bbox_xyxy"][3]) for item in cluster])), 3),
            }
            for cluster in springline_clusters
        ],
    }
    return [family], categories


# =============================================================================
# PILASTER FAMILY CLEANUP
# =============================================================================
def pilaster_distance(first: dict[str, Any], second: dict[str, Any], width: int, height: int) -> float:
    width_term = logarithmic_difference(float(first["width"]), float(second["width"]))
    height_term = logarithmic_difference(float(first["height"]), float(second["height"]))
    center_y_term = abs(float(first["center"][1]) - float(second["center"][1])) / max(1, height)
    return math.sqrt(width_term ** 2 + height_term ** 2 + (center_y_term * 2.0) ** 2)


def build_pilaster_family(pilasters: list[dict[str, Any]], width: int, height: int, cfg: Config) -> list[dict[str, Any]]:
    intact = [dict(item, integrity="intact") for item in pilasters if float(item["damage_fraction"]) <= cfg.intact_damage_max]
    fragmented = [
        dict(item, integrity="fragmented")
        for item in pilasters
        if cfg.intact_damage_max < float(item["damage_fraction"]) <= cfg.fragmented_damage_max
    ]
    uncertain = [
        dict(item, integrity="heavily_fragmented")
        for item in pilasters
        if float(item["damage_fraction"]) > cfg.fragmented_damage_max
    ]

    if intact:
        median_width = float(np.median([float(item["width"]) for item in intact]))
        median_height = float(np.median([float(item["height"]) for item in intact]))
        median_center_y = float(np.median([float(item["center"][1]) for item in intact]))
    else:
        median_width = median_height = median_center_y = 0.0

    intact_members: list[dict[str, Any]] = []
    for item in intact:
        enriched = dict(item)
        enriched["classification"] = "intact_prototype_member"
        enriched["normalized_width_by_family_median"] = round(float(item["width"]) / max(1e-6, median_width), 6)
        enriched["normalized_height_by_family_median"] = round(float(item["height"]) / max(1e-6, median_height), 6)
        intact_members.append(enriched)

    fragmented_members: list[dict[str, Any]] = []
    fragmented_unassigned: list[dict[str, Any]] = []
    prototype_reference = {
        "width": median_width,
        "height": median_height,
        "center": [0.0, median_center_y],
    }
    for item in fragmented:
        distance = pilaster_distance(item, prototype_reference, width, height) if intact else float("inf")
        enriched = dict(item)
        enriched["family_distance"] = None if not math.isfinite(distance) else round(distance, 6)
        if distance <= cfg.pilaster_size_log_distance_max:
            enriched["classification"] = "fragmented_member_of_primary_family"
            fragmented_members.append(enriched)
        else:
            enriched["classification"] = "fragmented_unassigned_pilaster_candidate"
            fragmented_unassigned.append(enriched)

    for item in uncertain:
        item["classification"] = "heavily_fragmented_unassigned_pilaster_candidate"
        fragmented_unassigned.append(item)

    return [
        {
            "family_id": "pilaster_family_01",
            "family_type": "pilaster_family",
            "prototype_status": "clean_intact_family" if intact_members else "no_clean_prototype",
            "intact_count": len(intact_members),
            "fragmented_count": len(fragmented_members),
            "prototype": {
                "source": "unique_low_damage_members_only",
                "width": robust_stats([float(item["width"]) for item in intact_members]),
                "height": robust_stats([float(item["height"]) for item in intact_members]),
                "center_y": robust_stats([float(item["center"][1]) for item in intact_members]),
            },
            "members_intact": intact_members,
            "members_fragmented": fragmented_members,
            "members_unassigned": fragmented_unassigned,
        }
    ]


# =============================================================================
# RHYTHM ANALYSIS
# =============================================================================
def spacing_cv(spacings: list[float]) -> float | None:
    if not spacings:
        return None
    array = np.asarray(spacings, dtype=np.float64)
    mean = float(np.mean(array))
    if mean <= 1e-9:
        return None
    return float(np.std(array) / mean)


def analyze_family_rhythm(family: dict[str, Any], cfg: Config) -> dict[str, Any]:
    members = list(family.get("members_intact", []))
    centers = sorted(float(item["center"][0]) for item in members)
    spacings = [round(centers[index + 1] - centers[index], 3) for index in range(len(centers) - 1)]
    cv = spacing_cv(spacings)

    reliable = (
        len(centers) >= cfg.rhythm_min_members
        and cv is not None
        and cv <= cfg.rhythm_max_spacing_cv
    )

    reasons: list[str] = []
    if len(centers) < cfg.rhythm_min_members:
        reasons.append("insufficient_unique_intact_members")
    if cv is not None and cv > cfg.rhythm_max_spacing_cv:
        reasons.append("adjacent_spacings_are_irregular")
    if cv is None:
        reasons.append("spacing_variation_not_computable")

    return {
        "family_id": family["family_id"],
        "member_count": len(centers),
        "centers_x": centers,
        "observed_adjacent_spacings_px": spacings,
        "spacing_cv": None if cv is None else round(cv, 6),
        "rhythm_status": "reliable_observed_rhythm" if reliable else "insufficient_or_irregular_observed_rhythm",
        "trusted_spacing_px": round(float(np.median(spacings)), 6) if reliable and spacings else None,
        "unreliable_summary_mean_suppressed": not reliable,
        "reasons": reasons,
        "missing_element_inference_created": 0,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def add_header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 35), (18, 18, 18), -1)
    cv2.putText(
        output,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def draw_box_label(
    image: np.ndarray,
    box: list[int],
    label: str,
    color: tuple[int, int, int],
    thickness: int,
    text_scale: float,
    offset_index: int = 0,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text_y = max(49, y1 - 5 - (offset_index % 3) * 14)
    cv2.putText(
        image,
        label,
        (max(2, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_cleanup_diagnostics(
    original: np.ndarray,
    arch_categories: dict[str, Any],
    pilaster_families: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    duplicate_relations: list[dict[str, Any]],
    all_elements_by_id: dict[str, dict[str, Any]],
    cfg: Config,
) -> np.ndarray:
    image = add_header(
        original,
        "STAGE 3B-v3 CLEANUP DIAGNOSTICS - EXISTING BOXES ONLY / NO COMPLETION",
    )

    family = arch_categories["primary_family"]
    for index, item in enumerate(family["members_intact"]):
        draw_box_label(image, item["bbox_xyxy"], f"ARCH FAMILY {item['id']}", COLORS["arch_intact_family"], 2, cfg.text_scale, index)
    for index, item in enumerate(family["members_fragmented_assigned"]):
        draw_box_label(image, item["bbox_xyxy"], f"ARCH FRAGMENT {item['id']}", COLORS["arch_fragmented_assigned"], 2, cfg.text_scale, index)
    for index, item in enumerate(arch_categories["central_fragmented_arches"]):
        draw_box_label(image, item["bbox_xyxy"], f"CENTRAL FRAGMENT {item['id']}", COLORS["arch_central_fragment"], 2, cfg.text_scale, index)
    for index, item in enumerate(arch_categories["upper_outlier_arches"]):
        draw_box_label(image, item["bbox_xyxy"], f"UPPER OUTLIER {item['id']}", COLORS["arch_upper_outlier"], 2, cfg.text_scale, index)
    for index, item in enumerate(arch_categories["other_arch_candidates"]):
        draw_box_label(image, item["bbox_xyxy"], f"ARCH UNCERTAIN {item['id']}", COLORS["arch_uncertain"], 1, cfg.text_scale, index)

    for family_record in pilaster_families:
        for index, item in enumerate(family_record["members_intact"]):
            draw_box_label(image, item["bbox_xyxy"], f"PILASTER FAMILY {item['id']}", COLORS["pilaster_intact"], 2, cfg.text_scale, index)
        for index, item in enumerate(family_record["members_fragmented"]):
            draw_box_label(image, item["bbox_xyxy"], f"PILASTER FRAGMENT {item['id']}", COLORS["pilaster_fragmented"], 2, cfg.text_scale, index)
        for index, item in enumerate(family_record["members_unassigned"]):
            draw_box_label(image, item["bbox_xyxy"], f"PILASTER UNCERTAIN {item['id']}", COLORS["arch_uncertain"], 1, cfg.text_scale, index)

    for index, item in enumerate(openings):
        aliases = "/".join(item.get("class_aliases", [item.get("class", "opening")]))
        draw_box_label(image, item["bbox_xyxy"], f"OPENING {aliases} {item['id']}", COLORS["opening"], 2, cfg.text_scale, index)

    for index, relation in enumerate(duplicate_relations):
        removed = all_elements_by_id.get(str(relation["removed_id"]))
        if removed is None:
            continue
        draw_box_label(
            image,
            removed["bbox_xyxy"],
            f"REMOVED DUPLICATE {removed['id']}",
            COLORS["duplicate_removed"],
            1,
            0.36,
            index,
        )
    return image


def draw_clean_families(
    original: np.ndarray,
    arch_categories: dict[str, Any],
    pilaster_families: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    cfg: Config,
) -> np.ndarray:
    image = add_header(
        original,
        "STAGE 3B-v3 CLEAN EXISTING FAMILIES - PROTOTYPE MEMBERS ONLY / NO NEW OBJECTS",
    )
    family = arch_categories["primary_family"]
    for index, item in enumerate(family["members_intact"]):
        draw_box_label(image, item["bbox_xyxy"], f"{family['family_id']} {item['id']}", COLORS["arch_intact_family"], 2, cfg.text_scale, index)
    for index, item in enumerate(family["members_fragmented_assigned"]):
        draw_box_label(image, item["bbox_xyxy"], f"{family['family_id']} FRAGMENT {item['id']}", COLORS["arch_fragmented_assigned"], 2, cfg.text_scale, index)
    for index, item in enumerate(arch_categories["central_fragmented_arches"]):
        draw_box_label(image, item["bbox_xyxy"], f"CENTRAL FRAGMENTED ARCH {item['id']}", COLORS["arch_central_fragment"], 2, cfg.text_scale, index)
    for index, item in enumerate(arch_categories["upper_outlier_arches"]):
        draw_box_label(image, item["bbox_xyxy"], f"ISOLATED UPPER OUTLIER {item['id']}", COLORS["arch_upper_outlier"], 2, cfg.text_scale, index)

    for family_record in pilaster_families:
        for index, item in enumerate(family_record["members_intact"]):
            draw_box_label(image, item["bbox_xyxy"], f"{family_record['family_id']} {item['id']}", COLORS["pilaster_intact"], 2, cfg.text_scale, index)
        for index, item in enumerate(family_record["members_fragmented"]):
            draw_box_label(image, item["bbox_xyxy"], f"{family_record['family_id']} FRAGMENT {item['id']}", COLORS["pilaster_fragmented"], 2, cfg.text_scale, index)

    for index, item in enumerate(openings):
        draw_box_label(image, item["bbox_xyxy"], f"OPENING {item['id']}", COLORS["opening"], 2, cfg.text_scale, index)
    return image


def draw_family_schematic(
    shape: tuple[int, ...],
    arch_categories: dict[str, Any],
    pilaster_families: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    cfg: Config,
) -> np.ndarray:
    height, width = shape[:2]
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    canvas = add_header(canvas, "STAGE 3B-v3 CLEAN FAMILY SCHEMATIC - OBSERVED EVIDENCE ONLY")

    family = arch_categories["primary_family"]
    for item in family["members_intact"]:
        cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["arch_intact_family"], 2)
    for item in family["members_fragmented_assigned"]:
        cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["arch_fragmented_assigned"], 2)
    for item in arch_categories["central_fragmented_arches"]:
        cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["arch_central_fragment"], 2)
    for item in arch_categories["upper_outlier_arches"]:
        cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["arch_upper_outlier"], 2)

    for family_record in pilaster_families:
        for item in family_record["members_intact"]:
            cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["pilaster_intact"], 2)
        for item in family_record["members_fragmented"]:
            cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["pilaster_fragmented"], 2)

    for item in openings:
        cv2.rectangle(canvas, tuple(item["bbox_xyxy"][:2]), tuple(item["bbox_xyxy"][2:]), COLORS["opening"], 2)
    return canvas


# =============================================================================
# SERIALIZATION CLEANUP
# =============================================================================
def without_raw(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_raw(item) for key, item in value.items() if key != "raw"}
    if isinstance(value, list):
        return [without_raw(item) for item in value]
    return value


# =============================================================================
# MAIN
# =============================================================================
def run(cfg: Config) -> Path:
    parent = find_parent_run(cfg)
    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started = time.perf_counter()
    meta_path = run_dir / "run_meta.json"

    source_script = Path(__file__)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage3b_v3_family_cleanup",
        "status": "running",
        "started_at": now_iso(),
        "parent_run": {
            "run_id": parent.name,
            "path": str(parent.resolve()),
        },
        "source_script": {
            "filename": source_script.name,
            "path": str(source_script.resolve()),
            "sha256": sha256_file(source_script),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "config": asdict(cfg),
        "safety": {
            "new_objects_created": 0,
            "new_axes_created": 0,
            "missing_elements_inferred": 0,
            "prototype_texture_created": 0,
            "note": "Yalnız mevcut kutular temizlenmiş ve ailelere ayrılmıştır.",
        },
        "copied_inputs": {},
        "summary": {},
        "outputs": [],
    }

    def flush_meta() -> None:
        write_json(meta_path, metadata)

    try:
        logger.info("=" * 78)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent.resolve()}")
        logger.info("[NEW RUN] Önceki klasörün üzerine yazılmayacak")
        logger.info("[SAFE] Yeni kemer, pilaster, eksen veya kayıp eleman oluşturulmayacak")
        logger.info("[STAGE] Stage 3B-v3 — aile temizleme ve güvenilir ritim kontrolü")
        logger.info("=" * 78)

        metadata["copied_inputs"] = copy_parent_inputs(parent, run_dir, cfg)
        flush_meta()

        original = read_rgb(run_dir / "input.png")
        height, width = original.shape[:2]
        completion_mask = load_binary_mask(run_dir / "completion_mask.png", (width, height))
        damage_component = largest_damage_component(completion_mask)

        validated_document = read_json(run_dir / "source_validated_elements.json")
        tentative_document = read_json(run_dir / "source_tentative_elements.json")
        elements = normalize_elements(
            validated_document,
            tentative_document,
            completion_mask,
            width,
            height,
        )
        all_elements_by_id = {str(item["id"]): item for item in elements}

        arches = [item for item in elements if item["class"] == "arch" and float(item["score"]) >= cfg.min_arch_score]
        pilasters = [item for item in elements if item["class"] == "pilaster" and float(item["score"]) >= cfg.min_pilaster_score]
        raw_openings = [
            item for item in elements
            if item["class"] in ("window", "door") and float(item["score"]) >= cfg.min_opening_score
        ]

        clean_arches, arch_duplicate_relations = cleanup_arch_duplicates(arches, width, height, cfg)
        clean_pilasters, pilaster_duplicate_relations = cleanup_general_duplicates(pilasters, cfg)
        clean_openings, opening_alias_relations = cleanup_opening_aliases(raw_openings)
        duplicate_relations = [
            *arch_duplicate_relations,
            *pilaster_duplicate_relations,
            *opening_alias_relations,
        ]

        logger.info(
            f"[CLEANUP] arch={len(arches)}->{len(clean_arches)}, "
            f"pilaster={len(pilasters)}->{len(clean_pilasters)}, "
            f"opening={len(raw_openings)}->{len(clean_openings)}, "
            f"removed_or_merged={len(duplicate_relations)}"
        )

        arch_families, arch_categories = build_arch_family_cleanup(
            clean_arches,
            damage_component,
            width,
            height,
            cfg,
        )
        pilaster_families = build_pilaster_family(clean_pilasters, width, height, cfg)

        arch_rhythm = analyze_family_rhythm(arch_families[0], cfg)
        pilaster_rhythm = analyze_family_rhythm(pilaster_families[0], cfg)
        rhythm_document = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "policy": {
                "unique_intact_members_only": True,
                "unreliable_mean_spacing_is_suppressed": True,
                "missing_element_inference_created": 0,
            },
            "arch_rhythm": arch_rhythm,
            "pilaster_rhythm": pilaster_rhythm,
        }

        diagnostics = draw_cleanup_diagnostics(
            original,
            arch_categories,
            pilaster_families,
            clean_openings,
            duplicate_relations,
            all_elements_by_id,
            cfg,
        )
        clean_overlay = draw_clean_families(
            original,
            arch_categories,
            pilaster_families,
            clean_openings,
            cfg,
        )
        schematic = draw_family_schematic(
            original.shape,
            arch_categories,
            pilaster_families,
            clean_openings,
            cfg,
        )

        save_rgb(run_dir / "stage3b_v3_cleanup_diagnostics.png", diagnostics)
        save_rgb(run_dir / "stage3b_v3_clean_families.png", clean_overlay)
        save_rgb(run_dir / "stage3b_v3_family_schematic.png", schematic)

        arch_output = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "damage_component": damage_component,
            "families": without_raw(arch_families),
            "central_fragmented_arches": without_raw(arch_categories["central_fragmented_arches"]),
            "upper_outlier_arches": without_raw(arch_categories["upper_outlier_arches"]),
            "other_arch_candidates": without_raw(arch_categories["other_arch_candidates"]),
            "springline_clusters": arch_categories["springline_clusters"],
        }
        pilaster_output = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "families": without_raw(pilaster_families),
        }
        openings_output = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "openings": without_raw(clean_openings),
        }

        write_json(run_dir / "stage3b_v3_arch_families.json", arch_output)
        write_json(run_dir / "stage3b_v3_pilaster_families.json", pilaster_output)
        write_json(run_dir / "stage3b_v3_openings.json", openings_output)
        write_json(run_dir / "stage3b_v3_duplicate_relations.json", duplicate_relations)
        write_json(run_dir / "stage3b_v3_rhythm_analysis.json", rhythm_document)

        primary_arch_family = arch_families[0]
        primary_pilaster_family = pilaster_families[0]
        report_lines = [
            "STAGE 3B-v3 — FAMILY CLEANUP REPORT",
            f"Run: {run_id}",
            f"Parent: {parent.resolve()}",
            "",
            "GÜVENLİK:",
            "- Yeni kemer/pilaster/eksen oluşturulmadı.",
            "- Kayıp eleman tahmini yapılmadı.",
            "- Yalnız mevcut nesne kutuları temizlendi ve sınıflandırıldı.",
            "",
            "KAYNAK VE TEMİZLİK:",
            f"- Kaynak eleman: {len(elements)}",
            f"- Kaynak kemer: {len(arches)} -> temiz benzersiz kemer: {len(clean_arches)}",
            f"- Kaynak pilaster: {len(pilasters)} -> temiz benzersiz pilaster: {len(clean_pilasters)}",
            f"- Kaynak açıklık: {len(raw_openings)} -> birleşik açıklık: {len(clean_openings)}",
            f"- Kaldırılan/birleştirilen alternatif kutu: {len(duplicate_relations)}",
            "",
            "KEMER AİLESİ:",
            f"- Aile: {primary_arch_family['family_id']}",
            f"- Sağlam prototip üyesi: {primary_arch_family['intact_count']}",
            f"- Aileye atanmış parçalı üye: {primary_arch_family['fragmented_assigned_count']}",
            f"- Merkez ağır hasarlı kemer adayı: {len(arch_categories['central_fragmented_arches'])}",
            f"- Üst outlier: {len(arch_categories['upper_outlier_arches'])}",
            "- Prototip yalnız aynı kotta bulunan benzersiz düşük-hasarlı kutulardan hesaplandı.",
            "",
            "PİLASTER AİLESİ:",
            f"- Sağlam prototip üyesi: {primary_pilaster_family['intact_count']}",
            f"- Parçalı aile üyesi: {primary_pilaster_family['fragmented_count']}",
            "",
            "RİTİM:",
            f"- Kemer ritim durumu: {arch_rhythm['rhythm_status']}",
            f"- Kemer gözlenen aralıkları: {arch_rhythm['observed_adjacent_spacings_px']}",
            f"- Güvenilir kemer aralığı: {arch_rhythm['trusted_spacing_px']}",
            f"- Pilaster ritim durumu: {pilaster_rhythm['rhythm_status']}",
            f"- Güvenilir pilaster aralığı: {pilaster_rhythm['trusted_spacing_px']}",
            "",
            "SONUÇ:",
            "- Ritmin güvenilir olmadığı durumda ortalama aralık restorasyon girdisi olarak kaydedilmez.",
            "- Stage 3C ancak temiz aile prototipi ve kullanıcı incelemesinden sonra hazırlanmalıdır.",
        ]
        (run_dir / "stage3b_v3_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

        metadata["summary"] = {
            "source_element_count": len(elements),
            "source_arch_count": len(arches),
            "clean_unique_arch_count": len(clean_arches),
            "removed_or_merged_relation_count": len(duplicate_relations),
            "clean_arch_prototype_member_count": int(primary_arch_family["intact_count"]),
            "central_fragmented_arch_count": len(arch_categories["central_fragmented_arches"]),
            "upper_outlier_arch_count": len(arch_categories["upper_outlier_arches"]),
            "clean_pilaster_prototype_member_count": int(primary_pilaster_family["intact_count"]),
            "fragmented_pilaster_member_count": int(primary_pilaster_family["fragmented_count"]),
            "arch_rhythm_status": arch_rhythm["rhythm_status"],
            "arch_trusted_spacing_px": arch_rhythm["trusted_spacing_px"],
            "new_objects_created": 0,
            "missing_elements_inferred": 0,
        }
        metadata["outputs"] = [
            {"file": "stage3b_v3_cleanup_diagnostics.png", "type": "cleanup_diagnostics"},
            {"file": "stage3b_v3_clean_families.png", "type": "clean_family_overlay"},
            {"file": "stage3b_v3_family_schematic.png", "type": "clean_family_schematic"},
            {"file": "stage3b_v3_arch_families.json", "type": "clean_arch_families"},
            {"file": "stage3b_v3_pilaster_families.json", "type": "clean_pilaster_families"},
            {"file": "stage3b_v3_openings.json", "type": "merged_openings"},
            {"file": "stage3b_v3_duplicate_relations.json", "type": "removed_and_merged_relations"},
            {"file": "stage3b_v3_rhythm_analysis.json", "type": "reliability_checked_rhythm"},
            {"file": "stage3b_v3_report.txt", "type": "human_readable_report"},
            {"file": "run.log", "type": "log"},
            {"file": "run_meta.json", "type": "metadata"},
        ]
        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush_meta()

        logger.info("[OK] Stage 3B-v3 tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            f"[COUNT] clean_arch_family_members={primary_arch_family['intact_count']}, "
            f"central_fragment={len(arch_categories['central_fragmented_arches'])}, "
            f"upper_outlier={len(arch_categories['upper_outlier_arches'])}, "
            f"removed_or_merged={len(duplicate_relations)}, "
            "new_objects=0, missing_inference=0"
        )
        logger.info(
            "[FILES] stage3b_v3_clean_families.png, "
            "stage3b_v3_duplicate_relations.json, stage3b_v3_rhythm_analysis.json"
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
        flush_meta()
        logger.error(f"[FAIL] {exc}")
        logger.error(traceback.format_exc())
        raise


def main() -> None:
    run_dir = run(Config())
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()