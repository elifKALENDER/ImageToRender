#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3b_element_families.py
========================================================
V8 / STAGE 3B — Mevcut mimari elemanları ailelere ayırma ve ritim analizi

Amaç
-----
Stage 3A'nın doğrulanmış/geçici mevcut kanıtlarını kullanarak:
- alt cephedeki benzer kemerleri ailelere ayırmak,
- pilaster/sütun adaylarını ailelere ayırmak,
- her ailede ölçü, kot, oran ve merkez aralıklarını hesaplamak,
- tekil ve ölçek/kot bakımından aykırı tespitleri ayrı tutmak,
- üstteki büyük şüpheli kemeri alt kattaki tekrar ailesine karıştırmamak.

KESİN GÜVENLİK KURALLARI
------------------------
- Yeni kemer, pilaster, sütun, pencere, kapı veya eksen ÜRETMEZ.
- İki eleman arasına yeni üye EKLEMEZ.
- Kayıp eleman veya restorasyon hipotezi ÜRETMEZ.
- Yalnız Stage 3A'da zaten bulunan mevcut elemanları gruplandırır.
- Her çalıştırmada yeni outputs/restore_YYYYMMDD_HHMMSS klasörü açar.
- Önceki run klasörlerinin üzerine yazmaz.

Çıktılar
--------
input.png
completion_mask.png
source_stage3a_existing_evidence_skeleton.json
source_validated_elements.json                    varsa
source_tentative_elements.json                    varsa
source_stage2d_rhythm_groups.json                 varsa
stage3b_element_families.png
stage3b_family_schematic.png
stage3b_arch_families.json
stage3b_pilaster_families.json
stage3b_opening_families.json
stage3b_rhythm_analysis.json
stage3b_family_report.txt
run.log
run_meta.json

Bu aşamada Stable Diffusion / ControlNet / inpainting YOKTUR.
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

    # Stage 3A mevcut kanıtlarını yeniden elemeden önce yinelenen kutuları azalt.
    duplicate_iou_threshold: float = 0.56
    duplicate_center_tolerance_ratio: float = 0.030
    duplicate_size_similarity_min: float = 0.64

    # Kemer ailesi uyumluluğu.
    arch_center_y_tolerance_ratio: float = 0.145
    arch_width_ratio_min: float = 0.43
    arch_width_ratio_max: float = 2.35
    arch_height_ratio_min: float = 0.40
    arch_height_ratio_max: float = 2.50
    arch_pair_distance_max: float = 1.36

    # Pilaster/sütun ailesi uyumluluğu.
    pilaster_center_y_tolerance_ratio: float = 0.185
    pilaster_width_ratio_min: float = 0.34
    pilaster_width_ratio_max: float = 2.90
    pilaster_height_ratio_min: float = 0.40
    pilaster_height_ratio_max: float = 2.50
    pilaster_pair_distance_max: float = 1.48

    # Pencere/kapı ailesi uyumluluğu.
    opening_center_y_tolerance_ratio: float = 0.19
    opening_pair_distance_max: float = 1.45

    # Aile ve ritim değerlendirmesi.
    repeated_family_min_members: int = 2
    rhythm_cv_good: float = 0.24
    rhythm_cv_acceptable: float = 0.42
    singleton_outlier_level_gap_ratio: float = 0.15
    singleton_outlier_size_ratio: float = 1.75

    # Görselleştirme.
    damage_overlay_alpha: float = 0.10
    draw_member_ids: bool = True
    family_box_thickness: int = 2


# =============================================================================
# PATH / LOG / JSON
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
    logger = logging.getLogger(f"stage3b_families_{run_id}")
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
        return parent

    root = Path(cfg.output_root)
    candidates = sorted(
        (path for path in root.glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if (
            (candidate / "input.png").exists()
            and (candidate / "completion_mask.png").exists()
            and (candidate / "stage3a_existing_evidence_skeleton.json").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "stage3a_existing_evidence_skeleton.json içeren uygun restore_* "
        f"klasörü bulunamadı: {root.resolve()}"
    )


def copy_parent_inputs(parent: Path, run_dir: Path) -> dict[str, Any]:
    required = {
        "input.png": "input.png",
        "completion_mask.png": "completion_mask.png",
        "stage3a_existing_evidence_skeleton.json":
            "source_stage3a_existing_evidence_skeleton.json",
    }
    optional = {
        "source_validated_elements.json": "source_validated_elements.json",
        "source_tentative_elements.json": "source_tentative_elements.json",
        "source_stage2d_rhythm_groups.json": "source_stage2d_rhythm_groups.json",
        "stage3a_skeleton_overlay.png": "source_stage3a_skeleton_overlay.png",
        "stage3a_existing_evidence_overlay.png":
            "source_stage3a_existing_evidence_overlay.png",
    }

    copied: dict[str, Any] = {"required": {}, "optional": {}}
    for source_name, target_name in required.items():
        source = parent / source_name
        if not source.exists():
            raise FileNotFoundError(f"Parent girdi eksik: {source.resolve()}")
        target = run_dir / target_name
        shutil.copy2(source, target)
        copied["required"][source_name] = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "run_copy": target_name,
            "run_copy_sha256": sha256_file(target),
        }

    for source_name, target_name in optional.items():
        source = parent / source_name
        if not source.exists():
            continue
        target = run_dir / target_name
        shutil.copy2(source, target)
        copied["optional"][source_name] = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "run_copy": target_name,
            "run_copy_sha256": sha256_file(target),
        }
    return copied


# =============================================================================
# GEOMETRY / NORMALIZATION
# =============================================================================
def clamp_box(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def safe_ratio(a: float, b: float) -> float:
    small = max(1e-6, min(abs(a), abs(b)))
    large = max(abs(a), abs(b))
    return large / small


def size_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    width_similarity = min(a["width"], b["width"]) / max(1.0, max(a["width"], b["width"]))
    height_similarity = min(a["height"], b["height"]) / max(1.0, max(a["height"], b["height"]))
    return float(math.sqrt(width_similarity * height_similarity))


def canonical_class(label: str) -> str:
    label = str(label).strip().lower()
    if label in ("pilaster", "column"):
        return "pilaster"
    if label in ("window", "door"):
        return label
    return label


def normalize_stage3a_guides(
    document: dict[str, Any],
    damage_mask: np.ndarray,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    raw_guides = document.get("existing_guides", document.get("guides", []))
    normalized: list[dict[str, Any]] = []

    for index, guide in enumerate(raw_guides):
        if not isinstance(guide, dict):
            continue
        label = canonical_class(str(guide.get("class", "unknown")))
        if label not in ("arch", "pilaster", "window", "door"):
            continue
        box = guide.get("bbox_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        box = clamp_box(box, width, height)
        x1, y1, x2, y2 = box
        box_mask = damage_mask[y1:y2, x1:x2] > 0
        recomputed_damage = float(np.count_nonzero(box_mask)) / max(1, box_mask.size)

        score_value = guide.get("score", 0.0)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        source_group = str(guide.get("source_group", "unknown")).lower()
        status_priority = 2 if source_group == "validated" else 1
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        box_width = float(x2 - x1)
        box_height = float(y2 - y1)

        normalized.append(
            {
                "source_index": index,
                "source_id": guide.get(
                    "source_id",
                    guide.get("id", f"{label}_{index + 1:03d}"),
                ),
                "class": label,
                "source_group": source_group,
                "status_priority": status_priority,
                "score": round(score, 6),
                "bbox_xyxy": box,
                "center_x": round(center_x, 3),
                "center_y": round(center_y, 3),
                "width": round(box_width, 3),
                "height": round(box_height, 3),
                "aspect_ratio": round(box_width / max(1.0, box_height), 6),
                "slenderness": round(box_height / max(1.0, box_width), 6),
                "area_ratio": round((box_width * box_height) / max(1.0, width * height), 8),
                "damage_fraction": round(recomputed_damage, 6),
                "supporting_axis_id": guide.get("supporting_axis_id"),
                "supporting_axis_x": guide.get("supporting_axis_x"),
                "raw_status": guide.get("raw_status"),
            }
        )
    return normalized


def deduplicate_elements(
    elements: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagonal = math.hypot(image_width, image_height)
    ordered = sorted(
        elements,
        key=lambda item: (
            int(item["status_priority"]),
            float(item["score"]),
        ),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate_of: dict[str, Any] | None = None
        for accepted in kept:
            if candidate["class"] != accepted["class"]:
                continue
            iou = box_iou(candidate["bbox_xyxy"], accepted["bbox_xyxy"])
            center_distance = math.hypot(
                float(candidate["center_x"]) - float(accepted["center_x"]),
                float(candidate["center_y"]) - float(accepted["center_y"]),
            )
            similarity = size_similarity(candidate, accepted)
            if (
                iou >= cfg.duplicate_iou_threshold
                or (
                    center_distance <= cfg.duplicate_center_tolerance_ratio * diagonal
                    and similarity >= cfg.duplicate_size_similarity_min
                )
            ):
                duplicate_of = accepted
                break

        if duplicate_of is None:
            kept.append(candidate)
        else:
            removed.append(
                {
                    "source_id": candidate["source_id"],
                    "class": candidate["class"],
                    "duplicate_of": duplicate_of["source_id"],
                    "reason": "overlapping_or_near_identical_existing_detection",
                }
            )

    kept.sort(key=lambda item: (item["class"], float(item["center_x"])))
    return kept, removed


# =============================================================================
# FAMILY CLUSTERING
# =============================================================================
class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def logarithmic_difference(a: float, b: float) -> float:
    return abs(math.log(max(1e-6, a) / max(1e-6, b)))


def arch_pair_compatible(
    a: dict[str, Any],
    b: dict[str, Any],
    image_height: int,
    cfg: Config,
) -> tuple[bool, float]:
    width_ratio = safe_ratio(float(a["width"]), float(b["width"]))
    height_ratio = safe_ratio(float(a["height"]), float(b["height"]))
    center_y_gap = abs(float(a["center_y"]) - float(b["center_y"])) / image_height
    aspect_gap = logarithmic_difference(
        float(a["aspect_ratio"]),
        float(b["aspect_ratio"]),
    )

    if not (cfg.arch_width_ratio_min <= 1.0 / width_ratio <= 1.0 or width_ratio <= cfg.arch_width_ratio_max):
        return False, 999.0
    if width_ratio > cfg.arch_width_ratio_max:
        return False, 999.0
    if height_ratio > cfg.arch_height_ratio_max:
        return False, 999.0
    if center_y_gap > cfg.arch_center_y_tolerance_ratio:
        return False, 999.0

    distance = (
        0.92 * logarithmic_difference(float(a["width"]), float(b["width"]))
        + 0.78 * logarithmic_difference(float(a["height"]), float(b["height"]))
        + 2.25 * center_y_gap
        + 0.52 * aspect_gap
    )
    return distance <= cfg.arch_pair_distance_max, distance


def pilaster_pair_compatible(
    a: dict[str, Any],
    b: dict[str, Any],
    image_height: int,
    cfg: Config,
) -> tuple[bool, float]:
    width_ratio = safe_ratio(float(a["width"]), float(b["width"]))
    height_ratio = safe_ratio(float(a["height"]), float(b["height"]))
    center_y_gap = abs(float(a["center_y"]) - float(b["center_y"])) / image_height
    slender_gap = logarithmic_difference(
        float(a["slenderness"]),
        float(b["slenderness"]),
    )

    if width_ratio > cfg.pilaster_width_ratio_max:
        return False, 999.0
    if height_ratio > cfg.pilaster_height_ratio_max:
        return False, 999.0
    if center_y_gap > cfg.pilaster_center_y_tolerance_ratio:
        return False, 999.0

    distance = (
        1.00 * logarithmic_difference(float(a["width"]), float(b["width"]))
        + 0.62 * logarithmic_difference(float(a["height"]), float(b["height"]))
        + 1.85 * center_y_gap
        + 0.58 * slender_gap
    )
    return distance <= cfg.pilaster_pair_distance_max, distance


def opening_pair_compatible(
    a: dict[str, Any],
    b: dict[str, Any],
    image_height: int,
    cfg: Config,
) -> tuple[bool, float]:
    center_y_gap = abs(float(a["center_y"]) - float(b["center_y"])) / image_height
    if center_y_gap > cfg.opening_center_y_tolerance_ratio:
        return False, 999.0
    distance = (
        0.90 * logarithmic_difference(float(a["width"]), float(b["width"]))
        + 0.75 * logarithmic_difference(float(a["height"]), float(b["height"]))
        + 1.85 * center_y_gap
    )
    return distance <= cfg.opening_pair_distance_max, distance


def cluster_class_elements(
    elements: list[dict[str, Any]],
    class_name: str,
    image_height: int,
    cfg: Config,
) -> list[list[dict[str, Any]]]:
    members = [item for item in elements if item["class"] == class_name]
    if not members:
        return []

    union_find = UnionFind(len(members))
    for first_index in range(len(members)):
        for second_index in range(first_index + 1, len(members)):
            first = members[first_index]
            second = members[second_index]
            if class_name == "arch":
                compatible, _ = arch_pair_compatible(first, second, image_height, cfg)
            elif class_name == "pilaster":
                compatible, _ = pilaster_pair_compatible(first, second, image_height, cfg)
            else:
                compatible, _ = opening_pair_compatible(first, second, image_height, cfg)
            if compatible:
                union_find.union(first_index, second_index)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, member in enumerate(members):
        groups.setdefault(union_find.find(index), []).append(member)

    clustered = list(groups.values())
    for group in clustered:
        group.sort(key=lambda item: float(item["center_x"]))
    clustered.sort(
        key=lambda group: (
            -len(group),
            float(np.mean([float(item["center_y"]) for item in group])),
            float(np.mean([float(item["center_x"]) for item in group])),
        )
    )
    return clustered


# =============================================================================
# FAMILY STATISTICS / OUTLIERS / RHYTHM
# =============================================================================
def robust_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "mad": None,
            "min": None,
            "max": None,
            "coefficient_of_variation": None,
        }
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    median = float(np.median(array))
    std = float(np.std(array))
    mad = float(np.median(np.abs(array - median)))
    coefficient = std / abs(mean) if abs(mean) > 1e-8 else None
    return {
        "count": len(values),
        "mean": round(mean, 6),
        "median": round(median, 6),
        "std": round(std, 6),
        "mad": round(mad, 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "coefficient_of_variation": None if coefficient is None else round(coefficient, 6),
    }


def linear_trend(xs: list[float], ys: list[float], image_width: int) -> dict[str, Any]:
    if len(xs) < 2 or len(set(round(value, 4) for value in xs)) < 2:
        return {
            "available": False,
            "slope_per_image_width": None,
            "intercept": None,
            "r_squared": None,
        }
    x_array = np.asarray(xs, dtype=np.float64) / max(1.0, float(image_width))
    y_array = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(x_array, y_array, 1)
    predicted = slope * x_array + intercept
    residual = float(np.sum((y_array - predicted) ** 2))
    total = float(np.sum((y_array - np.mean(y_array)) ** 2))
    r_squared = 1.0 - residual / total if total > 1e-10 else 1.0
    return {
        "available": True,
        "slope_per_image_width": round(float(slope), 6),
        "intercept": round(float(intercept), 6),
        "r_squared": round(float(r_squared), 6),
    }


def rhythm_quality(spacings: list[float], cfg: Config) -> tuple[str, float]:
    if len(spacings) < 1:
        return "insufficient_members", 0.0
    mean = float(np.mean(spacings))
    std = float(np.std(spacings))
    cv = std / mean if mean > 1e-8 else 1.0
    if cv <= cfg.rhythm_cv_good:
        label = "regular"
    elif cv <= cfg.rhythm_cv_acceptable:
        label = "approximately_regular"
    else:
        label = "irregular_or_perspective_affected"
    score = float(np.clip(math.exp(-2.8 * cv), 0.0, 1.0))
    return label, round(score, 6)


def build_family_record(
    family_id: str,
    class_name: str,
    members: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> dict[str, Any]:
    members = sorted(members, key=lambda item: float(item["center_x"]))
    centers_x = [float(item["center_x"]) for item in members]
    centers_y = [float(item["center_y"]) for item in members]
    widths = [float(item["width"]) for item in members]
    heights = [float(item["height"]) for item in members]
    aspects = [float(item["aspect_ratio"]) for item in members]
    damages = [float(item["damage_fraction"]) for item in members]
    scores = [float(item["score"]) for item in members]
    spacings = [centers_x[index + 1] - centers_x[index] for index in range(len(centers_x) - 1)]
    rhythm_label, rhythm_score = rhythm_quality(spacings, cfg)

    family_state = (
        "repeated_family"
        if len(members) >= cfg.repeated_family_min_members
        else "singleton_family"
    )
    damage_state = (
        "mostly_fragmented"
        if damages and float(np.mean(damages)) >= 0.30
        else "partly_fragmented"
        if damages and float(np.max(damages)) >= 0.12
        else "mostly_visible"
    )

    return {
        "family_id": family_id,
        "class": class_name,
        "member_count": len(members),
        "family_state": family_state,
        "damage_state": damage_state,
        "member_ids": [item["source_id"] for item in members],
        "members": members,
        "statistics": {
            "center_x": robust_stats(centers_x),
            "center_y": robust_stats(centers_y),
            "width": robust_stats(widths),
            "height": robust_stats(heights),
            "aspect_ratio": robust_stats(aspects),
            "damage_fraction": robust_stats(damages),
            "score": robust_stats(scores),
        },
        "perspective_trends": {
            "width_by_x": linear_trend(centers_x, widths, image_width),
            "height_by_x": linear_trend(centers_x, heights, image_width),
            "center_y_by_x": linear_trend(centers_x, centers_y, image_width),
        },
        "rhythm": {
            "centers_x": [round(value, 3) for value in centers_x],
            "adjacent_spacings_px": [round(value, 3) for value in spacings],
            "spacing_statistics": robust_stats(spacings),
            "quality": rhythm_label,
            "regularity_score": rhythm_score,
        },
        "image_normalized": {
            "mean_center_y_ratio": round(float(np.mean(centers_y)) / image_height, 6),
            "median_width_ratio": round(float(np.median(widths)) / image_width, 6),
            "median_height_ratio": round(float(np.median(heights)) / image_height, 6),
        },
        "outlier_status": "not_evaluated",
        "outlier_reasons": [],
    }


def identify_family_outliers(
    families: list[dict[str, Any]],
    image_height: int,
    cfg: Config,
) -> None:
    repeated = [
        family for family in families
        if int(family["member_count"]) >= cfg.repeated_family_min_members
    ]
    if not families:
        return

    primary: dict[str, Any] | None = None
    if repeated:
        primary = sorted(
            repeated,
            key=lambda family: (
                -int(family["member_count"]),
                float(family["statistics"]["center_y"]["std"] or 0.0),
                -float(family["rhythm"]["regularity_score"]),
            ),
        )[0]

    for family in families:
        reasons: list[str] = []
        if family is primary:
            family["outlier_status"] = "primary_repeated_family"
            family["outlier_reasons"] = []
            continue

        if primary is None:
            family["outlier_status"] = (
                "singleton_unresolved"
                if int(family["member_count"]) == 1
                else "repeated_family_without_primary_reference"
            )
            family["outlier_reasons"] = []
            continue

        family_center_y = float(family["statistics"]["center_y"]["median"])
        primary_center_y = float(primary["statistics"]["center_y"]["median"])
        level_gap = abs(family_center_y - primary_center_y) / image_height

        family_width = float(family["statistics"]["width"]["median"])
        primary_width = float(primary["statistics"]["width"]["median"])
        family_height = float(family["statistics"]["height"]["median"])
        primary_height = float(primary["statistics"]["height"]["median"])
        size_ratio = max(
            safe_ratio(family_width, primary_width),
            safe_ratio(family_height, primary_height),
        )

        if level_gap >= cfg.singleton_outlier_level_gap_ratio:
            reasons.append("different_vertical_level_from_primary_repeated_family")
        if size_ratio >= cfg.singleton_outlier_size_ratio:
            reasons.append("different_scale_from_primary_repeated_family")

        if int(family["member_count"]) == 1 and reasons:
            family["outlier_status"] = "isolated_outlier_candidate"
        elif reasons:
            family["outlier_status"] = "separate_family"
        else:
            family["outlier_status"] = "compatible_secondary_family"
        family["outlier_reasons"] = reasons


def build_families_for_class(
    elements: list[dict[str, Any]],
    class_name: str,
    prefix: str,
    image_width: int,
    image_height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    groups = cluster_class_elements(elements, class_name, image_height, cfg)
    families = [
        build_family_record(
            family_id=f"{prefix}_{index + 1:02d}",
            class_name=class_name,
            members=members,
            image_width=image_width,
            image_height=image_height,
            cfg=cfg,
        )
        for index, members in enumerate(groups)
    ]
    identify_family_outliers(families, image_height, cfg)
    return families


# =============================================================================
# VISUALIZATION
# =============================================================================
PALETTE = [
    (55, 220, 80),
    (60, 170, 255),
    (255, 190, 55),
    (220, 85, 220),
    (70, 225, 220),
    (255, 105, 80),
    (150, 110, 255),
    (120, 215, 120),
]


def overlay_damage(image: np.ndarray, damage_mask: np.ndarray, alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    red = np.zeros_like(output)
    red[..., 0] = 255
    mask = (damage_mask > 0)[..., None].astype(np.float32)
    output = output * (1.0 - alpha * mask) + red * (alpha * mask)
    return np.clip(output, 0, 255).astype(np.uint8)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 35), (20, 20, 20), -1)
    cv2.putText(
        output,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def all_families_with_colors(
    arch_families: list[dict[str, Any]],
    pilaster_families: list[dict[str, Any]],
    opening_families: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], tuple[int, int, int]]]:
    result: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    index = 0
    for family in [*arch_families, *pilaster_families, *opening_families]:
        result.append((family, PALETTE[index % len(PALETTE)]))
        index += 1
    return result


def draw_family_overlay(
    original: np.ndarray,
    damage_mask: np.ndarray,
    family_color_pairs: list[tuple[dict[str, Any], tuple[int, int, int]]],
    cfg: Config,
) -> np.ndarray:
    image = overlay_damage(original, damage_mask, cfg.damage_overlay_alpha)
    image = add_header(
        image,
        "STAGE 3B EXISTING ELEMENT FAMILIES - NO COMPLETION / NO NEW OBJECTS",
    )

    for family, color in family_color_pairs:
        status = str(family["outlier_status"])
        draw_color = (160, 160, 160) if status == "isolated_outlier_candidate" else color
        thickness = cfg.family_box_thickness
        for member_index, member in enumerate(family["members"]):
            x1, y1, x2, y2 = [int(value) for value in member["bbox_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), draw_color, thickness)
            label = family["family_id"]
            if cfg.draw_member_ids:
                label += f" {member['source_id']}"
            cv2.putText(
                image,
                label,
                (x1 + 2, max(46, y1 - 5 - (member_index % 2) * 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.37,
                draw_color,
                1,
                cv2.LINE_AA,
            )

        centers = [
            (int(round(float(member["center_x"]))), int(round(float(member["center_y"]))))
            for member in family["members"]
        ]
        if len(centers) >= 2:
            cv2.polylines(
                image,
                [np.asarray(centers, dtype=np.int32)],
                False,
                draw_color,
                1,
                cv2.LINE_AA,
            )

        if family["members"]:
            first = family["members"][0]
            x = int(round(float(first["center_x"])))
            y = max(55, int(round(float(first["bbox_xyxy"][1]))) - 21)
            family_label = (
                f"{family['family_id']} n={family['member_count']} "
                f"{family['outlier_status']}"
            )
            cv2.putText(
                image,
                family_label,
                (max(3, min(image.shape[1] - 280, x - 50)), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                draw_color,
                1,
                cv2.LINE_AA,
            )
    return image


def draw_family_schematic(
    image_shape: tuple[int, int, int],
    family_color_pairs: list[tuple[dict[str, Any], tuple[int, int, int]]],
) -> np.ndarray:
    height, width = image_shape[:2]
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    canvas = add_header(
        canvas,
        "FAMILY SCHEMATIC - ONLY OBSERVED MEMBER CENTERS AND BOXES",
    )

    for family, color in family_color_pairs:
        draw_color = (
            (160, 160, 160)
            if family["outlier_status"] == "isolated_outlier_candidate"
            else color
        )
        for member in family["members"]:
            x1, y1, x2, y2 = [int(value) for value in member["bbox_xyxy"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), draw_color, 2)
            center = (int(round(float(member["center_x"]))), int(round(float(member["center_y"]))))
            cv2.circle(canvas, center, 3, draw_color, -1)
        centers = [
            (int(round(float(member["center_x"]))), int(round(float(member["center_y"]))))
            for member in family["members"]
        ]
        if len(centers) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(centers, dtype=np.int32)],
                False,
                draw_color,
                1,
                cv2.LINE_AA,
            )
    return canvas


# =============================================================================
# REPORT
# =============================================================================
def family_report_lines(title: str, families: list[dict[str, Any]]) -> list[str]:
    lines = [title]
    if not families:
        lines.append("- Aile bulunamadı.")
        return lines

    for family in families:
        width = family["statistics"]["width"]
        height = family["statistics"]["height"]
        center_y = family["statistics"]["center_y"]
        rhythm = family["rhythm"]
        lines.append(
            f"- {family['family_id']}: uye={family['member_count']}, "
            f"durum={family['family_state']}, aykiri={family['outlier_status']}, "
            f"genislik_medyan={width['median']}, yukseklik_medyan={height['median']}, "
            f"kot_medyan={center_y['median']}, ritim={rhythm['quality']}, "
            f"ritim_puani={rhythm['regularity_score']:.2f}"
        )
        if family["outlier_reasons"]:
            lines.append(
                "  neden=" + ", ".join(str(reason) for reason in family["outlier_reasons"])
            )
        lines.append("  uyeler=" + ", ".join(family["member_ids"]))
    return lines


# =============================================================================
# MAIN
# =============================================================================
def run(cfg: Config) -> Path:
    parent_run = find_parent_run(cfg)
    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started = time.perf_counter()
    meta_path = run_dir / "run_meta.json"
    source_script = Path(__file__)

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage3b_element_families",
        "status": "running",
        "started_at": now_iso(),
        "parent_run": {
            "run_id": parent_run.name,
            "path": str(parent_run.resolve()),
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
            "restoration_hypotheses_created": 0,
            "note": "Yalnız Stage 3A mevcut elemanları ailelere ayrılmıştır.",
        },
        "summary": {},
        "outputs": [],
    }

    def flush_meta() -> None:
        write_json(meta_path, metadata)

    try:
        logger.info("=" * 78)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info("[NEW RUN] Önceki klasörün üzerine yazılmayacak")
        logger.info("[SAFE] Yeni mimari eleman veya eksen oluşturulmayacak")
        logger.info("[STAGE] Mevcut kemer/pilaster aileleri ve ritim analizi")
        logger.info("=" * 78)

        metadata["copied_inputs"] = copy_parent_inputs(parent_run, run_dir)
        flush_meta()

        original = read_rgb(run_dir / "input.png")
        height, width = original.shape[:2]
        damage_mask = load_binary_mask(
            run_dir / "completion_mask.png",
            (width, height),
        )
        stage3a_document = read_json(
            run_dir / "source_stage3a_existing_evidence_skeleton.json"
        )

        raw_elements = normalize_stage3a_guides(
            stage3a_document,
            damage_mask,
            width,
            height,
        )
        elements, duplicates = deduplicate_elements(
            raw_elements,
            width,
            height,
            cfg,
        )
        logger.info(
            f"[ELEMENTS] kaynak={len(raw_elements)}, "
            f"benzersiz={len(elements)}, yinelenen={len(duplicates)}"
        )

        arch_families = build_families_for_class(
            elements,
            "arch",
            "arch_family",
            width,
            height,
            cfg,
        )
        pilaster_families = build_families_for_class(
            elements,
            "pilaster",
            "pilaster_family",
            width,
            height,
            cfg,
        )
        window_families = build_families_for_class(
            elements,
            "window",
            "window_family",
            width,
            height,
            cfg,
        )
        door_families = build_families_for_class(
            elements,
            "door",
            "door_family",
            width,
            height,
            cfg,
        )
        opening_families = [*window_families, *door_families]

        logger.info(
            f"[FAMILIES] arch={len(arch_families)}, "
            f"pilaster={len(pilaster_families)}, opening={len(opening_families)}"
        )

        arch_document = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "safety": metadata["safety"],
            "class": "arch",
            "families": arch_families,
        }
        pilaster_document = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "safety": metadata["safety"],
            "class": "pilaster",
            "families": pilaster_families,
        }
        opening_document = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "safety": metadata["safety"],
            "classes": ["window", "door"],
            "families": opening_families,
        }

        write_json(run_dir / "stage3b_arch_families.json", arch_document)
        write_json(run_dir / "stage3b_pilaster_families.json", pilaster_document)
        write_json(run_dir / "stage3b_opening_families.json", opening_document)

        all_families = [*arch_families, *pilaster_families, *opening_families]
        rhythm_analysis = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "safety": metadata["safety"],
            "deduplication": {
                "source_element_count": len(raw_elements),
                "unique_element_count": len(elements),
                "removed_duplicates": duplicates,
            },
            "families": [
                {
                    "family_id": family["family_id"],
                    "class": family["class"],
                    "member_count": family["member_count"],
                    "family_state": family["family_state"],
                    "outlier_status": family["outlier_status"],
                    "outlier_reasons": family["outlier_reasons"],
                    "rhythm": family["rhythm"],
                    "perspective_trends": family["perspective_trends"],
                }
                for family in all_families
            ],
        }
        write_json(run_dir / "stage3b_rhythm_analysis.json", rhythm_analysis)

        family_color_pairs = all_families_with_colors(
            arch_families,
            pilaster_families,
            opening_families,
        )
        overlay = draw_family_overlay(
            original,
            damage_mask,
            family_color_pairs,
            cfg,
        )
        schematic = draw_family_schematic(original.shape, family_color_pairs)
        save_rgb(run_dir / "stage3b_element_families.png", overlay)
        save_rgb(run_dir / "stage3b_family_schematic.png", schematic)

        repeated_arch_families = sum(
            1 for family in arch_families
            if family["family_state"] == "repeated_family"
        )
        arch_outliers = sum(
            1 for family in arch_families
            if family["outlier_status"] == "isolated_outlier_candidate"
        )
        repeated_pilaster_families = sum(
            1 for family in pilaster_families
            if family["family_state"] == "repeated_family"
        )

        metadata["summary"] = {
            "source_existing_element_count": len(raw_elements),
            "unique_existing_element_count": len(elements),
            "removed_duplicate_count": len(duplicates),
            "arch_element_count": sum(1 for item in elements if item["class"] == "arch"),
            "pilaster_element_count": sum(1 for item in elements if item["class"] == "pilaster"),
            "window_element_count": sum(1 for item in elements if item["class"] == "window"),
            "door_element_count": sum(1 for item in elements if item["class"] == "door"),
            "arch_family_count": len(arch_families),
            "repeated_arch_family_count": repeated_arch_families,
            "isolated_arch_outlier_family_count": arch_outliers,
            "pilaster_family_count": len(pilaster_families),
            "repeated_pilaster_family_count": repeated_pilaster_families,
            "opening_family_count": len(opening_families),
            "new_objects_created": 0,
            "missing_elements_inferred": 0,
        }

        report_lines = [
            "V8 STAGE 3B — MİMARİ ELEMAN AİLELERİ VE RİTİM RAPORU",
            f"Run: {run_id}",
            f"Parent run: {parent_run.resolve()}",
            "",
            "GÜVENLİK:",
            "- Yeni kemer, pilaster, sütun, pencere, kapı veya eksen oluşturulmadı.",
            "- Kayıp eleman veya restorasyon hipotezi üretilmedi.",
            "- Yalnız Stage 3A'da mevcut olan tespitler ailelere ayrıldı.",
            "",
            f"Stage 3A mevcut eleman: {len(raw_elements)}",
            f"Benzersiz mevcut eleman: {len(elements)}",
            f"Yinelenen kutu olarak ayrılan: {len(duplicates)}",
            f"Kemer ailesi: {len(arch_families)}",
            f"Tekrar eden kemer ailesi: {repeated_arch_families}",
            f"Tekil/aykırı kemer ailesi: {arch_outliers}",
            f"Pilaster ailesi: {len(pilaster_families)}",
            f"Tekrar eden pilaster ailesi: {repeated_pilaster_families}",
            "",
        ]
        report_lines.extend(family_report_lines("KEMER AİLELERİ:", arch_families))
        report_lines.append("")
        report_lines.extend(family_report_lines("PİLASTER AİLELERİ:", pilaster_families))
        report_lines.append("")
        report_lines.extend(family_report_lines("PENCERE/KAPI AİLELERİ:", opening_families))
        report_lines.extend(
            [
                "",
                "YORUM KURALI:",
                "- primary_repeated_family: yapı içinde en güçlü tekrar ailesi.",
                "- isolated_outlier_candidate: ölçek veya kotu ana aileden belirgin farklı tekil tespit.",
                "- Bu etiketler tarihî doğruluk hükmü değildir; sonraki aşama için yapı-içi kanıt düzenidir.",
            ]
        )
        (run_dir / "stage3b_family_report.txt").write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        metadata["outputs"] = [
            {"file": "input.png", "type": "parent_input_copy"},
            {"file": "completion_mask.png", "type": "parent_mask_copy"},
            {
                "file": "source_stage3a_existing_evidence_skeleton.json",
                "type": "stage3a_existing_evidence_copy",
            },
            {"file": "stage3b_element_families.png", "type": "family_overlay"},
            {"file": "stage3b_family_schematic.png", "type": "family_schematic"},
            {"file": "stage3b_arch_families.json", "type": "arch_families"},
            {"file": "stage3b_pilaster_families.json", "type": "pilaster_families"},
            {"file": "stage3b_opening_families.json", "type": "opening_families"},
            {"file": "stage3b_rhythm_analysis.json", "type": "rhythm_and_trends"},
            {"file": "stage3b_family_report.txt", "type": "human_readable_report"},
            {"file": "run.log", "type": "log"},
            {"file": "run_meta.json", "type": "metadata"},
        ]
        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush_meta()

        logger.info("[OK] Stage 3B tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            f"[COUNT] arch_family={len(arch_families)}, "
            f"pilaster_family={len(pilaster_families)}, "
            f"new_objects=0, missing_inference=0"
        )
        logger.info(
            "[FILES] stage3b_element_families.png, "
            "stage3b_arch_families.json, stage3b_rhythm_analysis.json"
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
