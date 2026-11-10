#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage2d_cleaned_axes.py
====================================================
V8 / STAGE 2D — DINOv2 tekrar eksenlerini gözlenen cephe kanıtıyla temizleme

AMAÇ
----
Stage 2C'nin ``repetition_axes.json`` dosyasındaki mevcut eksenleri yeniden
sınıflandırır. Yeni eksen, pilaster, kemer veya modül oluşturmaz.

GİRDİLER (parent run)
---------------------
input.png
completion_mask.png
repetition_axes.json
elements.json                         isteğe bağlı

ÇIKTILAR (yeni restore_* run)
-----------------------------
input.png
completion_mask.png
source_repetition_axes.json
source_elements.json                  varsa
stage2d_facade_mask.png
stage2d_axis_classification.png
stage2d_structural_axes.png
stage2d_rhythm_groups.json
stage2d_report.txt
run.log
run_meta.json

KESİN KURAL
-----------
- Kaynak eksen sayısı değiştirilmez.
- Kaynak eksen konumu değiştirilmez.
- Yeni mimari nesne veya eksen üretilmez.
- Hasarlı boşluğa geometri çizilmez.
- Stable Diffusion / ControlNet kullanılmaz.
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
    parent_run_dir: str = ""  # boşsa repetition_axes.json içeren en yeni run

    # Cephe ROI / maskesi
    roi_fallback_top_ratio: float = 0.10
    roi_fallback_bottom_ratio: float = 0.88
    grabcut_iterations: int = 5
    axis_seed_half_width_px_min: int = 8
    axis_seed_half_width_ratio: float = 0.012
    element_seed_expand_ratio: float = 0.12
    facade_close_kernel: int = 17
    facade_open_kernel: int = 5
    facade_min_component_ratio: float = 0.002
    facade_keep_components: int = 6

    # Eksen şerit ölçümü
    axis_strip_half_width_px_min: int = 9
    axis_strip_half_width_ratio: float = 0.018
    vertical_bins: int = 8
    dark_threshold: int = 82
    canny_low: int = 45
    canny_high: int = 135
    edge_dilate_kernel: int = 3

    # Ritim / sınıflandırma
    dominant_spacing_tolerance_ratio: float = 0.18
    valid_structural_min_score: float = 0.69
    fragmented_structural_min_score: float = 0.57
    weak_axis_min_score: float = 0.43
    opening_score_threshold: float = 0.72
    facade_occupancy_min: float = 0.22
    fragmented_damage_fraction: float = 0.08
    heavy_damage_fraction: float = 0.72

    # Görselleştirme
    overlay_damage_alpha: float = 0.12
    line_thickness: int = 2
    font_scale: float = 0.48


COLORS: dict[str, tuple[int, int, int]] = {
    "valid_structural_axis": (52, 214, 85),
    "fragmented_structural_axis": (255, 190, 45),
    "opening_edge_axis": (76, 170, 255),
    "background_axis": (185, 90, 230),
    "weak_axis": (170, 170, 170),
}


# =============================================================================
# GENERAL / IO
# =============================================================================
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Görsel okunamadı: {path.resolve()}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Görsel kaydedilemedi: {path.resolve()}")


def load_binary_mask(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    width, height = size_wh
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadı: {path.resolve()}")
    if (mask.shape[1], mask.shape[0]) != (width, height):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


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
    logger = logging.getLogger(f"stage2d_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
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
        required = ("input.png", "completion_mask.png", "repetition_axes.json")
        if not parent.is_dir() or not all((parent / name).exists() for name in required):
            raise FileNotFoundError(
                f"Geçerli Stage 2C run değil: {parent.resolve()}"
            )
        return parent

    output_root = Path(cfg.output_root)
    candidates = sorted(
        (path for path in output_root.glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if all(
            (candidate / name).exists()
            for name in ("input.png", "completion_mask.png", "repetition_axes.json")
        ):
            return candidate
    raise FileNotFoundError(
        "input.png, completion_mask.png ve repetition_axes.json içeren "
        f"restore_* bulunamadı: {output_root.resolve()}"
    )


def copy_with_record(source: Path, target: Path) -> dict[str, Any]:
    shutil.copy2(source, target)
    return {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "run_copy": target.name,
        "run_copy_sha256": sha256_file(target),
    }


# =============================================================================
# NORMALIZATION
# =============================================================================
def clip_box(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def source_axes(document: dict[str, Any]) -> list[dict[str, Any]]:
    axes = document.get("axes", [])
    if not isinstance(axes, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(axes):
        if not isinstance(raw, dict):
            continue
        try:
            axis_x = float(raw.get("axis_x"))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                **raw,
                "id": str(raw.get("id") or f"repetition_axis_{index + 1:02d}"),
                "axis_x": axis_x,
                "source_index": index,
            }
        )
    normalized.sort(key=lambda item: float(item["axis_x"]))
    return normalized


def flatten_elements(document: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not document:
        return []
    for key in ("elements", "items", "detections"):
        value = document.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def element_box(item: dict[str, Any]) -> list[float] | None:
    for key in ("bbox_xyxy", "box_xyxy", "bbox", "box"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [float(part) for part in value]
            except (TypeError, ValueError):
                return None
    return None


def resolve_roi(
    axes_document: dict[str, Any],
    damage_mask: np.ndarray,
    width: int,
    height: int,
    cfg: Config,
) -> list[int]:
    raw_roi = axes_document.get("facade_roi")
    if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
        return clip_box(raw_roi, width, height)

    ys, xs = np.where(damage_mask > 0)
    if xs.size:
        x1 = max(0, int(xs.min() - width * 0.28))
        x2 = min(width, int(xs.max() + width * 0.28))
        y1 = max(0, int(ys.min() - height * 0.22))
        y2 = min(height, int(ys.max() + height * 0.14))
        return clip_box([x1, y1, x2, y2], width, height)

    return clip_box(
        [
            0,
            int(height * cfg.roi_fallback_top_ratio),
            width,
            int(height * cfg.roi_fallback_bottom_ratio),
        ],
        width,
        height,
    )


# =============================================================================
# FACADE MASK
# =============================================================================
def make_seed_mask(
    axes: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    roi: list[int],
    damage_mask: np.ndarray,
    width: int,
    height: int,
    cfg: Config,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    seed = np.zeros((height, width), dtype=np.uint8)

    half_width = max(
        cfg.axis_seed_half_width_px_min,
        int(round(width * cfg.axis_seed_half_width_ratio)),
    )
    for axis in axes:
        x = int(round(float(axis["axis_x"])))
        cv2.rectangle(
            seed,
            (max(x1, x - half_width), y1),
            (min(x2 - 1, x + half_width), y2 - 1),
            255,
            -1,
        )

    for item in elements:
        raw_box = element_box(item)
        if raw_box is None:
            continue
        bx1, by1, bx2, by2 = clip_box(raw_box, width, height)
        bw = bx2 - bx1
        bh = by2 - by1
        ex = int(round(bw * cfg.element_seed_expand_ratio))
        ey = int(round(bh * cfg.element_seed_expand_ratio))
        cv2.rectangle(
            seed,
            (max(x1, bx1 - ex), max(y1, by1 - ey)),
            (min(x2 - 1, bx2 + ex), min(y2 - 1, by2 + ey)),
            255,
            -1,
        )

    seed[:y1, :] = 0
    seed[y2:, :] = 0
    seed[:, :x1] = 0
    seed[:, x2:] = 0
    seed[damage_mask > 0] = 0
    return seed


def build_facade_mask(
    rgb: np.ndarray,
    damage_mask: np.ndarray,
    axes: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    roi: list[int],
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = roi
    seed = make_seed_mask(axes, elements, roi, damage_mask, width, height, cfg)

    grab_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grab_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    probable_background = (
        ((saturation > 145) & (value > 35))
        | ((value > 245) & (saturation < 45))
    )
    grab_mask[probable_background] = cv2.GC_PR_BGD
    grab_mask[damage_mask > 0] = cv2.GC_PR_BGD
    grab_mask[seed > 0] = cv2.GC_FGD

    # GrabCut en az bir foreground pikseli ister. Eksen yoksa ROI merkezi seed olur.
    if not np.any(grab_mask == cv2.GC_FGD):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius_x = max(4, (x2 - x1) // 20)
        radius_y = max(4, (y2 - y1) // 20)
        grab_mask[
            max(y1, cy - radius_y):min(y2, cy + radius_y),
            max(x1, cx - radius_x):min(x2, cx + radius_x),
        ] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(
            bgr,
            grab_mask,
            None,
            bg_model,
            fg_model,
            max(1, int(cfg.grabcut_iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
        facade = np.where(
            (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        # Güvenli fallback: ROI + seed; yeni geometri üretmez.
        facade = np.zeros((height, width), dtype=np.uint8)
        facade[y1:y2, x1:x2] = 255

    facade[damage_mask > 0] = 0
    facade[:y1, :] = 0
    facade[y2:, :] = 0
    facade[:, :x1] = 0
    facade[:, x2:] = 0

    close_size = max(3, int(cfg.facade_close_kernel) | 1)
    open_size = max(3, int(cfg.facade_open_kernel) | 1)
    facade = cv2.morphologyEx(
        facade,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    facade = cv2.morphologyEx(
        facade,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )

    binary = (facade > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = int(max(30, width * height * cfg.facade_min_component_ratio))
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            components.append((label, area))
    components.sort(key=lambda item: item[1], reverse=True)
    keep = {label for label, _ in components[: cfg.facade_keep_components]}
    cleaned = np.zeros_like(binary)
    for label in keep:
        cleaned[labels == label] = 1

    # Kaynak eksen çevresindeki seed'ler küçük bileşen olarak atılmasın.
    cleaned[seed > 0] = 1
    cleaned[damage_mask > 0] = 0
    facade = (cleaned * 255).astype(np.uint8)

    roi_area = max(1, (x2 - x1) * (y2 - y1))
    metrics = {
        "roi": roi,
        "seed_pixels": int(np.count_nonzero(seed)),
        "facade_pixels": int(np.count_nonzero(facade)),
        "facade_ratio_in_roi": round(float(np.count_nonzero(facade[y1:y2, x1:x2]) / roi_area), 6),
    }
    return facade, metrics


# =============================================================================
# AXIS METRICS / CLASSIFICATION
# =============================================================================
def longest_run_fraction(binary: np.ndarray) -> float:
    if binary.size == 0:
        return 0.0
    best = 0
    height = binary.shape[0]
    for column in range(binary.shape[1]):
        run = 0
        for value in binary[:, column]:
            if value:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return float(best / max(1, height))


def robust_dominant_spacings(axes: list[dict[str, Any]]) -> list[float]:
    xs = sorted(float(axis["axis_x"]) for axis in axes)
    if len(xs) < 2:
        return []
    differences: list[float] = []
    for i in range(len(xs)):
        for j in range(i + 1, min(len(xs), i + 4)):
            diff = xs[j] - xs[i]
            if diff > 5:
                differences.append(diff)
    if not differences:
        return []

    differences.sort()
    clusters: list[list[float]] = []
    for value in differences:
        placed = False
        for cluster in clusters:
            center = float(np.median(cluster))
            tolerance = max(4.0, center * 0.14)
            if abs(value - center) <= tolerance:
                cluster.append(value)
                placed = True
                break
        if not placed:
            clusters.append([value])

    # Temel mimari ritim çoğu kez daha büyük aralıkların bölenidir. Yalnız
    # yeterince tekrar eden kümeleri al ve en küçük iki güçlü aralığı seç.
    eligible = [cluster for cluster in clusters if len(cluster) >= 3]
    if not eligible:
        eligible = clusters
    eligible.sort(key=lambda cluster: (float(np.median(cluster)), -len(cluster)))
    result: list[float] = []
    for cluster in eligible:
        center = float(np.median(cluster))
        if center >= 10:
            result.append(center)
        if len(result) >= 2:
            break
    return sorted(result)


def spacing_support_for_axis(
    axis_x: float,
    all_x: list[float],
    dominant_spacings: list[float],
    cfg: Config,
) -> tuple[int, int]:
    spacing_support = 0
    family_support = 0
    for other_x in all_x:
        if math.isclose(axis_x, other_x, abs_tol=1e-6):
            continue
        distance = abs(other_x - axis_x)
        for spacing in dominant_spacings:
            tolerance = max(4.0, spacing * cfg.dominant_spacing_tolerance_ratio)
            if abs(distance - spacing) <= tolerance:
                spacing_support += 1
                family_support += 1
                break
            if abs(distance - 2.0 * spacing) <= tolerance * 1.35:
                family_support += 1
                break
    return min(spacing_support, 3), min(family_support, 4)


def measure_axis(
    axis: dict[str, Any],
    rgb: np.ndarray,
    damage_mask: np.ndarray,
    facade_mask: np.ndarray,
    roi: list[int],
    all_x: list[float],
    dominant_spacings: list[float],
    cfg: Config,
) -> dict[str, Any]:
    height, width = rgb.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    axis_x = int(round(float(axis["axis_x"])))
    raw_strip_width = float(axis.get("strip_width_original_px", 0.0) or 0.0)
    half_width = max(
        cfg.axis_strip_half_width_px_min,
        int(round(width * cfg.axis_strip_half_width_ratio)),
        int(round(raw_strip_width / 2.0)),
    )
    x1 = max(rx1, axis_x - half_width)
    x2 = min(rx2, axis_x + half_width + 1)
    y1, y2 = ry1, ry2
    if x2 <= x1 or y2 <= y1:
        x1, x2 = max(0, axis_x - 1), min(width, axis_x + 2)
        y1, y2 = 0, height

    strip_rgb = rgb[y1:y2, x1:x2]
    strip_damage = damage_mask[y1:y2, x1:x2] > 0
    strip_facade = facade_mask[y1:y2, x1:x2] > 0
    visible = ~strip_damage

    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high) > 0
    if cfg.edge_dilate_kernel > 1:
        kernel = np.ones((cfg.edge_dilate_kernel, 1), dtype=np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1) > 0

    visible_count = max(1, int(np.count_nonzero(visible)))
    facade_occupancy = float(np.count_nonzero(strip_facade & visible) / visible_count)
    dark_fraction = float(np.count_nonzero((gray < cfg.dark_threshold) & visible) / visible_count)
    vertical_activity = float(np.count_nonzero(edges & visible) / visible_count)
    longest_run = longest_run_fraction(edges & visible)
    pixel_damage_fraction = float(np.count_nonzero(strip_damage) / max(1, strip_damage.size))
    source_damage_fraction = float(axis.get("damage_fraction", 0.0) or 0.0)
    effective_damage_fraction = max(pixel_damage_fraction, source_damage_fraction)

    visible_vertical_bins = 0
    bin_edges = np.linspace(0, strip_rgb.shape[0], cfg.vertical_bins + 1).astype(int)
    for index in range(cfg.vertical_bins):
        by1, by2 = bin_edges[index], bin_edges[index + 1]
        if by2 <= by1:
            continue
        bin_visible = visible[by1:by2]
        bin_facade = strip_facade[by1:by2]
        if np.mean(bin_visible) >= 0.25 and np.mean(bin_facade & bin_visible) >= 0.10:
            visible_vertical_bins += 1

    spacing_support, family_support = spacing_support_for_axis(
        float(axis["axis_x"]), all_x, dominant_spacings, cfg
    )
    match_support = min(2, len(axis.get("matched_axes") or []))

    source_repetition = float(axis.get("repetition_score", 0.0) or 0.0)
    source_similarity = float(axis.get("best_similarity", 0.0) or 0.0)
    source_prominence = float(axis.get("prominence", 0.0) or 0.0)
    source_state = str(axis.get("state") or "observed_repeated_motif")

    repetition_component = np.clip(
        0.55 * source_repetition + 0.35 * source_similarity + 0.10 * min(1.0, source_prominence * 8.0),
        0.0,
        1.0,
    )
    vertical_component = np.clip(0.65 * vertical_activity * 2.2 + 0.35 * longest_run * 2.0, 0.0, 1.0)
    rhythm_component = np.clip(
        0.30 * match_support + 0.20 * min(1.0, spacing_support / 2.0) + 0.50 * min(1.0, family_support / 3.0),
        0.0,
        1.0,
    )
    structural_score = float(
        np.clip(
            0.40 * repetition_component
            + 0.24 * facade_occupancy
            + 0.18 * vertical_component
            + 0.18 * rhythm_component,
            0.0,
            1.0,
        )
    )

    opening_score = float(
        np.clip(
            0.52 * dark_fraction
            + 0.25 * max(0.0, 0.55 - facade_occupancy) / 0.55
            + 0.23 * max(0.0, vertical_activity - 0.38) / 0.62,
            0.0,
            1.0,
        )
    )

    reasons = ["dinov2_tekrar_kaniti"]
    if facade_occupancy >= cfg.facade_occupancy_min:
        reasons.append("cephe_yuzeyi_destegi")
    if vertical_component >= 0.30:
        reasons.append("dikey_sureklilik")
    if spacing_support or family_support:
        reasons.append("ritim_ailesi_destegi")
    if effective_damage_fraction >= cfg.fragmented_damage_fraction:
        reasons.append("hasar_ile_kesintili")

    if facade_occupancy < cfg.facade_occupancy_min and structural_score < cfg.weak_axis_min_score:
        classification = "background_axis"
        reasons.append("cephe_maskesi_zayif")
    elif opening_score >= cfg.opening_score_threshold and structural_score < cfg.valid_structural_min_score:
        classification = "opening_edge_axis"
        reasons.append("karanlik_aciklik_kenari")
    elif (
        structural_score >= cfg.valid_structural_min_score
        and vertical_component >= 0.30
        and effective_damage_fraction < cfg.fragmented_damage_fraction
        and source_state != "fragmented_repeated_motif"
    ):
        classification = "valid_structural_axis"
    elif (
        structural_score >= cfg.fragmented_structural_min_score
        and vertical_component >= 0.30
        and effective_damage_fraction < cfg.heavy_damage_fraction
        and (
            effective_damage_fraction >= cfg.fragmented_damage_fraction
            or source_state == "fragmented_repeated_motif"
        )
    ):
        classification = "fragmented_structural_axis"
    elif (
        structural_score >= cfg.valid_structural_min_score
        and vertical_component >= 0.30
        and (spacing_support > 0 or family_support > 0)
    ):
        classification = "valid_structural_axis"
    else:
        classification = "weak_axis"
        if effective_damage_fraction >= cfg.fragmented_damage_fraction:
            reasons.append("hasar_var_ama_yapisal_kanit_yetersiz")
        if vertical_component < 0.30:
            reasons.append("dikey_sureklilik_yetersiz")
        if spacing_support == 0 and family_support == 0:
            reasons.append("ritim_ailesi_yetersiz")
        reasons.append("yapisal_kanit_yetersiz")

    return {
        "source_index": int(axis.get("source_index", 0)),
        "source_id": str(axis.get("id")),
        "axis_x": round(float(axis["axis_x"]), 2),
        "source_state": source_state,
        "source_repetition_score": round(source_repetition, 6),
        "source_best_similarity": round(source_similarity, 6),
        "classification": classification,
        "classification_reasons": list(dict.fromkeys(reasons)),
        "metrics": {
            "strip_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "facade_occupancy": round(facade_occupancy, 6),
            "dark_fraction": round(dark_fraction, 6),
            "vertical_activity": round(vertical_activity, 6),
            "longest_vertical_run": round(longest_run, 6),
            "source_damage_fraction": round(source_damage_fraction, 6),
            "pixel_damage_fraction": round(pixel_damage_fraction, 6),
            "effective_damage_fraction": round(effective_damage_fraction, 6),
            "visible_vertical_bins": int(visible_vertical_bins),
            "match_support": int(match_support),
            "spacing_support": int(spacing_support),
            "family_support": int(family_support),
            "opening_score": round(opening_score, 6),
            "structural_score": round(structural_score, 6),
        },
    }


# =============================================================================
# VISUALS
# =============================================================================
def overlay_damage(rgb: np.ndarray, damage_mask: np.ndarray, alpha: float) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    selected = damage_mask > 0
    tint = np.array([255.0, 35.0, 35.0], dtype=np.float32)
    output[selected] = output[selected] * (1.0 - alpha) + tint * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_axes(
    rgb: np.ndarray,
    damage_mask: np.ndarray,
    axes: list[dict[str, Any]],
    roi: list[int],
    title: str,
    structural_only: bool,
    cfg: Config,
) -> np.ndarray:
    canvas = overlay_damage(rgb, damage_mask, cfg.overlay_damage_alpha)
    x1, y1, x2, y2 = roi
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (235, 235, 235), 1)
    for index, axis in enumerate(axes, start=1):
        classification = str(axis["classification"])
        if structural_only and classification not in {
            "valid_structural_axis",
            "fragmented_structural_axis",
        }:
            continue
        x = int(round(float(axis["axis_x"])))
        color = COLORS.get(classification, (180, 180, 180))
        cv2.line(canvas, (x, y1), (x, y2 - 1), color, cfg.line_thickness)
        label = f"{axis['source_id']} {classification.replace('_axis', '')}"
        cv2.putText(
            canvas,
            label,
            (max(2, min(canvas.shape[1] - 235, x + 3)), max(16, y1 + 16 + (index % 3) * 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (12, 12, 12), -1)
    cv2.putText(
        canvas,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


# =============================================================================
# MAIN
# =============================================================================
def run(cfg: Config) -> Path:
    parent = find_parent_run(cfg)
    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started = time.perf_counter()
    source_script = Path(__file__).resolve()
    meta_path = run_dir / "run_meta.json"

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage2d_cleaned_axes",
        "status": "running",
        "started_at": now_iso(),
        "parent_run": {"run_id": parent.name, "path": str(parent.resolve())},
        "source_script": {
            "filename": source_script.name,
            "path": str(source_script),
            "sha256": sha256_file(source_script),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", None),
        },
        "config": asdict(cfg),
        "safety": {
            "new_axes_created": 0,
            "invented_columns": 0,
            "invented_pilasters": 0,
            "invented_arches": 0,
            "invented_modules": 0,
            "legacy_stage2_files_read": [],
        },
        "copied_inputs": {},
        "summary": {},
        "outputs": [],
    }

    def flush() -> None:
        write_json(meta_path, metadata)

    try:
        logger.info("=" * 78)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent.resolve()}")
        logger.info("[STAGE] Stage 2D — kaynak DINOv2 eksenlerini temizleme")
        logger.info("[SAFE] Yeni eksen/nesne/modül üretilmeyecek")
        logger.info("=" * 78)

        copied: dict[str, Any] = {}
        copied["input"] = copy_with_record(parent / "input.png", run_dir / "input.png")
        copied["completion_mask"] = copy_with_record(
            parent / "completion_mask.png", run_dir / "completion_mask.png"
        )
        copied["repetition_axes"] = copy_with_record(
            parent / "repetition_axes.json", run_dir / "source_repetition_axes.json"
        )
        if (parent / "elements.json").exists():
            copied["elements"] = copy_with_record(
                parent / "elements.json", run_dir / "source_elements.json"
            )
        metadata["copied_inputs"] = copied
        flush()

        rgb = read_rgb(run_dir / "input.png")
        height, width = rgb.shape[:2]
        damage_mask = load_binary_mask(
            run_dir / "completion_mask.png", (width, height)
        )
        axes_document = read_json(run_dir / "source_repetition_axes.json")
        axes = source_axes(axes_document)
        if not axes:
            raise RuntimeError("repetition_axes.json içinde geçerli kaynak eksen yok")

        elements_document = (
            read_json(run_dir / "source_elements.json")
            if (run_dir / "source_elements.json").exists()
            else None
        )
        elements = flatten_elements(elements_document)
        roi = resolve_roi(axes_document, damage_mask, width, height, cfg)
        facade_mask, facade_metrics = build_facade_mask(
            rgb, damage_mask, axes, elements, roi, cfg
        )
        Image.fromarray(facade_mask, mode="L").save(run_dir / "stage2d_facade_mask.png")

        dominant_spacings = robust_dominant_spacings(axes)
        all_x = [float(axis["axis_x"]) for axis in axes]
        classified = [
            measure_axis(
                axis,
                rgb,
                damage_mask,
                facade_mask,
                roi,
                all_x,
                dominant_spacings,
                cfg,
            )
            for axis in axes
        ]

        counts: dict[str, int] = {}
        for axis in classified:
            counts[axis["classification"]] = counts.get(axis["classification"], 0) + 1
        structural_count = sum(
            1
            for axis in classified
            if axis["classification"]
            in {"valid_structural_axis", "fragmented_structural_axis"}
        )

        output_document = {
            "run_id": run_id,
            "parent_run": {"run_id": parent.name, "path": str(parent.resolve())},
            "image_size": [width, height],
            "facade_roi": roi,
            "safety": metadata["safety"],
            "facade_mask": facade_metrics,
            "summary": {
                "source_axis_count": len(axes),
                "structural_axis_candidate_count": structural_count,
                "counts": {
                    "valid_structural_axis": counts.get("valid_structural_axis", 0),
                    "fragmented_structural_axis": counts.get("fragmented_structural_axis", 0),
                    "opening_edge_axis": counts.get("opening_edge_axis", 0),
                    "background_axis": counts.get("background_axis", 0),
                    "weak_axis": counts.get("weak_axis", 0),
                },
                "dominant_spacings_px": [round(value, 6) for value in dominant_spacings],
                "new_axes_created": 0,
                "invented_columns": 0,
                "invented_pilasters": 0,
                "invented_modules": 0,
            },
            "axes": classified,
        }
        write_json(run_dir / "stage2d_rhythm_groups.json", output_document)

        all_overlay = draw_axes(
            rgb,
            damage_mask,
            classified,
            roi,
            "STAGE 2D - ALL SOURCE AXES CLASSIFIED / NO NEW AXIS",
            False,
            cfg,
        )
        structural_overlay = draw_axes(
            rgb,
            damage_mask,
            classified,
            roi,
            "STAGE 2D - OBSERVED STRUCTURAL AXIS CANDIDATES ONLY",
            True,
            cfg,
        )
        save_rgb(run_dir / "stage2d_axis_classification.png", all_overlay)
        save_rgb(run_dir / "stage2d_structural_axes.png", structural_overlay)

        report_lines = [
            "STAGE 2D — CLEANED OBSERVED AXES REPORT",
            f"Run: {run_id}",
            f"Parent: {parent.resolve()}",
            "",
            "GÜVENLİK:",
            "- Kaynak eksen konumları değiştirilmedi.",
            "- Yeni eksen, pilaster, kemer veya modül oluşturulmadı.",
            "- Yalnız kaynak DINOv2 eksenleri sınıflandırıldı.",
            "",
            f"Kaynak eksen: {len(axes)}",
            f"Yapısal aday: {structural_count}",
            f"Sınıf sayıları: {output_document['summary']['counts']}",
            f"Baskın aralıklar: {output_document['summary']['dominant_spacings_px']}",
            "",
        ]
        for axis in classified:
            report_lines.append(
                f"- {axis['source_id']}: x={axis['axis_x']:.2f}, "
                f"class={axis['classification']}, "
                f"structural={axis['metrics']['structural_score']:.3f}, "
                f"damage={axis['metrics']['effective_damage_fraction']:.3f}"
            )
        (run_dir / "stage2d_report.txt").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )

        metadata["summary"] = output_document["summary"]
        metadata["outputs"] = [
            "input.png",
            "completion_mask.png",
            "source_repetition_axes.json",
            *(["source_elements.json"] if (run_dir / "source_elements.json").exists() else []),
            "stage2d_facade_mask.png",
            "stage2d_axis_classification.png",
            "stage2d_structural_axes.png",
            "stage2d_rhythm_groups.json",
            "stage2d_report.txt",
            "run.log",
            "run_meta.json",
        ]
        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        flush()

        logger.info(
            f"[COUNT] source_axes={len(axes)}, structural_candidates={structural_count}, "
            "new_axes=0"
        )
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info("[OK] Stage 2D tamamlandı")
        return run_dir

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        metadata["traceback"] = traceback.format_exc()
        flush()
        logger.exception("[FAILED] Stage 2D")
        raise


def main() -> None:
    run(Config())


if __name__ == "__main__":
    main()
