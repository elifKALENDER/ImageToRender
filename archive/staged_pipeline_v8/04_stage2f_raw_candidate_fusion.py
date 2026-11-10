#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage2f_raw_candidate_fusion.py
=======================================================
V8 / STAGE 2F — Ham nesne adayları + sınıfa özel kanıt füzyonu

Bu aşamanın amacı, daha önce üretilmiş iki bağımsız kanıt katmanını birleştirmektir:

1. Stage 1B OWL-ViT + SAM nesne adayları
   - arch, pilaster, column, base, capital, window, door kutuları/maskeleri

2. Stage 2D temizlenmiş DINOv2 tekrar eksenleri
   - valid_structural_axis
   - fragmented_structural_axis
   - opening_edge_axis
   - background_axis
   - weak_axis

KESİN GÜVENLİK KURALLARI
------------------------
- Yeni sütun, pilaster, kemer, pencere veya kapı ÜRETMEZ.
- Yeni eksen ÜRETMEZ.
- İki kutu arasındaki boşluğa eleman EKLEMEZ.
- Yalnız mevcut nesne adaylarını doğrular, zayıflatır veya reddeder.
- Parçalı zincir oluştururken yalnız gerçekten mevcut kutuları gruplar.
- Her çalıştırmada yeni outputs/restore_YYYYMMDD_HHMMSS klasörü açar.
- Önceki run klasörlerinin üzerine yazmaz.

Yeni run çıktıları
------------------
input.png
completion_mask.png
source_stage2d_rhythm_groups.json
source_elements.json
source_stage2d_facade_mask.png
fusion_all_candidates.png
fused_architectural_evidence.png
fragment_evidence_chains.png
validated_elements.json
tentative_elements.json
rejected_elements.json
fragment_chains.json
fusion_report.txt
run.log
run_meta.json

Bu aşamada Stable Diffusion / ControlNet / restorasyon üretimi YOKTUR.
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

    # Boşsa en yeni stage2d_rhythm_groups.json içeren restore_* klasörü seçilir.
    stage2d_run_dir: str = ""

    # Boşsa input SHA256 eşleşmesine göre elements.json içeren run bulunur.
    stage1_run_dir: str = ""

    supported_classes: tuple[str, ...] = (
        "arch",
        "pilaster",
        "column",
        "base",
        "capital",
        "window",
        "door",
    )

    structural_classes: tuple[str, ...] = (
        "pilaster",
        "column",
        "base",
        "capital",
    )

    opening_classes: tuple[str, ...] = (
        "arch",
        "window",
        "door",
    )

    # Ön filtreleme
    minimum_detector_confidence: float = 0.0065
    maximum_box_area_ratio: float = 0.38
    minimum_box_area_ratio: float = 0.00020
    maximum_damage_for_intact: float = 0.34
    maximum_damage_for_fragment: float = 0.72

    # Eksen yakınlığı
    axis_center_tolerance_ratio: float = 0.035
    arch_side_axis_tolerance_ratio: float = 0.055
    opening_axis_tolerance_ratio: float = 0.045

    # Sınıflandırma eşikleri
    validated_score_min: float = 0.64
    tentative_score_min: float = 0.50
    fragment_score_min: float = 0.57

    # Görsel geometri
    dark_pixel_threshold: int = 68
    edge_active_threshold: float = 0.27
    facade_occupancy_min: float = 0.22
    fragment_min_visible_fraction: float = 0.18

    # Parça zinciri
    chain_axis_tolerance_ratio: float = 0.042
    chain_min_members: int = 2
    chain_min_vertical_separation_ratio: float = 0.035

    # Görselleştirme
    show_rejected_on_all_candidates: bool = True
    overlay_damage_alpha: float = 0.11


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
    logger = logging.getLogger(f"stage2e_{run_id}")
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


def find_stage2d_run(cfg: Config) -> Path:
    if cfg.stage2d_run_dir:
        run = Path(cfg.stage2d_run_dir)
        if not run.is_dir():
            raise FileNotFoundError(f"Stage 2D run klasörü yok: {run.resolve()}")
        if not (run / "stage2d_rhythm_groups.json").exists():
            raise FileNotFoundError(
                f"stage2d_rhythm_groups.json yok: {run.resolve()}"
            )
        return run

    root = Path(cfg.output_root)
    candidates = sorted(
        (path for path in root.glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if (
            (candidate / "input.png").exists()
            and (candidate / "completion_mask.png").exists()
            and (candidate / "stage2d_rhythm_groups.json").exists()
        ):
            return candidate
    raise FileNotFoundError(
        f"stage2d_rhythm_groups.json içeren restore_* bulunamadı: {root.resolve()}"
    )


def elements_count(path: Path) -> int:
    try:
        payload = read_json(path)
        return len(payload.get("elements", []))
    except Exception:
        return 0


def find_stage1_run(
    cfg: Config,
    stage2d_run: Path,
    input_sha256: str,
) -> Path:
    if cfg.stage1_run_dir:
        run = Path(cfg.stage1_run_dir)
        if not run.is_dir() or not (run / "elements.json").exists():
            raise FileNotFoundError(
                f"Geçerli Stage 1 run değil: {run.resolve()}"
            )
        return run

    # Stage 2D klasöründe kaynak elements kopyası varsa onu tercih et.
    if (stage2d_run / "source_elements.json").exists():
        return stage2d_run
    if (stage2d_run / "elements.json").exists():
        return stage2d_run

    # run_meta parent zincirinde ara.
    meta_path = stage2d_run / "run_meta.json"
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
            parent_path = meta.get("parent_run", {}).get("path")
            if parent_path:
                parent = Path(parent_path)
                if (parent / "elements.json").exists():
                    return parent
        except Exception:
            pass

    # Aynı input SHA ile en yeni elements.json run'ını bul.
    root = Path(cfg.output_root)
    candidates = sorted(
        (path for path in root.glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        element_path = candidate / "elements.json"
        input_path = candidate / "input.png"
        if not element_path.exists() or not input_path.exists():
            continue
        if elements_count(element_path) == 0:
            continue
        try:
            if sha256_file(input_path) == input_sha256:
                return candidate
        except Exception:
            continue

    raise FileNotFoundError(
        "Aynı input görüntüsüne ait elements.json içeren Stage 1 run bulunamadı"
    )


def resolve_elements_path(stage1_run: Path) -> Path:
    source = stage1_run / "source_elements.json"
    if source.exists():
        return source
    source = stage1_run / "elements.json"
    if source.exists():
        return source
    raise FileNotFoundError(f"elements.json bulunamadı: {stage1_run.resolve()}")


def copy_with_record(source: Path, target: Path) -> dict[str, Any]:
    shutil.copy2(source, target)
    return {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "run_copy": target.name,
        "run_copy_sha256": sha256_file(target),
    }


# =============================================================================
# GEOMETRY / MASK UTILITIES
# =============================================================================
def clip_box(
    box: Iterable[float],
    width: int,
    height: int,
) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_width(box: list[int]) -> int:
    return max(1, box[2] - box[0])


def box_height(box: list[int]) -> int:
    return max(1, box[3] - box[1])


def box_center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_area_ratio(box: list[int], width: int, height: int) -> float:
    return float(box_width(box) * box_height(box)) / float(width * height)


def fraction_in_box(mask: np.ndarray, box: list[int], positive: bool = True) -> float:
    x1, y1, x2, y2 = box
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    if positive:
        return float(np.count_nonzero(roi > 0)) / float(roi.size)
    return float(np.count_nonzero(roi == 0)) / float(roi.size)


def robust_normalize(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low = float(np.percentile(finite, 15))
    high = float(np.percentile(finite, 96))
    if high <= low + 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def refine_facade_mask(
    source_mask: np.ndarray,
    damage_mask: np.ndarray,
    boxes: list[list[int]],
    axes: list[dict[str, Any]],
    width: int,
    height: int,
) -> np.ndarray:
    """Gürültülü Stage 2D cephe maskesini kanıt çalışma alanına sınırlar."""
    xs: list[float] = []
    ys: list[float] = []

    for box in boxes:
        xs.extend([box[0], box[2]])
        ys.extend([box[1], box[3]])
    for axis in axes:
        xs.append(float(axis.get("axis_x", width / 2)))

    damage_y, damage_x = np.where(damage_mask > 0)
    if damage_x.size:
        xs.extend([float(damage_x.min()), float(damage_x.max())])
        ys.extend([float(damage_y.min()), float(damage_y.max())])

    if xs and ys:
        x1 = max(0, int(min(xs) - 0.10 * width))
        x2 = min(width, int(max(xs) + 0.10 * width))
        y1 = max(0, int(min(ys) - 0.08 * height))
        y2 = min(height, int(max(ys) + 0.08 * height))
    else:
        x1, y1, x2, y2 = 0, 0, width, height

    bounded = np.zeros_like(source_mask, dtype=np.uint8)
    bounded[y1:y2, x1:x2] = np.where(
        source_mask[y1:y2, x1:x2] > 0,
        255,
        0,
    ).astype(np.uint8)

    # Küçük kaya/gürültü adacıklarını at; en büyük bileşenleri koru.
    binary = (bounded > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        areas = [
            (label, int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, count)
        ]
        areas.sort(key=lambda item: item[1], reverse=True)
        minimum_area = max(40, int(width * height * 0.0015))
        keep = {label for label, area in areas[:5] if area >= minimum_area}
        cleaned = np.zeros_like(binary)
        for label in keep:
            cleaned[labels == label] = 1
        binary = cleaned

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary[damage_mask > 0] = 0
    return binary * 255


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================
def load_source_elements(
    elements_path: Path,
    width: int,
    height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    """
    Stage 1 dosyasındaki yalnız temizlenmiş ``elements`` listesini değil,
    ``clean_summary.rejected_examples`` içindeki ham adayları da geri alır.

    Böylece kemerler, kapılar, pencereler ve hasarla çakıştığı için daha önce
    elenmiş parçalar yeniden sınıfa özel kanıtlarla değerlendirilir. Burada
    yeni bir nesne üretilmez; yalnız Stage 1'in gerçekten kaydettiği kutular
    yeniden okunur.
    """
    payload = read_json(elements_path)
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    clean_items = list(payload.get("elements", []))
    rejected_items = list(
        payload.get("clean_summary", {}).get("rejected_examples", [])
    )

    for item in clean_items:
        copied = dict(item)
        copied["source_filter_status"] = "clean"
        copied["source_filter_reason"] = copied.get(
            "geometry_filter_reason", "ok"
        )
        merged.append(copied)
        seen_ids.add(str(copied.get("id", "")))

    for item in rejected_items:
        source_id = str(item.get("id", ""))
        if source_id and source_id in seen_ids:
            continue
        copied = dict(item)
        copied["source_filter_status"] = "previously_rejected"
        copied["source_filter_reason"] = copied.get("reason", "unknown")
        merged.append(copied)
        if source_id:
            seen_ids.add(source_id)

    elements: list[dict[str, Any]] = []
    for index, item in enumerate(merged):
        cls = str(item.get("class", "")).strip().lower()
        if cls not in cfg.supported_classes:
            continue

        confidence = float(item.get("confidence", 0.0))
        # Hasar nedeniyle elenen mimari açıklıkları düşük güven yüzünden
        # yeniden kaybetmemek için ham adaylarda daha düşük taban kullanılır.
        class_floor = cfg.minimum_detector_confidence
        if cls in {"arch", "window", "door"}:
            class_floor = min(class_floor, 0.0045)
        if confidence < class_floor:
            continue

        raw_box = item.get("bbox_xyxy") or item.get("bbox")
        if not raw_box or len(raw_box) != 4:
            continue
        box = clip_box(raw_box, width, height)
        area_ratio = box_area_ratio(box, width, height)
        if area_ratio < cfg.minimum_box_area_ratio:
            continue
        # Büyük kutular doğrudan atılmaz; özellikle hasarlı ana kemer/kapı
        # adayları için sınır genişletilir. Çok aşırı kutular yine elenir.
        class_max_area = cfg.maximum_box_area_ratio
        if cls in {"arch", "door"}:
            class_max_area = max(class_max_area, 0.62)
        if area_ratio > class_max_area:
            continue

        cx, cy = box_center(box)
        elements.append(
            {
                "source_index": index,
                "source_id": item.get("id", f"{cls}_{index + 1:03d}"),
                "class": cls,
                "confidence": confidence,
                "quality": item.get("quality", "raw_recovered"),
                "bbox_xyxy": box,
                "center": [round(cx, 3), round(cy, 3)],
                "width": box_width(box),
                "height": box_height(box),
                "area_ratio": round(area_ratio, 7),
                "source_damage_overlap": float(
                    item.get("damage_overlap", 0.0)
                ),
                "mask_file": item.get("mask_file"),
                "source_view": item.get("source_view"),
                "source_filter_status": item.get(
                    "source_filter_status", "unknown"
                ),
                "source_filter_reason": item.get(
                    "source_filter_reason", "unknown"
                ),
            }
        )

    return elements


def load_axes(stage2d_path: Path) -> list[dict[str, Any]]:
    payload = read_json(stage2d_path)
    return list(payload.get("axes", []))


# =============================================================================
# SCORE HELPERS
# =============================================================================
def class_relative_detector_scores(
    elements: list[dict[str, Any]],
) -> dict[str, float]:
    by_class: dict[str, list[float]] = {}
    for element in elements:
        by_class.setdefault(element["class"], []).append(float(element["confidence"]))

    result: dict[str, float] = {}
    for element in elements:
        values = np.asarray(by_class[element["class"]], dtype=np.float32)
        confidence = float(element["confidence"])
        if values.size == 1:
            normalized = min(1.0, confidence / 0.025)
        else:
            low = float(np.percentile(values, 10))
            high = float(np.percentile(values, 90))
            if high <= low + 1e-8:
                normalized = min(1.0, confidence / 0.025)
            else:
                normalized = float(np.clip((confidence - low) / (high - low), 0.0, 1.0))
        result[element["source_id"]] = normalized
    return result


def structural_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        axis
        for axis in axes
        if axis.get("classification")
        in {"valid_structural_axis", "fragmented_structural_axis"}
    ]


def opening_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        axis
        for axis in axes
        if axis.get("classification") == "opening_edge_axis"
    ]


def nearest_axis(
    x: float,
    axes: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if not axes:
        return None, math.inf
    ranked = sorted(
        ((axis, abs(float(axis.get("axis_x", 0.0)) - x)) for axis in axes),
        key=lambda item: item[1],
    )
    return ranked[0]


def axes_inside_box(
    box: list[int],
    axes: list[dict[str, Any]],
    margin: float = 0.0,
) -> list[dict[str, Any]]:
    x1, _, x2, _ = box
    return [
        axis
        for axis in axes
        if x1 - margin <= float(axis.get("axis_x", 0.0)) <= x2 + margin
    ]


def edge_metrics(
    gray: np.ndarray,
    edge_norm: np.ndarray,
    box: list[int],
    cfg: Config,
) -> dict[str, float]:
    x1, y1, x2, y2 = box
    roi_gray = gray[y1:y2, x1:x2]
    roi_edge = edge_norm[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return {
            "dark_fraction": 0.0,
            "vertical_activity": 0.0,
            "horizontal_activity": 0.0,
            "longest_vertical_run": 0.0,
            "upper_edge_ratio": 0.0,
        }

    dark_fraction = float(np.mean(roi_gray < cfg.dark_pixel_threshold))

    row_activity = np.mean(roi_edge, axis=1) >= cfg.edge_active_threshold
    col_activity = np.mean(roi_edge, axis=0) >= cfg.edge_active_threshold
    vertical_activity = float(np.mean(row_activity))
    horizontal_activity = float(np.mean(col_activity))
    longest_vertical = float(longest_true_run(row_activity)) / max(1, len(row_activity))

    split = max(1, roi_edge.shape[0] // 2)
    upper = float(np.mean(roi_edge[:split])) if split > 0 else 0.0
    lower = float(np.mean(roi_edge[split:])) if split < roi_edge.shape[0] else upper
    upper_edge_ratio = upper / max(1e-6, upper + lower)

    return {
        "dark_fraction": round(dark_fraction, 6),
        "vertical_activity": round(vertical_activity, 6),
        "horizontal_activity": round(horizontal_activity, 6),
        "longest_vertical_run": round(longest_vertical, 6),
        "upper_edge_ratio": round(upper_edge_ratio, 6),
    }


def geometry_score_for_class(
    cls: str,
    box: list[int],
    edge: dict[str, float],
    width: int,
    height: int,
) -> tuple[float, list[str]]:
    bw = box_width(box)
    bh = box_height(box)
    aspect = bw / max(1.0, float(bh))
    slenderness = bh / max(1.0, float(bw))
    reasons: list[str] = []

    if cls == "arch":
        aspect_score = float(np.exp(-abs(math.log(max(0.15, aspect) / 1.15))))
        dark_support = min(1.0, edge["dark_fraction"] / 0.38)
        upper_support = min(1.0, edge["upper_edge_ratio"] / 0.62)
        score = 0.42 * aspect_score + 0.32 * dark_support + 0.26 * upper_support
        if 0.45 <= aspect <= 3.2:
            reasons.append("kemer_kutusu_orani_uyumlu")
        if edge["dark_fraction"] >= 0.18:
            reasons.append("aciklik_golgesi_destegi")
        if edge["upper_edge_ratio"] >= 0.48:
            reasons.append("ust_yarida_kenar_destegi")
        return float(np.clip(score, 0.0, 1.0)), reasons

    if cls in {"pilaster", "column"}:
        slender_score = float(np.clip((slenderness - 1.1) / 4.2, 0.0, 1.0))
        vertical_score = min(1.0, edge["vertical_activity"] / 0.42)
        run_score = min(1.0, edge["longest_vertical_run"] / 0.32)
        score = 0.45 * slender_score + 0.32 * vertical_score + 0.23 * run_score
        if slenderness >= 1.7:
            reasons.append("ince_dikey_oran")
        if edge["vertical_activity"] >= 0.20:
            reasons.append("dikey_kenar_surekliligi")
        return float(np.clip(score, 0.0, 1.0)), reasons

    if cls in {"base", "capital"}:
        horizontal_score = min(1.0, edge["horizontal_activity"] / 0.55)
        aspect_score = float(np.exp(-abs(math.log(max(0.15, aspect) / 1.55))))
        size_score = 1.0 if bw / width <= 0.18 and bh / height <= 0.22 else 0.35
        score = 0.44 * horizontal_score + 0.36 * aspect_score + 0.20 * size_score
        if aspect >= 0.75:
            reasons.append("yatay_parca_orani")
        if edge["horizontal_activity"] >= 0.22:
            reasons.append("yatay_profil_kenari")
        return float(np.clip(score, 0.0, 1.0)), reasons

    if cls in {"window", "door"}:
        target_aspect = 0.55 if cls == "door" else 0.78
        aspect_score = float(np.exp(-abs(math.log(max(0.15, aspect) / target_aspect))))
        dark_support = min(1.0, edge["dark_fraction"] / 0.42)
        vertical_score = min(1.0, edge["vertical_activity"] / 0.40)
        score = 0.35 * aspect_score + 0.42 * dark_support + 0.23 * vertical_score
        if edge["dark_fraction"] >= 0.20:
            reasons.append("karanlik_aciklik_destegi")
        return float(np.clip(score, 0.0, 1.0)), reasons

    return 0.0, reasons


# =============================================================================
# FUSION
# =============================================================================
def evaluate_elements(
    elements: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    rgb: np.ndarray,
    damage_mask: np.ndarray,
    facade_mask: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_norm = robust_normalize(0.58 * sobel_x + 0.42 * sobel_y)

    structural = structural_axes(axes)
    openings = opening_axes(axes)
    detector_norm = class_relative_detector_scores(elements)
    evaluated: list[dict[str, Any]] = []

    center_tolerance = width * cfg.axis_center_tolerance_ratio
    opening_tolerance = width * cfg.opening_axis_tolerance_ratio
    arch_side_tolerance = width * cfg.arch_side_axis_tolerance_ratio

    for element in elements:
        cls = element["class"]
        box = element["bbox_xyxy"]
        cx, cy = box_center(box)
        damage = fraction_in_box(damage_mask, box, positive=True)
        visible_fraction = 1.0 - damage
        facade = fraction_in_box(facade_mask, box, positive=True)
        edge = edge_metrics(gray, edge_norm, box, cfg)
        geometry_score, geometry_reasons = geometry_score_for_class(
            cls, box, edge, width, height
        )

        axis_support = 0.0
        axis_reasons: list[str] = []
        supporting_axes: list[str] = []

        if cls in cfg.structural_classes:
            nearest, axis_distance = nearest_axis(cx, structural)
            if nearest is not None and axis_distance <= center_tolerance:
                distance_score = 1.0 - axis_distance / max(1.0, center_tolerance)
                source_structural = float(
                    nearest.get("metrics", {}).get("structural_score", 0.5)
                )
                axis_support = float(np.clip(0.62 * distance_score + 0.38 * source_structural, 0, 1))
                supporting_axes.append(str(nearest.get("source_id", "axis")))
                axis_reasons.append("yapisal_eksen_merkez_destegi")

        elif cls == "arch":
            inside = axes_inside_box(box, structural, margin=arch_side_tolerance)
            left_side = [a for a in inside if float(a.get("axis_x", cx)) <= cx]
            right_side = [a for a in inside if float(a.get("axis_x", cx)) >= cx]
            opening_inside = axes_inside_box(box, openings, margin=opening_tolerance)

            side_count = int(bool(left_side)) + int(bool(right_side))
            axis_support = 0.28 * side_count + 0.26 * min(1, len(opening_inside))
            if side_count == 2:
                axis_reasons.append("iki_yanda_yapisal_eksen")
            elif side_count == 1:
                axis_reasons.append("tek_yanda_yapisal_eksen")
            if opening_inside:
                axis_reasons.append("aciklik_kenari_ekseni")
            supporting_axes.extend(
                str(a.get("source_id", "axis"))
                for a in (left_side[:1] + right_side[:1] + opening_inside[:1])
            )
            axis_support = float(np.clip(axis_support, 0.0, 1.0))

        elif cls in {"window", "door"}:
            inside = axes_inside_box(box, openings, margin=opening_tolerance)
            nearest, distance_px = nearest_axis(cx, openings)
            if inside:
                axis_support = min(1.0, 0.60 + 0.15 * len(inside))
                supporting_axes.extend(str(a.get("source_id", "axis")) for a in inside[:2])
                axis_reasons.append("aciklik_ekseni_kutu_icinde")
            elif nearest is not None and distance_px <= opening_tolerance:
                axis_support = 1.0 - distance_px / opening_tolerance
                supporting_axes.append(str(nearest.get("source_id", "axis")))
                axis_reasons.append("aciklik_ekseni_yakinda")

        facade_score = float(np.clip((facade - 0.10) / 0.55, 0.0, 1.0))
        damage_score = float(np.clip(1.0 - damage / cfg.maximum_damage_for_fragment, 0.0, 1.0))
        detector_score = float(detector_norm[element["source_id"]])
        edge_score = float(
            np.clip(
                0.55 * edge["vertical_activity"]
                + 0.25 * edge["horizontal_activity"]
                + 0.20 * edge["longest_vertical_run"],
                0.0,
                1.0,
            )
        )

        final_score = float(
            np.clip(
                0.22 * detector_score
                + 0.24 * axis_support
                + 0.18 * facade_score
                + 0.20 * geometry_score
                + 0.09 * edge_score
                + 0.07 * damage_score,
                0.0,
                1.0,
            )
        )

        hard_reasons: list[str] = []
        if (
            facade < cfg.facade_occupancy_min
            and damage < cfg.maximum_damage_for_intact
        ):
            hard_reasons.append("cephe_destegi_cok_dusuk")

        # Hasarlı açıklıklarda görünür oran doğal olarak çok düşük olabilir.
        # Bu nedenle yüksek hasar artık otomatik ret değildir.
        minimum_visible = cfg.fragment_min_visible_fraction
        if cls in {"arch", "window", "door"}:
            minimum_visible = 0.06
        if visible_fraction < minimum_visible:
            hard_reasons.append("gorunur_parca_yetersiz")

        class_gate = True
        if cls in cfg.structural_classes:
            class_gate = axis_support >= 0.28 and geometry_score >= 0.25
        elif cls == "arch":
            # Kemer için iki yandaki yapısal eksen, üst kenar ve açıklık gölgesi
            # birlikte değerlendirilir. Açıklık ekseni bulunmaması tek başına ret değildir.
            arch_visual_support = (
                edge["upper_edge_ratio"] >= 0.34
                or edge["dark_fraction"] >= 0.12
            )
            class_gate = (
                geometry_score >= 0.22
                and arch_visual_support
                and (axis_support >= 0.18 or damage >= 0.20)
            )
        elif cls in {"window", "door"}:
            class_gate = (
                geometry_score >= 0.22
                and (
                    axis_support >= 0.16
                    or edge["dark_fraction"] >= 0.14
                    or damage >= 0.25
                )
            )

        # Parçalı elemanlar için puan: hasar cezalandırılmaz; tersine görünür
        # iz + sınıf geometrisi + eksen/kenar desteği aranır.
        fragment_evidence_score = float(
            np.clip(
                0.24 * detector_score
                + 0.24 * axis_support
                + 0.24 * geometry_score
                + 0.16 * edge_score
                + 0.12 * facade_score,
                0.0,
                1.0,
            )
        )

        if hard_reasons:
            status = "rejected"
        elif damage >= cfg.maximum_damage_for_intact:
            if class_gate and fragment_evidence_score >= 0.43:
                status = "fragment_supported"
            elif class_gate and fragment_evidence_score >= 0.34:
                status = "tentative"
            else:
                status = "rejected"
        elif final_score >= cfg.validated_score_min and class_gate:
            status = "validated"
        elif final_score >= cfg.tentative_score_min and class_gate:
            status = "tentative"
        else:
            status = "rejected"

        reasons = []
        reasons.extend(geometry_reasons)
        reasons.extend(axis_reasons)
        if facade >= cfg.facade_occupancy_min:
            reasons.append("cephe_maskesi_destegi")
        if detector_score >= 0.60:
            reasons.append("sinif_icinde_guclu_dedektor_sirasi")
        if damage >= cfg.maximum_damage_for_intact:
            reasons.append("hasarli_parca_olarak_degerlendirildi")
        reasons.extend(hard_reasons)

        evaluated.append(
            {
                **element,
                "status": status,
                "final_score": round(final_score, 6),
                "fragment_evidence_score": round(fragment_evidence_score, 6),
                "detector_relative_score": round(detector_score, 6),
                "axis_support_score": round(axis_support, 6),
                "facade_occupancy": round(facade, 6),
                "facade_score": round(facade_score, 6),
                "damage_fraction": round(damage, 6),
                "visible_fraction": round(visible_fraction, 6),
                "geometry_score": round(geometry_score, 6),
                "edge_score": round(edge_score, 6),
                "edge_metrics": edge,
                "supporting_axis_ids": sorted(set(supporting_axes)),
                "evidence": sorted(set(reasons)),
            }
        )

    return evaluated


# =============================================================================
# FRAGMENT CHAINS — ONLY EXISTING BOXES
# =============================================================================
def build_fragment_chains(
    evaluated: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    structural = structural_axes(axes)
    candidates = [
        item
        for item in evaluated
        if item["class"] in cfg.structural_classes
        and item["status"] in {"validated", "tentative", "fragment_supported"}
    ]

    tolerance = width * cfg.chain_axis_tolerance_ratio
    grouped: dict[str, list[dict[str, Any]]] = {}

    for item in candidates:
        cx = float(item["center"][0])
        nearest, distance_px = nearest_axis(cx, structural)
        if nearest is None or distance_px > tolerance:
            continue
        axis_id = str(nearest.get("source_id", "axis"))
        grouped.setdefault(axis_id, []).append(item)

    chains: list[dict[str, Any]] = []
    for axis_id, members in grouped.items():
        if len(members) < cfg.chain_min_members:
            continue
        members = sorted(members, key=lambda item: float(item["center"][1]))
        distinct_classes = sorted(set(item["class"] for item in members))
        vertical_span = (
            max(item["bbox_xyxy"][3] for item in members)
            - min(item["bbox_xyxy"][1] for item in members)
        ) / float(height)
        if vertical_span < cfg.chain_min_vertical_separation_ratio:
            continue

        mean_score = float(np.mean([item["final_score"] for item in members]))
        chain_status = "fragment_chain_supported" if (
            len(distinct_classes) >= 2
            or any(item["status"] == "fragment_supported" for item in members)
        ) else "same_axis_candidates"

        chains.append(
            {
                "id": f"fragment_chain_{len(chains) + 1:02d}",
                "axis_id": axis_id,
                "status": chain_status,
                "member_ids": [item["source_id"] for item in members],
                "member_classes": [item["class"] for item in members],
                "distinct_classes": distinct_classes,
                "member_count": len(members),
                "vertical_span_ratio": round(vertical_span, 6),
                "mean_member_score": round(mean_score, 6),
                "safety_note": (
                    "Bu zincir yeni pilaster üretmez; yalnız aynı mevcut eksen "
                    "yakınındaki mevcut nesne kutularını gruplar."
                ),
            }
        )

    return chains


# =============================================================================
# VISUALS
# =============================================================================
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "validated": (50, 220, 75),
    "fragment_supported": (255, 165, 25),
    "tentative": (245, 220, 45),
    "rejected": (170, 170, 170),
}


def overlay_damage(rgb: np.ndarray, damage_mask: np.ndarray, alpha: float) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    red = np.zeros_like(output)
    red[..., 0] = 255
    binary = (damage_mask > 0)[..., None].astype(np.float32)
    output = output * (1.0 - binary * alpha) + red * (binary * alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 31), (20, 20, 20), -1)
    cv2.putText(
        output,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def draw_elements(
    rgb: np.ndarray,
    damage_mask: np.ndarray,
    elements: list[dict[str, Any]],
    include_statuses: set[str],
    title: str,
) -> np.ndarray:
    visual = add_header(overlay_damage(rgb, damage_mask, 0.11), title)
    for index, element in enumerate(elements):
        if element["status"] not in include_statuses:
            continue
        color = STATUS_COLORS[element["status"]]
        x1, y1, x2, y2 = element["bbox_xyxy"]
        cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{element['class']} {element['status']} "
            f"F={element['final_score']:.2f}"
        )
        y = min(visual.shape[0] - 7, max(42, y1 + 15 + (index % 3) * 14))
        cv2.putText(
            visual,
            label,
            (max(2, min(visual.shape[1] - 215, x1)), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return visual


def draw_fragment_chains(
    rgb: np.ndarray,
    damage_mask: np.ndarray,
    chains: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    axes: list[dict[str, Any]],
) -> np.ndarray:
    visual = add_header(
        overlay_damage(rgb, damage_mask, 0.10),
        "EXISTING FRAGMENT BOX CHAINS ONLY - NO COMPLETION / NO NEW PILASTER",
    )
    by_id = {item["source_id"]: item for item in evaluated}
    axis_by_id = {str(axis.get("source_id", "")): axis for axis in axes}

    for chain_index, chain in enumerate(chains):
        color = (255, 170, 30) if chain["status"] == "fragment_chain_supported" else (235, 210, 40)
        member_boxes = []
        for member_id in chain["member_ids"]:
            item = by_id.get(member_id)
            if item is None:
                continue
            box = item["bbox_xyxy"]
            member_boxes.append(box)
            x1, y1, x2, y2 = box
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                visual,
                f"{item['class']}:{member_id}",
                (x1 + 2, min(visual.shape[0] - 8, y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        # Tam boy çizgi çizme; yalnız kutular arasındaki kısa bağlantıyı noktalı göster.
        if len(member_boxes) >= 2:
            axis = axis_by_id.get(chain["axis_id"])
            axis_x = int(round(float(axis.get("axis_x", 0)))) if axis else int(
                np.mean([box_center(box)[0] for box in member_boxes])
            )
            sorted_boxes = sorted(member_boxes, key=lambda box: box[1])
            for first, second in zip(sorted_boxes[:-1], sorted_boxes[1:]):
                start_y = first[3]
                end_y = second[1]
                if end_y <= start_y:
                    continue
                for y in range(start_y + 3, end_y, 9):
                    cv2.circle(visual, (axis_x, y), 1, color, -1)

        if member_boxes:
            min_x = min(box[0] for box in member_boxes)
            min_y = min(box[1] for box in member_boxes)
            cv2.putText(
                visual,
                chain["id"],
                (min_x, max(42, min_y - 5 - chain_index % 3 * 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

    return visual


# =============================================================================
# MAIN
# =============================================================================
def run(cfg: Config) -> Path:
    stage2d_run = find_stage2d_run(cfg)
    stage2d_input = stage2d_run / "input.png"
    input_hash = sha256_file(stage2d_input)
    stage1_run = find_stage1_run(cfg, stage2d_run, input_hash)
    elements_path = resolve_elements_path(stage1_run)

    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started = time.perf_counter()
    meta_path = run_dir / "run_meta.json"

    source_script = Path(__file__)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage2f_raw_candidate_fusion",
        "status": "running",
        "started_at": now_iso(),
        "parent_runs": {
            "stage2d": {
                "run_id": stage2d_run.name,
                "path": str(stage2d_run.resolve()),
            },
            "stage1": {
                "run_id": stage1_run.name,
                "path": str(stage1_run.resolve()),
            },
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
            "invented_pilasters": 0,
            "invented_columns": 0,
            "invented_arches": 0,
            "invented_modules": 0,
            "note": (
                "Yalnız mevcut Stage 1 nesne adayları ve mevcut Stage 2D "
                "eksenleri birleştirilmiştir."
            ),
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
        logger.info(f"[STAGE2D] {stage2d_run.resolve()}")
        logger.info(f"[STAGE1] {stage1_run.resolve()}")
        logger.info("[NEW RUN] Önceki klasörlerin üzerine yazılmayacak")
        logger.info("[SAFE] Yeni nesne/eksen/modül oluşturulmayacak")
        logger.info("=" * 78)

        copy_records: dict[str, Any] = {}
        copy_records["input"] = copy_with_record(
            stage2d_run / "input.png", run_dir / "input.png"
        )
        copy_records["completion_mask"] = copy_with_record(
            stage2d_run / "completion_mask.png", run_dir / "completion_mask.png"
        )
        copy_records["rhythm_groups"] = copy_with_record(
            stage2d_run / "stage2d_rhythm_groups.json",
            run_dir / "source_stage2d_rhythm_groups.json",
        )
        copy_records["elements"] = copy_with_record(
            elements_path, run_dir / "source_elements.json"
        )

        facade_source = stage2d_run / "stage2d_facade_mask.png"
        if facade_source.exists():
            copy_records["facade_mask"] = copy_with_record(
                facade_source, run_dir / "source_stage2d_facade_mask.png"
            )

        metadata["copied_inputs"] = copy_records
        flush()

        rgb = read_rgb(run_dir / "input.png")
        height, width = rgb.shape[:2]
        damage_mask = load_binary_mask(
            run_dir / "completion_mask.png", (width, height)
        )
        axes = load_axes(run_dir / "source_stage2d_rhythm_groups.json")
        elements = load_source_elements(
            run_dir / "source_elements.json", width, height, cfg
        )
        boxes = [element["bbox_xyxy"] for element in elements]

        if (run_dir / "source_stage2d_facade_mask.png").exists():
            source_facade = load_binary_mask(
                run_dir / "source_stage2d_facade_mask.png", (width, height)
            )
        else:
            source_facade = np.full((height, width), 255, dtype=np.uint8)

        facade_mask = refine_facade_mask(
            source_facade, damage_mask, boxes, axes, width, height
        )
        Image.fromarray(facade_mask, mode="L").save(
            run_dir / "fusion_refined_facade_mask.png"
        )

        logger.info(f"[SOURCE] nesne_adayi={len(elements)}, eksen={len(axes)}")
        logger.info(
            f"[AXES] structural={len(structural_axes(axes))}, "
            f"opening={len(opening_axes(axes))}"
        )

        evaluated = evaluate_elements(
            elements, axes, rgb, damage_mask, facade_mask, cfg
        )
        fragment_chains = build_fragment_chains(
            evaluated, axes, width, height, cfg
        )

        validated = [item for item in evaluated if item["status"] in {"validated", "fragment_supported"}]
        tentative = [item for item in evaluated if item["status"] == "tentative"]
        rejected = [item for item in evaluated if item["status"] == "rejected"]

        all_statuses = {"validated", "fragment_supported", "tentative", "rejected"}
        accepted_statuses = {"validated", "fragment_supported", "tentative"}

        save_rgb(
            run_dir / "fusion_all_candidates.png",
            draw_elements(
                rgb,
                damage_mask,
                evaluated,
                all_statuses,
                "ALL STAGE-1 OBJECT CANDIDATES - RAW + CLEAN - CLASS SPECIFIC FUSION",
            ),
        )
        save_rgb(
            run_dir / "fused_architectural_evidence.png",
            draw_elements(
                rgb,
                damage_mask,
                evaluated,
                accepted_statuses,
                "VALIDATED / FRAGMENT-SUPPORTED / TENTATIVE - NO NEW OBJECTS",
            ),
        )
        save_rgb(
            run_dir / "fragment_evidence_chains.png",
            draw_fragment_chains(rgb, damage_mask, fragment_chains, evaluated, axes),
        )

        common_header = {
            "run_id": run_id,
            "stage1_run": metadata["parent_runs"]["stage1"],
            "stage2d_run": metadata["parent_runs"]["stage2d"],
            "safety": metadata["safety"],
        }
        write_json(
            run_dir / "validated_elements.json",
            {**common_header, "elements": validated},
        )
        write_json(
            run_dir / "tentative_elements.json",
            {**common_header, "elements": tentative},
        )
        write_json(
            run_dir / "rejected_elements.json",
            {**common_header, "elements": rejected},
        )
        write_json(
            run_dir / "fragment_chains.json",
            {**common_header, "chains": fragment_chains},
        )

        class_summary: dict[str, dict[str, int]] = {}
        for item in evaluated:
            cls = item["class"]
            class_summary.setdefault(
                cls,
                {"validated": 0, "fragment_supported": 0, "tentative": 0, "rejected": 0},
            )
            class_summary[cls][item["status"]] += 1

        metadata["summary"] = {
            "source_object_candidate_count": len(elements),
            "recovered_previously_rejected_count": sum(
                1 for item in elements
                if item.get("source_filter_status") == "previously_rejected"
            ),
            "source_axis_count": len(axes),
            "structural_axis_count": len(structural_axes(axes)),
            "opening_axis_count": len(opening_axes(axes)),
            "validated_count": sum(1 for item in evaluated if item["status"] == "validated"),
            "fragment_supported_count": sum(1 for item in evaluated if item["status"] == "fragment_supported"),
            "tentative_count": len(tentative),
            "rejected_count": len(rejected),
            "fragment_chain_count": len(fragment_chains),
            "class_summary": class_summary,
            "new_objects_created": 0,
            "new_axes_created": 0,
        }

        report = [
            "V8 STAGE 2F — HAM ADAY KANIT FÜZYON RAPORU",
            f"Run: {run_id}",
            f"Stage 1 parent: {stage1_run.resolve()}",
            f"Stage 2D parent: {stage2d_run.resolve()}",
            "",
            "GÜVENLİK:",
            "- Yeni nesne veya eksen oluşturulmadı.",
            "- Aradaki boşluğa sütun/pilaster/kemer eklenmedi.",
            "- Parça zincirleri yalnız mevcut kutuları gruplar.",
            "",
            f"Kaynak nesne adayı (ham + temiz): {len(elements)}",
            f"Önceden reddedilmiş olup yeniden okunan: {sum(1 for e in elements if e.get('source_filter_status') == 'previously_rejected')}",
            f"Kaynak eksen: {len(axes)}",
            f"Validated: {metadata['summary']['validated_count']}",
            f"Fragment-supported: {metadata['summary']['fragment_supported_count']}",
            f"Tentative: {len(tentative)}",
            f"Rejected: {len(rejected)}",
            f"Fragment chain: {len(fragment_chains)}",
            "",
            "SINIF ÖZETİ:",
        ]
        for cls, counts in sorted(class_summary.items()):
            report.append(
                f"- {cls}: validated={counts['validated']}, "
                f"fragment={counts['fragment_supported']}, "
                f"tentative={counts['tentative']}, rejected={counts['rejected']}"
            )

        report.extend(["", "KABUL EDİLEN / GEÇİCİ NESNELER:"])
        for item in validated + tentative:
            report.append(
                f"- {item['source_id']}: class={item['class']}, "
                f"status={item['status']}, final={item['final_score']:.2f}, "
                f"detector={item['detector_relative_score']:.2f}, "
                f"axis={item['axis_support_score']:.2f}, "
                f"geometry={item['geometry_score']:.2f}, "
                f"facade={item['facade_occupancy']:.2f}, "
                f"damage={item['damage_fraction']:.2f}; "
                f"evidence={','.join(item['evidence']) or 'yok'}"
            )

        (run_dir / "fusion_report.txt").write_text(
            "\n".join(report), encoding="utf-8"
        )

        metadata["outputs"] = [
            {"file": "input.png", "type": "input_copy"},
            {"file": "completion_mask.png", "type": "mask_copy"},
            {"file": "source_elements.json", "type": "stage1_object_candidates_copy"},
            {"file": "source_stage2d_rhythm_groups.json", "type": "stage2d_axes_copy"},
            {"file": "fusion_refined_facade_mask.png", "type": "fusion_working_facade_mask"},
            {"file": "fusion_all_candidates.png", "type": "all_existing_candidates_classified"},
            {"file": "fused_architectural_evidence.png", "type": "accepted_existing_candidates"},
            {"file": "fragment_evidence_chains.png", "type": "existing_fragment_groups"},
            {"file": "validated_elements.json", "type": "validated_and_fragment_supported"},
            {"file": "tentative_elements.json", "type": "tentative"},
            {"file": "rejected_elements.json", "type": "rejected"},
            {"file": "fragment_chains.json", "type": "existing_box_groups"},
            {"file": "fusion_report.txt", "type": "human_readable_report"},
            {"file": "run.log", "type": "log"},
            {"file": "run_meta.json", "type": "metadata"},
        ]

        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush()

        logger.info("[OK] Stage 2F tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            f"[COUNT] kaynak_nesne={len(elements)}, "
            f"validated={metadata['summary']['validated_count']}, "
            f"fragment={metadata['summary']['fragment_supported_count']}, "
            f"tentative={len(tentative)}, rejected={len(rejected)}, "
            "yeni_nesne=0, yeni_eksen=0"
        )
        logger.info(
            "[FILES] fusion_all_candidates.png, fused_architectural_evidence.png, "
            "fragment_evidence_chains.png"
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
        flush()
        logger.error(f"[FAIL] {exc}")
        logger.error(traceback.format_exc())
        raise


def main() -> None:
    run_dir = run(Config())
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
