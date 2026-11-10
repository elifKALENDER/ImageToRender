#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ImageToRender2Restoration_v8_stage3c_c_reconstruction_hypotheses_expert_editor.py
================================================================================
V8 / STAGE 3C-C — Rekonstrüksiyon hipotezleri + uzman onay editörü

AMAÇ
----
Stage 3C-B'nin gözlenen cephe organizasyonu, destek/bay sırası ve rol kayıtlarını
okur. Gözlenen parçalı elemanlar ile maskeyle belgelenmiş kurs/bay kesintilerinden
rekonstrüksiyon hipotezleri üretir ve bunları uzman arayüzünde incelemeye açar.

Bu dosya Stage 3'ün SON adımıdır. Ayrı bir Stage 3C-D yoktur.

AKIŞ
----
1) Stage 3C-B parent run bulunur.
2) Otomatik hipotez adayları oluşturulur:
   - parçalı kemerin tam geometri hipotezi,
   - parçalı pilasterın tam geometri hipotezi,
   - doğrudan kemer gözlenmeyen hasarlı bay için kemer hipotezi,
   - çok geniş, doğrudan kemersiz ve ağır hasarlı bay için destek hipotezi,
   - hasarla kesilen korniş/base kursu için devam segmenti hipotezi.
3) Uzman arayüzü açılır:
   - kabul et / reddet,
   - sil,
   - yeni hipotez ekle,
   - taşı / yeniden boyutlandır / yeniden çiz,
   - sınıf, güven ve not düzenle,
   - geri al,
   - taslak kaydet veya kilitle.
4) Kilitlemede yalnız uzman onaylı hipotezler ayrı JSON'a yazılır.

TEMEL POLİTİKA
--------------
- expert_validated_inventory.json içindeki gözlenen elemanlar değiştirilemez.
- Gözlenen elemanlarla rekonstrüksiyon hipotezleri aynı listeye karıştırılmaz.
- Otomatik adayların hiçbiri uzman onayı olmadan Stage 4'e geçmez.
- Stable Diffusion / inpaint kullanılmaz.
- Üst üste gelen ayrı kemerler duplicate sayılmaz.
- Gözlenen parçalı elemanın kendisi korunur; hipotez yalnız eksik/tam geometri
  katmanını tanımlar.

GİRDİLER (Stage 3C-B parent run)
--------------------------------
input.png
completion_mask.png
source_expert_validated_inventory.json
source_stage3c_horizontal_courses.json
stage3c_b_facade_sectors.json
stage3c_b_support_axis_sequence.json
stage3c_b_ordered_bay_sequence.json
stage3c_b_arch_role_subfamilies.json
stage3c_b_pilaster_role_subfamilies.json

ÇIKTILAR
--------
Her çalıştırmada yeni klasör:
    outputs/restore_YYYYMMDD_HHMMSS

source_* girdiler
stage3c_c_auto_hypotheses.json
stage3c_c_auto_hypothesis_overlay.png
stage3c_c_expert_hypotheses_draft.json            (taslak kaydedilirse)
expert_approved_reconstruction_hypotheses.json    (kilitlenirse)
expert_rejected_reconstruction_hypotheses.json
expert_hypothesis_review_actions.json
stage3c_c_approved_hypotheses.png
stage3c_c_rejected_hypotheses.png
stage3c_c_evidence_and_approved_overlay.png
stage3c_c_report.txt
run.log
run_meta.json

KULLANIM
--------
Windows:
    C:\Users\ekale.ELIFK\.conda\envs\ImageToRender_GPU\python.exe \
    E:\ImageToRender\ImageToRenderStage3C_C.py

Editör açmadan otomatik adayları test etmek için:
    python ImageToRenderStage3C_C.py --auto-only
"""

from __future__ import annotations

import argparse
import copy
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
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception as exc:  # pragma: no cover - Windows runtime dependency
    raise RuntimeError(f"Tkinter yüklenemedi: {exc}") from exc


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    require_locked_inventory: bool = True
    require_stage3c_b_policy: bool = True

    reference_width_px: int = 628
    reference_height_px: int = 904

    # Otomatik hipotez eşikleri.
    fragment_damage_min: float = 0.04
    missing_arch_damage_min: float = 0.24
    missing_support_damage_min: float = 0.58
    missing_support_width_factor: float = 1.75
    course_gap_damage_min: float = 0.20

    # Görsel / editör.
    editor_width: int = 1460
    editor_height: int = 900
    right_panel_width: int = 390
    initial_fit_padding: int = 24
    zoom_min: float = 0.20
    zoom_max: float = 6.0
    zoom_step: float = 1.16
    handle_radius_px: int = 5
    hit_tolerance_px: int = 6
    nudge_px: int = 1
    nudge_fast_px: int = 5
    overlay_damage_alpha: int = 72
    show_observed_default: bool = True
    show_damage_default: bool = True
    show_labels_default: bool = True

    default_new_class: str = "arch"
    default_new_confidence: float = 0.50


CFG = Config()
STAGE_NAME = "v8_stage3c_c_reconstruction_hypotheses_expert_editor"
STAGE_TITLE = "Stage 3C-C — rekonstrüksiyon hipotezleri ve uzman onayı"

ALLOWED_CLASSES = (
    "arch",
    "pilaster",
    "opening",
    "cornice",
    "base",
    "column",
    "other",
)

STATUS_STYLE: dict[str, dict[str, Any]] = {
    "candidate_needs_review": {
        "label": "İNCELE",
        "color": "#f59e0b",
        "rgb": (245, 158, 11),
        "dash": (7, 4),
    },
    "expert_approved": {
        "label": "ONAYLI",
        "color": "#22c55e",
        "rgb": (34, 197, 94),
        "dash": None,
    },
    "expert_rejected": {
        "label": "RED",
        "color": "#ef4444",
        "rgb": (239, 68, 68),
        "dash": (4, 4),
    },
}

CLASS_RGB: dict[str, tuple[int, int, int]] = {
    "arch": (34, 197, 94),
    "pilaster": (14, 165, 233),
    "opening": (168, 85, 247),
    "cornice": (236, 72, 153),
    "base": (245, 158, 11),
    "column": (6, 182, 212),
    "other": (148, 163, 184),
}


# =============================================================================
# FILE / LOG UTILITIES
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
    logger = logging.getLogger(f"stage3c_c_{run_id}")
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


def copy_with_hash(source: Path, target: Path) -> dict[str, Any]:
    shutil.copy2(source, target)
    return {
        "source": str(source.resolve()),
        "copied_to": target.name,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def source_script_info() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "filename": source.name,
        "path": str(source),
        "sha256": sha256_file(source),
    }


# =============================================================================
# PARENT RESOLUTION / VALIDATION
# =============================================================================
REQUIRED_3CB_FILES = (
    "input.png",
    "completion_mask.png",
    "source_expert_validated_inventory.json",
    "source_stage3c_horizontal_courses.json",
    "stage3c_b_facade_sectors.json",
    "stage3c_b_support_axis_sequence.json",
    "stage3c_b_ordered_bay_sequence.json",
    "stage3c_b_arch_role_subfamilies.json",
    "stage3c_b_pilaster_role_subfamilies.json",
)


def is_compatible_parent(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_3CB_FILES)


def find_parent_run(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not is_compatible_parent(parent):
            missing = [name for name in REQUIRED_3CB_FILES if not (parent / name).exists()]
            raise FileNotFoundError(
                "Belirtilen parent run Stage 3C-B için uyumlu değil. Eksikler: "
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
        "Stage 3C-B çıktılarını içeren uyumlu restore_* klasörü bulunamadı."
    )


def validate_parent_docs(
    inventory_doc: dict[str, Any],
    sectors_doc: dict[str, Any],
    support_doc: dict[str, Any],
    bay_doc: dict[str, Any],
    cfg: Config,
) -> None:
    if cfg.require_locked_inventory:
        if not inventory_doc.get("locked") or inventory_doc.get("inventory_state") != "finalized":
            raise RuntimeError("Stage 3C-C yalnız kilitlenmiş uzman envanteriyle çalışır.")
        if inventory_doc.get("unresolved_review"):
            raise RuntimeError("Uzman envanterinde çözümlenmemiş review kaydı var.")

    if cfg.require_stage3c_b_policy:
        for doc in (sectors_doc, support_doc, bay_doc):
            policy = doc.get("policy", {})
            if not policy.get("expert_ids_are_authoritative", False):
                raise RuntimeError("Stage 3C-B expert ID otoritesi politikası eksik.")
            if not policy.get("instances_never_auto_merged", False):
                raise RuntimeError("Stage 3C-B auto-merge yasağı politikası eksik.")
            if not policy.get("no_missing_element_hypothesis", False):
                raise RuntimeError("Parent Stage 3C-B gözlenen-organizasyon aşaması değil.")


# =============================================================================
# GEOMETRY / DATA HELPERS
# =============================================================================
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


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ix1 = max(float(first[0]), float(second[0]))
    iy1 = max(float(first[1]), float(second[1]))
    ix2 = min(float(first[2]), float(second[2]))
    iy2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    union = box_area(first) + box_area(second) - intersection
    return float(intersection / max(1e-9, union))


def union_boxes(first: Sequence[float], second: Sequence[float]) -> list[int]:
    return [
        int(min(first[0], second[0])),
        int(min(first[1], second[1])),
        int(max(first[2], second[2])),
        int(max(first[3], second[3])),
    ]


def scaled_px(cfg: Config, value: float, width: int, height: int) -> float:
    scale = min(
        width / max(1.0, cfg.reference_width_px),
        height / max(1.0, cfg.reference_height_px),
    )
    return max(1.0, float(value) * scale)


def mask_ratio(mask: np.ndarray, box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(mask.shape[1] - 1, x1))
    x2 = max(x1 + 1, min(mask.shape[1], x2))
    y1 = max(0, min(mask.shape[0] - 1, y1))
    y2 = max(y1 + 1, min(mask.shape[0], y2))
    crop = mask[y1:y2, x1:x2]
    return float(np.mean(crop > 0)) if crop.size else 0.0


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.where(np.asarray(image) >= 128, 255, 0).astype(np.uint8)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalize_inventory(
    inventory_doc: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(inventory_doc.get("elements", []), start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", ""))
        if status not in {"observed_intact", "observed_fragmented"}:
            continue
        element_id = str(item.get("id") or f"element_{index:03d}")
        if element_id in seen:
            raise RuntimeError(f"Yinelenen gözlenen eleman ID'si: {element_id}")
        seen.add(element_id)
        box = clip_box(item.get("bbox_xyxy", [0, 0, 1, 1]), width, height)
        elements.append(
            {
                **item,
                "id": element_id,
                "display_number": int(item.get("display_number", index)),
                "class": str(item.get("class", "other")).lower(),
                "status": status,
                "bbox_xyxy": box,
                "center": list(box_center(box)),
                "width": box_width(box),
                "height": box_height(box),
            }
        )
    return elements


def element_lookup(elements: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in elements}


def median_geometry(elements: Iterable[dict[str, Any]]) -> dict[str, float]:
    pool = list(elements)
    if not pool:
        return {
            "width": 40.0,
            "height": 80.0,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 40.0,
            "y2": 80.0,
        }
    return {
        "width": float(median([box_width(item["bbox_xyxy"]) for item in pool])),
        "height": float(median([box_height(item["bbox_xyxy"]) for item in pool])),
        "x1": float(median([item["bbox_xyxy"][0] for item in pool])),
        "y1": float(median([item["bbox_xyxy"][1] for item in pool])),
        "x2": float(median([item["bbox_xyxy"][2] for item in pool])),
        "y2": float(median([item["bbox_xyxy"][3] for item in pool])),
    }


def next_hypothesis_id(records: list[dict[str, Any]], class_name: str) -> str:
    prefix = f"hyp_{class_name}_"
    numbers = []
    for item in records:
        if str(item.get("id", "")).startswith(prefix):
            try:
                numbers.append(int(str(item["id"]).rsplit("_", 1)[-1]))
            except ValueError:
                pass
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def make_record(
    records: list[dict[str, Any]],
    class_name: str,
    bbox: Sequence[float],
    width: int,
    height: int,
    hypothesis_type: str,
    source: str,
    confidence: float,
    evidence: list[str],
    related_observed_ids: list[str] | None = None,
    donor_ids: list[str] | None = None,
    sector_id: str | None = None,
    bay_id: str | None = None,
    course_id: str | None = None,
    geometry_basis: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    record_id = next_hypothesis_id(records, class_name)
    display_number = max((int(item.get("display_number", 0)) for item in records), default=0) + 1
    clipped = clip_box(bbox, width, height)
    return {
        "id": record_id,
        "display_number": display_number,
        "class": class_name,
        "status": "candidate_needs_review",
        "bbox_xyxy": clipped,
        "original_bbox_xyxy": list(clipped),
        "hypothesis_type": hypothesis_type,
        "source": source,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
        "evidence": list(dict.fromkeys(evidence)),
        "related_observed_ids": [item for item in (related_observed_ids or []) if item],
        "donor_ids": [item for item in (donor_ids or []) if item],
        "sector_id": sector_id,
        "bay_id": bay_id,
        "course_id": course_id,
        "geometry_basis": geometry_basis or {},
        "expert_action": "unreviewed_auto_candidate",
        "expert_note": note,
        "expert_reviewed": False,
        "deleted": False,
    }


# =============================================================================
# AUTOMATIC HYPOTHESIS GENERATION
# =============================================================================
def support_position_lookup(supports: list[dict[str, Any]]) -> dict[int, float]:
    return {int(item["sequence_index"]): float(item["x"]) for item in supports}


def build_arch_reference_model(
    arch_records: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    observed_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    x_by_index = support_position_lookup(supports)
    donors = [
        record
        for record in arch_records
        if record.get("status") == "observed_intact"
        and record.get("span_intervals") == 1
        and record.get("left_support_index") in x_by_index
        and record.get("right_support_index") in x_by_index
        and record.get("reference_role") in {
            "strong_observed_reference",
            "observed_reference_with_constraints",
        }
    ]
    if not donors:
        donors = [
            record
            for record in arch_records
            if record.get("status") == "observed_intact"
            and record.get("span_intervals") == 1
            and record.get("left_support_index") in x_by_index
            and record.get("right_support_index") in x_by_index
        ]

    left_offsets: list[float] = []
    right_offsets: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    donor_ids: list[str] = []

    for record in donors:
        box = record["bbox_xyxy"]
        left_x = x_by_index[int(record["left_support_index"])]
        right_x = x_by_index[int(record["right_support_index"])]
        left_offsets.append(float(box[0]) - left_x)
        right_offsets.append(float(box[2]) - right_x)
        tops.append(float(box[1]))
        bottoms.append(float(box[3]))
        widths.append(box_width(box))
        heights.append(box_height(box))
        donor_ids.append(record["arch_id"])

    observed_arches = [item for item in observed_lookup.values() if item.get("class") == "arch"]
    fallback = median_geometry(observed_arches)
    return {
        "left_offset": float(median(left_offsets)) if left_offsets else -fallback["width"] * 0.08,
        "right_offset": float(median(right_offsets)) if right_offsets else fallback["width"] * 0.08,
        "top": float(median(tops)) if tops else fallback["y1"],
        "bottom": float(median(bottoms)) if bottoms else fallback["y2"],
        "width": float(median(widths)) if widths else fallback["width"],
        "height": float(median(heights)) if heights else fallback["height"],
        "donor_ids": donor_ids,
        "donor_count": len(donor_ids),
    }


def build_pilaster_reference_model(
    pilaster_records: list[dict[str, Any]],
    observed_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    donors = [
        record
        for record in pilaster_records
        if record.get("status") == "observed_intact"
        and record.get("reference_role") in {
            "strong_observed_reference",
            "observed_reference_with_constraints",
        }
    ]
    if not donors:
        donors = [record for record in pilaster_records if record.get("status") == "observed_intact"]

    donor_elements = [observed_lookup[record["pilaster_id"]] for record in donors if record["pilaster_id"] in observed_lookup]
    fallback_elements = [
        item for item in observed_lookup.values() if item.get("class") in {"pilaster", "column"}
    ]
    geom = median_geometry(donor_elements or fallback_elements)
    return {
        "width": geom["width"],
        "top": geom["y1"],
        "bottom": geom["y2"],
        "height": geom["height"],
        "donor_ids": [record["pilaster_id"] for record in donors],
        "donor_count": len(donors),
    }


def arch_box_for_span(
    left_x: float,
    right_x: float,
    model: dict[str, Any],
    width: int,
    height: int,
) -> list[int]:
    x1 = left_x + float(model["left_offset"])
    x2 = right_x + float(model["right_offset"])
    if x2 <= x1 + 4:
        center = (left_x + right_x) / 2.0
        target_width = max(20.0, float(model["width"]))
        x1, x2 = center - target_width / 2.0, center + target_width / 2.0
    return clip_box([x1, model["top"], x2, model["bottom"]], width, height)


def pilaster_box_for_axis(
    axis_x: float,
    model: dict[str, Any],
    width: int,
    height: int,
) -> list[int]:
    half = max(5.0, float(model["width"]) / 2.0)
    return clip_box(
        [axis_x - half, model["top"], axis_x + half, model["bottom"]],
        width,
        height,
    )


def generate_auto_hypotheses(
    image: Image.Image,
    damage_mask: np.ndarray,
    observed_elements: list[dict[str, Any]],
    sectors_doc: dict[str, Any],
    support_doc: dict[str, Any],
    bay_doc: dict[str, Any],
    arch_roles_doc: dict[str, Any],
    pilaster_roles_doc: dict[str, Any],
    courses_doc: dict[str, Any],
    cfg: Config,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    width, height = image.size
    observed = element_lookup(observed_elements)
    supports = support_doc.get("supports", [])
    intervals = bay_doc.get("adjacent_intervals", [])
    arch_records = arch_roles_doc.get("arch_records", [])
    pilaster_records = pilaster_roles_doc.get("pilaster_records", [])
    x_by_index = support_position_lookup(supports)

    arch_model = build_arch_reference_model(arch_records, supports, observed)
    pilaster_model = build_pilaster_reference_model(pilaster_records, observed)

    records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # A) Gözlenen parçalı kemerlerin tam geometri hipotezi.
    # ------------------------------------------------------------------
    for arch_record in arch_records:
        if arch_record.get("status") != "observed_fragmented":
            continue
        arch_id = arch_record.get("arch_id")
        if arch_id not in observed:
            continue
        left_index = arch_record.get("left_support_index")
        right_index = arch_record.get("right_support_index")
        fragment_box = observed[arch_id]["bbox_xyxy"]
        if left_index in x_by_index and right_index in x_by_index:
            predicted = arch_box_for_span(
                x_by_index[int(left_index)],
                x_by_index[int(right_index)],
                arch_model,
                width,
                height,
            )
            basis = "bounded observed support span + intact single-interval arch model"
        else:
            geom = median_geometry(
                item for item in observed.values() if item.get("class") == "arch" and item.get("status") == "observed_intact"
            )
            cx, _ = box_center(fragment_box)
            predicted = clip_box(
                [
                    cx - geom["width"] / 2.0,
                    geom["y1"],
                    cx + geom["width"] / 2.0,
                    geom["y2"],
                ],
                width,
                height,
            )
            basis = "intact arch median fallback"
        predicted = clip_box(union_boxes(predicted, fragment_box), width, height)
        damage = max(
            float(arch_record.get("damage_overlap_ratio", 0.0)),
            mask_ratio(damage_mask, predicted),
        )
        confidence = 0.70 + min(0.12, arch_model["donor_count"] * 0.025) + min(0.10, damage * 0.12)
        record = make_record(
            records,
            "arch",
            predicted,
            width,
            height,
            "fragmented_arch_full_geometry_hypothesis",
            "stage3c_c_fragment_completion",
            confidence,
            [
                "expert_observed_fragment",
                "bounded_by_observed_supports" if left_index in x_by_index and right_index in x_by_index else "family_geometry_fallback",
                "intact_arch_reference_geometry",
                "damage_mask_overlap",
            ],
            related_observed_ids=[arch_id, arch_record.get("left_support_id"), arch_record.get("right_support_id")],
            donor_ids=arch_model["donor_ids"],
            sector_id=arch_record.get("sector_id"),
            bay_id=(
                f"bay_interval_{int(left_index):02d}_{int(right_index):02d}"
                if left_index is not None and right_index is not None
                else None
            ),
            geometry_basis={
                "basis": basis,
                "fragment_bbox_xyxy": fragment_box,
                "reference_model": arch_model,
                "damage_overlap_ratio": round(damage, 6),
            },
        )
        records.append(record)

    # ------------------------------------------------------------------
    # B) Gözlenen parçalı pilasterların tam geometri hipotezi.
    # ------------------------------------------------------------------
    for pilaster_record in pilaster_records:
        if pilaster_record.get("status") != "observed_fragmented":
            continue
        pilaster_id = pilaster_record.get("pilaster_id")
        if pilaster_id not in observed:
            continue
        fragment_box = observed[pilaster_id]["bbox_xyxy"]
        axis_x = float(pilaster_record.get("x", box_center(fragment_box)[0]))
        predicted = pilaster_box_for_axis(axis_x, pilaster_model, width, height)
        predicted = clip_box(union_boxes(predicted, fragment_box), width, height)
        damage = max(
            float(pilaster_record.get("damage_overlap_ratio", 0.0)),
            mask_ratio(damage_mask, predicted),
        )
        confidence = 0.68 + min(0.14, pilaster_model["donor_count"] * 0.035) + min(0.08, damage * 0.10)
        record = make_record(
            records,
            "pilaster",
            predicted,
            width,
            height,
            "fragmented_pilaster_full_geometry_hypothesis",
            "stage3c_c_fragment_completion",
            confidence,
            [
                "expert_observed_fragment",
                "observed_support_axis",
                "intact_pilaster_reference_geometry",
                "damage_mask_overlap",
            ],
            related_observed_ids=[pilaster_id],
            donor_ids=pilaster_model["donor_ids"],
            sector_id=pilaster_record.get("sector_id"),
            geometry_basis={
                "basis": "observed support axis + intact pilaster median top/bottom/width",
                "fragment_bbox_xyxy": fragment_box,
                "support_axis_x": axis_x,
                "reference_model": pilaster_model,
                "damage_overlap_ratio": round(damage, 6),
            },
        )
        records.append(record)

    # ------------------------------------------------------------------
    # C) Doğrudan kemeri olmayan hasarlı bay için kemer hipotezi.
    # ------------------------------------------------------------------
    for interval in intervals:
        direct_ids = list(interval.get("direct_arch_ids", []))
        damage = float(interval.get("damage_overlap_ratio", 0.0))
        if direct_ids or damage < cfg.missing_arch_damage_min:
            continue
        left_index = int(interval["left_support_index"])
        right_index = int(interval["right_support_index"])
        if left_index not in x_by_index or right_index not in x_by_index:
            continue
        predicted = arch_box_for_span(
            x_by_index[left_index],
            x_by_index[right_index],
            arch_model,
            width,
            height,
        )
        confidence = 0.48 + min(0.14, damage * 0.18)
        # Komşu baylardaki gözlenen tek aralıklı kemerler güveni artırır.
        seq = int(interval.get("sequence_index", left_index))
        neighbors = [item for item in intervals if abs(int(item.get("sequence_index", 0)) - seq) == 1]
        neighbor_arch_count = sum(bool(item.get("direct_arch_ids")) for item in neighbors)
        confidence += neighbor_arch_count * 0.06
        records.append(
            make_record(
                records,
                "arch",
                predicted,
                width,
                height,
                "missing_arch_between_observed_supports_hypothesis",
                "stage3c_c_missing_bay_inference",
                confidence,
                [
                    "no_direct_observed_arch_in_bay",
                    "two_observed_bounding_supports",
                    "damage_mask_overlap",
                    "neighboring_bay_rhythm" if neighbor_arch_count else "limited_neighbor_evidence",
                ],
                related_observed_ids=[interval.get("left_support_id"), interval.get("right_support_id")],
                donor_ids=arch_model["donor_ids"],
                sector_id=interval.get("sector_id"),
                bay_id=interval.get("id"),
                geometry_basis={
                    "basis": "empty damaged adjacent-support interval",
                    "interval": interval,
                    "reference_model": arch_model,
                    "neighbor_arch_count": neighbor_arch_count,
                },
            )
        )

    # ------------------------------------------------------------------
    # D) Çok geniş, doğrudan kemersiz ve ağır hasarlı bay için destek hipotezi.
    #     Doğrudan parçalı kemer bulunan geniş baylarda destek icat ETMEZ.
    # ------------------------------------------------------------------
    reference_widths = [
        float(item.get("width_px", 0.0))
        for item in intervals
        if item.get("coverage_state") == "direct_intact_arch_observed"
        and float(item.get("damage_overlap_ratio", 0.0)) <= 0.30
        and float(item.get("width_px", 0.0)) > 0
    ]
    median_bay_width = float(median(reference_widths)) if reference_widths else 0.0
    if median_bay_width > 0:
        for interval in intervals:
            if interval.get("direct_arch_ids"):
                continue
            damage = float(interval.get("damage_overlap_ratio", 0.0))
            interval_width = float(interval.get("width_px", 0.0))
            if damage < cfg.missing_support_damage_min:
                continue
            if interval_width < median_bay_width * cfg.missing_support_width_factor:
                continue
            left_index = int(interval["left_support_index"])
            right_index = int(interval["right_support_index"])
            if left_index not in x_by_index or right_index not in x_by_index:
                continue
            axis_x = (x_by_index[left_index] + x_by_index[right_index]) / 2.0
            predicted = pilaster_box_for_axis(axis_x, pilaster_model, width, height)
            confidence = 0.42 + min(0.14, damage * 0.16)
            records.append(
                make_record(
                    records,
                    "pilaster",
                    predicted,
                    width,
                    height,
                    "missing_support_in_abnormally_wide_damaged_bay_hypothesis",
                    "stage3c_c_missing_support_inference",
                    confidence,
                    [
                        "abnormally_wide_bay_interval",
                        "no_direct_observed_arch",
                        "heavy_damage_mask_overlap",
                        "observed_side_bay_width_reference",
                    ],
                    related_observed_ids=[interval.get("left_support_id"), interval.get("right_support_id")],
                    donor_ids=pilaster_model["donor_ids"],
                    sector_id=interval.get("sector_id"),
                    bay_id=interval.get("id"),
                    geometry_basis={
                        "basis": "wide damaged interval subdivision",
                        "interval_width_px": interval_width,
                        "median_reference_bay_width_px": median_bay_width,
                        "inferred_axis_x": axis_x,
                        "reference_model": pilaster_model,
                    },
                )
            )

    # ------------------------------------------------------------------
    # E) Hasarla kesilen yatay kurslar.
    # ------------------------------------------------------------------
    for course in courses_doc.get("courses", []):
        class_name = str(course.get("class", "other"))
        if class_name not in {"cornice", "base"}:
            continue
        for gap_index, gap in enumerate(course.get("gaps", []), start=1):
            if not gap.get("interrupted_by_damage", False):
                continue
            gap_box = gap.get("gap_bbox_xyxy")
            if not isinstance(gap_box, list) or len(gap_box) != 4:
                continue
            damage = float(gap.get("damage_mask_ratio", 0.0))
            if damage < cfg.course_gap_damage_min:
                continue
            hypothesis_type = f"missing_{class_name}_course_segment_hypothesis"
            records.append(
                make_record(
                    records,
                    class_name,
                    gap_box,
                    width,
                    height,
                    hypothesis_type,
                    "stage3c_c_horizontal_course_gap",
                    0.72 + min(0.16, damage * 0.18),
                    [
                        "same_observed_horizontal_course",
                        "gap_between_distinct_observed_segments",
                        "documented_damage_interruption",
                    ],
                    related_observed_ids=[gap.get("left_member_id"), gap.get("right_member_id")],
                    donor_ids=[gap.get("left_member_id"), gap.get("right_member_id")],
                    sector_id="cornice_transition" if class_name == "cornice" else "base_course",
                    course_id=course.get("id"),
                    geometry_basis={
                        "basis": "damage-supported gap inside observed horizontal course",
                        "gap_index": gap_index,
                        "course": course,
                    },
                )
            )

    # Aynı sınıfta aşırı çakışan otomatik adayları güven sırasıyla ayıkla.
    deduped: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        duplicate = any(
            record["class"] == kept["class"]
            and record.get("related_observed_ids") == kept.get("related_observed_ids")
            and box_iou(record["bbox_xyxy"], kept["bbox_xyxy"]) >= 0.78
            for kept in deduped
        )
        if not duplicate:
            deduped.append(record)

    deduped.sort(key=lambda item: (item["class"], item["bbox_xyxy"][0], item["bbox_xyxy"][1]))
    for index, record in enumerate(deduped, start=1):
        record["display_number"] = index

    logger.info(
        "[AUTO] candidates=%d | arch=%d | pilaster=%d | course=%d",
        len(deduped),
        sum(item["class"] == "arch" for item in deduped),
        sum(item["class"] in {"pilaster", "column"} for item in deduped),
        sum(item["class"] in {"cornice", "base"} for item in deduped),
    )
    return deduped


# =============================================================================
# OUTPUT VISUALS / REPORT
# =============================================================================
def add_header(image: Image.Image, title: str, height: int = 42) -> Image.Image:
    output = Image.new("RGBA", (image.width, image.height + height), (15, 23, 42, 255))
    output.alpha_composite(image.convert("RGBA"), (0, height))
    draw = ImageDraw.Draw(output, "RGBA")
    draw.text((9, 11), title, fill=(248, 250, 252, 255), font=load_font(14))
    return output


def draw_observed_evidence(
    draw: ImageDraw.ImageDraw,
    observed_elements: list[dict[str, Any]],
    shift_y: int,
    labels: bool = True,
) -> None:
    font = load_font(12, bold=True)
    for item in observed_elements:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        y1 += shift_y
        y2 += shift_y
        color = (100, 116, 139, 180) if item["status"] == "observed_intact" else (148, 163, 184, 230)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if labels:
            draw.text(
                (x1 + 2, y1 + 2),
                f"O#{item['display_number']}",
                fill=color,
                font=font,
            )


def make_output_visual(
    image: Image.Image,
    damage_mask: np.ndarray,
    observed_elements: list[dict[str, Any]],
    records: list[dict[str, Any]],
    statuses: set[str],
    title: str,
    show_damage: bool = True,
) -> Image.Image:
    header_height = 42
    if show_damage:
        mask_pil = Image.fromarray(damage_mask, mode="L")
        base = image.convert("RGBA")
        red = Image.new("RGBA", image.size, (239, 68, 68, 0))
        red.putalpha(mask_pil.point(lambda value: 62 if value > 0 else 0))
        image_layer = Image.alpha_composite(base, red).convert("RGB")
    else:
        image_layer = image

    canvas = add_header(image_layer, title, header_height)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_observed_evidence(draw, observed_elements, header_height, labels=True)
    font = load_font(13, bold=True)

    for record in records:
        if record.get("deleted") or record.get("status") not in statuses:
            continue
        x1, y1, x2, y2 = record["bbox_xyxy"]
        y1 += header_height
        y2 += header_height
        style = STATUS_STYLE[record["status"]]
        rgb = style["rgb"]
        draw.rectangle([x1, y1, x2, y2], outline=(*rgb, 255), width=4)
        label = f"H#{record['display_number']} {record['class']} {record['confidence']:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        draw.rectangle([x1, max(header_height, y1 - 20), x1 + text_width + 8, max(header_height, y1 - 2)], fill=(*rgb, 230))
        draw.text((x1 + 4, max(header_height, y1 - 19)), label, fill=(15, 23, 42, 255), font=font)
    return canvas.convert("RGB")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    active = [item for item in records if not item.get("deleted")]
    by_status = defaultdict(int)
    by_class = defaultdict(int)
    by_source = defaultdict(int)
    for item in active:
        by_status[item["status"]] += 1
        by_class[item["class"]] += 1
        by_source[item["source"]] += 1
    return {
        "active_count": len(active),
        "approved": by_status["expert_approved"],
        "rejected": by_status["expert_rejected"],
        "unresolved": by_status["candidate_needs_review"],
        "by_class": dict(sorted(by_class.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def build_report(
    run_id: str,
    parent_run: Path,
    records: list[dict[str, Any]],
    finalized: bool,
) -> str:
    summary = summarize_records(records)
    lines = [
        "STAGE 3C-C — RECONSTRUCTION HYPOTHESES / EXPERT REVIEW",
        f"Run: {run_id}",
        f"Parent Stage 3C-B: {parent_run.resolve()}",
        f"State: {'FINALIZED / LOCKED' if finalized else 'DRAFT'}",
        "",
        "NON-NEGOTIABLE POLICY:",
        "- Observed expert inventory was not modified.",
        "- Reconstruction hypotheses are stored in a separate layer.",
        "- No candidate becomes Stage 4 geometry without expert approval.",
        "- Rejected and deleted candidates are excluded from approved hypotheses.",
        "- Stable Diffusion was not used.",
        "",
        f"Active hypotheses: {summary['active_count']}",
        f"Approved: {summary['approved']}",
        f"Rejected: {summary['rejected']}",
        f"Unresolved: {summary['unresolved']}",
        f"By class: {summary['by_class']}",
        f"By source: {summary['by_source']}",
        "",
        "APPROVED HYPOTHESES:",
    ]
    approved = [item for item in records if not item.get("deleted") and item["status"] == "expert_approved"]
    if not approved:
        lines.append("- none")
    for item in approved:
        lines.append(
            f"- {item['id']} | {item['class']} | {item['hypothesis_type']} | "
            f"bbox={item['bbox_xyxy']} | confidence={item['confidence']:.3f} | "
            f"related={item.get('related_observed_ids', [])}"
        )
    return "\n".join(lines) + "\n"


# =============================================================================
# ADD DIALOG
# =============================================================================
class AddHypothesisDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, cfg: Config) -> None:
        super().__init__(parent)
        self.title("Yeni rekonstrüksiyon hipotezi")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result: dict[str, Any] | None = None

        self.class_var = tk.StringVar(value=cfg.default_new_class)
        self.type_var = tk.StringVar(value="expert_added_reconstruction_hypothesis")
        self.conf_var = tk.DoubleVar(value=cfg.default_new_confidence)
        self.note_var = tk.StringVar(value="")

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Sınıf").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.class_var, values=ALLOWED_CLASSES, state="readonly", width=30).grid(row=0, column=1, pady=4)
        ttk.Label(frame, text="Hipotez türü").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.type_var, width=34).grid(row=1, column=1, pady=4)
        ttk.Label(frame, text="Güven (0–1)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, textvariable=self.conf_var, from_=0.0, to=1.0, increment=0.05, width=10).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Uzman notu").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.note_var, width=34).grid(row=3, column=1, pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="İptal", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="Ekle", command=self.accept).pack(side="right", padx=4)
        self.bind("<Return>", lambda _e: self.accept())
        self.bind("<Escape>", lambda _e: self.destroy())

    def accept(self) -> None:
        try:
            confidence = max(0.0, min(1.0, float(self.conf_var.get())))
        except Exception:
            messagebox.showerror("Geçersiz güven", "Güven 0 ile 1 arasında sayı olmalı.", parent=self)
            return
        self.result = {
            "class": self.class_var.get(),
            "hypothesis_type": self.type_var.get().strip() or "expert_added_reconstruction_hypothesis",
            "confidence": confidence,
            "note": self.note_var.get().strip(),
        }
        self.destroy()


# =============================================================================
# EXPERT EDITOR
# =============================================================================
class HypothesisEditor:
    def __init__(
        self,
        root: tk.Tk,
        cfg: Config,
        run_id: str,
        run_dir: Path,
        parent_run: Path,
        logger: logging.Logger,
        image: Image.Image,
        damage_mask: np.ndarray,
        observed_elements: list[dict[str, Any]],
        records: list[dict[str, Any]],
        copied_inputs: dict[str, Any],
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.run_id = run_id
        self.run_dir = run_dir
        self.parent_run = parent_run
        self.logger = logger
        self.image = image.convert("RGB")
        self.image_width, self.image_height = self.image.size
        self.damage_mask = damage_mask
        self.observed_elements = observed_elements
        self.records = records
        self.copied_inputs = copied_inputs
        self.actions: list[dict[str, Any]] = []
        self.history: list[tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]] = []
        self.selected_id: str | None = None
        self.mode = "select"
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.photo_image: ImageTk.PhotoImage | None = None
        self.temp_draw_box: list[int] | None = None
        self.draw_start_image: tuple[float, float] | None = None
        self.drag_mode: str | None = None
        self.drag_start_image: tuple[float, float] | None = None
        self.drag_original_box: list[int] | None = None
        self.active_handle: str | None = None
        self.pan_start: tuple[float, float] | None = None
        self.pan_offset_start: tuple[float, float] | None = None
        self.finalized = False
        self.saved_once = False

        self.root.title("Stage 3C-C — Rekonstrüksiyon Hipotezleri / Uzman Onayı")
        self.root.geometry(f"{cfg.editor_width}x{cfg.editor_height}")
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.show_damage_var = tk.BooleanVar(value=cfg.show_damage_default)
        self.show_observed_var = tk.BooleanVar(value=cfg.show_observed_default)
        self.show_labels_var = tk.BooleanVar(value=cfg.show_labels_default)
        self.status_message_var = tk.StringVar(value="Hazır")
        self.selected_status_var = tk.StringVar(value="-")
        self.class_var = tk.StringVar(value="")
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.note_var = tk.StringVar(value="")
        self.type_var = tk.StringVar(value="")
        self.id_var = tk.StringVar(value="")
        self.related_var = tk.StringVar(value="")

        self.build_ui()
        self.refresh_tree()
        self.root.after(120, self.fit_image)
        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Escape>", lambda _e: self.set_mode("select"))
        # A/R, Delete ve ok tuşları yalnız tuval odaktayken çalışır. Böylece
        # uzman notu veya tür alanına yazı yazarken yanlışlıkla karar verilmez.
        self.canvas.bind("<Delete>", lambda _e: self.delete_selected())
        self.canvas.bind("<KeyPress-a>", lambda _e: self.approve_selected())
        self.canvas.bind("<KeyPress-r>", lambda _e: self.reject_selected())
        self.canvas.bind("<Left>", lambda e: self.nudge_selected(-1, 0, e))
        self.canvas.bind("<Right>", lambda e: self.nudge_selected(1, 0, e))
        self.canvas.bind("<Up>", lambda e: self.nudge_selected(0, -1, e))
        self.canvas.bind("<Down>", lambda e: self.nudge_selected(0, 1, e))

    # ------------------------------------------------------------------ UI
    def build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(6, 6, 6, 3))
        top.pack(fill="x")
        ttk.Button(top, text="Seç / Taşı", command=lambda: self.set_mode("select")).pack(side="left", padx=2)
        ttk.Button(top, text="Yeni Hipotez Ekle", command=lambda: self.set_mode("add")).pack(side="left", padx=2)
        ttk.Button(top, text="Kutuyu Yeniden Çiz", command=self.start_redraw).pack(side="left", padx=2)
        ttk.Button(top, text="Kalıcı Sil", command=self.delete_selected).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=7)
        ttk.Button(top, text="ONAYLA (A)", command=self.approve_selected).pack(side="left", padx=2)
        ttk.Button(top, text="REDDET (R)", command=self.reject_selected).pack(side="left", padx=2)
        ttk.Button(top, text="Geri Al", command=self.undo).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=7)
        ttk.Button(top, text="Sığdır", command=self.fit_image).pack(side="left", padx=2)
        ttk.Button(top, text="%100", command=self.zoom_100).pack(side="left", padx=2)
        ttk.Checkbutton(top, text="Hasar", variable=self.show_damage_var, command=self.redraw_canvas).pack(side="left", padx=4)
        ttk.Checkbutton(top, text="Gözlenen", variable=self.show_observed_var, command=self.redraw_canvas).pack(side="left", padx=4)
        ttk.Checkbutton(top, text="Etiket", variable=self.show_labels_var, command=self.redraw_canvas).pack(side="left", padx=4)
        ttk.Button(top, text="Taslak Kaydet", command=lambda: self.save_outputs(finalize=False)).pack(side="right", padx=3)
        ttk.Button(top, text="KİLİTLE / STAGE 3'Ü BİTİR", command=lambda: self.save_outputs(finalize=True)).pack(side="right", padx=3)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=3)

        left = ttk.Frame(body)
        right = ttk.Frame(body, width=self.cfg.right_panel_width)
        body.add(left, weight=4)
        body.add(right, weight=1)

        self.canvas = tk.Canvas(left, bg="#111827", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_press)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self.on_linux_wheel(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self.on_linux_wheel(event, -1))
        self.canvas.bind("<Configure>", lambda _e: self.redraw_canvas())
        self.canvas.focus_set()

        ttk.Label(right, text="Hipotez Listesi", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(5, 2))
        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True, padx=6)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("no", "status", "class", "conf"),
            show="headings",
            height=13,
        )
        self.tree.heading("no", text="#")
        self.tree.heading("status", text="Durum")
        self.tree.heading("class", text="Sınıf")
        self.tree.heading("conf", text="Güven")
        self.tree.column("no", width=38, anchor="center")
        self.tree.column("status", width=82, anchor="center")
        self.tree.column("class", width=80, anchor="center")
        self.tree.column("conf", width=58, anchor="center")
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        inspector = ttk.LabelFrame(right, text="Seçili Hipotez", padding=8)
        inspector.pack(fill="x", padx=6, pady=6)
        self._inspector_row(inspector, 0, "ID", ttk.Label(inspector, textvariable=self.id_var))
        self._inspector_row(inspector, 1, "Durum", ttk.Label(inspector, textvariable=self.selected_status_var))
        class_combo = ttk.Combobox(inspector, textvariable=self.class_var, values=ALLOWED_CLASSES, state="readonly", width=22)
        self._inspector_row(inspector, 2, "Sınıf", class_combo)
        type_entry = ttk.Entry(inspector, textvariable=self.type_var, width=25)
        self._inspector_row(inspector, 3, "Tür", type_entry)
        conf_spin = ttk.Spinbox(inspector, textvariable=self.confidence_var, from_=0.0, to=1.0, increment=0.05, width=9)
        self._inspector_row(inspector, 4, "Güven", conf_spin)
        note_entry = ttk.Entry(inspector, textvariable=self.note_var, width=25)
        self._inspector_row(inspector, 5, "Not", note_entry)
        self._inspector_row(inspector, 6, "İlişkili", ttk.Label(inspector, textvariable=self.related_var, wraplength=210))
        ttk.Button(inspector, text="Alanları Uygula", command=self.apply_inspector).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        info = ttk.LabelFrame(right, text="Kullanım", padding=8)
        info.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            info,
            text=(
                "Turuncu: inceleme adayı\n"
                "Yeşil: onaylı\n"
                "Kırmızı: reddedilmiş\n\n"
                "Seçili kutu içinden sürüklenir. Beyaz tutamaçlarla boyutlandırılır.\n"
                "Tekerlek: zoom | Orta fare: pan\n"
                "A: onay | R: red | Del: sil | Ctrl+Z: geri al"
            ),
            justify="left",
            wraplength=300,
        ).pack(anchor="w")

        bottom = ttk.Frame(self.root, padding=(8, 3, 8, 6))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_message_var).pack(side="left")
        ttk.Label(bottom, text=f"Run: {self.run_id}").pack(side="right")

    def _inspector_row(self, parent: ttk.LabelFrame, row: int, label: str, widget: tk.Widget) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=3)
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ RECORDS
    def visible_records(self) -> list[dict[str, Any]]:
        return [item for item in self.records if not item.get("deleted")]

    def get_record(self, record_id: str | None) -> dict[str, Any] | None:
        if record_id is None:
            return None
        return next((item for item in self.records if item["id"] == record_id and not item.get("deleted")), None)

    def push_history(self) -> None:
        self.history.append((copy.deepcopy(self.records), copy.deepcopy(self.actions), self.selected_id))
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self) -> None:
        if not self.history:
            self.status_message_var.set("Geri alınacak işlem yok.")
            return
        self.records, self.actions, self.selected_id = self.history.pop()
        self.refresh_tree()
        self.update_inspector()
        self.redraw_canvas()
        self.status_message_var.set("Son işlem geri alındı.")

    def record_action(self, action: str, record: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        self.actions.append(
            {
                "time": now_iso(),
                "action": action,
                "record_id": record["id"],
                "display_number": record.get("display_number"),
                **(extra or {}),
            }
        )

    def refresh_tree(self) -> None:
        current = self.selected_id
        self.tree.delete(*self.tree.get_children())
        for item in sorted(self.visible_records(), key=lambda x: int(x.get("display_number", 0))):
            style = STATUS_STYLE[item["status"]]
            self.tree.insert(
                "",
                "end",
                iid=item["id"],
                values=(
                    item["display_number"],
                    style["label"],
                    item["class"],
                    f"{float(item['confidence']):.2f}",
                ),
            )
        if current and self.tree.exists(current):
            self.tree.selection_set(current)
            self.tree.see(current)

    def select_record(self, record_id: str | None) -> None:
        self.selected_id = record_id
        if record_id and self.tree.exists(record_id):
            self.tree.selection_set(record_id)
            self.tree.see(record_id)
        else:
            self.tree.selection_remove(*self.tree.selection())
        self.update_inspector()
        self.redraw_canvas()

    def on_tree_select(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        self.selected_id = selection[0] if selection else None
        self.update_inspector()
        self.redraw_canvas()

    def update_inspector(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            self.id_var.set("")
            self.selected_status_var.set("-")
            self.class_var.set("")
            self.type_var.set("")
            self.confidence_var.set(0.5)
            self.note_var.set("")
            self.related_var.set("")
            return
        self.id_var.set(record["id"])
        self.selected_status_var.set(STATUS_STYLE[record["status"]]["label"])
        self.class_var.set(record["class"])
        self.type_var.set(record.get("hypothesis_type", ""))
        self.confidence_var.set(float(record.get("confidence", 0.5)))
        self.note_var.set(record.get("expert_note", ""))
        self.related_var.set(", ".join(record.get("related_observed_ids", [])) or "-")

    def apply_inspector(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        self.push_history()
        old = {
            "class": record["class"],
            "hypothesis_type": record.get("hypothesis_type"),
            "confidence": record.get("confidence"),
            "expert_note": record.get("expert_note"),
        }
        record["class"] = self.class_var.get() or record["class"]
        record["hypothesis_type"] = self.type_var.get().strip() or record["hypothesis_type"]
        try:
            record["confidence"] = round(max(0.0, min(1.0, float(self.confidence_var.get()))), 6)
        except Exception:
            record["confidence"] = old["confidence"]
        record["expert_note"] = self.note_var.get().strip()
        record["expert_action"] = "expert_fields_corrected"
        record["expert_reviewed"] = True
        self.record_action("fields_applied", record, {"before": old})
        self.refresh_tree()
        self.update_inspector()
        self.redraw_canvas()

    def approve_selected(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        self.push_history()
        previous = record["status"]
        record["status"] = "expert_approved"
        record["expert_action"] = "expert_approved_reconstruction_hypothesis"
        record["expert_reviewed"] = True
        self.record_action("hypothesis_approved", record, {"previous_status": previous})
        self.refresh_tree()
        self.update_inspector()
        self.redraw_canvas()
        self.status_message_var.set(f"H#{record['display_number']} onaylandı.")

    def reject_selected(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        self.push_history()
        previous = record["status"]
        record["status"] = "expert_rejected"
        record["expert_action"] = "expert_rejected_reconstruction_hypothesis"
        record["expert_reviewed"] = True
        self.record_action("hypothesis_rejected", record, {"previous_status": previous})
        self.refresh_tree()
        self.update_inspector()
        self.redraw_canvas()
        self.status_message_var.set(f"H#{record['display_number']} reddedildi.")

    def delete_selected(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        if not messagebox.askyesno(
            "Hipotezi kalıcı sil",
            f"H#{record['display_number']} ({record['class']}) listeden tamamen silinsin mi?",
            parent=self.root,
        ):
            return
        self.push_history()
        record["deleted"] = True
        record["expert_action"] = "record_deleted_permanently_from_hypothesis_inventory"
        self.record_action("record_deleted_permanently_from_hypothesis_inventory", record)
        self.selected_id = None
        self.refresh_tree()
        self.update_inspector()
        self.redraw_canvas()

    def nudge_selected(self, dx: int, dy: int, event: tk.Event) -> None:
        record = self.get_record(self.selected_id)
        if record is None or self.mode != "select":
            return
        factor = self.cfg.nudge_fast_px if (event.state & 0x0001) else self.cfg.nudge_px
        self.push_history()
        x1, y1, x2, y2 = record["bbox_xyxy"]
        record["bbox_xyxy"] = clip_box(
            [x1 + dx * factor, y1 + dy * factor, x2 + dx * factor, y2 + dy * factor],
            self.image_width,
            self.image_height,
        )
        record["expert_action"] = "bbox_corrected"
        record["expert_reviewed"] = True
        self.record_action("bbox_nudged", record, {"dx": dx * factor, "dy": dy * factor})
        self.redraw_canvas()

    # ------------------------------------------------------------------ VIEW
    def fit_image(self) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        padding = self.cfg.initial_fit_padding
        self.scale = min(
            (canvas_width - 2 * padding) / self.image_width,
            (canvas_height - 2 * padding) / self.image_height,
        )
        self.scale = max(self.cfg.zoom_min, min(self.cfg.zoom_max, self.scale))
        self.offset_x = (canvas_width - self.image_width * self.scale) / 2.0
        self.offset_y = (canvas_height - self.image_height * self.scale) / 2.0
        self.redraw_canvas()

    def zoom_100(self) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.scale = 1.0
        self.offset_x = (canvas_width - self.image_width) / 2.0
        self.offset_y = (canvas_height - self.image_height) / 2.0
        self.redraw_canvas()

    def zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        new_scale = max(self.cfg.zoom_min, min(self.cfg.zoom_max, self.scale * factor))
        if abs(new_scale - self.scale) < 1e-9:
            return
        self.scale = new_scale
        self.offset_x = canvas_x - image_x * self.scale
        self.offset_y = canvas_y - image_y * self.scale
        self.redraw_canvas()

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset_x) / max(self.scale, 1e-9), (y - self.offset_y) / max(self.scale, 1e-9)

    def on_mousewheel(self, event: tk.Event) -> None:
        factor = self.cfg.zoom_step if event.delta > 0 else 1.0 / self.cfg.zoom_step
        self.zoom_at(event.x, event.y, factor)

    def on_linux_wheel(self, event: tk.Event, direction: int) -> None:
        factor = self.cfg.zoom_step if direction > 0 else 1.0 / self.cfg.zoom_step
        self.zoom_at(event.x, event.y, factor)

    def on_pan_press(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y)
        self.pan_offset_start = (self.offset_x, self.offset_y)

    def on_pan_drag(self, event: tk.Event) -> None:
        if self.pan_start is None or self.pan_offset_start is None:
            return
        self.offset_x = self.pan_offset_start[0] + event.x - self.pan_start[0]
        self.offset_y = self.pan_offset_start[1] + event.y - self.pan_start[1]
        self.redraw_canvas()

    # ------------------------------------------------------------------ DRAW
    def redraw_canvas(self) -> None:
        if not self.canvas.winfo_exists():
            return
        self.canvas.delete("all")
        display_width = max(1, int(round(self.image_width * self.scale)))
        display_height = max(1, int(round(self.image_height * self.scale)))
        resized = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        if self.show_damage_var.get():
            mask_pil = Image.fromarray(self.damage_mask, mode="L").resize(
                (display_width, display_height), Image.Resampling.NEAREST
            )
            overlay = Image.new("RGBA", resized.size, (239, 68, 68, 0))
            overlay.putalpha(mask_pil.point(lambda value: self.cfg.overlay_damage_alpha if value > 0 else 0))
            resized = Image.alpha_composite(resized.convert("RGBA"), overlay).convert("RGB")
        self.photo_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_image, anchor="nw", tags=("base_image",))

        if self.show_observed_var.get():
            for item in sorted(self.observed_elements, key=lambda x: box_area(x["bbox_xyxy"]), reverse=True):
                self.draw_observed_record(item)

        for record in sorted(self.visible_records(), key=lambda x: box_area(x["bbox_xyxy"]), reverse=True):
            self.draw_hypothesis_record(record)

        selected = self.get_record(self.selected_id)
        if selected:
            self.draw_handles(selected)

        if self.temp_draw_box is not None:
            x1, y1, x2, y2 = self.temp_draw_box
            cx1, cy1 = self.image_to_canvas(x1, y1)
            cx2, cy2 = self.image_to_canvas(x2, y2)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#38bdf8", width=2, dash=(6, 4))

    def draw_observed_record(self, item: dict[str, Any]) -> None:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        cx1, cy1 = self.image_to_canvas(x1, y1)
        cx2, cy2 = self.image_to_canvas(x2, y2)
        color = "#64748b" if item["status"] == "observed_intact" else "#94a3b8"
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=1, dash=(3, 3))
        if self.show_labels_var.get():
            self.canvas.create_text(cx1 + 3, cy1 + 3, text=f"O#{item['display_number']}", anchor="nw", fill="#e2e8f0", font=("Segoe UI", 8))

    def draw_hypothesis_record(self, record: dict[str, Any]) -> None:
        style = STATUS_STYLE[record["status"]]
        x1, y1, x2, y2 = record["bbox_xyxy"]
        cx1, cy1 = self.image_to_canvas(x1, y1)
        cx2, cy2 = self.image_to_canvas(x2, y2)
        selected = record["id"] == self.selected_id
        width = 4 if selected else 2
        self.canvas.create_rectangle(
            cx1,
            cy1,
            cx2,
            cy2,
            outline="#38bdf8" if selected else style["color"],
            width=width,
            dash=style["dash"],
            tags=("hypothesis", f"record:{record['id']}"),
        )
        label = f"H{record['display_number']}"
        if self.show_labels_var.get():
            label += f" {record['class']} {record['confidence']:.2f}"
        text_id = self.canvas.create_text(
            cx1 + 4,
            cy1 + 3,
            text=label,
            anchor="nw",
            fill="#ffffff",
            font=("Segoe UI Semibold", 9),
        )
        bounds = self.canvas.bbox(text_id)
        if bounds:
            bg = self.canvas.create_rectangle(
                bounds[0] - 3,
                bounds[1] - 2,
                bounds[2] + 3,
                bounds[3] + 2,
                fill="#0f172a",
                outline=style["color"],
                width=1,
            )
            self.canvas.tag_lower(bg, text_id)

    def handle_positions(self, box: list[int]) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        points = {
            "nw": (x1, y1), "n": (cx, y1), "ne": (x2, y1),
            "e": (x2, cy), "se": (x2, y2), "s": (cx, y2),
            "sw": (x1, y2), "w": (x1, cy),
        }
        return {name: self.image_to_canvas(*point) for name, point in points.items()}

    def draw_handles(self, record: dict[str, Any]) -> None:
        radius = self.cfg.handle_radius_px
        for name, (x, y) in self.handle_positions(record["bbox_xyxy"]).items():
            self.canvas.create_rectangle(
                x - radius, y - radius, x + radius, y + radius,
                fill="#ffffff", outline="#0284c7", width=2,
                tags=("handle", f"handle:{name}"),
            )

    def hit_handle(self, canvas_x: float, canvas_y: float) -> str | None:
        record = self.get_record(self.selected_id)
        if record is None:
            return None
        tolerance = self.cfg.handle_radius_px + 3
        for name, (x, y) in self.handle_positions(record["bbox_xyxy"]).items():
            if abs(canvas_x - x) <= tolerance and abs(canvas_y - y) <= tolerance:
                return name
        return None

    def hit_record(self, canvas_x: float, canvas_y: float) -> dict[str, Any] | None:
        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        tolerance = self.cfg.hit_tolerance_px / max(self.scale, 1e-6)
        candidates = []
        for record in self.visible_records():
            x1, y1, x2, y2 = record["bbox_xyxy"]
            if x1 - tolerance <= image_x <= x2 + tolerance and y1 - tolerance <= image_y <= y2 + tolerance:
                candidates.append(record)
        return min(candidates, key=lambda item: box_area(item["bbox_xyxy"])) if candidates else None

    # ------------------------------------------------------------------ POINTER
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.temp_draw_box = None
        self.draw_start_image = None
        messages = {
            "select": "Seçim modu: kutuyu taşıyın veya tutamaçlarla boyutlandırın.",
            "add": "Yeni hipotez: görüntü üzerinde bir dikdörtgen çizin.",
            "redraw": "Yeniden çizim: seçili hipotezin doğru kutusunu çizin.",
        }
        self.status_message_var.set(messages.get(mode, "Hazır"))
        self.canvas.configure(cursor="crosshair" if mode in {"add", "redraw"} else "arrow")
        self.redraw_canvas()

    def start_redraw(self) -> None:
        if self.get_record(self.selected_id) is None:
            messagebox.showinfo("Kutu seçilmedi", "Önce yeniden çizilecek hipotezi seçin.", parent=self.root)
            return
        self.set_mode("redraw")

    def on_left_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.mode in {"add", "redraw"}:
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix = max(0.0, min(self.image_width - 1.0, ix))
            iy = max(0.0, min(self.image_height - 1.0, iy))
            self.draw_start_image = (ix, iy)
            self.temp_draw_box = [int(ix), int(iy), int(ix) + 1, int(iy) + 1]
            self.drag_mode = "draw"
            return

        handle = self.hit_handle(event.x, event.y)
        if handle:
            record = self.get_record(self.selected_id)
            if record:
                self.push_history()
                self.active_handle = handle
                self.drag_mode = "resize"
                self.drag_start_image = self.canvas_to_image(event.x, event.y)
                self.drag_original_box = list(record["bbox_xyxy"])
            return

        record = self.hit_record(event.x, event.y)
        if record:
            self.select_record(record["id"])
            self.push_history()
            self.drag_mode = "move"
            self.drag_start_image = self.canvas_to_image(event.x, event.y)
            self.drag_original_box = list(record["bbox_xyxy"])
        else:
            self.select_record(None)

    def on_left_drag(self, event: tk.Event) -> None:
        if self.drag_mode == "draw" and self.draw_start_image is not None:
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix = max(0.0, min(self.image_width, ix))
            iy = max(0.0, min(self.image_height, iy))
            sx, sy = self.draw_start_image
            self.temp_draw_box = clip_box([min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy)], self.image_width, self.image_height)
            self.redraw_canvas()
            return

        record = self.get_record(self.selected_id)
        if record is None or self.drag_start_image is None or self.drag_original_box is None:
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        sx, sy = self.drag_start_image
        dx, dy = ix - sx, iy - sy
        x1, y1, x2, y2 = self.drag_original_box

        if self.drag_mode == "move":
            new_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        elif self.drag_mode == "resize" and self.active_handle:
            new_box = [x1, y1, x2, y2]
            if "w" in self.active_handle:
                new_box[0] = x1 + dx
            if "e" in self.active_handle:
                new_box[2] = x2 + dx
            if "n" in self.active_handle:
                new_box[1] = y1 + dy
            if "s" in self.active_handle:
                new_box[3] = y2 + dy
            if new_box[2] < new_box[0]:
                new_box[0], new_box[2] = new_box[2], new_box[0]
            if new_box[3] < new_box[1]:
                new_box[1], new_box[3] = new_box[3], new_box[1]
        else:
            return
        record["bbox_xyxy"] = clip_box(new_box, self.image_width, self.image_height)
        self.redraw_canvas()

    def on_left_release(self, _event: tk.Event) -> None:
        if self.drag_mode == "draw" and self.temp_draw_box is not None:
            box = list(self.temp_draw_box)
            self.temp_draw_box = None
            self.draw_start_image = None
            self.drag_mode = None
            if box_area(box) < 25:
                self.set_mode("select")
                return
            if self.mode == "add":
                self.finish_add_box(box)
            else:
                self.finish_redraw_box(box)
            return

        record = self.get_record(self.selected_id)
        if record is not None and self.drag_mode in {"move", "resize"}:
            record["expert_action"] = "bbox_corrected"
            record["expert_reviewed"] = True
            self.record_action(
                "bbox_moved" if self.drag_mode == "move" else "bbox_resized",
                record,
                {"before": self.drag_original_box, "after": list(record["bbox_xyxy"])},
            )
        self.drag_mode = None
        self.drag_start_image = None
        self.drag_original_box = None
        self.active_handle = None
        self.refresh_tree()
        self.update_inspector()

    def finish_add_box(self, box: list[int]) -> None:
        dialog = AddHypothesisDialog(self.root, self.cfg)
        self.root.wait_window(dialog)
        if dialog.result is None:
            self.set_mode("select")
            return
        self.push_history()
        record = make_record(
            self.records,
            dialog.result["class"],
            box,
            self.image_width,
            self.image_height,
            dialog.result["hypothesis_type"],
            "expert_added_reconstruction_hypothesis",
            dialog.result["confidence"],
            ["expert_architectural_judgement"],
            geometry_basis={"basis": "expert-drawn reconstruction hypothesis"},
            note=dialog.result["note"],
        )
        record["expert_action"] = "expert_added_reconstruction_hypothesis"
        record["expert_reviewed"] = True
        self.records.append(record)
        self.record_action("hypothesis_added", record)
        self.refresh_tree()
        self.select_record(record["id"])
        self.set_mode("select")

    def finish_redraw_box(self, box: list[int]) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            self.set_mode("select")
            return
        self.push_history()
        before = list(record["bbox_xyxy"])
        record["bbox_xyxy"] = clip_box(box, self.image_width, self.image_height)
        record["expert_action"] = "bbox_redrawn"
        record["expert_reviewed"] = True
        self.record_action("bbox_redrawn", record, {"before": before, "after": list(record["bbox_xyxy"])})
        self.refresh_tree()
        self.update_inspector()
        self.set_mode("select")

    # ------------------------------------------------------------------ SAVE
    def build_document(self, finalize: bool) -> dict[str, Any]:
        active = [copy.deepcopy(item) for item in self.records if not item.get("deleted")]
        approved = [item for item in active if item["status"] == "expert_approved"]
        rejected = [item for item in active if item["status"] == "expert_rejected"]
        unresolved = [item for item in active if item["status"] == "candidate_needs_review"]
        return {
            "run_id": self.run_id,
            "stage": STAGE_NAME,
            "inventory_state": "finalized" if finalize else "draft",
            "locked": bool(finalize),
            "parent_run": {
                "run_id": self.parent_run.name,
                "path": str(self.parent_run.resolve()),
            },
            "policy": {
                "observed_inventory_untouched": True,
                "hypotheses_are_separate_layer": True,
                "expert_approval_required_for_stage4": True,
                "expert_may_add_reconstruction_hypothesis": True,
                "rejected_or_deleted_hypotheses_excluded_from_stage4": True,
                "overlapping_observed_arches_remain_distinct": True,
                "stable_diffusion_used": False,
            },
            "summary": {
                **summarize_records(self.records),
                "expert_added": sum(item.get("source") == "expert_added_reconstruction_hypothesis" for item in active),
                "bbox_corrected": sum(item.get("bbox_xyxy") != item.get("original_bbox_xyxy") for item in active),
                "deleted_during_session": sum(action.get("action") == "record_deleted_permanently_from_hypothesis_inventory" for action in self.actions),
            },
            "hypotheses": approved if finalize else active,
            "approved_hypotheses": approved,
            "rejected_hypotheses": rejected,
            "unresolved_review": unresolved,
        }

    def save_outputs(self, finalize: bool) -> None:
        document = self.build_document(finalize)
        unresolved = document["unresolved_review"]
        if finalize and unresolved:
            messagebox.showwarning(
                "Çözümlenmemiş hipotezler",
                f"{len(unresolved)} hipotez hâlâ İNCELE durumunda.\n"
                "Her birini onayla, reddet veya sil; sonra tekrar kilitle.",
                parent=self.root,
            )
            return
        if finalize and not messagebox.askyesno(
            "Stage 3'ü bitir",
            "Onaylı rekonstrüksiyon hipotezleri kilitlensin ve Stage 3 tamamlansın mı?",
            parent=self.root,
        ):
            return

        try:
            if finalize:
                write_json(self.run_dir / "expert_approved_reconstruction_hypotheses.json", document)
            else:
                write_json(self.run_dir / "stage3c_c_expert_hypotheses_draft.json", document)

            write_json(
                self.run_dir / "expert_rejected_reconstruction_hypotheses.json",
                {
                    "run_id": self.run_id,
                    "stage": STAGE_NAME,
                    "hypotheses": document["rejected_hypotheses"],
                },
            )
            write_json(
                self.run_dir / "expert_hypothesis_review_actions.json",
                {
                    "run_id": self.run_id,
                    "action_count": len(self.actions),
                    "actions": self.actions,
                },
            )

            make_output_visual(
                self.image,
                self.damage_mask,
                self.observed_elements,
                self.records,
                {"expert_approved"},
                "STAGE 3C-C — EXPERT-APPROVED RECONSTRUCTION HYPOTHESES",
            ).save(self.run_dir / "stage3c_c_approved_hypotheses.png")
            make_output_visual(
                self.image,
                self.damage_mask,
                self.observed_elements,
                self.records,
                {"expert_rejected"},
                "STAGE 3C-C — EXPERT-REJECTED RECONSTRUCTION HYPOTHESES",
            ).save(self.run_dir / "stage3c_c_rejected_hypotheses.png")
            make_output_visual(
                self.image,
                self.damage_mask,
                self.observed_elements,
                self.records,
                {"expert_approved"},
                "STAGE 3C-C — OBSERVED EVIDENCE + APPROVED HYPOTHESIS LAYER",
            ).save(self.run_dir / "stage3c_c_evidence_and_approved_overlay.png")

            (self.run_dir / "stage3c_c_report.txt").write_text(
                build_report(self.run_id, self.parent_run, self.records, finalize),
                encoding="utf-8",
            )

            metadata = read_json(self.run_dir / "run_meta.json")
            metadata.update(
                {
                    "status": "completed" if finalize else "draft_saved",
                    "finished_at": now_iso(),
                    "inventory_state": "finalized" if finalize else "draft",
                    "locked": bool(finalize),
                    "summary": document["summary"],
                    "outputs": sorted(path.name for path in self.run_dir.iterdir() if path.is_file()),
                }
            )
            write_json(self.run_dir / "run_meta.json", metadata)
            self.saved_once = True

            if finalize:
                self.finalized = True
                self.logger.info(
                    "[FINAL] approved=%d rejected=%d deleted=%d",
                    document["summary"]["approved"],
                    document["summary"]["rejected"],
                    document["summary"]["deleted_during_session"],
                )
                messagebox.showinfo(
                    "Stage 3 tamamlandı",
                    "Rekonstrüksiyon hipotezleri kilitlendi.\n\n"
                    f"Çıktı: {self.run_dir.resolve()}\n\n"
                    "Sonraki adım Stage 4 geometrik rekonstrüksiyondur.",
                    parent=self.root,
                )
                self.root.destroy()
            else:
                self.logger.info("[DRAFT] expert hypothesis draft saved")
                messagebox.showinfo("Taslak kaydedildi", str(self.run_dir.resolve()), parent=self.root)
        except Exception as exc:
            self.logger.exception("[SAVE FAIL] Stage 3C-C save failed")
            (self.run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            messagebox.showerror("Kayıt hatası", str(exc), parent=self.root)

    def on_close(self) -> None:
        if self.finalized:
            self.root.destroy()
            return
        if messagebox.askyesno(
            "Editörü kapat",
            "Stage 3C-C kilitlenmedi. Editör kapatılsın mı?\n"
            "Taslak kaydetmek için önce Taslak Kaydet düğmesini kullan.",
            parent=self.root,
        ):
            metadata = read_json(self.run_dir / "run_meta.json")
            metadata["status"] = "editor_closed_without_finalization"
            metadata["finished_at"] = now_iso()
            write_json(self.run_dir / "run_meta.json", metadata)
            self.root.destroy()


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_stage(cfg: Config, auto_only: bool = False) -> Path:
    parent_run = find_parent_run(cfg)
    input_path = parent_run / "input.png"
    mask_path = parent_run / "completion_mask.png"
    inventory_path = parent_run / "source_expert_validated_inventory.json"
    courses_path = parent_run / "source_stage3c_horizontal_courses.json"
    sectors_path = parent_run / "stage3c_b_facade_sectors.json"
    supports_path = parent_run / "stage3c_b_support_axis_sequence.json"
    bays_path = parent_run / "stage3c_b_ordered_bay_sequence.json"
    arch_roles_path = parent_run / "stage3c_b_arch_role_subfamilies.json"
    pilaster_roles_path = parent_run / "stage3c_b_pilaster_role_subfamilies.json"

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    damage_mask = load_binary_mask(mask_path, image.size)
    inventory_doc = read_json(inventory_path)
    courses_doc = read_json(courses_path)
    sectors_doc = read_json(sectors_path)
    support_doc = read_json(supports_path)
    bay_doc = read_json(bays_path)
    arch_roles_doc = read_json(arch_roles_path)
    pilaster_roles_doc = read_json(pilaster_roles_path)

    validate_parent_docs(inventory_doc, sectors_doc, support_doc, bay_doc, cfg)
    observed_elements = normalize_inventory(inventory_doc, width, height)

    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started_at = now_iso()

    try:
        logger.info("=" * 84)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info(f"[STAGE] {STAGE_TITLE}")
        logger.info("[RULE] observed inventory immutable / hypotheses separate / expert approval required")
        logger.info(f"[INPUT] elements={len(observed_elements)} image={width}x{height}")
        logger.info("=" * 84)

        copied_inputs = {
            "input": copy_with_hash(input_path, run_dir / "input.png"),
            "completion_mask": copy_with_hash(mask_path, run_dir / "completion_mask.png"),
            "expert_inventory": copy_with_hash(inventory_path, run_dir / "source_expert_validated_inventory.json"),
            "stage3c_horizontal_courses": copy_with_hash(courses_path, run_dir / "source_stage3c_horizontal_courses.json"),
            "stage3c_b_facade_sectors": copy_with_hash(sectors_path, run_dir / "source_stage3c_b_facade_sectors.json"),
            "stage3c_b_support_axis_sequence": copy_with_hash(supports_path, run_dir / "source_stage3c_b_support_axis_sequence.json"),
            "stage3c_b_ordered_bay_sequence": copy_with_hash(bays_path, run_dir / "source_stage3c_b_ordered_bay_sequence.json"),
            "stage3c_b_arch_role_subfamilies": copy_with_hash(arch_roles_path, run_dir / "source_stage3c_b_arch_role_subfamilies.json"),
            "stage3c_b_pilaster_role_subfamilies": copy_with_hash(pilaster_roles_path, run_dir / "source_stage3c_b_pilaster_role_subfamilies.json"),
        }

        records = generate_auto_hypotheses(
            image,
            damage_mask,
            observed_elements,
            sectors_doc,
            support_doc,
            bay_doc,
            arch_roles_doc,
            pilaster_roles_doc,
            courses_doc,
            cfg,
            logger,
        )

        auto_doc = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "parent_run": str(parent_run.resolve()),
            "policy": {
                "observed_inventory_untouched": True,
                "auto_candidates_require_expert_review": True,
                "hypotheses_are_not_observed_evidence": True,
                "stable_diffusion_used": False,
            },
            "candidate_count": len(records),
            "hypotheses": records,
        }
        write_json(run_dir / "stage3c_c_auto_hypotheses.json", auto_doc)
        make_output_visual(
            image,
            damage_mask,
            observed_elements,
            records,
            {"candidate_needs_review"},
            "STAGE 3C-C — AUTOMATIC HYPOTHESIS CANDIDATES / EXPERT REVIEW REQUIRED",
        ).save(run_dir / "stage3c_c_auto_hypothesis_overlay.png")

        metadata = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "status": "auto_candidates_ready" if auto_only else "editor_open",
            "started_at": started_at,
            "parent_run": {
                "run_id": parent_run.name,
                "path": str(parent_run.resolve()),
            },
            "source_script": source_script_info(),
            "inputs": copied_inputs,
            "config": asdict(cfg),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pillow": getattr(Image, "__version__", "unknown"),
                "tk": tk.TkVersion,
            },
            "policy": auto_doc["policy"],
            "summary": {
                "observed_elements_preserved": len(observed_elements),
                "auto_hypothesis_candidates": len(records),
                "fragment_completion_candidates": sum(item["source"] == "stage3c_c_fragment_completion" for item in records),
                "missing_bay_candidates": sum(item["source"] == "stage3c_c_missing_bay_inference" for item in records),
                "missing_support_candidates": sum(item["source"] == "stage3c_c_missing_support_inference" for item in records),
                "course_gap_candidates": sum(item["source"] == "stage3c_c_horizontal_course_gap" for item in records),
            },
            "outputs": [
                "stage3c_c_auto_hypotheses.json",
                "stage3c_c_auto_hypothesis_overlay.png",
            ],
        }
        write_json(run_dir / "run_meta.json", metadata)

        if auto_only:
            logger.info("[AUTO ONLY] Editör açılmadı.")
            logger.info(f"[OUT] {run_dir.resolve()}")
            return run_dir

        root = tk.Tk()
        editor = HypothesisEditor(
            root,
            cfg,
            run_id,
            run_dir,
            parent_run,
            logger,
            image,
            damage_mask,
            observed_elements,
            records,
            copied_inputs,
        )
        root.mainloop()
        logger.info(f"[OUT] {run_dir.resolve()}")
        return run_dir

    except Exception as exc:
        logger.exception("[FAIL] Stage 3C-C failed")
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        failure_meta = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "status": "failed",
            "started_at": started_at,
            "finished_at": now_iso(),
            "parent_run": str(parent_run.resolve()),
            "source_script": source_script_info(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(cfg),
        }
        write_json(run_dir / "run_meta.json", failure_meta)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=STAGE_TITLE)
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Yalnız otomatik hipotez adaylarını üretir; uzman editörünü açmaz.",
    )
    parser.add_argument(
        "--parent-run",
        default="",
        help="İsteğe bağlı Stage 3C-B restore_* klasörü.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(parent_run_dir=args.parent_run)
    try:
        run_dir = run_stage(cfg, auto_only=args.auto_only)
        print(f"\nStage 3C-C çalışma klasörü: {run_dir.resolve()}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
