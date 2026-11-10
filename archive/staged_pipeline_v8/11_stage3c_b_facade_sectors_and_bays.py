#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3c_b_facade_sectors_and_bays.py
================================================================
V8 / STAGE 3C-B — Cephe sektörleri, destek dizisi ve gözlenen açıklık sırası

AMAÇ
----
Stage 3C-A tarafından üretilmiş gözlenen mimari ilişki verilerini okur ve:

- yapının fotoğrafta görülen cephe sektörlerini tanımlar,
- pilaster/düşey destek eksenlerini soldan sağa sıralar,
- her iki komşu destek arasındaki gözlenen açıklık aralığını kaydeder,
- kemerleri destek aralıklarına bağlar,
- birden fazla destek aralığını geçen ayrı kemerleri korur,
- kemer ve pilasterlar için mimari rol etiketleri üretir.

Bu aşama:
- hiçbir uzman elemanını silmez veya birleştirmez,
- kesişen/üst üste gelen iki kemeri duplicate saymaz,
- eksik kemer, pilaster, aks veya açıklık üretmez,
- hasarlı boşluğu doldurmaz,
- Stable Diffusion kullanmaz.

ÖNEMLİ MİMARİ AYRIM
-------------------
Bir kemer yalnız tek komşu pilaster aralığını geçebileceği gibi, iki veya daha
fazla gözlenen destek aralığını da kapsayabilir. Bu tür bir kemer:

    multi_interval_spanning_arch

olarak kaydedilir. Altındaki tek aralıklı kemerlerle birleştirilmez ve onların
hatalı kopyası sayılmaz. Böylece mimar/restoratörün ayrı eleman kararı korunur.

GİRDİLER (Stage 3C-A parent run)
--------------------------------
input.png
completion_mask.png
source_expert_validated_inventory.json  (veya expert_validated_inventory.json)
stage3c_element_families.json
stage3c_alignment_axes.json
stage3c_arch_pilaster_relations.json
stage3c_horizontal_courses.json

ÇIKTILAR
--------
Her çalıştırmada yeni klasör:
    outputs/restore_YYYYMMDD_HHMMSS

input.png
completion_mask.png
source_expert_validated_inventory.json
source_stage3c_element_families.json
source_stage3c_alignment_axes.json
source_stage3c_arch_pilaster_relations.json
source_stage3c_horizontal_courses.json
stage3c_b_facade_sectors.json
stage3c_b_support_axis_sequence.json
stage3c_b_ordered_bay_sequence.json
stage3c_b_arch_role_subfamilies.json
stage3c_b_pilaster_role_subfamilies.json
stage3c_b_sector_overlay.png
stage3c_b_bay_sequence_overlay.png
stage3c_b_role_overlay.png
stage3c_b_report.txt
run.log
run_meta.json

KULLANIM
--------
Windows örneği:
    C:\\Users\\ekale.ELIFK\\.conda\\envs\\ImageToRender_GPU\\python.exe \\
    E:\\ImageToRender\\ImageToRenderStage3C_B.py
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
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    require_stage3c_a_policy: bool = True
    require_locked_inventory: bool = True

    reference_width_px: int = 628
    reference_height_px: int = 904

    # Merkezi hasar sektörü için yatay kurs boşluğu tercih edilir. Kurs verisi
    # yoksa completion mask içindeki merkezî büyük bileşen kullanılır.
    prefer_cornice_damage_gap_for_central_sector: bool = True
    central_component_min_area_ratio: float = 0.003
    central_component_center_bonus: float = 1.35

    # Rol sınıflandırma toleransları.
    overlap_horizontal_ratio_min: float = 0.28
    overlap_iou_min: float = 0.025
    strong_reference_support_score_min: float = 0.68
    low_damage_overlap_ratio_max: float = 0.25

    # Görsel çıktı.
    output_header_height: int = 42
    output_box_thickness: int = 3
    output_line_thickness: int = 2
    output_label_font_size: int = 14
    sector_fill_alpha: int = 48
    draw_element_numbers: bool = True


STAGE_NAME = "v8_stage3c_b_facade_sectors_and_bays"
STAGE_TITLE = "Stage 3C-B — cephe sektörleri ve gözlenen açıklık dizisi"

CLASS_COLORS: dict[str, tuple[int, int, int, int]] = {
    "arch": (34, 197, 94, 255),
    "pilaster": (14, 165, 233, 255),
    "opening": (168, 85, 247, 255),
    "window": (168, 85, 247, 255),
    "door": (139, 92, 246, 255),
    "cornice": (236, 72, 153, 255),
    "base": (245, 158, 11, 255),
    "capital": (20, 184, 166, 255),
    "column": (6, 182, 212, 255),
    "other": (148, 163, 184, 255),
}

SECTOR_STYLE: dict[str, dict[str, Any]] = {
    "upper_body": {
        "short": "S1 UPPER",
        "color": (168, 85, 247, 255),
        "description": "Korniş geçişinin üstündeki gözlenen üst gövde.",
    },
    "cornice_transition": {
        "short": "S2 CORNICE",
        "color": (236, 72, 153, 255),
        "description": "Üst gövde ile alt kemerli düzen arasındaki korniş kuşağı.",
    },
    "lower_left_arcade": {
        "short": "S3 LEFT",
        "color": (20, 184, 166, 255),
        "description": "Merkezi hasar kesintisinin solundaki gözlenen alt kemerli sektör.",
    },
    "central_damaged_arcade": {
        "short": "S4 CENTRAL",
        "color": (239, 68, 68, 255),
        "description": "Belgelendirilmiş merkezi hasar ile kesilen alt kemerli sektör.",
    },
    "lower_right_arcade": {
        "short": "S5 RIGHT",
        "color": (56, 189, 248, 255),
        "description": "Merkezi hasar kesintisinin sağındaki gözlenen alt kemerli sektör.",
    },
    "base_course": {
        "short": "S6 BASE",
        "color": (245, 158, 11, 255),
        "description": "Alt kaide/plinth parçalarının gözlenen yatay kuşağı.",
    },
}


# =============================================================================
# DATA MODEL
# =============================================================================
@dataclass(frozen=True)
class Element:
    id: str
    display_number: int
    cls: str
    status: str
    bbox: tuple[int, int, int, int]
    center_x: float
    center_y: float
    width: float
    height: float
    source: str
    usable_as_prototype: bool
    expert_action: str
    expert_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_number": self.display_number,
            "class": self.cls,
            "status": self.status,
            "bbox_xyxy": list(self.bbox),
            "center": [round(self.center_x, 4), round(self.center_y, 4)],
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "source": self.source,
            "usable_as_prototype": self.usable_as_prototype,
            "expert_action": self.expert_action,
            "expert_note": self.expert_note,
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
    logger = logging.getLogger(f"stage3c_b_{run_id}")
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


def resolve_inventory_path(parent: Path) -> Path | None:
    for name in (
        "source_expert_validated_inventory.json",
        "expert_validated_inventory.json",
    ):
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def is_compatible_parent(path: Path) -> bool:
    required = (
        "input.png",
        "completion_mask.png",
        "stage3c_element_families.json",
        "stage3c_alignment_axes.json",
        "stage3c_arch_pilaster_relations.json",
        "stage3c_horizontal_courses.json",
    )
    return (
        path.is_dir()
        and all((path / name).exists() for name in required)
        and resolve_inventory_path(path) is not None
    )


def find_parent_run(cfg: Config) -> tuple[Path, Path]:
    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not is_compatible_parent(parent):
            raise FileNotFoundError(
                "Belirtilen parent run Stage 3C-A için gerekli dosyaları içermiyor: "
                f"{parent.resolve()}"
            )
        inventory_path = resolve_inventory_path(parent)
        assert inventory_path is not None
        return parent, inventory_path

    candidates = sorted(
        (path for path in Path(cfg.output_root).glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if is_compatible_parent(candidate):
            inventory_path = resolve_inventory_path(candidate)
            assert inventory_path is not None
            return candidate, inventory_path

    raise FileNotFoundError(
        "Stage 3C-A çıktılarını içeren uyumlu restore_* klasörü bulunamadı."
    )


def copy_with_hash(source: Path, target: Path) -> dict[str, Any]:
    shutil.copy2(source, target)
    return {
        "source": str(source.resolve()),
        "copied_to": target.name,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.where(np.asarray(image) >= 128, 255, 0).astype(np.uint8)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def clip_box(
    box: Iterable[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return x1, y1, x2, y2


def box_area(box: tuple[int, int, int, int]) -> int:
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ix1 = max(first[0], second[0])
    iy1 = max(first[1], second[1])
    ix2 = min(first[2], second[2])
    iy2 = min(first[3], second[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    union = box_area(first) + box_area(second) - intersection
    return float(intersection / max(1, union))


def horizontal_overlap_ratio(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    denominator = max(1, min(first[2] - first[0], second[2] - second[0]))
    return float(overlap / denominator)


def mask_ratio(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.mean(crop > 0))


def union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    if not boxes:
        return 0, 0, 1, 1
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


# =============================================================================
# INPUT NORMALIZATION / VALIDATION
# =============================================================================
def normalize_inventory(
    inventory_doc: dict[str, Any],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> list[Element]:
    if cfg.require_locked_inventory:
        if not inventory_doc.get("locked") or inventory_doc.get("inventory_state") != "finalized":
            raise RuntimeError("Stage 3C-B yalnız kilitlenmiş uzman envanteriyle çalışır.")

    unresolved = inventory_doc.get("unresolved_review", [])
    if unresolved:
        raise RuntimeError("Uzman envanterinde çözümlenmemiş review kaydı bulunuyor.")

    raw_elements = inventory_doc.get("elements", [])
    if not isinstance(raw_elements, list) or not raw_elements:
        raise RuntimeError("Uzman envanterinde eleman bulunamadı.")

    elements: list[Element] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_elements, start=1):
        if not isinstance(item, dict):
            continue
        element_id = str(item.get("id", f"element_{index:03d}"))
        if element_id in seen_ids:
            raise RuntimeError(f"Yinelenen uzman eleman ID'si: {element_id}")
        seen_ids.add(element_id)

        bbox_raw = item.get("bbox_xyxy")
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            raise RuntimeError(f"Geçersiz bbox: {element_id}")
        bbox = clip_box(bbox_raw, image_width, image_height)
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        elements.append(
            Element(
                id=element_id,
                display_number=int(item.get("display_number", index)),
                cls=str(item.get("class", "other")).strip().lower(),
                status=str(item.get("status", "observed_fragmented")).strip().lower(),
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                source=str(item.get("source", "unknown")),
                usable_as_prototype=bool(item.get("usable_as_prototype", False)),
                expert_action=str(item.get("expert_action", "unmodified")),
                expert_note=str(item.get("expert_note", "")),
            )
        )

    return elements


def validate_stage3c_a_docs(
    families_doc: dict[str, Any],
    axes_doc: dict[str, Any],
    arch_relations_doc: dict[str, Any],
    courses_doc: dict[str, Any],
    cfg: Config,
) -> None:
    docs = [families_doc, arch_relations_doc, courses_doc]
    if cfg.require_stage3c_a_policy:
        for doc in docs:
            policy = doc.get("policy", {})
            if not policy.get("expert_ids_are_authoritative", False):
                raise RuntimeError("Parent Stage 3C-A uzman ID otoritesi politikasını taşımıyor.")
            if not policy.get("instances_never_auto_merged", False):
                raise RuntimeError("Parent Stage 3C-A auto-merge yasağını taşımıyor.")
            if not policy.get("no_missing_element_hypothesis", False):
                raise RuntimeError("Parent Stage 3C-A eksik eleman üretmeme politikasını taşımıyor.")

    if not isinstance(axes_doc.get("vertical_axes", []), list):
        raise RuntimeError("stage3c_alignment_axes.json düşey eksenleri geçersiz.")
    if not isinstance(arch_relations_doc.get("relations", []), list):
        raise RuntimeError("stage3c_arch_pilaster_relations.json geçersiz.")
    if not isinstance(courses_doc.get("courses", []), list):
        raise RuntimeError("stage3c_horizontal_courses.json geçersiz.")


# =============================================================================
# SECTOR DERIVATION
# =============================================================================
def largest_mask_component_box(
    damage_mask: np.ndarray,
    cfg: Config,
) -> tuple[int, int, int, int] | None:
    binary = np.where(damage_mask > 0, 1, 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None

    image_height, image_width = binary.shape
    min_area = image_width * image_height * cfg.central_component_min_area_ratio
    image_center_x = image_width / 2.0
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < min_area:
            continue
        center_x = float(centroids[label][0])
        centrality = 1.0 - min(1.0, abs(center_x - image_center_x) / max(1.0, image_width / 2.0))
        score = float(area) * (1.0 + cfg.central_component_center_bonus * centrality)
        box = (x, y, x + width, y + height)
        if best is None or score > best[0]:
            best = (score, box)
    return None if best is None else best[1]


def find_course(courses: list[dict[str, Any]], class_name: str) -> dict[str, Any] | None:
    candidates = [course for course in courses if course.get("class") == class_name]
    if not candidates:
        return None
    return max(candidates, key=lambda item: int(item.get("member_count", 0)))


def central_damage_gap_from_courses(
    courses: list[dict[str, Any]],
) -> tuple[int, int, int, int] | None:
    cornice = find_course(courses, "cornice")
    if cornice is None:
        return None
    gaps = [
        gap
        for gap in cornice.get("gaps", [])
        if gap.get("interrupted_by_damage")
        and isinstance(gap.get("gap_bbox_xyxy"), list)
        and len(gap["gap_bbox_xyxy"]) == 4
    ]
    if not gaps:
        return None
    chosen = max(
        gaps,
        key=lambda gap: (
            (gap["gap_bbox_xyxy"][2] - gap["gap_bbox_xyxy"][0])
            * float(gap.get("damage_mask_ratio", 0.0))
        ),
    )
    return tuple(int(value) for value in chosen["gap_bbox_xyxy"])


def derive_facade_sectors(
    elements: list[Element],
    courses: list[dict[str, Any]],
    damage_mask: np.ndarray,
    image_width: int,
    image_height: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    element_lookup = {element.id: element for element in elements}
    architectural = [
        element
        for element in elements
        if element.cls in {"arch", "pilaster", "opening", "cornice", "base", "column"}
    ]
    structure_box = union_box([element.bbox for element in architectural])

    cornice_course = find_course(courses, "cornice")
    base_course = find_course(courses, "base")

    central_gap = None
    central_basis = ""
    if cfg.prefer_cornice_damage_gap_for_central_sector:
        central_gap = central_damage_gap_from_courses(courses)
        if central_gap is not None:
            central_basis = "damage-supported gap between expert cornice segments"

    if central_gap is None:
        central_gap = largest_mask_component_box(damage_mask, cfg)
        central_basis = "largest central completion-mask component"

    if central_gap is None:
        central_x1 = int(round(image_width * 0.38))
        central_x2 = int(round(image_width * 0.62))
        central_basis = "conservative image-center fallback"
    else:
        central_x1, _, central_x2, _ = central_gap

    # Yapının görünür eleman sınırlarından çok uzaklaşmayalım.
    structure_x1 = max(0, structure_box[0])
    structure_x2 = min(image_width, structure_box[2])
    central_x1 = max(structure_x1 + 1, min(structure_x2 - 2, central_x1))
    central_x2 = max(central_x1 + 1, min(structure_x2 - 1, central_x2))

    if cornice_course:
        cornice_top = int(round(float(cornice_course.get("median_top_y", 0))))
        cornice_bottom = int(round(float(cornice_course.get("median_bottom_y", 0))))
    else:
        cornices = [element for element in elements if element.cls == "cornice"]
        cornice_top = min((element.bbox[1] for element in cornices), default=int(image_height * 0.28))
        cornice_bottom = max((element.bbox[3] for element in cornices), default=int(image_height * 0.39))

    if base_course:
        base_top = int(round(float(base_course.get("median_top_y", image_height * 0.68))))
        base_bottom = int(round(float(base_course.get("median_bottom_y", image_height * 0.79))))
    else:
        bases = [element for element in elements if element.cls == "base"]
        base_top = min((element.bbox[1] for element in bases), default=int(image_height * 0.68))
        base_bottom = max((element.bbox[3] for element in bases), default=int(image_height * 0.79))

    cornice_top = max(0, min(image_height - 1, cornice_top))
    cornice_bottom = max(cornice_top + 1, min(image_height, cornice_bottom))
    base_top = max(cornice_bottom + 1, min(image_height - 1, base_top))
    base_bottom = max(base_top + 1, min(image_height, base_bottom))

    upper_elements = [
        element
        for element in elements
        if element.center_y < cornice_top and element.cls not in {"cornice", "base"}
    ]
    upper_y1 = max(0, min((element.bbox[1] for element in upper_elements), default=0) - 60)

    sector_boxes = {
        "upper_body": (structure_x1, upper_y1, structure_x2, cornice_top),
        "cornice_transition": (structure_x1, cornice_top, structure_x2, cornice_bottom),
        "lower_left_arcade": (structure_x1, cornice_bottom, central_x1, base_top),
        "central_damaged_arcade": (central_x1, cornice_bottom, central_x2, base_top),
        "lower_right_arcade": (central_x2, cornice_bottom, structure_x2, base_top),
        "base_course": (structure_x1, base_top, structure_x2, base_bottom),
    }

    def choose_sector(element: Element) -> str:
        if element.cls == "cornice":
            return "cornice_transition"
        if element.cls == "base":
            return "base_course"
        if element.center_y < cornice_top:
            return "upper_body"
        if element.center_x < central_x1:
            return "lower_left_arcade"
        if element.center_x > central_x2:
            return "lower_right_arcade"
        return "central_damaged_arcade"

    element_to_sector = {element.id: choose_sector(element) for element in elements}
    sectors: list[dict[str, Any]] = []
    for order, sector_id in enumerate(
        (
            "upper_body",
            "cornice_transition",
            "lower_left_arcade",
            "central_damaged_arcade",
            "lower_right_arcade",
            "base_course",
        ),
        start=1,
    ):
        member_ids = [
            element.id
            for element in elements
            if element_to_sector[element.id] == sector_id
        ]
        style = SECTOR_STYLE[sector_id]
        sectors.append(
            {
                "id": sector_id,
                "order": order,
                "short_label": style["short"],
                "description": style["description"],
                "bbox_xyxy": list(sector_boxes[sector_id]),
                "member_ids": member_ids,
                "member_count": len(member_ids),
                "observed_only": True,
                "no_missing_geometry_inferred": True,
            }
        )

    diagnostics = {
        "central_sector_basis": central_basis,
        "central_gap_bbox_xyxy": list(central_gap) if central_gap else None,
        "central_x_bounds": [central_x1, central_x2],
        "cornice_bounds_y": [cornice_top, cornice_bottom],
        "base_bounds_y": [base_top, base_bottom],
        "structure_x_extent": [structure_x1, structure_x2],
        "mask_ratio_central_sector": round(
            mask_ratio(damage_mask, sector_boxes["central_damaged_arcade"]), 6
        ),
        "member_display_numbers": {
            sector_id: [element_lookup[item_id].display_number for item_id in member_ids]
            for sector_id in sector_boxes
            for member_ids in [[
                element.id
                for element in elements
                if element_to_sector[element.id] == sector_id
            ]]
        },
    }
    return sectors, element_to_sector, diagnostics


# =============================================================================
# SUPPORT AXIS / BAY SEQUENCE
# =============================================================================
def build_support_axis_sequence(
    elements: list[Element],
    axes_doc: dict[str, Any],
    element_to_sector: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    element_lookup = {element.id: element for element in elements}
    supports: list[dict[str, Any]] = []
    used_element_ids: set[str] = set()

    for axis in axes_doc.get("vertical_axes", []):
        member_ids = [
            member_id
            for member_id in axis.get("member_ids", [])
            if member_id in element_lookup
            and element_lookup[member_id].cls in {"pilaster", "column"}
        ]
        if not member_ids:
            continue
        # Stage 3C-A'da her gözlenen pilaster ekseni bağımsızdır. Birden fazla
        # üye varsa x'e en yakın üye seçilir; diğer kimlikler kaybolmaz, members'ta kalır.
        axis_x = float(axis.get("x", element_lookup[member_ids[0]].center_x))
        primary_id = min(
            member_ids,
            key=lambda member_id: abs(element_lookup[member_id].center_x - axis_x),
        )
        primary = element_lookup[primary_id]
        supports.append(
            {
                "axis_id": str(axis.get("id", f"support_axis_{len(supports)+1:02d}")),
                "x": round(axis_x, 4),
                "primary_element_id": primary_id,
                "member_ids": member_ids,
                "display_number": primary.display_number,
                "status": primary.status,
                "sector_id": element_to_sector.get(primary_id),
                "certainty": axis.get("certainty", "observed_geometry"),
            }
        )
        used_element_ids.update(member_ids)

    # Eksen dosyasında bulunmayan uzman pilasterı varsa kaybetmeyelim.
    for element in elements:
        if element.cls not in {"pilaster", "column"} or element.id in used_element_ids:
            continue
        supports.append(
            {
                "axis_id": f"fallback_axis_{element.id}",
                "x": round(element.center_x, 4),
                "primary_element_id": element.id,
                "member_ids": [element.id],
                "display_number": element.display_number,
                "status": element.status,
                "sector_id": element_to_sector.get(element.id),
                "certainty": "expert_bbox_center_fallback",
            }
        )

    supports.sort(key=lambda item: (float(item["x"]), item["primary_element_id"]))
    support_index_by_element: dict[str, int] = {}
    for index, support in enumerate(supports, start=1):
        support["sequence_index"] = index
        support["short_id"] = f"P{index}"
        for member_id in support["member_ids"]:
            support_index_by_element[member_id] = index

    return supports, support_index_by_element


def build_arch_spans(
    elements: list[Element],
    arch_relations_doc: dict[str, Any],
    support_index_by_element: dict[str, int],
    element_to_sector: dict[str, str],
    damage_mask: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    element_lookup = {element.id: element for element in elements}
    relations_by_arch = {
        relation.get("feature_id"): relation
        for relation in arch_relations_doc.get("relations", [])
        if relation.get("feature_id") in element_lookup
    }

    arch_elements = [element for element in elements if element.cls == "arch"]
    overlap_map: dict[str, list[str]] = defaultdict(list)
    for first_index, first in enumerate(arch_elements):
        for second in arch_elements[first_index + 1:]:
            iou = box_iou(first.bbox, second.bbox)
            horizontal = horizontal_overlap_ratio(first.bbox, second.bbox)
            if iou >= cfg.overlap_iou_min or horizontal >= cfg.overlap_horizontal_ratio_min:
                overlap_map[first.id].append(second.id)
                overlap_map[second.id].append(first.id)

    arch_spans: list[dict[str, Any]] = []
    for arch in sorted(arch_elements, key=lambda item: (item.center_x, item.bbox[1])):
        relation = relations_by_arch.get(arch.id, {})
        left_id = relation.get("left_pilaster_id")
        right_id = relation.get("right_pilaster_id")
        left_index = support_index_by_element.get(left_id)
        right_index = support_index_by_element.get(right_id)

        if left_index is not None and right_index is not None and left_index > right_index:
            left_index, right_index = right_index, left_index
            left_id, right_id = right_id, left_id

        span_intervals = (
            right_index - left_index
            if left_index is not None and right_index is not None
            else None
        )
        support_confidence_values = [
            float(value)
            for value in (relation.get("left_score"), relation.get("right_score"))
            if value is not None
        ]
        support_confidence = (
            min(support_confidence_values) if support_confidence_values else None
        )
        sector_id = element_to_sector.get(arch.id, "central_damaged_arcade")
        damage_overlap = mask_ratio(damage_mask, arch.bbox)

        role_tags: list[str] = []
        if span_intervals == 1:
            role_tags.append("single_interval_bay_arch")
        elif span_intervals is not None and span_intervals > 1:
            role_tags.append("multi_interval_spanning_arch")
        else:
            role_tags.append("support_span_unresolved")

        if sector_id in {"lower_left_arcade", "lower_right_arcade"}:
            role_tags.append("side_arcade_arch")
        if sector_id == "central_damaged_arcade":
            role_tags.append("central_damage_arch_evidence")
        if arch.status == "observed_fragmented":
            role_tags.append("fragmented_arch_evidence")
        else:
            role_tags.append("intact_observed_arch")
        if overlap_map.get(arch.id):
            role_tags.append("overlapping_distinct_arch")
        if left_index == 1 or right_index == max(support_index_by_element.values(), default=0):
            role_tags.append("outer_edge_arch")

        if arch.status == "observed_fragmented":
            reference_role = "fragment_evidence_only"
        elif span_intervals is not None and span_intervals > 1:
            reference_role = "context_reference_only"
        elif (
            support_confidence is not None
            and support_confidence >= cfg.strong_reference_support_score_min
            and damage_overlap <= cfg.low_damage_overlap_ratio_max
            and sector_id in {"lower_left_arcade", "lower_right_arcade"}
        ):
            reference_role = "strong_observed_reference"
        else:
            reference_role = "observed_reference_with_constraints"

        arch_spans.append(
            {
                "arch_id": arch.id,
                "display_number": arch.display_number,
                "status": arch.status,
                "bbox_xyxy": list(arch.bbox),
                "center": [round(arch.center_x, 4), round(arch.center_y, 4)],
                "sector_id": sector_id,
                "left_support_id": left_id,
                "right_support_id": right_id,
                "left_support_index": left_index,
                "right_support_index": right_index,
                "span_intervals": span_intervals,
                "support_relation_state": relation.get("relation_state", "unresolved"),
                "support_confidence": (
                    round(support_confidence, 6)
                    if support_confidence is not None
                    else None
                ),
                "damage_overlap_ratio": round(damage_overlap, 6),
                "overlapping_distinct_arch_ids": sorted(overlap_map.get(arch.id, [])),
                "role_tags": role_tags,
                "reference_role": reference_role,
                "safe_for_direct_copy": False,
                "not_a_completion_instruction": True,
            }
        )

    arch_spans.sort(
        key=lambda item: (
            item["left_support_index"] if item["left_support_index"] is not None else 999,
            item["span_intervals"] if item["span_intervals"] is not None else 999,
            item["center"][0],
        )
    )
    for order, span in enumerate(arch_spans, start=1):
        span["sequence_order"] = order
    return arch_spans


def sector_for_x(
    x: float,
    sector_diagnostics: dict[str, Any],
) -> str:
    central_x1, central_x2 = sector_diagnostics["central_x_bounds"]
    if x < central_x1:
        return "lower_left_arcade"
    if x > central_x2:
        return "lower_right_arcade"
    return "central_damaged_arcade"


def build_adjacent_intervals(
    supports: list[dict[str, Any]],
    arch_spans: list[dict[str, Any]],
    damage_mask: np.ndarray,
    sector_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    cornice_bottom = int(sector_diagnostics["cornice_bounds_y"][1])
    base_top = int(sector_diagnostics["base_bounds_y"][0])

    for zero_index in range(len(supports) - 1):
        left = supports[zero_index]
        right = supports[zero_index + 1]
        left_index = zero_index + 1
        right_index = zero_index + 2
        direct = [
            span
            for span in arch_spans
            if span["left_support_index"] == left_index
            and span["right_support_index"] == right_index
        ]
        spanning = [
            span
            for span in arch_spans
            if span["left_support_index"] is not None
            and span["right_support_index"] is not None
            and span["left_support_index"] <= left_index
            and span["right_support_index"] >= right_index
            and not (
                span["left_support_index"] == left_index
                and span["right_support_index"] == right_index
            )
        ]

        statuses = {span["status"] for span in direct}
        if not direct:
            coverage_state = "no_direct_observed_arch"
        elif statuses == {"observed_intact"}:
            coverage_state = "direct_intact_arch_observed"
        elif statuses == {"observed_fragmented"}:
            coverage_state = "direct_fragmented_arch_observed"
        else:
            coverage_state = "mixed_direct_arch_evidence"

        x1 = int(round(float(left["x"])))
        x2 = int(round(float(right["x"])))
        interval_box = (min(x1, x2), cornice_bottom, max(x1, x2), base_top)
        midpoint_x = (float(left["x"]) + float(right["x"])) / 2.0

        intervals.append(
            {
                "id": f"bay_interval_{left_index:02d}_{right_index:02d}",
                "sequence_index": left_index,
                "short_id": f"B{left_index}",
                "left_support_index": left_index,
                "right_support_index": right_index,
                "left_support_id": left["primary_element_id"],
                "right_support_id": right["primary_element_id"],
                "x_range": [round(float(left["x"]), 4), round(float(right["x"]), 4)],
                "width_px": round(abs(float(right["x"]) - float(left["x"])), 4),
                "midpoint_x": round(midpoint_x, 4),
                "sector_id": sector_for_x(midpoint_x, sector_diagnostics),
                "direct_arch_ids": [span["arch_id"] for span in direct],
                "direct_arch_display_numbers": [span["display_number"] for span in direct],
                "spanning_arch_ids": [span["arch_id"] for span in spanning],
                "spanning_arch_display_numbers": [span["display_number"] for span in spanning],
                "coverage_state": coverage_state,
                "damage_overlap_ratio": round(mask_ratio(damage_mask, interval_box), 6),
                "observed_interval_only": True,
                "absence_is_not_missing_element_inference": not bool(direct),
            }
        )

    return intervals


def build_primary_sequence_tokens(
    supports: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for index, support in enumerate(supports):
        tokens.append(
            {
                "token_type": "support",
                "token": support["short_id"],
                "support_index": support["sequence_index"],
                "element_id": support["primary_element_id"],
                "display_number": support["display_number"],
                "status": support["status"],
            }
        )
        if index < len(intervals):
            interval = intervals[index]
            tokens.append(
                {
                    "token_type": "bay_interval",
                    "token": interval["short_id"],
                    "interval_id": interval["id"],
                    "direct_arch_ids": interval["direct_arch_ids"],
                    "direct_arch_display_numbers": interval["direct_arch_display_numbers"],
                    "spanning_arch_ids": interval["spanning_arch_ids"],
                    "coverage_state": interval["coverage_state"],
                }
            )
    return tokens


# =============================================================================
# ROLE SUBFAMILIES
# =============================================================================
def make_role_groups(
    records: list[dict[str, Any]],
    id_key: str,
    role_names: list[str],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for role in role_names:
        members = [record[id_key] for record in records if role in record.get("role_tags", [])]
        if not members:
            continue
        groups.append(
            {
                "id": role,
                "role": role,
                "member_ids": members,
                "member_count": len(members),
                "non_exclusive_membership": True,
                "distinct_instances_preserved": True,
            }
        )
    return groups


def build_arch_role_subfamilies(
    arch_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    role_order = [
        "single_interval_bay_arch",
        "multi_interval_spanning_arch",
        "side_arcade_arch",
        "central_damage_arch_evidence",
        "outer_edge_arch",
        "overlapping_distinct_arch",
        "intact_observed_arch",
        "fragmented_arch_evidence",
        "support_span_unresolved",
    ]
    return {
        "role_group_count": len(make_role_groups(arch_spans, "arch_id", role_order)),
        "role_groups": make_role_groups(arch_spans, "arch_id", role_order),
        "arch_records": arch_spans,
        "reference_policy": {
            "strong_observed_reference_is_not_direct_copy_permission": True,
            "fragment_evidence_only_cannot_be_prototype": True,
            "multi_interval_arches_remain_distinct": True,
            "overlap_is_not_duplicate_evidence": True,
        },
    }


def build_pilaster_roles(
    elements: list[Element],
    supports: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    arch_spans: list[dict[str, Any]],
    element_to_sector: dict[str, str],
    damage_mask: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    element_lookup = {element.id: element for element in elements}
    direct_left_count: dict[int, int] = defaultdict(int)
    direct_right_count: dict[int, int] = defaultdict(int)
    multi_anchor_indices: set[int] = set()

    for span in arch_spans:
        left_index = span.get("left_support_index")
        right_index = span.get("right_support_index")
        if left_index is None or right_index is None:
            continue
        if span.get("span_intervals") == 1:
            direct_right_count[left_index] += 1
            direct_left_count[right_index] += 1
        elif span.get("span_intervals", 0) > 1:
            multi_anchor_indices.add(left_index)
            multi_anchor_indices.add(right_index)

    records: list[dict[str, Any]] = []
    support_count = len(supports)
    for support in supports:
        index = int(support["sequence_index"])
        element_id = support["primary_element_id"]
        element = element_lookup[element_id]
        role_tags: list[str] = []

        if index in {1, support_count}:
            role_tags.append("outer_boundary_support")
        if 1 < index < support_count:
            role_tags.append("intermediate_support")
        if direct_left_count[index] > 0 and direct_right_count[index] > 0:
            role_tags.append("shared_support_between_adjacent_bays")
        if index in multi_anchor_indices:
            role_tags.append("multi_interval_arch_anchor")
        sector_id = element_to_sector.get(element_id, "central_damaged_arcade")
        if sector_id == "central_damaged_arcade":
            role_tags.append("central_damage_support")
        elif sector_id in {"lower_left_arcade", "lower_right_arcade"}:
            role_tags.append("side_arcade_support")
        if element.status == "observed_fragmented":
            role_tags.append("fragmented_support_evidence")
        else:
            role_tags.append("intact_observed_support")

        damage_overlap = mask_ratio(damage_mask, element.bbox)
        if element.status == "observed_fragmented":
            reference_role = "fragment_evidence_only"
        elif (
            damage_overlap <= cfg.low_damage_overlap_ratio_max
            and sector_id in {"lower_left_arcade", "lower_right_arcade"}
        ):
            reference_role = "strong_observed_reference"
        else:
            reference_role = "observed_reference_with_constraints"

        records.append(
            {
                "pilaster_id": element_id,
                "display_number": element.display_number,
                "support_index": index,
                "support_axis_id": support["axis_id"],
                "x": support["x"],
                "status": element.status,
                "bbox_xyxy": list(element.bbox),
                "sector_id": sector_id,
                "direct_bays_on_left": direct_left_count[index],
                "direct_bays_on_right": direct_right_count[index],
                "damage_overlap_ratio": round(damage_overlap, 6),
                "role_tags": role_tags,
                "reference_role": reference_role,
                "safe_for_direct_copy": False,
                "not_a_completion_instruction": True,
            }
        )

    return records


def build_pilaster_role_subfamilies(
    pilaster_records: list[dict[str, Any]],
) -> dict[str, Any]:
    role_order = [
        "outer_boundary_support",
        "intermediate_support",
        "shared_support_between_adjacent_bays",
        "multi_interval_arch_anchor",
        "side_arcade_support",
        "central_damage_support",
        "intact_observed_support",
        "fragmented_support_evidence",
    ]
    groups = make_role_groups(pilaster_records, "pilaster_id", role_order)
    return {
        "role_group_count": len(groups),
        "role_groups": groups,
        "pilaster_records": pilaster_records,
        "reference_policy": {
            "strong_observed_reference_is_not_direct_copy_permission": True,
            "central_damage_supports_are_evidence_not_completion": True,
            "fragmented_supports_cannot_define_full_geometry_alone": True,
        },
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
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


def add_header(image: Image.Image, title: str, cfg: Config) -> Image.Image:
    header_height = cfg.output_header_height
    output = Image.new("RGBA", (image.width, image.height + header_height), (15, 23, 42, 255))
    output.alpha_composite(image.convert("RGBA"), (0, header_height))
    draw = ImageDraw.Draw(output, "RGBA")
    font = load_font(max(12, cfg.output_label_font_size), bold=False)
    draw.text((9, 11), title, fill=(248, 250, 252, 255), font=font)
    return output


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    padding = 3
    draw.rectangle(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        fill=(15, 23, 42, 225),
        outline=color,
        width=1,
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int = 8,
    gap: int = 5,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    position = 0.0
    while position < length:
        segment_end = min(length, position + dash)
        draw.line(
            [
                (x1 + dx * position, y1 + dy * position),
                (x1 + dx * segment_end, y1 + dy * segment_end),
            ],
            fill=fill,
            width=width,
        )
        position += dash + gap


def make_sector_overlay(
    image: Image.Image,
    elements: list[Element],
    sectors: list[dict[str, Any]],
    element_to_sector: dict[str, str],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(
        image,
        "STAGE 3C-B FACADE SECTORS — OBSERVED ORGANIZATION / NO COMPLETION",
        cfg,
    )
    shift_y = cfg.output_header_height
    font = load_font(cfg.output_label_font_size, bold=True)

    # Şeffaf dolgular ayrı katmanda alpha-composite edilir. Böylece sektör
    # renkleri fotoğrafı kapatmaz; yalnız okunabilir bir tonlama verir.
    sector_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sector_draw = ImageDraw.Draw(sector_layer, "RGBA")
    for sector in sectors:
        sector_id = sector["id"]
        color = SECTOR_STYLE[sector_id]["color"]
        x1, y1, x2, y2 = sector["bbox_xyxy"]
        sector_draw.rectangle(
            [x1, y1 + shift_y, x2, y2 + shift_y],
            fill=(*color[:3], cfg.sector_fill_alpha),
            outline=color,
            width=2,
        )
    canvas = Image.alpha_composite(canvas, sector_layer)
    draw = ImageDraw.Draw(canvas, "RGBA")

    for sector in sectors:
        sector_id = sector["id"]
        color = SECTOR_STYLE[sector_id]["color"]
        x1, y1, _, _ = sector["bbox_xyxy"]
        draw_label(
            draw,
            (x1 + 5, y1 + shift_y + 5),
            sector["short_label"],
            color,
            font,
        )

    for element in elements:
        sector_id = element_to_sector.get(element.id, "central_damaged_arcade")
        color = SECTOR_STYLE[sector_id]["color"]
        x1, y1, x2, y2 = element.bbox
        draw.rectangle(
            [x1, y1 + shift_y, x2, y2 + shift_y],
            outline=color,
            width=cfg.output_box_thickness,
        )
        if cfg.draw_element_numbers:
            # Korniş/base kutularında sektör etiketiyle çakışmayı azalt.
            label_offset_y = 24 if element.cls in {"cornice", "base"} else 3
            draw_label(
                draw,
                (x1 + 4, y1 + shift_y + label_offset_y),
                f"#{element.display_number}",
                color,
                font,
            )

    return canvas.convert("RGB")


def make_bay_sequence_overlay(
    image: Image.Image,
    elements: list[Element],
    supports: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    arch_spans: list[dict[str, Any]],
    sector_diagnostics: dict[str, Any],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(
        image,
        "STAGE 3C-B SUPPORT / BAY SEQUENCE — DISTINCT ARCH SPANS PRESERVED",
        cfg,
    )
    shift_y = cfg.output_header_height
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(cfg.output_label_font_size, bold=True)
    element_lookup = {element.id: element for element in elements}

    y_top = max(0, int(sector_diagnostics["cornice_bounds_y"][1]) - 18)
    y_bottom = min(image.height, int(sector_diagnostics["base_bounds_y"][1]) + 25)
    support_color = (56, 189, 248, 230)
    interval_color = (148, 163, 184, 220)

    for support in supports:
        x = int(round(float(support["x"])))
        draw.line(
            [(x, y_top + shift_y), (x, y_bottom + shift_y)],
            fill=support_color,
            width=cfg.output_line_thickness,
        )
        draw_label(draw, (x + 3, y_top + shift_y + 4), support["short_id"], support_color, font)

    band_y = min(image.height - 12, y_bottom - 5)
    for interval in intervals:
        x1, x2 = [int(round(value)) for value in interval["x_range"]]
        draw.line(
            [(x1, band_y + shift_y), (x2, band_y + shift_y)],
            fill=interval_color,
            width=max(1, cfg.output_line_thickness),
        )
        label_x = int(round((x1 + x2) / 2.0)) - 8
        draw_label(draw, (label_x, band_y + shift_y - 22), interval["short_id"], interval_color, font)

    # Önce kutular.
    for span in arch_spans:
        arch = element_lookup[span["arch_id"]]
        color = (245, 158, 11, 255) if arch.status == "observed_fragmented" else CLASS_COLORS["arch"]
        x1, y1, x2, y2 = arch.bbox
        draw.rectangle(
            [x1, y1 + shift_y, x2, y2 + shift_y],
            outline=color,
            width=cfg.output_box_thickness,
        )

    # Sonra destek span çizgileri. Aynı bölgede üst üste gelen kemerleri ayırmak
    # için sequence_order ile küçük düşey ofset uygulanır.
    for span in arch_spans:
        left_index = span.get("left_support_index")
        right_index = span.get("right_support_index")
        if left_index is None or right_index is None:
            continue
        left_x = float(supports[left_index - 1]["x"])
        right_x = float(supports[right_index - 1]["x"])
        arch = element_lookup[span["arch_id"]]
        line_y = max(8, arch.bbox[1] - 10 - ((span["sequence_order"] - 1) % 3) * 7)

        if span.get("span_intervals", 0) > 1:
            color = (236, 72, 153, 255)
            width = cfg.output_line_thickness + 2
        elif arch.status == "observed_fragmented":
            color = (245, 158, 11, 255)
            width = cfg.output_line_thickness + 1
        else:
            color = (34, 197, 94, 255)
            width = cfg.output_line_thickness + 1

        if arch.status == "observed_fragmented":
            draw_dashed_line(
                draw,
                (left_x, line_y + shift_y),
                (right_x, line_y + shift_y),
                color,
                width,
            )
        else:
            draw.line(
                [(left_x, line_y + shift_y), (right_x, line_y + shift_y)],
                fill=color,
                width=width,
            )
        short_role = "M" if span.get("span_intervals", 0) > 1 else "A"
        draw_label(
            draw,
            (int((left_x + right_x) / 2.0) - 12, line_y + shift_y - 18),
            f"{short_role}#{span['display_number']}",
            color,
            font,
        )

    return canvas.convert("RGB")


def make_role_overlay(
    image: Image.Image,
    elements: list[Element],
    arch_spans: list[dict[str, Any]],
    pilaster_records: list[dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(
        image,
        "STAGE 3C-B ARCH / PILASTER ROLES — OBSERVED ROLE TAGS ONLY",
        cfg,
    )
    shift_y = cfg.output_header_height
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(cfg.output_label_font_size, bold=True)
    element_lookup = {element.id: element for element in elements}

    for span in arch_spans:
        element = element_lookup[span["arch_id"]]
        if "multi_interval_spanning_arch" in span["role_tags"]:
            code = "M"
            color = (236, 72, 153, 255)
        elif "central_damage_arch_evidence" in span["role_tags"]:
            code = "C"
            color = (245, 158, 11, 255)
        elif "outer_edge_arch" in span["role_tags"]:
            code = "O"
            color = (132, 204, 22, 255)
        else:
            code = "S"
            color = CLASS_COLORS["arch"]
        x1, y1, x2, y2 = element.bbox
        draw.rectangle([x1, y1 + shift_y, x2, y2 + shift_y], outline=color, width=3)
        draw_label(draw, (x1 + 4, y1 + shift_y + 3), f"A#{element.display_number} {code}", color, font)

    for record in pilaster_records:
        element = element_lookup[record["pilaster_id"]]
        if "central_damage_support" in record["role_tags"]:
            code = "C"
            color = (239, 68, 68, 255)
        elif "outer_boundary_support" in record["role_tags"]:
            code = "O"
            color = (132, 204, 22, 255)
        elif "multi_interval_arch_anchor" in record["role_tags"]:
            code = "M"
            color = (236, 72, 153, 255)
        else:
            code = "I"
            color = CLASS_COLORS["pilaster"]
        x1, y1, x2, y2 = element.bbox
        draw.rectangle([x1, y1 + shift_y, x2, y2 + shift_y], outline=color, width=3)
        draw_label(draw, (x1 + 3, y2 + shift_y - 19), f"P{record['support_index']} {code}", color, font)

    return canvas.convert("RGB")


# =============================================================================
# REPORT
# =============================================================================
def build_report(
    run_id: str,
    parent_run: Path,
    elements: list[Element],
    sectors: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    arch_spans: list[dict[str, Any]],
    pilaster_records: list[dict[str, Any]],
    sector_diagnostics: dict[str, Any],
) -> str:
    single_spans = [span for span in arch_spans if span.get("span_intervals") == 1]
    multi_spans = [span for span in arch_spans if (span.get("span_intervals") or 0) > 1]
    unresolved_spans = [span for span in arch_spans if span.get("span_intervals") is None]
    central_arches = [
        span for span in arch_spans if span.get("sector_id") == "central_damaged_arcade"
    ]
    missing_direct = [
        interval for interval in intervals if not interval.get("direct_arch_ids")
    ]

    lines = [
        "STAGE 3C-B — FACADE SECTORS AND OBSERVED BAY SEQUENCE",
        f"Run: {run_id}",
        f"Parent: {parent_run.resolve()}",
        "",
        "NON-NEGOTIABLE POLICY:",
        "- Locked expert element IDs remain authoritative and distinct.",
        "- No element was deleted, merged, completed or newly invented.",
        "- A multi-interval arch is preserved as a separate observed architectural instance.",
        "- Overlap between arches is not duplicate evidence.",
        "- Empty support intervals are recorded only as observation state; they do not create missing elements.",
        "- Stable Diffusion was not used.",
        "",
        "INPUT:",
        f"- Expert elements: {len(elements)}",
        f"- Arches: {sum(element.cls == 'arch' for element in elements)}",
        f"- Pilasters/columns: {sum(element.cls in {'pilaster', 'column'} for element in elements)}",
        "",
        "DERIVED OBSERVED ORGANIZATION:",
        f"- Facade sectors: {len(sectors)}",
        f"- Ordered support axes: {len(supports)}",
        f"- Adjacent support intervals: {len(intervals)}",
        f"- Direct single-interval arch spans: {len(single_spans)}",
        f"- Multi-interval spanning arches: {len(multi_spans)}",
        f"- Unresolved support spans: {len(unresolved_spans)}",
        f"- Central damage arch evidence: {len(central_arches)}",
        f"- Intervals without direct observed arch: {len(missing_direct)}",
        "",
        "CENTRAL SECTOR:",
        f"- Basis: {sector_diagnostics['central_sector_basis']}",
        f"- X bounds: {sector_diagnostics['central_x_bounds']}",
        f"- Damage ratio: {sector_diagnostics['mask_ratio_central_sector']}",
        "",
        "ARCHITECTURAL INTERPRETATION LIMIT:",
        "- 'single_interval_bay_arch' describes an observed arch between adjacent supports.",
        "- 'multi_interval_spanning_arch' describes a distinct observed arch crossing more than one support interval.",
        "- 'strong_observed_reference' is evidence quality, not permission to copy geometry directly.",
        "- Central fragmented arches and supports remain fragment evidence only.",
        "",
        "NEXT REVIEW:",
        "- Confirm the left-to-right support sequence and every direct bay interval.",
        "- Confirm that multi-interval arches are correctly preserved as separate spans.",
        "- Confirm sector boundaries before any missing-element hypothesis stage.",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_stage(cfg: Config) -> Path:
    parent_run, inventory_path = find_parent_run(cfg)
    input_path = parent_run / "input.png"
    mask_path = parent_run / "completion_mask.png"
    families_path = parent_run / "stage3c_element_families.json"
    axes_path = parent_run / "stage3c_alignment_axes.json"
    arch_relations_path = parent_run / "stage3c_arch_pilaster_relations.json"
    courses_path = parent_run / "stage3c_horizontal_courses.json"

    image = Image.open(input_path).convert("RGB")
    image_width, image_height = image.size
    damage_mask = load_binary_mask(mask_path, image.size)
    inventory_doc = read_json(inventory_path)
    families_doc = read_json(families_path)
    axes_doc = read_json(axes_path)
    arch_relations_doc = read_json(arch_relations_path)
    courses_doc = read_json(courses_path)

    validate_stage3c_a_docs(
        families_doc,
        axes_doc,
        arch_relations_doc,
        courses_doc,
        cfg,
    )
    elements = normalize_inventory(inventory_doc, image_width, image_height, cfg)

    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started_at = now_iso()

    try:
        logger.info("=" * 80)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info(f"[STAGE] {STAGE_TITLE}")
        logger.info("[RULE] Expert IDs distinct / no merge / no missing hypothesis")
        logger.info(f"[INPUT] elements={len(elements)}, image={image_width}x{image_height}")
        logger.info("=" * 80)

        copied_inputs = {
            "input": copy_with_hash(input_path, run_dir / "input.png"),
            "completion_mask": copy_with_hash(mask_path, run_dir / "completion_mask.png"),
            "expert_inventory": copy_with_hash(
                inventory_path,
                run_dir / "source_expert_validated_inventory.json",
            ),
            "stage3c_element_families": copy_with_hash(
                families_path,
                run_dir / "source_stage3c_element_families.json",
            ),
            "stage3c_alignment_axes": copy_with_hash(
                axes_path,
                run_dir / "source_stage3c_alignment_axes.json",
            ),
            "stage3c_arch_pilaster_relations": copy_with_hash(
                arch_relations_path,
                run_dir / "source_stage3c_arch_pilaster_relations.json",
            ),
            "stage3c_horizontal_courses": copy_with_hash(
                courses_path,
                run_dir / "source_stage3c_horizontal_courses.json",
            ),
        }

        sectors, element_to_sector, sector_diagnostics = derive_facade_sectors(
            elements,
            courses_doc.get("courses", []),
            damage_mask,
            image_width,
            image_height,
            cfg,
        )
        supports, support_index_by_element = build_support_axis_sequence(
            elements,
            axes_doc,
            element_to_sector,
        )
        arch_spans = build_arch_spans(
            elements,
            arch_relations_doc,
            support_index_by_element,
            element_to_sector,
            damage_mask,
            cfg,
        )
        intervals = build_adjacent_intervals(
            supports,
            arch_spans,
            damage_mask,
            sector_diagnostics,
        )
        primary_tokens = build_primary_sequence_tokens(supports, intervals)
        pilaster_records = build_pilaster_roles(
            elements,
            supports,
            intervals,
            arch_spans,
            element_to_sector,
            damage_mask,
            cfg,
        )
        arch_roles = build_arch_role_subfamilies(arch_spans)
        pilaster_roles = build_pilaster_role_subfamilies(pilaster_records)

        common_policy = {
            "expert_ids_are_authoritative": True,
            "instances_never_auto_merged": True,
            "overlapping_arches_remain_distinct": True,
            "multi_interval_arches_remain_distinct": True,
            "observed_organization_only": True,
            "no_missing_element_hypothesis": True,
            "no_missing_axis_inference": True,
            "stable_diffusion_used": False,
        }

        write_json(
            run_dir / "stage3c_b_facade_sectors.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "parent_run": str(parent_run.resolve()),
                "policy": common_policy,
                "sector_count": len(sectors),
                "sectors": sectors,
                "element_to_sector": element_to_sector,
                "diagnostics": sector_diagnostics,
            },
        )
        write_json(
            run_dir / "stage3c_b_support_axis_sequence.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                "support_count": len(supports),
                "supports": supports,
                "element_to_support_index": support_index_by_element,
            },
        )
        write_json(
            run_dir / "stage3c_b_ordered_bay_sequence.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                "support_count": len(supports),
                "adjacent_interval_count": len(intervals),
                "arch_span_count": len(arch_spans),
                "supports": supports,
                "adjacent_intervals": intervals,
                "arch_spans": arch_spans,
                "primary_sequence_tokens": primary_tokens,
                "multi_interval_spans": [
                    span for span in arch_spans if (span.get("span_intervals") or 0) > 1
                ],
                "unresolved_spans": [
                    span for span in arch_spans if span.get("span_intervals") is None
                ],
            },
        )
        write_json(
            run_dir / "stage3c_b_arch_role_subfamilies.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                **arch_roles,
            },
        )
        write_json(
            run_dir / "stage3c_b_pilaster_role_subfamilies.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                **pilaster_roles,
            },
        )

        make_sector_overlay(
            image,
            elements,
            sectors,
            element_to_sector,
            cfg,
        ).save(run_dir / "stage3c_b_sector_overlay.png")
        make_bay_sequence_overlay(
            image,
            elements,
            supports,
            intervals,
            arch_spans,
            sector_diagnostics,
            cfg,
        ).save(run_dir / "stage3c_b_bay_sequence_overlay.png")
        make_role_overlay(
            image,
            elements,
            arch_spans,
            pilaster_records,
            cfg,
        ).save(run_dir / "stage3c_b_role_overlay.png")

        report = build_report(
            run_id,
            parent_run,
            elements,
            sectors,
            supports,
            intervals,
            arch_spans,
            pilaster_records,
            sector_diagnostics,
        )
        (run_dir / "stage3c_b_report.txt").write_text(report, encoding="utf-8")

        source_script = Path(__file__).resolve()
        output_files = [
            "input.png",
            "completion_mask.png",
            "source_expert_validated_inventory.json",
            "source_stage3c_element_families.json",
            "source_stage3c_alignment_axes.json",
            "source_stage3c_arch_pilaster_relations.json",
            "source_stage3c_horizontal_courses.json",
            "stage3c_b_facade_sectors.json",
            "stage3c_b_support_axis_sequence.json",
            "stage3c_b_ordered_bay_sequence.json",
            "stage3c_b_arch_role_subfamilies.json",
            "stage3c_b_pilaster_role_subfamilies.json",
            "stage3c_b_sector_overlay.png",
            "stage3c_b_bay_sequence_overlay.png",
            "stage3c_b_role_overlay.png",
            "stage3c_b_report.txt",
            "run.log",
            "run_meta.json",
        ]
        metadata = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "stage_title": STAGE_TITLE,
            "status": "success",
            "started_at": started_at,
            "finished_at": now_iso(),
            "parent_run": {
                "run_id": parent_run.name,
                "path": str(parent_run.resolve()),
            },
            "source_script": {
                "filename": source_script.name,
                "path": str(source_script),
                "sha256": sha256_file(source_script),
            },
            "inputs": copied_inputs,
            "config": asdict(cfg),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pillow": Image.__version__,
                "opencv": cv2.__version__,
            },
            "policy": common_policy,
            "summary": {
                "expert_elements_preserved": len(elements),
                "facade_sectors": len(sectors),
                "support_axes": len(supports),
                "adjacent_bay_intervals": len(intervals),
                "arch_spans": len(arch_spans),
                "single_interval_arches": sum(
                    span.get("span_intervals") == 1 for span in arch_spans
                ),
                "multi_interval_arches": sum(
                    (span.get("span_intervals") or 0) > 1 for span in arch_spans
                ),
                "unresolved_arch_spans": sum(
                    span.get("span_intervals") is None for span in arch_spans
                ),
                "intervals_without_direct_arch": sum(
                    not interval.get("direct_arch_ids") for interval in intervals
                ),
            },
            "outputs": output_files,
        }
        write_json(run_dir / "run_meta.json", metadata)

        logger.info(
            f"[COUNT] elements={len(elements)}, sectors={len(sectors)}, "
            f"supports={len(supports)}, intervals={len(intervals)}, arches={len(arch_spans)}"
        )
        logger.info(
            "[SPAN] "
            f"single={metadata['summary']['single_interval_arches']}, "
            f"multi={metadata['summary']['multi_interval_arches']}, "
            f"unresolved={metadata['summary']['unresolved_arch_spans']}"
        )
        logger.info("[RULE] 0 instances merged / 0 missing hypotheses generated")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info("[OK] Stage 3C-B completed")
        return run_dir

    except Exception as exc:
        logger.exception("[FAIL] Stage 3C-B failed")
        failure_meta = {
            "run_id": run_id,
            "stage": STAGE_NAME,
            "status": "failed",
            "started_at": started_at,
            "finished_at": now_iso(),
            "parent_run": str(parent_run.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(cfg),
        }
        write_json(run_dir / "run_meta.json", failure_meta)
        raise


# =============================================================================
# CLI ENTRY
# =============================================================================
def main() -> int:
    cfg = Config()
    try:
        run_dir = run_stage(cfg)
        print(f"\nStage 3C-B tamamlandı: {run_dir.resolve()}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
