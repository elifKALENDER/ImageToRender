#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3c_a_architectural_relations.py
================================================================
V8 / STAGE 3C-A — Uzman envanterinden mimari ilişki ve gramer grafiği

AMAÇ
----
Stage 3B-v5/v5b tarafından kilitlenmiş uzman gözlenen-eleman envanterini
okur ve yalnız fotoğrafta doğrulanmış elemanlar arasında geometrik/mimari
ilişkiler kurar.

Bu aşama:
- yeni veya eksik mimari eleman üretmez,
- hasarlı boşluğa geometri çizmez,
- Stable Diffusion kullanmaz,
- uzman tarafından ayrı kimlik verilmiş elemanları birleştirmez,
- kutu çakışmasını duplicate kanıtı saymaz,
- uzman envanterini değiştirmez.

Üretilen ilişkiler:
- eleman aileleri (kimlikler korunarak),
- aynı yatay kursta bulunan segmentler,
- kemer taç/üzengi hizaları,
- pilaster merkez eksenleri ve üst/alt hizaları,
- kemer–pilaster geometrik destek adayları,
- açıklık–pilaster yan sınır ilişkileri,
- aynı sınıftaki sağ/sol komşuluklar,
- bbox çakışmaları (yalnız mekânsal ilişki; duplicate değildir),
- yatay kurslar arasındaki hasar kesintileri.

GİRDİLER (parent run)
---------------------
input.png
completion_mask.png
expert_validated_inventory.json

ÇIKTILAR
--------
Her çalıştırmada yeni klasör:
    outputs/restore_YYYYMMDD_HHMMSS

input.png
completion_mask.png
source_expert_validated_inventory.json
stage3c_element_families.json
stage3c_alignment_axes.json
stage3c_arch_pilaster_relations.json
stage3c_opening_pilaster_relations.json
stage3c_horizontal_courses.json
stage3c_architecture_graph.json
stage3c_relationship_overlay.png
stage3c_family_overlay.png
stage3c_alignment_overlay.png
stage3c_damage_interruption_overlay.png
stage3c_report.txt
run.log
run_meta.json

KULLANIM
--------
Windows örneği:
    C:\\Users\\ekale.ELIFK\\.conda\\envs\\ImageToRender_GPU\\python.exe \
    E:\\ImageToRender\\ImageToRenderStage3C_A.py
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
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    require_locked_inventory: bool = True
    require_no_unresolved_review: bool = True

    # Referans görüntü 628x904. Toleranslar gerçek görüntü boyutuna ölçeklenir.
    reference_width_px: int = 628
    reference_height_px: int = 904

    family_threshold_arch: float = 0.56
    family_threshold_pilaster: float = 0.50
    family_threshold_opening: float = 0.72
    family_threshold_cornice: float = 0.58
    family_threshold_base: float = 0.56
    family_threshold_default: float = 0.64

    fragment_family_assignment_min_score: float = 0.36

    horizontal_course_tolerance_px: int = 30
    arch_spring_tolerance_px: int = 22
    arch_crown_tolerance_px: int = 30
    pilaster_top_tolerance_px: int = 26
    pilaster_bottom_tolerance_px: int = 38
    vertical_axis_cluster_tolerance_px: int = 9

    arch_support_x_tolerance_px: int = 46
    arch_support_y_tolerance_px: int = 78
    arch_support_min_score: float = 0.24

    opening_flank_x_tolerance_px: int = 52
    opening_flank_min_score: float = 0.22

    bbox_overlap_iou_min: float = 0.035
    damage_gap_mask_ratio_min: float = 0.08
    damage_gap_min_width_px: int = 10

    output_box_thickness: int = 3
    output_line_thickness: int = 2
    output_label_font_size: int = 15
    output_header_height: int = 42

    # Görselde uzun metinler yerine kısa numara/ilişki kodları kullanılır.
    draw_element_numbers: bool = True
    draw_support_relation_codes: bool = False


STAGE_NAME = "v8_stage3c_a_architectural_relations"
STAGE_TITLE = "Stage 3C-A — uzman envanterinden mimari ilişki ve gramer grafiği"

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

FAMILY_PALETTE: list[tuple[int, int, int, int]] = [
    (34, 197, 94, 255),
    (14, 165, 233, 255),
    (245, 158, 11, 255),
    (236, 72, 153, 255),
    (168, 85, 247, 255),
    (20, 184, 166, 255),
    (239, 68, 68, 255),
    (132, 204, 22, 255),
]


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
    logger = logging.getLogger(f"stage3c_a_{run_id}")
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
    required = (
        "input.png",
        "completion_mask.png",
        "expert_validated_inventory.json",
    )

    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not parent.is_dir():
            raise FileNotFoundError(
                f"Parent run klasörü bulunamadı: {parent.resolve()}"
            )
        missing = [name for name in required if not (parent / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Parent run zorunlu dosyaları eksik: " + ", ".join(missing)
            )
        return parent

    candidates = sorted(
        (
            path
            for path in Path(cfg.output_root).glob("restore_*")
            if path.is_dir()
        ),
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate

    raise FileNotFoundError(
        "expert_validated_inventory.json içeren uyumlu restore_* klasörü bulunamadı."
    )


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L").resize(
        (width, height),
        Image.Resampling.NEAREST,
    )
    return np.where(np.asarray(image) >= 128, 255, 0).astype(np.uint8)


def copy_with_hash(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "copied_to": destination.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


# =============================================================================
# GEOMETRY
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


def interval_overlap_ratio(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    denominator = max(1.0, min(first_end - first_start, second_end - second_start))
    return overlap / denominator


def exp_score(delta: float, scale: float) -> float:
    return math.exp(-abs(float(delta)) / max(1e-6, float(scale)))


def ratio_score(first: float, second: float, log_scale: float = 0.70) -> float:
    first = max(1e-6, float(first))
    second = max(1e-6, float(second))
    return math.exp(-abs(math.log(first / second)) / max(1e-6, log_scale))


def scaled_px(
    cfg: Config,
    value: int | float,
    image_width: int,
    image_height: int,
) -> float:
    scale = min(
        image_width / max(1.0, cfg.reference_width_px),
        image_height / max(1.0, cfg.reference_height_px),
    )
    return max(1.0, float(value) * scale)


# =============================================================================
# ELEMENT MODEL / VALIDATION
# =============================================================================
@dataclass
class Element:
    id: str
    display_number: int
    cls: str
    status: str
    bbox: list[int]
    center_x: float
    center_y: float
    width: float
    height: float
    source: str
    usable_as_prototype: bool
    expert_action: str
    expert_note: str
    raw: dict[str, Any]

    @property
    def is_intact(self) -> bool:
        return self.status == "observed_intact"

    @property
    def is_fragmented(self) -> bool:
        return self.status == "observed_fragmented"

    def as_node(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": "element",
            "display_number": self.display_number,
            "class": self.cls,
            "status": self.status,
            "bbox_xyxy": list(self.bbox),
            "center": [round(self.center_x, 3), round(self.center_y, 3)],
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "source": self.source,
            "usable_as_prototype": self.usable_as_prototype,
            "expert_action": self.expert_action,
            "expert_note": self.expert_note,
        }


def normalize_inventory(
    inventory_doc: dict[str, Any],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> list[Element]:
    if cfg.require_locked_inventory:
        locked = bool(inventory_doc.get("locked"))
        finalized = str(inventory_doc.get("inventory_state", "")).lower() == "finalized"
        if not (locked and finalized):
            raise RuntimeError(
                "Uzman envanteri kilitli/finalized değil. Önce Stage 3B editöründe "
                "Envanteri Kilitle ile kesin kayıt alın."
            )

    unresolved = inventory_doc.get("unresolved_review", [])
    if cfg.require_no_unresolved_review and unresolved:
        raise RuntimeError(
            f"Uzman envanterinde {len(unresolved)} çözümlenmemiş review kaydı var."
        )

    raw_elements = inventory_doc.get("elements", [])
    if not isinstance(raw_elements, list):
        raise TypeError("expert_validated_inventory.json içindeki elements liste değil.")

    elements: list[Element] = []
    seen_ids: set[str] = set()
    allowed_statuses = {"observed_intact", "observed_fragmented"}

    for index, item in enumerate(raw_elements, start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status not in allowed_statuses:
            # Stage 3C-A yalnız kesin gözlenen elemanları kullanır.
            continue
        element_id = str(item.get("id") or f"element_{index:03d}")
        if element_id in seen_ids:
            raise RuntimeError(f"Uzman envanterinde yinelenen ID var: {element_id}")
        seen_ids.add(element_id)

        box_value = item.get("bbox_xyxy")
        if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
            raise ValueError(f"Geçersiz bbox: {element_id}")
        bbox = clip_box(box_value, image_width, image_height)
        center_x, center_y = box_center(bbox)
        elements.append(
            Element(
                id=element_id,
                display_number=int(item.get("display_number", index)),
                cls=str(item.get("class", "other")).strip().lower(),
                status=status,
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                width=box_width(bbox),
                height=box_height(bbox),
                source=str(item.get("source", "unknown")),
                usable_as_prototype=bool(item.get("usable_as_prototype", False)),
                expert_action=str(item.get("expert_action", "unmodified")),
                expert_note=str(item.get("expert_note", "")),
                raw=item,
            )
        )

    if not elements:
        raise RuntimeError("Stage 3C-A için kullanılabilir gözlenen eleman yok.")
    elements.sort(key=lambda item: (item.display_number, item.id))
    return elements


# =============================================================================
# UNION-FIND / CLUSTERING
# =============================================================================
class UnionFind:
    def __init__(self, ids: Iterable[str]) -> None:
        id_list = list(ids)
        self.parent = {item: item for item in id_list}
        self.rank = {item: 0 for item in id_list}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1

    def groups(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            grouped[self.find(item)].append(item)
        return list(grouped.values())


def family_threshold(cfg: Config, class_name: str) -> float:
    return {
        "arch": cfg.family_threshold_arch,
        "pilaster": cfg.family_threshold_pilaster,
        "opening": cfg.family_threshold_opening,
        "window": cfg.family_threshold_opening,
        "door": cfg.family_threshold_opening,
        "cornice": cfg.family_threshold_cornice,
        "base": cfg.family_threshold_base,
    }.get(class_name, cfg.family_threshold_default)


def family_pair_similarity(
    first: Element,
    second: Element,
    image_width: int,
    image_height: int,
) -> tuple[float, dict[str, float]]:
    if first.cls != second.cls:
        return 0.0, {"class_match": 0.0}

    width_score = ratio_score(first.width, second.width)
    height_score = ratio_score(first.height, second.height)
    aspect_score = ratio_score(
        first.width / max(1.0, first.height),
        second.width / max(1.0, second.height),
        log_scale=0.55,
    )
    center_y_score = exp_score(first.center_y - second.center_y, image_height * 0.055)
    top_score = exp_score(first.bbox[1] - second.bbox[1], image_height * 0.052)
    bottom_score = exp_score(first.bbox[3] - second.bbox[3], image_height * 0.050)

    if first.cls == "arch":
        score = (
            0.23 * width_score
            + 0.15 * height_score
            + 0.18 * aspect_score
            + 0.16 * center_y_score
            + 0.10 * top_score
            + 0.18 * bottom_score
        )
    elif first.cls in {"pilaster", "column"}:
        score = (
            0.34 * width_score
            + 0.22 * height_score
            + 0.17 * aspect_score
            + 0.14 * top_score
            + 0.13 * bottom_score
        )
    elif first.cls in {"cornice", "base"}:
        vertical_overlap = interval_overlap_ratio(
            first.bbox[1], first.bbox[3], second.bbox[1], second.bbox[3]
        )
        score = (
            0.42 * center_y_score
            + 0.20 * height_score
            + 0.15 * aspect_score
            + 0.23 * vertical_overlap
        )
    elif first.cls in {"opening", "window", "door"}:
        score = (
            0.26 * width_score
            + 0.28 * height_score
            + 0.24 * aspect_score
            + 0.22 * center_y_score
        )
    else:
        score = (
            0.25 * width_score
            + 0.25 * height_score
            + 0.25 * aspect_score
            + 0.25 * center_y_score
        )

    evidence = {
        "width_similarity": round(width_score, 5),
        "height_similarity": round(height_score, 5),
        "aspect_similarity": round(aspect_score, 5),
        "center_y_similarity": round(center_y_score, 5),
        "top_similarity": round(top_score, 5),
        "bottom_similarity": round(bottom_score, 5),
    }
    return float(score), evidence


def select_family_representative(members: list[Element]) -> str | None:
    intact = [item for item in members if item.is_intact]
    if not intact:
        return None
    preferred = [item for item in intact if item.usable_as_prototype]
    pool = preferred or intact
    median_width = median([item.width for item in pool])
    median_height = median([item.height for item in pool])
    selected = min(
        pool,
        key=lambda item: (
            abs(math.log(max(1.0, item.width) / max(1.0, median_width)))
            + abs(math.log(max(1.0, item.height) / max(1.0, median_height))),
            item.display_number,
        ),
    )
    return selected.id


def fragment_assignment_gate(
    fragment: Element,
    intact_member: Element,
    image_width: int,
    image_height: int,
) -> bool:
    """Parçalı bir öğenin alakasız sağlam aileye bağlanmasını önler."""
    if fragment.cls != intact_member.cls:
        return False
    center_y_delta = abs(fragment.center_y - intact_member.center_y)
    if fragment.cls in {"opening", "window", "door"}:
        return (
            center_y_delta <= image_height * 0.12
            and ratio_score(fragment.width, intact_member.width) >= 0.30
            and ratio_score(fragment.height, intact_member.height) >= 0.30
        )
    if fragment.cls == "arch":
        return abs(fragment.bbox[3] - intact_member.bbox[3]) <= image_height * 0.10
    if fragment.cls in {"pilaster", "column"}:
        return ratio_score(fragment.width, intact_member.width) >= 0.28
    if fragment.cls in {"cornice", "base"}:
        return center_y_delta <= image_height * 0.10
    return center_y_delta <= image_height * 0.16


def build_element_families(
    elements: list[Element],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    by_class: dict[str, list[Element]] = defaultdict(list)
    for element in elements:
        by_class[element.cls].append(element)

    families: list[dict[str, Any]] = []
    membership: dict[str, str] = {}
    pair_diagnostics: list[dict[str, Any]] = []

    for class_name in sorted(by_class):
        class_elements = by_class[class_name]
        intact = [item for item in class_elements if item.is_intact]
        fragmented = [item for item in class_elements if item.is_fragmented]

        # Sağlam gözlemler aile çekirdeğini oluşturur. Uzman ID'leri asla birleşmez;
        # union-find yalnız aynı family ID'sine üyelik verir.
        seed_elements = intact or fragmented
        union_find = UnionFind(item.id for item in seed_elements)
        threshold = family_threshold(cfg, class_name)

        for first_index, first in enumerate(seed_elements):
            for second in seed_elements[first_index + 1 :]:
                score, evidence = family_pair_similarity(
                    first, second, image_width, image_height
                )
                accepted = score >= threshold
                pair_diagnostics.append(
                    {
                        "class": class_name,
                        "first_id": first.id,
                        "second_id": second.id,
                        "score": round(score, 6),
                        "threshold": threshold,
                        "same_family_link": accepted,
                        "evidence": evidence,
                        "policy": "family link only; instances remain distinct",
                    }
                )
                if accepted:
                    union_find.union(first.id, second.id)

        id_lookup = {item.id: item for item in class_elements}
        groups = [
            sorted(group, key=lambda item_id: id_lookup[item_id].display_number)
            for group in union_find.groups()
        ]
        groups.sort(key=lambda group: min(id_lookup[item_id].center_x for item_id in group))

        class_family_records: list[dict[str, Any]] = []
        for family_index, group_ids in enumerate(groups, start=1):
            group_members = [id_lookup[item_id] for item_id in group_ids]
            family_id = f"{class_name}_family_{family_index:02d}"
            for item_id in group_ids:
                membership[item_id] = family_id
            class_family_records.append(
                {
                    "id": family_id,
                    "class": class_name,
                    "member_ids": list(group_ids),
                    "intact_member_ids": [
                        item.id for item in group_members if item.is_intact
                    ],
                    "fragmented_member_ids": [
                        item.id for item in group_members if item.is_fragmented
                    ],
                    "representative_id": select_family_representative(group_members),
                    "member_count": len(group_members),
                    "basis": "expert class + conservative geometric similarity",
                    "distinct_instances": True,
                    "merge_allowed": False,
                    "geometry_summary": {
                        "median_width": round(median([item.width for item in group_members]), 3),
                        "median_height": round(median([item.height for item in group_members]), 3),
                        "median_center_y": round(
                            median([item.center_y for item in group_members]), 3
                        ),
                    },
                }
            )

        # Sağlam çekirdek varsa parçalı kayıtları en yakın aileye yalnız üyelik olarak ata.
        if intact and fragmented:
            for fragment in fragmented:
                if fragment.id in membership:
                    # Fragmented öğeler seed olarak kullanılmadığı durumda buraya düşer;
                    # intact yoksa yukarıdaki groups zaten hepsini kapsar.
                    continue
                scored_families: list[tuple[float, dict[str, Any]]] = []
                for family_record in class_family_records:
                    intact_members = [
                        id_lookup[item_id]
                        for item_id in family_record["intact_member_ids"]
                    ]
                    if not intact_members:
                        continue
                    member_scores = [
                        family_pair_similarity(
                            fragment, member, image_width, image_height
                        )[0]
                        for member in intact_members
                        if fragment_assignment_gate(
                            fragment, member, image_width, image_height
                        )
                    ]
                    if member_scores:
                        scored_families.append((max(member_scores), family_record))
                if scored_families:
                    best_score, best_family = max(scored_families, key=lambda item: item[0])
                    if best_score >= cfg.fragment_family_assignment_min_score:
                        best_family["member_ids"].append(fragment.id)
                        best_family["fragmented_member_ids"].append(fragment.id)
                        best_family["member_count"] += 1
                        membership[fragment.id] = best_family["id"]
                        best_family.setdefault("fragment_assignments", []).append(
                            {
                                "element_id": fragment.id,
                                "score": round(best_score, 6),
                                "certainty": "derived_geometry_only",
                            }
                        )

        # Atanmamış parçalı öğeler ayrı aile olur.
        unassigned_fragments = [item for item in fragmented if item.id not in membership]
        for fragment in unassigned_fragments:
            family_index = len(class_family_records) + 1
            family_id = f"{class_name}_family_{family_index:02d}"
            membership[fragment.id] = family_id
            class_family_records.append(
                {
                    "id": family_id,
                    "class": class_name,
                    "member_ids": [fragment.id],
                    "intact_member_ids": [],
                    "fragmented_member_ids": [fragment.id],
                    "representative_id": None,
                    "member_count": 1,
                    "basis": "isolated observed fragment; no intact family assignment",
                    "distinct_instances": True,
                    "merge_allowed": False,
                    "geometry_summary": {
                        "median_width": round(fragment.width, 3),
                        "median_height": round(fragment.height, 3),
                        "median_center_y": round(fragment.center_y, 3),
                    },
                }
            )

        families.extend(class_family_records)

    return families, membership, pair_diagnostics


# =============================================================================
# SCALAR ALIGNMENT CLUSTERS
# =============================================================================
def cluster_scalar_items(
    items: list[tuple[str, float]],
    tolerance: float,
) -> list[list[tuple[str, float]]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item[1])
    clusters: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = [ordered[0]]
    for item in ordered[1:]:
        current_median = median([value for _, value in current])
        if abs(item[1] - current_median) <= tolerance:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)
    return clusters


def build_alignment_axes(
    elements: list[Element],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> dict[str, Any]:
    by_class: dict[str, list[Element]] = defaultdict(list)
    for element in elements:
        by_class[element.cls].append(element)

    vertical_axes: list[dict[str, Any]] = []
    horizontal_axes: list[dict[str, Any]] = []

    pilasters = by_class.get("pilaster", []) + by_class.get("column", [])
    x_tolerance = scaled_px(
        cfg, cfg.vertical_axis_cluster_tolerance_px, image_width, image_height
    )
    x_clusters = cluster_scalar_items(
        [(item.id, item.center_x) for item in pilasters], x_tolerance
    )
    for index, cluster in enumerate(x_clusters, start=1):
        vertical_axes.append(
            {
                "id": f"pilaster_vertical_axis_{index:02d}",
                "axis_type": "vertical_center_axis",
                "class_scope": ["pilaster", "column"],
                "x": round(median([value for _, value in cluster]), 3),
                "member_ids": [item_id for item_id, _ in cluster],
                "member_count": len(cluster),
                "tolerance_px": round(x_tolerance, 3),
                "certainty": "observed_geometry",
            }
        )

    # Kemer taç ve üzengi çizgileri yalnız sağlam kemerlerle tanımlanır.
    intact_arches = [item for item in by_class.get("arch", []) if item.is_intact]
    crown_tolerance = scaled_px(
        cfg, cfg.arch_crown_tolerance_px, image_width, image_height
    )
    spring_tolerance = scaled_px(
        cfg, cfg.arch_spring_tolerance_px, image_width, image_height
    )
    crown_clusters = cluster_scalar_items(
        [(item.id, float(item.bbox[1])) for item in intact_arches], crown_tolerance
    )
    spring_clusters = cluster_scalar_items(
        [(item.id, float(item.bbox[3])) for item in intact_arches], spring_tolerance
    )
    for index, cluster in enumerate(crown_clusters, start=1):
        horizontal_axes.append(
            {
                "id": f"arch_crown_alignment_{index:02d}",
                "axis_type": "arch_crown_y",
                "y": round(median([value for _, value in cluster]), 3),
                "member_ids": [item_id for item_id, _ in cluster],
                "member_count": len(cluster),
                "tolerance_px": round(crown_tolerance, 3),
                "certainty": "observed_geometry",
            }
        )
    for index, cluster in enumerate(spring_clusters, start=1):
        horizontal_axes.append(
            {
                "id": f"arch_spring_alignment_{index:02d}",
                "axis_type": "arch_spring_y",
                "y": round(median([value for _, value in cluster]), 3),
                "member_ids": [item_id for item_id, _ in cluster],
                "member_count": len(cluster),
                "tolerance_px": round(spring_tolerance, 3),
                "certainty": "observed_geometry",
            }
        )

    intact_pilasters = [item for item in pilasters if item.is_intact]
    top_tolerance = scaled_px(
        cfg, cfg.pilaster_top_tolerance_px, image_width, image_height
    )
    bottom_tolerance = scaled_px(
        cfg, cfg.pilaster_bottom_tolerance_px, image_width, image_height
    )
    top_clusters = cluster_scalar_items(
        [(item.id, float(item.bbox[1])) for item in intact_pilasters], top_tolerance
    )
    bottom_clusters = cluster_scalar_items(
        [(item.id, float(item.bbox[3])) for item in intact_pilasters], bottom_tolerance
    )
    for index, cluster in enumerate(top_clusters, start=1):
        horizontal_axes.append(
            {
                "id": f"pilaster_top_alignment_{index:02d}",
                "axis_type": "pilaster_top_y",
                "y": round(median([value for _, value in cluster]), 3),
                "member_ids": [item_id for item_id, _ in cluster],
                "member_count": len(cluster),
                "tolerance_px": round(top_tolerance, 3),
                "certainty": "observed_geometry",
            }
        )
    for index, cluster in enumerate(bottom_clusters, start=1):
        horizontal_axes.append(
            {
                "id": f"pilaster_bottom_alignment_{index:02d}",
                "axis_type": "pilaster_bottom_y",
                "y": round(median([value for _, value in cluster]), 3),
                "member_ids": [item_id for item_id, _ in cluster],
                "member_count": len(cluster),
                "tolerance_px": round(bottom_tolerance, 3),
                "certainty": "observed_geometry",
            }
        )

    return {
        "policy": {
            "observed_only": True,
            "fragmented_elements_do_not_define_primary_alignment": True,
            "no_missing_axis_inference": True,
        },
        "vertical_axes": vertical_axes,
        "horizontal_axes": horizontal_axes,
    }


# =============================================================================
# HORIZONTAL COURSES / DAMAGE INTERRUPTIONS
# =============================================================================
def mask_ratio(mask: np.ndarray, box: Sequence[int]) -> float:
    height, width = mask.shape[:2]
    clipped = clip_box(box, width, height)
    x1, y1, x2, y2 = clipped
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.mean(crop > 0))


def build_horizontal_courses(
    elements: list[Element],
    damage_mask: np.ndarray,
    image_width: int,
    image_height: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    course_classes = {"cornice", "base"}
    tolerance = scaled_px(
        cfg, cfg.horizontal_course_tolerance_px, image_width, image_height
    )
    course_records: list[dict[str, Any]] = []
    membership: dict[str, str] = {}

    for class_name in sorted(course_classes):
        class_items = [item for item in elements if item.cls == class_name]
        clusters = cluster_scalar_items(
            [(item.id, item.center_y) for item in class_items], tolerance
        )
        lookup = {item.id: item for item in class_items}
        for index, cluster in enumerate(clusters, start=1):
            member_ids = [item_id for item_id, _ in cluster]
            members = sorted(
                [lookup[item_id] for item_id in member_ids],
                key=lambda item: item.bbox[0],
            )
            course_id = f"{class_name}_course_{index:02d}"
            for item in members:
                membership[item.id] = course_id

            gaps: list[dict[str, Any]] = []
            for left, right in zip(members, members[1:]):
                gap_x1 = left.bbox[2]
                gap_x2 = right.bbox[0]
                if gap_x2 - gap_x1 < scaled_px(
                    cfg, cfg.damage_gap_min_width_px, image_width, image_height
                ):
                    continue
                y1 = int(round(median([left.bbox[1], right.bbox[1]])))
                y2 = int(round(median([left.bbox[3], right.bbox[3]])))
                if y2 <= y1:
                    y1 = min(left.bbox[1], right.bbox[1])
                    y2 = max(left.bbox[3], right.bbox[3])
                gap_box = clip_box([gap_x1, y1, gap_x2, y2], image_width, image_height)
                ratio = mask_ratio(damage_mask, gap_box)
                gaps.append(
                    {
                        "left_member_id": left.id,
                        "right_member_id": right.id,
                        "gap_bbox_xyxy": gap_box,
                        "damage_mask_ratio": round(ratio, 6),
                        "interrupted_by_damage": ratio
                        >= cfg.damage_gap_mask_ratio_min,
                        "interpretation": (
                            "observed course segments separated across documented damage"
                            if ratio >= cfg.damage_gap_mask_ratio_min
                            else "observed gap; damage support below threshold"
                        ),
                    }
                )

            course_records.append(
                {
                    "id": course_id,
                    "class": class_name,
                    "member_ids": [item.id for item in members],
                    "member_count": len(members),
                    "center_y": round(median([item.center_y for item in members]), 3),
                    "median_top_y": round(median([item.bbox[1] for item in members]), 3),
                    "median_bottom_y": round(median([item.bbox[3] for item in members]), 3),
                    "x_extent": [
                        min(item.bbox[0] for item in members),
                        max(item.bbox[2] for item in members),
                    ],
                    "gaps": gaps,
                    "same_course_relation": len(members) > 1,
                    "instances_remain_distinct": True,
                    "certainty": "expert class + observed vertical alignment",
                }
            )

    return course_records, membership


# =============================================================================
# ARCH / OPENING — PILASTER RELATIONS
# =============================================================================
def support_side_score(
    feature: Element,
    pilaster: Element,
    side: str,
    image_width: int,
    image_height: int,
    cfg: Config,
) -> tuple[float, dict[str, float]]:
    x_tolerance = scaled_px(
        cfg, cfg.arch_support_x_tolerance_px, image_width, image_height
    )
    y_tolerance = scaled_px(
        cfg, cfg.arch_support_y_tolerance_px, image_width, image_height
    )
    target_x = float(feature.bbox[0] if side == "left" else feature.bbox[2])
    dx = abs(pilaster.center_x - target_x)
    dy = abs(float(pilaster.bbox[1]) - float(feature.bbox[3]))

    # Pilaster merkezinin kemerin ilgili yarısında veya hemen dışında olması beklenir.
    half_margin = max(feature.width * 0.16, x_tolerance * 0.25)
    if side == "left":
        side_ok = pilaster.center_x <= feature.center_x + half_margin
    else:
        side_ok = pilaster.center_x >= feature.center_x - half_margin
    if not side_ok:
        return 0.0, {
            "horizontal_distance_px": round(dx, 3),
            "vertical_distance_px": round(dy, 3),
            "side_gate": 0.0,
        }

    x_score = exp_score(dx, max(x_tolerance, feature.width * 0.50))

    if feature.cls in {"opening", "window", "door"}:
        vertical_overlap = interval_overlap_ratio(
            feature.bbox[1], feature.bbox[3], pilaster.bbox[1], pilaster.bbox[3]
        )
        # Üst cephedeki bir açıklığın alt cephe pilasterlarına yalnız x yakınlığıyla
        # bağlanmasını engeller. Açıklık ilişkisi için gerçek düşey çakışma gerekir.
        if vertical_overlap < 0.12:
            return 0.0, {
                "horizontal_distance_px": round(dx, 3),
                "vertical_distance_px": round(dy, 3),
                "horizontal_score": round(x_score, 6),
                "vertical_overlap": round(vertical_overlap, 6),
                "vertical_gate": 0.0,
                "side_gate": 1.0,
            }
        y_score = vertical_overlap
    else:
        vertical_overlap = interval_overlap_ratio(
            feature.bbox[1],
            feature.bbox[3] + y_tolerance,
            pilaster.bbox[1],
            pilaster.bbox[3],
        )
        if dy > y_tolerance * 1.75 and vertical_overlap < 0.08:
            return 0.0, {
                "horizontal_distance_px": round(dx, 3),
                "vertical_distance_px": round(dy, 3),
                "horizontal_score": round(x_score, 6),
                "vertical_overlap": round(vertical_overlap, 6),
                "vertical_gate": 0.0,
                "side_gate": 1.0,
            }
        y_score = exp_score(dy, y_tolerance)

    status_factor = 1.0 if pilaster.is_intact else 0.88
    score = status_factor * (
        0.58 * x_score + 0.27 * y_score + 0.15 * vertical_overlap
    )
    return float(score), {
        "horizontal_distance_px": round(dx, 3),
        "vertical_distance_px": round(dy, 3),
        "horizontal_score": round(x_score, 6),
        "vertical_score": round(y_score, 6),
        "vertical_overlap": round(vertical_overlap, 6),
        "pilaster_status_factor": status_factor,
        "side_gate": 1.0,
    }


def build_feature_pilaster_relations(
    features: list[Element],
    pilasters: list[Element],
    image_width: int,
    image_height: int,
    cfg: Config,
    relation_prefix: str,
    min_score: float,
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for feature in features:
        selected: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
        all_candidates: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        for side in ("left", "right"):
            for pilaster in pilasters:
                score, evidence = support_side_score(
                    feature,
                    pilaster,
                    side,
                    image_width,
                    image_height,
                    cfg,
                )
                candidate = {
                    "pilaster_id": pilaster.id,
                    "score": round(score, 6),
                    "evidence": evidence,
                }
                all_candidates[side].append(candidate)
            all_candidates[side].sort(key=lambda item: item["score"], reverse=True)
            if all_candidates[side] and all_candidates[side][0]["score"] >= min_score:
                selected[side] = all_candidates[side][0]

        left = selected["left"]
        right = selected["right"]
        relations.append(
            {
                "id": f"{relation_prefix}_{feature.id}",
                "feature_id": feature.id,
                "feature_class": feature.cls,
                "feature_status": feature.status,
                "left_pilaster_id": left["pilaster_id"] if left else None,
                "right_pilaster_id": right["pilaster_id"] if right else None,
                "left_score": left["score"] if left else None,
                "right_score": right["score"] if right else None,
                "relation_state": (
                    "bounded_by_two_observed_pilasters"
                    if left and right and left["pilaster_id"] != right["pilaster_id"]
                    else "one_observed_flank_candidate"
                    if left or right
                    else "no_reliable_observed_flank"
                ),
                "certainty": "derived_from_expert_validated_geometry",
                "not_a_completion_hypothesis": True,
                "candidate_diagnostics": {
                    "left_top3": all_candidates["left"][:3],
                    "right_top3": all_candidates["right"][:3],
                },
            }
        )
    return relations


def build_arch_pilaster_relations(
    elements: list[Element],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    arches = [item for item in elements if item.cls == "arch"]
    pilasters = [item for item in elements if item.cls in {"pilaster", "column"}]
    return build_feature_pilaster_relations(
        arches,
        pilasters,
        image_width,
        image_height,
        cfg,
        relation_prefix="arch_support_relation",
        min_score=cfg.arch_support_min_score,
    )


def build_opening_pilaster_relations(
    elements: list[Element],
    image_width: int,
    image_height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    openings = [
        item for item in elements if item.cls in {"opening", "window", "door"}
    ]
    pilasters = [item for item in elements if item.cls in {"pilaster", "column"}]
    return build_feature_pilaster_relations(
        openings,
        pilasters,
        image_width,
        image_height,
        cfg,
        relation_prefix="opening_flank_relation",
        min_score=cfg.opening_flank_min_score,
    )


# =============================================================================
# GRAPH
# =============================================================================
def build_overlap_relations(
    elements: list[Element], cfg: Config
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for first_index, first in enumerate(elements):
        for second in elements[first_index + 1 :]:
            iou = box_iou(first.bbox, second.bbox)
            if iou < cfg.bbox_overlap_iou_min:
                continue
            relations.append(
                {
                    "first_id": first.id,
                    "second_id": second.id,
                    "iou": round(iou, 6),
                    "relation": "bbox_overlap_observed",
                    "distinct_instances": True,
                    "duplicate_inference_forbidden": True,
                }
            )
    return relations


def build_same_class_adjacency(elements: list[Element]) -> list[dict[str, Any]]:
    by_class: dict[str, list[Element]] = defaultdict(list)
    for element in elements:
        by_class[element.cls].append(element)
    relations: list[dict[str, Any]] = []
    for class_name, items in sorted(by_class.items()):
        ordered = sorted(items, key=lambda item: (item.center_x, item.center_y))
        for left, right in zip(ordered, ordered[1:]):
            relations.append(
                {
                    "left_id": left.id,
                    "right_id": right.id,
                    "class": class_name,
                    "relation": "left_of_same_class",
                    "center_distance_px": round(right.center_x - left.center_x, 3),
                    "certainty": "observed_geometry",
                }
            )
    return relations


def build_architecture_graph(
    elements: list[Element],
    families: list[dict[str, Any]],
    family_membership: dict[str, str],
    alignments: dict[str, Any],
    courses: list[dict[str, Any]],
    course_membership: dict[str, str],
    arch_relations: list[dict[str, Any]],
    opening_relations: list[dict[str, Any]],
    overlap_relations: list[dict[str, Any]],
    adjacency_relations: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [item.as_node() for item in elements]
    edges: list[dict[str, Any]] = []

    for family in families:
        nodes.append(
            {
                "id": family["id"],
                "node_type": "element_family",
                "class": family["class"],
                "member_count": family["member_count"],
                "representative_id": family.get("representative_id"),
            }
        )
        for member_id in family["member_ids"]:
            edges.append(
                {
                    "source": member_id,
                    "target": family["id"],
                    "relation": "member_of_family",
                    "certainty": "derived_geometry",
                    "merge_allowed": False,
                }
            )

    for course in courses:
        nodes.append(
            {
                "id": course["id"],
                "node_type": "horizontal_course",
                "class": course["class"],
                "center_y": course["center_y"],
                "member_count": course["member_count"],
            }
        )
        for member_id in course["member_ids"]:
            edges.append(
                {
                    "source": member_id,
                    "target": course["id"],
                    "relation": "member_of_horizontal_course",
                    "certainty": "expert_class_and_observed_alignment",
                }
            )
        for gap_index, gap in enumerate(course.get("gaps", []), start=1):
            if not gap.get("interrupted_by_damage"):
                continue
            gap_node_id = f"{course['id']}_damage_gap_{gap_index:02d}"
            nodes.append(
                {
                    "id": gap_node_id,
                    "node_type": "observed_damage_gap",
                    "bbox_xyxy": gap["gap_bbox_xyxy"],
                    "damage_mask_ratio": gap["damage_mask_ratio"],
                }
            )
            edges.append(
                {
                    "source": course["id"],
                    "target": gap_node_id,
                    "relation": "interrupted_by_documented_damage",
                    "certainty": "mask_supported_geometry",
                    "no_completion_generated": True,
                }
            )

    for axis in alignments["vertical_axes"] + alignments["horizontal_axes"]:
        nodes.append(
            {
                "id": axis["id"],
                "node_type": "alignment_axis",
                "axis_type": axis["axis_type"],
                "x": axis.get("x"),
                "y": axis.get("y"),
                "member_count": axis["member_count"],
            }
        )
        for member_id in axis["member_ids"]:
            edges.append(
                {
                    "source": member_id,
                    "target": axis["id"],
                    "relation": "aligned_to_axis",
                    "certainty": axis["certainty"],
                }
            )

    for relation in arch_relations:
        arch_id = relation["feature_id"]
        if relation.get("left_pilaster_id"):
            edges.append(
                {
                    "source": relation["left_pilaster_id"],
                    "target": arch_id,
                    "relation": "left_support_candidate_for_arch",
                    "score": relation["left_score"],
                    "certainty": relation["certainty"],
                }
            )
        if relation.get("right_pilaster_id"):
            edges.append(
                {
                    "source": relation["right_pilaster_id"],
                    "target": arch_id,
                    "relation": "right_support_candidate_for_arch",
                    "score": relation["right_score"],
                    "certainty": relation["certainty"],
                }
            )

    for relation in opening_relations:
        opening_id = relation["feature_id"]
        if relation.get("left_pilaster_id"):
            edges.append(
                {
                    "source": relation["left_pilaster_id"],
                    "target": opening_id,
                    "relation": "left_flank_candidate_for_opening",
                    "score": relation["left_score"],
                    "certainty": relation["certainty"],
                }
            )
        if relation.get("right_pilaster_id"):
            edges.append(
                {
                    "source": relation["right_pilaster_id"],
                    "target": opening_id,
                    "relation": "right_flank_candidate_for_opening",
                    "score": relation["right_score"],
                    "certainty": relation["certainty"],
                }
            )

    for overlap in overlap_relations:
        edges.append(
            {
                "source": overlap["first_id"],
                "target": overlap["second_id"],
                "relation": "bbox_overlap_observed",
                "iou": overlap["iou"],
                "distinct_instances": True,
                "duplicate_inference_forbidden": True,
            }
        )

    for adjacency in adjacency_relations:
        edges.append(
            {
                "source": adjacency["left_id"],
                "target": adjacency["right_id"],
                "relation": adjacency["relation"],
                "class": adjacency["class"],
                "center_distance_px": adjacency["center_distance_px"],
                "certainty": adjacency["certainty"],
            }
        )

    return {
        "policy": {
            "expert_ids_are_authoritative": True,
            "instances_never_auto_merged": True,
            "bbox_overlap_is_not_duplicate_evidence": True,
            "observed_relations_only": True,
            "no_missing_element_hypothesis": True,
            "stable_diffusion_used": False,
        },
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "element_node_count": len(elements),
            "family_node_count": len(families),
            "course_node_count": len(courses),
            "alignment_node_count": len(
                alignments["vertical_axes"] + alignments["horizontal_axes"]
            ),
        },
        "nodes": nodes,
        "edges": edges,
        "lookup": {
            "element_to_family": family_membership,
            "element_to_horizontal_course": course_membership,
        },
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
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


def make_relationship_overlay(
    image: Image.Image,
    elements: list[Element],
    arch_relations: list[dict[str, Any]],
    opening_relations: list[dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(
        image,
        "STAGE 3C-A RELATIONSHIPS — OBSERVED ELEMENTS / NO COMPLETION",
        cfg,
    )
    shift_y = cfg.output_header_height
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(cfg.output_label_font_size, bold=True)
    element_lookup = {item.id: item for item in elements}

    for element in elements:
        color = CLASS_COLORS.get(element.cls, CLASS_COLORS["other"])
        x1, y1, x2, y2 = element.bbox
        y1 += shift_y
        y2 += shift_y
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=cfg.output_box_thickness,
        )
        if cfg.draw_element_numbers:
            draw_label(draw, (x1 + 4, y1 + 3), f"#{element.display_number}", color, font)

    relation_color = (56, 189, 248, 235)
    opening_color = (192, 132, 252, 235)
    for relation_index, relation in enumerate(arch_relations, start=1):
        arch = element_lookup[relation["feature_id"]]
        arch_point = (int(round(arch.center_x)), int(round(arch.bbox[3] + shift_y)))
        for side_key, pilaster_key in (
            ("L", "left_pilaster_id"),
            ("R", "right_pilaster_id"),
        ):
            pilaster_id = relation.get(pilaster_key)
            if not pilaster_id or pilaster_id not in element_lookup:
                continue
            pilaster = element_lookup[pilaster_id]
            pilaster_point = (
                int(round(pilaster.center_x)),
                int(round(pilaster.bbox[1] + shift_y)),
            )
            draw.line(
                [arch_point, pilaster_point],
                fill=relation_color,
                width=cfg.output_line_thickness,
            )
            if cfg.draw_support_relation_codes:
                mid = (
                    int((arch_point[0] + pilaster_point[0]) / 2),
                    int((arch_point[1] + pilaster_point[1]) / 2),
                )
                draw_label(draw, mid, f"A{relation_index}{side_key}", relation_color, font)

    for relation_index, relation in enumerate(opening_relations, start=1):
        opening = element_lookup[relation["feature_id"]]
        opening_point = (
            int(round(opening.center_x)),
            int(round(opening.center_y + shift_y)),
        )
        for side_key, pilaster_key in (
            ("L", "left_pilaster_id"),
            ("R", "right_pilaster_id"),
        ):
            pilaster_id = relation.get(pilaster_key)
            if not pilaster_id or pilaster_id not in element_lookup:
                continue
            pilaster = element_lookup[pilaster_id]
            pilaster_point = (
                int(round(pilaster.center_x)),
                int(round(pilaster.center_y + shift_y)),
            )
            draw.line(
                [opening_point, pilaster_point],
                fill=opening_color,
                width=max(1, cfg.output_line_thickness - 1),
            )
            if cfg.draw_support_relation_codes:
                mid = (
                    int((opening_point[0] + pilaster_point[0]) / 2),
                    int((opening_point[1] + pilaster_point[1]) / 2),
                )
                draw_label(draw, mid, f"O{relation_index}{side_key}", opening_color, font)

    return canvas.convert("RGB")


def make_family_overlay(
    image: Image.Image,
    elements: list[Element],
    families: list[dict[str, Any]],
    membership: dict[str, str],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(
        image,
        "STAGE 3C-A ELEMENT FAMILIES — DISTINCT EXPERT INSTANCES PRESERVED",
        cfg,
    )
    shift_y = cfg.output_header_height
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(cfg.output_label_font_size, bold=True)
    family_color = {
        family["id"]: FAMILY_PALETTE[index % len(FAMILY_PALETTE)]
        for index, family in enumerate(families)
    }
    family_short = {
        family["id"]: f"F{index + 1}"
        for index, family in enumerate(families)
    }
    for element in elements:
        family_id = membership.get(element.id)
        color = family_color.get(family_id, CLASS_COLORS["other"])
        x1, y1, x2, y2 = element.bbox
        y1 += shift_y
        y2 += shift_y
        dash_width = cfg.output_box_thickness
        draw.rectangle([x1, y1, x2, y2], outline=color, width=dash_width)
        label = f"#{element.display_number} {family_short.get(family_id, 'F?')}"
        draw_label(draw, (x1 + 4, y1 + 3), label, color, font)
    return canvas.convert("RGB")


def make_alignment_overlay(
    image: Image.Image,
    elements: list[Element],
    alignments: dict[str, Any],
    cfg: Config,
) -> Image.Image:
    canvas = add_header(
        image,
        "STAGE 3C-A ALIGNMENTS — OBSERVED AXES ONLY / NO INFERRED MISSING AXES",
        cfg,
    )
    shift_y = cfg.output_header_height
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(cfg.output_label_font_size, bold=True)

    for element in elements:
        color = CLASS_COLORS.get(element.cls, CLASS_COLORS["other"])
        x1, y1, x2, y2 = element.bbox
        draw.rectangle(
            [x1, y1 + shift_y, x2, y2 + shift_y],
            outline=(*color[:3], 150),
            width=max(1, cfg.output_box_thickness - 1),
        )

    vertical_color = (56, 189, 248, 220)
    horizontal_color = (244, 114, 182, 220)
    element_lookup = {item.id: item for item in elements}
    for index, axis in enumerate(alignments["vertical_axes"], start=1):
        x = int(round(axis["x"]))
        members = [element_lookup[item_id] for item_id in axis["member_ids"] if item_id in element_lookup]
        if not members:
            continue
        y1 = max(0, min(item.bbox[1] for item in members) - 35) + shift_y
        y2 = min(image.height, max(item.bbox[3] for item in members) + 35) + shift_y
        draw.line(
            [(x, y1), (x, y2)],
            fill=vertical_color,
            width=cfg.output_line_thickness,
        )
        draw_label(draw, (x + 3, y1 + 3), f"V{index}", vertical_color, font)

    for index, axis in enumerate(alignments["horizontal_axes"], start=1):
        y = int(round(axis["y"] + shift_y))
        members = [element_lookup[item_id] for item_id in axis["member_ids"] if item_id in element_lookup]
        if not members:
            continue
        x1 = max(0, min(item.bbox[0] for item in members) - 28)
        x2 = min(image.width, max(item.bbox[2] for item in members) + 28)
        draw.line(
            [(x1, y), (x2, y)],
            fill=horizontal_color,
            width=cfg.output_line_thickness,
        )
        draw_label(draw, (x1 + 3, y + 2), f"H{index}", horizontal_color, font)

    return canvas.convert("RGB")


def make_damage_interruption_overlay(
    image: Image.Image,
    damage_mask: np.ndarray,
    elements: list[Element],
    courses: list[dict[str, Any]],
    cfg: Config,
) -> Image.Image:
    mask_pil = Image.fromarray(damage_mask, mode="L")
    base = image.convert("RGBA")
    damage_layer = Image.new("RGBA", image.size, (239, 68, 68, 0))
    damage_layer.putalpha(mask_pil.point(lambda value: 62 if value > 0 else 0))
    base = Image.alpha_composite(base, damage_layer)
    canvas = add_header(
        base,
        "STAGE 3C-A HORIZONTAL COURSES / DOCUMENTED DAMAGE INTERRUPTIONS",
        cfg,
    )
    shift_y = cfg.output_header_height
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(cfg.output_label_font_size, bold=True)
    element_lookup = {item.id: item for item in elements}

    for course_index, course in enumerate(courses, start=1):
        color = CLASS_COLORS.get(course["class"], CLASS_COLORS["other"])
        y = int(round(course["center_y"] + shift_y))
        x1, x2 = course["x_extent"]
        draw.line([(x1, y), (x2, y)], fill=color, width=cfg.output_line_thickness + 1)
        draw_label(draw, (x1 + 4, y + 3), f"C{course_index}", color, font)
        for member_id in course["member_ids"]:
            element = element_lookup[member_id]
            bx1, by1, bx2, by2 = element.bbox
            draw.rectangle(
                [bx1, by1 + shift_y, bx2, by2 + shift_y],
                outline=color,
                width=cfg.output_box_thickness,
            )
        for gap_index, gap in enumerate(course.get("gaps", []), start=1):
            if not gap.get("interrupted_by_damage"):
                continue
            gx1, gy1, gx2, gy2 = gap["gap_bbox_xyxy"]
            gap_rect = [gx1, gy1 + shift_y, gx2, gy2 + shift_y]
            draw.rectangle(
                gap_rect,
                outline=(239, 68, 68, 255),
                width=cfg.output_box_thickness,
            )
            # Şeffaf dolgu yerine seyrek tarama: fotoğraf ve maskeyi örtmez.
            hatch_step = 18
            hatch_start = gx1 - (gy2 - gy1)
            for hx in range(hatch_start, gx2 + hatch_step, hatch_step):
                start_x = max(gx1, hx)
                start_y = gy1 + shift_y + max(0, gx1 - hx)
                end_x = min(gx2, hx + (gy2 - gy1))
                end_y = gy1 + shift_y + min(gy2 - gy1, gx2 - hx)
                if end_x > start_x and end_y > start_y:
                    draw.line(
                        [(start_x, start_y), (end_x, end_y)],
                        fill=(239, 68, 68, 150),
                        width=1,
                    )
            draw_label(
                draw,
                (gx1 + 4, gy1 + shift_y + 4),
                f"D{course_index}.{gap_index}",
                (239, 68, 68, 255),
                font,
            )

    return canvas.convert("RGB")


# =============================================================================
# REPORT
# =============================================================================
def build_report(
    run_id: str,
    parent_run: Path,
    elements: list[Element],
    families: list[dict[str, Any]],
    alignments: dict[str, Any],
    courses: list[dict[str, Any]],
    arch_relations: list[dict[str, Any]],
    opening_relations: list[dict[str, Any]],
    overlap_relations: list[dict[str, Any]],
    graph: dict[str, Any],
) -> str:
    counts_by_class: dict[str, int] = defaultdict(int)
    counts_by_status: dict[str, int] = defaultdict(int)
    for element in elements:
        counts_by_class[element.cls] += 1
        counts_by_status[element.status] += 1

    two_support_arches = sum(
        relation["relation_state"] == "bounded_by_two_observed_pilasters"
        for relation in arch_relations
    )
    damage_gaps = [
        gap
        for course in courses
        for gap in course.get("gaps", [])
        if gap.get("interrupted_by_damage")
    ]

    lines = [
        "STAGE 3C-A — ARCHITECTURAL RELATION AND GRAMMAR GRAPH",
        f"Run: {run_id}",
        f"Parent: {parent_run.resolve()}",
        "",
        "NON-NEGOTIABLE POLICY:",
        "- Locked expert element IDs are authoritative and remain distinct.",
        "- No element was deleted, merged, completed or newly invented.",
        "- Bbox overlap is recorded only as a spatial relation; it is not duplicate evidence.",
        "- No missing-element hypothesis was generated.",
        "- Stable Diffusion was not used.",
        "",
        "OBSERVED INVENTORY:",
        f"- Elements: {len(elements)}",
        f"- Intact: {counts_by_status['observed_intact']}",
        f"- Fragmented: {counts_by_status['observed_fragmented']}",
    ]
    for class_name in sorted(counts_by_class):
        lines.append(f"- {class_name}: {counts_by_class[class_name]}")

    lines.extend(
        [
            "",
            "DERIVED RELATIONS:",
            f"- Element families: {len(families)}",
            f"- Vertical axes: {len(alignments['vertical_axes'])}",
            f"- Horizontal alignment axes: {len(alignments['horizontal_axes'])}",
            f"- Horizontal courses: {len(courses)}",
            f"- Damage-supported course gaps: {len(damage_gaps)}",
            f"- Arch–pilaster relation records: {len(arch_relations)}",
            f"- Arches bounded by two observed pilasters: {two_support_arches}",
            f"- Opening–pilaster relation records: {len(opening_relations)}",
            f"- Observed bbox-overlap pairs: {len(overlap_relations)}",
            f"- Graph nodes: {graph['summary']['node_count']}",
            f"- Graph edges: {graph['summary']['edge_count']}",
            "",
            "INTERPRETATION LIMIT:",
            "- 'support_candidate' means observed geometric compatibility, not structural proof.",
            "- 'same_family' means family membership while preserving every expert instance ID.",
            "- 'interrupted_by_documented_damage' records a mask-supported gap; it does not fill the gap.",
            "",
            "NEXT REVIEW:",
            "- Inspect relationship, family, alignment and damage-interruption overlays.",
            "- Stage 3C-B missing-element hypotheses must not start until these relations are accepted.",
        ]
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_stage(cfg: Config) -> Path:
    parent_run = find_parent_run(cfg)
    input_path = parent_run / "input.png"
    mask_path = parent_run / "completion_mask.png"
    inventory_path = parent_run / "expert_validated_inventory.json"

    image = Image.open(input_path).convert("RGB")
    image_width, image_height = image.size
    damage_mask = load_binary_mask(mask_path, image.size)
    inventory_doc = read_json(inventory_path)
    elements = normalize_inventory(
        inventory_doc,
        image_width,
        image_height,
        cfg,
    )

    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started_at = now_iso()

    try:
        logger.info("=" * 80)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info(f"[STAGE] {STAGE_TITLE}")
        logger.info("[RULE] Expert IDs remain distinct; no auto merge / no completion")
        logger.info(f"[INPUT] elements={len(elements)}, image={image_width}x{image_height}")
        logger.info("=" * 80)

        copied_inputs = {
            "input": copy_with_hash(input_path, run_dir / "input.png"),
            "completion_mask": copy_with_hash(
                mask_path, run_dir / "completion_mask.png"
            ),
            "expert_inventory": copy_with_hash(
                inventory_path,
                run_dir / "source_expert_validated_inventory.json",
            ),
        }

        families, family_membership, family_pair_diagnostics = build_element_families(
            elements,
            image_width,
            image_height,
            cfg,
        )
        alignments = build_alignment_axes(
            elements,
            image_width,
            image_height,
            cfg,
        )
        courses, course_membership = build_horizontal_courses(
            elements,
            damage_mask,
            image_width,
            image_height,
            cfg,
        )
        arch_relations = build_arch_pilaster_relations(
            elements,
            image_width,
            image_height,
            cfg,
        )
        opening_relations = build_opening_pilaster_relations(
            elements,
            image_width,
            image_height,
            cfg,
        )
        overlap_relations = build_overlap_relations(elements, cfg)
        adjacency_relations = build_same_class_adjacency(elements)

        graph = build_architecture_graph(
            elements,
            families,
            family_membership,
            alignments,
            courses,
            course_membership,
            arch_relations,
            opening_relations,
            overlap_relations,
            adjacency_relations,
        )

        common_policy = {
            "expert_ids_are_authoritative": True,
            "instances_never_auto_merged": True,
            "bbox_overlap_is_not_duplicate_evidence": True,
            "observed_evidence_only": True,
            "no_missing_element_hypothesis": True,
            "stable_diffusion_used": False,
        }

        write_json(
            run_dir / "stage3c_element_families.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "parent_run": str(parent_run.resolve()),
                "policy": common_policy,
                "family_count": len(families),
                "families": families,
                "element_to_family": family_membership,
                "pair_diagnostics": family_pair_diagnostics,
            },
        )
        write_json(
            run_dir / "stage3c_alignment_axes.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                **alignments,
            },
        )
        write_json(
            run_dir / "stage3c_arch_pilaster_relations.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                "relation_count": len(arch_relations),
                "relations": arch_relations,
            },
        )
        write_json(
            run_dir / "stage3c_opening_pilaster_relations.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                "relation_count": len(opening_relations),
                "relations": opening_relations,
            },
        )
        write_json(
            run_dir / "stage3c_horizontal_courses.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "policy": common_policy,
                "course_count": len(courses),
                "courses": courses,
                "element_to_course": course_membership,
            },
        )
        write_json(
            run_dir / "stage3c_architecture_graph.json",
            {
                "run_id": run_id,
                "stage": STAGE_NAME,
                "parent_run": str(parent_run.resolve()),
                **graph,
                "overlap_relations": overlap_relations,
                "same_class_adjacency": adjacency_relations,
            },
        )

        make_relationship_overlay(
            image,
            elements,
            arch_relations,
            opening_relations,
            cfg,
        ).save(run_dir / "stage3c_relationship_overlay.png")
        make_family_overlay(
            image,
            elements,
            families,
            family_membership,
            cfg,
        ).save(run_dir / "stage3c_family_overlay.png")
        make_alignment_overlay(
            image,
            elements,
            alignments,
            cfg,
        ).save(run_dir / "stage3c_alignment_overlay.png")
        make_damage_interruption_overlay(
            image,
            damage_mask,
            elements,
            courses,
            cfg,
        ).save(run_dir / "stage3c_damage_interruption_overlay.png")

        report = build_report(
            run_id,
            parent_run,
            elements,
            families,
            alignments,
            courses,
            arch_relations,
            opening_relations,
            overlap_relations,
            graph,
        )
        (run_dir / "stage3c_report.txt").write_text(report, encoding="utf-8")

        source_script = Path(__file__).resolve()
        output_files = [
            "input.png",
            "completion_mask.png",
            "source_expert_validated_inventory.json",
            "stage3c_element_families.json",
            "stage3c_alignment_axes.json",
            "stage3c_arch_pilaster_relations.json",
            "stage3c_opening_pilaster_relations.json",
            "stage3c_horizontal_courses.json",
            "stage3c_architecture_graph.json",
            "stage3c_relationship_overlay.png",
            "stage3c_family_overlay.png",
            "stage3c_alignment_overlay.png",
            "stage3c_damage_interruption_overlay.png",
            "stage3c_report.txt",
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
            },
            "policy": common_policy,
            "summary": {
                "expert_elements_preserved": len(elements),
                "element_families": len(families),
                "vertical_axes": len(alignments["vertical_axes"]),
                "horizontal_axes": len(alignments["horizontal_axes"]),
                "horizontal_courses": len(courses),
                "arch_pilaster_relations": len(arch_relations),
                "opening_pilaster_relations": len(opening_relations),
                "bbox_overlap_relations": len(overlap_relations),
                "graph_nodes": graph["summary"]["node_count"],
                "graph_edges": graph["summary"]["edge_count"],
            },
            "outputs": output_files,
        }
        write_json(run_dir / "run_meta.json", metadata)

        logger.info(
            f"[COUNT] elements={len(elements)}, families={len(families)}, "
            f"axes={len(alignments['vertical_axes']) + len(alignments['horizontal_axes'])}, "
            f"courses={len(courses)}, arch_relations={len(arch_relations)}"
        )
        logger.info(
            f"[GRAPH] nodes={graph['summary']['node_count']}, "
            f"edges={graph['summary']['edge_count']}"
        )
        logger.info("[RULE] 0 instances merged / 0 completion hypotheses generated")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info("[OK] Stage 3C-A completed")
        return run_dir

    except Exception as exc:
        logger.exception("[FAIL] Stage 3C-A failed")
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


def main() -> None:
    cfg = Config()
    run_stage(cfg)


if __name__ == "__main__":
    main()