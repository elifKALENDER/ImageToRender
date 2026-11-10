#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3a_existing_evidence_skeleton.py
=================================================================
V8 / STAGE 3A — Yalnız mevcut tanınmış elemanlardan kanıt iskeleti

Bu sürümde düzeltilen ana hata:
- Nesne sınıfı önce "class" alanından okunur.

KESİN GÜVENLİK KURALLARI
------------------------
- Yeni sütun/pilaster/kemer/eksen oluşturmaz.
- İki pilaster arasına otomatik kemer eklemez.
- Eksen aralıklarından yeni modül üretmez.
- Yalnız validated_elements.json ve tentative_elements.json içindeki
  mevcut nesne kutularını görselleştirir.
- Yalnız source_stage2d_rhythm_groups.json / stage2d_rhythm_groups.json
  içindeki mevcut eksenleri yardımcı kanıt olarak okur.
- Hasar maskesi içindeki görünmeyen bölgeye çizgi uzatmaz.
- Her çalıştırmada yeni outputs/restore_YYYYMMDD_HHMMSS klasörü açar.

Çıktılar
--------
- input.png
- completion_mask.png
- source_validated_elements.json
- source_tentative_elements.json
- source_stage2d_rhythm_groups.json
- stage3a_skeleton_overlay.png
- stage3a_completion_guides.png
- stage3a_existing_evidence_skeleton.json
- stage3a_report.txt
- run.log
- run_meta.json
"""

from __future__ import annotations

import hashlib
import json
import logging
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

    # Validated/tentative nesneler zaten füzyondan geçtiği için eşikler yumuşak.
    pilaster_min_score: float = 0.45
    arch_min_score: float = 0.42
    opening_min_score: float = 0.42

    # Yalnız mevcut eksenlerden destek aranır; yeni eksen üretilmez.
    axis_min_structural_score: float = 0.55
    axis_near_box_tolerance_ratio: float = 0.045

    # Görselleştirme
    guide_thickness: int = 2
    tentative_thickness: int = 1
    damage_overlay_alpha: float = 0.11
    draw_tentative: bool = True


# =============================================================================
# GENERAL UTILITIES
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
    logger = logging.getLogger(f"stage3a_existing_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def find_parent_run(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not parent.is_dir():
            raise FileNotFoundError(f"Parent run klasörü bulunamadı: {parent.resolve()}")
        return parent

    candidates = sorted(
        (path for path in Path(cfg.output_root).glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        rhythm_exists = (
            (candidate / "stage2d_rhythm_groups.json").exists()
            or (candidate / "source_stage2d_rhythm_groups.json").exists()
        )
        if (
            (candidate / "input.png").exists()
            and (candidate / "completion_mask.png").exists()
            and (candidate / "validated_elements.json").exists()
            and (candidate / "tentative_elements.json").exists()
            and rhythm_exists
        ):
            return candidate

    raise FileNotFoundError(
        "input.png, completion_mask.png, validated_elements.json, "
        "tentative_elements.json ve rhythm_groups içeren uygun parent run bulunamadı."
    )


def copy_parent_inputs(parent: Path, run_dir: Path) -> dict[str, Any]:
    required = {
        "input.png": "input.png",
        "completion_mask.png": "completion_mask.png",
        "validated_elements.json": "source_validated_elements.json",
        "tentative_elements.json": "source_tentative_elements.json",
    }

    copied: dict[str, Any] = {"required": {}, "rhythm": {}}
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

    rhythm_source = parent / "stage2d_rhythm_groups.json"
    if not rhythm_source.exists():
        rhythm_source = parent / "source_stage2d_rhythm_groups.json"
    if not rhythm_source.exists():
        raise FileNotFoundError("Parent run içinde rhythm_groups dosyası bulunamadı")

    rhythm_target = run_dir / "source_stage2d_rhythm_groups.json"
    shutil.copy2(rhythm_source, rhythm_target)
    copied["rhythm"] = {
        "source_path": str(rhythm_source.resolve()),
        "source_sha256": sha256_file(rhythm_source),
        "run_copy": rhythm_target.name,
        "run_copy_sha256": sha256_file(rhythm_target),
    }
    return copied


# =============================================================================
# SCHEMA NORMALIZATION
# =============================================================================
def flatten_elements(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]

    if isinstance(document, dict):
        for key in (
            "elements",
            "validated_elements",
            "tentative_elements",
            "items",
            "candidates",
        ):
            value = document.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def element_label(element: dict[str, Any]) -> str:
    # DÜZELTME: gerçek JSON şemasındaki "class" ilk sırada okunur.
    for key in ("class", "label", "class_name", "category", "type"):
        value = element.get(key)
        if value:
            return str(value).strip().lower()
    return "unknown"


def element_score(element: dict[str, Any]) -> float:
    for key in ("final_score", "fused_score", "score", "confidence"):
        value = element.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def element_box(element: dict[str, Any]) -> list[int] | None:
    for key in ("bbox_xyxy", "box_xyxy", "bbox", "box"):
        value = element.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        try:
            return [int(round(float(number))) for number in value]
        except (TypeError, ValueError):
            continue
    return None


def clamp_box(box: Iterable[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def normalize_elements(
    validated_document: Any,
    tentative_document: Any,
    width: int,
    height: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    groups = [("validated", validated_document)]
    if cfg.draw_tentative:
        groups.append(("tentative", tentative_document))

    for source_group, document in groups:
        for source_index, element in enumerate(flatten_elements(document)):
            label = element_label(element)
            score = element_score(element)
            box = element_box(element)
            if box is None:
                continue
            box = clamp_box(box, width, height)

            if label in ("pilaster", "column"):
                minimum = cfg.pilaster_min_score
            elif label == "arch":
                minimum = cfg.arch_min_score
            elif label in ("window", "door"):
                minimum = cfg.opening_min_score
            else:
                continue

            if score < minimum:
                continue

            x1, y1, x2, y2 = box
            normalized.append(
                {
                    "source_group": source_group,
                    "source_index": source_index,
                    "source_id": element.get(
                        "source_id",
                        element.get("id", f"{label}_{source_index + 1:03d}"),
                    ),
                    "class": label,
                    "score": round(score, 6),
                    "bbox_xyxy": box,
                    "center": [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "raw_status": element.get("status", source_group),
                }
            )

    return normalized


def normalize_axes(rhythm_document: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    axes_out: list[dict[str, Any]] = []
    for index, axis in enumerate(rhythm_document.get("axes", [])):
        classification = str(axis.get("classification", "")).lower()
        if classification not in (
            "valid_structural_axis",
            "fragmented_structural_axis",
        ):
            continue

        metrics = axis.get("metrics", {})
        try:
            structural_score = float(metrics.get("structural_score", 0.0))
            axis_x = float(axis.get("axis_x"))
        except (TypeError, ValueError):
            continue

        if structural_score < cfg.axis_min_structural_score:
            continue

        axes_out.append(
            {
                "source_index": index,
                "source_id": axis.get("source_id", f"axis_{index + 1:02d}"),
                "axis_x": round(axis_x, 3),
                "classification": classification,
                "structural_score": round(structural_score, 6),
            }
        )

    axes_out.sort(key=lambda item: float(item["axis_x"]))
    return axes_out


# =============================================================================
# EXISTING-EVIDENCE GUIDES ONLY
# =============================================================================
def nearest_existing_axis(
    center_x: float,
    axes: list[dict[str, Any]],
    image_width: int,
    cfg: Config,
) -> tuple[dict[str, Any] | None, float | None]:
    if not axes:
        return None, None

    nearest = min(axes, key=lambda axis: abs(float(axis["axis_x"]) - center_x))
    distance = abs(float(nearest["axis_x"]) - center_x)
    tolerance = cfg.axis_near_box_tolerance_ratio * image_width
    if distance <= tolerance:
        return nearest, distance
    return None, distance


def build_existing_guides(
    elements: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    damage_mask: np.ndarray,
    image_width: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    guides: list[dict[str, Any]] = []

    for element in elements:
        x1, y1, x2, y2 = element["bbox_xyxy"]
        center_x, center_y = element["center"]
        axis, axis_distance = nearest_existing_axis(center_x, axes, image_width, cfg)

        box_damage = damage_mask[y1:y2, x1:x2] > 0
        damage_fraction = (
            float(np.count_nonzero(box_damage)) / max(1, box_damage.size)
        )

        guide: dict[str, Any] = {
            **element,
            "damage_fraction": round(damage_fraction, 6),
            "supporting_axis_id": None if axis is None else axis["source_id"],
            "supporting_axis_x": None if axis is None else axis["axis_x"],
            "axis_distance": None if axis_distance is None else round(axis_distance, 3),
            "guide_policy": "existing_bbox_only_no_damage_extension",
        }

        if element["class"] in ("pilaster", "column"):
            guide["guide_type"] = "existing_vertical_element"
            guide["guide_axis_x"] = int(round(center_x if axis is None else axis["axis_x"]))

        elif element["class"] == "arch":
            guide["guide_type"] = "existing_arch_bbox"
            guide["arch_geometry"] = {
                "center_x": int(round((x1 + x2) / 2.0)),
                "spring_y": int(round(y2)),
                "radius_x": max(1, int(round((x2 - x1) / 2.0))),
                "rise_y": max(1, int(round(y2 - y1))),
            }

        else:
            guide["guide_type"] = "existing_opening_bbox"

        guides.append(guide)

    return guides


def overlay_damage(image: np.ndarray, damage_mask: np.ndarray, alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    red = np.zeros_like(output)
    red[..., 0] = 255
    mask = (damage_mask > 0)[..., None].astype(np.float32)
    output = output * (1.0 - alpha * mask) + red * (alpha * mask)
    return np.clip(output, 0, 255).astype(np.uint8)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        output,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def masked_line_segments(
    mask: np.ndarray,
    x: int,
    y1: int,
    y2: int,
) -> list[tuple[int, int]]:
    """Yalnız maskenin siyah olduğu görünür kısımları döndürür."""
    height, width = mask.shape[:2]
    x = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if y2 < y1:
        y1, y2 = y2, y1

    visible = mask[y1 : y2 + 1, x] == 0
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for offset, is_visible in enumerate(visible):
        y = y1 + offset
        if is_visible and start is None:
            start = y
        elif not is_visible and start is not None:
            segments.append((start, y - 1))
            start = None
    if start is not None:
        segments.append((start, y2))
    return segments


def draw_masked_arch(
    image: np.ndarray,
    mask: np.ndarray,
    geometry: dict[str, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    center_x = int(geometry["center_x"])
    spring_y = int(geometry["spring_y"])
    radius_x = int(geometry["radius_x"])
    rise_y = int(geometry["rise_y"])

    angles = np.linspace(180.0, 360.0, 181)
    radians = np.deg2rad(angles)
    xs = np.round(center_x + radius_x * np.cos(radians)).astype(int)
    ys = np.round(spring_y + rise_y * np.sin(radians)).astype(int)

    height, width = mask.shape[:2]
    current: list[tuple[int, int]] = []
    for x, y in zip(xs, ys):
        inside = 0 <= x < width and 0 <= y < height
        visible = inside and mask[y, x] == 0
        if visible:
            current.append((x, y))
        elif len(current) >= 2:
            cv2.polylines(
                image,
                [np.asarray(current, dtype=np.int32)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )
            current = []
        else:
            current = []

    if len(current) >= 2:
        cv2.polylines(
            image,
            [np.asarray(current, dtype=np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def render_guides(
    original: np.ndarray,
    damage_mask: np.ndarray,
    guides: list[dict[str, Any]],
    cfg: Config,
    white_canvas: bool,
) -> np.ndarray:
    if white_canvas:
        image = np.full_like(original, 246)
        header = "EXISTING DETECTED EVIDENCE ONLY - NO COMPLETION / NO NEW OBJECT"
    else:
        image = overlay_damage(original, damage_mask, cfg.damage_overlay_alpha)
        header = "STAGE 3A EXISTING EVIDENCE SKELETON - NO HYPOTHESIS OBJECTS"

    image = add_header(image, header)

    for guide in guides:
        source_group = guide["source_group"]
        validated = source_group == "validated"
        color = (55, 220, 80) if validated else (255, 205, 55)
        thickness = cfg.guide_thickness if validated else cfg.tentative_thickness
        x1, y1, x2, y2 = guide["bbox_xyxy"]

        # Kutu zaten mevcut tanımanın sınırıdır; hasarın içine yeni şekil uzatılmaz.
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if guide["guide_type"] == "existing_vertical_element":
            axis_x = int(guide["guide_axis_x"])
            axis_x = max(x1, min(x2 - 1, axis_x))
            for segment_y1, segment_y2 in masked_line_segments(
                damage_mask,
                axis_x,
                y1,
                y2,
            ):
                cv2.line(
                    image,
                    (axis_x, segment_y1),
                    (axis_x, segment_y2),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        elif guide["guide_type"] == "existing_arch_bbox":
            draw_masked_arch(
                image,
                damage_mask,
                guide["arch_geometry"],
                color,
                thickness,
            )

        label = (
            f"{guide['source_id']} {guide['class']} "
            f"{source_group} {guide['score']:.2f}"
        )
        cv2.putText(
            image,
            label,
            (x1 + 2, max(45, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    return image


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
        "stage": "v8_stage3a_existing_evidence_skeleton",
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
            "inferred_arches_created": 0,
            "pair_based_arches_created": 0,
            "damage_area_extensions_created": 0,
            "note": (
                "Yalnız validated/tentative mevcut nesneler görselleştirildi. "
                "Hasar alanına yeni geometri uzatılmadı."
            ),
        },
        "summary": {},
        "guides": [],
        "outputs": [],
    }

    def flush_meta() -> None:
        write_json(meta_path, metadata)

    try:
        logger.info("=" * 76)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent_run.resolve()}")
        logger.info("[NEW RUN] Önceki klasörün üzerine yazılmayacak")
        logger.info("[FIX] Nesne sınıfı 'class' alanından okunacak")
        logger.info("[SAFE] İki pilaster arasına kemer eklenmeyecek")
        logger.info("[SAFE] Hasar alanına guide uzatılmayacak")
        logger.info("=" * 76)

        metadata["copied_inputs"] = copy_parent_inputs(parent_run, run_dir)
        flush_meta()

        original = read_rgb(run_dir / "input.png")
        height, width = original.shape[:2]
        damage_mask = load_binary_mask(
            run_dir / "completion_mask.png",
            (width, height),
        )

        validated_document = read_json(run_dir / "source_validated_elements.json")
        tentative_document = read_json(run_dir / "source_tentative_elements.json")
        rhythm_document = read_json(run_dir / "source_stage2d_rhythm_groups.json")

        elements = normalize_elements(
            validated_document,
            tentative_document,
            width,
            height,
            cfg,
        )
        axes = normalize_axes(rhythm_document, cfg)
        guides = build_existing_guides(
            elements,
            axes,
            damage_mask,
            width,
            cfg,
        )

        counts_by_class: dict[str, int] = {}
        counts_by_group: dict[str, int] = {}
        for guide in guides:
            counts_by_class[guide["class"]] = counts_by_class.get(guide["class"], 0) + 1
            counts_by_group[guide["source_group"]] = counts_by_group.get(guide["source_group"], 0) + 1

        metadata["guides"] = guides
        metadata["summary"] = {
            "existing_axis_count": len(axes),
            "existing_element_count": len(elements),
            "existing_guide_count": len(guides),
            "counts_by_class": counts_by_class,
            "counts_by_group": counts_by_group,
            "new_objects_created": 0,
            "inferred_arches_created": 0,
            "pair_based_arches_created": 0,
        }

        skeleton_document = {
            "run_id": run_id,
            "parent_run": metadata["parent_run"],
            "safety": metadata["safety"],
            "summary": metadata["summary"],
            "existing_axes": axes,
            "existing_guides": guides,
        }
        write_json(
            run_dir / "stage3a_existing_evidence_skeleton.json",
            skeleton_document,
        )

        overlay = render_guides(
            original,
            damage_mask,
            guides,
            cfg,
            white_canvas=False,
        )
        canvas = render_guides(
            original,
            damage_mask,
            guides,
            cfg,
            white_canvas=True,
        )

        # Eski isimler kıyaslama için korunuyor; içerik artık yalnız mevcut kanıt.
        save_rgb(run_dir / "stage3a_skeleton_overlay.png", overlay)
        save_rgb(run_dir / "stage3a_completion_guides.png", canvas)
        save_rgb(run_dir / "stage3a_existing_evidence_overlay.png", overlay)
        save_rgb(run_dir / "stage3a_existing_evidence_guides.png", canvas)

        report_lines = [
            "V8 STAGE 3A — MEVCUT KANIT İSKELETİ RAPORU",
            f"Run: {run_id}",
            f"Parent: {parent_run.resolve()}",
            "",
            "GÜVENLİK:",
            "- Yeni nesne oluşturulmadı.",
            "- Yeni eksen oluşturulmadı.",
            "- İki pilaster arasına kemer eklenmedi.",
            "- Hasar alanına yeni geometri uzatılmadı.",
            "- Yalnız mevcut validated/tentative kutular gösterildi.",
            "",
            f"Okunan mevcut eksen: {len(axes)}",
            f"Okunan mevcut eleman: {len(elements)}",
            f"Üretilen mevcut-kanıt guide: {len(guides)}",
            f"Sınıf özeti: {counts_by_class}",
            f"Durum özeti: {counts_by_group}",
            "",
            "GUIDES:",
        ]
        for guide in guides:
            report_lines.append(
                f"- {guide['source_id']}: class={guide['class']}, "
                f"group={guide['source_group']}, score={guide['score']:.3f}, "
                f"damage={guide['damage_fraction']:.3f}, "
                f"axis={guide['supporting_axis_id']}"
            )

        (run_dir / "stage3a_report.txt").write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        metadata["outputs"] = [
            {"file": "input.png", "type": "parent_input_copy"},
            {"file": "completion_mask.png", "type": "parent_mask_copy"},
            {
                "file": "source_validated_elements.json",
                "type": "parent_validated_elements_copy",
            },
            {
                "file": "source_tentative_elements.json",
                "type": "parent_tentative_elements_copy",
            },
            {
                "file": "source_stage2d_rhythm_groups.json",
                "type": "parent_existing_axes_copy",
            },
            {
                "file": "stage3a_skeleton_overlay.png",
                "type": "existing_evidence_overlay",
            },
            {
                "file": "stage3a_completion_guides.png",
                "type": "existing_evidence_guides_canvas",
            },
            {
                "file": "stage3a_existing_evidence_skeleton.json",
                "type": "existing_evidence_records",
            },
            {"file": "stage3a_report.txt", "type": "human_readable_report"},
            {"file": "run.log", "type": "log"},
            {"file": "run_meta.json", "type": "metadata"},
        ]

        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush_meta()

        logger.info("[OK] Stage 3A tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            f"[COUNT] elements={len(elements)}, guides={len(guides)}, "
            "new_objects=0, inferred_arches=0"
        )
        logger.info(
            "[FILES] stage3a_skeleton_overlay.png, "
            "stage3a_completion_guides.png, stage3a_report.txt"
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
