#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3b_v4_observed_instance_recovery.py
=====================================================================
V8 / STAGE 3B-v4 — Görünen mimari eleman örneklerini geri kazanma

Amaç
----
Stage 3B-v3 yalnız temiz aile prototiplerini seçmişti. Bu aşama o güvenilir
prototipleri arama örneği olarak kullanır ve fotoğrafta gerçekten görünen fakat
önceki nesne dedektörünün kaçırdığı kemer / pilaster örneklerini arar.

KESİN GÜVENLİK KURALLARI
------------------------
- Hasar alanına yeni kemer, pilaster, sütun veya eksen yerleştirmez.
- Eksik eleman tahmini yapmaz.
- Yalnız görüntüde mevcut piksel kanıtı bulunan adayları raporlar.
- Yeni bulunan adaylar doğrudan tarihsel gerçek kabul edilmez.
- Sonuçlar observed_intact / observed_fragmented / candidate_needs_review /
  rejected_similarity olarak ayrılır.
- Her çalıştırmada yeni outputs/restore_YYYYMMDD_HHMMSS klasörü açılır.
- Önceki run klasörlerine yazılmaz.

Yöntem
------
1. V3 temiz kemer ve pilaster prototiplerini okur.
2. OpenCV kenar/gradient şablon eşleştirmesi ile alt cephe bandını tarar.
3. DINOv2 mevcutsa adayları prototiplere görsel benzerlikle doğrular.
4. Stage 2D eksenleri yalnız aday desteği olarak kullanılır; tek başına nesne
   oluşturamaz.
5. Hasar oranı, görünür kenar, şablon skoru, DINO benzerliği ve eksen desteği
   birlikte değerlendirilir.

Ana çıktılar
------------
stage3b_v4_all_observed_instances.png
stage3b_v4_arch_instances.png
stage3b_v4_pilaster_instances.png
stage3b_v4_coverage_map.png
stage3b_v4_observed_instances.json
stage3b_v4_review_candidates.json
stage3b_v4_rejected_candidates.json
stage3b_v4_report.txt
run.log
run_meta.json
"""

from __future__ import annotations

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
from typing import Any, Iterable, Sequence

# Daha önce kullanıcı ortamında sorun çıkaran hızlı HF indirme seçeneği kapalıdır.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    # Zorunlu temel girdiler. Pilaster ailesi için aşağıda alternatif adlar var.
    required_parent_files: tuple[str, ...] = (
        "input.png",
        "completion_mask.png",
        "stage3b_v3_arch_families.json",
        "source_stage2d_rhythm_groups.json",
    )
    pilaster_family_candidates: tuple[str, ...] = (
        "stage3b_v3_pilaster_families.json",
        "source_stage3b_v2_pilaster_families.json",
        "stage3b_v2_pilaster_families.json",
    )

    # DINOv2 doğrulaması. Model daha önce Stage 2C'de indirildiyse cache'ten açılır.
    use_dinov2: bool = True
    dinov2_model_id: str = "facebook/dinov2-small"
    dinov2_local_files_only: bool = True
    dinov2_batch_size: int = 8
    dinov2_crop_padding_ratio: float = 0.10
    dinov2_arch_min_similarity: float = 0.68
    dinov2_pilaster_min_similarity: float = 0.66
    dinov2_review_min_similarity: float = 0.58

    # Arama bandı. Prototip kotundan otomatik türetilir; bunlar sınır emniyetidir.
    lower_facade_y_min_ratio: float = 0.33
    lower_facade_y_max_ratio: float = 0.73
    facade_x_margin_ratio: float = 0.08

    # Kemer şablon taraması.
    arch_template_top_fraction: float = 0.58
    arch_scale_min: float = 0.68
    arch_scale_max: float = 1.48
    arch_scale_step: float = 0.08
    arch_raw_template_threshold: float = 0.18
    arch_strong_template_threshold: float = 0.31
    arch_accept_fusion_threshold: float = 0.61
    arch_review_fusion_threshold: float = 0.49
    arch_springline_tolerance_ratio: float = 0.050
    arch_width_min_family_ratio: float = 0.58
    arch_width_max_family_ratio: float = 1.55
    arch_height_min_family_ratio: float = 0.58
    arch_height_max_family_ratio: float = 1.55
    arch_axis_edge_tolerance_ratio: float = 0.050
    arch_nms_iou: float = 0.30
    arch_max_raw_candidates: int = 120

    # Pilaster şablon taraması.
    pilaster_scale_w_min: float = 0.62
    pilaster_scale_w_max: float = 1.42
    pilaster_scale_h_min: float = 0.72
    pilaster_scale_h_max: float = 1.32
    pilaster_scale_step: float = 0.10
    pilaster_raw_template_threshold: float = 0.23
    pilaster_strong_template_threshold: float = 0.43
    pilaster_accept_fusion_threshold: float = 0.64
    pilaster_review_fusion_threshold: float = 0.51
    pilaster_center_y_tolerance_ratio: float = 0.105
    pilaster_axis_center_tolerance_ratio: float = 0.060
    pilaster_vertical_activity_min: float = 0.22
    pilaster_nms_iou: float = 0.33
    pilaster_max_raw_candidates: int = 160

    # Hasar ve görünürlük.
    observed_intact_damage_max: float = 0.14
    observed_fragmented_damage_max: float = 0.68
    reject_damage_above: float = 0.82
    minimum_visible_fraction: float = 0.18
    known_box_duplicate_iou: float = 0.55
    recovered_duplicate_iou: float = 0.40

    # Aday sayısı ve çizim.
    max_review_per_class: int = 10
    max_rejected_per_class: int = 50
    box_thickness: int = 2
    text_scale: float = 0.40


COLORS: dict[str, tuple[int, int, int]] = {
    "known_prototype": (40, 210, 70),
    "observed_intact": (35, 225, 220),
    "observed_fragmented": (255, 170, 35),
    "candidate_needs_review": (240, 210, 45),
    "rejected_similarity": (145, 145, 145),
    "damage": (255, 55, 55),
    "axis": (110, 180, 255),
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
    logger = logging.getLogger(f"stage3b_v4_{run_id}")
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


def find_parent_run(cfg: Config) -> tuple[Path, str]:
    def valid_parent(path: Path) -> tuple[bool, str]:
        if not path.is_dir():
            return False, ""
        if not all((path / name).exists() for name in cfg.required_parent_files):
            return False, ""
        pilaster_name = next(
            (name for name in cfg.pilaster_family_candidates if (path / name).exists()),
            "",
        )
        return bool(pilaster_name), pilaster_name

    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        ok, pilaster_name = valid_parent(parent)
        if not ok:
            raise FileNotFoundError(
                "Parent run gerekli V3 girdilerini içermiyor: "
                f"{parent.resolve()}"
            )
        return parent, pilaster_name

    candidates = sorted(
        (path for path in Path(cfg.output_root).glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        ok, pilaster_name = valid_parent(candidate)
        if ok:
            return candidate, pilaster_name

    raise FileNotFoundError(
        "Stage 3B-v3 girdilerini içeren uygun restore_* klasörü bulunamadı: "
        f"{Path(cfg.output_root).resolve()}"
    )


def copy_inputs(
    parent: Path,
    pilaster_family_name: str,
    run_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    mapping = {
        "input.png": "input.png",
        "completion_mask.png": "completion_mask.png",
        "stage3b_v3_arch_families.json": "source_stage3b_v3_arch_families.json",
        pilaster_family_name: "source_stage3b_v3_pilaster_families.json",
        "source_stage2d_rhythm_groups.json": "source_stage2d_rhythm_groups.json",
    }
    optional = {
        "stage3b_v3_openings.json": "source_stage3b_v3_openings.json",
        "source_validated_elements.json": "source_validated_elements.json",
        "source_tentative_elements.json": "source_tentative_elements.json",
        "stage3b_v3_clean_families.png": "source_stage3b_v3_clean_families.png",
    }

    copied: dict[str, Any] = {}
    for source_name, target_name in mapping.items():
        source = parent / source_name
        if not source.exists():
            raise FileNotFoundError(f"Parent girdi eksik: {source.resolve()}")
        target = run_dir / target_name
        shutil.copy2(source, target)
        copied[target_name] = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "copy_sha256": sha256_file(target),
        }

    for source_name, target_name in optional.items():
        source = parent / source_name
        if not source.exists():
            continue
        target = run_dir / target_name
        shutil.copy2(source, target)
        copied[target_name] = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "copy_sha256": sha256_file(target),
        }
    return copied


# =============================================================================
# IMAGE / GEOMETRY HELPERS
# =============================================================================
def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L").resize(
        (width, height),
        Image.Resampling.NEAREST,
    )
    return np.where(np.asarray(image) >= 128, 255, 0).astype(np.uint8)


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


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = box_width(first) * box_height(first)
    second_area = box_width(second) * box_height(second)
    return intersection / max(1e-9, first_area + second_area - intersection)


def horizontal_overlap_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = max(0.0, min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])))
    return overlap / max(1.0, min(box_width(first), box_width(second)))


def mask_fraction(mask: np.ndarray, box: Sequence[int]) -> float:
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0
    return float(np.count_nonzero(crop > 0) / crop.size)


def crop_with_padding(
    image: np.ndarray,
    box: Sequence[int],
    padding_ratio: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    padding_x = int(round((x2 - x1) * padding_ratio))
    padding_y = int(round((y2 - y1) * padding_ratio))
    padded = clip_box(
        [x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y],
        width,
        height,
    )
    px1, py1, px2, py2 = padded
    return image[py1:py2, px1:px2].copy()


def robust_normalize(image: np.ndarray, low: float = 5.0, high: float = 96.0) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo + 1e-9:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def make_gradient_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    normalized = robust_normalize(magnitude, 20, 97)
    return cv2.GaussianBlur(normalized, (3, 3), 0)


def make_edge_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    lower = int(max(20, 0.60 * median))
    upper = int(min(230, 1.35 * median))
    edge = cv2.Canny(gray, lower, upper)
    return cv2.dilate(edge, np.ones((3, 3), dtype=np.uint8), iterations=1)


def local_maxima(response: np.ndarray, threshold: float, kernel_size: int = 9) -> list[tuple[int, int, float]]:
    if response.size == 0:
        return []
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    dilated = cv2.dilate(response, kernel)
    rows, cols = np.where((response >= threshold) & (response >= dilated - 1e-8))
    return [(int(col), int(row), float(response[row, col])) for row, col in zip(rows, cols)]


def nms_candidates(
    candidates: list[dict[str, Any]],
    iou_threshold: float,
    max_count: int,
) -> list[dict[str, Any]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: float(item.get("proposal_score", item.get("template_score", 0.0))),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for candidate in sorted_candidates:
        if any(box_iou(candidate["bbox_xyxy"], item["bbox_xyxy"]) >= iou_threshold for item in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_count:
            break
    return kept


def remove_known_duplicates(
    candidates: list[dict[str, Any]],
    known_boxes: list[list[int]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    novel: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for candidate in candidates:
        maximum_iou = max((box_iou(candidate["bbox_xyxy"], box) for box in known_boxes), default=0.0)
        candidate_center = box_center(candidate["bbox_xyxy"])
        center_duplicate = False
        for known_box in known_boxes:
            known_center = box_center(known_box)
            x_tolerance = max(7.0, min(box_width(candidate["bbox_xyxy"]), box_width(known_box)) * 0.48)
            y_tolerance = max(9.0, min(box_height(candidate["bbox_xyxy"]), box_height(known_box)) * 0.46)
            if (
                abs(candidate_center[0] - known_center[0]) <= x_tolerance
                and abs(candidate_center[1] - known_center[1]) <= y_tolerance
            ):
                center_duplicate = True
                break
        candidate["maximum_known_iou"] = round(maximum_iou, 6)
        candidate["known_center_duplicate"] = center_duplicate
        if maximum_iou >= iou_threshold or center_duplicate:
            duplicate = dict(candidate)
            duplicate["rejection_reason"] = (
                "overlaps_known_family_member"
                if maximum_iou >= iou_threshold
                else "same_center_as_known_family_member"
            )
            duplicates.append(duplicate)
        else:
            novel.append(candidate)
    return novel, duplicates


# =============================================================================
# LOAD FAMILY PROTOTYPES / AXES
# =============================================================================
def normalize_member(member: dict[str, Any], class_name: str) -> dict[str, Any]:
    raw_box = member.get("bbox_xyxy", member.get("box"))
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError(f"{class_name} aile üyesinde kutu bulunamadı: {member}")
    return {
        "id": str(member.get("id", member.get("source_id", "unknown"))),
        "class": class_name,
        "bbox_xyxy": [int(round(float(value))) for value in raw_box],
        "score": float(member.get("score", member.get("final_score", 0.0))),
        "damage_fraction": float(member.get("damage_fraction", 0.0)),
        "source": "existing_family_member",
        "status": "known_prototype",
    }


def load_arch_prototypes(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    document = read_json(path)
    families = document.get("families", [])
    if not families:
        raise RuntimeError("V3 kemer ailesi boş")
    primary = families[0]
    prototypes = [normalize_member(item, "arch") for item in primary.get("members_intact", [])]
    if not prototypes:
        raise RuntimeError("Temiz kemer prototip üyesi bulunamadı")

    known_fragments: list[dict[str, Any]] = []
    for item in primary.get("members_fragmented_assigned", []):
        record = normalize_member(item, "arch")
        record["status"] = "observed_fragmented"
        record["source"] = "stage3b_v3_known_fragment"
        known_fragments.append(record)
    for item in document.get("central_fragmented_arches", []):
        record = normalize_member(item, "arch")
        record["status"] = "observed_fragmented"
        record["source"] = "stage3b_v3_central_fragment"
        known_fragments.append(record)
    return prototypes, known_fragments, document


def load_pilaster_prototypes(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    document = read_json(path)
    families = document.get("families", [])
    if not families:
        raise RuntimeError("Pilaster ailesi boş")
    primary = families[0]
    prototypes = [normalize_member(item, "pilaster") for item in primary.get("members_intact", [])]
    if not prototypes:
        raise RuntimeError("Temiz pilaster prototip üyesi bulunamadı")

    known_fragments: list[dict[str, Any]] = []
    for key in ("members_fragmented", "members_fragmented_assigned"):
        for item in primary.get(key, []):
            record = normalize_member(item, "pilaster")
            record["status"] = "observed_fragmented"
            record["source"] = "stage3b_v3_known_fragment"
            known_fragments.append(record)
    return prototypes, known_fragments, document


def load_axes(path: Path) -> list[dict[str, Any]]:
    document = read_json(path)
    axes: list[dict[str, Any]] = []
    for index, record in enumerate(document.get("axes", [])):
        try:
            x = float(record.get("axis_x"))
        except (TypeError, ValueError):
            continue
        classification = str(record.get("classification", "weak_axis"))
        metrics = record.get("metrics", {})
        axes.append(
            {
                "id": str(record.get("source_id", f"axis_{index + 1:02d}")),
                "x": x,
                "classification": classification,
                "structural_score": float(metrics.get("structural_score", 0.0)),
            }
        )
    return axes


def family_medians(members: list[dict[str, Any]]) -> dict[str, float]:
    widths = [box_width(item["bbox_xyxy"]) for item in members]
    heights = [box_height(item["bbox_xyxy"]) for item in members]
    centers_y = [box_center(item["bbox_xyxy"])[1] for item in members]
    tops = [float(item["bbox_xyxy"][1]) for item in members]
    bottoms = [float(item["bbox_xyxy"][3]) for item in members]
    return {
        "width": float(np.median(widths)),
        "height": float(np.median(heights)),
        "center_y": float(np.median(centers_y)),
        "top_y": float(np.median(tops)),
        "bottom_y": float(np.median(bottoms)),
    }


# =============================================================================
# DINOv2 OPTIONAL VERIFIER
# =============================================================================
class Dinov2Verifier:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.available = False
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.torch = None

        if not cfg.use_dinov2:
            logger.info("[DINOv2] Yapılandırmada kapalı; OpenCV-only devam")
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"[DINOv2] Yükleniyor: {cfg.dinov2_model_id} | "
                f"device={self.device} | local_only={cfg.dinov2_local_files_only}"
            )
            self.processor = AutoImageProcessor.from_pretrained(
                cfg.dinov2_model_id,
                local_files_only=cfg.dinov2_local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                cfg.dinov2_model_id,
                local_files_only=cfg.dinov2_local_files_only,
            ).to(self.device)
            self.model.eval()
            self.available = True
        except Exception as exc:
            logger.warning(
                "[DINOv2] Kullanılamadı; OpenCV kanıtlarıyla devam edilecek: "
                f"{exc}"
            )
            self.available = False

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray | None:
        if not self.available or not crops:
            return None
        assert self.processor is not None
        assert self.model is not None
        assert self.torch is not None
        torch = self.torch
        embeddings: list[np.ndarray] = []

        for start in range(0, len(crops), self.cfg.dinov2_batch_size):
            batch = crops[start:start + self.cfg.dinov2_batch_size]
            pil_images = [Image.fromarray(crop.astype(np.uint8), mode="RGB") for crop in batch]
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                vector = outputs.last_hidden_state[:, 0, :]
                vector = torch.nn.functional.normalize(vector, dim=-1)
            embeddings.append(vector.detach().float().cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def release(self) -> None:
        if not self.available:
            return
        try:
            del self.model
            self.model = None
            if self.torch is not None and self.device == "cuda":
                self.torch.cuda.empty_cache()
        except Exception:
            pass


def add_dino_similarity(
    candidates: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    image: np.ndarray,
    verifier: Dinov2Verifier,
    cfg: Config,
) -> dict[str, Any]:
    if not verifier.available:
        for candidate in candidates:
            candidate["dinov2_similarity"] = None
        return {"available": False, "prototype_self_similarity": None}

    prototype_crops = [
        crop_with_padding(image, item["bbox_xyxy"], cfg.dinov2_crop_padding_ratio)
        for item in prototypes
    ]
    candidate_crops = [
        crop_with_padding(image, item["bbox_xyxy"], cfg.dinov2_crop_padding_ratio)
        for item in candidates
    ]
    prototype_embeddings = verifier.embed_crops(prototype_crops)
    candidate_embeddings = verifier.embed_crops(candidate_crops)
    if prototype_embeddings is None or candidate_embeddings is None:
        for candidate in candidates:
            candidate["dinov2_similarity"] = None
        return {"available": False, "prototype_self_similarity": None}

    centroid = np.mean(prototype_embeddings, axis=0)
    centroid /= max(1e-9, float(np.linalg.norm(centroid)))
    self_similarities = prototype_embeddings @ centroid
    candidate_similarities = candidate_embeddings @ centroid
    for candidate, similarity in zip(candidates, candidate_similarities):
        candidate["dinov2_similarity"] = round(float(similarity), 6)

    return {
        "available": True,
        "prototype_self_similarity": {
            "min": round(float(np.min(self_similarities)), 6),
            "mean": round(float(np.mean(self_similarities)), 6),
            "max": round(float(np.max(self_similarities)), 6),
        },
    }


# =============================================================================
# TEMPLATE SEARCH
# =============================================================================
def match_template_multiscale(
    search_map: np.ndarray,
    template_map: np.ndarray,
    search_box: list[int],
    scales_w: Iterable[float],
    scales_h: Iterable[float],
    threshold: float,
    class_name: str,
    prototype_id: str,
    top_fraction: float | None = None,
) -> list[dict[str, Any]]:
    sx1, sy1, sx2, sy2 = search_box
    search_crop = search_map[sy1:sy2, sx1:sx2]
    template = template_map
    if top_fraction is not None:
        top_height = max(8, int(round(template.shape[0] * top_fraction)))
        template = template[:top_height]

    proposals: list[dict[str, Any]] = []
    for scale_w in scales_w:
        for scale_h in scales_h:
            target_width = max(8, int(round(template.shape[1] * scale_w)))
            target_height = max(8, int(round(template.shape[0] * scale_h)))
            if target_width >= search_crop.shape[1] or target_height >= search_crop.shape[0]:
                continue
            resized = cv2.resize(
                template,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA if scale_w < 1.0 or scale_h < 1.0 else cv2.INTER_LINEAR,
            )
            if float(np.std(resized)) < 2.0:
                continue
            response = cv2.matchTemplate(search_crop, resized, cv2.TM_CCOEFF_NORMED)
            for local_x, local_y, score in local_maxima(response, threshold, 9):
                x1 = sx1 + local_x
                y1 = sy1 + local_y
                if top_fraction is None:
                    full_width = target_width
                    full_height = target_height
                else:
                    full_width = target_width
                    full_height = max(target_height, int(round(target_height / top_fraction)))
                proposals.append(
                    {
                        "class": class_name,
                        "bbox_xyxy": [x1, y1, x1 + full_width, y1 + full_height],
                        "template_score": round(score, 6),
                        "proposal_score": round(score, 6),
                        "prototype_id": prototype_id,
                        "template_scale_w": round(float(scale_w), 4),
                        "template_scale_h": round(float(scale_h), 4),
                        "source": "prototype_template_search",
                    }
                )
    return proposals


def arch_axis_support(candidate: dict[str, Any], axes: list[dict[str, Any]], width: int, cfg: Config) -> dict[str, Any]:
    box = candidate["bbox_xyxy"]
    left = float(box[0])
    right = float(box[2])
    tolerance = width * cfg.arch_axis_edge_tolerance_ratio

    def nearest(edge_x: float) -> tuple[dict[str, Any] | None, float]:
        if not axes:
            return None, float("inf")
        record = min(axes, key=lambda axis: abs(float(axis["x"]) - edge_x))
        return record, abs(float(record["x"]) - edge_x)

    left_axis, left_distance = nearest(left)
    right_axis, right_distance = nearest(right)
    left_supported = left_axis is not None and left_distance <= tolerance
    right_supported = right_axis is not None and right_distance <= tolerance

    strength = 0.0
    supporting_ids: list[str] = []
    for supported, axis in ((left_supported, left_axis), (right_supported, right_axis)):
        if not supported or axis is None:
            continue
        supporting_ids.append(str(axis["id"]))
        classification_bonus = 1.0 if axis["classification"] in {
            "valid_structural_axis",
            "fragmented_structural_axis",
        } else 0.55
        strength += 0.5 * classification_bonus

    return {
        "axis_support_score": round(min(1.0, strength), 6),
        "supporting_axis_ids": supporting_ids,
        "left_axis_distance": None if not math.isfinite(left_distance) else round(left_distance, 3),
        "right_axis_distance": None if not math.isfinite(right_distance) else round(right_distance, 3),
    }


def pilaster_axis_support(candidate: dict[str, Any], axes: list[dict[str, Any]], width: int, cfg: Config) -> dict[str, Any]:
    center_x = box_center(candidate["bbox_xyxy"])[0]
    if not axes:
        return {
            "axis_support_score": 0.0,
            "supporting_axis_ids": [],
            "axis_distance": None,
        }
    nearest = min(axes, key=lambda axis: abs(float(axis["x"]) - center_x))
    distance = abs(float(nearest["x"]) - center_x)
    tolerance = width * cfg.pilaster_axis_center_tolerance_ratio
    if distance > tolerance:
        score = 0.0
        ids: list[str] = []
    else:
        base = max(0.0, 1.0 - distance / max(1e-6, tolerance))
        classification_weight = 1.0 if nearest["classification"] in {
            "valid_structural_axis",
            "fragmented_structural_axis",
        } else 0.60
        score = base * classification_weight
        ids = [str(nearest["id"])]
    return {
        "axis_support_score": round(float(score), 6),
        "supporting_axis_ids": ids,
        "axis_distance": round(float(distance), 3),
    }


def candidate_edge_metrics(
    candidate: dict[str, Any],
    rgb: np.ndarray,
    gradient: np.ndarray,
    edge: np.ndarray,
    damage_mask: np.ndarray,
) -> dict[str, Any]:
    x1, y1, x2, y2 = candidate["bbox_xyxy"]
    gray = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    gradient_crop = gradient[y1:y2, x1:x2]
    edge_crop = edge[y1:y2, x1:x2]
    damage_crop = damage_mask[y1:y2, x1:x2]
    if gray.size == 0:
        return {
            "damage_fraction": 1.0,
            "visible_fraction": 0.0,
            "edge_density": 0.0,
            "gradient_activity": 0.0,
            "vertical_activity": 0.0,
            "dark_fraction": 0.0,
            "upper_edge_density": 0.0,
        }

    visible = damage_crop == 0
    visible_count = int(np.count_nonzero(visible))
    total_count = max(1, int(damage_crop.size))
    damage_fraction = float(np.count_nonzero(damage_crop > 0) / total_count)
    visible_fraction = 1.0 - damage_fraction

    if visible_count:
        edge_density = float(np.count_nonzero((edge_crop > 0) & visible) / visible_count)
        gradient_activity = float(np.mean(gradient_crop[visible]) / 255.0)
        dark_threshold = max(40, int(np.percentile(gray[visible], 28)))
        dark_fraction = float(np.count_nonzero((gray <= dark_threshold) & visible) / visible_count)
    else:
        edge_density = gradient_activity = dark_fraction = 0.0

    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_x = robust_normalize(sobel_x, 20, 96)
    row_activity = np.mean(sobel_x / 255.0, axis=1)
    vertical_activity = float(np.mean(row_activity >= 0.22))

    upper_height = max(1, int(round(edge_crop.shape[0] * 0.58)))
    upper_visible = visible[:upper_height]
    upper_visible_count = int(np.count_nonzero(upper_visible))
    upper_edge_density = (
        float(np.count_nonzero((edge_crop[:upper_height] > 0) & upper_visible) / upper_visible_count)
        if upper_visible_count
        else 0.0
    )

    return {
        "damage_fraction": round(damage_fraction, 6),
        "visible_fraction": round(visible_fraction, 6),
        "edge_density": round(edge_density, 6),
        "gradient_activity": round(gradient_activity, 6),
        "vertical_activity": round(vertical_activity, 6),
        "dark_fraction": round(dark_fraction, 6),
        "upper_edge_density": round(upper_edge_density, 6),
    }


def generate_arch_candidates(
    rgb: np.ndarray,
    edge_map: np.ndarray,
    prototypes: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height, width = rgb.shape[:2]
    medians = family_medians(prototypes)
    search_box = [
        int(width * cfg.facade_x_margin_ratio),
        max(int(height * cfg.lower_facade_y_min_ratio), int(medians["top_y"] - height * 0.09)),
        int(width * (1.0 - cfg.facade_x_margin_ratio)),
        min(int(height * cfg.lower_facade_y_max_ratio), int(medians["bottom_y"] + height * 0.12)),
    ]
    search_box = clip_box(search_box, width, height)

    scales = np.arange(
        cfg.arch_scale_min,
        cfg.arch_scale_max + cfg.arch_scale_step / 2.0,
        cfg.arch_scale_step,
    )
    raw: list[dict[str, Any]] = []
    for prototype in prototypes:
        x1, y1, x2, y2 = prototype["bbox_xyxy"]
        template = edge_map[y1:y2, x1:x2]
        raw.extend(
            match_template_multiscale(
                edge_map,
                template,
                search_box,
                scales,
                scales,
                cfg.arch_raw_template_threshold,
                "arch",
                prototype["id"],
                cfg.arch_template_top_fraction,
            )
        )

    filtered: list[dict[str, Any]] = []
    springline_tolerance = height * cfg.arch_springline_tolerance_ratio
    for candidate in raw:
        candidate["bbox_xyxy"] = clip_box(candidate["bbox_xyxy"], width, height)
        box = candidate["bbox_xyxy"]
        candidate_width = box_width(box)
        candidate_height = box_height(box)
        springline_gap = abs(float(box[3]) - medians["bottom_y"])
        if not (
            medians["width"] * cfg.arch_width_min_family_ratio
            <= candidate_width
            <= medians["width"] * cfg.arch_width_max_family_ratio
        ):
            continue
        if not (
            medians["height"] * cfg.arch_height_min_family_ratio
            <= candidate_height
            <= medians["height"] * cfg.arch_height_max_family_ratio
        ):
            continue
        if springline_gap > springline_tolerance:
            continue
        axis_info = arch_axis_support(candidate, axes, width, cfg)
        candidate.update(axis_info)
        candidate["springline_gap_px"] = round(springline_gap, 3)
        candidate["proposal_score"] = round(
            0.72 * float(candidate["template_score"])
            + 0.28 * float(axis_info["axis_support_score"]),
            6,
        )
        filtered.append(candidate)

    return (
        nms_candidates(filtered, cfg.arch_nms_iou, cfg.arch_max_raw_candidates),
        {"search_box": search_box, "family_medians": medians, "raw_count": len(raw)},
    )


def generate_pilaster_candidates(
    rgb: np.ndarray,
    gradient_map: np.ndarray,
    prototypes: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height, width = rgb.shape[:2]
    medians = family_medians(prototypes)
    search_box = [
        int(width * cfg.facade_x_margin_ratio),
        max(int(height * cfg.lower_facade_y_min_ratio), int(medians["top_y"] - height * 0.08)),
        int(width * (1.0 - cfg.facade_x_margin_ratio)),
        min(int(height * cfg.lower_facade_y_max_ratio), int(medians["bottom_y"] + height * 0.09)),
    ]
    search_box = clip_box(search_box, width, height)

    scales_w = np.arange(
        cfg.pilaster_scale_w_min,
        cfg.pilaster_scale_w_max + cfg.pilaster_scale_step / 2.0,
        cfg.pilaster_scale_step,
    )
    scales_h = np.arange(
        cfg.pilaster_scale_h_min,
        cfg.pilaster_scale_h_max + cfg.pilaster_scale_step / 2.0,
        cfg.pilaster_scale_step,
    )
    raw: list[dict[str, Any]] = []
    for prototype in prototypes:
        x1, y1, x2, y2 = prototype["bbox_xyxy"]
        template = gradient_map[y1:y2, x1:x2]
        raw.extend(
            match_template_multiscale(
                gradient_map,
                template,
                search_box,
                scales_w,
                scales_h,
                cfg.pilaster_raw_template_threshold,
                "pilaster",
                prototype["id"],
                None,
            )
        )

    # Eksen merkezli ek adaylar: eksen tek başına nesne üretmez; en iyi lokal şablon skoru gerekir.
    median_width = int(round(medians["width"]))
    median_height = int(round(medians["height"]))
    for axis in axes:
        center_x = int(round(float(axis["x"])))
        if not (search_box[0] <= center_x <= search_box[2]):
            continue
        candidate = clip_box(
            [
                center_x - median_width / 2,
                medians["center_y"] - median_height / 2,
                center_x + median_width / 2,
                medians["center_y"] + median_height / 2,
            ],
            width,
            height,
        )
        x1, y1, x2, y2 = candidate
        best_score = 0.0
        best_prototype = ""
        crop = gradient_map[y1:y2, x1:x2]
        for prototype in prototypes:
            px1, py1, px2, py2 = prototype["bbox_xyxy"]
            template = gradient_map[py1:py2, px1:px2]
            resized = cv2.resize(template, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
            if crop.shape != resized.shape or float(np.std(resized)) < 2.0:
                continue
            score = float(cv2.matchTemplate(crop, resized, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_score = score
                best_prototype = prototype["id"]
        if best_score >= cfg.pilaster_raw_template_threshold * 0.72:
            raw.append(
                {
                    "class": "pilaster",
                    "bbox_xyxy": candidate,
                    "template_score": round(best_score, 6),
                    "proposal_score": round(best_score, 6),
                    "prototype_id": best_prototype,
                    "template_scale_w": 1.0,
                    "template_scale_h": 1.0,
                    "source": "axis_centered_prototype_check",
                }
            )

    center_y_tolerance = height * cfg.pilaster_center_y_tolerance_ratio
    filtered: list[dict[str, Any]] = []
    for candidate in raw:
        candidate["bbox_xyxy"] = clip_box(candidate["bbox_xyxy"], width, height)
        center_y = box_center(candidate["bbox_xyxy"])[1]
        center_y_gap = abs(center_y - medians["center_y"])
        if center_y_gap > center_y_tolerance:
            continue
        axis_info = pilaster_axis_support(candidate, axes, width, cfg)
        candidate.update(axis_info)
        candidate["center_y_gap_px"] = round(center_y_gap, 3)
        candidate["proposal_score"] = round(
            0.70 * float(candidate["template_score"])
            + 0.30 * float(axis_info["axis_support_score"]),
            6,
        )
        filtered.append(candidate)

    return (
        nms_candidates(filtered, cfg.pilaster_nms_iou, cfg.pilaster_max_raw_candidates),
        {"search_box": search_box, "family_medians": medians, "raw_count": len(raw)},
    )


# =============================================================================
# CANDIDATE SCORING / CLASSIFICATION
# =============================================================================
def classify_candidates(
    candidates: list[dict[str, Any]],
    class_name: str,
    image: np.ndarray,
    gradient: np.ndarray,
    edge: np.ndarray,
    damage_mask: np.ndarray,
    cfg: Config,
    dino_available: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["candidate_id"] = f"recovered_{class_name}_{index + 1:03d}"
        metrics = candidate_edge_metrics(item, image, gradient, edge, damage_mask)
        item["metrics"] = metrics
        damage = float(metrics["damage_fraction"])
        visible = float(metrics["visible_fraction"])
        template_score = float(item.get("template_score", 0.0))
        axis_score = float(item.get("axis_support_score", 0.0))
        dino_similarity = item.get("dinov2_similarity")

        if class_name == "arch":
            geometry_score = min(1.0, 2.5 * float(metrics["upper_edge_density"]))
            visual_score = min(1.0, 2.0 * float(metrics["gradient_activity"]))
            strong_template = cfg.arch_strong_template_threshold
            accept_threshold = cfg.arch_accept_fusion_threshold
            review_threshold = cfg.arch_review_fusion_threshold
            dino_min = cfg.dinov2_arch_min_similarity
        else:
            geometry_score = min(1.0, float(metrics["vertical_activity"]) / 0.55)
            visual_score = min(1.0, 2.2 * float(metrics["gradient_activity"]))
            strong_template = cfg.pilaster_strong_template_threshold
            accept_threshold = cfg.pilaster_accept_fusion_threshold
            review_threshold = cfg.pilaster_review_fusion_threshold
            dino_min = cfg.dinov2_pilaster_min_similarity

        if dino_available and dino_similarity is not None:
            dino_score = np.clip(
                (float(dino_similarity) - cfg.dinov2_review_min_similarity)
                / max(1e-6, 1.0 - cfg.dinov2_review_min_similarity),
                0.0,
                1.0,
            )
            weights = (0.29, 0.31, 0.17, 0.13, 0.10)
            fusion_score = (
                weights[0] * min(1.0, template_score / max(1e-6, strong_template))
                + weights[1] * dino_score
                + weights[2] * axis_score
                + weights[3] * geometry_score
                + weights[4] * visual_score
            )
            dino_gate = float(dino_similarity) >= dino_min
        else:
            weights = (0.45, 0.0, 0.23, 0.19, 0.13)
            fusion_score = (
                weights[0] * min(1.0, template_score / max(1e-6, strong_template))
                + weights[2] * axis_score
                + weights[3] * geometry_score
                + weights[4] * visual_score
            )
            dino_gate = True

        item["evidence"] = {
            "template_score": round(template_score, 6),
            "dinov2_similarity": dino_similarity,
            "axis_support_score": round(axis_score, 6),
            "geometry_score": round(float(geometry_score), 6),
            "visual_activity_score": round(float(visual_score), 6),
            "fusion_score": round(float(fusion_score), 6),
            "dinov2_gate_passed": bool(dino_gate),
        }

        reasons: list[str] = []
        if damage > cfg.reject_damage_above:
            reasons.append("damage_overlap_too_high_for_observed_recovery")
        if visible < cfg.minimum_visible_fraction:
            reasons.append("visible_evidence_too_low")
        if class_name == "pilaster" and float(metrics["vertical_activity"]) < cfg.pilaster_vertical_activity_min:
            reasons.append("vertical_activity_too_low")
        if dino_available and not dino_gate:
            reasons.append("dinov2_similarity_below_acceptance_gate")

        if class_name == "arch":
            class_gate = (
                template_score >= cfg.arch_raw_template_threshold
                and (axis_score >= 0.24 or template_score >= cfg.arch_strong_template_threshold)
                and float(metrics["upper_edge_density"]) >= 0.075
            )
        else:
            class_gate = (
                template_score >= cfg.pilaster_raw_template_threshold
                and axis_score >= 0.18
                and float(metrics["vertical_activity"]) >= cfg.pilaster_vertical_activity_min
            )

        if not class_gate:
            reasons.append("class_specific_visual_gate_failed")

        hard_reject = (
            damage > cfg.reject_damage_above
            or visible < cfg.minimum_visible_fraction
            or not class_gate
        )

        # OpenCV-only durumda yeni örnek otomatik kabul edilmez. DINOv2 yoksa
        # güçlü görünen adaylar yalnız review katmanına gider.
        auto_accept_allowed = dino_available and dino_gate

        if (
            not hard_reject
            and auto_accept_allowed
            and fusion_score >= accept_threshold
        ):
            if damage <= cfg.observed_intact_damage_max:
                item["status"] = "observed_intact"
            elif damage <= cfg.observed_fragmented_damage_max:
                item["status"] = "observed_fragmented"
            else:
                item["status"] = "candidate_needs_review"
            item["decision_reasons"] = [
                "prototype_template_support",
                "dinov2_prototype_similarity",
                "visible_pixel_evidence",
                "class_specific_geometry_gate",
            ]
            observed.append(item)
        elif not hard_reject and fusion_score >= review_threshold:
            item["status"] = "candidate_needs_review"
            if not dino_available:
                reasons.append("dinov2_unavailable_no_auto_acceptance")
            item["decision_reasons"] = reasons or ["evidence_is_plausible_but_not_sufficient"]
            review.append(item)
        else:
            item["status"] = "rejected_similarity"
            item["decision_reasons"] = reasons or ["combined_similarity_below_threshold"]
            rejected.append(item)

    return observed, review, rejected


def dedupe_recovered(
    observed: list[dict[str, Any]],
    review: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined = sorted(
        observed + review,
        key=lambda item: float(item["evidence"]["fusion_score"]),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for item in combined:
        if any(box_iou(item["bbox_xyxy"], previous["bbox_xyxy"]) >= cfg.recovered_duplicate_iou for previous in kept):
            continue
        kept.append(item)
    kept_ids = {item["candidate_id"] for item in kept}
    return (
        [item for item in observed if item["candidate_id"] in kept_ids],
        [item for item in review if item["candidate_id"] in kept_ids],
    )


def suppress_nearby_alternatives(
    items: list[dict[str, Any]],
    class_name: str,
    family_stats: dict[str, float],
    max_count: int,
) -> list[dict[str, Any]]:
    """Aynı görünen eleman çevresindeki farklı ölçekli alternatif kutuları tekilleştirir."""
    ordered = sorted(
        items,
        key=lambda item: float(item.get("evidence", {}).get("fusion_score", 0.0)),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    if class_name == "arch":
        center_x_tolerance = max(12.0, family_stats["width"] * 0.62)
        center_y_tolerance = max(10.0, family_stats["height"] * 0.55)
    else:
        center_x_tolerance = max(8.0, family_stats["width"] * 0.78)
        center_y_tolerance = max(18.0, family_stats["height"] * 0.38)

    for item in ordered:
        center_x, center_y = box_center(item["bbox_xyxy"])
        same_instance = False
        for previous in kept:
            prev_x, prev_y = box_center(previous["bbox_xyxy"])
            if (
                abs(center_x - prev_x) <= center_x_tolerance
                and abs(center_y - prev_y) <= center_y_tolerance
            ):
                same_instance = True
                break
        if not same_instance:
            kept.append(item)
        if len(kept) >= max_count:
            break
    return kept


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
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def draw_item(
    image: np.ndarray,
    item: dict[str, Any],
    label_prefix: str,
    cfg: Config,
    offset_index: int = 0,
) -> None:
    status = str(item.get("status", "candidate_needs_review"))
    color = COLORS.get(status, COLORS["candidate_needs_review"])
    x1, y1, x2, y2 = item["bbox_xyxy"]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, cfg.box_thickness)
    evidence = item.get("evidence", {})
    fusion = evidence.get("fusion_score")
    suffix = f" F={float(fusion):.2f}" if fusion is not None else ""
    identifier = item.get("candidate_id", item.get("id", ""))
    status_short = {
        "known_prototype": "KNOWN",
        "observed_intact": "OBSERVED",
        "observed_fragmented": "FRAGMENT",
        "candidate_needs_review": "REVIEW",
        "rejected_similarity": "REJECT",
    }.get(status, status.upper())
    text = f"{label_prefix} {identifier} {status_short}{suffix}"
    text_y = max(49, y1 - 5 - (offset_index % 4) * 13)
    cv2.putText(
        image,
        text,
        (max(2, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.text_scale,
        color,
        1,
        cv2.LINE_AA,
    )


def make_visual(
    original: np.ndarray,
    damage_mask: np.ndarray,
    known_arches: list[dict[str, Any]],
    known_pilasters: list[dict[str, Any]],
    observed_arches: list[dict[str, Any]],
    observed_pilasters: list[dict[str, Any]],
    review_arches: list[dict[str, Any]],
    review_pilasters: list[dict[str, Any]],
    show_arches: bool,
    show_pilasters: bool,
    header: str,
    cfg: Config,
) -> np.ndarray:
    image = original.astype(np.float32).copy()
    red = np.zeros_like(image)
    red[..., 0] = 255
    mask = (damage_mask > 0)[..., None].astype(np.float32)
    image = image * (1.0 - mask * 0.08) + red * mask * 0.08
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = add_header(image, header)

    if show_arches:
        for index, item in enumerate(known_arches):
            known = dict(item)
            known["status"] = "known_prototype"
            draw_item(image, known, "KNOWN ARCH", cfg, index)
        for index, item in enumerate(observed_arches):
            draw_item(image, item, "RECOVERED ARCH", cfg, index)
        for index, item in enumerate(review_arches):
            draw_item(image, item, "REVIEW ARCH", cfg, index)

    if show_pilasters:
        for index, item in enumerate(known_pilasters):
            known = dict(item)
            known["status"] = "known_prototype"
            draw_item(image, known, "KNOWN PILASTER", cfg, index)
        for index, item in enumerate(observed_pilasters):
            draw_item(image, item, "RECOVERED PILASTER", cfg, index)
        for index, item in enumerate(review_pilasters):
            draw_item(image, item, "REVIEW PILASTER", cfg, index)
    return image


def make_coverage_map(
    original: np.ndarray,
    damage_mask: np.ndarray,
    axes: list[dict[str, Any]],
    arch_search: dict[str, Any],
    pilaster_search: dict[str, Any],
    all_items: list[dict[str, Any]],
    cfg: Config,
) -> np.ndarray:
    image = add_header(
        original,
        "STAGE 3B-v4 COVERAGE MAP - SEARCH ZONES / AXES / OBSERVED EVIDENCE",
    )
    ax1, ay1, ax2, ay2 = arch_search["search_box"]
    px1, py1, px2, py2 = pilaster_search["search_box"]
    cv2.rectangle(image, (ax1, ay1), (ax2, ay2), (60, 220, 80), 1)
    cv2.rectangle(image, (px1, py1), (px2, py2), (220, 80, 230), 1)

    for axis in axes:
        x = int(round(float(axis["x"])))
        color = COLORS["axis"] if axis["classification"] != "weak_axis" else (150, 150, 150)
        cv2.line(image, (x, min(ay1, py1)), (x, max(ay2, py2)), color, 1, cv2.LINE_AA)

    for item in all_items:
        draw_item(image, item, str(item.get("class", "OBJECT")).upper(), cfg)

    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, COLORS["damage"], 1)
    return image


# =============================================================================
# SERIALIZATION
# =============================================================================
def compact_known(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "class": item["class"],
        "bbox_xyxy": item["bbox_xyxy"],
        "score": round(float(item.get("score", 0.0)), 6),
        "damage_fraction": round(float(item.get("damage_fraction", 0.0)), 6),
        "status": "known_prototype",
        "source": "stage3b_v3_clean_family",
    }


def summarize_status(items: list[dict[str, Any]]) -> dict[str, int]:
    statuses = {
        "known_prototype": 0,
        "observed_intact": 0,
        "observed_fragmented": 0,
        "candidate_needs_review": 0,
        "rejected_similarity": 0,
    }
    for item in items:
        status = str(item.get("status", "candidate_needs_review"))
        statuses[status] = statuses.get(status, 0) + 1
    return statuses


# =============================================================================
# MAIN
# =============================================================================
def run(cfg: Config) -> Path:
    parent, pilaster_family_name = find_parent_run(cfg)
    run_id, run_dir = create_new_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir, run_id)
    started = time.perf_counter()
    meta_path = run_dir / "run_meta.json"
    source_script = Path(__file__)

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": "v8_stage3b_v4_observed_instance_recovery",
        "status": "running",
        "started_at": now_iso(),
        "parent_run": {"run_id": parent.name, "path": str(parent.resolve())},
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
            "new_missing_elements_inferred": 0,
            "new_geometry_inside_damage_created": 0,
            "new_axes_created": 0,
            "stable_diffusion_used": False,
            "note": "Yalnız fotoğrafta görünen piksel kanıtı taşıyan adaylar aranmıştır.",
        },
        "copied_inputs": {},
        "dinov2": {},
        "summary": {},
        "outputs": [],
    }

    def flush_meta() -> None:
        write_json(meta_path, metadata)

    verifier: Dinov2Verifier | None = None

    try:
        logger.info("=" * 82)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[PARENT] {parent.resolve()}")
        logger.info("[NEW RUN] Önceki klasörün üzerine yazılmayacak")
        logger.info("[SAFE] Hasar alanına yeni eleman veya geometri yerleştirilmeyecek")
        logger.info("[STAGE] Stage 3B-v4 — görünen örnekleri prototiplerle geri kazanma")
        logger.info("=" * 82)

        metadata["copied_inputs"] = copy_inputs(parent, pilaster_family_name, run_dir, cfg)
        flush_meta()

        image = load_rgb(run_dir / "input.png")
        height, width = image.shape[:2]
        damage_mask = load_binary_mask(run_dir / "completion_mask.png", (width, height))
        gradient = make_gradient_map(image)
        edge = make_edge_map(image)

        arch_prototypes, known_arch_fragments, arch_document = load_arch_prototypes(
            run_dir / "source_stage3b_v3_arch_families.json"
        )
        pilaster_prototypes, known_pilaster_fragments, pilaster_document = load_pilaster_prototypes(
            run_dir / "source_stage3b_v3_pilaster_families.json"
        )
        axes = load_axes(run_dir / "source_stage2d_rhythm_groups.json")

        logger.info(
            f"[PROTOTYPE] arch={len(arch_prototypes)}, pilaster={len(pilaster_prototypes)}, "
            f"known_fragments={len(known_arch_fragments) + len(known_pilaster_fragments)}, "
            f"axes={len(axes)}"
        )

        arch_candidates, arch_search = generate_arch_candidates(
            image,
            edge,
            arch_prototypes,
            axes,
            cfg,
        )
        pilaster_candidates, pilaster_search = generate_pilaster_candidates(
            image,
            gradient,
            pilaster_prototypes,
            axes,
            cfg,
        )
        logger.info(
            f"[PROPOSAL] arch={len(arch_candidates)}, pilaster={len(pilaster_candidates)}"
        )

        known_arch_boxes = [item["bbox_xyxy"] for item in arch_prototypes + known_arch_fragments]
        known_pilaster_boxes = [item["bbox_xyxy"] for item in pilaster_prototypes + known_pilaster_fragments]
        arch_candidates, arch_known_duplicates = remove_known_duplicates(
            arch_candidates,
            known_arch_boxes,
            cfg.known_box_duplicate_iou,
        )
        pilaster_candidates, pilaster_known_duplicates = remove_known_duplicates(
            pilaster_candidates,
            known_pilaster_boxes,
            cfg.known_box_duplicate_iou,
        )

        verifier = Dinov2Verifier(cfg, logger)
        arch_dino = add_dino_similarity(
            arch_candidates,
            arch_prototypes,
            image,
            verifier,
            cfg,
        )
        pilaster_dino = add_dino_similarity(
            pilaster_candidates,
            pilaster_prototypes,
            image,
            verifier,
            cfg,
        )
        metadata["dinov2"] = {
            "available": bool(verifier.available),
            "model_id": cfg.dinov2_model_id,
            "arch": arch_dino,
            "pilaster": pilaster_dino,
        }

        observed_arches, review_arches, rejected_arches = classify_candidates(
            arch_candidates,
            "arch",
            image,
            gradient,
            edge,
            damage_mask,
            cfg,
            verifier.available,
        )
        observed_pilasters, review_pilasters, rejected_pilasters = classify_candidates(
            pilaster_candidates,
            "pilaster",
            image,
            gradient,
            edge,
            damage_mask,
            cfg,
            verifier.available,
        )

        observed_arches, review_arches = dedupe_recovered(observed_arches, review_arches, cfg)
        observed_pilasters, review_pilasters = dedupe_recovered(
            observed_pilasters,
            review_pilasters,
            cfg,
        )
        arch_stats = family_medians(arch_prototypes)
        pilaster_stats = family_medians(pilaster_prototypes)
        observed_arches = suppress_nearby_alternatives(observed_arches, "arch", arch_stats, 10)
        review_arches = suppress_nearby_alternatives(review_arches, "arch", arch_stats, cfg.max_review_per_class)
        observed_pilasters = suppress_nearby_alternatives(observed_pilasters, "pilaster", pilaster_stats, 12)
        review_pilasters = suppress_nearby_alternatives(review_pilasters, "pilaster", pilaster_stats, cfg.max_review_per_class)
        review_arches = sorted(
            review_arches,
            key=lambda item: float(item["evidence"]["fusion_score"]),
            reverse=True,
        )[:cfg.max_review_per_class]
        review_pilasters = sorted(
            review_pilasters,
            key=lambda item: float(item["evidence"]["fusion_score"]),
            reverse=True,
        )[:cfg.max_review_per_class]
        rejected_arches = sorted(
            rejected_arches + arch_known_duplicates,
            key=lambda item: float(item.get("evidence", {}).get("fusion_score", item.get("template_score", 0.0))),
            reverse=True,
        )[:cfg.max_rejected_per_class]
        rejected_pilasters = sorted(
            rejected_pilasters + pilaster_known_duplicates,
            key=lambda item: float(item.get("evidence", {}).get("fusion_score", item.get("template_score", 0.0))),
            reverse=True,
        )[:cfg.max_rejected_per_class]

        known_arch_records = [compact_known(item) for item in arch_prototypes]
        known_arch_fragment_records = [dict(item) for item in known_arch_fragments]
        known_pilaster_records = [compact_known(item) for item in pilaster_prototypes]
        known_pilaster_fragment_records = [dict(item) for item in known_pilaster_fragments]
        observed_records = (
            known_arch_records
            + known_arch_fragment_records
            + known_pilaster_records
            + known_pilaster_fragment_records
            + observed_arches
            + observed_pilasters
        )
        review_records = review_arches + review_pilasters
        rejected_records = rejected_arches + rejected_pilasters

        write_json(
            run_dir / "stage3b_v4_observed_instances.json",
            {
                "run_id": run_id,
                "parent_run": metadata["parent_run"],
                "policy": {
                    "existing_prototypes_preserved": True,
                    "missing_element_inference": False,
                    "damage_area_completion": False,
                    "recovered_candidates_require_visible_pixel_evidence": True,
                },
                "search": {"arch": arch_search, "pilaster": pilaster_search},
                "known_prototypes": {
                    "arches": known_arch_records,
                    "arch_fragments": known_arch_fragment_records,
                    "pilasters": known_pilaster_records,
                    "pilaster_fragments": known_pilaster_fragment_records,
                },
                "recovered_observed": {
                    "arches": observed_arches,
                    "pilasters": observed_pilasters,
                },
                "summary": summarize_status(observed_records),
            },
        )
        write_json(
            run_dir / "stage3b_v4_review_candidates.json",
            {
                "run_id": run_id,
                "note": "Bu adaylar otomatik kabul edilmemiştir; görsel inceleme gerekir.",
                "arches": review_arches,
                "pilasters": review_pilasters,
            },
        )
        write_json(
            run_dir / "stage3b_v4_rejected_candidates.json",
            {
                "run_id": run_id,
                "arches": rejected_arches,
                "pilasters": rejected_pilasters,
            },
        )

        all_visual = make_visual(
            image,
            damage_mask,
            arch_prototypes,
            pilaster_prototypes,
            known_arch_fragments + observed_arches,
            known_pilaster_fragments + observed_pilasters,
            [],
            [],
            True,
            True,
            "STAGE 3B-v4 OBSERVED INSTANCES - NO COMPLETION / ACCEPTED EVIDENCE ONLY",
            cfg,
        )
        arch_visual = make_visual(
            image,
            damage_mask,
            arch_prototypes,
            [],
            known_arch_fragments + observed_arches,
            [],
            [],
            [],
            True,
            False,
            "STAGE 3B-v4 ARCH INSTANCE RECOVERY - VISIBLE PIXEL EVIDENCE ONLY",
            cfg,
        )
        pilaster_visual = make_visual(
            image,
            damage_mask,
            [],
            pilaster_prototypes,
            [],
            known_pilaster_fragments + observed_pilasters,
            [],
            [],
            False,
            True,
            "STAGE 3B-v4 PILASTER INSTANCE RECOVERY - VISIBLE PIXEL EVIDENCE ONLY",
            cfg,
        )
        review_visual = make_visual(
            image,
            damage_mask,
            [],
            [],
            [],
            [],
            review_arches,
            review_pilasters,
            True,
            True,
            "STAGE 3B-v4 REVIEW CANDIDATES - NOT ACCEPTED / NO COMPLETION",
            cfg,
        )
        coverage = make_coverage_map(
            image,
            damage_mask,
            axes,
            arch_search,
            pilaster_search,
            observed_records + review_records,
            cfg,
        )
        save_rgb(run_dir / "stage3b_v4_all_observed_instances.png", all_visual)
        save_rgb(run_dir / "stage3b_v4_arch_instances.png", arch_visual)
        save_rgb(run_dir / "stage3b_v4_pilaster_instances.png", pilaster_visual)
        save_rgb(run_dir / "stage3b_v4_review_candidates.png", review_visual)
        save_rgb(run_dir / "stage3b_v4_coverage_map.png", coverage)

        summary = {
            "known_arch_prototypes": len(arch_prototypes),
            "known_arch_fragments": len(known_arch_fragments),
            "known_pilaster_prototypes": len(pilaster_prototypes),
            "known_pilaster_fragments": len(known_pilaster_fragments),
            "recovered_arch_observed": len(observed_arches),
            "recovered_pilaster_observed": len(observed_pilasters),
            "arch_review_candidates": len(review_arches),
            "pilaster_review_candidates": len(review_pilasters),
            "arch_rejected": len(rejected_arches),
            "pilaster_rejected": len(rejected_pilasters),
            "new_missing_elements_inferred": 0,
            "new_geometry_inside_damage_created": 0,
        }
        metadata["summary"] = summary

        report_lines = [
            "STAGE 3B-v4 — OBSERVED INSTANCE RECOVERY REPORT",
            f"Run: {run_id}",
            f"Parent: {parent.resolve()}",
            "",
            "GÜVENLİK:",
            "- Hasar alanına yeni eleman yerleştirilmedi.",
            "- Eksik kemer/pilaster tahmini yapılmadı.",
            "- Yalnız görüntüde mevcut piksel kanıtı taşıyan adaylar arandı.",
            "- Review adayları otomatik doğru kabul edilmedi.",
            "",
            f"Known arch prototypes: {len(arch_prototypes)}",
            f"Known arch fragments: {len(known_arch_fragments)}",
            f"Known pilaster prototypes: {len(pilaster_prototypes)}",
            f"Known pilaster fragments: {len(known_pilaster_fragments)}",
            f"Recovered observed arches: {len(observed_arches)}",
            f"Recovered observed pilasters: {len(observed_pilasters)}",
            f"Arch review candidates: {len(review_arches)}",
            f"Pilaster review candidates: {len(review_pilasters)}",
            f"DINOv2 available: {verifier.available}",
            "",
            "KULLANIM:",
            "- Turkuaz kutular: görüntü kanıtıyla geri kazanılan sağlam örnekler.",
            "- Turuncu kutular: görüntüde kalan fakat hasarla kesilen örnekler.",
            "- Sarı kutular: kullanıcı/restoratör incelemesi gereken adaylar.",
            "- Yeşil kutular: V3'ten gelen güvenilir prototipler.",
            "",
            "SONRAKİ AŞAMA KOŞULU:",
            "- Review adayları kontrol edilmeden eksik-eleman hipotezine geçilmemelidir.",
        ]
        (run_dir / "stage3b_v4_report.txt").write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        metadata["outputs"] = [
            {"file": "stage3b_v4_all_observed_instances.png", "type": "combined_observed_and_review"},
            {"file": "stage3b_v4_arch_instances.png", "type": "arch_instance_recovery"},
            {"file": "stage3b_v4_pilaster_instances.png", "type": "pilaster_instance_recovery"},
            {"file": "stage3b_v4_review_candidates.png", "type": "manual_review_visual"},
            {"file": "stage3b_v4_coverage_map.png", "type": "search_and_axis_coverage"},
            {"file": "stage3b_v4_observed_instances.json", "type": "accepted_observed_instances"},
            {"file": "stage3b_v4_review_candidates.json", "type": "manual_review_candidates"},
            {"file": "stage3b_v4_rejected_candidates.json", "type": "rejected_diagnostics"},
            {"file": "stage3b_v4_report.txt", "type": "human_readable_report"},
            {"file": "run.log", "type": "log"},
            {"file": "run_meta.json", "type": "metadata"},
        ]
        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush_meta()

        logger.info("[OK] Stage 3B-v4 tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(
            "[COUNT] "
            f"known_arch={len(arch_prototypes)}, recovered_arch={len(observed_arches)}, "
            f"known_pilaster={len(pilaster_prototypes)}, recovered_pilaster={len(observed_pilasters)}, "
            f"review={len(review_records)}, inferred_missing=0"
        )
        logger.info(
            "[FILES] stage3b_v4_all_observed_instances.png, "
            "stage3b_v4_observed_instances.json, stage3b_v4_review_candidates.json"
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
    finally:
        if verifier is not None:
            verifier.release()


def main() -> None:
    run_dir = run(Config())
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
