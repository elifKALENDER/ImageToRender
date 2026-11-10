#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ImageToRender2Restoration_v8_stage3b_element_families_v2.py
============================================================

STAGE 3B-v2 — Element Families (Perspective-normalized)
Amaç:
- Stage 3A / evidence çıktılarındaki mevcut elemanları analiz etmek
- özellikle kemer ailelerini perspektife göre normalize ederek gruplamak
- sağlam prototip üyeleri, parçalı üyeleri ve outlier'ları ayırmak
- yeni eleman üretmemek

KESİN KURALLAR
--------------
- yeni mimari eleman oluşturmaz
- eksik yere kemer/pilaster eklemez
- opening kutularını kemer ailesi prototipi olarak kullanmaz
- her çalıştırmada yeni restore_... klasörü açar
"""

from __future__ import annotations

import json
import math
import shutil
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    # Girdi klasörü için aranan dosyalar
    require_files: tuple = (
        "input.png",
        "completion_mask.png",
        "source_validated_elements.json",
        "source_tentative_elements.json",
        "source_stage2d_rhythm_groups.json",
        "stage3a_existing_evidence_skeleton.json",
    )

    # Aday filtreleri
    min_arch_score: float = 0.40
    min_pilaster_score: float = 0.40
    min_opening_score: float = 0.35

    # Hasar durumları
    intact_damage_max: float = 0.18
    fragmented_damage_min: float = 0.18
    fragmented_damage_max: float = 0.75

    # Outlier kuralları
    top_outlier_y_ratio: float = 0.36
    huge_width_ratio: float = 0.22
    huge_height_ratio: float = 0.16

    # Aile eşleme toleransları
    arch_family_norm_width_tol: float = 0.24
    arch_family_norm_height_tol: float = 0.28
    arch_family_springline_tol: float = 0.10
    arch_fragment_assign_tol: float = 0.34

    pilaster_family_width_tol: float = 0.30
    pilaster_family_height_tol: float = 0.30
    pilaster_family_center_tol: float = 0.18

    # Çizim
    line_thickness: int = 2
    text_scale: float = 0.48

    # Rapor
    max_report_items_per_family: int = 20


# =========================================================
# BASIC HELPERS
# =========================================================

def now_id() -> str:
    return datetime.now().strftime("restore_%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, img: np.ndarray) -> None:
    Image.fromarray(img.astype(np.uint8)).save(path)


def load_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    w, h = size_xy
    img = Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    arr = np.asarray(img)
    return np.where(arr >= 128, 255, 0).astype(np.uint8)


def create_run_dir(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    while True:
        run_dir = root / now_id()
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            time.sleep(1.0)


def draw_header(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (18, 18, 18), -1)
    cv2.putText(
        out, text, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.54,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def flatten_any_elements(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in (
            "elements",
            "validated_elements",
            "tentative_elements",
            "items",
        ):
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]

    return []


def get_label(elem: dict) -> str:
    for key in ("class", "label", "class_name", "category", "type"):
        val = elem.get(key)
        if val:
            return str(val).strip().lower()
    return "unknown"


def get_score(elem: dict) -> float:
    for key in ("final_score", "score", "confidence", "fused_score"):
        val = elem.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    return 0.0


def get_box(elem: dict):
    for key in ("box_xyxy", "bbox_xyxy", "box", "bbox"):
        val = elem.get(key)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [int(round(float(v))) for v in val]
    return None


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return [x1, y1, x2, y2]


def box_size(box):
    x1, y1, x2, y2 = box
    return x2 - x1, y2 - y1


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_damage_fraction(box, mask: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi > 0) / roi.size)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_elements(items: list[dict], iou_threshold=0.72) -> tuple[list[dict], int]:
    kept = []
    removed = 0

    for item in sorted(items, key=lambda z: z["score"], reverse=True):
        duplicate = False
        for prev in kept:
            if item["label"] == prev["label"] and iou(item["box"], prev["box"]) >= iou_threshold:
                duplicate = True
                removed += 1
                break
        if not duplicate:
            kept.append(item)

    return kept, removed


def family_color(index: int):
    palette = [
        (60, 200, 70),
        (70, 170, 255),
        (255, 180, 60),
        (210, 90, 255),
        (50, 230, 200),
        (255, 100, 120),
    ]
    return palette[index % len(palette)]


# =========================================================
# FIND PARENT / COPY INPUTS
# =========================================================

def find_parent_run(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        p = Path(cfg.parent_run_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Parent klasörü bulunamadı: {p}")
        return p

    candidates = sorted(
        [p for p in Path(cfg.output_root).glob("restore_*") if p.is_dir()],
        reverse=True
    )

    for p in candidates:
        ok = True
        for name in cfg.require_files:
            if not (p / name).exists():
                ok = False
                break
        if ok:
            return p

    raise FileNotFoundError("Stage 3A başarılı parent run bulunamadı.")


def copy_required_inputs(parent: Path, run_dir: Path, cfg: Config):
    for name in cfg.require_files:
        src = parent / name
        dst = run_dir / name
        shutil.copy2(src, dst)


# =========================================================
# LOAD / NORMALIZE ELEMENTS
# =========================================================

def load_raw_elements(run_dir: Path, damage_mask: np.ndarray, w: int, h: int):
    validated = read_json(run_dir / "source_validated_elements.json")
    tentative = read_json(run_dir / "source_tentative_elements.json")

    items = []

    for group_name, data in (("validated", validated), ("tentative", tentative)):
        for idx, elem in enumerate(flatten_any_elements(data)):
            label = get_label(elem)
            score = get_score(elem)
            box = get_box(elem)
            if box is None:
                continue

            box = clip_box(box, w, h)
            cx, cy = box_center(box)
            bw, bh = box_size(box)

            items.append({
                "id": elem.get("id", f"{group_name}_{idx:03d}"),
                "source_group": group_name,
                "label": label,
                "score": score,
                "box": box,
                "cx": cx,
                "cy": cy,
                "w": bw,
                "h": bh,
                "damage_fraction": box_damage_fraction(box, damage_mask),
                "raw": elem,
            })

    items, duplicate_count = dedupe_elements(items, iou_threshold=0.72)
    return items, duplicate_count


def split_elements(items: list[dict], cfg: Config):
    arches = []
    pilasters = []
    openings = []
    others = []

    for e in items:
        label = e["label"]

        if "arch" in label:
            if e["score"] >= cfg.min_arch_score:
                arches.append(e)
        elif "pilaster" in label or "column" in label:
            if e["score"] >= cfg.min_pilaster_score:
                pilasters.append(e)
        elif "opening" in label or "door" in label or "window" in label:
            if e["score"] >= cfg.min_opening_score:
                openings.append(e)
        else:
            others.append(e)

    return arches, pilasters, openings, others


# =========================================================
# RHYTHM / AXIS
# =========================================================

def load_axes(run_dir: Path):
    rhythm = read_json(run_dir / "source_stage2d_rhythm_groups.json")
    axes = []

    for i, ax in enumerate(rhythm.get("axes", [])):
        cls = str(ax.get("classification", "")).lower()
        score = safe_float(ax.get("metrics", {}).get("structural_score", 0.0), 0.0)

        if cls not in ("valid_structural_axis", "fragmented_structural_axis"):
            continue

        x = safe_float(ax.get("axis_x", 0.0), None)
        if x is None:
            continue

        axes.append({
            "id": ax.get("source_id", f"axis_{i+1:02d}"),
            "axis_x": float(x),
            "classification": cls,
            "score": score,
        })

    axes.sort(key=lambda z: z["axis_x"])
    return axes


def nearest_pilaster_support_spacing(arch: dict, pilasters: list[dict]):
    """
    Kemer genişliğini perspektife göre normalize etmek için:
    kemerin merkezinin sol ve sağında en yakın pilaster merkezlerini bulmaya çalışır.
    """
    if len(pilasters) < 2:
        return None, None, None

    cx = arch["cx"]
    pil_sorted = sorted(pilasters, key=lambda z: z["cx"])

    left_candidates = [p for p in pil_sorted if p["cx"] < cx]
    right_candidates = [p for p in pil_sorted if p["cx"] > cx]

    if not left_candidates or not right_candidates:
        return None, None, None

    left = left_candidates[-1]
    right = right_candidates[0]
    spacing = float(right["cx"] - left["cx"])

    if spacing <= 1:
        return None, None, None

    return left, right, spacing


# =========================================================
# CLASSIFY ARCHES
# =========================================================

def classify_arch_integrity(arch: dict, cfg: Config, img_h: int, img_w: int):
    """
    arch -> intact / fragmented / isolated_outlier_candidate
    """
    x1, y1, x2, y2 = arch["box"]
    damage = arch["damage_fraction"]

    # Üstteki büyük şüpheli kemer
    if (
        y1 <= img_h * cfg.top_outlier_y_ratio
        and arch["w"] >= img_w * cfg.huge_width_ratio
        and arch["h"] >= img_h * cfg.huge_height_ratio
    ):
        return "isolated_outlier_candidate"

    if damage <= cfg.intact_damage_max:
        return "intact"

    if cfg.fragmented_damage_min <= damage <= cfg.fragmented_damage_max:
        return "fragmented"

    return "uncertain"


def build_arch_features(arch: dict, pilasters: list[dict], img_h: int):
    left, right, support_spacing = nearest_pilaster_support_spacing(arch, pilasters)

    norm_width = arch["w"] / support_spacing if support_spacing and support_spacing > 1 else None
    norm_height = arch["h"] / support_spacing if support_spacing and support_spacing > 1 else None

    # springline olarak kabaca alt kot kullanıyoruz
    springline_y = arch["box"][3] / max(1, img_h)
    top_y = arch["box"][1] / max(1, img_h)
    center_y = arch["cy"] / max(1, img_h)

    return {
        "support_left_id": left["id"] if left else None,
        "support_right_id": right["id"] if right else None,
        "support_spacing_px": round(float(support_spacing), 4) if support_spacing else None,
        "norm_width_by_support": round(float(norm_width), 6) if norm_width else None,
        "norm_height_by_support": round(float(norm_height), 6) if norm_height else None,
        "springline_y_ratio": round(float(springline_y), 6),
        "top_y_ratio": round(float(top_y), 6),
        "center_y_ratio": round(float(center_y), 6),
    }


def arch_feature_distance(a_feat: dict, b_feat: dict) -> float:
    """
    Kemer aile benzerliği: support aralığına göre normalize edilmiş genişlik/yükseklik + springline
    """
    vals = []

    for key, weight in (
        ("norm_width_by_support", 1.0),
        ("norm_height_by_support", 0.9),
        ("springline_y_ratio", 0.7),
    ):
        av = a_feat.get(key)
        bv = b_feat.get(key)
        if av is not None and bv is not None:
            vals.append(abs(av - bv) * weight)

    if not vals:
        # support spacing yoksa kaba fallback
        ay = a_feat.get("center_y_ratio", 0.0)
        by = b_feat.get("center_y_ratio", 0.0)
        return abs(ay - by) * 2.0 + 0.4

    return float(sum(vals) / len(vals))


def cluster_intact_arches(intact_arches: list[dict], cfg: Config):
    families = []
    family_counter = 1

    for arch in intact_arches:
        assigned = False

        for fam in families:
            dist = arch_feature_distance(arch["features"], fam["prototype_features"])

            if dist <= cfg.arch_fragment_assign_tol:
                fam["members"].append(arch)
                assigned = True
                break

        if not assigned:
            families.append({
                "family_id": f"arch_family_{family_counter:02d}",
                "members": [arch],
                "prototype_features": arch["features"],
            })
            family_counter += 1

    # Aile prototiplerini güncelle
    for fam in families:
        fam["prototype_features"] = average_arch_features(fam["members"])

    return families


def average_arch_features(members: list[dict]) -> dict:
    keys = [
        "norm_width_by_support",
        "norm_height_by_support",
        "springline_y_ratio",
        "top_y_ratio",
        "center_y_ratio",
    ]
    out = {}
    for key in keys:
        vals = []
        for m in members:
            v = m["features"].get(key)
            if v is not None:
                vals.append(float(v))
        out[key] = round(float(np.mean(vals)), 6) if vals else None
    return out


def assign_fragmented_arches(fragmented_arches: list[dict], families: list[dict], cfg: Config):
    orphan = []

    for arch in fragmented_arches:
        best_idx = None
        best_dist = 1e9

        for i, fam in enumerate(families):
            dist = arch_feature_distance(arch["features"], fam["prototype_features"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None and best_dist <= cfg.arch_fragment_assign_tol:
            families[best_idx].setdefault("fragmented_members", []).append(arch)
        else:
            orphan.append(arch)

    return orphan


def finalize_arch_families(families: list[dict], outliers: list[dict], orphan_fragmented: list[dict]):
    final = []

    for fam in families:
        intact_members = fam.get("members", [])
        fragmented = fam.get("fragmented_members", [])

        final.append({
            "family_id": fam["family_id"],
            "family_type": "arch_family",
            "prototype_status": "intact_repeated_family",
            "prototype_features": fam["prototype_features"],
            "intact_count": len(intact_members),
            "fragmented_count": len(fragmented),
            "members_intact": [serialize_element_summary(x) for x in intact_members],
            "members_fragmented": [serialize_element_summary(x) for x in fragmented],
        })

    # outlier ailesi
    if outliers:
        final.append({
            "family_id": "arch_outlier_family",
            "family_type": "arch_family",
            "prototype_status": "isolated_outlier_candidate",
            "prototype_features": None,
            "intact_count": 0,
            "fragmented_count": 0,
            "members_intact": [],
            "members_fragmented": [],
            "members_outlier": [serialize_element_summary(x) for x in outliers],
        })

    # family dışı fragmented
    if orphan_fragmented:
        final.append({
            "family_id": "arch_fragmented_unassigned",
            "family_type": "arch_family",
            "prototype_status": "fragmented_arch_candidate",
            "prototype_features": None,
            "intact_count": 0,
            "fragmented_count": len(orphan_fragmented),
            "members_intact": [],
            "members_fragmented": [serialize_element_summary(x) for x in orphan_fragmented],
        })

    return final


# =========================================================
# PILASTER FAMILIES
# =========================================================

def classify_pilaster_integrity(p: dict, cfg: Config):
    damage = p["damage_fraction"]
    if damage <= cfg.intact_damage_max:
        return "intact"
    if damage <= cfg.fragmented_damage_max:
        return "fragmented"
    return "uncertain"


def cluster_pilasters(pilasters: list[dict], img_h: int, img_w: int, cfg: Config):
    # Bu örnekte çoğu pilaster tek ailede toplanabilir.
    # Yine de basit özelliklere göre kümeliyoruz.
    intact = []
    fragmented = []
    uncertain = []

    for p in pilasters:
        integrity = classify_pilaster_integrity(p, cfg)
        p["integrity"] = integrity
        p["features"] = {
            "width_ratio": round(p["w"] / max(1, img_w), 6),
            "height_ratio": round(p["h"] / max(1, img_h), 6),
            "center_y_ratio": round(p["cy"] / max(1, img_h), 6),
        }
        if integrity == "intact":
            intact.append(p)
        elif integrity == "fragmented":
            fragmented.append(p)
        else:
            uncertain.append(p)

    families = []
    counter = 1

    for p in intact:
        assigned = False
        for fam in families:
            proto = fam["prototype_features"]
            dist = pilaster_feature_distance(p["features"], proto)
            if dist <= 0.18:
                fam["members"].append(p)
                assigned = True
                break
        if not assigned:
            families.append({
                "family_id": f"pilaster_family_{counter:02d}",
                "members": [p],
                "prototype_features": p["features"],
            })
            counter += 1

    # prototipleri ortala
    for fam in families:
        fam["prototype_features"] = average_pilaster_features(fam["members"])

    # fragmented üyeleri ata
    orphan = []
    for p in fragmented:
        best_idx = None
        best_dist = 1e9
        for i, fam in enumerate(families):
            dist = pilaster_feature_distance(p["features"], fam["prototype_features"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist <= 0.22:
            families[best_idx].setdefault("fragmented_members", []).append(p)
        else:
            orphan.append(p)

    final = []

    for fam in families:
        final.append({
            "family_id": fam["family_id"],
            "family_type": "pilaster_family",
            "prototype_status": "intact_repeated_family",
            "prototype_features": fam["prototype_features"],
            "intact_count": len(fam.get("members", [])),
            "fragmented_count": len(fam.get("fragmented_members", [])),
            "members_intact": [serialize_element_summary(x) for x in fam.get("members", [])],
            "members_fragmented": [serialize_element_summary(x) for x in fam.get("fragmented_members", [])],
        })

    if orphan:
        final.append({
            "family_id": "pilaster_fragmented_unassigned",
            "family_type": "pilaster_family",
            "prototype_status": "fragmented_pilaster_candidate",
            "prototype_features": None,
            "intact_count": 0,
            "fragmented_count": len(orphan),
            "members_intact": [],
            "members_fragmented": [serialize_element_summary(x) for x in orphan],
        })

    return final, intact, fragmented, uncertain


def pilaster_feature_distance(a: dict, b: dict) -> float:
    vals = [
        abs(a["width_ratio"] - b["width_ratio"]) * 1.1,
        abs(a["height_ratio"] - b["height_ratio"]) * 0.9,
        abs(a["center_y_ratio"] - b["center_y_ratio"]) * 0.7,
    ]
    return float(sum(vals) / len(vals))


def average_pilaster_features(members: list[dict]):
    keys = ["width_ratio", "height_ratio", "center_y_ratio"]
    out = {}
    for k in keys:
        vals = [float(m["features"][k]) for m in members if k in m["features"]]
        out[k] = round(float(np.mean(vals)), 6) if vals else None
    return out


# =========================================================
# SERIALIZATION
# =========================================================

def serialize_element_summary(e: dict):
    return {
        "id": e["id"],
        "label": e["label"],
        "source_group": e["source_group"],
        "score": round(float(e["score"]), 6),
        "box": e["box"],
        "center": [round(float(e["cx"]), 3), round(float(e["cy"]), 3)],
        "size": [int(e["w"]), int(e["h"])],
        "damage_fraction": round(float(e["damage_fraction"]), 6),
        "integrity": e.get("integrity"),
        "features": e.get("features", {}),
    }


# =========================================================
# RHYTHM ANALYSIS
# =========================================================

def rhythm_analysis(arch_families: list[dict], pilaster_families: list[dict]):
    """
    Basit ritim özeti: yalnız intact family üyelerinden
    """
    analysis = {
        "arch_rhythm": [],
        "pilaster_rhythm": [],
    }

    for fam in arch_families:
        if fam.get("prototype_status") != "intact_repeated_family":
            continue

        xs = []
        for m in fam.get("members_intact", []):
            xs.append(float(m["center"][0]))
        xs = sorted(xs)

        spacings = [round(xs[i+1] - xs[i], 3) for i in range(len(xs)-1)]

        analysis["arch_rhythm"].append({
            "family_id": fam["family_id"],
            "member_count": len(xs),
            "centers_x": xs,
            "center_spacings_px": spacings,
            "mean_spacing_px": round(float(np.mean(spacings)), 6) if spacings else None,
        })

    for fam in pilaster_families:
        if fam.get("prototype_status") != "intact_repeated_family":
            continue

        xs = []
        for m in fam.get("members_intact", []):
            xs.append(float(m["center"][0]))
        xs = sorted(xs)

        spacings = [round(xs[i+1] - xs[i], 3) for i in range(len(xs)-1)]

        analysis["pilaster_rhythm"].append({
            "family_id": fam["family_id"],
            "member_count": len(xs),
            "centers_x": xs,
            "center_spacings_px": spacings,
            "mean_spacing_px": round(float(np.mean(spacings)), 6) if spacings else None,
        })

    return analysis


# =========================================================
# VISUALIZATION
# =========================================================

def build_member_family_map(families: list[dict]):
    mapping = {}
    outlier = set()
    fragmented_unassigned = set()

    for fi, fam in enumerate(families):
        fam_id = fam["family_id"]
        for m in fam.get("members_intact", []):
            mapping[m["id"]] = (fam_id, fi, "intact")
        for m in fam.get("members_fragmented", []):
            mapping[m["id"]] = (fam_id, fi, "fragmented")

        for m in fam.get("members_outlier", []):
            outlier.add(m["id"])

    return mapping, outlier, fragmented_unassigned


def draw_family_overlay(
    original: np.ndarray,
    arches: list[dict],
    pilasters: list[dict],
    openings: list[dict],
    arch_families: list[dict],
    pilaster_families: list[dict]
):
    img = original.copy()
    img = draw_header(img, "STAGE 3B-v2 EXISTING ELEMENT FAMILIES - PERSPECTIVE NORMALIZED / NO COMPLETION")

    arch_map, arch_outliers, _ = build_member_family_map(arch_families)
    pil_map, _, _ = build_member_family_map(pilaster_families)

    # openings sadece açık mavi ile
    for op in openings:
        x1, y1, x2, y2 = op["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 255), 2)
        cv2.putText(
            img, f"opening {op['id']}",
            (x1, max(48, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (80, 200, 255), 1, cv2.LINE_AA
        )

    # pilasterlar
    for p in pilasters:
        x1, y1, x2, y2 = p["box"]
        if p["id"] in pil_map:
            fam_id, fi, status = pil_map[p["id"]]
            color = family_color(fi + 3)
            if status == "fragmented":
                color = tuple(int(min(255, c + 25)) for c in color)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, f"{fam_id} {p['id']}",
                (x1, max(48, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA
            )

    # kemerler
    for a in arches:
        x1, y1, x2, y2 = a["box"]

        if a["id"] in arch_outliers:
            color = (180, 180, 180)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, f"{a['id']} isolated_outlier_candidate",
                (x1, max(48, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA
            )
            continue

        if a["id"] in arch_map:
            fam_id, fi, status = arch_map[a["id"]]
            color = family_color(fi)
            if status == "fragmented":
                color = tuple(int(min(255, c + 25)) for c in color)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, f"{fam_id} {a['id']}",
                (x1, max(48, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA
            )
        else:
            # family'ye girmeyen kemer
            color = (255, 120, 120)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, f"{a['id']} fragmented_arch_candidate",
                (x1, max(48, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA
            )

    return img


def draw_family_schematic(
    image_shape,
    arch_families: list[dict],
    pilaster_families: list[dict],
    openings: list[dict]
):
    h, w = image_shape[:2]
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    canvas = draw_header(canvas, "STAGE 3B-v2 FAMILY SCHEMATIC - EXISTING EVIDENCE ONLY")

    # Kemer aileleri
    fi = 0
    for fam in arch_families:
        color = family_color(fi)
        fi += 1

        for m in fam.get("members_intact", []):
            x1, y1, x2, y2 = m["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        for m in fam.get("members_fragmented", []):
            x1, y1, x2, y2 = m["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        for m in fam.get("members_outlier", []):
            x1, y1, x2, y2 = m["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (170, 170, 170), 2)

    # Pilaster aileleri
    fi = 3
    for fam in pilaster_families:
        color = family_color(fi)
        fi += 1

        for m in fam.get("members_intact", []):
            x1, y1, x2, y2 = m["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        for m in fam.get("members_fragmented", []):
            x1, y1, x2, y2 = m["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    # openings
    for op in openings:
        x1, y1, x2, y2 = op["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 200, 255), 1)

    return canvas


# =========================================================
# MAIN RUN
# =========================================================

def run(cfg: Config) -> Path:
    parent = find_parent_run(cfg)
    run_dir = create_run_dir(cfg.output_root)

    started = time.perf_counter()

    meta = {
        "run_id": run_dir.name,
        "stage": "v8_stage3b_element_families_v2",
        "status": "running",
        "started_at": now_iso(),
        "parent_run": str(parent.resolve()),
        "config": asdict(cfg),
        "safety": {
            "new_objects": 0,
            "missing_inference": 0,
            "new_axes": 0,
        }
    }

    try:
        copy_required_inputs(parent, run_dir, cfg)

        original = load_rgb(run_dir / "input.png")
        h, w = original.shape[:2]
        damage_mask = load_mask(run_dir / "completion_mask.png", (w, h))
        axes = load_axes(run_dir)

        raw_items, duplicate_count = load_raw_elements(run_dir, damage_mask, w, h)
        arches, pilasters, openings, others = split_elements(raw_items, cfg)

        # Arch integrity / features
        arch_outliers = []
        arch_intact = []
        arch_fragmented = []
        arch_uncertain = []

        for a in arches:
            a["integrity"] = classify_arch_integrity(a, cfg, h, w)
            a["features"] = build_arch_features(a, pilasters, h)

            if a["integrity"] == "isolated_outlier_candidate":
                arch_outliers.append(a)
            elif a["integrity"] == "intact":
                arch_intact.append(a)
            elif a["integrity"] == "fragmented":
                arch_fragmented.append(a)
            else:
                arch_uncertain.append(a)

        arch_family_core = cluster_intact_arches(arch_intact, cfg)
        orphan_fragmented = assign_fragmented_arches(arch_fragmented, arch_family_core, cfg)
        orphan_fragmented.extend(arch_uncertain)
        arch_families = finalize_arch_families(arch_family_core, arch_outliers, orphan_fragmented)

        pilaster_families, pil_intact, pil_fragmented, pil_uncertain = cluster_pilasters(pilasters, h, w, cfg)

        rhythm = rhythm_analysis(arch_families, pilaster_families)

        # Visuals
        overlay = draw_family_overlay(
            original=original,
            arches=arches,
            pilasters=pilasters,
            openings=openings,
            arch_families=arch_families,
            pilaster_families=pilaster_families,
        )

        schematic = draw_family_schematic(
            image_shape=original.shape,
            arch_families=arch_families,
            pilaster_families=pilaster_families,
            openings=openings,
        )

        save_rgb(run_dir / "stage3b_v2_element_families.png", overlay)
        save_rgb(run_dir / "stage3b_v2_family_schematic.png", schematic)

        write_json(run_dir / "stage3b_v2_arch_families.json", {
            "run_id": run_dir.name,
            "parent_run": str(parent.resolve()),
            "families": arch_families,
        })

        write_json(run_dir / "stage3b_v2_pilaster_families.json", {
            "run_id": run_dir.name,
            "parent_run": str(parent.resolve()),
            "families": pilaster_families,
        })

        write_json(run_dir / "stage3b_v2_rhythm_analysis.json", {
            "run_id": run_dir.name,
            "parent_run": str(parent.resolve()),
            "axes_count": len(axes),
            "arch_family_count": sum(1 for f in arch_families if f["prototype_status"] == "intact_repeated_family"),
            "pilaster_family_count": sum(1 for f in pilaster_families if f["prototype_status"] == "intact_repeated_family"),
            "rhythm": rhythm,
        })

        # Report
        arch_family_count = sum(1 for f in arch_families if f["prototype_status"] == "intact_repeated_family")
        pilaster_family_count = sum(1 for f in pilaster_families if f["prototype_status"] == "intact_repeated_family")
        opening_family_count = 1 if openings else 0

        report_lines = [
            "STAGE 3B-v2 — PERSPECTIVE NORMALIZED ELEMENT FAMILY REPORT",
            f"Run: {run_dir.name}",
            f"Parent: {parent.resolve()}",
            "",
            "SAFE RULES:",
            "- New architectural object creation: 0",
            "- Missing inference creation: 0",
            "- New rhythm axis creation: 0",
            "",
            f"Raw source elements: {len(raw_items)}",
            f"Duplicates removed: {duplicate_count}",
            f"Arch candidates: {len(arches)}",
            f"Pilaster candidates: {len(pilasters)}",
            f"Opening candidates: {len(openings)}",
            "",
            f"Arch families (intact repeated): {arch_family_count}",
            f"Pilaster families (intact repeated): {pilaster_family_count}",
            f"Opening families (reported only): {opening_family_count}",
            "",
            "ARCH FAMILY DETAILS:",
        ]

        for fam in arch_families:
            report_lines.append(
                f"- {fam['family_id']} | status={fam['prototype_status']} | "
                f"intact={fam.get('intact_count', 0)} | fragmented={fam.get('fragmented_count', 0)}"
            )

        report_lines.append("")
        report_lines.append("PILASTER FAMILY DETAILS:")
        for fam in pilaster_families:
            report_lines.append(
                f"- {fam['family_id']} | status={fam['prototype_status']} | "
                f"intact={fam.get('intact_count', 0)} | fragmented={fam.get('fragmented_count', 0)}"
            )

        report_lines.append("")
        report_lines.append("INTERPRETATION:")
        report_lines.append("- Side arches should tend to merge under one perspective-normalized family when support spacing is similar.")
        report_lines.append("- Central heavily damaged arch should remain fragmented_arch_candidate unless it has strong intact family similarity.")
        report_lines.append("- Upper huge arch-like detection should remain isolated_outlier_candidate and must not define the prototype family.")

        (run_dir / "stage3b_v2_family_report.txt").write_text(
            "\n".join(report_lines),
            encoding="utf-8"
        )

        meta["status"] = "completed"
        meta["finished_at"] = now_iso()
        meta["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        meta["summary"] = {
            "source_elements": len(raw_items),
            "duplicate_removed": duplicate_count,
            "arches": len(arches),
            "pilasters": len(pilasters),
            "openings": len(openings),
            "arch_family_count": arch_family_count,
            "pilaster_family_count": pilaster_family_count,
            "opening_family_count": opening_family_count,
            "new_objects": 0,
            "missing_inference": 0,
        }

        write_json(run_dir / "run_meta.json", meta)

        log_lines = [
            "=" * 78,
            f"[RUN] {run_dir.name}",
            f"[PARENT] {parent.resolve()}",
            "[NEW RUN] Önceki klasörün üzerine yazılmayacak",
            "[SAFE] Yeni mimari eleman veya eksen oluşturulmayacak",
            "[STAGE] Perspective-normalized kemer/pilaster aileleri ve ritim analizi",
            "=" * 78,
            f"[ELEMENTS] kaynak={len(raw_items)}, benzersiz={len(raw_items)}, yinelenen={duplicate_count}",
            f"[FAMILIES] arch={arch_family_count}, pilaster={pilaster_family_count}, opening={opening_family_count}",
            "[OK] Stage 3B-v2 tamamlandı",
            f"[OUT] {run_dir.resolve()}",
            f"[COUNT] arch_family={arch_family_count}, pilaster_family={pilaster_family_count}, new_objects=0, missing_inference=0",
            "[FILES] stage3b_v2_element_families.png, stage3b_v2_arch_families.json, stage3b_v2_rhythm_analysis.json",
        ]
        (run_dir / "run.log").write_text("\n".join(log_lines), encoding="utf-8")

        print(f"\nBİTTİ: {run_dir.resolve()}")
        return run_dir

    except Exception as e:
        meta["status"] = "failed"
        meta["finished_at"] = now_iso()
        meta["error"] = {
            "type": e.__class__.__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "run_meta.json", meta)
        raise


def main():
    cfg = Config()
    run(cfg)


if __name__ == "__main__":
    main()