"""Step 3 — damage: label each element intact or fragmented and split the damage mask into regions.

Input:
    data/structures/<id>/damage_mask.png   white = damaged or missing (draw with tools/mask_editor.py)
    elements.json and element_masks/       from detect / review (or tools/editor_adapter.py)

Rule (docs/ARCHITECTURE.md §5.1): for every element that is not rejected,
    overlap = |element mask AND damage mask| / |element mask|
    overlap <  damage.intact_max_overlap  ->  "intact"
    overlap >= damage.intact_max_overlap  ->  "fragmented"
Rejected elements get no status. A damage region is an 8-connected component of the damage mask;
components smaller than damage.min_region_px are listed as ignored, not as regions.

Output in the run folder:
    damage_mask.png            copy of the mask used (so later steps read only the run folder)
    elements.json              each active element gains damage_status, damage_overlap, damage_box_overlap
    damage_regions.json        regions, per-region element overlap, uncovered pixels, summary
    damage_regions/<id>.png    one 0/255 mask per region
    damage_overlay.png         damage tinted, intact elements green, fragmented elements red

``damage_box_overlap`` (the same share for the element's box) is recorded for diagnosis only. A SAM
mask covers the visible part of an element, so a cut element can have a small mask overlap and a large
box overlap; the label uses the mask, as §5.1 says.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from i2r.detect import ELEMENTS_FILE, load_elements, validate_elements
from i2r.runio import ContractError, write_json

if TYPE_CHECKING:
    import argparse

    from i2r.runio import Run

REGIONS_FILE = "damage_regions.json"
REGION_DIR = "damage_regions"
MASK_COPY = "damage_mask.png"
OVERLAY = "damage_overlay.png"

INTACT = "intact"
FRAGMENTED = "fragmented"

GREEN = (40, 220, 90)
RED = (240, 50, 50)
DAMAGE_TINT = np.array([255.0, 40.0, 40.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pure functions (tested without files)
# ---------------------------------------------------------------------------

def load_binary_mask(path: Path, size: tuple[int, int], what: str) -> np.ndarray:
    """0/1 uint8 mask of the given (width, height); ContractError if missing or of another size."""
    if not path.is_file():
        raise ContractError(f"{what} not found: {path}")
    mask = Image.open(path).convert("L")
    if mask.size != tuple(size):
        raise ContractError(f"{path}: size {mask.size} differs from the image size {tuple(size)}")
    return (np.asarray(mask) >= 128).astype(np.uint8)


def check_damage_mask(damage: np.ndarray, min_fraction: float, path: Path) -> float:
    fraction = float(damage.mean())
    if fraction < min_fraction:
        raise ContractError(
            f"{path}: damage mask covers {fraction * 100:.3f} % of the image, below "
            f"damage.min_mask_fraction ({min_fraction * 100:.3f} %). Draw it with "
            "`python tools/mask_editor.py --structure <id>`."
        )
    return fraction


def overlap_share(region: np.ndarray, damage: np.ndarray) -> float:
    """Share of ``region``'s pixels that lie in ``damage``. ContractError if ``region`` is empty."""
    area = int(region.sum())
    if area == 0:
        raise ContractError("element mask is empty")
    return float(np.logical_and(region > 0, damage > 0).sum()) / area


def box_mask(box: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = 1
    return out


def label(overlap: float, intact_max_overlap: float) -> str:
    return INTACT if overlap < intact_max_overlap else FRAGMENTED


def split_regions(damage: np.ndarray, min_region_px: int) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, int]]:
    """8-connected components, largest first. Returns (records, masks, ignored summary)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(damage.astype(np.uint8), connectivity=8)
    kept = []
    ignored = {"count": 0, "pixels": 0}
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_region_px:
            ignored["count"] += 1
            ignored["pixels"] += area
            continue
        kept.append((area, i, [x, y, x + w, y + h], centroids[i]))
    kept.sort(key=lambda item: -item[0])
    records, masks = [], []
    for number, (area, i, box, centroid) in enumerate(kept, start=1):
        records.append({
            "id": f"region_{number:02d}",
            "box": box,
            "area_px": area,
            "centroid": [round(float(centroid[0]), 1), round(float(centroid[1]), 1)],
        })
        masks.append((labels == i).astype(np.uint8))
    return records, masks, ignored


def assess_elements(
    elements: list[dict[str, Any]],
    element_masks: dict[str, np.ndarray],
    damage: np.ndarray,
    intact_max_overlap: float,
) -> dict[str, int]:
    """Write damage_status / damage_overlap / damage_box_overlap into each element (in place)."""
    counts = {INTACT: 0, FRAGMENTED: 0, "rejected_skipped": 0}
    for element in elements:
        if element.get("rejected"):
            for key in ("damage_status", "damage_overlap", "damage_box_overlap"):
                element.pop(key, None)
            counts["rejected_skipped"] += 1
            continue
        overlap = overlap_share(element_masks[element["id"]], damage)
        box_overlap = overlap_share(box_mask(element["box"], damage.shape), damage)
        element["damage_status"] = label(overlap, intact_max_overlap)
        element["damage_overlap"] = round(overlap, 4)
        element["damage_box_overlap"] = round(box_overlap, 4)
        counts[element["damage_status"]] += 1
    return counts


def region_contents(
    regions: list[dict[str, Any]],
    region_masks: list[np.ndarray],
    elements: list[dict[str, Any]],
    element_masks: dict[str, np.ndarray],
) -> None:
    """Per region: which active elements' masks it touches, and how many pixels no element covers."""
    active = [e for e in elements if not e.get("rejected")]
    for record, rmask in zip(regions, region_masks):
        covered = np.zeros_like(rmask, dtype=bool)
        touching = []
        for element in active:
            emask = element_masks[element["id"]] > 0
            inter = int(np.logical_and(rmask > 0, emask).sum())
            if inter:
                touching.append({
                    "id": element["id"],
                    "class": element["class"],
                    "overlap_px": inter,
                    "share_of_region": round(inter / record["area_px"], 4),
                })
                covered |= emask
        touching.sort(key=lambda t: -t["overlap_px"])
        record["elements"] = touching
        record["uncovered_px"] = int(np.logical_and(rmask > 0, ~covered).sum())


def expert_agreement(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare damage_status with expert_condition (set by tools/editor_adapter.py) where both exist."""
    rows = [(e["id"], e["expert_condition"], e["damage_status"]) for e in elements
            if not e.get("rejected") and e.get("expert_condition") in (INTACT, FRAGMENTED)]
    disagreements = [{"id": i, "expert": x, "computed": c} for i, x, c in rows if x != c]
    return {
        "compared": len(rows),
        "agree": len(rows) - len(disagreements),
        "disagreements": disagreements,
    }


def draw_overlay(rgb: np.ndarray, damage: np.ndarray, elements: list[dict[str, Any]]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    sel = damage > 0
    out[sel] = out[sel] * 0.5 + DAMAGE_TINT * 0.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    for element in elements:
        status = element.get("damage_status")
        if status is None:
            continue
        color = GREEN if status == INTACT else RED
        x1, y1, x2, y2 = element["box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
        text = f"{element['id']} {element['damage_overlap']:.2f}"
        cv2.putText(out, text, (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def run_step(run: "Run", opts: "argparse.Namespace") -> None:
    log = run.logger
    dcfg = run.config["damage"]
    intact_max_overlap = float(dcfg["intact_max_overlap"])

    doc = load_elements(run.run_dir)
    size = tuple(doc["image_size"])
    damage = load_binary_mask(
        run.damage_mask_path, size,
        "Damage mask (draw it with `python tools/mask_editor.py --structure "
        f"{run.structure_id}`)",
    )
    fraction = check_damage_mask(damage, float(dcfg["min_mask_fraction"]), run.damage_mask_path)
    shutil.copy2(run.damage_mask_path, run.path(MASK_COPY))

    elements = doc["elements"]
    element_masks: dict[str, np.ndarray] = {}
    for element in elements:
        if element.get("rejected"):
            continue
        path = run.path(element["mask_file"])
        element_masks[element["id"]] = load_binary_mask(path, size, f"Mask of element {element['id']}")
        if not element_masks[element["id"]].any():
            raise ContractError(f"{path}: mask of element {element['id']} is empty")

    counts = assess_elements(elements, element_masks, damage, intact_max_overlap)
    regions, region_masks, ignored = split_regions(damage, int(dcfg["min_region_px"]))
    region_contents(regions, region_masks, elements, element_masks)

    region_dir = run.path(REGION_DIR)
    if region_dir.exists():
        shutil.rmtree(region_dir)
    region_dir.mkdir()
    for record, rmask in zip(regions, region_masks):
        record["mask_file"] = f"{REGION_DIR}/{record['id']}.png"
        Image.fromarray(rmask * 255, mode="L").save(run.path(record["mask_file"]))

    agreement = expert_agreement(elements)
    mask_kinds = sorted({e.get("mask_kind", "segmentation") for e in elements if not e.get("rejected")})
    summary = {
        "damage_fraction": round(fraction, 6),
        "regions": len(regions),
        "ignored_small_regions": ignored,
        "elements": counts,
        "element_mask_kinds": mask_kinds,
        "expert_agreement": agreement,
    }
    write_json(run.path(REGIONS_FILE), {
        "structure_id": run.structure_id,
        "image_size": list(size),
        "damage_mask": {"file": MASK_COPY, "sha256": sha256_file(run.path(MASK_COPY))},
        "settings": {
            "intact_max_overlap": intact_max_overlap,
            "min_mask_fraction": float(dcfg["min_mask_fraction"]),
            "min_region_px": int(dcfg["min_region_px"]),
        },
        "summary": summary,
        "regions": regions,
    })

    doc.setdefault("summary", {})["damage"] = counts
    validate_elements(doc)
    write_json(run.path(ELEMENTS_FILE), doc)

    rgb = np.asarray(Image.open(run.image_path).convert("RGB"))
    Image.fromarray(draw_overlay(rgb, damage, elements)).save(run.path(OVERLAY))

    log.info("[damage] mask covers %.1f %% of the image; %d region(s), %d small one(s) ignored",
             fraction * 100, len(regions), ignored["count"])
    log.info("[damage] elements: %d intact, %d fragmented (%d rejected skipped)",
             counts[INTACT], counts[FRAGMENTED], counts["rejected_skipped"])
    if "box" in mask_kinds:
        log.warning("[damage] some element masks are boxes (editor import); overlap is overstated for them")
    if agreement["compared"]:
        log.info("[damage] agreement with expert_condition: %d / %d", agreement["agree"], agreement["compared"])
        for row in agreement["disagreements"]:
            log.info("[damage]   %s: expert %s, computed %s", row["id"], row["expert"], row["computed"])
    run.meta["steps"].setdefault("damage", {})["summary"] = {
        k: summary[k] for k in ("damage_fraction", "regions", "elements")
    }
