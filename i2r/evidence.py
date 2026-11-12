"""Step 4 — evidence: one record per target, each with exactly one tier (docs/ARCHITECTURE.md §5.2).

Tier A, for each element labelled ``fragmented`` by the damage step (§5.1):

For each element labelled ``fragmented`` by the damage step (docs/ARCHITECTURE.md §5.1, §5.2):

1. Candidate donors pass a list of filters (``DONOR_FILTERS``): same class, intact, not rejected, not
   the target itself, width and height within ``evidence.size_tolerance``.
2. Each candidate is scored by a list of scorers (``SCORERS``; today one: DINOv2-small cosine
   similarity). The target crop has its damaged pixels replaced by the mean of its visible pixels. The
   donor crop is resized to the target box, optionally flipped, and with
   ``evidence.mask_donor_like_target`` the same pixels are hidden on it, so both are compared on the
   part of the target that survives.
3. The best candidate (flipped or not) at or above ``evidence.min_similarity`` makes the target tier A,
   with the next ``evidence.alternatives_kept`` as alternatives. Otherwise the target is tier C with the
   reason. Confidence for tier A is the similarity (§5.2). A target whose visible share of its box
   is below ``evidence.min_visible_fraction`` is tier C without scoring: its crops would be mostly
   fill colour on both sides, and the similarity would measure the fill.

Tier B, implied slots with no visible fragment (geometry only, ``TIER_B_RULES``):
    B1  a gap between two observed same-class neighbours of a row (at least ``min_row_size``), about
        k >= 2 whole spacings wide; each implied slot must lie in the damage mask
        (``slot_min_damage``) and be free of same-class elements (``slot_free_max_overlap``). The donor
        is the nearest intact row member. Confidence = row regularity x donor score. Never beyond a
        row's end.
    B2  the mirror of an intact element across the facade's vertical axis (``symmetry_axis_x`` from
        structure.yaml, else estimated from same-class pairs) under the same slot conditions; the donor
        is flipped. Confidence = axis agreement x donor score. Switched off, and logged, when the
        agreement is below ``symmetry_min_agreement``.
Tier C: every damage pixel outside all A and B target boxes, as connected regions of at least
``damage.min_region_px``, confidence 0. Fragmented elements without a donor are also tier C records.

This module never edits elements.json.

Output in the run folder:
    evidence.json          one record per target, every candidate with its score or rejection reason
    evidence_c_regions/    one 0/255 mask per tier C region
    evidence_overlay.png   donor -> target arrows (green A, blue B), tier C hatched grey
    evidence_sheet.png     per A or B target: target crop, chosen donor as transferred, alternatives
"""
from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image

from i2r.detect import load_elements
from i2r.runio import ContractError, read_json, write_json

if TYPE_CHECKING:
    import argparse

    from i2r.runio import Run

EVIDENCE_FILE = "evidence.json"
OVERLAY = "evidence_overlay.png"
SHEET = "evidence_sheet.png"
DAMAGE_MASK = "damage_mask.png"      # copy written by the damage step

TIER_A, TIER_B, TIER_C = "A", "B", "C"
C_REGION_DIR = "evidence_c_regions"

Embed = Callable[[list[np.ndarray]], np.ndarray]


# ---------------------------------------------------------------------------
# Donor filters: (target, donor, settings) -> rejection reason or None
# ---------------------------------------------------------------------------

def filter_not_self(target: dict, donor: dict, ecfg: dict) -> Optional[str]:
    return "same element" if donor["id"] == target["id"] else None


def filter_same_class(target: dict, donor: dict, ecfg: dict) -> Optional[str]:
    return None if donor["class"] == target["class"] else "other class"


def filter_usable(target: dict, donor: dict, ecfg: dict) -> Optional[str]:
    if donor.get("rejected"):
        return "rejected"
    if donor.get("damage_status") != "intact":
        return f"not intact ({donor.get('damage_status')})"
    return None


def size_ratios(target_box: list[int], donor_box: list[int]) -> tuple[float, float]:
    tw, th = target_box[2] - target_box[0], target_box[3] - target_box[1]
    dw, dh = donor_box[2] - donor_box[0], donor_box[3] - donor_box[1]
    return dw / tw, dh / th


def filter_size(target: dict, donor: dict, ecfg: dict) -> Optional[str]:
    tol = float(ecfg["size_tolerance"])
    rw, rh = size_ratios(target["box"], donor["box"])
    if abs(rw - 1.0) > tol or abs(rh - 1.0) > tol:
        return f"size ratio w {rw:.2f}, h {rh:.2f} outside 1 +/- {tol:.2f}"
    return None


# Order matters only for the reason that is reported (the first failing filter).
DONOR_FILTERS: list[Callable[[dict, dict, dict], Optional[str]]] = [
    filter_not_self, filter_same_class, filter_usable, filter_size,
]


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------

def crop(array: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return array[y1:y2, x1:x2]


def fill_hidden(rgb: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """Replace hidden pixels by the mean colour of the visible ones."""
    out = rgb.copy()
    visible = ~hidden
    if hidden.any() and visible.any():
        out[hidden] = rgb[visible].mean(axis=0).astype(np.uint8)
    return out


def target_crop(rgb: np.ndarray, damage: np.ndarray, box: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(crop with damaged pixels filled, boolean hole mask) of the target box."""
    hole = crop(damage, box) > 0
    return fill_hidden(crop(rgb, box), hole), hole


def donor_crop(
    rgb: np.ndarray, damage: np.ndarray, donor_box: list[int], target_box: list[int],
    hole: np.ndarray | None, flipped: bool,
) -> np.ndarray:
    """Donor resized to the target box (as it would be transferred), optionally flipped, with its own
    damaged pixels and, if ``hole`` is given, the target's hole filled."""
    tw, th = target_box[2] - target_box[0], target_box[3] - target_box[1]
    patch = cv2.resize(crop(rgb, donor_box), (tw, th), interpolation=cv2.INTER_AREA)
    own = cv2.resize(crop(damage, donor_box), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    if flipped:
        patch, own = patch[:, ::-1].copy(), own[:, ::-1].copy()
    hidden = own | hole if hole is not None else own
    return fill_hidden(patch, hidden)


def to_model_size(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    donor: dict
    flipped: bool
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def similarity(self) -> float:
        return self.scores["dinov2_cosine"]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def score_dinov2_cosine(target_vec: np.ndarray, donor_vec: np.ndarray) -> float:
    return cosine(target_vec, donor_vec)


# name -> function(target embedding, donor embedding) -> score. A learned scorer is one more entry
# (docs/ARCHITECTURE.md §9). The tier decision uses "dinov2_cosine".
SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "dinov2_cosine": score_dinov2_cosine,
}


# ---------------------------------------------------------------------------
# Evidence (pure: image arrays and element records in, records out)
# ---------------------------------------------------------------------------

def no_donor_reason(element_class: str, rejected: list[dict[str, str]]) -> str:
    """Why no same-class donor is left, e.g. 'no usable donor of class 'arch': 2 not intact, 1 size'."""
    if not any(r["reason"].startswith("size ratio") for r in rejected):
        return f"no intact element of class {element_class!r}"
    counts: dict[str, int] = {}
    for r in rejected:
        key = "size" if r["reason"].startswith("size ratio") else r["reason"].split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
    parts = ", ".join(f"{n} {k}" for k, n in counts.items())
    return f"no usable donor of class {element_class!r}: {parts}"


def find_tier_a(
    elements: list[dict[str, Any]],
    rgb: np.ndarray,
    damage: np.ndarray,
    ecfg: dict[str, Any],
    embed: Embed,
) -> list[dict[str, Any]]:
    """One record per fragmented element, in element order."""
    size = int(ecfg["crop_px"])
    targets = [e for e in elements if not e.get("rejected") and e.get("damage_status") == "fragmented"]

    # 1. Filter donors and collect every crop to embed, so the model runs in a few batches.
    crops: list[np.ndarray] = []
    plans = []
    for target in targets:
        t_img, hole = target_crop(rgb, damage, target["box"])
        visible = float((~hole).mean())
        plan = {"target": target, "visible": visible, "rejected": [], "pairs": [], "t_index": None}
        if visible < float(ecfg["min_visible_fraction"]) or visible == 0.0:
            plans.append(plan)
            continue
        plan["t_index"] = len(crops)
        crops.append(to_model_size(t_img, size))
        donor_hole = hole if ecfg["mask_donor_like_target"] else None
        for donor in elements:
            reason = next((r for f in DONOR_FILTERS if (r := f(target, donor, ecfg))), None)
            if reason is not None:
                if donor["class"] == target["class"] and donor["id"] != target["id"]:
                    plan["rejected"].append({"donor_id": donor["id"], "reason": reason})
                continue
            for flipped in ([False, True] if ecfg["try_flipped_donor"] else [False]):
                d_img = donor_crop(rgb, damage, donor["box"], target["box"], donor_hole, flipped)
                plan["pairs"].append((Candidate(donor, flipped), len(crops)))
                crops.append(to_model_size(d_img, size))
        plans.append(plan)

    vectors = embed(crops) if crops else np.zeros((0, 1))
    if len(vectors) != len(crops):
        raise ContractError(f"embedding returned {len(vectors)} vectors for {len(crops)} crops")

    # 2. Score, rank, decide.
    threshold = float(ecfg["min_similarity"])
    keep = int(ecfg["alternatives_kept"])
    records = []
    for number, plan in enumerate(plans, start=1):
        target = plan["target"]
        candidates = []
        for cand, index in plan["pairs"]:
            cand.scores = {name: round(fn(vectors[plan["t_index"]], vectors[index]), 4)
                           for name, fn in SCORERS.items()}
            candidates.append(cand)
        # best orientation per donor, then rank donors
        best_per_donor: dict[str, Candidate] = {}
        for cand in candidates:
            current = best_per_donor.get(cand.donor["id"])
            if current is None or cand.similarity > current.similarity:
                best_per_donor[cand.donor["id"]] = cand
        ranked = sorted(best_per_donor.values(), key=lambda c: -c.similarity)

        record: dict[str, Any] = {
            "target_id": f"t_{number:03d}",
            "kind": "fragmented_element",
            "element_id": target["id"],
            "class": target["class"],
            "target_box": target["box"],
            "visible_fraction": round(plan["visible"], 4),
        }
        if plan["t_index"] is None:
            reason = (f"visible part {plan['visible']:.2f} of the box below min_visible_fraction "
                      f"{float(ecfg['min_visible_fraction']):.2f}")
        elif not ranked:
            reason = no_donor_reason(target["class"], plan["rejected"])
        elif ranked[0].similarity < threshold:
            reason = f"best similarity {ranked[0].similarity:.3f} below min_similarity {threshold:.2f}"
        else:
            reason = None

        best = ranked[0] if ranked else None
        record.update({
            "tier": TIER_A if reason is None else TIER_C,
            "reason": reason,
            "donor_id": best.donor["id"] if best and reason is None else None,
            "donor_box": best.donor["box"] if best and reason is None else None,
            "flipped": best.flipped if best and reason is None else None,
            "similarity": best.similarity if best else None,
            "confidence": best.similarity if best and reason is None else 0.0,
            "alternatives": [
                {"donor_id": c.donor["id"], "similarity": c.similarity, "flipped": c.flipped}
                for c in ranked[1:1 + keep]
            ] if reason is None else [],
            "donor_expert_prototype": (
                (best.donor.get("editor_record") or {}).get("usable_as_prototype")
                if best and reason is None else None
            ),
            "candidates": [
                {"donor_id": c.donor["id"], "flipped": c.flipped, "scores": c.scores}
                for c in sorted(candidates, key=lambda c: -c.similarity)
            ],
            "rejected_donors": plan["rejected"],
        })
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Tier B (geometry only, no model)
# ---------------------------------------------------------------------------

def box_size(box: list[int]) -> tuple[int, int]:
    return box[2] - box[0], box[3] - box[1]


def box_centre(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_from_centre(cx: float, cy: float, w: float, h: float, width: int, height: int,
                    min_inside: float) -> list[int] | None:
    """Integer box clipped to the image; None when less than ``min_inside`` of its area lies inside
    (``evidence.tier_b.slot_min_inside_image``)."""
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    cx1, cy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    cx2, cy2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < min_inside * w * h:
        return None
    return [cx1, cy1, cx2, cy2]


def intersection(a: list[int], b: list[int]) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def iou(a: list[int], b: list[int]) -> float:
    inter = intersection(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def damage_share(box: list[int], damage: np.ndarray) -> float:
    region = crop(damage, box)
    return float(region.mean()) if region.size else 0.0


def slot_is_free(box: list[int], element_class: str, active: list[dict], max_overlap: float) -> bool:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return all(intersection(box, e["box"]) / area <= max_overlap
               for e in active if e["class"] == element_class)


def active_elements(elements: list[dict]) -> list[dict]:
    return [e for e in elements if not e.get("rejected") and e.get("damage_status") in ("intact", "fragmented")]


def group_rows(elements: list[dict], tol: float) -> list[list[dict]]:
    """Same-class elements whose vertical centres and heights agree within ``tol`` (relative to the
    row's mean height), each row sorted left to right."""
    rows: list[list[dict]] = []
    for e in sorted(elements, key=lambda e: (e["class"], box_centre(e["box"])[1])):
        _, cy = box_centre(e["box"])
        _, h = box_size(e["box"])
        for row in rows:
            if row[0]["class"] != e["class"]:
                continue
            mean_h = float(np.mean([box_size(r["box"])[1] for r in row]))
            mean_cy = float(np.mean([box_centre(r["box"])[1] for r in row]))
            if abs(cy - mean_cy) <= tol * mean_h and abs(h - mean_h) <= tol * mean_h:
                row.append(e)
                break
        else:
            rows.append([e])
    return [sorted(row, key=lambda e: box_centre(e["box"])[0]) for row in rows]


def row_spacing(row: list[dict], tol: float) -> tuple[float, list[float]] | None:
    """Unit spacing of a row: the median of the centre distances that are close to the smallest one."""
    xs = [box_centre(e["box"])[0] for e in row]
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0]
    if not gaps:
        return None
    smallest = min(gaps)
    unit = float(np.median([g for g in gaps if g <= smallest * (1 + tol)]))
    return unit, gaps


def tier_b1_rows(elements: list[dict], damage: np.ndarray, bcfg: dict) -> tuple[list[dict], dict]:
    """B1: a gap between two observed same-class neighbours, about k whole spacings wide (k >= 2),
    whose implied slots lie in the damage mask with no element. Only between neighbours, never beyond
    a row's end."""
    tol = float(bcfg["row_tolerance"])
    height, width = damage.shape
    active = active_elements(elements)
    targets, rows_info = [], []
    for row in group_rows(active, tol):
        info = {"class": row[0]["class"], "elements": [e["id"] for e in row]}
        rows_info.append(info)
        if len(row) < int(bcfg["min_row_size"]):
            info["skipped"] = f"fewer than {bcfg['min_row_size']} elements"
            continue
        spacing = row_spacing(row, tol)
        if spacing is None:
            info["skipped"] = "no positive spacing"
            continue
        unit, gaps = spacing
        multiples = [g / unit for g in gaps]
        deviation = [abs(m - round(m)) for m in multiples]
        regularity = max(0.0, 1.0 - 2.0 * float(np.mean(deviation)))  # 1 = exact multiples, 0 = half-way
        info.update({"spacing_px": round(unit, 1), "gap_multiples": [round(m, 2) for m in multiples],
                     "regularity": round(regularity, 3)})
        intact = [e for e in row if e["damage_status"] == "intact"]
        for left, right, m, dev in zip(row, row[1:], multiples, deviation):
            k = int(round(m))
            if k < 2 or dev > tol:
                continue
            (lx, ly), (rx, ry) = box_centre(left["box"]), box_centre(right["box"])
            w = (box_size(left["box"])[0] + box_size(right["box"])[0]) / 2
            h = (box_size(left["box"])[1] + box_size(right["box"])[1]) / 2
            for j in range(1, k):
                cx, cy = lx + (rx - lx) * j / k, ly + (ry - ly) * j / k
                slot = box_from_centre(cx, cy, w, h, width, height, float(bcfg["slot_min_inside_image"]))
                if slot is None:
                    continue
                if damage_share(slot, damage) < float(bcfg["slot_min_damage"]):
                    continue
                if not slot_is_free(slot, left["class"], active, float(bcfg["slot_free_max_overlap"])):
                    continue
                if not intact:
                    continue
                donor = min(intact, key=lambda e: abs(box_centre(e["box"])[0] - cx))
                targets.append(implied_target("B1", left["class"], slot, donor, False,
                                              regularity * float(donor["score"]),
                                              {"row": [e["id"] for e in row], "between": [left["id"], right["id"]],
                                               "spacing_px": round(unit, 1), "regularity": round(regularity, 3)}))
    return targets, {"rows": rows_info}


def mirror_box(box: list[int], axis_x: float, width: int, height: int, min_inside: float) -> list[int] | None:
    w, h = box_size(box)
    cx, cy = box_centre(box)
    return box_from_centre(2 * axis_x - cx, cy, w, h, width, height, min_inside)


def axis_agreement(axis_x: float, active: list[dict], damage: np.ndarray, bcfg: dict) -> tuple[float, int]:
    """Share of checkable elements whose mirror matches a same-class element. An element is checkable
    when its mirror lies in the image and mostly outside the damage mask."""
    height, width = damage.shape
    checked = matched = 0
    for e in active:
        m = mirror_box(e["box"], axis_x, width, height, float(bcfg["slot_min_inside_image"]))
        if m is None or damage_share(m, damage) >= float(bcfg["slot_min_damage"]):
            continue
        checked += 1
        if any(o["class"] == e["class"] and iou(m, o["box"]) >= float(bcfg["mirror_match_iou"]) for o in active):
            matched += 1
    return (matched / checked if checked else 0.0), checked


def estimate_axis(active: list[dict], damage: np.ndarray, bcfg: dict) -> dict:
    """Candidate axes are the midpoints of same-class pairs at a similar height and size; the one with
    the highest agreement wins (ties: more checked elements)."""
    tol = float(bcfg["row_tolerance"])
    candidates = set()
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            if a["class"] != b["class"]:
                continue
            (ax, ay), (bx, by) = box_centre(a["box"]), box_centre(b["box"])
            (aw, ah), (bw, bh) = box_size(a["box"]), box_size(b["box"])
            mean_h = (ah + bh) / 2
            if abs(ay - by) <= tol * mean_h and abs(ah - bh) <= tol * mean_h and abs(aw - bw) <= tol * max(aw, bw):
                candidates.add(round((ax + bx) / 2))
    best = {"axis_x": None, "agreement": 0.0, "checked": 0, "candidates": len(candidates)}
    for x in sorted(candidates):
        agreement, checked = axis_agreement(float(x), active, damage, bcfg)
        if checked < int(bcfg["symmetry_min_elements"]):
            continue
        if (agreement, checked) > (best["agreement"], best["checked"]):
            best.update({"axis_x": float(x), "agreement": round(agreement, 3), "checked": checked})
    return best


def tier_b2_symmetry(elements: list[dict], damage: np.ndarray, bcfg: dict) -> tuple[list[dict], dict]:
    """B2: the mirror of an intact element across the facade axis falls in the damage mask with no
    element. Switched off when the axis agreement is below symmetry_min_agreement."""
    height, width = damage.shape
    active = active_elements(elements)
    given = bcfg.get("symmetry_axis_x")
    if given is not None:
        agreement, checked = axis_agreement(float(given), active, damage, bcfg)
        axis = {"axis_x": float(given), "source": "structure.yaml", "agreement": round(agreement, 3), "checked": checked}
    else:
        axis = {**estimate_axis(active, damage, bcfg), "source": "estimated"}
    minimum = float(bcfg["symmetry_min_agreement"])
    if axis["axis_x"] is None or axis["agreement"] < minimum:
        axis["enabled"] = False
        axis["reason"] = (f"no axis with at least {bcfg['symmetry_min_elements']} checkable elements"
                          if axis["axis_x"] is None else
                          f"agreement {axis['agreement']:.2f} below symmetry_min_agreement {minimum:.2f}")
        return [], {"axis": axis}
    axis["enabled"] = True
    targets = []
    for e in active:
        if e["damage_status"] != "intact":
            continue
        m = mirror_box(e["box"], axis["axis_x"], width, height, float(bcfg["slot_min_inside_image"]))
        # off the image, or the element sits on the axis
        if m is None or iou(m, e["box"]) > float(bcfg["on_axis_iou"]):
            continue
        if damage_share(m, damage) < float(bcfg["slot_min_damage"]):
            continue
        if not slot_is_free(m, e["class"], active, float(bcfg["slot_free_max_overlap"])):
            continue
        targets.append(implied_target("B2", e["class"], m, e, True, axis["agreement"] * float(e["score"]),
                                      {"axis_x": axis["axis_x"], "agreement": axis["agreement"]}))
    return targets, {"axis": axis}


def implied_target(rule: str, element_class: str, box: list[int], donor: dict, flipped: bool,
                   confidence: float, basis: dict) -> dict:
    return {
        "kind": "implied_slot", "rule": rule, "element_id": None, "class": element_class,
        "tier": TIER_B, "reason": None, "target_box": box,
        "donor_id": donor["id"], "donor_box": donor["box"], "flipped": flipped,
        "similarity": None, "confidence": round(confidence, 4), "alternatives": [], "basis": basis,
    }


# Each rule: (elements, damage, tier_b settings) -> (targets, diagnostics). Rules B3, B4 (bays, courses)
# would be further entries (docs/ARCHITECTURE.md §9).
TIER_B_RULES: dict[str, tuple[str, Callable[[list[dict], np.ndarray, dict], tuple[list[dict], dict]]]] = {
    "B1": ("rows_enabled", tier_b1_rows),
    "B2": ("symmetry_enabled", tier_b2_symmetry),
}


def find_tier_b(elements: list[dict], damage: np.ndarray, bcfg: dict) -> tuple[list[dict], dict]:
    """Run the enabled rules; when two implied slots of the same class overlap (IoU above ``same_slot_iou``) keep the one
    with the higher confidence and name the other rule in ``also_implied_by``."""
    found: list[dict] = []
    diagnostics: dict[str, Any] = {}
    for name, (switch, rule) in TIER_B_RULES.items():
        if not bcfg.get(switch, False):
            diagnostics[name] = {"enabled": False, "reason": f"tier_b.{switch} is false"}
            continue
        targets, diag = rule(elements, damage, bcfg)
        diagnostics[name] = diag
        found.extend(targets)
    kept: list[dict] = []
    for t in sorted(found, key=lambda t: -t["confidence"]):
        twin = next((k for k in kept if k["class"] == t["class"]
                     and iou(k["target_box"], t["target_box"]) > float(bcfg["same_slot_iou"])), None)
        if twin is None:
            kept.append(t)
        else:
            twin.setdefault("also_implied_by", []).append({"rule": t["rule"], "donor_id": t["donor_id"],
                                                          "confidence": t["confidence"]})
    return kept, diagnostics


# ---------------------------------------------------------------------------
# Tier C regions
# ---------------------------------------------------------------------------

def tier_c_regions(damage: np.ndarray, covered_boxes: list[list[int]], min_region_px: int
                   ) -> tuple[list[dict], list[np.ndarray]]:
    """Damage pixels outside every A and B target box, as connected regions (largest first)."""
    remaining = damage.astype(bool).copy()
    for x1, y1, x2, y2 in covered_boxes:
        remaining[y1:y2, x1:x2] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(remaining.astype(np.uint8), connectivity=8)
    parts = sorted((int(stats[i][4]), i) for i in range(1, n) if stats[i][4] >= min_region_px)[::-1]
    records, masks = [], []
    for area, i in parts:
        x, y, w, h = (int(v) for v in stats[i][:4])
        records.append({
            "kind": "uncovered_region", "rule": None, "element_id": None, "class": None,
            "tier": TIER_C, "reason": "no element evidence for these damage pixels",
            "target_box": [x, y, x + w, y + h], "area_px": area,
            "donor_id": None, "donor_box": None, "flipped": None,
            "similarity": None, "confidence": 0.0, "alternatives": [],
        })
        masks.append((labels == i).astype(np.uint8))
    return records, masks


# ---------------------------------------------------------------------------
# DINOv2 (torch imported only here)
# ---------------------------------------------------------------------------

class Dinov2Embedder:
    """DINOv2 loading and inference, ported from archive/staged_pipeline_v8/02_stage2c_dinov2_repetition.py
    (processor without resize or centre crop, fp16 on CUDA). Global vectors instead of a dense grid."""

    def __init__(self, ecfg: dict[str, Any], log) -> None:
        import torch
        from transformers import AutoImageProcessor, Dinov2Model

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = self.device == "cuda" and bool(ecfg["use_fp16_if_cuda"])
        self.pooling = ecfg["pooling"]
        if self.pooling not in ("mean_patch", "cls"):
            raise ContractError(f"evidence.pooling must be mean_patch or cls, not {self.pooling!r}")
        self.batch = int(ecfg["batch_size"])
        log.info("[evidence] loading %s on %s%s", ecfg["dino_model_id"], self.device, " fp16" if self.fp16 else "")
        self.processor = AutoImageProcessor.from_pretrained(ecfg["dino_model_id"])
        self.model = Dinov2Model.from_pretrained(ecfg["dino_model_id"]).to(self.device).eval()
        if self.fp16:
            self.model.half()

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        for start in range(0, len(images), self.batch):
            chunk = [Image.fromarray(img) for img in images[start:start + self.batch]]
            inputs = self.processor(images=chunk, return_tensors="pt", do_resize=False, do_center_crop=False)
            pixels = inputs["pixel_values"].to(self.device)
            if self.fp16:
                pixels = pixels.half()
            with torch.inference_mode():
                hidden = self.model(pixel_values=pixels).last_hidden_state.float()
            vec = hidden[:, 1:, :].mean(dim=1) if self.pooling == "mean_patch" else hidden[:, 0, :]
            vec = torch.nn.functional.normalize(vec, dim=-1)
            out.append(vec.cpu().numpy())
        return np.concatenate(out, axis=0)

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

GREEN, GREY, RED, BLUE = (40, 220, 90), (170, 170, 170), (240, 60, 60), (60, 140, 255)
TIER_COLOR = {TIER_A: GREEN, TIER_B: BLUE, TIER_C: GREY}


def centre(box: list[int]) -> tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def draw_overlay(rgb: np.ndarray, damage: np.ndarray, records: list[dict[str, Any]],
                 c_masks: list[np.ndarray] | None = None) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    sel = damage > 0
    out[sel] = out[sel] * 0.6 + np.array(RED, dtype=np.float32) * 0.4
    out = np.clip(out, 0, 255).astype(np.uint8)
    if c_masks:
        h, w = damage.shape
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((xx + yy) % 8) < 2
        for m in c_masks:
            out[(m > 0) & hatch] = GREY
    for r in records:
        if r["kind"] == "uncovered_region":
            continue
        color = TIER_COLOR[r["tier"]]
        x1, y1, x2, y2 = r["target_box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
        if r["donor_box"]:
            dx1, dy1, dx2, dy2 = r["donor_box"]
            cv2.rectangle(out, (dx1, dy1), (dx2 - 1, dy2 - 1), color, 1, cv2.LINE_AA)
            cv2.arrowedLine(out, centre(r["donor_box"]), centre(r["target_box"]), color, 1, cv2.LINE_AA, tipLength=0.04)
        label = r["tier"] if r["tier"] != TIER_B else r["rule"]
        cv2.putText(out, f"{r['target_id']} {label}", (x1 + 2, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def draw_sheet(rgb: np.ndarray, damage: np.ndarray, records: list[dict[str, Any]],
               elements: dict[str, dict], tile_h: int = 150) -> np.ndarray:
    """Rows: fragment (hole tinted) | chosen donor as transferred | alternatives."""
    rows = []
    for r in records:
        if r["kind"] == "uncovered_region":
            continue
        box = r["target_box"]
        tw, th = box[2] - box[0], box[3] - box[1]
        frag = crop(rgb, box).astype(np.float32).copy()
        hole = crop(damage, box) > 0
        frag[hole] = frag[hole] * 0.5 + np.array(RED, dtype=np.float32) * 0.5
        name = r["element_id"] or f"{r['rule']} slot"
        tiles = [(frag.astype(np.uint8), f"{r['target_id']} {name} {r['tier']}")]
        chosen = ([{"donor_id": r["donor_id"], "similarity": r["similarity"], "flipped": r["flipped"]}]
                  if r["donor_id"] else []) + r["alternatives"]
        for c in chosen:
            img = donor_crop(rgb, damage, elements[c["donor_id"]]["box"], box, None, c["flipped"])
            score = f"{c['similarity']:.2f}" if c["similarity"] is not None else f"conf {r['confidence']:.2f}"
            tiles.append((img, f"{c['donor_id']} {score}{' F' if c['flipped'] else ''}"))
        if not chosen and r["reason"]:
            tiles.append((np.full((th, max(tw, 1), 3), 40, np.uint8), r["reason"][:40]))
        scale = tile_h / th
        width = max(1, int(round(tw * scale)))
        row = []
        for img, text in tiles:
            tile = cv2.resize(img, (width, tile_h), interpolation=cv2.INTER_AREA)
            if width < 120:  # pad, never stretch: the label needs room, the element keeps its shape
                left = (120 - width) // 2
                tile = np.pad(tile, ((0, 0), (left, 120 - width - left), (0, 0)))
            band = np.full((16, tile.shape[1], 3), 25, np.uint8)
            cv2.putText(band, text[:28], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (235, 235, 235), 1, cv2.LINE_AA)
            row.append(np.vstack([band, tile]))
            row.append(np.full((tile_h + 16, 4, 3), 0, np.uint8))
        rows.append(np.hstack(row))
    if not rows:
        return np.zeros((10, 10, 3), np.uint8)
    width = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows]
    return np.vstack(padded)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def run_step(run: "Run", opts: "argparse.Namespace") -> None:
    log = run.logger
    ecfg = run.config["evidence"]
    doc = load_elements(run.run_dir)
    elements = doc["elements"]
    if not any("damage_status" in e for e in elements):
        raise ContractError(f"{run.path('elements.json')}: no damage_status; run the damage step first")
    regions = read_json(run.path("damage_regions.json"))  # contract check: damage step ran in this run
    mask_path = run.path(DAMAGE_MASK)
    if not mask_path.is_file():
        raise ContractError(f"Damage mask copy not found: {mask_path}; run the damage step first")
    damage = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8)
    rgb = np.asarray(Image.open(run.image_path).convert("RGB"))
    if rgb.shape[:2] != damage.shape:
        raise ContractError(f"{mask_path}: size differs from {run.image_path}")

    embedder = Dinov2Embedder(ecfg, log)
    try:
        records = find_tier_a(elements, rgb, damage, ecfg, embedder)
    finally:
        embedder.close()

    a_records = records
    b_records, b_diag = find_tier_b(elements, damage, ecfg["tier_b"])
    covered = [r["target_box"] for r in a_records if r["tier"] == TIER_A] + [r["target_box"] for r in b_records]
    c_records, c_masks = tier_c_regions(damage, covered, int(run.config["damage"]["min_region_px"]))
    records = a_records + b_records + c_records
    for number, r in enumerate(records, start=1):
        r["target_id"] = f"t_{number:03d}"
    c_dir = run.path(C_REGION_DIR)
    if c_dir.exists():
        shutil.rmtree(c_dir)
    c_dir.mkdir()
    for r, m in zip(c_records, c_masks):
        r["mask_file"] = f"{C_REGION_DIR}/{r['target_id']}.png"
        Image.fromarray(m * 255, mode="L").save(run.path(r["mask_file"]))

    by_tier: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    for r in records:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        if r["class"]:
            by_class.setdefault(r["class"], {}).setdefault(r["tier"], 0)
            by_class[r["class"]][r["tier"]] += 1
    flagged = [r["target_id"] for r in records if r.get("donor_expert_prototype") is False]
    damage_px = int(damage.sum())
    c_px = int(sum(int(m.sum()) for m in c_masks))
    write_json(run.path(EVIDENCE_FILE), {
        "structure_id": run.structure_id,
        "image_size": doc["image_size"],
        "damage_mask_sha256": regions["damage_mask"]["sha256"],
        "model": {"id": ecfg["dino_model_id"], "pooling": ecfg["pooling"], "crop_px": ecfg["crop_px"],
                  "device": embedder.device, "fp16": embedder.fp16},
        "settings": {**{k: ecfg[k] for k in ("min_similarity", "min_visible_fraction", "size_tolerance",
                                            "alternatives_kept", "try_flipped_donor", "mask_donor_like_target")},
                     "tier_b": ecfg["tier_b"]},
        "summary": {"targets": len(records), "by_tier": by_tier, "by_class": by_class,
                    "fragmented_elements": len(a_records),
                    "tier_c_regions": len(c_records),
                    "damage_px": damage_px,
                    "tier_c_region_px": c_px,
                    "tier_c_share_of_damage": round(c_px / damage_px, 4) if damage_px else 0.0,
                    "donor_not_marked_as_prototype_by_expert": flagged},
        "tier_b_diagnostics": b_diag,
        "targets": records,
    })
    by_id = {e["id"]: e for e in elements}
    Image.fromarray(draw_overlay(rgb, damage, records, c_masks)).save(run.path(OVERLAY))
    Image.fromarray(draw_sheet(rgb, damage, records, by_id)).save(run.path(SHEET))

    log.info("[evidence] %d target(s): %s", len(records), by_tier)
    for r in records:
        if r["kind"] == "fragmented_element" and r["tier"] == TIER_A:
            log.info("[evidence]   %s %-24s A  donor %s%s sim %.3f", r["target_id"], r["element_id"],
                     r["donor_id"], " (flipped)" if r["flipped"] else "", r["similarity"])
        elif r["kind"] == "fragmented_element":
            log.info("[evidence]   %s %-24s C  %s", r["target_id"], r["element_id"], r["reason"])
        elif r["kind"] == "implied_slot":
            log.info("[evidence]   %s %-24s %s donor %s conf %.3f box %s", r["target_id"], r["class"] + " slot",
                     r["rule"], r["donor_id"], r["confidence"], r["target_box"])
    for name, diag in b_diag.items():
        if "axis" in diag:
            axis = diag["axis"]
            log.info("[evidence] B2 axis (%s): x=%s agreement %.2f over %d element(s)%s", axis["source"],
                     axis["axis_x"], axis["agreement"], axis["checked"],
                     "" if axis["enabled"] else f" -> B2 off: {axis['reason']}")
        elif "rows" in diag:
            usable = [r for r in diag["rows"] if "skipped" not in r]
            log.info("[evidence] B1: %d row(s), %d with at least min_row_size elements", len(diag["rows"]), len(usable))
        elif not diag.get("enabled", True):
            log.info("[evidence] %s off: %s", name, diag["reason"])
    log.info("[evidence] tier C: %d region(s), %.1f %% of the damage pixels", len(c_records),
             100 * c_px / damage_px if damage_px else 0.0)
    if flagged:
        log.info("[evidence] donors the expert did not mark usable_as_prototype: %s", flagged)
    run.meta["steps"].setdefault("evidence", {})["summary"] = {"targets": len(records), "by_tier": by_tier}
