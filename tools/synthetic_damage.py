#!/usr/bin/env python3
"""Synthetic damage cases (T19, docs/ARCHITECTURE.md §7.1).

Takes a finished run of one structure (after the damage step, so every element has damage_status) and
hides things the photograph shows, so the pipeline can be checked against what is really there:

    partial      a band of ``synthetic.partial_fraction`` of an intact element's box is hidden;
                 the element stays in the inventory                               -> expected tier A
    full         the whole box plus ``synthetic.full_margin_px`` is hidden and the element is marked
                 ``rejected: true`` (``synthetic_hidden: true``; nothing is deleted)  -> expected tier B
    background   ``synthetic.background_cases`` fixed rectangles on wall with no element and no damage
                                                                                  -> expected tier C,
                 and the restoration view must leave them open

"Hidden" means: added to the damage mask **and** painted with ``synthetic.hidden_fill`` in the case
image. Without the paint, a pixel the pipeline leaves open would still show the original and score
perfectly. Detection is not re-run (§7.1); the inventory and element masks are copied from the run.

    python tools/synthetic_damage.py --run outputs/s01_test_church/20261112_121244

Output: ``outputs/synthetic/<structure>/<timestamp>/``
    manifest.json             base run, settings, case list, skipped elements
    cases_overview.png        hidden areas on the photograph (yellow partial, magenta full, cyan background)
    <case_id>/image.png       photograph with the hidden area painted
    <case_id>/damage_mask.png original damage mask plus the hidden area
    <case_id>/hidden_mask.png the hidden area only (0/255)
    <case_id>/ground_truth.json
    <case_id>/seed/           elements.json and element_masks/ for the damage step

Run them with ``python evaluation/evaluate.py completion --cases <that folder>``.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from i2r import runio  # noqa: E402
from i2r.detect import ELEMENTS_FILE, MASK_DIR, load_elements  # noqa: E402
from i2r.runio import ContractError, write_json  # noqa: E402

MANIFEST = "manifest.json"
GROUND_TRUTH = "ground_truth.json"
SEED_DIR = "seed"
HIDDEN_MASK = "hidden_mask.png"
OVERVIEW = "cases_overview.png"

VARIANTS = ("partial", "full", "background")
EXPECTED_TIER = {"partial": "A", "full": "B", "background": "C"}
VARIANT_COLOR = {"partial": (255, 220, 0), "full": (255, 0, 255), "background": (0, 230, 255)}
SIDES = ("top", "bottom", "left", "right")


# ---------------------------------------------------------------------------
# Geometry (pure, tested)
# ---------------------------------------------------------------------------

def partial_band(box: list[int], fraction: float, side: str) -> list[int]:
    """The part of ``box`` to hide: a band of ``fraction`` of its height (top/bottom) or width."""
    if side not in SIDES:
        raise ContractError(f"synthetic.partial_side must be one of {', '.join(SIDES)}, not {side!r}")
    if not 0.0 < fraction < 1.0:
        raise ContractError(f"synthetic.partial_fraction must lie in (0, 1), not {fraction}")
    x1, y1, x2, y2 = box
    h = max(1, int(round((y2 - y1) * fraction)))
    w = max(1, int(round((x2 - x1) * fraction)))
    return {
        "top": [x1, y1, x2, y1 + h],
        "bottom": [x1, y2 - h, x2, y2],
        "left": [x1, y1, x1 + w, y2],
        "right": [x2 - w, y1, x2, y2],
    }[side]


def expand_box(box: list[int], margin: int, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [max(0, x1 - margin), max(0, y1 - margin), min(width, x2 + margin), min(height, y2 + margin)]


def rect_mask(box: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = 1
    return out


def background_boxes(
    element_boxes: list[list[int]], damage: np.ndarray, size: tuple[int, int], clearance: int, grid: int, count: int,
) -> list[list[int]]:
    """Up to ``count`` rectangles of ``size`` (w, h) inside the convex hull of the element boxes (a proxy
    for the facade; the bounding box reached the rock beside the building on s01) that keep ``clearance``
    from every element box and from every damage pixel. The first is the one nearest the facade centre;
    each next one is the farthest from those already chosen (deterministic)."""
    if not element_boxes or count <= 0:
        return []
    height, width = damage.shape
    bw, bh = size
    fx1 = max(0, min(b[0] for b in element_boxes))
    fy1 = max(0, min(b[1] for b in element_boxes))
    fx2 = min(width, max(b[2] for b in element_boxes))
    fy2 = min(height, max(b[3] for b in element_boxes))
    corners = np.array([[x, y] for b in element_boxes for x in (b[0], b[2] - 1) for y in (b[1], b[3] - 1)],
                       dtype=np.int32)
    inside = np.zeros_like(damage, dtype=np.uint8)
    cv2.fillConvexPoly(inside, cv2.convexHull(corners), 1)
    blocked = ((damage > 0) | (inside == 0)).astype(np.uint8)
    for b in element_boxes:
        blocked[max(0, b[1]):max(0, b[3]), max(0, b[0]):max(0, b[2])] = 1
    if clearance > 0:
        k = 2 * clearance + 1
        blocked = cv2.dilate(blocked, np.ones((k, k), np.uint8))
    integral = cv2.integral(blocked)
    candidates = []
    for y in range(fy1, fy2 - bh + 1, max(1, grid)):
        for x in range(fx1, fx2 - bw + 1, max(1, grid)):
            hit = integral[y + bh, x + bw] - integral[y, x + bw] - integral[y + bh, x] + integral[y, x]
            if hit == 0:
                candidates.append((x + bw / 2.0, y + bh / 2.0, [x, y, x + bw, y + bh]))
    if not candidates:
        return []
    cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    chosen = [min(candidates, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)]
    while len(chosen) < count:
        best = max(candidates, key=lambda c: min((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 for d in chosen))
        if any(best[2] == d[2] for d in chosen):
            break
        chosen.append(best)
    return [c[2] for c in chosen]


def paint(rgb: np.ndarray, hidden: np.ndarray, fill: list[int]) -> np.ndarray:
    out = rgb.copy()
    out[hidden > 0] = np.asarray(fill, dtype=np.uint8)
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def plan_cases(doc: dict[str, Any], damage: np.ndarray, scfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Case specifications (no files) and skipped elements with the reason."""
    height, width = damage.shape
    active = [e for e in doc["elements"] if not e.get("rejected")]
    intact = [e for e in active if e.get("damage_status") == "intact"]
    if not any("damage_status" in e for e in active):
        raise ContractError("elements.json has no damage_status; give a run in which the damage step ran")
    cases, skipped = [], []
    for e in intact:
        band = partial_band(e["box"], float(scfg["partial_fraction"]), str(scfg["partial_side"]))
        cases.append({"variant": "partial", "element": e, "hidden_box": band})
        full = expand_box(e["box"], int(scfg["full_margin_px"]), width, height)
        cases.append({"variant": "full", "element": e, "hidden_box": full})
    for e in active:
        if e.get("damage_status") != "intact":
            skipped.append({"id": e["id"], "class": e["class"], "reason": f"not intact ({e.get('damage_status')})"})
    rects = background_boxes([e["box"] for e in active], damage, tuple(scfg["background_size_px"]),
                             int(scfg["background_clearance_px"]), int(scfg["background_grid_px"]),
                             int(scfg["background_cases"]))
    if len(rects) < int(scfg["background_cases"]):
        skipped.append({"id": "background", "class": None,
                        "reason": f"only {len(rects)} free rectangle(s) of {scfg['background_cases']} found"})
    for number, rect in enumerate(rects, start=1):
        cases.append({"variant": "background", "element": None, "hidden_box": rect, "number": number})
    for c in cases:
        c["case_id"] = (f"background__bg{c['number']:02d}" if c["variant"] == "background"
                        else f"{c['variant']}__{c['element']['id']}")
    return cases, skipped


def case_inventory(doc: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """elements.json for the case: a full case marks its element rejected (never deleted)."""
    out = copy.deepcopy(doc)
    if spec["variant"] == "full":
        for e in out["elements"]:
            if e["id"] == spec["element"]["id"]:
                e["rejected"] = True
                e["synthetic_hidden"] = True
                e["review_status"] = "synthetic_hidden"
    return out


def ground_truth(spec: dict[str, Any], structure_id: str, base_run: str, hidden_px: int) -> dict[str, Any]:
    element = spec["element"]
    gt = {
        "case_id": spec["case_id"],
        "structure_id": structure_id,
        "base_run": base_run,
        "variant": spec["variant"],
        "expected_tier": EXPECTED_TIER[spec["variant"]],
        "element_id": element["id"] if element else None,
        "class": element["class"] if element else None,
        "element_box": element["box"] if element else None,
        "hidden_box": spec["hidden_box"],
        "hidden_px": hidden_px,
        "hidden_mask_file": HIDDEN_MASK,
    }
    if spec["variant"] == "full":
        gt["note"] = ("Tier B needs a B1 row (evidence.tier_b.min_row_size, interpolation only) or a B2 mirror "
                      "partner; without either the current rules give tier C.")
    if spec["variant"] == "background":
        gt["note"] = "Plain wall: the restoration view must leave it open (abstention)."
    return gt


def write_case(folder: Path, spec: dict[str, Any], rgb: np.ndarray, damage: np.ndarray, doc: dict[str, Any],
               run_dir: Path, structure_id: str, fill: list[int]) -> dict[str, Any]:
    hidden = rect_mask(spec["hidden_box"], damage.shape) & (1 - damage)
    folder.mkdir(parents=True)
    Image.fromarray(paint(rgb, hidden, fill)).save(folder / "image.png")
    Image.fromarray(((damage | hidden) * 255).astype(np.uint8), mode="L").save(folder / "damage_mask.png")
    Image.fromarray((hidden * 255).astype(np.uint8), mode="L").save(folder / HIDDEN_MASK)
    seed = folder / SEED_DIR
    seed.mkdir()
    write_json(seed / ELEMENTS_FILE, case_inventory(doc, spec))
    shutil.copytree(run_dir / MASK_DIR, seed / MASK_DIR)
    gt = ground_truth(spec, structure_id, runio.relative(run_dir), int(hidden.sum()))
    write_json(folder / GROUND_TRUTH, gt)
    return gt


def draw_overview(rgb: np.ndarray, cases: list[dict[str, Any]]) -> np.ndarray:
    out = rgb.copy()
    for c in cases:
        x1, y1, x2, y2 = c["hidden_box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), VARIANT_COLOR[c["variant"]], 1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create synthetic damage cases from a run (T19).")
    parser.add_argument("--run", required=True, help="run folder in which the damage step ran")
    parser.add_argument("--config", default=str(runio.DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    try:
        run_dir = Path(args.run).resolve()
        doc = load_elements(run_dir)
        structure_id = doc["structure_id"]
        cfg = runio.load_config(args.config)
        sdir = runio.structure_dir(cfg, structure_id)
        cfg, _ = runio.apply_structure_config(cfg, sdir)
        scfg = cfg["synthetic"]
        mask_path = run_dir / "damage_mask.png"
        if not mask_path.is_file():
            raise ContractError(f"{mask_path} not found; give a run in which the damage step ran")
        damage = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8)
        rgb = np.asarray(Image.open(sdir / cfg["paths"]["image_file"]).convert("RGB"))
        if rgb.shape[:2] != damage.shape:
            raise ContractError(f"{mask_path}: size differs from the structure image")
        cases, skipped = plan_cases(doc, damage, scfg)
        out_root = runio.REPO_ROOT / cfg["paths"]["output_root"] / "synthetic" / structure_id
        out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=False)
        listed = []
        for spec in cases:
            gt = write_case(out_dir / spec["case_id"], spec, rgb, damage, doc, run_dir, structure_id,
                            list(scfg["hidden_fill"]))
            listed.append({k: gt[k] for k in ("case_id", "variant", "element_id", "class", "expected_tier",
                                              "hidden_px")})
        write_json(out_dir / MANIFEST, {
            "structure_id": structure_id,
            "base_run": runio.relative(run_dir),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": runio.git_commit(),
            "settings": scfg,
            "cases": listed,
            "skipped": skipped,
        })
        Image.fromarray(draw_overview(rgb, cases)).save(out_dir / OVERVIEW)
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    counts = {v: sum(1 for c in listed if c["variant"] == v) for v in VARIANTS}
    print(f"[synthetic] {len(listed)} case(s) {counts} in {runio.relative(out_dir)}")
    for s in skipped:
        print(f"[synthetic]   skipped {s['id']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
