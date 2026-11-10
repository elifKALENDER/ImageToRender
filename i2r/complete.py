"""Step 5 — complete: donor transfer into the damage mask, per view rule.

Donor transfer (docs/ARCHITECTURE.md §5.5), deterministic, no model:
1. Donor pixels = its element mask minus its own damaged pixels, cropped to its box.
2. Affine map donor box -> target box (scale + translation), flipped when the evidence says so.
3. Tier A only: search a small shift and scale that best fit the fragment's visible part
   (grey-level normalised correlation over visible pixels the donor covers).
4. Optional mean-colour shift to the fragment's visible part, then blend **only inside the damage mask**;
   alpha ramps from 0 at the mask edge to 1 ``complete.feather_px`` inside it.
5. Every changed pixel gets the target's label in the source layer.

View rules (§5.3, §5.4) are a table, ``VIEW_RULES``: record -> (complete?, reason). The restoration
rule completes tier A, tier B only with ``restoration.include_tier_b``, never tier C. The
first-construction rule is T18; asking for that view logs it as not implemented.

At the end of every view: no pixel outside the damage mask may differ from the photograph
(``assert_preserved``); otherwise the run fails with ``PreservationError``.

Targets are painted in increasing confidence, so where two overlap the more confident one is on top.

Output in the run folder, per view <v>:
    <v>_completed.png    the photograph with completed targets (no marking; render.py adds it)
    <v>_source.png       16-bit label image: 0 = photograph, n = n-th entry of "labels"
    completion.json      per view: rule, per-target status, alignment, pixels changed, labels
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np
from PIL import Image

from i2r.detect import load_elements
from i2r.runio import ContractError, read_json, write_json

if TYPE_CHECKING:
    import argparse

    from i2r.runio import Run

COMPLETION_FILE = "completion.json"
DAMAGE_MASK = "damage_mask.png"


class PreservationError(ContractError):
    """A pixel outside the damage mask was changed."""


# ---------------------------------------------------------------------------
# View rules: (record, config) -> (complete it?, reason)
# ---------------------------------------------------------------------------

def rule_restoration(record: dict, config: dict) -> tuple[bool, str]:
    tier = record["tier"]
    if tier == "A":
        return True, "tier A: direct evidence"
    if tier == "B":
        if config["restoration"]["include_tier_b"]:
            return True, "tier B: included by restoration.include_tier_b"
        return False, "tier B: position implied, not restored (Venice Charter, §5.3)"
    return False, "tier C: no evidence"


VIEW_RULES: dict[str, Callable[[dict, dict], tuple[bool, str]]] = {
    "restoration": rule_restoration,
}


# ---------------------------------------------------------------------------
# Geometry and transfer (pure functions)
# ---------------------------------------------------------------------------

def donor_patch(rgb: np.ndarray, damage: np.ndarray, donor_mask: np.ndarray, box: list[int],
                flipped: bool) -> tuple[np.ndarray, np.ndarray]:
    """(RGB crop, boolean usable mask) of the donor box; usable = element mask and not damaged."""
    x1, y1, x2, y2 = box
    patch = rgb[y1:y2, x1:x2].copy()
    usable = (donor_mask[y1:y2, x1:x2] > 0) & (damage[y1:y2, x1:x2] == 0)
    if flipped:
        patch, usable = patch[:, ::-1].copy(), usable[:, ::-1].copy()
    return patch, usable


def placement(target_box: list[int], scale: float, dx: int, dy: int) -> tuple[float, float, float, float]:
    """(x1, y1, width, height) of the donor's image after scaling about the target centre and shifting."""
    x1, y1, x2, y2 = target_box
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
    return cx - w / 2, cy - h / 2, w, h


def warp(patch: np.ndarray, usable: np.ndarray, place: tuple[float, float, float, float],
         window: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Donor patch and usable mask mapped into ``window`` (x1, y1, x2, y2 in image pixels)."""
    ph, pw = usable.shape
    px, py, w, h = place
    wx1, wy1, wx2, wy2 = window
    matrix = np.float32([[w / pw, 0, px - wx1], [0, h / ph, py - wy1]])
    size = (wx2 - wx1, wy2 - wy1)
    image = cv2.warpAffine(patch, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(usable.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return image, mask


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-9 else 0.0


def grey(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def window_for(target_box: list[int], ccfg: dict, shape: tuple[int, int]) -> list[int]:
    """Target box grown enough to hold every searched placement, clipped to the image."""
    x1, y1, x2, y2 = target_box
    w, h = x2 - x1, y2 - y1
    grow = max(ccfg["refine_scales"]) - 1.0
    mx = int(np.ceil(ccfg["refine_shift_frac"] * w + grow * w / 2)) + 1
    my = int(np.ceil(ccfg["refine_shift_frac"] * h + grow * h / 2)) + 1
    height, width = shape
    return [max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)]


def align(rgb: np.ndarray, damage: np.ndarray, patch: np.ndarray, usable: np.ndarray,
          target_box: list[int], window: list[int], ccfg: dict) -> dict[str, Any]:
    """Best (scale, dx, dy) by NCC on visible fragment pixels inside the target box."""
    wx1, wy1, wx2, wy2 = window
    x1, y1, x2, y2 = target_box
    in_box = np.zeros((wy2 - wy1, wx2 - wx1), dtype=bool)
    in_box[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = True
    visible = in_box & (damage[wy1:wy2, wx1:wx2] == 0)
    target_grey = grey(rgb[wy1:wy2, wx1:wx2])
    w, h = x2 - x1, y2 - y1
    sx, sy = int(round(ccfg["refine_shift_frac"] * w)), int(round(ccfg["refine_shift_frac"] * h))
    minimum = int(ccfg["refine_min_visible_px"])

    def score(scale: float, dx: int, dy: int) -> tuple[float, int]:
        image, mask = warp(patch, usable, placement(target_box, scale, dx, dy), window)
        sel = visible & mask
        n = int(sel.sum())
        return (ncc(grey(image)[sel], target_grey[sel]) if n >= minimum else float("-inf")), n

    base, base_n = score(1.0, 0, 0)
    result = {"scale": 1.0, "dx": 0, "dy": 0, "score_before": None if base == float("-inf") else round(base, 4),
              "score": None, "visible_px": base_n, "searched": False}
    if base == float("-inf"):
        result["note"] = f"fewer than refine_min_visible_px ({minimum}) visible pixels; not aligned"
        return result
    best = (base, 1.0, 0, 0)
    for scale in ccfg["refine_scales"]:
        for dy in range(-sy, sy + 1):
            for dx in range(-sx, sx + 1):
                value, _ = score(float(scale), dx, dy)
                # strictly better only: ties keep the unshifted placement
                if value > best[0] + 1e-9:
                    best = (value, float(scale), dx, dy)
    result.update({"scale": best[1], "dx": best[2], "dy": best[3], "score": round(best[0], 4), "searched": True})
    return result


def feather_alpha(fill: np.ndarray, damage_window: np.ndarray, feather_px: float) -> np.ndarray:
    """Alpha 0..1: 0 wherever ``fill`` is False; inside, ramps up from the damage-mask edge."""
    if feather_px <= 0:
        return fill.astype(np.float32)
    padded = np.pad(damage_window.astype(np.uint8), 1, constant_values=1)  # window border is not an edge
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    return np.where(fill, alpha, 0.0).astype(np.float32)


def paint(out: np.ndarray, labels: np.ndarray, window: list[int], image: np.ndarray, alpha: np.ndarray,
          label: int) -> int:
    """Blend ``image`` into ``out`` inside ``window``; returns how many pixels got this label."""
    wx1, wy1, wx2, wy2 = window
    region = out[wy1:wy2, wx1:wx2].astype(np.float32)
    a = alpha[..., None]
    out[wy1:wy2, wx1:wx2] = np.clip(region * (1 - a) + image.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
    changed = alpha > 0
    labels[wy1:wy2, wx1:wx2][changed] = label
    return int(changed.sum())


def transfer(out: np.ndarray, labels: np.ndarray, rgb: np.ndarray, damage: np.ndarray,
             record: dict, donor_mask: np.ndarray, label: int, ccfg: dict) -> dict[str, Any]:
    """Transfer one donor into one target. ``rgb`` is the untouched photograph (alignment and colour
    are measured on it); ``out`` and ``labels`` are written."""
    target_box = record["target_box"]
    patch, usable = donor_patch(rgb, damage, donor_mask, record["donor_box"], bool(record["flipped"]))
    window = window_for(target_box, ccfg, damage.shape)
    if record["tier"] == "A":
        alignment = align(rgb, damage, patch, usable, target_box, window, ccfg)
    else:
        alignment = {"scale": 1.0, "dx": 0, "dy": 0, "searched": False,
                     "note": "tier B: no fragment to align to; placed at the implied box"}
    image, mask = warp(patch, usable, placement(target_box, alignment["scale"], alignment["dx"], alignment["dy"]),
                       window)
    wx1, wy1, wx2, wy2 = window
    damage_w = damage[wy1:wy2, wx1:wx2] > 0
    colour_shift = [0.0, 0.0, 0.0]
    visible = mask & ~damage_w
    if ccfg["match_colour"] and int(visible.sum()) >= int(ccfg["refine_min_visible_px"]):
        shift = rgb[wy1:wy2, wx1:wx2][visible].astype(np.float32).mean(0) - image[visible].astype(np.float32).mean(0)
        image = np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        colour_shift = [round(float(v), 1) for v in shift]
    fill = mask & damage_w
    alpha = feather_alpha(fill, damage_w, float(ccfg["feather_px"]))
    painted = paint(out, labels, window, image, alpha, label)
    x1, y1, x2, y2 = target_box
    box_damage = int(damage[y1:y2, x1:x2].sum())
    return {"alignment": alignment, "colour_shift": colour_shift, "pixels_painted": painted,
            "damage_px_in_box": box_damage,
            "damage_px_in_box_left_unfilled": int(box_damage - (fill[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1]).sum())}


def assert_preserved(original: np.ndarray, result: np.ndarray, damage: np.ndarray, view: str) -> None:
    changed = np.any(original != result, axis=-1) & (damage == 0)
    n = int(changed.sum())
    if n:
        ys, xs = np.nonzero(changed)
        raise PreservationError(f"[{view}] {n} pixel(s) outside the damage mask changed, "
                                f"first at x={int(xs[0])}, y={int(ys[0])}")


def complete_view(view: str, rgb: np.ndarray, damage: np.ndarray, records: list[dict],
                  donor_masks: dict[str, np.ndarray], config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply one view's rule. Returns (completed image, uint16 label image, report)."""
    rule = VIEW_RULES[view]
    ccfg = config["complete"]
    out = rgb.copy()
    labels = np.zeros(damage.shape, dtype=np.uint16)
    label_names: dict[str, str] = {}
    results = []
    for record in sorted(records, key=lambda r: (r["confidence"] or 0.0)):
        do_it, why = rule(record, config)
        entry = {"target_id": record["target_id"], "tier": record["tier"], "kind": record["kind"],
                 "donor_id": record.get("donor_id"), "rule_reason": why}
        if not do_it:
            entry.update({"status": "not_completed", "method": "none", "pixels_changed": 0})
        elif not record.get("donor_id"):
            entry.update({"status": "not_completed", "method": "none", "pixels_changed": 0,
                          "note": "rule allows completion but the record has no donor"})
        else:
            label = len(label_names) + 1
            label_names[str(label)] = record["target_id"]
            info = transfer(out, labels, rgb, damage, record, donor_masks[record["donor_id"]], label, ccfg)
            entry.update({"status": "completed", "method": "donor_transfer", "label": label,
                          "flipped": bool(record["flipped"]), **info})
        results.append(entry)
    for entry in results:  # final count after overlaps: pixels whose visible label is this target
        if entry.get("label"):
            entry["pixels_changed"] = int((labels == entry["label"]).sum())
            if entry["pixels_changed"] == 0:
                entry["status"] = "completed_but_empty"
    assert_preserved(rgb, out, damage, view)
    order = {r["target_id"]: i for i, r in enumerate(records)}
    results.sort(key=lambda e: order[e["target_id"]])
    report = {"rule": rule.__name__, "labels": label_names, "targets": results,
              "pixels_changed": int((labels > 0).sum()),
              "completed": sum(1 for e in results if e["status"] == "completed")}
    return out, labels, report


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def load_donor_masks(run: "Run", elements: list[dict], records: list[dict], size: tuple[int, int]) -> dict[str, np.ndarray]:
    by_id = {e["id"]: e for e in elements}
    masks = {}
    for donor_id in {r["donor_id"] for r in records if r.get("donor_id")}:
        if donor_id not in by_id:
            raise ContractError(f"evidence.json names donor {donor_id!r}, which is not in elements.json")
        path = run.path(by_id[donor_id]["mask_file"])
        if not path.is_file():
            raise ContractError(f"Mask of donor {donor_id} not found: {path}")
        mask = Image.open(path).convert("L")
        if mask.size != size:
            raise ContractError(f"{path}: size {mask.size} differs from the image size {size}")
        masks[donor_id] = (np.asarray(mask) >= 128).astype(np.uint8)
    return masks


def run_step(run: "Run", opts: "argparse.Namespace") -> None:
    log = run.logger
    evidence = read_json(run.path("evidence.json"))
    if "targets" not in evidence:
        raise ContractError(f"{run.path('evidence.json')}: missing key 'targets'")
    records = evidence["targets"]
    for r in records:
        for key in ("target_id", "kind", "tier", "target_box", "donor_id", "donor_box", "flipped", "confidence"):
            if key not in r:
                raise ContractError(f"evidence.json target {r.get('target_id', '?')}: missing key {key!r}")
    elements = load_elements(run.run_dir)["elements"]
    mask_path = run.path(DAMAGE_MASK)
    if not mask_path.is_file():
        raise ContractError(f"Damage mask copy not found: {mask_path}; run the damage step first")
    damage = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8)
    rgb = np.asarray(Image.open(run.image_path).convert("RGB"))
    if rgb.shape[:2] != damage.shape:
        raise ContractError(f"{mask_path}: size differs from {run.image_path}")
    size = (rgb.shape[1], rgb.shape[0])
    donor_masks = load_donor_masks(run, elements, records, size)

    report: dict[str, Any] = {"structure_id": run.structure_id, "views": {}}
    for view in getattr(opts, "views", ["restoration"]):
        if view not in VIEW_RULES:
            log.info("[complete] view %s: not implemented yet (T18)", view)
            report["views"][view] = {"status": "not_implemented"}
            continue
        out, labels, view_report = complete_view(view, rgb, damage, records, donor_masks, run.config)
        Image.fromarray(out).save(run.path(f"{view}_completed.png"))
        Image.fromarray(labels).save(run.path(f"{view}_source.png"))
        report["views"][view] = view_report
        tier_a = sum(1 for r in records if r["tier"] == "A")
        log.info("[complete] %s: %d target(s) completed (tier A records: %d), %d pixel(s) changed; "
                 "pixels outside the damage mask unchanged (checked)",
                 view, view_report["completed"], tier_a, view_report["pixels_changed"])
        for e in view_report["targets"]:
            if e["status"].startswith("completed"):
                al = e["alignment"]
                log.info("[complete]   %s %s donor %s%s: %d px, shift (%d, %d) scale %.2f, ncc %s -> %s, "
                         "%d damage px in box left unfilled", e["target_id"], e["tier"], e["donor_id"],
                         " flipped" if e["flipped"] else "", e["pixels_changed"], al["dx"], al["dy"],
                         al["scale"], al.get("score_before"), al.get("score"), e["damage_px_in_box_left_unfilled"])
    write_json(run.path(COMPLETION_FILE), report)
    run.meta["steps"].setdefault("complete", {})["views"] = {
        v: {k: r[k] for k in ("completed", "pixels_changed")} for v, r in report["views"].items() if "completed" in r
    }
