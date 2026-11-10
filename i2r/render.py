"""Step 6 — render: the marked view images, the provenance record and the source maps.

Per view <v> that the complete step produced:
    <v>.png              the completed photograph, marked (§5.3): completed pixels get a slight tone shift
                         and a thin inner outline; tier B areas left open are hatched blue and labelled
                         "position implied, not restored"; every other damage pixel left open is hatched
                         and labelled "no evidence". A bar below the photograph carries the label
                         "AI proposal, not a restoration decision" and the legend.
    <v>_source_map.png   every changed pixel coloured by method and tier, with target IDs; the rest of
                         the photograph in faded grey. Same bar.
Once per run:
    provenance.json      one entry per evidence target (§5.6) with its status in every view
    targets_overlay.png  elements (thin, by class) and targets (by tier), with the bar
    legend.png           the bar alone

Every marking is drawn on a separate layer and copied into the photograph **only inside the damage
mask**, so in each view image every pixel outside the mask equals the photograph (tested). The label
and legend sit in the added bar, not on the photograph.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from i2r.detect import class_color, load_elements
from i2r.runio import ContractError, read_json, write_json

if TYPE_CHECKING:
    import argparse

    from i2r.runio import Run

PROVENANCE_FILE = "provenance.json"
OVERLAY = "targets_overlay.png"
LEGEND = "legend.png"
DAMAGE_MASK = "damage_mask.png"

TIER_COLOUR = {"A": (40, 200, 90), "B": (70, 130, 230), "C": (220, 60, 60)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Marking (pure)
# ---------------------------------------------------------------------------

def hatch_pattern(shape: tuple[int, int], spacing: int, offset: int = 0, width: int = 1) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + yy + offset) % max(2, spacing)) < max(1, width)


def inner_outline(region: np.ndarray, px: int) -> np.ndarray:
    """Pixels of ``region`` within ``px`` of its edge (drawn inside the region, never outside)."""
    if px <= 0:
        return np.zeros_like(region, dtype=bool)
    eroded = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=px)
    return region & (eroded == 0)


def put_label(layer: np.ndarray, text: str, region: np.ndarray, colour: tuple[int, int, int]) -> bool:
    """Text on a dark band at the point of ``region`` farthest from its edge, only when the whole band
    lies inside ``region`` (a label is never cut by the mask). Returns whether it was drawn."""
    scale = 0.38
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    if not region.any():
        return False
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    x1, y1 = cx - tw // 2 - 3, cy - th // 2 - 4
    x2, y2 = cx + tw // 2 + 3, cy + th // 2 + 4
    h, w = region.shape
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h or not region[y1:y2 + 1, x1:x2 + 1].all():
        return False
    cv2.rectangle(layer, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.putText(layer, text, (cx - tw // 2, cy + th // 2), FONT, scale, colour, 1, cv2.LINE_AA)
    return True


def mark_view(completed: np.ndarray, damage: np.ndarray, labels: np.ndarray, records: list[dict],
              statuses: dict[str, str], rcfg: dict) -> tuple[np.ndarray, dict[str, int]]:
    """The marked photograph (same size) and pixel counts per marking."""
    damage_b = damage > 0
    changed = labels > 0
    layer = completed.astype(np.float32).copy()

    # completed pixels: tone shift towards white, then a thin inner outline per target
    t = float(rcfg["tone_shift"])
    layer[changed] = layer[changed] + (255.0 - layer[changed]) * t
    outline = np.zeros_like(changed)
    for value in np.unique(labels[changed]):
        outline |= inner_outline(labels == value, int(rcfg["outline_px"]))
    layer[outline] = rcfg["outline_color"]

    # open areas: B targets not completed, then everything else still damaged
    open_b = np.zeros_like(damage_b)
    for r in records:
        if r["tier"] == "B" and statuses.get(r["target_id"]) != "completed":
            x1, y1, x2, y2 = r["target_box"]
            open_b[y1:y2, x1:x2] = True
    open_b &= damage_b & ~changed
    open_c = damage_b & ~changed & ~open_b
    spacing = int(rcfg["hatch_spacing_px"])
    # tier B: opposite diagonal, two pixels wide, so it reads over dark interiors and differs from C
    b_lines = hatch_pattern(damage.shape[::-1], spacing, width=2).T[:, ::-1]
    layer[open_b & b_lines] = rcfg["hatch_color_b"]
    layer[open_c & hatch_pattern(damage.shape, spacing, offset=spacing // 2)] = rcfg["hatch_color"]

    layer = np.clip(layer, 0, 255).astype(np.uint8)
    labelled = 0
    for r in records:
        x1, y1, x2, y2 = r["target_box"]
        in_box = np.zeros_like(damage_b)
        in_box[y1:y2, x1:x2] = True
        if r["tier"] == "B" and statuses.get(r["target_id"]) != "completed":
            labelled += put_label(layer, "B: implied", open_b & in_box, (255, 255, 255))
        elif r["kind"] == "uncovered_region":
            labelled += put_label(layer, "no evidence", open_c & in_box, (255, 255, 255))

    out = completed.copy()
    out[damage_b] = layer[damage_b]          # markings live inside the damage mask only
    counts = {"completed_px": int(changed.sum()), "outline_px": int(outline.sum()),
              "open_tier_b_px": int(open_b.sum()), "open_no_evidence_px": int(open_c.sum()),
              "labels_drawn": int(labelled)}
    return out, counts


def source_map(photo: np.ndarray, labels: np.ndarray, label_targets: dict[str, dict],
               generative_label: int | None = None) -> np.ndarray:
    """Faded grey photograph; changed pixels coloured by tier (method donor_transfer); target IDs."""
    g = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.repeat((g * 0.45 + 120)[..., None], 3, axis=2).astype(np.uint8)
    for value, record in label_targets.items():
        sel = labels == int(value)
        if not sel.any():
            continue
        out[sel] = TIER_COLOUR[record["tier"]]
        ys, xs = np.nonzero(sel)
        cv2.putText(out, record["target_id"], (int(xs.min()), max(10, int(ys.min()) - 2)), FONT, 0.33,
                    (20, 20, 20), 1, cv2.LINE_AA)
    return out


def targets_overlay(photo: np.ndarray, elements: list[dict], records: list[dict]) -> np.ndarray:
    out = photo.copy()
    for e in elements:
        if e.get("rejected"):
            continue
        x1, y1, x2, y2 = e["box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), class_color(e["class"]), 1, cv2.LINE_AA)
    for r in records:
        x1, y1, x2, y2 = r["target_box"]
        colour = TIER_COLOUR[r["tier"]]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), colour, 2, cv2.LINE_AA)
        cv2.putText(out, f"{r['target_id']} {r.get('rule') or r['tier']}", (x1 + 2, max(10, y1 - 3)), FONT, 0.33,
                    colour, 1, cv2.LINE_AA)
    return out


SOURCE_MAP_LEGEND = [
    (TIER_COLOUR["A"], "donor transfer, tier A (direct)"),
    (TIER_COLOUR["B"], "donor transfer, tier B (positional)"),
    ((200, 60, 200), "generative (conjecture)"),
]


def bar(width: int, rcfg: dict, view: str | None, items: list | None = None) -> np.ndarray:
    """Label and legend strip placed under every image."""
    out = np.full((22 + 20 * 4, width, 3), 20, np.uint8)     # room for up to four legend lines
    title = rcfg["label"] + (f" | {view.replace('_', ' ')} view" if view else "")
    cv2.putText(out, title, (6, 16), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    items = items or [
        ((240, 240, 240), "completed from a donor (outlined)"),
        (tuple(rcfg["hatch_color_b"]), "hatched: B, position implied, not restored"),
        (tuple(rcfg["hatch_color"]), "hatched: no evidence"),
    ]
    x, y = 6, 32
    for colour, text in items:
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.36, 1)
        if x + 16 + tw > width - 4:
            x, y = 6, y + 20
        cv2.rectangle(out, (x, y), (x + 10, y + 10), colour, -1)
        cv2.putText(out, text, (x + 14, y + 9), FONT, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        x += 24 + tw
    return out[: max(y + 16, 40)]


def with_bar(image: np.ndarray, rcfg: dict, view: str | None, items: list | None = None) -> np.ndarray:
    return np.vstack([image, bar(image.shape[1], rcfg, view, items)])


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def provenance_entries(records: list[dict], completion: dict) -> list[dict]:
    """One entry per evidence target (§5.6), with its status and pixel count in every view."""
    per_view = {view: {e["target_id"]: e for e in rep.get("targets", [])}
                for view, rep in completion["views"].items()}
    entries = []
    for r in records:
        views, reasons, pixels, alignment = {}, {}, {}, {}
        method = "none"
        for view, by_id in per_view.items():
            if not by_id:
                views[view] = "not_rendered"
                continue
            e = by_id.get(r["target_id"])
            if e is None:
                raise ContractError(f"completion.json view {view}: no entry for target {r['target_id']}")
            views[view] = "completed" if e["status"] == "completed" else "not_completed"
            reasons[view] = e.get("note") or e["rule_reason"]
            pixels[view] = e["pixels_changed"]
            if e["status"] == "completed":
                method = e["method"]
                alignment[view] = {k: e["alignment"].get(k) for k in ("dx", "dy", "scale", "score_before", "score")}
        entries.append({
            "target_id": r["target_id"],
            "kind": r["kind"],
            "rule": r.get("rule"),
            "element_id": r.get("element_id"),
            "class": r["class"],
            "tier": r["tier"],
            "target_box": r["target_box"],
            "donor_id": r["donor_id"],
            "donor_box": r["donor_box"],
            "similarity": r.get("similarity"),
            "confidence": r["confidence"],
            "alternatives": r.get("alternatives", []),
            "method": method,
            "flipped": r["flipped"],
            "evidence_reason": r.get("reason"),
            "views": views,
            "view_reasons": reasons,
            "pixels_changed": pixels,
            "alignment": alignment,
        })
    return entries


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def run_step(run: "Run", opts: "argparse.Namespace") -> None:
    log = run.logger
    rcfg = run.config["restoration"]
    evidence = read_json(run.path("evidence.json"))
    completion = read_json(run.path("completion.json"))
    for key, doc, name in (("targets", evidence, "evidence.json"), ("views", completion, "completion.json")):
        if key not in doc:
            raise ContractError(f"{run.path(name)}: missing key {key!r}")
    records = evidence["targets"]
    elements = load_elements(run.run_dir)["elements"]
    photo = np.asarray(Image.open(run.image_path).convert("RGB"))
    damage = (np.asarray(Image.open(run.path(DAMAGE_MASK)).convert("L")) >= 128).astype(np.uint8)
    by_id = {r["target_id"]: r for r in records}

    rendered = {}
    for view, report in completion["views"].items():
        if "targets" not in report:
            log.info("[render] view %s: nothing to render (%s)", view, report.get("status"))
            continue
        completed = np.asarray(Image.open(run.path(f"{view}_completed.png")).convert("RGB"))
        labels = np.asarray(Image.open(run.path(f"{view}_source.png")))
        if completed.shape != photo.shape or labels.shape != damage.shape:
            raise ContractError(f"[render] {view}: completed or source image size differs from the photograph")
        unexplained = int((np.any(completed != photo, axis=-1) & (labels == 0)).sum())
        if unexplained:
            raise ContractError(f"[render] {view}: {unexplained} changed pixel(s) have no source label")
        statuses = {e["target_id"]: e["status"] for e in report["targets"]}
        marked, counts = mark_view(completed, damage, labels, records, statuses, rcfg)
        Image.fromarray(with_bar(marked, rcfg, view)).save(run.path(f"{view}.png"))
        label_targets = {k: by_id[v] for k, v in report["labels"].items()}
        Image.fromarray(with_bar(source_map(photo, labels, label_targets), rcfg, view,
                                        SOURCE_MAP_LEGEND)).save(
            run.path(f"{view}_source_map.png"))
        rendered[view] = counts
        log.info("[render] %s: %s.png and %s_source_map.png | %s", view, view, view, counts)

    entries = provenance_entries(records, completion)
    write_json(run.path(PROVENANCE_FILE), {
        "statement": rcfg["label"],
        "structure_id": run.structure_id,
        "run_id": run.meta.get("run_id"),
        "git_commit": run.meta.get("git_commit"),
        "source_image": {"file": str(run.image_path.name), "sha256": sha256(run.image_path)},
        "damage_mask": {"file": DAMAGE_MASK, "sha256": sha256(run.path(DAMAGE_MASK))},
        "views": {v: {"image": f"{v}.png", "source_map": f"{v}_source_map.png",
                      "source_labels": f"{v}_source.png", "labels": completion["views"][v]["labels"],
                      "markings_px": rendered[v]} for v in rendered},
        "method_values": ["donor_transfer", "generative", "none"],
        "targets": entries,
    })
    Image.fromarray(with_bar(targets_overlay(photo, elements, records), rcfg, None)).save(run.path(OVERLAY))
    Image.fromarray(bar(photo.shape[1], rcfg, None)).save(run.path(LEGEND))
    summary = {v: sum(1 for e in entries if e["views"].get(v) == "completed") for v in rendered}
    log.info("[render] provenance.json: %d target(s); completed per view: %s", len(entries), summary)
    run.meta["steps"].setdefault("render", {})["completed_per_view"] = summary
