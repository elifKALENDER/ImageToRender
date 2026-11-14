"""Step 6 — render: the marked view images, the provenance record and the source maps.

Per view <v> that the complete step produced:
    <v>.png              the completed photograph, marked (§5.3): completed pixels get a slight tone shift
                         and a thin inner outline; tier B areas left open are hatched blue and labelled
                         "position implied, not restored"; every other damage pixel left open is hatched
                         and labelled "no evidence". A bar below the photograph carries the label
                         "AI proposal, not a restoration decision" and the legend.
    <v>_source_map.png   every changed pixel coloured by method and tier, with target IDs; the rest of
                         the photograph in faded grey. Same bar.
    <v>_ai_guess.png     only when the generative pass ran and first_construction.generative_display is
                         "separate" (T26): the view with the generated pixels, striped and dash-outlined
                         in the conjecture colour, under a banner. <v>.png then holds no generated pixel.
                         With "inline" the generated pixels stay in <v>.png with the same marking. They
                         are never outlined or tone-shifted like donor pixels.
Once per run:
    provenance.json      one entry per evidence target (§5.6) with its status in every view
    first_construction_conjecture_overlay.png
                         (first-construction view only) generative pixels in magenta over the faded
                         photograph, open damage hatched; empty of magenta while the generative fill is
                         off. Delivered with first_construction.png, never apart (§5.4).
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
from i2r.complete import GENERATIVE_LABEL_NAME, OUTSIDE_FACADE, load_facade_mask

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


def text_thickness(ts: float) -> int:
    return max(1, int(round(ts)))


def put_label(layer: np.ndarray, text: str, region: np.ndarray, colour: tuple[int, int, int],
              ts: float = 1.0) -> bool:
    """Text on a dark band at the point of ``region`` farthest from its edge, only when the whole band
    lies inside ``region`` (a label is never cut by the mask). Returns whether it was drawn.
    ``ts`` is restoration.text_scale (scaled with the image)."""
    scale = 0.38 * ts
    thick = text_thickness(ts)
    pad = max(3, int(round(4 * ts)))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    if not region.any():
        return False
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    x1, y1 = cx - tw // 2 - pad + 1, cy - th // 2 - pad
    x2, y2 = cx + tw // 2 + pad - 1, cy + th // 2 + pad
    h, w = region.shape
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h or not region[y1:y2 + 1, x1:x2 + 1].all():
        return False
    cv2.rectangle(layer, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.putText(layer, text, (cx - tw // 2, cy + th // 2), FONT, scale, colour, thick, cv2.LINE_AA)
    return True


def dash_pattern(shape: tuple[int, int], dash: int) -> np.ndarray:
    """Checkerboard of ``dash`` px cells: a line drawn through it comes out dashed in any direction."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = max(1, dash)
    return ((xx // d + yy // d) % 2) == 0


def mark_view(completed: np.ndarray, damage: np.ndarray, labels: np.ndarray, records: list[dict],
              statuses: dict[str, str], rcfg: dict, generative_label: int | None = None,
              gcfg: dict | None = None, facade: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """The marked photograph (same size) and pixel counts per marking.

    Open damage pixels outside ``facade`` (1 inside; the run's facade limit) are hatched in their own
    colour: they are not "no evidence" but "not part of the building" (T27).

    Pixels of ``generative_label`` are an AI guess, not a donor completion: they get no tone shift and no
    donor outline, but stripes and a dashed outline in the conjecture colour (``gcfg`` =
    config ``first_construction``, the ``guess_*`` keys)."""
    damage_b = damage > 0
    changed = labels > 0
    guess = (labels == int(generative_label)) if generative_label else np.zeros_like(changed)
    donor = changed & ~guess
    layer = completed.astype(np.float32).copy()

    # donor-completed pixels: tone shift towards white, then a thin inner outline per target
    t = float(rcfg["tone_shift"])
    layer[donor] = layer[donor] + (255.0 - layer[donor]) * t
    outline = np.zeros_like(changed)
    for value in np.unique(labels[donor]):
        outline |= inner_outline(labels == value, int(rcfg["outline_px"]))
    layer[outline] = rcfg["outline_color"]

    # generated pixels: stripes blended towards the conjecture colour and a dashed inner outline
    guess_marks = np.zeros_like(changed)
    if guess.any():
        if gcfg is None:
            raise ContractError("mark_view: a generative label needs the first_construction settings")
        stripes = guess & hatch_pattern(damage.shape, int(gcfg["guess_stripe_spacing_px"]),
                                        width=int(gcfg["guess_stripe_width_px"]))
        a = float(gcfg["guess_stripe_alpha"])
        layer[stripes] = layer[stripes] * (1.0 - a) + np.asarray(CONJECTURE_COLOUR, np.float32) * a
        dashes = inner_outline(guess, int(gcfg["guess_outline_px"])) & dash_pattern(damage.shape,
                                                                                   int(gcfg["guess_dash_px"]))
        layer[dashes] = CONJECTURE_COLOUR
        guess_marks = stripes | dashes

    # open areas: B targets not completed, then everything else still damaged
    open_b = np.zeros_like(damage_b)
    for r in records:
        if r["tier"] == "B" and statuses.get(r["target_id"]) != "completed":
            x1, y1, x2, y2 = r["target_box"]
            open_b[y1:y2, x1:x2] = True
    open_b &= damage_b & ~changed
    open_c = damage_b & ~changed & ~open_b
    open_out = open_c & (facade == 0) if facade is not None else np.zeros_like(open_c)
    open_c &= ~open_out
    spacing = int(rcfg["hatch_spacing_px"])
    # tier B: opposite diagonal, wider lines, so it reads over dark interiors and differs from C
    b_lines = hatch_pattern(damage.shape[::-1], spacing, width=int(rcfg["hatch_width_b_px"])).T[:, ::-1]
    layer[open_b & b_lines] = rcfg["hatch_color_b"]
    layer[open_c & hatch_pattern(damage.shape, spacing, offset=spacing // 2,
                                 width=int(rcfg["hatch_width_px"]))] = rcfg["hatch_color"]
    if open_out.any():
        layer[open_out & hatch_pattern(damage.shape, spacing, offset=spacing // 2,
                                       width=int(rcfg["hatch_width_px"]))] = rcfg["hatch_color_outside"]

    layer = np.clip(layer, 0, 255).astype(np.uint8)
    labelled = 0
    for r in records:
        x1, y1, x2, y2 = r["target_box"]
        in_box = np.zeros_like(damage_b)
        in_box[y1:y2, x1:x2] = True
        if r["tier"] == "B" and statuses.get(r["target_id"]) != "completed":
            labelled += put_label(layer, "B: implied", open_b & in_box, (255, 255, 255), float(rcfg["text_scale"]))
        elif r["kind"] == "uncovered_region":
            labelled += put_label(layer, "no evidence", open_c & in_box, (255, 255, 255), float(rcfg["text_scale"]))
        elif r["kind"] == OUTSIDE_FACADE:
            labelled += put_label(layer, "outside facade", open_out & in_box, (255, 255, 255),
                                  float(rcfg["text_scale"]))
    if guess.any():
        labelled += put_label(layer, "AI guess", guess, (255, 255, 255), float(rcfg["text_scale"]))

    out = completed.copy()
    out[damage_b] = layer[damage_b]          # markings live inside the damage mask only
    counts = {"completed_px": int(donor.sum()), "outline_px": int(outline.sum()),
              "ai_guess_px": int(guess.sum()), "ai_guess_marks_px": int(guess_marks.sum()),
              "open_tier_b_px": int(open_b.sum()), "open_no_evidence_px": int(open_c.sum()),
              "open_outside_facade_px": int(open_out.sum()), "labels_drawn": int(labelled)}
    return out, counts


def source_map(photo: np.ndarray, labels: np.ndarray, label_targets: dict[str, dict],
               generative_label: int | None = None, ts: float = 1.0) -> np.ndarray:
    """Faded grey photograph; changed pixels coloured by tier (method donor_transfer); target IDs."""
    g = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.repeat((g * 0.45 + 120)[..., None], 3, axis=2).astype(np.uint8)
    if generative_label is not None:
        out[labels == int(generative_label)] = CONJECTURE_COLOUR
    for value, record in label_targets.items():
        sel = labels == int(value)
        if not sel.any():
            continue
        out[sel] = TIER_COLOUR[record["tier"]]
        ys, xs = np.nonzero(sel)
        cv2.putText(out, record["target_id"], (int(xs.min()), max(int(10 * ts), int(ys.min()) - 2)), FONT,
                    0.33 * ts, (20, 20, 20), text_thickness(ts), cv2.LINE_AA)
    return out


def targets_overlay(photo: np.ndarray, elements: list[dict], records: list[dict], ts: float = 1.0) -> np.ndarray:
    out = photo.copy()
    for e in elements:
        if e.get("rejected"):
            continue
        x1, y1, x2, y2 = e["box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), class_color(e["class"]), text_thickness(ts), cv2.LINE_AA)
    for r in records:
        x1, y1, x2, y2 = r["target_box"]
        colour = TIER_COLOUR[r["tier"]]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), colour, 2 * text_thickness(ts), cv2.LINE_AA)
        cv2.putText(out, f"{r['target_id']} {r.get('rule') or r['tier']}", (x1 + 2, max(int(10 * ts), y1 - 3)), FONT,
                    0.33 * ts, colour, text_thickness(ts), cv2.LINE_AA)
    return out


def source_map_legend(generative: bool) -> list:
    """Legend of a source map; the generative item only when the map holds generated pixels."""
    items = [(TIER_COLOUR["A"], "donor transfer, tier A (direct)"),
             (TIER_COLOUR["B"], "donor transfer, tier B (positional)")]
    if generative:
        items.append((CONJECTURE_COLOUR, "generative (AI guess, no evidence)"))
    return items


def bar(width: int, rcfg: dict, view: str | None, items: list | None = None) -> np.ndarray:
    """Label and legend strip placed under every image (sizes follow restoration.text_scale)."""
    ts = float(rcfg["text_scale"])
    thick = text_thickness(ts)

    def u(v: float) -> int:
        return int(round(v * ts))

    out = np.full((u(22) + u(20) * 4, width, 3), 20, np.uint8)     # room for up to four legend lines
    title = rcfg["label"] + (f" | {view.replace('_', ' ')} view" if view else "")
    cv2.putText(out, title, (u(6), u(16)), FONT, 0.45 * ts, (255, 255, 255), thick, cv2.LINE_AA)
    items = items if items is not None else [
        ((240, 240, 240), "completed from a donor (outlined)"),
        (tuple(rcfg["hatch_color_b"]), "hatched: B, position implied, not restored"),
        (tuple(rcfg["hatch_color"]), "hatched: no evidence"),
    ]
    x, y = u(6), u(32)
    for colour, text in items:
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.36 * ts, thick)
        if x + u(16) + tw > width - 4:
            x, y = u(6), y + u(20)
        cv2.rectangle(out, (x, y), (x + u(10), y + u(10)), colour, -1)
        cv2.putText(out, text, (x + u(14), y + u(9)), FONT, 0.36 * ts, (220, 220, 220), thick, cv2.LINE_AA)
        x += u(24) + tw
    return out[: max(y + u(16), u(40))]


CONJECTURE_COLOUR = (200, 60, 200)


def conjecture_legend(rcfg: dict) -> list:
    return [
        (CONJECTURE_COLOUR, "generative fill (conjecture)"),
        (tuple(rcfg["hatch_color"]), "hatched: open, no evidence"),
    ]


def view_legend(rcfg: dict, counts: dict[str, int]) -> list:
    """Legend of one marked view: only the markings that occur in it (pixel count above 0)."""
    items = []
    if counts["completed_px"]:
        items.append(((240, 240, 240), "completed from a donor (outlined)"))
    if counts.get("ai_guess_px"):
        items.append((CONJECTURE_COLOUR, AI_GUESS_LEGEND))
    if counts["open_tier_b_px"]:
        items.append((tuple(rcfg["hatch_color_b"]), "hatched: B, position implied, not restored"))
    if counts["open_no_evidence_px"]:
        items.append((tuple(rcfg["hatch_color"]), "hatched: no evidence"))
    if counts.get("open_outside_facade_px"):
        items.append((tuple(rcfg["hatch_color_outside"]), OUTSIDE_FACADE_LEGEND))
    return items


OUTSIDE_FACADE_LEGEND = "hatched: outside the facade limit, not filled"
AI_GUESS_LEGEND = "striped: AI guess (generative), no evidence"
AI_GUESS_BANNER = "AI GUESS: no evidence in the photograph"
GENERATIVE_DISPLAYS = ("separate", "inline")


def banner(width: int, text: str, ts: float) -> np.ndarray:
    """Strip in the conjecture colour placed above the AI-guess image."""
    out = np.full((int(round(24 * ts)), width, 3), CONJECTURE_COLOUR, np.uint8)
    cv2.putText(out, text, (int(round(6 * ts)), int(round(17 * ts))), FONT, 0.5 * ts, (255, 255, 255),
                text_thickness(ts), cv2.LINE_AA)
    return out


def conjecture_overlay(photo: np.ndarray, damage: np.ndarray, labels: np.ndarray, report: dict,
                       rcfg: dict) -> tuple[np.ndarray, int]:
    """Faded photograph; generative pixels magenta; damage left open hatched. Returns (image, generative px)."""
    g = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.repeat((g * 0.45 + 120)[..., None], 3, axis=2).astype(np.uint8)
    generative = np.zeros(damage.shape, dtype=bool)
    gen = report.get("generative")
    if gen and gen.get("label"):
        generative |= labels == gen["label"]
    out[generative] = CONJECTURE_COLOUR
    open_damage = (damage > 0) & (labels == 0)
    spacing = int(rcfg["hatch_spacing_px"])
    out[open_damage & hatch_pattern(damage.shape, spacing, width=int(rcfg["hatch_width_px"]))] = rcfg["hatch_color"]
    n = int(generative.sum())
    if n == 0:
        put_label(out, "no generative fill", open_damage, (255, 255, 255), float(rcfg["text_scale"]))
    return out, n


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
            "area_px": r.get("area_px"),
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
    facade = load_facade_mask(run, evidence, damage.shape)
    fcfg = run.config["first_construction"]
    display = fcfg["generative_display"]
    if display not in GENERATIVE_DISPLAYS:
        raise ContractError(f"config first_construction.generative_display: {display!r}; "
                            f"expected one of {', '.join(GENERATIVE_DISPLAYS)}")

    rendered: dict[str, dict] = {}
    extras: dict[str, dict] = {}
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
        gen = report.get("generative")
        gen_label = gen["label"] if gen else None
        map_labels, map_gen = labels, gen_label
        if gen_label and display == "separate":
            # The view image holds no generated pixel: it is the image the same run gives with the fill
            # off. The guess is a second file, marked and with a banner (FAILURE_ANALYSIS §10, option 1).
            guess = labels == gen_label
            base, map_labels, map_gen = completed.copy(), labels.copy(), None
            base[guess] = photo[guess]
            map_labels[guess] = 0
            marked, counts = mark_view(base, damage, map_labels, records, statuses, rcfg, facade=facade)
            guess_image, guess_counts = mark_view(completed, damage, labels, records, statuses, rcfg, gen_label, fcfg,
                                                facade)
            guess_file = f"{view}_ai_guess.png"
            Image.fromarray(np.vstack([
                banner(photo.shape[1], AI_GUESS_BANNER, float(rcfg["text_scale"])),
                with_bar(guess_image, rcfg, f"{view} (AI guess)", view_legend(rcfg, guess_counts)),
            ])).save(run.path(guess_file))
            extras[view] = {"ai_guess_image": guess_file, "ai_guess_markings_px": guess_counts}
        else:
            marked, counts = mark_view(completed, damage, labels, records, statuses, rcfg, gen_label, fcfg,
                                                facade)
        if gen_label:
            extras.setdefault(view, {})["generative_display"] = display
        Image.fromarray(with_bar(marked, rcfg, view, view_legend(rcfg, counts))).save(run.path(f"{view}.png"))
        label_targets = {k: by_id[v] for k, v in report["labels"].items() if v != GENERATIVE_LABEL_NAME}
        Image.fromarray(with_bar(source_map(photo, map_labels, label_targets, map_gen, ts=float(rcfg["text_scale"])),
                                 rcfg, view, source_map_legend(map_gen is not None))).save(
            run.path(f"{view}_source_map.png"))
        if view == "first_construction":
            overlay, generative_px = conjecture_overlay(photo, damage, labels, report, rcfg)
            Image.fromarray(with_bar(overlay, rcfg, view, conjecture_legend(rcfg))).save(
                run.path("first_construction_conjecture_overlay.png"))
            counts["generative_px"] = generative_px   # generated in this view's completion (in the image only when inline)
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
                      "markings_px": rendered[v], **extras.get(v, {}),
                      **({"conjecture_overlay": "first_construction_conjecture_overlay.png"}
                         if v == "first_construction" else {})} for v in rendered},
        # Facade limit of this run; damage outside it is listed as targets of kind outside_facade (T27).
        "facade": evidence.get("facade"),
        "method_values": ["donor_transfer", "generative", "none"],
        "targets": entries,
        # One conjecture entry per view that ran the tier C pass: no donor, no evidence (§5.4, §5.6).
        "generative_fill": {v: {"method": "generative", "donor_id": None, "evidence": "conjecture",
                                **{k: rep["generative"][k] for k in rep["generative"] if k != "debug"}}
                            for v, rep in completion["views"].items() if rep.get("generative")},
    })
    Image.fromarray(with_bar(targets_overlay(photo, elements, records, float(rcfg["text_scale"])), rcfg, None)).save(run.path(OVERLAY))
    Image.fromarray(bar(photo.shape[1], rcfg, None)).save(run.path(LEGEND))
    summary = {v: sum(1 for e in entries if e["views"].get(v) == "completed") for v in rendered}
    log.info("[render] provenance.json: %d target(s); completed per view: %s", len(entries), summary)
    run.meta["steps"].setdefault("render", {})["completed_per_view"] = summary
