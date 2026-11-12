#!/usr/bin/env python3
"""Evaluation commands.

    python evaluation/evaluate.py detection --pred <run>/elements.json --ref <inventory.json>
        [--out <report.md>] [--iou 0.5]

    python evaluation/evaluate.py completion --cases outputs/synthetic/<structure>/<timestamp>
        [--config config.yaml] [--only <case_id>,...] [--no-perceptual]

`detection` compares detector output with a reference inventory (docs/ARCHITECTURE.md §7.3):
per-class recall, precision and mean box IoU of matched pairs.

`completion` runs the pipeline from the damage step (no review, both views) on every case made by
tools/synthetic_damage.py and computes the §7.1 metrics. The steps run in this process (the same
run_step functions run.py calls), one run folder per case under ``<cases>/eval_<timestamp>/``, with the
current config.yaml plus the structure's structure.yaml, pixel settings scaled to the image. Writes
``completion.csv`` and ``completion.md`` there. Definitions used (§7.1, made computable):
    recovered     partial/full: a tier A or B target of the hidden element's class with box IoU >=
                  synthetic.recovery_iou against the element's box (a tier C record does not count)
    tier          partial/full: tier of that best-IoU same-class target, else "C"; background: "A"/"B"
                  if an A/B target box covers synthetic.background_covered_share of the rectangle, else "C"
    filled        share of hidden pixels the view changed
    SSIM          scikit-image SSIM map (grey) averaged over the hidden pixels, view vs original photo;
                  ``ssim_open`` is the same for the case image (hidden area still painted) as a floor
    perceptual    1 - DINOv2 cosine between the hidden area's box in the view and in the original
    preservation  max absolute pixel difference outside the case damage mask (must be 0)
    abstention    background, restoration view: nothing filled
Donor validity needs the author; the donor of each case is listed for that.

Reference formats accepted:
    - elements.json of this pipeline (elements[].class, elements[].box)
    - the archived Stage 3B-v5 expert inventory (elements[].class, elements[].bbox_xyxy, status)

Matching is one-to-one and greedy by IoU: among all (reference, prediction) pairs whose classes are
compatible and whose IoU reaches the threshold, the highest-IoU pair is matched first.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from i2r.runio import ContractError, read_json, require_keys  # noqa: E402

# Reference classes that do not exist in config.yaml, mapped to the predicted classes that count as
# a correct detection. The archived inventory used "opening" for door-or-window openings.
CLASS_ALIASES: dict[str, set[str]] = {
    "opening": {"door", "window"},
}

# Reference statuses that are not visible evidence and must not be counted.
EXCLUDED_STATUSES = {"rejected", "candidate_needs_review"}


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def load_items(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return [{id, class, box, score}] from either supported format, plus the raw document."""
    doc = read_json(path)
    require_keys(doc, ("elements",), str(path))
    items = []
    for index, element in enumerate(doc["elements"]):
        where = f"{path}: elements[{index}]"
        if str(element.get("status", "")).lower() in EXCLUDED_STATUSES or element.get("rejected"):
            continue
        box_key = "box" if "box" in element else "bbox_xyxy"
        require_keys(element, ("id", "class", box_key), where)
        items.append({
            "id": str(element["id"]),
            "class": str(element["class"]),
            "box": [float(v) for v in element[box_key]],
            "score": float(element.get("score", element.get("ai_score", 0.0)) or 0.0),
            "status": element.get("status"),
        })
    return items, doc


def compatible(ref_class: str, pred_class: str) -> bool:
    return pred_class == ref_class or pred_class in CLASS_ALIASES.get(ref_class, set())


def match(
    refs: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float, class_aware: bool = True
) -> list[tuple[int, int, float]]:
    pairs = []
    for ri, ref in enumerate(refs):
        for pi, pred in enumerate(preds):
            if class_aware and not compatible(ref["class"], pred["class"]):
                continue
            iou = box_iou(ref["box"], pred["box"])
            if iou >= iou_threshold:
                pairs.append((iou, ri, pi))
    pairs.sort(reverse=True)
    used_r: set[int] = set()
    used_p: set[int] = set()
    matched = []
    for iou, ri, pi in pairs:
        if ri in used_r or pi in used_p:
            continue
        used_r.add(ri)
        used_p.add(pi)
        matched.append((ri, pi, iou))
    return matched


def evaluate_detection(
    refs: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float
) -> dict[str, Any]:
    matched = match(refs, preds, iou_threshold)
    ref_hit = {ri: (pi, iou) for ri, pi, iou in matched}
    pred_hit = {pi for _, pi, _ in matched}

    ref_classes = sorted({r["class"] for r in refs})
    pred_classes = sorted({p["class"] for p in preds})
    # Predicted classes the reference could confirm: the reference classes and their aliases.
    judged_pred_classes = set(ref_classes)
    for ref_class in ref_classes:
        judged_pred_classes |= CLASS_ALIASES.get(ref_class, set())

    per_ref_class = []
    for name in ref_classes:
        idx = [i for i, r in enumerate(refs) if r["class"] == name]
        hits = [ref_hit[i][1] for i in idx if i in ref_hit]
        per_ref_class.append({
            "class": name,
            "reference": len(idx),
            "found": len(hits),
            "recall": len(hits) / len(idx) if idx else None,
            "mean_iou": sum(hits) / len(hits) if hits else None,
        })

    per_pred_class = []
    for name in pred_classes:
        idx = [i for i, p in enumerate(preds) if p["class"] == name]
        hits = sum(1 for i in idx if i in pred_hit)
        judged = name in judged_pred_classes
        per_pred_class.append({
            "class": name,
            "predicted": len(idx),
            "correct": hits if judged else None,
            "precision": (hits / len(idx)) if judged and idx else None,
            "judged": judged,
        })

    judged_preds = [i for i, p in enumerate(preds) if p["class"] in judged_pred_classes]
    agnostic = match(refs, preds, iou_threshold, class_aware=False)
    lenient = match(refs, preds, 0.3)

    best_any = []
    for ref in refs:
        best = max(preds, key=lambda p: box_iou(ref["box"], p["box"]), default=None)
        best_any.append({
            "ref_id": ref["id"],
            "ref_class": ref["class"],
            "status": ref.get("status"),
            "matched": refs.index(ref) in ref_hit,
            "best_iou_any_class": round(box_iou(ref["box"], best["box"]), 3) if best else 0.0,
            "best_class": best["class"] if best else None,
        })

    return {
        "iou_threshold": iou_threshold,
        "reference_total": len(refs),
        "predicted_total": len(preds),
        "matched": len(matched),
        "recall": len(matched) / len(refs) if refs else None,
        "precision_on_judged_classes": (
            sum(1 for i in judged_preds if i in pred_hit) / len(judged_preds) if judged_preds else None
        ),
        "judged_predictions": len(judged_preds),
        "recall_class_agnostic": len(agnostic) / len(refs) if refs else None,
        "recall_at_iou_0_3": len(lenient) / len(refs) if refs else None,
        "mean_iou_matched": sum(i for *_, i in matched) / len(matched) if matched else None,
        "per_reference_class": per_ref_class,
        "per_predicted_class": per_pred_class,
        "per_reference_element": best_any,
    }


def fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{100 * value:.0f} %"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def detection_markdown(result: dict[str, Any], pred_path: str, ref_path: str) -> str:
    r = result
    lines = [
        f"Prediction: `{pred_path}`  ",
        f"Reference: `{ref_path}`  ",
        f"IoU threshold: {r['iou_threshold']}. Class aliases: "
        + ", ".join(f"{k} = {' or '.join(sorted(v))}" for k, v in CLASS_ALIASES.items()),
        "",
        "| Overall | Value |",
        "|---|---|",
        f"| Reference elements | {r['reference_total']} |",
        f"| Predicted elements | {r['predicted_total']} |",
        f"| Matched (class-aware) | {r['matched']} |",
        f"| Recall (class-aware) | {fmt(r['recall'], True)} |",
        f"| Recall, class ignored | {fmt(r['recall_class_agnostic'], True)} |",
        f"| Recall at IoU 0.3 (class-aware) | {fmt(r['recall_at_iou_0_3'], True)} |",
        f"| Precision on classes the reference covers | {fmt(r['precision_on_judged_classes'], True)} "
        f"({r['judged_predictions']} predictions judged) |",
        f"| Mean IoU of matched pairs | {fmt(r['mean_iou_matched'])} |",
        "",
        "| Reference class | Reference | Found | Recall | Mean IoU |",
        "|---|---|---|---|---|",
    ]
    for row in r["per_reference_class"]:
        lines.append(f"| {row['class']} | {row['reference']} | {row['found']} | {fmt(row['recall'], True)} | "
                     f"{fmt(row['mean_iou'])} |")
    lines += [
        "",
        "| Predicted class | Predicted | Correct | Precision |",
        "|---|---|---|---|",
    ]
    for row in r["per_predicted_class"]:
        note = "" if row["judged"] else " (not in reference)"
        lines.append(f"| {row['class']}{note} | {row['predicted']} | {fmt(row['correct'])} | "
                     f"{fmt(row['precision'], True)} |")
    lines += [
        "",
        "| Reference element | Class | Status | Matched | Best IoU (any class) | Best class |",
        "|---|---|---|---|---|---|",
    ]
    for row in r["per_reference_element"]:
        lines.append(f"| {row['ref_id']} | {row['ref_class']} | {row['status'] or '—'} | "
                     f"{'yes' if row['matched'] else 'no'} | {row['best_iou_any_class']:.2f} | {row['best_class']} |")
    return "\n".join(lines) + "\n"


def cmd_detection(args: argparse.Namespace) -> int:
    preds, pred_doc = load_items(Path(args.pred))
    refs, _ = load_items(Path(args.ref))
    result = evaluate_detection(refs, preds, args.iou)
    table = detection_markdown(result, args.pred, args.ref)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# completion (synthetic damage, §7.1)
# ---------------------------------------------------------------------------

COMPLETION_STEPS = ("damage", "evidence", "complete", "render")
VIEW_NAMES = ("restoration", "first_construction")


def match_hidden_element(gt: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict | None, float]:
    """Best-IoU target of the hidden element's class (uncovered regions have no class)."""
    best, best_iou = None, 0.0
    for r in records:
        if r.get("class") != gt["class"] or r["kind"] == "uncovered_region":
            continue
        iou = box_iou(gt["element_box"], r["target_box"])
        if iou > best_iou:
            best, best_iou = r, iou
    return best, best_iou


def box_cover_share(rect: Sequence[int], box: Sequence[int]) -> float:
    iw = max(0, min(rect[2], box[2]) - max(rect[0], box[0]))
    ih = max(0, min(rect[3], box[3]) - max(rect[1], box[1]))
    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    return iw * ih / area if area > 0 else 0.0


def assign_case(gt: dict[str, Any], records: list[dict[str, Any]], scfg: dict[str, Any]) -> dict[str, Any]:
    """Tier given to the hidden thing, recovered or not, and the target that decided it."""
    if gt["variant"] == "background":
        covering = [r for r in records if r["tier"] in ("A", "B")
                    and box_cover_share(gt["hidden_box"], r["target_box"]) >= float(scfg["background_covered_share"])]
        r = max(covering, key=lambda t: box_cover_share(gt["hidden_box"], t["target_box"]), default=None)
        return {"tier": r["tier"] if r else "C", "recovered": None, "target": r, "iou": None}
    r, iou = match_hidden_element(gt, records)
    hit = r is not None and iou >= float(scfg["recovery_iou"])
    tier = r["tier"] if hit else "C"
    return {"tier": tier, "recovered": bool(hit and tier in ("A", "B")), "target": r if hit else None,
            "iou": round(iou, 3) if r is not None else 0.0}


def masked_ssim(a: np.ndarray, b: np.ndarray, region: np.ndarray, pad: int = 8) -> float | None:
    """Mean of the SSIM map (grey) over ``region`` pixels, computed on the region's padded box."""
    from skimage.metrics import structural_similarity

    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    h, w = region.shape
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    ga = cv2.cvtColor(a[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    win = min(7, ga.shape[0], ga.shape[1])
    win -= (win + 1) % 2  # odd
    if win < 3:
        return None
    _, smap = structural_similarity(ga, gb, win_size=win, data_range=255, full=True)
    return float(smap[region[y1:y2, x1:x2] > 0].mean())


def view_metrics(original: np.ndarray, case_image: np.ndarray, completed: np.ndarray, hidden: np.ndarray,
                 case_damage: np.ndarray) -> dict[str, Any]:
    changed = np.any(completed != case_image, axis=2)
    n_hidden = int(hidden.sum())
    outside = case_damage == 0
    diff = np.abs(completed.astype(np.int16) - case_image.astype(np.int16)).max(axis=2)
    return {
        "filled": round(float((changed & (hidden > 0)).sum()) / n_hidden, 4) if n_hidden else None,
        "ssim": masked_ssim(completed, original, hidden),
        "preservation": int(diff[outside].max()) if outside.any() else 0,
    }


def hidden_crop_box(hidden: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(hidden)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def run_case(case_dir: Path, eval_dir: Path, cfg: dict[str, Any], structure_id: str) -> tuple[Path, str]:
    """Copy the seed into a fresh run folder and run the steps from damage. Returns (run folder, status)."""
    from argparse import Namespace

    from i2r import runio

    run_dir = eval_dir / case_dir.name
    shutil.copytree(case_dir / "seed", run_dir)
    with (run_dir / "config.yaml").open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    logger = runio.setup_logger(run_dir)
    run = runio.Run(structure_id, case_dir, run_dir, cfg, logger,
                    {"run_id": f"{structure_id}/synthetic/{case_dir.name}", "case_dir": runio.relative(case_dir),
                     "steps": {}, "git_commit": runio.git_commit()})
    opts = Namespace(views=list(VIEW_NAMES), no_review=True, view="both", from_step="damage")
    status = "completed"
    try:
        for name in COMPLETION_STEPS:
            importlib.import_module(f"i2r.{name}").run_step(run, opts)
            run.meta["steps"].setdefault(name, {})["status"] = "done"
    except ContractError as exc:
        status = f"failed: {exc}"
        logger.error("%s", exc)
    finally:
        run.meta["status"] = status
        runio.save_meta(run)
        runio.close_logger(run)
    return run_dir, status


def evaluate_case(case_dir: Path, run_dir: Path, status: str, original: np.ndarray, scfg: dict[str, Any],
                  embed: Any) -> dict[str, Any]:
    from PIL import Image

    gt = read_json(case_dir / "ground_truth.json")
    require_keys(gt, ("case_id", "variant", "expected_tier", "element_id", "class", "element_box", "hidden_box"),
                 str(case_dir / "ground_truth.json"))
    row: dict[str, Any] = {k: gt[k] for k in ("case_id", "variant", "element_id", "class", "expected_tier")}
    row["status"] = "completed" if status == "completed" else status
    if status != "completed":
        return row
    evidence = read_json(run_dir / "evidence.json")
    a = assign_case(gt, evidence["targets"], scfg)
    t = a["target"]
    row.update({
        "tier": a["tier"],
        "tier_correct": a["tier"] == gt["expected_tier"],
        "recovered": a["recovered"],
        "target_id": t["target_id"] if t else None,
        "target_iou": a["iou"],
        "rule": t.get("rule") if t else None,
        "donor_id": t.get("donor_id") if t else None,
        "similarity": t.get("similarity") if t else None,
        "reason": (t.get("reason") if t else None),
    })
    if t is None and gt["variant"] != "background":
        own = next((r for r in evidence["targets"] if r.get("element_id") == gt["element_id"]), None)
        row["reason"] = own.get("reason") if own else "no target of this class at the element"
    case_image = np.asarray(Image.open(case_dir / "image.png").convert("RGB"))
    hidden = (np.asarray(Image.open(case_dir / "hidden_mask.png").convert("L")) >= 128).astype(np.uint8)
    case_damage = (np.asarray(Image.open(case_dir / "damage_mask.png").convert("L")) >= 128).astype(np.uint8)
    row["ssim_open"] = masked_ssim(case_image, original, hidden)
    box = hidden_crop_box(hidden) if hidden.any() else None
    for view in VIEW_NAMES:
        completed = np.asarray(Image.open(run_dir / f"{view}_completed.png").convert("RGB"))
        m = view_metrics(original, case_image, completed, hidden, case_damage)
        for k, v in m.items():
            row[f"{view}_{k}"] = v
        if embed is not None and box is not None:
            x1, y1, x2, y2 = box
            size = int(embed.crop_px)
            crops = [cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
                     for img in (completed, original)]
            vec = embed(crops)
            row[f"{view}_perceptual"] = round(float(1.0 - np.dot(vec[0], vec[1])), 4)
    if gt["variant"] == "background":
        row["abstained"] = row["restoration_filled"] == 0
    return row


def mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in ("partial", "full", "background"):
        rs = [r for r in rows if r["variant"] == variant]
        done = [r for r in rs if r["status"] == "completed"]
        if not rs:
            continue
        s: dict[str, Any] = {"variant": variant, "cases": len(rs), "failed": len(rs) - len(done),
                             "expected_tier": rs[0]["expected_tier"]}
        s["tier_accuracy"] = mean([r["tier_correct"] for r in done])
        s["recovery"] = mean([r["recovered"] for r in done]) if variant != "background" else None
        s["abstention"] = mean([r["abstained"] for r in done]) if variant == "background" else None
        s["tiers"] = {k: sum(1 for r in done if r["tier"] == k) for k in ("A", "B", "C")}
        for view in VIEW_NAMES:
            s[f"{view}_filled"] = mean([r.get(f"{view}_filled") for r in done])
            s[f"{view}_ssim"] = mean([r.get(f"{view}_ssim") for r in done])
            s[f"{view}_perceptual"] = mean([r.get(f"{view}_perceptual") for r in done])
        s["ssim_open"] = mean([r.get("ssim_open") for r in done])
        s["preservation_max"] = max((max(r["restoration_preservation"], r["first_construction_preservation"])
                                     for r in done), default=None)
        out.append(s)
    return out


CSV_COLUMNS = ("case_id", "variant", "element_id", "class", "expected_tier", "status", "tier", "tier_correct",
               "recovered", "target_id", "target_iou", "rule", "donor_id", "similarity", "reason", "ssim_open",
               *(f"{v}_{k}" for v in VIEW_NAMES for k in ("filled", "ssim", "perceptual", "preservation")),
               "abstained")


def completion_markdown(rows: list[dict[str, Any]], summary: list[dict[str, Any]], header: dict[str, Any]) -> str:
    def f(v: Any, pct: bool = False) -> str:
        if isinstance(v, bool):
            return "yes" if v else "no"
        return fmt(v, pct)

    lines = [
        f"Cases: `{header['cases']}`  ",
        f"Base run: `{header['base_run']}`  ",
        f"Structures: {header['structures']} (n = {header['structures']}). Thresholds are the current config "
        "values (initial guesses until T20).  ",
        f"Config: `{header['config']}`, git {header['git']}",
        "",
        "R = restoration view, F = first-construction view. SSIM and perceptual distance are over the hidden "
        "pixels against the original photograph; \"SSIM open\" is the score of leaving the area painted.",
        "",
        "| Variant | Cases | Failed | Expected tier | Tier accuracy | Tiers A/B/C | Recovery | Abstention (R) | "
        "Filled R | Filled F | SSIM R | SSIM F | SSIM open | Perceptual R | Perceptual F | Preservation max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        t = s["tiers"]
        lines.append(
            f"| {s['variant']} | {s['cases']} | {s['failed']} | {s['expected_tier']} | {f(s['tier_accuracy'], True)} | "
            f"{t['A']}/{t['B']}/{t['C']} | {f(s['recovery'], True)} | {f(s['abstention'], True)} | "
            f"{f(s['restoration_filled'], True)} | {f(s['first_construction_filled'], True)} | "
            f"{f(s['restoration_ssim'])} | {f(s['first_construction_ssim'])} | {f(s['ssim_open'])} | "
            f"{f(s['restoration_perceptual'])} | {f(s['first_construction_perceptual'])} | {f(s['preservation_max'])} |")
    lines += [
        "",
        "| Case | Class | Expected | Tier | Recovered | IoU | Rule | Donor (author to judge) | Sim. | Filled R | "
        "Filled F | SSIM F | SSIM open | Pres. R/F | Reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] != "completed":
            lines.append(f"| {r['case_id']} | {r['class'] or '—'} | {r['expected_tier']} | failed | | | | | | | | | | | "
                         f"{r['status'][:80]} |")
            continue
        lines.append(
            f"| {r['case_id']} | {r['class'] or '—'} | {r['expected_tier']} | {r['tier']} | {f(r['recovered'])} | "
            f"{f(r['target_iou'])} | {r['rule'] or '—'} | {r['donor_id'] or '—'} | {f(r['similarity'])} | "
            f"{f(r['restoration_filled'], True)} | {f(r['first_construction_filled'], True)} | "
            f"{f(r['first_construction_ssim'])} | {f(r['ssim_open'])} | "
            f"{r['restoration_preservation']}/{r['first_construction_preservation']} | {(r['reason'] or '')[:70]} |")
    return "\n".join(lines) + "\n"


class _Embed:
    """DINOv2 from the evidence step, loaded once for the perceptual distance."""

    def __init__(self, ecfg: dict[str, Any]) -> None:
        import logging

        from i2r.evidence import Dinov2Embedder

        self.crop_px = int(ecfg["crop_px"])
        self.model = Dinov2Embedder(ecfg, logging.getLogger("i2r.evaluate"))

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        return self.model(crops)

    def close(self) -> None:
        self.model.close()


def cmd_completion(args: argparse.Namespace) -> int:
    import csv
    from datetime import datetime

    from PIL import Image

    from i2r import runio

    cases_dir = Path(args.cases).resolve()
    manifest = read_json(cases_dir / "manifest.json")
    require_keys(manifest, ("structure_id", "base_run", "cases"), str(cases_dir / "manifest.json"))
    structure_id = manifest["structure_id"]
    cfg, sdir, _ = runio.effective_config(args.config, structure_id)
    scfg = cfg["synthetic"]
    original = np.asarray(Image.open(sdir / cfg["paths"]["image_file"]).convert("RGB"))
    only = {c.strip() for c in args.only.split(",") if c.strip()} if args.only else None
    case_ids = [c["case_id"] for c in manifest["cases"] if only is None or c["case_id"] in only]
    if not case_ids:
        raise ContractError("no case selected")
    eval_dir = cases_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_dir.mkdir()

    runs = []
    for number, case_id in enumerate(case_ids, start=1):
        print(f"[completion] case {number}/{len(case_ids)}: {case_id}", flush=True)
        runs.append((cases_dir / case_id, *run_case(cases_dir / case_id, eval_dir, cfg, structure_id)))
    embed = None if args.no_perceptual else _Embed(cfg["evidence"])
    try:
        rows = [evaluate_case(c, r, s, original, scfg, embed) for c, r, s in runs]
    finally:
        if embed is not None:
            embed.close()
    summary = summarise(rows)
    git = runio.git_commit()
    header = {"cases": runio.relative(cases_dir), "base_run": manifest["base_run"], "structures": 1,
              "config": runio.relative(Path(args.config)),
              "git": f"{(git['hash'] or '?')[:7]}{' (dirty)' if git['dirty'] else ''}"}
    with (eval_dir / "completion.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    table = completion_markdown(rows, summary, header)
    (eval_dir / "completion.md").write_text(table, encoding="utf-8")
    runio.write_json(eval_dir / "completion_summary.json", {"header": header, "summary": summary, "rows": rows})
    print(table)
    print(f"[completion] written to {runio.relative(eval_dir)}")
    worst = max((s["preservation_max"] or 0) for s in summary) if summary else 0
    return 0 if worst == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ImageToRender evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    det = sub.add_parser("detection", help="compare elements.json with a reference inventory")
    det.add_argument("--pred", required=True)
    det.add_argument("--ref", required=True)
    det.add_argument("--iou", type=float, default=0.5)
    det.add_argument("--out", default="")
    det.set_defaults(func=cmd_detection)
    com = sub.add_parser("completion", help="run and score synthetic damage cases (§7.1)")
    com.add_argument("--cases", required=True, help="folder made by tools/synthetic_damage.py")
    com.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    com.add_argument("--only", default="", help="comma-separated case ids (quick checks)")
    com.add_argument("--no-perceptual", action="store_true", help="skip the DINOv2 perceptual distance")
    com.set_defaults(func=cmd_completion)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
