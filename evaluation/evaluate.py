#!/usr/bin/env python3
"""Evaluation commands.

    python evaluation/evaluate.py detection --pred <run>/elements.json --ref <inventory.json>
        [--out <report.md>] [--iou 0.5]

`detection` compares detector output with a reference inventory (docs/ARCHITECTURE.md §7.3):
per-class recall, precision and mean box IoU of matched pairs.

Reference formats accepted:
    - elements.json of this pipeline (elements[].class, elements[].box)
    - the archived Stage 3B-v5 expert inventory (elements[].class, elements[].bbox_xyxy, status)

Matching is one-to-one and greedy by IoU: among all (reference, prediction) pairs whose classes are
compatible and whose IoU reaches the threshold, the highest-IoU pair is matched first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ImageToRender evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    det = sub.add_parser("detection", help="compare elements.json with a reference inventory")
    det.add_argument("--pred", required=True)
    det.add_argument("--ref", required=True)
    det.add_argument("--iou", type=float, default=0.5)
    det.add_argument("--out", default="")
    det.set_defaults(func=cmd_detection)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
