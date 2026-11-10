#!/usr/bin/env python3
"""Replace box masks with SAM masks (T16b).

Elements imported from the inventory editor have rectangular masks (``mask_kind: "box"``). A donor
transferred with a box mask carries the wall around the element and ends in straight edges. This tool
prompts SAM (the same model and code as the detect step) with each box, clips the result to the box and
keeps it when it fills at least ``refine_masks.min_fill`` of the box.

It never edits an existing run. It creates a new run of the same structure holding only the files of
the steps before damage (elements.json, element masks, editor/review/detect files), so the next command
starts cleanly from the damage step:

    python tools/refine_masks.py --run outputs/s01_test_church/20261110_171025
    python run.py --structure s01_test_church --from damage

Per element: the new mask is ``element_masks/<id>_sam.png``; ``mask_file`` points to it, ``mask_kind``
becomes ``sam_from_box`` and ``previous_mask_file`` keeps the box. Rejected elements are left alone.
Writes ``refine_masks_log.json`` and ``refine_masks_overlay.png`` (old box, new mask) in the new run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from i2r import runio  # noqa: E402
from i2r.detect import (ELEMENTS_FILE, MASK_DIR, class_color, load_elements,  # noqa: E402
                        validate_elements)
from i2r.runio import ContractError, read_json, write_json  # noqa: E402

LOG_FILE = "refine_masks_log.json"
OVERLAY = "refine_masks_overlay.png"
UPSTREAM_PREFIXES = ("editor_", "review_", "detect_", "elements_")


def copy_upstream(source: Path, target: Path) -> list[str]:
    """Copy only the files of the steps before damage."""
    copied = []
    for item in source.iterdir():
        name = item.name
        if name == ELEMENTS_FILE or name.startswith(UPSTREAM_PREFIXES) or name == MASK_DIR:
            if item.is_dir():
                shutil.copytree(item, target / name)
            else:
                shutil.copy2(item, target / name)
            copied.append(name)
    if ELEMENTS_FILE not in copied:
        raise ContractError(f"{source}: no {ELEMENTS_FILE}")
    return sorted(copied)


def clip_to_box(mask: np.ndarray, box: list[int]) -> np.ndarray:
    out = np.zeros_like(mask)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def decide(mask: np.ndarray, box: list[int], min_fill: float) -> tuple[bool, float]:
    """Keep the SAM mask when it fills at least ``min_fill`` of the box."""
    x1, y1, x2, y2 = box
    fill = float((mask[y1:y2, x1:x2] > 0).mean())
    return fill >= min_fill, fill


def refine(elements: list[dict], masks: list[np.ndarray], targets: list[dict], run_dir: Path,
           min_fill: float) -> list[dict[str, Any]]:
    """Apply SAM masks to ``targets`` (subset of ``elements``) in place; returns log rows."""
    rows = []
    for element, raw in zip(targets, masks):
        mask = clip_to_box(raw, element["box"])
        keep, fill = decide(mask, element["box"], min_fill)
        row = {"id": element["id"], "class": element["class"], "box": element["box"],
               "sam_fill_of_box": round(fill, 4), "kept": keep}
        if keep:
            mask_file = f"{MASK_DIR}/{element['id']}_sam.png"
            Image.fromarray(mask, mode="L").save(run_dir / mask_file)
            element["previous_mask_file"] = element["mask_file"]
            element["mask_file"] = mask_file
            element["mask_kind"] = "sam_from_box"
            row["mask_file"] = mask_file
        else:
            row["reason"] = f"SAM mask fills {fill:.2f} of the box, below refine_masks.min_fill {min_fill:.2f}"
        rows.append(row)
    return rows


def draw_overlay(rgb: np.ndarray, elements: list[dict], run_dir: Path) -> np.ndarray:
    import cv2

    out = rgb.copy()
    for e in elements:
        if e.get("mask_kind") != "sam_from_box":
            continue
        colour = np.array(class_color(e["class"]), dtype=np.float32)
        mask = np.asarray(Image.open(run_dir / e["mask_file"])) > 0
        out[mask] = (out[mask] * 0.55 + colour * 0.45).astype(np.uint8)
    for e in elements:
        if e.get("rejected"):
            continue
        x1, y1, x2, y2 = e["box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), class_color(e["class"]), 1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replace box masks with SAM masks in a new run.")
    parser.add_argument("--run", required=True, help="source run folder (outputs/<structure>/<timestamp>)")
    parser.add_argument("--config", default=str(runio.DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    source = Path(args.run).resolve()
    if not (source / ELEMENTS_FILE).is_file():
        print(f"[ERROR] {source / ELEMENTS_FILE} not found", file=sys.stderr)
        return 2
    structure_id = load_elements(source)["structure_id"]
    run = runio.create_run(structure_id, args.config)
    log = run.logger
    status = "failed"
    try:
        import torch

        from i2r.detect import segment_sam

        copied = copy_upstream(source, run.run_dir)
        doc = load_elements(run.run_dir)
        rcfg = run.config["refine_masks"]
        kinds = set(rcfg["kinds"])
        targets = [e for e in doc["elements"]
                   if not e.get("rejected") and e.get("mask_kind", "segmentation") in kinds]
        image = Image.open(run.image_path).convert("RGB")
        if list(image.size) != doc["image_size"]:
            raise ContractError(f"{run.image_path} is {image.size}, elements.json says {doc['image_size']}")
        log.info("[refine] %d of %d element(s) have mask kind %s", len(targets), len(doc["elements"]), sorted(kinds))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = device == "cuda" and bool(run.config["detect"]["use_fp16_if_cuda"])
        masks = segment_sam(image, [e["box"] for e in targets], run.config["detect"], device, fp16, log)
        rows = refine(doc["elements"], masks, targets, run.run_dir, float(rcfg["min_fill"]))

        validate_elements(doc)
        write_json(run.path(ELEMENTS_FILE), doc)
        write_json(run.path(LOG_FILE), {"source_run": runio.relative(source), "copied": copied,
                                        "min_fill": rcfg["min_fill"], "elements": rows})
        Image.fromarray(draw_overlay(np.asarray(image), doc["elements"], run.run_dir)).save(run.path(OVERLAY))
        kept = sum(r["kept"] for r in rows)
        run.meta["steps"]["refine_masks"] = {"status": "done", "source_run": runio.relative(source),
                                             "refined": kept, "kept_box": len(rows) - kept}
        log.info("[refine] SAM mask kept for %d, box kept for %d", kept, len(rows) - kept)
        for r in rows:
            if not r["kept"]:
                log.info("[refine]   %s: %s", r["id"], r["reason"])
        log.info("[refine] new run: %s  (next: python run.py --structure %s --from damage)",
                 runio.relative(run.run_dir), structure_id)
        status = "completed"
    finally:
        run.meta["status"] = status
        runio.save_meta(run)
        runio.close_logger(run)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
