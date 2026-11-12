"""Step 1 — detect architectural elements.

Multi-scale OWL-ViT (zero-shot, text prompts per class) proposes boxes; SAM turns each box into a
pixel mask. The two models are loaded one after the other and freed in between, so the step fits in
6 GB of VRAM.

Ported from ``archive/staged_pipeline_v8/01_stage1_multiscale_detection.py``: view tiling, the
per-prompt dynamic threshold, class-aware NMS and SAM box prompting are kept; class list and
thresholds now come from ``config.yaml``.

Output in the run folder:
    elements.json             one record per element (format: ELEMENT_KEYS)
    element_masks/<id>.png    single-channel 0/255 mask per element
    detect_diagnostics.json   per-view, per-prompt scores (for debugging thresholds)
    detect_views.png          the detection views drawn on the image
    detect_elements.png       boxes and masks drawn on the image
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from i2r.runio import ContractError, read_json, require_keys, write_json

if TYPE_CHECKING:
    import argparse

    from i2r.runio import Run

ELEMENTS_FILE = "elements.json"
MASK_DIR = "element_masks"

DOCUMENT_KEYS = ("structure_id", "image_size", "classes", "elements")
ELEMENT_KEYS = ("id", "class", "box", "score", "mask_file", "source")
VALID_SOURCES = ("detector", "expert")


# ---------------------------------------------------------------------------
# Format of elements.json (shared by every later step)
# ---------------------------------------------------------------------------

def validate_elements(doc: Any, where: str = ELEMENTS_FILE) -> None:
    """Check the elements.json contract. Raises ContractError naming the problem."""
    require_keys(doc, DOCUMENT_KEYS, where)
    width, height = doc["image_size"]
    classes = set(doc["classes"])
    seen: set[str] = set()
    if not isinstance(doc["elements"], list):
        raise ContractError(f"{where}: 'elements' must be a list")
    for index, element in enumerate(doc["elements"]):
        label = f"{where}: elements[{index}]"
        require_keys(element, ELEMENT_KEYS, label)
        if element["id"] in seen:
            raise ContractError(f"{label}: duplicate id {element['id']!r}")
        seen.add(element["id"])
        if element["class"] not in classes:
            raise ContractError(f"{label}: unknown class {element['class']!r}")
        if element["source"] not in VALID_SOURCES:
            raise ContractError(f"{label}: source must be one of {VALID_SOURCES}")
        box = element["box"]
        if (
            not isinstance(box, list) or len(box) != 4
            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
        ):
            raise ContractError(f"{label}: box {box} is not x1<x2, y1<y2 inside {width}x{height}")
        if not 0.0 <= float(element["score"]) <= 1.0:
            raise ContractError(f"{label}: score {element['score']} outside 0..1")


def load_elements(run_dir: Path) -> dict[str, Any]:
    doc = read_json(run_dir / ELEMENTS_FILE)
    validate_elements(doc, str(run_dir / ELEMENTS_FILE))
    return doc


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSpec:
    text: str
    element_class: str


@dataclass(frozen=True)
class View:
    view_id: str
    box: tuple[int, int, int, int]
    reason: str


def prompt_specs(classes_cfg: dict[str, Any]) -> list[PromptSpec]:
    specs = []
    for name, entry in classes_cfg.items():
        require_keys(entry, ("prompts",), f"config.yaml classes.{name}")
        specs.extend(PromptSpec(str(text), name) for text in entry["prompts"])
    if not specs:
        raise ContractError("config.yaml: no class prompts defined")
    return specs


def clamp_box(box: Iterable[float], width: int, height: int) -> list[int]:
    """Round to integer pixels and keep x1<x2<=width, y1<y2<=height."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = int(np.clip(round(x1), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    x2 = int(np.clip(round(x2), x1 + 1, width))
    y2 = int(np.clip(round(y2), y1 + 1, height))
    return [x1, y1, x2, y2]


def box_area_ratio(box: Sequence[float], width: int, height: int) -> float:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return area / max(1.0, float(width * height))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def evenly_spaced(full: int, crop: int, overlap: float) -> list[int]:
    if crop >= full:
        return [0]
    step = max(1, int(round(crop * (1.0 - overlap))))
    positions = list(range(0, full - crop + 1, step))
    if positions[-1] != full - crop:
        positions.append(full - crop)
    return positions


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_around(
    box: tuple[int, int, int, int], width: int, height: int, expand: float, min_side: int
) -> tuple[int, int, int, int]:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    side = min(max(max(box[2] - box[0], box[3] - box[1]) * expand, min_side), max(width, height))
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))
    if x1 < 0:
        x2, x1 = x2 - x1, 0
    if y1 < 0:
        y2, y1 = y2 - y1, 0
    if x2 > width:
        x1, x2 = x1 - (x2 - width), width
    if y2 > height:
        y1, y2 = y1 - (y2 - height), height
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def build_views(width: int, height: int, damage: np.ndarray | None, dcfg: dict[str, Any]) -> list[View]:
    views: list[View] = []
    if dcfg["use_full_image_view"]:
        views.append(View("full", (0, 0, width, height), "full_image"))

    damage_box = mask_bbox(damage) if damage is not None else None
    if dcfg["use_damage_context_view"] and damage_box is not None:
        min_side = int(round(min(width, height) * dcfg["damage_context_min_fraction"]))
        views.append(View(
            "damage_context",
            square_around(damage_box, width, height, dcfg["damage_context_expand"], min_side),
            "expanded_damage_context",
        ))

    side = max(int(dcfg["min_tile_px"]), min(int(round(min(width, height) * dcfg["tile_fraction_of_short_side"])), width, height))
    index = 0
    for y in evenly_spaced(height, side, dcfg["tile_overlap"]):
        for x in evenly_spaced(width, side, dcfg["tile_overlap"]):
            index += 1
            views.append(View(f"tile_{index:02d}", (x, y, x + side, y + side), "overlapping_tile"))

    unique: list[View] = []
    for view in views:
        if all(box_iou(view.box, kept.box) < float(dcfg["duplicate_view_iou"]) for kept in unique):
            unique.append(view)
    return unique[: max(1, int(dcfg["max_detection_views"]))]


def select_candidates(
    scores: list[float],
    labels: list[int],
    boxes: list[list[float]],
    view: View,
    specs: list[PromptSpec],
    dcfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per prompt, keep boxes scoring at least max(floor, best * ratio) — or one fallback box."""
    vw, vh = view.box[2] - view.box[0], view.box[3] - view.box[1]
    by_prompt: dict[int, list[tuple[float, list[float]]]] = {i: [] for i in range(len(specs))}
    for score, label, box in zip(scores, labels, boxes):
        ratio = box_area_ratio(box, vw, vh)
        if 0 <= label < len(specs) and dcfg["min_box_area_ratio"] <= ratio <= dcfg["max_box_area_ratio"]:
            by_prompt[int(label)].append((float(score), box))

    selected, diagnostics = [], []
    for index, spec in enumerate(specs):
        candidates = sorted(by_prompt[index], key=lambda item: item[0], reverse=True)
        best = candidates[0][0] if candidates else 0.0
        threshold = max(dcfg["absolute_score_floor"], best * dcfg["relative_score_ratio"])
        chosen = [c for c in candidates if c[0] >= threshold][: dcfg["max_candidates_per_prompt_per_view"]]
        quality = "primary"
        if not chosen and candidates and best >= dcfg["fallback_score_floor"]:
            chosen, quality = candidates[:1], "fallback"
        diagnostics.append({
            "prompt": spec.text,
            "class": spec.element_class,
            "best_score": round(best, 6),
            "threshold": round(threshold, 6),
            "selected": len(chosen),
            "quality": quality if chosen else "none",
        })
        for score, (lx1, ly1, lx2, ly2) in chosen:
            selected.append({
                "class": spec.element_class,
                "prompt": spec.text,
                "score": score,
                "box": [lx1 + view.box[0], ly1 + view.box[1], lx2 + view.box[0], ly2 + view.box[1]],
                "view": view.view_id,
                "quality": quality,
            })
    return selected, diagnostics


def check_class_merge(merge: dict[str, str], classes: Iterable[str]) -> None:
    known = set(classes)
    for source, target in merge.items():
        if source not in known or target not in known:
            raise ContractError(f"config detect.class_merge: {source!r} -> {target!r} uses an unknown class")
        if target in merge:
            raise ContractError(f"config detect.class_merge: {target!r} is both a target and a source")


def apply_class_merge(detections: list[dict[str, Any]], merge: dict[str, str]) -> int:
    """Relabel detections in place; the detector's own label is kept as ``detected_class``."""
    changed = 0
    for det in detections:
        det.setdefault("detected_class", det["class"])
        if det["class"] in merge:
            det["class"] = merge[det["class"]]
            changed += 1
    return changed


def max_area_ratio(element_class: str, config: dict[str, Any]) -> float:
    """Largest allowed box area (share of the image) for a class."""
    override = config["classes"].get(element_class, {}).get("max_area_ratio")
    return float(override if override is not None else config["detect"]["max_element_area_ratio"])


def split_oversized(
    detections: list[dict[str, Any]], width: int, height: int, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate detections whose box covers too much of the image. Returns (kept, dropped)."""
    kept, dropped = [], []
    for det in detections:
        box = clamp_box(det["box"], width, height)
        if box_area_ratio(box, width, height) > max_area_ratio(det["class"], config):
            dropped.append({"class": det["class"], "box": box, "score": round(float(det["score"]), 6)})
        else:
            kept.append(det)
    return kept, dropped


def class_aware_nms(
    detections: list[dict[str, Any]], iou_threshold: float, max_per_class: int, max_total: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        grouped.setdefault(det["class"], []).append(det)
    kept: list[dict[str, Any]] = []
    for group in grouped.values():
        accepted: list[dict[str, Any]] = []
        for det in sorted(group, key=lambda d: d["score"], reverse=True):
            if all(box_iou(det["box"], other["box"]) < iou_threshold for other in accepted):
                accepted.append(det)
            if len(accepted) >= max_per_class:
                break
        kept.extend(accepted)
    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:max_total]


# ---------------------------------------------------------------------------
# Models (torch imported lazily so the format helpers above need no GPU stack)
# ---------------------------------------------------------------------------

def _free_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_device(batch: dict[str, Any], device: str, fp16: bool) -> dict[str, Any]:
    import torch

    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if fp16 and value.dtype.is_floating_point:
                value = value.half()
        out[key] = value
    return out


def detect_owlvit(
    image: Image.Image, views: list[View], specs: list[PromptSpec], config: dict[str, Any],
    device: str, fp16: bool, log: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    dcfg = config["detect"]

    log.info("[detect] loading OWL-ViT %s on %s", dcfg["owlvit_model_id"], device)
    processor = OwlViTProcessor.from_pretrained(dcfg["owlvit_model_id"])
    model = OwlViTForObjectDetection.from_pretrained(dcfg["owlvit_model_id"]).to(device).eval()
    if fp16:
        model.half()

    texts = [spec.text for spec in specs]
    found: list[dict[str, Any]] = []
    view_diagnostics: list[dict[str, Any]] = []
    for number, view in enumerate(views, start=1):
        crop = image.crop(view.box)
        inputs = _to_device(dict(processor(text=[texts], images=crop, return_tensors="pt")), device, fp16)
        with torch.inference_mode():
            outputs = model(**inputs)
        target = torch.tensor([crop.size[::-1]], dtype=outputs.pred_boxes.dtype, device=outputs.pred_boxes.device)
        result = processor.post_process_object_detection(outputs=outputs, threshold=0.0, target_sizes=target)[0]
        selected, diagnostics = select_candidates(
            result["scores"].float().cpu().tolist(),
            result["labels"].cpu().tolist(),
            result["boxes"].float().cpu().tolist(),
            view, specs, dcfg,
        )
        found.extend(selected)
        view_diagnostics.append({"view": view.view_id, "box": list(view.box), "reason": view.reason,
                                 "selected": len(selected), "prompts": diagnostics})
        log.info("[detect] view %d/%d %-14s selected %d", number, len(views), view.view_id, len(selected))
        del inputs, outputs, result
        _free_cuda()

    del model, processor
    _free_cuda()

    # Merge classes the detector cannot tell apart, then drop oversized boxes. Both happen before
    # NMS: merging first lets NMS reduce two boxes on one element (one per original class) to one,
    # and dropping first keeps an oversized box from suppressing a correct one or using up the
    # per-class limit.
    merged = apply_class_merge(found, dcfg.get("class_merge") or {})
    if merged:
        log.info("[detect] class merge %s applied to %d candidate(s)", dcfg["class_merge"], merged)
    width, height = image.size
    found, dropped_large = split_oversized(found, width, height, config)
    if dropped_large:
        log.info("[detect] dropped %d oversized candidate(s)", len(dropped_large))

    # Cap each class before NMS so overlapping tiles cannot flood one class.
    capped: list[dict[str, Any]] = []
    per_class: dict[str, list[dict[str, Any]]] = {}
    for det in found:
        per_class.setdefault(det["class"], []).append(det)
    for group in per_class.values():
        capped.extend(sorted(group, key=lambda d: d["score"], reverse=True)[: dcfg["max_candidates_per_class_before_nms"]])

    final = class_aware_nms(capped, dcfg["nms_iou_threshold"], dcfg["max_detections_per_class"],
                            dcfg["max_total_detections"])
    diagnostics_extra = {"dropped_oversized": dropped_large, "class_merge_changed": merged}
    log.info("[detect] candidates %d, after class cap %d, after NMS %d", len(found), len(capped), len(final))
    return final, view_diagnostics, diagnostics_extra


def segment_sam(
    image: Image.Image, boxes: list[list[int]], dcfg: dict[str, Any], device: str, fp16: bool, log: Any
) -> list[np.ndarray]:
    if not boxes:
        return []
    import torch
    from transformers import SamModel, SamProcessor

    log.info("[detect] loading SAM %s on %s", dcfg["sam_model_id"], device)
    processor = SamProcessor.from_pretrained(dcfg["sam_model_id"])
    model = SamModel.from_pretrained(dcfg["sam_model_id"]).to(device).eval()
    if fp16:
        model.half()

    base = processor(images=image, return_tensors="pt")
    pixel_values = base["pixel_values"].to(device)
    if fp16:
        pixel_values = pixel_values.half()
    with torch.inference_mode():
        embeddings = model.get_image_embeddings(pixel_values)

    masks: list[np.ndarray] = []
    batch = max(1, int(dcfg["sam_batch_size"]))
    for start in range(0, len(boxes), batch):
        chunk = boxes[start:start + batch]
        prompt = processor(images=image, input_boxes=[[list(map(float, b)) for b in chunk]], return_tensors="pt")
        input_boxes = prompt["input_boxes"].to(device)
        if fp16:
            input_boxes = input_boxes.half()
        with torch.inference_mode():
            out = model(image_embeddings=embeddings, input_boxes=input_boxes, multimask_output=True)
        post = processor.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), base["original_sizes"].cpu(), base["reshaped_input_sizes"].cpu(),
            binarize=True,
        )[0]
        iou = out.iou_scores.float().cpu()[0]
        for i in range(len(chunk)):
            best = int(torch.argmax(iou[i]).item())
            masks.append(np.where(post[i, best].numpy() > 0, 255, 0).astype(np.uint8))
        del prompt, input_boxes, out, post, iou
        _free_cuda()

    del embeddings, pixel_values, base, model, processor
    _free_cuda()
    log.info("[detect] SAM masks: %d", len(masks))
    return masks


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def class_color(name: str) -> tuple[int, int, int]:
    """Stable, distinct colour per class name."""
    hue = (sum(ord(c) * 37 for c in name) % 180)
    hsv = np.uint8([[[hue, 200, 235]]])
    r, g, b = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


def draw_elements(rgb: np.ndarray, elements: list[dict[str, Any]], masks: dict[str, np.ndarray]) -> np.ndarray:
    out = rgb.copy()
    for element in elements:
        color = np.array(class_color(element["class"]), dtype=np.float32)
        mask = masks.get(element["id"])
        if mask is not None:
            sel = mask > 0
            out[sel] = (out[sel] * 0.65 + color * 0.35).astype(np.uint8)
    for element in elements:
        color = class_color(element["class"])
        x1, y1, x2, y2 = element["box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
        text = f"{element['id']} {element['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, top), (min(out.shape[1] - 1, x1 + tw + 4), top + th + 6), color, -1)
        cv2.putText(out, text, (x1 + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_views(rgb: np.ndarray, views: list[View]) -> np.ndarray:
    out = rgb.copy()
    for number, view in enumerate(views, start=1):
        x1, y1, x2, y2 = view.box
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 220, 40), 2)
        cv2.putText(out, f"{number}:{view.view_id}", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 220, 40), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "element"


def load_damage_mask(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = Image.open(path).convert("L")
    if mask.size != size:
        raise ContractError(f"{path}: size {mask.size} differs from the image size {size}")
    return np.where(np.asarray(mask) >= 128, 255, 0).astype(np.uint8)


def run_step(run: "Run", opts: "argparse.Namespace") -> None:
    import torch

    log = run.logger
    dcfg = run.config["detect"]
    specs = prompt_specs(run.config["classes"])
    check_class_merge(dcfg.get("class_merge") or {}, run.config["classes"])

    if not run.image_path.is_file():
        raise ContractError(f"Image not found: {run.image_path}")
    image = Image.open(run.image_path).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image)

    damage = load_damage_mask(run.damage_mask_path, image.size)
    if damage is None:
        log.warning("[detect] no damage mask at %s; the damage-context view is skipped", run.damage_mask_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda" and bool(dcfg["use_fp16_if_cuda"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log.info("[detect] image %dx%d, %d prompts in %d classes", width, height, len(specs), len(run.config["classes"]))

    views = build_views(width, height, damage, dcfg)
    Image.fromarray(draw_views(rgb, views)).save(run.path("detect_views.png"))

    detections, view_diagnostics, extra = detect_owlvit(image, views, specs, run.config, device, fp16, log)
    boxes = [clamp_box(det["box"], width, height) for det in detections]
    masks = segment_sam(image, boxes, dcfg, device, fp16, log)
    if len(masks) != len(boxes):
        raise ContractError(f"SAM returned {len(masks)} masks for {len(boxes)} boxes")

    mask_dir = run.path(MASK_DIR)
    mask_dir.mkdir(exist_ok=True)
    elements: list[dict[str, Any]] = []
    element_masks: dict[str, np.ndarray] = {}
    dropped_small = 0
    counters: dict[str, int] = {}
    for det, box, mask in zip(detections, boxes, masks):
        area = float((mask > 0).mean())
        if area < dcfg["min_mask_area_ratio"]:
            dropped_small += 1
            continue
        counters[det["class"]] = counters.get(det["class"], 0) + 1
        element_id = f"{slug(det['class'])}_{counters[det['class']]:02d}"
        mask_file = f"{MASK_DIR}/{element_id}.png"
        Image.fromarray(mask, mode="L").save(run.path(mask_file))
        element_masks[element_id] = mask
        elements.append({
            "id": element_id,
            "class": det["class"],
            "box": box,
            "score": round(float(det["score"]), 6),
            "mask_file": mask_file,
            "source": "detector",
            "detected_class": det.get("detected_class", det["class"]),
            "prompt": det["prompt"],
            "view": det["view"],
            "quality": det["quality"],
            "mask_area_ratio": round(area, 6),
        })

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0.0
    by_class = {name: sum(1 for e in elements if e["class"] == name) for name in run.config["classes"]}
    doc = {
        "structure_id": run.structure_id,
        "image_size": [width, height],
        "classes": list(run.config["classes"]),
        "elements": elements,
        "summary": {
            "total": len(elements),
            "by_class": by_class,
            "classes_with_no_detection": [name for name, n in by_class.items() if n == 0],
            "dropped_small_masks": dropped_small,
            "dropped_oversized_candidates": len(extra["dropped_oversized"]),
            "class_merge": dcfg.get("class_merge") or {},
            "views": len(views),
            "device": device,
            "peak_vram_gb": round(peak_gb, 3),
        },
    }
    validate_elements(doc)
    write_json(run.path(ELEMENTS_FILE), doc)
    write_json(run.path("detect_diagnostics.json"), {"views": view_diagnostics, **extra})
    Image.fromarray(draw_elements(rgb, elements, element_masks)).save(run.path("detect_elements.png"))

    log.info("[detect] elements: %d  %s", len(elements), {k: v for k, v in by_class.items() if v})
    log.info("[detect] classes with no detection: %s", doc["summary"]["classes_with_no_detection"] or "none")
    log.info("[detect] peak VRAM: %.2f GB", peak_gb)
    run.meta["steps"].setdefault("detect", {})["peak_vram_gb"] = round(peak_gb, 3)
