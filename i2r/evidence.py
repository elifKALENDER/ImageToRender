"""Step 4 — evidence, tier A: find an intact donor for every fragmented element.

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

Tier B and the tier C regions are T15. This module never edits elements.json.

Output in the run folder:
    evidence.json          one record per target, every candidate with its score or rejection reason
    evidence_overlay.png   donor -> target arrows (green tier A, grey tier C)
    evidence_sheet.png     per target: fragment, chosen donor as transferred, alternatives
"""
from __future__ import annotations

import gc
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

TIER_A, TIER_C = "A", "C"

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
            "confidence": best.similarity if best and reason is None else None,
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

GREEN, GREY, RED = (40, 220, 90), (170, 170, 170), (240, 60, 60)


def centre(box: list[int]) -> tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def draw_overlay(rgb: np.ndarray, damage: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    sel = damage > 0
    out[sel] = out[sel] * 0.6 + np.array(RED, dtype=np.float32) * 0.4
    out = np.clip(out, 0, 255).astype(np.uint8)
    for r in records:
        color = GREEN if r["tier"] == TIER_A else GREY
        x1, y1, x2, y2 = r["target_box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
        if r["donor_box"]:
            dx1, dy1, dx2, dy2 = r["donor_box"]
            cv2.rectangle(out, (dx1, dy1), (dx2 - 1, dy2 - 1), color, 1, cv2.LINE_AA)
            cv2.arrowedLine(out, centre(r["donor_box"]), centre(r["target_box"]), color, 1, cv2.LINE_AA, tipLength=0.04)
        cv2.putText(out, f"{r['target_id']} {r['tier']}", (x1 + 2, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def draw_sheet(rgb: np.ndarray, damage: np.ndarray, records: list[dict[str, Any]],
               elements: dict[str, dict], tile_h: int = 150) -> np.ndarray:
    """Rows: fragment (hole tinted) | chosen donor as transferred | alternatives."""
    rows = []
    for r in records:
        box = r["target_box"]
        tw, th = box[2] - box[0], box[3] - box[1]
        frag = crop(rgb, box).astype(np.float32).copy()
        hole = crop(damage, box) > 0
        frag[hole] = frag[hole] * 0.5 + np.array(RED, dtype=np.float32) * 0.5
        tiles = [(frag.astype(np.uint8), f"{r['target_id']} {r['element_id']} {r['tier']}")]
        chosen = ([{"donor_id": r["donor_id"], "similarity": r["similarity"], "flipped": r["flipped"]}]
                  if r["donor_id"] else []) + r["alternatives"]
        for c in chosen:
            img = donor_crop(rgb, damage, elements[c["donor_id"]]["box"], box, None, c["flipped"])
            tiles.append((img, f"{c['donor_id']} {c['similarity']:.2f}{' F' if c['flipped'] else ''}"))
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

    by_tier: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    for r in records:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        by_class.setdefault(r["class"], {}).setdefault(r["tier"], 0)
        by_class[r["class"]][r["tier"]] += 1
    flagged = [r["target_id"] for r in records if r["donor_expert_prototype"] is False]
    write_json(run.path(EVIDENCE_FILE), {
        "structure_id": run.structure_id,
        "image_size": doc["image_size"],
        "damage_mask_sha256": regions["damage_mask"]["sha256"],
        "model": {"id": ecfg["dino_model_id"], "pooling": ecfg["pooling"], "crop_px": ecfg["crop_px"],
                  "device": embedder.device, "fp16": embedder.fp16},
        "settings": {k: ecfg[k] for k in ("min_similarity", "min_visible_fraction", "size_tolerance", "alternatives_kept",
                                         "try_flipped_donor", "mask_donor_like_target")},
        "summary": {"targets": len(records), "by_tier": by_tier, "by_class": by_class,
                    "donor_not_marked_as_prototype_by_expert": flagged,
                    "tier_b_and_c_regions": "not computed yet (T15)"},
        "targets": records,
    })
    by_id = {e["id"]: e for e in elements}
    Image.fromarray(draw_overlay(rgb, damage, records)).save(run.path(OVERLAY))
    Image.fromarray(draw_sheet(rgb, damage, records, by_id)).save(run.path(SHEET))

    log.info("[evidence] %d fragmented element(s): %s", len(records), by_tier)
    for r in records:
        if r["tier"] == TIER_A:
            log.info("[evidence]   %s %-24s A  donor %s%s sim %.3f", r["target_id"], r["element_id"],
                     r["donor_id"], " (flipped)" if r["flipped"] else "", r["similarity"])
        else:
            log.info("[evidence]   %s %-24s C  %s", r["target_id"], r["element_id"], r["reason"])
    if flagged:
        log.info("[evidence] donors the expert did not mark usable_as_prototype: %s", flagged)
    run.meta["steps"].setdefault("evidence", {})["summary"] = {"targets": len(records), "by_tier": by_tier}
