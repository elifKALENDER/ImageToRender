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
first-construction rule completes tiers A and B by donor transfer. With ``first_construction.generative``
(T21) one ControlNet inpaint pass (``generative_fill``) then fills the damage pixels the donors left open,
on a context crop of the donor-completed image; only those pixels are replaced, and they get one label
named ``generative_fill``. The restoration view never runs it. Debug images: ``generative_*.png``.

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


OUTSIDE_FACADE = "outside_facade"    # evidence record kind: damage pixels outside the facade limit (T27)
OUTSIDE_FACADE_REASON = "tier C: outside the facade limit; not part of the building, never filled"


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
    if record.get("kind") == OUTSIDE_FACADE:
        return False, OUTSIDE_FACADE_REASON
    return False, "tier C: no evidence"


def rule_first_construction(record: dict, config: dict) -> tuple[bool, str]:
    tier = record["tier"]
    if tier == "A":
        return True, "tier A: direct evidence"
    if tier == "B":
        return True, "tier B: positional evidence"
    if record.get("kind") == OUTSIDE_FACADE:
        return False, OUTSIDE_FACADE_REASON
    if config["first_construction"]["generative"]:
        return False, "tier C: no donor; covered by the single generative pass (provenance: generative_fill)"
    return False, "tier C: no evidence; generative fill off (first_construction.generative: false)"


VIEW_RULES: dict[str, Callable[[dict, dict], tuple[bool, str]]] = {
    "restoration": rule_restoration,
    "first_construction": rule_first_construction,
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


# ---------------------------------------------------------------------------
# Tier C generative fill (T21; first-construction view only)
# ---------------------------------------------------------------------------

GENERATIVE_LABEL_NAME = "generative_fill"


class GenerativeError(ContractError):
    """The generative models could not be loaded or run."""


def open_area(damage: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Damage pixels no donor filled: the area the generative pass may change."""
    return (damage > 0) & (labels == 0)


def context_box(area: np.ndarray, min_px: int) -> list[int]:
    """Bounding box of ``area`` grown (about its centre) to at least ``min_px`` per side, kept inside
    the image; the whole image side when the image is smaller."""
    ys, xs = np.nonzero(area)
    if len(xs) == 0:
        raise ContractError("context_box: empty area")
    height, width = area.shape

    def span(lo: int, hi: int, size: int) -> tuple[int, int]:
        need = min(size, max(hi - lo, min_px))
        start = (lo + hi) // 2 - need // 2
        start = max(0, min(start, size - need))
        return start, start + need

    x1, x2 = span(int(xs.min()), int(xs.max()) + 1, width)
    y1, y2 = span(int(ys.min()), int(ys.max()) + 1, height)
    return [x1, y1, x2, y2]


def generation_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """(w, h) with the long side ``max_side``, both multiples of 8 and at least 64."""
    scale = max_side / max(width, height)
    return (max(64, int(round(width * scale / 8)) * 8), max(64, int(round(height * scale / 8)) * 8))


def paste_area(out: np.ndarray, labels: np.ndarray, generated: np.ndarray, area: np.ndarray, label: int) -> int:
    """Copy ``generated`` into ``out`` only where ``area`` is set; label those pixels. Returns the count."""
    out[area] = generated[area]
    labels[area] = label
    return int(area.sum())


def neutralize_inside(control: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Control image with the masked pixels set to the median colour outside the mask (archived
    inpaint_v3: the collapse would otherwise steer the geometry)."""
    out = control.copy()
    outside = out[~mask]
    fill = np.median(outside, axis=0).astype(np.uint8) if outside.size else np.array([127, 127, 127], np.uint8)
    out[mask] = fill
    return out


def generative_fill(image: np.ndarray, area: np.ndarray, config: dict, log) -> tuple[np.ndarray, dict[str, Any]]:
    """One ControlNet inpaint pass (depth + lineart, SD 1.5) over a context crop around ``area``,
    set up as in archive/inpaint/inpaint_v3.py. Returns (full-size image whose ``area`` pixels are the
    generated ones, elsewhere a copy of ``image``; info)."""
    import gc
    import time

    import torch
    from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from PIL import ImageFilter
    from transformers import AutoImageProcessor, DPTForDepthEstimation

    fcfg, mcfg = config["first_construction"], config["models"]
    hub = {"cache_dir": mcfg["cache_dir"], "local_files_only": bool(mcfg["local_files_only"])}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    t0 = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    box = context_box(area, int(fcfg["context_min_px"]))
    x1, y1, x2, y2 = box
    crop, crop_area = image[y1:y2, x1:x2], area[y1:y2, x1:x2]
    gw, gh = generation_size(x2 - x1, y2 - y1, int(fcfg["generation_max_side_px"]))
    photo = Image.fromarray(crop).resize((gw, gh), Image.LANCZOS)
    mask_small = cv2.resize(crop_area.astype(np.uint8) * 255, (gw, gh), interpolation=cv2.INTER_NEAREST)
    mask_img = Image.fromarray(mask_small, "L")
    for _ in range(max(0, int(fcfg["mask_expand_px"]))):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    if float(fcfg["mask_feather_px"]) > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(float(fcfg["mask_feather_px"])))
    log.info("[complete] generative: crop %s (%d x %d) -> %d x %d, open area %d px", box, x2 - x1, y2 - y1,
             gw, gh, int(crop_area.sum()))

    try:
        # Conditioning from the donor-completed image (archived prepare_depth / prepare_lineart).
        depth_model = DPTForDepthEstimation.from_pretrained(mcfg["depth_estimator"], **hub).to(device)
        depth_proc = AutoImageProcessor.from_pretrained(mcfg["depth_estimator"], **hub)
        inputs = {k: v.to(device) for k, v in depth_proc(images=photo, return_tensors="pt").items()}
        with torch.no_grad():
            pred = depth_model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(gh, gw), mode="bicubic", align_corners=False)
        arr = pred.squeeze().float().cpu().numpy()
        arr -= arr.min()
        arr = (arr * 255.0 / (float(arr.max()) or 1.0)).clip(0, 255).astype(np.uint8)
        depth = np.repeat(arr[..., None], 3, axis=2)
        del depth_model, depth_proc, inputs, pred
        gc.collect()

        from controlnet_aux.lineart import LineartDetector

        detector = LineartDetector.from_pretrained(mcfg["annotator_repo"], cache_dir=mcfg["cache_dir"],
                                                   local_files_only=bool(mcfg["local_files_only"]))
        lineart = np.asarray(detector(photo).convert("RGB").resize((gw, gh), Image.LANCZOS))
        del detector
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if fcfg["neutralize_control_in_mask"]:
            hard = mask_small > 127
            depth, lineart = neutralize_inside(depth, hard), neutralize_inside(lineart, hard)

        # Pipeline (archived build_pipeline, without the IP-Adapter).
        controls = [ControlNetModel.from_pretrained(mcfg["depth_controlnet"], torch_dtype=dtype, **hub),
                    ControlNetModel.from_pretrained(mcfg["lineart_controlnet"], torch_dtype=dtype, **hub)]
        vae = AutoencoderKL.from_pretrained(mcfg["vae"], torch_dtype=dtype, **hub)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            mcfg["inpaint_base"], controlnet=controls, vae=vae, torch_dtype=dtype, safety_checker=None,
            use_safetensors=bool(mcfg["use_safetensors"]), **hub)
    except OSError as exc:
        raise GenerativeError(f"generative fill: a model is missing from {mcfg['cache_dir']} ({exc})") from exc
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    attention = "xformers"
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("[complete] generative: xformers unavailable (%s); attention slicing", exc)
        pipe.enable_attention_slicing("max")
        attention = "slicing"
    pipe.enable_vae_slicing()
    if device == "cuda" and mcfg["enable_cpu_offload"]:
        pipe.enable_model_cpu_offload()
        gen_device = "cuda"
    else:
        pipe = pipe.to(device)
        gen_device = device
    generator = torch.Generator(device=gen_device).manual_seed(int(fcfg["seed"]))
    result = pipe(
        prompt=fcfg["prompt"], negative_prompt=fcfg["negative_prompt"], image=photo, mask_image=mask_img,
        control_image=[Image.fromarray(depth), Image.fromarray(lineart)],
        controlnet_conditioning_scale=[float(fcfg["depth_scale"]), float(fcfg["lineart_scale"])],
        control_guidance_end=[float(v) for v in fcfg["guidance_end"]], strength=float(fcfg["strength"]),
        num_inference_steps=int(fcfg["steps"]), guidance_scale=float(fcfg["guidance_scale"]),
        height=gh, width=gw, generator=generator,
    ).images[0].convert("RGB")
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None
    del pipe, controls, vae
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    generated = cv2.resize(np.asarray(result), (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    full = image.copy()
    full[y1:y2, x1:x2] = generated
    info = {
        "method": "generative", "context_box": box, "generation_size": [gw, gh], "seed": int(fcfg["seed"]),
        "strength": float(fcfg["strength"]), "steps": int(fcfg["steps"]),
        "guidance_scale": float(fcfg["guidance_scale"]),
        "controlnet_scales": [float(fcfg["depth_scale"]), float(fcfg["lineart_scale"])],
        "guidance_end": [float(v) for v in fcfg["guidance_end"]],
        "models": {k: mcfg[k] for k in ("inpaint_base", "vae", "depth_controlnet", "lineart_controlnet",
                                         "depth_estimator", "annotator_repo")},
        "device": device, "attention": attention, "peak_vram_gb": peak,
        "seconds": round(time.perf_counter() - t0, 1),
        "debug": {"depth": depth, "lineart": lineart, "raw": np.asarray(result), "mask": np.asarray(mask_img)},
    }
    return full, info


def complete_view(view: str, rgb: np.ndarray, damage: np.ndarray, records: list[dict],
                  donor_masks: dict[str, np.ndarray], config: dict,
                  generate: Callable[[np.ndarray, np.ndarray, dict], tuple[np.ndarray, dict]] | None = None,
                  facade: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply one view's rule. Returns (completed image, uint16 label image, report).

    ``generate(image, open_area, config) -> (image, info)`` is the tier C pass; it runs only for the
    first-construction view when ``first_construction.generative`` is true. Open damage pixels outside
    ``facade`` (1 inside; the evidence step's facade limit) are never given to it."""
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
    generative = None
    if view == "first_construction" and config["first_construction"]["generative"] and generate is not None:
        area = open_area(damage, labels)
        excluded = 0
        if facade is not None:
            excluded = int((area & (facade == 0)).sum())
            area &= facade > 0
        if area.any():
            before = out.copy()
            filled, info = generate(out, area, config)
            label = len(label_names) + 1
            label_names[str(label)] = GENERATIVE_LABEL_NAME
            info["label"] = label
            info["pixels_changed"] = paste_area(out, labels, filled, area, label)
            info["excluded_outside_facade_px"] = excluded
            # the pass may change only the open area (accept criterion of T21)
            if np.any(np.any(before != out, axis=-1) & ~area):
                raise PreservationError(f"[{view}] generative fill changed pixels outside the open area")
            generative = info
    assert_preserved(rgb, out, damage, view)
    order = {r["target_id"]: i for i, r in enumerate(records)}
    results.sort(key=lambda e: order[e["target_id"]])
    report = {"rule": rule.__name__, "labels": label_names, "targets": results,
              "pixels_changed": int((labels > 0).sum()),
              "completed": sum(1 for e in results if e["status"] == "completed"),
              "generative": generative}
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


def load_facade_mask(run: "Run", evidence: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """The facade limit the evidence step wrote for this run (1 inside), or None when it set none."""
    name = (evidence.get("facade") or {}).get("mask_file")
    if not name:
        return None
    path = run.path(name)
    if not path.is_file():
        raise ContractError(f"evidence.json names the facade mask {name!r}, which is not in the run folder")
    mask = np.asarray(Image.open(path).convert("L")) >= 128
    if mask.shape != shape:
        raise ContractError(f"{path}: size differs from the damage mask")
    return mask.astype(np.uint8)


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
    facade = load_facade_mask(run, evidence, damage.shape)

    report: dict[str, Any] = {"structure_id": run.structure_id, "views": {}}
    for view in getattr(opts, "views", ["restoration"]):
        if view not in VIEW_RULES:
            log.info("[complete] view %s: no rule in VIEW_RULES", view)
            report["views"][view] = {"status": "not_implemented"}
            continue
        def generate(image: np.ndarray, area: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
            return generative_fill(image, area, config, log)

        out, labels, view_report = complete_view(view, rgb, damage, records, donor_masks, run.config, generate,
                                                 facade)
        gen = view_report.get("generative")
        if gen is not None:
            debug = gen.pop("debug")
            for name, arr in debug.items():
                Image.fromarray(arr).save(run.path(f"generative_{name}.png"))
            log.info("[complete] generative: %d px replaced in the open area, %.1f s, peak VRAM %s GB, %s",
                     gen["pixels_changed"], gen["seconds"], gen["peak_vram_gb"], gen["attention"])
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
    # Whether the generative pass ran, so a run folder with generated pixels can be told apart (T26).
    gen = {v: r["generative"] for v, r in report["views"].items() if r.get("generative")}
    run.meta["steps"]["complete"]["generative_fill"] = {
        "enabled": bool(run.config["first_construction"]["generative"]),
        "ran": bool(gen),
        "pixels_changed": {v: g["pixels_changed"] for v, g in gen.items()},
    }
