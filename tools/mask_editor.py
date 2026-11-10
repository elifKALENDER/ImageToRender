#!/usr/bin/env python3
"""Draw the damage mask of a building: white = damaged / missing, black = keep.

Copied from ``archive/inpaint/inpaint_v3.py`` (``create_mask_with_editor``,
``generate_auto_mask_proposal``, ``auto_mask_preview``) and made standalone: no torch, no diffusers,
no pipeline imports. Changes from the archived version: the window can start from an existing mask
or from the automatic proposal; texts are in English; the proposal settings come from the
``mask_proposal`` section of ``config.yaml`` instead of a dataclass.

Usage:
    python tools/mask_editor.py --structure s02_test_arch               # data/structures/<id>/image.png -> damage_mask.png
    python tools/mask_editor.py --structure s02_test_arch --propose     # start from the dark-region proposal
    python tools/mask_editor.py --image photo.jpg --out mask.png
    python tools/mask_editor.py --structure s02_test_arch --propose --no-gui   # write the proposal only

Mouse: left button paints damage, right button erases, wheel changes the brush size.
If the output file already exists, the window opens with it loaded (nothing is overwritten until
"Save" is pressed).

The proposal marks dark regions that are darker than their surroundings. Real doors, windows and
arch openings are dark too and cannot be told apart from losses automatically, so the proposal is
only a starting point: always correct it by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

NEAREST = Image.Resampling.NEAREST
LANCZOS = Image.Resampling.LANCZOS


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProposalSettings:
    dark_percentile: float
    max_gray: int
    local_contrast: int
    min_area_ratio: float
    max_component_ratio: float
    close_px: int
    expand_px: int
    drop_border_components: bool


@dataclass(frozen=True)
class EditorSettings:
    brush_px: int
    max_width: int
    max_height: int


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(cfg: dict[str, Any], config_path: Path) -> tuple[ProposalSettings, EditorSettings]:
    """Read the ``mask_proposal`` and ``mask_editor`` sections of config.yaml."""
    try:
        proposal = ProposalSettings(**cfg["mask_proposal"])
        editor = EditorSettings(**cfg["mask_editor"])
    except KeyError as exc:
        raise SystemExit(f"[ERROR] {config_path}: missing section {exc}") from exc
    except TypeError as exc:
        raise SystemExit(f"[ERROR] {config_path}: mask_proposal / mask_editor keys: {exc}") from exc
    return proposal, editor


# ---------------------------------------------------------------------------
# Mask helpers (no GUI)
# ---------------------------------------------------------------------------

def binarise(mask: Image.Image) -> Image.Image:
    """Hard 0/255 mask: grey pixels must not produce half-completed regions later."""
    arr = np.asarray(mask.convert("L"))
    return Image.fromarray(np.where(arr >= 128, 255, 0).astype(np.uint8), "L")


def mask_area_ratio(mask: Image.Image) -> float:
    return float((np.asarray(mask.convert("L")) > 127).mean())


def load_existing_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        raise SystemExit(f"[ERROR] {path}: size {mask.size} differs from the image size {size}")
    return binarise(mask)


def preview(photo: Image.Image, mask: Image.Image) -> Image.Image:
    """The mask drawn semi-transparent red over the photograph."""
    rgb = np.asarray(photo.convert("RGB"), dtype=np.float32).copy()
    sel = np.asarray(mask.resize(photo.size, NEAREST).convert("L")) > 127
    tint = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    rgb[sel] = rgb[sel] * 0.42 + tint * 0.58
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def propose_dark_regions(photo: Image.Image, cfg: ProposalSettings) -> tuple[Image.Image, dict[str, Any]]:
    """Dark interior regions clearly darker than their surroundings, as a damage proposal.

    The OpenCV path (connected components with area and border filters) is the archived default.
    OpenCV is optional here: without it the archived Pillow fallback runs, which has no component
    filter and therefore marks more.
    """
    gray = np.asarray(photo.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    percentile_threshold = float(np.percentile(gray, cfg.dark_percentile))
    dark_threshold = int(min(float(cfg.max_gray), percentile_threshold))
    sigma = max(3.0, min(h, w) / 65.0)
    diagnostics: dict[str, Any] = {
        "method": "dark_local_contrast_connected_components",
        "settings": cfg.__dict__,
        "percentile_threshold": round(percentile_threshold, 3),
        "resolved_dark_threshold": dark_threshold,
        "sigma": round(sigma, 3),
    }

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        diagnostics["path"] = "opencv"
        local = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        contrast = local.astype(np.int16) - gray.astype(np.int16)
        dark = gray <= dark_threshold
        very_dark = gray <= max(24, min(52, dark_threshold - 12))
        binary = (((dark & (contrast >= cfg.local_contrast)) | very_dark).astype(np.uint8)) * 255
        if cfg.close_px > 0:
            k = cfg.close_px * 2 + 1
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
        min_area = max(40, int(h * w * cfg.min_area_ratio))
        max_area = max(min_area + 1, int(h * w * cfg.max_component_ratio))
        kept = np.zeros((h, w), dtype=np.uint8)
        components = []
        rejected_border = 0
        for i in range(1, n):
            x, y, cw, ch, area = (int(v) for v in stats[i])
            if area < min_area or area > max_area:
                continue
            if cfg.drop_border_components and (x <= 0 or y <= 0 or x + cw >= w or y + ch >= h):
                rejected_border += 1
                continue
            kept[labels == i] = 255
            components.append({"x": x, "y": y, "w": cw, "h": ch, "area": area})
        if cfg.expand_px > 0:
            k = cfg.expand_px * 2 + 1
            kept = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
        proposal = Image.fromarray(kept, "L")
        diagnostics.update({
            "min_component_area_px": min_area,
            "max_component_area_px": max_area,
            "kept_component_count": len(components),
            "rejected_border_components": rejected_border,
            "components": components,
        })
    else:
        diagnostics["path"] = "pillow_fallback"
        local = np.asarray(Image.fromarray(gray, "L").filter(ImageFilter.GaussianBlur(sigma)), dtype=np.int16)
        contrast = local - gray.astype(np.int16)
        candidate = (gray <= dark_threshold) & (contrast >= cfg.local_contrast)
        proposal = Image.fromarray(candidate.astype(np.uint8) * 255, "L")
        for _ in range(max(0, cfg.close_px)):
            proposal = proposal.filter(ImageFilter.MaxFilter(3))
        for _ in range(max(0, cfg.close_px // 2)):
            proposal = proposal.filter(ImageFilter.MinFilter(3))
        for _ in range(max(0, cfg.expand_px)):
            proposal = proposal.filter(ImageFilter.MaxFilter(3))

    proposal = binarise(proposal)
    area = mask_area_ratio(proposal)
    diagnostics["area_ratio"] = round(area, 6)
    warnings = []
    if area < 0.0005:
        warnings.append("proposal is almost empty; the thresholds may not suit this photograph")
    if area > 0.55:
        warnings.append("proposal is very large; do not use it without correcting it by hand")
    diagnostics["warnings"] = warnings
    return proposal, diagnostics


# ---------------------------------------------------------------------------
# Editor window
# ---------------------------------------------------------------------------

def edit_mask(
    photo_path: Path,
    output_path: Path,
    settings: EditorSettings,
    initial: Image.Image | None = None,
) -> bool:
    """Open the drawing window. Returns True if a mask was saved, False on cancel.

    ``initial``: a 0/255 mask at the photograph's size to start from (existing mask or proposal).
    """
    import tkinter as tk
    from tkinter import messagebox

    from PIL import ImageTk

    original = Image.open(photo_path).convert("RGB")
    ow, oh = original.size
    scale = min(settings.max_width / ow, settings.max_height / oh, 1.0)
    dw, dh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
    display_photo = original.resize((dw, dh), LANCZOS)
    display_mask = (
        initial.resize((dw, dh), NEAREST) if initial is not None else Image.new("L", (dw, dh), 0)
    )

    root = tk.Tk()
    root.title(f"Damage mask: {photo_path.name}")
    root.configure(bg="#202020")
    state = {"brush": max(4, int(round(settings.brush_px * scale))), "saved": False,
             "painting": False, "erasing": False}

    tk.Label(
        root,
        text=("Red = damaged or missing, to be completed. Left button paints, right button erases, "
              "wheel changes the brush. Do not paint real door, window or arch openings."),
        fg="white", bg="#202020", font=("Segoe UI", 10), wraplength=max(520, dw), justify="left",
    ).pack(fill="x", padx=10, pady=(10, 6))
    canvas = tk.Canvas(root, width=dw, height=dh, highlightthickness=0, bg="black")
    canvas.pack(padx=10, pady=4)
    status_var = tk.StringVar()
    tk.Label(root, textvariable=status_var, fg="white", bg="#202020").pack(fill="x", padx=10, pady=(2, 4))
    holder: dict[str, Any] = {"image": None}

    def update_status() -> None:
        brush = max(1, int(round(state["brush"] / max(scale, 1e-6))))
        area = mask_area_ratio(display_mask) * 100
        status_var.set(f"Brush: about {brush}px | marked: {area:.1f} % of the image | "
                       "Left: add | Right: erase | Wheel: size")

    def redraw() -> None:
        red = Image.new("RGBA", (dw, dh), (255, 36, 36, 0))
        red.putalpha(display_mask.point(lambda p: 150 if p > 0 else 0))
        shown = Image.alpha_composite(display_photo.convert("RGBA"), red).convert("RGB")
        holder["image"] = ImageTk.PhotoImage(shown)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=holder["image"])

    def paint_at(x: int, y: int, value: int) -> None:
        r = state["brush"] / 2
        ImageDraw.Draw(display_mask).ellipse((x - r, y - r, x + r, y + r), fill=value)
        redraw()

    def press(value: int, key: str):
        def handler(event) -> None:
            state[key] = True
            paint_at(event.x, event.y, value)
        return handler

    def move(value: int, key: str):
        def handler(event) -> None:
            if state[key]:
                paint_at(event.x, event.y, value)
        return handler

    def release(key: str):
        def handler(_event) -> None:
            state[key] = False
            update_status()
        return handler

    def wheel(event) -> None:
        state["brush"] = max(4, min(180, state["brush"] + (3 if event.delta > 0 else -3)))
        update_status()

    def clear() -> None:
        display_mask.paste(0, (0, 0, dw, dh))
        redraw()
        update_status()

    def save() -> None:
        if not np.any(np.asarray(display_mask) > 0):
            messagebox.showwarning("Empty mask", "Nothing is marked. Paint the damaged areas with the left button.")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        binarise(display_mask.resize((ow, oh), NEAREST)).save(output_path)
        state["saved"] = True
        root.destroy()

    canvas.bind("<ButtonPress-1>", press(255, "painting"))
    canvas.bind("<B1-Motion>", move(255, "painting"))
    canvas.bind("<ButtonRelease-1>", release("painting"))
    canvas.bind("<ButtonPress-3>", press(0, "erasing"))
    canvas.bind("<B3-Motion>", move(0, "erasing"))
    canvas.bind("<ButtonRelease-3>", release("erasing"))
    canvas.bind("<MouseWheel>", wheel)

    buttons = tk.Frame(root, bg="#202020")
    buttons.pack(fill="x", padx=10, pady=(4, 10))
    tk.Button(buttons, text="Clear", command=clear).pack(side="left")
    tk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(6, 0))
    tk.Button(buttons, text="Save mask", command=save, font=("Segoe UI", 9, "bold")).pack(side="right")

    update_status()
    redraw()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return bool(state["saved"] and output_path.exists())


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def resolve_paths(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[Path, Path]:
    if args.structure:
        paths = cfg.get("paths") or {}
        folder = REPO_ROOT / paths.get("data_root", "data/structures") / args.structure
        if not folder.is_dir():
            raise SystemExit(f"[ERROR] structure folder not found: {folder}")
        image = Path(args.image) if args.image else folder / paths.get("image_file", "image.png")
        out = Path(args.out) if args.out else folder / paths.get("damage_mask_file", "damage_mask.png")
    else:
        if not (args.image and args.out):
            raise SystemExit("[ERROR] give --structure, or both --image and --out")
        image, out = Path(args.image), Path(args.out)
    if not image.is_file():
        raise SystemExit(f"[ERROR] image not found: {image}")
    return image, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw a damage mask (white = damaged).")
    parser.add_argument("--structure", help="folder name under data/structures/ (sets --image and --out)")
    parser.add_argument("--image", help="photograph")
    parser.add_argument("--out", help="mask PNG to write")
    parser.add_argument("--propose", action="store_true",
                        help="start from the automatic dark-region proposal (replaces a loaded mask)")
    parser.add_argument("--no-gui", action="store_true",
                        help="with --propose: write the proposal and a preview next to --out, open no window")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = read_config(config_path)
    image_path, out_path = resolve_paths(args, cfg)
    proposal_cfg, editor_cfg = load_settings(cfg, config_path)
    photo = Image.open(image_path).convert("RGB")

    initial = None
    if args.propose:
        initial, diagnostics = propose_dark_regions(photo, proposal_cfg)
        print(f"[propose] {diagnostics['path']}: {diagnostics['area_ratio'] * 100:.2f} % of the image, "
              f"{diagnostics.get('kept_component_count', '?')} region(s)")
        for warning in diagnostics["warnings"]:
            print(f"[propose] WARNING: {warning}")
        if args.no_gui:
            stem = out_path.with_suffix("")
            initial.save(f"{stem}_proposal.png")
            preview(photo, initial).save(f"{stem}_proposal_preview.png")
            with open(f"{stem}_proposal.json", "w", encoding="utf-8") as fh:
                json.dump(diagnostics, fh, indent=2)
            print(f"[propose] written: {stem}_proposal.png, _proposal_preview.png, _proposal.json")
            return 0
    elif args.no_gui:
        raise SystemExit("[ERROR] --no-gui needs --propose")
    elif out_path.is_file():
        initial = load_existing_mask(out_path, photo.size)
        print(f"[edit] loaded existing mask {out_path} ({mask_area_ratio(initial) * 100:.2f} % of the image)")

    if edit_mask(image_path, out_path, editor_cfg, initial):
        saved = Image.open(out_path)
        print(f"[saved] {out_path} ({mask_area_ratio(saved) * 100:.2f} % of the image)")
        return 0
    print("[cancelled] nothing written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
