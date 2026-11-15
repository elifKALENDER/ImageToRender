#!/usr/bin/env python3
"""Box intact elements on another photograph of the same building, to be used as donors (T37).

Where the photograph of a building shows no intact element of a class (no base, no whole arch seen from
the front), another photograph of the same building may. This tool stores such elements in

    data/structures/<id>/reference_donors.json       boxes (tracked)
    data/structures/<id>/references/<name>.png       the photograph (local, not in git)

and the evidence step of the next run offers them as donors (``i2r/references.py``). A placement from a
reference donor is always a guess for the expert.

Usage:
    python tools/reference_donors.py --structure <id> --photo <image file> --name other_side      # window
    python tools/reference_donors.py --structure <id> --name other_side                           # window again
    python tools/reference_donors.py --structure <id> --list
    python tools/reference_donors.py --structure <id> --name other_side --add base 120 600 190 650 [--note "..."]
    python tools/reference_donors.py --structure <id> --remove base_01
    python tools/reference_donors.py --structure <id> --masks        # SAM mask per box (GPU), instead of the box
Then:
    python run.py --structure <id> --from evidence --no-review

Window: choose the class, drag a rectangle around one intact element. Right click on a box removes it.
Closing the window saves. Without ``--masks`` the whole box is copied (wall included); with it only the
element SAM finds in the box, when that fills at least ``refine_masks.min_fill`` of the box. A mask
belongs to the box it was made for: after redrawing a box run ``--masks`` again.
torch is imported only for ``--masks``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from i2r import runio  # noqa: E402
from i2r.references import REFERENCE_FILE, read_reference_file  # noqa: E402
from i2r.runio import ContractError, write_json  # noqa: E402

PHOTO_DIR = "references"
MIN_BOX_PX = 4                       # a smaller drag is a slip of the hand, not a box
ABOUT = ("Intact elements boxed on other photographs of the same building, offered as donors "
         "(tools/reference_donors.py, i2r/references.py). Boxes are in pixels of the named photo.")


def load(structure_dir: Path, classes: list[str]) -> dict[str, Any]:
    return read_reference_file(structure_dir / REFERENCE_FILE, classes)


def save(structure_dir: Path, doc: dict[str, Any]) -> None:
    write_json(structure_dir / REFERENCE_FILE, {"structure_id": structure_dir.name, "about": ABOUT,
                                                "photos": doc["photos"], "donors": doc["donors"]})


def add_photo(structure_dir: Path, doc: dict[str, Any], name: str, source: Path, note: str = "") -> None:
    """Copy ``source`` into the building's references folder as <name>.png and register it."""
    if not source.is_file():
        raise ContractError(f"Photograph not found: {source}")
    folder = structure_dir / PHOTO_DIR
    folder.mkdir(exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(folder / f"{name}.png")
    doc["photos"][name] = {"file": f"{PHOTO_DIR}/{name}.png", "original_name": source.name,
                           "note": note or doc["photos"].get(name, {}).get("note", "")}


def photo_path(structure_dir: Path, doc: dict[str, Any], name: str) -> Path:
    if name not in doc["photos"]:
        known = ", ".join(doc["photos"]) or "none"
        raise ContractError(f"No reference photo named {name!r} (known: {known}); add it with --photo <file>")
    return structure_dir / doc["photos"][name]["file"]


def add_donor(structure_dir: Path, doc: dict[str, Any], name: str, element_class: str, box: list[int],
              classes: list[str], note: str = "") -> dict[str, Any]:
    if element_class not in classes:
        raise ContractError(f"Unknown class {element_class!r}. Known: {', '.join(classes)}")
    with Image.open(photo_path(structure_dir, doc, name)) as image:
        width, height = image.size
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2, y1, y2 = max(0, min(x1, x2)), min(width, max(x1, x2)), max(0, min(y1, y2)), min(height, max(y1, y2))
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        raise ContractError(f"Box {box} is smaller than {MIN_BOX_PX} px inside the photo ({width} x {height})")
    used = {d["id"] for d in doc["donors"]}
    number = 1
    while f"{element_class}_{number:02d}" in used:
        number += 1
    donor = {"id": f"{element_class}_{number:02d}", "class": element_class, "photo": name, "box": [x1, y1, x2, y2],
             "note": note}
    doc["donors"].append(donor)
    return donor


def remove_donor(doc: dict[str, Any], donor_id: str) -> None:
    if not any(d["id"] == donor_id for d in doc["donors"]):
        raise ContractError(f"No reference donor {donor_id!r}. Known: {', '.join(d['id'] for d in doc['donors']) or 'none'}")
    doc["donors"] = [d for d in doc["donors"] if d["id"] != donor_id]


def make_masks(structure_dir: Path, doc: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """SAM mask (prompted with the box, clipped to it) for every donor; returns one line per donor."""
    import logging

    import numpy as np
    import torch

    from i2r.detect import segment_sam

    log = logging.getLogger("reference_donors")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda" and bool(cfg["detect"]["use_fp16_if_cuda"])
    min_fill = float(cfg["refine_masks"]["min_fill"])
    folder = structure_dir / PHOTO_DIR / "masks"
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in doc["photos"]:
        donors = [d for d in doc["donors"] if d["photo"] == name]
        if not donors:
            continue
        image = Image.open(photo_path(structure_dir, doc, name)).convert("RGB")
        masks = segment_sam(image, [d["box"] for d in donors], cfg["detect"], device, fp16, log)
        for d, mask in zip(donors, masks):
            x1, y1, x2, y2 = d["box"]
            clipped = np.zeros(mask.shape, np.uint8)
            clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2] > 0
            fill = float(clipped[y1:y2, x1:x2].mean())
            if fill >= min_fill:
                Image.fromarray(clipped * 255, mode="L").save(folder / f"{d['id']}.png")
                d["mask_file"], d["mask_box"] = f"{PHOTO_DIR}/masks/{d['id']}.png", list(d["box"])
                lines.append(f"{d['id']}: SAM mask kept, fills {fill:.2f} of the box")
            else:
                d.pop("mask_file", None)
                d.pop("mask_box", None)
                lines.append(f"{d['id']}: box kept, the SAM mask fills only {fill:.2f} of it (min_fill {min_fill:.2f})")
    return lines


def table(doc: dict[str, Any]) -> str:
    rows = [f"photo {name}: {p['file']}" + (f"  ({p['note']})" if p.get("note") else "") for name, p in doc["photos"].items()]
    rows += [f"  {d['id']:16s} {d['class']:10s} {d['photo']:14s} {d['box']}  "
             + ("mask" if d.get("mask_file") and d.get("mask_box") == d["box"] else "box ")
             + (f"  {d['note']}" if d.get("note") else "") for d in doc["donors"]]
    return "\n".join(rows) if rows else "no reference donors"


def window(structure_dir: Path, doc: dict[str, Any], name: str, classes: list[str]) -> None:
    import tkinter as tk
    from tkinter import ttk

    from PIL import ImageTk

    image = Image.open(photo_path(structure_dir, doc, name)).convert("RGB")
    root = tk.Tk()
    root.title(f"Reference donors: {structure_dir.name} / {name}")
    zoom = max(0.25, min(3.0, (root.winfo_screenheight() - 200) / image.height, (root.winfo_screenwidth() - 80) / image.width))
    shown = ImageTk.PhotoImage(image.resize((round(image.width * zoom), round(image.height * zoom))))
    bar = tk.Frame(root)
    bar.pack(fill="x", padx=6, pady=4)
    tk.Label(bar, text="Class:").pack(side="left")
    chosen = ttk.Combobox(bar, state="readonly", values=classes, width=14)
    chosen.set("base" if "base" in classes else classes[0])
    chosen.pack(side="left", padx=4)
    status = tk.Label(bar, anchor="w", text="drag: new box   right click on a box: remove   close the window: save")
    status.pack(side="left", padx=10)
    canvas = tk.Canvas(root, width=shown.width(), height=shown.height(), cursor="tcross")
    canvas.pack(padx=6, pady=(0, 6))
    canvas.create_image(0, 0, anchor="nw", image=shown)
    drag: dict[str, Any] = {"start": None, "item": None}

    def redraw() -> None:
        canvas.delete("box")
        for d in doc["donors"]:
            if d["photo"] != name:
                continue
            x1, y1, x2, y2 = (v * zoom for v in d["box"])
            canvas.create_rectangle(x1, y1, x2, y2, outline="#ffd400", width=2, tags="box")
            canvas.create_text(x1 + 2, y1 - 2, anchor="sw", text=d["id"], fill="#ffd400", tags="box",
                               font=("Consolas", 9))

    def press(event: Any) -> None:
        drag["start"] = (event.x, event.y)
        drag["item"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00e0ff", width=1)

    def move(event: Any) -> None:
        if drag["start"]:
            canvas.coords(drag["item"], *drag["start"], event.x, event.y)

    def release(event: Any) -> None:
        if not drag["start"]:
            return
        (sx, sy), drag["start"] = drag["start"], None
        canvas.delete(drag["item"])
        box = [round(sx / zoom), round(sy / zoom), round(event.x / zoom), round(event.y / zoom)]
        try:
            donor = add_donor(structure_dir, doc, name, chosen.get(), box, classes)
            status.config(text=f"added {donor['id']} {donor['box']}")
        except ContractError as exc:
            status.config(text=str(exc))
        redraw()

    def remove(event: Any) -> None:
        x, y = event.x / zoom, event.y / zoom
        hit = [d for d in doc["donors"] if d["photo"] == name and d["box"][0] <= x <= d["box"][2]
               and d["box"][1] <= y <= d["box"][3]]
        if hit:
            smallest = min(hit, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            remove_donor(doc, smallest["id"])
            status.config(text=f"removed {smallest['id']}")
            redraw()

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<ButtonRelease-1>", release)
    canvas.bind("<Button-3>", remove)
    root.bind("<Escape>", lambda _e: root.destroy())
    redraw()
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Box intact elements on another photograph, as donors.")
    parser.add_argument("--structure", required=True, help="folder name under data/structures/")
    parser.add_argument("--name", help="short name of the reference photo, e.g. other_side")
    parser.add_argument("--photo", type=Path, help="image file to add (copied into the building's folder) under --name")
    parser.add_argument("--note", default="", help="note stored with --photo or --add")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--add", nargs=5, metavar=("CLASS", "X1", "Y1", "X2", "Y2"), help="add one box, no window")
    parser.add_argument("--remove", nargs="+", default=[], metavar="ID")
    parser.add_argument("--masks", action="store_true", help="make a SAM mask for every box (GPU)")
    parser.add_argument("--config", default=str(runio.DEFAULT_CONFIG))
    opts = parser.parse_args(argv)
    try:
        cfg = runio.load_config(opts.config)
        structure_dir = runio.structure_dir(cfg, opts.structure)
        classes = list(cfg["classes"])
        doc = load(structure_dir, classes)
        if opts.photo:
            if not opts.name:
                raise ContractError("--photo needs --name")
            add_photo(structure_dir, doc, opts.name, opts.photo.resolve(), opts.note)
        for donor_id in opts.remove:
            remove_donor(doc, donor_id)
        if opts.add:
            if not opts.name:
                raise ContractError("--add needs --name (the photo the box is on)")
            add_donor(structure_dir, doc, opts.name, opts.add[0], [int(v) for v in opts.add[1:]], classes, opts.note)
        if opts.masks:
            print("\n".join(make_masks(structure_dir, doc, cfg)))
        if not (opts.list or opts.add or opts.remove or opts.masks):
            if not opts.name:
                raise ContractError("give --name (and --photo the first time), or --list")
            window(structure_dir, doc, opts.name, classes)
        if opts.photo or opts.add or opts.remove or opts.masks or not opts.list:
            save(structure_dir, doc)
        print(table(doc))
        print(f"\nFile: {runio.relative(structure_dir / REFERENCE_FILE)}")
        print(f"Use them: python run.py --structure {opts.structure} --from evidence --no-review")
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
