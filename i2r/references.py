"""Reference donors: intact elements boxed on another photograph of the same building (T37, was R5c).

The method copies what the photograph shows elsewhere. Where the photograph shows no intact element of a
class (s01: no base on the damaged side), another photograph of the same building may. The author boxes
such elements with ``tools/reference_donors.py``; they are stored in the building's folder:

    data/structures/<id>/reference_donors.json
    data/structures/<id>/references/<photo file>          (images, not in git)

    {"photos": {"other_side": {"file": "references/other_side.png", "note": "..."}},
     "donors": [{"id": "base_01", "class": "base", "photo": "other_side", "box": [x1, y1, x2, y2],
                 "note": "..."}]}

In the pipeline a reference donor is one more intact element, with these differences:

- its ID is ``ref:<id>``; its mask is its box (no SAM mask);
- sizes on two photographs are not comparable, so the size filter compares proportions only;
- it is used only when no donor of the same photograph qualifies (``evidence.find_tier_a``);
- a placement from it is a guess for the expert, never an automatic completion
  (``complete.reference_donor_is_guess``): the element is from the same building, but nothing registers
  the two photographs, and light, angle and resolution differ.

The evidence step copies the file and the photographs into the run folder (``reference_donors/``), so
later steps and a later ``--from`` read only the run folder.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from i2r.runio import ContractError, read_json, write_json

REFERENCE_FILE = "reference_donors.json"       # in the building's folder
RUN_DIR = "reference_donors"                   # in the run folder: the file and the photographs
PREFIX = "ref:"


def read_reference_file(path: Path, classes: list[str] | None = None) -> dict[str, Any]:
    """The checked content of a reference_donors.json ({"photos": {}, "donors": []} when missing)."""
    if not path.is_file():
        return {"photos": {}, "donors": []}
    doc = read_json(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("photos"), dict) or not isinstance(doc.get("donors"), list):
        raise ContractError(f"{path}: expected an object with 'photos' (object) and 'donors' (list)")
    for name, photo in doc["photos"].items():
        if not isinstance(photo, dict) or not isinstance(photo.get("file"), str):
            raise ContractError(f"{path}: photo {name!r} needs 'file'")
    seen = set()
    for d in doc["donors"]:
        if not isinstance(d, dict) or any(k not in d for k in ("id", "class", "photo", "box")):
            raise ContractError(f"{path}: every donor needs 'id', 'class', 'photo' and 'box'")
        if d["id"] in seen:
            raise ContractError(f"{path}: donor ID {d['id']!r} is used twice")
        seen.add(d["id"])
        if d["photo"] not in doc["photos"]:
            raise ContractError(f"{path}: donor {d['id']} names the unknown photo {d['photo']!r}")
        if classes is not None and d["class"] not in classes:
            raise ContractError(f"{path}: donor {d['id']} has the unknown class {d['class']!r}")
        box = d["box"]
        if (not isinstance(box, list) or len(box) != 4 or any(not isinstance(v, int) for v in box)
                or box[2] <= box[0] or box[3] <= box[1] or min(box) < 0):
            raise ContractError(f"{path}: donor {d['id']} needs 'box' [x1, y1, x2, y2] in pixels, x2 > x1, y2 > y1")
    return doc


def copy_into_run(structure_dir: Path, run_dir: Path, classes: list[str] | None = None) -> int:
    """Copy the building's reference donors and their photographs into the run folder (replacing what an
    earlier run left there). Returns the number of donors."""
    target = run_dir / RUN_DIR
    if target.exists():
        shutil.rmtree(target)
    doc = read_reference_file(structure_dir / REFERENCE_FILE, classes)
    if not doc["donors"]:
        return 0
    target.mkdir()
    photos = {}
    for name, photo in doc["photos"].items():
        source = structure_dir / photo["file"]
        if not source.is_file():
            raise ContractError(f"{structure_dir / REFERENCE_FILE}: photo {name!r} not found: {source}")
        with Image.open(source) as image:
            image.convert("RGB").save(target / f"{name}.png")
        photos[name] = {**photo, "file": f"{name}.png", "source_file": photo["file"]}
    write_json(target / REFERENCE_FILE, {"photos": photos, "donors": doc["donors"]})
    return len(doc["donors"])


def load_from_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """(donors as element-like records, {donor ID: RGB array of its photograph}); ([], {}) when the run
    has no reference donor."""
    folder = run_dir / RUN_DIR
    if not (folder / REFERENCE_FILE).is_file():
        return [], {}
    doc = read_reference_file(folder / REFERENCE_FILE)
    arrays = {}
    for name, photo in doc["photos"].items():
        path = folder / photo["file"]
        if not path.is_file():
            raise ContractError(f"Reference photograph not found in the run folder: {path}")
        arrays[name] = np.asarray(Image.open(path).convert("RGB"))
    donors, images = [], {}
    for d in doc["donors"]:
        rgb = arrays[d["photo"]]
        if d["box"][2] > rgb.shape[1] or d["box"][3] > rgb.shape[0]:
            raise ContractError(f"reference donor {d['id']}: box {d['box']} lies outside photo {d['photo']!r} "
                                f"({rgb.shape[1]} x {rgb.shape[0]})")
        donor_id = PREFIX + d["id"]
        donors.append({"id": donor_id, "class": d["class"], "box": d["box"], "damage_status": "intact",
                       "source": "reference", "reference_photo": d["photo"], "note": d.get("note", "")})
        images[donor_id] = rgb
    return donors, images


def is_reference(donor_id: str | None) -> bool:
    return bool(donor_id) and str(donor_id).startswith(PREFIX)
