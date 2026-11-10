#!/usr/bin/env python3
"""Connect the expert inventory editor (``tools/inventory_editor.py``) to the pipeline's run folders.

The editor is the archived Stage 3B-v5 tool and is used unchanged. It reads and writes the old Stage 3B
file formats; this adapter translates between those and ``elements.json``.

    python tools/editor_adapter.py edit   --run <run>                  export, open the editor, import
    python tools/editor_adapter.py export --run <run>                  write <run>/editor_input/
    python tools/editor_adapter.py import --run <run> --inventory <editor save folder or json>
    python tools/editor_adapter.py import --structure <id> --inventory <old locked inventory>

``<run>`` is a pipeline run folder, e.g. ``outputs/s01_test_church/20261110_165746``.

Export (elements.json -> editor input, in ``<run>/editor_input/``):
    input.png, completion_mask.png (black if the structure has no damage mask yet),
    stage3b_v4_observed_instances.json   elements confirmed in review, preset to observed_intact
    stage3b_v4_review_candidates.json    unconfirmed elements (to decide) and rejected ones (to revisit)
The editor saves each session into a new ``<run>/editor_saves/restore_<timestamp>/`` folder.

Import into a run (editor save -> the run's elements.json, edited in place; the previous file is kept
as ``elements_before_editor*.json`` and every change is appended to ``editor_import_log.json``):
    - elements are matched by id; class, box and status changes are applied;
    - a changed box gets a rectangular mask (``mask_kind: "box"``); the segmentation mask is kept;
    - boxes the expert drew get ``source: "expert"``, ``score: 1.0`` and a rectangular mask;
    - elements the editor removed or merged are kept with ``rejected: true`` (nothing is deleted);
    - the editor's preset status is not an expert decision: ``expert_condition`` (intact or
      fragmented) is recorded only for records the expert actually touched.
  With no edits the elements come back unchanged.

Import from an old inventory (``--structure``): creates a new run of that structure whose
``elements.json`` holds the inventory's accepted elements, all with rectangular masks. This is how the
expert-locked 23-element inventory of s01 (``outputs/restore_20260731_162315``) enters the pipeline.

Editor classes that are not in ``config.yaml`` (the old inventory used ``opening``) are translated by
``editor.class_map`` in the config or the structure's ``structure.yaml``; an untranslated class stops
the import.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from i2r import runio  # noqa: E402
from i2r.detect import ELEMENTS_FILE, MASK_DIR, load_elements, validate_elements  # noqa: E402
from i2r.runio import ContractError, read_json, write_json  # noqa: E402

EDITOR_SCRIPT = Path(__file__).resolve().parent / "inventory_editor.py"

EXPORT_DIR = "editor_input"
SAVES_DIR = "editor_saves"
IMPORT_LOG = "editor_import_log.json"

OBSERVED_FILE = "stage3b_v4_observed_instances.json"
REVIEW_FILE = "stage3b_v4_review_candidates.json"
INVENTORY_FILE = "expert_validated_inventory.json"
REJECTED_FILE = "expert_rejected_elements.json"

OBSERVED = {"observed_intact": "intact", "observed_fragmented": "fragmented"}
EXPERT_SOURCE = "expert_added_observed_evidence"
EXPERT_SCORE = 1.0

# Editor fields kept on an element as ``editor_record`` when the editor changed it.
EDITOR_RECORD_KEYS = (
    "id", "class", "status", "source", "expert_action", "expert_note", "usable_as_prototype",
    "merged_from", "ai_score",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_editor_module() -> ModuleType:
    """Import tools/inventory_editor.py by path (it needs Tkinter; imported only when used)."""
    name = "inventory_editor"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EDITOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise ContractError(f"Cannot load the editor from {EDITOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while the class is created
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


# ---------------------------------------------------------------------------
# Classes and masks
# ---------------------------------------------------------------------------

def map_class(name: str, classes: Iterable[str], class_map: dict[str, str], where: str) -> str:
    classes = list(classes)
    if name in classes:
        return name
    if name in class_map:
        target = class_map[name]
        if target not in classes:
            raise ContractError(f"{where}: editor.class_map sends {name!r} to unknown class {target!r}")
        return target
    raise ContractError(
        f"{where}: editor class {name!r} is not in config.yaml classes. Add it to editor.class_map "
        f"(in config.yaml or the structure's structure.yaml) with the reason."
    )


def box_mask(box: list[int], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


def save_box_mask(run_dir: Path, name: str, box: list[int], size: tuple[int, int]) -> str:
    mask_file = f"{MASK_DIR}/{name}.png"
    (run_dir / MASK_DIR).mkdir(exist_ok=True)
    Image.fromarray(box_mask(box, size), mode="L").save(run_dir / mask_file)
    return mask_file


def unique_id(base: str, taken: set[str]) -> str:
    candidate, suffix = base, 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def editor_status(element: dict[str, Any]) -> str:
    if element.get("rejected"):
        return "rejected"
    if element.get("confirmed"):
        return "observed_intact"
    return "candidate_needs_review"


def editor_item(element: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": element["id"],
        "class": element["class"],
        "bbox_xyxy": list(element["box"]),
        "score": element["score"],
        "status": editor_status(element),
    }


def export_run(run_dir: Path, image_path: Path, damage_path: Path | None) -> Path:
    """Write the editor's input files into ``<run>/editor_input/``. Returns that folder."""
    doc = load_elements(run_dir)
    width, height = doc["image_size"]
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        raise ContractError(f"{image_path}: size {image.size} differs from elements.json {width}x{height}")

    out = run_dir / EXPORT_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    image.save(out / "input.png")
    if damage_path is not None and damage_path.is_file():
        damage = Image.open(damage_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        damage_note = str(damage_path)
    else:
        damage = Image.new("L", (width, height), 0)
        damage_note = "none (black placeholder)"
    damage.save(out / "completion_mask.png")

    confirmed = [editor_item(e) for e in doc["elements"] if editor_status(e) == "observed_intact"]
    to_review = [editor_item(e) for e in doc["elements"] if editor_status(e) != "observed_intact"]
    if not confirmed and not to_review:
        raise ContractError(f"{run_dir / ELEMENTS_FILE}: no elements to edit")
    write_json(out / OBSERVED_FILE, {
        "run_id": run_dir.name,
        "note": "Exported by tools/editor_adapter.py. 'observed_intact' is a preset for elements "
                "confirmed in review, not an expert judgement of condition.",
        "known_prototypes": {"arches": confirmed},
        "recovered_observed": {},
    })
    write_json(out / REVIEW_FILE, {
        "note": "Unconfirmed elements (decide) and rejected elements (revisit).",
        "arches": to_review,
    })
    write_json(out / "export_manifest.json", {
        "exported_at": now(),
        "run": runio.relative(run_dir),
        "elements_sha256": sha256(run_dir / ELEMENTS_FILE),
        "image": str(image_path),
        "damage_mask": damage_note,
        "counts": {"confirmed": len(confirmed), "to_review": len(to_review)},
    })
    return out


# ---------------------------------------------------------------------------
# Reading an editor save
# ---------------------------------------------------------------------------

def read_editor_save(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (inventory document, every record: accepted, unresolved and rejected)."""
    inventory_path = path / INVENTORY_FILE if path.is_dir() else path
    inventory = read_json(inventory_path)
    runio.require_keys(inventory, ("elements",), str(inventory_path))
    records = list(inventory["elements"]) + list(inventory.get("unresolved_review") or [])
    rejected_path = inventory_path.parent / REJECTED_FILE
    if rejected_path.is_file():
        records += list(read_json(rejected_path).get("elements") or [])
    for index, record in enumerate(records):
        runio.require_keys(record, ("id", "class", "status", "bbox_xyxy"), f"{inventory_path}: record {index}")
    inventory["_path"] = str(inventory_path)
    return inventory, records


def editor_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in EDITOR_RECORD_KEYS if key in record}


# ---------------------------------------------------------------------------
# Import into an existing run
# ---------------------------------------------------------------------------

def merge_into_run(
    run_dir: Path,
    inventory: dict[str, Any],
    records: list[dict[str, Any]],
    class_map: dict[str, str],
) -> dict[str, Any]:
    """Apply an editor save to ``<run>/elements.json``. Returns the log entry."""
    doc = load_elements(run_dir)
    size = tuple(doc["image_size"])
    classes = doc["classes"]
    by_id = {e["id"]: e for e in doc["elements"]}
    taken = set(by_id)
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()
    merged_into: dict[str, str] = {}

    for record in records:
        where = f"{inventory['_path']}: {record['id']}"
        key = record.get("original_candidate_id") or (
            None if record.get("source") == EXPERT_SOURCE else record["id"]
        )
        for merged in record.get("merged_from") or []:
            if merged != key:
                merged_into[merged] = key or record["id"]
        new_class = map_class(record["class"], classes, class_map, where)
        box = [int(v) for v in record["bbox_xyxy"]]

        if key is not None and key in by_id:
            seen.add(key)
            changed = apply_record(by_id[key], record, new_class, box, run_dir, size)
            if changed:
                changes.append({"id": key, "changed": changed})
            continue

        element_id = unique_id(record["id"], taken)
        status = record["status"]
        element = {
            "id": element_id,
            "class": new_class,
            "box": box,
            "score": EXPERT_SCORE,
            "mask_file": save_box_mask(run_dir, element_id, box, size),
            "source": "expert",
            "mask_kind": "box",
            "confirmed": status in OBSERVED,
            "rejected": status == "rejected",
            "review_status": "editor_added",
            "editor_record": editor_record(record),
        }
        if status in OBSERVED:
            element["expert_condition"] = OBSERVED[status]
        doc["elements"].append(element)
        changes.append({"id": element_id, "changed": ["added"]})

    for element in doc["elements"]:
        if element["id"] in seen or element["id"] not in by_id or element.get("rejected"):
            continue
        element["rejected"] = True
        element["confirmed"] = False
        if element["id"] in merged_into:
            element["review_status"] = "editor_merged"
            element["merged_into"] = merged_into[element["id"]]
        else:
            element["review_status"] = "editor_removed"
        changes.append({"id": element["id"], "changed": [element["review_status"]]})

    validate_elements(doc, str(run_dir / ELEMENTS_FILE))
    entry = {
        "imported_at": now(),
        "editor_save": inventory["_path"],
        "inventory_state": inventory.get("inventory_state"),
        "locked": bool(inventory.get("locked")),
        "class_map": class_map,
        "changes": changes,
    }
    if changes:
        backup = run_dir / "elements_before_editor.json"
        if backup.exists():
            backup = run_dir / f"elements_before_editor_{datetime.now():%Y%m%d_%H%M%S}.json"
        shutil.copy2(run_dir / ELEMENTS_FILE, backup)
        entry["previous_elements"] = backup.name
        write_json(run_dir / ELEMENTS_FILE, doc)
    log_path = run_dir / IMPORT_LOG
    log = read_json(log_path) if log_path.is_file() else []
    log.append(entry)
    write_json(log_path, log)
    return entry


def apply_record(
    element: dict[str, Any],
    record: dict[str, Any],
    new_class: str,
    box: list[int],
    run_dir: Path,
    size: tuple[int, int],
) -> list[str]:
    """Apply one editor record to its element in place. Returns the names of what changed."""
    changed: list[str] = []
    if new_class != element["class"]:
        element.setdefault("original_class", element["class"])
        element["class"] = new_class
        changed.append("class")
    if box != list(element["box"]):
        element.setdefault("original_box", list(element["box"]))
        element["box"] = box
        element["mask_file"] = save_box_mask(run_dir, f"{element['id']}_editor", box, size)
        element["mask_kind"] = "box"
        changed.append("box")

    status = record["status"]
    touched = record.get("expert_action", "unmodified") != "unmodified"
    if status == "rejected" and not element.get("rejected"):
        element["rejected"] = True
        element["confirmed"] = False
        changed.append("rejected")
    elif status in OBSERVED and touched:
        if element.get("rejected") or not element.get("confirmed"):
            element["rejected"] = False
            element["confirmed"] = True
            changed.append("confirmed")
        if element.get("expert_condition") != OBSERVED[status]:
            element["expert_condition"] = OBSERVED[status]
            changed.append("condition")

    if changed:
        element["review_status"] = "editor_" + changed[0]
        element["editor_record"] = editor_record(record)
    return changed


# ---------------------------------------------------------------------------
# Import from an old inventory into a new run
# ---------------------------------------------------------------------------

def elements_from_inventory(
    inventory: dict[str, Any],
    records: list[dict[str, Any]],
    structure_id: str,
    image_size: tuple[int, int],
    classes: list[str],
    class_map: dict[str, str],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Build an elements.json document from accepted editor records.

    Returns (document, {element id: box}) — the caller writes one rectangular mask per element.
    Unresolved and rejected records are not imported.
    """
    width, height = image_size
    elements: list[dict[str, Any]] = []
    boxes: dict[str, list[int]] = {}
    taken: set[str] = set()
    skipped = 0
    for record in records:
        status = record["status"]
        if status not in OBSERVED:
            skipped += 1
            continue
        where = f"{inventory['_path']}: {record['id']}"
        new_class = map_class(record["class"], classes, class_map, where)
        box = [int(v) for v in record["bbox_xyxy"]]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ContractError(f"{where}: box {box} is outside the {width}x{height} image")
        element_id = unique_id(str(record["id"]), taken)
        expert = record.get("source") == EXPERT_SOURCE
        score = record.get("ai_score")
        element = {
            "id": element_id,
            "class": new_class,
            "box": box,
            "score": EXPERT_SCORE if expert or score is None else float(score),
            "mask_file": f"{MASK_DIR}/{element_id}.png",
            "source": "expert" if expert else "detector",
            "mask_kind": "box",
            "confirmed": True,
            "rejected": False,
            "review_status": "editor_inventory",
            "expert_condition": OBSERVED[status],
            "editor_record": editor_record(record),
        }
        if new_class != record["class"]:
            element["original_class"] = record["class"]
        elements.append(element)
        boxes[element_id] = box

    doc = {
        "structure_id": structure_id,
        "image_size": [width, height],
        "classes": list(classes),
        "elements": elements,
        "summary": {
            "total": len(elements),
            "by_class": {c: sum(1 for e in elements if e["class"] == c) for c in classes},
            "by_condition": {c: sum(1 for e in elements if e["expert_condition"] == c)
                             for c in OBSERVED.values()},
            "expert_added": sum(1 for e in elements if e["source"] == "expert"),
            "not_imported": skipped,
        },
        "editor_import": {
            "inventory": inventory["_path"],
            "inventory_state": inventory.get("inventory_state"),
            "locked": bool(inventory.get("locked")),
            "class_map": class_map,
            "note": "Scores of detector-origin elements are the old Stage 3B scores, not comparable "
                    "with this detector's scores. Masks are rectangles from the boxes.",
        },
    }
    validate_elements(doc)
    return doc, boxes


def import_inventory_as_run(structure_id: str, inventory_path: Path, config_path: Path) -> Path:
    inventory, records = read_editor_save(inventory_path)
    run = runio.create_run(structure_id, config_path)
    log = run.logger
    status = "failed"
    try:
        image = Image.open(run.image_path)
        source_image = Path(inventory["_path"]).parent / "input.png"
        if source_image.is_file():
            old = Image.open(source_image)
            if old.size != image.size:
                raise ContractError(
                    f"{source_image} is {old.size}, {run.image_path} is {image.size}: not the same photograph"
                )
            if not np.array_equal(np.asarray(old.convert("RGB")), np.asarray(image.convert("RGB"))):
                log.warning("[editor] %s and %s differ in pixels (same size); boxes may be off",
                            source_image, run.image_path)
        class_map = (run.config.get("editor") or {}).get("class_map") or {}
        doc, boxes = elements_from_inventory(
            inventory, records, structure_id, image.size, list(run.config["classes"]), class_map
        )
        for element_id, box in boxes.items():
            save_box_mask(run.run_dir, element_id, box, image.size)
        write_json(run.path(ELEMENTS_FILE), doc)
        shutil.copy2(inventory["_path"], run.path("editor_source_inventory.json"))
        run.meta["steps"]["editor_import"] = {
            "status": "done",
            "inventory": inventory["_path"],
            "inventory_sha256": sha256(Path(inventory["_path"])),
            "elements": len(doc["elements"]),
        }
        log.info("[editor] imported %d elements from %s %s", len(doc["elements"]), inventory["_path"],
                 {k: v for k, v in doc["summary"]["by_class"].items() if v})
        if class_map:
            log.info("[editor] class map applied: %s", class_map)
        status = "completed"
    finally:
        run.meta["status"] = status
        run.meta["finished_at"] = now()
        runio.save_meta(run)
        runio.close_logger(run)
    return run.run_dir


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run_paths(run_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    """Effective config of a run and its image and damage-mask paths."""
    config_file = run_dir / "config.yaml"
    if not config_file.is_file():
        raise ContractError(f"{run_dir} is not a run folder (no config.yaml)")
    with config_file.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    structure_id = load_elements(run_dir)["structure_id"]
    sdir = runio.structure_dir(cfg, structure_id)
    return cfg, sdir / cfg["paths"]["image_file"], sdir / cfg["paths"]["damage_mask_file"]


def class_map_of(cfg: dict[str, Any]) -> dict[str, str]:
    return dict((cfg.get("editor") or {}).get("class_map") or {})


def open_editor(run_dir: Path, cfg: dict[str, Any], export_dir: Path) -> Path | None:
    editor = load_editor_module()
    classes = tuple(cfg["classes"])
    editor_cfg = editor.Config(
        output_root=str(run_dir / SAVES_DIR),
        parent_run_dir=str(export_dir),
        class_options=classes,
        default_new_class="pilaster" if "pilaster" in classes else classes[0],
    )
    return editor.run_editor(editor_cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge between elements.json and the inventory editor.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("edit", "export"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, type=Path)
    p = sub.add_parser("import")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="merge into this run's elements.json")
    target.add_argument("--structure", help="create a new run of this structure from an old inventory")
    p.add_argument("--inventory", required=True, type=Path,
                   help="editor save folder or its expert_validated_inventory.json")
    p.add_argument("--config", type=Path, default=runio.DEFAULT_CONFIG)
    opts = parser.parse_args(argv)

    try:
        if opts.command == "import" and opts.structure:
            run_dir = import_inventory_as_run(opts.structure, opts.inventory, opts.config)
            print(f"New run: {runio.relative(run_dir)}")
            return 0

        run_dir = opts.run.resolve()
        cfg, image_path, damage_path = run_paths(run_dir)
        if opts.command == "import":
            inventory, records = read_editor_save(opts.inventory)
            entry = merge_into_run(run_dir, inventory, records, class_map_of(cfg))
            print(f"Imported {inventory['_path']}: {len(entry['changes'])} element(s) changed")
            return 0

        export_dir = export_run(run_dir, image_path, damage_path)
        print(f"Editor input written to {runio.relative(export_dir)}")
        if opts.command == "export":
            return 0
        saved = open_editor(run_dir, cfg, export_dir)
        if saved is None:
            print("Editor closed without saving; elements.json unchanged.")
            return 0
        inventory, records = read_editor_save(Path(saved))
        entry = merge_into_run(run_dir, inventory, records, class_map_of(cfg))
        print(f"Imported {runio.relative(Path(saved))}: {len(entry['changes'])} element(s) changed")
        return 0
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
