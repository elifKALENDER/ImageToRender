"""Tests for tools/editor_adapter.py.

The editor itself is a Tkinter program, so the tests do not open it. They use the editor's own loader
(``load_editor_records``) and its own record normaliser to read what the adapter exported, then write
the save files in the layout of ``ExpertInventoryEditor.save_inventory``.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from i2r.runio import REPO_ROOT, ContractError, write_json
from tools import editor_adapter as adapter

SIZE = (120, 80)
OLD_INVENTORY = REPO_ROOT / "outputs" / "restore_20260731_162315" / "expert_validated_inventory.json"


def make_run(folder: Path) -> tuple[Path, Path, dict]:
    run_dir = folder / "run"
    (run_dir / "element_masks").mkdir(parents=True)
    image_path = folder / "image.png"
    Image.fromarray(np.full((SIZE[1], SIZE[0], 3), 128, np.uint8)).save(image_path)
    elements = [
        {"id": "arch_01", "class": "arch", "box": [10, 10, 40, 40], "score": 0.04,
         "confirmed": True, "rejected": False, "review_status": "confirmed"},
        {"id": "pilaster_01", "class": "pilaster", "box": [50, 10, 60, 70], "score": 0.03,
         "confirmed": False, "rejected": False, "review_status": "not_reviewed"},
        {"id": "window_01", "class": "window", "box": [70, 20, 90, 50], "score": 0.02,
         "confirmed": False, "rejected": True, "review_status": "rejected"},
    ]
    for element in elements:
        element.update(mask_file=f"element_masks/{element['id']}.png", source="detector",
                       detected_class=element["class"])
        Image.fromarray(np.zeros((SIZE[1], SIZE[0]), np.uint8)).save(run_dir / element["mask_file"])
    doc = {"structure_id": "demo", "image_size": list(SIZE),
           "classes": ["arch", "pilaster", "window", "door"], "elements": elements}
    write_json(run_dir / "elements.json", doc)
    return run_dir, image_path, doc


def editor_load(export_dir: Path) -> list[dict]:
    editor = adapter.load_editor_module()
    records, _ = editor.load_editor_records(
        export_dir / adapter.OBSERVED_FILE, export_dir / adapter.REVIEW_FILE, *SIZE)
    return records


def editor_save(records: list[dict], folder: Path, finalize: bool) -> Path:
    """Write what ExpertInventoryEditor.save_inventory writes (the JSON part)."""
    editor = adapter.load_editor_module()
    folder.mkdir(parents=True, exist_ok=True)
    normalized = [editor.ExpertInventoryEditor.normalized_output_record(None, r)
                  for r in records if not r.get("hidden")]
    write_json(folder / adapter.INVENTORY_FILE, {
        "inventory_state": "finalized" if finalize else "draft",
        "locked": finalize,
        "elements": [r for r in normalized if r["status"] in adapter.OBSERVED],
        "unresolved_review": [r for r in normalized if r["status"] == "candidate_needs_review"],
    })
    write_json(folder / adapter.REJECTED_FILE,
               {"elements": [r for r in normalized if r["status"] == "rejected"]})
    return folder


class RoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.run_dir, self.image_path, self.original = make_run(self.folder)
        self.export_dir = adapter.export_run(self.run_dir, self.image_path, None)
        self.records = editor_load(self.export_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def elements(self) -> dict[str, dict]:
        doc = json.loads((self.run_dir / "elements.json").read_text(encoding="utf-8"))
        return {e["id"]: e for e in doc["elements"]}

    def import_save(self, finalize: bool, class_map: dict | None = None) -> dict:
        save = editor_save(self.records, self.folder / "save", finalize)
        inventory, records = adapter.read_editor_save(save)
        return adapter.merge_into_run(self.run_dir, inventory, records, class_map or {})

    def record(self, record_id: str) -> dict:
        return next(r for r in self.records if r["id"] == record_id)

    def test_export_puts_every_element_in_the_editor(self) -> None:
        statuses = {r["id"]: r["status"] for r in self.records}
        self.assertEqual(statuses, {"arch_01": "observed_intact",
                                    "pilaster_01": "candidate_needs_review",
                                    "window_01": "rejected"})
        self.assertEqual(self.record("arch_01")["bbox_xyxy"], [10, 10, 40, 40])

    def test_round_trip_without_edits_returns_the_same_elements(self) -> None:
        entry = self.import_save(finalize=False)
        self.assertEqual(entry["changes"], [])
        self.assertEqual(list(self.elements().values()), self.original["elements"])
        self.assertFalse((self.run_dir / "elements_before_editor.json").exists())

    def test_edits_are_applied_and_nothing_is_deleted(self) -> None:
        # pilaster_01: box moved and decided as fragmented; arch_01: class changed;
        # window_01 removed in the editor; one box drawn by the expert.
        pilaster = self.record("pilaster_01")
        pilaster.update(bbox_xyxy=[52, 12, 62, 72], status="observed_fragmented", expert_action="bbox_nudged")
        self.record("arch_01").update({"class": "door", "expert_action": "class_corrected"})
        self.records = [r for r in self.records if r["id"] != "window_01"]
        self.records.append({**copy.deepcopy(pilaster), "id": "expert_window_004", "class": "window",
                             "status": "observed_intact", "bbox_xyxy": [95, 5, 110, 30],
                             "source": adapter.EXPERT_SOURCE, "original_candidate_id": None,
                             "expert_action": "added", "ai_score": None})
        self.import_save(finalize=True)
        elements = self.elements()

        pilaster = elements["pilaster_01"]
        self.assertEqual(pilaster["box"], [52, 12, 62, 72])
        self.assertEqual(pilaster["original_box"], [50, 10, 60, 70])
        self.assertEqual(pilaster["mask_kind"], "box")
        self.assertTrue(pilaster["confirmed"])
        self.assertEqual(pilaster["expert_condition"], "fragmented")
        mask = np.asarray(Image.open(self.run_dir / pilaster["mask_file"]))
        self.assertEqual(int(mask[40, 55]), 255)
        self.assertEqual(int(mask[5, 5]), 0)
        self.assertTrue((self.run_dir / "element_masks" / "pilaster_01.png").exists())

        self.assertEqual((elements["arch_01"]["class"], elements["arch_01"]["original_class"]),
                         ("door", "arch"))
        self.assertNotIn("expert_condition", elements["window_01"])  # already rejected: untouched
        self.assertTrue(elements["window_01"]["rejected"])

        added = elements["expert_window_004"]
        self.assertEqual((added["source"], added["score"], added["confirmed"]), ("expert", 1.0, True))
        self.assertTrue((self.run_dir / added["mask_file"]).exists())
        self.assertTrue((self.run_dir / "elements_before_editor.json").exists())
        log = json.loads((self.run_dir / adapter.IMPORT_LOG).read_text(encoding="utf-8"))
        self.assertTrue(log[-1]["locked"])

    def test_removed_confirmed_element_is_kept_as_rejected(self) -> None:
        self.records = [r for r in self.records if r["id"] != "arch_01"]
        self.import_save(finalize=False)
        arch = self.elements()["arch_01"]
        self.assertTrue(arch["rejected"])
        self.assertEqual(arch["review_status"], "editor_removed")

    def test_merged_element_points_to_the_survivor(self) -> None:
        survivor = self.record("pilaster_01")
        survivor.update(merged_from=["pilaster_01", "arch_01"], expert_action="merged")
        self.records = [r for r in self.records if r["id"] != "arch_01"]
        self.import_save(finalize=False)
        arch = self.elements()["arch_01"]
        self.assertEqual((arch["review_status"], arch["merged_into"]), ("editor_merged", "pilaster_01"))

    def test_preset_status_is_not_recorded_as_expert_condition(self) -> None:
        self.import_save(finalize=True)
        self.assertNotIn("expert_condition", self.elements()["arch_01"])

    def test_unknown_editor_class_needs_a_class_map(self) -> None:
        self.record("arch_01").update({"class": "opening", "expert_action": "class_corrected"})
        with self.assertRaises(ContractError) as ctx:
            self.import_save(finalize=False)
        self.assertIn("class_map", str(ctx.exception))
        self.import_save(finalize=False, class_map={"opening": "window"})
        self.assertEqual(self.elements()["arch_01"]["class"], "window")


class OldInventoryTest(unittest.TestCase):
    def inventory(self) -> dict:
        return {"_path": "old.json", "inventory_state": "finalized", "locked": True}

    def test_accepted_records_become_elements(self) -> None:
        records = [
            {"id": "arch_004", "class": "arch", "status": "observed_intact", "bbox_xyxy": [1, 1, 10, 10],
             "source": "ai_known_prototype", "ai_score": 0.79, "expert_action": "unmodified"},
            {"id": "expert_opening_018", "class": "opening", "status": "observed_fragmented",
             "bbox_xyxy": [20, 20, 30, 40], "source": adapter.EXPERT_SOURCE, "ai_score": None,
             "expert_action": "added"},
            {"id": "x", "class": "arch", "status": "rejected", "bbox_xyxy": [1, 1, 5, 5]},
        ]
        doc, boxes = adapter.elements_from_inventory(
            self.inventory(), records, "demo", SIZE, ["arch", "window"], {"opening": "window"})
        by_id = {e["id"]: e for e in doc["elements"]}
        self.assertEqual(set(by_id), {"arch_004", "expert_opening_018"})
        self.assertEqual((by_id["arch_004"]["source"], by_id["arch_004"]["score"]), ("detector", 0.79))
        opening = by_id["expert_opening_018"]
        self.assertEqual((opening["class"], opening["original_class"], opening["source"]),
                         ("window", "opening", "expert"))
        self.assertEqual(opening["expert_condition"], "fragmented")
        self.assertEqual(doc["summary"]["not_imported"], 1)
        self.assertEqual(boxes["arch_004"], [1, 1, 10, 10])

    @unittest.skipUnless(OLD_INVENTORY.is_file(), "local expert inventory not present")
    def test_locked_s01_inventory_gives_23_elements(self) -> None:
        inventory, records = adapter.read_editor_save(OLD_INVENTORY)
        with Image.open(OLD_INVENTORY.parent / "input.png") as image:
            size = image.size
        classes = ["window", "door", "arch", "pilaster", "column", "cornice", "base"]
        doc, _ = adapter.elements_from_inventory(
            inventory, records, "s01_test_church", size, classes, {"opening": "window"})
        self.assertEqual(len(doc["elements"]), 23)
        self.assertEqual(doc["summary"]["expert_added"], 8)
        self.assertEqual(doc["summary"]["by_condition"], {"intact": 12, "fragmented": 11})


if __name__ == "__main__":
    unittest.main()
