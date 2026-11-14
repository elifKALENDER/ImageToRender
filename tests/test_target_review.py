"""Tests for expert decisions per target (T31): complete applies them, tools/target_review.py writes them.

No GUI and no GPU: the tool's functions are called directly on a synthetic run folder."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import tests.test_complete as scene_module
from i2r import complete, render
from i2r.runio import REPO_ROOT, ContractError, read_json
from tests.test_complete import config, record, scene
from tests.test_render import build_run
from tools import target_review


def guess_scene():
    """A scene whose fragment sits 10 px off its box: the placement fails the acceptance tests."""
    with mock.patch.object(scene_module, "SHIFT", (10, 2)):
        return scene()


def decision_for(rgb, damage, donor_mask, value: str, note: str = "") -> dict:
    """The decision an expert would store after seeing this placement."""
    _, _, report = complete.complete_view("first_construction", rgb, damage, [record("A")], {"donor": donor_mask},
                                          config())
    e = report["targets"][0]
    placement = complete.placement_signature({"donor_id": e["donor_id"], "flipped": e["flipped"]}, e["alignment"])
    return {complete.decision_key(record("A")): {"decision": value, "note": note, "placement": placement}}


class DecisionsInCompleteTest(unittest.TestCase):
    def test_an_approved_guess_is_completed_in_the_restoration_view(self) -> None:
        rgb, damage, donor_mask, _ = guess_scene()
        decisions = decision_for(rgb, damage, donor_mask, "approve", "checked on site")
        out, labels, report = complete.complete_view("restoration", rgb, damage, [record("A")],
                                                     {"donor": donor_mask}, config(), decisions=decisions)
        e = report["targets"][0]
        self.assertEqual(e["status"], "completed")
        self.assertEqual((e["acceptance"]["accepted"], e["acceptance"]["expert"]), (True, "approve"))
        self.assertTrue(e["acceptance"]["reasons"])                 # the automatic verdict stays on record
        self.assertIn("approved by the expert although", e["note"])
        self.assertIn("checked on site", e["note"])
        self.assertGreater(e["pixels_changed"], 0)
        self.assertTrue(np.array_equal(out[damage == 0], rgb[damage == 0]))
        self.assertEqual(report["expert_decisions"], {"approve": 1, "reject": 0, "stale": 0})

    def test_a_rejected_placement_is_withheld_in_every_view_even_if_it_passed(self) -> None:
        rgb, damage, donor_mask, _ = scene()                        # a good placement
        key = complete.decision_key(record("A"))
        _, _, auto = complete.complete_view("restoration", rgb, damage, [record("A")], {"donor": donor_mask}, config())
        e = auto["targets"][0]
        self.assertEqual(e["status"], "completed")
        placement = complete.placement_signature({"donor_id": e["donor_id"], "flipped": e["flipped"]}, e["alignment"])
        decisions = {key: {"decision": "reject", "note": "wrong element", "placement": placement}}
        for view in ("restoration", "first_construction"):
            out, labels, report = complete.complete_view(view, rgb, damage, [record("A")], {"donor": donor_mask},
                                                         config(), decisions=decisions)
            e = report["targets"][0]
            self.assertEqual(e["status"], complete.REJECTED)
            self.assertIn("wrong element", e["note"])
            self.assertTrue(np.array_equal(out, rgb))
            self.assertFalse(labels.any())

    def test_a_decision_for_another_placement_is_ignored(self) -> None:
        rgb, damage, donor_mask, _ = guess_scene()
        decisions = decision_for(rgb, damage, donor_mask, "approve")
        key = next(iter(decisions))
        decisions[key]["placement"]["dx"] += 1                      # the donor was placed differently back then
        _, _, report = complete.complete_view("restoration", rgb, damage, [record("A")], {"donor": donor_mask},
                                              config(), decisions=decisions)
        e = report["targets"][0]
        self.assertEqual(e["status"], complete.GUESS)
        self.assertEqual(e["acceptance"]["expert"], "stale")
        self.assertIn("another placement", e["note"])

    def test_decision_key_survives_renumbering(self) -> None:
        a = {"target_id": "t_003", "element_id": "arch_1", "kind": "fragmented_element", "target_box": [1, 2, 3, 4]}
        self.assertEqual(complete.decision_key(a), complete.decision_key({**a, "target_id": "t_009"}))
        slot = {"target_id": "t_010", "element_id": None, "kind": "implied_slot", "rule": "B1", "class": "pilaster",
                "target_box": [1, 2, 3, 4]}
        self.assertEqual(complete.decision_key(slot), "implied_slot:B1:pilaster:1,2,3,4")

    def test_a_malformed_decisions_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / complete.DECISIONS_FILE
            path.write_text('{"decisions": {"element:x": {"decision": "maybe", "placement": {}}}}', encoding="utf-8")
            with self.assertRaises(ContractError):
                complete.read_decisions(path)
            self.assertEqual(complete.read_decisions(Path(tmp) / "absent.json"), {})


class ReviewToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = build_run(Path(self._tmp.name))
        self.opts = types.SimpleNamespace(views=["restoration", "first_construction"])
        complete.run_step(self.run, self.opts)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_items_panels_and_logged_placement(self) -> None:
        items = target_review.review_items(self.run.run_dir)
        self.assertEqual([i["target_id"] for i in items], ["t_001", "t_002"])     # tier A and the tier B slot
        a = items[0]
        self.assertEqual(a["key"], "element:target")
        self.assertEqual(set(a["status"]), {"restoration", "first_construction"})
        for view in ("restoration", "first_construction"):
            panel = self.run.path(f"targets/t_001_{view}.png")
            self.assertTrue(panel.is_file())
            with Image.open(panel) as image:
                self.assertGreater(image.width, 4 * 30)                            # four panels side by side
        entry = read_json(self.run.path("completion.json"))["views"]["restoration"]["targets"][0]
        self.assertEqual(len(entry["placement_box"]), 4)
        px = entry["donor_px"]
        self.assertEqual(px["placed"], px["used_in_damage"] + px["discarded_on_surviving"])
        self.assertAlmostEqual(px["share_used"] + px["share_discarded"], 1.0, places=3)

    def test_reject_through_the_tool_is_applied_by_the_next_complete_step(self) -> None:
        item = target_review.review_items(self.run.run_dir)[0]
        target_review.set_decision(self.run.structure_dir, item, "reject", " not this element ", self.run.run_dir)
        stored = read_json(self.run.structure_dir / complete.DECISIONS_FILE)["decisions"]["element:target"]
        self.assertEqual((stored["decision"], stored["note"]), ("reject", "not this element"))
        complete.run_step(self.run, self.opts)
        render.run_step(self.run, self.opts)
        self.assertTrue(self.run.path(complete.DECISIONS_FILE).is_file())          # copied into the run folder
        prov = {e["target_id"]: e for e in read_json(self.run.path("provenance.json"))["targets"]}
        self.assertEqual(prov["t_001"]["views"], {"restoration": complete.REJECTED,
                                                  "first_construction": complete.REJECTED})
        self.assertEqual(prov["t_001"]["pixels_changed"], {"restoration": 0, "first_construction": 0})
        self.assertIn("rejected", target_review.table(target_review.review_items(self.run.run_dir),
                                                      complete.read_decisions(self.run.structure_dir
                                                                              / complete.DECISIONS_FILE)))
        # clearing it brings the automatic result back
        target_review.set_decision(self.run.structure_dir, item, None, "", self.run.run_dir)
        complete.run_step(self.run, self.opts)
        entry = read_json(self.run.path("completion.json"))["views"]["restoration"]["targets"][0]
        self.assertEqual(entry["status"], "completed")
        self.assertFalse(self.run.path(complete.DECISIONS_FILE).exists())

    def test_unknown_decision_value(self) -> None:
        item = target_review.review_items(self.run.run_dir)[0]
        with self.assertRaises(ContractError):
            target_review.set_decision(self.run.structure_dir, item, "maybe", "", self.run.run_dir)

    def test_tool_does_not_import_torch(self) -> None:
        code = "import sys; import tools.target_review; sys.exit(1 if 'torch' in sys.modules else 0)"
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT).returncode, 0)


if __name__ == "__main__":
    unittest.main()
