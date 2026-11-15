"""Tests for reference donors (T37): elements boxed on another photograph of the building.

No GUI and no GPU: fake embeddings, a synthetic run folder, the tool's functions called directly."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from i2r import complete, evidence, references, render
from i2r.runio import REPO_ROOT, ContractError, load_config, read_json, write_json
from tests.test_evidence_a import GREEN, RED, Scene, element, fake_embed, settings
from tests.test_render import build_run
from tools import reference_donors, target_review

CLASSES = list(load_config()["classes"])


def other_photo(colour: tuple[int, int, int] = RED) -> np.ndarray:
    """A 60 x 80 photograph with one coloured block at [10, 10, 30, 50]."""
    rgb = np.full((80, 60, 3), 128, np.uint8)
    rgb[10:50, 10:30] = colour
    return rgb


def ref_donor(colour: tuple[int, int, int] = RED, box: list[int] | None = None) -> tuple[list[dict], dict]:
    donor = {"id": "ref:base_01", "class": "base", "box": box or [10, 10, 30, 50], "damage_status": "intact",
             "source": "reference", "reference_photo": "other"}
    return [donor], {"ref:base_01": other_photo(colour)}


def base_scene() -> Scene:
    """A half-lost base and no intact base on the photograph."""
    s = Scene()
    s.add(element("stub", [10, 10, 30, 50], "fragmented", cls="base"), RED)
    s.damage_box([10, 30, 30, 50])
    return s


class EvidenceTest(unittest.TestCase):
    def test_a_reference_donor_gives_a_fragment_its_donor(self) -> None:
        s = base_scene()
        (without,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed)
        self.assertEqual(without["tier"], "C")
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed, ref_donor())
        self.assertEqual((record["tier"], record["donor_id"], record["donor_photo"]), ("A", "ref:base_01", "other"))

    def test_a_donor_of_the_same_photograph_goes_first(self) -> None:
        s = base_scene()
        s.add(element("own", [40, 10, 60, 50], "intact", cls="base"), RED)
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed, ref_donor())
        self.assertEqual((record["donor_id"], record["donor_photo"]), ("own", None))
        self.assertEqual([a["donor_id"] for a in record["alternatives"]], ["ref:base_01"])

    def test_a_dissimilar_reference_donor_is_not_used(self) -> None:
        s = base_scene()
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed, ref_donor(GREEN))
        self.assertEqual(record["tier"], "C")
        self.assertIn("below min_similarity", record["reason"])

    def test_only_proportions_are_compared_across_photographs(self) -> None:
        target = {"id": "t", "class": "base", "box": [0, 0, 20, 40]}
        twice = {"id": "ref:a", "class": "base", "box": [0, 0, 40, 80], "reference_photo": "other"}
        self.assertIsNone(evidence.filter_size(target, twice, settings()))                 # other scale, same shape
        self.assertIsNotNone(evidence.filter_size(target, {**twice, "reference_photo": None}, settings()))
        wide = {**twice, "box": [0, 0, 80, 40]}
        self.assertIn("proportion", evidence.filter_size(target, wide, settings()))


class FileAndToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = build_run(Path(self._tmp.name))
        self.opts = types.SimpleNamespace(views=["restoration", "first_construction"])
        self.photo = Path(self._tmp.name) / "elsewhere.jpg"
        Image.fromarray(other_photo((150, 120, 90))).save(self.photo)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def add(self) -> dict:
        sdir = self.run.structure_dir
        doc = reference_donors.load(sdir, CLASSES)
        reference_donors.add_photo(sdir, doc, "other", self.photo, "test")
        donor = reference_donors.add_donor(sdir, doc, "other", "pilaster", [30, 50, 10, 10], CLASSES)   # any corner order
        reference_donors.save(sdir, doc)
        return donor

    def test_tool_stores_photo_and_box_and_removes_them(self) -> None:
        donor = self.add()
        self.assertEqual((donor["id"], donor["box"]), ("pilaster_01", [10, 10, 30, 50]))
        sdir = self.run.structure_dir
        self.assertTrue((sdir / "references" / "other.png").is_file())
        doc = reference_donors.load(sdir, CLASSES)
        with self.assertRaises(ContractError):
            reference_donors.add_donor(sdir, doc, "other", "pilaster", [5, 5, 6, 6], CLASSES)           # too small
        with self.assertRaises(ContractError):
            reference_donors.add_donor(sdir, doc, "nowhere", "pilaster", [0, 0, 20, 20], CLASSES)
        with self.assertRaises(ContractError):
            reference_donors.add_donor(sdir, doc, "other", "gargoyle", [0, 0, 20, 20], CLASSES)
        reference_donors.remove_donor(doc, "pilaster_01")
        self.assertEqual(doc["donors"], [])

    def test_run_folder_gets_its_own_copy(self) -> None:
        self.add()
        self.assertEqual(references.copy_into_run(self.run.structure_dir, self.run.run_dir, CLASSES), 1)
        donors, images = references.load_from_run(self.run.run_dir)
        self.assertEqual(donors[0]["id"], "ref:pilaster_01")
        self.assertEqual(images["ref:pilaster_01"].shape, (80, 60, 3))
        (self.run.structure_dir / references.REFERENCE_FILE).unlink()
        self.assertEqual(references.copy_into_run(self.run.structure_dir, self.run.run_dir, CLASSES), 0)
        self.assertEqual(references.load_from_run(self.run.run_dir), ([], {}))             # stale copy removed

    def test_a_box_outside_the_photo_or_a_bad_file_is_refused(self) -> None:
        path = self.run.structure_dir / references.REFERENCE_FILE
        write_json(path, {"photos": {"other": {"file": "x.png"}},
                          "donors": [{"id": "a", "class": "pilaster", "photo": "missing", "box": [0, 0, 5, 5]}]})
        with self.assertRaises(ContractError):
            references.read_reference_file(path, CLASSES)
        path.unlink()
        self.add()
        doc = read_json(path)
        doc["donors"][0]["box"] = [10, 10, 300, 50]
        write_json(path, doc)
        references.copy_into_run(self.run.structure_dir, self.run.run_dir, CLASSES)
        with self.assertRaises(ContractError):
            references.load_from_run(self.run.run_dir)

    def test_named_by_the_expert_it_is_placed_as_a_guess_and_recorded(self) -> None:
        self.add()
        references.copy_into_run(self.run.structure_dir, self.run.run_dir, CLASSES)
        complete.run_step(self.run, self.opts)
        item = target_review.review_items(self.run.run_dir)[0]
        self.assertIn("ref:pilaster_01", [o["donor_id"] for o in item["options"]])
        self.assertEqual(next(o for o in item["options"] if o["donor_id"] == "ref:pilaster_01")["photo"], "other")
        target_review.set_donor(self.run.structure_dir, item, "ref:pilaster_01", None, "", self.run.run_dir)
        complete.run_step(self.run, self.opts)
        render.run_step(self.run, self.opts)
        views = read_json(self.run.path("completion.json"))["views"]
        entry = views["first_construction"]["targets"][0]
        self.assertEqual((entry["donor_id"], entry["donor_photo"]), ("ref:pilaster_01", "other"))
        self.assertEqual(entry["status"], complete.GUESS)
        self.assertIn(complete.REFERENCE_GUESS_REASON, entry["acceptance"]["reasons"])
        self.assertEqual(views["restoration"]["targets"][0]["pixels_changed"], 0)          # withheld there
        self.assertTrue(self.run.path(entry["panel_file"]).is_file())
        prov = read_json(self.run.path("provenance.json"))["targets"][0]
        self.assertEqual((prov["donor_id"], prov["donor_photo"]), ("ref:pilaster_01", "other"))
        self.assertIn("another photograph", self.run.path("report.html").read_text(encoding="utf-8"))

    def test_a_fragment_without_a_donor_can_be_given_one(self) -> None:
        records = read_json(self.run.path("evidence.json"))["targets"]
        open_one = {**records[0], "tier": "C", "donor_id": None, "donor_box": None, "flipped": None,
                    "confidence": 0.0, "reason": "best similarity 0.70 below min_similarity 0.80"}
        elements = read_json(self.run.path("elements.json"))["elements"]
        out, messages = complete.apply_donor_choices([open_one], elements,
                                                     {"element:target": {"donor_id": "donor", "flipped": False}})
        self.assertEqual((out[0]["tier"], out[0]["donor_id"], out[0]["reason"]), ("A", "donor", None))
        self.assertEqual(out[0]["donor_choice"]["automatic_tier"], "C")
        self.assertIn("instead of none (tier C)", messages[0])

    def test_tools_do_not_import_torch(self) -> None:
        code = "import sys; import tools.reference_donors; sys.exit(1 if 'torch' in sys.modules else 0)"
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT).returncode, 0)


if __name__ == "__main__":
    unittest.main()
