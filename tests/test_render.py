"""Tests for i2r.render: marking stays inside the mask, every changed pixel has a source, provenance keys.

Runs complete.run_step and render.run_step on a synthetic run folder (no GPU)."""
from __future__ import annotations

import copy
import logging
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from i2r import complete, render
from i2r.runio import Run, load_config, read_json, write_json
from tests.test_complete import DONOR_BOX, TARGET_BOX, scene

PROVENANCE_KEYS = {"target_id", "class", "tier", "target_box", "donor_id", "donor_box", "similarity",
                   "confidence", "alternatives", "method", "flipped", "views", "pixels_changed"}


def build_run(tmp: Path, include_tier_b: bool = False) -> Run:
    rgb, damage, donor_mask, _ = scene()
    h, w = damage.shape
    structure, run_dir = tmp / "structure", tmp / "run"
    structure.mkdir()
    (run_dir / "element_masks").mkdir(parents=True)
    Image.fromarray(rgb).save(structure / "image.png")
    Image.fromarray(damage * 255).save(run_dir / "damage_mask.png")
    Image.fromarray(donor_mask * 255).save(run_dir / "element_masks" / "donor.png")
    target_mask = np.zeros_like(damage)
    x1, y1, x2, y2 = TARGET_BOX
    target_mask[y1:y2, x1:x2] = 1
    Image.fromarray(target_mask * 255).save(run_dir / "element_masks" / "target.png")
    cfg = copy.deepcopy(load_config())
    cfg["restoration"]["include_tier_b"] = include_tier_b
    write_json(run_dir / "elements.json", {
        "structure_id": "synthetic", "image_size": [w, h], "classes": list(cfg["classes"]),
        "elements": [
            {"id": "donor", "class": "pilaster", "box": DONOR_BOX, "score": 1.0, "mask_file": "element_masks/donor.png",
             "source": "expert", "damage_status": "intact"},
            {"id": "target", "class": "pilaster", "box": TARGET_BOX, "score": 1.0,
             "mask_file": "element_masks/target.png", "source": "expert", "damage_status": "fragmented"},
        ]})
    base = {"alternatives": [], "similarity": None, "reason": None, "rule": None, "element_id": None}
    write_json(run_dir / "evidence.json", {"targets": [
        {**base, "target_id": "t_001", "kind": "fragmented_element", "element_id": "target", "class": "pilaster",
         "tier": "A", "target_box": TARGET_BOX, "donor_id": "donor", "donor_box": DONOR_BOX, "flipped": False,
         "similarity": 0.9, "confidence": 0.9},
        {**base, "target_id": "t_002", "kind": "implied_slot", "rule": "B1", "class": "pilaster", "tier": "B",
         "target_box": [95, 40, 115, 80], "donor_id": "donor", "donor_box": DONOR_BOX, "flipped": False,
         "confidence": 0.5},
        {**base, "target_id": "t_003", "kind": "uncovered_region", "class": None, "tier": "C",
         "target_box": [95, 40, 115, 80], "donor_id": None, "donor_box": None, "flipped": None,
         "confidence": 0.0, "reason": "no element evidence for these damage pixels"},
    ]})
    # a second damaged area for the B slot
    damage[40:80, 95:115] = 1
    Image.fromarray(damage * 255).save(run_dir / "damage_mask.png")
    logger = logging.getLogger(f"test_render.{id(tmp)}")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return Run("synthetic", structure, run_dir, cfg, logger, {"steps": {}, "run_id": "synthetic/test"})


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def render(self, include_tier_b: bool = False, views: tuple[str, ...] = ("restoration",)) -> Run:
        run = build_run(self.tmp, include_tier_b)
        opts = types.SimpleNamespace(views=list(views))
        complete.run_step(run, opts)
        render.run_step(run, opts)
        return run

    def test_files_and_provenance(self) -> None:
        run = self.render()
        for name in ("restoration.png", "restoration_source_map.png", "provenance.json", "targets_overlay.png",
                     "legend.png", "restoration_completed.png", "restoration_source.png", "completion.json"):
            self.assertTrue(run.path(name).is_file(), name)
        prov = read_json(run.path("provenance.json"))
        self.assertEqual(prov["statement"], "AI proposal, not a restoration decision")
        self.assertEqual([e["target_id"] for e in prov["targets"]], ["t_001", "t_002", "t_003"])
        for entry in prov["targets"]:
            self.assertTrue(PROVENANCE_KEYS <= set(entry), PROVENANCE_KEYS - set(entry))
        by_id = {e["target_id"]: e for e in prov["targets"]}
        self.assertEqual(by_id["t_001"]["views"], {"restoration": "completed"})
        self.assertEqual(by_id["t_001"]["method"], "donor_transfer")
        self.assertEqual(by_id["t_002"]["views"], {"restoration": "not_completed"})
        self.assertEqual(by_id["t_003"]["method"], "none")

    def test_every_changed_pixel_has_a_completed_source(self) -> None:
        run = self.render()
        photo = np.asarray(Image.open(run.image_path).convert("RGB"))
        completed = np.asarray(Image.open(run.path("restoration_completed.png")))
        labels = np.asarray(Image.open(run.path("restoration_source.png")))
        changed = np.any(completed != photo, axis=-1)
        self.assertTrue(changed.any())
        self.assertTrue(np.all(labels[changed] > 0))
        prov = read_json(run.path("provenance.json"))
        names = prov["views"]["restoration"]["labels"]
        status = {e["target_id"]: e["views"]["restoration"] for e in prov["targets"]}
        for value in np.unique(labels[labels > 0]):
            self.assertEqual(status[names[str(int(value))]], "completed")

    def test_marking_stays_inside_the_damage_mask(self) -> None:
        run = self.render()
        photo = np.asarray(Image.open(run.image_path).convert("RGB"))
        damage = np.asarray(Image.open(run.path("damage_mask.png"))) > 0
        marked = np.asarray(Image.open(run.path("restoration.png")).convert("RGB"))
        h = photo.shape[0]
        self.assertGreater(marked.shape[0], h)                       # label bar added below
        self.assertTrue(np.array_equal(marked[:h][~damage], photo[~damage]))
        self.assertFalse(np.array_equal(marked[:h][damage], photo[damage]))

    def test_open_tier_b_is_hatched_blue_and_included_b_is_not(self) -> None:
        run = self.render()
        marked = np.asarray(Image.open(run.path("restoration.png")).convert("RGB"))
        blue = tuple(run.config["restoration"]["hatch_color_b"])
        region = marked[40:80, 95:115].reshape(-1, 3)
        self.assertTrue(any(tuple(p) == blue for p in region))

    def test_both_views(self) -> None:
        run = self.render(views=("restoration", "first_construction"))
        for name in ("first_construction.png", "first_construction_source_map.png",
                     "first_construction_conjecture_overlay.png"):
            self.assertTrue(run.path(name).is_file(), name)
        prov = read_json(run.path("provenance.json"))
        by_id = {e["target_id"]: e for e in prov["targets"]}
        self.assertEqual(by_id["t_002"]["views"], {"restoration": "not_completed", "first_construction": "completed"})
        self.assertEqual(by_id["t_003"]["views"], {"restoration": "not_completed",
                                                   "first_construction": "not_completed"})
        self.assertEqual(prov["views"]["first_construction"]["markings_px"]["generative_px"], 0)
        # the marked images differ only inside the tier B target
        a = np.asarray(Image.open(run.path("restoration.png")).convert("RGB"))
        b = np.asarray(Image.open(run.path("first_construction.png")).convert("RGB"))
        h = np.asarray(Image.open(run.image_path)).shape[0]
        differ = np.any(a[:h] != b[:h], axis=-1)
        inside_b = np.zeros_like(differ)
        inside_b[40:80, 95:115] = True
        self.assertTrue(differ.any())
        self.assertFalse((differ & ~inside_b).any())

    def test_tier_b_switch_completes_and_records_it(self) -> None:
        run = self.render(include_tier_b=True)
        prov = read_json(run.path("provenance.json"))
        by_id = {e["target_id"]: e for e in prov["targets"]}
        self.assertEqual(by_id["t_002"]["views"]["restoration"], "completed")


if __name__ == "__main__":
    unittest.main()
